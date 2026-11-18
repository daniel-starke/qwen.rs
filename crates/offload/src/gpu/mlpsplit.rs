//! A share of the NPU engine's decode MLPs on a discrete GPU.
//!
//! WHY
//! ---
//! NPU-engine decode is bound by reading 12.2 GB of weights per pass from
//! DRAM (~58 GB/s for NPU and cores together). Weight bytes resident in VRAM
//! leave that bus. The MLP (`crates/npu/src/mlp.rs`) splits without any
//! exchange of intermediates: gate / up rows `[0, R)` and down COLUMNS
//! `[0, R)` are a narrower MLP of their own, and the block rotation before
//! the down matrix stays inside it when `R` is a multiple of 256. So the GPU
//! computes `W_down[:, :R] H (silu(W_gate[:R] a) * W_up[:R] b * d3[:R])`
//! from the two rotated activations and returns `[s][5120]` floats to add.
//!
//! Per layer in VRAM: `7680 R` bytes of codes (R = 9216 of 17408 rows for 64
//! layers in ~4.5 GB). One submission per layer: two GEMVs
//! (`i4_dp4a.glsl`, int8 activations as on the cores), SiLU + Hadamard +
//! quantization (`mlp_act.glsl`), the down GEMV into a host-visible buffer. Every failure is an `Err`: the caller computes the share on the
//! CPU and builds a new `MlpSplit` later (another application may have
//! taken the VRAM).

use super::*;

const W: usize = 5120;
const F: usize = 17408;
const MAXS: usize = 4;
/// Code bytes per GPU row and layer: gate + up rows, the down column pair.
const ROW_BYTES: u64 = (W / 2 * 2 + W / 2) as u64;
/// Host activation buffer: per matrix (gate, up) the int8 bytes, the block
/// scales and the bias corrections, at these offsets.
const XREG: usize = 65536;
const XINV: usize = 32768;
const XCORR: usize = 36864;
const CORR: usize = 128;
/// Rows per warp of `i4_dp4a.glsl` (its `NR`).
const NR: u32 = 8;

/// See the module docs.
pub struct MlpSplit {
	dev: GpuDevice,
	sm: [vk::ShaderModule; 2],
	pl: [vk::Pipeline; 2],
	pool: vk::DescriptorPool,
	params: [(vk::Buffer, vk::DeviceMemory); 2],
	bufs: Vec<(vk::Buffer, vk::DeviceMemory)>,
	stage: (vk::Buffer, vk::DeviceMemory, *mut u8),
	/// g, u (f32 `[MAXS][rows]`), then the down activation: bytes, block
	/// scales, corrections
	scratch: [vk::Buffer; 5],
	xin: (vk::Buffer, *mut u8),
	yout: (vk::Buffer, *const f32),
	sets: Vec<[vk::DescriptorSet; 4]>,
	rows: usize,
	s_set: (usize, usize),
	cmd: Option<vk::CommandBuffer>,
}

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for MlpSplit {}

impl MlpSplit {
	/// On the Vulkan adapter `pick` names, for `layers` layers and at most
	/// `max_rows` GPU rows per layer; fewer when the device's memory budget
	/// less `reserve` bytes holds fewer (an error when it holds none).
	pub fn new(
		pick: Option<&str>,
		layers: usize,
		max_rows: usize,
		reserve: u64,
	) -> Result<MlpSplit, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let mut rows = max_rows.min(F) / 1024 * 1024;
		if let Some((budget, usage)) = dev.vram_budget() {
			let free = budget.saturating_sub(usage).saturating_sub(reserve);
			rows = rows.min((free / (layers as u64 * (ROW_BYTES + 12)) / 1024 * 1024) as usize);
		}
		if rows == 0 {
			return Err("mlp split: no VRAM for a share of the MLP weights".into());
		}
		if dev.pipeline_gemm_q4k_dp4a.is_none() {
			return Err("mlp split: the device has no (working) integer dot product".into());
		}
		let d = &dev.device;
		let main = CString::new("main").unwrap();
		let sm = [
			create_sm(d, &load_spv(include_bytes!("../shaders/i4_dp4a.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/mlp_act.spv")))?,
		];
		let pl = [
			compute_pipeline(d, dev.aux_pipeline_layout, sm[0], &main)?,
			compute_pipeline(d, dev.aux_pipeline_layout, sm[1], &main)?,
		];
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: (24 * layers) as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: (4 * layers) as u32,
			},
		];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: (4 * layers) as u32,
					pool_size_count: 2,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("mlp split: descriptor pool: {e}"))?;
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
		let st = vk::BufferUsageFlags::STORAGE_BUFFER;
		let mut bufs = Vec::new();
		let mut mk =
			|size: usize, usage, props| -> Result<(vk::Buffer, vk::DeviceMemory), String> {
				let b = dev
					.try_create_gpu_buffer(size as u64, usage, props)
					.ok_or("mlp split: buffer allocation failed")?;
				bufs.push(b);
				Ok(b)
			};
		let pu = vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
		let params = [mk(64, pu, local)?, mk(64, pu, local)?];
		let stage = mk(rows * W / 2, vk::BufferUsageFlags::TRANSFER_SRC, host)?;
		let pcorr = mk(MAXS * CORR * 4, st, local)?.0;
		let scratch = [
			mk(MAXS * rows * 4, st, local)?.0,
			mk(MAXS * rows * 4, st, local)?.0,
			mk(MAXS * rows, st, local)?.0,
			mk(MAXS * rows / 256 * 4, st, local)?.0,
			pcorr,
		];
		let xin = mk(2 * XREG, st, host | vk::MemoryPropertyFlags::HOST_CACHED)
			.or_else(|_| mk(2 * XREG, st, host))?;
		let yout = mk(
			MAXS * W * 4,
			st,
			host | vk::MemoryPropertyFlags::HOST_CACHED,
		)
		.or_else(|_| mk(MAXS * W * 4, st, host))?;
		let map = |m: vk::DeviceMemory| {
			unsafe { d.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
				.map_err(|e| format!("mlp split: map: {e}"))
		};
		let (sp, xp, yp) = (
			map(stage.1)? as *mut u8,
			map(xin.1)? as *mut u8,
			map(yout.1)? as *const f32,
		);
		Ok(MlpSplit {
			sm,
			pl,
			pool,
			params,
			stage: (stage.0, stage.1, sp),
			scratch,
			xin: (xin.0, xp),
			yout: (yout.0, yp),
			sets: Vec::new(),
			rows,
			s_set: (0, 0),
			cmd: None,
			bufs,
			dev,
		})
	}

	/// GPU rows per layer (a multiple of 1024).
	pub fn rows(&self) -> usize {
		self.rows
	}

	#[allow(clippy::wrong_self_convention)]
	fn to_vram(
		&mut self,
		fill: impl FnOnce(&mut [u8]),
		bytes: usize,
	) -> Result<vk::Buffer, String> {
		let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
		let b = self
			.dev
			.try_create_gpu_buffer(bytes as u64, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
			.ok_or("mlp split: VRAM allocation failed")?;
		self.bufs.push(b);
		// SAFETY: the staging map holds `rows * W / 2` bytes, the largest upload
		fill(unsafe { std::slice::from_raw_parts_mut(self.stage.2, bytes) });
		let (src, d) = (self.stage.0, &self.dev.device);
		self.dev
			.submit_and_wait(|cmd| unsafe {
				d.cmd_copy_buffer(
					cmd,
					src,
					b.0,
					&[vk::BufferCopy {
						src_offset: 0,
						dst_offset: 0,
						size: bytes as u64,
					}],
				)
			})
			.map_err(|e| format!("mlp split: upload: {e}"))?;
		Ok(b.0)
	}

	/// The next layer's weights (layers are added in order): `gate` / `up`
	/// `[F][W / 2]` and `down` `[W][F / 2]` code bytes (nibbles biased by
	/// 8), row scales `sg` / `su` `[F]` and `sd` `[W]`, the diagonal `d3`
	/// `[F]` before the last rotation.
	#[allow(clippy::too_many_arguments)]
	pub fn add_layer(
		&mut self,
		gate: &[u8],
		up: &[u8],
		down: &[u8],
		sg: &[f32],
		su: &[f32],
		d3: &[f32],
		sd: &[f32],
	) -> Result<(), String> {
		let r = self.rows;
		assert!(
			gate.len() == F * W / 2 && up.len() == gate.len() && down.len() == gate.len() && r <= F
		);
		let gb = self.to_vram(|m| m.copy_from_slice(&gate[..r * W / 2]), r * W / 2)?;
		let ub = self.to_vram(|m| m.copy_from_slice(&up[..r * W / 2]), r * W / 2)?;
		let db = self.to_vram(
			|m| {
				for (o, i) in m.chunks_exact_mut(r / 2).zip(down.chunks_exact(F / 2)) {
					o.copy_from_slice(&i[..r / 2]);
				}
			},
			W * r / 2,
		)?;
		let cb = self.to_vram(
			|m| {
				let f: Vec<u8> = sg[..r]
					.iter()
					.chain(&su[..r])
					.chain(&d3[..r])
					.chain(sd)
					.flat_map(|v| v.to_le_bytes())
					.collect();
				m.copy_from_slice(&f);
			},
			(3 * r + W) * 4,
		)?;
		let (r4, y4) = ((r * 4) as u64, (MAXS * W * 4) as u64);
		let s4 = (MAXS * r * 4) as u64;
		let [g, u, px, pinv, pcorr] = self.scratch;
		let x = self.xin.0;
		let xr = |which: usize| {
			let o = (which * XREG) as u64;
			[
				(x, o, (MAXS * W) as u64),
				(x, o + XINV as u64, (MAXS * W / 256 * 4) as u64),
				(x, o + XCORR as u64, (MAXS * CORR * 4) as u64),
			]
		};
		let (xa, xb) = (xr(0), xr(1));
		let pc = (pcorr, 0, vk::WHOLE_SIZE);
		let pi = (pinv, 0, vk::WHOLE_SIZE);
		let sets = [
			self.make_set(
				&[
					(gb, 0, vk::WHOLE_SIZE),
					xa[0],
					(g, 0, s4),
					(cb, 0, r4),
					xa[1],
					xa[2],
				],
				0,
			)?,
			self.make_set(
				&[
					(ub, 0, vk::WHOLE_SIZE),
					xb[0],
					(u, 0, s4),
					(cb, r4, r4),
					xb[1],
					xb[2],
				],
				0,
			)?,
			self.make_set(
				&[
					(g, 0, s4),
					(u, 0, s4),
					(cb, 2 * r4, r4),
					(px, 0, vk::WHOLE_SIZE),
					pi,
					pc,
				],
				0,
			)?,
			self.make_set(
				&[
					(db, 0, vk::WHOLE_SIZE),
					(px, 0, vk::WHOLE_SIZE),
					(self.yout.0, 0, y4),
					(cb, 3 * r4, (W * 4) as u64),
					pi,
					pc,
				],
				1,
			)?,
		];
		self.sets.push(sets);
		Ok(())
	}

	fn make_set(
		&self,
		b6: &[(vk::Buffer, u64, u64); 6],
		params: usize,
	) -> Result<vk::DescriptorSet, String> {
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("mlp split: descriptor set: {e}"))?[0];
		let info = |b: vk::Buffer, off: u64, range: u64| vk::DescriptorBufferInfo {
			buffer: b,
			offset: off,
			range,
		};
		let infos: Vec<vk::DescriptorBufferInfo> = b6
			.iter()
			.map(|x| info(x.0, x.1, x.2))
			.chain(Some(info(self.params[params].0, 0, 64)))
			.collect();
		let writes: Vec<vk::WriteDescriptorSet> = (0..7)
			.map(|i| vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: if i < 6 { i as u32 } else { 8 },
				descriptor_type: if i < 6 {
					vk::DescriptorType::STORAGE_BUFFER
				} else {
					vk::DescriptorType::UNIFORM_BUFFER
				},
				descriptor_count: 1,
				p_buffer_info: &infos[i],
				..Default::default()
			})
			.collect();
		unsafe { d.update_descriptor_sets(&writes, &[]) };
		Ok(set)
	}

	/// Start layer `l`'s share for `s` positions on the rotated int8
	/// activations of the gate and the up matrix: per position the even
	/// elements, the odd elements (`[s][W / 2]` each), the scale back to
	/// f32 and 8 x the sum of the values. Only rows / columns `[0, used)`
	/// are computed (a multiple of 1024, at most [`Self::rows`]). Returns at
	/// once; [`Self::finish`] must follow.
	#[allow(clippy::type_complexity)]
	pub fn begin(
		&mut self,
		l: usize,
		s: usize,
		used: usize,
		ab: [(&[i8], &[i8], &[f32], &[i32]); 2],
	) -> Result<(), String> {
		if used == 0
			|| used > self.rows
			|| !used.is_multiple_of(1024)
			|| self.cmd.is_some()
			|| l >= self.sets.len()
			|| s == 0 || s > MAXS
			|| ab.iter().any(|a| {
				a.0.len() < s * W / 2 || a.1.len() < s * W / 2 || a.2.len() < s || a.3.len() < s
			}) {
			return Err("mlp split: bad call".into());
		}
		let d = &self.dev.device;
		use vk::{AccessFlags as A, PipelineStageFlags as P};
		let barrier = |cmd, a, b, sa, da| unsafe {
			d.cmd_pipeline_barrier(
				cmd,
				a,
				b,
				vk::DependencyFlags::empty(),
				&[vk::MemoryBarrier {
					src_access_mask: sa,
					dst_access_mask: da,
					..Default::default()
				}],
				&[],
				&[],
			)
		};
		if (s, used) != self.s_set {
			let pb = |v: [u32; 4]| -> Vec<u8> {
				v.iter()
					.chain(&[0u32; 12])
					.flat_map(|x| x.to_le_bytes())
					.collect()
			};
			let (pa, pd) = (
				pb([used as u32, W as u32, s as u32, (W / 32) as u32]),
				pb([W as u32, used as u32, s as u32, (self.rows / 32) as u32]),
			);
			let (ba, bd) = (self.params[0].0, self.params[1].0);
			self.dev
				.submit_and_wait(|cmd| unsafe {
					d.cmd_update_buffer(cmd, ba, 0, &pa);
					d.cmd_update_buffer(cmd, bd, 0, &pd);
					barrier(
						cmd,
						P::TRANSFER,
						P::COMPUTE_SHADER,
						A::TRANSFER_WRITE,
						A::UNIFORM_READ,
					);
				})
				.map_err(|e| format!("mlp split: params: {e}"))?;
			self.s_set = (s, used);
		}
		// per 32 elements 16 even bytes, then 16 odd bytes; the position's
		// scale for every block, its bias in block 0
		for (which, (even, odd, inv, bias)) in ab.into_iter().enumerate() {
			// SAFETY: the map holds 2 x XREG bytes
			let reg = unsafe { std::slice::from_raw_parts_mut(self.xin.1.add(which * XREG), XREG) };
			let i8s =
				|v: &[i8]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
			for ((o, e), d) in reg[..s * W]
				.chunks_exact_mut(32)
				.zip(i8s(&even[..s * W / 2]).chunks_exact(16))
				.zip(i8s(&odd[..s * W / 2]).chunks_exact(16))
			{
				o[..16].copy_from_slice(e);
				o[16..].copy_from_slice(d);
			}
			for j in 0..s {
				for blk in 0..W / 256 {
					reg[XINV + (j * (W / 256) + blk) * 4..][..4]
						.copy_from_slice(&inv[j].to_le_bytes());
				}
				reg[XCORR + j * CORR * 4..][..CORR * 4].fill(0);
				reg[XCORR + j * CORR * 4..][..4]
					.copy_from_slice(&(-inv[j] * bias[j] as f32).to_le_bytes());
			}
		}
		let (pl, layout, sets, rows) = (
			self.pl,
			self.dev.aux_pipeline_layout,
			self.sets[l],
			used as u32,
		);
		let bind = |cmd, set: vk::DescriptorSet| unsafe {
			d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[])
		};
		let cmd = self
			.dev
			.submit_nowait(|cmd| unsafe {
				let rw = |cmd| {
					barrier(
						cmd,
						P::COMPUTE_SHADER,
						P::COMPUTE_SHADER,
						A::SHADER_WRITE,
						A::SHADER_READ,
					)
				};
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[0]);
				bind(cmd, sets[0]);
				d.cmd_dispatch(cmd, rows / NR, 1, 1);
				bind(cmd, sets[1]);
				d.cmd_dispatch(cmd, rows / NR, 1, 1);
				rw(cmd);
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[1]);
				bind(cmd, sets[2]);
				d.cmd_dispatch(cmd, rows / 256, s as u32, 1);
				rw(cmd);
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[0]);
				bind(cmd, sets[3]);
				d.cmd_dispatch(cmd, W as u32 / NR, 1, 1);
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("mlp split: submit: {e}"))?;
		self.cmd = Some(cmd);
		Ok(())
	}

	/// True once the share begun is done (never blocks).
	pub fn done(&self) -> bool {
		self.cmd.is_none()
			|| matches!(
				unsafe { self.dev.device.get_fence_status(self.dev.fence) },
				Ok(true)
			)
	}

	/// Wait for the share begun and add it to `y` `[s][W]`.
	pub fn finish(&mut self, s: usize, y: &mut [f32]) -> Result<(), String> {
		let cmd = self.cmd.take().ok_or("mlp split: nothing begun")?;
		self.dev
			.wait_submitted(cmd)
			.map_err(|e| format!("mlp split: {e}"))?;
		// SAFETY: the map holds MAXS x W floats; the submission is done
		let g = unsafe { std::slice::from_raw_parts(self.yout.1, s * W) };
		for (o, v) in y[..s * W].iter_mut().zip(g) {
			*o += v;
		}
		Ok(())
	}
}

impl Drop for MlpSplit {
	fn drop(&mut self) {
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			for i in 0..2 {
				d.destroy_pipeline(self.pl[i], None);
				d.destroy_shader_module(self.sm[i], None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			for &(b, m) in &self.bufs {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
		}
		let _ = self.stage.1;
	}
}
