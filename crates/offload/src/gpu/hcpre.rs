//! The attention hyper-connection of the Qwen3.8-Flash-Next NPU engine's
//! delta layers in prefill, on a Vulkan device (the iGPU).
//!
//! WHY
//! ---
//! A delta layer's prefill is NPU-bound (`delta_pre` 10.5 + `post_delta`
//! 9.3 ms per 256 positions) while the iGPU, which runs only the
//! recurrence between them, idles ~16 of every 20 ms. The hyper-connection
//! at the start of `delta_pre` (norm per stream, the down matrix with the
//! injection rows, SiLU, the up matrix, the gated mean of the streams) is
//! ~3.5 ms of that graph and reads only the streams, so it runs here
//! (`hc_norm_npu.glsl`, `hc_down_npu.glsl`, `hc_up_npu.glsl`, f32 sums over
//! the graphs' int8 weights) and the NPU's `delta_pre_m` graph starts from
//! its outputs.
//!
//! Every layer's weights live in host-visible memory of the device, uploaded
//! once ([`HcPre::add_layer`]); a call reads the streams where the NPU put
//! them (its host buffers, imported once with `VK_EXT_external_memory_host`
//! as [`super::scan::DeltaScan`] does, a descriptor set per buffer) and
//! copies the mixed input / injection weights out. `QWEN_FLASH_HC_IMPORT=0`
//! (or a device without the extension) copies the streams in instead.
//!
//! On a device with `dp4a` the down and up matrices run on int8 inputs
//! (`hc_norm_dp4a.glsl`, `hc_down_dp4a.glsl`, `hc_up_dp4a.glsl`: the normed
//! streams rounded per position and stream, the down output per position;
//! exact int32 sums), ~2x the float kernels, which are bound by their FMAs
//! and the int8 -> float conversions. `QWEN_FLASH_HC_DP4A=0` is the A/B.

use super::*;

const W: usize = 2560;
const HCW: usize = 4 * W;
/// Most stream buffers imported (one descriptor set each); past it a call
/// copies.
const MAX_IMPORTS: usize = 64;
/// Rows of the down matrix with the injection rows folded in.
const RD: usize = 324;
const RANK: usize = 320;
/// Positions per call.
pub const HC_MAXN: usize = 256;
/// Positions per workgroup of the down and the up pass: the dp4a kernels'
/// (`hc_down_dp4a.glsl`'s 4 PT, `hc_up_dp4a.glsl`'s PP), the float ones'.
const POS_I8: (u32, u32) = (32, 16);
const POS_F32: (u32, u32) = (16, 4);
/// Bytes of a layer's codes: down + injection `[324][10240]`, up
/// `[10240][320]` (the kernels' offsets).
const WBYTES: usize = RD * HCW + HCW * RANK;
/// Floats of a layer's vectors: norm `[10240]`, down scales `[336]`, up
/// scales `[10240]`.
const PFLOATS: usize = HCW + 336 + HCW;

struct Buf {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	map: *mut u8,
}

/// See the module docs.
pub struct HcPre {
	dev: GpuDevice,
	sm: [vk::ShaderModule; 3],
	pl: [vk::Pipeline; 3],
	pool: vk::DescriptorPool,
	set: vk::DescriptorSet,
	params: (vk::Buffer, vk::DeviceMemory),
	/// x, weights, vectors, xn, m, mixed
	bufs: [Buf; 6],
	layers: usize,
	/// positions per workgroup of the down and the up pass
	pos: (u32, u32),
	/// the submission [`HcPre::begin`] left in flight and its positions
	pending: Option<(vk::CommandBuffer, usize)>,
	/// `QWEN_NPU_SCAN_TS=1`: a timestamp pool (start, end) and the period;
	/// each call prints its GPU interval (absolute, microseconds) under
	/// [`HcPre::set_tag`]'s tag
	ts: Option<(vk::QueryPool, f64)>,
	tag: &'static str,
	/// the host-import alignment, when the streams are imported
	align: Option<usize>,
	/// imported stream buffers by (base, length) and their descriptor sets
	/// by the streams' (address, bytes)
	imports: Vec<((usize, usize), vk::Buffer, vk::DeviceMemory)>,
	xsets: HashMap<(usize, usize), vk::DescriptorSet>,
}

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for HcPre {}

impl HcPre {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU), for
	/// `layers` delta layers.
	pub fn new(pick: Option<&str>, layers: usize) -> Result<HcPre, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let main = CString::new("main").unwrap();
		let d = &dev.device;
		let i8 = dev.use_dp4a && std::env::var("QWEN_FLASH_HC_DP4A").map_or(true, |v| v != "0");
		let spv: [&[u8]; 3] = if i8 {
			[
				include_bytes!("../shaders/hc_norm_dp4a.spv"),
				include_bytes!("../shaders/hc_down_dp4a.spv"),
				include_bytes!("../shaders/hc_up_dp4a.spv"),
			]
		} else {
			[
				include_bytes!("../shaders/hc_norm_npu.spv"),
				include_bytes!("../shaders/hc_down_npu.spv"),
				include_bytes!("../shaders/hc_up_npu.spv"),
			]
		};
		let sm = [
			create_sm(d, &load_spv(spv[0]))?,
			create_sm(d, &load_spv(spv[1]))?,
			create_sm(d, &load_spv(spv[2]))?,
		];
		let mut pl = [vk::Pipeline::null(); 3];
		for (p, &m) in pl.iter_mut().zip(&sm) {
			*p = compute_pipeline(d, dev.aux_pipeline_layout, m, &main)?;
		}
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 6 * (1 + MAX_IMPORTS as u32),
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 1 + MAX_IMPORTS as u32,
			},
		];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 1 + MAX_IMPORTS as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("hyper-connection descriptor pool: {e}"))?;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: pool,
				descriptor_set_count: 1,
				p_set_layouts: &dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("hyper-connection descriptor set: {e}"))?[0];
		let params = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let sizes = [
			HC_MAXN * HCW * 2,
			layers * WBYTES,
			layers * PFLOATS * 4,
			// `xn` f32; the dp4a form: `xq` int8, then per (position,
			// stream) its scale and the norm's rsqrt
			HC_MAXN * HCW * 4,
			HC_MAXN * RD * 4,
			HC_MAXN * W * 2,
		];
		let mut bufs = Vec::with_capacity(6);
		for &b in &sizes {
			let (buf, mem) = dev
				.try_create_readback_buffer(b as u64)
				.ok_or(format!("hyper-connection: no memory for {b} bytes"))?;
			let map = dev.map(mem, b) as *mut u8;
			bufs.push(Buf { buf, mem, map });
		}
		let bufs: [Buf; 6] = bufs.try_into().ok().unwrap();
		let mut infos: Vec<vk::DescriptorBufferInfo> = bufs
			.iter()
			.map(|b| vk::DescriptorBufferInfo {
				buffer: b.buf,
				offset: 0,
				range: vk::WHOLE_SIZE,
			})
			.collect();
		infos.push(vk::DescriptorBufferInfo {
			buffer: params.0,
			offset: 0,
			range: 64,
		});
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
		let align = dev
			.host_import_align()
			.filter(|_| std::env::var("QWEN_FLASH_HC_IMPORT").map_or(true, |v| v != "0"));
		let ts = std::env::var_os("QWEN_NPU_SCAN_TS")
			.and_then(|_| unsafe {
				dev.device
					.create_query_pool(
						&vk::QueryPoolCreateInfo {
							query_type: vk::QueryType::TIMESTAMP,
							query_count: 2,
							..Default::default()
						},
						None,
					)
					.ok()
			})
			.map(|qp| {
				let p = unsafe { dev.instance.get_physical_device_properties(dev.physical) };
				(qp, p.limits.timestamp_period as f64)
			});
		Ok(HcPre {
			align,
			imports: Vec::new(),
			xsets: HashMap::new(),
			dev,
			sm,
			pl,
			pool,
			set,
			params,
			bufs,
			layers,
			pos: if i8 { POS_I8 } else { POS_F32 },
			pending: None,
			ts,
			tag: "hc",
		})
	}

	/// The name `QWEN_NPU_SCAN_TS`'s lines carry.
	pub fn set_tag(&mut self, tag: &'static str) {
		self.tag = tag;
	}

	/// Delta layer `di`'s weights: `norm` `[10240]`, `down` the int8 codes
	/// `[324][10240]` (`fl{L}.hca.dinj.w`) with their effective row scales
	/// (the graphs' `codes x 2^-7 x s`), `up` `[10240][320]` likewise.
	pub fn add_layer(
		&mut self,
		di: usize,
		norm: &[f32],
		down: &[u8],
		down_s: &[f32],
		up: &[u8],
		up_s: &[f32],
	) -> Result<(), String> {
		if di >= self.layers
			|| norm.len() != HCW
			|| down.len() != RD * HCW
			|| down_s.len() != RD
			|| up.len() != HCW * RANK
			|| up_s.len() != HCW
		{
			return Err(format!("hyper-connection layer {di}: bad sizes"));
		}
		// SAFETY: the mappings hold `layers` layers
		unsafe {
			let w = self.bufs[1].map.add(di * WBYTES);
			ptr::copy_nonoverlapping(down.as_ptr(), w, down.len());
			ptr::copy_nonoverlapping(up.as_ptr(), w.add(down.len()), up.len());
			let p = (self.bufs[2].map as *mut f32).add(di * PFLOATS);
			ptr::copy_nonoverlapping(norm.as_ptr(), p, HCW);
			ptr::copy_nonoverlapping(down_s.as_ptr(), p.add(HCW), RD);
			ptr::copy_nonoverlapping(up_s.as_ptr(), p.add(HCW + 336), HCW);
		}
		Ok(())
	}

	/// The descriptor set that reads the streams from `x` in place,
	/// importing its pages on first use; `None` when it cannot (no import,
	/// too many, or the import failed: the caller copies).
	fn import_set(&mut self, x: &[u16]) -> Option<vk::DescriptorSet> {
		let a = self.align?;
		let p = x.as_ptr() as usize;
		let len = x.len() * 2;
		if let Some(&s) = self.xsets.get(&(p, len)) {
			return Some(s);
		}
		if self.xsets.len() >= MAX_IMPORTS {
			return None;
		}
		let (base, end) = (p & !(a - 1), (p + len + a - 1) & !(a - 1));
		let (buf, off) = match self
			.imports
			.iter()
			.find(|(k, ..)| k.0 <= base && k.0 + k.1 >= end)
		{
			Some(&(k, b, _)) => (b, (p - k.0) as u64),
			None => {
				let (b, m) = self.dev.import_host_range(base, end - base).ok()?;
				self.imports.push(((base, end - base), b, m));
				(b, (p - base) as u64)
			}
		};
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.ok()?[0];
		let mut infos: Vec<vk::DescriptorBufferInfo> = self
			.bufs
			.iter()
			.map(|b| vk::DescriptorBufferInfo {
				buffer: b.buf,
				offset: 0,
				range: vk::WHOLE_SIZE,
			})
			.collect();
		infos[0] = vk::DescriptorBufferInfo {
			buffer: buf,
			offset: off,
			range: len as u64,
		};
		infos.push(vk::DescriptorBufferInfo {
			buffer: self.params.0,
			offset: 0,
			range: 64,
		});
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
		self.xsets.insert((p, len), set);
		Some(set)
	}

	/// Delta layer `di` on the streams `x` `[n][10240]` f16, left in flight
	/// until [`HcPre::end`] (one at a time: a second `begin` waits for the
	/// first and drops its result). `x` is read in place: it must be host
	/// memory at an unchanging address that outlives this object (the NPU's
	/// host buffers), left unchanged until [`HcPre::end`] returns.
	pub fn begin(&mut self, di: usize, x: &[u16], n: usize) -> Result<(), String> {
		// one in flight: an earlier one nobody collected is dropped
		if let Some((cmd, _)) = self.pending.take() {
			self.dev
				.wait_submitted(cmd)
				.map_err(|e| format!("hyper-connection: {e}"))?;
		}
		if di >= self.layers || n == 0 || n > HC_MAXN || x.len() < n * HCW {
			return Err(format!(
				"hyper-connection: layer {di}, {n} positions: bad sizes"
			));
		}
		let set = match self.import_set(x) {
			Some(s) => s,
			None => {
				// SAFETY: the mapping holds HC_MAXN positions
				unsafe {
					ptr::copy_nonoverlapping(
						x.as_ptr() as *const u8,
						self.bufs[0].map,
						n * HCW * 2,
					);
				}
				self.set
			}
		};
		let d = &self.dev.device;
		let (layout, params, pl, pos) = (
			self.dev.aux_pipeline_layout,
			self.params.0,
			self.pl,
			self.pos,
		);
		let mut pr = [0u32; 16];
		pr[0] = n as u32;
		pr[1] = (di * WBYTES / 16) as u32;
		pr[2] = (di * PFLOATS) as u32;
		let pb: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
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
		let n32 = n as u32;
		let qp = self.ts.map(|t| t.0);
		let cmd = self
			.dev
			.submit_nowait(|cmd| unsafe {
				if let Some(qp) = qp {
					d.cmd_reset_query_pool(cmd, qp, 0, 2);
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
				}
				d.cmd_update_buffer(cmd, params, 0, &pb);
				barrier(
					cmd,
					P::TRANSFER | P::HOST,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE | A::HOST_WRITE,
					A::UNIFORM_READ | A::SHADER_READ,
				);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					layout,
					0,
					&[set],
					&[],
				);
				let groups = [
					(n32 * 4, 1),
					((RD as u32).div_ceil(32), n32.div_ceil(pos.0)),
					(W as u32 / 256, n32.div_ceil(pos.1)),
				];
				for (k, &(gx, gy)) in groups.iter().enumerate() {
					if k > 0 {
						barrier(
							cmd,
							P::COMPUTE_SHADER,
							P::COMPUTE_SHADER,
							A::SHADER_WRITE,
							A::SHADER_READ,
						);
					}
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[k]);
					d.cmd_dispatch(cmd, gx, gy, 1);
				}
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
				if let Some(qp) = qp {
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
				}
			})
			.map_err(|e| format!("hyper-connection: {e}"))?;
		self.pending = Some((cmd, n));
		Ok(())
	}

	/// Wait for [`HcPre::begin`]'s submission: the mixed input `mixed`
	/// `[n][2560]` and the injection weights `inj` `[n][4]`, f16.
	pub fn end(&mut self, mixed: &mut [u16], inj: &mut [u16]) -> Result<(), String> {
		let (cmd, n) = self
			.pending
			.take()
			.ok_or("hyper-connection: nothing in flight")?;
		self.dev
			.wait_submitted(cmd)
			.map_err(|e| format!("hyper-connection: {e}"))?;
		if let Some((qp, period)) = self.ts {
			let mut t = [0u64; 2];
			// SAFETY: both queries were written by the submission waited for
			if unsafe {
				self.dev.device.get_query_pool_results(
					qp,
					0,
					&mut t,
					vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
				)
			}
			.is_ok()
			{
				eprintln!(
					"[igts] {} n {n}: {:.1} {:.1}",
					self.tag,
					t[0] as f64 * period * 1e-3,
					t[1] as f64 * period * 1e-3
				);
			}
		}
		if mixed.len() < n * W || inj.len() < n * 4 {
			return Err("hyper-connection: output too small".into());
		}
		// SAFETY: the mappings hold HC_MAXN positions
		unsafe {
			ptr::copy_nonoverlapping(self.bufs[5].map as *const u16, mixed.as_mut_ptr(), n * W);
			let m = std::slice::from_raw_parts(self.bufs[4].map as *const f32, n * RD);
			for p in 0..n {
				for c in 0..4 {
					inj[p * 4 + c] = quant::f32_to_fp16(m[p * RD + RANK + c]);
				}
			}
		}
		Ok(())
	}
}

impl Drop for HcPre {
	fn drop(&mut self) {
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			for &(_, b, m) in &self.imports {
				hostimp::free_import(d, b, m);
			}
			for b in &self.bufs {
				d.unmap_memory(b.mem);
				d.destroy_buffer(b.buf, None);
				d.free_memory(b.mem, None);
			}
			for (&p, &m) in self.pl.iter().zip(&self.sm) {
				d.destroy_pipeline(p, None);
				d.destroy_shader_module(m, None);
			}
			if let Some((qp, _)) = self.ts {
				d.destroy_query_pool(qp, None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_buffer(self.params.0, None);
			d.free_memory(self.params.1, None);
		}
	}
}
