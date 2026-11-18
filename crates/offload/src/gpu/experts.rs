//! A share of the routed experts of the Qwen3.8-Flash-Next NPU engine's
//! prefill on a Vulkan device (the iGPU).
//!
//! WHY
//! ---
//! The expert pass after each layer's dense part is ~30 % of a prefill
//! (~130 ms per layer at 3.2k tokens), NPU-bound, and the iGPU idles
//! through all of it. Here it runs some of the slots (one expert, up to 64
//! of its tokens) at the same time: `exp_gemm.glsl` (gate / up and down,
//! int8 x int4 `dp4a`, exact int32 sums) and `exp_act.glsl` (SwiGLU, the
//! down input's sign / range, Hadamard, int8), the math of
//! `npu::flash::cpuexp` on the expert pools the NPU binds, imported
//! zero-copy ([`Experts::add_layer`]). `exp_xq.glsl` rounds the MoE input
//! rows to int8 first, once per token.
//!
//! A call ([`Experts::begin`]) copies the MoE input rows in and leaves one
//! submission in flight; [`Experts::end`] returns each slot row's expert
//! output, f16 with the row scale applied (not the routing weight).

use super::*;

const W: usize = 2560;
const FE: usize = 640;
/// Rows of a slot.
pub const EXP_SLOT: usize = 64;
/// Rows of a GEMM workgroup (`exp_gemm.glsl`'s `RW`), every slot size.
const RW: usize = 64;
/// The GEMMs' slot sizes (`exp_gemm.glsl`'s `TOK`), widest first: a slot
/// runs on the narrowest that holds it. A workgroup computes all `TOK`
/// tokens whether the slot has them or not, so a slot of 3 tokens on the
/// 32-token kernel cost as much as one of 32 (~0.16 ms on the Arrow Lake
/// iGPU); short prefill pieces are mostly such slots.
const TOKS: [usize; 4] = [64, 32, 16, 8];
const NT: usize = TOKS.len();
/// The kernels: the MoE input to int8, gate / up per slot size, act, down
/// per slot size.
const NK: usize = 2 + 2 * NT;
/// Kernel index of gate / up and of down for slot size class `c`.
const fn k_gu(c: usize) -> usize {
	1 + c
}
const K_ACT: usize = 1 + NT;
const fn k_down(c: usize) -> usize {
	2 + NT + c
}
/// The slot size class of a slot of `n` tokens.
fn class(n: usize) -> usize {
	(0..NT).rev().find(|&c| TOKS[c] >= n).unwrap_or(0)
}

struct Buf {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	map: *mut u8,
	len: usize,
}

/// One layer's imported pools and descriptor sets (one per kernel).
struct Layer {
	sets: [vk::DescriptorSet; NK],
	/// codes / scales offsets: gate / up words, its scales (f16), d.pre
	/// (f16), down words, its scales (f16)
	ofs: [u32; 5],
}

/// See the module docs.
pub struct Experts {
	dev: GpuDevice,
	sm: [vk::ShaderModule; NK],
	pl: [vk::Pipeline; NK],
	pool: vk::DescriptorPool,
	/// one 256-byte region per kernel
	params: (vk::Buffer, vk::DeviceMemory),
	/// x (f16 rows), slots + token lists, G, Hq, Y, Xq (x as int8)
	bufs: [Buf; 6],
	imports: Vec<(vk::Buffer, vk::DeviceMemory)>,
	layers: Vec<Option<Layer>>,
	max_rows: usize,
	max_tok: usize,
	/// the submission in flight and its slot rows
	pending: Option<(vk::CommandBuffer, usize)>,
	/// two timestamps around each submission (`None`: the device has no
	/// compute timestamps) and the timestamp period, ns
	ts: Option<(vk::QueryPool, f64)>,
	/// the device time of the last submission [`Experts::end`] waited for, ms
	last_ms: Option<f64>,
}

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for Experts {}

impl Experts {
	/// On the Vulkan adapter `pick` names, for `layers` layers, calls of up
	/// to `max_tok` tokens and `max_rows` slot rows.
	pub fn new(
		pick: Option<&str>,
		layers: usize,
		max_tok: usize,
		max_rows: usize,
	) -> Result<Experts, String> {
		let dev = GpuDevice::init_pick(pick)?;
		if dev.host_import_align().is_none() {
			return Err("experts: the device does not import host memory".into());
		}
		let main = CString::new("main").unwrap();
		let d = &dev.device;
		let sm = [
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_xq.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_gu.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_gu32.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_gu16.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_gu8.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_act.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_down.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_down32.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_down16.spv")))?,
			create_sm(d, &load_spv(include_bytes!("../shaders/exp_down8.spv")))?,
		];
		let mut pl = [vk::Pipeline::null(); NK];
		for (p, &m) in pl.iter_mut().zip(&sm) {
			*p = compute_pipeline(d, dev.aux_pipeline_layout, m, &main)?;
		}
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: (layers * NK * 6) as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: (layers * NK) as u32,
			},
		];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: (layers * NK) as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("experts: descriptor pool: {e}"))?;
		let params = dev.create_gpu_buffer(
			(NK * 256) as u64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let sizes = [
			max_tok * W * 2,
			(4 * max_rows + max_rows) * 4,
			max_rows * 2 * FE * 4,
			max_rows * FE,
			max_rows * W * 2,
			max_tok * W,
		];
		let mut bufs = Vec::with_capacity(6);
		for &b in &sizes {
			let (buf, mem) = dev
				.try_create_readback_buffer(b as u64)
				.ok_or(format!("experts: no memory for {b} bytes"))?;
			let map = dev.map(mem, b) as *mut u8;
			bufs.push(Buf {
				buf,
				mem,
				map,
				len: b,
			});
		}
		let bufs: [Buf; 6] = bufs.try_into().ok().unwrap();
		// the device time per call, for the split's calibration
		// (`npu::flash::engine`, `quant::calib`)
		let limits = unsafe { dev.instance.get_physical_device_properties(dev.physical) }.limits;
		let ts = (limits.timestamp_compute_and_graphics == vk::TRUE)
			.then(|| unsafe {
				d.create_query_pool(
					&vk::QueryPoolCreateInfo {
						query_type: vk::QueryType::TIMESTAMP,
						query_count: 2,
						..Default::default()
					},
					None,
				)
			})
			.and_then(|r| r.ok())
			.map(|qp| (qp, limits.timestamp_period as f64));
		Ok(Experts {
			dev,
			sm,
			pl,
			pool,
			params,
			bufs,
			imports: Vec::new(),
			layers: (0..layers).map(|_| None).collect(),
			max_rows,
			max_tok,
			pending: None,
			ts,
			last_ms: None,
		})
	}

	/// The device's `maxStorageBufferRange` (an expert pool is imported
	/// as one buffer: ~840 MB of gate / up codes per layer).
	pub fn max_range(&self) -> u64 {
		unsafe {
			self.dev
				.instance
				.get_physical_device_properties(self.dev.physical)
		}
		.limits
		.max_storage_buffer_range as u64
	}

	/// Whether the driver reports signed / mixed 4 x 8-bit packed dot
	/// products as accelerated (else `dp4a` is emulated).
	pub fn dp4a_accelerated(&self) -> (bool, bool) {
		let mut ip = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
		let mut p2 = vk::PhysicalDeviceProperties2 {
			p_next: &mut ip as *mut _ as *mut c_void,
			..Default::default()
		};
		unsafe {
			self.dev
				.instance
				.get_physical_device_properties2(self.dev.physical, &mut p2)
		};
		(
			ip.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE,
			ip.integer_dot_product_accumulating_saturating4x8_bit_packed_signed_accelerated
				== vk::TRUE,
		)
	}

	/// `(budget, usage)` of the device's largest heap (`VK_EXT_memory_budget`).
	pub fn budget(&self) -> Option<(u64, u64)> {
		self.dev.vram_budget()
	}

	fn import(&mut self, b: &[u8]) -> Result<(vk::Buffer, u32), String> {
		let a = self
			.dev
			.host_import_align()
			.ok_or("experts: no host import")?;
		let p = b.as_ptr() as usize;
		let (base, end) = (p & !(a - 1), (p + b.len() + a - 1) & !(a - 1));
		let (buf, mem) = self.dev.import_host_range(base, end - base)?;
		self.imports.push((buf, mem));
		let o = p - base;
		if !o.is_multiple_of(4) {
			return Err("experts: pool not 4-byte aligned".into());
		}
		Ok((buf, (o / 4) as u32))
	}

	/// Layer `l`'s expert pools as the NPU binds them (`cpuexp::Pool`):
	/// gate / up codes `[NE][1280][1280]`, their scales `[NE][1280]` f16,
	/// down codes `[NE][2560][320]`, their scales `[NE][2560]` f16, `d.pre`
	/// `[NE][640]` f16. The memory must outlive `self`.
	pub fn add_layer(
		&mut self,
		l: usize,
		gu_w: &[u8],
		gu_s: &[u16],
		d_w: &[u8],
		d_s: &[u16],
		d_pre: &[u16],
	) -> Result<(), String> {
		if l >= self.layers.len() {
			return Err(format!("experts: layer {l} of {}", self.layers.len()));
		}
		let u8s =
			|h: &[u16]| unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2) };
		let (gw, gwo) = self.import(gu_w)?;
		let (gs, gso) = self.import(u8s(gu_s))?;
		let (dw, dwo) = self.import(d_w)?;
		let (dsb, dso) = self.import(u8s(d_s))?;
		let (dp, dpo) = self.import(u8s(d_pre))?;
		let d = &self.dev.device;
		let sets = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: NK as u32,
				p_set_layouts: [self.dev.aux_layout; NK].as_ptr(),
				..Default::default()
			})
		}
		.map_err(|e| format!("experts: descriptor sets: {e}"))?;
		let b = |buf| vk::DescriptorBufferInfo {
			buffer: buf,
			offset: 0,
			range: vk::WHOLE_SIZE,
		};
		let [x, sl, g, hq, y, xq] = [0, 1, 2, 3, 4, 5].map(|i| self.bufs[i].buf);
		// per kernel: storage bindings 0.. in order, then the uniform
		let binds: [Vec<vk::Buffer>; NK] = std::array::from_fn(|k| match k {
			0 => vec![x, xq],
			K_ACT => vec![dp, sl, g, hq],
			k if k < K_ACT => vec![gw, gs, xq, sl, g],
			_ => vec![dw, dsb, hq, sl, y],
		});
		let infos: Vec<Vec<vk::DescriptorBufferInfo>> = binds
			.iter()
			.enumerate()
			.map(|(k, v)| {
				let mut i: Vec<_> = v.iter().map(|&bf| b(bf)).collect();
				i.push(vk::DescriptorBufferInfo {
					buffer: self.params.0,
					offset: (k * 256) as u64,
					range: 256,
				});
				i
			})
			.collect();
		let mut writes = Vec::new();
		for (k, inf) in infos.iter().enumerate() {
			for (i, info) in inf.iter().enumerate() {
				let uni = i + 1 == inf.len();
				writes.push(vk::WriteDescriptorSet {
					dst_set: sets[k],
					dst_binding: if uni { 8 } else { i as u32 },
					descriptor_type: if uni {
						vk::DescriptorType::UNIFORM_BUFFER
					} else {
						vk::DescriptorType::STORAGE_BUFFER
					},
					descriptor_count: 1,
					p_buffer_info: info,
					..Default::default()
				});
			}
		}
		unsafe { d.update_descriptor_sets(&writes, &[]) };
		self.layers[l] = Some(Layer {
			sets: sets.try_into().unwrap(),
			ofs: [gwo, gso * 2, dpo * 2, dwo, dso * 2],
		});
		Ok(())
	}

	/// Layer `l`'s experts on the MoE input rows `x` (`[tokens][2560]` f16,
	/// as bytes) for `slots` (expert, its tokens' rows of `x`, at most
	/// [`EXP_SLOT`]), left in flight until [`Experts::end`]. Slot rows are
	/// numbered in slot order, then token order. Only the rows of `x` the
	/// slots use are copied to the device (compacted in token order).
	pub fn begin(&mut self, l: usize, x: &[u8], slots: &[(usize, &[u32])]) -> Result<(), String> {
		if let Some((cmd, _)) = self.pending.take() {
			self.dev
				.wait_submitted(cmd)
				.map_err(|e| format!("experts: {e}"))?;
		}
		let ly = self
			.layers
			.get(l)
			.and_then(|v| v.as_ref())
			.ok_or(format!("experts: layer {l} not added"))?;
		let (sets, ofs) = (ly.sets, ly.ofs);
		let nin = x.len() / (W * 2);
		// output rows in the caller's slot order; the table holds the slots
		// by size class, widest first (each class's kernels from its base)
		let mut rowof = Vec::with_capacity(slots.len());
		let mut row = 0;
		// each used token's row in the device's copy (u32::MAX: unused)
		let mut map = vec![u32::MAX; nin];
		for (i, &(_, toks)) in slots.iter().enumerate() {
			if toks.is_empty() || toks.len() > EXP_SLOT || toks.iter().any(|&k| k as usize >= nin) {
				return Err(format!("experts: slot {i}: bad tokens"));
			}
			for &k in toks {
				map[k as usize] = 0;
			}
			rowof.push(row);
			row += toks.len();
		}
		let mut ntok = 0;
		for m in map.iter_mut().filter(|m| **m != u32::MAX) {
			*m = ntok as u32;
			ntok += 1;
		}
		let rows: usize = slots.iter().map(|s| s.1.len()).sum();
		if ntok > self.max_tok || rows > self.max_rows || slots.is_empty() {
			return Err(format!(
				"experts: {ntok} tokens, {rows} rows, {} slots: bad sizes",
				slots.len()
			));
		}
		let mut order: Vec<usize> = (0..slots.len()).collect();
		order.sort_by_key(|&i| class(slots[i].1.len()));
		// per class its slot count and first slot
		let mut cnt = [0u32; NT];
		for s in slots {
			cnt[class(s.1.len())] += 1;
		}
		let base: [u32; NT] = std::array::from_fn(|c| cnt[..c].iter().sum());
		// SAFETY: the mappings hold max_tok tokens, max_rows rows
		unsafe {
			let rb = W * 2;
			for (t, &m) in map.iter().enumerate().filter(|(_, m)| **m != u32::MAX) {
				ptr::copy_nonoverlapping(
					x.as_ptr().add(t * rb),
					self.bufs[0].map.add(m as usize * rb),
					rb,
				);
			}
			let t =
				std::slice::from_raw_parts_mut(self.bufs[1].map as *mut u32, self.bufs[1].len / 4);
			let mut lo = 4 * slots.len();
			for (j, &i) in order.iter().enumerate() {
				let (e, toks) = slots[i];
				t[4 * j..4 * j + 4].copy_from_slice(&[
					e as u32,
					toks.len() as u32,
					lo as u32,
					rowof[i] as u32,
				]);
				for (o, &k) in t[lo..lo + toks.len()].iter_mut().zip(toks) {
					*o = map[k as usize];
				}
				lo += toks.len();
			}
		}
		let d = &self.dev.device;
		// per kernel (wofs, sofs, sbase)
		let mut pb = [0u32; NK * 64];
		for k in 0..NK {
			let v = match k {
				0 => [(ntok * W / 8) as u32, 0, 0],
				K_ACT => [0, ofs[2], 0],
				k if k < K_ACT => [ofs[0], ofs[1], base[k - k_gu(0)]],
				k => [ofs[3], ofs[4], base[k - k_down(0)]],
			};
			pb[k * 64..k * 64 + 3].copy_from_slice(&v);
		}
		let pb: Vec<u8> = pb.iter().flat_map(|x| x.to_le_bytes()).collect();
		let (layout, params, pl) = (self.dev.aux_pipeline_layout, self.params.0, self.pl);
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
		let ns = slots.len() as u32;
		let qp = self.ts.map(|t| t.0);
		let cmd = self
			.dev
			.submit_nowait(|cmd| unsafe {
				if let Some(qp) = qp {
					d.cmd_reset_query_pool(cmd, qp, 0, 2);
					d.cmd_write_timestamp(cmd, P::TOP_OF_PIPE, qp, 0);
				}
				d.cmd_update_buffer(cmd, params, 0, &pb);
				barrier(
					cmd,
					P::TRANSFER | P::HOST,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE | A::HOST_WRITE,
					A::UNIFORM_READ | A::SHADER_READ,
				);
				// the sizes of each GEMM need no barrier between them
				let grid: [(u32, u32); NK] = std::array::from_fn(|k| match k {
					0 => ((ntok * W / 8).div_ceil(64) as u32, 1),
					K_ACT => (EXP_SLOT as u32, ns),
					k if k < K_ACT => ((2 * FE / RW) as u32, cnt[k - k_gu(0)]),
					k => ((W / RW) as u32, cnt[k - k_down(0)]),
				});
				let stage: [u32; NK] = std::array::from_fn(|k| match k {
					0 => 0,
					K_ACT => 2,
					k if k < K_ACT => 1,
					_ => 3,
				});
				let mut last = 0;
				for k in 0..NK {
					if grid[k].0 * grid[k].1 == 0 {
						continue;
					}
					if stage[k] != last {
						last = stage[k];
						barrier(
							cmd,
							P::COMPUTE_SHADER,
							P::COMPUTE_SHADER,
							A::SHADER_WRITE,
							A::SHADER_READ,
						);
					}
					d.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						layout,
						0,
						&[sets[k]],
						&[],
					);
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[k]);
					d.cmd_dispatch(cmd, grid[k].0, grid[k].1, 1);
				}
				if let Some(qp) = qp {
					d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, 1);
				}
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("experts: {e}"))?;
		self.pending = Some((cmd, rows));
		Ok(())
	}

	/// Whether [`Experts::begin`]'s submission is done (without waiting).
	pub fn done(&self) -> bool {
		self.pending.is_none()
			|| unsafe { self.dev.device.get_fence_status(self.dev.fence) }.unwrap_or(true)
	}

	/// The device time of the submission the last [`Experts::end`] waited
	/// for, ms (GPU timestamps: queueing and the host's part excluded).
	pub fn last_ms(&self) -> Option<f64> {
		self.last_ms
	}

	/// Wait for [`Experts::begin`]'s submission: the slot rows' outputs
	/// `[rows][2560]` f16, valid until the next `begin`.
	pub fn end(&mut self) -> Result<&[u16], String> {
		let (cmd, rows) = self.pending.take().ok_or("experts: nothing in flight")?;
		self.dev
			.wait_submitted(cmd)
			.map_err(|e| format!("experts: {e}"))?;
		self.last_ms = self.ts.and_then(|(qp, period)| {
			let mut t = [0u64; 2];
			// SAFETY: both queries were written by the submission waited for
			unsafe {
				self.dev.device.get_query_pool_results(
					qp,
					0,
					&mut t,
					vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
				)
			}
			.ok()
			.map(|()| t[1].saturating_sub(t[0]) as f64 * period * 1e-6)
		});
		// SAFETY: the mapping holds max_rows rows
		Ok(unsafe { std::slice::from_raw_parts(self.bufs[4].map as *const u16, rows * W) })
	}
}

impl Drop for Experts {
	fn drop(&mut self) {
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			if let Some((qp, _)) = self.ts {
				d.destroy_query_pool(qp, None);
			}
			for b in &self.bufs {
				d.unmap_memory(b.mem);
				d.destroy_buffer(b.buf, None);
				d.free_memory(b.mem, None);
			}
			for &(b, m) in &self.imports {
				hostimp::free_import(d, b, m);
			}
			for (&p, &m) in self.pl.iter().zip(&self.sm) {
				d.destroy_pipeline(p, None);
				d.destroy_shader_module(m, None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_buffer(self.params.0, None);
			d.free_memory(self.params.1, None);
		}
	}
}
