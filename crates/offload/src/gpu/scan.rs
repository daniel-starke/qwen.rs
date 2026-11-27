//! The NPU prefill's delta-net recurrence on a Vulkan device (the iGPU).
//!
//! WHY
//! ---
//! The NPU spends ~10.7 ms of a ~37 ms delta layer (256 positions) in the
//! chunked recurrence with its engines mostly idle: a long chain of small
//! dependent ops. `delta_prefill_npu.glsl` runs the same scan on the iGPU in
//! ~3.1 ms, with the state in registers. The layer is then two NPU graphs
//! (`build_delta.py --cut pre` / `--cut post`) with this scan in between.
//!
//! The tensors stay where the NPU put them: its host buffers (4 KiB-aligned
//! Level Zero host allocations, alive as long as the engine) are imported as
//! Vulkan buffers once (`VK_EXT_external_memory_host`), so a scan reads `qn`
//! / `kn` / `v` / `gb` and writes `o` in place. Only the state is the GPU's:
//! one f32 `[48][128][128]` buffer per delta layer, kept across chunks.

use super::*;

const S: usize = 128;
const HV: usize = 48;
const LANES: usize = 8;
const ST_BYTES: u64 = (HV * S * S * 4) as u64;

struct Imported {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	/// `DeltaScan::tick` of its last use
	last: u64,
}

/// A conv queued for the next scan ([`DeltaScan::conv_next`]): its layer,
/// positions and the (address, length) of `cpast`, `qkv`, `qn`, `kn`, `v`.
#[derive(Clone, Copy)]
struct ConvJob {
	di: usize,
	n: usize,
	r: [(usize, usize); 5],
}

impl ConvJob {
	fn key(&self) -> [usize; 6] {
		let r = &self.r;
		[self.di, r[0].0, r[1].0, r[2].0, r[3].0, r[4].0]
	}
}

/// See the module docs.
pub struct DeltaScan {
	dev: GpuDevice,
	sm: vk::ShaderModule,
	pipeline: vk::Pipeline,
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	states: Vec<(vk::Buffer, vk::DeviceMemory)>,
	/// by (base address, length)
	imports: std::collections::BTreeMap<(usize, usize), Imported>,
	/// most imports kept: each is a device memory allocation, and a far
	/// context imports two per K / V block (16 x 257 of them at 262144
	/// positions) against a driver limit that can be 4096
	max_imports: usize,
	tick: u64,
	sets: HashMap<[usize; 6], vk::DescriptorSet>,
	/// the gated norm (`gn_npu.glsl`, [`Self::scan_begin_gn`]): its
	/// pipeline, per delta layer its factors `[2][6144]` f32 (made by
	/// [`Self::set_gnorm`]), its descriptor sets by (layer, `o`, `z`)
	gn_sm: vk::ShaderModule,
	gn_pl: vk::Pipeline,
	gn_w: Vec<Option<(vk::Buffer, vk::DeviceMemory)>>,
	gsets: HashMap<[usize; 3], vk::DescriptorSet>,
	/// the short conv, Swish and q / k l2 norms ahead of the recurrence
	/// (`conv_npu.glsl`, [`Self::conv_next`]): its pipeline, per delta
	/// layer its taps and row scale (`[10240][4]` then `[10240]` f32, made
	/// by [`Self::set_conv`]), its descriptor sets by (layer, `cpast`,
	/// `qkv`, `qn`, `kn`, `v`), the set the next scan submission runs first
	cv_sm: vk::ShaderModule,
	cv_pl: vk::Pipeline,
	cv_w: Vec<Option<(vk::Buffer, vk::DeviceMemory)>>,
	csets: HashMap<[usize; 6], vk::DescriptorSet>,
	cv_next: Option<ConvJob>,
	align: usize,
	/// far attention (`attn_far_qks_npu.glsl` + `attn_far_pv_npu.glsl`, or
	/// with `QWEN_NPU_FAR_FUSED=0` the older `attn_far_qk_npu.glsl` +
	/// `attn_far_npu.glsl`): the two passes' pipelines, the scores between
	/// them (and the fused form's per-row factors after them), the host
	/// block behind the running `o` / `ml` (imported), the submission in
	/// flight
	far_sm: [vk::ShaderModule; 2],
	far_pl: [vk::Pipeline; 2],
	far_sc: (vk::Buffer, vk::DeviceMemory),
	far_mem: *mut u8,
	far_cmd: Option<vk::CommandBuffer>,
	/// every layer's state in host memory, mapped (`set_states` /
	/// `get_states`; made on first use)
	bulk: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
	/// `QWEN_NPU_SCAN_TS=1`: a timestamp pool (start, end, scan start, scan
	/// end) and the period; each scan prints its GPU time (and that of the
	/// conv before and the gated norm after) next to the host's wait for it
	ts: Option<(vk::QueryPool, f64)>,
	/// [`Self::scan_begin`]'s submission in flight (its positions and when
	/// it was submitted)
	scan_cmd: Option<(vk::CommandBuffer, usize, std::time::Instant)>,
}

/// Rows of the far attention state (24 heads x 256 positions) and its bytes:
/// `o` `[FAR_ROWS][256]` f32, then `ml` `[FAR_ROWS][2]` f32.
const FAR_ROWS: usize = 24 * 256;
const FAR_O: usize = FAR_ROWS * 256 * 4;
const FAR_BYTES: usize = FAR_O + FAR_ROWS * 8;
/// Largest K / V block (keys) of the far attention's score buffer.
const FAR_CAP: usize = 1024;
/// Channels of a delta layer's conv (`qkv`: q 2048, k 2048, v 6144).
const CONV_C: usize = 10240;
/// The score buffer: `[FAR_ROWS][FAR_CAP]` f32 scores, then the fused
/// form's `[FAR_ROWS]` f32 factors exp(m_old - m_new) from vec4 `FAR_F4` on.
const FAR_F4: usize = FAR_ROWS * FAR_CAP / 4;
const FAR_SC_BYTES: usize = (FAR_ROWS * FAR_CAP + FAR_ROWS) * 4;
/// Descriptor sets the pool holds. The cache of sets (one per delta layer x
/// buffer set, one per far block x query buffer) can outgrow it at long
/// contexts: 16 attention layers x 256 blocks per query buffer, and every
/// prefill set has its own; the pool is then reset and the sets made again.
const MAX_SETS: usize = 10240;

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for DeltaScan {}

impl DeltaScan {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU), with
	/// `layers` delta-layer states (zero).
	pub fn new(pick: Option<&str>, layers: usize) -> Result<DeltaScan, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let align = dev
			.host_import_align()
			.ok_or("the Vulkan device has no VK_EXT_external_memory_host")?;
		let sm = create_sm(
			&dev.device,
			&load_spv(quant::packed!("delta_prefill_npu.spv")),
		)?;
		let main = CString::new("main").unwrap();
		let pipeline = compute_pipeline(&dev.device, dev.aux_pipeline_layout, sm, &main)?;
		let gn_sm = create_sm(&dev.device, &load_spv(quant::packed!("gn_npu.spv")))?;
		let gn_pl = compute_pipeline(&dev.device, dev.aux_pipeline_layout, gn_sm, &main)?;
		let cv_sm = create_sm(&dev.device, &load_spv(quant::packed!("conv_npu.spv")))?;
		let cv_pl = compute_pipeline(&dev.device, dev.aux_pipeline_layout, cv_sm, &main)?;
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 6 * MAX_SETS as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: MAX_SETS as u32,
			},
		];
		let pool = unsafe {
			dev.device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: MAX_SETS as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("scan descriptor pool: {e}"))?;
		let params = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let usage = vk::BufferUsageFlags::STORAGE_BUFFER
			| vk::BufferUsageFlags::TRANSFER_SRC
			| vk::BufferUsageFlags::TRANSFER_DST;
		let states: Vec<_> = (0..layers)
			.map(|_| dev.create_gpu_buffer(ST_BYTES, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL))
			.collect();
		// `QWEN_NPU_FAR_FUSED=0`: the older pair, softmax in pass 2 (the A/B)
		let fused = !matches!(
			std::env::var("QWEN_NPU_FAR_FUSED").as_deref(),
			Ok("0" | "off" | "no" | "false")
		);
		let far_spv: [&[u8]; 2] = if fused {
			[
				quant::packed!("attn_far_qks_npu.spv"),
				quant::packed!("attn_far_pv_npu.spv"),
			]
		} else {
			[
				quant::packed!("attn_far_qk_npu.spv"),
				quant::packed!("attn_far_npu.spv"),
			]
		};
		let far_sm = [
			create_sm(&dev.device, &load_spv(far_spv[0]))?,
			create_sm(&dev.device, &load_spv(far_spv[1]))?,
		];
		let far_pl = [
			compute_pipeline(&dev.device, dev.aux_pipeline_layout, far_sm[0], &main)?,
			compute_pipeline(&dev.device, dev.aux_pipeline_layout, far_sm[1], &main)?,
		];
		let far_sc = dev.create_gpu_buffer(
			FAR_SC_BYTES as u64,
			vk::BufferUsageFlags::STORAGE_BUFFER,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		// SAFETY: a non-zero size and a power-of-two alignment
		let far_mem = unsafe {
			std::alloc::alloc_zeroed(
				std::alloc::Layout::from_size_align(FAR_BYTES, align.max(4096)).unwrap(),
			)
		};
		if far_mem.is_null() {
			return Err("scan: far attention state: out of memory".into());
		}
		// the driver's limit less the allocations of our own and of the
		// device (`QWEN_NPU_SCAN_IMPORTS=N`: at most N, a test of eviction)
		let limit = unsafe { dev.instance.get_physical_device_properties(dev.physical) }
			.limits
			.max_memory_allocation_count as usize;
		let max_imports = std::env::var("QWEN_NPU_SCAN_IMPORTS")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.unwrap_or(usize::MAX)
			.min(limit.saturating_sub(256 + layers))
			.max(16);
		let ts = std::env::var_os("QWEN_NPU_SCAN_TS")
			.and_then(|_| unsafe {
				dev.device
					.create_query_pool(
						&vk::QueryPoolCreateInfo {
							query_type: vk::QueryType::TIMESTAMP,
							query_count: 4,
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
		let mut sc = DeltaScan {
			ts,
			scan_cmd: None,
			dev,
			sm,
			pipeline,
			pool,
			params,
			states,
			imports: Default::default(),
			max_imports,
			tick: 0,
			sets: HashMap::new(),
			gn_sm,
			gn_pl,
			gn_w: (0..layers).map(|_| None).collect(),
			gsets: HashMap::new(),
			cv_sm,
			cv_pl,
			cv_w: (0..layers).map(|_| None).collect(),
			csets: HashMap::new(),
			cv_next: None,
			align,
			far_sm,
			far_pl,
			far_sc,
			far_mem,
			far_cmd: None,
			bulk: None,
		};
		for i in 0..layers {
			sc.set_state(i, None)?;
		}
		Ok(sc)
	}

	/// Delta layer `di`'s state: `None` = zero, else `[48][128][128]` f32
	/// (row = key dim), true scale.
	pub fn set_state(&mut self, di: usize, st: Option<&[f32]>) -> Result<(), String> {
		self.scan_end()?;
		let (buf, _) = self.states[di];
		match st {
			None => self
				.dev
				.submit_and_wait(|cmd| unsafe {
					self.dev.device.cmd_fill_buffer(cmd, buf, 0, ST_BYTES, 0)
				})
				.map_err(|e| format!("scan: zero state: {e}")),
			Some(v) => {
				assert_eq!(v.len() * 4, ST_BYTES as usize);
				let (b, m, sb, sm, p) = self
					.dev
					.create_staged_buffer(ST_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER);
				unsafe {
					ptr::copy_nonoverlapping(
						v.as_ptr() as *const u8,
						p.0 as *mut u8,
						ST_BYTES as usize,
					)
				};
				let r = self.dev.submit_and_wait(|cmd| unsafe {
					self.dev.device.cmd_copy_buffer(
						cmd,
						sb,
						buf,
						&[vk::BufferCopy {
							src_offset: 0,
							dst_offset: 0,
							size: ST_BYTES,
						}],
					)
				});
				unsafe {
					self.dev.device.unmap_memory(sm);
					for (x, y) in [(b, m), (sb, sm)] {
						self.dev.device.destroy_buffer(x, None);
						self.dev.device.free_memory(y, None);
					}
				}
				r.map_err(|e| format!("scan: upload state: {e}"))
			}
		}
	}

	/// The host buffer of [`Self::set_states`] / [`Self::get_states`].
	fn bulk(&mut self) -> Result<(vk::Buffer, *mut u8), String> {
		if self.bulk.is_none() {
			let size = ST_BYTES * self.states.len() as u64;
			let host =
				vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
			let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
			let (b, m) = self
				.dev
				.try_create_gpu_buffer(size, usage, host | vk::MemoryPropertyFlags::HOST_CACHED)
				.or_else(|| self.dev.try_create_gpu_buffer(size, usage, host))
				.ok_or("scan: no host buffer for the states")?;
			let p = self.dev.map(m, size as usize) as *mut u8;
			self.bulk = Some((b, m, p));
		}
		let (b, _, p) = self.bulk.unwrap();
		Ok((b, p))
	}

	/// Layers `0..n`'s states in one submission (a prefill's hand-over
	/// moves all 48: per layer a staging buffer and a wait each took ~100
	/// ms in all): `fill(di, dst)` writes layer `di`'s state as
	/// [`Self::set_state`] takes it, or returns false for a zero state.
	pub fn set_states(
		&mut self,
		n: usize,
		fill: &mut dyn FnMut(usize, &mut [f32]) -> bool,
	) -> Result<(), String> {
		self.scan_end()?;
		assert!(n <= self.states.len());
		let (hb, p) = self.bulk()?;
		let live: Vec<bool> = (0..n)
			.map(|di| {
				// SAFETY: the mapped buffer holds `states.len()` states; no
				// submission uses it now
				let dst = unsafe {
					std::slice::from_raw_parts_mut(
						p.add(di * ST_BYTES as usize) as *mut f32,
						HV * S * S,
					)
				};
				fill(di, dst)
			})
			.collect();
		let (d, states) = (&self.dev.device, &self.states);
		self.dev
			.submit_and_wait(|cmd| unsafe {
				for (di, &(buf, _)) in states.iter().enumerate().take(n) {
					if live[di] {
						let r = vk::BufferCopy {
							src_offset: di as u64 * ST_BYTES,
							dst_offset: 0,
							size: ST_BYTES,
						};
						d.cmd_copy_buffer(cmd, hb, buf, &[r]);
					} else {
						d.cmd_fill_buffer(cmd, buf, 0, ST_BYTES, 0);
					}
				}
			})
			.map_err(|e| format!("scan: upload states: {e}"))
	}

	/// Layers `0..n`'s states in one submission, each handed to
	/// `take(di, state)` (as [`Self::state`] returns it).
	pub fn get_states(
		&mut self,
		n: usize,
		take: &mut dyn FnMut(usize, &[f32]),
	) -> Result<(), String> {
		self.scan_end()?;
		assert!(n <= self.states.len());
		let (hb, p) = self.bulk()?;
		let (d, states) = (&self.dev.device, &self.states);
		self.dev
			.submit_and_wait(|cmd| unsafe {
				for (di, &(buf, _)) in states.iter().enumerate().take(n) {
					let r = vk::BufferCopy {
						src_offset: 0,
						dst_offset: di as u64 * ST_BYTES,
						size: ST_BYTES,
					};
					d.cmd_copy_buffer(cmd, buf, hb, &[r]);
				}
			})
			.map_err(|e| format!("scan: read states: {e}"))?;
		for di in 0..n {
			// SAFETY: as in `set_states`; the copies are done
			let src = unsafe {
				std::slice::from_raw_parts(p.add(di * ST_BYTES as usize) as *const f32, HV * S * S)
			};
			take(di, src);
		}
		Ok(())
	}

	/// Delta layer `di`'s state, as [`Self::set_state`] takes it.
	pub fn state(&mut self, di: usize) -> Result<Vec<f32>, String> {
		self.scan_end()?;
		let (buf, _) = self.states[di];
		let (rb, rbm) = self.dev.create_readback_buffer(ST_BYTES);
		let r = self.dev.submit_and_wait(|cmd| unsafe {
			self.dev.device.cmd_copy_buffer(
				cmd,
				buf,
				rb,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: ST_BYTES,
				}],
			)
		});
		let mut out = vec![0.0f32; HV * S * S];
		if r.is_ok() {
			let p = self.dev.map(rbm, ST_BYTES as usize);
			unsafe {
				ptr::copy_nonoverlapping(p as *const f32, out.as_mut_ptr(), out.len());
				self.dev.device.unmap_memory(rbm);
			}
		}
		unsafe {
			self.dev.device.destroy_buffer(rb, None);
			self.dev.device.free_memory(rbm, None);
		}
		r.map_err(|e| format!("scan: read state: {e}"))?;
		Ok(out)
	}

	/// The recurrence of delta layer `di` over the first `n` positions of
	/// the NPU's `qn` / `kn` [n][2048], `v` [n][6144], `gb` [n][96] (f16);
	/// writes `o` [n][6144] f16 x `oscale`. All five must be host memory that
	/// outlives this object at an unchanging address (they are imported once).
	#[allow(clippy::too_many_arguments)]
	pub fn scan(
		&mut self,
		di: usize,
		n: usize,
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		o: &mut [u8],
		oscale: f32,
	) -> Result<(), String> {
		self.scan_begin(di, n, qn, kn, v, gb, o, oscale)?;
		self.scan_end()
	}

	/// [`Self::scan`] submitted without waiting: `o` is written once
	/// [`Self::scan_end`] returns, and nothing else may run on this device
	/// before (each other call finishes it first).
	#[allow(clippy::too_many_arguments)]
	pub fn scan_begin(
		&mut self,
		di: usize,
		n: usize,
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		o: &mut [u8],
		oscale: f32,
	) -> Result<(), String> {
		self.scan_end()?;
		if n == 0 {
			self.cv_next = None;
			return Ok(());
		}
		if qn.len() < n * 4096
			|| kn.len() < n * 4096
			|| v.len() < n * 12288
			|| gb.len() < n * 192
			|| o.len() < n * 12288
		{
			return Err(format!("scan: buffers too small for {n} positions"));
		}
		let key = [
			di,
			qn.as_ptr() as usize,
			kn.as_ptr() as usize,
			v.as_ptr() as usize,
			gb.as_ptr() as usize,
			o.as_ptr() as usize,
		];
		let ranges = [qn, kn, v, gb, &*o].map(|s| (s.as_ptr() as usize, s.len()));
		let cj = self.conv_room(n, &ranges, usize::from(!self.sets.contains_key(&key)))?;
		let mut bi = [(vk::Buffer::null(), 0u64, 0u64); 5];
		for (k, &(p, len)) in ranges.iter().enumerate() {
			let (b, off) = self.import(p, len)?;
			bi[k] = (b, off, len as u64);
		}
		let cv = cj.map(|j| self.conv_set(&j)).transpose()?;
		let set = match self.sets.get(&key) {
			Some(&s) => s,
			None => {
				let s = self.make_set(self.states[di].0, &bi)?;
				self.sets.insert(key, s);
				s
			}
		};
		let cmd = self.submit_run(set, n, oscale, None, cv)?;
		self.scan_cmd = Some((cmd, n, std::time::Instant::now()));
		Ok(())
	}

	/// [`Self::room`] for a scan over `n` positions with `ranges` and `sets`
	/// new descriptor sets, and for the conv [`Self::conv_next`] queued
	/// for it (taken: returned).
	fn conv_room(
		&mut self,
		n: usize,
		ranges: &[(usize, usize)],
		sets: usize,
	) -> Result<Option<ConvJob>, String> {
		let cj = self.cv_next.take();
		let Some(j) = cj else {
			self.room(ranges, sets)?;
			return Ok(None);
		};
		if j.n != n {
			return Err(format!(
				"scan: conv queued for {} positions, scan over {n}",
				j.n
			));
		}
		let mut all = ranges.to_vec();
		all.extend_from_slice(&j.r);
		self.room(&all, sets + usize::from(!self.csets.contains_key(&j.key())))?;
		Ok(Some(j))
	}

	/// The descriptor set of conv job `j` (its ranges in room: [`Self::conv_room`]).
	fn conv_set(&mut self, j: &ConvJob) -> Result<vk::DescriptorSet, String> {
		let Some((wbuf, _)) = self.cv_w.get(j.di).copied().flatten() else {
			return Err(format!("scan: no conv taps for layer {}", j.di));
		};
		let mut bi = [(vk::Buffer::null(), 0u64, 0u64); 5];
		for (k, &(p, len)) in j.r.iter().enumerate() {
			let (b, off) = self.import(p, len)?;
			bi[k] = (b, off, len as u64);
		}
		let key = j.key();
		if let Some(&s) = self.csets.get(&key) {
			return Ok(s);
		}
		let w = (wbuf, 0, (5 * CONV_C * 4) as u64);
		let s = self.make_set_raw(&[w, bi[0], bi[1], bi[2], bi[3], bi[4]])?;
		self.csets.insert(key, s);
		Ok(s)
	}

	/// Delta layer `di`'s gated-norm factors `w` `[2][6144]` (see
	/// [`Self::scan_begin_gn`]).
	pub fn set_gnorm(&mut self, di: usize, w: &[f32]) -> Result<(), String> {
		self.scan_end()?;
		if w.len() != 2 * HV * S {
			return Err(format!(
				"scan: gated norm factors: {} values, wants {}",
				w.len(),
				2 * HV * S
			));
		}
		let mut gw = std::mem::take(&mut self.gn_w);
		let r = self.upload(&mut gw[di], w);
		self.gn_w = gw;
		r.map_err(|e| format!("scan: upload gated norm factors: {e}"))
	}

	/// Delta layer `di`'s conv for [`Self::conv_next`]: `w` = the taps
	/// `[10240][4]` (the checkpoint's), then `qkv`'s row scale `[10240]`,
	/// f32 (the conv's input is `qkv * qs`).
	pub fn set_conv(&mut self, di: usize, w: &[f32]) -> Result<(), String> {
		self.scan_end()?;
		if w.len() != 5 * CONV_C {
			return Err(format!(
				"scan: conv taps: {} values, wants {}",
				w.len(),
				5 * CONV_C
			));
		}
		let mut cw = std::mem::take(&mut self.cv_w);
		let r = self.upload(&mut cw[di], w);
		self.cv_w = cw;
		r.map_err(|e| format!("scan: upload conv taps: {e}"))
	}

	/// Queue delta layer `di`'s conv, Swish and q / k l2 norms
	/// (`conv_npu.glsl`, taps from [`Self::set_conv`]) of `n` positions
	/// ahead of the next scan submission ([`Self::scan_begin`] /
	/// [`Self::scan_begin_gn`] over the same `n`): from `cpast` [3][10240]
	/// and the raw `qkv` [n][10240] (f16) into `qn`, `kn` [n][2048] and `v`
	/// [n][6144] f16, the buffers that scan then reads. All host memory as
	/// the scan's.
	#[allow(clippy::too_many_arguments)]
	pub fn conv_next(
		&mut self,
		di: usize,
		n: usize,
		cpast: &[u8],
		qkv: &[u8],
		qn: &mut [u8],
		kn: &mut [u8],
		v: &mut [u8],
	) -> Result<(), String> {
		self.scan_end()?;
		if self.cv_w.get(di).is_none_or(|w| w.is_none()) {
			return Err(format!("scan: no conv taps for layer {di}"));
		}
		if cpast.len() < 3 * CONV_C * 2
			|| qkv.len() < n * CONV_C * 2
			|| qn.len() < n * 4096
			|| kn.len() < n * 4096
			|| v.len() < n * 12288
		{
			return Err(format!("scan: conv buffers too small for {n} positions"));
		}
		let r = [cpast, qkv, &*qn, &*kn, &*v].map(|s| (s.as_ptr() as usize, s.len()));
		self.cv_next = Some(ConvJob { di, n, r });
		Ok(())
	}

	/// `w` into the device buffer `slot` (made on first use).
	fn upload(
		&self,
		slot: &mut Option<(vk::Buffer, vk::DeviceMemory)>,
		w: &[f32],
	) -> Result<(), String> {
		let bytes = (w.len() * 4) as u64;
		if slot.is_none() {
			*slot = Some(self.dev.create_gpu_buffer(
				bytes,
				vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			));
		}
		let (buf, _) = slot.unwrap();
		let (b, m, sb, sm, p) = self
			.dev
			.create_staged_buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER);
		// SAFETY: the staging buffer is mapped and `bytes` long
		unsafe {
			ptr::copy_nonoverlapping(w.as_ptr() as *const u8, p.0 as *mut u8, bytes as usize)
		};
		let r = self.dev.submit_and_wait(|cmd| unsafe {
			self.dev.device.cmd_copy_buffer(
				cmd,
				sb,
				buf,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: bytes,
				}],
			)
		});
		unsafe {
			self.dev.device.unmap_memory(sm);
			for (x, y) in [(b, m), (sb, sm)] {
				self.dev.device.destroy_buffer(x, None);
				self.dev.device.free_memory(y, None);
			}
		}
		r.map_err(|e| e.to_string())
	}

	/// [`Self::scan_begin`], then in the same submission the delta layer's
	/// gated norm in place of `o`: per (position, head)
	/// `o / sqrt(mean(o^2) + 1e-6 oscale^2) * w0 * sigmoid(z * w1)`, rotated
	/// in Hadamard blocks of 256 (`w` from [`Self::set_gnorm`]; `z` [n][6144] f16, host memory
	/// as the other buffers).
	#[allow(clippy::too_many_arguments)]
	pub fn scan_begin_gn(
		&mut self,
		di: usize,
		n: usize,
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		z: &[u8],
		o: &mut [u8],
		oscale: f32,
	) -> Result<(), String> {
		self.scan_end()?;
		if n == 0 {
			self.cv_next = None;
			return Ok(());
		}
		let Some((wbuf, _)) = self.gn_w.get(di).copied().flatten() else {
			return Err(format!("scan: no gated norm factors for layer {di}"));
		};
		if qn.len() < n * 4096
			|| kn.len() < n * 4096
			|| v.len() < n * 12288
			|| gb.len() < n * 192
			|| o.len() < n * 12288
			|| z.len() < n * 12288
		{
			return Err(format!("scan: buffers too small for {n} positions"));
		}
		let key = [
			di,
			qn.as_ptr() as usize,
			kn.as_ptr() as usize,
			v.as_ptr() as usize,
			gb.as_ptr() as usize,
			o.as_ptr() as usize,
		];
		let gkey = [di, o.as_ptr() as usize, z.as_ptr() as usize];
		let ranges = [qn, kn, v, gb, &*o, z].map(|s| (s.as_ptr() as usize, s.len()));
		let new = usize::from(!self.sets.contains_key(&key))
			+ usize::from(!self.gsets.contains_key(&gkey));
		let cj = self.conv_room(n, &ranges, new)?;
		let mut bi = [(vk::Buffer::null(), 0u64, 0u64); 6];
		for (k, &(p, len)) in ranges.iter().enumerate() {
			let (b, off) = self.import(p, len)?;
			bi[k] = (b, off, len as u64);
		}
		let cv = cj.map(|j| self.conv_set(&j)).transpose()?;
		let set = match self.sets.get(&key) {
			Some(&s) => s,
			None => {
				let s = self.make_set(self.states[di].0, &[bi[0], bi[1], bi[2], bi[3], bi[4]])?;
				self.sets.insert(key, s);
				s
			}
		};
		let gset = match self.gsets.get(&gkey) {
			Some(&s) => s,
			None => {
				// bindings 0 factors, 1 o, 2 z (3..5 unused)
				let w = (wbuf, 0, (2 * HV * S * 4) as u64);
				let s = self.make_set_raw(&[w, bi[4], bi[5], w, w, w])?;
				self.gsets.insert(gkey, s);
				s
			}
		};
		let cmd = self.submit_run(set, n, oscale, Some(gset), cv)?;
		self.scan_cmd = Some((cmd, n, std::time::Instant::now()));
		Ok(())
	}

	/// Wait for [`Self::scan_begin`]'s submission, if one is in flight.
	pub fn scan_end(&mut self) -> Result<(), String> {
		let Some((cmd, n, t0)) = self.scan_cmd.take() else {
			return Ok(());
		};
		self.dev
			.wait_submitted(cmd)
			.map_err(|e| format!("scan: {e}"))?;
		if let (Some((qp, period)), true) = (self.ts, n > 0) {
			let mut t = [0u64; 4];
			// SAFETY: all queries were written by the submission waited for
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
				let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period * 1e-6;
				eprintln!(
					"[scants] n {n}: gpu {:.2} ms (conv {:.2}, scan {:.2}, norm {:.2}), host {:.2} ms",
					ms(t[0], t[1]),
					ms(t[0], t[2]),
					ms(t[2], t[3]),
					ms(t[3], t[1]),
					t0.elapsed().as_secs_f64() * 1e3
				);
				eprintln!(
					"[igts] scan n {n}: {:.1} {:.1}",
					t[0] as f64 * period * 1e-3,
					t[1] as f64 * period * 1e-3
				);
			}
		}
		Ok(())
	}

	/// A scan over no positions on any set already made (the state is read
	/// and written back unchanged): keeps the device out of its idle power
	/// state while the caller's other work would leave it idle for long.
	pub fn poke(&mut self) -> Result<(), String> {
		if self.far_cmd.is_some() {
			return Ok(());
		}
		self.scan_end()?;
		match self.sets.values().next().copied() {
			Some(set) => {
				let cmd = self.submit_run(set, 0, 1.0, None, None)?;
				self.dev
					.wait_submitted(cmd)
					.map_err(|e| format!("scan: {e}"))
			}
			None => Ok(()),
		}
	}

	/// The scan over `n` positions on `set`, with `gn` the gated norm
	/// after it on that set ([`Self::scan_begin_gn`]; same parameters).
	fn submit_run(
		&mut self,
		set: vk::DescriptorSet,
		n: usize,
		oscale: f32,
		gn: Option<vk::DescriptorSet>,
		cv: Option<vk::DescriptorSet>,
	) -> Result<vk::CommandBuffer, String> {
		let mut pr = [0u32; 16];
		pr[0] = n as u32;
		pr[1] = oscale.to_bits();
		let pb: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
		let d = &self.dev.device;
		let (pipeline, layout, params, gn_pl) = (
			self.pipeline,
			self.dev.aux_pipeline_layout,
			self.params.0,
			self.gn_pl,
		);
		// a workgroup: 256 invocations, LANES per 4 value columns
		let groups = (HV * S * LANES / 1024) as u32;
		let qp = self.ts.map(|t| t.0);
		let cv_pl = self.cv_pl;
		self.dev
			.submit_nowait(|cmd| unsafe {
				if let Some(qp) = qp {
					d.cmd_reset_query_pool(cmd, qp, 0, 4);
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
				}
				d.cmd_update_buffer(cmd, params, 0, &pb);
				d.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&[vk::MemoryBarrier {
						src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
						dst_access_mask: vk::AccessFlags::UNIFORM_READ,
						..Default::default()
					}],
					&[],
					&[],
				);
				if let Some(cs) = cv {
					// the scan's q / k / v: one group per head row and position
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cv_pl);
					d.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						layout,
						0,
						&[cs],
						&[],
					);
					d.cmd_dispatch(cmd, (CONV_C / 128) as u32, n as u32, 1);
					d.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: vk::AccessFlags::SHADER_READ,
							..Default::default()
						}],
						&[],
						&[],
					);
				}
				if let Some(qp) = qp {
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, qp, 2);
				}
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					layout,
					0,
					&[set],
					&[],
				);
				d.cmd_dispatch(cmd, groups, 1, 1);
				if let Some(qp) = qp {
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, qp, 3);
				}
				if let Some(gs) = gn {
					// in place of the scan's `o`: one group per position and
					// block of 256 channels
					d.cmd_pipeline_barrier(
						cmd,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::PipelineStageFlags::COMPUTE_SHADER,
						vk::DependencyFlags::empty(),
						&[vk::MemoryBarrier {
							src_access_mask: vk::AccessFlags::SHADER_WRITE,
							dst_access_mask: vk::AccessFlags::SHADER_READ
								| vk::AccessFlags::SHADER_WRITE,
							..Default::default()
						}],
						&[],
						&[],
					);
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gn_pl);
					d.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						layout,
						0,
						&[gs],
						&[],
					);
					d.cmd_dispatch(cmd, (HV * S / 256) as u32, n as u32, 1);
				}
				// the NPU reads `o` from host memory after the fence
				d.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::COMPUTE_SHADER,
					vk::PipelineStageFlags::HOST,
					vk::DependencyFlags::empty(),
					&[vk::MemoryBarrier {
						src_access_mask: vk::AccessFlags::SHADER_WRITE,
						dst_access_mask: vk::AccessFlags::HOST_READ,
						..Default::default()
					}],
					&[],
					&[],
				);
				if let Some(qp) = qp {
					d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
				}
			})
			.map_err(|e| format!("scan: {e}"))
	}

	/// Far attention blocks `(k, v)` (`[cap][4][256]` f16, keys `[0, nk)`)
	/// for the queries `qg` `[4][6 * s][256]` f16, one dispatch per block,
	/// submitted without waiting; the running result is [`Self::far_out`].
	/// `fresh` starts a new result (else the blocks join the one there).
	/// Nothing else may run on the device until [`Self::far_poll`] returns
	/// true or [`Self::far_wait`] returns. All buffers as [`Self::scan`]'s.
	pub fn far_begin(
		&mut self,
		s: usize,
		nk: usize,
		qg: &[u8],
		blocks: &[(&[u8], &[u8])],
		fresh: bool,
	) -> Result<(), String> {
		self.scan_end()?;
		if self.far_cmd.is_some() {
			return Err("scan: far attention already in flight".into());
		}
		let cap = blocks.first().map_or(0, |b| b.0.len() / 2048);
		if !s.is_multiple_of(4)
			|| s * 24 > FAR_ROWS
			|| qg.len() < s * 24 * 512
			|| !cap.is_multiple_of(4)
			|| cap > FAR_CAP
		{
			return Err(format!(
				"scan: far attention over {s} positions, blocks of {cap} keys"
			));
		}
		if blocks
			.iter()
			.any(|b| b.0.len() != cap * 2048 || b.1.len() != cap * 2048)
		{
			return Err("scan: far attention blocks of different sizes".into());
		}
		if fresh {
			// SAFETY: our allocation; no submission is in flight
			let ml = unsafe {
				std::slice::from_raw_parts_mut(self.far_mem.add(FAR_O) as *mut [f32; 2], FAR_ROWS)
			};
			ml[..s * 24].fill([-1e30, 0.0]);
		}
		let key = |k: &[u8], v: &[u8]| {
			[
				usize::MAX,
				qg.as_ptr() as usize,
				k.as_ptr() as usize,
				v.as_ptr() as usize,
				0,
				0,
			]
		};
		// room for all of this call's sets and imports first: a reset inside
		// the loop would free the ones it collected already
		let missing = blocks
			.iter()
			.filter(|(k, v)| !self.sets.contains_key(&key(k, v)))
			.count();
		let mut ranges = vec![
			(self.far_mem as usize, FAR_BYTES),
			(qg.as_ptr() as usize, qg.len()),
		];
		for (k, v) in blocks {
			ranges.push((k.as_ptr() as usize, k.len()));
			ranges.push((v.as_ptr() as usize, v.len()));
		}
		self.room(&ranges, missing)?;
		let (stb, sto) = self.import(self.far_mem as usize, FAR_BYTES)?;
		let (qb, qo) = self.import(qg.as_ptr() as usize, qg.len())?;
		let mut sets = Vec::with_capacity(blocks.len());
		for (k, v) in blocks {
			let key = key(k, v);
			let set = match self.sets.get(&key) {
				Some(&x) => x,
				None => {
					let (kb, ko) = self.import(k.as_ptr() as usize, k.len())?;
					let (vb, vo) = self.import(v.as_ptr() as usize, v.len())?;
					// bindings 0 q, 1 k, 2 v, 3 o, 4 ml, 5 scores
					let ml = (stb, sto + FAR_O as u64, (FAR_ROWS * 8) as u64);
					let sc = (self.far_sc.0, 0, FAR_SC_BYTES as u64);
					let x = self.make_set_raw(&[
						(qb, qo, qg.len() as u64),
						(kb, ko, k.len() as u64),
						(vb, vo, v.len() as u64),
						(stb, sto, FAR_O as u64),
						ml,
						sc,
					])?;
					self.sets.insert(key, x);
					x
				}
			};
			sets.push(set);
		}
		let mut pr = [0u32; 16];
		pr[0] = s as u32;
		pr[1] = nk as u32;
		pr[2] = cap as u32;
		pr[3] = FAR_F4 as u32;
		let pb: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
		let d = &self.dev.device;
		let (pl, layout, params) = (self.far_pl, self.dev.aux_pipeline_layout, self.params.0);
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
		let cmd = self
			.dev
			.submit_nowait(|cmd| unsafe {
				d.cmd_update_buffer(cmd, params, 0, &pb);
				barrier(
					cmd,
					P::TRANSFER,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE,
					A::UNIFORM_READ,
				);
				for (i, set) in sets.iter().enumerate() {
					let rw = |cmd| {
						barrier(
							cmd,
							P::COMPUTE_SHADER,
							P::COMPUTE_SHADER,
							A::SHADER_WRITE,
							A::SHADER_READ | A::SHADER_WRITE,
						)
					};
					if i > 0 {
						rw(cmd);
					}
					d.cmd_bind_descriptor_sets(
						cmd,
						vk::PipelineBindPoint::COMPUTE,
						layout,
						0,
						&[*set],
						&[],
					);
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[0]);
					d.cmd_dispatch(cmd, 1, s as u32, 4);
					rw(cmd);
					d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[1]);
					d.cmd_dispatch(cmd, (s / 4) as u32, 4, 1);
				}
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("scan: far attention: {e}"))?;
		self.far_cmd = Some(cmd);
		Ok(())
	}

	/// True once no far attention submission is in flight (never blocks).
	pub fn far_poll(&mut self) -> Result<bool, String> {
		if self.far_cmd.is_none() {
			return Ok(true);
		}
		match unsafe { self.dev.device.get_fence_status(self.dev.fence) } {
			Ok(true) => self.far_wait().map(|_| true),
			Ok(false) => Ok(false),
			Err(e) => Err(format!("scan: far attention: {e}")),
		}
	}

	/// Waits for the far attention submission in flight, if any.
	pub fn far_wait(&mut self) -> Result<(), String> {
		match self.far_cmd.take() {
			Some(cmd) => self
				.dev
				.wait_submitted(cmd)
				.map_err(|e| format!("scan: far attention: {e}")),
			None => Ok(()),
		}
	}

	/// The far attention result over the blocks begun since the last
	/// `fresh`: `o` `[24 s][256]` (sum of exp(score - m) v, not normalized)
	/// and `[m, l]` per row (max score, sum of exp(score - m)). Read it only
	/// with no submission in flight.
	pub fn far_out(&self) -> (&[f32], &[[f32; 2]]) {
		// SAFETY: our allocation, FAR_BYTES long, aligned for f32
		unsafe {
			(
				std::slice::from_raw_parts(self.far_mem as *const f32, FAR_O / 4),
				std::slice::from_raw_parts(self.far_mem.add(FAR_O) as *const [f32; 2], FAR_ROWS),
			)
		}
	}

	/// A buffer over host bytes `[ptr, ptr + len)` and `ptr`'s offset in it,
	/// importing the covering pages on first use.
	fn import(&mut self, ptr: usize, len: usize) -> Result<(vk::Buffer, u64), String> {
		if let Some(k) = self.covering(ptr, len) {
			let e = self.imports.get_mut(&k).unwrap();
			e.last = self.tick;
			return Ok((e.buf, (ptr - k.0) as u64));
		}
		let a = self.align;
		let (base, end) = (ptr & !(a - 1), (ptr + len + a - 1) & !(a - 1));
		let (buf, mem) = self.dev.import_host_range(base, end - base)?;
		self.imports.insert(
			(base, end - base),
			Imported {
				buf,
				mem,
				last: self.tick,
			},
		);
		Ok((buf, (ptr - base) as u64))
	}

	/// The key of an import that covers `[ptr, ptr + len)`, if any.
	fn covering(&self, ptr: usize, len: usize) -> Option<(usize, usize)> {
		let a = self.align;
		let (base, end) = (ptr & !(a - 1), (ptr + len + a - 1) & !(a - 1));
		self.imports
			.range(..=(base, usize::MAX))
			.next_back()
			.map(|(&k, _)| k)
			.filter(|k| k.0 + k.1 >= end)
	}

	/// Makes room for a call that imports `ranges` (host bytes) and makes
	/// `sets` new descriptor sets. When the sets would not fit the pool, or
	/// the imports `max_imports`, frees every set (the pool is reset; they
	/// are made again on use) and, for the imports, the least recently used
	/// ones not in `ranges` down to half of `max_imports`. Only with no
	/// submission in flight (`scan` waits for its own; `far_begin` refuses
	/// to start over one).
	fn room(&mut self, ranges: &[(usize, usize)], sets: usize) -> Result<(), String> {
		self.tick += 1;
		let mut new = 0;
		for &(p, len) in ranges {
			match self.covering(p, len) {
				Some(k) => self.imports.get_mut(&k).unwrap().last = self.tick,
				None => new += 1,
			}
		}
		let over = self.imports.len() + new > self.max_imports;
		if !over && self.sets.len() + self.gsets.len() + self.csets.len() + sets <= MAX_SETS {
			return Ok(());
		}
		if sets > MAX_SETS || new > self.max_imports {
			return Err(format!(
				"scan: one call needs {sets} descriptor sets (pool: {MAX_SETS}), {new} imports (at most {})",
				self.max_imports
			));
		}
		if self.far_cmd.is_some() {
			return Err(
				"scan: descriptor pool or imports full while far attention is in flight".into(),
			);
		}
		let d = &self.dev.device;
		unsafe { d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()) }
			.map_err(|e| format!("scan: reset descriptor pool: {e}"))?;
		self.sets.clear();
		self.gsets.clear();
		self.csets.clear();
		if over {
			let keep = (self.max_imports / 2).min(self.max_imports - new);
			let mut old: Vec<_> = self
				.imports
				.iter()
				.filter(|(_, e)| e.last < self.tick)
				.map(|(&k, e)| (e.last, k))
				.collect();
			old.sort_unstable();
			let drop = self.imports.len().saturating_sub(keep).min(old.len());
			for &(_, k) in &old[..drop] {
				let e = self.imports.remove(&k).unwrap();
				// SAFETY: no submission in flight, and the sets that bound
				// it are gone with the pool reset
				unsafe { hostimp::free_import(d, e.buf, e.mem) };
			}
		}
		Ok(())
	}

	fn make_set(
		&self,
		st: vk::Buffer,
		bi: &[(vk::Buffer, u64, u64); 5],
	) -> Result<vk::DescriptorSet, String> {
		// bindings 0 state, 1 q, 2 k, 3 v, 4 o, 5 gb, 8 params
		self.make_set_raw(&[(st, 0, ST_BYTES), bi[0], bi[1], bi[2], bi[4], bi[3]])
	}

	/// A descriptor set with (buffer, offset, range) on bindings 0..6 and
	/// the parameter block on 8.
	fn make_set_raw(&self, b6: &[(vk::Buffer, u64, u64); 6]) -> Result<vk::DescriptorSet, String> {
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("scan descriptor set: {e}"))?[0];
		let info = |b: vk::Buffer, off: u64, range: u64| vk::DescriptorBufferInfo {
			buffer: b,
			offset: off,
			range,
		};
		let mut infos = [info(self.params.0, 0, 64); 7];
		for (i, x) in b6.iter().enumerate() {
			infos[i] = info(x.0, x.1, x.2);
		}
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
}

impl Drop for DeltaScan {
	fn drop(&mut self) {
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			if let Some((qp, _)) = self.ts {
				d.destroy_query_pool(qp, None);
			}
			d.destroy_pipeline(self.pipeline, None);
			d.destroy_shader_module(self.sm, None);
			d.destroy_pipeline(self.gn_pl, None);
			d.destroy_shader_module(self.gn_sm, None);
			d.destroy_pipeline(self.cv_pl, None);
			d.destroy_shader_module(self.cv_sm, None);
			for (b, m) in self.gn_w.iter().chain(&self.cv_w).flatten() {
				d.destroy_buffer(*b, None);
				d.free_memory(*m, None);
			}
			for i in 0..2 {
				d.destroy_pipeline(self.far_pl[i], None);
				d.destroy_shader_module(self.far_sm[i], None);
			}
			d.destroy_buffer(self.far_sc.0, None);
			d.free_memory(self.far_sc.1, None);
			std::alloc::dealloc(
				self.far_mem,
				std::alloc::Layout::from_size_align(FAR_BYTES, self.align.max(4096)).unwrap(),
			);
			d.destroy_descriptor_pool(self.pool, None);
			if let Some((b, m, _)) = self.bulk.take() {
				d.unmap_memory(m);
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
			for &(b, m) in self.states.iter().chain(std::iter::once(&self.params)) {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
			for e in self.imports.values() {
				hostimp::free_import(d, e.buf, e.mem);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Host memory at a page boundary, as the NPU's buffers are (imports
	/// cover whole pages).
	struct Page(*mut u8, usize);

	impl Page {
		fn new(len: usize) -> Page {
			let l = std::alloc::Layout::from_size_align(len.max(1), 4096).unwrap();
			// SAFETY: a non-zero size
			Page(unsafe { std::alloc::alloc_zeroed(l) }, len)
		}
		fn b(&mut self) -> &mut [u8] {
			// SAFETY: `len` bytes allocated
			unsafe { std::slice::from_raw_parts_mut(self.0, self.1) }
		}
		fn h(&mut self) -> &mut [u16] {
			// SAFETY: page aligned, `len / 2` u16
			unsafe { std::slice::from_raw_parts_mut(self.0 as *mut u16, self.1 / 2) }
		}
	}

	impl Drop for Page {
		fn drop(&mut self) {
			let l = std::alloc::Layout::from_size_align(self.1.max(1), 4096).unwrap();
			// SAFETY: allocated with this layout
			unsafe { std::alloc::dealloc(self.0, l) }
		}
	}

	/// `conv_npu.glsl` (queued by `conv_next` ahead of a scan) against the
	/// host: per channel the 4-tap conv over [cpast; qkv * qs], Swish, and
	/// per q / k head of 128 the l2 norm.
	#[test]
	fn conv_matches_host() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let Ok(mut sc) = DeltaScan::new(pick.as_deref().or(Some("intel")), 1) else {
			return;
		};
		let n = 37;
		let f = |i: usize, m: usize, d: f32| ((i * m % 97) as f32 - 48.0) / d;
		let w: Vec<f32> = (0..4 * CONV_C)
			.map(|i| f(i, 7, 160.0))
			.chain((0..CONV_C).map(|c| 0.5 + (c % 13) as f32 / 8.0))
			.collect();
		sc.set_conv(0, &w).unwrap();
		let (mut cp, mut qkv) = (Page::new(3 * CONV_C * 2), Page::new(n * CONV_C * 2));
		cp.h()
			.iter_mut()
			.enumerate()
			.for_each(|(i, x)| *x = quant::f32_to_fp16(f(i, 5, 40.0)));
		qkv.h()
			.iter_mut()
			.enumerate()
			.for_each(|(i, x)| *x = quant::f32_to_fp16(f(i, 3, 30.0)));
		let (mut qn, mut kn, mut v) = (
			Page::new(n * 4096),
			Page::new(n * 4096),
			Page::new(n * 12288),
		);
		let (mut gb, mut o) = (Page::new(n * 192), Page::new(n * 12288));
		let (cpb, qkvb) = (cp.b().to_vec(), qkv.b().to_vec());
		let (cph, qkvh) = (cp.h().to_vec(), qkv.h().to_vec());
		sc.conv_next(0, n, cp.b(), qkv.b(), qn.b(), kn.b(), v.b())
			.unwrap();
		sc.scan(0, n, qn.b(), kn.b(), v.b(), gb.b(), o.b(), 1.0)
			.unwrap();
		assert_eq!((cp.b().to_vec(), qkv.b().to_vec()), (cpb, qkvb));
		let qs = &w[4 * CONV_C..];
		let x = |r: usize, c: usize| -> f32 {
			if r < 3 {
				quant::fp16_to_f32(cph[r * CONV_C + c])
			} else {
				quant::fp16_to_f32(qkvh[(r - 3) * CONV_C + c]) * qs[c]
			}
		};
		let mut worst = 0f32;
		let mut y = vec![0f32; CONV_C];
		for t in 0..n {
			for (c, yc) in y.iter_mut().enumerate() {
				let a: f32 = (0..4).map(|k| w[4 * c + k] * x(t + k, c)).sum();
				*yc = a / (1.0 + (-a).exp());
			}
			for r in y[..4096].chunks_exact_mut(128) {
				let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
				r.iter_mut().for_each(|a| *a *= inv);
			}
			let got = qn.h()[t * 2048..][..2048]
				.iter()
				.chain(&kn.h()[t * 2048..][..2048])
				.chain(&v.h()[t * 6144..][..6144])
				.map(|&h| quant::fp16_to_f32(h))
				.collect::<Vec<_>>();
			for (g, r) in got.iter().zip(&y) {
				worst = worst.max((g - r).abs() / (r.abs() + 0.05));
			}
		}
		assert!(worst < 2e-3, "conv: worst relative error {worst}");
	}
}
