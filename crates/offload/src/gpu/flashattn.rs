//! The prefill attention of the Qwen3.8-Flash-Next NPU engine on a Vulkan
//! device (the iGPU).
//!
//! WHY
//! ---
//! The engine's prefill runs on the NPU + iGPU and keeps the cores free,
//! but its attention layers
//! ran on the cores: 2.1-2.2 s per 2048 tokens. Here one kernel
//! (`flash_attn_npu.glsl`, an online softmax over tiles of 64 keys) takes a
//! sub-chunk's queries over each one's key list: dense causal while the
//! indexer selects every complete block, else the 512 blocks it selected
//! plus the tail. The selection itself (block scores, top 512) stays on the
//! host.
//!
//! The cache is a mirror of the engine's host cache (f16 both), per
//! attention layer in host-visible memory: each call uploads the rows the
//! mirror lacks, so decode steps (which do not come here) and a reset
//! ([`FlashAttn::invalidate`]) need nothing else. It grows by a quarter at
//! a time, keeping the rows it has (a growth by 4096 positions that
//! re-uploaded the whole layer each time was quadratic in the context), up
//! to the context ([`FlashAttn::set_max_rows`]): the mirrors are system
//! memory the iGPU keeps resident, ~24 KiB a position over the 12 layers,
//! and a quarter past the context would be ~1.5 GiB more at 262144.
//!
//! The block indexer's scores (`idx_score_npu.glsl`, [`FlashAttn::score_begin`])
//! run here too, on the second fence: the NPU, the bound of the attention
//! layers at depth, ran them as a graph (~6.6 ms of a ~24 ms cycle per
//! sub-chunk at 53k positions) while the iGPU idled between attentions.
//! Their pooled block keys are a second mirror per layer, kept like the
//! K / V one.

use super::*;

const HQ: usize = 24;
const HKV: usize = 2;
const DA: usize = 256;
/// `sel` entries per query (count + 512 blocks).
const SEL: usize = 513;
/// Query heads per workgroup (the kernel's `G`).
const HG: usize = 6;
/// Indexer heads and their width.
const IH: usize = 4;
const ID: usize = 128;

struct Buf {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	map: *mut u8,
	bytes: usize,
}

/// One attention layer's K / V mirror: `cap` positions, `len` of them set.
struct Layer {
	k: Option<Buf>,
	v: Option<Buf>,
	cap: usize,
	len: usize,
}

/// See the module docs.
pub struct FlashAttn {
	dev: GpuDevice,
	sm: vk::ShaderModule,
	pl: vk::Pipeline,
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	/// q, sel, out, lse for `rows` queries
	io: Option<[Buf; 4]>,
	rows: usize,
	layers: Vec<Layer>,
	/// the most positions a mirror grows to ([`FlashAttn::set_max_rows`])
	max_rows: usize,
	/// a descriptor set per layer, made for the current buffers
	sets: Vec<vk::DescriptorSet>,
	/// the submission [`FlashAttn::attend_begin`] left in flight and its
	/// query count
	pending: Option<(vk::CommandBuffer, usize)>,
	/// host seconds and bytes of the mirror's row uploads
	/// ([`FlashAttn::take_fill`])
	fill: (f64, u64),
	/// the indexer's scores ([`FlashAttn::score_begin`])
	ix: Idx,
}

/// A mirror's next capacity for `total` entries from `old`: a quarter
/// more, in steps of 4096, but not past `max` (rounded up to 4096) unless
/// `total` is.
fn grown(total: usize, old: usize, max: usize) -> usize {
	let max = max.next_multiple_of(4096);
	if total > max {
		return total;
	}
	total
		.max((old + old / 4).next_multiple_of(4096).min(max))
		.next_multiple_of(4096)
}

/// `r` as int8 at absmax / 127 into `d`; returns the scale.
fn q8_row(r: &[f32], d: &mut [i8]) -> f32 {
	let m = r.iter().fold(0f32, |m, v| m.max(v.abs()));
	if m == 0.0 {
		d.fill(0);
		return 0.0;
	}
	let inv = 127.0 / m;
	for (d, &v) in d.iter_mut().zip(r) {
		*d = (v * inv).round() as i8;
	}
	m / 127.0
}

/// The indexer's kernel, buffers and pooled-key mirrors.
struct Idx {
	sm: vk::ShaderModule,
	pl: vk::Pipeline,
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	/// per attention layer: the pooled keys `[cap][128]` int8 and their
	/// f32 scales `[cap]`, cap, rows set
	keys: Vec<(Option<[Buf; 2]>, usize, usize)>,
	/// q `[rows][4][128]` int8, its scales `[rows][4]` f32, out
	/// `[rows][cols]` u32
	io: Option<[Buf; 3]>,
	rows: usize,
	cols: usize,
	sets: Vec<vk::DescriptorSet>,
	/// in flight on the second fence: (cmd, layer, queries, blocks)
	pending: Option<(vk::CommandBuffer, usize, usize, usize)>,
	/// waited for early (a mirror had to grow): (layer, queries, blocks)
	done: Option<(usize, usize, usize)>,
}

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for FlashAttn {}

impl FlashAttn {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU), for
	/// `layers` attention layers.
	pub fn new(pick: Option<&str>, layers: usize) -> Result<FlashAttn, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let main = CString::new("main").unwrap();
		let sm = create_sm(&dev.device, &load_spv(quant::packed!("flash_attn_npu.spv")))?;
		let pl = compute_pipeline(&dev.device, dev.aux_pipeline_layout, sm, &main)?;
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 6 * layers as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: layers as u32,
			},
		];
		let pool = unsafe {
			dev.device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: layers as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("flash attention descriptor pool: {e}"))?;
		let params = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		let ism = create_sm(&dev.device, &load_spv(quant::packed!("idx_score_npu.spv")))?;
		let ipl = compute_pipeline(&dev.device, dev.aux_pipeline_layout, ism, &main)?;
		let ipool = unsafe {
			dev.device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: layers as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("indexer descriptor pool: {e}"))?;
		let iparams = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		Ok(FlashAttn {
			dev,
			sm,
			pl,
			pool,
			params,
			io: None,
			rows: 0,
			layers: (0..layers)
				.map(|_| Layer {
					k: None,
					v: None,
					cap: 0,
					len: 0,
				})
				.collect(),
			max_rows: 1 << 40,
			sets: Vec::new(),
			pending: None,
			fill: (0.0, 0),
			ix: Idx {
				sm: ism,
				pl: ipl,
				pool: ipool,
				params: iparams,
				keys: (0..layers).map(|_| (None, 0, 0)).collect(),
				io: None,
				rows: 0,
				cols: 0,
				sets: Vec::new(),
				pending: None,
				done: None,
			},
		})
	}

	/// Grow no mirror past `rows` positions (the context); a call past it
	/// still gets the rows it needs.
	pub fn set_max_rows(&mut self, rows: usize) {
		self.max_rows = rows.clamp(1, 1 << 40);
	}

	fn grown(&self, total: usize, old: usize, per: usize) -> usize {
		grown(total, old, self.max_rows.div_ceil(per))
	}

	/// The mirror's row uploads since the last call: (host seconds, bytes).
	/// After a reset or a restore the next prefill uploads every row again.
	pub fn take_fill(&mut self) -> (f64, u64) {
		std::mem::take(&mut self.fill)
	}

	/// Forget every mirrored row (the engine's cache was reset or replaced).
	pub fn invalidate(&mut self) {
		if let Some((cmd, _)) = self.pending.take() {
			let _ = self.dev.wait_submitted(cmd);
		}
		self.score_wait();
		self.ix.done = None;
		self.layers.iter_mut().for_each(|l| l.len = 0);
		self.ix.keys.iter_mut().for_each(|k| k.2 = 0);
	}

	/// Wait for the scores in flight, keeping their result.
	fn score_wait(&mut self) {
		if let Some((cmd, a, n, nb)) = self.ix.pending.take() {
			let _ = self.dev.wait_submitted_spin_bank(cmd, 200, 1);
			self.ix.done = Some((a, n, nb));
		}
	}

	/// The pooled block keys the indexer's mirror of layer `a` holds.
	pub fn idx_len(&self, a: usize) -> usize {
		self.ix.keys[a].2
	}

	/// Set layer `a`'s pooled keys from block `at` (at most
	/// [`Self::idx_len`]) on: `rows` f32 `[..][128]`; the mirror then
	/// holds `at + rows.len() / 128` blocks.
	pub fn idx_append(&mut self, a: usize, at: usize, rows: &[f32]) -> Result<(), String> {
		if at > self.ix.keys[a].2 {
			return Err(format!(
				"indexer mirror: block {at} past {}",
				self.ix.keys[a].2
			));
		}
		let total = at + rows.len() / ID;
		if total > self.ix.keys[a].1 {
			// (a job in flight reads the old buffer)
			self.score_wait();
			let cap = self.grown(total, self.ix.keys[a].1, 4);
			let nb = [self.alloc(cap * ID)?, self.alloc(cap * 4)?];
			if let Some(ob) = self.ix.keys[a].0.take() {
				for (o, (n, w)) in ob.into_iter().zip(nb.iter().zip([ID, 4])) {
					// SAFETY: both mappings hold at least `at` rows
					unsafe { ptr::copy_nonoverlapping(o.map, n.map, at * w) };
					self.free(o);
				}
			}
			self.ix.keys[a] = (Some(nb), cap, at);
			self.ix.sets.clear();
		}
		let (b, _, len) = &mut self.ix.keys[a];
		let b = b.as_ref().unwrap();
		// SAFETY: the mappings hold `cap >= total` rows. Rows past the
		// blocks a job in flight scores are not read by it.
		let (d, s) = unsafe {
			(
				std::slice::from_raw_parts_mut(b[0].map as *mut i8, b[0].bytes),
				std::slice::from_raw_parts_mut(b[1].map as *mut f32, b[1].bytes / 4),
			)
		};
		for (j, r) in rows.chunks_exact(ID).enumerate() {
			s[at + j] = q8_row(r, &mut d[(at + j) * ID..][..ID]);
		}
		*len = total;
		Ok(())
	}

	/// Queue the indexer's scores of layer `a`'s first `nb` pooled blocks
	/// (at most [`Self::idx_len`]) against `n` queries `iq`
	/// `[n][4][128]` f16, on the second fence (beside one attention
	/// submission); [`Self::score_end`] collects them.
	pub fn score_begin(&mut self, a: usize, iq: &[u16], n: usize, nb: usize) -> Result<(), String> {
		self.score_wait();
		self.ix.done = None;
		if n == 0 || nb == 0 || nb > self.ix.keys[a].2 || iq.len() < n * IH * ID {
			return Err(format!(
				"indexer: layer {a}, {n} queries, {nb} blocks: bad sizes"
			));
		}
		let cols = nb.next_multiple_of(64);
		if n > self.ix.rows || cols > self.ix.cols {
			if let Some(io) = self.ix.io.take() {
				for b in io {
					self.free(b);
				}
			}
			let rows = n.max(self.ix.rows).next_multiple_of(256);
			let cols = self.grown(cols, self.ix.cols, 4);
			self.ix.io = Some([
				self.alloc(rows * IH * ID)?,
				self.alloc(rows * IH * 4)?,
				self.alloc(rows * cols * 4)?,
			]);
			(self.ix.rows, self.ix.cols) = (rows, cols);
			self.ix.sets.clear();
		}
		if self.ix.sets.is_empty() {
			self.make_idx_sets()?;
		}
		let io = self.ix.io.as_ref().unwrap();
		// SAFETY: the mappings hold `rows >= n` queries
		let (d, s) = unsafe {
			(
				std::slice::from_raw_parts_mut(io[0].map as *mut i8, n * IH * ID),
				std::slice::from_raw_parts_mut(io[1].map as *mut f32, n * IH),
			)
		};
		let mut r = [0f32; ID];
		for (j, q) in iq[..n * IH * ID].chunks_exact(ID).enumerate() {
			for (r, &h) in r.iter_mut().zip(q) {
				*r = quant::fp16_to_f32(h);
			}
			s[j] = q8_row(&r, &mut d[j * ID..][..ID]);
		}
		let d = &self.dev.device;
		let (pl, layout, params, set) = (
			self.ix.pl,
			self.dev.aux_pipeline_layout,
			self.ix.params.0,
			self.ix.sets[a],
		);
		let mut pr = [0u32; 16];
		pr[0] = n as u32;
		pr[1] = nb as u32;
		pr[2] = self.ix.cols as u32;
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
		let cmd = self
			.dev
			.submit_nowait_bank(1, |cmd| unsafe {
				d.cmd_update_buffer(cmd, params, 0, &pb);
				barrier(
					cmd,
					P::TRANSFER | P::HOST,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE | A::HOST_WRITE,
					A::UNIFORM_READ | A::SHADER_READ,
				);
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					layout,
					0,
					&[set],
					&[],
				);
				d.cmd_dispatch(cmd, nb.div_ceil(64) as u32, n.div_ceil(16) as u32, 1);
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("indexer: {e}"))?;
		self.ix.pending = Some((cmd, a, n, nb));
		Ok(())
	}

	/// Wait for [`Self::score_begin`]'s scores: `(layer, queries, blocks,
	/// row stride, scores)`, the scores `[queries][stride]` as the f32
	/// bits of each (>= +0); `None` when nothing was queued.
	#[allow(clippy::type_complexity)]
	pub fn score_end(&mut self) -> Result<Option<(usize, usize, usize, usize, &[u32])>, String> {
		if let Some((cmd, a, n, nb)) = self.ix.pending.take() {
			self.dev
				.wait_submitted_spin_bank(cmd, 200, 1)
				.map_err(|e| format!("indexer: {e}"))?;
			self.ix.done = Some((a, n, nb));
		}
		let Some((a, n, nb)) = self.ix.done.take() else {
			return Ok(None);
		};
		let io = self.ix.io.as_ref().unwrap();
		let cols = self.ix.cols;
		// SAFETY: the mapping holds `rows * cols` u32
		let s = unsafe { std::slice::from_raw_parts(io[2].map as *const u32, n * cols) };
		Ok(Some((a, n, nb, cols, s)))
	}

	/// A descriptor set per layer for the indexer (layers without keys get
	/// the q buffer in their place; they are not dispatched before they
	/// have some).
	fn make_idx_sets(&mut self) -> Result<(), String> {
		let d = &self.dev.device;
		unsafe { d.reset_descriptor_pool(self.ix.pool, vk::DescriptorPoolResetFlags::empty()) }
			.map_err(|e| format!("indexer: {e}"))?;
		let io = self.ix.io.as_ref().ok_or("indexer: no buffers")?;
		let layouts = vec![self.dev.aux_layout; self.ix.keys.len()];
		self.ix.sets = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.ix.pool,
				descriptor_set_count: layouts.len() as u32,
				p_set_layouts: layouts.as_ptr(),
				..Default::default()
			})
		}
		.map_err(|e| format!("indexer descriptor sets: {e}"))?;
		let info = |b: vk::Buffer| vk::DescriptorBufferInfo {
			buffer: b,
			offset: 0,
			range: vk::WHOLE_SIZE,
		};
		for (k, &set) in self.ix.keys.iter().zip(&self.ix.sets) {
			let (kb, ks) =
				k.0.as_ref()
					.map_or((io[0].buf, io[1].buf), |b| (b[0].buf, b[1].buf));
			let infos = [
				info(io[0].buf),
				info(kb),
				info(ks),
				info(io[1].buf),
				info(io[2].buf),
				info(io[0].buf),
				vk::DescriptorBufferInfo {
					buffer: self.ix.params.0,
					offset: 0,
					range: 64,
				},
			];
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
		}
		Ok(())
	}

	fn alloc(&self, bytes: usize) -> Result<Buf, String> {
		let (buf, mem) = self
			.dev
			.try_create_readback_buffer(bytes as u64)
			.ok_or(format!("flash attention: no memory for {bytes} bytes"))?;
		let map = self.dev.map(mem, bytes) as *mut u8;
		Ok(Buf {
			buf,
			mem,
			map,
			bytes,
		})
	}

	fn free(&self, b: Buf) {
		unsafe {
			self.dev.device.unmap_memory(b.mem);
			self.dev.device.destroy_buffer(b.buf, None);
			self.dev.device.free_memory(b.mem, None);
		}
	}

	/// Attention layer `a`, queries at positions `[p0, p0 + n)`: `k` / `v`
	/// the layer's whole host cache `[p0 + n][2][256]` f16, as contiguous
	/// runs of rows in order, `q` `[n][24][256]`
	/// f16, `sel` `[n][513]` (see the kernel; a dense query starts at key
	/// `kmin`) -> `out` `[n][24][256]` f16.
	#[allow(clippy::too_many_arguments)]
	pub fn attend(
		&mut self,
		a: usize,
		k: &[&[u16]],
		v: &[&[u16]],
		p0: usize,
		n: usize,
		q: &[u16],
		sel: &[u32],
		kmin: usize,
		out: &mut [u16],
	) -> Result<(), String> {
		self.attend_begin(a, k, v, p0, n, q, sel, kmin)?;
		self.attend_end(out, None)
	}

	/// [`FlashAttn::attend`] up to the submission, which is left in flight:
	/// the engine copies its sub-chunks' buffers meanwhile. One at a time,
	/// each followed by [`FlashAttn::attend_end`].
	#[allow(clippy::too_many_arguments)]
	pub fn attend_begin(
		&mut self,
		a: usize,
		k: &[&[u16]],
		v: &[&[u16]],
		p0: usize,
		n: usize,
		q: &[u16],
		sel: &[u32],
		kmin: usize,
	) -> Result<(), String> {
		if self.pending.is_some() {
			return Err("flash attention: a submission is already in flight".into());
		}
		let total = p0 + n;
		let row = HKV * DA;
		let len = |x: &[&[u16]]| x.iter().map(|s| s.len()).sum::<usize>();
		if a >= self.layers.len()
			|| n == 0 || len(k) < total * row
			|| len(v) < total * row
			|| q.len() < n * HQ * DA
			|| sel.len() < n * SEL
			|| kmin > p0
		{
			return Err(format!(
				"flash attention: layer {a}, {n} queries at {p0}: bad sizes"
			));
		}
		let mut remake = false;
		if n > self.rows {
			if let Some(io) = self.io.take() {
				for b in io {
					self.free(b);
				}
			}
			let rows = n.next_multiple_of(256);
			self.io = Some([
				self.alloc(rows * HQ * DA * 2)?,
				self.alloc(rows * SEL * 4)?,
				self.alloc(rows * HQ * DA * 2)?,
				self.alloc(rows * HQ * 4)?,
			]);
			self.rows = rows;
			remake = true;
		}
		if total > self.layers[a].cap {
			let cap = self.grown(total, self.layers[a].cap, 1);
			let keep = self.layers[a].len.min(total);
			for i in 0..2 {
				let nb = self.alloc(cap * row * 2)?;
				let la = &mut self.layers[a];
				let ob = if i == 0 {
					la.k.replace(nb)
				} else {
					la.v.replace(nb)
				};
				if let Some(ob) = ob {
					let nb = if i == 0 { la.k.as_ref() } else { la.v.as_ref() }.unwrap();
					// SAFETY: both mappings hold at least `keep` rows
					unsafe { ptr::copy_nonoverlapping(ob.map, nb.map, keep * row * 2) };
					self.free(ob);
				}
			}
			let la = &mut self.layers[a];
			(la.cap, la.len) = (cap, keep);
			remake = true;
		}
		// the rows the mirror lacks
		{
			let t = std::time::Instant::now();
			let la = &mut self.layers[a];
			let from = la.len.min(total);
			for (src, dst) in [(k, la.k.as_ref().unwrap()), (v, la.v.as_ref().unwrap())] {
				// SAFETY: the mapping holds `cap * row` f16 and `total <= cap`
				let d =
					unsafe { std::slice::from_raw_parts_mut(dst.map as *mut u16, dst.bytes / 2) };
				let (mut at, end) = (from * row, total * row);
				let mut base = 0;
				for s in src {
					if at >= end {
						break;
					}
					if at < base + s.len() {
						let n = (base + s.len()).min(end) - at;
						d[at..at + n].copy_from_slice(&s[at - base..][..n]);
						at += n;
					}
					base += s.len();
				}
			}
			la.len = total;
			if total > from {
				self.fill.0 += t.elapsed().as_secs_f64();
				self.fill.1 += ((total - from) * row * 4) as u64;
			}
		}
		if remake || self.sets.is_empty() {
			self.make_sets()?;
		}
		let io = self.io.as_ref().unwrap();
		// SAFETY: the mappings hold `rows >= n` queries
		unsafe {
			ptr::copy_nonoverlapping(q.as_ptr() as *const u8, io[0].map, n * HQ * DA * 2);
			ptr::copy_nonoverlapping(sel.as_ptr() as *const u8, io[1].map, n * SEL * 4);
		}
		let d = &self.dev.device;
		let (pl, layout, params, set) = (
			self.pl,
			self.dev.aux_pipeline_layout,
			self.params.0,
			self.sets[a],
		);
		let mut pr = [0u32; 16];
		pr[0] = p0 as u32;
		pr[1] = n as u32;
		pr[2] = (1.0 / (DA as f32).sqrt()).to_bits();
		pr[3] = kmin as u32;
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
		let cmd = self
			.dev
			.submit_nowait(|cmd| unsafe {
				d.cmd_update_buffer(cmd, params, 0, &pb);
				barrier(
					cmd,
					P::TRANSFER | P::HOST,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE | A::HOST_WRITE,
					A::UNIFORM_READ | A::SHADER_READ,
				);
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					layout,
					0,
					&[set],
					&[],
				);
				d.cmd_dispatch(cmd, n as u32, HKV as u32, (HQ / HKV / HG) as u32);
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("flash attention: {e}"))?;
		self.pending = Some((cmd, n));
		Ok(())
	}

	/// Wait for [`FlashAttn::attend_begin`]'s submission: its `out`
	/// `[n][24][256]` f16 and, if asked, `lse` `[n][24]` (max + ln sum of
	/// the scaled scores).
	pub fn attend_end(&mut self, out: &mut [u16], lse: Option<&mut [f32]>) -> Result<(), String> {
		let (cmd, n) = self
			.pending
			.take()
			.ok_or("flash attention: nothing in flight")?;
		self.dev
			.wait_submitted(cmd)
			.map_err(|e| format!("flash attention: {e}"))?;
		if out.len() < n * HQ * DA {
			return Err(format!("flash attention: out holds {} values", out.len()));
		}
		let io = self.io.as_ref().unwrap();
		// SAFETY: the mapping holds `rows >= n` queries
		unsafe {
			ptr::copy_nonoverlapping(io[2].map as *const u16, out.as_mut_ptr(), n * HQ * DA);
		}
		if let Some(l) = lse {
			if l.len() < n * HQ {
				return Err(format!("flash attention: lse holds {} values", l.len()));
			}
			// SAFETY: as above
			unsafe {
				ptr::copy_nonoverlapping(io[3].map as *const f32, l.as_mut_ptr(), n * HQ);
			}
		}
		Ok(())
	}

	/// A descriptor set per layer (layers without a cache yet get the q
	/// buffer in its place; they are not dispatched before they have one).
	fn make_sets(&mut self) -> Result<(), String> {
		let d = &self.dev.device;
		unsafe { d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()) }
			.map_err(|e| format!("flash attention: {e}"))?;
		let io = self.io.as_ref().ok_or("flash attention: no buffers")?;
		let layouts = vec![self.dev.aux_layout; self.layers.len()];
		self.sets = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: layouts.len() as u32,
				p_set_layouts: layouts.as_ptr(),
				..Default::default()
			})
		}
		.map_err(|e| format!("flash attention descriptor sets: {e}"))?;
		let info = |b: vk::Buffer| vk::DescriptorBufferInfo {
			buffer: b,
			offset: 0,
			range: vk::WHOLE_SIZE,
		};
		for (la, &set) in self.layers.iter().zip(&self.sets) {
			let kb = la.k.as_ref().map_or(io[0].buf, |b| b.buf);
			let vb = la.v.as_ref().map_or(io[0].buf, |b| b.buf);
			let infos = [
				info(io[0].buf),
				info(kb),
				info(vb),
				info(io[1].buf),
				info(io[2].buf),
				info(io[3].buf),
				vk::DescriptorBufferInfo {
					buffer: self.params.0,
					offset: 0,
					range: 64,
				},
			];
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
		}
		Ok(())
	}
}

impl Drop for FlashAttn {
	fn drop(&mut self) {
		unsafe {
			let _ = self.dev.device.device_wait_idle();
		}
		let mut bufs: Vec<Buf> = self.io.take().map(Vec::from).unwrap_or_default();
		for la in &mut self.layers {
			bufs.extend(la.k.take());
			bufs.extend(la.v.take());
		}
		bufs.extend(self.ix.io.take().map(Vec::from).unwrap_or_default());
		for k in &mut self.ix.keys {
			bufs.extend(k.0.take().into_iter().flatten());
		}
		for b in bufs {
			self.free(b);
		}
		let d = &self.dev.device;
		unsafe {
			d.destroy_pipeline(self.pl, None);
			d.destroy_shader_module(self.sm, None);
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_buffer(self.params.0, None);
			d.free_memory(self.params.1, None);
			d.destroy_pipeline(self.ix.pl, None);
			d.destroy_shader_module(self.ix.sm, None);
			d.destroy_descriptor_pool(self.ix.pool, None);
			d.destroy_buffer(self.ix.params.0, None);
			d.free_memory(self.ix.params.1, None);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A mirror grows by a quarter in steps of 4096 and stops at the
	/// context, unless a call needs more.
	#[test]
	fn mirror_growth_stops_at_the_context() {
		assert_eq!(grown(100, 0, 262144), 4096);
		assert_eq!(grown(8193, 8192, 262144), 12288);
		assert_eq!(grown(240000, 229376, 262144), 262144);
		assert_eq!(grown(258000, 245760, 262144), 262144);
		assert_eq!(grown(262144, 262144, 262144), 262144);
		assert_eq!(grown(262145, 262144, 262144), 262145);
		// the indexer's blocks of 4: 65536 at 262144 positions
		assert_eq!(grown(60000, 57344, 262144usize.div_ceil(4)), 65536);
	}

	/// A mirror that grows past its capacity keeps the rows it had: the
	/// same output as a mirror uploaded from scratch.
	#[test]
	fn flash_attention_mirror_grows_in_place() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let Ok(mut fa) = FlashAttn::new(pick.as_deref(), 1) else {
			return;
		};
		let row = HKV * DA;
		let (p0, n) = (4300usize, 64usize);
		let total = p0 + n;
		let h = |i: usize, m: usize| quant::f32_to_fp16(((i * m % 101) as f32 - 50.0) / 64.0);
		let k: Vec<u16> = (0..total * row).map(|i| h(i, 7)).collect();
		let v: Vec<u16> = (0..total * row).map(|i| h(i, 13)).collect();
		let q: Vec<u16> = (0..n * HQ * DA).map(|i| h(i, 3)).collect();
		// dense queries over every earlier position (the indexer would select
		// blocks past 2048; the kernel takes what `sel` says)
		let mut sel = vec![0u32; n * SEL];
		(0..n).for_each(|t| sel[t * SEL] = u32::MAX);
		let mut first = vec![0u16; 256 * HQ * DA];
		let sel0: Vec<u32> = (0..256 * SEL)
			.map(|i| if i % SEL == 0 { u32::MAX } else { 0 })
			.collect();
		let q0: Vec<u16> = (0..256 * HQ * DA).map(|i| h(i, 5)).collect();
		// 256 rows mirrored (capacity 4096), then a call that needs 4364
		fa.attend(
			0,
			&[&k[..256 * row]],
			&[&v[..256 * row]],
			0,
			256,
			&q0,
			&sel0,
			0,
			&mut first,
		)
		.unwrap();
		let mut grown = vec![0u16; n * HQ * DA];
		let (ka, kb) = k.split_at(1000 * row);
		let (va, vb) = v.split_at(1000 * row);
		fa.attend(0, &[ka, kb], &[va, vb], p0, n, &q, &sel, 0, &mut grown)
			.unwrap();
		assert_eq!(fa.layers[0].cap, 8192);
		fa.invalidate();
		let mut fresh = vec![0u16; n * HQ * DA];
		fa.attend(0, &[&k[..]], &[&v[..]], p0, n, &q, &sel, 0, &mut fresh)
			.unwrap();
		assert_eq!(grown, fresh);
	}

	/// The kernel against an f64 softmax over each query's key list, on the
	/// default device (`QWEN_TEST_ADAPTER` picks one): 70 queries at position 2030 (some dense, some past
	/// 2051 with 512 selected blocks), then 5 more after rows appended.
	#[test]
	fn flash_attention_matches_reference() {
		// `QWEN_TEST_ADAPTER=intel`: on the iGPU
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let Ok(mut fa) = FlashAttn::new(pick.as_deref(), 2) else {
			return;
		};
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut val = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
		};
		let row = HKV * DA;
		let r16 = |x: f32| quant::fp16_to_f32(quant::f32_to_fp16(x));
		let run =
			|fa: &mut FlashAttn, p0: usize, n: usize, kmin: usize, val: &mut dyn FnMut() -> f32| {
				let total = p0 + n;
				let k: Vec<f32> = (0..total * row).map(|_| r16(val())).collect();
				let v: Vec<f32> = (0..total * row).map(|_| r16(val())).collect();
				let q: Vec<u16> = (0..n * HQ * DA)
					.map(|_| quant::f32_to_fp16(val() * 3.0))
					.collect();
				let mut sel = vec![0u32; n * SEL];
				for t in 0..n {
					sel[t * SEL] = u32::MAX;
				}
				let mut keys: Vec<Vec<usize>> = Vec::new();
				for t in 0..n {
					let p = p0 + t;
					let nb = (p + 1) / 4;
					let mut ks: Vec<usize> = if nb <= 512 {
						(kmin..nb * 4).collect()
					} else {
						// every other block from a per-query offset, then the rest
						let mut bs: Vec<usize> = (0..nb).filter(|b| (b + t) % 2 == 0).collect();
						bs.truncate(512);
						let mut b = 0;
						while bs.len() < 512 {
							if !bs.contains(&b) {
								bs.push(b);
							}
							b += 1;
						}
						bs.sort_unstable();
						sel[t * SEL] = 512;
						for (i, &b) in bs.iter().enumerate() {
							sel[t * SEL + 1 + i] = b as u32;
						}
						bs.iter().flat_map(|&b| b * 4..b * 4 + 4).collect()
					};
					ks.extend((nb * 4).max(kmin)..=p);
					keys.push(ks);
				}
				let mut out = vec![0u16; n * HQ * DA];
				let mut lse = vec![0f32; n * HQ];
				let (kh, vh): (Vec<u16>, Vec<u16>) = (
					k.iter().map(|&x| quant::f32_to_fp16(x)).collect(),
					v.iter().map(|&x| quant::f32_to_fp16(x)).collect(),
				);
				// (split into runs, as the engine's segments are)
				let (k2, v2) = (kh.split_at(kh.len() / 3), vh.split_at(vh.len() / 3));
				fa.attend_begin(1, &[k2.0, k2.1], &[v2.0, v2.1], p0, n, &q, &sel, kmin)
					.unwrap();
				fa.attend_end(&mut out, Some(&mut lse)).unwrap();
				for t in 0..n {
					for h in 0..HQ {
						let kh = h / (HQ / HKV);
						let qf = |d: usize| quant::fp16_to_f32(q[(t * HQ + h) * DA + d]) as f64;
						let sc: Vec<f64> = keys[t]
							.iter()
							.map(|&u| {
								(0..DA)
									.map(|d| qf(d) * k[(u * HKV + kh) * DA + d] as f64)
									.sum::<f64>() / 16.0
							})
							.collect();
						let m = sc.iter().cloned().fold(f64::MIN, f64::max);
						let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
						let lw = m + z.ln();
						let lg = lse[t * HQ + h] as f64;
						assert!(
							(lg - lw).abs() < 2e-3 * (1.0 + lw.abs()),
							"query {t} head {h}: lse {lg} vs {lw}"
						);
						for d in (0..DA).step_by(7) {
							let want: f64 = keys[t]
								.iter()
								.zip(&sc)
								.map(|(&u, s)| {
									(s - m).exp() / z * v[(u * HKV + kh) * DA + d] as f64
								})
								.sum();
							let got = quant::fp16_to_f32(out[(t * HQ + h) * DA + d]) as f64;
							assert!(
								(got - want).abs() < 2e-3 * (1.0 + want.abs()),
								"query {t} head {h} dim {d}: {got} vs {want}"
							);
						}
					}
				}
			};
		run(&mut fa, 2030, 70, 0, &mut val);
		// the dense queries from key 1022 (the keys before on the NPU)
		fa.invalidate();
		run(&mut fa, 1990, 70, 1022, &mut val);
		// a new history: invalidated, shorter
		fa.invalidate();
		run(&mut fa, 3, 5, 0, &mut val);
	}

	/// `cargo test --release -p offload --features vulkan --lib
	/// flash_attention_bench -- --ignored --nocapture` (`QWEN_TEST_ADAPTER`,
	/// `QWEN_TEST_SPV=FILE` loads another build of the kernel): ms per
	/// 256-query call, dense at 2048 keys and selected (512 blocks) at 3k.
	#[test]
	#[ignore]
	fn flash_attention_bench() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let mut fa = FlashAttn::new(pick.as_deref(), 1).unwrap();
		if let Ok(f) = std::env::var("QWEN_TEST_SPV") {
			let spv = std::fs::read(f).unwrap();
			let d = &fa.dev.device;
			let sm = create_sm(d, &load_spv(&spv)).unwrap();
			let main = CString::new("main").unwrap();
			fa.pl = compute_pipeline(d, fa.dev.aux_pipeline_layout, sm, &main).unwrap();
		}
		let n = 256;
		for (p0, sparse) in [(1792usize, false), (2900, true)] {
			let total = p0 + n;
			let k: Vec<u16> = (0..total * HKV * DA)
				.map(|i| quant::f32_to_fp16(((i % 97) as f32 - 48.0) / 64.0))
				.collect();
			let q: Vec<u16> = (0..n * HQ * DA)
				.map(|i| quant::f32_to_fp16(((i % 89) as f32 - 44.0) / 32.0))
				.collect();
			let mut sel = vec![0u32; n * SEL];
			for t in 0..n {
				sel[t * SEL] = u32::MAX;
			}
			if sparse {
				for t in 0..n {
					sel[t * SEL] = 512;
					let nb = (p0 + t + 1) / 4;
					for b in 0..512 {
						sel[t * SEL + 1 + b] = (b * nb / 512) as u32;
					}
				}
			}
			let mut out = vec![0u16; n * HQ * DA];
			fa.attend(0, &[&k[..]], &[&k[..]], p0, n, &q, &sel, 0, &mut out)
				.unwrap();
			let t = std::time::Instant::now();
			let it = 10;
			for _ in 0..it {
				fa.attend(0, &[&k[..]], &[&k[..]], p0, n, &q, &sel, 0, &mut out)
					.unwrap();
			}
			let ms = t.elapsed().as_secs_f64() * 1e3 / it as f64;
			let keys = if sparse { 2048.0 } else { (p0 + n / 2) as f64 };
			let gf = n as f64 * HQ as f64 * keys * DA as f64 * 4.0 / ms / 1e6;
			let cs: f64 = out.iter().map(|&h| quant::fp16_to_f32(h) as f64).sum();
			println!(
				"p0 {p0} sparse {sparse}: {ms:.2} ms per call, {gf:.0} GFLOP/s, checksum {cs:.4}"
			);
		}
	}

	fn idx_data(n: usize, nb: usize) -> (Vec<u16>, Vec<f32>) {
		let mut s = 99u64;
		let mut rnd = || {
			s = s
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
		};
		let q: Vec<u16> = (0..n * IH * ID)
			.map(|_| quant::f32_to_fp16(rnd() * 2.0))
			.collect();
		// keys rounded to f16 as the mirror holds them
		let k: Vec<f32> = (0..nb * ID)
			.map(|_| quant::fp16_to_f32(quant::f32_to_fp16(rnd() * 3.0)))
			.collect();
		(q, k)
	}

	/// The indexer's scores against a host reference, across a mirror that
	/// grows between appends and a query count off the tile size.
	#[test]
	fn indexer_scores_match_reference() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let Ok(mut fa) = FlashAttn::new(pick.as_deref(), 2) else {
			return;
		};
		let (n, nb) = (77usize, 5000usize);
		let (q, k) = idx_data(n, nb);
		fa.idx_append(1, 0, &k[..1000 * ID]).unwrap();
		fa.idx_append(1, 1000, &k[1000 * ID..]).unwrap();
		assert_eq!(fa.idx_len(1), nb);
		let nbs = nb - 3;
		fa.score_begin(1, &q, n, nbs).unwrap();
		let (a, rn, rb, stride, s) = fa.score_end().unwrap().unwrap();
		assert_eq!((a, rn, rb), (1, n, nbs));
		// the int8 rows the kernel sees, then its float steps
		let q8 = |r: &[f32]| {
			let mut d = vec![0i8; ID];
			let s = q8_row(r, &mut d);
			(d, s)
		};
		let qq: Vec<(Vec<i8>, f32)> = q
			.chunks_exact(ID)
			.map(|r| q8(&r.iter().map(|&h| quant::fp16_to_f32(h)).collect::<Vec<_>>()))
			.collect();
		let kq: Vec<(Vec<i8>, f32)> = k.chunks_exact(ID).map(q8).collect();
		let (mut md, mut mf) = (0f32, 0f32);
		for t in 0..n {
			for b in 0..nbs {
				let m: Vec<f32> = (0..IH)
					.map(|h| {
						let (qa, qs) = &qq[t * IH + h];
						let d: i32 = qa
							.iter()
							.zip(&kq[b].0)
							.map(|(&x, &y)| x as i32 * y as i32)
							.sum();
						(d as f32 * qs).max(0.0)
					})
					.collect();
				let r = (m[0] + m[1] + m[2] + m[3]) * kq[b].1;
				let g = f32::from_bits(s[t * stride + b]);
				md = md.max((g - r).abs() / (1e-3 + r.abs()));
				// against the f32 scores: the int8 rounding
				let f: f32 = (0..IH)
					.map(|h| {
						(0..ID)
							.map(|d| quant::fp16_to_f32(q[(t * IH + h) * ID + d]) * k[b * ID + d])
							.sum::<f32>()
							.max(0.0)
					})
					.sum();
				mf = mf.max((g - f).abs());
			}
		}
		assert!(md < 1e-5, "max rel diff {md}");
		eprintln!("indexer: max |int8 - f32| {mf:.4}");
		assert!(fa.score_end().unwrap().is_none());
	}

	/// iGPU time of a 256-query indexer call at 53k / 82k positions.
	#[test]
	#[ignore]
	fn indexer_bench() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let mut fa = FlashAttn::new(pick.as_deref(), 1).unwrap();
		if let Ok(f) = std::env::var("QWEN_TEST_SPV") {
			let spv = std::fs::read(f).unwrap();
			let d = &fa.dev.device;
			let sm = create_sm(d, &load_spv(&spv)).unwrap();
			let main = CString::new("main").unwrap();
			fa.ix.pl = compute_pipeline(d, fa.dev.aux_pipeline_layout, sm, &main).unwrap();
		}
		let n = 256;
		let (q, k) = idx_data(n, 20736);
		fa.idx_append(0, 0, &k).unwrap();
		for nb in [3328usize, 13312, 20736] {
			fa.score_begin(0, &q, n, nb).unwrap();
			fa.score_end().unwrap();
			let it = 20;
			let t = std::time::Instant::now();
			for _ in 0..it {
				fa.score_begin(0, &q, n, nb).unwrap();
				fa.score_end().unwrap();
			}
			let ms = t.elapsed().as_secs_f64() * 1e3 / it as f64;
			let gf = 2.0 * (n * IH * ID * nb) as f64 / ms / 1e6;
			eprintln!("nb {nb}: {ms:.2} ms per call, {gf:.0} GFLOP/s");
		}
	}
}
