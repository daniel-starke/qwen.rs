//! The MiniMax-H3-Image transformer's attention on a discrete card (the
//! GTX 1060), for a share of each block's query chunks.
//!
//! `imagegen::h3::npudit` runs the blocks' per-row work on the NPU and the
//! iGPU and the attention (every row over every row) on the cores, chunk by
//! chunk, overlapped; the attention is the transformer's bound. The card is
//! otherwise idle and has its own memory bandwidth, so it takes query
//! chunks while the cores take the others (`h3_attn.glsl`).
//!
//! Unlike the iGPU kernels it does not read host memory in place: a key
//! block is read once per 48 query rows, far too often for PCIe. A block's
//! K and V go to VRAM once (the host stages them row by row as they are
//! finished, [`H3Attn::stage_kv`]; the copy rides on the block's first
//! submission), queries go up and outputs come back per submission
//! through mapped slots.
//!
//! VRAM is held only between [`H3Attn::reserve`] and [`H3Attn::release`]
//! (one job): K and V of the job's keys plus [`SLOTS`] query / output
//! slots, refused past `QWEN_H3_ATTN_GPU_MB` (default 1024) or past what
//! `VK_EXT_memory_budget` leaves with a margin for other applications.

use super::*;

/// f16 values of a head.
const HD: usize = 128;
/// Submissions in flight at once.
pub const H3_ATTN_SLOTS: usize = 3;
/// VRAM left free for other applications beyond this one's use (MB).
const MARGIN_MB: u64 = 512;

struct Mem {
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	/// Mapped address (host-visible buffers), else null.
	ptr: *mut u8,
	size: usize,
}

struct Slot {
	q_host: Mem,
	q_dev: Mem,
	o_dev: Mem,
	o_host: Mem,
	set: vk::DescriptorSet,
	fence: vk::Fence,
	cmd: Option<vk::CommandBuffer>,
	nq: usize,
}

/// See the module docs.
pub struct H3Attn {
	dev: GpuDevice,
	dsl: vk::DescriptorSetLayout,
	layout: vk::PipelineLayout,
	sm: vk::ShaderModule,
	pl: vk::Pipeline,
	pool: vk::DescriptorPool,
	/// Two timestamps per slot: its submission's start and end on the card.
	qpool: vk::QueryPool,
	/// Seconds per timestamp tick (0: no timestamps on this queue).
	tick: f64,
	/// Values of a row (heads x 128).
	row: usize,
	/// Query rows a slot holds.
	slot_rows: usize,
	/// Keys the K / V buffers hold (0: nothing reserved).
	cap: usize,
	k_host: Option<Mem>,
	v_host: Option<Mem>,
	k_dev: Option<Mem>,
	v_dev: Option<Mem>,
	slots: Vec<Slot>,
}

// SAFETY: used from one thread at a time; the mapped pointers belong to
// the buffers it owns.
unsafe impl Send for H3Attn {}

/// `QWEN_H3_ATTN_GPU_MB`: the most VRAM a job's attention buffers may take.
fn cap_bytes() -> u64 {
	std::env::var("QWEN_H3_ATTN_GPU_MB")
		.ok()
		.and_then(|v| v.trim().parse::<u64>().ok())
		.unwrap_or(1024)
		<< 20
}

impl H3Attn {
	/// On the Vulkan adapter `pick` names, for `heads` heads of 128 and
	/// submissions of up to `slot_rows` query rows.
	pub fn new(pick: Option<&str>, heads: usize, slot_rows: usize) -> Result<H3Attn, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let props = unsafe { dev.instance.get_physical_device_properties(dev.physical) };
		if props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
			return Err("the Vulkan adapter is not a discrete card".into());
		}
		let qf = unsafe {
			dev.instance
				.get_physical_device_queue_family_properties(dev.physical)
		};
		let tick = match qf.get(dev.qfamily as usize) {
			Some(f) if f.timestamp_valid_bits > 0 => props.limits.timestamp_period as f64 * 1e-9,
			_ => 0.0,
		};
		let d = &dev.device;
		let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..4)
			.map(|b| vk::DescriptorSetLayoutBinding {
				binding: b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				..Default::default()
			})
			.collect();
		let e = |e: vk::Result| format!("h3 attention: {e}");
		let dsl = unsafe {
			d.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo {
					binding_count: bindings.len() as u32,
					p_bindings: bindings.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(e)?;
		let pcr = vk::PushConstantRange {
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			offset: 0,
			size: 16,
		};
		let layout = unsafe {
			d.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo {
					set_layout_count: 1,
					p_set_layouts: &dsl,
					push_constant_range_count: 1,
					p_push_constant_ranges: &pcr,
					..Default::default()
				},
				None,
			)
		}
		.map_err(e)?;
		// `QWEN_H3_ATTN_SPV=FILE` (kernel experiments): a build from disk
		let spv = match std::env::var_os("QWEN_H3_ATTN_SPV") {
			Some(f) => std::fs::read(&f).map_err(|e| format!("{}: {e}", f.to_string_lossy()))?,
			None => quant::packed!("h3_attn.spv").to_vec(),
		};
		let sm = create_sm(d, &load_spv(&spv))?;
		let main = CString::new("main").unwrap();
		let pl = compute_pipeline_sg32(d, layout, sm, &main)?;
		let sizes = [vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: (4 * H3_ATTN_SLOTS) as u32,
		}];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: H3_ATTN_SLOTS as u32,
					pool_size_count: 1,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(e)?;
		let qpool = unsafe {
			d.create_query_pool(
				&vk::QueryPoolCreateInfo {
					query_type: vk::QueryType::TIMESTAMP,
					query_count: (2 * H3_ATTN_SLOTS) as u32,
					..Default::default()
				},
				None,
			)
		}
		.map_err(e)?;
		Ok(H3Attn {
			dev,
			dsl,
			layout,
			sm,
			pl,
			pool,
			qpool,
			tick,
			row: heads * HD,
			slot_rows,
			cap: 0,
			k_host: None,
			v_host: None,
			k_dev: None,
			v_dev: None,
			slots: Vec::new(),
		})
	}

	/// VRAM a job of `nk` keys takes.
	pub fn vram_need(&self, nk: usize) -> u64 {
		(2 * nk * self.row * 2 + H3_ATTN_SLOTS * self.slot_rows * self.row * (4 + 2)) as u64
	}

	/// The buffers for a job of `nk` keys (VRAM and mapped host memory);
	/// `Err` when the card cannot spare the VRAM.
	pub fn reserve(&mut self, nk: usize) -> Result<(), String> {
		self.release();
		let need = self.vram_need(nk);
		let cap = cap_bytes();
		if need > cap {
			return Err(format!(
				"{} MB of VRAM needed, QWEN_H3_ATTN_GPU_MB allows {}",
				need >> 20,
				cap >> 20
			));
		}
		if let Some((budget, usage)) = self.dev.vram_budget() {
			let free = budget.saturating_sub(usage);
			if need + (MARGIN_MB << 20) > free {
				return Err(format!(
					"{} MB of VRAM needed, {} MB free of the budget (keeping {MARGIN_MB} MB for other applications)",
					need >> 20,
					free >> 20
				));
			}
		}
		let r = self.alloc(nk);
		if r.is_err() {
			self.release();
		}
		r
	}

	fn alloc(&mut self, nk: usize) -> Result<(), String> {
		use vk::{BufferUsageFlags as U, MemoryPropertyFlags as M};
		let kv = nk * self.row * 2;
		let (q, o) = (self.slot_rows * self.row * 4, self.slot_rows * self.row * 2);
		let local = |s: usize, u: U| -> Result<Mem, String> {
			let (buf, mem) = self
				.dev
				.try_create_gpu_buffer(s as u64, u, M::DEVICE_LOCAL)
				.ok_or("h3 attention: out of device memory")?;
			Ok(Mem {
				buf,
				mem,
				ptr: std::ptr::null_mut(),
				size: s,
			})
		};
		let host = |s: usize, read: bool| -> Result<Mem, String> {
			let (buf, mem) = if read {
				self.dev.try_create_readback_buffer(s as u64)
			} else {
				self.dev.try_create_gpu_buffer(
					s as u64,
					U::TRANSFER_SRC,
					M::HOST_VISIBLE | M::HOST_COHERENT,
				)
			}
			.ok_or("h3 attention: no host-visible memory")?;
			Ok(Mem {
				buf,
				mem,
				ptr: self.dev.map(mem, s) as *mut u8,
				size: s,
			})
		};
		self.k_host = Some(host(kv, false)?);
		self.v_host = Some(host(kv, false)?);
		self.k_dev = Some(local(kv, U::STORAGE_BUFFER | U::TRANSFER_DST)?);
		self.v_dev = Some(local(kv, U::STORAGE_BUFFER | U::TRANSFER_DST)?);
		self.cap = nk;
		for _ in 0..H3_ATTN_SLOTS {
			let q_host = host(q, false)?;
			let q_dev = local(q, U::STORAGE_BUFFER | U::TRANSFER_DST)?;
			let o_dev = local(o, U::STORAGE_BUFFER | U::TRANSFER_SRC)?;
			let o_host = host(o, true)?;
			let d = &self.dev.device;
			let set = unsafe {
				d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
					descriptor_pool: self.pool,
					descriptor_set_count: 1,
					p_set_layouts: &self.dsl,
					..Default::default()
				})
			}
			.map_err(|e| format!("h3 attention descriptor set: {e}"))?[0];
			let bufs = [
				&q_dev,
				self.k_dev.as_ref().unwrap(),
				self.v_dev.as_ref().unwrap(),
				&o_dev,
			];
			let infos: Vec<vk::DescriptorBufferInfo> = bufs
				.iter()
				.map(|m| vk::DescriptorBufferInfo {
					buffer: m.buf,
					offset: 0,
					range: m.size as u64,
				})
				.collect();
			let writes: Vec<vk::WriteDescriptorSet> = (0..4)
				.map(|i| vk::WriteDescriptorSet {
					dst_set: set,
					dst_binding: i as u32,
					descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
					descriptor_count: 1,
					p_buffer_info: &infos[i],
					..Default::default()
				})
				.collect();
			unsafe { d.update_descriptor_sets(&writes, &[]) };
			let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }
				.map_err(|e| format!("h3 attention fence: {e}"))?;
			self.slots.push(Slot {
				q_host,
				q_dev,
				o_dev,
				o_host,
				set,
				fence,
				cmd: None,
				nq: 0,
			});
		}
		Ok(())
	}

	/// Keys the reserved buffers hold (0: none).
	pub fn capacity(&self) -> usize {
		self.cap
	}

	/// Query rows a submission takes.
	pub fn slot_rows(&self) -> usize {
		self.slot_rows
	}

	/// Copy K / V rows (`[n][row]` f16) to keys `k0 .. k0 + n` of the
	/// staging buffers; they go to VRAM with the next submission that asks
	/// for it. Nothing in flight may read the staging buffers.
	pub fn stage_kv(&mut self, k0: usize, k: &[u16], v: &[u16]) {
		let (kh, vh) = (self.k_host.as_ref().unwrap(), self.v_host.as_ref().unwrap());
		assert!(k.len() == v.len() && (k0 * self.row + k.len()) * 2 <= kh.size);
		unsafe {
			let o = k0 * self.row;
			std::ptr::copy_nonoverlapping(k.as_ptr(), (kh.ptr as *mut u16).add(o), k.len());
			std::ptr::copy_nonoverlapping(v.as_ptr(), (vh.ptr as *mut u16).add(o), v.len());
		}
	}

	/// The query rows of slot `s` (`[slot_rows][row]` f32, write only:
	/// host-visible memory without caching).
	pub fn queries(&mut self, s: usize) -> &mut [f32] {
		let m = &self.slots[s].q_host;
		unsafe { std::slice::from_raw_parts_mut(m.ptr as *mut f32, m.size / 4) }
	}

	/// Submit slot `s`: its first `nq` query rows over keys `0 .. nk`,
	/// scores scaled by `scale`; with `upload`, the staged keys `0 .. nk`
	/// go to VRAM first. The slot must be idle ([`H3Attn::wait`]).
	pub fn submit(
		&mut self,
		s: usize,
		nq: usize,
		nk: usize,
		scale: f32,
		upload: bool,
	) -> Result<(), String> {
		if nq == 0 || nq > self.slot_rows || nk == 0 || nk > self.cap || self.slots[s].cmd.is_some()
		{
			return Err(format!(
				"h3 attention: slot {s}: {nq} queries over {nk} keys: bad submission"
			));
		}
		let d = &self.dev.device;
		let row = self.row;
		let sl = &self.slots[s];
		let (kh, vh, kd, vd) = (
			self.k_host.as_ref().unwrap(),
			self.v_host.as_ref().unwrap(),
			self.k_dev.as_ref().unwrap(),
			self.v_dev.as_ref().unwrap(),
		);
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
		let copy = |cmd, src: vk::Buffer, dst: vk::Buffer, size: usize| unsafe {
			d.cmd_copy_buffer(
				cmd,
				src,
				dst,
				&[vk::BufferCopy {
					src_offset: 0,
					dst_offset: 0,
					size: size as u64,
				}],
			)
		};
		let pc: Vec<u8> = [nq as u32, nk as u32, scale.to_bits(), (row / 4) as u32]
			.iter()
			.flat_map(|x| x.to_le_bytes())
			.collect();
		let (pl, layout, set, qp, q0) = (self.pl, self.layout, sl.set, self.qpool, 2 * s as u32);
		let cmd = self
			.dev
			.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, |cmd| unsafe {
				d.cmd_reset_query_pool(cmd, qp, q0, 2);
				// after the host's writes and the earlier submissions' reads
				barrier(
					cmd,
					P::HOST | P::COMPUTE_SHADER | P::TRANSFER,
					P::TRANSFER,
					A::HOST_WRITE | A::SHADER_READ | A::TRANSFER_READ,
					A::TRANSFER_READ | A::TRANSFER_WRITE,
				);
				// written once the earlier submissions are done: the start of
				// this one's own time on the card
				d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, q0);
				if upload {
					copy(cmd, kh.buf, kd.buf, nk * row * 2);
					copy(cmd, vh.buf, vd.buf, nk * row * 2);
				}
				copy(cmd, sl.q_host.buf, sl.q_dev.buf, nq * row * 4);
				barrier(
					cmd,
					P::TRANSFER,
					P::COMPUTE_SHADER,
					A::TRANSFER_WRITE,
					A::SHADER_READ,
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
				d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
				d.cmd_dispatch(cmd, nq.div_ceil(48) as u32, (row / HD) as u32, 1);
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::TRANSFER,
					A::SHADER_WRITE,
					A::TRANSFER_READ,
				);
				copy(cmd, sl.o_dev.buf, sl.o_host.buf, nq * row * 2);
				barrier(cmd, P::TRANSFER, P::HOST, A::TRANSFER_WRITE, A::HOST_READ);
				d.cmd_write_timestamp(cmd, P::BOTTOM_OF_PIPE, qp, q0 + 1);
			});
		let r = unsafe {
			d.reset_fences(&[sl.fence]).ok();
			d.queue_submit(
				self.dev.queue,
				&[vk::SubmitInfo {
					command_buffer_count: 1,
					p_command_buffers: &cmd,
					..Default::default()
				}],
				sl.fence,
			)
		};
		if let Err(e) = r {
			unsafe { d.free_command_buffers(self.dev.cmd_pool, &[cmd]) };
			return Err(format!("h3 attention: {e}"));
		}
		let sl = &mut self.slots[s];
		sl.cmd = Some(cmd);
		sl.nq = nq;
		Ok(())
	}

	/// Whether slot `s` has nothing in flight (finished or never submitted).
	pub fn done(&self, s: usize) -> bool {
		let sl = &self.slots[s];
		sl.cmd.is_none()
			|| matches!(
				unsafe { self.dev.device.get_fence_status(sl.fence) },
				Ok(true)
			)
	}

	/// Whether slot `s` holds a submission (finished or not).
	pub fn busy(&self, s: usize) -> bool {
		self.slots[s].cmd.is_some()
	}

	/// Wait for slot `s`'s submission (none: Ok); its output rows are then
	/// [`H3Attn::output`].
	pub fn wait(&mut self, s: usize) -> Result<(), String> {
		let Some(cmd) = self.slots[s].cmd.take() else {
			return Ok(());
		};
		let r = self.dev.wait_fence_spin(self.slots[s].fence, 50);
		unsafe {
			self.dev
				.device
				.free_command_buffers(self.dev.cmd_pool, &[cmd])
		};
		r.map_err(|e| format!("h3 attention: {e}"))
	}

	/// Seconds slot `s`'s last submission took on the card, transfers
	/// included and the earlier submissions' time not (after
	/// [`H3Attn::wait`]; `None` without timestamps).
	pub fn seconds(&self, s: usize) -> Option<f64> {
		if self.tick == 0.0 {
			return None;
		}
		let mut t = [0u64; 2];
		unsafe {
			self.dev.device.get_query_pool_results(
				self.qpool,
				2 * s as u32,
				&mut t,
				vk::QueryResultFlags::TYPE_64,
			)
		}
		.ok()?;
		Some(t[1].wrapping_sub(t[0]) as f64 * self.tick)
	}

	/// Slot `s`'s output rows of its last submission (`[nq][row]` f16).
	pub fn output(&self, s: usize) -> &[u16] {
		let sl = &self.slots[s];
		unsafe { std::slice::from_raw_parts(sl.o_host.ptr as *const u16, sl.nq * self.row) }
	}

	/// Free the job's buffers (waits for anything in flight).
	pub fn release(&mut self) {
		for s in 0..self.slots.len() {
			let _ = self.wait(s);
		}
		let d = &self.dev.device;
		let free = |m: Mem| unsafe {
			d.destroy_buffer(m.buf, None);
			d.free_memory(m.mem, None);
		};
		for sl in self.slots.drain(..) {
			unsafe { d.destroy_fence(sl.fence, None) };
			for m in [sl.q_host, sl.q_dev, sl.o_dev, sl.o_host] {
				free(m);
			}
		}
		for m in [
			self.k_host.take(),
			self.v_host.take(),
			self.k_dev.take(),
			self.v_dev.take(),
		]
		.into_iter()
		.flatten()
		{
			free(m);
		}
		unsafe {
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
		}
		self.cap = 0;
	}
}

impl Drop for H3Attn {
	fn drop(&mut self) {
		self.release();
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			d.destroy_pipeline(self.pl, None);
			d.destroy_shader_module(self.sm, None);
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_query_pool(self.qpool, None);
			d.destroy_pipeline_layout(self.layout, None);
			d.destroy_descriptor_set_layout(self.dsl, None);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The kernel against an f64 softmax(Q K^T s) V on the default
	/// (discrete) adapter: 4 heads, 101 queries (not a multiple of 48), 150
	/// keys (not a multiple of 64), large values as H3's rows have; then the
	/// time of 512 queries over 4200 keys at 56 heads.
	#[test]
	fn h3_attention_matches_reference() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let Ok(mut a) = H3Attn::new(pick.as_deref(), 4, 128) else {
			return;
		};
		let (heads, nq, nk) = (4usize, 101usize, 150usize);
		let row = heads * HD;
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut val = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
		};
		let q: Vec<f32> = (0..nq * row).map(|_| val()).collect();
		let k: Vec<u16> = (0..nk * row).map(|_| quant::f32_to_fp16(val())).collect();
		let v: Vec<u16> = (0..nk * row)
			.map(|_| quant::f32_to_fp16(val() * 40.0))
			.collect();
		let scale = 1.0 / (HD as f32).sqrt();
		a.reserve(nk).unwrap();
		a.stage_kv(0, &k[..77 * row], &v[..77 * row]);
		a.stage_kv(77, &k[77 * row..], &v[77 * row..]);
		// two submissions: queries 0..64 with the upload, then 64..101
		a.queries(0)[..64 * row].copy_from_slice(&q[..64 * row]);
		a.queries(1)[..(nq - 64) * row].copy_from_slice(&q[64 * row..]);
		a.submit(0, 64, nk, scale, true).unwrap();
		a.submit(1, nq - 64, nk, scale, false).unwrap();
		a.wait(0).unwrap();
		a.wait(1).unwrap();
		let mut o = a.output(0).to_vec();
		o.extend_from_slice(a.output(1));
		let f = |s: &[u16], i: usize| quant::fp16_to_f32(s[i]) as f64;
		let mut worst = 0f64;
		for t in 0..nq {
			for h in 0..heads {
				let sc: Vec<f64> = (0..nk)
					.map(|u| {
						(0..HD)
							.map(|d| q[t * row + h * HD + d] as f64 * f(&k, u * row + h * HD + d))
							.sum::<f64>() * scale as f64
					})
					.collect();
				let m = sc.iter().cloned().fold(f64::MIN, f64::max);
				let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
				for d in 0..HD {
					let want: f64 = (0..nk)
						.map(|u| (sc[u] - m).exp() / z * f(&v, u * row + h * HD + d))
						.sum();
					let got = f(&o, t * row + h * HD + d);
					worst = worst.max((got - want).abs() / (1.0 + want.abs()));
					assert!(
						(got - want).abs() < 2e-3 * (1.0 + want.abs()),
						"query {t} head {h} dim {d}: {got} vs {want}"
					);
				}
			}
		}
		eprintln!("h3 attention: max rel error {worst:.2e}");
	}

	/// The time of 512 queries over 4200 keys at 56 heads (a 4 MP job's
	/// shape), 3 slots in flight; `QWEN_H3_ATTN_SPV` measures a variant.
	#[test]
	#[ignore]
	fn h3_attention_bench() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").ok();
		let (heads, nq, nk) = (56usize, 512usize, 4200usize);
		let Ok(mut a) = H3Attn::new(pick.as_deref(), heads, nq) else {
			return;
		};
		a.reserve(nk).unwrap();
		let scale = 1.0 / (HD as f32).sqrt();
		let run = |a: &mut H3Attn, reps: usize| {
			let t = std::time::Instant::now();
			for i in 0..reps {
				let s = i % H3_ATTN_SLOTS;
				a.wait(s).unwrap();
				a.submit(s, nq, nk, scale, i == 0).unwrap();
			}
			for s in 0..H3_ATTN_SLOTS {
				a.wait(s).unwrap();
			}
			t.elapsed().as_secs_f64() / reps as f64
		};
		run(&mut a, 3);
		let s = run(&mut a, 24);
		eprintln!(
			"h3 attention: last submission {:.1} ms on the card",
			a.seconds(0).unwrap_or(0.0) * 1e3
		);
		eprintln!(
			"h3 attention: 512 x 4200 x 56 heads: {:.1} ms, {:.2} TFLOP/s",
			s * 1e3,
			4.0 * (nq * nk * heads * HD) as f64 / s / 1e12
		);
	}
}
