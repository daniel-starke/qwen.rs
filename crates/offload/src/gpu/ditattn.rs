//! The Qwen-Image transformer's attention on a Vulkan device (the iGPU),
//! for a share of each block's query chunks.
//!
//! WHY
//! ---
//! `imagegen::qimage::npudit` runs the transformer's per-token work on the NPU and
//! the attention (target rows over [prefix, target] keys) on the cores,
//! chunk by chunk, overlapped. In an edit the reference images' rows make
//! the attention the longer part (1 MP, one reference: 132 s of the
//! cores against ~19 s of NPU waits), so the idle iGPU takes the last
//! chunks of each block (`dit_attn_npu.glsl`, an online softmax over tiles
//! of 64 keys) while the cores run the first ones.
//!
//! As in [`VisAttn`](super::VisAttn) the tensors stay where the NPU put
//! them: its host buffers are imported on first use
//! (`VK_EXT_external_memory_host`), queries and K / V are read and the
//! output rows written in place, as f16. [`DitAttn::forget`] drops the
//! imports before the buffers are freed.

use super::*;

/// f16 values of a position: 32 heads of 128.
const ROW: usize = 32 * 128;
/// Queries per workgroup (the kernel's `G`).
const TQ: usize = 8;
/// Descriptor sets the pool holds (one per (q, k, v, o) buffer set).
const SETS: usize = 16;

struct Imported {
	base: usize,
	len: usize,
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
}

/// See the module docs.
pub struct DitAttn {
	dev: GpuDevice,
	sm: vk::ShaderModule,
	pl: vk::Pipeline,
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	imports: Vec<Imported>,
	align: usize,
	/// descriptor sets by the (q, k, v, o) buffers' addresses
	sets: Vec<([usize; 4], vk::DescriptorSet)>,
	pending: Option<vk::CommandBuffer>,
}

// SAFETY: used from one thread at a time.
unsafe impl Send for DitAttn {}

impl DitAttn {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU).
	pub fn new(pick: Option<&str>) -> Result<DitAttn, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let align = dev
			.host_import_align()
			.ok_or("the Vulkan device has no VK_EXT_external_memory_host")?;
		let main = CString::new("main").unwrap();
		// `QWEN_DIT_ATTN_SPV=FILE` (kernel experiments): a build from disk
		let spv = match std::env::var_os("QWEN_DIT_ATTN_SPV") {
			Some(f) => std::fs::read(&f).map_err(|e| format!("{}: {e}", f.to_string_lossy()))?,
			None => quant::packed!("dit_attn_npu.spv").to_vec(),
		};
		let sm = create_sm(&dev.device, &load_spv(&spv))?;
		let pl = compute_pipeline(&dev.device, dev.aux_pipeline_layout, sm, &main)?;
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: (4 * SETS) as u32,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: SETS as u32,
			},
		];
		let pool = unsafe {
			dev.device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: SETS as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("dit attention descriptor pool: {e}"))?;
		let params = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		Ok(DitAttn {
			dev,
			sm,
			pl,
			pool,
			params,
			imports: Vec::new(),
			align,
			sets: Vec::new(),
			pending: None,
		})
	}

	/// Submit the attention of queries `q0 .. q0 + nq` of `q` (`[*][32][128]`
	/// f16) over keys `0 .. nk` of `k` / `v` (`[nk][32][128]` f16) with
	/// scores scaled by `scale`, into rows `o0 .. o0 + nq` of `o`; one
	/// submission in flight, finished by [`DitAttn::end`]. The four buffers
	/// must be host memory that stays at its address until
	/// [`DitAttn::forget`] (they are imported on first use), and nothing may
	/// write `q`, `k`, `v` or the output rows until `end`.
	#[allow(clippy::too_many_arguments)]
	pub fn begin(
		&mut self,
		q: &[u16],
		q0: usize,
		nq: usize,
		k: &[u16],
		v: &[u16],
		nk: usize,
		o: &[u16],
		o0: usize,
		scale: f32,
	) -> Result<(), String> {
		if self.pending.is_some() {
			return Err("dit attention: a submission is already in flight".into());
		}
		if nq == 0
			|| nk == 0
			|| q.len() < (q0 + nq) * ROW
			|| k.len() < nk * ROW
			|| v.len() < nk * ROW
			|| o.len() < (o0 + nq) * ROW
		{
			return Err(format!(
				"dit attention: {nq} queries at {q0} over {nk} keys: bad sizes"
			));
		}
		let key = [q, k, v, o].map(|s| s.as_ptr() as usize);
		let set = match self.sets.iter().find(|e| e.0 == key) {
			Some(e) => e.1,
			None => {
				if self.sets.len() == SETS {
					unsafe {
						self.dev
							.device
							.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
					}
					.map_err(|e| format!("dit attention: {e}"))?;
					self.sets.clear();
				}
				let mut b = [(vk::Buffer::null(), 0u64, 0u64); 4];
				for (i, s) in [q, k, v, o].iter().enumerate() {
					let (buf, off) = self.import(s.as_ptr() as usize, s.len() * 2)?;
					b[i] = (buf, off, (s.len() * 2) as u64);
				}
				let set = self.make_set(&b)?;
				self.sets.push((key, set));
				set
			}
		};
		let mut pr = [0u32; 16];
		pr[0] = q0 as u32;
		pr[1] = nq as u32;
		pr[2] = nk as u32;
		pr[3] = scale.to_bits();
		pr[4] = o0 as u32;
		let pb: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
		let tq = std::env::var("QWEN_DIT_ATTN_G")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(TQ);
		let d = &self.dev.device;
		let (pl, layout, params) = (self.pl, self.dev.aux_pipeline_layout, self.params.0);
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
				d.cmd_dispatch(cmd, nq.div_ceil(tq) as u32, 32, 1);
				// the NPU reads the rows from host memory after the fence
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("dit attention: {e}"))?;
		self.pending = Some(cmd);
		Ok(())
	}

	/// Wait for [`DitAttn::begin`]'s submission (nothing in flight: Ok).
	pub fn end(&mut self) -> Result<(), String> {
		match self.pending.take() {
			Some(cmd) => self
				.dev
				.wait_submitted(cmd)
				.map_err(|e| format!("dit attention: {e}")),
			None => Ok(()),
		}
	}

	/// Drop every import (before the host buffers are freed).
	pub fn forget(&mut self) {
		let _ = self.end();
		let d = &self.dev.device;
		unsafe {
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
			for e in self.imports.drain(..) {
				hostimp::free_import(d, e.buf, e.mem);
			}
		}
		self.sets.clear();
	}

	/// A buffer over host bytes `[ptr, ptr + len)` and `ptr`'s offset in it,
	/// importing the covering pages on first use.
	fn import(&mut self, ptr: usize, len: usize) -> Result<(vk::Buffer, u64), String> {
		let a = self.align;
		let (base, end) = (ptr & !(a - 1), (ptr + len + a - 1) & !(a - 1));
		if let Some(e) = self
			.imports
			.iter()
			.find(|e| e.base <= base && end <= e.base + e.len)
		{
			return Ok((e.buf, (ptr - e.base) as u64));
		}
		let (buf, mem) = self.dev.import_host_range(base, end - base)?;
		self.imports.push(Imported {
			base,
			len: end - base,
			buf,
			mem,
		});
		Ok((buf, (ptr - base) as u64))
	}

	/// A descriptor set with (buffer, offset, range) on bindings 0..4 and
	/// the parameter block on 8.
	fn make_set(&self, b4: &[(vk::Buffer, u64, u64); 4]) -> Result<vk::DescriptorSet, String> {
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("dit attention descriptor set: {e}"))?[0];
		let info = |b: vk::Buffer, off: u64, range: u64| vk::DescriptorBufferInfo {
			buffer: b,
			offset: off,
			range,
		};
		let mut infos = [info(self.params.0, 0, 64); 5];
		for (i, x) in b4.iter().enumerate() {
			infos[i] = info(x.0, x.1, x.2);
		}
		let writes: Vec<vk::WriteDescriptorSet> = (0..5)
			.map(|i| vk::WriteDescriptorSet {
				dst_set: set,
				dst_binding: if i < 4 { i as u32 } else { 8 },
				descriptor_type: if i < 4 {
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

impl Drop for DitAttn {
	fn drop(&mut self) {
		self.forget();
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			d.destroy_pipeline(self.pl, None);
			d.destroy_shader_module(self.sm, None);
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_buffer(self.params.0, None);
			d.free_memory(self.params.1, None);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The kernel against an f64 softmax(Q K^T s) V, on the default device:
	/// 37 queries (not a multiple of 32) from row 5, 150 keys (not a
	/// multiple of 64), output at row 3; and the time at 256 queries over
	/// 8192 keys.
	#[test]
	fn dit_attention_matches_reference() {
		let Ok(mut da) = DitAttn::new(std::env::var("QWEN_TEST_ADAPTER").ok().as_deref()) else {
			return;
		};
		let (q0, nq, nk, o0) = (5usize, 37usize, 150usize, 3usize);
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut val = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
		};
		// 4 KiB aligned host vectors (imports take whole pages)
		let page = |n: usize| -> Vec<u16> {
			let mut v = vec![0u16; n + 2048];
			let off = (v.as_ptr() as usize).next_multiple_of(4096) - v.as_ptr() as usize;
			v.drain(..off / 2);
			v.truncate(n);
			v
		};
		let mut q = page((q0 + nq) * ROW);
		let mut k = page(nk * ROW);
		let mut v = page(nk * ROW);
		for x in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
			*x = quant::f32_to_fp16(val());
		}
		let o = page((o0 + nq) * ROW);
		let scale = 1.0 / (128f32).sqrt();
		da.begin(&q, q0, nq, &k, &v, nk, &o, o0, scale).unwrap();
		da.end().unwrap();
		let f = |s: &[u16], i: usize| quant::fp16_to_f32(s[i]) as f64;
		let mut worst = 0f64;
		for t in 0..nq {
			for h in [0usize, 7, 31] {
				let sc: Vec<f64> = (0..nk)
					.map(|u| {
						(0..128)
							.map(|d| {
								f(&q, (q0 + t) * ROW + h * 128 + d) * f(&k, u * ROW + h * 128 + d)
							})
							.sum::<f64>() * scale as f64
					})
					.collect();
				let m = sc.iter().cloned().fold(f64::MIN, f64::max);
				let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
				for d in 0..128 {
					let want: f64 = (0..nk)
						.map(|u| (sc[u] - m).exp() / z * f(&v, u * ROW + h * 128 + d))
						.sum();
					let got = f(&o, (o0 + t) * ROW + h * 128 + d);
					worst = worst.max((got - want).abs());
					assert!(
						(got - want).abs() < 2e-3 * (1.0 + want.abs()),
						"query {t} head {h} dim {d}: {got} vs {want}"
					);
				}
			}
		}
		assert!(o[..o0 * ROW].iter().all(|&x| x == 0));
		eprintln!("dit attention: max abs error {worst:.2e}");
		da.forget();
		// time: 256 queries over 8192 keys
		let (nq, nk) = (256usize, 8192usize);
		let q = page(nq * ROW);
		let k = page(nk * ROW);
		let o = page(nq * ROW);
		da.begin(&q, 0, nq, &k, &k, nk, &o, 0, scale).unwrap();
		da.end().unwrap();
		let t = std::time::Instant::now();
		for _ in 0..4 {
			da.begin(&q, 0, nq, &k, &k, nk, &o, 0, scale).unwrap();
			da.end().unwrap();
		}
		let s = t.elapsed().as_secs_f64() / 4.0;
		eprintln!(
			"dit attention: 256 x 8192: {:.1} ms, {:.2} TFLOP/s",
			s * 1e3,
			4.0 * (nq * nk * ROW) as f64 / s / 1e12
		);
	}
}
