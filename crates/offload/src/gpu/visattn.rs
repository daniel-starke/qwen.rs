//! The attention of the NPU engine's vision tower on a Vulkan device (the
//! iGPU).
//!
//! WHY
//! ---
//! The tower's attention is over every patch of an image, so it has no
//! static shape and cannot be an NPU graph; everything else of a block can
//! (`npubuild::vis`). On the cores it is most of an image's time (4000
//! patches: 10.5 s of attention against 1.5 s of NPU jobs). Here it runs as
//! four kernels per block: 2-D RoPE of Q and K into f32 buffers laid out per
//! head (`vis_rope_npu.glsl`), then per tile of queries the scores
//! (`vis_qk_npu.glsl`), their exponentials (`vis_sm_npu.glsl`) and P V
//! (`vis_pv_npu.glsl`). 960 patches: 8 ms per block (the cores: 19), 4096:
//! ~160 ms (390); `visbench` checks and times it.
//!
//! As in [`DeltaScan`](super::DeltaScan) the tensors stay where the NPU put
//! them: its host buffers are imported once (`VK_EXT_external_memory_host`),
//! `qkv` is read and the output rows are written in place, as f16.

use super::*;

/// Dims of a head (the kernels' `HD4 * 4`).
const HD: usize = 72;
/// Bytes of the score buffer at most (a tile of queries x every key).
const SC_BYTES: usize = 128 << 20;

struct Imported {
	base: usize,
	len: usize,
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
}

/// See the module docs.
pub struct VisAttn {
	dev: GpuDevice,
	sm: [vk::ShaderModule; 4],
	pl: [vk::Pipeline; 4],
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	imports: Vec<Imported>,
	align: usize,
	/// rotated Q and K, V (all three f32, per head), a tile's scores and
	/// their rows' largest one and sum (in parts), the cos / sin table (host
	/// visible, mapped at `cs_map`), for `cap` patches
	bufs: [(vk::Buffer, vk::DeviceMemory); 6],
	cs_map: *mut f32,
	cap: usize,
	/// the positions the table holds
	hw: Vec<(u32, u32)>,
	/// (qkv, out, patches) the descriptor sets were made for
	bound: (usize, usize, usize),
	sets: [vk::DescriptorSet; 2],
}

const Q: usize = 0;
const K: usize = 1;
const V: usize = 2;
const SC: usize = 3;
const SI: usize = 4;
const CS: usize = 5;

// SAFETY: used from one thread at a time (the NPU engine's).
unsafe impl Send for VisAttn {}

impl VisAttn {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU).
	pub fn new(pick: Option<&str>) -> Result<VisAttn, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let align = dev
			.host_import_align()
			.ok_or("the Vulkan device has no VK_EXT_external_memory_host")?;
		let main = CString::new("main").unwrap();
		let spv: [&[u8]; 4] = [
			include_bytes!("../shaders/vis_rope_npu.spv"),
			include_bytes!("../shaders/vis_qk_npu.spv"),
			include_bytes!("../shaders/vis_sm_npu.spv"),
			include_bytes!("../shaders/vis_pv_npu.spv"),
		];
		let mut sm = [vk::ShaderModule::null(); 4];
		let mut pl = [vk::Pipeline::null(); 4];
		for i in 0..4 {
			sm[i] = create_sm(&dev.device, &load_spv(spv[i]))?;
			pl[i] = compute_pipeline(&dev.device, dev.aux_pipeline_layout, sm[i], &main)?;
		}
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 12,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 2,
			},
		];
		let pool = unsafe {
			dev.device.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: 2,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("vision attention descriptor pool: {e}"))?;
		let params = dev.create_gpu_buffer(
			64,
			vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		);
		Ok(VisAttn {
			dev,
			sm,
			pl,
			pool,
			params,
			imports: Vec::new(),
			align,
			bufs: [(vk::Buffer::null(), vk::DeviceMemory::null()); 6],
			cs_map: ptr::null_mut(),
			cap: 0,
			hw: Vec::new(),
			bound: (0, 0, 0),
			sets: [vk::DescriptorSet::null(); 2],
		})
	}

	/// Queries per tile for `n` patches (a multiple of 8).
	fn tile(n: usize) -> usize {
		let nq = n.next_multiple_of(8);
		(SC_BYTES / (64 * nq) / 8 * 8).clamp(8, nq)
	}

	fn free_bufs(&mut self) {
		let d = &self.dev.device;
		unsafe {
			if !self.cs_map.is_null() {
				d.unmap_memory(self.bufs[CS].1);
				self.cs_map = ptr::null_mut();
			}
			for b in self.bufs.iter_mut().filter(|b| b.0 != vk::Buffer::null()) {
				d.destroy_buffer(b.0, None);
				d.free_memory(b.1, None);
				b.0 = vk::Buffer::null();
			}
		}
		self.cap = 0;
	}

	/// The device buffers for `n` patches of `heads` heads.
	fn ensure(&mut self, n: usize, heads: usize) -> Result<(), String> {
		if n <= self.cap {
			return Ok(());
		}
		self.free_bufs();
		let nq = n.next_multiple_of(8);
		let tile = Self::tile(n);
		let qb = heads * nq * HD * 4;
		let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		for (i, sz, props) in [
			(Q, qb, local),
			(K, qb, local),
			(V, qb, local),
			(SC, heads * tile * nq * 4, local),
			(SI, heads * tile * 64, local),
			(CS, n * HD * 4, host),
		] {
			self.bufs[i] = self
				.dev
				.try_create_gpu_buffer(sz as u64, vk::BufferUsageFlags::STORAGE_BUFFER, props)
				.ok_or(format!("vision attention: no device memory for {sz} bytes"))?;
		}
		self.cs_map = self.dev.map(self.bufs[CS].1, n * HD * 4) as *mut f32;
		self.cap = n;
		self.hw.clear();
		self.bound = (0, 0, 0);
		Ok(())
	}

	/// The cos / sin table of patches at (row, column) `hw`: per patch 36
	/// cosines then 36 sines; half the rotary channels carry the row, half
	/// the column (`vision::tower::rope_tables`).
	fn table(&mut self, hw: &[(u32, u32)]) {
		if self.hw == hw {
			return;
		}
		let (half, quarter) = (HD / 2, HD / 4);
		let inv: Vec<f64> = (0..quarter)
			.map(|k| 1.0 / 10000f64.powf(2.0 * k as f64 / half as f64))
			.collect();
		// SAFETY: the mapping holds `cap * HD` floats and `hw.len() <= cap`
		let t = unsafe { std::slice::from_raw_parts_mut(self.cs_map, hw.len() * HD) };
		for (s, &(hp, wp)) in hw.iter().enumerate() {
			for j in 0..half {
				let (pos, k) = if j < quarter {
					(hp, j)
				} else {
					(wp, j - quarter)
				};
				let ang = pos as f64 * inv[k];
				t[s * HD + j] = ang.cos() as f32;
				t[s * HD + half + j] = ang.sin() as f32;
			}
		}
		self.hw = hw.to_vec();
	}

	/// One block's attention over the `n = hw.len()` patches of an image:
	/// `qkv` `[n][3][heads][72]` f16 (per patch q, k, v, before RoPE) ->
	/// `out` `[n][heads][72]` f16. Both must be host memory that outlives
	/// this object at an unchanging address (they are imported on first
	/// use), aligned to 4 KiB.
	pub fn attend(
		&mut self,
		heads: usize,
		hw: &[(u32, u32)],
		qkv: &[u8],
		out: &mut [u8],
	) -> Result<(), String> {
		let n = hw.len();
		if n == 0 || qkv.len() != n * 3 * heads * HD * 2 || out.len() != n * heads * HD * 2 {
			return Err(format!(
				"vision attention: {n} patches, {} and {} bytes",
				qkv.len(),
				out.len()
			));
		}
		self.ensure(n, heads)?;
		self.table(hw);
		let key = (qkv.as_ptr() as usize, out.as_ptr() as usize, n);
		if self.bound != key {
			// (the descriptors' ranges are the tensors' sizes)
			let (xb, xo) = self.import(qkv.as_ptr() as usize, qkv.len())?;
			let (ob, oo) = self.import(out.as_ptr() as usize, out.len())?;
			unsafe {
				self.dev
					.device
					.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
			}
			.map_err(|e| format!("vision attention: {e}"))?;
			let b = |i: usize| (self.bufs[i].0, 0, vk::WHOLE_SIZE);
			// pass 1: 0 x, 1 cs, 2 q, 3 k, 4 v; passes 2 to 4: 0 q, 1 k, 2 v, 3 o, 4 max / sum, 5 scores
			self.sets = [
				self.make_set(&[(xb, xo, qkv.len() as u64), b(CS), b(Q), b(K), b(V), b(Q)])?,
				self.make_set(&[b(Q), b(K), b(V), (ob, oo, out.len() as u64), b(SI), b(SC)])?,
			];
			self.bound = key;
		}
		let nq = n.next_multiple_of(8);
		let tile = Self::tile(n);
		let d = &self.dev.device;
		let (pl, layout, params, sets) = (
			self.pl,
			self.dev.aux_pipeline_layout,
			self.params.0,
			self.sets,
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
		let set_params = |cmd, t0: usize, tq: usize| unsafe {
			let mut pr = [0u32; 16];
			pr[0] = n as u32;
			pr[1] = heads as u32;
			pr[2] = nq as u32;
			pr[3] = t0 as u32;
			pr[4] = tq as u32;
			pr[5] = (1.0 / (HD as f32).sqrt()).to_bits();
			let pb: Vec<u8> = pr.iter().flat_map(|x| x.to_le_bytes()).collect();
			// (after the compute work that read the block and wrote the rows)
			barrier(
				cmd,
				P::COMPUTE_SHADER,
				P::TRANSFER,
				A::SHADER_READ | A::SHADER_WRITE,
				A::TRANSFER_WRITE,
			);
			d.cmd_update_buffer(cmd, params, 0, &pb);
			barrier(
				cmd,
				P::TRANSFER,
				P::COMPUTE_SHADER,
				A::TRANSFER_WRITE,
				A::UNIFORM_READ | A::SHADER_READ | A::SHADER_WRITE,
			);
		};
		let pass = |cmd, i: usize, g: [usize; 3]| unsafe {
			d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[i]);
			d.cmd_bind_descriptor_sets(
				cmd,
				vk::PipelineBindPoint::COMPUTE,
				layout,
				0,
				&[sets[i.min(1)]],
				&[],
			);
			d.cmd_dispatch(cmd, g[0] as u32, g[1] as u32, g[2] as u32);
		};
		// the workgroups of pass `i` for a tile of `tq` queries
		let groups = |i: usize, tq: usize| -> [usize; 3] {
			match i {
				0 => [(heads * 9).div_ceil(48), nq, 1],
				1 => [1, tq / 8, heads],
				2 => [1, tq, heads],
				_ => [(9 * tq / 8).div_ceil(8), 1, heads],
			}
		};
		let tiles: Vec<(usize, usize)> = (0..nq)
			.step_by(tile)
			.map(|t0| (t0, tile.min(nq - t0)))
			.collect();
		// `QWEN_VIS_ATTN_TIMING=1` (diagnostic): every pass as its own
		// submission, with its time
		if std::env::var_os("QWEN_VIS_ATTN_TIMING").is_some() {
			let mut ms = [0f64; 4];
			let mut run = |i: usize, t0: usize, tq: usize| -> Result<(), String> {
				let t = std::time::Instant::now();
				self.dev
					.submit_and_wait(|cmd| {
						set_params(cmd, t0, tq);
						pass(cmd, i, groups(i, tq));
						barrier(
							cmd,
							P::COMPUTE_SHADER,
							P::HOST,
							A::SHADER_WRITE,
							A::HOST_READ,
						);
					})
					.map_err(|e| format!("vision attention: {e}"))?;
				ms[i] += t.elapsed().as_secs_f64() * 1e3;
				Ok(())
			};
			run(0, 0, 0)?;
			for &(t0, tq) in &tiles {
				for i in 1..4 {
					run(i, t0, tq)?;
				}
			}
			eprintln!(
				"  [vis attn] {n} patches: rope {:.1} ms, scores {:.1}, softmax {:.1}, P V {:.1}",
				ms[0], ms[1], ms[2], ms[3]
			);
			return Ok(());
		}
		self.dev
			.submit_and_wait(|cmd| {
				set_params(cmd, 0, 0);
				pass(cmd, 0, groups(0, 0));
				for &(t0, tq) in &tiles {
					set_params(cmd, t0, tq);
					for i in 1..4 {
						if i > 1 {
							barrier(
								cmd,
								P::COMPUTE_SHADER,
								P::COMPUTE_SHADER,
								A::SHADER_WRITE,
								A::SHADER_READ | A::SHADER_WRITE,
							);
						}
						pass(cmd, i, groups(i, tq));
					}
				}
				// the NPU reads the rows from host memory after the fence
				barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::HOST,
					A::SHADER_WRITE,
					A::HOST_READ,
				);
			})
			.map_err(|e| format!("vision attention: {e}"))
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

	/// A descriptor set with (buffer, offset, range) on bindings 0..6 and
	/// the parameter block on 8.
	fn make_set(&self, b6: &[(vk::Buffer, u64, u64); 6]) -> Result<vk::DescriptorSet, String> {
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("vision attention descriptor set: {e}"))?[0];
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

impl Drop for VisAttn {
	fn drop(&mut self) {
		unsafe {
			let _ = self.dev.device.device_wait_idle();
		}
		self.free_bufs();
		let d = &self.dev.device;
		unsafe {
			for i in 0..4 {
				d.destroy_pipeline(self.pl[i], None);
				d.destroy_shader_module(self.sm[i], None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_buffer(self.params.0, None);
			d.free_memory(self.params.1, None);
			for e in &self.imports {
				hostimp::free_import(d, e.buf, e.mem);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The four kernels against an f64 softmax(Q K^T / sqrt(72)) V with 2-D
	/// RoPE, on the default device: 20 patches (not a multiple of the
	/// kernels' 8 queries and 4 keys), 2 heads.
	#[test]
	fn vision_attention_matches_reference() {
		let Ok(mut va) = VisAttn::new(None) else {
			return;
		};
		let (h, gw, gh) = (2usize, 5usize, 4usize);
		let n = gw * gh;
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut val = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
		};
		let x: Vec<u16> = (0..n * 3 * h * HD)
			.map(|_| quant::f32_to_fp16(val()))
			.collect();
		let f = |i: usize| quant::fp16_to_f32(x[i]) as f64;
		let hw: Vec<(u32, u32)> = (0..n).map(|i| ((i / gw) as u32, (i % gw) as u32)).collect();
		let mut out = vec![0u16; n * h * HD];
		// SAFETY: plain views of the two vectors, which outlive `va`
		let (xb, ob) = unsafe {
			(
				std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2),
				std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2),
			)
		};
		va.attend(h, &hw, xb, ob).unwrap();
		let rot = |t: usize, part: usize, hd: usize| -> Vec<f64> {
			let r: Vec<f64> = (0..HD)
				.map(|d| f(((t * 3 + part) * h + hd) * HD + d))
				.collect();
			let mut o = vec![0f64; HD];
			for j in 0..36 {
				let (pos, k) = if j < 18 {
					(hw[t].0, j)
				} else {
					(hw[t].1, j - 18)
				};
				let (s, c) = (pos as f64 / 10000f64.powf(2.0 * k as f64 / 36.0)).sin_cos();
				o[j] = r[j] * c - r[j + 36] * s;
				o[j + 36] = r[j + 36] * c + r[j] * s;
			}
			o
		};
		for t in 0..n {
			for hd in 0..h {
				let q = rot(t, 0, hd);
				let sc: Vec<f64> = (0..n)
					.map(|u| {
						rot(u, 1, hd)
							.iter()
							.zip(&q)
							.map(|(a, b)| a * b)
							.sum::<f64>() / (HD as f64).sqrt()
					})
					.collect();
				let m = sc.iter().cloned().fold(f64::MIN, f64::max);
				let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
				for d in 0..HD {
					let want: f64 = (0..n)
						.map(|u| (sc[u] - m).exp() / z * f(((u * 3 + 2) * h + hd) * HD + d))
						.sum();
					let got = quant::fp16_to_f32(out[(t * h + hd) * HD + d]) as f64;
					assert!(
						(got - want).abs() < 2e-3 * (1.0 + want.abs()),
						"patch {t} head {hd} dim {d}: {got} vs {want}"
					);
				}
			}
		}
		// another image on the same object (other buffers, fewer patches)
		let hw2: Vec<(u32, u32)> = hw[..8].to_vec();
		let mut out2 = vec![0u16; 8 * h * HD];
		// SAFETY: as above
		let ob2 =
			unsafe { std::slice::from_raw_parts_mut(out2.as_mut_ptr() as *mut u8, out2.len() * 2) };
		va.attend(h, &hw2, &xb[..8 * 3 * h * HD * 2], ob2).unwrap();
		assert!(
			out2.iter().all(|&v| quant::fp16_to_f32(v).is_finite()) && out2 != out[..8 * h * HD]
		);
	}
}
