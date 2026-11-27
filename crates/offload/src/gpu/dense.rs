//! Dense f32 matrices in VRAM and one product on them, `C (-)= A B^T`
//! (`shaders/gemm_f32.glsl`): what a blocked Cholesky factorization and a
//! triangular inverse of a large matrix are made of. `crates/npubuild` runs
//! GPTQ's 17408 x 17408 Hessian factorization on it.
//!
//! Every failure is an `Err`; the caller falls back to the CPU.

use super::*;
use std::collections::HashMap;

/// Floats of the host staging buffer (uploads and downloads go through it
/// in pieces).
const STAGE: usize = 32 << 20;
/// Tile of `gemm_f32.glsl`: rows of A x rows of B per workgroup.
const BM: u32 = 128;
const BN: u32 = 64;

/// A region of a matrix buffer: (buffer, offset, row stride), in floats.
pub type Region = (usize, usize, usize);

/// See the module docs.
pub struct Dense {
	dev: GpuDevice,
	sm: vk::ShaderModule,
	pl: vk::Pipeline,
	pool: vk::DescriptorPool,
	params: (vk::Buffer, vk::DeviceMemory),
	stage: (vk::Buffer, vk::DeviceMemory, *mut f32),
	bufs: Vec<Option<(vk::Buffer, vk::DeviceMemory, usize)>>,
	sets: HashMap<[usize; 3], vk::DescriptorSet>,
}

// SAFETY: used from one thread at a time.
unsafe impl Send for Dense {}

impl Dense {
	/// On the Vulkan adapter `pick` names (`None`: the default one).
	pub fn new(pick: Option<&str>) -> Result<Dense, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let d = &dev.device;
		let main = CString::new("main").unwrap();
		let sm = create_sm(d, &load_spv(quant::packed!("gemm_f32.spv")))?;
		let pl = compute_pipeline(d, dev.aux_pipeline_layout, sm, &main)?;
		let sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 6 * 64,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: 64,
			},
		];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
					max_sets: 64,
					pool_size_count: 2,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("dense: descriptor pool: {e}"))?;
		let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
		let params = dev
			.try_create_gpu_buffer(
				64,
				vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			)
			.ok_or("dense: params buffer")?;
		let su = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
		let stage = dev
			.try_create_gpu_buffer(
				(STAGE * 4) as u64,
				su,
				host | vk::MemoryPropertyFlags::HOST_CACHED,
			)
			.or_else(|| dev.try_create_gpu_buffer((STAGE * 4) as u64, su, host))
			.ok_or("dense: staging buffer")?;
		let map = unsafe { d.map_memory(stage.1, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
			.map_err(|e| format!("dense: map: {e}"))? as *mut f32;
		Ok(Dense {
			sm,
			pl,
			pool,
			params,
			stage: (stage.0, stage.1, map),
			bufs: Vec::new(),
			sets: HashMap::new(),
			dev,
		})
	}

	/// Bytes of VRAM free within the device's budget, when it reports one.
	pub fn free_bytes(&self) -> Option<u64> {
		self.dev.vram_budget().map(|(b, u)| b.saturating_sub(u))
	}

	/// A buffer of `floats` floats in VRAM (contents undefined).
	pub fn alloc(&mut self, floats: usize) -> Result<usize, String> {
		let usage = vk::BufferUsageFlags::STORAGE_BUFFER
			| vk::BufferUsageFlags::TRANSFER_DST
			| vk::BufferUsageFlags::TRANSFER_SRC;
		let b = self
			.dev
			.try_create_gpu_buffer(
				(floats * 4) as u64,
				usage,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			)
			.ok_or(format!("dense: no VRAM for {} MB", (floats * 4) >> 20))?;
		self.bufs.push(Some((b.0, b.1, floats)));
		Ok(self.bufs.len() - 1)
	}

	/// Release buffer `i`.
	pub fn free(&mut self, i: usize) {
		if let Some((b, m, _)) = self.bufs[i].take() {
			let d = &self.dev.device;
			let stale: Vec<[usize; 3]> = self
				.sets
				.keys()
				.filter(|k| k.contains(&i))
				.copied()
				.collect();
			for k in stale {
				let s = self.sets.remove(&k).unwrap();
				let _ = unsafe { d.free_descriptor_sets(self.pool, &[s]) };
			}
			unsafe {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
		}
	}

	fn buf(&self, i: usize) -> Result<(vk::Buffer, usize), String> {
		self.bufs
			.get(i)
			.copied()
			.flatten()
			.map(|b| (b.0, b.2))
			.ok_or_else(|| "dense: no such buffer".to_string())
	}

	/// `data` into buffer `i` from float `off` on.
	pub fn upload(&mut self, i: usize, off: usize, data: &[f32]) -> Result<(), String> {
		let (b, n) = self.buf(i)?;
		if off + data.len() > n {
			return Err("dense: upload past the buffer".into());
		}
		let (src, d) = (self.stage.0, &self.dev.device);
		for (ci, c) in data.chunks(STAGE).enumerate() {
			// SAFETY: the map holds STAGE floats; no submission is in flight
			unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), self.stage.2, c.len()) };
			let r = vk::BufferCopy {
				src_offset: 0,
				dst_offset: ((off + ci * STAGE) * 4) as u64,
				size: (c.len() * 4) as u64,
			};
			self.dev
				.submit_and_wait(|cmd| unsafe { d.cmd_copy_buffer(cmd, src, b, &[r]) })
				.map_err(|e| format!("dense: upload: {e}"))?;
		}
		Ok(())
	}

	/// Zero buffer `i`.
	pub fn zero(&mut self, i: usize) -> Result<(), String> {
		let (b, _) = self.buf(i)?;
		let d = &self.dev.device;
		self.dev
			.submit_and_wait(|cmd| unsafe { d.cmd_fill_buffer(cmd, b, 0, vk::WHOLE_SIZE, 0) })
			.map_err(|e| format!("dense: fill: {e}"))
	}

	/// `data` (`rows` rows of `cols` floats, packed) into buffer `i` at
	/// stride `ld` from float `off` on.
	pub fn upload_rows(
		&mut self,
		i: usize,
		off: usize,
		ld: usize,
		rows: usize,
		cols: usize,
		data: &[f32],
	) -> Result<(), String> {
		let (b, n) = self.buf(i)?;
		if rows == 0 || cols == 0 {
			return Ok(());
		}
		if off + (rows - 1) * ld + cols > n || data.len() != rows * cols || data.len() > STAGE {
			return Err("dense: bad upload".into());
		}
		// SAFETY: the map holds STAGE floats; no submission is in flight
		unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.stage.2, data.len()) };
		let (src, d) = (self.stage.0, &self.dev.device);
		let regions: Vec<vk::BufferCopy> = (0..rows)
			.map(|r| vk::BufferCopy {
				src_offset: (r * cols * 4) as u64,
				dst_offset: ((off + r * ld) * 4) as u64,
				size: (cols * 4) as u64,
			})
			.collect();
		self.dev
			.submit_and_wait(|cmd| unsafe { d.cmd_copy_buffer(cmd, src, b, &regions) })
			.map_err(|e| format!("dense: upload: {e}"))
	}

	/// `rows` rows of `cols` floats at stride `ld` from float `off` of
	/// buffer `i`, packed into `out`.
	pub fn download(
		&mut self,
		i: usize,
		off: usize,
		ld: usize,
		rows: usize,
		cols: usize,
		out: &mut [f32],
	) -> Result<(), String> {
		let (b, n) = self.buf(i)?;
		if rows == 0 || cols == 0 {
			return Ok(());
		}
		if off + (rows - 1) * ld + cols > n || out.len() != rows * cols || cols > STAGE {
			return Err("dense: bad download".into());
		}
		let (dst, d) = (self.stage.0, &self.dev.device);
		let per = STAGE / cols;
		for (ci, o) in out.chunks_mut(per * cols).enumerate() {
			let regions: Vec<vk::BufferCopy> = (0..o.len() / cols)
				.map(|r| vk::BufferCopy {
					src_offset: ((off + (ci * per + r) * ld) * 4) as u64,
					dst_offset: (r * cols * 4) as u64,
					size: (cols * 4) as u64,
				})
				.collect();
			self.dev
				.submit_and_wait(|cmd| unsafe { d.cmd_copy_buffer(cmd, b, dst, &regions) })
				.map_err(|e| format!("dense: download: {e}"))?;
			// SAFETY: the map holds STAGE floats; the copy is done
			unsafe {
				std::ptr::copy_nonoverlapping(self.stage.2 as *const f32, o.as_mut_ptr(), o.len())
			};
		}
		Ok(())
	}

	fn set_for(&mut self, key: [usize; 3]) -> Result<vk::DescriptorSet, String> {
		if let Some(&s) = self.sets.get(&key) {
			return Ok(s);
		}
		if self.sets.len() >= 60 {
			return Err("dense: too many buffer combinations".into());
		}
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dev.aux_layout,
				..Default::default()
			})
		}
		.map_err(|e| format!("dense: descriptor set: {e}"))?[0];
		let info = |b: vk::Buffer, range: u64| vk::DescriptorBufferInfo {
			buffer: b,
			offset: 0,
			range,
		};
		let b3 = [
			self.buf(key[0])?.0,
			self.buf(key[1])?.0,
			self.buf(key[2])?.0,
		];
		// bindings 3..5 are not read by the kernel: any buffer will do
		let infos: Vec<vk::DescriptorBufferInfo> = (0..6)
			.map(|i| info(b3[i.min(2)], vk::WHOLE_SIZE))
			.chain(Some(info(self.params.0, 64)))
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
		self.sets.insert(key, set);
		Ok(set)
	}

	/// `C[m][n] (-)= A[m][k] B[n][k]^T` on regions of buffers. `sub`:
	/// subtract from `C` instead of storing; `tb`: `B` is read transposed
	/// (its `[k][n]` rows hold the `n` vectors as columns); `lower`: only
	/// the lower triangle of `C` is needed (tiles wholly above the diagonal
	/// are skipped). The region written must not overlap one that is read.
	#[allow(clippy::too_many_arguments)]
	pub fn gemm(
		&mut self,
		a: Region,
		m: usize,
		b: Region,
		n: usize,
		k: usize,
		c: Region,
		sub: bool,
		tb: bool,
		lower: bool,
	) -> Result<(), String> {
		if m == 0 || n == 0 || k == 0 {
			return Ok(());
		}
		let last = |r: Region, rows: usize, cols: usize| r.1 + (rows - 1) * r.2 + cols;
		let (bn, bk) = if tb { (k, n) } else { (n, k) };
		if last(a, m, k) > self.buf(a.0)?.1
			|| last(b, bn, bk) > self.buf(b.0)?.1
			|| last(c, m, n) > self.buf(c.0)?.1
		{
			return Err("dense: a region runs past its buffer".into());
		}
		let set = self.set_for([a.0, b.0, c.0])?;
		let p: Vec<u8> = [
			a.1,
			a.2,
			b.1,
			b.2,
			c.1,
			c.2,
			m,
			n,
			k,
			sub as usize | (tb as usize) << 1 | (lower as usize) << 2,
		]
		.iter()
		.chain(&[0usize; 6])
		.flat_map(|&v| (v as u32).to_le_bytes())
		.collect();
		let (d, pl, layout, pb) = (
			&self.dev.device,
			self.pl,
			self.dev.aux_pipeline_layout,
			self.params.0,
		);
		use vk::{AccessFlags as A, PipelineStageFlags as P};
		self.dev
			.submit_and_wait(|cmd| unsafe {
				d.cmd_update_buffer(cmd, pb, 0, &p);
				d.cmd_pipeline_barrier(
					cmd,
					P::TRANSFER,
					P::COMPUTE_SHADER,
					vk::DependencyFlags::empty(),
					&[vk::MemoryBarrier {
						src_access_mask: A::TRANSFER_WRITE,
						dst_access_mask: A::UNIFORM_READ | A::SHADER_READ | A::SHADER_WRITE,
						..Default::default()
					}],
					&[],
					&[],
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
				d.cmd_dispatch(cmd, (m as u32).div_ceil(BM), (n as u32).div_ceil(BN), 1);
				d.cmd_pipeline_barrier(
					cmd,
					P::COMPUTE_SHADER,
					P::COMPUTE_SHADER | P::TRANSFER,
					vk::DependencyFlags::empty(),
					&[vk::MemoryBarrier {
						src_access_mask: A::SHADER_WRITE,
						dst_access_mask: A::SHADER_READ | A::TRANSFER_READ,
						..Default::default()
					}],
					&[],
					&[],
				);
			})
			.map_err(|e| format!("dense: gemm: {e}"))
	}
}

impl Drop for Dense {
	fn drop(&mut self) {
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			d.destroy_pipeline(self.pl, None);
			d.destroy_shader_module(self.sm, None);
			d.destroy_descriptor_pool(self.pool, None);
			for &(b, m, _) in self.bufs.iter().flatten() {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
			for (b, m) in [self.params, (self.stage.0, self.stage.1)] {
				d.destroy_buffer(b, None);
				d.free_memory(m, None);
			}
		}
	}
}
