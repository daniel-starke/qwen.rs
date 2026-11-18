//! The Qwen-Image transformer's blocks on a Vulkan device (the iGPU), for
//! a share of each block's row chunks next to the NPU.
//!
//! WHY
//! ---
//! Text to image on the NPU backend (`imagegen::npudit`) is NPU-bound:
//! the cores' attention takes about a third of each block and the iGPU
//! idles. Here the iGPU runs the NPU's `dit_mid` job (block l - 1 after its
//! attention, block l up to its attention) for some row chunks, the
//! NPU graphs' math (`npubuild::qimage::graph_form`, folded form) on the
//! same int8 weights: `dit_gemm_npu.glsl` (int8 x int8 `dp4a`, exact int32
//! sums), `dit_row_npu.glsl` (residual + gate, LayerNorm x modulation,
//! pre-scale, Hadamard, int8), `dit_act_npu.glsl` (SwiGLU, per-row int8),
//! `dit_qkv_npu.glsl` (q / k / v to f16 where the NPU puts them). f32
//! between the matmuls where the NPU has f16.
//!
//! The tensors stay where the NPU put them: its weight buffers and the
//! pass' host buffers are imported (`VK_EXT_external_memory_host`) on
//! first use; [`DitBlk::forget_rows`] drops the pass' imports before those
//! buffers are freed.

use super::*;
use std::collections::HashMap;

const D: usize = 4096;
const FF: usize = 12288;
/// Rows of one pass through the kernels (the NPU's chunk).
pub const DIT_CHUNK: usize = 256;
/// f32 vectors of a block in [`DitBlk`]'s vector buffer: `o.pre`, `gu.pre`,
/// `qkv.pre`, `gu.sg`, `o.s`, `d.s`.
const VB: usize = 5 * D + FF;
const O_PRE: usize = 0;
const GU_PRE: usize = D;
const QKV_PRE: usize = 2 * D;
const GU_SG: usize = 3 * D;
const O_S: usize = 3 * D + FF;
const D_S: usize = 4 * D + FF;
const NONE: u32 = !0;
/// Descriptor sets the pool holds.
const SETS: usize = 1024;

/// The kernels.
const K_GEMM: usize = 0;
const K_ROW0: usize = 1;
const K_ROW1: usize = 2;
const K_ROW2: usize = 3;
const K_ACT: usize = 4;
const K_QKV: usize = 5;

struct Imported {
	base: usize,
	len: usize,
	buf: vk::Buffer,
	mem: vk::DeviceMemory,
	rows: bool,
}

/// A page-aligned host vector of `T`, imported once.
struct HostVec<T> {
	v: Vec<T>,
	off: usize,
	len: usize,
}

impl<T: Copy + Default> HostVec<T> {
	fn new(len: usize) -> HostVec<T> {
		let sz = std::mem::size_of::<T>();
		let v = vec![T::default(); len + 4096 / sz];
		let off = ((v.as_ptr() as usize).next_multiple_of(4096) - v.as_ptr() as usize) / sz;
		HostVec { v, off, len }
	}
	fn s(&self) -> &[T] {
		&self.v[self.off..self.off + self.len]
	}
	fn m(&mut self) -> &mut [T] {
		&mut self.v[self.off..self.off + self.len]
	}
}

/// One `dit_mid` job over rows `r0 .. r1`: block `l - 1` after its
/// attention and block `l` up to its attention. The slices are the whole
/// host buffers (imported on first use).
pub struct DitMid<'a> {
	/// The pre block (`l`; the post block is `l - 1`).
	pub l: usize,
	/// The residual in, `X[cur]`, f16 `[rows][D]`.
	pub x: &'a [u8],
	/// The attention output, f16 `[rows][D]`.
	pub a: &'a [u8],
	/// The residual out, `X[1 - cur]`.
	pub xo: &'a [u8],
	/// q, f16 `[rows][D]`; k / v, f16 `[kv rows][D]`.
	pub q: &'a [u8],
	pub k: &'a [u8],
	pub v: &'a [u8],
	pub r0: usize,
	pub r1: usize,
	/// The K / V row of row `r0`.
	pub krow: usize,
	/// Block `l - 1`'s `o.w`, `gu.w`, `d.w`; block `l`'s `qkv.w` (int8).
	pub w_o: &'a [u8],
	pub w_gu: &'a [u8],
	pub w_d: &'a [u8],
	pub w_qkv: &'a [u8],
}

/// See the module docs.
pub struct DitBlk {
	dev: GpuDevice,
	dsl: vk::DescriptorSetLayout,
	layout: vk::PipelineLayout,
	sm: Vec<vk::ShaderModule>,
	pl: Vec<vk::Pipeline>,
	pool: vk::DescriptorPool,
	imports: Vec<Imported>,
	align: usize,
	sets: HashMap<[usize; 6], vk::DescriptorSet>,
	/// Submissions in flight (no fence; [`DitBlk::end`] fences them).
	pending: Vec<vk::CommandBuffer>,
	layers: usize,
	/// per block [`VB`] floats, then the modulation (ms1, mg1, ms2, mg2),
	/// then a factor per row of a chunk
	vecs: HostVec<f32>,
	y: HostVec<f32>,
	x1: HostVec<f32>,
	xq: HostVec<u8>,
}

// SAFETY: used from one thread at a time.
unsafe impl Send for DitBlk {}

impl DitBlk {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU), for a
	/// transformer of `layers` blocks.
	pub fn new(pick: Option<&str>, layers: usize) -> Result<DitBlk, String> {
		let dev = GpuDevice::init_pick(pick)?;
		let align = dev
			.host_import_align()
			.ok_or("the Vulkan device has no VK_EXT_external_memory_host")?;
		let d = &dev.device;
		let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6)
			.map(|b| vk::DescriptorSetLayoutBinding {
				binding: b,
				descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: 1,
				stage_flags: vk::ShaderStageFlags::COMPUTE,
				..Default::default()
			})
			.collect();
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
		.map_err(|e| format!("dit block: {e}"))?;
		let pcr = vk::PushConstantRange {
			stage_flags: vk::ShaderStageFlags::COMPUTE,
			offset: 0,
			size: 32,
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
		.map_err(|e| format!("dit block: {e}"))?;
		let main = CString::new("main").unwrap();
		let gemm = match std::env::var_os("QWEN_DIT_GEMM_SPV") {
			Some(f) => std::fs::read(&f).map_err(|e| format!("{}: {e}", f.to_string_lossy()))?,
			None => include_bytes!("../shaders/dit_gemm_npu.spv").to_vec(),
		};
		let spvs: [&[u8]; 6] = [
			&gemm,
			include_bytes!("../shaders/dit_row0_npu.spv"),
			include_bytes!("../shaders/dit_row1_npu.spv"),
			include_bytes!("../shaders/dit_row2_npu.spv"),
			include_bytes!("../shaders/dit_act_npu.spv"),
			include_bytes!("../shaders/dit_qkv_npu.spv"),
		];
		let mut sm = Vec::new();
		let mut pl = Vec::new();
		for s in spvs {
			let m = create_sm(d, &load_spv(s))?;
			pl.push(compute_pipeline(d, layout, m, &main)?);
			sm.push(m);
		}
		let sizes = [vk::DescriptorPoolSize {
			ty: vk::DescriptorType::STORAGE_BUFFER,
			descriptor_count: (6 * SETS) as u32,
		}];
		let pool = unsafe {
			d.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo {
					max_sets: SETS as u32,
					pool_size_count: sizes.len() as u32,
					p_pool_sizes: sizes.as_ptr(),
					..Default::default()
				},
				None,
			)
		}
		.map_err(|e| format!("dit block descriptor pool: {e}"))?;
		Ok(DitBlk {
			dsl,
			layout,
			sm,
			pl,
			pool,
			imports: Vec::new(),
			align,
			sets: HashMap::new(),
			pending: Vec::new(),
			layers,
			vecs: HostVec::new(layers * VB + 4 * D + DIT_CHUNK),
			y: HostVec::new(DIT_CHUNK * 2 * FF),
			x1: HostVec::new(DIT_CHUNK * D),
			xq: HostVec::new(DIT_CHUNK * FF),
			dev,
		})
	}

	/// Block `l`'s vectors (f32): `o.pre`, `gu.pre`, `qkv.pre` ([D]),
	/// `gu.sg` ([FF]), `o.s`, `d.s` ([D]).
	pub fn set_block(&mut self, l: usize, v: [&[f32]; 6]) {
		let b = &mut self.vecs.m()[l * VB..(l + 1) * VB];
		let mut o = 0;
		for x in v {
			b[o..o + x.len()].copy_from_slice(x);
			o += x.len();
		}
		assert_eq!(o, VB);
	}

	/// The step's modulation: `1 + scale1`, `gate1`, `1 + scale2`, `gate2`
	/// ([D] each). Nothing may be in flight.
	pub fn set_mods(&mut self, m: [&[f32]; 4]) {
		let o = self.layers * VB;
		for (i, x) in m.iter().enumerate() {
			self.vecs.m()[o + i * D..o + (i + 1) * D].copy_from_slice(x);
		}
	}

	/// Submit job `j` (no fence; several may be in flight, in queue order),
	/// finished by [`DitBlk::end`].
	/// Nothing may touch rows `r0 .. r1` of `a`, `xo`, `q`, the K / V rows
	/// or the weights meanwhile.
	pub fn mid_begin(&mut self, j: &DitMid<'_>) -> Result<(), String> {
		let rows = j.r1 - j.r0;
		let ldw_d = j.w_d.len() / D;
		if j.l == 0
			|| j.l >= self.layers
			|| rows == 0
			|| [j.x, j.a, j.xo, j.q].iter().any(|b| b.len() < j.r1 * D * 2)
			|| [j.k, j.v].iter().any(|b| b.len() < (j.krow + rows) * D * 2)
			|| j.w_o.len() != D * D
			|| j.w_gu.len() != 2 * FF * D
			|| j.w_d.len() < D * FF
			|| !j.w_d.len().is_multiple_of(D * 16)
			|| j.w_qkv.len() != 3 * D * D
		{
			return Err(format!(
				"dit block: job of block {} rows {}..{}: bad sizes",
				j.l, j.r0, j.r1
			));
		}
		let (vecs, y, x1, xq) = (
			self.vecs.s().as_ptr() as usize,
			self.y.s().as_ptr() as usize,
			self.x1.s().as_ptr() as usize,
			self.xq.s().as_ptr() as usize,
		);
		let (lv, ly, lx1, lxq) = (
			self.vecs.len * 4,
			self.y.len * 4,
			self.x1.len * 4,
			self.xq.len,
		);
		let p = |s: &[u8]| (s.as_ptr() as usize, s.len());
		let (vv, yy, xx1, xxq) = ((vecs, lv), (y, ly), (x1, lx1), (xq, lxq));
		// the sets: bindings 0..6 (unused ones bound to the vectors)
		let s_row0 = self.set([p(j.a), yy, xx1, vv, xxq, vv])?;
		let s_row1 = self.set([p(j.x), yy, xx1, vv, xxq, vv])?;
		let s_row2 = self.set([vv, yy, xx1, vv, xxq, p(j.xo)])?;
		let s_act = self.set([vv, yy, vv, vv, xxq, vv])?;
		let s_qkv = self.set([p(j.q), yy, vv, vv, p(j.k), p(j.v)])?;
		let mut s_w = [vk::DescriptorSet::null(); 4];
		for (i, w) in [j.w_o, j.w_gu, j.w_d, j.w_qkv].into_iter().enumerate() {
			s_w[i] = self.set([p(w), xxq, yy, vv, vv, vv])?;
		}
		let (lp, lq) = ((j.l - 1) * VB, j.l * VB);
		let mods = self.layers * VB;
		let (ms1, mg1, ms2, mg2) = (mods, mods + D, mods + 2 * D, mods + 3 * D);
		let rowf = mods + 4 * D;
		let d = &self.dev.device;
		let (pl, layout) = (&self.pl, self.layout);
		let rec = |cmd: vk::CommandBuffer| unsafe {
			use vk::{AccessFlags as A, PipelineStageFlags as P};
			let bar = |a, b, sa, da| {
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
			let step = |k: usize, set: vk::DescriptorSet, pc: [u32; 8], gx: usize, gy: usize| {
				d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl[k]);
				d.cmd_bind_descriptor_sets(
					cmd,
					vk::PipelineBindPoint::COMPUTE,
					layout,
					0,
					&[set],
					&[],
				);
				let b: Vec<u8> = pc.iter().flat_map(|x| x.to_le_bytes()).collect();
				d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &b);
				d.cmd_dispatch(cmd, gx as u32, gy as u32, 1);
				bar(
					P::COMPUTE_SHADER,
					P::COMPUTE_SHADER,
					A::SHADER_WRITE,
					A::SHADER_READ | A::SHADER_WRITE,
				);
			};
			// after the host's writes and the submissions before this one
			// (they share the scratch vectors)
			bar(
				P::HOST | P::COMPUTE_SHADER,
				P::COMPUTE_SHADER,
				A::HOST_WRITE | A::SHADER_WRITE,
				A::SHADER_READ | A::SHADER_WRITE,
			);
			let u = |x: usize| x as u32;
			let mut r = j.r0;
			while r < j.r1 {
				let n = (j.r1 - r).min(DIT_CHUNK);
				let tiles = n.div_ceil(64);
				let gemm = |set, k: usize, nn: usize, sofs: u32, fofs: u32| {
					step(
						K_GEMM,
						set,
						[u(k), u(nn), u(n), u(nn), sofs, fofs, 0, 0],
						nn / 64,
						tiles,
					)
				};
				step(
					K_ROW0,
					s_row0,
					[u(r), u(D), u(lp + O_PRE), 0, 0, 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[0], D, D, u(lp + O_S), NONE);
				step(
					K_ROW1,
					s_row1,
					[u(r), u(D), u(lp + GU_PRE), u(mg1), u(ms2), 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[1], D, 2 * FF, NONE, NONE);
				step(
					K_ACT,
					s_act,
					[u(lp + GU_SG), u(rowf), 0, 0, 0, 0, 0, 0],
					n,
					1,
				);
				// d's input is FF wide: K = FF, y [n][D]; its rows may carry
				// zero pad columns (the NPU graph's `rowq` pad lanes)
				step(
					K_GEMM,
					s_w[2],
					[u(FF), u(D), u(n), u(D), u(lp + D_S), u(rowf), u(ldw_d), 0],
					D / 64,
					tiles,
				);
				step(
					K_ROW2,
					s_row2,
					[u(r), u(D), u(lq + QKV_PRE), u(mg2), u(ms1), 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[3], D, 3 * D, NONE, NONE);
				step(
					K_QKV,
					s_qkv,
					[u(r), u(j.krow + r - j.r0), 0, 0, 0, 0, 0, 0],
					D / 2 / 256,
					n,
				);
				r += n;
			}
			// the NPU and the cores read the rows from host memory after the
			// fence
			bar(P::COMPUTE_SHADER, P::HOST, A::SHADER_WRITE, A::HOST_READ);
		};
		let _ = mg1;
		let cmd = self
			.dev
			.record_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, rec);
		if let Err(e) = self.dev.submit_cmd_nofence(cmd) {
			unsafe { d.free_command_buffers(self.dev.cmd_pool, &[cmd]) };
			return Err(format!("dit block: {e}"));
		}
		self.pending.push(cmd);
		Ok(())
	}

	/// Wait for the submissions in flight (none: Ok): an empty one on the
	/// fence after them (a fence covers every earlier submission).
	pub fn end(&mut self) -> Result<(), String> {
		if self.pending.is_empty() {
			return Ok(());
		}
		let r = match self.dev.submit_nowait(|_| {}) {
			Ok(cmd) => self.dev.wait_submitted(cmd),
			Err(e) => {
				unsafe { self.dev.device.queue_wait_idle(self.dev.queue).ok() };
				Err(e)
			}
		};
		let cmds: Vec<vk::CommandBuffer> = std::mem::take(&mut self.pending);
		unsafe {
			self.dev
				.device
				.free_command_buffers(self.dev.cmd_pool, &cmds)
		};
		r.map_err(|e| format!("dit block: {e}"))
	}

	/// Drop the imports of the pass' buffers (before they are freed); the
	/// weights' and the own buffers' stay.
	pub fn forget_rows(&mut self) {
		let _ = self.end();
		let d = &self.dev.device;
		unsafe {
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
		}
		self.sets.clear();
		let mut keep = Vec::new();
		for e in self.imports.drain(..) {
			if e.rows {
				unsafe { hostimp::free_import(d, e.buf, e.mem) };
			} else {
				keep.push(e);
			}
		}
		self.imports = keep;
	}

	/// The descriptor set over host ranges `b` (bindings 0..6).
	fn set(&mut self, b: [(usize, usize); 6]) -> Result<vk::DescriptorSet, String> {
		let key = b.map(|x| x.0);
		if let Some(s) = self.sets.get(&key) {
			return Ok(*s);
		}
		if self.sets.len() == SETS {
			// the sets of the submissions in flight go with the pool
			self.end()?;
			unsafe {
				self.dev
					.device
					.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
			}
			.map_err(|e| format!("dit block: {e}"))?;
			self.sets.clear();
		}
		let mut infos = [vk::DescriptorBufferInfo::default(); 6];
		for (i, &(ptr, len)) in b.iter().enumerate() {
			let (buf, off) = self.import(ptr, len)?;
			infos[i] = vk::DescriptorBufferInfo {
				buffer: buf,
				offset: off,
				range: len as u64,
			};
		}
		let d = &self.dev.device;
		let set = unsafe {
			d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
				descriptor_pool: self.pool,
				descriptor_set_count: 1,
				p_set_layouts: &self.dsl,
				..Default::default()
			})
		}
		.map_err(|e| format!("dit block descriptor set: {e}"))?[0];
		let writes: Vec<vk::WriteDescriptorSet> = (0..6)
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
		self.sets.insert(key, set);
		Ok(set)
	}

	/// A buffer over host bytes `[ptr, ptr + len)` and `ptr`'s offset in it,
	/// importing the covering pages on first use. Weights (int8 matrices)
	/// and the own buffers are kept across passes.
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
		let own = [
			self.vecs.s().as_ptr() as usize,
			self.y.s().as_ptr() as usize,
			self.x1.s().as_ptr() as usize,
			self.xq.s().as_ptr() as usize,
		];
		let weight = [D * D, 2 * FF * D, D * FF, 3 * D * D].contains(&len);
		let (buf, mem) = self.dev.import_host_range(base, end - base)?;
		self.imports.push(Imported {
			base,
			len: end - base,
			buf,
			mem,
			rows: !weight && !own.contains(&ptr),
		});
		Ok((buf, (ptr - base) as u64))
	}
}

impl Drop for DitBlk {
	fn drop(&mut self) {
		let _ = self.end();
		let d = &self.dev.device;
		unsafe {
			let _ = d.device_wait_idle();
			let _ = d.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
			for e in self.imports.drain(..) {
				hostimp::free_import(d, e.buf, e.mem);
			}
			for &p in &self.pl {
				d.destroy_pipeline(p, None);
			}
			for &m in &self.sm {
				d.destroy_shader_module(m, None);
			}
			d.destroy_descriptor_pool(self.pool, None);
			d.destroy_pipeline_layout(self.layout, None);
			d.destroy_descriptor_set_layout(self.dsl, None);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn page<T: Copy + Default>(n: usize) -> HostVec<T> {
		HostVec::new(n)
	}

	fn rnd(seed: &mut u64) -> f32 {
		*seed ^= *seed << 13;
		*seed ^= *seed >> 7;
		*seed ^= *seed << 17;
		(*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
	}

	fn h(v: f32) -> [u8; 2] {
		quant::f32_to_fp16(v).to_le_bytes()
	}
	fn fh(b: &[u8], i: usize) -> f32 {
		quant::fp16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
	}

	fn fwht(v: &mut [f32]) {
		for blk in v.chunks_exact_mut(256) {
			let mut hh = 1;
			while hh < 256 {
				for i in (0..256).step_by(2 * hh) {
					for j in i..i + hh {
						let (x, y) = (blk[j], blk[j + hh]);
						blk[j] = x + y;
						blk[j + hh] = x - y;
					}
				}
				hh *= 2;
			}
			for x in blk {
				*x /= 16.0;
			}
		}
	}

	fn q8(t: &[f32], f: f32) -> Vec<i32> {
		t.iter()
			.map(|&x| ((x * f).round_ties_even().clamp(-128.0, 127.0)) as i32)
			.collect()
	}

	fn mm(xq: &[i32], w: &[u8], n: usize, k: usize) -> Vec<f32> {
		(0..n)
			.map(|r| {
				let s: i64 = (0..k)
					.map(|p| xq[p] as i64 * (w[r * k + p] as i8) as i64)
					.sum();
				s as f32 / 16384.0
			})
			.collect()
	}

	fn mvn(x: &[f32]) -> Vec<f32> {
		let n = x.len() as f64;
		let m = x.iter().map(|&v| v as f64).sum::<f64>() / n;
		let var = x.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / n;
		let inv = 1.0 / (var + 1e-6).sqrt();
		x.iter().map(|&v| ((v as f64 - m) * inv) as f32).collect()
	}

	/// One `dit_mid` job on random weights against a host model of the
	/// same math (two rows checked, the rest of a 300-row job timed), on
	/// `QWEN_TEST_ADAPTER` (`intel`).
	#[test]
	fn dit_mid_matches_reference() {
		// the engine pins the iGPU (`npudit`); on the GTX 1060 this kernel loses the device
		let pick = std::env::var("QWEN_TEST_ADAPTER").unwrap_or_else(|_| "intel".into());
		let Ok(mut g) = DitBlk::new(Some(&pick), 2) else {
			return;
		};
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut wq = |n: usize| -> HostVec<u8> {
			let mut w = page::<u8>(n);
			for b in w.m() {
				*b = ((rnd(&mut seed) * 254.0) as i8) as u8;
			}
			w
		};
		let (w_o, w_gu, w_d, w_qkv) = (wq(D * D), wq(2 * FF * D), wq(D * FF), wq(3 * D * D));
		let mut seed = 7u64;
		let mut vec = |n: usize, a: f32, b: f32| -> Vec<f32> {
			(0..n).map(|_| a + b * rnd(&mut seed)).collect()
		};
		let blk: Vec<Vec<f32>> = vec![
			vec(D, 0.0, 0.2),
			vec(D, 0.0, 0.2),
			vec(D, 0.0, 0.2),
			vec(FF, 1.0, 0.5),
			vec(D, 0.5, 0.5),
			vec(D, 0.5, 0.5),
		];
		for l in 0..2 {
			g.set_block(l, [&blk[0], &blk[1], &blk[2], &blk[3], &blk[4], &blk[5]]);
		}
		let mods: Vec<Vec<f32>> = (0..4).map(|_| vec(D, 1.0, 0.5)).collect();
		g.set_mods([&mods[0], &mods[1], &mods[2], &mods[3]]);
		let rows = 300;
		let mut x = page::<u8>(rows * D * 2);
		let mut a = page::<u8>(rows * D * 2);
		for i in 0..rows * D {
			x.m()[2 * i..2 * i + 2].copy_from_slice(&h(rnd(&mut seed) * 8.0));
			a.m()[2 * i..2 * i + 2].copy_from_slice(&h(rnd(&mut seed) * 4.0));
		}
		let (xo, q, k, v) = (
			page::<u8>(rows * D * 2),
			page::<u8>(rows * D * 2),
			page::<u8>((rows + 10) * D * 2),
			page::<u8>((rows + 10) * D * 2),
		);
		let job = DitMid {
			l: 1,
			x: x.s(),
			a: a.s(),
			xo: xo.s(),
			q: q.s(),
			k: k.s(),
			v: v.s(),
			r0: 0,
			r1: rows,
			krow: 10,
			w_o: w_o.s(),
			w_gu: w_gu.s(),
			w_d: w_d.s(),
			w_qkv: w_qkv.s(),
		};
		g.mid_begin(&job).unwrap();
		g.end().unwrap();
		let t = std::time::Instant::now();
		for _ in 0..3 {
			g.mid_begin(&job).unwrap();
			g.end().unwrap();
		}
		eprintln!(
			"dit mid: {rows} rows {:.1} ms",
			t.elapsed().as_secs_f64() / 3.0 * 1e3
		);
		// `d` with 32 pad columns per row (the NPU graph's `rowq` pad
		// lanes; not zero here, so a wrong stride shows): the same output
		let mut w_dp = page::<u8>(D * (FF + 32));
		for (o, i) in w_dp
			.m()
			.chunks_exact_mut(FF + 32)
			.zip(w_d.s().chunks_exact(FF))
		{
			o[..FF].copy_from_slice(i);
			o[FF..].fill(0x55);
		}
		let want = (xo.s().to_vec(), q.s().to_vec());
		g.mid_begin(&DitMid {
			w_d: w_dp.s(),
			..job
		})
		.unwrap();
		g.end().unwrap();
		assert!(
			xo.s() == &want.0[..] && q.s() == &want.1[..],
			"padded d differs"
		);
		for r in [3usize, 271] {
			let row = |b: &[u8]| -> Vec<f32> { (0..D).map(|i| fh(b, r * D + i)).collect() };
			let (xr, ar) = (row(x.s()), row(a.s()));
			let mut t: Vec<f32> = ar.iter().zip(&blk[0]).map(|(a, p)| a * p).collect();
			fwht(&mut t);
			let o: Vec<f32> = mm(&q8(&t, 128.0), w_o.s(), D, D)
				.iter()
				.zip(&blk[4])
				.map(|(y, s)| y * s)
				.collect();
			let x1: Vec<f32> = (0..D).map(|i| xr[i] + o[i] * mods[1][i]).collect();
			let mut t: Vec<f32> = mvn(&x1)
				.iter()
				.enumerate()
				.map(|(i, v)| v * mods[2][i] * blk[1][i])
				.collect();
			fwht(&mut t);
			let gu = mm(&q8(&t, 128.0), w_gu.s(), 2 * FF, D);
			let mut hid: Vec<f32> = (0..FF)
				.map(|i| {
					let gg = gu[i] * blk[3][i];
					gg / (1.0 + (-gg).exp()) * gu[FF + i]
				})
				.collect();
			fwht(&mut hid);
			let m = hid.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
			let dn: Vec<f32> = mm(&q8(&hid, 127.0 / m), w_d.s(), D, FF)
				.iter()
				.zip(&blk[5])
				.map(|(y, s)| y * s * m * 128.0 / 127.0)
				.collect();
			let x2: Vec<f32> = (0..D).map(|i| x1[i] + dn[i] * mods[3][i]).collect();
			let mut t: Vec<f32> = mvn(&x2)
				.iter()
				.enumerate()
				.map(|(i, v)| v * mods[0][i] * blk[2][i])
				.collect();
			fwht(&mut t);
			let qkv = mm(&q8(&t, 128.0), w_qkv.s(), 3 * D, D);
			let rel = |got: &[f32], want: &[f32]| {
				let (mut e, mut w) = (0f64, 0f64);
				for (a, b) in got.iter().zip(want) {
					e += ((a - b) as f64).powi(2);
					w += (*b as f64).powi(2);
				}
				(e / w).sqrt()
			};
			let gx: Vec<f32> = (0..D).map(|i| fh(xo.s(), r * D + i)).collect();
			let gq: Vec<f32> = (0..D).map(|i| fh(q.s(), r * D + i)).collect();
			let gk: Vec<f32> = (0..D).map(|i| fh(k.s(), (10 + r) * D + i)).collect();
			let gv: Vec<f32> = (0..D).map(|i| fh(v.s(), (10 + r) * D + i)).collect();
			let e = [
				rel(&gx, &x2),
				rel(&gq, &qkv[..D]),
				rel(&gk, &qkv[D..2 * D]),
				rel(&gv, &qkv[2 * D..]),
			];
			eprintln!(
				"dit mid row {r}: rel. error xo / q / k / v {:.2e} {:.2e} {:.2e} {:.2e}",
				e[0], e[1], e[2], e[3]
			);
			assert!(e.iter().all(|&x| x < 2e-2), "row {r}: {e:?}");
		}
	}
}
