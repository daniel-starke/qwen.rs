//! The Qwen-Image transformer's blocks on a Vulkan device (the iGPU), for
//! a share of each block's row chunks next to the NPU.
//!
//! WHY
//! ---
//! Text to image on the NPU backend (`imagegen::qimage::npudit`) is NPU-bound:
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

use super::hostkern::{HostVec, Rig};
use super::*;

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

/// The kernels.
const K_GEMM: usize = 0;
const K_ROW0: usize = 1;
const K_ROW1: usize = 2;
const K_ROW2: usize = 3;
const K_ACT: usize = 4;
const K_QKV: usize = 5;

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
	rig: Rig,
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
		let gemm = match std::env::var_os("QWEN_DIT_GEMM_SPV") {
			Some(f) => std::fs::read(&f).map_err(|e| format!("{}: {e}", f.to_string_lossy()))?,
			None => quant::packed!("dit_gemm_npu.spv").to_vec(),
		};
		let spvs: [&[u8]; 6] = [
			&gemm,
			quant::packed!("dit_row0_npu.spv"),
			quant::packed!("dit_row1_npu.spv"),
			quant::packed!("dit_row2_npu.spv"),
			quant::packed!("dit_act_npu.spv"),
			quant::packed!("dit_qkv_npu.spv"),
		];
		let mut rig = Rig::new(pick, &spvs, 32, "dit block")?;
		let vecs = HostVec::new(layers * VB + 4 * D + DIT_CHUNK);
		let y = HostVec::new(DIT_CHUNK * 2 * FF);
		let x1 = HostVec::new(DIT_CHUNK * D);
		let xq = HostVec::new(DIT_CHUNK * FF);
		rig.own = vec![vecs.range().0, y.range().0, x1.range().0, xq.range().0];
		rig.keep_lens = vec![D * D, 2 * FF * D, D * FF, 3 * D * D];
		Ok(DitBlk {
			rig,
			layers,
			vecs,
			y,
			x1,
			xq,
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
		let (vv, yy, xx1, xxq) = (
			self.vecs.range(),
			self.y.range(),
			self.x1.range(),
			self.xq.range(),
		);
		let p = |s: &[u8]| (s.as_ptr() as usize, s.len());
		let rig = &mut self.rig;
		// the sets: bindings 0..6 (unused ones bound to the vectors)
		let s_row0 = rig.set([p(j.a), yy, xx1, vv, xxq, vv])?;
		let s_row1 = rig.set([p(j.x), yy, xx1, vv, xxq, vv])?;
		let s_row2 = rig.set([vv, yy, xx1, vv, xxq, p(j.xo)])?;
		let s_act = rig.set([vv, yy, vv, vv, xxq, vv])?;
		let s_qkv = rig.set([p(j.q), yy, vv, vv, p(j.k), p(j.v)])?;
		let mut s_w = [vk::DescriptorSet::null(); 4];
		for (i, w) in [j.w_o, j.w_gu, j.w_d, j.w_qkv].into_iter().enumerate() {
			s_w[i] = rig.set([p(w), xxq, yy, vv, vv, vv])?;
		}
		let (lp, lq) = ((j.l - 1) * VB, j.l * VB);
		let mods = self.layers * VB;
		let (ms1, mg1, ms2, mg2) = (mods, mods + D, mods + 2 * D, mods + 3 * D);
		let rowf = mods + 4 * D;
		let u = |x: usize| x as u32;
		rig.submit(|rig, cmd| {
			let step = |k: usize, set, pc: [u32; 8], gx: usize, gy: usize| {
				rig.dispatch(cmd, k, set, &pc, gx, gy)
			};
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
		})
		.map_err(|e| format!("dit block: {e}"))
	}

	/// Wait for the submissions in flight.
	pub fn end(&mut self) -> Result<(), String> {
		self.rig.end()
	}

	/// Drop the imports of the pass' buffers (before they are freed); the
	/// weights' and the own buffers' stay.
	pub fn forget_rows(&mut self) {
		self.rig.forget_rows()
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
