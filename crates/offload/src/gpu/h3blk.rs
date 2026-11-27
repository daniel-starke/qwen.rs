//! MiniMax-H3-Image's transformer blocks on a Vulkan device (the iGPU), for
//! a share of each block's row chunks next to the NPU (`imagegen::h3::npudit`;
//! the text encoder's layers, the same math with grouped queries and one
//! modulation entry, `imagegen::h3::npute`).
//!
//! A job is the NPU's `h3/mid` over some rows: block l - 1 from its
//! attention on and block l up to its attention, the graphs' math
//! (`npubuild::h3img::dit_graph`) on the same int8 weights:
//! `dit_gemm_npu.glsl` (int8 x int8 `dp4a`, exact int32 sums),
//! `h3_row_npu.glsl` (scaled residual + out projection, RMSNorm x
//! modulation + shift, pre-scale, Hadamard, int8), `h3_act_npu.glsl`
//! (SwiGLU, per-row int8), `h3_qkv_npu.glsl` (raw q / k / v to f16 where the
//! NPU puts them). f32 between the matmuls where the NPU has f16.
//!
//! The tensors stay where the NPU put them: its weight buffers and the
//! pass' host buffers are imported on first use; [`H3Blk::forget_rows`]
//! drops the pass' imports before those buffers are freed. The modulation
//! entries a step uses are copied in as f32 ([`H3Blk::set_entry`]).

use super::hostkern::{HostVec, Rig};
use super::*;

/// The transformer's widths, which its kernels are built for: the
/// residual, the attention, the MLP's hidden state (k / v as wide as q).
pub const H3_D: usize = 5376;
pub const H3_A: usize = 7168;
pub const H3_FF: usize = 14336;
/// The text encoder's: residual, q, k / v, the MLP's hidden state.
const TE: [usize; 4] = [5120, 8192, 1024, 25600];
/// Rows of one pass through the kernels (the NPU's chunk).
const CHUNK: usize = 256;
/// The `rowq` pad columns of the down projection's rows.
const PAD: usize = 32;
/// Modulation entries per block a step can use.
pub const H3_SLOTS: usize = 8;
/// Parts of a modulation entry: `ms1`, `sh1`, `ms2`, `sh2`, `o.s`, `d.s`,
/// `xr` (`npubuild::h3img::mod_part`).
const MP: usize = 7;
const NONE: u32 = !0;

const K_GEMM: usize = 0;
const K_ROW0: usize = 1;
const K_ROW1: usize = 2;
const K_ROW2: usize = 3;
const K_ACT: usize = 4;
const K_QKV: usize = 5;

/// One `h3/mid` job over padded rows `r0 .. r1`. The slices are the whole
/// host buffers (imported on first use).
pub struct H3Mid<'a> {
	/// The pre block (`l`; the post block is `l - 1`).
	pub l: usize,
	/// The stored residual in, `X[cur]`, f16 `[rows][D]`.
	pub x: &'a [u8],
	/// The attention output, f16 `[rows][A]`.
	pub a: &'a [u8],
	/// The residual out, `X[1 - cur]`.
	pub xo: &'a [u8],
	/// Raw q, f16 `[rows][A]`, k / v `[rows][KV]`, written at the job's rows.
	pub q: &'a [u8],
	pub k: &'a [u8],
	pub v: &'a [u8],
	pub r0: usize,
	pub r1: usize,
	/// Block `l - 1`'s `o.w`, `gu.w`, `d.w`; block `l`'s `qkv.w` (int8).
	pub w_o: &'a [u8],
	pub w_gu: &'a [u8],
	pub w_d: &'a [u8],
	pub w_qkv: &'a [u8],
	/// The rows' modulation slot ([`H3Blk::set_entry`]).
	pub slot: usize,
}

/// See the module docs.
pub struct H3Blk {
	rig: Rig,
	layers: usize,
	/// The widths: residual, q and the attention, k / v, the MLP's hidden state.
	d: usize,
	a: usize,
	kv: usize,
	ff: usize,
	/// per block [`BV`] floats, then per block and slot the [`MP`] parts,
	/// then a factor per row of a chunk
	vecs: HostVec<f32>,
	y: HostVec<f32>,
	x1: HostVec<f32>,
	xq: HostVec<u8>,
}

// SAFETY: used from one thread at a time.
unsafe impl Send for H3Blk {}

impl H3Blk {
	/// On the Vulkan adapter `pick` names (`"intel"` for the iGPU), for
	/// `layers` blocks of the widths `d` (residual), `inner` (q and the
	/// attention), `kv` (k and v), `ff` (an error unless kernels are built
	/// for them: the transformer's or the text encoder's).
	pub fn new(
		pick: Option<&str>,
		layers: usize,
		d: usize,
		inner: usize,
		kv: usize,
		ff: usize,
	) -> Result<H3Blk, String> {
		let spvs: [&[u8]; 6] = if [d, inner, kv, ff] == [H3_D, H3_A, H3_A, H3_FF] {
			[
				quant::packed!("dit_gemm_npu.spv"),
				quant::packed!("h3_row0_npu.spv"),
				quant::packed!("h3_row1_npu.spv"),
				quant::packed!("h3_row2_npu.spv"),
				quant::packed!("h3_act_npu.spv"),
				quant::packed!("h3_qkv_npu.spv"),
			]
		} else if [d, inner, kv, ff] == TE {
			[
				quant::packed!("dit_gemm_npu.spv"),
				quant::packed!("h3te_row0_npu.spv"),
				quant::packed!("h3te_row1_npu.spv"),
				quant::packed!("h3te_row2_npu.spv"),
				quant::packed!("h3te_act_npu.spv"),
				quant::packed!("h3te_qkv_npu.spv"),
			]
		} else {
			return Err(format!(
				"h3 block: no kernels for {d} / {inner} / {kv} / {ff}"
			));
		};
		let (a, bv) = (inner, inner + 2 * d + ff);
		let mut rig = Rig::new(pick, &spvs, 32, "h3 block")?;
		let vecs = HostVec::new(layers * bv + layers * H3_SLOTS * MP * d + CHUNK);
		let y = HostVec::new(CHUNK * (2 * ff).max(a + 2 * kv));
		let x1 = HostVec::new(CHUNK * d);
		let xq = HostVec::new(CHUNK * ff.max(a).max(d));
		rig.own = vec![vecs.range().0, y.range().0, x1.range().0, xq.range().0];
		rig.keep_lens = vec![d * a, 2 * ff * d, d * (ff + PAD), (a + 2 * kv) * d];
		Ok(H3Blk {
			rig,
			layers,
			d,
			a,
			kv,
			ff,
			vecs,
			y,
			x1,
			xq,
		})
	}

	/// Floats of a block's vectors.
	fn bv(&self) -> usize {
		self.a + 2 * self.d + self.ff
	}

	/// Block `l`'s vectors (f32): `o.pre` [A], `gu.pre`, `qkv.pre` [D],
	/// `gu.sg` [FF].
	pub fn set_block(&mut self, l: usize, v: [&[f32]; 4]) {
		let bv = self.bv();
		let b = &mut self.vecs.m()[l * bv..(l + 1) * bv];
		let mut o = 0;
		for x in v {
			b[o..o + x.len()].copy_from_slice(x);
			o += x.len();
		}
		assert_eq!(o, bv);
	}

	/// Block `l`'s modulation entry in `slot`: the [`MP`] parts of `D`
	/// f32 each. Nothing may be in flight.
	pub fn set_entry(&mut self, l: usize, slot: usize, parts: &[f32]) {
		let d = self.d;
		assert!(slot < H3_SLOTS && parts.len() == MP * d);
		let o = self.mofs(l, slot, 0);
		self.vecs.m()[o..o + MP * d].copy_from_slice(parts);
	}

	fn mofs(&self, l: usize, slot: usize, part: usize) -> usize {
		self.layers * self.bv() + ((l * H3_SLOTS + slot) * MP + part) * self.d
	}

	/// Submit job `j` (no fence; several may be in flight, in queue order),
	/// finished by [`H3Blk::end`]. Nothing may touch rows `r0 .. r1` of `a`,
	/// `xo`, `q`, `k`, `v` or the weights meanwhile.
	pub fn mid_begin(&mut self, j: &H3Mid<'_>) -> Result<(), String> {
		let rows = j.r1 - j.r0;
		let (d, a, kv, ff, bv) = (self.d, self.a, self.kv, self.ff, self.bv());
		let (o_pre, gu_pre, qkv_pre, gu_sg) = (0, a, a + d, a + 2 * d);
		if j.l == 0
			|| j.l >= self.layers
			|| rows == 0
			|| j.slot >= H3_SLOTS
			|| [j.x, j.xo].iter().any(|b| b.len() < j.r1 * d * 2)
			|| [j.a, j.q].iter().any(|b| b.len() < j.r1 * a * 2)
			|| [j.k, j.v].iter().any(|b| b.len() < j.r1 * kv * 2)
			|| j.w_o.len() != d * a
			|| j.w_gu.len() != 2 * ff * d
			|| j.w_d.len() != d * (ff + PAD)
			|| j.w_qkv.len() != (a + 2 * kv) * d
		{
			return Err(format!(
				"h3 block: job of block {} rows {}..{}: bad sizes",
				j.l, j.r0, j.r1
			));
		}
		let (vv, yy, xx1, xxq) = (
			self.vecs.range(),
			self.y.range(),
			self.x1.range(),
			self.xq.range(),
		);
		let (lp, lq) = ((j.l - 1) * bv, j.l * bv);
		let mp = |part: usize| self.mofs(j.l - 1, j.slot, part) as u32;
		let mq = |part: usize| self.mofs(j.l, j.slot, part) as u32;
		let (ms1, sh1) = (mq(0), mq(1));
		let (ms2, sh2, os, ds, xr) = (mp(2), mp(3), mp(4), mp(5), mp(6));
		let rowf = (self.layers * bv + self.layers * H3_SLOTS * MP * d) as u32;
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
		let u = |x: usize| x as u32;
		rig.submit(|rig, cmd| {
			let step = |k: usize, set, pc: [u32; 8], gx: usize, gy: usize| {
				rig.dispatch(cmd, k, set, &pc, gx, gy)
			};
			let mut r = j.r0;
			while r < j.r1 {
				let n = (j.r1 - r).min(CHUNK);
				let tiles = n.div_ceil(64);
				// y [n][nn] = xq [n][k] . w [nn][k] (row stride ldw bytes)
				let gemm = |set, k: usize, nn: usize, sofs: u32, fofs: u32, ldw: usize| {
					step(
						K_GEMM,
						set,
						[u(k), u(nn), u(n), u(nn), sofs, fofs, u(ldw), 0],
						nn / 64,
						tiles,
					)
				};
				step(
					K_ROW0,
					s_row0,
					[u(r), u(lp + o_pre), 0, 0, 0, 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[0], a, d, os, NONE, 0);
				step(
					K_ROW1,
					s_row1,
					[u(r), u(lp + gu_pre), xr, ms2, sh2, 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[1], d, 2 * ff, NONE, NONE, 0);
				step(K_ACT, s_act, [u(lp + gu_sg), rowf, 0, 0, 0, 0, 0, 0], n, 1);
				gemm(s_w[2], ff, d, ds, rowf, ff + PAD);
				step(
					K_ROW2,
					s_row2,
					[u(r), u(lq + qkv_pre), 0, ms1, sh1, 0, 0, 0],
					n,
					1,
				);
				gemm(s_w[3], d, a + 2 * kv, NONE, NONE, 0);
				step(
					K_QKV,
					s_qkv,
					[u(r), 0, 0, 0, 0, 0, 0, 0],
					(a + 2 * kv) / 2 / 256,
					n,
				);
				r += n;
			}
		})
	}

	/// Submissions in flight.
	pub fn in_flight(&self) -> usize {
		self.rig.in_flight()
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

	const D: usize = H3_D;
	const A: usize = H3_A;
	const FF: usize = H3_FF;

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

	/// `xq . w[r]` (rows `ld` bytes apart) x 2^-14.
	fn mm(xq: &[i32], w: &[u8], n: usize, k: usize, ld: usize) -> Vec<f32> {
		(0..n)
			.map(|r| {
				let s: i64 = (0..k)
					.map(|p| xq[p] as i64 * (w[r * ld + p] as i8) as i64)
					.sum();
				s as f32 / 16384.0
			})
			.collect()
	}

	fn rms(x: &[f32]) -> Vec<f32> {
		let ms = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64;
		let inv = 1.0 / (ms + 6.2e-5).sqrt();
		x.iter().map(|&v| (v as f64 * inv) as f32).collect()
	}

	/// One `h3/mid` job on random weights against a host model of the same
	/// math (two rows checked, a 300-row job timed), on `QWEN_TEST_ADAPTER`
	/// (`intel`): the transformer's widths and the text encoder's.
	#[test]
	fn h3_mid_matches_reference() {
		mid_check([D, A, A, FF], "transformer");
		mid_check(TE, "text encoder");
	}

	fn mid_check([dd, aa, kv, ff]: [usize; 4], what: &str) {
		let pick = std::env::var("QWEN_TEST_ADAPTER").unwrap_or_else(|_| "intel".into());
		let Ok(mut g) = H3Blk::new(Some(&pick), 2, dd, aa, kv, ff) else {
			return;
		};
		let mut seed = 0x9e37_79b9_7f4a_7c15u64;
		let mut wq = |n: usize| -> HostVec<u8> {
			let mut w = HostVec::<u8>::new(n);
			for b in w.m() {
				*b = ((rnd(&mut seed) * 254.0) as i8) as u8;
			}
			w
		};
		let (w_o, w_gu, w_d, w_qkv) = (
			wq(dd * aa),
			wq(2 * ff * dd),
			wq(dd * (ff + PAD)),
			wq((aa + 2 * kv) * dd),
		);
		let mut seed = 7u64;
		let mut vec = |n: usize, a: f32, b: f32| -> Vec<f32> {
			(0..n).map(|_| a + b * rnd(&mut seed)).collect()
		};
		let blk: Vec<Vec<f32>> = vec![
			vec(aa, 0.0, 0.2),
			vec(dd, 0.0, 0.2),
			vec(dd, 0.0, 0.2),
			vec(ff, 1.0, 0.5),
		];
		for l in 0..2 {
			g.set_block(l, [&blk[0], &blk[1], &blk[2], &blk[3]]);
		}
		// entries: ms1, sh1, ms2, sh2, o.s, d.s, xr (slot 1 of each block)
		let ent: Vec<Vec<f32>> = (0..2)
			.map(|_| {
				let mut e = vec(dd, 1.0, 0.5);
				e.extend(vec(dd, 0.0, 0.3));
				e.extend(vec(dd, 1.0, 0.5));
				e.extend(vec(dd, 0.0, 0.3));
				e.extend(vec(dd, 0.5, 0.5));
				e.extend(vec(dd, 0.5, 0.5));
				e.extend(vec(dd, 0.8, 0.2));
				e
			})
			.collect();
		for l in 0..2 {
			g.set_entry(l, 1, &ent[l]);
			g.set_entry(l, 0, &vec(MP * dd, 9.0, 1.0));
		}
		let part = |l: usize, p: usize| &ent[l][p * dd..(p + 1) * dd];
		let rows = 300;
		let mut x = HostVec::<u8>::new(rows * dd * 2);
		let mut a = HostVec::<u8>::new(rows * aa * 2);
		for i in 0..rows * dd {
			x.m()[2 * i..2 * i + 2].copy_from_slice(&h(rnd(&mut seed) * 8.0));
		}
		for i in 0..rows * aa {
			a.m()[2 * i..2 * i + 2].copy_from_slice(&h(rnd(&mut seed) * 4.0));
		}
		let (xo, q, k, v) = (
			HostVec::<u8>::new(rows * dd * 2),
			HostVec::<u8>::new(rows * aa * 2),
			HostVec::<u8>::new(rows * kv * 2),
			HostVec::<u8>::new(rows * kv * 2),
		);
		let job = H3Mid {
			l: 1,
			x: x.s(),
			a: a.s(),
			xo: xo.s(),
			q: q.s(),
			k: k.s(),
			v: v.s(),
			r0: 0,
			r1: rows,
			w_o: w_o.s(),
			w_gu: w_gu.s(),
			w_d: w_d.s(),
			w_qkv: w_qkv.s(),
			slot: 1,
		};
		g.mid_begin(&job).unwrap();
		g.end().unwrap();
		let t = std::time::Instant::now();
		for _ in 0..3 {
			g.mid_begin(&job).unwrap();
			g.end().unwrap();
		}
		eprintln!(
			"h3 mid ({what}): {rows} rows {:.1} ms",
			t.elapsed().as_secs_f64() / 3.0 * 1e3
		);
		let rel = |got: &[f32], want: &[f32]| {
			let (mut e, mut w) = (0f64, 0f64);
			for (a, b) in got.iter().zip(want) {
				e += ((a - b) as f64).powi(2);
				w += (*b as f64).powi(2);
			}
			(e / w).sqrt()
		};
		for r in [3usize, 271] {
			let xr: Vec<f32> = (0..dd).map(|i| fh(x.s(), r * dd + i)).collect();
			let ar: Vec<f32> = (0..aa).map(|i| fh(a.s(), r * aa + i)).collect();
			let mut t: Vec<f32> = ar.iter().zip(&blk[0]).map(|(a, p)| a * p).collect();
			fwht(&mut t);
			let o = mm(&q8(&t, 128.0), w_o.s(), dd, aa, aa);
			let x1: Vec<f32> = (0..dd)
				.map(|i| xr[i] * part(0, 6)[i] + o[i] * part(0, 4)[i])
				.collect();
			let mut t: Vec<f32> = rms(&x1)
				.iter()
				.enumerate()
				.map(|(i, v)| (v * part(0, 2)[i] + part(0, 3)[i]) * blk[1][i])
				.collect();
			fwht(&mut t);
			let gu = mm(&q8(&t, 128.0), w_gu.s(), 2 * ff, dd, dd);
			let mut hid: Vec<f32> = (0..ff)
				.map(|i| {
					let gg = gu[i] * blk[3][i];
					gg / (1.0 + (-gg).exp()) * (gu[ff + i] / 256.0)
				})
				.collect();
			fwht(&mut hid);
			let m = hid.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-3);
			let dn: Vec<f32> = mm(&q8(&hid, 127.0 / m), w_d.s(), dd, ff, ff + PAD)
				.iter()
				.zip(part(0, 5))
				.map(|(y, s)| y * s * m * 128.0 / 127.0)
				.collect();
			let x2: Vec<f32> = (0..dd).map(|i| x1[i] + dn[i]).collect();
			let mut t: Vec<f32> = rms(&x2)
				.iter()
				.enumerate()
				.map(|(i, v)| (v * part(1, 0)[i] + part(1, 1)[i]) * blk[2][i])
				.collect();
			fwht(&mut t);
			let qkv = mm(&q8(&t, 128.0), w_qkv.s(), aa + 2 * kv, dd, dd);
			let gx: Vec<f32> = (0..dd).map(|i| fh(xo.s(), r * dd + i)).collect();
			let gr =
				|b: &[u8], n: usize| -> Vec<f32> { (0..n).map(|i| fh(b, r * n + i)).collect() };
			let e = [
				rel(&gx, &x2),
				rel(&gr(q.s(), aa), &qkv[..aa]),
				rel(&gr(k.s(), kv), &qkv[aa..aa + kv]),
				rel(&gr(v.s(), kv), &qkv[aa + kv..]),
			];
			eprintln!(
				"h3 mid ({what}) row {r}: rel. error xo / q / k / v {:.2e} {:.2e} {:.2e} {:.2e}",
				e[0], e[1], e[2], e[3]
			);
			assert!(e.iter().all(|&x| x < 2e-2), "row {r}: {e:?}");
		}
	}
}
