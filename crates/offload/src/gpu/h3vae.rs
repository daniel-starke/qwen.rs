//! MiniMax-H3-Image's VAE decoder ViT on a Vulkan device (the iGPU), for a
//! share of the tiles next to the NPU (`imagegen::h3::npuvae`).
//!
//! Tiles are independent through the decoder, so the iGPU takes whole
//! tiles through all layers and the head while the NPU runs the others:
//! it reads their tokens (f16, residual scale included) from the NPU's
//! input buffer and writes their head output (f16) into the NPU's output
//! buffer. The NPU graphs' math (`npubuild::h3img::vdec_graph`,
//! `vout_graph`) on the same folded f16 weights (imported zero-copy), in
//! f32: `h3v_norm` (RMSNorm; the head's LayerNorm), `h3v_gemm` (f16
//! weights, bias, optional residual add), `h3v_qk` (q / k RMSNorm and
//! RoPE), `h3v_attn` (attention within a tile), `h3v_elem` (f16 in / out,
//! SwiGLU). The residual stays f32 on the device between layers.

use super::hostkern::{HostVec, Rig};

/// Tiles of one pass (the scratch buffers' size).
const B: usize = 8;
/// Latent tokens per tile (the head's rows).
const TOK: usize = 256;
const HD: usize = 64;
const HALF: usize = 16;

const K_GEMM: usize = 0;
const K_ATTN: usize = 1;
const K_QK: usize = 2;
const K_RMS: usize = 3;
const K_LN: usize = 4;
const K_IN: usize = 5;
const K_ACT: usize = 6;
const K_OUT: usize = 7;

/// The decoder's shape.
#[derive(Clone, Copy, Debug)]
pub struct H3VaeShape {
	/// Width (heads x 64).
	pub d: usize,
	pub ff: usize,
	/// Tokens per tile.
	pub s: usize,
	/// Output values per latent token.
	pub out: usize,
	pub layers: usize,
	/// The layers' RMSNorm and the head's LayerNorm epsilon (on the
	/// scaled residual), and q / k's.
	pub eps: f32,
	pub eps_qk: f32,
}

/// See the module docs.
pub struct H3Vae {
	rig: Rig,
	c: H3VaeShape,
	/// Per layer its weights' host ranges: `wqkv`, `bqkv`, `wo`, `bo`, `w0`,
	/// `b0`, `w2`, `b2` (f16); the head's `g`, `b`, `w`, `bo`.
	w: Vec<[(usize, usize); 8]>,
	head: [(usize, usize); 4],
	/// `cos`, `sin` [s][16]
	vecs: HostVec<f32>,
	x: HostVec<f32>,
	h: HostVec<f32>,
	qkv: HostVec<f32>,
	att: HostVec<f32>,
	gu: HostVec<f32>,
	act: HostVec<f32>,
	y: HostVec<f32>,
}

// SAFETY: used from one thread at a time.
unsafe impl Send for H3Vae {}

fn rng(s: &[u8]) -> (usize, usize) {
	(s.as_ptr() as usize, s.len())
}

impl H3Vae {
	/// On the Vulkan adapter `pick` names, with the tokens' rotary
	/// `cos` / `sin` (`[s][16]`), per layer its folded f16 weights and the
	/// head's (`npubuild::h3img::VAE_WNAMES` order; `g`, `b`, `w`, `bo`).
	/// The weights must outlive this.
	pub fn new(
		pick: Option<&str>,
		c: H3VaeShape,
		cos: &[f32],
		sin: &[f32],
		w: &[[&[u8]; 8]],
		head: [&[u8]; 4],
	) -> Result<H3Vae, String> {
		let d = c.d;
		if d != 1024
			|| !d.is_multiple_of(HD)
			|| cos.len() != c.s * HALF
			|| sin.len() != c.s * HALF
			|| w.len() != c.layers
		{
			return Err(format!(
				"h3 vae: kernels for a 1024-wide decoder with {HALF} rotary pairs per head"
			));
		}
		let sizes = [
			3 * d * d,
			3 * d,
			d * d,
			d,
			2 * c.ff * d,
			2 * c.ff,
			d * c.ff,
			d,
		];
		if w.iter()
			.any(|l| l.iter().zip(&sizes).any(|(b, &n)| b.len() != 2 * n))
			|| head[2].len() != 2 * c.out * d
			|| !c.out.is_multiple_of(64)
		{
			return Err("h3 vae: weights of unexpected sizes".into());
		}
		let spvs: [&[u8]; 8] = [
			quant::packed!("h3v_gemm.spv"),
			quant::packed!("h3v_attn.spv"),
			quant::packed!("h3v_qk.spv"),
			quant::packed!("h3v_norm0.spv"),
			quant::packed!("h3v_norm1.spv"),
			quant::packed!("h3v_elem0.spv"),
			quant::packed!("h3v_elem1.spv"),
			quant::packed!("h3v_elem2.spv"),
		];
		let mut rig = Rig::new(pick, &spvs, 32, "h3 vae")?;
		let rows = B * c.s;
		let mut vecs = HostVec::new(2 * c.s * HALF);
		vecs.m()[..c.s * HALF].copy_from_slice(cos);
		vecs.m()[c.s * HALF..].copy_from_slice(sin);
		let x = HostVec::new(rows * d);
		let h = HostVec::new(rows * d);
		let qkv = HostVec::new(rows * 3 * d);
		let att = HostVec::new(rows * d);
		let gu = HostVec::new(rows * 2 * c.ff);
		let act = HostVec::new(rows * c.ff);
		let y = HostVec::new(B * TOK * c.out);
		let wr: Vec<[(usize, usize); 8]> = w.iter().map(|l| l.map(rng)).collect();
		let hr = head.map(rng);
		rig.own = [&vecs, &x, &h, &qkv, &att, &gu, &act, &y]
			.iter()
			.map(|v| v.range().0)
			.collect();
		rig.own.extend(wr.iter().flatten().chain(&hr).map(|r| r.0));
		Ok(H3Vae {
			rig,
			c,
			w: wr,
			head: hr,
			vecs,
			x,
			h,
			qkv,
			att,
			gu,
			act,
			y,
		})
	}

	/// Submit tiles `t0 .. t1` (no fence; finished by [`H3Vae::end`]): their
	/// tokens from `xin` (f16 `[tiles][s][d]`), their head output into
	/// `yout` (f16 `[tiles][256][out]`). Nothing may touch those rows
	/// meanwhile.
	pub fn decode_begin(
		&mut self,
		xin: &[u8],
		yout: &[u8],
		t0: usize,
		t1: usize,
	) -> Result<(), String> {
		let c = self.c;
		let (d, s, f) = (c.d, c.s, c.ff);
		if t1 <= t0 || xin.len() < t1 * s * d * 2 || yout.len() < t1 * TOK * c.out * 2 {
			return Err(format!("h3 vae: tiles {t0}..{t1}: bad sizes"));
		}
		let (vv, xx, hh, qq, aa, gg, ac, yy) = (
			self.vecs.range(),
			self.x.range(),
			self.h.range(),
			self.qkv.range(),
			self.att.range(),
			self.gu.range(),
			self.act.range(),
			self.y.range(),
		);
		let (xi, yo) = (rng(xin), rng(yout));
		let u = |x: usize| x as u32;
		let fb = |x: f32| x.to_bits();
		let groups = |n: usize| n.div_ceil(256);
		let mut t = t0;
		while t < t1 {
			let nt = (t1 - t).min(B);
			let rows = nt * s;
			for l in 0..=c.layers {
				let rig = &mut self.rig;
				if l < c.layers {
					let w = self.w[l];
					let s_in = rig.set([xi, vv, xx, vv, vv, vv])?;
					let s_rms = rig.set([vv, xx, hh, vv, vv, vv])?;
					let s_qkv = rig.set([w[0], hh, qq, w[1], vv, vv])?;
					let s_qk = rig.set([vv, qq, vv, vv, vv, vv])?;
					let s_attn = rig.set([vv, qq, aa, vv, vv, vv])?;
					let s_o = rig.set([w[2], aa, xx, w[3], vv, vv])?;
					let s_gu = rig.set([w[4], hh, gg, w[5], vv, vv])?;
					let s_act = rig.set([vv, gg, ac, vv, vv, vv])?;
					let s_dn = rig.set([w[6], ac, xx, w[7], vv, vv])?;
					let qs = 1.0 / (HD as f32).sqrt();
					rig.submit(|rig, cmd| {
						let gemm = |set, k: usize, n: usize, add: u32| {
							rig.dispatch(
								cmd,
								K_GEMM,
								set,
								&[u(k), u(n), u(rows), add],
								n / 64,
								rows.div_ceil(64),
							)
						};
						if l == 0 {
							let pairs = rows * d / 2;
							rig.dispatch(
								cmd,
								K_IN,
								s_in,
								&[u(pairs), u(t * s * d / 2)],
								groups(pairs),
								1,
							);
						}
						rig.dispatch(cmd, K_RMS, s_rms, &[fb(c.eps), u(s)], rows, 1);
						gemm(s_qkv, d, 3 * d, 0);
						let qk = rows * 2 * (d / HD);
						rig.dispatch(
							cmd,
							K_QK,
							s_qk,
							&[u(rows), u(s), u(d), 0, u(s * HALF), fb(c.eps_qk), fb(qs)],
							qk.div_ceil(64),
							1,
						);
						rig.dispatch3(
							cmd,
							K_ATTN,
							s_attn,
							&[u(s), u(d)],
							[s.div_ceil(64), d / HD, nt],
						);
						gemm(s_o, d, d, 1);
						rig.dispatch(cmd, K_RMS, s_rms, &[fb(c.eps), u(s)], rows, 1);
						gemm(s_gu, d, 2 * f, 0);
						rig.dispatch(cmd, K_ACT, s_act, &[u(rows * f), u(f)], groups(rows * f), 1);
						gemm(s_dn, f, d, 1);
					})?;
				} else {
					let hd = self.head;
					let s_ln = rig.set([vv, xx, hh, vv, hd[0], hd[1]])?;
					let s_w = rig.set([hd[2], hh, yy, hd[3], vv, vv])?;
					let s_out = rig.set([yo, yy, vv, vv, vv, vv])?;
					let ho = nt * TOK;
					rig.submit(|rig, cmd| {
						rig.dispatch(cmd, K_LN, s_ln, &[fb(c.eps), u(s)], ho, 1);
						rig.dispatch(
							cmd,
							K_GEMM,
							s_w,
							&[u(d), u(c.out), u(ho), 0],
							c.out / 64,
							ho.div_ceil(64),
						);
						let pairs = ho * c.out / 2;
						rig.dispatch(
							cmd,
							K_OUT,
							s_out,
							&[u(pairs), u(t * TOK * c.out / 2)],
							groups(pairs),
							1,
						);
					})?;
				}
			}
			t += nt;
		}
		Ok(())
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

	fn rnd(seed: &mut u64) -> f32 {
		*seed ^= *seed << 13;
		*seed ^= *seed >> 7;
		*seed ^= *seed << 17;
		(*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
	}

	/// f16 host buffer of `n` values `a + b rnd`, and the values it holds.
	fn f16v(seed: &mut u64, n: usize, a: f32, b: f32) -> (HostVec<u8>, Vec<f32>) {
		let mut hb = HostVec::<u8>::new(n * 2);
		let mut v = Vec::with_capacity(n);
		for i in 0..n {
			let h = quant::f32_to_fp16(a + b * rnd(seed));
			hb.m()[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
			v.push(quant::fp16_to_f32(h));
		}
		(hb, v)
	}

	/// `y[m][n] = x[m] . w[n] + b[n]`.
	fn lin(x: &[f32], w: &[f32], b: &[f32], k: usize) -> Vec<f32> {
		let n = b.len();
		let mut y = Vec::with_capacity(x.len() / k * n);
		for row in x.chunks_exact(k) {
			for (r, wr) in w.chunks_exact(k).enumerate() {
				y.push(row.iter().zip(wr).map(|(a, c)| (a * c) as f64).sum::<f64>() as f32 + b[r]);
			}
		}
		y
	}

	fn rmsn(x: &[f32], eps: f32) -> Vec<f32> {
		let ms = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64;
		let inv = 1.0 / (ms + eps as f64).sqrt();
		x.iter().map(|&v| (v as f64 * inv) as f32).collect()
	}

	/// Two tiles through one random layer and the head against a host model
	/// of the same math (rows of the second tile checked), on
	/// `QWEN_TEST_ADAPTER` (`intel`).
	#[test]
	fn h3_vae_matches_reference() {
		let pick = std::env::var("QWEN_TEST_ADAPTER").unwrap_or_else(|_| "intel".into());
		let (d, f, s, out) = (1024, 4096, 261, 768);
		let c = H3VaeShape {
			d,
			ff: f,
			s,
			out,
			layers: 1,
			eps: 1e-5 * 256.0,
			eps_qk: 6.2e-5,
		};
		let mut seed = 0x1234_5678_9abc_def1u64;
		let sc = 1.0 / (d as f32).sqrt();
		let (wqkv, vqkv) = f16v(&mut seed, 3 * d * d, 0.0, 2.0 * sc);
		let (bqkv, vbqkv) = f16v(&mut seed, 3 * d, 0.0, 0.2);
		let (wo, vwo) = f16v(&mut seed, d * d, 0.0, 2.0 * sc);
		let (bo, vbo) = f16v(&mut seed, d, 0.0, 0.2);
		let (w0, vw0) = f16v(&mut seed, 2 * f * d, 0.0, 2.0 * sc);
		let (b0, vb0) = f16v(&mut seed, 2 * f, 0.0, 0.2);
		let (w2, vw2) = f16v(&mut seed, d * f, 0.0, sc);
		let (b2, vb2) = f16v(&mut seed, d, 0.0, 0.2);
		let (hg, vhg) = f16v(&mut seed, d, 1.0, 0.5);
		let (hb, vhb) = f16v(&mut seed, d, 0.0, 0.2);
		let (hw, vhw) = f16v(&mut seed, out * d, 0.0, 2.0 * sc);
		let (hbo, vhbo) = f16v(&mut seed, out, 0.0, 0.2);
		let cos: Vec<f32> = (0..s * HALF).map(|i| (i as f32 * 0.37).cos()).collect();
		let sin: Vec<f32> = (0..s * HALF).map(|i| (i as f32 * 0.37).sin()).collect();
		let lw = [[
			wqkv.s(),
			bqkv.s(),
			wo.s(),
			bo.s(),
			w0.s(),
			b0.s(),
			w2.s(),
			b2.s(),
		]];
		let Ok(mut g) = H3Vae::new(
			Some(&pick),
			c,
			&cos,
			&sin,
			&lw,
			[hg.s(), hb.s(), hw.s(), hbo.s()],
		) else {
			return;
		};
		let tiles = 2;
		let (xin, vx) = f16v(&mut seed, tiles * s * d, 0.0, 16.0);
		let yout = HostVec::<u8>::new(tiles * TOK * out * 2);
		g.decode_begin(xin.s(), yout.s(), 0, tiles).unwrap();
		g.end().unwrap();
		let t = std::time::Instant::now();
		g.decode_begin(xin.s(), yout.s(), 0, tiles).unwrap();
		g.end().unwrap();
		eprintln!(
			"h3 vae: {tiles} tiles, one layer and the head {:.1} ms",
			t.elapsed().as_secs_f64() * 1e3
		);
		// the host model on tile 1
		let x = &vx[s * d..2 * s * d];
		let h: Vec<f32> = x.chunks_exact(d).flat_map(|r| rmsn(r, c.eps)).collect();
		let mut qkv = lin(&h, &vqkv, &vbqkv, d);
		for row in qkv.chunks_exact_mut(3 * d).enumerate() {
			let (r, row) = row;
			for (j, t) in row[..2 * d].chunks_exact_mut(HD).enumerate() {
				let n = rmsn(t, c.eps_qk);
				t.copy_from_slice(&n);
				for i in 0..HALF {
					let (a, b) = (t[i], t[i + HALF]);
					let (co, si) = (cos[r * HALF + i], sin[r * HALF + i]);
					t[i] = a * co - b * si;
					t[i + HALF] = b * co + a * si;
				}
				if j < d / HD {
					t.iter_mut().for_each(|v| *v /= 8.0);
				}
			}
		}
		let mut att = vec![0f32; s * d];
		for hh in 0..d / HD {
			for qi in 0..s {
				let q = &qkv[qi * 3 * d + hh * HD..][..HD];
				let sc: Vec<f64> = (0..s)
					.map(|k| {
						q.iter()
							.zip(&qkv[k * 3 * d + d + hh * HD..][..HD])
							.map(|(a, b)| (a * b) as f64)
							.sum()
					})
					.collect();
				let m = sc.iter().cloned().fold(f64::MIN, f64::max);
				let p: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
				let l: f64 = p.iter().sum();
				for c0 in 0..HD {
					let v: f64 = (0..s)
						.map(|k| p[k] * qkv[k * 3 * d + 2 * d + hh * HD + c0] as f64)
						.sum();
					att[qi * d + hh * HD + c0] = (v / l) as f32;
				}
			}
		}
		let rel = |got: &[f32], want: &[f32]| {
			let (mut e, mut w) = (0f64, 0f64);
			for (a, b) in got.iter().zip(want) {
				e += ((a - b) as f64).powi(2);
				w += (*b as f64).powi(2);
			}
			(e / w).sqrt()
		};
		for r in [0usize, 100, 255] {
			let o = lin(&att[r * d..(r + 1) * d], &vwo, &vbo, d);
			let x1: Vec<f32> = (0..d).map(|i| x[r * d + i] + o[i]).collect();
			let gu = lin(&rmsn(&x1, c.eps), &vw0, &vb0, d);
			let act: Vec<f32> = (0..f)
				.map(|i| gu[i] * (gu[f + i] / (1.0 + (-gu[f + i]).exp())))
				.collect();
			let dn = lin(&act, &vw2, &vb2, f);
			let x2: Vec<f32> = (0..d).map(|i| x1[i] + dn[i]).collect();
			let m = x2.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
			let var = x2.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / d as f64;
			let inv = 1.0 / (var + c.eps as f64).sqrt();
			let ln: Vec<f32> = (0..d)
				.map(|i| ((x2[i] as f64 - m) * inv) as f32 * vhg[i] + vhb[i])
				.collect();
			let want = lin(&ln, &vhw, &vhbo, d);
			let o0 = (TOK + r) * out;
			let got: Vec<f32> = (0..out)
				.map(|i| {
					quant::fp16_to_f32(u16::from_le_bytes([
						yout.s()[2 * (o0 + i)],
						yout.s()[2 * (o0 + i) + 1],
					]))
				})
				.collect();
			let e = rel(&got, &want);
			eprintln!("h3 vae tile 1 row {r}: rel. error {e:.2e}");
			assert!(e < 5e-3, "row {r}: {e}");
		}
	}
}
