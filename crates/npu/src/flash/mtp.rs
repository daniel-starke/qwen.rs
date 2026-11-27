//! The MTP head on the host (F9, `.npu` key `mtp = DIR`, the files of
//! `npubuild`'s `mtp` step): one attention + MoE layer that drafts the
//! token after next from the main model's streams after its last layer
//! (SGLang `sgl_qwen4_exp_mtp.py`, `_fuse_residual_linear_shared`):
//!
//! ```text
//! e     = fc_embedding(rms(emb(tok_q)) (1 + w_e))          [W]
//! x[c]  = e + fc_hidden(rms(h_{q-1}) (1 + w_h))[c]         per stream c
//! x     = layer(x)          (hyper-connections, attention, MoE)
//! logit = lm_head(final mixer(x));  x is the next draft's h
//! ```
//!
//! `rms(h)` is one norm over all `HC * W` values (not per stream). Every
//! matrix is int8 per row with f32 activations; the attention is dense
//! over the head's own cache of the last [`WIN`] positions (the indexer
//! selects all blocks below 2052 keys, so up to there this is the model's).
//! Its cache holds rows made from the main model's streams only; a chain's
//! draft rows are dropped after it ([`Mtp::truncate`]).
//!
//! With the dense GPU holding the head ([`GpuDec::add_mtp`],
//! `QWEN_FLASH_MTP_GPU`, default on) [`Mtp::run_on`] runs the dense part
//! there in two chained submissions per row; the attention, the routed
//! experts (2.5 GB, no room in VRAM) and the final mixer stay here. Per
//! draft row ~2.1 ms instead of ~2.6 (the experts' 49 MB from DRAM are
//! 0.9 of it). Drafts only, so the sums need not match the host's.

use std::path::Path;

use super::gpudec::GpuDec;
use super::{HC, HC_RANK, W};
use quant::parallel::par_for;

const HCW: usize = HC * W;
const HQ: usize = 24;
const HKV: usize = 2;
const DA: usize = 256;
const RD: usize = 64;
const NE: usize = 512;
const FE: usize = 640;
/// Positions the head's attention keeps.
pub const WIN: usize = 2048;

/// `[n][k]` int8 codes per row, f32 row scales.
struct M {
	c: Vec<i8>,
	s: Vec<f32>,
	n: usize,
	k: usize,
}

impl M {
	fn load(d: &Path, name: &str, n: usize, k: usize) -> Result<M, String> {
		let rd = |x: &str| {
			let p = d.join(format!("{name}.{x}"));
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		let c = rd("i8")?;
		let s: Vec<f32> = rd("s")?
			.chunks_exact(4)
			.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
			.collect();
		if c.len() != n * k || s.len() != n {
			return Err(format!("mtp/{name}: not [{n}][{k}]"));
		}
		// SAFETY: u8 and i8 have the same layout
		let c = unsafe { std::mem::transmute::<Vec<u8>, Vec<i8>>(c) };
		Ok(M { c, s, n, k })
	}

	/// `y[p] = W x[p]` for every position, each row read once.
	fn mv(&self, xs: &[&[f32]]) -> Vec<Vec<f32>> {
		mv_rows(&self.c, &self.s, self.k, 0..self.n, xs)
	}
}

/// Rows `rows` of `[.][k]` codes `c` / scales `s` times every `xs[p]`.
fn mv_rows(
	c: &[i8],
	s: &[f32],
	k: usize,
	rows: std::ops::Range<usize>,
	xs: &[&[f32]],
) -> Vec<Vec<f32>> {
	let (n, np) = (rows.len(), xs.len());
	let mut y = vec![0f32; np * n];
	#[derive(Clone, Copy)]
	struct P(*mut f32);
	unsafe impl Send for P {}
	unsafe impl Sync for P {}
	let py = P(y.as_mut_ptr());
	const RB: usize = 16;
	let r0 = rows.start;
	par_for(n.div_ceil(RB), |b| {
		let py = py;
		for r in b * RB..((b + 1) * RB).min(n) {
			let w = &c[(r0 + r) * k..][..k];
			for (p, x) in xs.iter().enumerate() {
				// SAFETY: AVX2 / FMA checked at load; block b owns rows
				// [b RB, (b + 1) RB) of every position's output
				unsafe {
					*py.0.add(p * n + r) = super::engine::dot_i8(w, x) * s[r0 + r];
				}
			}
		}
	});
	y.chunks_exact(n).map(|v| v.to_vec()).collect()
}

/// Weights per int4 group scale (`npubuild`'s `Q4G`).
const Q4G: usize = 32;

/// A routed expert matrix stack: int8 codes per row with f32 row scales,
/// or int4 codes (even element in the low nibble, value + 8) with an f16
/// scale per [`Q4G`] weights (`QWEN_FLASH_MTP_Q4=0`: the int8 files).
enum Ex {
	I8(Vec<i8>, Vec<f32>),
	Q4(Vec<u8>, Vec<u16>),
}

impl Ex {
	/// `NAME.q4` / `NAME.q4s` where the build wrote them (`q4done`) and
	/// they are not switched off, else `NAME.i8` / `NAME.s`.
	fn load(d: &Path, name: &str, n: usize, k: usize) -> Result<Ex, String> {
		let q4 = d.join("q4done").exists()
			&& std::env::var("QWEN_FLASH_MTP_Q4").map_or(true, |v| v != "0");
		if !q4 {
			let m = M::load(d, name, n, k)?;
			return Ok(Ex::I8(m.c, m.s));
		}
		let rd = |x: &str| {
			let p = d.join(format!("{name}.{x}"));
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		let c = rd("q4")?;
		let s: Vec<u16> = rd("q4s")?
			.chunks_exact(2)
			.map(|b| u16::from_le_bytes([b[0], b[1]]))
			.collect();
		if c.len() != n * k / 2 || s.len() != n * k / Q4G {
			return Err(format!("mtp/{name}: not [{n}][{k}] int4"));
		}
		Ok(Ex::Q4(c, s))
	}

	/// Rows `rows` (of `k` weights) times every `xs[p]`.
	fn mv_rows(&self, k: usize, rows: std::ops::Range<usize>, xs: &[&[f32]]) -> Vec<Vec<f32>> {
		match self {
			Ex::I8(c, s) => mv_rows(c, s, k, rows, xs),
			Ex::Q4(c, s) => mv_rows_q4(c, s, k, rows, xs),
		}
	}
}

/// [`mv_rows`] of int4 rows ([`Ex::Q4`]).
fn mv_rows_q4(
	c: &[u8],
	s: &[u16],
	k: usize,
	rows: std::ops::Range<usize>,
	xs: &[&[f32]],
) -> Vec<Vec<f32>> {
	let (n, np) = (rows.len(), xs.len());
	let mut y = vec![0f32; np * n];
	#[derive(Clone, Copy)]
	struct P(*mut f32);
	unsafe impl Send for P {}
	unsafe impl Sync for P {}
	let py = P(y.as_mut_ptr());
	const RB: usize = 16;
	let r0 = rows.start;
	par_for(n.div_ceil(RB), |b| {
		let py = py;
		for r in b * RB..((b + 1) * RB).min(n) {
			let w = &c[(r0 + r) * k / 2..][..k / 2];
			let sc = &s[(r0 + r) * k / Q4G..][..k / Q4G];
			for (p, x) in xs.iter().enumerate() {
				// SAFETY: AVX2 / FMA / F16C checked at load; block b owns
				// rows [b RB, (b + 1) RB) of every position's output
				unsafe {
					*py.0.add(p * n + r) = dot_q4(w, sc, x);
				}
			}
		}
	});
	y.chunks_exact(n).map(|v| v.to_vec()).collect()
}

/// An int4 row ([`Ex::Q4`]: `w` codes, `s` group scales) dotted with `x`.
///
/// # Safety
/// AVX2, FMA and F16C; `x` holds `2 w.len()` values, a multiple of [`Q4G`].
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4(w: &[u8], s: &[u16], x: &[f32]) -> f32 {
	use core::arch::x86_64::*;
	let m = _mm_set1_epi8(0x0f);
	let eight = _mm_set1_epi8(8);
	let mut acc = _mm256_setzero_ps();
	for (g, &sh) in s.iter().enumerate() {
		let b = _mm_loadu_si128(w.as_ptr().add(g * Q4G / 2) as *const __m128i);
		let lo = _mm_sub_epi8(_mm_and_si128(b, m), eight);
		let hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), m), eight);
		// elements 0..16 and 16..32 in order
		let e0 = _mm_unpacklo_epi8(lo, hi);
		let e1 = _mm_unpackhi_epi8(lo, hi);
		let xp = x.as_ptr().add(g * Q4G);
		let f = |q: __m128i| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
		let mut a = _mm256_mul_ps(f(e0), _mm256_loadu_ps(xp));
		a = _mm256_fmadd_ps(f(_mm_srli_si128(e0, 8)), _mm256_loadu_ps(xp.add(8)), a);
		a = _mm256_fmadd_ps(f(e1), _mm256_loadu_ps(xp.add(16)), a);
		a = _mm256_fmadd_ps(f(_mm_srli_si128(e1, 8)), _mm256_loadu_ps(xp.add(24)), a);
		let d = _mm256_cvtph_ps(_mm_set1_epi16(sh as i16));
		acc = _mm256_fmadd_ps(a, d, acc);
	}
	let mut t = [0f32; 8];
	_mm256_storeu_ps(t.as_mut_ptr(), acc);
	t.iter().sum()
}

fn vecf(d: &Path, name: &str, n: usize) -> Result<Vec<f32>, String> {
	let p = d.join(format!("{name}.f32"));
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	if b.len() != 4 * n {
		return Err(format!("mtp/{name}: not {n} values"));
	}
	Ok(b.chunks_exact(4)
		.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
		.collect())
}

fn sigmoid(x: f32) -> f32 {
	1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
	x * sigmoid(x)
}

/// `x / rms(x) * w` in place, over the whole of `x`.
fn rmsw(x: &mut [f32], w: &[f32]) {
	let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
	let r = 1.0 / (ms + 1e-6).sqrt();
	x.iter_mut().zip(w).for_each(|(a, w)| *a *= r * w);
}

struct Hc {
	norm: Vec<f32>,
	down: M,
	up: M,
	inj: Option<M>,
}

impl Hc {
	fn load(d: &Path, n: &str, inj: bool) -> Result<Hc, String> {
		Ok(Hc {
			norm: vecf(d, &format!("{n}.norm"), HCW)?,
			down: M::load(d, &format!("{n}.down"), HC_RANK, HCW)?,
			up: M::load(d, &format!("{n}.up"), HCW, HC_RANK)?,
			inj: if inj {
				Some(M::load(d, &format!("{n}.inj"), HC, HCW)?)
			} else {
				None
			},
		})
	}

	/// Per position: the mixed input `[W]`, the injection weights.
	fn apply(&self, xs: &[Vec<f32>]) -> Vec<(Vec<f32>, [f32; HC])> {
		let xn: Vec<Vec<f32>> = xs
			.iter()
			.map(|x| {
				let mut y = x.clone();
				for (r, w) in y.chunks_exact_mut(W).zip(self.norm.chunks_exact(W)) {
					rmsw(r, w);
				}
				y
			})
			.collect();
		let xr: Vec<&[f32]> = xn.iter().map(|v| v.as_slice()).collect();
		let mut m = self.down.mv(&xr);
		for v in m.iter_mut().flatten() {
			*v = silu(*v / HC as f32);
		}
		let mr: Vec<&[f32]> = m.iter().map(|v| v.as_slice()).collect();
		let u = self.up.mv(&mr);
		let j = self.inj.as_ref().map(|w| w.mv(&xr));
		(0..xs.len())
			.map(|p| {
				let mut mixed = vec![0f32; W];
				for c in 0..HC {
					for i in 0..W {
						let t = c * W + i;
						mixed[i] += sigmoid(u[p][t]) * xn[p][t] / HC as f32;
					}
				}
				let mut inj = [0f32; HC];
				if let Some(j) = &j {
					for c in 0..HC {
						inj[c] = 2.0 * sigmoid(j[p][c] / HC as f32);
					}
				}
				(mixed, inj)
			})
			.collect()
	}
}

/// One input row of the head: the token's embedding `[W]`, the streams
/// `h` `[HCW]` it follows, the rotary (cos, sin) of its position.
pub struct Row<'a> {
	pub emb: &'a [f32],
	pub h: &'a [f32],
	pub rope: [(f32, f32); RD / 2],
}

pub struct Mtp {
	pne: Vec<f32>,
	pnh: Vec<f32>,
	fce: M,
	fch: M,
	hcf: Hc,
	hca: Hc,
	hcm: Hc,
	q: M,
	k: M,
	v: M,
	o: M,
	qn: Vec<f32>,
	kn: Vec<f32>,
	router: M,
	sg: M,
	su: M,
	sd: M,
	sgate: Vec<f32>,
	/// `[NE][2 FE][W]` / `[NE][W][FE]`
	egu: Ex,
	ed: Ex,
	/// the attention cache, `[rows][HKV DA]` each (roped keys)
	kc: Vec<f32>,
	vc: Vec<f32>,
}

impl Mtp {
	pub fn load(d: &Path) -> Result<Mtp, String> {
		let t = std::time::Instant::now();
		let m = |n: &str, r: usize, k: usize| M::load(d, n, r, k);
		let egu = Ex::load(d, "egu", NE * 2 * FE, W)?;
		let ed = Ex::load(d, "ed", NE * W, FE)?;
		let h = Mtp {
			pne: vecf(d, "pne", W)?,
			pnh: vecf(d, "pnh", HCW)?,
			fce: m("fce", W, W)?,
			fch: m("fch", W, W)?,
			hcf: Hc::load(d, "hcf", false)?,
			hca: Hc::load(d, "hca", true)?,
			hcm: Hc::load(d, "hcm", true)?,
			q: m("q", 2 * HQ * DA, W)?,
			k: m("k", HKV * DA, W)?,
			v: m("v", HKV * DA, W)?,
			o: m("o", W, HQ * DA)?,
			qn: vecf(d, "qn", DA)?,
			kn: vecf(d, "kn", DA)?,
			router: m("router", NE, W)?,
			sg: m("sg", FE, W)?,
			su: m("su", FE, W)?,
			sd: m("sd", W, FE)?,
			sgate: vecf(d, "sgate", W)?,
			egu,
			ed,
			kc: Vec::new(),
			vc: Vec::new(),
		};
		eprintln!(
			"flash: MTP head loaded in {:.1} s (experts {})",
			t.elapsed().as_secs_f64(),
			if matches!(h.egu, Ex::Q4(..)) {
				"int4"
			} else {
				"int8"
			}
		);
		Ok(h)
	}

	/// Cached positions.
	pub fn len(&self) -> usize {
		self.kc.len() / (HKV * DA)
	}

	pub fn is_empty(&self) -> bool {
		self.kc.is_empty()
	}

	/// Keep the first `n` cached positions.
	pub fn truncate(&mut self, n: usize) {
		self.kc.truncate(n * HKV * DA);
		self.vc.truncate(n * HKV * DA);
	}

	pub fn clear(&mut self) {
		self.truncate(0);
	}

	/// Drop all but the last [`WIN`] positions once twice that many are
	/// cached.
	pub fn trim(&mut self) {
		if self.len() > 2 * WIN {
			let d = (self.len() - WIN) * HKV * DA;
			self.kc.drain(..d);
			self.vc.drain(..d);
		}
	}

	/// The rows in order, each appended to the cache; the last one attends
	/// to the cache and the rows before it. Returns the last row's streams
	/// after the layer (the next draft's `h`) and its final mixer's output
	/// (the lm_head's input). The rows before the last only leave their key
	/// and value in the cache: input fusion, the attention hyper-connection
	/// and the k / v projections, nothing of the attention or the MoE.
	pub fn run(&mut self, rows: &[Row<'_>]) -> (Vec<f32>, Vec<f32>) {
		let np = rows.len();
		let last = np - 1;
		// input fusion
		let en: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut e = r.emb.to_vec();
				rmsw(&mut e, &self.pne);
				e
			})
			.collect();
		let hn: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut h = r.h.to_vec();
				rmsw(&mut h, &self.pnh);
				h
			})
			.collect();
		let er: Vec<&[f32]> = en.iter().map(|v| v.as_slice()).collect();
		let e = self.fce.mv(&er);
		// fc_hidden per stream: every (position, stream) one input
		let hr: Vec<&[f32]> = hn.iter().flat_map(|h| h.chunks_exact(W)).collect();
		let hs = self.fch.mv(&hr);
		let xs: Vec<Vec<f32>> = (0..np)
			.map(|p| {
				let mut x = vec![0f32; HCW];
				for c in 0..HC {
					for i in 0..W {
						x[c * W + i] = e[p][i] + hs[p * HC + c][i];
					}
				}
				x
			})
			.collect();
		// attention: every row's key and value, the last row's query
		let mut ha = self.hca.apply(&xs);
		let ar: Vec<&[f32]> = ha.iter().map(|v| v.0.as_slice()).collect();
		let (k, v) = (self.k.mv(&ar), self.v.mv(&ar));
		let qg = self.q.mv(&ar[last..]).pop().unwrap();
		let rope = |t: &mut [f32], p: usize| {
			for i in 0..RD / 2 {
				let (c, s) = rows[p].rope[i];
				let (a, b) = (t[i], t[i + RD / 2]);
				t[i] = a * c - b * s;
				t[i + RD / 2] = b * c + a * s;
			}
		};
		for p in 0..np {
			let mut kr = k[p].clone();
			for h in 0..HKV {
				let r = &mut kr[h * DA..][..DA];
				rmsw(r, &self.kn);
				rope(r, p);
			}
			self.kc.extend_from_slice(&kr);
			self.vc.extend_from_slice(&v[p]);
		}
		let mut q = vec![0f32; HQ * DA];
		for h in 0..HQ {
			let r = &mut q[h * DA..][..DA];
			r.copy_from_slice(&qg[h * 2 * DA..][..DA]);
			rmsw(r, &self.qn);
			rope(r, last);
		}
		let att = self.attend(&q, &qg);
		// from here on the last row alone
		let mut x = xs.into_iter().next_back().unwrap();
		let (_, inj) = ha.pop().unwrap();
		let y = self.o.mv(&[&att]).pop().unwrap();
		for c in 0..HC {
			for i in 0..W {
				x[c * W + i] += inj[c] * y[i];
			}
		}
		// MoE
		let xl = std::slice::from_ref(&x);
		let (mixed, inj) = self.hcm.apply(xl).pop().unwrap();
		let mr = [mixed.as_slice()];
		let lg = self.router.mv(&mr);
		let y = self.moe(&mr, &lg).pop().unwrap();
		for c in 0..HC {
			for i in 0..W {
				x[c * W + i] += inj[c] * y[i];
			}
		}
		let (m, _) = self.hcf.apply(std::slice::from_ref(&x)).pop().unwrap();
		(x, m)
	}

	/// [`Self::run`] with the dense part on `g` when it holds the head
	/// ([`GpuDec::add_mtp`]); the attention, the routed experts and the
	/// final mixer stay on the host.
	pub fn run_on(
		&mut self,
		rows: &[Row<'_>],
		g: Option<&mut GpuDec>,
	) -> Result<(Vec<f32>, Vec<f32>), String> {
		match g.filter(|g| g.has_mtp()) {
			Some(g) => self.run_gpu(rows, g),
			None => Ok(self.run(rows)),
		}
	}

	fn run_gpu(
		&mut self,
		rows: &[Row<'_>],
		g: &mut GpuDec,
	) -> Result<(Vec<f32>, Vec<f32>), String> {
		let last = rows.len() - 1;
		let en: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut e = r.emb.to_vec();
				rmsw(&mut e, &self.pne);
				e
			})
			.collect();
		let hn: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut h = r.h.to_vec();
				rmsw(&mut h, &self.pnh);
				h
			})
			.collect();
		let mut pre = g.mtp_pre(&en, &hn)?;
		let rope = |t: &mut [f32], p: usize| {
			for i in 0..RD / 2 {
				let (c, s) = rows[p].rope[i];
				let (a, b) = (t[i], t[i + RD / 2]);
				t[i] = a * c - b * s;
				t[i + RD / 2] = b * c + a * s;
			}
		};
		for (p, r) in pre.iter_mut().enumerate() {
			for h in 0..HKV {
				let t = &mut r.k[h * DA..][..DA];
				rmsw(t, &self.kn);
				rope(t, p);
			}
			self.kc.extend_from_slice(&r.k);
			self.vc.extend_from_slice(&r.v);
		}
		let pl = pre.pop().unwrap();
		let mut q = vec![0f32; HQ * DA];
		for h in 0..HQ {
			let r = &mut q[h * DA..][..DA];
			r.copy_from_slice(&pl.q[h * 2 * DA..][..DA]);
			rmsw(r, &self.qn);
			rope(r, last);
		}
		let att = self.attend(&q, &pl.q);
		let post = g.mtp_post(&att, &pl.x, &pl.inj)?;
		let mut x = post.x;
		let gt: f32 = post.m.iter().zip(&self.sgate).map(|(a, b)| a * b).sum();
		let mut y = vec![post
			.shared
			.iter()
			.map(|v| sigmoid(gt) * v)
			.collect::<Vec<f32>>()];
		self.routed(&[&post.m], std::slice::from_ref(&post.logits), &mut y);
		for c in 0..HC {
			for i in 0..W {
				x[c * W + i] += post.inj[c] * y[0][i];
			}
		}
		let (m, _) = self.hcf.apply(std::slice::from_ref(&x)).pop().unwrap();
		Ok((x, m))
	}

	/// The shared expert plus the routed experts of every position.
	fn moe(&self, xs: &[&[f32]], lg: &[Vec<f32>]) -> Vec<Vec<f32>> {
		let np = xs.len();
		let g = self.sg.mv(xs);
		let u = self.su.mv(xs);
		let a: Vec<Vec<f32>> = g
			.iter()
			.zip(&u)
			.map(|(g, u)| g.iter().zip(u).map(|(g, u)| silu(*g) * u).collect())
			.collect();
		let ar: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
		let so = self.sd.mv(&ar);
		let mut y: Vec<Vec<f32>> = (0..np)
			.map(|p| {
				let gt: f32 = xs[p].iter().zip(&self.sgate).map(|(a, b)| a * b).sum();
				so[p].iter().map(|v| sigmoid(gt) * v).collect()
			})
			.collect();
		self.routed(xs, lg, &mut y);
		y
	}

	/// Add the routed experts of every position `xs[p]` (router logits
	/// `lg[p]`) to `y[p]`.
	fn routed(&self, xs: &[&[f32]], lg: &[Vec<f32>], y: &mut [Vec<f32>]) {
		// per expert hit, its positions
		let sels: Vec<Vec<(usize, f32)>> = lg.iter().map(|l| super::engine::route(l)).collect();
		let mut users: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
		for (p, sel) in sels.iter().enumerate() {
			for &(e, w) in sel {
				match users.iter_mut().find(|u| u.0 == e) {
					Some(u) => u.1.push((p, w)),
					None => users.push((e, vec![(p, w)])),
				}
			}
		}
		for (e, us) in &users {
			let ux: Vec<&[f32]> = us.iter().map(|&(p, _)| xs[p]).collect();
			let gu = self.egu.mv_rows(W, e * 2 * FE..(e + 1) * 2 * FE, &ux);
			let h: Vec<Vec<f32>> = gu
				.iter()
				.map(|v| (0..FE).map(|i| silu(v[i]) * v[FE + i]).collect())
				.collect();
			let hr: Vec<&[f32]> = h.iter().map(|v| v.as_slice()).collect();
			let o = self.ed.mv_rows(FE, e * W..(e + 1) * W, &hr);
			for (&(p, w), o) in us.iter().zip(&o) {
				for (a, b) in y[p].iter_mut().zip(o) {
					*a += w * b;
				}
			}
		}
	}

	/// The attention of the last row's normed, roped query `q` `[HQ DA]`
	/// over the cache, gated by `qg`'s gate halves: `[HQ DA]`.
	fn attend(&self, q: &[f32], qg: &[f32]) -> Vec<f32> {
		let scale = 1.0 / (DA as f32).sqrt();
		let (kc, vc) = (&self.kc, &self.vc);
		let mut att = vec![0f32; HQ * DA];
		{
			let end = self.len();
			let start = end.saturating_sub(WIN);
			#[derive(Clone, Copy)]
			struct P(*mut f32);
			unsafe impl Send for P {}
			unsafe impl Sync for P {}
			let po = P(att.as_mut_ptr());
			par_for(HQ, |h| {
				let po = po;
				let kh = h / (HQ / HKV);
				let qr = &q[h * DA..][..DA];
				let sc: Vec<f32> = (start..end)
					.map(|u| {
						qr.iter()
							.zip(&kc[(u * HKV + kh) * DA..][..DA])
							.map(|(a, b)| a * b)
							.sum::<f32>() * scale
					})
					.collect();
				let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
				let sum: f32 = e.iter().sum();
				// SAFETY: task h owns att[h DA..][..DA]
				let o = unsafe { std::slice::from_raw_parts_mut(po.0.add(h * DA), DA) };
				for (u, w) in (start..end).zip(&e) {
					let w = w / sum;
					for (a, b) in o.iter_mut().zip(&vc[(u * HKV + kh) * DA..][..DA]) {
						*a += w * b;
					}
				}
				let g = &qg[h * 2 * DA + DA..][..DA];
				for (a, g) in o.iter_mut().zip(g) {
					*a *= sigmoid(*g);
				}
			});
		}
		att
	}
}

#[cfg(test)]
impl Mtp {
	/// Every row through the whole layer (the reference for [`Mtp::run`]).
	fn run_full(&mut self, rows: &[Row<'_>]) -> Vec<(Vec<f32>, Vec<f32>)> {
		let np = rows.len();
		// input fusion
		let en: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut e = r.emb.to_vec();
				rmsw(&mut e, &self.pne);
				e
			})
			.collect();
		let hn: Vec<Vec<f32>> = rows
			.iter()
			.map(|r| {
				let mut h = r.h.to_vec();
				rmsw(&mut h, &self.pnh);
				h
			})
			.collect();
		let er: Vec<&[f32]> = en.iter().map(|v| v.as_slice()).collect();
		let e = self.fce.mv(&er);
		// fc_hidden per stream: every (position, stream) one input
		let hr: Vec<&[f32]> = hn.iter().flat_map(|h| h.chunks_exact(W)).collect();
		let hs = self.fch.mv(&hr);
		let mut xs: Vec<Vec<f32>> = (0..np)
			.map(|p| {
				let mut x = vec![0f32; HCW];
				for c in 0..HC {
					for i in 0..W {
						x[c * W + i] = e[p][i] + hs[p * HC + c][i];
					}
				}
				x
			})
			.collect();
		// attention
		let ha = self.hca.apply(&xs);
		let ar: Vec<&[f32]> = ha.iter().map(|v| v.0.as_slice()).collect();
		let (qg, k, v) = (self.q.mv(&ar), self.k.mv(&ar), self.v.mv(&ar));
		let base = self.len();
		let mut qs = Vec::with_capacity(np);
		for p in 0..np {
			let rope = |t: &mut [f32]| {
				for i in 0..RD / 2 {
					let (c, s) = rows[p].rope[i];
					let (a, b) = (t[i], t[i + RD / 2]);
					t[i] = a * c - b * s;
					t[i + RD / 2] = b * c + a * s;
				}
			};
			let mut q = vec![0f32; HQ * DA];
			for h in 0..HQ {
				let r = &mut q[h * DA..][..DA];
				r.copy_from_slice(&qg[p][h * 2 * DA..][..DA]);
				rmsw(r, &self.qn);
				rope(r);
			}
			let mut kr = k[p].clone();
			for h in 0..HKV {
				let r = &mut kr[h * DA..][..DA];
				rmsw(r, &self.kn);
				rope(r);
			}
			self.kc.extend_from_slice(&kr);
			self.vc.extend_from_slice(&v[p]);
			qs.push(q);
		}
		let scale = 1.0 / (DA as f32).sqrt();
		let (kc, vc) = (&self.kc, &self.vc);
		let mut att = vec![vec![0f32; HQ * DA]; np];
		for (p, out) in att.iter_mut().enumerate() {
			let end = base + p + 1;
			let start = end.saturating_sub(WIN);
			#[derive(Clone, Copy)]
			struct P(*mut f32);
			unsafe impl Send for P {}
			unsafe impl Sync for P {}
			let po = P(out.as_mut_ptr());
			let q = &qs[p];
			par_for(HQ, |h| {
				let po = po;
				let kh = h / (HQ / HKV);
				let qr = &q[h * DA..][..DA];
				let sc: Vec<f32> = (start..end)
					.map(|u| {
						qr.iter()
							.zip(&kc[(u * HKV + kh) * DA..][..DA])
							.map(|(a, b)| a * b)
							.sum::<f32>() * scale
					})
					.collect();
				let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
				let sum: f32 = e.iter().sum();
				// SAFETY: task h owns out[h DA..][..DA]
				let o = unsafe { std::slice::from_raw_parts_mut(po.0.add(h * DA), DA) };
				for (u, w) in (start..end).zip(&e) {
					let w = w / sum;
					for (a, b) in o.iter_mut().zip(&vc[(u * HKV + kh) * DA..][..DA]) {
						*a += w * b;
					}
				}
				let g = &qg[p][h * 2 * DA + DA..][..DA];
				for (a, g) in o.iter_mut().zip(g) {
					*a *= sigmoid(*g);
				}
			});
		}
		let atr: Vec<&[f32]> = att.iter().map(|v| v.as_slice()).collect();
		let y = self.o.mv(&atr);
		for ((x, (_, inj)), y) in xs.iter_mut().zip(&ha).zip(&y) {
			for c in 0..HC {
				for i in 0..W {
					x[c * W + i] += inj[c] * y[i];
				}
			}
		}
		// MoE
		let hm = self.hcm.apply(&xs);
		let mr: Vec<&[f32]> = hm.iter().map(|v| v.0.as_slice()).collect();
		let lg = self.router.mv(&mr);
		let y = self.moe(&mr, &lg);
		for ((x, (_, inj)), y) in xs.iter_mut().zip(&hm).zip(&y) {
			for c in 0..HC {
				for i in 0..W {
					x[c * W + i] += inj[c] * y[i];
				}
			}
		}
		let f = self.hcf.apply(&xs);
		xs.into_iter().zip(f).map(|(x, (m, _))| (x, m)).collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// `cargo test --release -p npu flash::mtp::tests::bench -- --ignored
	/// --nocapture`: the head's layer on the build's files
	/// (`QWEN_FLASH_MTP_DIR`: a Flash-Next build's `mtp`) after 200
	/// cached rows; ms per run of 1 / 3 rows, and of its matrices.
	#[test]
	#[ignore]
	fn bench() {
		let d = std::env::var("QWEN_FLASH_MTP_DIR")
			.expect("set QWEN_FLASH_MTP_DIR to a Flash-Next build's mtp directory");
		let mut m = Mtp::load(Path::new(&d)).unwrap();
		let _w = quant::parallel::limit_width(quant::parallel::decode_threads());
		let mut seed = 7u64;
		let mut rnd = move || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
		};
		let embs: Vec<Vec<f32>> = (0..8).map(|_| (0..W).map(|_| rnd()).collect()).collect();
		let hs: Vec<Vec<f32>> = (0..8).map(|_| (0..HCW).map(|_| rnd()).collect()).collect();
		let row = |i: usize| Row {
			emb: &embs[i % 8],
			h: &hs[i % 8],
			rope: [((i as f32).cos(), (i as f32).sin()); RD / 2],
		};
		for i in 0..200 {
			m.run(&[row(i)]);
		}
		let ms = |f: &mut dyn FnMut()| {
			let mut v: Vec<f64> = (0..15)
				.map(|_| {
					let t = std::time::Instant::now();
					f();
					t.elapsed().as_secs_f64() * 1e3
				})
				.collect();
			v.sort_by(|a, b| a.partial_cmp(b).unwrap());
			v[v.len() / 2]
		};
		let base = m.len();
		for np in [1usize, 3] {
			let rows: Vec<Row<'_>> = (0..np).map(&row).collect();
			let t = ms(&mut || {
				m.run(&rows);
				m.truncate(base);
			});
			eprintln!("run of {np}: {t:.2} ms");
		}
		let x: Vec<f32> = (0..HCW).map(|_| rnd()).collect();
		let xs = [&x[..W]];
		let x6: Vec<f32> = (0..HQ * DA).map(|_| rnd()).collect();
		let xf: Vec<f32> = (0..FE).map(|_| rnd()).collect();
		let hx = vec![x.clone()];
		let t = |n: &str, f: &mut dyn FnMut()| eprintln!("  {n}: {:.3} ms", ms(f));
		t("fce", &mut || drop(m.fce.mv(&xs)));
		t("fch x4", &mut || {
			drop(
				m.fch
					.mv(&[&x[..W], &x[W..2 * W], &x[2 * W..3 * W], &x[3 * W..]]),
			)
		});
		t("hca", &mut || drop(m.hca.apply(&hx)));
		t("q", &mut || drop(m.q.mv(&xs)));
		t("k+v", &mut || {
			m.k.mv(&xs);
			m.v.mv(&xs);
		});
		t("o", &mut || drop(m.o.mv(&[&x6])));
		t("router", &mut || drop(m.router.mv(&xs)));
		t("sg+su+sd", &mut || {
			m.sg.mv(&xs);
			m.su.mv(&xs);
			m.sd.mv(&[&xf]);
		});
		let lg = m.router.mv(&xs);
		t("moe", &mut || drop(m.moe(&xs, &lg)));
	}
	/// `cargo test --release -p npu flash::mtp::tests::last_row -- --ignored
	/// --nocapture` (`QWEN_FLASH_MTP_DIR`): [`Mtp::run`] gives the last row
	/// what the whole layer over every row gives it, and the same cache.
	#[test]
	#[ignore]
	fn last_row() {
		let d = std::env::var("QWEN_FLASH_MTP_DIR").expect("QWEN_FLASH_MTP_DIR");
		let mut m = Mtp::load(Path::new(&d)).unwrap();
		let mut seed = 11u64;
		let mut rnd = move || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
		};
		let embs: Vec<Vec<f32>> = (0..5).map(|_| (0..W).map(|_| rnd()).collect()).collect();
		let hs: Vec<Vec<f32>> = (0..5).map(|_| (0..HCW).map(|_| rnd()).collect()).collect();
		let rows: Vec<Row<'_>> = (0..5)
			.map(|i| Row {
				emb: &embs[i],
				h: &hs[i],
				rope: [((i as f32).cos(), (i as f32).sin()); RD / 2],
			})
			.collect();
		let want = m.run_full(&rows).pop().unwrap();
		let (kc, vc) = (m.kc.clone(), m.vc.clone());
		m.clear();
		let got = m.run(&rows);
		assert!(kc == m.kc && vc == m.vc, "cache differs");
		for (a, b) in [(&want.0, &got.0), (&want.1, &got.1)] {
			let mx = a.iter().fold(0f32, |m, v| m.max(v.abs()));
			let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
			eprintln!("max {mx} diff {d}");
			assert!(d <= 1e-5 * mx, "{d} of {mx}");
		}
	}
}
