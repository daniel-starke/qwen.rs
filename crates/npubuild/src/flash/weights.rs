//! Qwen3.8-Flash-Next calibration and quantization: the float reference
//! ([`super::refm`]) runs the calibration chunks layer by layer; every
//! quantized MatMul input gets its Hessian and the static int8 range of its
//! rotated form; the matrices are quantized per row (the routed experts by
//! int4 GPTQ, the others int8).
//!
//! The files (`OUT/w/fl{L}.<name>.in`, the graphs' bound inputs) follow the
//! engine's v1 convention: a quantized linear with
//! input width `k` reads `<in>.pre` `[k]` f16 (rotation signs x `127 / r`),
//! `<m>.w` codes of `W R` `[n][k]` (int8 per row, round to nearest; int4
//! GPTQ for the routed experts) and `<m>.s` `[n]` f16 (`scale * r / 127 *
//! 256`); in the graph `fq(rotate(x * pre)) @ (w * 2^-8)^T * s`. The routed
//! experts' down input has a range per expert (`ed.pre` `[NE][FE]`). The
//! hyper-connection matrices, the router, norms and the small per-layer
//! vectors stay f16. `cal/F{L}.cal` (text: the ranges) marks a layer done.

use std::path::{Path, PathBuf};

use super::refm::{self as fr, Layer, Mixer, D, DA, FE, H, HCW, HKV, HQ, ID, IH, NE, NQKV};
use crate::g::f16_bits;
use crate::gptq::{self, signs};
use crate::la::{self, Acc};
use crate::st::Checkpoint;
use crate::weights::write;
use npu::flash::{ngram, NL, W};

/// Positions per calibration chunk.
pub const CHUNK: usize = 256;

/// Calibration tokens a routed expert needs for a down_proj input range of
/// its own, and the headroom on it.
const ED_MIN_TOKENS: usize = 16;
const ED_HEADROOM: f32 = 1.25;

/// The rotation block for an input of width `k`.
pub fn rot_block(k: usize) -> usize {
	[256, 128, 64]
		.into_iter()
		.find(|b| k.is_multiple_of(*b))
		.expect("input width not a multiple of 64")
}

/// The rotation's signs of a `k`-wide input (the 27B's sequence).
pub fn sg(k: usize) -> Vec<f32> {
	signs(k)
}

/// Running statistics of one quantized input: Hessian, rows seen, the
/// largest magnitude of the rotated input.
pub struct Stat {
	pub k: usize,
	pub h: Vec<f32>,
	pub n: usize,
	pub r: f32,
}

impl Stat {
	pub fn new(k: usize) -> Stat {
		Stat {
			k,
			h: vec![0f32; k * k],
			n: 0,
			r: 0.0,
		}
	}

	/// Add the rows `x` `[s][k]`.
	pub fn add(&mut self, x: &[f32], s: usize) {
		let k = self.k;
		if s == 0 {
			return;
		}
		let xt = la::transpose(x, s, k);
		// SAFETY: xt is [k][s], h [k][k]
		unsafe {
			la::gemm_nt_raw(
				xt.as_ptr(),
				s,
				k,
				xt.as_ptr(),
				s,
				k,
				s,
				self.h.as_mut_ptr(),
				k,
				Acc::Add,
			)
		};
		self.n += s;
		let mut xr = x.to_vec();
		gptq::rotate_rows_b(&mut xr, k, &sg(k), rot_block(k));
		self.r = xr.iter().fold(self.r, |m, v| m.max(v.abs()));
	}
}

/// The GPTQ factor of the rotated Hessian `h` `[k][k]`.
fn factor(mut h: Vec<f32>, k: usize) -> Result<gptq::Factor, String> {
	gptq::rotate_sym_b(&mut h, k, &sg(k), rot_block(k));
	gptq::factor(h, k)
}

/// `w R` `[n][k]` quantized against the factor `f`: (packed int4 codes, row
/// scales).
fn q4(mut w: Vec<f32>, n: usize, k: usize, f: &gptq::Factor) -> (Vec<u8>, Vec<f32>) {
	gptq::rotate_rows_b(&mut w, k, &sg(k), rot_block(k));
	let sc = gptq::row_scales(&w, n, k);
	let q = gptq::gptq(w, n, f, &sc);
	(gptq::pack4(&q), sc)
}

/// `w R` `[n][k]` as int8 per row, round to nearest: (codes, row scales).
fn q8(mut w: Vec<f32>, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
	gptq::rotate_rows_b(&mut w, k, &sg(k), rot_block(k));
	let (q, sc) = gptq::quant_i8(&w, n, k);
	(q.iter().map(|&v| v as u8).collect(), sc)
}

fn f16s(v: impl Iterator<Item = f64>) -> Vec<u8> {
	v.flat_map(|x| f16_bits(x).to_le_bytes()).collect()
}

fn f16v(v: &[f32]) -> Vec<u8> {
	f16s(v.iter().map(|&x| x as f64))
}

/// A hyper-connection matrix `[n][k]` as int8 per row for the graphs'
/// weight-only form (`graphs::wlin`: codes x 2^-7, activation f16): (codes,
/// f16 row scales `max |w| 128 / 127`).
pub fn hc_i8(w: &[f32], k: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
	let mut c = Vec::with_capacity(w.len());
	let mut sc = Vec::with_capacity(w.len() / k);
	for r in w.chunks_exact(k) {
		let m = r.iter().fold(0f32, |a, v| a.max(v.abs()));
		let st = if m > 0.0 { m / 127.0 } else { 1.0 };
		c.extend(
			r.iter()
				.map(|&v| (v / st).round().clamp(-127.0, 127.0) as i8 as u8),
		);
		let s = st as f64 * 128.0;
		if s < 6.2e-5 {
			return Err(format!(
				"hyper-connection row scale {s:e} is an f16 subnormal"
			));
		}
		sc.push(s);
	}
	Ok((c, f16s(sc.into_iter())))
}

/// `{p}.dinj.w` / `.s` (`graphs::HC_FOLD_INJ`): the down matrix's int8
/// codes and row scales (`hc_i8`) followed by the injection matrix's
/// (f16 `[HC][HCW]`), quantized the same way.
pub fn dinj(down_w: &[u8], down_s: &[u8], inj: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
	let v: Vec<f32> = inj
		.chunks_exact(2)
		.map(|c| crate::g::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect();
	let (c, s) = hc_i8(&v, HCW)?;
	Ok(([down_w, &c].concat(), [down_s, &s].concat()))
}

/// Writes one layer's files: `name("qkv.w")` is the path of `fl{L}.qkv.w.in`.
struct Out<'a> {
	name: &'a dyn Fn(&str) -> PathBuf,
}

impl Out<'_> {
	fn put(&self, n: &str, b: &[u8]) -> Result<(), String> {
		write(&(self.name)(n), b)
	}

	/// `<n>.pre`: signs x 127 / r of a `k`-wide input.
	fn pre(&self, n: &str, k: usize, r: f32) -> Result<(), String> {
		let s = sg(k);
		self.put(
			&format!("{n}.pre"),
			&f16s(s.iter().map(|&v| v as f64 * 127.0 / r as f64)),
		)
	}

	/// `<n>.w` / `<n>.s` of an int8 matrix whose input range is `r`: the
	/// scales x 2^16 (the graph takes the codes x 2^-16; at 2^8 most scales
	/// of small matrices were f16 subnormals, zero on the NPU).
	fn mat(&self, n: &str, codes: &[u8], sc: &[f32], r: f32) -> Result<(), String> {
		self.put(&format!("{n}.w"), codes)?;
		self.put(
			&format!("{n}.s"),
			&f16s(sc.iter().map(|&s| s as f64 * r as f64 / 127.0 * 65536.0)),
		)
	}

	fn hc(&self, n: &str, h: &fr::Hc) -> Result<(), String> {
		self.put(&format!("{n}.norm"), &f16v(&h.norm))?;
		self.put(&format!("{n}.down"), &f16v(&h.down))?;
		self.put(&format!("{n}.up"), &f16v(&h.up))?;
		for (m, w, k) in [("down", &h.down, HCW), ("up", &h.up, npu::flash::HC_RANK)] {
			let (c, sc) = hc_i8(w, k)?;
			self.put(&format!("{n}.{m}.w"), &c)?;
			self.put(&format!("{n}.{m}.s"), &sc)?;
		}
		if let Some(i) = &h.inject {
			self.put(&format!("{n}.inj"), &f16v(i))?;
			let (dc, ds) = hc_i8(&h.down, HCW)?;
			let (c, s) = dinj(&dc, &ds, &f16v(i))?;
			self.put(&format!("{n}.dinj.w"), &c)?;
			self.put(&format!("{n}.dinj.s"), &s)?;
		}
		Ok(())
	}
}

/// What calibration collects of one layer.
struct LayerStats {
	proj: Stat,
	out: Stat,
	moe: Stat,
	sdown: Stat,
	/// per expert: Hessian `[FE][FE]` and token count (allocated on use)
	edown: Vec<Option<Stat>>,
	/// the PLE layer's input (the n-gram embeddings)
	ple: Option<Stat>,
}

impl LayerStats {
	fn new(ple: bool) -> LayerStats {
		LayerStats {
			proj: Stat::new(W),
			out: Stat::new(H * D),
			moe: Stat::new(W),
			sdown: Stat::new(FE),
			edown: (0..NE).map(|_| None).collect(),
			ple: ple.then(|| Stat::new(W)),
		}
	}
}

/// Quantize layer `l` and write its files; returns a one-line summary.
fn layer_files(
	ck: &Checkpoint,
	layer: Layer,
	st: LayerStats,
	out: &Out<'_>,
) -> Result<String, String> {
	let LayerStats {
		proj,
		out: so,
		moe,
		sdown,
		edown,
		ple,
	} = st;
	let (rp, ro, rm, rs) = (proj.r, so.r, moe.r, sdown.r);
	out.hc("hca", &layer.hc_a)?;
	out.hc("hcm", &layer.hc_m)?;
	// the mixer
	out.pre("proj", W, rp)?;
	out.pre("out", H * D, ro)?;
	match layer.mixer {
		Mixer::Delta(d) => {
			let (c, s) = q8(d.wqkv, NQKV, W);
			out.mat("qkv", &c, &s, rp)?;
			let (c, s) = q8(d.wz, H * D, W);
			out.mat("z", &c, &s, rp)?;
			// in_proj_a / b: int8 per row, round to nearest
			let mut ab = d.wa;
			ab.extend_from_slice(&d.wb);
			gptq::rotate_rows_b(&mut ab, W, &sg(W), rot_block(W));
			let (qab, sab) = gptq::quant_i8(&ab, 2 * H, W);
			out.put("ab.w", &qab.iter().map(|&v| v as u8).collect::<Vec<u8>>())?;
			out.put(
				"ab.s",
				&f16s(sab.iter().map(|&s| s as f64 * rp as f64 / 127.0 * 65536.0)),
			)?;
			out.put("conv", &f16v(&d.conv))?;
			out.put("dtb", &f16v(&d.dtb))?;
			out.put("acoef", &f16v(&d.acoef))?;
			out.put("gnw", &f16v(&d.gnw))?;
			let (c, s) = q8(d.wo, W, H * D);
			out.mat("o", &c, &s, ro)?;
		}
		Mixer::Attn(a) => {
			for (n, m, rows) in [
				("q", a.wq, 2 * HQ * DA),
				("k", a.wk, HKV * DA),
				("v", a.wv, HKV * DA),
				("iqk", a.wi, (IH + 1) * ID),
			] {
				let (c, s) = q8(m, rows, W);
				out.mat(n, &c, &s, rp)?;
			}
			out.put("qn", &f16v(&a.qnw))?;
			out.put("kn", &f16v(&a.knw))?;
			out.put("iqn", &f16v(&a.iqn))?;
			out.put("ikn", &f16v(&a.ikn))?;
			let (c, s) = q8(a.wo, W, HQ * DA);
			out.mat("o", &c, &s, ro)?;
		}
	}
	// the MoE: router f16, shared expert and routed experts int4
	let m = layer.moe;
	out.put("router", &f16v(&m.gate))?;
	out.put("sgate", &f16v(&m.sgate))?;
	out.pre("moe", W, rm)?;
	let fm = factor(moe.h, W)?;
	let mut sgu = m.sg.clone();
	sgu.extend_from_slice(&m.su);
	let (c, s) = q8(sgu, 2 * FE, W);
	out.mat("sgu", &c, &s, rm)?;
	out.pre("sd", FE, rs)?;
	let (c, s) = q8(m.sd.clone(), W, FE);
	out.mat("sd", &c, &s, rs)?;
	// routed experts: gate_up on the pooled factor, down on its own
	// Hessian blended with the pooled down Hessian
	let mut hpool = vec![0f32; FE * FE];
	let mut npool = 0usize;
	let mut re = 0f32;
	for e in edown.iter().flatten() {
		hpool.iter_mut().zip(&e.h).for_each(|(a, b)| *a += b);
		npool += e.n;
		re = re.max(e.r);
	}
	let mut ed_pre = Vec::with_capacity(NE * FE * 2);
	let (mut egu_w, mut egu_s, mut ed_w, mut ed_s) = (
		Vec::with_capacity(NE * 2 * FE * W / 2),
		Vec::with_capacity(NE * 2 * FE * 2),
		Vec::with_capacity(NE * W * FE / 2),
		Vec::with_capacity(NE * W * 2),
	);
	let mut unseen = 0;
	for (e, st) in edown.into_iter().enumerate() {
		let (gu, dn) = fr::expert(ck, &m, e)?;
		let (c, s) = q4(gu, 2 * FE, W, &fm);
		egu_w.extend_from_slice(&c);
		egu_s.extend(f16s(
			s.iter().map(|&v| v as f64 * rm as f64 / 127.0 * 256.0),
		));
		let mut h = vec![0f32; FE * FE];
		let pw = 0.1 / npool.max(1) as f32;
		match &st {
			Some(st) if st.n > 0 => {
				let ow = 1.0 / st.n as f32;
				for ((a, b), c) in h.iter_mut().zip(&st.h).zip(&hpool) {
					*a = b * ow + c * pw;
				}
			}
			_ => {
				unseen += 1;
				h.iter_mut().zip(&hpool).for_each(|(a, c)| *a = c * pw);
			}
		}
		let f = factor(h, FE)?;
		let (c, s) = q4(dn, W, FE, &f);
		// the down input's range is the expert's own (with headroom: it is
		// seen on few tokens), the layer's for rarely seen experts: one range
		// per layer put the input quantization error of a typical expert at
		// 4-8 % (others ~1 %)
		let r = match &st {
			Some(st) if st.n >= ED_MIN_TOKENS => (st.r * ED_HEADROOM).min(re),
			_ => re,
		};
		ed_pre.extend(f16s(sg(FE).iter().map(|&v| v as f64 * 127.0 / r as f64)));
		ed_w.extend_from_slice(&c);
		ed_s.extend(f16s(s.iter().map(|&v| v as f64 * r as f64 / 127.0 * 256.0)));
	}
	out.put("egu.w", &egu_w)?;
	out.put("egu.s", &egu_s)?;
	out.put("ed.w", &ed_w)?;
	out.put("ed.s", &ed_s)?;
	out.put("ed.pre", &ed_pre)?;
	// PLE (layer 1)
	let mut rple = 0f32;
	if let (Some(p), Some(st)) = (layer.ple, ple) {
		rple = st.r;
		out.pre("ple", W, st.r)?;
		let (c, s) = q8(p.key, HCW, W);
		out.mat("pk", &c, &s, st.r)?;
		let (c, s) = q8(p.value, W, W);
		out.mat("pv", &c, &s, st.r)?;
		out.put("nkey", &f16v(&p.nkey))?;
		out.put("nquery", &f16v(&p.nquery))?;
		out.put("nconv", &f16v(&p.nconv))?;
		out.put("pconv", &f16v(&p.conv))?;
	}
	Ok(format!(
		"ranges proj {rp:.2} out {ro:.2} moe {rm:.2} sdown {rs:.2} edown {re:.2}{}; {unseen} experts unseen",
		if rple > 0.0 {
			format!(" ple {rple:.2}")
		} else {
			String::new()
		}
	))
}

/// The record that marks layer `l` done.
pub fn cal_path(cal: &Path, l: usize) -> PathBuf {
	cal.join(format!("F{l}.cal"))
}

/// Calibrate and quantize every layer on `chunks` (each [`CHUNK`] ids):
/// `wdir/fl{L}.*.in`, `cal/F{L}.cal`. Finished layers are skipped (the
/// streams after the last one are kept in `cal/fresid.f32`).
pub fn calibrate(
	ck: &Checkpoint,
	table: &ngram::Table,
	chunks: &[Vec<u32>],
	cal: &Path,
	wdir: &Path,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	for d in [cal, wdir] {
		std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
	}
	let done = (0..NL).take_while(|&l| cal_path(cal, l).exists()).count();
	if done == NL {
		return Ok(());
	}
	let hash = ngram::Hash::new(npu::flash::VOCAB as u64, 0);
	let es: Vec<Vec<f32>> = chunks
		.iter()
		.map(|c| fr::ngram_embed(&hash, table, c))
		.collect::<Result<_, _>>()?;
	let resid = cal.join("fresid.f32");
	let n = chunks.len();
	let mut xs: Vec<Vec<f32>> = Vec::new();
	let mut start = 0;
	if done > 0 {
		if let Ok(b) = std::fs::read(&resid) {
			if b.len() == 16 + n * CHUNK * HCW * 4
				&& u64::from_le_bytes(b[..8].try_into().unwrap()) == done as u64
				&& u64::from_le_bytes(b[8..16].try_into().unwrap()) == n as u64
			{
				xs = b[16..]
					.chunks_exact(CHUNK * HCW * 4)
					.map(|c| {
						c.chunks_exact(4)
							.map(|v| f32::from_le_bytes(v.try_into().unwrap()))
							.collect()
					})
					.collect();
				start = done;
			}
		}
	}
	if xs.is_empty() {
		for c in chunks {
			assert_eq!(c.len(), CHUNK);
			xs.push(fr::embed(ck, c)?);
		}
	}
	for l in start..NL {
		let t0 = std::time::Instant::now();
		let layer = fr::load_layer(ck, l)?;
		let mut st = LayerStats::new(layer.ple.is_some());
		for (x, e) in xs.iter_mut().zip(&es) {
			if let Some(p) = st.ple.as_mut() {
				p.add(e, CHUNK);
			}
			let taps = fr::forward(ck, &layer, x, Some(e), CHUNK)?;
			if l < done {
				continue;
			}
			st.proj.add(&taps.proj, CHUNK);
			st.out.add(&taps.out, CHUNK);
			st.moe.add(&taps.moe, CHUNK);
			st.sdown.add(&taps.sdown, CHUNK);
			for (ei, (toks, a)) in taps.edown.iter().enumerate() {
				if !toks.is_empty() {
					st.edown[ei]
						.get_or_insert_with(|| Stat::new(FE))
						.add(a, toks.len());
				}
			}
		}
		let tf = t0.elapsed().as_secs_f64();
		if l < done {
			continue;
		}
		let name = |n: &str| wdir.join(format!("fl{l}.{n}.in"));
		let summary = layer_files(ck, layer, st, &Out { name: &name })?;
		let mut b = Vec::with_capacity(16 + n * CHUNK * HCW * 4);
		b.extend((l as u64 + 1).to_le_bytes());
		b.extend((n as u64).to_le_bytes());
		for x in &xs {
			x.iter().for_each(|v| b.extend(v.to_le_bytes()));
		}
		write(&resid, &b)?;
		write(&cal_path(cal, l), format!("{summary}\n").as_bytes())?;
		log(&format!(
			"layer {l}: forward {tf:.0} s, quantized in {:.0} s; {summary}",
			t0.elapsed().as_secs_f64() - tf
		));
	}
	// the final mixer (f16)
	let fm = fr::Hc::load_final(ck)?;
	let name = |n: &str| wdir.join(format!("flf.{n}.in"));
	Out { name: &name }.hc("hcf", &fm)?;
	Ok(())
}
