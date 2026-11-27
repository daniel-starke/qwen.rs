//! MiniMax-H3-Image's text encoder on the NPU (`cli H3_DIR --npu-build OUT`,
//! step `te`; `docs/h3image.md` "NPU backend").
//!
//! The text encoder's decoder layer is the transformer block of
//! [`crate::h3img`] without the modulation and with grouped queries, so it
//! runs the same graphs ([`h3img::dit_graph_ty`] over a [`DShape`] whose k
//! / v are [`DShape::kv`] wide) and the same files: the residual stored as
//! `x_l / c_l`, W8A8 with the 256-wide Hadamard rotation, static ranges for
//! QKV, out and gate / up, per row for down, q / k / v raw out of the graph
//! (the head norms, MRoPE and the causal attention on the cores). The
//! modulation file holds one entry: the RMSNorm weights as `ms1` / `ms2`,
//! zero shifts, the out and down rows' scales over `c_l` as `o.s` /
//! `d.s`.
//!
//! **Codes.** int8 per row by default; `QWEN_H3_TE_W4` at build time
//! (`mlp`: gate / up and down int4, `all`: every matrix int4; into its own
//! output directory) writes int4 codes per row (GPTQ on the rotated rows,
//! the 27B's clip-searched scales, `[-8, 7]`), recorded in
//! `w/te/te.codes` so the graphs are built to match.
//!
//! The calibration (the float layers over frozen prompts, edits and texts,
//! layer by layer, each layer's ranges and Hessians and then its GPTQ
//! codes) lives in `imagegen` (it needs the tokenizer and the vision
//! tower); this module quantizes and writes a layer
//! ([`write_layer`]) and builds the graphs.

use crate::g::{Graph, Ty};
use crate::gptq;
use crate::h3img::{self, DShape, HS, MPARTS};
use crate::qimage::{self, f16f, f16v, mulf16, ROT};
use crate::st::parse_json;
use crate::weights::Accel;
use crate::{mkdir, say};
use std::path::{Path, PathBuf};

/// The layers that run: the conditioning is `hidden_states[50]`.
pub const LAYERS: usize = 50;

/// The graph stems: [`h3img::S`] rows, then [`h3img::S_SMALL`] rows.
pub const TE_STEMS: [&str; 6] = [
	"h3te/pre",
	"h3te/mid",
	"h3te/post",
	"h3te/pre64",
	"h3te/mid64",
	"h3te/post64",
];

/// The text encoder's shape: `text_config` of `text_encoder/config.json`,
/// cut to its first `layers` layers.
pub fn shape(src: &Path, layers: usize) -> Result<DShape, String> {
	let p = src.join("text_encoder").join("config.json");
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	let j = parse_json(&b)?;
	let t = j
		.get("text_config")
		.ok_or(format!("{}: no text_config", p.display()))?;
	let u = |k: &str| -> Result<usize, String> {
		t.get(k)
			.and_then(|v| v.num())
			.map(|v| v as usize)
			.ok_or(format!("{}: no {k}", p.display()))
	};
	let hd = u("head_dim")?;
	if u("num_hidden_layers")? < layers {
		return Err(format!("{}: fewer than {layers} layers", p.display()));
	}
	Ok(DShape {
		d: u("hidden_size")?,
		heads: u("num_attention_heads")?,
		hd,
		ff: u("intermediate_size")?,
		layers,
		kv: u("num_key_value_heads")? * hd,
	})
}

/// The weights' directory (`OUT/w/te`).
pub fn wdir(out: &Path) -> PathBuf {
	out.join("w").join("te")
}

/// Layer `l`'s file (`te{l}.<name>.in`).
pub fn wfile(wdir: &Path, l: usize, name: &str) -> PathBuf {
	wdir.join(format!("te{l}.{name}.in"))
}

/// Layer `l`'s calibration record (`te{l}.cal`): the ranges of the rotated
/// QKV, out and gate / up inputs (headroom included), the residual's
/// largest magnitude after the layer, the MLP gate's and hidden state's.
pub fn cal_file(wdir: &Path, l: usize) -> PathBuf {
	wdir.join(format!("te{l}.cal"))
}

/// The input's largest magnitude (`te.in`), the stored scales (`te.scale.f32`:
/// `c_in`, then `c_l` per layer), the code types (`te.codes`).
pub fn in_file(wdir: &Path) -> PathBuf {
	wdir.join("te.in")
}

pub fn scale_file(wdir: &Path) -> PathBuf {
	wdir.join("te.scale.f32")
}

fn codes_file(wdir: &Path) -> PathBuf {
	wdir.join("te.codes")
}

/// The matrices of [`h3img::HNAMES`] in order.
pub const NAMES: [&str; 4] = h3img::HNAMES;

/// The code types `QWEN_H3_TE_W4` asks for (module docs).
pub fn codes_env() -> Result<[Ty; 4], String> {
	let (a, b) = (Ty::I8, Ty::I4);
	match std::env::var("QWEN_H3_TE_W4").as_deref() {
		Err(_) | Ok("0") | Ok("") => Ok([a; 4]),
		Ok("mlp") => Ok([a, a, b, b]),
		Ok("1") | Ok("all") => Ok([b; 4]),
		Ok(v) => Err(format!("QWEN_H3_TE_W4={v}: 0, mlp or all")),
	}
}

fn ty_name(t: Ty) -> &'static str {
	if t == Ty::I4 {
		"i4"
	} else {
		"i8"
	}
}

/// The code types of the set in `out` (`te.codes`), else [`codes_env`].
pub fn codes(out: &Path) -> Result<[Ty; 4], String> {
	let p = codes_file(&wdir(out));
	let Ok(t) = std::fs::read_to_string(&p) else {
		return codes_env();
	};
	let v: Vec<&str> = t.split_whitespace().collect();
	if v.len() != 4 {
		return Err(format!("{}: four code types", p.display()));
	}
	let mut r = [Ty::I8; 4];
	for (o, s) in r.iter_mut().zip(v) {
		*o = match s {
			"i8" => Ty::I8,
			"i4" => Ty::I4,
			_ => return Err(format!("{}: code type {s}", p.display())),
		};
	}
	Ok(r)
}

/// Record the code types the weights step uses (an error when the
/// directory already holds layers of other types).
pub fn set_codes(out: &Path, c: [Ty; 4]) -> Result<(), String> {
	let w = wdir(out);
	mkdir(&w)?;
	let p = codes_file(&w);
	if p.exists() {
		let have = codes(out)?;
		if have != c {
			return Err(format!(
				"{}: built with codes {:?}, QWEN_H3_TE_W4 asks for {:?} (use another output directory)",
				p.display(),
				have.map(ty_name),
				c.map(ty_name)
			));
		}
		return Ok(());
	}
	let t = c.map(ty_name).join(" ") + "\n";
	std::fs::write(&p, t).map_err(|e| format!("{}: {e}", p.display()))
}

/// Whether layer `l`'s files are complete (its modulation file is written
/// last).
pub fn layer_done(out: &Path, l: usize) -> bool {
	let w = wdir(out);
	wfile(&w, l, "mod").exists() && cal_file(&w, l).exists()
}

/// Whether every layer, the scales and the code types are there.
pub fn complete(out: &Path, c: &DShape) -> bool {
	let w = wdir(out);
	(0..c.layers).all(|l| layer_done(out, l)) && scale_file(&w).exists() && codes_file(&w).exists()
}

/// Layer `l`'s calibration record.
pub fn read_cal(out: &Path, l: usize) -> Result<[f32; 6], String> {
	let p = cal_file(&wdir(out), l);
	let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	let v: Vec<f32> = t
		.split_whitespace()
		.filter_map(|x| x.parse().ok())
		.collect();
	if v.len() != 6 || v[..4].iter().any(|&x| !(x > 0.0)) {
		return Err(format!("{}: bad record", p.display()));
	}
	Ok([v[0], v[1], v[2], v[3], v[4], v[5]])
}

/// The input rows' largest magnitude, written once by the calibration.
pub fn read_in(out: &Path) -> Result<f32, String> {
	let p = in_file(&wdir(out));
	let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	t.trim()
		.parse::<f32>()
		.ok()
		.filter(|&v| v > 0.0)
		.ok_or(format!("{}: bad value", p.display()))
}

pub fn write_in(out: &Path, xin: f32) -> Result<(), String> {
	let w = wdir(out);
	mkdir(&w)?;
	std::fs::write(in_file(&w), format!("{xin}\n")).map_err(|e| e.to_string())
}

/// The stored scale of the input (`c_in`) and of layer `l`'s output.
pub fn scale_in(out: &Path) -> Result<f32, String> {
	Ok(read_in(out)? / h3img::XS)
}

pub fn scale_of(out: &Path, l: usize) -> Result<f32, String> {
	Ok(read_cal(out, l)?[3] / h3img::XS)
}

/// One layer's float weights (row-major `[out][in]`).
pub struct LayerW {
	pub ln1: Vec<f32>,
	pub q: Vec<f32>,
	pub k: Vec<f32>,
	pub v: Vec<f32>,
	pub nq: Vec<f32>,
	pub nk: Vec<f32>,
	pub o: Vec<f32>,
	pub ln2: Vec<f32>,
	pub gate: Vec<f32>,
	pub up: Vec<f32>,
	pub down: Vec<f32>,
}

/// Codes and `{w}.s` (f16) of `w` `[n][k]` as type `ty` against the
/// unrotated input Hessian `h` `[k][k]` (GPTQ; `None`: round to nearest),
/// for an input of range `r` (0: per row): the graph computes `fq(x R) .
/// (codes 2^e)^T . s` (`e` = -7 for int8, -3 for int4).
#[allow(clippy::too_many_arguments)]
pub fn quantize(
	w: &[f32],
	n: usize,
	k: usize,
	r: f32,
	h: Option<Vec<f32>>,
	ty: Ty,
	accel: &Accel,
) -> Result<(Vec<u8>, Vec<u8>), String> {
	let sg = gptq::signs(k);
	let mut wr = w.to_vec();
	gptq::rotate_rows_b(&mut wr, k, &sg, ROT);
	let i4 = ty == Ty::I4;
	let (sc, ge) = if i4 {
		(gptq::row_scales(&wr, n, k), 8.0)
	} else {
		let s: Vec<f32> = wr
			.chunks_exact(k)
			.map(|row| {
				row.iter()
					.fold(0f32, |a, &v| a.max(v.abs()))
					.max(f32::MIN_POSITIVE)
					/ 127.0
			})
			.collect();
		(s, 128.0)
	};
	let (lo, hi) = if i4 { (-8.0, 7.0) } else { (-127.0, 127.0) };
	let q = match h {
		Some(mut h) => {
			gptq::rotate_sym_b(&mut h, k, &sg, ROT);
			let mut p = gptq::prepare(h, k);
			let x = match accel.inv_chol(&p.hr, k) {
				Some(x) => x,
				None => gptq::inv_chol(std::mem::take(&mut p.hr), k)?,
			};
			let f = p.finish(x);
			gptq::gptq_range(wr, n, &f, &sc, lo, hi)
		}
		None => wr
			.chunks_exact(k)
			.zip(&sc)
			.flat_map(|(row, &s)| {
				row.iter()
					.map(move |&v| (v / s).round_ties_even().clamp(lo, hi) as i8)
			})
			.collect(),
	};
	let fr = if r > 0.0 { 128.0 * r / 127.0 } else { 1.0 };
	let so: Vec<f32> = sc.iter().map(|&s| s * ge * fr).collect();
	let codes = if i4 {
		gptq::pack4(&q)
	} else {
		q.into_iter().map(|c| c as u8).collect()
	};
	Ok((codes, f16v(&so)))
}

/// Write layer `l`'s files (module docs) from its float weights `w`, its
/// calibration record `cal` ([`cal_file`]'s values), its inputs' Hessians
/// `hess` (full `[k][k]`, [`NAMES`] order; `None`: round to nearest), the
/// stored scales of its input `cp` and output `cl`. The calibration record
/// is written first, the modulation file last ([`layer_done`]).
#[allow(clippy::too_many_arguments)]
pub fn write_layer(
	out: &Path,
	l: usize,
	c: &DShape,
	w: LayerW,
	cal: [f32; 6],
	hess: Option<[Vec<f32>; 4]>,
	cp: f32,
	cl: f32,
	codes: [Ty; 4],
	accel: &Accel,
) -> Result<(), String> {
	let t = std::time::Instant::now();
	let (d, inner, kv, ff) = (c.d, c.inner(), c.kv, c.ff);
	let wd = wdir(out);
	mkdir(&wd)?;
	std::fs::write(
		cal_file(&wd, l),
		format!(
			"{} {} {} {} {} {}\n",
			cal[0], cal[1], cal[2], cal[3], cal[4], cal[5]
		),
	)
	.map_err(|e| e.to_string())?;
	let put = |name: &str, b: &[u8]| crate::weights::write(&wfile(&wd, l, name), b);
	let [r_qkv, r_o, r_gu, ..] = cal;
	let by_gptq = hess.is_some();
	let [h_qkv, h_o, h_gu, h_d] = match hess {
		Some(h) => h.map(Some),
		None => [None, None, None, None],
	};
	let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
	// q / k / v
	let mut qkv = w.q;
	qkv.extend(&w.k);
	qkv.extend(&w.v);
	drop((w.k, w.v));
	let (cq, sq) = quantize(&qkv, c.qkv(), d, r_qkv, h_qkv, codes[0], accel)?;
	drop(qkv);
	put("qkv.w", &cq)?;
	put("qkv.s", &sq)?;
	put("qkv.pre", &qimage::pre(d, r_qkv))?;
	let sv = f16f(&sq[(inner + kv) * 2..]);
	put("nq", &f32b(&w.nq))?;
	put("nk", &f32b(&w.nk))?;
	// the attention's output: v's row scales per channel, each head's
	// value head repeated over its query heads
	let group = c.heads * c.hd / kv;
	let sva: Vec<f32> = (0..inner)
		.map(|j| {
			let (h, i) = (j / c.hd, j % c.hd);
			sv[(h / group) * c.hd + i]
		})
		.collect();
	let (co, so) = quantize(&w.o, d, inner, r_o, h_o, codes[1], accel)?;
	put("o.w", &co)?;
	put("o.pre", &mulf16(&qimage::pre(inner, r_o), &sva))?;
	// SwiGLU: gate rows, then the up rows x the down input's rotation
	// signs; the up half's row scales into the columns of `d`
	let mut gu = w.gate;
	let sg = gptq::signs(ff);
	for (row, &s) in w.up.chunks_exact(d).zip(&sg) {
		gu.extend(row.iter().map(|&v| v * s));
	}
	drop(w.up);
	let (cg, sgu) = quantize(&gu, 2 * ff, d, r_gu, h_gu, codes[2], accel)?;
	drop(gu);
	let su = f16f(&sgu[ff * 2..]);
	put("gu.sg", &sgu[..ff * 2])?;
	put("gu.w", &cg)?;
	put("gu.pre", &qimage::pre(d, r_gu))?;
	let mut dw = w.down;
	for row in dw.chunks_exact_mut(ff) {
		row.iter_mut().zip(&su).for_each(|(w, &f)| *w *= f);
	}
	// the down input in the graph is the hidden state / the up rows' scales
	let h_d = h_d.map(|mut h| {
		for (a, row) in h.chunks_exact_mut(ff).enumerate() {
			row.iter_mut().zip(&su).for_each(|(v, &b)| *v /= su[a] * b);
		}
		h
	});
	let (cd, sd) = quantize(&dw, d, ff, 0.0, h_d, codes[3], accel)?;
	drop(dw);
	// int8: zero columns for the `rowq` kernel's pad lanes
	let cd = if codes[3] == Ty::I8 && qimage::rowq() {
		h3img::pad_cols(&cd, ff)
	} else {
		cd
	};
	put("d.w", &cd)?;
	// the one modulation entry
	let (so, sd) = (f16f(&so), f16f(&sd));
	let ms = h3img::mstride(d);
	let parts: [Vec<f32>; MPARTS] = [
		w.ln1,
		vec![0.0; d],
		w.ln2,
		vec![0.0; d],
		so.iter().map(|&s| s / cl).collect(),
		sd.iter().map(|&s| s / cl / HS as f32).collect(),
		vec![cp / cl; d],
	];
	let mut m = vec![0u8; MPARTS * ms];
	for (i, v) in parts.iter().enumerate() {
		m[i * ms..i * ms + d * 2].copy_from_slice(&f16v(v));
	}
	put("mod", &m)?;
	let how = |t: Ty| if t == Ty::I4 { "int4" } else { "int8" };
	say(&format!(
		"H3 text encoder layer {l}: attention {}, MLP {}{} in {:.1} s",
		how(codes[0]),
		how(codes[2]),
		if by_gptq { " GPTQ" } else { "" },
		t.elapsed().as_secs_f64()
	));
	Ok(())
}

/// Write the scale file once every layer's record is there.
pub fn write_scales(out: &Path, c: &DShape) -> Result<(), String> {
	let mut v = vec![scale_in(out)?];
	for l in 0..c.layers {
		v.push(scale_of(out, l)?);
	}
	let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
	crate::weights::write(&scale_file(&wdir(out)), &b)
}

/// The text encoder's graphs (`OUT/graphs/h3te/*.xml`), for the code types
/// of the weights.
pub fn step_graphs(out: &Path, c: &DShape) -> Result<(), String> {
	let gdir = out.join("graphs");
	let ty = codes(out)?;
	for (i, stem) in TE_STEMS.iter().enumerate() {
		let rows = if i < 3 { h3img::S } else { h3img::S_SMALL };
		let g: Graph = h3img::dit_graph_ty(c, i % 3 != 0, i % 3 != 2, rows, Some(ty));
		crate::sets::save(&gdir, stem, &g)?;
	}
	say(&format!(
		"H3 text encoder graphs ({}) in {}",
		ty.map(ty_name).join(" "),
		gdir.join("h3te").display()
	));
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The graph computes `fq(x R) . (codes 2^e)^T . s`; for an input
	/// exactly on the activation grid that is `x W^T` up to the weights'
	/// rounding.
	#[test]
	fn quantize_matches_the_graph_form() {
		let (n, k) = (8, 512);
		let mut seed = 1u64;
		let mut rnd = || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
		};
		let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
		let x: Vec<f32> = (0..k).map(|_| rnd() * 3.0).collect();
		let (accel, _) = Accel::new(false);
		for ty in [Ty::I8, Ty::I4] {
			let r = 4.0f32;
			let (codes, s) = quantize(&w, n, k, r, None, ty, &accel).unwrap();
			let s = f16f(&s);
			let codes: Vec<i8> = if ty == Ty::I4 {
				gptq::unpack4(&codes)
			} else {
				codes.iter().map(|&c| c as i8).collect()
			};
			let e = if ty == Ty::I4 { 1.0 / 8.0 } else { 1.0 / 128.0 };
			// the graph: x * pre (signs x 127 / 128 / r), rotated, on the grid
			let sg = gptq::signs(k);
			let mut t: Vec<f32> = x
				.iter()
				.zip(&sg)
				.map(|(v, s)| v * s * 127.0 / 128.0 / r)
				.collect();
			gptq::rotate_rows_b(&mut t, k, &vec![1.0; k], ROT);
			let tq: Vec<f32> = t
				.iter()
				.map(|v| (v * 128.0).round().clamp(-128.0, 127.0) / 128.0)
				.collect();
			let (mut err, mut norm) = (0f64, 0f64);
			for j in 0..n {
				let y: f32 = (0..k)
					.map(|i| tq[i] * codes[j * k + i] as f32 * e)
					.sum::<f32>() * s[j];
				let want: f32 = (0..k).map(|i| x[i] * w[j * k + i]).sum();
				err += ((y - want) as f64).powi(2);
				norm += (want as f64).powi(2);
			}
			let rel = (err / norm).sqrt();
			assert!(
				rel < if ty == Ty::I4 { 0.15 } else { 0.02 },
				"{ty:?}: {rel}"
			);
		}
	}
}
