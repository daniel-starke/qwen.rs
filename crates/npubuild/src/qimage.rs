//! Qwen-Image's transformer on the NPU (`cli QWEN_IMAGE_DIR --npu-build
//! OUT`, `docs/image.md` "NPU backend").
//!
//! The transformer's 32 single-stream blocks are cut at each attention:
//! the attention runs over every row of the sequence (prefix and target),
//! whose length is the image's, so it is not an NPU graph; everything per
//! row is, over [`S`] rows at a time. Three graphs, one blob each for all
//! blocks (every per-block value is an input bound to its file,
//! `w/dit{L}.<name>.in`):
//!
//! ```text
//!   dit_pre   x                 -> q, k, v       block 0 up to its attention
//!   dit_mid   x, a              -> xo, q, k, v   block l - 1 from its attention on, block l up to its
//!   dit_post  x, a              -> xo            block 31 from its attention on
//! ```
//!
//! plus the step's modulation (`ms1` = 1 + scale1, `mg1` = gate1, `ms2`,
//! `mg2`; [1, 1, 4096], the same for every block: one modulation for all)
//! and `cos` / `sin` [1, S, 128] (RoPE per row). `q` / `k` are normed and
//! rotated, `[1, S, 32 x 128]` like `v`; `a` is the attention's output
//! before `to_out`.
//!
//! **RoPE layout**: the checkpoint rotates adjacent pairs `(2i, 2i + 1)`.
//! Each head's `q` / `k` rows (and the QK norm weights) are stored
//! permuted to `[0, 2, .., 126, 1, 3, .., 127]`, so the graph rotates the
//! halves (`x1 cos - x2 sin`, `x1 sin + x2 cos`) with `cos` / `sin` of
//! pair `i` at lanes `i` and `64 + i`; a dot product of two permuted rows
//! is that of the originals, so the attention is unchanged.
//!
//! **Weights and activations int8** (W8A8: 20 ms per job against 44 with
//! f16 activations): each matrix is stored as `W R` (`R`: random signs and
//! normalized Hadamard blocks of 256 along the input), int8 per row
//! ([`q8_rows`]), and its input is rotated the same way in the graph and
//! quantized ([`Act`]): at a calibrated range for QKV, out and gate / up
//! (`cal/dit.cal`), per row for down. The rotation spreads the residual
//! stream's few outlier channels (hundreds of times the rest) over a
//! block, as on the CPU backend. QKV are one matrix (`qkv`, rows q, k, v),
//! as are gate and up (`gu`).

use crate::g::{Graph, Ty, V};
use crate::st::Checkpoint;
use crate::{mkdir, say};
use std::path::{Path, PathBuf};

/// Rows per NPU job.
pub const S: usize = 256;
/// Width, heads, head dim, MLP width of Qwen-Image-2.1's transformer.
pub const D: usize = 4096;
pub const NH: usize = 32;
pub const HD: usize = 128;
pub const FF: usize = 12288;
pub const EPS: f64 = 1e-6;

/// True when `dir` is a Qwen-Image diffusers checkpoint
/// (`model_index.json` and `transformer/`).
pub fn is_qwen_image(dir: &Path) -> bool {
	dir.join("model_index.json").is_file() && dir.join("transformer").is_dir()
}

/// The graph kinds: `(name, graph)`.
pub fn graph_kinds() -> Vec<(&'static str, Graph)> {
	vec![
		("dit_pre", graph(false, true, S)),
		("dit_mid", graph(true, true, S)),
		("dit_post", graph(true, false, S)),
	]
}

/// The outputs of a kind, in result order (the driver names an output
/// after the op that computes it: they go by order).
pub fn outputs(kind: &str) -> &'static [&'static str] {
	if kind == "dit_post" {
		&["xo"]
	} else if kind == "dit_pre" {
		&["q", "k", "v"]
	} else {
		&["xo", "q", "k", "v"]
	}
}

/// Whether input `name` of a graph is a weight of the block after the cut
/// (the pre part: `qkv`, the QK norms) rather than of the one before.
pub fn is_pre_weight(name: &str) -> bool {
	name.starts_with("qkv.") || name == "nq" || name == "nk"
}

/// Inputs that are not weights (the job's rows, the step's modulation).
pub fn is_runtime(name: &str) -> bool {
	matches!(
		name,
		"x" | "a" | "cos" | "sin" | "ms1" | "mg1" | "ms2" | "mg2"
	)
}

fn p16(g: &mut Graph, name: &str, shape: &[usize]) -> V {
	g.parameter(name, Ty::F16, shape)
}

/// How a linear's activation is quantized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
	/// int8 at a calibrated range: `fq(rotate(x * {w}.pre))` (`pre` = the
	/// rotation's signs x `127 / 128 / r`), the range folded into `{w}.s`.
	Static,
	/// int8 per row: `x` rotated, each row scaled by `127 / 128 / max
	/// |row|`, the output by `max |row| 128 / 127` (no range to calibrate; ~2 + 7 ms of
	/// eltwise per job for the four matrices of a block).
	Dynamic,
}

/// The activation quantization of each matrix (`qkv`, `o`, `gu`: inputs
/// of a LayerNorm or of the attention, bounded; `d`: the MLP's hidden
/// state, whose outliers vary).
pub fn act(m: &str) -> Act {
	if m == "d" {
		Act::Dynamic
	} else {
		Act::Static
	}
}

/// `x W^T`, int8 x int8 ([`Act`]), against the int8 codes `{w}.w` [n, k]
/// of `W R` (x 2^-7; `R` the rotation: signs, then normalized Hadamard
/// blocks of [`ROT`]) with row scales `{w}.s` [n].
///
/// The activation's grid is `k / 128` (|t| < 1), not the integers: the
/// MatMul's f16 output before the row scale is then at most the row's
/// code sum / 128. On the integer grid the residual's outlier channel
/// (1407) overflowed f16 to inf in block 0 at the last step of a 512x512
/// image (17 rows of NaN velocity, an empty image).
fn qlin(g: &mut Graph, x: &V, w: &str, n: usize, k: usize) -> V {
	qlin_f(g, x, w, n, k, true, true)
}

/// [`qlin`] without the row scale `{w}.s` (`scale` false: applied or
/// folded elsewhere) and, for [`Act::Dynamic`], without the rotation's
/// signs (`signs` false: folded into the rows that produce `x`).
pub(crate) fn qlin_f(
	g: &mut Graph,
	x: &V,
	w: &str,
	n: usize,
	k: usize,
	scale: bool,
	signs: bool,
) -> V {
	qlin_ty(
		g,
		x,
		w,
		n,
		k,
		scale,
		signs,
		if w4() { Ty::I4 } else { Ty::I8 },
		false,
	)
}

/// [`qlin_f`] with the codes' type `ty` (`I8`: x 2^-7; `I4`: x 2^-3, and
/// a per-row activation without the [`rowq`] kernel). A per-row
/// activation's MatMul output takes the row maximum before the row scale
/// when `max_first` (row scales far above 1 overflow f16 in between).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qlin_ty(
	g: &mut Graph,
	x: &V,
	w: &str,
	n: usize,
	k: usize,
	scale: bool,
	signs: bool,
	ty: Ty,
	max_first: bool,
) -> V {
	let e = if ty == Ty::I4 { -3 } else { -7 };
	let rowq = rowq() && ty == Ty::I8;
	// [`rowq`]: the codes carry zero columns for the kernel's pad lanes
	let kc = if act(w) == Act::Dynamic && rowq {
		k + ROWQ_PAD
	} else {
		k
	};
	let c = g.parameter(&format!("{w}.w"), ty, &[n, kc]);
	let c = g.convert(&c, Ty::F16);
	let sc = g.s16(2f64.powi(e));
	let c = g.mul(&c, &sc);
	let rs = |g: &mut Graph, y: V| -> V {
		if !scale {
			return y;
		}
		let s = p16(g, &format!("{w}.s"), &[n]);
		g.mul(&y, &s)
	};
	if act(w) == Act::Static {
		let pre = p16(g, &format!("{w}.pre"), &[k]);
		let t = g.mul(x, &pre);
		let t = crate::flash::graphs::rotate(g, &t, ROT);
		let tq = g.fake_quantize(&t, -1.0, 127.0 / 128.0, 256);
		let y = g.matmul(&tq, &c, false, true);
		return rs(g, y);
	}
	let t = if signs {
		let sg: Vec<f64> = crate::gptq::signs(k).iter().map(|&v| v as f64).collect();
		let sgc = g.c16(&[k], &sg);
		g.mul(x, &sgc)
	} else {
		x.clone()
	};
	let t = crate::flash::graphs::rotate(g, &t, ROT);
	if rowq {
		// `shave/rowq.sk` (carrier SoftMax): the scaled row and its max
		// in ROWQ_PAD pad lanes; the MatMul reads the pad lanes against
		// zero columns, so the quantized tensor is never sliced
		let s = t.shape[t.shape.len() - 2];
		let z = g.c16(&[1, s, ROWQ_PAD], &vec![0.0; s * ROWQ_PAD]);
		let tp = g.concat(&[t, z], 2);
		let y = g.softmax(&tp, 2);
		let m = g.slice(&y, k, k + 1, 1, 2);
		let yq = g.fake_quantize(&y, -1.0, 127.0 / 128.0, 256);
		let y = g.matmul(&yq, &c, false, true);
		let iq = g.s16(128.0 / 127.0);
		let m = g.mul(&m, &iq);
		return if max_first {
			let y = g.mul(&y, &m);
			rs(g, y)
		} else {
			let y = rs(g, y);
			g.mul(&y, &m)
		};
	}
	let a = g.abs(&t);
	let m = g.reduce_max(&a, -1);
	let lo = g.s16(1e-3);
	let m = g.maximum(&m, &lo);
	let q = g.s16(127.0 / 128.0);
	let r = g.div(&q, &m);
	let t = g.mul(&t, &r);
	let tq = g.fake_quantize(&t, -1.0, 127.0 / 128.0, 256);
	let y = g.matmul(&tq, &c, false, true);
	let iq = g.s16(128.0 / 127.0);
	let m = g.mul(&m, &iq);
	if max_first {
		let y = g.mul(&y, &m);
		rs(g, y)
	} else {
		let y = rs(g, y);
		g.mul(&y, &m)
	}
}

/// The folded form (default; `QWEN_IMAGE_FOLD=0` at build time is the A/B,
/// into its own output directory): two exact rewrites that take ~4 ms of
/// eltwise work off each `dit_mid` job (25.3 -> 21.0 ms).
///
/// - **`gu`**: only the gate half is scaled in the graph (`gu.sg`, [FF]);
///   the up rows carry the down input's rotation signs, and the up half's
///   row scales are folded into the columns of `d`, so `hid = swish(g) *
///   u` needs neither the signs Multiply nor the up half's scale.
/// - **`q` / `k` / `v` raw**: the MatMul's output before its row scale;
///   the cores apply `q` / `k`'s row scales (`qkv.s`), head norms (`nq`,
///   `nk`) and RoPE in f32 (`imagegen::qimage::npudit`), and `v`'s row scales are
///   folded into `o.pre` (a per-channel factor of V passes through the
///   attention unchanged). No `cos` / `sin` inputs.
///
/// The engine tells the forms apart by the graphs' inputs (`gu.sg`).
pub fn fold() -> bool {
	std::env::var("QWEN_IMAGE_FOLD").map_or(true, |v| v != "0")
}

/// The down projection's per-row activation scale ([`Act::Dynamic`]) in our
/// SHAVE kernel `shave/rowq.sk` (carrier SoftMax over the row, built into
/// [`ROWQ_KERN`] with `-DPAD`): one pass scales each row to a max |y| of
/// 127 / 128 and writes the row's max into [`ROWQ_PAD`] lanes appended to
/// it; the down MatMul reads those lanes against zero weight columns
/// (`d.w` is [D, FF + ROWQ_PAD]) and the max for the output's rescale is
/// sliced from the kernel's f16 output. The stock chain (Abs, ReduceMax,
/// Maximum, Divide, Multiply) sat between the gate / up and the down
/// MatMuls: `dit_mid` 21.0 -> 19.45 ms. Default on; `QWEN_IMAGE_ROWQ=0` at
/// build time is the stock form (into its own output directory: the
/// weights differ). Not with [`w4`].
///
/// Slicing the data lanes off the quantized tensor instead does not compile
/// (the compiler propagates the FakeQuantize through the split and rejects
/// its own i8 / u8 types), and an elementwise carrier (Mish) does not keep
/// rows whole: the compiler tiled it into 64-row x 3080-column pieces.
pub fn rowq() -> bool {
	!w4() && std::env::var("QWEN_IMAGE_ROWQ").map_or(true, |v| v != "0")
}

/// The pad lanes of [`rowq`] per row.
pub const ROWQ_PAD: usize = 32;

/// Where [`step_compile`] builds the graphs' SHAVE kernels.
pub const ROWQ_KERN: &str = "kern/dit";

/// `QWEN_IMAGE_W4=1` at build time: the weights int4 per row, round to
/// nearest (codes x 2^-3 in the graph), an A/B against int8: the same
/// speed (`dit_mid` 25.0 vs 25.3 ms: the graphs are compute-bound), 3.5 GB
/// less RAM, see `docs/image.md` "NPU backend" for its error. Build
/// it into its own output directory.
pub fn w4() -> bool {
	std::env::var("QWEN_IMAGE_W4").is_ok_and(|v| v == "1")
}

/// The rotation block of the int8 activations.
pub const ROT: usize = 256;

/// LayerNorm without affine, `* m` (`m` = 1 + scale).
fn ln_mod(g: &mut Graph, x: &V, m: &V) -> V {
	let y = g.mvn(x, EPS);
	g.mul(&y, m)
}

fn rms(g: &mut Graph, x: &V) -> V {
	let two = g.sf(x.ty, 2.0);
	let sq = g.power(x, &two);
	let ms = g.reduce_mean(&sq, -1);
	let e = g.sf(x.ty, EPS);
	let t = g.add(&ms, &e);
	let t = g.sqrt(&t);
	let one = g.sf(x.ty, 1.0);
	let r = g.div(&one, &t);
	g.mul(x, &r)
}

/// Head norm (weight `w`, permuted) and RoPE (halves) of `t` [1, s, NH *
/// HD] with `cos` / `sin` [1, s, 1, HD].
fn norm_rope(g: &mut Graph, t: &V, w: &str, s: usize, cos: &V, sin: &V) -> V {
	let t = g.reshape(t, &[1, s as i64, NH as i64, HD as i64]);
	let t = rms(g, &t);
	let wv = p16(g, w, &[HD]);
	let t = g.mul(&t, &wv);
	let h = g.variadic_split(&t, 3, &[HD / 2, HD / 2]);
	let neg = g.negative(&h[1]);
	let rot = g.concat(&[neg, h[0].clone()], 3);
	let a = g.mul(&t, cos);
	let b = g.mul(&rot, sin);
	let y = g.add(&a, &b);
	g.reshape(&y, &[1, s as i64, D as i64])
}

/// See the module docs: `post` = the block before the cut from its
/// attention on, `pre` = the block after it up to its attention; in the
/// [`fold`] form unless that is off.
pub fn graph(post: bool, pre: bool, s: usize) -> Graph {
	graph_form(post, pre, s, fold())
}

/// [`graph`], folded ([`fold`]) or not.
pub fn graph_form(post: bool, pre: bool, s: usize, fold: bool) -> Graph {
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, s, D]);
	let x = if post {
		let a = p16(&mut g, "a", &[1, s, D]);
		let mg1 = p16(&mut g, "mg1", &[1, 1, D]);
		let ms2 = p16(&mut g, "ms2", &[1, 1, D]);
		let mg2 = p16(&mut g, "mg2", &[1, 1, D]);
		let o = qlin(&mut g, &a, "o", D, D);
		let o = g.mul(&o, &mg1);
		let x1 = g.add(&x, &o);
		let h = ln_mod(&mut g, &x1, &ms2);
		let gu = qlin_f(&mut g, &h, "gu", 2 * FF, D, !fold, true);
		let gu = g.variadic_split(&gu, 2, &[FF, FF]);
		let gate = if fold {
			let sg = p16(&mut g, "gu.sg", &[FF]);
			g.mul(&gu[0], &sg)
		} else {
			gu[0].clone()
		};
		let sw = g.swish1(&gate);
		let hid = g.mul(&sw, &gu[1]);
		let dn = qlin_f(&mut g, &hid, "d", D, FF, true, !fold);
		let dn = g.mul(&dn, &mg2);
		g.add(&x1, &dn)
	} else {
		x
	};
	if post {
		g.result(&x, "xo");
	}
	if pre {
		let ms1 = p16(&mut g, "ms1", &[1, 1, D]);
		if fold {
			let h = ln_mod(&mut g, &x, &ms1);
			let qkv = qlin_f(&mut g, &h, "qkv", 3 * D, D, false, true);
			let p = g.variadic_split(&qkv, 2, &[D, D, D]);
			g.result(&p[0], "q");
			g.result(&p[1], "k");
			g.result(&p[2], "v");
			return g;
		}
		let cos = p16(&mut g, "cos", &[1, s, HD]);
		let sin = p16(&mut g, "sin", &[1, s, HD]);
		let cs = g.unsqueeze(&cos, 2);
		let sn = g.unsqueeze(&sin, 2);
		let h = ln_mod(&mut g, &x, &ms1);
		let qkv = qlin(&mut g, &h, "qkv", 3 * D, D);
		let p = g.variadic_split(&qkv, 2, &[D, D, D]);
		let q = norm_rope(&mut g, &p[0], "nq", s, &cs, &sn);
		let k = norm_rope(&mut g, &p[1], "nk", s, &cs, &sn);
		g.result(&q, "q");
		g.result(&k, "k");
		g.result(&p[2], "v");
	}
	g
}

/// The permutation of a head's dims: evens, then odds.
pub fn head_perm() -> Vec<usize> {
	(0..HD / 2)
		.map(|i| 2 * i)
		.chain((0..HD / 2).map(|i| 2 * i + 1))
		.collect()
}

/// int8 per-row codes of `W R` and the f16 row scales of the graph's
/// convention: `max |row| / 127 x 128` (the codes' 2^-7), times `128 r /
/// 127` for an input of calibrated range `r` ([`Act::Static`]: the
/// activation grid's `1 / 128`), times 1 for [`Act::Dynamic`] (the graph
/// applies `max |row| 128 / 127`).
pub fn q8_rows(w: &[f32], n: usize, k: usize, r: f32) -> (Vec<u8>, Vec<u8>) {
	let mut w = w.to_vec();
	crate::gptq::rotate_rows_b(&mut w, k, &crate::gptq::signs(k), ROT);
	let w = &w[..];
	let mut codes = vec![0u8; n * k];
	let mut sc = vec![0f32; n];
	let (pc, ps) = (P(codes.as_mut_ptr()), P(sc.as_mut_ptr()));
	let f = if r > 0.0 { 128.0 * r / 127.0 } else { 1.0 };
	// int8: codes in +-127, x 2^-7 in the graph; int4: +-7, x 2^-3
	let (qm, ge) = if w4() {
		(7.0f32, 8.0f32)
	} else {
		(127.0, 128.0)
	};
	quant::par_for(n, move |row| {
		let (pc, ps) = (pc, ps);
		let x = &w[row * k..(row + 1) * k];
		let m = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
		let inv = if m > 0.0 { qm / m } else { 0.0 };
		// SAFETY: task `row` owns row `row` of both outputs
		let c = unsafe { std::slice::from_raw_parts_mut(pc.0.add(row * k), k) };
		for (o, &v) in c.iter_mut().zip(x) {
			*o = ((v * inv).round().clamp(-qm, qm) as i8) as u8;
		}
		unsafe { *ps.0.add(row) = m / qm * ge * f };
	});
	if w4() {
		let q: Vec<i8> = codes.iter().map(|&c| c as i8).collect();
		return (crate::gptq::pack4(&q), f16v(&sc));
	}
	(codes, f16v(&sc))
}

/// `{w}.pre` of an input of width `k` and range `r`: signs x `127 / 128 /
/// r`.
pub fn pre(k: usize, r: f32) -> Vec<u8> {
	let v: Vec<f32> = crate::gptq::signs(k)
		.iter()
		.map(|&s| s * 127.0 / 128.0 / r)
		.collect();
	f16v(&v)
}

/// The calibrated input ranges per block, `cal/dit.cal`: a line `L r_qkv
/// r_o r_gu` per block (the largest magnitude of the rotated input,
/// headroom included).
pub fn cal_path(out: &Path) -> PathBuf {
	out.join("cal").join("dit.cal")
}

/// Read [`cal_path`]: `[qkv, o, gu]` per block.
pub fn read_cal(out: &Path, layers: usize) -> Result<Vec<[f32; 3]>, String> {
	let p = cal_path(out);
	let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	let mut v = vec![[0f32; 3]; layers];
	let mut seen = vec![false; layers];
	for line in t
		.lines()
		.map(str::trim)
		.filter(|l| !l.is_empty() && !l.starts_with('#'))
	{
		let f: Vec<&str> = line.split_whitespace().collect();
		let bad = || format!("{}: bad line {line}", p.display());
		if f.len() != 4 {
			return Err(bad());
		}
		let l: usize = f[0].parse().map_err(|_| bad())?;
		if l >= layers {
			return Err(bad());
		}
		for i in 0..3 {
			v[l][i] = f[i + 1].parse().map_err(|_| bad())?;
			if !(v[l][i] > 0.0) {
				return Err(bad());
			}
		}
		seen[l] = true;
	}
	if let Some(l) = seen.iter().position(|s| !s) {
		return Err(format!("{}: no line for block {l}", p.display()));
	}
	Ok(v)
}

#[derive(Clone, Copy)]
struct P<T>(*mut T);
// SAFETY: every user writes disjoint rows.
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}

pub(crate) fn f16v(v: &[f32]) -> Vec<u8> {
	v.iter()
		.flat_map(|&x| crate::g::f16_bits(x as f64).to_le_bytes())
		.collect()
}

/// f16 bytes -> f32.
pub fn f16f(b: &[u8]) -> Vec<f32> {
	b.chunks_exact(2)
		.map(|c| crate::g::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect()
}

/// The f16 product of an f16 vector (bytes) and `b`.
pub(crate) fn mulf16(a: &[u8], b: &[f32]) -> Vec<u8> {
	let v: Vec<f32> = f16f(a).iter().zip(b).map(|(x, y)| x * y).collect();
	f16v(&v)
}

/// The block's files under `w/` (`dit{l}.<name>.in`).
pub fn wfile(wdir: &Path, l: usize, name: &str) -> PathBuf {
	wdir.join(format!("dit{l}.{name}.in"))
}

/// `OUT/w/dit{L}.*.in` of every block (skipped per block when its last
/// file exists).
fn step_weights(src: &Path, out: &Path) -> Result<(), String> {
	let ck = Checkpoint::open_dir(&src.join("transformer"))?;
	let layers = num_layers(src)?;
	let cal = read_cal(out, layers)?;
	let wdir = out.join("w");
	mkdir(&wdir)?;
	let perm = head_perm();
	let fold = fold();
	if fold && w4() {
		return Err("QWEN_IMAGE_W4=1 needs QWEN_IMAGE_FOLD=0 (the fold negates int8 codes)".into());
	}
	for l in 0..layers {
		if wfile(&wdir, l, "d.s").exists() {
			continue;
		}
		let t = std::time::Instant::now();
		let p = |n: &str| format!("transformer_blocks.{l}.{n}");
		let put = |name: &str, b: &[u8]| crate::weights::write(&wfile(&wdir, l, name), b);
		// q / k rows permuted per head
		let permute = |w: Vec<f32>| -> Vec<f32> {
			let mut o = vec![0f32; w.len()];
			for h in 0..NH {
				for (j, &src) in perm.iter().enumerate() {
					o[(h * HD + j) * D..(h * HD + j + 1) * D]
						.copy_from_slice(&w[(h * HD + src) * D..(h * HD + src + 1) * D]);
				}
			}
			o
		};
		let mut qkv = permute(ck.mat(&p("attn.to_q.weight"), D, D)?);
		qkv.extend(permute(ck.mat(&p("attn.to_k.weight"), D, D)?));
		qkv.extend(ck.mat(&p("attn.to_v.weight"), D, D)?);
		let [r_qkv, r_o, r_gu] = cal[l];
		let (c, s) = q8_rows(&qkv, 3 * D, D, r_qkv);
		drop(qkv);
		put("qkv.w", &c)?;
		put("qkv.s", &s)?;
		// v's row scales as the graph would apply them, for `o.pre`
		let sv = f16f(&s[2 * D * 2..]);
		put("qkv.pre", &pre(D, r_qkv))?;
		for (n, name) in [("nq", "attn.norm_q.weight"), ("nk", "attn.norm_k.weight")] {
			let v = ck.vec(&p(name), HD)?;
			let v: Vec<f32> = perm.iter().map(|&i| v[i]).collect();
			put(n, &f16v(&v))?;
		}
		let (c, s) = q8_rows(&ck.mat(&p("attn.to_out.0.weight"), D, D)?, D, D, r_o);
		put("o.w", &c)?;
		put("o.s", &s)?;
		let op = pre(D, r_o);
		put("o.pre", &if fold { mulf16(&op, &sv) } else { op })?;
		let mut gu = ck.mat(&p("img_mlp.gate_layer.weight"), FF, D)?;
		gu.extend(ck.mat(&p("img_mlp.proj.weight"), FF, D)?);
		let (mut c, s) = q8_rows(&gu, 2 * FF, D, r_gu);
		drop(gu);
		let mut dw = ck.mat(&p("img_mlp.out.weight"), D, FF)?;
		if fold {
			// the up rows x the down input's rotation signs, the up half's
			// row scales into the columns of `d`
			for (j, &v) in crate::gptq::signs(FF).iter().enumerate() {
				if v < 0.0 {
					c[(FF + j) * D..(FF + j + 1) * D]
						.iter_mut()
						.for_each(|b| *b = (*b as i8).wrapping_neg() as u8);
				}
			}
			let su = f16f(&s[FF * 2..]);
			for row in dw.chunks_exact_mut(FF) {
				row.iter_mut().zip(&su).for_each(|(w, &f)| *w *= f);
			}
			put("gu.sg", &s[..FF * 2])?;
		} else {
			put("gu.s", &s)?;
		}
		put("gu.w", &c)?;
		put("gu.pre", &pre(D, r_gu))?;
		let (c, s) = q8_rows(&dw, D, FF, 0.0);
		drop(dw);
		let c = if rowq() {
			// zero columns for the kernel's pad lanes
			let mut p = vec![0u8; D * (FF + ROWQ_PAD)];
			for (o, i) in p.chunks_exact_mut(FF + ROWQ_PAD).zip(c.chunks_exact(FF)) {
				o[..FF].copy_from_slice(i);
			}
			p
		} else {
			c
		};
		put("d.w", &c)?;
		put("d.s", &s)?;
		say(&format!(
			"transformer block {l}: int8 in {:.1} s",
			t.elapsed().as_secs_f64()
		));
	}
	Ok(())
}

/// The transformer's block count (`transformer/config.json`).
pub fn num_layers(src: &Path) -> Result<usize, String> {
	let p = src.join("transformer").join("config.json");
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	crate::st::parse_json(&b)?
		.get("num_layers")
		.and_then(|v| v.num())
		.map(|v| v as usize)
		.ok_or(format!("{}: no num_layers", p.display()))
}

/// `OUT/graphs/dit/<kind>.xml` + `.args` / `.uses`.
fn step_graphs(out: &Path) -> Result<(), String> {
	let gdir = out.join("graphs");
	for (n, g) in graph_kinds() {
		crate::sets::save(&gdir, &format!("dit/{n}"), &g)?;
	}
	say(&format!(
		"transformer graphs in {}",
		gdir.join("dit").display()
	));
	Ok(())
}

/// Compile the graphs whose blob is missing (our compiler, the prefill
/// options, no weight swizzling: the engine binds the plain files).
fn step_compile(o: &crate::Options) -> Result<(), String> {
	let t = crate::sets::Tools::find(o.compiler.as_deref(), o.jobs)?;
	let gdir = o.out.join("graphs");
	if rowq() {
		crate::sets::write_kernel(
			&gdir.join(ROWQ_KERN),
			include_str!("../../npu/shave/rowq.sk"),
			"softmax",
			&[("PAD".to_string(), ROWQ_PAD as i64)],
		)?;
	}
	for (n, _) in graph_kinds() {
		let stem = format!("dit/{n}");
		if gdir.join(format!("{stem}.blob")).exists() {
			continue;
		}
		say(&format!("compiling {stem}"));
		let kern = if rowq() { ROWQ_KERN } else { "" };
		crate::sets::compile_graph_kern(&t, &gdir, &stem, Some(kern), false)?;
	}
	crate::qvae::step_compile(&t, &o.src, &o.out)
}

/// `OUT/image.npu`.
fn step_sets(o: &crate::Options) -> Result<(), String> {
	// (absolute, symlinks kept: the checkpoint's canonical path is a link)
	let src = std::path::absolute(&o.src).unwrap_or_else(|_| o.src.clone());
	let src = src.to_string_lossy();
	let t = format!(
		"# Qwen-Image: the transformer on the NPU (int8 per row), its attention on the cores;
# the VAE decoder's 3x3 convolutions on the NPU (f16)
arch       = qwen-image
checkpoint = {src}
graphs     = graphs/dit
weights    = w
vae        = graphs/vae
"
	);
	let t = if o.out.join("sets/vis/vis.cfg").is_file() {
		t + "# edits: the vision tower's per-patch work on the NPU, its attention on the cores\nvision     = sets/vis\n"
	} else {
		t
	};
	let p = o.out.join("image.npu");
	std::fs::write(&p, t).map_err(|e| format!("{}: {e}", p.display()))?;
	say(&format!("done: {}", p.display()));
	Ok(())
}

/// Writes [`cal_path`] (the CPU reference over calibration prompts; it
/// lives in `imagegen`, which depends on this crate).
pub type Calibrate<'a> = &'a dyn Fn(&Path, &Path, usize) -> Result<Vec<[f32; 3]>, String>;

/// The Qwen-Image build: `cal` (the activation ranges: `calibrate(src,
/// out, layers)`), `weights`, `graphs`, `compile`, `vis` (the text
/// encoder's vision tower: [`crate::sets::vis_set`]), `sets`.
pub fn build(o: &crate::Options, calibrate: Option<Calibrate>) -> Result<(), String> {
	mkdir(&o.out)?;
	let layers = num_layers(&o.src)?;
	if o.wants("cal") && read_cal(&o.out, layers).is_err() {
		let f = calibrate.ok_or(
			"the activation ranges (cal/dit.cal) are made by `cli QWEN_IMAGE_DIR --npu-build OUT`",
		)?;
		let r = f(&o.src, &o.out, layers)?;
		mkdir(&o.out.join("cal"))?;
		let mut t = String::from(
			"# block, input ranges of qkv, o, gu (rotated, headroom included)
",
		);
		for (l, v) in r.iter().enumerate() {
			t += &format!(
				"{l} {} {} {}
",
				v[0], v[1], v[2]
			);
		}
		std::fs::write(cal_path(&o.out), t).map_err(|e| e.to_string())?;
	}
	if o.wants("weights") {
		step_weights(&o.src, &o.out)?;
	}
	if o.wants("graphs") {
		step_graphs(&o.out)?;
		crate::qvae::step_graphs(&o.src, &o.out)?;
	}
	if o.wants("compile") {
		step_compile(o)?;
	}
	if o.wants("vis") {
		// the text encoder's vision tower (edits' condition images), with
		// its deepstack mergers: the 27B's tower graphs (driver-compiled)
		let ck = Checkpoint::open_dir(&o.src.join("text_encoder"))?;
		crate::sets::vis_set(&ck, &o.out, &mut |m| say(m))?;
	}
	if o.wants("sets") {
		step_sets(o)?;
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The codes and row scales give back `W R` as the graph reads them
	/// (`c 2^-7 s`, the dynamic form: `s` = `max |row| 128 / 127`).
	#[test]
	fn int8_rows_reconstruct_the_rotated_weight() {
		let (n, k) = (8, 512);
		let w: Vec<f32> = (0..n * k)
			.map(|i| {
				((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * (1.0 + (i % k == 3) as u8 as f32 * 40.0)
			})
			.collect();
		let (c, s) = q8_rows(&w, n, k, 0.0);
		let mut wr = w.clone();
		crate::gptq::rotate_rows_b(&mut wr, k, &crate::gptq::signs(k), ROT);
		for r in 0..n {
			let sr = crate::g::f16_to_f32(u16::from_le_bytes([s[2 * r], s[2 * r + 1]]));
			let (mut e, mut nn) = (0f64, 0f64);
			for j in 0..k {
				let got = (c[r * k + j] as i8) as f32 / 128.0 * sr;
				e += (got - wr[r * k + j]).powi(2) as f64;
				nn += wr[r * k + j].powi(2) as f64;
			}
			assert!((e / nn).sqrt() < 1e-2, "row {r}: {}", (e / nn).sqrt());
		}
		// a static input's range scales the row scale by 128 r / 127
		let (_, s2) = q8_rows(&w, n, k, 3.0);
		let a = crate::g::f16_to_f32(u16::from_le_bytes([s[0], s[1]]));
		let b = crate::g::f16_to_f32(u16::from_le_bytes([s2[0], s2[1]]));
		assert!((b / a - 3.0 * 128.0 / 127.0).abs() < 2e-3);
	}

	/// A dot product of two rows permuted per head is the original one.
	#[test]
	fn head_permutation_keeps_dot_products() {
		let p = head_perm();
		let mut seen = [false; HD];
		p.iter().for_each(|&i| seen[i] = true);
		assert!(seen.iter().all(|&s| s));
		assert_eq!((p[0], p[1], p[HD / 2]), (0, 2, 1));
	}

	#[test]
	fn calibration_file_parses() {
		let d = std::env::temp_dir().join(format!("qwen-rs-ditcal-{}", std::process::id()));
		std::fs::create_dir_all(d.join("cal")).unwrap();
		std::fs::write(cal_path(&d), "# x\n0 1 2 3\n1 4.5 5 6\n").unwrap();
		assert_eq!(
			read_cal(&d, 2).unwrap(),
			vec![[1.0, 2.0, 3.0], [4.5, 5.0, 6.0]]
		);
		assert!(read_cal(&d, 3).is_err(), "a block without a line");
		std::fs::write(cal_path(&d), "0 1 2\n").unwrap();
		assert!(read_cal(&d, 1).is_err());
		let _ = std::fs::remove_dir_all(&d);
	}
}
