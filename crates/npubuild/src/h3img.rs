//! MiniMax-H3-Image on the NPU (`cli H3_DIR --npu-build OUT`,
//! `docs/h3image.md` "NPU backend"): the transformer's 50 blocks and the
//! VAE decoder's 24 ViT layers.
//!
//! **Transformer.** As Qwen-Image's (`crate::qimage`): the blocks are cut at
//! each attention (which runs on the cores over every row), W8A8 with the
//! Hadamard rotation, q / k / v raw out of the graph, three graphs over
//! [`S`] rows (`h3/pre`, `h3/mid`, `h3/post`) and the same three over
//! [`S_SMALL`] rows (`h3/pre64`, ...: the first and last jobs of a block,
//! where the NPU or the cores wait for the other), every per-block value an
//! input. What differs:
//!
//! - **The residual is stored scaled.** H3's residual stream reaches ~1.3e7
//!   (a few channels of a few rows), far past f16. Each block's output is
//!   kept as `x_l / c_l` with a calibrated `c_l` (`cal/h3dit.cal`: the
//!   block's largest magnitude over the calibration images / [`XS`]), so
//!   it stays below ~8192 with 8x headroom. The RMSNorms are scale
//!   invariant; the gates and `1 / c_l` are folded into the out and down
//!   projections' row scales, and the step from `c_{l-1}` to `c_l` is one
//!   multiply (`xr`).
//! - **The modulation differs per block and per row** (`(timestep,
//!   modality)` tables, RMSNorm weight folded into the scale): each block
//!   has a file `mod.in` of `[entries][7]` rows of `D` f16 (at a 4 KiB
//!   stride, so every bound part is aligned): `ms1` = `n1 (1 + scale_msa)`,
//!   `sh1`, `ms2`, `sh2`, `o.s` = the out rows' scale x `gate_msa / c_l`,
//!   `d.s` likewise with `gate_mlp`, `xr` = `c_{l-1} / c_l`. The engine
//!   orders the rows by entry, so a job's rows share one.
//!
//! ```text
//!   h3/pre   x                 -> q, k, v       block 0 up to its attention
//!   h3/mid   x, a              -> xo, q, k, v   block l - 1 from its attention on, block l up to its
//!   h3/post  x, a              -> xo            block 49 from its attention on
//! ```
//!
//! **VAE decoder.** One graph per ViT layer over [`VB`] tiles of `256 + 5`
//! tokens (a 256-pixel tile's 16 x 16 latents, the 4 registers, the zero
//! token), f16 weights (read from the checkpoint at load, the norms and
//! the per-channel residual scales folded in), the attention inside the
//! graph; `h3vae/out` is the final LayerNorm and `proj_out`. The residual
//! is kept x [`VS`] (its values start below 1e-2).

use crate::g::{Graph, Ty, V};
use crate::gptq::{self, Gram};
use crate::qimage::{self, f16f, f16v, mulf16, pre, q8_rows, ROWQ_KERN, ROWQ_PAD};
use crate::st::{parse_json, Checkpoint};
use crate::weights::Accel;
use crate::{mkdir, say};
use std::path::{Path, PathBuf};

/// Rows per transformer job.
pub const S: usize = 256;
/// Rows per small transformer job (the `*64` graphs).
pub const S_SMALL: usize = 64;
/// The stored residual's target magnitude: `c_l` = the block's largest
/// calibrated magnitude / `XS`.
pub const XS: f32 = 8192.0;
/// The graphs' RMSNorm epsilon on the scaled residual (f16's smallest
/// normal: a pad row is all zeros).
pub const EPS_G: f64 = 6.2e-5;
/// The MLP's up half is scaled by `HS` before the SwiGLU product (the
/// hidden state reaches ~3e5, past f16; the down input's per-row scale
/// undoes it, and `d.s` carries `1 / HS`).
pub const HS: f64 = 1.0 / 256.0;
/// Whether matrix `w` (`qkv`, `o`, `gu`) quantizes its input per row
/// ([`qlin_dyn`], as the down projection always does) instead of at its
/// calibrated range: `QWEN_H3_ACT` at build time (`dyn` = all, or a
/// list such as `qkv,o`; into its own output directory: the weights
/// differ). Default static: on the s512 reference per-row out and gate / up
/// raised the velocity error from 3.9-6.6e-2 to 7.4-10.6e-2 per step, and
/// per-row QKV left its mean unchanged (4.9e-2 vs 5.0e-2).
pub fn act_dyn(w: &str) -> bool {
	match std::env::var("QWEN_H3_ACT").as_deref() {
		Ok("dyn") => true,
		Ok(list) => list.split(',').any(|x| x == w),
		Err(_) => false,
	}
}

/// `x W^T` with the input quantized per row: `x * {w}.pre` (the rotation's
/// signs, times v's row scales for `o`), rotated, scaled to a row maximum of
/// 127 / 128 by the `rowq` SHAVE kernel (the row's maximum in [`ROWQ_PAD`]
/// pad lanes, read against zero weight columns), int8 x int8, the output
/// times `{w}.s` (when `scale`) and the row maximum.
fn qlin_dyn(g: &mut Graph, x: &V, w: &str, n: usize, k: usize, scale: bool) -> V {
	let c = g.parameter(&format!("{w}.w"), Ty::I8, &[n, k + ROWQ_PAD]);
	let c = g.convert(&c, Ty::F16);
	let sc = g.s16(2f64.powi(-7));
	let c = g.mul(&c, &sc);
	let pre = p16(g, &format!("{w}.pre"), &[k]);
	let t = g.mul(x, &pre);
	let t = crate::flash::graphs::rotate(g, &t, qimage::ROT);
	let rows = t.shape[t.shape.len() - 2];
	let z = g.c16(&[1, rows, ROWQ_PAD], &vec![0.0; rows * ROWQ_PAD]);
	let tp = g.concat(&[t, z], 2);
	let y = g.softmax(&tp, 2);
	let m = g.slice(&y, k, k + 1, 1, 2);
	let yq = g.fake_quantize(&y, -1.0, 127.0 / 128.0, 256);
	let y = g.matmul(&yq, &c, false, true);
	let y = if scale {
		let s = p16(g, &format!("{w}.s"), &[n]);
		g.mul(&y, &s)
	} else {
		y
	};
	let iq = g.s16(128.0 / 127.0);
	let m = g.mul(&m, &iq);
	g.mul(&y, &m)
}

/// A linear of [`dit_graph`]: per row ([`act_dyn`]) or static.
fn qlin(g: &mut Graph, x: &V, w: &str, n: usize, k: usize, scale: bool) -> V {
	if act_dyn(w) {
		qlin_dyn(g, x, w, n, k, scale)
	} else {
		qimage::qlin_f(g, x, w, n, k, scale, true)
	}
}

/// Zero columns for the `rowq` kernel's pad lanes: `[n][k]` -> `[n][k +
/// ROWQ_PAD]`.
pub(crate) fn pad_cols(c: &[u8], k: usize) -> Vec<u8> {
	let mut p = vec![0u8; c.len() / k * (k + ROWQ_PAD)];
	for (o, i) in p.chunks_exact_mut(k + ROWQ_PAD).zip(c.chunks_exact(k)) {
		o[..k].copy_from_slice(i);
	}
	p
}

/// Parts of a block's modulation entry (see the module docs).
pub const MPARTS: usize = 7;
/// Tiles per VAE decoder job.
pub const VB: usize = 4;
/// The VAE decoder residual's scale.
pub const VS: f32 = 16.0;
/// Latents per side of a VAE decoder tile (256 pixels / 16).
pub const VT: usize = 16;

/// True when `dir` is a MiniMax-H3-Image checkpoint (`model_index.json`'s
/// class).
pub fn is_h3(dir: &Path) -> bool {
	std::fs::read(dir.join("model_index.json"))
		.ok()
		.and_then(|b| parse_json(&b).ok())
		.and_then(|j| {
			j.get("_class_name")
				.and_then(|v| v.str().map(str::to_string))
		})
		.is_some_and(|c| c.starts_with("MiniMaxH3Image"))
}

/// The transformer's shape (`transformer_diffusers/config.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DShape {
	pub d: usize,
	pub heads: usize,
	pub hd: usize,
	pub ff: usize,
	pub layers: usize,
	/// The width of k and of v (`heads * hd` without grouped queries).
	pub kv: usize,
}

impl DShape {
	/// The width of q and of the attention's output.
	pub fn inner(&self) -> usize {
		self.heads * self.hd
	}

	/// The width of the fused q / k / v projection.
	pub fn qkv(&self) -> usize {
		self.inner() + 2 * self.kv
	}

	pub fn read(src: &Path) -> Result<DShape, String> {
		let p = src.join("transformer_diffusers").join("config.json");
		let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		let j = parse_json(&b)?;
		let u = |k: &str| -> Result<usize, String> {
			j.get(k)
				.and_then(|v| v.num())
				.map(|v| v as usize)
				.ok_or(format!("{}: no {k}", p.display()))
		};
		let (heads, hd) = (u("num_attention_heads")?, u("attention_head_dim")?);
		Ok(DShape {
			d: u("hidden_size")?,
			heads,
			hd,
			ff: u("ffn_dim")?,
			layers: u("num_layers")?,
			kv: heads * hd,
		})
	}
}

/// The VAE decoder's shape (`vae_diffusers/config.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VShape {
	pub layers: usize,
	pub heads: usize,
	pub hd: usize,
	pub ff: usize,
	pub regs: usize,
	pub theta: f64,
	pub rope_ratio: f64,
	pub eps: f64,
	/// Output values per token (`3 x 16 x 16`).
	pub out: usize,
}

impl VShape {
	pub fn d(&self) -> usize {
		self.heads * self.hd
	}

	/// Tokens per tile: the latents, the registers, the zero token.
	pub fn s(&self) -> usize {
		VT * VT + self.regs + 1
	}

	/// Rotary channels per head.
	pub fn rd(&self) -> usize {
		(self.hd as f64 * self.rope_ratio) as usize
	}

	pub fn read(src: &Path) -> Result<VShape, String> {
		let p = src.join("vae_diffusers").join("config.json");
		let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		let j = parse_json(&b)?;
		let n = |k: &str| -> Result<f64, String> {
			j.get(k)
				.and_then(|v| v.num())
				.ok_or(format!("{}: no {k}", p.display()))
		};
		let (heads, hd) = (
			n("decoder_num_attention_heads")? as usize,
			n("decoder_attention_head_dim")? as usize,
		);
		Ok(VShape {
			layers: n("decoder_num_layers")? as usize,
			heads,
			hd,
			ff: n("decoder_ffn_mult")? as usize * heads * hd,
			regs: n("decoder_num_register_tokens")? as usize,
			theta: n("decoder_rope_theta")?,
			rope_ratio: n("decoder_rope_dim_ratio")?,
			eps: n("decoder_norm_eps")?,
			out: 3 * 16 * 16,
		})
	}
}

fn p16(g: &mut Graph, name: &str, shape: &[usize]) -> V {
	g.parameter(name, Ty::F16, shape)
}

/// `x / sqrt(mean(x^2) + eps)` over the last axis (the compiler's
/// FuseRMSNorm form).
fn rms(g: &mut Graph, x: &V, eps: f64) -> V {
	let two = g.sf(x.ty, 2.0);
	let sq = g.power(x, &two);
	let ms = g.reduce_mean(&sq, -1);
	let e = g.sf(x.ty, eps);
	let t = g.add(&ms, &e);
	let t = g.sqrt(&t);
	let one = g.sf(x.ty, 1.0);
	let r = g.div(&one, &t);
	g.mul(x, &r)
}

/// `rms(x) * m + sh` with the row's modulation `m`, `sh` ([1, 1, D]).
fn rms_mod(g: &mut Graph, x: &V, m: &str, sh: &str, d: usize) -> V {
	let ms = p16(g, m, &[1, 1, d]);
	let shv = p16(g, sh, &[1, 1, d]);
	let h = rms(g, x, EPS_G);
	let h = g.mul(&h, &ms);
	g.add(&h, &shv)
}

/// The transformer's graph stems: [`S`] rows, then [`S_SMALL`] rows.
pub const DIT_STEMS: [&str; 6] = [
	"h3/pre",
	"h3/mid",
	"h3/post",
	"h3/pre64",
	"h3/mid64",
	"h3/post64",
];

/// The graph kinds of the transformer: `(name, graph)`.
pub fn dit_kinds(c: &DShape) -> Vec<(&'static str, Graph)> {
	DIT_STEMS
		.iter()
		.enumerate()
		.map(|(i, &n)| {
			let rows = if i < 3 { S } else { S_SMALL };
			(n, dit_graph(c, i % 3 != 0, i % 3 != 2, rows))
		})
		.collect()
}

/// A transformer graph's outputs in result order.
pub fn dit_outputs(kind: &str) -> &'static [&'static str] {
	match kind.trim_end_matches("64") {
		"h3/post" => &["xo"],
		"h3/pre" => &["q", "k", "v"],
		_ => &["xo", "q", "k", "v"],
	}
}

/// The modulation part a transformer graph input is, if it is one.
pub fn mod_part(name: &str) -> Option<usize> {
	["ms1", "sh1", "ms2", "sh2", "o.s", "d.s", "xr"]
		.iter()
		.position(|&n| n == name)
}

/// Whether a modulation part or weight input belongs to the block after
/// the cut (its pre part) rather than to the one before.
pub fn is_pre(name: &str) -> bool {
	name.starts_with("qkv.") || name == "ms1" || name == "sh1"
}

/// See the module docs: `post` = the block before the cut from its
/// attention on, `pre` = the block after it up to its attention, over
/// `rows` rows.
pub fn dit_graph(c: &DShape, post: bool, pre: bool, rows: usize) -> Graph {
	dit_graph_ty(c, post, pre, rows, None)
}

/// [`dit_graph`] with the code type of each matrix (`qkv`, `o`, `gu`, `d`;
/// the activations static but for `d`'s, per row), or (`None`) as
/// [`act_dyn`] and `QWEN_IMAGE_W4` choose.
pub fn dit_graph_ty(
	c: &DShape,
	post: bool,
	pre: bool,
	rows: usize,
	codes: Option<[Ty; 4]>,
) -> Graph {
	let (d, inner, ff, kv) = (c.d, c.inner(), c.ff, c.kv);
	let lin = |g: &mut Graph, x: &V, w: &str, n: usize, k: usize, scale: bool, i: usize| match codes
	{
		// the text encoder: its early layers' `d.s` reach ~1.5e3
		Some(t) => qimage::qlin_ty(g, x, w, n, k, scale, w != "d", t[i], true),
		None if w == "d" => qimage::qlin_f(g, x, w, n, k, scale, false),
		None => qlin(g, x, w, n, k, scale),
	};
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, rows, d]);
	let x = if post {
		let a = p16(&mut g, "a", &[1, rows, inner]);
		let xr = p16(&mut g, "xr", &[1, 1, d]);
		// o.s: the rows' scale x gate_msa / c_l
		let o = lin(&mut g, &a, "o", d, inner, true, 1);
		let xs = g.mul(&x, &xr);
		let x1 = g.add(&xs, &o);
		let h = rms_mod(&mut g, &x1, "ms2", "sh2", d);
		let gu = lin(&mut g, &h, "gu", 2 * ff, d, false, 2);
		let gu = g.variadic_split(&gu, 2, &[ff, ff]);
		let sg = p16(&mut g, "gu.sg", &[ff]);
		let gate = g.mul(&gu[0], &sg);
		let sw = g.swish1(&gate);
		let hs = g.s16(HS);
		let up = g.mul(&gu[1], &hs);
		let hid = g.mul(&sw, &up);
		// d.s: the rows' scale x gate_mlp / c_l
		let dn = lin(&mut g, &hid, "d", d, ff, true, 3);
		let xo = g.add(&x1, &dn);
		g.result(&xo, "xo");
		xo
	} else {
		x
	};
	if pre {
		let h = rms_mod(&mut g, &x, "ms1", "sh1", d);
		let qkv = lin(&mut g, &h, "qkv", c.qkv(), d, false, 0);
		let p = g.variadic_split(&qkv, 2, &[inner, kv, kv]);
		g.result(&p[0], "q");
		g.result(&p[1], "k");
		g.result(&p[2], "v");
	}
	g
}

/// `cos` / `sin` `[s][rd / 2]` of a tile's tokens: the 16 x 16 latents at
/// pixel centers in [-1, 1) (`h` then `w`, `rd / 4` frequencies each), the
/// registers and the zero token at angle 0.
pub fn vae_rope(c: &VShape) -> (Vec<f32>, Vec<f32>) {
	let rd = c.rd();
	let nf = rd / 4;
	let half = rd / 2;
	let inv: Vec<f32> = (0..nf)
		.map(|i| 1.0 / (c.theta as f32).powf((i * 4) as f32 / rd as f32))
		.collect();
	let s = c.s();
	let mut cos = vec![1f32; s * half];
	let mut sin = vec![0f32; s * half];
	for y in 0..VT {
		for x in 0..VT {
			let p = [
				2.0 * ((y as f32 + 0.5) / VT as f32) - 1.0,
				2.0 * ((x as f32 + 0.5) / VT as f32) - 1.0,
			];
			let r = y * VT + x;
			for ax in 0..2 {
				for (i, &f) in inv.iter().enumerate() {
					let a = 2.0 * std::f32::consts::PI * p[ax] * f;
					cos[r * half + ax * nf + i] = a.cos();
					sin[r * half + ax * nf + i] = a.sin();
				}
			}
		}
	}
	(cos, sin)
}

/// One VAE decoder layer over [`VB`] tiles: `x` [1, VB s, d] -> `xo`
/// (weights `wqkv`, `bqkv`, `wo`, `bo`, `w0`, `b0`, `w2`, `b2`, folded:
/// see [`vae_layer_weights`]).
pub fn vdec_graph(c: &VShape) -> Graph {
	let (d, nh, hd, f, s) = (c.d(), c.heads, c.hd, c.ff, c.s());
	let rows = VB * s;
	let rd = c.rd();
	let half = rd / 2;
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, rows, d]);
	let lin = |g: &mut Graph, x: &V, w: &str, b: &str, n: usize, k: usize| -> V {
		let wv = p16(g, w, &[n, k]);
		let bv = p16(g, b, &[n]);
		let y = g.matmul(x, &wv, false, true);
		g.add(&y, &bv)
	};
	let eps = c.eps * (VS as f64) * (VS as f64);
	let h = rms(&mut g, &x, eps);
	let qkv = lin(&mut g, &h, "wqkv", "bqkv", 3 * d, d);
	let p = g.variadic_split(&qkv, 2, &[d, d, d]);
	let (cs, sn) = vae_rope(c);
	let cs: Vec<f64> = cs.iter().map(|&v| v as f64).collect();
	let sn: Vec<f64> = sn.iter().map(|&v| v as f64).collect();
	let cs = g.c16(&[1, s, 1, half], &cs);
	let sn = g.c16(&[1, s, 1, half], &sn);
	let qk = |g: &mut Graph, t: &V, scale: f64| -> V {
		let t = g.reshape(t, &[VB as i64, s as i64, nh as i64, hd as i64]);
		let t = rms(g, &t, c.eps.max(EPS_G));
		let parts = g.variadic_split(&t, 3, &[half, half, hd - rd]);
		let a = g.mul(&parts[0], &cs);
		let b = g.mul(&parts[1], &sn);
		let y1 = g.sub(&a, &b);
		let a = g.mul(&parts[1], &cs);
		let b = g.mul(&parts[0], &sn);
		let y2 = g.add(&a, &b);
		let t = g.concat(&[y1, y2, parts[2].clone()], 3);
		let t = if scale != 1.0 {
			let sc = g.s16(scale);
			g.mul(&t, &sc)
		} else {
			t
		};
		let t = g.transpose(&t, &[0, 2, 1, 3]);
		g.reshape(&t, &[1, (VB * nh) as i64, s as i64, hd as i64])
	};
	let q = qk(&mut g, &p[0], 1.0 / (hd as f64).sqrt());
	let k = qk(&mut g, &p[1], 1.0);
	let v = g.reshape(&p[2], &[VB as i64, s as i64, nh as i64, hd as i64]);
	let v = g.transpose(&v, &[0, 2, 1, 3]);
	let v = g.reshape(&v, &[1, (VB * nh) as i64, s as i64, hd as i64]);
	let sc = g.matmul(&q, &k, false, true);
	let pr = g.softmax(&sc, 3);
	let o = g.matmul(&pr, &v, false, false);
	let o = g.reshape(&o, &[VB as i64, nh as i64, s as i64, hd as i64]);
	let o = g.transpose(&o, &[0, 2, 1, 3]);
	let o = g.reshape(&o, &[1, rows as i64, d as i64]);
	let o = lin(&mut g, &o, "wo", "bo", d, d);
	let x1 = g.add(&x, &o);
	let h = rms(&mut g, &x1, eps);
	let gu = lin(&mut g, &h, "w0", "b0", 2 * f, d);
	let gu = g.variadic_split(&gu, 2, &[f, f]);
	let sw = g.swish1(&gu[1]);
	let act = g.mul(&gu[0], &sw);
	let y = lin(&mut g, &act, "w2", "b2", d, f);
	let xo = g.add(&x1, &y);
	g.result(&xo, "xo");
	g
}

/// The VAE decoder's head over [`VB`] tiles: LayerNorm (affine `g`, `b`)
/// of the latent tokens and `proj_out` (`w`, `bo`): `x` [1, VB s, d] ->
/// `y` [1, VB 256, out].
pub fn vout_graph(c: &VShape) -> Graph {
	let (d, s, tok) = (c.d(), c.s(), VT * VT);
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, VB * s, d]);
	let t = g.reshape(&x, &[VB as i64, s as i64, d as i64]);
	let t = g.slice(&t, 0, tok, 1, 1);
	let t = g.reshape(&t, &[1, (VB * tok) as i64, d as i64]);
	let t = g.mvn(&t, c.eps * (VS as f64) * (VS as f64));
	let gv = p16(&mut g, "g", &[d]);
	let bv = p16(&mut g, "b", &[d]);
	let t = g.mul(&t, &gv);
	let t = g.add(&t, &bv);
	let w = p16(&mut g, "w", &[c.out, d]);
	let bo = p16(&mut g, "bo", &[c.out]);
	let y = g.matmul(&t, &w, false, true);
	let y = g.add(&y, &bo);
	g.result(&y, "y");
	g
}

/// Layer `l`'s folded f32 weights in the order of [`VAE_WNAMES`]: the
/// RMSNorm weights into the QKV and `ff.net.0` columns, the per-channel
/// residual scales (`scale1`, `scale2`) and [`VS`] into the out and
/// `ff.net.2` rows and biases.
pub fn vae_layer_weights(ck: &Checkpoint, c: &VShape, l: usize) -> Result<Vec<Vec<f32>>, String> {
	let (d, f) = (c.d(), c.ff);
	let p = |n: &str| format!("decoder.transformer_blocks.{l}.{n}");
	let n1 = ck.vec(&p("norm1.weight"), d)?;
	let n2 = ck.vec(&p("norm2.weight"), d)?;
	let s1 = ck.vec(&p("scale1"), d)?;
	let s2 = ck.vec(&p("scale2"), d)?;
	let cols = |w: &mut [f32], k: usize, n: &[f32]| {
		for row in w.chunks_exact_mut(k) {
			row.iter_mut().zip(n).for_each(|(v, &s)| *v *= s);
		}
	};
	let mut wqkv = ck.mat(&p("attn.to_q.weight"), d, d)?;
	wqkv.extend(ck.mat(&p("attn.to_k.weight"), d, d)?);
	wqkv.extend(ck.mat(&p("attn.to_v.weight"), d, d)?);
	cols(&mut wqkv, d, &n1);
	let mut bqkv = ck.vec(&p("attn.to_q.bias"), d)?;
	bqkv.extend(ck.vec(&p("attn.to_k.bias"), d)?);
	bqkv.extend(ck.vec(&p("attn.to_v.bias"), d)?);
	let rows = |w: &mut [f32], k: usize, b: &mut [f32], s: &[f32]| {
		for (r, row) in w.chunks_exact_mut(k).enumerate() {
			let f = s[r] * VS;
			row.iter_mut().for_each(|v| *v *= f);
			b[r] *= f;
		}
	};
	let mut wo = ck.mat(&p("attn.to_out.0.weight"), d, d)?;
	let mut bo = ck.vec(&p("attn.to_out.0.bias"), d)?;
	rows(&mut wo, d, &mut bo, &s1);
	let mut w0 = ck.mat(&p("ff.net.0.proj.weight"), 2 * f, d)?;
	cols(&mut w0, d, &n2);
	let b0 = ck.vec(&p("ff.net.0.proj.bias"), 2 * f)?;
	let mut w2 = ck.mat(&p("ff.net.2.weight"), d, f)?;
	let mut b2 = ck.vec(&p("ff.net.2.bias"), d)?;
	rows(&mut w2, f, &mut b2, &s2);
	Ok(vec![wqkv, bqkv, wo, bo, w0, b0, w2, b2])
}

/// The decoder layer graph's weight inputs, in [`vae_layer_weights`]' order.
pub const VAE_WNAMES: [&str; 8] = ["wqkv", "bqkv", "wo", "bo", "w0", "b0", "w2", "b2"];

/// The calibration: `in` (the blocks' input's largest magnitude) and per
/// block `[r_qkv, r_o, r_gu, xmax, gmax, hmax]`: the rotated QKV, out and
/// gate / up inputs' largest magnitudes (headroom included), the block
/// output's, and (checked, not used) the MLP gate's and hidden state's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cal {
	pub xin: f32,
	pub blocks: Vec<[f32; 6]>,
}

pub fn cal_path(out: &Path) -> PathBuf {
	out.join("cal").join("h3dit.cal")
}

/// The AdaLN tables the calibration writes (`cal/h3adaln.bin`): u32 count
/// of timesteps, blocks, modalities, width; the timesteps (f32); per block
/// `[ts][modality][6][width]` f32.
pub fn adaln_path(out: &Path) -> PathBuf {
	out.join("cal").join("h3adaln.bin")
}

impl Cal {
	pub fn write(&self, out: &Path) -> Result<(), String> {
		mkdir(&out.join("cal"))?;
		let mut t = String::from(
			"# MiniMax-H3-Image: the blocks' input max, then per block the input ranges of qkv, o, gu\n# (rotated, headroom included), the output's max, the MLP gate's and hidden state's max\n",
		);
		t += &format!("in {}\n", self.xin);
		for (l, v) in self.blocks.iter().enumerate() {
			t += &format!(
				"{l} {} {} {} {} {} {}\n",
				v[0], v[1], v[2], v[3], v[4], v[5]
			);
		}
		std::fs::write(cal_path(out), t).map_err(|e| e.to_string())
	}

	pub fn read(out: &Path, layers: usize) -> Result<Cal, String> {
		let p = cal_path(out);
		let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		let mut c = Cal {
			xin: 0.0,
			blocks: vec![[0.0; 6]; layers],
		};
		let mut seen = vec![false; layers];
		for line in t
			.lines()
			.map(str::trim)
			.filter(|l| !l.is_empty() && !l.starts_with('#'))
		{
			let f: Vec<&str> = line.split_whitespace().collect();
			let bad = || format!("{}: bad line {line}", p.display());
			if f.len() == 2 && f[0] == "in" {
				c.xin = f[1].parse().map_err(|_| bad())?;
				continue;
			}
			if f.len() != 7 {
				return Err(bad());
			}
			let l: usize = f[0].parse().map_err(|_| bad())?;
			if l >= layers {
				return Err(bad());
			}
			for i in 0..6 {
				c.blocks[l][i] = f[i + 1].parse().map_err(|_| bad())?;
			}
			if c.blocks[l][..4].iter().any(|&v| !(v > 0.0)) {
				return Err(bad());
			}
			seen[l] = true;
		}
		if !(c.xin > 0.0) {
			return Err(format!("{}: no input line", p.display()));
		}
		if let Some(l) = seen.iter().position(|s| !s) {
			return Err(format!("{}: no line for block {l}", p.display()));
		}
		Ok(c)
	}

	/// The residual's stored scale after each block (`c_l`), and the
	/// input's.
	pub fn scales(&self) -> (f32, Vec<f32>) {
		(
			self.xin / XS,
			self.blocks.iter().map(|b| b[3] / XS).collect(),
		)
	}
}

/// Write the AdaLN tables ([`adaln_path`]).
pub fn write_adaln(
	out: &Path,
	ts: &[f32],
	d: usize,
	modalities: usize,
	tables: &[Vec<f32>],
) -> Result<(), String> {
	mkdir(&out.join("cal"))?;
	let mut b =
		Vec::with_capacity(16 + ts.len() * 4 + tables.iter().map(|t| t.len() * 4).sum::<usize>());
	for v in [ts.len(), tables.len(), modalities, d] {
		b.extend((v as u32).to_le_bytes());
	}
	ts.iter().for_each(|v| b.extend(v.to_le_bytes()));
	for t in tables {
		if t.len() != ts.len() * modalities * 6 * d {
			return Err(format!("adaln table: {} values", t.len()));
		}
		t.iter().for_each(|v| b.extend(v.to_le_bytes()));
	}
	std::fs::write(adaln_path(out), b).map_err(|e| e.to_string())
}

/// Read [`adaln_path`]: (timesteps, modalities, per block its table).
pub fn read_adaln(
	out: &Path,
	layers: usize,
	d: usize,
) -> Result<(Vec<f32>, usize, Vec<Vec<f32>>), String> {
	let p = adaln_path(out);
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	let u = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
	if b.len() < 16 {
		return Err(format!("{}: truncated", p.display()));
	}
	let (nt, nl, nm, nd) = (u(0), u(1), u(2), u(3));
	if nl != layers || nd != d {
		return Err(format!(
			"{}: {nl} blocks of {nd}, expected {layers} of {d}",
			p.display()
		));
	}
	let per = nt * nm * 6 * d;
	if b.len() != 16 + nt * 4 + nl * per * 4 {
		return Err(format!("{}: size", p.display()));
	}
	let f = |o: usize, n: usize| -> Vec<f32> {
		b[o..o + n * 4]
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes(c.try_into().unwrap()))
			.collect()
	};
	let ts = f(16, nt);
	let o0 = 16 + nt * 4;
	Ok((ts, nm, (0..nl).map(|l| f(o0 + l * per * 4, per)).collect()))
}

/// A block's file under `w/` (`h3dit{l}.<name>.in`).
pub fn wfile(wdir: &Path, l: usize, name: &str) -> PathBuf {
	wdir.join(format!("h3dit{l}.{name}.in"))
}

/// The modulation file's part stride in bytes (`D` f16 rounded up to
/// 4 KiB).
pub fn mstride(d: usize) -> usize {
	(d * 2).next_multiple_of(4096)
}

/// The inputs whose Hessians (`X^T X`, [`Gram`]) the calibration records
/// per block, in file order: QKV, out, gate / up, and down (as the MLP's
/// hidden state `silu(gate) up`, before the up rows' scales).
pub const HNAMES: [&str; 4] = ["qkv", "o", "gu", "d"];

/// The widths of [`HNAMES`].
pub fn hess_widths(c: &DShape) -> [usize; 4] {
	[c.d, c.inner(), c.d, c.ff]
}

/// Block `l`'s Hessian of input `name` (`cal/h3hess/{l}.{name}.f32`: the
/// [`Gram`]'s stored floats).
pub fn hess_path(out: &Path, l: usize, name: &str) -> PathBuf {
	out.join("cal")
		.join("h3hess")
		.join(format!("{l}.{name}.f32"))
}

/// Whether the block weights are rounded by GPTQ against the calibrated
/// Hessians (default) or to nearest (`QWEN_H3_GPTQ=0`, into its own output
/// directory: the weights differ).
pub fn gptq_on() -> bool {
	std::env::var("QWEN_H3_GPTQ").map_or(true, |v| v != "0")
}

/// Write block `l`'s Hessians.
pub fn write_hess(out: &Path, l: usize, g: &[Gram; 4]) -> Result<(), String> {
	mkdir(&out.join("cal").join("h3hess"))?;
	for (n, g) in HNAMES.iter().zip(g) {
		crate::weights::write(&hess_path(out, l, n), &g.to_bytes())?;
	}
	Ok(())
}

/// Whether every block's Hessians are there, at their sizes.
pub fn hess_complete(out: &Path, c: &DShape) -> bool {
	let w = hess_widths(c);
	(0..c.layers).all(|l| {
		HNAMES.iter().zip(w).all(|(n, k)| {
			std::fs::metadata(hess_path(out, l, n))
				.is_ok_and(|m| m.len() as usize == Gram::new(k).h.len() * 4)
		})
	})
}

/// Block `l`'s Hessian `i` of [`HNAMES`] as the full `[k][k]` matrix.
fn read_hess(out: &Path, l: usize, i: usize, k: usize) -> Result<Vec<f32>, String> {
	let p = hess_path(out, l, HNAMES[i]);
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	Ok(Gram::from_bytes(k, &b)
		.map_err(|e| format!("{}: {e}", p.display()))?
		.full())
}

/// [`q8_rows`] (same output) with the codes rounded by GPTQ against the
/// unrotated input Hessian `h` `[k][k]` (consumed) instead of to nearest:
/// the rotated rows keep their scale `max|w| / 127`.
fn q8_gptq(
	w: &[f32],
	n: usize,
	k: usize,
	r: f32,
	h: Vec<f32>,
	accel: &Accel,
) -> Result<(Vec<u8>, Vec<u8>), String> {
	let sg = gptq::signs(k);
	let mut wr = w.to_vec();
	gptq::rotate_rows_b(&mut wr, k, &sg, qimage::ROT);
	let mut h = h;
	gptq::rotate_sym_b(&mut h, k, &sg, qimage::ROT);
	let mut p = gptq::prepare(h, k);
	let x = match accel.inv_chol(&p.hr, k) {
		Some(x) => x,
		None => gptq::inv_chol(std::mem::take(&mut p.hr), k)?,
	};
	let f = p.finish(x);
	let sc: Vec<f32> = wr
		.chunks_exact(k)
		.map(|row| {
			row.iter()
				.fold(0f32, |a, &v| a.max(v.abs()))
				.max(f32::MIN_POSITIVE)
				/ 127.0
		})
		.collect();
	let q = gptq::gptq_range(wr, n, &f, &sc, -127.0, 127.0);
	let fr = if r > 0.0 { 128.0 * r / 127.0 } else { 1.0 };
	let so: Vec<f32> = sc.iter().map(|&s| s * 128.0 * fr).collect();
	Ok((q.into_iter().map(|c| c as u8).collect(), f16v(&so)))
}

/// `w/h3scale.f32`: `c_in`, then `c_l` per block; `w/h3ts.f32`: the
/// modulation entries' timesteps (entry = timestep index x modalities +
/// modality).
pub fn scale_file(wdir: &Path) -> PathBuf {
	wdir.join("h3scale.f32")
}

pub fn ts_file(wdir: &Path) -> PathBuf {
	wdir.join("h3ts.f32")
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
	v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `OUT/w/h3dit{L}.*.in` of every block (skipped per block when its last
/// file exists), and the scale and timestep files.
fn step_weights(src: &Path, out: &Path, accel: &Accel) -> Result<(), String> {
	let c = DShape::read(src)?;
	let use_gptq = gptq_on() && !qimage::w4();
	let wdir = out.join("w");
	let todo = (0..c.layers).any(|l| !wfile(&wdir, l, "mod").exists());
	if use_gptq && todo && !hess_complete(out, &c) {
		return Err(format!(
			"{}: the Hessians are missing (run the cal step, or QWEN_H3_GPTQ=0 for round to nearest)",
			out.join("cal").join("h3hess").display()
		));
	}
	let (d, inner, ff) = (c.d, c.inner(), c.ff);
	let ck = Checkpoint::open_dir(&src.join("transformer_diffusers"))?;
	let cal = Cal::read(out, c.layers)?;
	let (ts, nm, tables) = read_adaln(out, c.layers, d)?;
	let (c_in, cs) = cal.scales();
	mkdir(&wdir)?;
	crate::weights::write(&scale_file(&wdir), &f32_bytes(&[&[c_in][..], &cs].concat()))?;
	crate::weights::write(&ts_file(&wdir), &f32_bytes(&ts))?;
	let ms = mstride(d);
	let entries = ts.len() * nm;
	for l in 0..c.layers {
		if wfile(&wdir, l, "mod").exists() {
			continue;
		}
		let t = std::time::Instant::now();
		let p = |n: &str| format!("transformer_blocks.{l}.{n}");
		let put = |name: &str, b: &[u8]| crate::weights::write(&wfile(&wdir, l, name), b);
		let [r_qkv, r_o, r_gu, _, _, _] = cal.blocks[l];
		// per-row activations: no range in the row scales, the signs as `pre`
		let dy = |w: &str| act_dyn(w);
		let r_qkv = if dy("qkv") { 0.0 } else { r_qkv };
		let r_o = if dy("o") { 0.0 } else { r_o };
		let r_gu = if dy("gu") { 0.0 } else { r_gu };
		let pre = |k: usize, r: f32| -> Vec<u8> {
			if r == 0.0 {
				f16v(&crate::gptq::signs(k))
			} else {
				pre(k, r)
			}
		};
		let padc = |c: Vec<u8>, k: usize, w: &str| if dy(w) { pad_cols(&c, k) } else { c };
		// GPTQ against Hessian `i` of HNAMES (`cs`: the input's column scales), else round to nearest
		let q8 = |w: &[f32], n: usize, k: usize, r: f32, i: usize, cs: Option<&[f32]>| {
			if !use_gptq {
				return Ok(q8_rows(w, n, k, r));
			}
			let mut h = read_hess(out, l, i, k)?;
			if let Some(cs) = cs {
				for (a, row) in h.chunks_exact_mut(k).enumerate() {
					row.iter_mut().zip(cs).for_each(|(v, &b)| *v /= cs[a] * b);
				}
			}
			q8_gptq(w, n, k, r, h, accel)
		};
		let mut qkv = ck.mat(&p("attn.to_q.weight"), inner, d)?;
		qkv.extend(ck.mat(&p("attn.to_k.weight"), inner, d)?);
		qkv.extend(ck.mat(&p("attn.to_v.weight"), inner, d)?);
		let (cq, sq) = q8(&qkv, 3 * inner, d, r_qkv, 0, None)?;
		drop(qkv);
		put("qkv.w", &padc(cq, d, "qkv"))?;
		put("qkv.s", &sq)?;
		put("qkv.pre", &pre(d, r_qkv))?;
		let sv = f16f(&sq[2 * inner * 2..]);
		put("nq", &f32_bytes(&ck.vec(&p("attn.norm_q.weight"), c.hd)?))?;
		put("nk", &f32_bytes(&ck.vec(&p("attn.norm_k.weight"), c.hd)?))?;
		let (co, so) = q8(
			&ck.mat(&p("attn.to_out.0.weight"), d, inner)?,
			d,
			inner,
			r_o,
			1,
			None,
		)?;
		put("o.w", &padc(co, inner, "o"))?;
		put("o.pre", &mulf16(&pre(inner, r_o), &sv))?;
		// SwiGLU: net.0.proj's first half is the value (up), the second the gate
		let ff0 = p("ff.net.0.proj.weight");
		let mut gu = ck.rows(&ff0, ff, 2 * ff)?;
		gu.extend(ck.rows(&ff0, 0, ff)?);
		let (mut cg, sg) = q8(&gu, 2 * ff, d, r_gu, 2, None)?;
		drop(gu);
		// the up rows x the down input's rotation signs, the up half's row
		// scales into the columns of `d`
		for (j, &v) in crate::gptq::signs(ff).iter().enumerate() {
			if v < 0.0 {
				cg[(ff + j) * d..(ff + j + 1) * d]
					.iter_mut()
					.for_each(|b| *b = (*b as i8).wrapping_neg() as u8);
			}
		}
		let su = f16f(&sg[ff * 2..]);
		let mut dw = ck.mat(&p("ff.net.2.weight"), d, ff)?;
		for row in dw.chunks_exact_mut(ff) {
			row.iter_mut().zip(&su).for_each(|(w, &f)| *w *= f);
		}
		put("gu.sg", &sg[..ff * 2])?;
		put("gu.w", &padc(cg, d, "gu"))?;
		put("gu.pre", &pre(d, r_gu))?;
		// the down input in the graph is the hidden state / the up rows' scales
		let (cd, sd) = q8(&dw, d, ff, 0.0, 3, Some(&su[..]))?;
		drop(dw);
		put("d.w", &pad_cols(&cd, ff))?;
		drop(cd);
		// the modulation entries (module docs)
		let (so, sd) = (f16f(&so), f16f(&sd));
		let n1 = ck.vec(&p("norm1.weight"), d)?;
		let n2 = ck.vec(&p("norm2.weight"), d)?;
		let cl = cs[l];
		let cp = if l == 0 { c_in } else { cs[l - 1] };
		let mut m = vec![0u8; entries * MPARTS * ms];
		for e in 0..entries {
			let tb = &tables[l][e * 6 * d..(e + 1) * 6 * d];
			let part = |i: usize| &tb[i * d..(i + 1) * d];
			let (sh_a, sc_a, g_a, sh_m, sc_m, g_m) =
				(part(0), part(1), part(2), part(3), part(4), part(5));
			let parts: [Vec<f32>; MPARTS] = [
				(0..d).map(|j| n1[j] * (1.0 + sc_a[j])).collect(),
				sh_a.to_vec(),
				(0..d).map(|j| n2[j] * (1.0 + sc_m[j])).collect(),
				sh_m.to_vec(),
				(0..d).map(|j| so[j] * g_a[j] / cl).collect(),
				(0..d).map(|j| sd[j] * g_m[j] / cl / HS as f32).collect(),
				vec![cp / cl; d],
			];
			for (i, v) in parts.iter().enumerate() {
				let o = (e * MPARTS + i) * ms;
				m[o..o + d * 2].copy_from_slice(&f16v(v));
			}
		}
		put("mod", &m)?;
		let how = if use_gptq { "int8 GPTQ" } else { "int8" };
		say(&format!(
			"H3 transformer block {l}: {how} in {:.1} s",
			t.elapsed().as_secs_f64()
		));
	}
	Ok(())
}

/// `OUT/graphs/h3/*.xml` and `OUT/graphs/h3vae/*.xml`.
fn step_graphs(src: &Path, out: &Path) -> Result<(), String> {
	let gdir = out.join("graphs");
	let c = DShape::read(src)?;
	for (n, g) in dit_kinds(&c) {
		crate::sets::save(&gdir, n, &g)?;
	}
	let tc = crate::h3te::shape(src, crate::h3te::LAYERS)?;
	if crate::h3te::complete(out, &tc) {
		crate::h3te::step_graphs(out, &tc)?;
	}
	let v = VShape::read(src)?;
	crate::sets::save(&gdir, "h3vae/dec", &vdec_graph(&v))?;
	crate::sets::save(&gdir, "h3vae/out", &vout_graph(&v))?;
	say(&format!("H3 graphs in {}", gdir.display()));
	Ok(())
}

/// Compile the graphs whose blob is missing: the transformer's with the
/// `rowq` kernel, the VAE decoder's without SHAVE kernels (its attention's
/// softmax is the kernel's carrier).
fn step_compile(o: &crate::Options) -> Result<(), String> {
	let t = crate::sets::Tools::find(o.compiler.as_deref(), o.jobs)?;
	let gdir = o.out.join("graphs");
	crate::sets::write_kernel(
		&gdir.join(ROWQ_KERN),
		include_str!("../../npu/shave/rowq.sk"),
		"softmax",
		&[("PAD".to_string(), ROWQ_PAD as i64)],
	)?;
	let te = crate::h3te::TE_STEMS
		.iter()
		.filter(|s| gdir.join(format!("{s}.xml")).exists());
	for stem in DIT_STEMS
		.iter()
		.chain(&["h3vae/dec", "h3vae/out"])
		.chain(te)
	{
		if gdir.join(format!("{stem}.blob")).exists() {
			continue;
		}
		say(&format!("compiling {stem}"));
		// the `rowq` kernel serves the int8 down projections
		let kern = if stem.starts_with("h3/")
			|| (stem.starts_with("h3te/") && crate::h3te::codes(&o.out)?[3] == Ty::I8)
		{
			ROWQ_KERN
		} else {
			""
		};
		crate::sets::compile_graph_kern(&t, &gdir, stem, Some(kern), false)?;
	}
	Ok(())
}

/// `OUT/image.npu`.
fn step_sets(o: &crate::Options) -> Result<(), String> {
	let src = std::path::absolute(&o.src).unwrap_or_else(|_| o.src.clone());
	let gdir = o.out.join("graphs");
	let te = crate::h3te::TE_STEMS
		.iter()
		.all(|s| gdir.join(format!("{s}.blob")).exists());
	let mut t = format!(
		"# MiniMax-H3-Image: the transformer on the NPU (int8 per row), its attention on the cores;
# the VAE decoder's ViT on the NPU (f16)
arch       = h3-image
checkpoint = {}
graphs     = graphs/h3
weights    = w
vae        = graphs/h3vae
",
		src.to_string_lossy()
	);
	if te {
		t += "# the text encoder on the NPU (its attention on the cores)
te         = w/te
te_graphs  = graphs/h3te
";
	}
	let p = o.out.join("image.npu");
	std::fs::write(&p, t).map_err(|e| format!("{}: {e}", p.display()))?;
	say(&format!("done: {}", p.display()));
	Ok(())
}

/// Writes [`cal_path`], [`adaln_path`] and, with [`gptq_on`], the
/// Hessians ([`write_hess`]) (the quantized CPU backend over
/// the calibration images; it lives in `imagegen`, which depends on this
/// crate).
pub type Calibrate<'a> = &'a dyn Fn(&Path, &Path) -> Result<(), String>;

/// The text encoder's calibration and weights (`crate::h3te`, `w/te`):
/// `(src, out, accel)`; it lives in `imagegen` too.
pub type TeBuild<'a> = &'a dyn Fn(&Path, &Path, &Accel) -> Result<(), String>;

/// The H3 build: `cal` (`calibrate(src, out)`), `weights`, `te`
/// (`te(src, out, accel)`), `graphs`, `compile`, `sets`.
pub fn build(
	o: &crate::Options,
	calibrate: Option<Calibrate>,
	te: Option<TeBuild>,
) -> Result<(), String> {
	mkdir(&o.out)?;
	let c = DShape::read(&o.src)?;
	// the Hessians are needed only while a block's weights are missing
	let wdir = o.out.join("w");
	let need_hess = gptq_on()
		&& !qimage::w4()
		&& (0..c.layers).any(|l| !wfile(&wdir, l, "mod").exists())
		&& !hess_complete(&o.out, &c);
	if o.wants("cal")
		&& (Cal::read(&o.out, c.layers).is_err()
			|| read_adaln(&o.out, c.layers, c.d).is_err()
			|| need_hess)
	{
		let f = calibrate
			.ok_or("the calibration (cal/h3dit.cal) is made by `cli H3_DIR --npu-build OUT`")?;
		f(&o.src, &o.out)?;
	}
	if o.wants("weights") {
		let (accel, what) = Accel::new(o.gpu);
		if gptq_on() {
			say(&format!("H3 GPTQ factorizations: {what}"));
		}
		step_weights(&o.src, &o.out, &accel)?;
	}
	if o.wants("te") {
		let tc = crate::h3te::shape(&o.src, crate::h3te::LAYERS)?;
		if !crate::h3te::complete(&o.out, &tc) {
			let f = te.ok_or(
				"the text encoder's weights (w/te) are made by `cli H3_DIR --npu-build OUT`",
			)?;
			let (accel, what) = Accel::new(o.gpu);
			say(&format!("H3 text encoder GPTQ factorizations: {what}"));
			f(&o.src, &o.out, &accel)?;
		}
	}
	if o.wants("graphs") {
		step_graphs(&o.src, &o.out)?;
	}
	if o.wants("compile") {
		step_compile(o)?;
	}
	if o.wants("sets") {
		step_sets(o)?;
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn calibration_file_roundtrips() {
		let d = std::env::temp_dir().join(format!("qwen-rs-h3cal-{}", std::process::id()));
		let c = Cal {
			xin: 128.5,
			blocks: vec![
				[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
				[1.5, 2.5, 3.5, 9e6, 0.0, 0.0],
			],
		};
		c.write(&d).unwrap();
		assert_eq!(Cal::read(&d, 2).unwrap(), c);
		assert!(Cal::read(&d, 3).is_err(), "a block without a line");
		let (ci, cs) = c.scales();
		assert_eq!(ci, 128.5 / XS);
		assert_eq!(cs[1], 9e6 / XS);
		let t = vec![vec![0.5f32; 2 * 3 * 6 * 4]; 2];
		write_adaln(&d, &[0.25, 0.75], 4, 3, &t).unwrap();
		let (ts, nm, r) = read_adaln(&d, 2, 4).unwrap();
		assert_eq!((ts, nm, r), (vec![0.25, 0.75], 3, t));
		let _ = std::fs::remove_dir_all(&d);
	}

	#[test]
	fn mod_parts_are_named() {
		assert_eq!(mod_part("ms1"), Some(0));
		assert_eq!(mod_part("xr"), Some(6));
		assert_eq!(mod_part("o.w"), None);
		assert!(is_pre("qkv.w") && is_pre("sh1") && !is_pre("o.s") && !is_pre("gu.sg"));
		assert_eq!(mstride(5376), 12288);
	}
}
