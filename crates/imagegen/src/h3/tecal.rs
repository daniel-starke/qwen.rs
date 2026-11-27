//! The NPU build's text encoder step (`npubuild::h3te`, `w/te`): the float
//! text encoder (BF16 widened to f32) runs frozen prompts, edits with one
//! and two references and stretches of the build's calibration texts layer
//! by layer; each layer records the ranges of its rotated QKV, out and
//! gate / up inputs, its residual's and MLP's largest magnitudes and the
//! Hessians of its four matrices' inputs, and is quantized (GPTQ) and
//! written before the next layer runs. The next layer is fed the float
//! layer's output.
//!
//! The references are the first two calibration images of the transformer
//! (`super::npucal`), made again by the quantized CPU backend (frozen
//! prompts and seeds: a calibration is reproducible). A layer whose files
//! exist only runs forward, so an interrupted step resumes.

use super::config::{Config, TAG_TEXT};
use super::pipeline::{Model, Reference, MIN_AREA_OVERRIDE};
use crate::cond::Cond;
use crate::job::{ImageSrc, Options};
use crate::te::TextEncoder;
use crate::WFmt;
use npubuild::gptq::{self, Gram};
use npubuild::h3te::{self, LayerW};
use npubuild::weights::Accel;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Text-to-image prompts (frozen): short and long, lettering, Chinese.
const PROMPTS: [&str; 12] = [
	"A red fox sitting in a snowy forest at dawn, soft golden light, photograph, 85 mm",
	"A vintage travel poster with the large title \"H3 CITY\" and a skyline at sunset, flat colours, bold lettering",
	"A portrait of an elderly fisherman with a weathered face and a knitted cap, harbour in the background, overcast light, photograph",
	"An isometric illustration of a tiny workshop with tools, plants and a cat, pastel palette, clean lines",
	"A bowl of ramen on a wooden counter, steam rising, a soft-boiled egg cut in half, slices of pork, green onions and nori, warm restaurant light from the left, shallow depth of field, food photography",
	"Product shot of a matte black wireless headphone on a white seamless background, studio lighting with a soft key light and a rim light, subtle reflection on the floor, high detail, commercial catalogue style",
	"A hand-lettered chalkboard menu outside a small cafe that reads \"TODAY: Lemon Tart 4.50 / Flat White 3.20 / Open until 6 pm\", a bicycle leaning against the wall, morning street scene, photograph",
	"A wide aerial view of a river delta at golden hour: braided channels of water reflecting the orange sky, sandbars, patches of green vegetation, a small fishing boat leaving a wake, mist over the horizon, documentary photograph taken from a drone at about 300 metres altitude",
	"Watercolor illustration for a children's book: a small grey elephant holding a red umbrella walks through a rainy village, puddles, cobblestones, cozy houses with lit windows, loose brush strokes, soft paper texture, gentle and whimsical mood",
	"A detailed technical cutaway diagram of a mechanical wristwatch, showing the mainspring, gear train, escapement and balance wheel, labelled with thin leader lines and small sans-serif captions, blueprint style on dark blue paper",
	"一只橘色的小猫趴在洒满阳光的窗台上睡觉，窗外是盛开的樱花树，柔和的自然光，温暖的色调，高清摄影",
	"夜晚的上海外滩，黄浦江两岸灯火辉煌，东方明珠电视塔倒映在水面上，长曝光摄影，天空呈深蓝色，画面左下角有白色文字\"上海之夜\"",
];

/// Edits (frozen): the prompt and its references (indices into the
/// generated images).
const EDITS: [(&str, &[usize]); 2] = [
	("Turn <Picture 1> into a watercolor painting at night, with the moon behind the trees", &[0]),
	(
		"Place the fox from <Picture 1> in front of the skyline of <Picture 2>, keeping the poster's flat colours and its title",
		&[0, 1],
	),
];

/// The generated references: `(prompt, seed)` at 512x512, the
/// transformer calibration's first two images.
const REFS: [(usize, u64); 2] = [(0, 1), (1, 2)];

/// Stretches of the build's calibration texts (`npubuild::flash_calib`):
/// this many of this many tokens each.
const TEXT_CHUNKS: usize = 2;
const TEXT_TOKENS: usize = 512;

/// Headroom on the largest rotated inputs seen.
const HEADROOM: f32 = 1.25;

/// One calibration sequence.
struct Seq {
	/// The residual `[s][d]`.
	x: Vec<f32>,
	cos: Vec<f32>,
	sin: Vec<f32>,
	/// Image rows and, per deepstack layer, their features.
	img: Vec<usize>,
	deep: Vec<Vec<f32>>,
}

fn rot_max(x: &[f32], k: usize) -> f32 {
	let mut t = x.to_vec();
	gptq::rotate_rows_b(&mut t, k, &gptq::signs(k), npubuild::qimage::ROT);
	t.iter().fold(0f32, |a, v| a.max(v.abs()))
}

fn amax(x: &[f32]) -> f32 {
	x.iter().fold(0f32, |a, v| a.max(v.abs()))
}

/// The two reference images (quantized CPU backend).
fn references(src: &Path, log: &mut dyn FnMut(&str)) -> Result<Vec<ImageSrc>, String> {
	log("text encoder calibration: the references (quantized CPU backend)");
	let m = Model::load(src, WFmt::Q6K, |_, _, _| {})?;
	MIN_AREA_OVERRIDE.store(1024, Ordering::Relaxed);
	let r = REFS
		.iter()
		.map(|&(p, seed)| {
			let o = m.generate(
				&Options {
					prompt: PROMPTS[p].to_string(),
					width: 512,
					height: 512,
					seed,
					..Options::default()
				},
				|_, _, _| {},
			)?;
			Ok(ImageSrc::Rgba(o.width, o.height, o.rgba))
		})
		.collect();
	MIN_AREA_OVERRIDE.store(0, Ordering::Relaxed);
	r
}

/// The calibration sequences' inputs.
fn sequences(
	src: &Path,
	tenc: &TextEncoder,
	log: &mut dyn FnMut(&str),
) -> Result<Vec<Seq>, String> {
	let mut v = Vec::new();
	let mut add = |ids: &[u32], vis: Option<&crate::te::VisionRows>| -> Result<(), String> {
		let grids: &[(usize, usize)] = vis.map_or(&[], |v| &v.grids);
		let pos = tenc.positions(ids, grids)?;
		let (x, img) = tenc.inputs(ids, vis)?;
		let (cos, sin) = tenc.rope_tables(&pos);
		v.push(Seq {
			x,
			cos,
			sin,
			img,
			deep: vis.map_or_else(Vec::new, |v| v.deep.clone()),
		});
		Ok(())
	};
	for p in PROMPTS {
		let (ids, tags) = super::te::presentation(&tenc.tok, p, &[])?;
		debug_assert!(tags.iter().all(|&t| t == TAG_TEXT));
		add(&ids, None)?;
	}
	for t in npubuild::flash_calib() {
		let ids = tenc.tok.encode(t);
		for c in ids.chunks(TEXT_TOKENS).take(TEXT_CHUNKS) {
			add(c, None)?;
		}
	}
	let refs = references(src, log)?;
	for (prompt, which) in EDITS {
		let conds: Vec<Cond> = which
			.iter()
			.map(|&i| Reference::new(&refs[i]).map(|r| r.vision))
			.collect::<Result<_, _>>()?;
		let vis = crate::cond::vision_rows(src, &tenc.ck, &conds, None)?;
		let pads: Vec<usize> = vis.grids.iter().map(|&(r, c)| r * c).collect();
		let (ids, _) = super::te::presentation(&tenc.tok, prompt, &pads)?;
		add(&ids, Some(&vis))?;
	}
	Ok(v)
}

/// Layer `l`'s float weights.
fn layer_w(tenc: &TextEncoder, c: &npubuild::h3img::DShape, l: usize) -> Result<LayerW, String> {
	let (d, inner, kv, ff) = (c.d, c.inner(), c.kv, c.ff);
	let n = |x: &str| tenc.tensor(&format!("layers.{l}.{x}"));
	let ck = &tenc.ck;
	Ok(LayerW {
		ln1: ck.f32(&n("input_layernorm.weight"))?,
		q: ck.mat(&n("self_attn.q_proj.weight"), inner, d)?,
		k: ck.mat(&n("self_attn.k_proj.weight"), kv, d)?,
		v: ck.mat(&n("self_attn.v_proj.weight"), kv, d)?,
		nq: ck.f32(&n("self_attn.q_norm.weight"))?,
		nk: ck.f32(&n("self_attn.k_norm.weight"))?,
		o: ck.mat(&n("self_attn.o_proj.weight"), d, inner)?,
		ln2: ck.f32(&n("post_attention_layernorm.weight"))?,
		gate: ck.mat(&n("mlp.gate_proj.weight"), ff, d)?,
		up: ck.mat(&n("mlp.up_proj.weight"), ff, d)?,
		down: ck.mat(&n("mlp.down_proj.weight"), d, ff)?,
	})
}

/// Rotate-half RoPE of `v` `[s][heads * hd]` with `cos` / `sin` `[s][hd / 2]`.
fn rope(v: &mut [f32], heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
	let half = hd / 2;
	for (p, row) in v.chunks_exact_mut(heads * hd).enumerate() {
		let (c, s) = (
			&cos[p * half..(p + 1) * half],
			&sin[p * half..(p + 1) * half],
		);
		for r in row.chunks_exact_mut(hd) {
			for i in 0..half {
				let (a, b) = (r[i], r[i + half]);
				r[i] = a * c[i] - b * s[i];
				r[i + half] = b * c[i] + a * s[i];
			}
		}
	}
}

/// The `te` step: see the module docs.
pub fn build(
	src: &Path,
	out: &Path,
	accel: &Accel,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	let cfg = Config::load(src)?;
	let c = h3te::shape(src, h3te::LAYERS)?;
	let tc = &cfg.te;
	if (
		tc.dim,
		tc.heads * tc.head_dim,
		tc.kv_heads * tc.head_dim,
		tc.ffn,
	) != (c.d, c.inner(), c.kv, c.ff)
	{
		return Err("text encoder: the build's shape differs from the pipeline's".into());
	}
	let codes = h3te::codes_env()?;
	h3te::set_codes(out, codes)?;
	let tenc = TextEncoder::open(src, tc)?;
	let t0 = std::time::Instant::now();
	let mut seqs = sequences(src, &tenc, log)?;
	let rows: usize = seqs.iter().map(|s| s.x.len() / c.d).sum();
	log(&format!(
		"text encoder calibration: {} sequences, {rows} rows ({:.0} s)",
		seqs.len(),
		t0.elapsed().as_secs_f64()
	));
	let xin = seqs.iter().fold(0f32, |a, s| a.max(amax(&s.x)));
	if h3te::read_in(out).is_err() {
		h3te::write_in(out, xin)?;
	}
	let (d, inner, kv, ff, hd) = (c.d, c.inner(), c.kv, c.ff, c.hd);
	let (heads, kvh) = (c.heads, c.kv / hd);
	let scale = 1.0 / (hd as f32).sqrt();
	let eps = tc.eps;
	let use_gptq = npubuild::h3img::gptq_on();
	for l in 0..c.layers {
		let t = std::time::Instant::now();
		let done = h3te::layer_done(out, l);
		let w = layer_w(&tenc, &c, l)?;
		let mut g = (!done && use_gptq)
			.then(|| [Gram::new(d), Gram::new(inner), Gram::new(d), Gram::new(ff)]);
		let mut cal = [0f32; 6];
		for sq in seqs.iter_mut() {
			let s = sq.x.len() / d;
			let mut h = sq.x.clone();
			crate::ops::rms(&mut h, &w.ln1, eps, false);
			if !done {
				cal[0] = cal[0].max(rot_max(&h, d));
				if let Some(g) = g.as_mut() {
					g[0].add(&h);
				}
			}
			let mut q = npubuild::la::linear(&h, s, d, &w.q, inner);
			let mut k = npubuild::la::linear(&h, s, d, &w.k, kv);
			let v = npubuild::la::linear(&h, s, d, &w.v, kv);
			drop(h);
			crate::ops::rms(&mut q, &w.nq, eps, false);
			crate::ops::rms(&mut k, &w.nk, eps, false);
			rope(&mut q, heads, hd, &sq.cos, &sq.sin);
			rope(&mut k, kvh, hd, &sq.cos, &sq.sin);
			let mut a = vec![0f32; s * inner];
			quant::attn_causal(&q, &k, &v, s, 0, heads, kvh, hd, scale, &mut a);
			drop((q, k, v));
			if !done {
				cal[1] = cal[1].max(rot_max(&a, inner));
				if let Some(g) = g.as_mut() {
					g[1].add(&a);
				}
			}
			let o = npubuild::la::linear(&a, s, inner, &w.o, d);
			drop(a);
			sq.x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
			drop(o);
			let mut h = sq.x.clone();
			crate::ops::rms(&mut h, &w.ln2, eps, false);
			if !done {
				cal[2] = cal[2].max(rot_max(&h, d));
				if let Some(g) = g.as_mut() {
					g[2].add(&h);
				}
			}
			let gt = npubuild::la::linear(&h, s, d, &w.gate, ff);
			let mut hid = npubuild::la::linear(&h, s, d, &w.up, ff);
			drop(h);
			cal[4] = cal[4].max(amax(&gt));
			hid.iter_mut()
				.zip(&gt)
				.for_each(|(u, &g)| *u *= g / (1.0 + (-g).exp()));
			drop(gt);
			cal[5] = cal[5].max(amax(&hid));
			if let Some(g) = g.as_mut() {
				g[3].add(&hid);
			}
			let dn = npubuild::la::linear(&hid, s, ff, &w.down, d);
			drop(hid);
			sq.x.iter_mut().zip(&dn).for_each(|(x, o)| *x += o);
			cal[3] = cal[3].max(amax(&sq.x));
			if let Some(f) = sq.deep.get(l) {
				for (j, &p) in sq.img.iter().enumerate() {
					sq.x[p * d..(p + 1) * d]
						.iter_mut()
						.zip(&f[j * d..(j + 1) * d])
						.for_each(|(a, b)| *a += b);
				}
				cal[3] = cal[3].max(amax(&sq.x));
			}
		}
		if done {
			log(&format!(
				"text encoder layer {l}: done before, forward in {:.1} s",
				t.elapsed().as_secs_f64()
			));
			continue;
		}
		for r in &mut cal[..3] {
			*r *= HEADROOM;
		}
		let t_fwd = t.elapsed().as_secs_f64();
		let cp = if l == 0 {
			h3te::scale_in(out)?
		} else {
			h3te::scale_of(out, l - 1)?
		};
		let cl = cal[3] / npubuild::h3img::XS;
		let hess = g.map(|g| g.map(|g| g.full()));
		h3te::write_layer(out, l, &c, w, cal, hess, cp, cl, codes, accel)?;
		log(&format!(
			"text encoder layer {l}: ranges {:.2} / {:.2} / {:.2}, residual max {:.3e}, gate max {:.3e}, hidden max {:.3e} (forward {t_fwd:.1} s, all {:.1} s)",
			cal[0],
			cal[1],
			cal[2],
			cal[3],
			cal[4],
			cal[5],
			t.elapsed().as_secs_f64()
		));
	}
	h3te::write_scales(out, &c)
}
