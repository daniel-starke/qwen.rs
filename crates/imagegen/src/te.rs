//! Text encoder: the Qwen3-VL text stack (`text_encoder/`), f32 on the
//! cores, weights converted from BF16 one layer at a time, shared by the
//! image models (each builds its own prompt: [`crate::qimage::te`],
//! [`crate::h3::te`]).
//!
//! Text-only positions are equal on all three MRoPE axes, so the interleaved
//! MRoPE is plain rotate-half RoPE at `theta = rope_theta`.
//!
//! An image's `<|image_pad|>` run is one token per 2x2 patch group,
//! embedded as the vision tower's output rows ([`VisionRows`]), at MRoPE
//! positions `(t, t + row, t + column)`, after which the text resumes at
//! `t + max(rows, columns)`; the deepstack features are added to those rows
//! after the first text layers (one merger per layer).

use crate::config::TeCfg;
use crate::ops::{self, Lin, WFmt};
use crate::wcache;
use npubuild::st::Checkpoint;
use std::path::Path;
use tokenizer::Tokenizer;

pub struct TextEncoder {
	pub cfg: TeCfg,
	pub ck: Checkpoint,
	pub tok: Tokenizer,
	pre: String,
	/// The text layers, quantized once ([`TextEncoder::load`]); `None` =
	/// the reference, which converts each layer from BF16 per use.
	resident: Option<Vec<Layer>>,
}

/// One text layer's weights.
struct Layer {
	ln1: Vec<f32>,
	q: Lin,
	k: Lin,
	v: Lin,
	nq: Vec<f32>,
	nk: Vec<f32>,
	o: Lin,
	ln2: Vec<f32>,
	gate: Lin,
	up: Lin,
	down: Lin,
}

impl Layer {
	fn save(&self, w: &mut wcache::Writer) -> Result<(), String> {
		w.f32s(&self.ln1)?;
		for l in [&self.q, &self.k, &self.v] {
			w.lin(l)?;
		}
		w.f32s(&self.nq)?;
		w.f32s(&self.nk)?;
		w.lin(&self.o)?;
		w.f32s(&self.ln2)?;
		for l in [&self.gate, &self.up, &self.down] {
			w.lin(l)?;
		}
		Ok(())
	}

	fn read(r: &mut wcache::Reader) -> Result<Layer, String> {
		Ok(Layer {
			ln1: r.f32s()?,
			q: r.lin()?,
			k: r.lin()?,
			v: r.lin()?,
			nq: r.f32s()?,
			nk: r.f32s()?,
			o: r.lin()?,
			ln2: r.f32s()?,
			gate: r.lin()?,
			up: r.lin()?,
			down: r.lin()?,
		})
	}

	fn bytes(&self) -> usize {
		[
			&self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
		]
		.iter()
		.map(|l| l.bytes())
		.sum()
	}
}

/// The condition images as the text encoder sees them: the vision tower's
/// rows of every image in order (`[tokens][dim]`), each deepstack merger's
/// rows likewise (one per text layer from 0), and each image's merged grid
/// (rows, columns).
pub struct VisionRows {
	pub embeds: Vec<f32>,
	pub deep: Vec<Vec<f32>>,
	pub grids: Vec<(usize, usize)>,
}

/// The MRoPE axis of rotary pair `i`: height for `i % 3 == 1`, width for
/// `i % 3 == 2`, each below `3 * sections[axis]`; time otherwise (the
/// interleaved layout of `mrope_section = [24, 20, 20]`).
fn mrope_axis(i: usize, sec: [usize; 3]) -> usize {
	match i % 3 {
		1 if i < 3 * sec[1] => 1,
		2 if i < 3 * sec[2] => 2,
		_ => 0,
	}
}

impl TextEncoder {
	pub fn open(dir: &Path, cfg: &TeCfg) -> Result<TextEncoder, String> {
		let ck = Checkpoint::open_dir(&dir.join("text_encoder"))?;
		let tok = Tokenizer::from_hf_json(&dir.join("processor/tokenizer.json"))?;
		let pre = "model.language_model.".to_string();
		if !ck.has(&format!("{pre}embed_tokens.weight")) {
			return Err("text_encoder: no model.language_model.embed_tokens".into());
		}
		Ok(TextEncoder {
			cfg: cfg.clone(),
			ck,
			tok,
			pre,
			resident: None,
		})
	}

	/// [`Self::open`] with every text layer's matmuls converted to `fmt`
	/// once and kept (`WFmt::F32`: nothing resident). `progress(layer)`
	/// after each.
	pub fn load(
		dir: &Path,
		cfg: &TeCfg,
		fmt: WFmt,
		mut progress: impl FnMut(usize),
	) -> Result<TextEncoder, String> {
		let mut t = Self::open(dir, cfg)?;
		if fmt != WFmt::F32 {
			let mut v = Vec::with_capacity(cfg.layers);
			for l in 0..cfg.layers {
				v.push(t.layer(l, fmt)?);
				progress(l);
			}
			t.resident = Some(v);
		}
		Ok(t)
	}

	/// [`Self::load`] through the weight cache file `file` (`None`: no
	/// cache): read when it was made from this checkpoint in `fmt`, else
	/// made after the load (a failed write is reported, not fatal).
	/// `progress(layer)` after each layer either way.
	pub fn load_cached(
		dir: &Path,
		cfg: &TeCfg,
		fmt: WFmt,
		file: Option<&Path>,
		mut progress: impl FnMut(usize),
	) -> Result<TextEncoder, String> {
		let Some(file) = file.filter(|_| fmt != WFmt::F32) else {
			return Self::load(dir, cfg, fmt, progress);
		};
		let key = wcache::key(
			&dir.join("text_encoder"),
			&format!("te;{fmt:?};layers {};rot {}", cfg.layers, ops::rot_on()),
		)?;
		if let Some(mut r) = wcache::Reader::open(file, &key) {
			let mut t = Self::open(dir, cfg)?;
			let mut v = Vec::with_capacity(cfg.layers);
			for l in 0..cfg.layers {
				v.push(Layer::read(&mut r)?);
				progress(l);
			}
			t.resident = Some(v);
			return Ok(t);
		}
		let t = Self::load(dir, cfg, fmt, progress)?;
		let save = || -> Result<(), String> {
			let mut w = wcache::Writer::create(file, &key)?;
			for l in t.resident.as_deref().unwrap_or_default() {
				l.save(&mut w)?;
			}
			w.finish()
		};
		if let Err(e) = save() {
			eprintln!("image: text encoder weights not cached: {e}");
		}
		Ok(t)
	}

	pub fn resident_bytes(&self) -> usize {
		self.resident
			.as_ref()
			.map_or(0, |v| v.iter().map(Layer::bytes).sum())
	}

	fn layer(&self, l: usize, fmt: WFmt) -> Result<Layer, String> {
		let c = &self.cfg;
		let (d, hd) = (c.dim, c.head_dim);
		let (nq, nk) = (c.heads * hd, c.kv_heads * hd);
		let n = |x: &str| format!("{}layers.{l}.{x}", self.pre);
		let lin = |x: &str, rows: usize, cols: usize| -> Result<Lin, String> {
			Ok(Lin::new(self.ck.mat(&n(x), rows, cols)?, rows, cols, fmt))
		};
		Ok(Layer {
			ln1: self.ck.f32(&n("input_layernorm.weight"))?,
			q: lin("self_attn.q_proj.weight", nq, d)?,
			k: lin("self_attn.k_proj.weight", nk, d)?,
			v: lin("self_attn.v_proj.weight", nk, d)?,
			nq: self.ck.f32(&n("self_attn.q_norm.weight"))?,
			nk: self.ck.f32(&n("self_attn.k_norm.weight"))?,
			o: lin("self_attn.o_proj.weight", d, nq)?,
			ln2: self.ck.f32(&n("post_attention_layernorm.weight"))?,
			gate: lin("mlp.gate_proj.weight", c.ffn, d)?,
			up: lin("mlp.up_proj.weight", c.ffn, d)?,
			down: lin("mlp.down_proj.weight", d, c.ffn)?,
		})
	}

	pub fn image_pad_id(&self) -> Result<u32, String> {
		self.tok
			.token_id("<|image_pad|>")
			.ok_or_else(|| "tokenizer: no <|image_pad|>".to_string())
	}

	/// MRoPE positions `[t, h, w]` of `ids`: text advances all three; an
	/// image's pad run (grid `rows x cols`, `grids` in order) lays out `(t,
	/// t + r, t + c)` row by row, then `t += max(rows, cols)`.
	pub fn positions(
		&self,
		ids: &[u32],
		grids: &[(usize, usize)],
	) -> Result<Vec<[i64; 3]>, String> {
		let pad = self.image_pad_id()?;
		let mut out = Vec::with_capacity(ids.len());
		let (mut t, mut i, mut g) = (0i64, 0usize, 0usize);
		while i < ids.len() {
			if ids[i] != pad {
				out.push([t; 3]);
				t += 1;
				i += 1;
				continue;
			}
			let &(rows, cols) = grids
				.get(g)
				.ok_or("text encoder: more image pad runs than images")?;
			let n = rows * cols;
			if ids.len() - i < n || ids[i..i + n].iter().any(|&x| x != pad) {
				return Err(format!(
					"text encoder: image {} pad run is not {rows} x {cols}",
					g + 1
				));
			}
			for r in 0..rows as i64 {
				for c in 0..cols as i64 {
					out.push([t, t + r, t + c]);
				}
			}
			t += rows.max(cols) as i64;
			i += n;
			g += 1;
		}
		if g != grids.len() {
			return Err(format!(
				"text encoder: {} images, {g} pad runs",
				grids.len()
			));
		}
		Ok(out)
	}

	/// The checkpoint's name of text-stack tensor `name` (`layers.0....`).
	pub fn tensor(&self, name: &str) -> String {
		format!("{}{name}", self.pre)
	}

	/// The stack's input rows of `ids` (`[s][dim]`: the embeddings, an
	/// image's `<|image_pad|>` run taking `vis`'s rows in order) and the
	/// rows of the pads.
	pub fn inputs(
		&self,
		ids: &[u32],
		vis: Option<&VisionRows>,
	) -> Result<(Vec<f32>, Vec<usize>), String> {
		let (s, d) = (ids.len(), self.cfg.dim);
		let pad = self.image_pad_id()?;
		let img_rows: Vec<usize> = (0..s).filter(|&p| ids[p] == pad).collect();
		let emb = self.tensor("embed_tokens.weight");
		let mut x = Vec::with_capacity(s * d);
		for &t in ids {
			x.extend(self.ck.rows(&emb, t as usize, t as usize + 1)?);
		}
		match vis {
			Some(v) => {
				if v.embeds.len() != img_rows.len() * d
					|| v.deep.iter().any(|f| f.len() != v.embeds.len())
				{
					return Err(format!(
						"text encoder: {} rows of vision features for {} pads",
						v.embeds.len() / d,
						img_rows.len()
					));
				}
				for (j, &p) in img_rows.iter().enumerate() {
					x[p * d..(p + 1) * d].copy_from_slice(&v.embeds[j * d..(j + 1) * d]);
				}
			}
			None if !img_rows.is_empty() => {
				return Err("text encoder: image pads in a prompt without images".into())
			}
			None => {}
		}
		Ok((x, img_rows))
	}

	/// The rotate-half RoPE's `cos` / `sin` `[s][head_dim / 2]` of the
	/// MRoPE positions `pos` (pair i at the position of its axis).
	pub fn rope_tables(&self, pos: &[[i64; 3]]) -> (Vec<f32>, Vec<f32>) {
		let c = &self.cfg;
		let (s, half) = (pos.len(), c.head_dim / 2);
		let inv: Vec<f32> = (0..half)
			.map(|i| 1.0 / (c.theta as f32).powf((2 * i) as f32 / c.head_dim as f32))
			.collect();
		let mut cos = vec![0f32; s * half];
		let mut sin = vec![0f32; s * half];
		for p in 0..s {
			for i in 0..half {
				let a = pos[p][mrope_axis(i, c.mrope)] as f32 * inv[i];
				cos[p * half + i] = a.cos();
				sin[p * half + i] = a.sin();
			}
		}
		(cos, sin)
	}

	/// The text stack over token ids `ids` (an image's `<|image_pad|>` run
	/// takes `vis`'s rows, in order): every row's output of the last layer
	/// of `cfg.layers`, before the final norm. `tap(layer, hidden)` sees each
	/// layer's output before its deepstack addition.
	pub fn encode_ids(
		&self,
		ids: &[u32],
		vis: Option<&VisionRows>,
		tap: &mut dyn FnMut(usize, &[f32]),
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let grids: &[(usize, usize)] = vis.map_or(&[], |v| &v.grids);
		let pos = self.positions(ids, grids)?;
		let (s, d, hd) = (ids.len(), c.dim, c.head_dim);
		let nq = c.heads * hd;
		let (mut x, img_rows) = self.inputs(ids, vis)?;
		let half = hd / 2;
		let (cos, sin) = self.rope_tables(&pos);
		let rope = |v: &mut [f32], heads: usize| {
			for p in 0..s {
				for h in 0..heads {
					let r = &mut v[(p * heads + h) * hd..(p * heads + h + 1) * hd];
					for i in 0..half {
						let (a, b) = (r[i], r[i + half]);
						let (cs, sn) = (cos[p * half + i], sin[p * half + i]);
						r[i] = a * cs - b * sn;
						r[i + half] = b * cs + a * sn;
					}
				}
			}
		};

		for l in 0..c.layers {
			let tmp;
			let w = match &self.resident {
				Some(v) => &v[l],
				None => {
					tmp = self.layer(l, WFmt::F32)?;
					&tmp
				}
			};
			let mut h = x.clone();
			ops::rms(&mut h, &w.ln1, c.eps, false);
			let mut qkv = ops::multi(&[&w.q, &w.k, &w.v], &h, s).into_iter();
			std::mem::drop(h);
			let (mut q, mut k, v) = (
				qkv.next().unwrap(),
				qkv.next().unwrap(),
				qkv.next().unwrap(),
			);
			ops::rms(&mut q, &w.nq, c.eps, false);
			ops::rms(&mut k, &w.nk, c.eps, false);
			rope(&mut q, c.heads);
			rope(&mut k, c.kv_heads);
			let mut a = vec![0f32; s * nq];
			let scale = 1.0 / (hd as f32).sqrt();
			quant::attn_causal(&q, &k, &v, s, 0, c.heads, c.kv_heads, hd, scale, &mut a);
			let o = w.o.apply(&a, s);
			x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);

			for c0 in (0..s).step_by(ops::MLP_CHUNK) {
				let r = ops::MLP_CHUNK.min(s - c0);
				let xc = &mut x[c0 * d..(c0 + r) * d];
				let mut h = xc.to_vec();
				ops::rms(&mut h, &w.ln2, c.eps, false);
				let mut gu = ops::multi(&[&w.gate, &w.up], &h, r).into_iter();
				let (g, u) = (gu.next().unwrap(), gu.next().unwrap());
				let o = ops::swiglu_down(&w.down, &g, u, r);
				xc.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
			}
			tap(l, &x);
			if let Some(f) = vis.and_then(|v| v.deep.get(l)) {
				for (j, &p) in img_rows.iter().enumerate() {
					x[p * d..(p + 1) * d]
						.iter_mut()
						.zip(&f[j * d..(j + 1) * d])
						.for_each(|(a, b)| *a += b);
				}
			}
		}
		Ok(x)
	}
}
