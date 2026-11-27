//! The transformer (`QwenImage21Transformer2DModel`), f32 on the cores,
//! weights converted from BF16 one layer at a time.
//!
//! Sequence ([`layout`]): the prompt's rows (`txt_in` of the text encoder's
//! output), with each condition image's latent tokens (`img_in`, raster
//! order) in place of its slots (one text-encoder slot = a 2x2 group of
//! latent tokens), then the target image's `h * w` latent tokens. The
//! prefix (text and condition images) modulates from `t = 0` and sees only
//! what is before it (text causal, an image block also all of itself), so
//! its per-layer K / V do not depend on the step: [`Prefix`] computes them
//! once and every step runs the target rows only, attending to `[prefix,
//! target]`.

use super::config::DitCfg;
use crate::ops::{self, Lin, WFmt};
use npubuild::st::Checkpoint;
use std::path::Path;

/// `QWEN_IMAGE_TIMING=1`: seconds per block phase (qkv, norm + rope,
/// attention, out, mlp), summed and printed per forward.
static PH: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

fn timing() -> bool {
	static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*T.get_or_init(|| std::env::var("QWEN_IMAGE_TIMING").is_ok_and(|v| v == "1"))
}

fn lap(i: usize, t: &mut std::time::Instant) {
	if timing() {
		PH[i].fetch_add(
			t.elapsed().as_nanos() as u64,
			std::sync::atomic::Ordering::Relaxed,
		);
		*t = std::time::Instant::now();
	}
}

/// Calibration of the NPU backend's activation ranges: while set, every
/// block records the largest magnitude of its rotated QKV, out and
/// gate / up inputs (`npubuild::qimage::Act::Static`), per block.
pub static CAL: std::sync::Mutex<Option<Vec<[f32; 3]>>> = std::sync::Mutex::new(None);

/// The largest magnitude of the rows of `x` (`k` wide) after the NPU
/// graphs' rotation.
fn rot_max(x: &[f32], k: usize) -> f32 {
	let mut r = x.to_vec();
	let sg = npubuild::gptq::signs(k);
	npubuild::gptq::rotate_rows_b(&mut r, k, &sg, npubuild::qimage::ROT);
	r.iter().fold(0f32, |m, v| m.max(v.abs()))
}

fn cal_add(l: usize, i: usize, x: &[f32], k: usize) {
	if CAL.lock().unwrap().is_none() {
		return;
	}
	let m = rot_max(x, k);
	if let Some(v) = CAL.lock().unwrap().as_mut() {
		if v.len() <= l {
			v.resize(l + 1, [0.0; 3]);
		}
		v[l][i] = v[l][i].max(m);
	}
}

pub struct Dit {
	pub cfg: DitCfg,
	ck: Checkpoint,
	/// The 32 blocks' weights, quantized once ([`Dit::load`]); `None` = the
	/// reference, which converts each block from BF16 per use.
	resident: Option<Vec<BlockW>>,
}

/// Per-layer K / V of the prefix rows (post norm and RoPE), `[len][dim]`.
pub struct Prefix {
	pub len: usize,
	/// The prefix's attention runs `(start, end, text)`: a text run is
	/// causal, an image block sees all of itself; both see what precedes.
	pub segments: Vec<(usize, usize, bool)>,
	pub k: Vec<Vec<f32>>,
	pub v: Vec<Vec<f32>>,
}

impl Prefix {
	pub fn new(l: &Layout) -> Prefix {
		Prefix {
			len: l.prefix,
			segments: l.segments.clone(),
			k: Vec::new(),
			v: Vec::new(),
		}
	}
}

/// Where a prefix row comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
	/// Row `r` of the text encoder's output.
	Text(usize),
	/// Latent token `t` of condition image `j`.
	Image(usize, usize),
}

/// The joint sequence (diffusers' `QwenImage21Rope` and
/// `_qwenimage21_prefix_segments`).
pub struct Layout {
	/// RoPE ids `(frame, h, w)` of every row, prefix then target.
	pub ids: Vec<[i64; 3]>,
	/// The prefix rows' sources.
	pub rows: Vec<Src>,
	/// The prefix's attention runs `(start, end, text)`.
	pub segments: Vec<(usize, usize, bool)>,
	pub prefix: usize,
}

/// Lay out the sequence of a text-encoder output whose rows are image slots
/// where `slot` is true, for condition images of `refs` latent tokens `(h,
/// w)` each (in order) and a target of `target`. Text advances the frame on
/// all three ids; an image block holds the frame where it starts, lays out
/// `(h, w)` centered on 0 (`-(h - h / 2) .. h / 2`, its own grid), and the
/// frame then advances by `max(h, w)`.
pub fn layout(
	slot: &[bool],
	refs: &[(usize, usize)],
	target: (usize, usize),
) -> Result<Layout, String> {
	fn grid(l: &mut Layout, pos: i64, h: usize, w: usize) {
		let (h0, w0) = ((h - h / 2) as i64, (w - w / 2) as i64);
		for y in 0..h as i64 {
			for x in 0..w as i64 {
				l.ids.push([pos, y - h0, x - w0]);
			}
		}
	}
	fn text(l: &mut Layout, slot: &[bool], r: &mut usize, pos: &mut i64) {
		let start = l.rows.len();
		while *r < slot.len() && !slot[*r] {
			l.ids.push([*pos; 3]);
			l.rows.push(Src::Text(*r));
			*pos += 1;
			*r += 1;
		}
		if l.rows.len() > start {
			l.segments.push((start, l.rows.len(), true));
		}
	}
	let mut l = Layout {
		ids: Vec::new(),
		rows: Vec::new(),
		segments: Vec::new(),
		prefix: 0,
	};
	let (mut pos, mut r) = (0i64, 0usize);
	for (j, &(h, w)) in refs.iter().enumerate() {
		text(&mut l, slot, &mut r, &mut pos);
		let n = slot[r..].iter().take_while(|&&s| s).count();
		if n * 4 != h * w {
			return Err(format!(
				"condition image {}: {n} slots for {h}x{w} latent tokens",
				j + 1
			));
		}
		let start = l.rows.len();
		grid(&mut l, pos, h, w);
		l.rows.extend((0..h * w).map(|t| Src::Image(j, t)));
		l.segments.push((start, l.rows.len(), false));
		pos += h.max(w) as i64;
		r += n;
	}
	text(&mut l, slot, &mut r, &mut pos);
	if r < slot.len() {
		return Err("more image slots than condition images".into());
	}
	l.prefix = l.rows.len();
	grid(&mut l, pos, target.0, target.1);
	Ok(l)
}

/// RoPE ids of a text-to-image sequence: text `(p, p, p)` for `p < txt`,
/// then the target grid `(txt, y, x)` centered on 0 (`y` in `-(h - h / 2) ..
/// h / 2`).
pub fn t2i_ids(txt: usize, h: usize, w: usize) -> Vec<[i64; 3]> {
	layout(&vec![false; txt], &[], (h, w))
		.expect("a text-only layout")
		.ids
}

/// One step's modulation: the target row (step `t`) and the prefix row
/// (`t = 0`), each `[scale1, gate1, scale2, gate2]` with the gates `tanh`'d,
/// and the output norm's scale for the target row.
pub struct Modulation {
	pub target: Vec<f32>,
	pub prefix: Vec<f32>,
	pub out_scale: Vec<f32>,
}

/// `(cos, sin)` per position and rotary pair (`head_dim / 2` pairs: the
/// frame axis' pairs, then height's, then width's), adjacent-pair layout.
pub struct Rope {
	pub cos: Vec<f32>,
	pub sin: Vec<f32>,
	pairs: usize,
}

impl Rope {
	/// `ids[p] = (frame, h, w)` for each position.
	pub fn new(cfg: &DitCfg, ids: &[[i64; 3]]) -> Rope {
		let pairs = cfg.head_dim / 2;
		let mut inv = Vec::with_capacity(pairs);
		for &dim in &cfg.axes {
			for i in (0..dim).step_by(2) {
				inv.push(1.0f32 / 10000f32.powf(i as f32 / dim as f32));
			}
		}
		assert_eq!(inv.len(), pairs);
		let mut cos = vec![0f32; ids.len() * pairs];
		let mut sin = vec![0f32; ids.len() * pairs];
		for (p, id) in ids.iter().enumerate() {
			let mut j = 0;
			for (ax, &dim) in cfg.axes.iter().enumerate() {
				for _ in 0..dim / 2 {
					let a = id[ax] as f32 * inv[j];
					cos[p * pairs + j] = a.cos();
					sin[p * pairs + j] = a.sin();
					j += 1;
				}
			}
		}
		Rope { cos, sin, pairs }
	}

	/// Rotate every head of rows `[n][heads * head_dim]` at positions
	/// `p0 ..`.
	fn apply(&self, x: &mut [f32], heads: usize, p0: usize) {
		let (pairs, hd) = (self.pairs, 2 * self.pairs);
		let n = x.len() / (heads * hd);
		let px = ops::Ptr(x.as_mut_ptr());
		quant::par_for(n, move |r| {
			let px = px;
			// SAFETY: task r owns row r
			let row =
				unsafe { std::slice::from_raw_parts_mut(px.0.add(r * heads * hd), heads * hd) };
			let (c, s) = (
				&self.cos[(p0 + r) * pairs..(p0 + r + 1) * pairs],
				&self.sin[(p0 + r) * pairs..(p0 + r + 1) * pairs],
			);
			for h in 0..heads {
				let v = &mut row[h * hd..(h + 1) * hd];
				for i in 0..pairs {
					let (a, b) = (v[2 * i], v[2 * i + 1]);
					v[2 * i] = a * c[i] - b * s[i];
					v[2 * i + 1] = a * s[i] + b * c[i];
				}
			}
		});
	}
}

/// Which keys a block's rows attend to.
enum Attn<'a> {
	/// The prefix rows themselves, by runs.
	Prefix(&'a [(usize, usize, bool)]),
	/// Target rows: the prefix's keys (`kv`) and every target row.
	Target(Option<(&'a [f32], &'a [f32])>),
}

impl Dit {
	pub fn open(dir: &Path, cfg: &DitCfg) -> Result<Dit, String> {
		let ck = Checkpoint::open_dir(&dir.join("transformer"))?;
		if !ck.has("transformer_blocks.0.img_mlp.gate_layer.weight") {
			return Err("transformer: unexpected tensor layout (no img_mlp.gate_layer)".into());
		}
		Ok(Dit {
			cfg: cfg.clone(),
			ck,
			resident: None,
		})
	}

	/// [`Self::open`] with every block's matmuls converted to `fmt` once
	/// and kept (`WFmt::F32`: nothing resident, the reference).
	/// `progress(layer)` after each block.
	pub fn load(
		dir: &Path,
		cfg: &DitCfg,
		fmt: WFmt,
		mut progress: impl FnMut(usize),
	) -> Result<Dit, String> {
		let mut d = Self::open(dir, cfg)?;
		if fmt != WFmt::F32 {
			let mut b = Vec::with_capacity(cfg.layers);
			for l in 0..cfg.layers {
				b.push(d.load_block(l, fmt)?);
				progress(l);
			}
			d.resident = Some(b);
		}
		Ok(d)
	}

	/// Bytes of the resident weights.
	pub fn resident_bytes(&self) -> usize {
		self.resident.as_ref().map_or(0, |b| {
			b.iter()
				.map(|w| {
					[&w.q, &w.k, &w.v, &w.o, &w.gate, &w.proj, &w.out]
						.iter()
						.map(|l| l.bytes())
						.sum::<usize>()
				})
				.sum()
		})
	}

	fn w(&self, name: &str) -> Result<Vec<f32>, String> {
		self.ck.f32(name)
	}

	/// `txt_in`: zero-centered RMSNorm -> Linear -> GELU(tanh) -> Linear.
	pub fn txt_in(&self, te: &[f32]) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let (k, d) = (c.ctx_dim, c.dim());
		let s = te.len() / k;
		let mut x = te.to_vec();
		ops::rms(&mut x, &self.w("txt_in.text_norm.weight")?, c.eps, true);
		let mut h = ops::linear(&x, s, k, &self.w("txt_in.in_layer.weight")?, d);
		ops::map(&mut h, ops::gelu_tanh);
		Ok(ops::linear(
			&h,
			s,
			d,
			&self.w("txt_in.out_layer.weight")?,
			d,
		))
	}

	/// `img_in` on token-major latents `[n][in_ch]`.
	pub fn img_in(&self, lat: &[f32]) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		Ok(ops::linear(
			lat,
			lat.len() / c.in_ch,
			c.in_ch,
			&self.w("img_in.weight")?,
			c.dim(),
		))
	}

	/// The prefix rows of `layout`: `txt_in` of the text encoder's rows
	/// (`te`, `[n][ctx_dim]`), `img_in` of the condition images' latents
	/// (`refs[j]`, `[h w][in_ch]` token-major).
	pub fn prefix_rows(
		&self,
		te: &[f32],
		layout: &Layout,
		refs: &[Vec<f32>],
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let (k, d) = (c.ctx_dim, c.dim());
		let mut tx = Vec::new();
		for s in &layout.rows {
			if let Src::Text(r) = *s {
				tx.extend_from_slice(
					te.get(r * k..(r + 1) * k)
						.ok_or("dit: text row out of range")?,
				);
			}
		}
		let t = self.txt_in(&tx)?;
		let im: Vec<Vec<f32>> = refs
			.iter()
			.map(|r| self.img_in(r))
			.collect::<Result<_, _>>()?;
		let mut out = Vec::with_capacity(layout.prefix * d);
		let mut ti = 0;
		for s in &layout.rows {
			match *s {
				Src::Text(_) => {
					out.extend_from_slice(&t[ti * d..(ti + 1) * d]);
					ti += 1;
				}
				Src::Image(j, tok) => out.extend_from_slice(
					im.get(j)
						.and_then(|m| m.get(tok * d..(tok + 1) * d))
						.ok_or("dit: condition latents shorter than the layout")?,
				),
			}
		}
		Ok(out)
	}

	/// Timestep embedding rows for `ts` (sinusoidal, `cos` half first, of
	/// `1000 t`) through `linear_1 -> SiLU -> linear_2`.
	pub fn temb(&self, ts: &[f32]) -> Result<Vec<f32>, String> {
		Ok(self.temb_with(
			ts,
			&self.w("time_text_embed.timestep_embedder.linear_1.weight")?,
			&self.w("time_text_embed.timestep_embedder.linear_2.weight")?,
		))
	}

	/// [`Self::temb`] with `linear_1` / `linear_2` given.
	fn temb_with(&self, ts: &[f32], l1: &[f32], l2: &[f32]) -> Vec<f32> {
		let d = self.cfg.dim();
		const TD: usize = 256;
		let half = TD / 2;
		let mut e = vec![0f32; ts.len() * TD];
		for (r, &t) in ts.iter().enumerate() {
			let t = 1000.0f32 * t;
			for i in 0..half {
				let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
				let a = t * f;
				e[r * TD + i] = a.cos();
				e[r * TD + half + i] = a.sin();
			}
		}
		let mut h = ops::linear(&e, ts.len(), TD, l1, d);
		ops::map(&mut h, ops::silu);
		ops::linear(&h, ts.len(), d, l2, d)
	}

	/// The f32 weights of the per-step pieces outside the blocks: the
	/// timestep embedding, the modulation and `norm_out`'s scale (~370 MB,
	/// from ~190 MB of BF16).
	fn step_weights(&self) -> Result<StepW, String> {
		Ok(StepW {
			l1: self.w("time_text_embed.timestep_embedder.linear_1.weight")?,
			l2: self.w("time_text_embed.timestep_embedder.linear_2.weight")?,
			m: self.w("modulation.1.weight")?,
			out: self.w("norm_out.linear.weight")?,
		})
	}

	/// The step's modulation from its timestep (the prefix row is `t = 0`),
	/// with the timestep embedding and the raw modulation rows (checks).
	pub fn modulation(&self, t: f32) -> Result<(Modulation, Vec<f32>, Vec<f32>), String> {
		Ok(self.modulation_with(t, &self.step_weights()?))
	}

	/// [`Self::modulation`] of every timestep in `ts`, the weights read
	/// and widened once (per step they were ~1 GB of reads and writes);
	/// the same arithmetic per step.
	pub fn modulations(&self, ts: &[f32]) -> Result<Vec<Modulation>, String> {
		let w = self.step_weights()?;
		Ok(ts.iter().map(|&t| self.modulation_with(t, &w).0).collect())
	}

	fn modulation_with(&self, t: f32, w: &StepW) -> (Modulation, Vec<f32>, Vec<f32>) {
		let d = self.cfg.dim();
		let temb = self.temb_with(&[t, 0.0], &w.l1, &w.l2);
		let mut a = temb.clone();
		ops::map(&mut a, ops::silu);
		let mut m = ops::linear(&a, 2, d, &w.m, 4 * d);
		let raw = m.clone();
		for r in 0..2 {
			for part in [1, 3] {
				for v in &mut m[r * 4 * d + part * d..r * 4 * d + (part + 1) * d] {
					*v = v.tanh();
				}
			}
		}
		let out_scale = ops::linear(&a[..d], 1, d, &w.out, d);
		(
			Modulation {
				target: m[..4 * d].to_vec(),
				prefix: m[4 * d..].to_vec(),
				out_scale,
			},
			temb,
			raw,
		)
	}

	/// One block over `x` (`n` rows at sequence positions `p0 ..`), with
	/// modulation row `m` and the keys of `attn`. The rows' own K / V are
	/// returned.
	#[allow(clippy::too_many_arguments)]
	fn block(
		&self,
		l: usize,
		x: &mut [f32],
		m: &[f32],
		rope: &Rope,
		p0: usize,
		attn: Attn,
		w: &BlockW,
	) -> (Vec<f32>, Vec<f32>) {
		let c = &self.cfg;
		let (d, hd, nh) = (c.dim(), c.head_dim, c.heads);
		let n = x.len() / d;
		let mut tt = std::time::Instant::now();
		let h = ops::ln_scale(x, &m[..d], c.eps);
		cal_add(l, 0, &h, d);
		let mut qkv = ops::multi(&[&w.q, &w.k, &w.v], &h, n).into_iter();
		drop(h);
		let (mut q, mut k, v) = (
			qkv.next().unwrap(),
			qkv.next().unwrap(),
			qkv.next().unwrap(),
		);
		lap(0, &mut tt);
		ops::rms(&mut q, &w.nq, c.eps, false);
		ops::rms(&mut k, &w.nk, c.eps, false);
		rope.apply(&mut q, nh, p0);
		rope.apply(&mut k, nh, p0);
		lap(1, &mut tt);
		let mut a = vec![0f32; n * d];
		let scale = 1.0 / (hd as f32).sqrt();
		match attn {
			Attn::Prefix(segs) => {
				for &(s0, s1, text) in segs {
					let (qs, as_) = (&q[s0 * d..s1 * d], &mut a[s0 * d..s1 * d]);
					let (ks, vs) = (&k[..s1 * d], &v[..s1 * d]);
					if text {
						quant::attn_causal(qs, ks, vs, s1 - s0, s0, nh, nh, hd, scale, as_);
					} else {
						quant::attn_block(qs, ks, vs, s1 - s0, s0, nh, nh, hd, scale, as_);
					}
				}
			}
			Attn::Target(kv_pre) => {
				let (kk, vv);
				let (ka, va): (&[f32], &[f32]) = match kv_pre {
					Some((kp, vp)) => {
						kk = [kp, &k[..]].concat();
						vv = [vp, &v[..]].concat();
						(&kk, &vv)
					}
					None => (&k, &v),
				};
				let t0 = ka.len() / d - n;
				quant::attn_block(&q, ka, va, n, t0, nh, nh, hd, scale, &mut a);
			}
		}
		drop(q);
		lap(2, &mut tt);
		cal_add(l, 1, &a, d);
		let o = w.o.apply(&a, n);
		drop(a);
		ops::add_gated(x, &o, &m[d..2 * d]);
		drop(o);
		lap(3, &mut tt);

		// the MLP by row chunks: rows are independent
		for c0 in (0..n).step_by(ops::MLP_CHUNK) {
			let r = ops::MLP_CHUNK.min(n - c0);
			let xc = &mut x[c0 * d..(c0 + r) * d];
			let h = ops::ln_scale(xc, &m[2 * d..3 * d], c.eps);
			cal_add(l, 2, &h, d);
			let mut gu = ops::multi(&[&w.gate, &w.proj], &h, r).into_iter();
			let (g, u) = (gu.next().unwrap(), gu.next().unwrap());
			let o = ops::swiglu_down(&w.out, &g, u, r);
			ops::add_gated(xc, &o, &m[3 * d..4 * d]);
		}
		lap(4, &mut tt);
		(k, v)
	}

	fn load_block(&self, l: usize, fmt: WFmt) -> Result<BlockW, String> {
		let p = |n: &str| format!("transformer_blocks.{l}.{n}");
		let d = self.cfg.dim();
		let f = self.cfg.mlp_ratio * d;
		let lin = |n: &str, rows: usize, cols: usize| -> Result<Lin, String> {
			Ok(Lin::new(self.ck.mat(&p(n), rows, cols)?, rows, cols, fmt))
		};
		Ok(BlockW {
			q: lin("attn.to_q.weight", d, d)?,
			k: lin("attn.to_k.weight", d, d)?,
			v: lin("attn.to_v.weight", d, d)?,
			o: lin("attn.to_out.0.weight", d, d)?,
			nq: self.w(&p("attn.norm_q.weight"))?,
			nk: self.w(&p("attn.norm_k.weight"))?,
			gate: lin("img_mlp.gate_layer.weight", f, d)?,
			proj: lin("img_mlp.proj.weight", f, d)?,
			out: lin("img_mlp.out.weight", d, f)?,
		})
	}

	/// The velocity for target latents `lat` (`[h w][in_ch]`) at modulation
	/// `m`. The first call (`pre.k` empty) also runs the prefix rows `txt`
	/// ([`Self::prefix_rows`]) and fills `pre`; `tap(layer, prefix_rows,
	/// target_rows)` sees each block's output.
	pub fn forward(
		&self,
		lat: &[f32],
		txt: &mut Option<Vec<f32>>,
		pre: &mut Prefix,
		rope: &Rope,
		m: &Modulation,
		mut tap: impl FnMut(usize, Option<&[f32]>, &[f32]),
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let d = c.dim();
		let mut x = self.img_in(lat)?;
		let fill = pre.k.is_empty();
		for l in 0..c.layers {
			let tmp;
			let w = match &self.resident {
				Some(b) => &b[l],
				None => {
					tmp = self.load_block(l, WFmt::F32)?;
					&tmp
				}
			};
			if fill {
				let t = txt
					.as_mut()
					.ok_or("dit: no prefix rows for the first step")?;
				let (k, v) = self.block(l, t, &m.prefix, rope, 0, Attn::Prefix(&pre.segments), w);
				pre.k.push(k);
				pre.v.push(v);
			}
			let kv = (pre.len > 0).then(|| (&pre.k[l][..], &pre.v[l][..]));
			self.block(l, &mut x, &m.target, rope, pre.len, Attn::Target(kv), w);
			tap(l, if fill { txt.as_deref() } else { None }, &x);
		}
		if fill {
			*txt = None;
		}
		if timing() {
			let s: Vec<String> = ["qkv", "norm+rope", "attn", "out", "mlp"]
				.iter()
				.zip(&PH)
				.map(|(n, v)| {
					format!(
						"{n} {:.2}",
						v.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
					)
				})
				.collect();
			eprintln!("[dit] {} rows: {} s", x.len() / d, s.join(", "));
		}
		self.out(&x, m)
	}

	/// The velocity from the target rows' final residual `x` (`[n][dim]`):
	/// `norm_out` (LayerNorm * (1 + scale)), `proj_out`.
	pub fn out(&self, x: &[f32], m: &Modulation) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let d = c.dim();
		let h = ops::ln_scale(x, &m.out_scale, c.eps);
		Ok(ops::linear(
			&h,
			x.len() / d,
			d,
			&self.w("proj_out.weight")?,
			c.out_ch,
		))
	}
}

/// [`Dit::step_weights`].
struct StepW {
	l1: Vec<f32>,
	l2: Vec<f32>,
	m: Vec<f32>,
	out: Vec<f32>,
}

struct BlockW {
	q: Lin,
	k: Lin,
	v: Lin,
	o: Lin,
	nq: Vec<f32>,
	nk: Vec<f32>,
	gate: Lin,
	proj: Lin,
	out: Lin,
}
