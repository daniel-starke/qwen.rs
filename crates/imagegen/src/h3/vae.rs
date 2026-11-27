//! The VAE (`AutoencoderKLMiniMaxH3Image`), f32 on the cores: the
//! MiniMax-H3 causal video CNN encoder run over one frame, and a 2-D ViT
//! decoder. Both work in overlapping 256-pixel tiles blended linearly
//! (on by default in the release: the released images are the blended
//! ones, so tiling is part of the model).
//!
//! Encoder, one frame: every causal 3x3x3 conv sees two zero frames and the
//! frame, so only its last temporal slice acts (a temporal stride changes
//! nothing); spatial padding is **reflect**, a stride-2 downsampler pads
//! right / bottom by one (reflect) first. GroupNorm(32) + SiLU before each
//! conv of a residual block; `conv_out` -> 48 moments, `quant_conv`; the
//! latent is a posterior sample `mean + exp(logvar / 2) * noise`.
//!
//! Decoder: `post_quant_conv` (1x1), one token per latent pixel
//! (`proj_in`), 4 learned registers and one zero token appended (rotary
//! position 0), 24 pre-norm blocks (RMSNorm, attention with q / k RMSNorm
//! without affine and a 2-axis RoPE over half of each head, SwiGLU MLP,
//! both residuals scaled per channel by `scale1` / `scale2`), LayerNorm,
//! `proj_out` to a 16x16x3 patch per token. Output is ImageNet-normalized
//! RGB.

use super::config::VaeCfg;
use crate::ops::{self, Ptr};
use npubuild::la::{gemm_nt_raw, Acc};
use npubuild::st::Checkpoint;
use std::path::Path;

/// Tile size and minimum overlap in pixels (`tile_sample_min_*`).
pub const TILE: usize = 256;
pub const TILE_OVERLAP: usize = 64;

/// A pixel-major feature map `[h][w][c]`.
#[derive(Clone)]
pub struct Map {
	pub h: usize,
	pub w: usize,
	pub c: usize,
	pub d: Vec<f32>,
}

struct Conv {
	cout: usize,
	cin: usize,
	k: usize,
	/// `[k k][cout][cin]` (tap-major).
	w: Vec<f32>,
	b: Vec<f32>,
}

/// One decoder block's weights: (weight, bias) pairs and vectors.
pub struct DecLayer {
	n1: Vec<f32>,
	q: (Vec<f32>, Vec<f32>),
	k: (Vec<f32>, Vec<f32>),
	v: (Vec<f32>, Vec<f32>),
	o: (Vec<f32>, Vec<f32>),
	s1: Vec<f32>,
	n2: Vec<f32>,
	f0: (Vec<f32>, Vec<f32>),
	f2: (Vec<f32>, Vec<f32>),
	s2: Vec<f32>,
}

/// The decoder's weights ([`Vae::dec_weights`]).
pub struct DecW {
	post_quant: Conv,
	proj_in: (Vec<f32>, Vec<f32>),
	regs: Vec<f32>,
	layers: Vec<DecLayer>,
	norm: (Vec<f32>, Vec<f32>),
	proj_out: (Vec<f32>, Vec<f32>),
}

pub struct Vae {
	pub cfg: VaeCfg,
	pub ck: Checkpoint,
}

/// `_split_tiles`: tile starts and the overlaps between consecutive tiles
/// over `len` pixels; the slack is spread round-robin over the overlaps in
/// steps of `ratio` so every boundary stays latent-aligned. One tile of
/// `len` when it fits.
pub fn split_tiles(
	len: usize,
	tile: usize,
	min_ov: usize,
	ratio: usize,
) -> (Vec<usize>, usize, Vec<usize>) {
	if tile >= len {
		return (vec![0], len, vec![]);
	}
	let mut n = len.div_ceil(tile);
	while tile * n < min_ov * (n - 1) + len {
		n += 1;
	}
	let mut ov = vec![min_ov; n - 1];
	let rem = tile * n - min_ov * (n - 1) - len;
	for i in 0..rem / ratio {
		ov[i % (n - 1)] += ratio;
	}
	let mut starts = vec![0];
	for o in &ov {
		starts.push(starts.last().unwrap() + tile - o);
	}
	(starts, tile, ov)
}

/// `_stitch_tiles` of `tiles[i][j]` (`[c][h][w]` channel-major, each
/// `th x tw`): every tile blends its top rows with the bottom of the
/// original tile above and its left columns with the right of the original
/// tile to the left (weights `p / e` for the lower / right one), then loses
/// its bottom / right overlap unless it is the last.
pub fn stitch(
	tiles: &[Vec<Vec<f32>>],
	c: usize,
	th: usize,
	tw: usize,
	hov: &[usize],
	wov: &[usize],
) -> (Vec<f32>, usize, usize) {
	let (ni, nj) = (tiles.len(), tiles[0].len());
	let crop_h = |i: usize| if i + 1 < ni { th - hov[i] } else { th };
	let crop_w = |j: usize| if j + 1 < nj { tw - wov[j] } else { tw };
	let oh: usize = (0..ni).map(crop_h).sum();
	let ow: usize = (0..nj).map(crop_w).sum();
	let mut out = vec![0f32; c * oh * ow];
	let mut y0 = 0;
	for i in 0..ni {
		let mut x0 = 0;
		for j in 0..nj {
			let mut t = tiles[i][j].clone();
			if i > 0 {
				let e = hov[i - 1].min(th);
				let a = &tiles[i - 1][j];
				for ch in 0..c {
					for p in 0..e {
						let (wa, wb) = (1.0 - p as f32 / e as f32, p as f32 / e as f32);
						for x in 0..tw {
							let ia = (ch * th + th - e + p) * tw + x;
							let ib = (ch * th + p) * tw + x;
							t[ib] = a[ia] * wa + t[ib] * wb;
						}
					}
				}
			}
			if j > 0 {
				let e = wov[j - 1].min(tw);
				let a = &tiles[i][j - 1];
				for ch in 0..c {
					for y in 0..th {
						for p in 0..e {
							let (wa, wb) = (1.0 - p as f32 / e as f32, p as f32 / e as f32);
							let ia = (ch * th + y) * tw + tw - e + p;
							let ib = (ch * th + y) * tw + p;
							t[ib] = a[ia] * wa + t[ib] * wb;
						}
					}
				}
			}
			let (ch_, cw_) = (crop_h(i), crop_w(j));
			for ch in 0..c {
				for y in 0..ch_ {
					let src = &t[(ch * th + y) * tw..(ch * th + y) * tw + cw_];
					out[(ch * oh + y0 + y) * ow + x0..][..cw_].copy_from_slice(src);
				}
			}
			x0 += cw_;
		}
		y0 += crop_h(i);
	}
	(out, oh, ow)
}

fn reflect(i: isize, n: usize) -> usize {
	let n = n as isize;
	let r = if i < 0 {
		-i
	} else if i >= n {
		2 * (n - 1) - i
	} else {
		i
	};
	r as usize
}

/// 3x3 convolution with reflect padding 1, by nine GEMMs over the padded
/// map computed over the padded width (two extra columns per row dropped).
fn conv3(cv: &Conv, x: &Map) -> Map {
	assert_eq!(x.c, cv.cin);
	let (h, w, ci, co) = (x.h, x.w, cv.cin, cv.cout);
	let pw = w + 2;
	let mut pad = quant::uninit_vec((h + 2) * pw * ci);
	{
		let pp = Ptr(pad.as_mut_ptr());
		quant::par_for(h + 2, move |r| {
			let pp = pp;
			// SAFETY: task r owns padded row r
			let row = unsafe { std::slice::from_raw_parts_mut(pp.0.add(r * pw * ci), pw * ci) };
			let y = reflect(r as isize - 1, h);
			for xx in 0..pw {
				let sx = reflect(xx as isize - 1, w);
				row[xx * ci..(xx + 1) * ci]
					.copy_from_slice(&x.d[(y * w + sx) * ci..(y * w + sx + 1) * ci]);
			}
		});
	}
	let m = (h - 1) * pw + w;
	let mut o = quant::uninit_vec(m * co);
	for t in 0..9 {
		let (dy, dx) = (t / 3, t % 3);
		// SAFETY: rows (dy pw + dx) .. + m of the padded map exist; o holds m rows
		unsafe {
			gemm_nt_raw(
				pad.as_ptr().add((dy * pw + dx) * ci),
				ci,
				m,
				cv.w.as_ptr().add(t * co * ci),
				ci,
				co,
				ci,
				o.as_mut_ptr(),
				co,
				if t == 0 { Acc::Set } else { Acc::Add },
			)
		};
	}
	let mut d = quant::uninit_vec(h * w * co);
	let dp = Ptr(d.as_mut_ptr());
	let o = &o;
	quant::par_for(h, move |y| {
		let dp = dp;
		for xx in 0..w {
			let src = &o[(y * pw + xx) * co..(y * pw + xx + 1) * co];
			// SAFETY: task y owns output row y
			let dst = unsafe { std::slice::from_raw_parts_mut(dp.0.add((y * w + xx) * co), co) };
			for ((dd, &s), &b) in dst.iter_mut().zip(src).zip(&cv.b) {
				*dd = s + b;
			}
		}
	});
	Map { h, w, c: co, d }
}

/// 3x3 stride-2 convolution after a reflect pad of one column / row on the
/// right / bottom: output `ceil(h / 2) x ceil(w / 2)`.
fn conv3_s2(cv: &Conv, x: &Map) -> Map {
	assert_eq!(x.c, cv.cin);
	let (ci, co) = (cv.cin, cv.cout);
	let (ph, pw) = (x.h + 1, x.w + 1);
	let (oh, ow) = ((ph - 3) / 2 + 1, (pw - 3) / 2 + 1);
	let m = oh * ow;
	let mut o = quant::uninit_vec(m * co);
	let mut g = quant::uninit_vec(m * ci);
	for t in 0..9 {
		let (dy, dx) = (t / 3, t % 3);
		let gp = Ptr(g.as_mut_ptr());
		quant::par_for(oh, move |y| {
			let gp = gp;
			for xx in 0..ow {
				let sy = reflect((2 * y + dy) as isize, x.h);
				let sx = reflect((2 * xx + dx) as isize, x.w);
				// SAFETY: task y owns gathered rows y ow ..
				let dst =
					unsafe { std::slice::from_raw_parts_mut(gp.0.add((y * ow + xx) * ci), ci) };
				dst.copy_from_slice(&x.d[(sy * x.w + sx) * ci..(sy * x.w + sx + 1) * ci]);
			}
		});
		// SAFETY: g holds m rows of ci, o m rows of co
		unsafe {
			gemm_nt_raw(
				g.as_ptr(),
				ci,
				m,
				cv.w.as_ptr().add(t * co * ci),
				ci,
				co,
				ci,
				o.as_mut_ptr(),
				co,
				if t == 0 { Acc::Set } else { Acc::Add },
			)
		};
	}
	ops::rows_mut(&mut o, co, |r| {
		r.iter_mut().zip(&cv.b).for_each(|(v, b)| *v += b)
	});
	Map {
		h: oh,
		w: ow,
		c: co,
		d: o,
	}
}

fn conv1(cv: &Conv, x: &Map) -> Map {
	assert_eq!((cv.k, x.c), (1, cv.cin));
	let mut d = ops::linear(&x.d, x.h * x.w, cv.cin, &cv.w, cv.cout);
	ops::rows_mut(&mut d, cv.cout, |r| {
		r.iter_mut().zip(&cv.b).for_each(|(v, b)| *v += b)
	});
	Map {
		h: x.h,
		w: x.w,
		c: cv.cout,
		d,
	}
}

/// GroupNorm over all pixels of each channel group, affine, then SiLU when
/// `act`.
fn group_norm(x: &mut Map, groups: usize, eps: f32, g: &[f32], b: &[f32], act: bool) {
	let (c, n) = (x.c, x.h * x.w);
	let cg = c / groups;
	let mut stat = vec![(0f64, 0f64); groups];
	for p in 0..n {
		let row = &x.d[p * c..(p + 1) * c];
		for (gi, s) in stat.iter_mut().enumerate() {
			for &v in &row[gi * cg..(gi + 1) * cg] {
				s.0 += v as f64;
				s.1 += v as f64 * v as f64;
			}
		}
	}
	let cnt = (n * cg) as f64;
	let ms: Vec<(f32, f32)> = stat
		.iter()
		.map(|&(s, q)| {
			let m = s / cnt;
			let var = (q / cnt - m * m).max(0.0);
			(m as f32, 1.0 / (var as f32 + eps).sqrt())
		})
		.collect();
	ops::rows_mut(&mut x.d, c, |r| {
		for (ch, v) in r.iter_mut().enumerate() {
			let (m, s) = ms[ch / cg];
			let y = (*v - m) * s * g[ch] + b[ch];
			*v = if act { ops::silu(y) } else { y };
		}
	});
}

impl Vae {
	pub fn open(dir: &Path, cfg: &VaeCfg) -> Result<Vae, String> {
		Ok(Vae {
			cfg: cfg.clone(),
			ck: Checkpoint::open_dir(&dir.join("vae_diffusers"))?,
		})
	}

	fn v(&self, name: &str) -> Result<Vec<f32>, String> {
		self.ck.f32(name)
	}

	/// A (causal 3-D or 2-D) conv's weights as a 2-D conv: the last
	/// temporal slice (`[cout][cin][kt][k][k]` -> tap-major).
	fn conv(&self, name: &str) -> Result<Conv, String> {
		let sh = self.ck.shape(&format!("{name}.weight"))?.to_vec();
		let (cout, cin, kt, k) = match sh[..] {
			[o, i, t, a, b] if a == b => (o, i, t, a),
			[o, i, a, b] if a == b => (o, i, 1, a),
			_ => return Err(format!("{name}: conv weight shape {sh:?}")),
		};
		let raw = self.v(&format!("{name}.weight"))?;
		let mut w = vec![0f32; cout * cin * k * k];
		for o in 0..cout {
			for i in 0..cin {
				for t in 0..k * k {
					w[(t * cout + o) * cin + i] = raw[((o * cin + i) * kt + kt - 1) * k * k + t];
				}
			}
		}
		Ok(Conv {
			cout,
			cin,
			k,
			w,
			b: self.ck.vec(&format!("{name}.bias"), cout)?,
		})
	}

	fn gn(&self, x: &mut Map, name: &str, act: bool) -> Result<(), String> {
		let (g, b) = (
			self.v(&format!("{name}.weight"))?,
			self.v(&format!("{name}.bias"))?,
		);
		group_norm(x, self.cfg.groups, self.cfg.norm_eps, &g, &b, act);
		Ok(())
	}

	fn resblock(&self, p: &str, x: Map) -> Result<Map, String> {
		let mut h = x.clone();
		self.gn(&mut h, &format!("{p}.norm1"), true)?;
		let h = conv3(&self.conv(&format!("{p}.conv1"))?, &h);
		let mut h = h;
		self.gn(&mut h, &format!("{p}.norm2"), true)?;
		let mut h = conv3(&self.conv(&format!("{p}.conv2"))?, &h);
		let res = if self.ck.has(&format!("{p}.conv_shortcut.weight")) {
			conv1(&self.conv(&format!("{p}.conv_shortcut"))?, &x)
		} else {
			x
		};
		h.d.iter_mut().zip(&res.d).for_each(|(a, b)| *a += b);
		Ok(h)
	}

	/// The encoder over one tile of ImageNet-normalized RGB (`[h][w][3]`):
	/// the 48 moments `[h / 16][w / 16][48]`. `tap(name, map)` sees each
	/// stage's output.
	pub fn encode_tile(&self, px: &Map, tap: &mut dyn FnMut(&str, &Map)) -> Result<Map, String> {
		let c = &self.cfg;
		let mut x = conv3(&self.conv("encoder.conv_in")?, px);
		tap("conv_in", &x);
		for (j, (&sd, &td)) in c.spatial_down.iter().zip(&c.temporal_down).enumerate() {
			for m in 0..c.layers_per_block {
				x = self.resblock(&format!("encoder.down_blocks.{j}.resnets.{m}"), x)?;
			}
			if sd * td > 1 {
				let cv = self.conv(&format!("encoder.down_blocks.{j}.downsamplers.0.conv"))?;
				if sd != 2 {
					return Err(format!(
						"vae: down block {j}: only spatial stride 2 is implemented"
					));
				}
				x = conv3_s2(&cv, &x);
			}
			tap(&format!("down{j}"), &x);
		}
		self.gn(&mut x, "encoder.norm_out", true)?;
		let x = conv3(&self.conv("encoder.conv_out")?, &x);
		tap("conv_out", &x);
		Ok(conv1(&self.conv("quant_conv")?, &x))
	}

	/// The moments of a whole image (`[3][h][w]` ImageNet-normalized,
	/// channel-major), tiled and blended: `[48][h / 16][w / 16]`.
	pub fn encode_moments(
		&self,
		px: &[f32],
		h: usize,
		w: usize,
		mut tap: impl FnMut(&str, &Map),
	) -> Result<(Vec<f32>, usize, usize), String> {
		let r = self.cfg.ratio();
		let (ys, th, hov) = split_tiles(h, TILE, TILE_OVERLAP, r);
		let (xs, tw, wov) = split_tiles(w, TILE, TILE_OVERLAP, r);
		let mut first = true;
		let mut rows = Vec::new();
		for &y0 in &ys {
			let mut row = Vec::new();
			for &x0 in &xs {
				let (ty, tx) = (th.min(h - y0), tw.min(w - x0));
				let mut m = Map {
					h: ty,
					w: tx,
					c: 3,
					d: vec![0f32; ty * tx * 3],
				};
				for y in 0..ty {
					for x in 0..tx {
						for ch in 0..3 {
							m.d[(y * tx + x) * 3 + ch] = px[(ch * h + y0 + y) * w + x0 + x];
						}
					}
				}
				let mut noop = |_: &str, _: &Map| {};
				let o = self.encode_tile(&m, if first { &mut tap } else { &mut noop })?;
				first = false;
				row.push(to_chw(&o));
			}
			rows.push(row);
		}
		let hl: Vec<usize> = hov.iter().map(|o| o / r).collect();
		let wl: Vec<usize> = wov.iter().map(|o| o / r).collect();
		Ok(stitch(&rows, 2 * self.cfg.latent, th / r, tw / r, &hl, &wl))
	}

	/// The decoder's weights, read once per decode.
	pub fn dec_weights(&self) -> Result<DecW, String> {
		let c = &self.cfg;
		let layers = (0..c.dec_layers)
			.map(|l| {
				let p = |x: &str| format!("decoder.transformer_blocks.{l}.{x}");
				let lin = |x: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
					Ok((
						self.v(&p(&format!("{x}.weight")))?,
						self.v(&p(&format!("{x}.bias")))?,
					))
				};
				Ok(DecLayer {
					n1: self.v(&p("norm1.weight"))?,
					q: lin("attn.to_q")?,
					k: lin("attn.to_k")?,
					v: lin("attn.to_v")?,
					o: lin("attn.to_out.0")?,
					s1: self.v(&p("scale1"))?,
					n2: self.v(&p("norm2.weight"))?,
					f0: lin("ff.net.0.proj")?,
					f2: lin("ff.net.2")?,
					s2: self.v(&p("scale2"))?,
				})
			})
			.collect::<Result<_, String>>()?;
		Ok(DecW {
			post_quant: self.conv("post_quant_conv")?,
			proj_in: (
				self.v("decoder.proj_in.weight")?,
				self.v("decoder.proj_in.bias")?,
			),
			regs: self.v("decoder.register_tokens")?,
			layers,
			norm: (
				self.v("decoder.norm_out.weight")?,
				self.v("decoder.norm_out.bias")?,
			),
			proj_out: (
				self.v("decoder.proj_out.weight")?,
				self.v("decoder.proj_out.bias")?,
			),
		})
	}

	/// What [`Self::tile_tokens`] needs of the decoder's weights (no blocks).
	pub fn dec_weights_in(&self) -> Result<DecW, String> {
		Ok(DecW {
			post_quant: self.conv("post_quant_conv")?,
			proj_in: (
				self.v("decoder.proj_in.weight")?,
				self.v("decoder.proj_in.bias")?,
			),
			regs: self.v("decoder.register_tokens")?,
			layers: Vec::new(),
			norm: (Vec::new(), Vec::new()),
			proj_out: (Vec::new(), Vec::new()),
		})
	}

	/// A tile's token rows before the blocks (`[n + registers + 1][d]`):
	/// `post_quant_conv`, `proj_in`, the registers, the zero token. `tap`
	/// sees `post_quant` and `proj_in`.
	pub fn tile_tokens(&self, w: &DecW, z: &Map, tap: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
		let c = &self.cfg;
		let d = c.dec_dim();
		let z = conv1(&w.post_quant, z);
		tap("post_quant", &to_chw(&z));
		let n = z.h * z.w;
		let s = n + c.dec_registers + 1;
		let mut x = Vec::with_capacity(s * d);
		x.extend(super::dit::linear_b(
			&z.d,
			n,
			c.latent,
			&w.proj_in.0,
			&w.proj_in.1,
		));
		tap("proj_in", &x);
		x.extend_from_slice(&w.regs);
		x.resize(s * d, 0.0);
		x
	}

	/// The `[3][16 h][16 w]` pixels of a tile's `proj_out` rows (`[h w][3 16
	/// 16]`): token (y, x) -> pixels (y 16 + py, x 16 + px).
	pub fn tile_pixels(&self, o: &[f32], h: usize, w: usize) -> Vec<f32> {
		let ps = self.cfg.ratio();
		let (oh, ow) = (h * ps, w * ps);
		let mut img = vec![0f32; 3 * oh * ow];
		for y in 0..h {
			for xx in 0..w {
				let t = &o[(y * w + xx) * 3 * ps * ps..];
				for ch in 0..3 {
					for py in 0..ps {
						for px in 0..ps {
							img[(ch * oh + y * ps + py) * ow + xx * ps + px] =
								t[(ch * ps + py) * ps + px];
						}
					}
				}
			}
		}
		img
	}

	/// The decoder over one tile of latents (`[h][w][latent]`, denormalized):
	/// `[3][16 h][16 w]` ImageNet-normalized RGB. `tap(name, rows)` sees each
	/// stage's token rows.
	pub fn decode_tile(
		&self,
		z: &Map,
		tap: &mut dyn FnMut(&str, &[f32]),
	) -> Result<Vec<f32>, String> {
		let w = self.dec_weights()?;
		Ok(self.decode_tile_w(&w, z, tap))
	}

	/// [`Self::decode_tile`] with the weights `dw`.
	pub fn decode_tile_w(&self, dw: &DecW, z: &Map, tap: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
		let c = &self.cfg;
		let (d, nh, hd) = (c.dec_dim(), c.dec_heads, c.dec_head_dim);
		let (h, w) = (z.h, z.w);
		let mut x = self.tile_tokens(dw, z, tap);
		let n = h * w;
		let s = n + c.dec_registers + 1;

		// rotary: (h, w) in [-1, 1) at pixel centers, registers and the zero token at 0
		let rd = (hd as f64 * c.dec_rope_ratio) as usize;
		let nf = rd / 4;
		let inv: Vec<f32> = (0..nf)
			.map(|i| 1.0 / (c.dec_rope_theta as f32).powf((i * 4) as f32 / rd as f32))
			.collect();
		let half = rd / 2;
		let mut cos = vec![1f32; s * half];
		let mut sin = vec![0f32; s * half];
		for y in 0..h {
			for xx in 0..w {
				let p = [
					2.0 * ((y as f32 + 0.5) / h as f32) - 1.0,
					2.0 * ((xx as f32 + 0.5) / w as f32) - 1.0,
				];
				let r = y * w + xx;
				for ax in 0..2 {
					for (i, &f) in inv.iter().enumerate() {
						let a = 2.0 * std::f32::consts::PI * p[ax] * f;
						cos[r * half + ax * nf + i] = a.cos();
						sin[r * half + ax * nf + i] = a.sin();
					}
				}
			}
		}
		let rope = super::dit::Rope { cos, sin, half };

		for (l, ly) in dw.layers.iter().enumerate() {
			let mut hn = x.clone();
			ops::rms(&mut hn, &ly.n1, c.dec_eps, false);
			let mut q = super::dit::linear_b(&hn, s, d, &ly.q.0, &ly.q.1);
			let mut k = super::dit::linear_b(&hn, s, d, &ly.k.0, &ly.k.1);
			let v = super::dit::linear_b(&hn, s, d, &ly.v.0, &ly.v.1);
			let ones = vec![1f32; hd];
			ops::rms(&mut q, &ones, c.dec_eps, false);
			ops::rms(&mut k, &ones, c.dec_eps, false);
			rope.apply(&mut q, nh, hd);
			rope.apply(&mut k, nh, hd);
			let mut a = vec![0f32; s * d];
			quant::attn_block(
				&q,
				&k,
				&v,
				s,
				0,
				nh,
				nh,
				hd,
				1.0 / (hd as f32).sqrt(),
				&mut a,
			);
			let o = super::dit::linear_b(&a, s, d, &ly.o.0, &ly.o.1);
			ops::rows_mut_idx(&mut x, d, |r, row| {
				for ((xv, &ov), &sc) in row.iter_mut().zip(&o[r * d..(r + 1) * d]).zip(&ly.s1) {
					*xv += ov * sc;
				}
			});
			let mut hn = x.clone();
			ops::rms(&mut hn, &ly.n2, c.dec_eps, false);
			let f = ly.f0.1.len() / 2;
			let gu = super::dit::linear_b(&hn, s, d, &ly.f0.0, &ly.f0.1);
			let mut act = vec![0f32; s * f];
			ops::rows_mut_idx(&mut act, f, |r, row| {
				let src = &gu[r * 2 * f..(r + 1) * 2 * f];
				for i in 0..f {
					row[i] = src[i] * ops::silu(src[f + i]);
				}
			});
			let o = super::dit::linear_b(&act, s, f, &ly.f2.0, &ly.f2.1);
			ops::rows_mut_idx(&mut x, d, |r, row| {
				for ((xv, &ov), &sc) in row.iter_mut().zip(&o[r * d..(r + 1) * d]).zip(&ly.s2) {
					*xv += ov * sc;
				}
			});
			tap(&format!("block{l}"), &x);
		}
		// LayerNorm with affine
		let (g, b) = (&dw.norm.0, &dw.norm.1);
		let mut hn = vec![0f32; n * d];
		ops::rows_mut_idx(&mut hn, d, |r, row| {
			let src = &x[r * d..(r + 1) * d];
			let mean = src.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
			let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
			let inv = 1.0 / (var as f32 + c.dec_eps).sqrt();
			for i in 0..d {
				row[i] = (src[i] - mean as f32) * inv * g[i] + b[i];
			}
		});
		tap("norm_out", &hn);
		let o = super::dit::linear_b(&hn, n, d, &dw.proj_out.0, &dw.proj_out.1);
		let img = self.tile_pixels(&o, h, w);
		tap("out", &img);
		img
	}

	/// The latent tiles of denormalized latents `[latent][h][w]`: (tiles
	/// row-major as `[lh][lw][latent]` maps, tiles per row, tile size in
	/// pixels, overlaps).
	#[allow(clippy::type_complexity)]
	pub fn tiles(
		&self,
		z: &[f32],
		h: usize,
		w: usize,
	) -> (Vec<Map>, usize, (usize, usize), (Vec<usize>, Vec<usize>)) {
		let c = &self.cfg;
		let r = c.ratio();
		let (ys, th, hov) = split_tiles(h * r, TILE, TILE_OVERLAP, r);
		let (xs, tw, wov) = split_tiles(w * r, TILE, TILE_OVERLAP, r);
		let mut out = Vec::with_capacity(ys.len() * xs.len());
		for &y0 in &ys {
			for &x0 in &xs {
				let (ly, lx, lh, lw) = (y0 / r, x0 / r, th / r, tw / r);
				let mut m = Map {
					h: lh,
					w: lw,
					c: c.latent,
					d: vec![0f32; lh * lw * c.latent],
				};
				for y in 0..lh {
					for x in 0..lw {
						for ch in 0..c.latent {
							m.d[(y * lw + x) * c.latent + ch] = z[(ch * h + ly + y) * w + lx + x];
						}
					}
				}
				out.push(m);
			}
		}
		(out, xs.len(), (th, tw), (hov, wov))
	}

	/// Blend decoded tiles (row-major, `per_row` a row) into the image.
	pub fn blend(
		imgs: Vec<Vec<f32>>,
		per_row: usize,
		size: (usize, usize),
		ov: &(Vec<usize>, Vec<usize>),
	) -> (Vec<f32>, usize, usize) {
		let mut rows = Vec::new();
		let mut it = imgs.into_iter();
		loop {
			let row: Vec<Vec<f32>> = it.by_ref().take(per_row).collect();
			if row.is_empty() {
				break;
			}
			rows.push(row);
		}
		stitch(&rows, 3, size.0, size.1, &ov.0, &ov.1)
	}

	/// Decode denormalized latents `[latent][h][w]` (tiled, blended):
	/// `[3][16 h][16 w]` ImageNet-normalized RGB. `tap` sees the first tile's
	/// stages.
	pub fn decode(
		&self,
		z: &[f32],
		h: usize,
		w: usize,
		mut tap: impl FnMut(&str, &[f32]),
	) -> Result<(Vec<f32>, usize, usize), String> {
		let dw = self.dec_weights()?;
		let (tiles, per_row, size, ov) = self.tiles(z, h, w);
		let mut imgs = Vec::with_capacity(tiles.len());
		for (i, m) in tiles.iter().enumerate() {
			let mut noop = |_: &str, _: &[f32]| {};
			imgs.push(self.decode_tile_w(&dw, m, if i == 0 { &mut tap } else { &mut noop }));
		}
		Ok(Self::blend(imgs, per_row, size, &ov))
	}
}

/// `[h][w][c]` -> `[c][h][w]`.
pub fn to_chw(m: &Map) -> Vec<f32> {
	let mut o = vec![0f32; m.d.len()];
	for p in 0..m.h * m.w {
		for ch in 0..m.c {
			o[ch * m.h * m.w + p] = m.d[p * m.c + ch];
		}
	}
	o
}

/// ImageNet-normalized RGB `[3][h][w]` -> RGB bytes `[h][w][3]`, as the
/// pipeline: `clamp(x std + mean, 0, 1) * 255`, truncated.
pub fn to_rgb8(img: &[f32], h: usize, w: usize) -> Vec<u8> {
	let mut out = vec![0u8; h * w * 3];
	for ch in 0..3 {
		let (m, s) = (super::config::PIXEL_MEAN[ch], super::config::PIXEL_STD[ch]);
		for p in 0..h * w {
			let v = (img[ch * h * w + p] * s + m).clamp(0.0, 1.0);
			out[p * 3 + ch] = (v * 255.0) as u8;
		}
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tiles_cover_exactly() {
		// 2048 px: 11 tiles of 256, overlaps 64 + 16 on the first eight
		let (s, t, o) = split_tiles(2048, 256, 64, 16);
		assert_eq!((s.len(), t), (11, 256));
		assert_eq!(o[..8], [80; 8]);
		assert_eq!(o[8..], [64, 64]);
		assert_eq!(s.last().unwrap() + t, 2048);
		assert_eq!(split_tiles(256, 256, 64, 16), (vec![0], 256, vec![]));
		assert_eq!(split_tiles(200, 256, 64, 16), (vec![0], 200, vec![]));
	}

	#[test]
	fn stitch_blends_linearly() {
		// two 1-channel 1x4 tiles side by side, overlap 2
		let a = vec![1.0f32; 4];
		let b = vec![3.0f32; 4];
		let (o, h, w) = stitch(&[vec![a, b]], 1, 1, 4, &[], &[2]);
		assert_eq!((h, w), (1, 6));
		assert_eq!(o, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
	}
}
