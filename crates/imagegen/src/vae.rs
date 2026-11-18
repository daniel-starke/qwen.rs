//! The VAE decoder (`AutoencoderKLQwenImage21`: Wan-2.2's residual decoder,
//! one frame, so every causal 3-D conv is a 2-D one), f32 on the cores.
//!
//! Activations are pixel-major (`[h][w][c]`), so a 3x3 convolution is nine
//! GEMMs over a zero-padded copy: for tap `(dy, dx)` the input rows of output
//! pixels `(y, x)` are padded rows `(y + dy) (w + 2) + x + dx`, which is one
//! strided matrix if the output is computed over the padded width too (the
//! two extra columns per row are discarded).
//!
//! Decoder: `post_quant_conv` (1x1), `conv_in` (z -> 8 dim), mid block (res,
//! single-head attention, res), up blocks of 3 residual blocks each; blocks
//! with an upsampler double the resolution (nearest, then a 3x3 conv) and add
//! a channel-to-space shortcut of their input (`DupUp3D` on the first chunk:
//! with a temporal factor 2 only the second temporal half is kept); then
//! RMS norm, SiLU, `conv_out` to 4 channels (RGBA), clamped to [-1, 1].
//! The `upsample3d` blocks' `time_conv` is never used for one frame.
//!
//! Encoder (edits: the condition images' latents): `conv_in` (RGBA ->
//! base dim), down blocks of 2 residual blocks each; blocks with a
//! downsampler halve the resolution (zero pad right / bottom, 3x3 stride 2)
//! and add a space-to-channel average of their input (`AvgDown3D`: with a
//! temporal factor 2 the missing first frame is zeros, so half of each
//! group averages zeros); mid block, RMS norm, SiLU, `conv_out` to 2 z,
//! `quant_conv`; the latent is the distribution's mean (the first z
//! channels), normalized with `latents_mean` / `latents_std`. The
//! `downsample3d` blocks' `time_conv` is not used for the first frame.

use std::borrow::Cow;

use crate::config::VaeCfg;
use crate::npuvae::{NpuVae, Src};
use crate::ops::{self, Ptr};
use npubuild::la::{gemm_nt_raw, Acc};
use npubuild::st::Checkpoint;
use std::path::Path;

/// A pixel-major feature map.
#[derive(Clone)]
pub struct Map {
	pub h: usize,
	pub w: usize,
	pub c: usize,
	pub d: Vec<f32>,
}

pub struct Vae {
	pub cfg: VaeCfg,
	ck: Checkpoint,
}

struct Conv {
	cout: usize,
	cin: usize,
	k: usize,
	/// `[k k][cout][cin]` (tap-major).
	w: Vec<f32>,
	b: Vec<f32>,
}

impl Vae {
	pub fn open(dir: &Path, cfg: &VaeCfg) -> Result<Vae, String> {
		let ck = Checkpoint::open_dir(&dir.join("vae"))?;
		Ok(Vae {
			cfg: cfg.clone(),
			ck,
		})
	}

	fn conv(&self, name: &str) -> Result<Conv, String> {
		let sh = self.ck.shape(&format!("{name}.weight"))?.to_vec();
		if sh.len() != 4 || sh[2] != sh[3] {
			return Err(format!("{name}: conv weight shape {sh:?}"));
		}
		let (cout, cin, k) = (sh[0], sh[1], sh[2]);
		let raw = self.ck.f32(&format!("{name}.weight"))?;
		let mut w = vec![0f32; raw.len()];
		for o in 0..cout {
			for i in 0..cin {
				for t in 0..k * k {
					w[(t * cout + o) * cin + i] = raw[(o * cin + i) * k * k + t];
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

	fn gamma(&self, name: &str) -> Result<Vec<f32>, String> {
		self.ck.f32(&format!("{name}.gamma"))
	}

	/// Decode normalized latents `[h w][z]` (token-major, as the transformer
	/// produces them) into `[h 16][w 16][4]` in [-1, 1]; `tap(name, map)`
	/// sees each stage's output.
	pub fn decode(
		&self,
		lat: &[f32],
		h: usize,
		w: usize,
		tap: impl FnMut(&str, &Map),
	) -> Result<Map, String> {
		self.decode_with(lat, h, w, None, tap)
	}

	/// [`Self::decode`] with the 3x3 convolutions `npu` has on the NPU.
	pub fn decode_with(
		&self,
		lat: &[f32],
		h: usize,
		w: usize,
		mut npu: Option<&mut NpuVae>,
		mut tap: impl FnMut(&str, &Map),
	) -> Result<Map, String> {
		let npu = &mut npu;
		let c = &self.cfg;
		let z = c.z;
		let mut x = Map {
			h,
			w,
			c: z,
			d: lat.to_vec(),
		};
		for p in 0..h * w {
			for ch in 0..z {
				let v = &mut x.d[p * z + ch];
				*v = *v * c.std[ch] + c.mean[ch];
			}
		}
		x = conv_apply(&self.conv("post_quant_conv")?, &x);
		x = self.conv3(npu, "decoder.conv_in", &x, Pre::None, None)?;
		tap("conv_in", &x);
		x = self.resblock_with(npu, "decoder.mid_block.resnets.0", Cow::Owned(x))?;
		x = self.attention("decoder.mid_block.attentions.0", x)?;
		x = self.resblock_with(npu, "decoder.mid_block.resnets.1", Cow::Owned(x))?;
		tap("mid_block", &x);
		for i in 0..c.dim_mult.len() {
			let p = format!("decoder.up_blocks.{i}");
			let up = self.ck.has(&format!("{p}.upsampler.resample.1.weight"));
			// the shortcut of an upsampling block reads its input: kept, and
			// lent to the first resnet (not copied)
			let input: Option<Map>;
			let mut y = if up {
				input = Some(x);
				Cow::Borrowed(input.as_ref().unwrap())
			} else {
				input = None;
				Cow::Owned(x)
			};
			for j in 0..=c.res_blocks {
				y = Cow::Owned(self.resblock_with(npu, &format!("{p}.resnets.{j}"), y)?);
			}
			x = y.into_owned();
			if let Some(input) = &input {
				x = self.conv3(npu, &format!("{p}.upsampler.resample.1"), &x, Pre::Up, None)?;
				let ft = if self.ck.has(&format!("{p}.upsampler.time_conv.weight")) {
					2
				} else {
					1
				};
				dup_up_add(&mut x, input, ft);
			}
			tap(&format!("up_blocks{i}"), &x);
		}
		rms_ch(&mut x, &self.gamma("decoder.norm_out")?);
		ops::map(&mut x.d, ops::silu);
		x = conv_apply(&self.conv("decoder.conv_out")?, &x);
		tap("conv_out", &x);
		x.d.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
		Ok(x)
	}

	/// Encode an image `[h][w][in_ch]` in [-1, 1] (pixel-major) into
	/// normalized latents `[h / 16 w / 16][z]` (token-major, as the
	/// transformer takes them); `tap(name, map)` sees each stage's output.
	pub fn encode(&self, px: &Map, tap: impl FnMut(&str, &Map)) -> Result<Vec<f32>, String> {
		self.encode_with(px, None, tap)
	}

	/// [`Self::encode`] with the 3x3 stride-1 convolutions `npu` has on the
	/// NPU (the `norm_out` stage is then not tapped: it is fused into
	/// `conv_out`'s input).
	pub fn encode_with(
		&self,
		px: &Map,
		mut npu: Option<&mut NpuVae>,
		mut tap: impl FnMut(&str, &Map),
	) -> Result<Vec<f32>, String> {
		let npu = &mut npu;
		let c = &self.cfg;
		if px.c != c.in_ch || !px.h.is_multiple_of(c.scale) || !px.w.is_multiple_of(c.scale) {
			return Err(format!(
				"vae encoder: {}x{}x{} input (want {} channels, sides x{})",
				px.w, px.h, px.c, c.in_ch, c.scale
			));
		}
		let mut x = conv_apply(&self.conv("encoder.conv_in")?, px);
		tap("conv_in", &x);
		let nb = c.dim_mult.len();
		for i in 0..nb {
			let p = format!("encoder.down_blocks.{i}");
			// the block's input, read by its shortcut: lent to the first
			// resnet (not copied)
			let input = x;
			let mut y = Cow::Borrowed(&input);
			for j in 0..c.res_blocks {
				y = Cow::Owned(self.resblock_with(npu, &format!("{p}.resnets.{j}"), y)?);
			}
			x = y.into_owned();
			let down = self.ck.has(&format!("{p}.downsampler.resample.1.weight"));
			if down {
				x = conv_s2(&self.conv(&format!("{p}.downsampler.resample.1"))?, &x);
			}
			let ft = if i + 1 < nb && c.t_down.get(i) == Some(&true) {
				2
			} else {
				1
			};
			avg_down_add(&mut x, &input, ft, if down { 2 } else { 1 });
			tap(&format!("down_blocks{i}"), &x);
		}
		x = self.resblock_with(npu, "encoder.mid_block.resnets.0", Cow::Owned(x))?;
		x = self.attention("encoder.mid_block.attentions.0", x)?;
		x = self.resblock_with(npu, "encoder.mid_block.resnets.1", Cow::Owned(x))?;
		tap("mid_block", &x);
		if npu.as_deref().is_some_and(|n| n.has("encoder.conv_out")) {
			x = self.conv3(
				npu,
				"encoder.conv_out",
				&x,
				Pre::NormSilu("encoder.norm_out"),
				None,
			)?;
		} else {
			rms_ch(&mut x, &self.gamma("encoder.norm_out")?);
			tap("norm_out", &x);
			ops::map(&mut x.d, ops::silu);
			x = conv_apply(&self.conv("encoder.conv_out")?, &x);
		}
		tap("conv_out", &x);
		x = conv_apply(&self.conv("quant_conv")?, &x);
		let z = c.z;
		if x.c != 2 * z {
			return Err(format!(
				"vae encoder: {} output channels, want {}",
				x.c,
				2 * z
			));
		}
		let mut lat = vec![0f32; x.h * x.w * z];
		for p in 0..x.h * x.w {
			for ch in 0..z {
				lat[p * z + ch] = x.d[p * 2 * z + ch];
			}
		}
		tap(
			"mean",
			&Map {
				h: x.h,
				w: x.w,
				c: z,
				d: lat.clone(),
			},
		);
		for p in 0..x.h * x.w {
			for ch in 0..z {
				let v = &mut lat[p * z + ch];
				*v = (*v - c.mean[ch]) / c.std[ch];
			}
		}
		Ok(lat)
	}

	/// 3x3 conv `name` of `x` after `pre`, plus `add`: on the NPU when `npu`
	/// has it, else on the cores.
	fn conv3(
		&self,
		npu: &mut Option<&mut NpuVae>,
		name: &str,
		x: &Map,
		pre: Pre,
		add: Option<&Map>,
	) -> Result<Map, String> {
		if let Some(n) = npu.as_deref_mut().filter(|n| n.has(name)) {
			let g: Vec<f32>;
			let src = match pre {
				Pre::None => Src::Map(x),
				Pre::Up => Src::Up(x),
				Pre::NormSilu(gn) => {
					let sc = (x.c as f32).sqrt();
					g = self.gamma(gn)?.iter().map(|v| v * sc).collect();
					Src::NormSilu(x, &g)
				}
			};
			return n.conv(name, &src, add);
		}
		let cv = self.conv(name)?;
		let mut y = match pre {
			Pre::None => conv_apply(&cv, x),
			Pre::Up => conv_up2(&cv, x),
			Pre::NormSilu(gn) => {
				let mut h = x.clone();
				rms_ch(&mut h, &self.gamma(gn)?);
				ops::map(&mut h.d, ops::silu);
				conv_apply(&cv, &h)
			}
		};
		if let Some(a) = add {
			y.d.iter_mut().zip(&a.d).for_each(|(v, b)| *v += b);
		}
		Ok(y)
	}

	/// [`Self::resblock`] with its convolutions through [`Self::conv3`].
	/// `x` is only read on the NPU path, so it may be borrowed.
	fn resblock_with(
		&self,
		npu: &mut Option<&mut NpuVae>,
		p: &str,
		x: Cow<'_, Map>,
	) -> Result<Map, String> {
		if npu.is_none() {
			return self.resblock(p, x.into_owned());
		}
		let sm;
		let short = if self.ck.has(&format!("{p}.conv_shortcut.weight")) {
			sm = conv_apply(&self.conv(&format!("{p}.conv_shortcut"))?, &x);
			&sm
		} else {
			&x
		};
		let n1 = format!("{p}.norm1");
		let n2 = format!("{p}.norm2");
		let h = self.conv3(npu, &format!("{p}.conv1"), &x, Pre::NormSilu(&n1), None)?;
		self.conv3(
			npu,
			&format!("{p}.conv2"),
			&h,
			Pre::NormSilu(&n2),
			Some(short),
		)
	}

	fn resblock(&self, p: &str, x: Map) -> Result<Map, String> {
		let short = if self.ck.has(&format!("{p}.conv_shortcut.weight")) {
			conv_apply(&self.conv(&format!("{p}.conv_shortcut"))?, &x)
		} else {
			x.clone()
		};
		let mut h = x;
		rms_ch(&mut h, &self.gamma(&format!("{p}.norm1"))?);
		ops::map(&mut h.d, ops::silu);
		h = conv_apply(&self.conv(&format!("{p}.conv1"))?, &h);
		rms_ch(&mut h, &self.gamma(&format!("{p}.norm2"))?);
		ops::map(&mut h.d, ops::silu);
		h = conv_apply(&self.conv(&format!("{p}.conv2"))?, &h);
		h.d.iter_mut().zip(&short.d).for_each(|(a, b)| *a += b);
		Ok(h)
	}

	/// Single-head self-attention over all pixels, channels as the head.
	fn attention(&self, p: &str, x: Map) -> Result<Map, String> {
		let (n, c) = (x.h * x.w, x.c);
		let mut h = x.clone();
		rms_ch(&mut h, &self.gamma(&format!("{p}.norm"))?);
		let qkv = conv_apply(&self.conv(&format!("{p}.to_qkv"))?, &h);
		let (mut q, mut k, mut v) = (vec![0f32; n * c], vec![0f32; n * c], vec![0f32; n * c]);
		for i in 0..n {
			let r = &qkv.d[i * 3 * c..(i + 1) * 3 * c];
			q[i * c..(i + 1) * c].copy_from_slice(&r[..c]);
			k[i * c..(i + 1) * c].copy_from_slice(&r[c..2 * c]);
			v[i * c..(i + 1) * c].copy_from_slice(&r[2 * c..]);
		}
		let mut a = vec![0f32; n * c];
		quant::attn_block(&q, &k, &v, n, 0, 1, 1, c, 1.0 / (c as f32).sqrt(), &mut a);
		h.d = a;
		let mut o = conv_apply(&self.conv(&format!("{p}.proj"))?, &h);
		o.d.iter_mut().zip(&x.d).for_each(|(a, b)| *a += b);
		Ok(o)
	}
}

/// What precedes a convolution.
#[derive(Clone, Copy)]
enum Pre<'a> {
	None,
	/// RMS norm with gamma `{name}.gamma`, then SiLU.
	NormSilu(&'a str),
	/// Nearest-neighbor 2x upsampling.
	Up,
}

/// RMS norm over the channels of each pixel: `x / max(|x|, 1e-12) sqrt(c)
/// gamma` (`F.normalize`).
fn rms_ch(x: &mut Map, g: &[f32]) {
	let c = x.c;
	assert_eq!(g.len(), c);
	let sc = (c as f32).sqrt();
	ops::rows_mut(&mut x.d, c, |r| {
		let n = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt() as f32;
		let f = sc / n.max(1e-12);
		r.iter_mut().zip(g).for_each(|(v, &gg)| *v = *v * f * gg);
	});
}

fn conv_apply(cv: &Conv, x: &Map) -> Map {
	assert_eq!(x.c, cv.cin);
	let (h, w, ci, co) = (x.h, x.w, cv.cin, cv.cout);
	if cv.k == 1 {
		let mut d = ops::linear(&x.d, h * w, ci, &cv.w, co);
		ops::rows_mut(&mut d, co, |r| {
			r.iter_mut().zip(&cv.b).for_each(|(v, b)| *v += b)
		});
		return Map { h, w, c: co, d };
	}
	assert_eq!(cv.k, 3);
	conv3_rows(cv, h, w, &|y, r| {
		r.copy_from_slice(&x.d[y * w * ci..(y + 1) * w * ci])
	})
}

/// Bytes of a padded input band of [`conv3_rows`].
const BAND_BYTES: usize = 64 << 20;

/// 3x3 convolution (zero padding 1) of an `h x w x cin` map whose row `y`
/// `fill(y, row)` writes, by bands of rows: each band's padded input is
/// built from `fill`, so neither a padded copy of the whole map nor (for an
/// upsampling conv) the upsampled map exists. The output of a band is nine
/// GEMMs over its padded rows (computed over the padded width; the two
/// extra columns per row are discarded).
fn conv3_rows(cv: &Conv, h: usize, w: usize, fill: &(dyn Fn(usize, &mut [f32]) + Sync)) -> Map {
	let (ci, co) = (cv.cin, cv.cout);
	let pw = w + 2;
	let band = (BAND_BYTES / (pw * ci * 4)).clamp(1, h);
	let mut pad = vec![0f32; (band + 2) * pw * ci];
	let mut o = vec![0f32; ((band - 1) * pw + w) * co];
	let mut d = quant::uninit_vec(h * w * co);
	for y0 in (0..h).step_by(band) {
		let bh = band.min(h - y0);
		{
			let pp = Ptr(pad.as_mut_ptr());
			quant::par_for(bh + 2, move |r| {
				let pp = pp;
				// SAFETY: task r owns padded row r of the band
				let row = unsafe { std::slice::from_raw_parts_mut(pp.0.add(r * pw * ci), pw * ci) };
				row[..ci].fill(0.0);
				row[(w + 1) * ci..].fill(0.0);
				match (y0 + r).checked_sub(1).filter(|&y| y < h) {
					Some(y) => fill(y, &mut row[ci..(w + 1) * ci]),
					None => row[ci..(w + 1) * ci].fill(0.0),
				}
			});
		}
		let m = (bh - 1) * pw + w;
		for t in 0..9 {
			let (dy, dx) = (t / 3, t % 3);
			// SAFETY: rows (dy pw + dx) .. + m of the padded band exist (the
			// last is (bh + 2) pw - 1 at most); o holds m rows of co.
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
		let dp = Ptr(d.as_mut_ptr());
		let o = &o;
		quant::par_for(bh, move |yy| {
			let dp = dp;
			for xx in 0..w {
				let src = &o[(yy * pw + xx) * co..(yy * pw + xx + 1) * co];
				// SAFETY: task yy owns output row y0 + yy
				let dst = unsafe {
					std::slice::from_raw_parts_mut(dp.0.add(((y0 + yy) * w + xx) * co), co)
				};
				for ((d, &s), &b) in dst.iter_mut().zip(src).zip(&cv.b) {
					*d = s + b;
				}
			}
		});
	}
	Map { h, w, c: co, d }
}

/// A 3x3 convolution of the nearest-neighbor 2x upsampling of `x`, without
/// the upsampled map.
fn conv_up2(cv: &Conv, x: &Map) -> Map {
	assert_eq!(x.c, cv.cin);
	let (w, c) = (x.w, x.c);
	conv3_rows(cv, 2 * x.h, 2 * w, &|y, r| {
		let src = &x.d[(y / 2) * w * c..(y / 2 + 1) * w * c];
		for xx in 0..2 * w {
			r[xx * c..(xx + 1) * c].copy_from_slice(&src[(xx / 2) * c..(xx / 2 + 1) * c]);
		}
	})
}

/// 3x3 convolution with stride 2 after zero padding one column / row on the
/// right / bottom (`ZeroPad2d((0, 1, 0, 1))`): even sides halve.
fn conv_s2(cv: &Conv, x: &Map) -> Map {
	assert_eq!(x.c, cv.cin);
	assert_eq!(cv.k, 3);
	let (ci, co) = (cv.cin, cv.cout);
	let (oh, ow) = (x.h / 2, x.w / 2);
	let m = oh * ow;
	let mut o = vec![0f32; m * co];
	let mut g = quant::uninit_vec(m * ci);
	for t in 0..9 {
		let (dy, dx) = (t / 3, t % 3);
		let gp = Ptr(g.as_mut_ptr());
		quant::par_for(oh, move |y| {
			let gp = gp;
			for xx in 0..ow {
				let (sy, sx) = (2 * y + dy, 2 * xx + dx);
				// SAFETY: task y owns gathered rows y ow ..
				let dst =
					unsafe { std::slice::from_raw_parts_mut(gp.0.add((y * ow + xx) * ci), ci) };
				if sy < x.h && sx < x.w {
					dst.copy_from_slice(&x.d[(sy * x.w + sx) * ci..(sy * x.w + sx + 1) * ci]);
				} else {
					dst.fill(0.0);
				}
			}
		});
		// SAFETY: g holds m rows of ci, o m rows of co, the tap's weights co x ci
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

/// `x += AvgDown3D(input)` (one frame, padded in front to `ft` frames with
/// zeros): channel-major groups of `ft fs fs` values per input channel,
/// `in_c ft fs fs / out_c` of them averaged per output channel.
fn avg_down_add(x: &mut Map, input: &Map, ft: usize, fs: usize) {
	let (co, ci) = (x.c, input.c);
	let f = ft * fs * fs;
	assert_eq!(ci * f % co, 0);
	assert_eq!((input.h / fs, input.w / fs), (x.h, x.w));
	let gsz = ci * f / co;
	let (w, iw) = (x.w, input.w);
	let xp = Ptr(x.d.as_mut_ptr());
	quant::par_for(x.h, move |y| {
		let xp = xp;
		for xx in 0..w {
			// SAFETY: task y owns row y of x
			let dst = unsafe { std::slice::from_raw_parts_mut(xp.0.add((y * w + xx) * co), co) };
			for (o, d) in dst.iter_mut().enumerate() {
				let mut s = 0f32;
				for k in o * gsz..(o + 1) * gsz {
					let (c, t, a, b) = (k / f, (k / (fs * fs)) % ft, (k / fs) % fs, k % fs);
					if t == ft - 1 {
						s += input.d[((fs * y + a) * iw + fs * xx + b) * ci + c];
					}
				}
				*d += s / gsz as f32;
			}
		}
	});
}

/// `x += DupUp3D(input)` (first chunk): output channel `c` at `(2y + a, 2x
/// + b)` is input channel `(c ft 4 + (ft - 1) 4 + 2a + b) / repeats`.
fn dup_up_add(x: &mut Map, input: &Map, ft: usize) {
	let (co, ci) = (x.c, input.c);
	let f = ft * 4;
	assert_eq!(co * f % ci, 0);
	let rep = co * f / ci;
	let (w2, iw) = (x.w, input.w);
	let xp = Ptr(x.d.as_mut_ptr());
	quant::par_for(x.h, move |y| {
		let xp = xp;
		for xx in 0..w2 {
			let (a, b) = (y % 2, xx % 2);
			let src = &input.d[((y / 2) * iw + xx / 2) * ci..((y / 2) * iw + xx / 2 + 1) * ci];
			// SAFETY: task y owns row y of x
			let dst = unsafe { std::slice::from_raw_parts_mut(xp.0.add((y * w2 + xx) * co), co) };
			for (c, d) in dst.iter_mut().enumerate() {
				*d += src[(c * f + (ft - 1) * 4 + 2 * a + b) / rep];
			}
		}
	});
}

/// `[-1, 1]` RGBA map to 8-bit RGBA bytes (`(v / 2 + 0.5) 255`, rounded, as
/// diffusers' `VaeImageProcessor.postprocess`).
pub fn to_rgba8(x: &Map) -> Vec<u8> {
	assert_eq!(x.c, 4);
	x.d.iter()
		.map(|&v| (((v * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8)
		.collect()
}
