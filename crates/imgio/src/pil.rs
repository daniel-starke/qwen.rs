//! Pillow's 8-bit resampling and alpha handling, bit for bit: what
//! diffusers' `VaeImageProcessor` does to an image generator's condition
//! image (`Image.resize(..., LANCZOS)` on RGBA, which Pillow runs on the
//! premultiplied `RGBa` form) and the pipeline's alpha-over-white copy for
//! the vision encoder (`Image.paste(img, mask=alpha)`).
//!
//! Unlike [`crate::resize`] (float taps, no intermediate rounding: close to
//! Pillow, kept for the vision path whose tests pin it), this reproduces
//! Pillow's fixed-point arithmetic: coefficients in `f64`, normalized, then
//! scaled to 22-bit integers (`PRECISION_BITS`), a horizontal pass stored
//! back to 8 bits, then the vertical pass (`libImaging/Resample.c`).

const PRECISION_BITS: u32 = 32 - 8 - 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
	/// `Image.BICUBIC` (Keys, a = -0.5), support 2.
	Bicubic,
	/// `Image.LANCZOS` (a = 3), support 3.
	Lanczos,
}

impl Filter {
	fn support(self) -> f64 {
		match self {
			Filter::Bicubic => 2.0,
			Filter::Lanczos => 3.0,
		}
	}

	fn eval(self, x: f64) -> f64 {
		match self {
			Filter::Bicubic => {
				const A: f64 = -0.5;
				let x = x.abs();
				if x < 1.0 {
					((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
				} else if x < 2.0 {
					(((x - 5.0) * x + 8.0) * x - 4.0) * A
				} else {
					0.0
				}
			}
			Filter::Lanczos => {
				let sinc = |x: f64| {
					if x == 0.0 {
						1.0
					} else {
						let x = x * std::f64::consts::PI;
						x.sin() / x
					}
				};
				if (-3.0..3.0).contains(&x) {
					sinc(x) * sinc(x / 3.0)
				} else {
					0.0
				}
			}
		}
	}
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel the
/// first source index and its integer taps.
fn coeffs(in_len: usize, out_len: usize, f: Filter) -> Vec<(usize, Vec<i32>)> {
	let scale = in_len as f64 / out_len as f64;
	let fscale = scale.max(1.0);
	let support = f.support() * fscale;
	let ss = 1.0 / fscale;
	(0..out_len)
		.map(|xx| {
			let center = (xx as f64 + 0.5) * scale;
			let xmin = ((center - support + 0.5) as i64).max(0) as usize;
			let xmax = ((center + support + 0.5) as i64).min(in_len as i64) as usize;
			let n = xmax.saturating_sub(xmin);
			let mut k: Vec<f64> = (0..n)
				.map(|x| f.eval((x as f64 + xmin as f64 - center + 0.5) * ss))
				.collect();
			let ww: f64 = k.iter().sum();
			if ww != 0.0 {
				k.iter_mut().for_each(|v| *v /= ww);
			}
			let one = (1u64 << PRECISION_BITS) as f64;
			let ki = k
				.iter()
				.map(|&v| {
					if v < 0.0 {
						(-0.5 + v * one) as i32
					} else {
						(0.5 + v * one) as i32
					}
				})
				.collect();
			(xmin, ki)
		})
		.collect()
}

#[inline]
fn clip8(v: i32) -> u8 {
	(v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resample interleaved 8-bit pixels of `ch` channels from `sw x sh` to
/// `dw x dh` as Pillow's `Image.resize` does on a mode without alpha
/// handling (`RGB`, `RGBa`, `L`). A same-size call is a copy.
pub fn resample(
	src: &[u8],
	ch: usize,
	sw: usize,
	sh: usize,
	dw: usize,
	dh: usize,
	f: Filter,
) -> Vec<u8> {
	assert_eq!(src.len(), sw * sh * ch);
	assert!(dw > 0 && dh > 0);
	if sw == dw && sh == dh {
		return src.to_vec();
	}
	let half = 1i32 << (PRECISION_BITS - 1);
	let vc = coeffs(sh, dh, f);
	// rows of the source the vertical pass reads
	let (y0, y1) = (vc[0].0, vc[dh - 1].0 + vc[dh - 1].1.len());
	let tmp: Vec<u8>;
	let (mid, mid_w, row0): (&[u8], usize, usize) = if sw != dw {
		let hc = coeffs(sw, dw, f);
		let mut t = vec![0u8; (y1 - y0) * dw * ch];
		let mut acc = vec![0i32; ch];
		for y in y0..y1 {
			let row = &src[y * sw * ch..(y + 1) * sw * ch];
			for (x, (xmin, k)) in hc.iter().enumerate() {
				acc.iter_mut().for_each(|a| *a = half);
				for (i, &kk) in k.iter().enumerate() {
					let p = &row[(xmin + i) * ch..(xmin + i + 1) * ch];
					for c in 0..ch {
						acc[c] = acc[c].wrapping_add(p[c] as i32 * kk);
					}
				}
				let o = ((y - y0) * dw + x) * ch;
				for c in 0..ch {
					t[o + c] = clip8(acc[c]);
				}
			}
		}
		tmp = t;
		(&tmp, dw, y0)
	} else {
		(src, sw, 0)
	};
	if sh == dh {
		return mid.to_vec();
	}
	let mut out = vec![0u8; dw * dh * ch];
	let mut acc = vec![0i32; ch];
	for (y, (ymin, k)) in vc.iter().enumerate() {
		for x in 0..mid_w {
			acc.iter_mut().for_each(|a| *a = half);
			for (i, &kk) in k.iter().enumerate() {
				let p = &mid[((ymin + i - row0) * mid_w + x) * ch..][..ch];
				for c in 0..ch {
					acc[c] = acc[c].wrapping_add(p[c] as i32 * kk);
				}
			}
			let o = (y * dw + x) * ch;
			for c in 0..ch {
				out[o + c] = clip8(acc[c]);
			}
		}
	}
	out
}

/// `RGBA` -> `RGBa` (`rgbA2rgba`: color x alpha / 255, rounded).
pub fn premultiply(rgba: &mut [u8]) {
	for p in rgba.chunks_exact_mut(4) {
		let a = p[3] as u32;
		for v in &mut p[..3] {
			let t = *v as u32 * a + 128;
			*v = (((t >> 8) + t) >> 8) as u8;
		}
	}
}

/// `RGBa` -> `RGBA` (`rgba2rgbA`: color x 255 / alpha, truncated; alpha 0
/// and 255 unchanged).
pub fn unpremultiply(rgba: &mut [u8]) {
	for p in rgba.chunks_exact_mut(4) {
		let a = p[3] as u32;
		if a != 0 && a != 255 {
			for v in &mut p[..3] {
				*v = (255 * *v as u32 / a).min(255) as u8;
			}
		}
	}
}

/// `Image.resize` of an `RGBA` image (through `RGBa`, as Pillow does for
/// every filter but nearest).
pub fn resize_rgba(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
	if sw == dw && sh == dh {
		return src.to_vec();
	}
	let mut p = src.to_vec();
	premultiply(&mut p);
	let mut out = resample(&p, 4, sw, sh, dw, dh, f);
	unpremultiply(&mut out);
	out
}

/// An `RGBA` image composited over white, `RGB` out: `Image.new("RGB",
/// size, white).paste(img, mask=alpha)` (`BLEND` with `DIV255`).
pub fn over_white(rgba: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
	for p in rgba.chunks_exact(4) {
		let m = p[3] as u32;
		for &v in &p[..3] {
			let t = 255 * (255 - m) + v as u32 * m + 128;
			out.push((((t >> 8) + t) >> 8) as u8);
		}
	}
	out
}
