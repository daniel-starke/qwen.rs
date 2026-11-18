//! Convolution resampling (bicubic by default), matching Pillow's `resize`.
//!
//! The reference preprocessor for Qwen-VL is `Qwen2VLImageProcessor`, which
//! resizes with `PILImageResampling.BICUBIC`. Pillow does **not** do a naive
//! 4x4 bicubic tap: when downscaling it widens the filter support by the
//! reduction factor, so a 4000px photo shrunk to 768px averages ~26 source
//! pixels per output pixel instead of 4. A naive bicubic aliases badly at those
//! ratios, which is visible to the vision tower, so the widening is
//! reproduced here.

/// Pillow's bicubic kernel (Keys, `a = -0.5`), support 2.0.
#[inline]
fn cubic(x: f32) -> f32 {
	const A: f32 = -0.5;
	let x = x.abs();
	if x < 1.0 {
		((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
	} else if x < 2.0 {
		(((x - 5.0) * x + 8.0) * x - 4.0) * A
	} else {
		0.0
	}
}

/// Per-output-pixel taps: `(first_source_index, weights)`.
fn coefficients(in_len: usize, out_len: usize, support: f32) -> Vec<(usize, Vec<f32>)> {
	let scale = in_len as f32 / out_len as f32;
	let filter_scale = scale.max(1.0);
	let support = support * filter_scale;
	let mut out = Vec::with_capacity(out_len);
	for xx in 0..out_len {
		let center = (xx as f32 + 0.5) * scale;
		// Pillow *truncates* both ends (`(int)(...)`), it does not floor/ceil.
		// Rounding the right edge up instead widens the window by one tap and
		// shows up as a few-LSB drift on upscales.
		let xmin = ((center - support + 0.5) as isize).max(0) as usize;
		let xmax = ((center + support + 0.5) as isize)
			.min(in_len as isize)
			.max(0) as usize;
		let xmax = xmax.max(xmin + 1).min(in_len);
		let mut w = Vec::with_capacity(xmax - xmin);
		let mut sum = 0.0f32;
		for i in xmin..xmax {
			let v = cubic((i as f32 + 0.5 - center) / filter_scale);
			w.push(v);
			sum += v;
		}
		if sum != 0.0 {
			for v in w.iter_mut() {
				*v /= sum;
			}
		} else {
			// Degenerate (all taps zero): fall back to nearest.
			let n = w.len();
			w.iter_mut().for_each(|v| *v = 0.0);
			w[n / 2] = 1.0;
		}
		out.push((xmin, w));
	}
	out
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
	if v <= 0.0 {
		0
	} else if v >= 255.0 {
		255
	} else {
		(v + 0.5) as u8
	}
}

/// Resample an interleaved 3-channel 8-bit image to `(dst_w, dst_h)`.
pub fn resize_rgb(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
	assert_eq!(src.len(), src_w * src_h * 3);
	assert!(dst_w > 0 && dst_h > 0);
	if src_w == dst_w && src_h == dst_h {
		return src.to_vec();
	}
	// Horizontal pass into an f32 intermediate of shape [src_h, dst_w, 3].
	let hc = coefficients(src_w, dst_w, 2.0);
	let mut mid = vec![0.0f32; src_h * dst_w * 3];
	for y in 0..src_h {
		let row = &src[y * src_w * 3..(y + 1) * src_w * 3];
		for (x, (xmin, w)) in hc.iter().enumerate() {
			let mut acc = [0.0f32; 3];
			for (i, &k) in w.iter().enumerate() {
				let s = (xmin + i) * 3;
				acc[0] += row[s] as f32 * k;
				acc[1] += row[s + 1] as f32 * k;
				acc[2] += row[s + 2] as f32 * k;
			}
			let o = (y * dst_w + x) * 3;
			mid[o] = acc[0];
			mid[o + 1] = acc[1];
			mid[o + 2] = acc[2];
		}
	}
	// Vertical pass.
	let vc = coefficients(src_h, dst_h, 2.0);
	let mut out = vec![0u8; dst_w * dst_h * 3];
	for (y, (ymin, w)) in vc.iter().enumerate() {
		for x in 0..dst_w {
			let mut acc = [0.0f32; 3];
			for (i, &k) in w.iter().enumerate() {
				let s = ((ymin + i) * dst_w + x) * 3;
				acc[0] += mid[s] * k;
				acc[1] += mid[s + 1] * k;
				acc[2] += mid[s + 2] * k;
			}
			let o = (y * dst_w + x) * 3;
			out[o] = clamp_u8(acc[0]);
			out[o + 1] = clamp_u8(acc[1]);
			out[o + 2] = clamp_u8(acc[2]);
		}
	}
	out
}
