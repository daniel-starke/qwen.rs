//! Image preprocessing: the Qwen-VL "smart resize", patchification and
//! normalization.
//!
//! Everything here is geometry, and every choice in it is dictated by what the
//! tower and the text model expect downstream:
//!
//! * The grid must be a whole number of **merge blocks** (`patch_size *
//!   spatial_merge_size` = 32 px here), because the projector consumes 2x2
//!   patch groups and the text side gets one token per group.
//! * Patches come out in **merge-block order** (block row, block column, then
//!   the 2x2 inside) not raster order. The projector's reshape assumes it, and
//!   so do the 2-D RoPE position ids and the interpolated position embedding.
//!   Raster order here produces plausible-looking garbage, not an error.

use imgio::Image;

/// Resolved grid geometry for one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
	/// Resized pixel dimensions (multiples of `patch * merge`).
	pub width: usize,
	pub height: usize,
	/// Patch counts (`width / patch`, `height / patch`).
	pub gw: usize,
	pub gh: usize,
	/// Merge-block counts: one text token each.
	pub mw: usize,
	pub mh: usize,
}

impl Grid {
	/// Patches in the image (`gw * gh`).
	pub fn patches(&self) -> usize {
		self.gw * self.gh
	}
	/// Text tokens the image occupies (`mw * mh`).
	pub fn tokens(&self) -> usize {
		self.mw * self.mh
	}
}

/// Qwen-VL `smart_resize`: round to a whole number of merge blocks, then scale
/// the whole image until its token count lands inside `[min_tokens,
/// max_tokens]`, preserving aspect ratio.
///
/// `max_tokens` is stated in **merged** tokens (what the prompt pays for),
/// which is the only unit a caller can reason about.
pub fn smart_resize(
	width: usize,
	height: usize,
	patch: usize,
	merge: usize,
	min_tokens: usize,
	max_tokens: usize,
) -> Result<Grid, String> {
	if width == 0 || height == 0 {
		return Err("vision: zero-sized image".to_string());
	}
	let factor = patch * merge;
	// Pixels per merged token.
	let per_token = (factor * factor) as f64;
	let max_pixels = max_tokens as f64 * per_token;
	let min_pixels = min_tokens.max(1) as f64 * per_token;

	let ratio = width.max(height) as f64 / width.min(height) as f64;
	if ratio > 200.0 {
		return Err(format!(
			"vision: aspect ratio {ratio:.0}:1 is beyond the 200:1 the preprocessor supports"
		));
	}

	let round_by = |v: f64| -> usize { (((v / factor as f64).round() as usize).max(1)) * factor };
	let floor_by = |v: f64| -> usize { (((v / factor as f64).floor() as usize).max(1)) * factor };
	let ceil_by = |v: f64| -> usize { (((v / factor as f64).ceil() as usize).max(1)) * factor };

	let (mut h, mut w) = (round_by(height as f64), round_by(width as f64));
	let area = (width * height) as f64;
	if (h * w) as f64 > max_pixels {
		let beta = (area / max_pixels).sqrt();
		h = floor_by(height as f64 / beta);
		w = floor_by(width as f64 / beta);
	} else if ((h * w) as f64) < min_pixels {
		let beta = (min_pixels / area).sqrt();
		h = ceil_by(height as f64 * beta);
		w = ceil_by(width as f64 * beta);
	}
	Ok(Grid {
		width: w,
		height: h,
		gw: w / patch,
		gh: h / patch,
		mw: w / factor,
		mh: h / factor,
	})
}

/// Resize `img` to `grid` and emit one normalized feature vector per patch, in
/// merge-block order.
///
/// Each vector is `3 * patch * patch` long, ordered channel-major then row then
/// column: the layout of the `v.patch_embd.weight` filters, so the patch
/// embedding is a plain matmul against it. The temporal axis is *not* expanded
/// here: a still image repeats its single frame across both temporal slices, so
/// the loader folds `patch_embd.weight.1` into `patch_embd.weight` instead
/// (see [`crate::VisionModel::load`]).
pub fn patchify(
	img: &Image,
	grid: &Grid,
	patch: usize,
	merge: usize,
	mean: [f32; 3],
	std: [f32; 3],
) -> Vec<f32> {
	let resized = img.resized(grid.width, grid.height);
	let plen = 3 * patch * patch;
	let mut out = vec![0.0f32; grid.patches() * plen];
	let mut s = 0usize;
	for by in 0..grid.mh {
		for bx in 0..grid.mw {
			for iy in 0..merge {
				for ix in 0..merge {
					let py = by * merge + iy;
					let px = bx * merge + ix;
					let dst = &mut out[s * plen..(s + 1) * plen];
					for c in 0..3 {
						for ky in 0..patch {
							let row = (py * patch + ky) * grid.width;
							for kx in 0..patch {
								let v = resized.rgb[(row + px * patch + kx) * 3 + c] as f32 / 255.0;
								dst[(c * patch + ky) * patch + kx] = (v - mean[c]) / std[c];
							}
						}
					}
					s += 1;
				}
			}
		}
	}
	out
}

/// Row and column index of each patch, in the same merge-block order
/// `patchify` emits. These are the 2-D RoPE position ids.
pub fn patch_positions(grid: &Grid, merge: usize) -> Vec<(u32, u32)> {
	let mut out = Vec::with_capacity(grid.patches());
	for by in 0..grid.mh {
		for bx in 0..grid.mw {
			for iy in 0..merge {
				for ix in 0..merge {
					out.push(((by * merge + iy) as u32, (bx * merge + ix) as u32));
				}
			}
		}
	}
	out
}
