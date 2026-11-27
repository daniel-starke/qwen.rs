//! The packed sequence of MiniMax-H3-Image: `[text | reference images |
//! generated image]` in one 1-D sequence with full (bidirectional)
//! attention.
//!
//! - Text rows carry the text encoder's rows (a reference's vision block
//!   among them, tagged as image) at rotary `(p, 0, 0)`, `p` the row index.
//! - Each image (references, then the target) is one frame on the rotary
//!   clock, starting where the text ends: frame `T + j`, and its `(h, w)`
//!   coordinates are an aspect-normalized grid centered on the unit square,
//!   scaled by 32 (a square spans `[0, 32)`), built in f64 like numpy's
//!   `linspace(.., endpoint=False)` and fed to the rotary in f32.
//! - Rows of an image are its latents patchified 2x2, row-major over the
//!   patch grid, each row `[c][py][px]` (`patchify_video_latents`).

use super::config::{DitCfg, REF_T, TAG_IMAGE};

pub struct Layout {
	/// Rotary `(t, h, w)` of every row, in f32 as the transformer reads it.
	pub pos: Vec<[f32; 3]>,
	/// Modality tag of every row.
	pub tags: Vec<u8>,
	pub text: usize,
	pub refs: usize,
	pub target: usize,
	/// Patch grid `(rows, cols)` of every reference, then the target.
	pub grids: Vec<(usize, usize)>,
}

impl Layout {
	pub fn len(&self) -> usize {
		self.text + self.refs + self.target
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// First image row (references, then the target).
	pub fn image0(&self) -> usize {
		self.text
	}

	/// First target row.
	pub fn target0(&self) -> usize {
		self.text + self.refs
	}

	/// The distinct timesteps of a step at `t` (sorted) and every row's index
	/// into them: references are held at [`REF_T`].
	pub fn row_timesteps(&self, t: f32) -> (Vec<f32>, Vec<u8>) {
		let mut u = vec![t];
		if self.refs > 0 && REF_T != t {
			u.push(REF_T);
		}
		u.sort_by(f32::total_cmp);
		let it = u.iter().position(|&x| x == t).unwrap() as u8;
		let ir = u.iter().position(|&x| x == REF_T).unwrap_or(0) as u8;
		let mut idx = vec![it; self.len()];
		idx[self.text..self.text + self.refs].fill(ir);
		(u, idx)
	}
}

/// One aspect-normalized spatial axis: `dim / patch` coordinates (numpy
/// `linspace(left, left + ratio, n, endpoint=False) * 32` in f64).
fn axis(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
	let ratio = dim as f64 / sqrt_area;
	let left = (1.0 - ratio) / 2.0;
	let n = dim / patch;
	let step = ((left + ratio) - left) / n as f64;
	(0..n).map(|i| (i as f64 * step + left) * 32.0).collect()
}

/// The `(h, w)` rotary grid of one latent frame (`lh x lw` latents).
fn frame(lh: usize, lw: usize, patch: usize) -> Vec<[f64; 2]> {
	let sa = ((lh * lw) as f64).sqrt();
	let (hs, ws) = (axis(lh, patch, sa), axis(lw, patch, sa));
	let mut out = Vec::with_capacity(hs.len() * ws.len());
	for &h in &hs {
		for &w in &ws {
			out.push([h, w]);
		}
	}
	out
}

/// The layout of `text_tags.len()` text rows, references of latent size
/// `refs[j] = (lh, lw)` and a target of `(lh, lw)` latents.
pub fn build(
	cfg: &DitCfg,
	text_tags: &[u8],
	refs: &[(usize, usize)],
	target: (usize, usize),
) -> Layout {
	let p = cfg.patch;
	let mut l = Layout {
		pos: Vec::new(),
		tags: text_tags.to_vec(),
		text: text_tags.len(),
		refs: 0,
		target: 0,
		grids: Vec::new(),
	};
	for i in 0..l.text {
		l.pos.push([i as f32, 0.0, 0.0]);
	}
	let mut time = l.text as f64;
	for (j, &(lh, lw)) in refs.iter().chain([&target]).enumerate() {
		let g = frame(lh, lw, p);
		for hw in &g {
			l.pos.push([time as f32, hw[0] as f32, hw[1] as f32]);
		}
		l.tags.extend(std::iter::repeat_n(TAG_IMAGE, g.len()));
		l.grids.push((lh / p, lw / p));
		if j < refs.len() {
			l.refs += g.len();
			time += 1.0;
		} else {
			l.target = g.len();
		}
	}
	l
}

/// Latents `[c][lh][lw]` -> rows `[(lh/p) (lw/p)][c p p]`.
pub fn patchify(lat: &[f32], c: usize, lh: usize, lw: usize, p: usize) -> Vec<f32> {
	let (gh, gw) = (lh / p, lw / p);
	let mut out = Vec::with_capacity(lat.len());
	for y in 0..gh {
		for x in 0..gw {
			for ch in 0..c {
				for py in 0..p {
					for px in 0..p {
						out.push(lat[(ch * lh + y * p + py) * lw + x * p + px]);
					}
				}
			}
		}
	}
	out
}

/// The inverse of [`patchify`].
pub fn unpatchify(rows: &[f32], c: usize, lh: usize, lw: usize, p: usize) -> Vec<f32> {
	let (gh, gw) = (lh / p, lw / p);
	let mut out = vec![0f32; rows.len()];
	let mut i = 0;
	for y in 0..gh {
		for x in 0..gw {
			for ch in 0..c {
				for py in 0..p {
					for px in 0..p {
						out[(ch * lh + y * p + py) * lw + x * p + px] = rows[i];
						i += 1;
					}
				}
			}
		}
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn square_spans_zero_to_32() {
		let a = axis(64, 2, 64.0);
		assert_eq!(a.len(), 32);
		assert_eq!(a[0], 0.0);
		assert_eq!(a[1], 1.0);
		assert_eq!(a[31], 31.0);
	}

	#[test]
	fn patchify_roundtrip() {
		let (c, lh, lw) = (3, 4, 6);
		let lat: Vec<f32> = (0..c * lh * lw).map(|i| i as f32).collect();
		let r = patchify(&lat, c, lh, lw, 2);
		// row 0 = patch (0, 0): channel 0's [0,1,6,7], then channel 1's
		assert_eq!(&r[..4], &[0.0, 1.0, 6.0, 7.0]);
		assert_eq!(r[4], 24.0);
		assert_eq!(unpatchify(&r, c, lh, lw, 2), lat);
	}
}
