//! Qwen-Image-2.1's use of the shared condition images ([`crate::cond`]):
//! the VAE encoder's input and the target size of an edit.

use super::vae::Map;
use crate::cond::{dims, Cond};

impl Cond {
	/// The VAE encoder's input: `[h][w][4]` in [-1, 1].
	pub fn vae_pixels(&self) -> Map {
		Map {
			h: self.h,
			w: self.w,
			c: 4,
			d: self
				.rgba
				.iter()
				.map(|&v| 2.0 * (v as f32 / 255.0) - 1.0)
				.collect(),
		}
	}
}

/// The target size for these conditions at `resolution`: the last image's
/// aspect, ~`resolution`^2 pixels.
pub fn target(conds: &[Cond], resolution: usize) -> Option<(usize, usize)> {
	let c = conds.last()?;
	let r = resolution as f64;
	let (w, h) = dims(r * r, c.orig.0 as f64 / c.orig.1 as f64);
	Some((w / 32 * 32, h / 32 * 32))
}
