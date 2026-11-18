//! Binary PPM (`P6`) writer.
//!
//! Decoding Netpbm is the `image` crate's job like every other container; what
//! lives here is the one direction it is not asked for. `--image-dump` writes
//! P6 because it is the format a human can pipe straight into ImageMagick, and
//! because the unit tests round-trip through it.

use super::Image;

/// Encode an [`Image`] as binary PPM (`P6`). Used by `--image-dump` to let a
/// human look at exactly the pixels the vision tower was handed.
pub fn encode_ppm(img: &Image) -> Vec<u8> {
	let mut out = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
	out.extend_from_slice(&img.rgb);
	out
}
