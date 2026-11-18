//! Image decoding for the vision path (M9), and the PNG writer of the image
//! generator.
//!
//! - **Author:** Daniel Starke
//! - **Copyright:** Copyright 2026 Daniel Starke
//!
//! Container decoding is delegated to the `image` crate (PNG, JPEG, BMP and
//! Netpbm), which is a production-grade, fuzzed implementation of exactly the
//! part of this crate that has no relationship to the model: parsing other
//! people's file formats. What stays hand-written is the part that *is*
//! model-visible: the Pillow-compatible bicubic resample the Qwen-VL
//! preprocessor specifies (`resize`), which no general-purpose resizer
//! reproduces bit-for-bit, and the P6 writer behind `--image-dump`.
//!
//! Everything here is pure pixel handling. Patchifying, mean/std normalization
//! and the grid geometry live in the `vision` crate, next to the tower that
//! consumes them.

use std::io::Cursor;

use image::{ImageFormat, ImageReader, Limits};

pub mod netpbm;
pub mod pil;
pub mod resize;
#[cfg(test)]
mod tests;

/// An 8-bit interleaved RGB image. Alpha is dropped at decode time.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
	pub width: usize,
	pub height: usize,
	/// `width * height * 3` bytes, row-major, R,G,B per pixel.
	pub rgb: Vec<u8>,
}

impl std::fmt::Debug for Image {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Image({}x{}, {} bytes)",
			self.width,
			self.height,
			self.rgb.len()
		)
	}
}

impl Image {
	/// Construct from raw interleaved RGB, checking the length.
	pub fn from_rgb(width: usize, height: usize, rgb: Vec<u8>) -> Result<Image, String> {
		if rgb.len() != width * height * 3 {
			return Err(format!(
				"image: {}x{} needs {} bytes, got {}",
				width,
				height,
				width * height * 3,
				rgb.len()
			));
		}
		Ok(Image { width, height, rgb })
	}

	/// Pillow-compatible bicubic resample to `(w, h)`.
	pub fn resized(&self, w: usize, h: usize) -> Image {
		Image {
			width: w,
			height: h,
			rgb: resize::resize_rgb(&self.rgb, self.width, self.height, w, h),
		}
	}

	/// Encode as binary PPM (`P6`): what `--image-dump` writes.
	pub fn to_ppm(&self) -> Vec<u8> {
		netpbm::encode_ppm(self)
	}
}

/// The formats the vision path accepts.
const SUPPORTED: [ImageFormat; 4] = [
	ImageFormat::Png,
	ImageFormat::Jpeg,
	ImageFormat::Bmp,
	ImageFormat::Pnm,
];

/// Sniff the container from its leading bytes.
///
/// `image::guess_format` carries no magic for Netpbm (its `P1`-`P6` headers
/// are two ASCII bytes, not a signature) so that one is matched here.
fn detect(src: &[u8]) -> Option<ImageFormat> {
	if src.len() >= 2 && src[0] == b'P' && (b'1'..=b'6').contains(&src[1]) {
		return Some(ImageFormat::Pnm);
	}
	let format = image::guess_format(src).ok()?;
	SUPPORTED.contains(&format).then_some(format)
}

/// Ceilings applied to every decode.
///
/// The server takes base64 image blocks straight off the network, so a decoder
/// must not be able to be talked into a huge allocation by a small header.
/// 512 MiB is ~4x the largest image the tower will ever be handed after the
/// `--image-max-tokens` clamp.
fn limits() -> Limits {
	let mut limits = Limits::default();
	limits.max_image_width = Some(1 << 16);
	limits.max_image_height = Some(1 << 16);
	limits.max_alloc = Some(512 << 20);
	limits
}

/// Detected container format, for error messages.
pub fn format_name(src: &[u8]) -> &'static str {
	match detect(src) {
		Some(ImageFormat::Png) => "PNG",
		Some(ImageFormat::Jpeg) => "JPEG",
		Some(ImageFormat::Bmp) => "BMP",
		Some(ImageFormat::Pnm) => "Netpbm",
		_ => "unknown",
	}
}

/// Decode any supported container from its bytes.
pub fn decode(src: &[u8]) -> Result<Image, String> {
	let format = detect(src)
		.ok_or_else(|| "image: unrecognized format (want PNG, JPEG, BMP or Netpbm)".to_string())?;
	let mut reader = ImageReader::with_format(Cursor::new(src), format);
	reader.limits(limits());
	let decoded = reader.decode().map_err(|e| format!("image: {e}"))?;
	// `to_rgb8` drops alpha without compositing and widens grayscale, which is
	// what `Image::convert("RGB")` does on the Pillow side of the fixtures.
	let rgb = decoded.to_rgb8();
	Image::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.into_raw())
}

/// Decode to 8-bit RGBA (`width`, `height`, `width * height * 4` bytes):
/// Pillow's `convert("RGBA")` (opaque alpha for images without one, gray
/// widened). The image generator's condition images keep their alpha.
pub fn decode_rgba(src: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
	let format = detect(src)
		.ok_or_else(|| "image: unrecognized format (want PNG, JPEG, BMP or Netpbm)".to_string())?;
	let mut reader = ImageReader::with_format(Cursor::new(src), format);
	reader.limits(limits());
	let rgba = reader
		.decode()
		.map_err(|e| format!("image: {e}"))?
		.to_rgba8();
	Ok((
		rgba.width() as usize,
		rgba.height() as usize,
		rgba.into_raw(),
	))
}

/// Read a file and [`decode_rgba`] it.
pub fn load_rgba(path: &str) -> Result<(usize, usize, Vec<u8>), String> {
	let bytes = std::fs::read(path).map_err(|e| format!("image: cannot read {path}: {e}"))?;
	decode_rgba(&bytes).map_err(|e| format!("{path}: {e}"))
}

/// Encode 8-bit RGBA pixels (`width * height * 4` bytes, row-major) as a
/// PNG: RGBA, or RGB with the alpha dropped when `alpha` is false (the
/// image generator's output).
pub fn encode_png(
	width: usize,
	height: usize,
	rgba: &[u8],
	alpha: bool,
) -> Result<Vec<u8>, String> {
	use image::ImageEncoder;
	if rgba.len() != width * height * 4 {
		return Err(format!(
			"png: {} bytes for {width}x{height} RGBA",
			rgba.len()
		));
	}
	let rgb;
	let (px, ty) = if alpha {
		(rgba, image::ExtendedColorType::Rgba8)
	} else {
		rgb = rgba
			.chunks_exact(4)
			.flat_map(|p| [p[0], p[1], p[2]])
			.collect::<Vec<u8>>();
		(&rgb[..], image::ExtendedColorType::Rgb8)
	};
	let mut out = Vec::new();
	image::codecs::png::PngEncoder::new(&mut out)
		.write_image(px, width as u32, height as u32, ty)
		.map_err(|e| format!("png: {e}"))?;
	Ok(out)
}

/// Write 8-bit RGBA pixels as a PNG file ([`encode_png`] with alpha).
pub fn save_png_rgba(path: &str, width: usize, height: usize, rgba: &[u8]) -> Result<(), String> {
	let b = encode_png(width, height, rgba, true)?;
	std::fs::write(path, b).map_err(|e| format!("{path}: {e}"))
}

/// Read and decode a file.
pub fn load(path: &str) -> Result<Image, String> {
	let bytes = std::fs::read(path).map_err(|e| format!("image: cannot read {path}: {e}"))?;
	decode(&bytes).map_err(|e| format!("{path}: {e}"))
}
