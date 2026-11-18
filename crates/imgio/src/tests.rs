//! Decoder and resampler tests.
//!
//! The fixtures in `tests/data/` are produced by `tests/gen_fixtures.py`
//! (Pillow), each together with a `.ref.ppm` holding **Pillow's own
//! decode** of the same file. PNG/BMP/Netpbm are lossless, so those must match
//! byte for byte; JPEG and the resampler are compared with a tolerance,
//! because a different IDCT rounding or chroma upsampler is a legitimate
//! implementation choice, not a bug.
//!
//! Decoding itself is the `image` crate's; these cases pin the *contract*
//! this crate exposes to `vision` (RGB8 out, alpha dropped, grayscale and
//! palette widened, Pillow-equivalent pixels) across a dependency bump.

use super::*;

macro_rules! fixture {
	($name:literal) => {
		include_bytes!(concat!("../tests/data/", $name)).as_slice()
	};
}

/// Decode a `.ref.ppm` fixture (Pillow's ground truth).
fn reference(bytes: &[u8]) -> Image {
	decode(bytes).expect("reference ppm")
}

/// Max and mean absolute per-channel difference between two same-size images.
fn diff(a: &Image, b: &Image) -> (u8, f64) {
	assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
	let mut max = 0u8;
	let mut sum = 0u64;
	for (x, y) in a.rgb.iter().zip(b.rgb.iter()) {
		let d = x.abs_diff(*y);
		max = max.max(d);
		sum += d as u64;
	}
	(max, sum as f64 / a.rgb.len() as f64)
}

#[test]
fn png_truecolor_is_exact() {
	let got = decode(fixture!("rgb8.png")).unwrap();
	assert_eq!(got, reference(fixture!("rgb8.ref.ppm")));
}

#[test]
fn png_adam7_interlaced_is_exact() {
	// Adam7-interlaced PNG decodes to exactly the same pixels as the
	// non-interlaced file of the same image: hence `rgb8.ref.ppm` as the
	// reference, not a fixture of its own.
	//
	// This case exists because it was *documented as refused* long after it had
	// stopped being: the claim survived the move from the hand-written decoders
	// to the `image` crate, which de-interlaces transparently. Pillow cannot
	// write interlaced PNG, so `gen_fixtures.py` encodes this one by hand
	// (IHDR interlace method 1, seven passes).
	let got = decode(fixture!("rgb8_adam7.png")).unwrap();
	assert_eq!(got, reference(fixture!("rgb8.ref.ppm")));
}

#[test]
fn png_grayscale_is_exact() {
	let got = decode(fixture!("gray8.png")).unwrap();
	assert_eq!(got, reference(fixture!("gray8.ref.ppm")));
}

#[test]
fn png_palette_is_exact() {
	let got = decode(fixture!("pal8.png")).unwrap();
	assert_eq!(got, reference(fixture!("pal8.ref.ppm")));
}

#[test]
fn png_rgba_drops_alpha() {
	let got = decode(fixture!("rgba8.png")).unwrap();
	assert_eq!(got, reference(fixture!("rgba8.ref.ppm")));
}

/// 1-bit depth: the sub-byte sample path, which is the one that silently
/// produces garbage if the shift is computed wrongly.
#[test]
fn png_one_bit_is_exact() {
	let got = decode(fixture!("bw1.png")).unwrap();
	assert_eq!(got, reference(fixture!("bw1.ref.ppm")));
}

#[test]
fn bmp_is_exact() {
	let got = decode(fixture!("rgb8.bmp")).unwrap();
	assert_eq!(got, reference(fixture!("rgb8_bmp.ref.ppm")));
}

#[test]
fn ppm_is_exact() {
	let got = decode(fixture!("rgb8.ppm")).unwrap();
	assert_eq!(got, reference(fixture!("rgb8.ref.ppm")));
}

#[test]
fn ppm_round_trips_through_our_encoder() {
	let img = decode(fixture!("rgb8.png")).unwrap();
	let back = decode(&img.to_ppm()).unwrap();
	assert_eq!(img, back);
}

#[test]
fn jpeg_444_matches_pillow_closely() {
	let got = decode(fixture!("rgb8_444.jpg")).unwrap();
	let (max, mean) = diff(&got, &reference(fixture!("rgb8_444.ref.ppm")));
	assert!(
		max <= 4 && mean < 0.6,
		"4:4:4 jpeg drift too large: max={max} mean={mean:.3}"
	);
}

/// 4:2:0 subsamples chroma, so the upsampler's filter is visible in the
/// output. This bound is what caught nearest-neighbor replication: with it
/// the mean sat at 11.3 LSB on this (deliberately high-frequency) fixture,
/// against ~1 for the center-aligned bilinear the reference decoders use.
#[test]
fn jpeg_420_matches_pillow_closely() {
	let got = decode(fixture!("rgb8_420.jpg")).unwrap();
	let (max, mean) = diff(&got, &reference(fixture!("rgb8_420.ref.ppm")));
	assert!(
		mean < 3.0,
		"4:2:0 jpeg drift too large: max={max} mean={mean:.3}"
	);
}

#[test]
fn jpeg_grayscale_matches_pillow_closely() {
	let got = decode(fixture!("gray8.jpg")).unwrap();
	let (max, mean) = diff(&got, &reference(fixture!("gray8.ref.jpg.ppm")));
	assert!(
		max <= 4 && mean < 0.6,
		"gray jpeg drift too large: max={max} mean={mean:.3}"
	);
}

/// Restart markers reset the DC predictor mid-scan; getting that wrong shows
/// up as banding rather than an error, so it needs its own fixture.
#[test]
fn jpeg_restart_markers() {
	let got = decode(fixture!("rgb8_rst.jpg")).unwrap();
	let (max, mean) = diff(&got, &reference(fixture!("rgb8_rst.ref.ppm")));
	assert!(
		mean < 3.0,
		"restart-marker jpeg drift too large: max={max} mean={mean:.3}"
	);
}

/// Progressive JPEG: the old hand-written decoder refused SOF2 outright. This
/// asserts it is decoded, and decoded *right*: a progressive scan that is
/// merely accepted but reassembled wrongly still returns a plausible image.
#[test]
fn jpeg_progressive_matches_pillow_closely() {
	let got = decode(fixture!("rgb8_prog.jpg")).unwrap();
	let (max, mean) = diff(&got, &reference(fixture!("rgb8_prog.ref.ppm")));
	assert!(
		mean < 3.0,
		"progressive jpeg drift too large: max={max} mean={mean:.3}"
	);
}

#[test]
fn bicubic_downscale_matches_pillow() {
	let src = decode(fixture!("resize_src.ppm")).unwrap();
	let got = src.resized(32, 24);
	let (max, mean) = diff(&got, &reference(fixture!("resize_down.ref.ppm")));
	assert!(
		max <= 2 && mean < 0.3,
		"downscale drift: max={max} mean={mean:.3}"
	);
}

/// On a smooth source (no hard edges for the bicubic kernel to overshoot at)
/// we and Pillow must agree to within rounding in **both** directions.
#[test]
fn bicubic_matches_pillow_on_a_smooth_source() {
	let src = decode(fixture!("smooth_src.ppm")).unwrap();
	let (max, mean) = diff(
		&src.resized(200, 150),
		&reference(fixture!("smooth_up.ref.ppm")),
	);
	assert!(
		max <= 2 && mean < 0.3,
		"smooth upscale drift: max={max} mean={mean:.3}"
	);
	let (max, mean) = diff(
		&src.resized(37, 29),
		&reference(fixture!("smooth_down.ref.ppm")),
	);
	assert!(
		max <= 2 && mean < 0.3,
		"smooth downscale drift: max={max} mean={mean:.3}"
	);
}

/// Upscaling the high-frequency fixture is where we deliberately differ:
/// Pillow's two-pass resample stores the horizontal result back into an 8-bit
/// image, so bicubic overshoot at a hard edge is **clipped** before the
/// vertical pass. We keep the intermediate in `f32`, which is strictly more
/// faithful, and costs a handful of LSBs next to a discontinuity. The mean is
/// what has to stay small; the max is the overshoot, and is expected.
#[test]
fn bicubic_upscale_tracks_pillow_apart_from_its_clipped_intermediate() {
	let src = decode(fixture!("resize_src.ppm")).unwrap();
	let got = src.resized(200, 150);
	let (max, mean) = diff(&got, &reference(fixture!("resize_up.ref.ppm")));
	assert!(mean < 0.6, "upscale drift: max={max} mean={mean:.3}");
}

#[test]
fn resize_to_same_size_is_identity() {
	let src = decode(fixture!("rgb8.png")).unwrap();
	assert_eq!(src, src.resized(src.width, src.height));
}

#[test]
fn unknown_format_is_an_error() {
	let e = decode(b"not an image at all").unwrap_err();
	assert!(e.contains("unrecognized"), "{e}");
}

/// A truncated file must not panic or return a short buffer: the server
/// decodes whatever arrives in a base64 block.
///
/// PNG is checksummed, so a cut file is an error. JPEG is not: a cut scan
/// leaves the remaining MCUs at their initial value, and every production
/// decoder (libjpeg included) returns that partial image rather than failing.
/// What we assert there is that it is still a *well-formed* image of the
/// declared size: the invariant `Image::from_rgb` exists to guarantee.
#[test]
fn truncated_input_does_not_produce_a_malformed_image() {
	let png = fixture!("rgb8.png");
	assert!(decode(&png[..png.len() / 2]).is_err());

	let jpg = fixture!("rgb8_444.jpg");
	if let Ok(img) = decode(&jpg[..jpg.len() / 2]) {
		assert_eq!(img.rgb.len(), img.width * img.height * 3);
		assert_eq!((img.width, img.height), (37, 23));
	}
}

/// A format the vision path does not accept must be refused by name rather
/// than reaching a decoder that was not compiled in.
#[test]
fn unsupported_container_is_refused() {
	// GIF's magic. `image` knows the format; we do not enable it.
	let mut src = b"GIF89a".to_vec();
	src.extend_from_slice(&[0u8; 32]);
	assert_eq!(format_name(&src), "unknown");
	assert!(decode(&src).unwrap_err().contains("unrecognized"));
}

/// A small header must not be able to ask for a large allocation.
#[test]
fn oversized_header_is_refused() {
	// A P6 claiming 100000x100000 (30 GB) with no data behind it.
	let src = b"P6
100000 100000
255
"
	.to_vec();
	assert!(decode(&src).is_err());
}

#[test]
fn format_name_identifies_each_container() {
	assert_eq!(format_name(fixture!("rgb8.png")), "PNG");
	assert_eq!(format_name(fixture!("rgb8_420.jpg")), "JPEG");
	assert_eq!(format_name(fixture!("rgb8.bmp")), "BMP");
	assert_eq!(format_name(fixture!("rgb8.ppm")), "Netpbm");
}
