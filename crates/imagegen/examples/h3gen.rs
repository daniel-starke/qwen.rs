//! MiniMax-H3-Image without the CLI: `h3gen H3_DIR PROMPT OUT.png [--size N |
//! --width W --height H] [--seed N] [--image REF]...` in the format
//! `QWEN_IMAGE_WEIGHTS` names (`q6k` default, `f32` = the reference). A
//! request below 4 MP is generated at 4 MP and resized back. `PROMPT` =
//! `@FILE` reads it from a file (UTF-8, trimmed).

use imagegen::h3::pipeline::generate;
use imagegen::{ImageSrc, Options};
use std::path::PathBuf;

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	if a.len() < 3 {
		return Err("usage: h3gen H3_DIR PROMPT OUT.png [--size N | --width W --height H] [--seed N] [--image REF]...".into());
	}
	let num = |k: &str, d: usize| -> Result<usize, String> {
		match a.iter().position(|x| x == k) {
			Some(i) => a
				.get(i + 1)
				.and_then(|v| v.parse().ok())
				.ok_or(format!("{k}: a number")),
			None => Ok(d),
		}
	};
	let size = num("--size", 0)?;
	let images: Vec<ImageSrc> = a
		.windows(2)
		.filter(|w| w[0] == "--image")
		.map(|w| ImageSrc::Path(PathBuf::from(&w[1])))
		.collect();
	let prompt = match a[1].strip_prefix('@') {
		Some(f) => std::fs::read_to_string(f)
			.map_err(|e| format!("{f}: {e}"))?
			.trim()
			.to_string(),
		None => a[1].clone(),
	};
	let o = Options {
		prompt,
		width: num("--width", size)?,
		height: num("--height", size)?,
		seed: num("--seed", 42)? as u64,
		images,
		..Options::default()
	};
	let t = std::time::Instant::now();
	let img = generate(&PathBuf::from(&a[0]), &o, |s, i, n| {
		if n == 0 || i == n || s == "denoise" {
			eprintln!("[{:7.1} s] {s} {i}/{n}", t.elapsed().as_secs_f64());
		}
	})?;
	let png = imgio::encode_png(img.width, img.height, &img.rgba, false)?;
	std::fs::write(&a[2], png).map_err(|e| format!("{}: {e}", a[2]))?;
	eprintln!(
		"wrote {} ({}x{}): text encoder {:.1} s, vae encoder {:.1} s, transformer {:.1} s, vae {:.1} s",
		a[2], img.width, img.height, img.secs[0], img.secs[1], img.secs[2], img.secs[3]
	);
	Ok(())
}
