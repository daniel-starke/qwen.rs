//! `cli QWEN_IMAGE_DIR --txt2img PROMPT -o FILE.png [--image FILE]...`: one
//! image (an edit with condition images) without a server (`imagegen`'s
//! CPU backend; `QWEN_IMAGE_WEIGHTS=f32` is the reference).

/// Run it; the process exit code.
pub fn main(dir: &str, args: &[String]) -> i32 {
	let mut prompt = None;
	let mut out = None;
	let (mut w, mut h) = (0usize, 0usize);
	let mut images = Vec::new();
	let mut resolution = 1024usize;
	let mut seed = None;
	let mut steps = 0usize;
	let mut noise = None;
	let mut it = args.iter();
	while let Some(a) = it.next() {
		let mut val = |what: &str| -> String {
			it.next().cloned().unwrap_or_else(|| {
				eprintln!("error: {what} needs a value");
				std::process::exit(2)
			})
		};
		match a.as_str() {
			"--txt2img" => prompt = Some(val("--txt2img")),
			"-o" | "--out" => out = Some(val("-o")),
			"--size" => {
				let v = val("--size");
				let p: Vec<usize> = v.split(['x', 'X']).filter_map(|s| s.parse().ok()).collect();
				match p[..] {
					[s] => (w, h) = (s, s),
					[a, b] => (w, h) = (a, b),
					_ => {
						eprintln!("error: --size {v}: want WxH");
						return 2;
					}
				}
			}
			"--seed" => seed = val("--seed").parse().ok(),
			"--steps" => match val("--steps").parse() {
				Ok(n) if n <= imagegen::MAX_STEPS => steps = n,
				_ => {
					eprintln!("error: --steps wants 0..={}", imagegen::MAX_STEPS);
					return 2;
				}
			},
			"--noise" => noise = Some(val("--noise")),
			"--progress" => quant::progress::set_enabled(true),
			"--image" => images.push(std::path::PathBuf::from(val("--image")).into()),
			"--resolution" => match val("--resolution").parse() {
				Ok(r) if r >= 32 => resolution = r,
				_ => {
					eprintln!("error: --resolution wants a pixel count >= 32");
					return 2;
				}
			},
			a => {
				eprintln!("error: unknown option {a} (with --txt2img)");
				return 2;
			}
		}
	}
	if images.len() > imagegen::MAX_IMAGES {
		eprintln!(
			"error: at most {} --image (got {})",
			imagegen::MAX_IMAGES,
			images.len()
		);
		return 2;
	}
	let (Some(prompt), Some(out)) = (prompt, out) else {
		eprintln!("error: --txt2img needs a prompt and -o FILE.png");
		return 2;
	};
	let seed = seed.unwrap_or_else(|| {
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map_or(0, |d| d.as_nanos() as u64)
			% 1_000_000_007
	});
	let noise = match noise {
		Some(p) => match std::fs::read(&p) {
			Ok(b) => Some(
				b.chunks_exact(4)
					.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
					.collect(),
			),
			Err(e) => {
				eprintln!("error: {p}: {e}");
				return 1;
			}
		},
		None => None,
	};
	let o = imagegen::Options {
		prompt,
		width: w,
		height: h,
		seed,
		steps,
		noise,
		images,
		resolution,
		cancel: None,
	};
	let t0 = std::time::Instant::now();
	// `--progress`: one bar line instead of a line per stage
	let bar = quant::progress::Stages::start("image");
	let quiet = quant::progress::enabled();
	let r = imagegen::generate(std::path::Path::new(dir), &o, |stage, i, n| {
		bar.update(stage, imagegen::steps_done(stage, i), n);
		if quiet || (stage.starts_with("load") && i != n) {
			return;
		}
		let t = quant::progress::fmt_secs(t0.elapsed().as_secs_f64(), 1);
		if n > 0 {
			eprintln!("[image] {stage} {i}/{n} ({t})");
		} else {
			eprintln!("[image] {stage} ({t})");
		}
	});
	drop(bar);
	match r {
		Ok(img) => {
			if let Err(e) = imgio::save_png_rgba(&out, img.width, img.height, &img.rgba) {
				eprintln!("error: {e}");
				return 1;
			}
			eprintln!(
				"[image] {out}: {}x{}, seed {}, text encoder {:.1} s, vae encoder {:.1} s, transformer {:.1} s, vae {:.1} s",
				img.width, img.height, img.seed, img.secs[0], img.secs[1], img.secs[2], img.secs[3]
			);
			0
		}
		Err(e) => {
			eprintln!("error: {e}");
			1
		}
	}
}
