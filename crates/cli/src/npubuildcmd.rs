//! `cli BF16_DIR --npu-build OUT ...`: build the NPU engine's files from the
//! BF16 checkpoint (`npubuild`).

/// Run the build; the process exit code.
pub fn main(src: &str, args: &[String]) -> i32 {
	if !npubuild::is_checkpoint(src) {
		eprintln!("error: {src} is not a checkpoint directory (no model.safetensors.index.json, no model_index.json + transformer/)");
		return 2;
	}
	let mut o = npubuild::Options::new(src, "");
	let mut it = args.iter();
	while let Some(a) = it.next() {
		let mut val = |what: &str| -> String {
			it.next().cloned().unwrap_or_else(|| {
				eprintln!("error: {what} needs a value");
				std::process::exit(2)
			})
		};
		match a.as_str() {
			"--npu-build" => o.out = val("--npu-build").into(),
			"--npu-calib" => o.calib.push(val("--npu-calib").into()),
			"--npu-calib-chunks" => o.max_chunks = val("--npu-calib-chunks").parse().unwrap_or(16),
			"--npu-steps" => {
				o.steps = val("--npu-steps")
					.split(',')
					.map(|s| s.to_string())
					.collect()
			}
			"--npu-compiler" => o.compiler = Some(val("--npu-compiler").into()),
			"--npu-jobs" => o.jobs = val("--npu-jobs").parse().unwrap_or(6),
			"--npu-no-gpu" => o.gpu = false,
			a => {
				eprintln!("error: unknown option {a} (with --npu-build)");
				return 2;
			}
		}
	}
	if o.out.as_os_str().is_empty() {
		eprintln!("error: --npu-build needs the output directory");
		return 2;
	}
	if npubuild::qimage::is_qwen_image(&o.src) {
		let cal = |src: &std::path::Path, _out: &std::path::Path, layers: usize| {
			imagegen::npucal::calibrate(src, layers, &mut |m| eprintln!("npu-build: {m}"))
		};
		return match npubuild::qimage::build(&o, Some(&cal)) {
			Ok(()) => 0,
			Err(e) => {
				eprintln!("error: {e}");
				1
			}
		};
	}
	match npubuild::build(&o) {
		Ok(()) => 0,
		Err(e) => {
			eprintln!("error: {e}");
			1
		}
	}
}
