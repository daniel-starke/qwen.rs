//! Deflates every `src/shaders/*.spv` into `OUT_DIR/<name>.spv.z`, embedded
//! with `quant::packed!("<name>.spv")`.

use std::env;
use std::path::PathBuf;

fn main() {
	let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
	println!("cargo::rerun-if-changed=src/shaders");
	for e in std::fs::read_dir("src/shaders").expect("src/shaders") {
		let p = e.expect("src/shaders").path();
		if p.extension().is_some_and(|x| x == "spv") {
			let raw = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
			let name = format!("{}.z", p.file_name().unwrap().to_string_lossy());
			std::fs::write(
				out.join(name),
				miniz_oxide::deflate::compress_to_vec(&raw, 10),
			)
			.expect("OUT_DIR");
		}
	}
}
