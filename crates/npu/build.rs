//! Deflates the SHAVE instruction and conflict tables into
//! `OUT_DIR/<name>.z`, embedded with `quant::packed_str!("<name>")`.

use std::env;
use std::path::PathBuf;

fn main() {
	let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
	for f in ["src/shave/isa.tbl", "src/shave/conflicts.tbl"] {
		println!("cargo::rerun-if-changed={f}");
		let raw = std::fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}"));
		let name = format!("{}.z", f.rsplit('/').next().unwrap());
		std::fs::write(
			out.join(name),
			miniz_oxide::deflate::compress_to_vec(&raw, 10),
		)
		.expect("OUT_DIR");
	}
}
