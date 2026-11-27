//! Writes a linker map beside the binary (`target/<profile>/cli.map`), the
//! input of `scripts/mapsize.py` for binary size analysis.

use std::env;
use std::path::PathBuf;

fn main() {
	println!("cargo::rerun-if-changed=build.rs");
	// OUT_DIR = <target>[/<triple>]/<profile>/build/cli-<hash>/out: the
	// profile directory, where the binary lands, is three levels up.
	let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
	let Some(dir) = out.ancestors().nth(3) else {
		return;
	};
	let map = dir.join("cli.map");
	match env::var("CARGO_CFG_TARGET_ENV").as_deref() {
		Ok("gnu") => println!("cargo::rustc-link-arg-bins=-Wl,-Map={}", map.display()),
		Ok("msvc") => println!("cargo::rustc-link-arg-bins=/MAP:{}", map.display()),
		_ => {}
	}
}
