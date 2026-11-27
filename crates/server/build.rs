//! Packs the `--tune-profile` corpus (`src/tune.rs`, `embedded_corpus`): the source
//! files its Read tool results show, deflated into `OUT_DIR/corpus.bin.z` (`quant::packed!`) so
//! the tuner needs nothing beside the binary at a quarter of the size.
//! Layout before deflate, per file: path length (u32 LE), path, text length
//! (u32 LE), text.

use std::env;
use std::path::PathBuf;

/// Workspace-relative paths, in the order the tuner reads them.
const FILES: &[&str] = &[
	"crates/server/src/lib.rs",
	"crates/server/src/openai.rs",
	"crates/server/src/tools.rs",
	"crates/prefixcache/src/lib.rs",
	"crates/model/src/forward.rs",
	"crates/model/src/mtp.rs",
	"crates/model/src/dflash.rs",
	"crates/model/src/suffix.rs",
	"crates/tokenizer/src/lib.rs",
	"crates/quant/src/linear.rs",
	"crates/quant/src/parallel.rs",
	"crates/offload/src/manager.rs",
	"crates/offload/src/gpu/device.rs",
	"crates/offload/src/gpu/stream.rs",
	"crates/offload/src/gpu/gemv.rs",
	"crates/offload/src/gpu/matmul.rs",
	"crates/offload/src/gpu/attn.rs",
	"crates/npu/src/engine.rs",
	"crates/npu/src/flash/engine.rs",
	"crates/npu/src/flash/gpudec.rs",
	"crates/ov/src/stateless.rs",
];

fn main() {
	let root =
		PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("../..");
	let mut raw = Vec::new();
	for f in FILES {
		let p = root.join(f);
		println!("cargo::rerun-if-changed={}", p.display());
		let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
		for part in [f.as_bytes(), text.as_bytes()] {
			raw.extend_from_slice(&(part.len() as u32).to_le_bytes());
			raw.extend_from_slice(part);
		}
	}
	let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("corpus.bin.z");
	std::fs::write(&out, miniz_oxide::deflate::compress_to_vec(&raw, 10)).expect("corpus.bin.z");
}
