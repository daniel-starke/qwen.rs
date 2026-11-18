//! Qwen-Image's VAE on the NPU (`docs/image.md` "NPU performance"):
//! the 3x3 stride-1 convolutions of its decoder and encoder ([`on_npu`]),
//! f16, over tiles of [`T`] x [`T`] output pixels, so a graph's shape does
//! not depend on the image size.
//!
//! One graph per (input, output) channel count, `vae/c{ci}_{co}`:
//!
//! ```text
//!   x [n, T + 2, T + 2, ci]  (pixel-major tiles, the 1-pixel halo included)
//!   w [co, ci, 3, 3], b [co] -> y [n, T, T, co]
//! ```
//!
//! The cores build the tiles (the zero padding at the map's edges, the
//! nearest-neighbor upsampling, and the RMS norm + SiLU that precede most
//! convolutions: in the graph the norm cost more than the convolution at
//! 144 channels) and add the residuals. `n` (tiles per job) makes a job
//! ~50 GFLOP: 2 tiles at 1152 channels, 64 at 144.

use crate::g::{Graph, Ty};
use crate::st::Checkpoint;
use crate::{mkdir, say};
use std::path::Path;

/// Output tile side.
pub const T: usize = 32;

/// Tiles per job of a `ci -> co` convolution (~50 GFLOP, at most 64).
pub fn tiles_per_job(ci: usize, co: usize) -> usize {
	let f = 2 * 9 * ci * co * T * T;
	(50_000_000_000 / f).clamp(1, 64)
}

/// The graph's name for a `ci -> co` convolution.
pub fn kind(ci: usize, co: usize) -> String {
	format!("c{ci}_{co}")
}

/// One 3x3 convolution over `n` tiles (see the module docs).
pub fn conv_graph(n: usize, ci: usize, co: usize) -> Graph {
	let mut g = Graph::new();
	let x = g.parameter("x", Ty::F16, &[n, T + 2, T + 2, ci]);
	let w = g.parameter("w", Ty::F16, &[co, ci, 3, 3]);
	let b = g.parameter("b", Ty::F16, &[co]);
	let h = g.transpose(&x, &[0, 3, 1, 2]);
	let y = g.conv2d(&h, &w);
	let y = g.transpose(&y, &[0, 2, 3, 1]);
	let y = g.add(&y, &b);
	g.result(&y, "y");
	g
}

/// Whether the VAE's convolution weight `name` (shape `sh`) runs on the
/// NPU: a 3x3 stride-1 one of the decoder or the encoder with at least 16
/// inputs and 64 outputs. The cores keep the decoder's `conv_out` (4
/// channels), the encoder's `conv_in` (4 inputs) and its stride-2
/// downsamplers.
pub fn on_npu(name: &str, sh: &[usize]) -> bool {
	(name.starts_with("decoder.") || name.starts_with("encoder."))
		&& name.ends_with(".weight")
		&& !name.contains("downsampler")
		&& sh.len() == 4
		&& sh[2] == 3
		&& sh[3] == 3
		&& sh[1] >= 16
		&& sh[0] >= 64
}

/// The `(ci, co)` of the VAE's NPU convolutions, from the checkpoint's
/// `vae/` (sorted, distinct).
pub fn conv_kinds(src: &Path) -> Result<Vec<(usize, usize)>, String> {
	let ck = Checkpoint::open_dir(&src.join("vae"))?;
	let mut v: Vec<(usize, usize)> = ck
		.names()
		.filter_map(|n| {
			ck.shape(n)
				.ok()
				.filter(|s| on_npu(n, s))
				.map(|s| (s[1], s[0]))
		})
		.collect();
	v.sort_unstable();
	v.dedup();
	Ok(v)
}

/// `OUT/graphs/vae/c{ci}_{co}.xml` + `.args` / `.uses`.
pub fn step_graphs(src: &Path, out: &Path) -> Result<(), String> {
	let gdir = out.join("graphs");
	mkdir(&gdir.join("vae"))?;
	let kinds = conv_kinds(src)?;
	for &(ci, co) in &kinds {
		crate::sets::save(
			&gdir,
			&format!("vae/{}", kind(ci, co)),
			&conv_graph(tiles_per_job(ci, co), ci, co),
		)?;
	}
	say(&format!(
		"VAE graphs ({} convolution kinds) in {}",
		kinds.len(),
		gdir.join("vae").display()
	));
	Ok(())
}

/// Compile the VAE graphs whose blob is missing.
pub fn step_compile(t: &crate::sets::Tools, src: &Path, out: &Path) -> Result<(), String> {
	let gdir = out.join("graphs");
	for (ci, co) in conv_kinds(src)? {
		let stem = format!("vae/{}", kind(ci, co));
		if gdir.join(format!("{stem}.blob")).exists() {
			continue;
		}
		say(&format!("compiling {stem}"));
		crate::sets::compile_graph(t, &gdir, &stem, true, false)?;
	}
	Ok(())
}
