//! Vision tower tests.
//!
//! The load path needs a real 0.9 GB mmproj, so it is exercised by the CLI
//! (`--image-probe`) rather than here. What *is* testable in-process is the
//! part that silently produces plausible nonsense when it is wrong: the grid
//! geometry, the merge-block patch order, and the tower arithmetic itself.
//!
//! `tower_matches_numpy_reference` is the real gate. `tests/data/tiny.ref` is
//! produced by `tests/gen_reference.py`, a NumPy transcription of the Qwen3-VL
//! vision spec driven by the same integer LCG that builds the weights here: so
//! nothing has to be serialized and the two implementations share only their
//! *inputs*. It cannot
//! catch a shared misreading of the spec, but it does catch every transposed
//! matrix, mis-ordered patch and wrong rotary pairing, which is what actually
//! goes wrong.

use super::*;
use imgio::Image;

const E: usize = 16;
const NH: usize = 2;
const NFF: usize = 24;
const NL: usize = 2;
const PATCH: usize = 4;
const MERGE: usize = 2;
const PROJ: usize = 8;
const SIDE: usize = 4;
const GH: usize = 4;
const GW: usize = 6;

/// The same LCG the NumPy reference uses.
struct Lcg(u64);

impl Lcg {
	fn f(&mut self) -> f32 {
		self.0 = (self.0.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
		((self.0 as f64 / 0x7fff_ffffu32 as f64) * 2.0 - 1.0) as f32
	}
	fn arr(&mut self, n: usize) -> Vec<f32> {
		(0..n).map(|_| self.f()).collect()
	}
}

fn tiny_model(r: &mut Lcg) -> VisionModel {
	let plen = 3 * PATCH * PATCH;
	let patch_w = r.arr(E * plen);
	let patch_b = r.arr(E);
	let pos_w = r.arr(SIDE * SIDE * E);
	let mut blocks = Vec::new();
	for _ in 0..NL {
		blocks.push(VisionBlock {
			ln1_w: r.arr(E),
			ln1_b: r.arr(E),
			ln2_w: r.arr(E),
			ln2_b: r.arr(E),
			qkv_w: r.arr(3 * E * E),
			qkv_b: r.arr(3 * E),
			out_w: r.arr(E * E),
			out_b: r.arr(E),
			up_w: r.arr(NFF * E),
			up_b: r.arr(NFF),
			down_w: r.arr(E * NFF),
			down_b: r.arr(E),
		});
	}
	let post_ln_w = r.arr(E);
	let post_ln_b = r.arr(E);
	let merged = E * MERGE * MERGE;
	let mm0_w = r.arr(merged * merged);
	let mm0_b = r.arr(merged);
	let mm2_w = r.arr(PROJ * merged);
	let mm2_b = r.arr(PROJ);
	VisionModel {
		cfg: VisionConfig {
			image_size: SIDE * PATCH * MERGE,
			patch_size: PATCH,
			n_embd: E,
			n_ff: NFF,
			n_layer: NL,
			n_head: NH,
			proj_dim: PROJ,
			merge_size: MERGE,
			eps: 1e-6,
			mean: [0.5; 3],
			std: [0.5; 3],
			n_pos: SIDE * SIDE,
		},
		blocks,
		patch_w,
		patch_b,
		pos_w,
		post_ln_w,
		post_ln_b,
		mm0_w,
		mm0_b,
		mm2_w,
		mm2_b,
		deepstack: Vec::new(),
		gelu_erf: false,
	}
}

#[test]
fn tower_matches_numpy_reference() {
	let mut r = Lcg(0x1234_5678);
	let m = tiny_model(&mut r);
	let grid = Grid {
		width: GW * PATCH,
		height: GH * PATCH,
		gw: GW,
		gh: GH,
		mw: GW / MERGE,
		mh: GH / MERGE,
	};
	// The reference draws the patch buffer from the same stream, immediately
	// after the weights.
	let patches = r.arr(grid.patches() * 3 * PATCH * PATCH);
	let got = tower::forward(&m, &patches, &grid);

	let raw = include_bytes!("../tests/data/tiny.ref");
	let want: Vec<f32> = raw
		.chunks_exact(4)
		.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect();
	assert_eq!(got.len(), want.len(), "token count");
	assert_eq!(got.len(), grid.tokens() * PROJ);

	let mut max = 0.0f32;
	for (a, b) in got.iter().zip(want.iter()) {
		max = max.max((a - b).abs());
	}
	// f32 accumulation order differs between NumPy's BLAS and our kernels, so
	// the bar is float noise, not equality.
	assert!(
		max < 2e-4,
		"tower diverges from the NumPy reference: max |diff| = {max:e}"
	);
}

#[test]
fn smart_resize_rounds_to_whole_merge_blocks() {
	// factor = patch * merge = 32 for the real model.
	let g = smart_resize(1000, 700, 16, 2, 4, 4096).unwrap();
	assert_eq!(g.width % 32, 0);
	assert_eq!(g.height % 32, 0);
	assert_eq!(g.gw, g.width / 16);
	assert_eq!(g.mw * 2, g.gw);
	assert_eq!(g.mh * 2, g.gh);
}

#[test]
fn smart_resize_caps_the_token_budget() {
	let g = smart_resize(4000, 3000, 16, 2, 4, 256).unwrap();
	assert!(g.tokens() <= 256, "{} tokens", g.tokens());
	// Aspect ratio is preserved to within one merge block per axis.
	let want = 4000.0 / 3000.0;
	let got = g.width as f64 / g.height as f64;
	assert!((got - want).abs() < 0.1, "aspect {got} vs {want}");
}

#[test]
fn smart_resize_lifts_a_thumbnail_to_the_floor() {
	let g = smart_resize(8, 8, 16, 2, 4, 1024).unwrap();
	assert!(g.tokens() >= 4, "{} tokens", g.tokens());
	assert_eq!(g.width % 32, 0);
}

#[test]
fn smart_resize_rejects_a_degenerate_aspect_ratio() {
	let e = smart_resize(4000, 3, 16, 2, 4, 1024).unwrap_err();
	assert!(e.contains("aspect ratio"), "{e}");
}

/// Patch order is merge-block major, not raster. Nothing downstream errors if
/// this is wrong: the model just attends to the wrong places.
#[test]
fn patch_order_is_merge_block_major() {
	let grid = Grid {
		width: 96,
		height: 64,
		gw: 6,
		gh: 4,
		mw: 3,
		mh: 2,
	};
	let pos = patch_positions(&grid, 2);
	assert_eq!(pos.len(), 24);
	assert_eq!(&pos[0..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
	assert_eq!(&pos[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
	// Second block row starts at patch row 2.
	assert_eq!(pos[12], (2, 0));
	// Every (row, col) appears exactly once.
	let mut seen: Vec<(u32, u32)> = pos.clone();
	seen.sort_unstable();
	seen.dedup();
	assert_eq!(seen.len(), 24);
}

/// `patchify` must lay each patch out channel-major then row then column:
/// the order of the `v.patch_embd.weight` filters.
#[test]
fn patchify_normalizes_and_orders_channels() {
	// 8x8 image; pixel (x, y) gets a value that identifies it exactly.
	let mut rgb = vec![0u8; 8 * 8 * 3];
	for y in 0..8 {
		for x in 0..8 {
			for c in 0..3 {
				rgb[(y * 8 + x) * 3 + c] = (y * 8 + x + c * 2) as u8;
			}
		}
	}
	let img = Image::from_rgb(8, 8, rgb).unwrap();
	let grid = Grid {
		width: 8,
		height: 8,
		gw: 2,
		gh: 2,
		mw: 1,
		mh: 1,
	};
	let out = patchify(&img, &grid, 4, 2, [0.0; 3], [1.0; 3]);
	let plen = 3 * 4 * 4;
	assert_eq!(out.len(), 4 * plen);
	// Patch 0 is the top-left 4x4 block; its channel-0 plane is rows 0..4.
	for ky in 0..4 {
		for kx in 0..4 {
			let want = (ky * 8 + kx) as f32 / 255.0;
			assert!((out[ky * 4 + kx] - want).abs() < 1e-6, "c0 ({kx},{ky})");
		}
	}
	// Channel 2 of the same patch is offset by 4 in the source value.
	assert!((out[2 * 16] - 4.0 / 255.0).abs() < 1e-6);
	// Patch 1 is the block to the *right* (merge-block inner order).
	assert!((out[plen] - 4.0 / 255.0).abs() < 1e-6);
	// Patch 2 is the block *below* the first.
	assert!((out[2 * plen] - (4.0 * 8.0) / 255.0).abs() < 1e-6);
}

#[test]
fn patchify_applies_mean_and_std() {
	let img = Image::from_rgb(4, 4, vec![255u8; 4 * 4 * 3]).unwrap();
	let grid = Grid {
		width: 4,
		height: 4,
		gw: 1,
		gh: 1,
		mw: 1,
		mh: 1,
	};
	// (1.0 - 0.5) / 0.5 == 1.0 for every channel.
	let out = patchify(&img, &grid, 4, 1, [0.5; 3], [0.5; 3]);
	assert!(out.iter().all(|v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn layer_norm_has_zero_mean_and_unit_variance() {
	let mut x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.37 - 3.0).collect();
	let w = vec![1.0f32; 16];
	let b = vec![0.0f32; 16];
	linalg::layer_norm(&mut x, 16, &w, &b, 1e-6);
	for row in x.chunks(16) {
		let mean = row.iter().sum::<f32>() / 16.0;
		let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 16.0;
		assert!(mean.abs() < 1e-5, "mean {mean}");
		assert!((var - 1.0).abs() < 1e-3, "var {var}");
	}
}

#[test]
fn matmul_bias_is_sequence_major() {
	// y[t, o] = dot(x[t], w[o]) + b[o]
	let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 positions x 2 dims
	let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 out rows x 2 dims
	let b = vec![10.0, 20.0, 30.0];
	let y = linalg::matmul_bias(&x, 2, 2, &w, Some(&b), 3);
	assert_eq!(y, vec![11.0, 22.0, 33.0, 13.0, 24.0, 37.0]);
}

#[test]
fn softmax_sums_to_one() {
	let mut r = vec![1.0f32, 2.0, 3.0, -50.0];
	linalg::softmax(&mut r);
	assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
	assert!(r[2] > r[1] && r[1] > r[0] && r[3] < 1e-20);
}
