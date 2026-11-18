//! The vision tower forward pass and the merger/projector.

use std::time::Instant;

use super::linalg::{gelu, gelu_erf, layer_norm, matmul_bias};
use super::{Grid, VisionConfig, VisionModel};
use quant::par_for;

/// `QWEN_VISION_TIMING=1` prints a per-phase split of the tower. Which phase
/// dominates flips with image size (the matmuls are O(patches) and attention
/// is O(patches^2)) so an end-to-end number cannot tell you where an image is
/// spending its seconds.
fn timing() -> bool {
	static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*F.get_or_init(|| std::env::var("QWEN_VISION_TIMING").is_ok())
}

/// Learned position embedding for `grid`, bilinearly interpolated from the
/// model's square `n_pos` grid and emitted in **merge-block order** so it lines
/// up with `patchify`'s patches.
///
/// This mirrors `fast_pos_embed_interpolate`: the source coordinates are
/// `linspace(0, side - 1, n)` per axis (endpoints included) not a
/// center-aligned resample. The two differ by half a cell at the edges, which
/// is small enough to look right and wrong enough to shift what the model
/// attends to.
pub fn position_embedding(cfg: &VisionConfig, pos_w: &[f32], grid: &Grid) -> Vec<f32> {
	let e = cfg.n_embd;
	let merge = cfg.merge_size;
	let side = (cfg.n_pos as f64).sqrt().round() as usize;

	let axis = |n: usize| -> Vec<(usize, usize, f32)> {
		(0..n)
			.map(|i| {
				let v = if n <= 1 {
					0.0
				} else {
					i as f64 * (side - 1) as f64 / (n - 1) as f64
				};
				let lo = v as usize; // truncation, matching the reference `.int()`
				let hi = (lo + 1).min(side - 1);
				(lo, hi, (v - lo as f64) as f32)
			})
			.collect()
	};
	let hs = axis(grid.gh);
	let ws = axis(grid.gw);

	let mut out = vec![0.0f32; grid.patches() * e];
	let mut s = 0usize;
	for by in 0..grid.mh {
		for bx in 0..grid.mw {
			for iy in 0..merge {
				for ix in 0..merge {
					let (h0, h1, dh) = hs[by * merge + iy];
					let (w0, w1, dw) = ws[bx * merge + ix];
					let taps = [
						(h0 * side + w0, (1.0 - dh) * (1.0 - dw)),
						(h0 * side + w1, (1.0 - dh) * dw),
						(h1 * side + w0, dh * (1.0 - dw)),
						(h1 * side + w1, dh * dw),
					];
					let dst = &mut out[s * e..(s + 1) * e];
					for (idx, wt) in taps {
						if wt == 0.0 {
							continue;
						}
						let src = &pos_w[idx * e..(idx + 1) * e];
						for k in 0..e {
							dst[k] += src[k] * wt;
						}
					}
					s += 1;
				}
			}
		}
	}
	out
}

/// 2-D rotary position embedding for the tower's attention.
///
/// Half the rotary channels carry the patch **row**, half the **column**, in
/// NEOX pairing `(i, i + half)`. The angle table is built once per image and
/// shared by every block and head.
pub fn rope_tables(cfg: &VisionConfig, grid: &Grid) -> (Vec<f32>, Vec<f32>) {
	rope_tables_at(cfg, &super::patch_positions(grid, cfg.merge_size))
}

/// [`rope_tables`] for patches at the given (row, column) positions.
pub fn rope_tables_at(cfg: &VisionConfig, positions: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
	let head_dim = cfg.head_dim();
	let half = head_dim / 2; // rotary channels; the pairing spans the head
	let quarter = half / 2; // frequencies per axis
	let mut cos = vec![0.0f32; positions.len() * half];
	let mut sin = vec![0.0f32; positions.len() * half];
	for (s, &(hp, wp)) in positions.iter().enumerate() {
		for j in 0..half {
			let (pos, k) = if j < quarter {
				(hp, j)
			} else {
				(wp, j - quarter)
			};
			let inv = 1.0f64 / 10000f64.powf(2.0 * k as f64 / half as f64);
			let ang = pos as f64 * inv;
			cos[s * half + j] = ang.cos() as f32;
			sin[s * half + j] = ang.sin() as f32;
		}
	}
	(cos, sin)
}

/// Apply the rotary tables in place to one `[n, n_head * head_dim]` block.
pub fn apply_rope(
	x: &mut [f32],
	n: usize,
	n_head: usize,
	head_dim: usize,
	cos: &[f32],
	sin: &[f32],
) {
	let half = head_dim / 2;
	let xp = x.as_mut_ptr() as usize;
	par_for(n, |t| {
		// SAFETY: positions are disjoint and each `t` is visited once.
		let row = unsafe {
			std::slice::from_raw_parts_mut(
				(xp as *mut f32).add(t * n_head * head_dim),
				n_head * head_dim,
			)
		};
		for h in 0..n_head {
			let head = &mut row[h * head_dim..(h + 1) * head_dim];
			for j in 0..half {
				let (c, s) = (cos[t * half + j], sin[t * half + j]);
				let (a, b) = (head[j], head[j + half]);
				head[j] = a * c - b * s;
				head[j + half] = b * c + a * s;
			}
		}
	});
}

/// Full (bidirectional) multi-head attention over every patch of the image
/// (`quant::attn_block` with the whole image as one block: the tiled
/// kernel; the scalar loop it replaced took 10.4 of the 12.2 s of a 4096
/// patch image on the NPU tower).
pub fn attention(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	n: usize,
	n_head: usize,
	head_dim: usize,
) -> Vec<f32> {
	let scale = 1.0 / (head_dim as f32).sqrt();
	let mut out = quant::uninit_vec(n * n_head * head_dim);
	// the kernel's SIMD form takes heads of a multiple of 16 dims: the
	// tower's 72 run padded to 80 with zeros (exact: zero terms in q k and
	// zero columns of v)
	let hp = head_dim.next_multiple_of(16);
	if hp == head_dim {
		quant::attn_block(q, k, v, n, 0, n_head, n_head, head_dim, scale, &mut out);
		return out;
	}
	let pad = |x: &[f32]| -> Vec<f32> {
		let mut y = vec![0f32; n * n_head * hp];
		let yp = y.as_mut_ptr() as usize;
		par_for(n, |t| {
			// SAFETY: task t owns row t
			let r = unsafe {
				std::slice::from_raw_parts_mut((yp as *mut f32).add(t * n_head * hp), n_head * hp)
			};
			for h in 0..n_head {
				r[h * hp..h * hp + head_dim].copy_from_slice(
					&x[(t * n_head + h) * head_dim..(t * n_head + h + 1) * head_dim],
				);
			}
		});
		y
	};
	let (qp, kp, vp) = (pad(q), pad(k), pad(v));
	let mut o = quant::uninit_vec(n * n_head * hp);
	quant::attn_block(&qp, &kp, &vp, n, 0, n_head, n_head, hp, scale, &mut o);
	let op = out.as_mut_ptr() as usize;
	par_for(n, |t| {
		// SAFETY: task t owns row t
		let r = unsafe {
			std::slice::from_raw_parts_mut(
				(op as *mut f32).add(t * n_head * head_dim),
				n_head * head_dim,
			)
		};
		for h in 0..n_head {
			r[h * head_dim..(h + 1) * head_dim]
				.copy_from_slice(&o[(t * n_head + h) * hp..(t * n_head + h) * hp + head_dim]);
		}
	});
	out
}

/// One block's attention on half-precision rows, for a caller that runs the
/// rest of the tower elsewhere (the NPU engine without an attention device):
/// `qkv` `[n][3 * n_embd]` f16 (per patch q, k, v, before RoPE), `hw` each
/// patch's (row, column) -> `out` `[n][n_embd]` f16.
pub fn attend_f16(cfg: &VisionConfig, hw: &[(u32, u32)], qkv: &[u8], out: &mut [u8]) {
	let (e, n) = (cfg.n_embd, hw.len());
	assert!(
		qkv.len() == n * 3 * e * 2 && out.len() == n * e * 2,
		"attention buffers do not match {n} patches"
	);
	// (host buffers are page aligned: the rows are u16 slices)
	let h16 = unsafe { std::slice::from_raw_parts(qkv.as_ptr() as *const u16, n * 3 * e) };
	let part = |k: usize| -> Vec<f32> {
		let mut v = quant::uninit_vec(n * e);
		let vp = v.as_mut_ptr() as usize;
		par_for(n, |t| {
			// SAFETY: task t owns row t
			let d = unsafe { std::slice::from_raw_parts_mut((vp as *mut f32).add(t * e), e) };
			quant::f16_to_f32_slice(&h16[(t * 3 + k) * e..(t * 3 + k + 1) * e], d);
		});
		v
	};
	let (mut q, mut k, v) = (part(0), part(1), part(2));
	let (cos, sin) = rope_tables_at(cfg, hw);
	apply_rope(&mut q, n, cfg.n_head, cfg.head_dim(), &cos, &sin);
	apply_rope(&mut k, n, cfg.n_head, cfg.head_dim(), &cos, &sin);
	let a = attention(&q, &k, &v, n, cfg.n_head, cfg.head_dim());
	let op = out.as_mut_ptr() as usize;
	par_for(n, |t| {
		// SAFETY: task t owns row t of `out` (page aligned host memory)
		let d = unsafe { std::slice::from_raw_parts_mut((op as *mut u16).add(t * e), e) };
		quant::f32_to_f16_slice(&a[t * e..(t + 1) * e], d);
	});
}

/// Run the tower over `patches` (`[grid.patches(), 3 * patch * patch]`,
/// merge-block order) and return `[grid.tokens(), proj_dim]`.
pub fn forward(m: &VisionModel, patches: &[f32], grid: &Grid) -> Vec<f32> {
	forward_deep(m, patches, grid).0
}

/// A merger over `x` (`[tokens][merged]`: each 2x2 patch group's rows,
/// contiguous in merge-block order): `fc2(GELU(fc1(x)))`.
fn merge_mlp(
	m: &VisionModel,
	x: &[f32],
	tokens: usize,
	fc1: (&[f32], &[f32]),
	fc2: (&[f32], &[f32]),
) -> Vec<f32> {
	let merged = x.len() / tokens;
	let mut y = matmul_bias(x, tokens, merged, fc1.0, Some(fc1.1), merged);
	let act = if m.gelu_erf { gelu_erf } else { gelu };
	let (yp, n) = (y.as_mut_ptr() as usize, y.len());
	quant::par_for(n.div_ceil(1 << 14), |i| {
		let yp = yp as *mut f32;
		let (a, b) = (i << 14, ((i + 1) << 14).min(n));
		for k in a..b {
			// SAFETY: task i owns [a, b)
			unsafe { *yp.add(k) = act(*yp.add(k)) };
		}
	});
	matmul_bias(&y, tokens, merged, fc2.0, Some(fc2.1), m.cfg.proj_dim)
}

/// [`forward`], plus each deepstack merger's rows (`m.deepstack` order,
/// `[grid.tokens(), proj_dim]` each).
pub fn forward_deep(m: &VisionModel, patches: &[f32], grid: &Grid) -> (Vec<f32>, Vec<Vec<f32>>) {
	let cfg = &m.cfg;
	let e = cfg.n_embd;
	let n = grid.patches();
	let head_dim = cfg.head_dim();
	let plen = 3 * cfg.patch_size * cfg.patch_size;
	assert_eq!(
		patches.len(),
		n * plen,
		"patch buffer does not match the grid"
	);

	// Patch embedding + learned position embedding.
	let mut x = matmul_bias(patches, n, plen, &m.patch_w, Some(&m.patch_b), e);
	let pos = position_embedding(cfg, &m.pos_w, grid);
	for (a, b) in x.iter_mut().zip(pos.iter()) {
		*a += *b;
	}

	let (cos, sin) = rope_tables(cfg, grid);
	let measure = timing();
	let t_all = Instant::now();
	let (mut ms_qkv, mut ms_attn, mut ms_proj, mut ms_mlp, mut ms_norm) = (0.0, 0.0, 0.0, 0.0, 0.0);
	let lap = |acc: &mut f64, t: Instant| {
		if measure {
			*acc += t.elapsed().as_secs_f64() * 1000.0;
		}
	};

	let merged = e * cfg.merge_size * cfg.merge_size;
	let tokens = grid.tokens();
	let mut deep = Vec::with_capacity(m.deepstack.len());
	for (bi, blk) in m.blocks.iter().enumerate() {
		// Attention sub-block.
		let t = Instant::now();
		let mut h = x.clone();
		layer_norm(&mut h, e, &blk.ln1_w, &blk.ln1_b, cfg.eps);
		lap(&mut ms_norm, t);
		let t = Instant::now();
		let qkv = matmul_bias(&h, n, e, &blk.qkv_w, Some(&blk.qkv_b), 3 * e);
		// Split the fused projection: Q rows, then K, then V.
		let mut q = quant::uninit_vec(n * e);
		let mut k = quant::uninit_vec(n * e);
		let mut v = quant::uninit_vec(n * e);
		for t in 0..n {
			let src = &qkv[t * 3 * e..(t + 1) * 3 * e];
			q[t * e..(t + 1) * e].copy_from_slice(&src[0..e]);
			k[t * e..(t + 1) * e].copy_from_slice(&src[e..2 * e]);
			v[t * e..(t + 1) * e].copy_from_slice(&src[2 * e..3 * e]);
		}
		apply_rope(&mut q, n, cfg.n_head, head_dim, &cos, &sin);
		apply_rope(&mut k, n, cfg.n_head, head_dim, &cos, &sin);
		lap(&mut ms_qkv, t);
		let t = Instant::now();
		let a = attention(&q, &k, &v, n, cfg.n_head, head_dim);
		lap(&mut ms_attn, t);
		let t = Instant::now();
		let a = matmul_bias(&a, n, e, &blk.out_w, Some(&blk.out_b), e);
		for (dst, add) in x.iter_mut().zip(a.iter()) {
			*dst += *add;
		}
		lap(&mut ms_proj, t);

		// MLP sub-block.
		let t = Instant::now();
		let mut h = x.clone();
		layer_norm(&mut h, e, &blk.ln2_w, &blk.ln2_b, cfg.eps);
		let mut up = matmul_bias(&h, n, e, &blk.up_w, Some(&blk.up_b), cfg.n_ff);
		for v in up.iter_mut() {
			*v = gelu(*v);
		}
		let down = matmul_bias(&up, n, cfg.n_ff, &blk.down_w, Some(&blk.down_b), e);
		for (dst, add) in x.iter_mut().zip(down.iter()) {
			*dst += *add;
		}
		lap(&mut ms_mlp, t);
		for dm in m.deepstack.iter().filter(|d| d.layer == bi) {
			// post-shuffle norm: over each token's merged row
			let mut h = x.clone();
			layer_norm(&mut h, merged, &dm.ln_w, &dm.ln_b, cfg.eps);
			deep.push(merge_mlp(
				m,
				&h,
				tokens,
				(&dm.fc1_w, &dm.fc1_b),
				(&dm.fc2_w, &dm.fc2_b),
			));
		}
	}
	if measure {
		eprintln!(
			"VISION patches={n} total_ms={:.1} norm={ms_norm:.1} qkv+rope={ms_qkv:.1} attn={ms_attn:.1} proj={ms_proj:.1} mlp={ms_mlp:.1}",
			t_all.elapsed().as_secs_f64() * 1000.0,
		);
	}

	// Merger: normalize each patch, concatenate each 2x2 group (already
	// contiguous thanks to the merge-block patch order), then the two-layer
	// projector.
	layer_norm(&mut x, e, &m.post_ln_w, &m.post_ln_b, cfg.eps);
	debug_assert_eq!(n, tokens * cfg.merge_size * cfg.merge_size);
	// `x` is already [tokens, merged] once reinterpreted, so no copy is needed.
	let out = merge_mlp(m, &x, tokens, (&m.mm0_w, &m.mm0_b), (&m.mm2_w, &m.mm2_b));
	(out, deep)
}
