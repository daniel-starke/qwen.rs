//! Small math helpers, activation functions, RoPE, and the gated-delta-net
//! per-head recurrence. Shared by both the incremental (`forward_step`) and
//! batched-prefill forward paths.

use super::*;

thread_local! {
	pub(crate) static MATMUL_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static LMHEAD_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static SSN_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static LAYER_LOOP_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static ATTN_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static NORM_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static FLEXOP_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	pub(crate) static AOUT_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	/// Elementwise activation work: SwiGLU, the SSM gate/norm loops, conv1d.
	pub(crate) static ELEM_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
	/// Attention core (scores/softmax/weighted sum) and the SSM q/k/v prep.
	pub(crate) static PREP_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(crate) fn phase_measure() -> bool {
	static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*M.get_or_init(|| std::env::var("QWEN_PHASE").is_ok())
}

pub(crate) fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
	let t = if phase_measure() {
		Some(Instant::now())
	} else {
		None
	};
	let mut y = vec![0.0f32; x.len()];
	rms_norm_into(x, w, eps, &mut y);
	if let Some(t) = t {
		NORM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
	}
	y
}

pub(crate) fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
	let t = if phase_measure() {
		Some(Instant::now())
	} else {
		None
	};
	let mut y = vec![0.0f32; x.len()];
	l2_norm_into(x, eps, &mut y);
	if let Some(t) = t {
		NORM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
	}
	y
}

/// [`rms_norm`] into a caller buffer. The batched prefill passes call the
/// norms once per row (or per head per row) from every pool worker at once;
/// a returned `Vec` per call was ~12M heap allocations per 2048-token chunk,
/// the 20-40 KB ones above the low-fragmentation heap's size classes, so they
/// serialized on the process heap lock.
pub(crate) fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
	let n = x.len();
	let mut ss = 0.0f32;
	for &v in x {
		ss += v * v;
	}
	let inv = 1.0 / (ss / n as f32 + eps).sqrt();
	for i in 0..n {
		y[i] = x[i] * inv * w[i];
	}
}

/// [`l2_norm`] into a caller buffer (see [`rms_norm_into`]).
pub(crate) fn l2_norm_into(x: &[f32], eps: f32, y: &mut [f32]) {
	let mut ss = 0.0f32;
	for &v in x {
		ss += v * v;
	}
	let inv = 1.0 / ss.sqrt().max(eps);
	for i in 0..x.len() {
		y[i] = x[i] * inv;
	}
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
	1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
	if x > 20.0 {
		x
	} else {
		(1.0 + x.exp()).ln()
	}
}

/// Value-column block width of one prefill scan task (`QWEN_SSM_COLS`,
/// default 32): 48 heads x 4 blocks = 192 tasks over 24 workers, each block's
/// `128 x 32` state slice (16 KB) resident in L1. Falls back to the whole
/// head when the width does not divide `s` or is not a multiple of 8.
pub(crate) fn ssm_col_block(s: usize) -> usize {
	static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	let b = *B.get_or_init(|| {
		std::env::var("QWEN_SSM_COLS")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(32)
	});
	if b >= 8 && b.is_multiple_of(8) && b <= s && s.is_multiple_of(b) {
		b
	} else {
		s
	}
}

/// One gated-delta-net head update: the O(S^2) recurrence that dominates the
/// SSM cost. The body mirrors the original nested-loop scan **exactly** (same
/// operation order, same float ops) so it is bit-identical; extracting it lets
/// `forward_step`/`prefill_chunk` run the `Hv` independent heads across the
/// thread pool. Each head owns a disjoint `S*S` slice
/// of `st` and a disjoint `S` slice of `o`, so parallel heads are race-free.
///
/// Math (per head, state `st` is `S x S`, `⊗` outer product, `sk = stᵀ k`):
///   st ← g·st
///   sk = stᵀ k
///   d  = β·(v − sk)
///   st ← st + k ⊗ d
///   o  = stᵀ q · (1/√S)
pub(crate) fn delta_net_head(
	st: &mut [f32],
	s: usize,
	k: &[f32],
	q: &[f32],
	v: &[f32],
	beta: f32,
	g: f32,
	qscale: f32,
	o: &mut [f32],
) {
	// Delegates to the column-block kernel over all `s` columns. Two passes
	// over `st` (decay + S^T k, then rank-1 update + S^T q), each row hot in
	// L1 between its read and its write; bit-identical to the four-pass form.
	quant::delta_net_cols(st, s, 0, k, q, &v[..s], beta, g, qscale, &mut o[..s]);
}

/// Should the **contiguous** MRoPE section layout be used instead of the
/// interleaved one?
///
/// Qwen ships two. The contiguous form (llama.cpp's `GGML_ROPE_TYPE_MROPE`)
/// gives the first `sections[0]` rotary channels to `t`, the next
/// `sections[1]` to `h`, the rest to `w`. The interleaved form
/// (`GGML_ROPE_TYPE_IMROPE`) round-robins `t, h, w` across channels instead, so
/// each axis gets a spread of frequencies rather than a band.
///
/// **This architecture uses the interleaved one**, and that is the default
/// here: `llama-model.cpp`'s `llama_model_rope_type` maps `LLM_ARCH_QWEN35`
/// (with `QWEN3VL`, `QWEN35MOE`, `QWEN3TTS`) to `LLAMA_ROPE_TYPE_IMROPE`, while
/// only `QWEN2VL`/`PADDLEOCR` get plain `MROPE`.
///
/// The two are **identical for text**, where `t == h == w`, so no text-only
/// measurement can distinguish them, which is why this went unnoticed until
/// the vision path existed, and why flipping it changes no text logits.
/// `QWEN_MROPE_CONTIGUOUS=1` selects the other variant for an A/B.
pub(crate) fn mrope_contiguous() -> bool {
	static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*F.get_or_init(|| std::env::var("QWEN_MROPE_CONTIGUOUS").is_ok())
}

/// MRoPE (Qwen-style multi-factor RoPE) applied in-place to one head vector
/// of `head_dim` channels. Only the first `n_rot` dims are rotated (NEOX
/// layout: pair `(c, c + n_rot/2)`).
///
/// `pos` is the `(t, h, w)` coordinate triple. For text all three are the
/// token position and the section layout drops out of the arithmetic; for an
/// image patch they are the frame index and the patch's row and column in the
/// merged grid, and the layout is what makes the model see 2-D structure.
pub(crate) fn rope_apply(head: &mut [f32], pos: [i64; 3], hp: &Hyperparams) {
	let n_rot = hp.n_rot;
	let half = n_rot / 2; // = 32
	let theta_scale = hp.rope_freq_base.powf(-2.0 / n_rot as f64);
	let s = hp.rope_sections;
	let sect_dims = (s[0] + s[1] + s[2] + s[3]).max(1);
	let contiguous = mrope_contiguous();
	// Text is the overwhelmingly common case and needs none of the section
	// arithmetic; keeping it on its own path also guarantees decode stays
	// bit-identical to the pre-vision build.
	let degenerate = pos[0] == pos[1] && pos[1] == pos[2];
	for p in 0..half {
		let base = if degenerate {
			pos[0]
		} else {
			let sector = (p as i64) % sect_dims;
			if contiguous {
				if sector < s[0] {
					pos[0]
				} else if sector < s[0] + s[1] {
					pos[1]
				} else {
					pos[2]
				}
			} else {
				// `ggml_mrope_cache_init`, `is_imrope` branch.
				match sector % 3 {
					0 if sector < 3 * s[0] => pos[0],
					1 if sector < 3 * s[1] => pos[1],
					2 if sector < 3 * s[2] => pos[2],
					// Reachable only if the sections do not cover the rotary
					// channels; llama.cpp uses a 4th position array here, which
					// this arch does not have (`sections[3] == 0`, and
					// 11/11/10 covers all 32 channel pairs).
					_ => pos[2],
				}
			}
		} as f64;
		let theta = hp.rope_freq_scale * base * theta_scale.powi(p as i32);
		let cos = (theta).cos() * hp.rope_attn_factor;
		let sin = (theta).sin() * hp.rope_attn_factor;
		let x0 = head[p];
		let x1 = head[p + half];
		head[p] = x0 * cos as f32 - x1 * sin as f32;
		head[p + half] = x0 * sin as f32 + x1 * cos as f32;
	}
}
