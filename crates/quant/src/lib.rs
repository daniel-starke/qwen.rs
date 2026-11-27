//! K-quant / f32 / f16 dequantization and CPU matrix-multiply for the qwen.rs
//! CPU (`NdArray`) path.
//!
//! Reference: llama.cpp `ggml/src/ggml-quants.c` (`dequantize_row_q4_K`,
//! `get_scale_min_k4`) and `ggml/src/ggml-common.h` (`block_q4_K`,
//! `block_q8_0`, `QK_K = 256`, `K_SCALE_SIZE = 12`).
//!
//! Dequantization is dependency-free and produces plain `f32` row-major
//! matrices that the model graph and GEMM routines consume. The GEMM is
//! parallelized with `std::thread` (no `rayon` dependency) and uses an AVX2
//! inner-product kernel on x86_64 when the CPU supports it.

pub use gguf::GgmlType;

/// Number of elements per `Q4_K` / `Q5_K` / `Q6_K` super-block.
pub const QK_K: usize = 256;
/// Byte size of the `scales` field in a `Q4_K` block.
pub const K_SCALE_SIZE: usize = 12;
/// Bytes per `Q6_K` super-block (128 `ql` + 64 `qh` + 16 `scales` + 2 `d`).
pub const fn q6k_block_bytes() -> usize {
	128 + 64 + 16 + 2
}

pub mod act;
pub mod alloc;
pub mod attn;
pub mod calib;
pub mod cputime;
pub mod delta;
pub mod dequant;
pub mod energy;
pub mod fp16;
pub mod gemm;
pub mod gemv_q4k;
pub mod gemv_q6k;
pub mod knob;
pub mod linear;
pub mod ltime;
pub mod packed;
pub mod parallel;
pub mod progress;
pub mod q8k;
pub mod snap;
pub mod sparse_cm;
#[cfg(test)]
mod tests;

// Public API of the crate (unchanged surface).
pub use act::{conv_silu, silu_mul, silu_mul_assign};
pub use attn::{
	attn_block, attn_block_rows, attn_block_rows_k8, attn_causal, attn_causal_groups,
	f16_to_f32_slice, f32_to_f16_slice, kv_extend, KvElem, K8,
};
pub use delta::{delta_net_cols, delta_net_cols_ld};
pub use dequant::{
	dequantize, dequantize_f16, dequantize_f32, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0,
	quantize_q4_k, quantize_q4_k_block, quantize_q6_k, quantize_q6_k_block,
};
pub use fp16::{f32_to_fp16, fp16_to_f32};
pub use gemm::{attn_scores, gemm, gemm_f16, gemv, gemv_f16};
pub use gemv_q4k::gemv_q4k;
pub(crate) use gemv_q4k::q4k_rows;
pub use gemv_q6k::gemv_q6k;
pub(crate) use gemv_q6k::q6k_rows;
pub use linear::{
	apply_linear, apply_linear_batch, apply_linear_group, apply_linear_group_exact,
	apply_linear_group_rows, apply_linear_group_rows_pre, concat_weights, exact_decode_forced,
	gemm_q4k, gemm_q4k_strided, gemm_q4k_strided_q8, gemm_q6k, gemm_q6k_strided,
	gemm_q6k_strided_q8, par_quantize_q8k, par_rows_q8k, par_silu_mul_q8k, set_exact_decode,
	LinearWeight,
};
pub use parallel::par_for;

/// An `out_dim`-long `Vec<f32>` whose contents are **not** initialized.
///
/// The GEMV/GEMM kernels write every element of their output before anything
/// reads it, so zeroing first is pure memset: at ~13 MB of matmul outputs per
/// decode token it costs ~2 ms. Only use this where the very next operation
/// fills the whole buffer: an accumulating kernel (`o[j] += ..`) still needs
/// `vec![0.0; n]`.
#[inline]
#[allow(clippy::uninit_vec)] // deliberate: see above
pub fn uninit_vec(n: usize) -> Vec<f32> {
	let mut v: Vec<f32> = Vec::with_capacity(n);
	// SAFETY: `f32` has no invalid bit patterns and no `Drop`, the capacity is
	// exactly `n`, and every caller overwrites all `n` elements before reading.
	unsafe { v.set_len(n) };
	v
}

/// Runtime CPU/env feature gates, resolved **once**.
///
/// These used to be re-read per GEMV call. Decode issues ~575 GEMVs per token,
/// and `std::env::var` on Windows is a `GetEnvironmentVariableW` plus a
/// `String` allocation each time, so the lookups alone cost milliseconds per
/// token on a path that is otherwise DRAM-bound.
pub(crate) struct Features {
	pub avx2: bool,
	pub f16c: bool,
	pub avxvnni: bool,
	pub phase: bool,
}

pub(crate) fn features() -> &'static Features {
	static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
	F.get_or_init(|| Features {
		avx2: is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
		f16c: is_x86_feature_detected!("f16c"),
		avxvnni: std::env::var("QWEN_AVXVNNI").is_ok() && is_x86_feature_detected!("avxvnni"),
		phase: std::env::var("QWEN_PHASE").is_ok(),
	})
}
pub use q8k::{q8_prof_ns, quantize_q8k, quantize_q8k_into, Q8K};

// Shared crate-internal helpers (reachable from sibling modules).
pub(crate) use dequant::get_scale_min_k4;
pub(crate) use gemv_q4k::{hsum_float_8, q4k_dot_q8k_scalar};
pub(crate) use gemv_q6k::{get_scale_shuffle_k6, q6k_dot_q8k_scalar};
pub(crate) use parallel::{par_gemm_collect, par_range, par_range_pre, par_rows};

// Helpers consumed only by the `#[cfg(test)]` suite.
#[cfg(test)]
pub(crate) use dequant::{dequantize_q4_k_block, dequantize_q6_k_block, dequantize_q8_0_block};
#[cfg(test)]
pub(crate) use gemv_q4k::q4k_dot_q8k_avxvnni;
#[cfg(test)]
pub(crate) use gemv_q6k::q6k_dot_q8k_avxvnni;
