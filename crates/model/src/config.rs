//! Model configuration and per-layer weight tensor layout.

use super::*;

/// Per-call (single forward_step thread) accumulator of time spent inside the
/// large matmul dispatchers (`lin`/`lin_batch`). Used by the `QWEN_PHASE`
/// diagnostic to split each token into matmul vs non-matmul cost.
pub struct Hyperparams {
	pub n_embd: usize,
	pub n_head: usize,
	pub n_head_kv: usize,
	pub head_dim: usize,
	pub n_rot: usize,
	pub rope_sections: [i64; 4],
	pub n_ff: usize,
	pub eps: f32,

	// rope
	pub rope_freq_base: f64,
	pub rope_freq_scale: f64,
	pub rope_attn_factor: f64,
	pub rope_ext_factor: f64,

	// ssm / linear attention
	pub ssm_state: usize,   // S  = head_k_dim = head_v_dim
	pub ssm_n_group: usize, // H_k
	pub ssm_dt_rank: usize, // H_v
	pub ssm_inner: usize,   // value_dim = S * H_v
	pub conv_kernel: usize,

	pub n_layer_main: usize,
	pub full_attn_interval: usize,
	pub n_vocab: usize,
	/// Native context length declared by the GGUF (`qwen35.context_length`).
	/// Used as the default `ctx` (KV-cache capacity) so the model's trained
	/// window (262144 here) is supported out of the box; `--ctx` can lower it.
	pub context_length: usize,
}

/// The MTP (multi-token prediction, "nextn") head: one extra full-attention
/// decoder block stored as `blk.{n_layer_main}` that predicts the token after
/// next from the next token's embedding and the main model's final hidden
/// state. Shares `token_embd` and the lm_head with the main model (this GGUF
/// has no `nextn.embed_tokens` / `nextn.shared_head_head`).
///
/// Reference: `llama.cpp/src/models/qwen35.cpp` (`graph_mtp`).
pub(crate) struct MtpTensors {
	/// `attn_*`, `post_attention_norm`, `ffn_*` of the block, as a
	/// full-attention [`LayerTensors`].
	pub(crate) layer: LayerTensors,
	/// `[2 * n_embd -> n_embd]`, input `concat(enorm(e), hnorm(h))`.
	pub(crate) eh_proj: LinearWeight,
	pub(crate) enorm: Vec<f32>,
	pub(crate) hnorm: Vec<f32>,
	/// Final norm before the shared lm_head (`output_norm` if absent).
	pub(crate) head_norm: Vec<f32>,
}

/// Per-layer weights, built lazily and shared (behind `Arc`) across tokens via
/// the layer cache.
pub(crate) struct LayerTensors {
	pub(crate) attn_norm: Vec<f32>,
	pub(crate) post_norm: Vec<f32>,
	// full attention (large matmul weights). `LinearWeight` keeps `Q4_K_M`
	// weights quantized (the GEMV dequantizes them on the fly: fused
	// dequant-GEMV), so the layer cache stays at the model's quantized size
	// (~16.8 GB for the 27B) instead of a 4x-larger `f32`/`f16` expansion, and
	// per-token weight traffic drops ~4x (see `quant::gemv_q4k`).
	pub(crate) wq: Option<LinearWeight>,
	pub(crate) wk: Option<LinearWeight>,
	pub(crate) wv: Option<LinearWeight>,
	pub(crate) wo: Option<LinearWeight>,
	pub(crate) q_norm: Option<Vec<f32>>,
	pub(crate) k_norm: Option<Vec<f32>>,
	// linear attention
	pub(crate) wqkv: Option<LinearWeight>,
	pub(crate) wqkv_gate: Option<LinearWeight>,
	pub(crate) conv1d: Option<Vec<f32>>,
	// `conv1d` transposed to tap-major `[ksz][conv_dim]` for `quant::conv_silu`.
	pub(crate) conv1d_t: Option<Vec<f32>>,
	pub(crate) ssm_dt: Option<Vec<f32>>,
	pub(crate) ssm_a: Option<Vec<f32>>,
	// Q4_K `[n_embd x Hv]` on disk. Kept in native form (not widened to f32):
	// as f32 these two cost 94 MB of reads per token across the 48 SSM layers,
	// against 13 MB quantized.
	pub(crate) ssm_beta: Option<LinearWeight>,
	pub(crate) ssm_alpha: Option<LinearWeight>,
	pub(crate) ssm_norm: Option<Vec<f32>>,
	pub(crate) ssm_out: Option<LinearWeight>,
	// ffn
	pub(crate) ffn_gate: LinearWeight,
	pub(crate) ffn_down: LinearWeight,
	pub(crate) ffn_up: LinearWeight,
}
