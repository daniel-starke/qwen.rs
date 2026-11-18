//! Device-agnostic (CPU + optional `ash` Vulkan offload) hybrid Qwen35 graph: token embedding,
//! RMSNorm, MRoPE, gated full attention (with Q/K norms + sigmoid gating),
//! gated-delta-net linear attention (recurrent), SwiGLU FFN, output norm and
//! tied LM head.
//!
//! Reference: llama.cpp's `src/models/qwen35.cpp` and
//! `src/models/delta-net-base.cpp` (the gated-delta-net math).
//!
//! We run the forward **token-by-token** (autoregressive / AR style), which is
//! the exact definition of the gated delta net and produces identical logits to
//! llama.cpp's chunked prefill.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use gguf::{GgmlType, GgufFile, TensorInfo};
use offload::{BatchJob, BatchJobSeq, GpuBlockInfo, OffloadManager, WeightId};
use quant::{apply_linear, apply_linear_batch, dequantize, f32_to_fp16, LinearWeight};

// Re-export math helpers, phase accumulators and the config/state types so the
// per-section modules can reach them via `use super::*;`.
pub use config::Hyperparams;
pub(crate) use config::{LayerTensors, MtpTensors};
pub(crate) use dflash::DflashTensors;
pub use dflash::{dflash_n_draft, DflashRun};
pub use forward::set_feature_dump;
pub use lin::gpu_prefill_requested;
pub(crate) use math::*;
pub use mtp::{Decoder, MtpStats};
pub use prompt::{build_image_prompt, ImageSpan, MultimodalPrompt, PromptInput};
pub use quant::progress;
pub(crate) use state::{spec_check, MtpEntry, SpecSnap, SsmReplay};
pub use state::{Checkpoint, KvCache, MtpRun, RunState};

pub mod config;
pub mod dflash;
pub mod draftvocab;
pub mod forward;
pub mod lin;
pub mod load;
pub mod math;
pub mod mtp;
pub mod pipe;
pub mod prompt;
pub mod sparse;
pub mod state;
pub mod suffix;
#[cfg(test)]
pub mod tests;
pub mod tune;
pub mod vec_fwd;

/// The central model struct: owns the GGUF file, embeddings, the (cached)
/// per-layer weights, the optional Vulkan offload manager and the run-state
/// config (`ctx` / `n_batch`).
pub struct Model {
	gguf: GgufFile,
	path: String,
	pub hp: Hyperparams,
	tok_embd: Vec<f32>,       // [n_embd, n_vocab] row-major
	output: LinearWeight, // [n_embd, n_vocab], kept in its native GGUF form (tied to token_embd if absent)
	output_row_sum: Vec<f32>, // [n_vocab] precomputed Σ_d output[o,d] (for DC logits)
	output_norm: Vec<f32>,
	pub full_attn_layers: Vec<bool>,
	/// Optional Vulkan offload manager. When `Some`, the large per-layer matmul
	/// weights for the last `n_gpu_layers` layers are dispatched on the GPU
	/// (fused f16 matmul). The KV cache and SSM recurrent state always stay on
	/// the CPU in `RunState` regardless, only the hidden-state vector crosses
	/// the bus at each matmul boundary. `None` ⇒ pure CPU (the verified path).
	pub offload: Option<Arc<OffloadManager>>,
	/// Cache of (still-quantized where possible) layer weights, populated lazily
	/// on first use (see `get_layer`). Big matmuls are kept as `LinearWeight`:
	/// `Q4_K_M` stays quantized (the fused GEMV dequantizes on the fly), so the
	/// whole 27B model resident footprint is ~16.8 GB (plus the small f32
	/// params), and per-token weight traffic is ~13.5 GB (not ~54 GB of an
	/// expanded `f16` matrix). Shared across threads behind a `Mutex`; once warm
	/// reads are lock-free `Arc` clones and the forward borrows through them, so
	/// no weight bytes are copied per token.
	layer_cache: Mutex<Vec<Option<Arc<LayerTensors>>>>,

	/// Maximum context length: KV-cache capacity (per full-attention layer) and
	/// the cap on positions. Defaults to `hp.context_length` (the model's native
	/// window) so large contexts work without tuning; `--ctx` can lower it.
	/// Set via `set_ctx`.
	ctx: usize,
	/// Prefill chunk size: the prompt is processed in batches of at most
	/// `n_batch` tokens (mirroring llama.cpp's `n_batch`, default 2048) so a
	/// 200k-token prompt never materializes the whole sequence's activations at
	/// once. Set via `set_n_batch`.
	n_batch: usize,
	/// MTP draft head, loaded by `enable_mtp`; `None` leaves every path as it
	/// was.
	mtp: Option<MtpTensors>,
	/// Tokens the MTP head drafts per verify pass (`enable_mtp`), or the
	/// DFlash2 drafts verified per pass (`enable_dflash`).
	n_draft: usize,
	/// The MTP head's reduced draft vocabulary (`enable_mtp`,
	/// `QWEN_MTP_VOCAB`); `None` drafts over the full lm_head.
	draft_vocab: Option<draftvocab::DraftVocab>,
	/// DFlash2 block drafter, loaded by `enable_dflash`; exclusive with `mtp`.
	dflash: Option<DflashTensors>,
}
