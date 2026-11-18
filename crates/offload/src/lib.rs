//! Dual-backend offload manager (M4).
//!
//! The offload manager uploads a budget of the *last* transformer layers'
//! large matmul weights to a Vulkan device (raw `ash` + precompiled SPIR-V) and
//! dispatches those matrix products on the GPU.
//! Everything else (the recurrent SSM state, the attention **KV cache, norms,
//! RoPE, elementwise glue**) stays on the CPU in `RunState` / `Model`. Per the
//! project decision, the **KV cache remains in CPU RAM**: only the hidden-state
//! vector (a single token's `[n_embd]` floats, a few KB) crosses the bus at each
//! matmul boundary, which is negligible compared to shuttling the multi-GB
//! weights.
//!
//! The GPU path keeps each big matmul weight in its **original quantized GGUF
//! form** (`Q4_K`/`Q4_K_M`, `Q6_K`, `Q8_0`, `F16`, `F32`) in VRAM and **fuses
//! dequantization into the matmul** via a precompiled SPIR-V compute shader per
//! quant type (the exact llama.cpp mechanism: weights never expand to a full
//! `f16`/`f32` matrix in VRAM). Independent matmuls that share an input are folded
//! into one submission per token (`matmul_batch`) so the H2D uploads and D2H
//! readbacks overlap and the per-matmul sync overhead is amortized. The path is
//! behind the `vulkan` feature; without it (or with no adapter) the manager still
//! exists and `matmul` transparently falls back to `quant::apply_linear`, so the
//! routing logic is exercisable and the existing test suite keeps passing with no
//! network/registry access. When `vulkan` is enabled and a Vulkan adapter is
//! present, the last `n_gpu_layers` layers run their big matmuls on the GPU;
//! otherwise it silently falls back to CPU.

pub mod manager;
pub use manager::{
	pack_f16_weights, reference_matmul, BatchJob, BatchJobSeq, Deferred, GpuBlockInfo, NextCall,
	OffloadManager, WeightId, SPEC_MAX_SEQ,
};

#[cfg(feature = "vulkan")]
mod gpu;
#[cfg(feature = "vulkan")]
pub use gpu::{
	host_import_budget, host_imported, DeltaScan, Dense, DitAttn, DitBlk, DitMid, Experts,
	FlashAttn, Gemv, GemvMat, HcPre, MlpSplit, Region, VisAttn, DELTA_STATE, DELTA_WIN, EXP_SLOT,
	GEMV_NP, GEMV_WAVE, GEMV_X, GEMV_Y, HC_MAXN,
};

#[cfg(test)]
mod tests;
