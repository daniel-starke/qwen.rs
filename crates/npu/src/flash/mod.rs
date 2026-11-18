//! Qwen3.8-Flash-Next (`qwen4_exp`): the host-side pieces the builder
//! (`npubuild`) and the engine share.
//!
//! - [`ngram`]: the PLE layer's hashed n-gram ids and the table on disk.
//! - [`cpuexp`]: a decode step's routed experts on the CPU cores.
//! - [`engine`]: the model on the NPU (prefill graphs, expert jobs).
//! - [`idx`]: a prefill attention layer's block indexer scores on the NPU.
//! - [`gpudec`]: a decode step's dense layers on a discrete GPU.
//! - [`mtp`]: the MTP head (drafts for the verify steps) on the cores.

pub mod cpuexp;
pub mod engine;
pub mod far;
pub mod gpudec;
pub mod idx;
mod kvcol;
pub mod mtp;
pub mod ngram;

/// Model dimensions (`config.json` of `Qwen/Qwen3.8-Flash-Next`).
pub const W: usize = 2560;
/// Hyper-connection streams; the residual is `[HC][W]`.
pub const HC: usize = 4;
pub const HC_RANK: usize = 320;
pub const NL: usize = 48;
/// Tokens of the widest expert slot of a prefill expert pass (the wide
/// expert graphs' `c`, `offload::EXP_SLOT`).
pub const EXP_SLOT: usize = 64;
pub const VOCAB: usize = 248320;
pub const EOS: u32 = 248044;
/// Layer index (0-based) of the PLE layer.
pub const PLE_LAYER: usize = 1;
/// Scale of the delta recurrence's output `o` going into a prefill set's
/// `post` graph (the gated norm absorbs it; `shave/hcmix.sk` mode G
/// flushes subnormal f16 inputs, so it must keep `o` out of them).
pub const OSCALE_P: usize = 64;
