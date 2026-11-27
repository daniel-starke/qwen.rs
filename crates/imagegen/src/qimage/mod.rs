//! Qwen-Image-2.1 (`docs/image.md`): text to image and edits with up to
//! [`pipeline::MAX_IMAGES`] condition images.
//!
//! M1: the CPU reference of the whole text-to-image pipeline in f32: the
//! text encoder's prompt ([`te`], on the shared stack [`crate::te`]), the
//! transformer ([`dit`]), the VAE ([`vae`]), the Turbo schedule ([`sched`])
//! and torch-compatible initial noise ([`crate::rng`]), driven by
//! [`pipeline`]. Weights are read from the BF16 checkpoint one layer at a
//! time (`npubuild::st::Checkpoint`), never a whole model as f32.
//!
//! M2: edits (condition images through the text encoder's vision tower with
//! deepstack, [`crate::cond`], and the VAE encoder, [`cond`], placed at
//! their slots of the transformer's sequence).
//!
//! The NPU backend (`image.npu` of `cli QWEN_IMAGE_DIR --npu-build OUT`):
//! the transformer ([`npudit`]), the VAE's convolutions ([`npuvae`]) and
//! the activation ranges of its build ([`npucal`]).

pub mod cond;
pub mod config;
pub mod dit;
pub mod npucal;
pub mod npudit;
pub mod npuvae;
pub mod pipeline;
pub mod sched;
pub mod te;
pub mod vae;

pub use config::Config;
pub use pipeline::{generate, Model, MAX_IMAGES};
