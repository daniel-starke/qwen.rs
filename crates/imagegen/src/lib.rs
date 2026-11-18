//! Qwen-Image-2.1 image generation (`docs/image.md`).
//!
//! M1: the CPU reference of the whole text-to-image pipeline in f32: the
//! text encoder ([`te`]), transformer ([`dit`]), VAE decoder ([`vae`]),
//! the Turbo schedule ([`sched`]) and torch-compatible initial noise
//! ([`rng`]), driven by [`pipeline`]. Weights are read from the BF16
//! checkpoint one layer at a time (`npubuild::st::Checkpoint`), never a
//! whole model as f32.
//!
//! M2: edits ([`cond`]: condition images through the text encoder's vision
//! tower with deepstack and the VAE encoder, placed at their slots of the
//! transformer's sequence).

pub mod cond;
pub mod config;
pub mod dit;
pub mod npucal;
pub mod npudit;
pub mod npuvae;
pub mod ops;
pub mod pipeline;
pub mod refdump;
pub mod rng;
pub mod sched;
pub mod te;
pub mod vae;

pub use config::Config;
pub use ops::WFmt;
pub use pipeline::{generate, steps_done, ImageSrc, Model, Options, Output, MAX_IMAGES, MAX_STEPS};
