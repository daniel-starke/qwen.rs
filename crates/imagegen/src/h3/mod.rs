//! MiniMax-H3-Image (`docs/h3image.md`): text to image and edits with up to
//! 16 reference images, on the same interfaces as Qwen-Image.
//!
//! M1: the f32 reference of every stage, weights read from the BF16 / F32
//! checkpoint one layer at a time: the text encoder (the Qwen3-VL stack of
//! [`crate::te`], 50 of 64 layers), the transformer ([`dit`]), the VAE
//! ([`vae`]), the packed sequence ([`layout`]). Gates: per-layer checks
//! against `scripts/h3refdump.py` dumps (`tests/h3ref.rs`).

pub mod config;
pub mod dit;
pub mod layout;
pub mod npucal;
pub mod npudit;
pub mod npute;
pub mod npuvae;
pub mod pipeline;
pub mod te;
pub mod tecal;
pub mod vae;

pub use config::Config;
