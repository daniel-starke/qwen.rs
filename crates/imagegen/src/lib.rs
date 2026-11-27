//! Image generation: Qwen-Image-2.1 ([`qimage`], `docs/image.md`) and
//! MiniMax-H3-Image ([`h3`], `docs/h3image.md`), one model per path
//! ([`model`]), behind one job interface ([`job`]).
//!
//! Shared by the models: the Qwen3-VL text stack both use as text encoder
//! ([`te`], its config in [`config`]), condition images and the vision
//! tower over them ([`cond`]), row-wise f32 ops and the quantized matmul
//! ([`ops`]), torch-compatible noise ([`rng`]), the converted-weight cache
//! ([`wcache`]), the NPU and its f16 host buffers ([`npuhost`]) and the
//! reference dumps of the gates ([`refdump`]).

pub mod cond;
pub mod config;
pub mod h3;
pub mod job;
pub mod model;
pub mod npuhost;
pub mod ops;
pub mod qimage;
pub mod refdump;
pub mod rng;
pub mod te;
pub mod wcache;

pub use job::{steps_done, ImageSrc, Options, Output, MAX_STEPS};
pub use model::{generate, ImageModel, Info, Kind};
pub use ops::WFmt;
