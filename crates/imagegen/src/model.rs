//! The image models behind one interface: the checkpoint (or `image.npu`)
//! path picks the model (`model_index.json`'s class), and [`Info`] holds
//! what the interfaces (CLI, MCP, OpenAI Images) need to know before it is
//! loaded: its name, sizes, reference count and knobs.

use crate::job::{Options, Output};
use crate::qimage::pipeline;
use crate::{h3, WFmt};
use std::path::Path;

/// Which model a path holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
	/// Qwen-Image-2.1 (`docs/image.md`): a checkpoint or an `image.npu`.
	QwenImage,
	/// MiniMax-H3-Image (`docs/h3image.md`): a checkpoint.
	H3,
}

/// What the interfaces show and check per model.
#[derive(Clone, Copy, Debug)]
pub struct Info {
	/// The id the OpenAI Images routes report.
	pub id: &'static str,
	/// The name in tool descriptions and logs.
	pub name: &'static str,
	/// Most reference images an edit takes.
	pub max_images: usize,
	/// Pixels of the default canvas (and of a size given as an aspect).
	pub area: usize,
	/// Explicit sides: multiples of 32 within this range.
	pub sides: (usize, usize),
	/// Most pixels of an explicit size.
	pub max_area: usize,
	/// Whether a job may choose its denoising steps.
	pub steps: bool,
	/// Whether the output can keep an alpha channel.
	pub alpha: bool,
}

impl Kind {
	/// The model at `path`, if it is one.
	pub fn of(path: &Path) -> Option<Kind> {
		if h3::pipeline::Model::is_model(path) {
			Some(Kind::H3)
		} else if pipeline::Model::is_model(path) {
			Some(Kind::QwenImage)
		} else {
			None
		}
	}

	pub fn info(self) -> Info {
		match self {
			Kind::QwenImage => Info {
				id: "qwen-image-2.1-turbo",
				name: "Qwen-Image-2.1",
				max_images: pipeline::MAX_IMAGES,
				area: 1024 * 1024,
				sides: (256, 2048),
				max_area: 2048 * 2048,
				steps: true,
				alpha: true,
			},
			Kind::H3 => Info {
				id: "minimax-h3-image",
				name: "MiniMax-H3-Image",
				max_images: h3::pipeline::MAX_IMAGES,
				area: h3::pipeline::MIN_AREA,
				sides: (256, 8192),
				max_area: h3::pipeline::MAX_AREA,
				steps: false,
				alpha: false,
			},
		}
	}
}

impl Info {
	/// Check an explicit size against [`Self::sides`] and [`Self::max_area`].
	pub fn check_size(&self, w: usize, h: usize) -> Result<(usize, usize), String> {
		let (lo, hi) = self.sides;
		if !w.is_multiple_of(32)
			|| !h.is_multiple_of(32)
			|| !(lo..=hi).contains(&w)
			|| !(lo..=hi).contains(&h)
		{
			return Err(format!(
				"size {w}x{h}: sides must be multiples of 32 between {lo} and {hi}"
			));
		}
		if w * h > self.max_area {
			return Err(format!(
				"size {w}x{h}: at most {:.1} megapixels",
				self.max_area as f64 / 1048576.0
			));
		}
		Ok((w, h))
	}
}

/// A loaded image model.
pub enum ImageModel {
	QwenImage(Box<pipeline::Model>),
	H3(Box<h3::pipeline::Model>),
}

impl ImageModel {
	/// True when `path` is something [`Self::load`] takes.
	pub fn is_model(path: &Path) -> bool {
		Kind::of(path).is_some()
	}

	/// Load the model `path` holds; `progress(stage, step, steps)` per layer.
	pub fn load(
		path: &Path,
		fmt: WFmt,
		progress: impl FnMut(&str, usize, usize),
	) -> Result<ImageModel, String> {
		match Kind::of(path) {
			Some(Kind::H3) => Ok(ImageModel::H3(Box::new(h3::pipeline::Model::load(
				path, fmt, progress,
			)?))),
			Some(Kind::QwenImage) => Ok(ImageModel::QwenImage(Box::new(pipeline::Model::load(
				path, fmt, progress,
			)?))),
			None => Err(format!(
				"{}: not an image model (a Qwen-Image or MiniMax-H3-Image checkpoint, or an image.npu)",
				path.display()
			)),
		}
	}

	pub fn kind(&self) -> Kind {
		match self {
			ImageModel::QwenImage(_) => Kind::QwenImage,
			ImageModel::H3(_) => Kind::H3,
		}
	}

	pub fn info(&self) -> Info {
		self.kind().info()
	}

	pub fn generate(
		&self,
		o: &Options,
		progress: impl FnMut(&str, usize, usize),
	) -> Result<Output, String> {
		match self {
			ImageModel::QwenImage(m) => m.generate(o, progress),
			ImageModel::H3(m) => m.generate(o, progress),
		}
	}

	/// Free device memory between jobs (the NPU graphs of an `image.npu`).
	pub fn release_devices(&self) {
		match self {
			ImageModel::QwenImage(m) => m.release_devices(),
			ImageModel::H3(m) => m.release_devices(),
		}
	}

	/// Bytes of the resident weights (text encoder, transformer).
	pub fn resident_bytes(&self) -> (usize, usize) {
		match self {
			ImageModel::QwenImage(m) => m.resident_bytes(),
			ImageModel::H3(m) => m.resident_bytes(),
		}
	}

	/// What runs the transformer, for the log.
	pub fn backend(&self) -> &'static str {
		match self {
			ImageModel::QwenImage(m) => m.backend(),
			ImageModel::H3(m) => m.backend(),
		}
	}

	/// Seconds the load took.
	pub fn load_secs(&self) -> f64 {
		match self {
			ImageModel::QwenImage(m) => m.load_secs,
			ImageModel::H3(m) => m.load_secs,
		}
	}
}

/// Load the model `path` holds in the format `QWEN_IMAGE_WEIGHTS` names
/// (default `q6k`) and generate one image.
pub fn generate(
	path: &Path,
	o: &Options,
	mut progress: impl FnMut(&str, usize, usize),
) -> Result<Output, String> {
	ImageModel::load(path, WFmt::from_env(), &mut progress)?.generate(o, progress)
}
