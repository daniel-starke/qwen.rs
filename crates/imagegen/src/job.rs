//! A job's interface, the same for every image model: the request
//! ([`Options`]), its condition images ([`ImageSrc`]), the result
//! ([`Output`]) and the progress reporting.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// A condition image: a file, or decoded RGBA pixels `(w, h, bytes)`.
#[derive(Clone)]
pub enum ImageSrc {
	Path(PathBuf),
	Rgba(usize, usize, Vec<u8>),
}

impl From<PathBuf> for ImageSrc {
	fn from(p: PathBuf) -> ImageSrc {
		ImageSrc::Path(p)
	}
}

pub struct Options {
	pub prompt: String,
	/// Pixels, multiples of 32; 0 = `resolution`^2 pixels at the last
	/// condition image's aspect (square without one).
	pub width: usize,
	pub height: usize,
	pub seed: u64,
	/// Denoising steps, 0 = the model's default (Qwen-Image Turbo: its 8
	/// saved sigmas plus a refinement step = 9; 8 = as saved; others
	/// resample the curve; without saved sigmas 40,
	/// [`crate::qimage::sched::Schedule::for_steps`]). A model with a fixed
	/// count refuses any other.
	pub steps: usize,
	/// Initial noise `[h w][64]` (token-major) instead of the seed's
	/// (Qwen-Image only).
	pub noise: Option<Vec<f32>>,
	/// Condition images (an edit), in order.
	pub images: Vec<ImageSrc>,
	/// Qwen-Image's `output_resolution`: condition images are resized to
	/// about this squared many pixels, as is a target without a size.
	pub resolution: usize,
	/// Set from another thread to stop the job at the next stage or step.
	pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for Options {
	fn default() -> Options {
		Options {
			prompt: String::new(),
			width: 0,
			height: 0,
			seed: 0,
			steps: 0,
			noise: None,
			images: Vec::new(),
			resolution: 1024,
			cancel: None,
		}
	}
}

pub struct Output {
	pub width: usize,
	pub height: usize,
	/// RGBA, 8 bits per channel.
	pub rgba: Vec<u8>,
	pub seed: u64,
	/// Seconds: text encoder (with the vision tower), VAE encoder,
	/// transformer (prefix + denoising), VAE decoder.
	pub secs: [f64; 4],
}

/// The process' working set and private bytes in GB (`QWEN_IMAGE_TIMING`
/// lines).
#[cfg(windows)]
pub fn mem_gb() -> (f64, f64) {
	#[repr(C)]
	#[derive(Default)]
	struct Pmc {
		cb: u32,
		faults: u32,
		peak_ws: usize,
		ws: usize,
		qpp: usize,
		qp: usize,
		qpnp: usize,
		qnp: usize,
		pagefile: usize,
		peak_pagefile: usize,
		private: usize,
	}
	extern "system" {
		fn GetCurrentProcess() -> isize;
		fn K32GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
	}
	let mut c = Pmc {
		cb: std::mem::size_of::<Pmc>() as u32,
		..Default::default()
	};
	// SAFETY: a plain query into a struct of the documented layout
	unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
	(
		c.ws as f64 / (1u64 << 30) as f64,
		c.private as f64 / (1u64 << 30) as f64,
	)
}

#[cfg(not(windows))]
pub fn mem_gb() -> (f64, f64) {
	(0.0, 0.0)
}

/// The most denoising steps a job may ask for.
pub const MAX_STEPS: usize = 50;

/// Steps finished when `progress(stage, i, _)` is reported: a `load ...`
/// stage reports after layer `i`, `denoise` before step `i` (1-based).
pub fn steps_done(stage: &str, i: usize) -> usize {
	if stage == "denoise" {
		i.saturating_sub(1)
	} else {
		i
	}
}
