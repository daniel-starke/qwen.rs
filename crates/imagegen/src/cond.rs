//! Condition images of an edit and the text encoder's vision tower over
//! them ([`vision_rows`]), shared by the image models.
//!
//! [`Cond::new`] prepares an image as Qwen-Image's pipeline does: converted
//! to RGBA and resized once, with Pillow's Lanczos (on the premultiplied
//! form, as Pillow resizes RGBA), to about `resolution`^2 pixels at its own
//! aspect, sides multiples of 32 ([`dims`]). That one copy feeds both of its
//! encoders (the VAE's input: [`crate::qimage::cond`]); the vision tower
//! reads it composited over white (Pillow's `paste` with the alpha as
//! mask), through Qwen3-VL's processor, whose own resize is the identity at
//! these sizes. MiniMax-H3-Image builds its [`Cond`]s at its own sizes.

use crate::te::VisionRows;
use imgio::pil;
use npubuild::st::{parse_json, Checkpoint, Json};
use std::path::Path;

/// Python's `round` (half to even).
fn round_even(x: f64) -> f64 {
	let r = x.round();
	if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
		r - x.signum()
	} else {
		r
	}
}

/// `calculate_dimensions(area, ratio)`: width x height of `area` pixels at
/// aspect `ratio`, each side rounded (half to even) to a multiple of 32.
pub fn dims(area: f64, ratio: f64) -> (usize, usize) {
	let w = (area * ratio).sqrt();
	let h = w / ratio;
	(
		(round_even(w / 32.0) * 32.0) as usize,
		(round_even(h / 32.0) * 32.0) as usize,
	)
}

/// One condition image after the pipeline's resize.
pub struct Cond {
	/// The original size (its aspect decides the target's).
	pub orig: (usize, usize),
	pub w: usize,
	pub h: usize,
	/// RGBA, `h * w * 4` bytes.
	pub rgba: Vec<u8>,
}

impl Cond {
	/// Decode `path` and resize it for `resolution`.
	pub fn load(path: &Path, resolution: usize) -> Result<Cond, String> {
		let (w, h, rgba) = imgio::load_rgba(&path.to_string_lossy())?;
		Cond::new(w, h, &rgba, resolution)
	}

	pub fn new(w: usize, h: usize, rgba: &[u8], resolution: usize) -> Result<Cond, String> {
		if w == 0 || h == 0 || rgba.len() != w * h * 4 {
			return Err(format!(
				"condition image: {w}x{h} with {} bytes",
				rgba.len()
			));
		}
		let r = resolution as f64;
		let (dw, dh) = dims(r * r, w as f64 / h as f64);
		if dw == 0 || dh == 0 {
			return Err(format!(
				"condition image: {w}x{h} is too narrow for {resolution}^2 pixels"
			));
		}
		Ok(Cond {
			orig: (w, h),
			w: dw,
			h: dh,
			rgba: pil::resize_rgba(rgba, w, h, dw, dh, pil::Filter::Lanczos),
		})
	}

	/// The vision tower's copy: alpha composited over white.
	pub fn te_image(&self) -> imgio::Image {
		imgio::Image::from_rgb(self.w, self.h, pil::over_white(&self.rgba))
			.expect("over_white keeps the size")
	}
}

/// The processor's pixel bounds (`processor_config.json`
/// `image_processor.size`: shortest / longest edge are pixel counts).
fn pixel_bounds(dir: &Path) -> (usize, usize) {
	let j = std::fs::read(dir.join("processor/processor_config.json"))
		.ok()
		.and_then(|b| parse_json(&b).ok());
	let size = |k: &str| {
		j.as_ref()
			.and_then(|j| j.get("image_processor"))
			.and_then(|p| p.get("size"))
			.and_then(|s| s.get(k))
			.and_then(Json::num)
			.map(|v| v as usize)
	};
	(
		size("shortest_edge").unwrap_or(65536),
		size("longest_edge").unwrap_or(16777216),
	)
}

/// The tower's attention on the cores, for the NPU tower.
struct CoreAttn(vision::VisionConfig);

impl npu::vis::VisAttn for CoreAttn {
	fn attend(
		&mut self,
		_n: usize,
		qkv: &[u8],
		hw: &[(u32, u32)],
		out: &mut [u8],
	) -> Result<(), String> {
		vision::tower::attend_f16(&self.0, hw, qkv, out);
		Ok(())
	}
	fn name(&self) -> String {
		"cores".into()
	}
}

/// Run the text encoder's vision tower (with its deepstack mergers) over
/// the conditions; `ck` is the text encoder's checkpoint. With `npu` (an
/// `image.npu`'s `vision` set) the per-patch work runs on the NPU, the
/// attention on the cores; else all of it on the cores.
pub fn vision_rows(
	dir: &Path,
	ck: &Checkpoint,
	conds: &[Cond],
	mut npu: Option<&mut npu::vis::NpuVision>,
) -> Result<VisionRows, String> {
	let m = match npu {
		Some(_) => None,
		None => Some(npubuild::vis::model_deepstack(ck)?),
	};
	let cfg = match &m {
		Some(m) => m.cfg.clone(),
		None => npubuild::vis::config_any(ck)?,
	};
	let ndeep = match (&m, npu.as_deref()) {
		(Some(m), _) => m.deepstack.len(),
		(None, Some(n)) => n.cfg.deep.len(),
		_ => 0,
	};
	let c = &cfg;
	let per_token = (c.patch_size * c.merge_size).pow(2);
	let (min_px, max_px) = pixel_bounds(dir);
	let mut rows = VisionRows {
		embeds: Vec::new(),
		deep: vec![Vec::new(); ndeep],
		grids: Vec::new(),
	};
	for cd in conds {
		let grid = vision::smart_resize(
			cd.w,
			cd.h,
			c.patch_size,
			c.merge_size,
			min_px / per_token,
			max_px / per_token,
		)?;
		let patches = vision::patchify(
			&cd.te_image(),
			&grid,
			c.patch_size,
			c.merge_size,
			c.mean,
			c.std,
		);
		let (e, deep) = match (&m, npu.as_deref_mut()) {
			(Some(m), _) => vision::tower::forward_deep(m, &patches, &grid),
			(None, Some(n)) => {
				let pos = vision::tower::position_embedding(c, &n.pos_w, &grid);
				let hw = vision::patch_positions(&grid, c.merge_size);
				n.encode_deep(&patches, &pos, &hw, &mut CoreAttn(c.clone()))?
			}
			_ => unreachable!(),
		};
		rows.embeds.extend(e);
		for (d, f) in rows.deep.iter_mut().zip(deep) {
			d.extend(f);
		}
		rows.grids.push((grid.mh, grid.mw));
	}
	Ok(rows)
}
