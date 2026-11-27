//! The MiniMax-H3-Image checkpoint's JSON configs (diffusers layout):
//! `modular_model_index.json` (the sampling grid), `transformer_diffusers/`,
//! `vae_diffusers/` and `text_encoder/config.json`.

use crate::config::{num, nums, read, TeCfg};
use npubuild::st::Json;
use std::path::{Path, PathBuf};

/// Which Qwen3-VL hidden state conditions the transformer: `hidden_states[50]`,
/// the output of decoder layer 49, before the final norm.
pub const TE_LAYER: usize = 50;

/// The `t` the reference rows of an edit are held at in every step.
pub const REF_T: f32 = 0.999;

/// Modality tags: they index the transformer's AdaLN table.
pub const TAG_IMAGE: u8 = 0;
pub const TAG_TEXT: u8 = 1;
/// Modalities per timestep in the AdaLN table (image, text, audio).
pub const MODALITIES: usize = 3;

/// Canvas sides are multiples of this (VAE ratio x patch).
pub const CANVAS_MULTIPLE: usize = 32;

/// Prompt (and per-reference label) tokens kept.
pub const MAX_PROMPT_TOKENS: usize = 4096;

/// ImageNet normalization of the VAE's pixels.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// `transformer_diffusers/config.json` (`MiniMaxH3ImageTransformer2DModel`).
#[derive(Clone, Debug)]
pub struct DitCfg {
	pub layers: usize,
	pub refiner_layers: usize,
	pub heads: usize,
	pub head_dim: usize,
	pub hidden: usize,
	pub ffn: usize,
	pub in_ch: usize,
	/// Spatial patch (2): a row is `in_ch * patch^2` latent values.
	pub patch: usize,
	pub text_dim: usize,
	pub freq_dim: usize,
	pub temb_hidden: usize,
	pub temb_dim: usize,
	/// Rotary frequencies per axis; `2 * 3 * rope_freq` channels of a head rotate.
	pub rope_freq: usize,
	pub rope_theta: f64,
	pub eps: f32,
	pub qk_eps: f32,
	pub final_eps: f32,
	pub pdd_heads: usize,
}

impl DitCfg {
	/// Attention width (`heads * head_dim`, wider than `hidden`).
	pub fn inner(&self) -> usize {
		self.heads * self.head_dim
	}

	/// Values per packed row.
	pub fn patch_dim(&self) -> usize {
		self.in_ch * self.patch * self.patch
	}
}

/// `vae_diffusers/config.json` (`AutoencoderKLMiniMaxH3Image`).
#[derive(Clone, Debug)]
pub struct VaeCfg {
	pub latent: usize,
	pub block_out: Vec<usize>,
	pub layers_per_block: usize,
	pub spatial_down: Vec<usize>,
	pub temporal_down: Vec<usize>,
	pub groups: usize,
	pub norm_eps: f32,
	pub dec_layers: usize,
	pub dec_heads: usize,
	pub dec_head_dim: usize,
	pub dec_registers: usize,
	pub dec_ffn_mult: usize,
	pub dec_rope_theta: f64,
	pub dec_rope_ratio: f64,
	pub dec_eps: f32,
	pub mean: Vec<f32>,
	pub std: Vec<f32>,
}

impl VaeCfg {
	/// Spatial compression (16).
	pub fn ratio(&self) -> usize {
		self.spatial_down.iter().product()
	}

	pub fn dec_dim(&self) -> usize {
		self.dec_heads * self.dec_head_dim
	}
}

#[derive(Clone, Debug)]
pub struct Config {
	pub dir: PathBuf,
	/// `sample_sigmas`, terminal 0 included: one transformer pass per
	/// interval, step `i` through output head `i`.
	pub sigmas: Vec<f64>,
	pub dit: DitCfg,
	pub vae: VaeCfg,
	/// The text stack, cut to the [`TE_LAYER`] layers that run.
	pub te: TeCfg,
}

/// Whether `dir` holds a MiniMax-H3-Image checkpoint.
pub fn is_checkpoint(dir: &Path) -> bool {
	read(&dir.join("model_index.json"))
		.ok()
		.and_then(|j| j.get("_class_name").and_then(Json::str).map(str::to_string))
		.is_some_and(|c| c.starts_with("MiniMaxH3Image"))
}

impl Config {
	pub fn load(dir: &Path) -> Result<Config, String> {
		if !is_checkpoint(dir) {
			return Err(format!(
				"{}: not a MiniMax-H3-Image checkpoint",
				dir.display()
			));
		}
		let idx = read(&dir.join("modular_model_index.json"))?;
		let sigmas = nums(&idx, "sample_sigmas");
		if sigmas.len() < 2 || *sigmas.last().unwrap() != 0.0 {
			return Err("modular_model_index.json: no sample_sigmas ending at 0".into());
		}

		let t = read(&dir.join("transformer_diffusers/config.json"))?;
		let ps = nums(&t, "patch_size");
		if ps.len() != 3 || ps[0] != 1.0 || ps[1] != ps[2] {
			return Err("transformer: expected a (1, p, p) patch".into());
		}
		let u = |k: &str| num(&t, k).map(|x| x as usize);
		let dit = DitCfg {
			layers: u("num_layers")?,
			refiner_layers: u("num_refiner_layers")?,
			heads: u("num_attention_heads")?,
			head_dim: u("attention_head_dim")?,
			hidden: u("hidden_size")?,
			ffn: u("ffn_dim")?,
			in_ch: u("in_channels")?,
			patch: ps[1] as usize,
			text_dim: u("text_dim")?,
			freq_dim: u("freq_dim")?,
			temb_hidden: u("time_embed_hidden_dim")?,
			temb_dim: u("time_embed_dim")?,
			rope_freq: u("rope_freq_dim")?,
			rope_theta: num(&t, "rope_theta")?,
			eps: num(&t, "norm_eps")? as f32,
			qk_eps: num(&t, "qk_norm_eps")? as f32,
			final_eps: num(&t, "final_norm_eps")? as f32,
			pdd_heads: u("num_pdd_heads")?,
		};
		if dit.pdd_heads != sigmas.len() - 1 {
			return Err(format!(
				"transformer: {} output heads for {} sampling steps",
				dit.pdd_heads,
				sigmas.len() - 1
			));
		}
		if 6 * dit.rope_freq > dit.head_dim {
			return Err("transformer: rotary channels exceed the head".into());
		}

		let v = read(&dir.join("vae_diffusers/config.json"))?;
		let uv = |k: &str| num(&v, k).map(|x| x as usize);
		let us = |k: &str| nums(&v, k).iter().map(|&x| x as usize).collect::<Vec<_>>();
		let vae = VaeCfg {
			latent: uv("latent_channels")?,
			block_out: us("block_out_channels"),
			layers_per_block: uv("layers_per_block")?,
			spatial_down: us("spatial_downsample_factors"),
			temporal_down: us("temporal_downsample_factors"),
			groups: uv("norm_num_groups")?,
			norm_eps: num(&v, "norm_eps")? as f32,
			dec_layers: uv("decoder_num_layers")?,
			dec_heads: uv("decoder_num_attention_heads")?,
			dec_head_dim: uv("decoder_attention_head_dim")?,
			dec_registers: uv("decoder_num_register_tokens")?,
			dec_ffn_mult: uv("decoder_ffn_mult")?,
			dec_rope_theta: num(&v, "decoder_rope_theta")?,
			dec_rope_ratio: num(&v, "decoder_rope_dim_ratio")?,
			dec_eps: num(&v, "decoder_norm_eps")? as f32,
			mean: nums(&v, "latents_mean").iter().map(|&x| x as f32).collect(),
			std: nums(&v, "latents_std").iter().map(|&x| x as f32).collect(),
		};
		if vae.block_out.len() != vae.spatial_down.len()
			|| vae.block_out.len() != vae.temporal_down.len()
			|| vae.mean.len() != vae.latent
			|| vae.std.len() != vae.latent
			|| vae.latent != dit.in_ch
		{
			return Err("vae: inconsistent config".into());
		}
		if v.get("spatial_padding_mode")
			.and_then(Json::str)
			.unwrap_or("reflect")
			!= "reflect"
		{
			return Err("vae: only reflect padding is implemented".into());
		}

		let mut te = TeCfg::load(&dir.join("text_encoder/config.json"))?;
		if te.layers <= TE_LAYER {
			return Err(format!(
				"text_encoder: {} layers, the transformer reads hidden_states[{TE_LAYER}]",
				te.layers
			));
		}
		te.layers = TE_LAYER;
		if te.dim != dit.text_dim {
			return Err("text_encoder: width differs from the transformer's text_dim".into());
		}
		Ok(Config {
			dir: dir.to_path_buf(),
			sigmas,
			dit,
			vae,
			te,
		})
	}

	/// The sampling grid as the released sampler steps on it: every sigma
	/// rounded through its f32 timestep `1 - sigma` and back (below 0.5
	/// that moves a sigma by an ulp). Returns `(timesteps, sigmas)`.
	pub fn schedule(&self) -> (Vec<f32>, Vec<f32>) {
		let ts: Vec<f32> = self.sigmas[..self.sigmas.len() - 1]
			.iter()
			.map(|&s| (1.0 - s) as f32)
			.collect();
		let mut sg: Vec<f32> = ts.iter().map(|&t| 1.0 - t).collect();
		sg.push(0.0);
		(ts, sg)
	}
}

/// The canvas a `height x width` request is generated on: below `min_area`
/// pixels both sides scale by `sqrt(min_area / (h w))`; either way they
/// round to [`CANVAS_MULTIPLE`] (Python's half-to-even `round`).
pub fn canvas(height: usize, width: usize, min_area: usize) -> (usize, usize) {
	let m = CANVAS_MULTIPLE as f64;
	let scale = (min_area as f64 / (width * height) as f64).sqrt().max(1.0);
	let side =
		|x: usize| ((round_half_even(x as f64 * scale / m) * m) as usize).max(CANVAS_MULTIPLE);
	(side(height), side(width))
}

/// Python's `round` (ties to even).
pub fn round_half_even(x: f64) -> f64 {
	let r = x.round();
	if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
		r - x.signum()
	} else {
		r
	}
}

/// The size a reference is resized to for one consumer: a short edge
/// within `[lo, hi]` keeps the image, otherwise it scales to the nearer
/// bound with both sides rounded to [`CANVAS_MULTIPLE`].
pub fn bound_short_edge(width: usize, height: usize, lo: usize, hi: usize) -> (usize, usize) {
	let short = width.min(height);
	if (lo..=hi).contains(&short) {
		return (width, height);
	}
	let scale = (if short < lo { lo } else { hi }) as f64 / short as f64;
	let m = CANVAS_MULTIPLE as f64;
	let side =
		|x: usize| ((round_half_even(x as f64 * scale / m) * m) as usize).max(CANVAS_MULTIPLE);
	(side(width), side(height))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn canvas_matches_the_pipeline() {
		// get_image_canvas_size: 1024x1024 at the default 4 MP -> 2048x2048
		assert_eq!(canvas(1024, 1024, 2048 * 2048), (2048, 2048));
		assert_eq!(canvas(3552, 4736, 2048 * 2048), (3552, 4736));
		assert_eq!(canvas(512, 512, 1024), (512, 512));
		assert_eq!(canvas(500, 700, 1024), (512, 704));
		assert_eq!(round_half_even(2.5), 2.0);
		assert_eq!(round_half_even(3.5), 4.0);
		assert_eq!(round_half_even(-2.5), -2.0);
	}

	#[test]
	fn reference_bounds() {
		assert_eq!(bound_short_edge(1500, 1200, 1024, 2048), (1500, 1200));
		assert_eq!(bound_short_edge(800, 600, 1024, 2048), (1376, 1024));
		assert_eq!(bound_short_edge(1500, 1200, 1024, 1024), (1280, 1024));
	}
}
