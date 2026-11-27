//! The checkpoints' JSON config helpers (diffusers layout) and the
//! Qwen3-VL text encoder's `text_config`, shared by the image models.

use npubuild::st::{parse_json, Json};
use std::path::Path;

pub(crate) fn read(p: &Path) -> Result<Json, String> {
	parse_json(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
}

pub(crate) fn num(j: &Json, k: &str) -> Result<f64, String> {
	j.get(k)
		.and_then(Json::num)
		.ok_or_else(|| format!("config: no number {k}"))
}

pub(crate) fn nums(j: &Json, k: &str) -> Vec<f64> {
	j.get(k)
		.map_or(&[][..], Json::arr)
		.iter()
		.filter_map(Json::num)
		.collect()
}

pub(crate) fn flag(j: &Json, k: &str) -> bool {
	matches!(j.get(k), Some(Json::Bool(true)))
}

/// `text_encoder/config.json` `text_config`.
#[derive(Clone, Debug)]
pub struct TeCfg {
	pub layers: usize,
	pub dim: usize,
	pub heads: usize,
	pub kv_heads: usize,
	pub head_dim: usize,
	pub ffn: usize,
	pub eps: f32,
	pub theta: f64,
	/// Rotary pairs per MRoPE axis (time, height, width), interleaved.
	pub mrope: [usize; 3],
}

impl TeCfg {
	/// `text_config` of a Qwen3-VL `text_encoder/config.json`.
	pub fn load(path: &Path) -> Result<TeCfg, String> {
		let tc = read(path)?;
		let tt = tc
			.get("text_config")
			.ok_or("text_encoder: no text_config")?;
		Ok(TeCfg {
			layers: num(tt, "num_hidden_layers")? as usize,
			dim: num(tt, "hidden_size")? as usize,
			heads: num(tt, "num_attention_heads")? as usize,
			kv_heads: num(tt, "num_key_value_heads")? as usize,
			head_dim: num(tt, "head_dim")? as usize,
			ffn: num(tt, "intermediate_size")? as usize,
			eps: num(tt, "rms_norm_eps")? as f32,
			theta: num(tt, "rope_theta")?,
			mrope: {
				let m: Vec<usize> = tt
					.get("rope_scaling")
					.map_or_else(Vec::new, |r| nums(r, "mrope_section"))
					.iter()
					.map(|&x| x as usize)
					.collect();
				if m.len() == 3 {
					[m[0], m[1], m[2]]
				} else {
					[24, 20, 20]
				}
			},
		})
	}
}
