//! The checkpoint's JSON configs (diffusers layout): `model_index.json`,
//! `transformer/`, `vae/`, `scheduler/` and `text_encoder/config.json`.

use npubuild::st::{parse_json, Json};
use std::path::{Path, PathBuf};

fn read(p: &Path) -> Result<Json, String> {
	parse_json(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
}

fn num(j: &Json, k: &str) -> Result<f64, String> {
	j.get(k)
		.and_then(Json::num)
		.ok_or_else(|| format!("config: no number {k}"))
}

fn nums(j: &Json, k: &str) -> Vec<f64> {
	j.get(k)
		.map_or(&[][..], Json::arr)
		.iter()
		.filter_map(Json::num)
		.collect()
}

fn flag(j: &Json, k: &str) -> bool {
	matches!(j.get(k), Some(Json::Bool(true)))
}

/// `transformer/config.json`.
#[derive(Clone, Debug)]
pub struct DitCfg {
	pub layers: usize,
	pub heads: usize,
	pub head_dim: usize,
	pub in_ch: usize,
	pub out_ch: usize,
	pub ctx_dim: usize,
	pub mlp_ratio: usize,
	pub axes: [usize; 3],
	pub eps: f32,
	pub causal_condition: bool,
}

impl DitCfg {
	pub fn dim(&self) -> usize {
		self.heads * self.head_dim
	}
}

/// `vae/config.json` (the decoder side).
#[derive(Clone, Debug)]
pub struct VaeCfg {
	pub z: usize,
	pub in_ch: usize,
	pub enc_dim: usize,
	/// Encoder down blocks with a temporal factor 2 (`temperal_downsample`).
	pub t_down: Vec<bool>,
	pub dec_dim: usize,
	pub dim_mult: Vec<usize>,
	pub res_blocks: usize,
	pub out_ch: usize,
	pub scale: usize,
	pub mean: Vec<f32>,
	pub std: Vec<f32>,
}

/// `scheduler/scheduler_config.json` (FlowMatchEulerDiscrete).
#[derive(Clone, Debug)]
pub struct SchedCfg {
	pub train_steps: f64,
	pub shift: f64,
	pub dynamic: bool,
	pub base_seq: f64,
	pub max_seq: f64,
	pub base_shift: f64,
	pub max_shift: f64,
	pub shift_terminal: Option<f64>,
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

/// The whole model's configuration.
#[derive(Clone, Debug)]
pub struct Config {
	pub dir: PathBuf,
	/// `model_index.json` `sample_sigmas` (Turbo): the default schedule.
	pub sample_sigmas: Option<Vec<f64>>,
	pub dit: DitCfg,
	pub vae: VaeCfg,
	pub sched: SchedCfg,
	pub te: TeCfg,
}

impl Config {
	pub fn load(dir: &Path) -> Result<Config, String> {
		let idx = read(&dir.join("model_index.json"))?;
		let class = idx.get("_class_name").and_then(Json::str).unwrap_or("");
		if !class.starts_with("QwenImage21") {
			return Err(format!(
				"{}: pipeline {class:?} is not QwenImage21*",
				dir.display()
			));
		}
		let sample_sigmas = idx
			.get("sample_sigmas")
			.map(|_| nums(&idx, "sample_sigmas"));

		let t = read(&dir.join("transformer/config.json"))?;
		let axes = nums(&t, "axes_dims_rope");
		if axes.len() != 3 || num(&t, "patch_size")? != 1.0 {
			return Err("transformer: expected 3 RoPE axes and patch size 1".into());
		}
		let dit = DitCfg {
			layers: num(&t, "num_layers")? as usize,
			heads: num(&t, "num_attention_heads")? as usize,
			head_dim: num(&t, "attention_head_dim")? as usize,
			in_ch: num(&t, "in_channels")? as usize,
			out_ch: num(&t, "out_channels")? as usize,
			ctx_dim: num(&t, "context_in_dim")? as usize,
			mlp_ratio: num(&t, "mlp_ratio")? as usize,
			axes: [axes[0] as usize, axes[1] as usize, axes[2] as usize],
			eps: num(&t, "eps")? as f32,
			causal_condition: flag(&t, "causal_condition"),
		};

		let v = read(&dir.join("vae/config.json"))?;
		let vae = VaeCfg {
			z: num(&v, "z_dim")? as usize,
			in_ch: num(&v, "in_channels")? as usize,
			enc_dim: num(&v, "base_dim")? as usize,
			t_down: v
				.get("temperal_downsample")
				.map_or(&[][..], Json::arr)
				.iter()
				.map(|b| matches!(b, Json::Bool(true)))
				.collect(),
			dec_dim: v
				.get("decoder_base_dim")
				.and_then(Json::num)
				.map_or_else(|| num(&v, "base_dim"), Ok)? as usize,
			dim_mult: nums(&v, "dim_mult").iter().map(|&x| x as usize).collect(),
			res_blocks: num(&v, "num_res_blocks")? as usize,
			out_ch: num(&v, "out_channels")? as usize,
			scale: num(&v, "scale_factor_spatial")? as usize,
			mean: nums(&v, "latents_mean").iter().map(|&x| x as f32).collect(),
			std: nums(&v, "latents_std").iter().map(|&x| x as f32).collect(),
		};
		if !flag(&v, "is_residual") || !nums(&v, "attn_scales").is_empty() {
			return Err("vae: expected the residual decoder without attention scales".into());
		}

		let s = read(&dir.join("scheduler/scheduler_config.json"))?;
		let sched = SchedCfg {
			train_steps: num(&s, "num_train_timesteps")?,
			shift: num(&s, "shift").unwrap_or(1.0),
			dynamic: flag(&s, "use_dynamic_shifting"),
			base_seq: num(&s, "base_image_seq_len").unwrap_or(256.0),
			max_seq: num(&s, "max_image_seq_len").unwrap_or(4096.0),
			base_shift: num(&s, "base_shift").unwrap_or(0.5),
			max_shift: num(&s, "max_shift").unwrap_or(1.15),
			shift_terminal: s.get("shift_terminal").and_then(Json::num),
		};
		if s.get("time_shift_type")
			.and_then(Json::str)
			.unwrap_or("exponential")
			!= "exponential"
		{
			return Err("scheduler: only the exponential time shift is implemented".into());
		}

		let tc = read(&dir.join("text_encoder/config.json"))?;
		let tt = tc
			.get("text_config")
			.ok_or("text_encoder: no text_config")?;
		let te = TeCfg {
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
		};
		Ok(Config {
			dir: dir.to_path_buf(),
			sample_sigmas,
			dit,
			vae,
			sched,
			te,
		})
	}
}
