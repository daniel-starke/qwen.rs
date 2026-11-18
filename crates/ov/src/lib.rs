//! OpenVINO backend: runs an OpenVINO IR export of the model (Hugging Face
//! `OpenVINO/Qwen3.8-27B-int4-ov`, NNCF INT4_ASYM group 128) on the Intel NPU
//! through the OpenVINO runtime, with that export's own weights. Text only.
//!
//! The export is two models:
//! - `openvino_text_embeddings_model`: `input` [B, S] i64 -> `inputs_embeds`
//!   [B, S, 5120], always run on the CPU plugin (it is a gather);
//! - `openvino_language_model`: **stateful**; `inputs_embeds`,
//!   `attention_mask` [B, past + S], `position_ids` [4, B, S] (row 0 the text
//!   position, rows 1-3 MRoPE t/h/w, all equal for text) and `beam_idx` [B]
//!   in, `logits` [B, S, V] out; the KV cache of the 16 attention layers and
//!   the conv / SSM state of the 48 linear-attention layers live in its 128
//!   state variables.
//!
//! **NPU** ([`npu`]): the plugin's own LLM mode (`NPUW_LLM`, what OpenVINO
//! GenAI uses) aborts the process compiling this architecture with OpenVINO
//! 2026.3.1 and 2026.4.0 (`LLVM ERROR: Failed to infer result type(s)` at
//! layer 0's linear-attention `in_proj_qkv`; GenAI's `VLMPipeline(…, "NPU")`
//! dies the same way). So the language model is rewritten stateless, reshaped
//! to static shapes and compiled through NPUW without LLM mode, and the state
//! is kept on the host.
//!
//! **CPU / GPU**: the stateful model as exported, dynamic shapes. Kept as the
//! reference the NPU path is checked against.

pub mod dbg;
mod ffi;
pub mod lock;
pub mod npu;
mod rt;
pub mod stateless;

use std::path::{Path, PathBuf};

pub use ffi::{ET_F16, ET_F32, ET_I32, ET_I4, ET_I64, ET_I8};
pub use rt::{Compiled, HostTensor, OvModel, Request, Runtime};

/// How to load and compile the export.
#[derive(Clone, Debug)]
pub struct Options {
	/// OpenVINO device for the language model: `NPU` (default), or `CPU` /
	/// `GPU` as a reference.
	pub device: String,
	/// Directory holding `openvino_c.dll` (see `ffi::Api::load`).
	pub lib_dir: Option<PathBuf>,
	/// Compiled-blob cache (`CACHE_DIR`): an NPU compile of this model is
	/// ~5 min per graph. `None` = `<temp>/qwen-rs-ovcache` (the model
	/// directory may be on a drive that must not be written).
	pub cache_dir: Option<PathBuf>,
	/// NPU: KV capacity in tokens (static past length; the context).
	pub ctx: usize,
	/// NPU: positions per prefill graph `infer` (0 or 1 = no prefill graph,
	/// prompts go through the 1-position graph); a shorter remainder runs
	/// right-padded through it too (`QWEN_OV_PAD`). CPU/GPU: positions per
	/// prefill `infer`, bounding the [S, 248320] f32 logits output.
	pub chunk: usize,
	/// Extra `KEY=VALUE` compile properties, appended last (so they override).
	pub props: Vec<(String, String)>,
}

impl Default for Options {
	fn default() -> Options {
		Options {
			device: "NPU".into(),
			lib_dir: None,
			cache_dir: None,
			ctx: 4096,
			chunk: 0,
			props: Vec::new(),
		}
	}
}

#[allow(clippy::large_enum_variant)] // one per process
enum Engine {
	Npu(npu::NpuLlm),
	Dynamic {
		req: Request,
		lm: Compiled,
		chunk: usize,
	},
}

pub struct OvLlm {
	// Drop order: engine (requests) before the embedding model and runtime.
	engine: Engine,
	emb_req: Request,
	_emb: Compiled,
	rt: Runtime,
	pos: usize,
	pub device: String,
	/// Fixed context on the NPU; `None` on dynamic-shape devices.
	pub ctx: Option<usize>,
}

fn check_export(dir: &Path) -> Result<(PathBuf, PathBuf), String> {
	let lm_xml = dir.join("openvino_language_model.xml");
	let emb_xml = dir.join("openvino_text_embeddings_model.xml");
	for f in [&lm_xml, &emb_xml] {
		if !f.is_file() {
			return Err(format!(
				"{} is missing: not an OpenVINO export directory",
				f.display()
			));
		}
		match std::fs::metadata(f.with_extension("bin")) {
			Ok(m) if m.len() > 1024 => {}
			_ => {
				return Err(format!(
					"{} is missing or a git-lfs pointer",
					f.with_extension("bin").display()
				))
			}
		}
	}
	Ok((lm_xml, emb_xml))
}

impl OvLlm {
	pub fn load(dir: &Path, o: &Options) -> Result<OvLlm, String> {
		let (lm_xml, emb_xml) = check_export(dir)?;
		let rt = Runtime::new(o.lib_dir.as_deref())?;
		eprintln!(
			"  openvino {} from {}",
			rt.api.version(),
			rt.api.dir.display()
		);
		let devs = rt.devices();
		let base = o.device.split([':', '.', ',']).next().unwrap_or("");
		if !devs.iter().any(|d| d.split('.').next() == Some(base)) {
			return Err(format!(
				"OpenVINO device {} not available (have: {})",
				o.device,
				devs.join(", ")
			));
		}
		let name = rt
			.property(&o.device, "FULL_DEVICE_NAME")
			.unwrap_or_default();
		eprintln!("  openvino device {}: {name}", o.device);

		let cache = o
			.cache_dir
			.clone()
			.unwrap_or_else(|| std::env::temp_dir().join("qwen-rs-ovcache"));
		let cache_s = std::fs::create_dir_all(&cache)
			.is_ok()
			.then(|| cache.to_string_lossy().into_owned());

		let t = std::time::Instant::now();
		let emb_m = rt.read_model(&emb_xml)?;
		let eprops: Vec<(&str, &str)> = cache_s.iter().map(|c| ("CACHE_DIR", c.as_str())).collect();
		let emb = rt.compile(&emb_m, "CPU", &eprops)?;
		drop(emb_m);
		let emb_req = emb.request()?;
		eprintln!(
			"  [perf] openvino embeddings ready on CPU in {:.1} s",
			t.elapsed().as_secs_f64()
		);

		let (engine, ctx) = if base == "NPU" {
			let opts = npu::NpuOptions {
				cap: o.ctx,
				chunk: o.chunk,
				cache_dir: cache_s.as_deref(),
				props: &o.props,
			};
			let bin = lm_xml.with_extension("bin");
			(
				Engine::Npu(npu::NpuLlm::load(&rt, &lm_xml, &bin, &opts)?),
				Some(o.ctx),
			)
		} else {
			let t = std::time::Instant::now();
			let lm_m = rt.read_model(&lm_xml)?;
			let mut props: Vec<(&str, &str)> =
				cache_s.iter().map(|c| ("CACHE_DIR", c.as_str())).collect();
			props.push(("PERFORMANCE_HINT", "LATENCY"));
			for (k, v) in &o.props {
				props.push((k, v));
			}
			let lm = rt.compile(&lm_m, &o.device, &props)?;
			drop(lm_m);
			eprintln!(
				"  [perf] openvino language model compiled for {} in {:.1} s",
				o.device,
				t.elapsed().as_secs_f64()
			);
			let req = lm.request()?;
			let chunk = if o.chunk > 1 { o.chunk } else { 1024 };
			(Engine::Dynamic { req, lm, chunk }, None)
		};
		Ok(OvLlm {
			engine,
			emb_req,
			_emb: emb,
			rt,
			pos: 0,
			device: o.device.clone(),
			ctx,
		})
	}

	pub fn runtime(&self) -> &Runtime {
		&self.rt
	}

	/// Tokens fed so far in this conversation.
	pub fn pos(&self) -> usize {
		self.pos
	}

	/// Start a new conversation.
	pub fn reset(&mut self) -> Result<(), String> {
		match &mut self.engine {
			Engine::Npu(n) => n.reset(),
			// The C API has no `reset_state`: a fresh request starts from the
			// variables' initial values.
			Engine::Dynamic { req, lm, .. } => *req = lm.request()?,
		}
		self.pos = 0;
		Ok(())
	}

	fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		let x: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
		self.emb_req.set("input", &[1, x.len() as i64], &x)?;
		self.emb_req.infer()?;
		let (shape, e) = self.emb_req.output_f32("inputs_embeds")?;
		if shape.len() != 3 || shape[1] as usize != ids.len() {
			return Err(format!("embeddings: unexpected output shape {shape:?}"));
		}
		Ok(e.to_vec())
	}

	/// Feed `ids` after everything fed so far; returns the logits of the
	/// last one.
	pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		if ids.is_empty() {
			return Err("forward: no tokens".into());
		}
		if let Some(c) = self.ctx {
			if self.pos + ids.len() > c {
				return Err(format!(
					"context full: {} + {} tokens > {c} (--ov-ctx)",
					self.pos,
					ids.len()
				));
			}
		}
		let emb = self.embed(ids)?;
		let out = match &mut self.engine {
			Engine::Npu(n) => n.forward(&emb)?,
			Engine::Dynamic { req, chunk, .. } => {
				let d = emb.len() / ids.len();
				let mut last = Vec::new();
				let mut pos = self.pos;
				for part in emb.chunks(*chunk * d) {
					last = dynamic_step(req, part, d, pos)?;
					pos += part.len() / d;
				}
				last
			}
		};
		self.pos += ids.len();
		Ok(out)
	}
}

fn dynamic_step(req: &mut Request, emb: &[f32], d: usize, past: usize) -> Result<Vec<f32>, String> {
	let n = emb.len() / d;
	let total = past + n;
	let mask = vec![1i64; total];
	let mut pos = Vec::with_capacity(4 * n);
	for _ in 0..4 {
		pos.extend((past..total).map(|p| p as i64));
	}
	req.set("inputs_embeds", &[1, n as i64, d as i64], emb)?;
	req.set("attention_mask", &[1, total as i64], &mask)?;
	req.set("position_ids", &[4, 1, n as i64], &pos)?;
	req.set("beam_idx", &[1], &[0i32])?;
	req.infer()?;
	let (shape, l) = req.output_f32("logits")?;
	let v = *shape.last().unwrap_or(&0) as usize;
	if v == 0 || l.len() % v != 0 {
		return Err(format!("logits: unexpected shape {shape:?}"));
	}
	if dbg::tf_enabled() {
		dbg::tf_dump(l);
	}
	Ok(l[l.len() - v..].to_vec())
}
