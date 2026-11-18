//! DFlash2 speculative decoding: a block-diffusion drafter
//! (`z-lab/Qwen3.8-27B-DFlash2`) trained for exactly this target.
//!
//! Unlike the MTP head, which chains one token at a time, the drafter proposes
//! a whole block in **one** pass: it reads `[x_n, MASK × 7]` at positions
//! `n .. n + 8`, attends (non-causally inside the block) to a context built
//! from the target's own hidden states, and emits 7 drafts. Its graph is
//! `dflash/model.py` (`DFlash2DraftModel`) of github.com/z-lab/dflash:
//!
//! ```text
//! ctx_p  = hidden_norm(fc(concat(h5_p, h19_p, h33_p, h47_p, h61_p))): target layer outputs
//! per layer (5 dense Qwen3 layers, sliding window 2048):
//!   K/V of ctx_p come straight from k_proj/v_proj(ctx_p); context rows do
//!   NOT pass through the layers, so they are computed once per position.
//!   x += conv_b(attn(conv_a(norm(x)))): two-tap dynamic causal convs
//!   x += conv_b(mlp(conv_a(norm(x))))            around each sublayer
//! logits = output(norm(x[1..])): the target's own lm_head
//! drafts = selector path through the top-16 of each row (low-rank bigram)
//! ```
//!
//! Verify, accept and rollback are MTP's (`Spec`, `spec_commit`): the verify
//! batch is the same `prefill_chunk` with SSM snapshots, and it also
//! **captures** the five feature rows of every position, so the accepted ones
//! become drafter context at the next step. Every emitted token is still
//! sampled from the main model's logits, so output is identical to plain
//! decode; the drafter only decides how many positions a verify pass covers.
//!
//! The drafter's context is **best effort**. Losing rows (a rewind past them,
//! a step that went through `forward_step`) only lowers acceptance, never
//! correctness, which is what lets [`DflashRun`] keep just the last
//! `sliding_window` positions and skip the prefix-cache disk format entirely.
//!
//! Weights ship as BF16 safetensors; they are quantized to `Q6_K` at load
//! (1.6 GB for the 1.65B matmul parameters; the two 248320 × 256 selector
//! codebooks stay BF16 and are only ever gathered row-wise).

#![allow(non_snake_case)]

use super::*;
use crate::mtp::Spec;
use std::io::{Read, Seek, SeekFrom};

/// Two-tap grouped dynamic causal convolution around a sublayer.
pub(crate) struct DfConv {
	/// `[2 * ksz * groups, n_embd]`: per-row dynamic taps, first half for
	/// `prepare`, second for `finish`.
	proj: LinearWeight,
	/// `[2][ksz][n_embd]` static taps.
	base: Vec<f32>,
}

pub(crate) struct DfLayer {
	in_norm: Vec<f32>,
	post_norm: Vec<f32>,
	q_norm: Vec<f32>,
	k_norm: Vec<f32>,
	/// `q|k|v` fused into one `[qd + 2 kvd, n]` matrix (one dispatch and one
	/// activation quantization instead of three), and `k|v` alone for the
	/// context rows, which have no query.
	qkv: LinearWeight,
	kv: LinearWeight,
	wo: LinearWeight,
	/// `gate|up` fused, `[2 ff, n]`.
	gate_up: LinearWeight,
	down: LinearWeight,
	aconv: DfConv,
	mconv: DfConv,
}

/// The loaded drafter.
pub(crate) struct DflashTensors {
	pub(crate) layers: Vec<DfLayer>,
	fc: LinearWeight,
	hidden_norm: Vec<f32>,
	norm: Vec<f32>,
	sel_proj: LinearWeight,
	/// BF16 `[n_vocab, rank]` codebooks.
	pre_cb: Vec<u16>,
	suc_cb: Vec<u16>,
	/// Drafting copy of the lm_head (`QWEN_DFLASH_HEAD`); `None` = the target's.
	head: Option<LinearWeight>,
	/// Reduced vocabulary the head runs over (`QWEN_DFLASH_VOCAB`), built
	/// from the drafting head's rows.
	vocab: Option<draftvocab::DraftVocab>,
	/// Target layers whose outputs form the context feature, in `fc` order.
	pub(crate) target_layers: Vec<usize>,
	pub(crate) block: usize,
	mask: u32,
	pub(crate) window: usize,
	theta: f64,
	n_head: usize,
	n_kv: usize,
	head_dim: usize,
	ff: usize,
	rank: usize,
	top_k: usize,
	group: usize,
	ksz: usize,
	eps: f32,
}

impl DflashTensors {
	pub(crate) fn kv_dim(&self) -> usize {
		self.n_kv * self.head_dim
	}
}

/// Per-run drafter context: K/V rows for positions `start .. start + rows`,
/// then captured feature rows not yet projected (`pending`).
#[derive(Default)]
pub struct DflashRun {
	/// Width of one K/V row (`n_kv * head_dim`).
	pub(crate) kv_dim: usize,
	pub(crate) start: usize,
	pub(crate) k: Vec<Vec<f32>>,
	pub(crate) v: Vec<Vec<f32>>,
	pub(crate) pending: Vec<f32>,
	/// While `Some`, `prefill_chunk` writes every row's feature into it.
	pub(crate) capture: Option<Vec<f32>>,
	pub(crate) snap: SpecSnap,
}

impl Clone for DflashRun {
	/// The rollback scratch and capture buffer are not state; they are rebuilt.
	fn clone(&self) -> Self {
		DflashRun {
			kv_dim: self.kv_dim,
			start: self.start,
			k: self.k.iter().map(|r| r.to_vec()).collect(),
			v: self.v.iter().map(|r| r.to_vec()).collect(),
			pending: self.pending.clone(),
			capture: None,
			snap: SpecSnap::default(),
		}
	}
}

impl DflashRun {
	pub(crate) fn new(n_layer: usize, kv_dim: usize) -> DflashRun {
		DflashRun {
			kv_dim,
			k: vec![Vec::new(); n_layer],
			v: vec![Vec::new(); n_layer],
			..Default::default()
		}
	}

	fn rows(&self, kv_dim: usize) -> usize {
		self.k.first().map_or(0, |k| k.len() / kv_dim)
	}

	/// First position not yet covered (projected or pending).
	fn end(&self, kv_dim: usize, feat: usize) -> usize {
		self.start + self.rows(kv_dim) + self.pending.len() / feat
	}

	fn reset(&mut self, pos: usize) {
		self.start = pos;
		self.k.iter_mut().for_each(Vec::clear);
		self.v.iter_mut().for_each(Vec::clear);
		self.pending.clear();
	}

	/// Forget positions `>= pos` (a rewind). Pending rows are dropped too; the
	/// context then has a gap the next draft resets over.
	pub(crate) fn truncate(&mut self, pos: usize, kv_dim: usize) {
		if pos < self.start {
			self.reset(pos);
			return;
		}
		let keep = (pos - self.start).min(self.rows(kv_dim));
		for k in self.k.iter_mut().chain(self.v.iter_mut()) {
			k.truncate(keep * kv_dim);
		}
		self.pending.clear();
	}

	pub(crate) fn bytes(&self) -> usize {
		let kv: usize = self.k.iter().chain(&self.v).map(|r| r.capacity()).sum();
		(kv + self.pending.capacity()) * std::mem::size_of::<f32>()
	}
}

// ---------------------------------------------------------------- loading

/// Just enough JSON for a safetensors header and a `config.json`.
#[derive(Debug, Clone)]
enum Json {
	Null,
	Bool,
	Num(f64),
	Str(String),
	Arr(Vec<Json>),
	Obj(Vec<(String, Json)>),
}

impl Json {
	fn get(&self, k: &str) -> Option<&Json> {
		match self {
			Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, v)| v),
			_ => None,
		}
	}
	fn num(&self) -> Option<f64> {
		if let Json::Num(n) = self {
			Some(*n)
		} else {
			None
		}
	}
	fn usize(&self, k: &str) -> Result<usize, String> {
		self.get(k)
			.and_then(Json::num)
			.map(|n| n as usize)
			.ok_or(format!("config: missing {k}"))
	}
	fn arr(&self) -> &[Json] {
		if let Json::Arr(v) = self {
			v
		} else {
			&[]
		}
	}
}

fn parse_json(s: &[u8]) -> Result<Json, String> {
	let mut p = 0usize;
	let v = json_value(s, &mut p)?;
	Ok(v)
}

fn json_ws(s: &[u8], p: &mut usize) {
	while *p < s.len() && s[*p].is_ascii_whitespace() {
		*p += 1;
	}
}

fn json_value(s: &[u8], p: &mut usize) -> Result<Json, String> {
	json_ws(s, p);
	let err = |p: usize| format!("json: unexpected input at byte {p}");
	match s.get(*p).copied().ok_or_else(|| err(*p))? {
		b'{' => {
			*p += 1;
			let mut v = Vec::new();
			loop {
				json_ws(s, p);
				if s.get(*p) == Some(&b'}') {
					*p += 1;
					return Ok(Json::Obj(v));
				}
				let Json::Str(k) = json_value(s, p)? else {
					return Err(err(*p));
				};
				json_ws(s, p);
				if s.get(*p) != Some(&b':') {
					return Err(err(*p));
				}
				*p += 1;
				v.push((k, json_value(s, p)?));
				json_ws(s, p);
				if s.get(*p) == Some(&b',') {
					*p += 1;
				}
			}
		}
		b'[' => {
			*p += 1;
			let mut v = Vec::new();
			loop {
				json_ws(s, p);
				if s.get(*p) == Some(&b']') {
					*p += 1;
					return Ok(Json::Arr(v));
				}
				v.push(json_value(s, p)?);
				json_ws(s, p);
				if s.get(*p) == Some(&b',') {
					*p += 1;
				}
			}
		}
		b'"' => {
			*p += 1;
			let mut out = Vec::new();
			while *p < s.len() && s[*p] != b'"' {
				if s[*p] == b'\\' {
					*p += 1;
					match s.get(*p) {
						Some(b'n') => out.push(b'\n'),
						Some(b't') => out.push(b'\t'),
						Some(&c) => out.push(c), // `\uXXXX` never occurs in these files
						None => return Err(err(*p)),
					}
				} else {
					out.push(s[*p]);
				}
				*p += 1;
			}
			*p += 1;
			Ok(Json::Str(String::from_utf8_lossy(&out).into_owned()))
		}
		b't' if s[*p..].starts_with(b"true") => {
			*p += 4;
			Ok(Json::Bool)
		}
		b'f' if s[*p..].starts_with(b"false") => {
			*p += 5;
			Ok(Json::Bool)
		}
		b'n' if s[*p..].starts_with(b"null") => {
			*p += 4;
			Ok(Json::Null)
		}
		_ => {
			let b = *p;
			while *p < s.len() && matches!(s[*p], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
				*p += 1;
			}
			std::str::from_utf8(&s[b..*p])
				.ok()
				.and_then(|t| t.parse().ok())
				.map(Json::Num)
				.ok_or_else(|| err(b))
		}
	}
}

/// A safetensors file: header parsed, data read tensor by tensor.
struct SafeTensors {
	file: std::fs::File,
	data_start: u64,
	header: Json,
}

impl SafeTensors {
	fn open(path: &std::path::Path) -> Result<SafeTensors, String> {
		let e = |x: std::io::Error| format!("{}: {x}", path.display());
		let mut file = std::fs::File::open(path).map_err(e)?;
		let mut n = [0u8; 8];
		file.read_exact(&mut n).map_err(e)?;
		let n = u64::from_le_bytes(n);
		if n > 1 << 26 {
			return Err(format!("{}: not a safetensors file", path.display()));
		}
		let mut h = vec![0u8; n as usize];
		file.read_exact(&mut h).map_err(e)?;
		Ok(SafeTensors {
			file,
			data_start: 8 + n,
			header: parse_json(&h)?,
		})
	}

	/// Tensor `name` as f32, with its shape. BF16 / F16 / F32 only.
	fn f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
		let info = self
			.header
			.get(name)
			.ok_or(format!("drafter: missing tensor {name}"))?;
		let dtype = match info.get("dtype") {
			Some(Json::Str(s)) => s.clone(),
			_ => return Err(format!("{name}: no dtype")),
		};
		let shape: Vec<usize> = info
			.get("shape")
			.map_or(&[][..], Json::arr)
			.iter()
			.filter_map(Json::num)
			.map(|n| n as usize)
			.collect();
		let off: Vec<u64> = info
			.get("data_offsets")
			.map_or(&[][..], Json::arr)
			.iter()
			.filter_map(Json::num)
			.map(|n| n as u64)
			.collect();
		if off.len() != 2 {
			return Err(format!("{name}: bad data_offsets"));
		}
		let mut raw = vec![0u8; (off[1] - off[0]) as usize];
		self.file
			.seek(SeekFrom::Start(self.data_start + off[0]))
			.map_err(|e| e.to_string())?;
		self.file.read_exact(&mut raw).map_err(|e| e.to_string())?;
		let n: usize = shape.iter().product();
		let v: Vec<f32> = match dtype.as_str() {
			"BF16" => raw
				.chunks_exact(2)
				.map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
				.collect(),
			"F16" => raw
				.chunks_exact(2)
				.map(|b| quant::fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
				.collect(),
			"F32" => raw
				.chunks_exact(4)
				.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
				.collect(),
			d => return Err(format!("{name}: unsupported dtype {d}")),
		};
		if v.len() != n {
			return Err(format!("{name}: {} values for shape {shape:?}", v.len()));
		}
		Ok((v, shape))
	}

	/// A BF16 tensor's raw `u16`s (the codebooks: gathered, never multiplied).
	fn bf16(&mut self, name: &str) -> Result<(Vec<u16>, Vec<usize>), String> {
		let (v, shape) = self.f32(name)?;
		Ok((
			v.into_iter().map(|x| (x.to_bits() >> 16) as u16).collect(),
			shape,
		))
	}

	/// A `[out, in]` matrix quantized to `Q4_K` or `Q6_K` (`QWEN_DFLASH_QUANT`).
	fn mat(&mut self, name: &str, out: usize, inn: usize) -> Result<LinearWeight, String> {
		let (v, shape) = self.f32(name)?;
		if shape != [out, inn] {
			return Err(format!("{name}: shape {shape:?}, expected [{out}, {inn}]"));
		}
		Ok(if dflash_q4k() {
			LinearWeight::Q4K {
				raw: quant::quantize_q4_k(&v, out, inn),
				out_dim: out,
				in_dim: inn,
			}
		} else {
			LinearWeight::Q6K {
				raw: quant::quantize_q6_k(&v, out, inn),
				out_dim: out,
				in_dim: inn,
			}
		})
	}

	fn vec(&mut self, name: &str, n: usize) -> Result<Vec<f32>, String> {
		let (v, _) = self.f32(name)?;
		if v.len() != n {
			return Err(format!("{name}: {} values, expected {n}", v.len()));
		}
		Ok(v)
	}
}

/// `QWEN_DFLASH_N`: drafts verified per DFlash2 step (default 4, of the 7 a
/// block yields). Each verified row costs ~16-37 ms of CPU GEMM here (the
/// verify pass turns compute-bound past ~4 rows), so the full block loses:
/// code at temp 0, N = 3 / 4 / 5 / 7 gave 10.08 / - / 9.59 / 9.19 tok/s,
/// and N = 4 beat N = 3 on prose, code and a tool call in interleaved runs.
pub fn dflash_n_draft() -> usize {
	std::env::var("QWEN_DFLASH_N")
		.ok()
		.and_then(|v| v.trim().parse().ok())
		.unwrap_or(4)
}

/// `QWEN_DFLASH_QUANT` (`q4k`, default / `q6k`): format the BF16 drafter
/// matmuls are quantized to at load. The draft is DRAM-bound: Q4_K reads
/// 0.95 GB instead of 1.35 GB per pass, 55 -> 48 ms, at no measured cost in
/// acceptance (prose 2.91 vs 2.91 tokens/step).
fn dflash_q4k() -> bool {
	std::env::var("QWEN_DFLASH_QUANT").map_or(true, |v| !v.eq_ignore_ascii_case("q6k"))
}

/// `QWEN_DFLASH_HEAD` (`q4k`, default / `q6k`): draft against a Q4_K copy of
/// the target's lm_head (0.7 GB, built at load) instead of the 1 GB Q6_K
/// original. Drafting only needs each row's top-16 candidates; verify still
/// uses the original. Another -7 ms per draft (Q4_K both: 55 -> 42 ms).
fn dflash_head_q4k() -> bool {
	std::env::var("QWEN_DFLASH_HEAD").map_or(true, |v| !v.eq_ignore_ascii_case("q6k"))
}

/// `QWEN_DFLASH_VOCAB`: base rows of the drafter's reduced head vocabulary
/// (see [`draftvocab`]), default 151643; `0` = the full 248k-row head.
/// 151643 is Qwen2's vocabulary size: this vocab's merge list carries on past
/// it to 248043 (mostly multilingual merges), and those 39 % of the head's
/// rows never reached the drafter's top-16 candidates; acceptance identical
/// to the full head on code and prose, draft ~40.6 -> ~35.5 ms. Unlike MTP's
/// argmax, the selector *does* lose acceptance on a shorter prefix (32768:
/// code 2.50 -> 2.38, prose 2.20 -> 2.00 tokens/step; 65536: 2.47 / 2.15),
/// so MTP's 32768 does not carry over.
fn dflash_vocab_k() -> usize {
	std::env::var("QWEN_DFLASH_VOCAB")
		.ok()
		.and_then(|v| v.trim().parse().ok())
		.unwrap_or(151643)
}

/// A Q4_K copy of a Q6_K matrix, requantized in row chunks so the f32
/// intermediate stays small.
fn requant_q4k(w: &LinearWeight) -> Option<LinearWeight> {
	let LinearWeight::Q6K {
		raw,
		out_dim,
		in_dim,
	} = w
	else {
		return None;
	};
	let nb = in_dim / quant::QK_K;
	let mut out = Vec::with_capacity(out_dim * nb * 144);
	for r0 in (0..*out_dim).step_by(4096) {
		let r1 = (r0 + 4096).min(*out_dim);
		let f = quant::dequantize_q6_k(&raw[r0 * nb * 210..r1 * nb * 210], (r1 - r0) * in_dim);
		out.extend_from_slice(&quant::quantize_q4_k(&f, r1 - r0, *in_dim));
	}
	Some(LinearWeight::Q4K {
		raw: out,
		out_dim: *out_dim,
		in_dim: *in_dim,
	})
}

/// `QWEN_DFLASH_PMIN`: verify drafts only up to the first whose drafter
/// probability is below this (the first is always verified). `0` = always
/// the full `QWEN_DFLASH_N`.
pub fn dflash_pmin() -> f32 {
	quant::knob::parse("QWEN_DFLASH_PMIN").unwrap_or(0.0)
}

impl Model {
	/// Whether decode drafts with a DFlash2 drafter (`enable_dflash`).
	pub fn dflash_enabled(&self) -> bool {
		self.dflash.is_some() && self.n_draft > 0
	}

	/// Load a DFlash2 drafter from `path` (its directory, or its
	/// `model.safetensors` with `config.json` beside it) and verify up to
	/// `n_draft` of its drafts per pass (clamped to the block size - 1).
	/// Mutually exclusive with MTP.
	pub fn enable_dflash(&mut self, path: &str, n_draft: usize) -> Result<(), String> {
		if self.mtp.is_some() {
			return Err("--dflash and --mtp are mutually exclusive".into());
		}
		let p = std::path::Path::new(path);
		let (dir, st) = if p.is_dir() {
			(p.to_path_buf(), p.join("model.safetensors"))
		} else {
			(
				p.parent()
					.unwrap_or(std::path::Path::new("."))
					.to_path_buf(),
				p.to_path_buf(),
			)
		};
		// The config, first found of: `<name>.json` beside the safetensors
		// file (`Qwen3.8-27B-DFlash2.safetensors` + `Qwen3.8-27B-DFlash2.json`),
		// `config.json` beside it, and `config.json` beside a symlinked file's target.
		let cfg_path = [
			Some(st.with_extension("json")),
			Some(dir.join("config.json")),
			std::fs::canonicalize(&st)
				.ok()
				.and_then(|t| t.parent().map(|d| d.join("config.json"))),
		]
		.into_iter()
		.flatten()
		.find(|c| c.exists())
		.unwrap_or_else(|| dir.join("config.json"));
		let cfg = std::fs::read(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
		let cfg = parse_json(&cfg)?;
		let dc = cfg
			.get("dflash_config")
			.ok_or("config: not a DFlash drafter (no dflash_config)")?;
		let n = self.hp.n_embd;
		let n_layer = cfg.usize("num_hidden_layers")?;
		let (n_head, n_kv, head_dim) = (
			cfg.usize("num_attention_heads")?,
			cfg.usize("num_key_value_heads")?,
			cfg.usize("head_dim")?,
		);
		let ff = cfg.usize("intermediate_size")?;
		if cfg.usize("hidden_size")? != n || cfg.usize("vocab_size")? != self.hp.n_vocab {
			return Err("drafter hidden_size / vocab_size do not match this model".into());
		}
		let target_layers: Vec<usize> = dc
			.get("target_layer_ids")
			.map_or(&[][..], Json::arr)
			.iter()
			.filter_map(Json::num)
			.map(|x| x as usize)
			.collect();
		if target_layers.is_empty() || target_layers.iter().any(|&l| l >= self.hp.n_layer_main) {
			return Err(format!(
				"drafter target_layer_ids {target_layers:?} do not fit this model"
			));
		}
		for k in [
			"input_embedding_scale",
			"output_multiplier",
			"final_logit_softcapping",
		] {
			if dc.get(k).or(cfg.get(k)).is_some() {
				return Err(format!("drafter uses {k}, which is not implemented"));
			}
		}
		let theta = cfg
			.get("rope_parameters")
			.and_then(|r| r.get("rope_theta"))
			.or(cfg.get("rope_theta"))
			.and_then(Json::num)
			.ok_or("config: no rope_theta")?;
		let group = dc.usize("conv_group_size")?;
		let ksz = dc.usize("conv_kernel_size")?;
		let rank = dc.usize("selector_rank")?;
		let t0 = Instant::now();
		let mut f = SafeTensors::open(&st)?;
		let conv = |f: &mut SafeTensors, p: &str| -> Result<DfConv, String> {
			Ok(DfConv {
				proj: f.mat(
					&format!("{p}.kernel_projection.weight"),
					2 * ksz * (n / group),
					n,
				)?,
				base: f.vec(&format!("{p}.base_kernel"), 2 * ksz * n)?,
			})
		};
		let mut layers = Vec::with_capacity(n_layer);
		for i in 0..n_layer {
			let l = format!("layers.{i}");
			let wq = f.mat(
				&format!("{l}.self_attn.q_proj.weight"),
				n_head * head_dim,
				n,
			)?;
			let wk = f.mat(&format!("{l}.self_attn.k_proj.weight"), n_kv * head_dim, n)?;
			let wv = f.mat(&format!("{l}.self_attn.v_proj.weight"), n_kv * head_dim, n)?;
			let gate = f.mat(&format!("{l}.mlp.gate_proj.weight"), ff, n)?;
			let up = f.mat(&format!("{l}.mlp.up_proj.weight"), ff, n)?;
			let cat = |a: &LinearWeight, b: &LinearWeight| {
				quant::concat_weights(a, b).expect("same-format drafter matrices")
			};
			let kv = cat(&wk, &wv);
			layers.push(DfLayer {
				qkv: cat(&wq, &kv),
				kv,
				gate_up: cat(&gate, &up),
				in_norm: f.vec(&format!("{l}.input_layernorm.weight"), n)?,
				post_norm: f.vec(&format!("{l}.post_attention_layernorm.weight"), n)?,
				q_norm: f.vec(&format!("{l}.self_attn.q_norm.weight"), head_dim)?,
				k_norm: f.vec(&format!("{l}.self_attn.k_norm.weight"), head_dim)?,
				wo: f.mat(
					&format!("{l}.self_attn.o_proj.weight"),
					n,
					n_head * head_dim,
				)?,
				down: f.mat(&format!("{l}.mlp.down_proj.weight"), n, ff)?,
				aconv: conv(&mut f, &format!("{l}.attention_conv"))?,
				mconv: conv(&mut f, &format!("{l}.mlp_conv"))?,
			});
		}
		let (pre_cb, ps) = f.bf16("candidate_selector.predecessor_codebook")?;
		let (suc_cb, ss) = f.bf16("candidate_selector.successor_codebook")?;
		if ps != [self.hp.n_vocab, rank] || ss != ps {
			return Err(format!("selector codebooks {ps:?} / {ss:?}"));
		}
		let head = if dflash_head_q4k() {
			requant_q4k(&self.output)
		} else {
			None
		};
		let specials = self.special_tokens();
		let vocab_k = dflash_vocab_k();
		let vocab = draftvocab::DraftVocab::new(
			head.as_ref().unwrap_or(&self.output),
			self.hp.n_vocab,
			n,
			vocab_k,
			&specials,
		);
		self.dflash = Some(DflashTensors {
			head,
			vocab,
			fc: f.mat("fc.weight", n, target_layers.len() * n)?,
			hidden_norm: f.vec("hidden_norm.weight", n)?,
			norm: f.vec("norm.weight", n)?,
			sel_proj: f.mat("candidate_selector.hidden_projection.weight", rank, n)?,
			layers,
			pre_cb,
			suc_cb,
			target_layers,
			block: dc.usize("block_size")?,
			mask: dc.usize("mask_token_id")? as u32,
			window: cfg.usize("sliding_window").unwrap_or(usize::MAX / 2),
			theta,
			n_head,
			n_kv,
			head_dim,
			ff,
			rank,
			top_k: dc.usize("selector_top_k")?,
			group,
			ksz,
			eps: cfg.get("rms_norm_eps").and_then(Json::num).unwrap_or(1e-6) as f32,
		});
		let df = self.dflash.as_ref().unwrap();
		self.n_draft = n_draft
			.clamp(1, df.block - 1)
			.min(offload::SPEC_MAX_SEQ - 1);
		eprintln!(
			"  dflash: {n_layer}-layer drafter, block {}, target layers {:?}, quantized to {}{}{} in {:.1} s",
			df.block, df.target_layers,
			if dflash_q4k() { "Q4_K" } else { "Q6_K" },
			if df.head.is_some() { " (+ Q4_K draft lm_head)" } else { "" },
			if df.vocab.is_some() { format!(", draft vocab {vocab_k}+") } else { String::new() },
			t0.elapsed().as_secs_f64()
		);
		Ok(())
	}

	/// Add `tokens` to the drafter's reduced vocabulary (prompt and output
	/// tokens, as MTP's).
	pub(crate) fn dflash_note(&self, tokens: &[u32]) {
		if let Some(df) = &self.dflash {
			if let Some(dv) = &df.vocab {
				dv.note(df.head.as_ref().unwrap_or(&self.output), tokens);
			}
		}
	}

	/// Feature width: `target_layers.len() * n_embd`.
	fn dflash_feat(&self) -> usize {
		self.dflash
			.as_ref()
			.map_or(0, |d| d.target_layers.len() * self.hp.n_embd)
	}

	/// `prefill_chunk` hook: copy layer `il`'s output rows into the capture
	/// buffer if the drafter reads that layer.
	pub(crate) fn dflash_capture(&self, rs: &mut RunState, il: usize, x: &[f32], N: usize) {
		let (Some(df), Some(d)) = (self.dflash.as_ref(), rs.dflash.as_mut()) else {
			return;
		};
		let Some(cap) = d.capture.as_mut() else {
			return;
		};
		let Some(j) = df.target_layers.iter().position(|&l| l == il) else {
			return;
		};
		let n = self.hp.n_embd;
		let f = df.target_layers.len() * n;
		cap.resize(N * f, 0.0);
		for t in 0..N {
			cap[t * f + j * n..t * f + (j + 1) * n].copy_from_slice(&x[t * n..(t + 1) * n]);
		}
	}

	/// Append feature rows for positions `pos0 ..` to the drafter context,
	/// resetting it first if they do not continue it.
	pub(crate) fn dflash_push(&self, rs: &mut RunState, pos0: usize, feats: &[f32]) {
		let (Some(df), Some(d)) = (self.dflash.as_ref(), rs.dflash.as_mut()) else {
			return;
		};
		let f = self.dflash_feat();
		if d.end(df.kv_dim(), f) != pos0 {
			d.reset(pos0);
		}
		d.pending.extend_from_slice(feats);
	}

	/// Project pending feature rows into per-layer K/V and drop rows no future
	/// query can see.
	pub(crate) fn dflash_absorb(&self, d: &mut DflashRun) {
		let df = self.dflash.as_ref().unwrap();
		let n = self.hp.n_embd;
		let f = self.dflash_feat();
		let kvd = df.kv_dim();
		let mut R = d.pending.len() / f;
		if R == 0 {
			return;
		}
		if R > df.window {
			// Only the last `window` rows are visible from the next query on.
			let skip = R - df.window;
			let rows = d.rows(kvd);
			d.k.iter_mut().chain(d.v.iter_mut()).for_each(Vec::clear);
			d.start += rows + skip;
			d.pending.drain(..skip * f);
			R = df.window;
		}
		let pos0 = d.start + d.rows(kvd);
		let c = apply_linear_batch(&df.fc, &d.pending, R, n, f);
		let c = rows_rms(&c, &df.hidden_norm, n, df.eps);
		for (li, l) in df.layers.iter().enumerate() {
			let kv = apply_linear_batch(&l.kv, &c, R, 2 * kvd, n);
			let [mut k, v] = split_cols(&kv, R, [kvd, kvd]);
			head_norm_rope(&mut k, &l.k_norm, df, df.n_kv, pos0, df.eps);
			d.k[li].extend_from_slice(&k);
			d.v[li].extend_from_slice(&v);
		}
		d.pending.clear();
		let rows = d.rows(kvd);
		if rows > df.window + 256 {
			let drop = rows - df.window;
			for r in d.k.iter_mut().chain(d.v.iter_mut()) {
				r.drain(..drop * kvd);
			}
			d.start += drop;
		}
	}

	/// One drafter pass for the block `[token, MASK, ...]` at position `p`;
	/// `d` must cover exactly the positions before `p`. Returns
	/// `block - 1` drafts, each with the drafter's own probability for it.
	fn dflash_draft(&self, d: &DflashRun, token: u32, p: usize) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
		let t_layers = Instant::now();
		let df = self.dflash.as_ref().unwrap();
		let n = self.hp.n_embd;
		// `QWEN_DFLASH_BLOCK`: rows actually run (anchor + masks), default
		// anchor + the drafts that will be verified. The model was trained on
		// full 8-row blocks and the extra rows do reach the earlier ones
		// through the in-block (non-causal) attention, but dropping them cost
		// no acceptance (prose 2.94 vs 2.88, code 3.61 vs 3.61 tokens/step at
		// 5 vs 8 rows) and saves ~3.4 ms per draft.
		let nd = self.draft_n();
		let B = std::env::var("QWEN_DFLASH_BLOCK")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(nd + 1)
			.clamp(nd + 1, df.block);
		let (hd, kvd, qd) = (df.head_dim, df.kv_dim(), df.n_head * df.head_dim);
		let mut x = Vec::with_capacity(B * n);
		x.extend_from_slice(&self.embedding(token));
		let me = self.embedding(df.mask);
		for _ in 1..B {
			x.extend_from_slice(&me);
		}
		let rows = d.rows(kvd);
		// Context rows a block query can see: q - k < window for the earliest
		// query `p`, i.e. positions > p - window.
		let lo = (p + 1).saturating_sub(df.window).max(d.start) - d.start;
		let nctx = rows - lo;
		let scale = 1.0 / (hd as f32).sqrt();
		for (li, l) in df.layers.iter().enumerate() {
			// --- attention
			let h = rows_rms(&x, &l.in_norm, n, df.eps);
			let (h, dy) = self.dconv_prepare(&l.aconv, &h, B);
			let qkv = apply_linear_batch(&l.qkv, &h, B, qd + 2 * kvd, n);
			let [mut q, mut k, v] = split_cols(&qkv, B, [qd, kvd, kvd]);
			head_norm_rope(&mut q, &l.q_norm, df, df.n_head, p, df.eps);
			head_norm_rope(&mut k, &l.k_norm, df, df.n_kv, p, df.eps);
			let (ck, cv) = (&d.k[li][lo * kvd..], &d.v[li][lo * kvd..]);
			let rep = df.n_head / df.n_kv;
			let mut att = vec![0.0f32; B * qd];
			let ap = att.as_mut_ptr() as usize;
			quant::par_for(B * df.n_head, |task| {
				let (t, hh) = (task / df.n_head, task % df.n_head);
				let kh = hh / rep;
				let qv = &q[t * qd + hh * hd..t * qd + (hh + 1) * hd];
				let key = |j: usize| {
					if j < nctx {
						&ck[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]
					} else {
						let j = j - nctx;
						&k[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]
					}
				};
				let val = |j: usize| {
					if j < nctx {
						&cv[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]
					} else {
						let j = j - nctx;
						&v[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]
					}
				};
				// Context row j sits at d.start + lo + j; it is visible while
				// (p + t) - pos < window.
				let first = (p + t + 1)
					.saturating_sub(df.window)
					.saturating_sub(d.start + lo)
					.min(nctx);
				let mut sc = Vec::with_capacity(nctx + B);
				let mut m = f32::NEG_INFINITY;
				for j in first..nctx + B {
					let s = qv.iter().zip(key(j)).map(|(a, b)| a * b).sum::<f32>() * scale;
					m = m.max(s);
					sc.push(s);
				}
				let mut o = vec![0.0f32; hd];
				let mut z = 0.0f32;
				for (i, j) in (first..nctx + B).enumerate() {
					let w = (sc[i] - m).exp();
					z += w;
					for (a, b) in o.iter_mut().zip(val(j)) {
						*a += w * b;
					}
				}
				// SAFETY: each task owns its (t, hh) slice of `att`.
				let dst = unsafe {
					std::slice::from_raw_parts_mut((ap as *mut f32).add(t * qd + hh * hd), hd)
				};
				for (a, b) in dst.iter_mut().zip(&o) {
					*a = b / z;
				}
			});
			let o = apply_linear_batch(&l.wo, &att, B, n, qd);
			let o = dconv(&o, &dy, &l.aconv.base, 1, df.ksz, df.group, B, n);
			for (a, b) in x.iter_mut().zip(&o) {
				*a += b;
			}
			// --- MLP
			let h = rows_rms(&x, &l.post_norm, n, df.eps);
			let (h, dy) = self.dconv_prepare(&l.mconv, &h, B);
			let ff = df.ff;
			let gu = apply_linear_batch(&l.gate_up, &h, B, 2 * ff, n);
			let mut a = vec![0f32; B * ff];
			for (r, o) in gu.chunks_exact(2 * ff).zip(a.chunks_exact_mut(ff)) {
				quant::act::silu_mul(&r[..ff], &r[ff..], o);
			}
			let m = apply_linear_batch(&l.down, &a, B, n, df.ff);
			let m = dconv(&m, &dy, &l.mconv.base, 1, df.ksz, df.group, B, n);
			for (a, b) in x.iter_mut().zip(&m) {
				*a += b;
			}
		}
		// --- head + candidate selector over rows 1..=n_draft: the lm_head
		// read is the same either way, but the rows past what will be verified
		// are wasted top-k / softmax work.
		let D = (B - 1).min(nd);
		let nv = self.hp.n_vocab;
		let hid = rows_rms(&x[n..(D + 1) * n], &df.norm, n, df.eps);
		let t_head = Instant::now();
		// Over the reduced vocabulary the logits have `nv` columns and column
		// `i` is token `id(i)`; the selector and the probabilities only ever
		// see the columns, so both run unchanged on either form.
		let head = df.head.as_ref().unwrap_or(&self.output);
		let (logits, nv, extra) = match &df.vocab {
			Some(dv) => {
				let (l, ids) = dv.logits_batch(head, &hid, D);
				let m = l.len() / D;
				(l, m, Some((dv, ids)))
			}
			None => (apply_linear_batch(head, &hid, D, nv, n), nv, None),
		};
		let id = |i: u32| match &extra {
			Some((dv, ids)) => dv.id(ids, i as usize),
			None => i,
		};
		let hp = apply_linear_batch(&df.sel_proj, &hid, D, df.rank, n);
		let r = df.rank;
		let row = |cb: &[u16], t: u32| -> Vec<f32> {
			cb[t as usize * r..(t as usize + 1) * r]
				.iter()
				.map(|&b| f32::from_bits((b as u32) << 16))
				.collect()
		};
		let mut pred = token;
		let mut path = Vec::with_capacity(D);
		let mut col = Vec::with_capacity(D);
		let mut alts = Vec::with_capacity(D);
		for j in 0..D {
			let cand = top_k(&logits[j * nv..(j + 1) * nv], df.top_k);
			let w: Vec<f32> = row(&df.pre_cb, pred)
				.iter()
				.zip(&hp[j * r..(j + 1) * r])
				.map(|(a, b)| a * b)
				.collect();
			let mut best = (f32::NEG_INFINITY, cand[0].1);
			let mut second = (f32::NEG_INFINITY, cand[0].1);
			for &(u, c) in &cand {
				let s = u + row(&df.suc_cb, id(c))
					.iter()
					.zip(&w)
					.map(|(a, b)| a * b)
					.sum::<f32>();
				if s > best.0 {
					second = best;
					best = (s, c);
				} else if s > second.0 {
					second = (s, c);
				}
			}
			alts.push(id(second.1));
			col.push(best.1);
			pred = id(best.1);
			path.push(pred);
		}
		// Softmax probability of each chosen token under its row's logits:
		// the confidence `QWEN_DFLASH_PMIN` cuts the verify batch on.
		let debug = std::env::var("QWEN_DFLASH_DEBUG").is_ok();
		let probs: Vec<f32> = (0..D)
			.map(|j| {
				if dflash_pmin() <= 0.0 && !debug && std::env::var("QWEN_TREE_LOG").is_err() {
					return 1.0;
				}
				let row = &logits[j * nv..(j + 1) * nv];
				let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
				let z: f32 = row.iter().map(|&x| (x - m).exp()).sum();
				(row[col[j] as usize] - m).exp() / z
			})
			.collect();
		if debug {
			eprintln!(
				"DFLASH draft: layers {:.1} ms, head+selector {:.1} ms",
				(t_head - t_layers).as_secs_f64() * 1e3,
				t_head.elapsed().as_secs_f64() * 1e3
			);
		}
		(path, probs, alts)
	}

	/// `GroupedDynamicCausalConv::prepare`: the dynamic taps for both halves,
	/// and the first convolution applied to `h`.
	fn dconv_prepare(&self, c: &DfConv, h: &[f32], B: usize) -> (Vec<f32>, Vec<f32>) {
		let df = self.dflash.as_ref().unwrap();
		let n = self.hp.n_embd;
		let w = 2 * df.ksz * (n / df.group);
		let dy = apply_linear_batch(&c.proj, h, B, w, n);
		(dconv(h, &dy, &c.base, 0, df.ksz, df.group, B, n), dy)
	}

	/// DFlash2 decode step: draft a block, then verify `[token, drafts…]` as
	/// one batch, capturing its feature rows. Like [`Model::spec_step`].
	pub fn dflash_step(
		&self,
		rs: &mut RunState,
		token: u32,
		forced: Option<Vec<u32>>,
		stats: &mut MtpStats,
	) -> Spec {
		let df = self.dflash.as_ref().unwrap();
		let f = self.dflash_feat();
		let base_pos = rs.pos;
		let base_mpos = rs.mpos;

		let t0 = Instant::now();
		let mut d = rs
			.dflash
			.take()
			.expect("dflash_step needs a run state with a drafter");
		if d.end(df.kv_dim(), f) != base_pos {
			d.reset(base_pos);
		}
		self.dflash_note(&[token]);
		let suffix = forced.is_some();
		let (mut alts, mut dprobs) = (Vec::new(), Vec::new());
		let drafts = match forced {
			// Context drafts: no drafter pass. The verify still captures the
			// feature rows, so the drafter's context stays complete.
			Some(drafts) => drafts,
			None => {
				self.dflash_absorb(&mut d);
				let (mut drafts, probs, a) = self.dflash_draft(&d, token, base_pos);
				alts = a;
				dprobs = probs.clone();
				if std::env::var("QWEN_DFLASH_DEBUG").is_ok() {
					eprintln!("DFLASH pos={base_pos} token={token} ctx=[{}, +{}) drafts={drafts:?} p={probs:.2?}", d.start, d.rows(df.kv_dim()));
				}
				// Keep the first draft always; stop at the first one the drafter
				// itself doubts (each verified row costs ~30 ms of CPU GEMM here).
				let pmin = dflash_pmin();
				let keep = 1 + probs[1..].iter().take_while(|&&p| p >= pmin).count();
				drafts.truncate(keep.min(self.draft_n()));
				drafts
			}
		};
		let t1 = Instant::now();

		d.capture = Some(Vec::with_capacity((drafts.len() + 1) * f));
		rs.dflash = Some(d);
		let (logits, hs, tokens) = self.spec_verify(rs, token, &drafts, None);
		let feats = rs.dflash.as_mut().unwrap().capture.take().unwrap();
		stats.note(&drafts, t1 - t0, t1.elapsed(), suffix);
		alts.truncate(drafts.len());
		dprobs.truncate(drafts.len());
		Spec {
			drafts,
			logits,
			hs,
			tokens,
			base_pos,
			base_mpos,
			mtp_kv_len: 0,
			feats,
			suffix,
			snap: None,
			alts,
			probs: dprobs,
		}
	}
}

/// Split `rows` rows of concatenated outputs (`[t * sum(w) + o]`) into one
/// row-major buffer per part.
fn split_cols<const K: usize>(y: &[f32], rows: usize, w: [usize; K]) -> [Vec<f32>; K] {
	let tot: usize = w.iter().sum();
	let mut off = 0;
	w.map(|wi| {
		let o = off;
		off += wi;
		(0..rows)
			.flat_map(|t| y[t * tot + o..t * tot + o + wi].iter().copied())
			.collect()
	})
}

/// Row-wise RMS norm of `rows * n` values.
fn rows_rms(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
	let mut y = vec![0.0f32; x.len()];
	for (a, b) in x.chunks_exact(n).zip(y.chunks_exact_mut(n)) {
		rms_norm_into(a, w, eps, b);
	}
	y
}

/// Per-head RMS norm then rotate-half RoPE, on rows of `heads * head_dim`;
/// row `t` is at position `pos0 + t`.
fn head_norm_rope(
	x: &mut [f32],
	w: &[f32],
	df: &DflashTensors,
	heads: usize,
	pos0: usize,
	eps: f32,
) {
	let hd = df.head_dim;
	let half = hd / 2;
	let mut tmp = vec![0.0f32; hd];
	for (t, row) in x.chunks_exact_mut(heads * hd).enumerate() {
		let pos = (pos0 + t) as f64;
		let cs: Vec<(f32, f32)> = (0..half)
			.map(|i| {
				let a = pos * df.theta.powf(-((2 * i) as f64) / hd as f64);
				(a.cos() as f32, a.sin() as f32)
			})
			.collect();
		for h in row.chunks_exact_mut(hd) {
			rms_norm_into(h, w, eps, &mut tmp);
			for i in 0..half {
				let (c, s) = cs[i];
				let (a, b) = (tmp[i], tmp[i + half]);
				h[i] = a * c - b * s;
				h[i + half] = b * c + a * s;
			}
		}
	}
}

/// `_grouped_dynamic_convolve` over a `B`-row block: `out[t][c] = Σ_o
/// (base[o][c] + dyn[t][o][c / group]) * h[t - o][c]`, zero before row 0.
/// `dy` rows hold both halves' taps (`[2][ksz][groups]`) and `base` is
/// `[2][ksz][n]`; `half` is 0 for `prepare`, 1 for `finish`.
#[allow(clippy::too_many_arguments)]
fn dconv(
	h: &[f32],
	dy: &[f32],
	base: &[f32],
	half: usize,
	ksz: usize,
	group: usize,
	B: usize,
	n: usize,
) -> Vec<f32> {
	let G = n / group;
	let w = 2 * ksz * G;
	let base = &base[half * ksz * n..(half + 1) * ksz * n];
	let mut out = vec![0.0f32; B * n];
	for t in 0..B {
		for o in 0..ksz.min(t + 1) {
			let src = &h[(t - o) * n..(t - o + 1) * n];
			let bo = &base[o * n..(o + 1) * n];
			let dt = &dy[t * w + half * ksz * G..(t + 1) * w];
			for c in 0..n {
				out[t * n + c] += (bo[c] + dt[o * G + c / group]) * src[c];
			}
		}
	}
	out
}

/// The `k` largest values of `v` as `(value, index)`.
fn top_k(v: &[f32], k: usize) -> Vec<(f32, u32)> {
	let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
	let mut floor = f32::NEG_INFINITY;
	for (i, &x) in v.iter().enumerate() {
		if best.len() < k || x > floor {
			let at = best.partition_point(|b| b.0 >= x);
			best.insert(at, (x, i as u32));
			best.truncate(k);
			if best.len() == k {
				floor = best[k - 1].0;
			}
		}
	}
	best
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn top_k_is_the_k_largest_in_descending_order() {
		let v = [0.5, -1.0, 3.0, 2.0, 3.0, 0.0, 7.0];
		let t = top_k(&v, 3);
		assert_eq!(t.iter().map(|x| x.1).collect::<Vec<_>>(), vec![6, 2, 4]);
		assert_eq!(top_k(&v, 10).len(), v.len());
	}

	#[test]
	fn dconv_matches_the_reference_formula() {
		// n = 4 channels in groups of 2, two taps, three rows. Taps are
		// distinct per (half, tap, group) so a wrong index shows up.
		let (B, n, ksz, group) = (3usize, 4usize, 2usize, 2usize);
		let G = n / group;
		let h: Vec<f32> = (0..B * n).map(|i| i as f32 + 1.0).collect();
		let dy: Vec<f32> = (0..B * 2 * ksz * G).map(|i| 0.01 * i as f32).collect();
		let base: Vec<f32> = (0..2 * ksz * n).map(|i| 0.1 * i as f32).collect();
		for half in 0..2 {
			let out = dconv(&h, &dy, &base, half, ksz, group, B, n);
			for t in 0..B {
				for c in 0..n {
					let mut want = 0.0f32;
					for o in 0..ksz {
						if o > t {
							continue;
						}
						let b = base[(half * ksz + o) * n + c];
						let d = dy[t * 2 * ksz * G + (half * ksz + o) * G + c / group];
						want += (b + d) * h[(t - o) * n + c];
					}
					assert!(
						(out[t * n + c] - want).abs() < 1e-5,
						"half {half} t {t} c {c}"
					);
				}
			}
		}
	}

	#[test]
	fn parses_a_safetensors_header() {
		let j = parse_json(br#"{"__metadata__":{"format":"pt"},"fc.weight":{"dtype":"BF16","shape":[5120,25600],"data_offsets":[0,262144000]},"x":[true,false,null,-1.5e2,"a\"b"]}"#).unwrap();
		let fc = j.get("fc.weight").unwrap();
		assert!(matches!(fc.get("dtype"), Some(Json::Str(s)) if s == "BF16"));
		let shape: Vec<f64> = fc
			.get("shape")
			.unwrap()
			.arr()
			.iter()
			.filter_map(Json::num)
			.collect();
		assert_eq!(shape, vec![5120.0, 25600.0]);
		assert_eq!(j.get("x").unwrap().arr()[3].num(), Some(-150.0));
		assert!(parse_json(b"{\"a\": }").is_err());
	}

	#[test]
	fn truncate_cuts_the_context_back_and_resets_before_its_start() {
		let mut d = DflashRun::new(2, 2);
		d.start = 10;
		for k in d.k.iter_mut().chain(d.v.iter_mut()) {
			k.extend_from_slice(&[0.0; 2 * 5]); // positions 10..15
		}
		d.pending.extend_from_slice(&[1.0; 6]);
		d.truncate(13, 2);
		assert_eq!((d.start, d.rows(2)), (10, 3));
		assert!(d.pending.is_empty());
		d.truncate(4, 2);
		assert_eq!((d.start, d.rows(2)), (4, 0));
	}
}
