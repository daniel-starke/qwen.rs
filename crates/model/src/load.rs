//! GGUF model loading, tensor accessors, layer construction and the
//! per-layer weight cache (`get_layer` / `warm`).

use super::*;

impl Model {
	pub fn load(path: &str) -> Result<Model, String> {
		let gguf = gguf::parse_file(path).map_err(|e| format!("parse: {e}"))?;

		let meta = |k: &str| -> Option<f64> {
			gguf.metadata
				.iter()
				.find(|e| e.key == k)
				.and_then(|e| match &e.value {
					gguf::MetadataValue::U32(v) => Some(*v as f64),
					gguf::MetadataValue::I32(v) => Some(*v as f64),
					gguf::MetadataValue::F32(v) => Some(*v as f64),
					gguf::MetadataValue::U64(v) => Some(*v as f64),
					gguf::MetadataValue::I64(v) => Some(*v as f64),
					gguf::MetadataValue::F64(v) => Some(*v),
					_ => None,
				})
		};
		let meta_arr = |k: &str| -> Option<Vec<i64>> {
			gguf.metadata
				.iter()
				.find(|e| e.key == k)
				.and_then(|e| match &e.value {
					gguf::MetadataValue::Array { values, .. } => Some(
						values
							.iter()
							.map(|v| match v {
								gguf::MetadataValue::I32(x) => *x as i64,
								gguf::MetadataValue::U32(x) => *x as i64,
								_ => 0,
							})
							.collect(),
					),
					_ => None,
				})
		};

		let n_embd = meta("qwen35.embedding_length").ok_or("missing embedding_length")? as usize;
		let n_head = meta("qwen35.attention.head_count").ok_or("missing head_count")? as usize;
		let n_head_kv =
			meta("qwen35.attention.head_count_kv").ok_or("missing head_count_kv")? as usize;
		let n_ff = meta("qwen35.feed_forward_length").ok_or("missing ffn")? as usize;
		let n_rot = meta("qwen35.rope.dimension_count").ok_or("missing rope dim")? as usize;
		let rope_sections =
			meta_arr("qwen35.rope.dimension_sections").ok_or("missing rope sections")?;
		let sections = [
			rope_sections[0],
			rope_sections[1],
			rope_sections[2],
			rope_sections[3],
		];
		let eps = meta("qwen35.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
		let rope_freq_base = meta("qwen35.rope.freq_base").unwrap_or(1e7);
		let rope_freq_scale = meta("qwen35.rope.freq_scaling_factor")
			.or_else(|| meta("qwen35.rope.scaling.factor"))
			.unwrap_or(1.0);
		let ssm_state = meta("qwen35.ssm.state_size").ok_or("missing ssm state")? as usize;
		let ssm_n_group = meta("qwen35.ssm.group_count").ok_or("missing ssm group")? as usize;
		let ssm_dt_rank = meta("qwen35.ssm.time_step_rank").ok_or("missing ssm dt rank")? as usize;
		let ssm_inner = meta("qwen35.ssm.inner_size").ok_or("missing ssm inner")? as usize;
		let conv_kernel = meta("qwen35.ssm.conv_kernel").ok_or("missing conv kernel")? as usize;
		let block_count = meta("qwen35.block_count").ok_or("missing block_count")? as usize;
		let nextn = meta("qwen35.nextn_predict_layers").unwrap_or(0.0) as usize;
		let n_layer_main = block_count - nextn;
		let full_attn_interval = meta("qwen35.full_attention_interval").unwrap_or(4.0) as usize;

		// head_dim: derive from a full-attention layer's qkv tensor if present,
		// else 256 (this model).
		let head_dim = {
			let mut hd = 256usize;
			for i in 0..n_layer_main {
				let name = format!("blk.{i}.attn_q.weight");
				if let Some(t) = gguf.tensors.iter().find(|t| t.name == name) {
					let out = t.dims[1] as usize;
					hd = out / (2 * n_head);
					break;
				}
			}
			hd
		};

		// find tok_embd and output to get vocab size
		let tok_t = gguf
			.tensors
			.iter()
			.find(|t| t.name == "token_embd.weight")
			.ok_or("missing token_embd")?;
		let n_vocab = tok_t.dims[1] as usize;
		let context_length = meta("qwen35.context_length")
			.map(|v| v as usize)
			.unwrap_or(262144);
		if std::env::var("QWEN_DEBUG").is_ok() {
			eprintln!(
				"DBG token_embd dims=[{} x {}] type={:?}",
				tok_t.dims[0], tok_t.dims[1], tok_t.ggml_type
			);
		}

		// full_attn_layers
		let mut full_attn_layers = vec![false; n_layer_main];
		for i in 0..n_layer_main {
			let is_full = (i + 1) % full_attn_interval == 0;
			// a full-attention layer has attn_q.weight; a recurrent one has attn_qkv.weight
			let has_q = gguf
				.tensors
				.iter()
				.any(|t| t.name == format!("blk.{i}.attn_q.weight"));
			full_attn_layers[i] = is_full && has_q;
		}

		let mut model = Model {
			gguf,
			path: path.to_string(),
			hp: Hyperparams {
				n_embd,
				n_head,
				n_head_kv,
				head_dim,
				n_rot,
				rope_sections: sections,
				n_ff,
				eps,
				rope_freq_base,
				rope_freq_scale,
				rope_attn_factor: 1.0,
				rope_ext_factor: 0.0,
				ssm_state,
				ssm_n_group,
				ssm_dt_rank,
				ssm_inner,
				conv_kernel,
				n_layer_main,
				full_attn_interval,
				n_vocab,
				context_length,
			},
			tok_embd: vec![],
			output: LinearWeight::F32(vec![]),
			output_row_sum: vec![],
			output_norm: vec![],
			full_attn_layers,
			layer_cache: Mutex::new(vec![None; n_layer_main]),
			offload: None,
			ctx: 0,
			n_batch: 0,
			mtp: None,
			n_draft: 0,
			draft_vocab: None,
			dflash: None,
		};
		// Default the context window to the model's native length and the prefill
		// batch to llama.cpp's default (2048). `--ctx`/`--batch` override these.
		model.ctx = model.hp.context_length;
		model.n_batch = 2048;

		// cache embeddings + output (5 GB each; acceptable within 128 GiB)
		let tok = model.tensor_f32("token_embd.weight")?;
		if std::env::var("QWEN_DEBUG").is_ok() {
			let mean = tok.iter().sum::<f32>() / tok.len() as f32;
			let absmax = tok.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
			eprintln!(
				"DBG tok_embd len={} mean={:.6} absmax={:.6} first8={:?}",
				tok.len(),
				mean,
				absmax,
				&tok[..8]
			);
		}
		if let Ok(p) = std::env::var("QWEN_TOKDUMP") {
			if let Ok(mut f) = std::fs::File::create(&p) {
				use std::io::Write;
				let _ = f.write_all(unsafe {
					std::slice::from_raw_parts(tok.as_ptr() as *const u8, tok.len() * 4)
				});
			}
		}
		// lm_head. Like the per-layer big matmuls it stays in its **native
		// form**: this model ships `output.weight` as `Q6_K`, so the per-token
		// logits GEMV streams 1.0 GB instead of the 5.1 GB an f32 expansion
		// would cost, and the f32 copy never enters RSS. This is unconditional:
		// an opt-in f16 switch does nothing for a K-quantized head and leaves
		// decode at ~half speed.
		let output: LinearWeight = match model.tensor_raw("output.weight") {
			Some((raw, GgmlType::Q4_K, od, id)) | Some((raw, GgmlType::Q4_K_M, od, id)) => {
				LinearWeight::Q4K {
					raw,
					out_dim: od,
					in_dim: id,
				}
			}
			Some((raw, GgmlType::Q6_K, od, id)) => LinearWeight::Q6K {
				raw,
				out_dim: od,
				in_dim: id,
			},
			Some((raw, GgmlType::F16, _, _)) => LinearWeight::F16(
				raw.chunks_exact(2)
					.map(|c| u16::from_le_bytes([c[0], c[1]]))
					.collect(),
			),
			Some((raw, ty, od, id)) => LinearWeight::F32(dequantize(ty, &raw, od * id)),
			// tied head (GGUFs that omit `output.weight` reuse `token_embd`)
			None => LinearWeight::F32(tok.clone()),
		};
		let output_norm = model.tensor_f32("output_norm.weight")?;

		// Precompute per-row sum of `output` so the DC-removed logits variant
		// (`logits_dc[o] = logits[o] - dc*Σ_d output[o,d]`) avoids re-reading the
		// full 5+ GB weight matrix a second time every token.
		let nv = model.hp.n_vocab;
		let ne = model.hp.n_embd;
		let output_row_sum = Self::output_row_sum_of(&output, nv, ne);

		Ok(Model {
			tok_embd: tok,
			output,
			output_row_sum,
			output_norm,
			..model
		})
	}

	fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
		self.gguf.tensors.iter().find(|t| t.name == name)
	}

	/// Per-row sum of the lm_head weight, widened to f32, used by the
	/// DC-removed logits variant (`logits_dc[o] = logits[o] - dc*Σ_d
	/// output[o,d]`). Computed once at load for whichever `LinearWeight` form
	/// the lm_head is held in (identical result for F16/F32 since F16 is just
	/// the widened f32 rounded to half precision).
	fn output_row_sum_of(w: &LinearWeight, nv: usize, ne: usize) -> Vec<f32> {
		let mut s = vec![0.0f32; nv];
		match w {
			LinearWeight::F32(v) => {
				for o in 0..nv {
					let base = o * ne;
					let mut a = 0.0f32;
					for d in 0..ne {
						a += v[base + d];
					}
					s[o] = a;
				}
			}
			LinearWeight::F16(v) => {
				for o in 0..nv {
					let base = o * ne;
					let mut a = 0.0f32;
					for d in 0..ne {
						a += quant::fp16_to_f32(v[base + d]);
					}
					s[o] = a;
				}
			}
			// Quantized heads: dequantize one row at a time (never the whole
			// matrix) and sum it. Rows are independent, so this fans out over
			// the shared pool; `nv` is ~248 k here, so a serial pass would add
			// seconds to load.
			LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => {
				let ty = match w {
					LinearWeight::Q4K { .. } => GgmlType::Q4_K,
					_ => GgmlType::Q6_K,
				};
				let row_bytes = raw.len() / nv;
				let base = s.as_mut_ptr() as usize;
				quant::par_for(nv, |o| {
					let row = &raw[o * row_bytes..(o + 1) * row_bytes];
					let a: f32 = dequantize(ty, row, ne).iter().sum();
					// Disjoint index per `o`; `par_for` joins before returning.
					unsafe { *(base as *mut f32).add(o) = a };
				});
			}
		}
		s
	}

	fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
		let t = self
			.tensor_info(name)
			.ok_or_else(|| format!("tensor {name} not found"))?;
		let raw = self
			.gguf
			.read_tensor_bytes(&self.path, t)
			.map_err(|e| format!("read {name}: {e}"))?;
		if name == "token_embd.weight" {
			if let Ok(p) = std::env::var("QWEN_TENSORDUMP") {
				if let Ok(mut f) = std::fs::File::create(&p) {
					use std::io::Write;
					let n = raw.len().min(288);
					let _ = f.write_all(&raw[..n]);
				}
			}
		}
		Ok(dequantize(t.ggml_type, &raw, t.num_elements()))
	}

	/// Read the raw (still-quantized) GGUF bytes of tensor `name`, returning them
	/// together with the matrix's `[out_dim, in_dim]`. Used by the Vulkan offload
	/// path to keep weights in their original `Q4_K_M` form on the GPU (so they
	/// stay compressed in VRAM and are dequantized on the fly).
	pub(crate) fn raw_weight_bytes(
		&self,
		name: &str,
	) -> Option<(Vec<u8>, quant::GgmlType, usize, usize)> {
		let (raw, ty, od, id) = self.tensor_raw(name)?;
		Some((raw, ty, od, id))
	}

	/// Read the raw GGUF bytes of tensor `name` together with its `GgmlType` and
	/// `[out_dim, in_dim]`. Used by `build_layer` to keep big matmuls in their
	/// native (often `Q4_K_M`) form for the fused dequant-GEMV.
	fn tensor_raw(&self, name: &str) -> Option<(Vec<u8>, GgmlType, usize, usize)> {
		let t = self.tensor_info(name)?;
		let raw = self.gguf.read_tensor_bytes(&self.path, t).ok()?;
		let dims = &t.dims;
		let out_dim = *dims.first()? as usize;
		let in_dim = if dims.len() >= 2 { dims[1] as usize } else { 1 };
		let raw = if name.starts_with("blk.") && name.ends_with(".attn_q.weight") {
			self.q_gate_rows_split(raw)
		} else {
			raw
		};
		Some((raw, t.ggml_type, out_dim, in_dim))
	}

	/// `attn_q` interleaves its two outputs per head: rows `[2h*hd, 2h*hd +
	/// hd)` are head `h`'s query, the next `hd` its output gate. Reordered here
	/// to every query row first, then every gate row, so the gate is one
	/// contiguous row range that prefill can compute separately while the CPU
	/// runs attention (it is read only after it). Every consumer of these bytes
	/// (CPU layers, the GPU split and layer uploads, the MTP head) comes
	/// through `tensor_raw`, so they all see this layout.
	fn q_gate_rows_split(&self, raw: Vec<u8>) -> Vec<u8> {
		let (hd, nh) = (self.hp.head_dim, self.hp.n_head);
		let rows = 2 * hd * nh;
		assert_eq!(
			raw.len() % rows,
			0,
			"attn_q: {} bytes over {rows} rows",
			raw.len()
		);
		let rb = raw.len() / rows;
		let mut out = vec![0u8; raw.len()];
		for h in 0..nh {
			let src = 2 * h * hd * rb;
			out[h * hd * rb..(h + 1) * hd * rb].copy_from_slice(&raw[src..src + hd * rb]);
			let g = (nh + h) * hd * rb;
			out[g..g + hd * rb].copy_from_slice(&raw[src + hd * rb..src + 2 * hd * rb]);
		}
		out
	}

	/// Build a layer's weights from the GGUF (uncached).
	///
	/// Big matmuls are kept as `LinearWeight`: `Q4_K_M` stays quantized (the
	/// GEMV dequantizes on the fly; fused dequant-GEMV, ~4x less per-token
	/// traffic and matching llama.cpp), and the rare non-Q4_K big matmuls are
	/// dequantized once to `f16`. Small per-head/SSM params stay `f32`.
	fn build_layer(&self, il: usize) -> Result<LayerTensors, String> {
		self.build_layer_as(il, self.full_attn_layers[il])
	}

	/// Load the MTP head (`blk.{n_layer_main}`, `qwen35.nextn_predict_layers`
	/// = 1) and draft `n_draft` tokens per verify pass from then on; every
	/// `RunState` created afterwards carries its state. `n_draft == 0`
	/// unloads it. Fails if the GGUF has no MTP block.
	pub fn enable_mtp(&mut self, n_draft: usize) -> Result<(), String> {
		if n_draft == 0 {
			self.mtp = None;
			self.n_draft = 0;
			return Ok(());
		}
		let il = self.hp.n_layer_main;
		let has = |m: &Model, s: &str| m.tensor_info(&format!("blk.{il}.{s}")).is_some();
		if !has(self, "nextn.eh_proj.weight") {
			return Err(format!(
				"this GGUF has no MTP head (blk.{il}.nextn.eh_proj.weight)"
			));
		}
		if has(self, "nextn.embed_tokens.weight") || has(self, "nextn.shared_head_head.weight") {
			// Neither is read below; a GGUF that ships them would silently be
			// run against the wrong embedding / lm_head.
			return Err(
				"MTP heads with their own embed_tokens / shared_head_head are not supported".into(),
			);
		}
		let layer = self.build_layer_as(il, true)?;
		let (raw, ty, od, id) = self
			.tensor_raw(&format!("blk.{il}.nextn.eh_proj.weight"))
			.ok_or("missing nextn.eh_proj")?;
		let eh_proj = match ty {
			GgmlType::Q4_K | GgmlType::Q4_K_M => LinearWeight::Q4K {
				raw,
				out_dim: od,
				in_dim: id,
			},
			GgmlType::Q6_K => LinearWeight::Q6K {
				raw,
				out_dim: od,
				in_dim: id,
			},
			_ => LinearWeight::F16(
				dequantize(ty, &raw, od * id)
					.into_iter()
					.map(f32_to_fp16)
					.collect(),
			),
		};
		let head_norm = if has(self, "nextn.shared_head_norm.weight") {
			self.tensor_f32(&format!("blk.{il}.nextn.shared_head_norm.weight"))?
		} else {
			self.output_norm.clone()
		};
		self.mtp = Some(MtpTensors {
			layer,
			eh_proj,
			enorm: self.tensor_f32(&format!("blk.{il}.nextn.enorm.weight"))?,
			hnorm: self.tensor_f32(&format!("blk.{il}.nextn.hnorm.weight"))?,
			head_norm,
		});
		self.n_draft = n_draft.min(offload::SPEC_MAX_SEQ - 1);
		let specials = self.special_tokens();
		self.draft_vocab = draftvocab::DraftVocab::new(
			&self.output,
			self.hp.n_vocab,
			self.hp.n_embd,
			draftvocab::draft_vocab_k(),
			&specials,
		);
		Ok(())
	}

	/// Special tokens (GGUF types 3 CONTROL / 4 USER_DEFINED), always in a
	/// reduced draft vocabulary: they end turns and frame tool calls.
	pub(crate) fn special_tokens(&self) -> Vec<u32> {
		match self
			.gguf
			.metadata
			.iter()
			.find(|e| e.key == "tokenizer.ggml.token_type")
			.map(|e| &e.value)
		{
			Some(gguf::MetadataValue::Array { values, .. }) => values
				.iter()
				.enumerate()
				.filter(|(_, v)| {
					matches!(
						v,
						gguf::MetadataValue::I32(3 | 4) | gguf::MetadataValue::U32(3 | 4)
					)
				})
				.map(|(i, _)| i as u32)
				.collect(),
			_ => Vec::new(),
		}
	}

	/// `build_layer` for layer `il`, with its kind given explicitly (the MTP
	/// block sits past `full_attn_layers`).
	fn build_layer_as(&self, il: usize, is_full: bool) -> Result<LayerTensors, String> {
		let p = |s: &str| format!("blk.{il}.{s}");
		let req = |m: &Model, s: &str| m.tensor_f32(&p(s));
		// Big matmul weights: keep in native (mostly Q4_K_M) form for the fused
		// dequant-GEMV; otherwise dequantize once to f16.
		let big = |m: &Model, s: &str| -> Result<LinearWeight, String> {
			let name = p(s);
			let (raw, ty, od, id) = m
				.tensor_raw(&name)
				.ok_or_else(|| format!("tensor {name} not found"))?;
			if ty == GgmlType::Q4_K || ty == GgmlType::Q4_K_M {
				return Ok(LinearWeight::Q4K {
					raw,
					out_dim: od,
					in_dim: id,
				});
			}
			if ty == GgmlType::Q6_K {
				return Ok(LinearWeight::Q6K {
					raw,
					out_dim: od,
					in_dim: id,
				});
			}
			let f = dequantize(ty, &raw, od * id);
			Ok(LinearWeight::F16(f.into_iter().map(f32_to_fp16).collect()))
		};

		let (wq, wk, wv, wo, q_norm, k_norm) = if is_full {
			(
				Some(big(self, "attn_q.weight")?),
				Some(big(self, "attn_k.weight")?),
				Some(big(self, "attn_v.weight")?),
				Some(big(self, "attn_output.weight")?),
				Some(req(self, "attn_q_norm.weight")?),
				Some(req(self, "attn_k_norm.weight")?),
			)
		} else {
			(None, None, None, None, None, None)
		};
		let (wqkv, wqkv_gate, conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out) =
			if !is_full {
				(
					Some(big(self, "attn_qkv.weight")?),
					Some(big(self, "attn_gate.weight")?),
					Some(req(self, "ssm_conv1d.weight")?),
					Some(req(self, "ssm_dt.bias")?),
					Some(req(self, "ssm_a")?),
					Some(big(self, "ssm_beta.weight")?),
					Some(big(self, "ssm_alpha.weight")?),
					Some(req(self, "ssm_norm.weight")?),
					Some(big(self, "ssm_out.weight")?),
				)
			} else {
				(None, None, None, None, None, None, None, None, None)
			};

		let ksz = self.hp.conv_kernel;
		let conv1d_t = conv1d.as_ref().map(|w: &Vec<f32>| {
			let c_n = w.len() / ksz;
			let mut t = vec![0.0f32; w.len()];
			for c in 0..c_n {
				for f in 0..ksz {
					t[f * c_n + c] = w[f + c * ksz];
				}
			}
			t
		});
		let lt = LayerTensors {
			attn_norm: req(self, "attn_norm.weight")?,
			post_norm: req(self, "post_attention_norm.weight")?,
			wq,
			wk,
			wv,
			wo,
			q_norm,
			k_norm,
			wqkv,
			wqkv_gate,
			conv1d,
			conv1d_t,
			ssm_dt,
			ssm_a,
			ssm_beta,
			ssm_alpha,
			ssm_norm,
			ssm_out,
			ffn_gate: big(self, "ffn_gate.weight")?,
			ffn_down: big(self, "ffn_down.weight")?,
			ffn_up: big(self, "ffn_up.weight")?,
		};
		Ok(lt)
	}

	/// Return layer `il`'s weights, using the layer cache so the (expensive)
	/// Q4_K decode is done **once** and the (large) weight bytes are shared, not
	/// copied, across tokens. The cache is populated lazily and shared across
	/// threads behind a `Mutex`; `forward_step` holds the returned `Arc` and
	/// borrows its weights, so no weight bytes are copied per token.
	pub(crate) fn get_layer(&self, il: usize) -> Arc<LayerTensors> {
		let no_cache = std::env::var("QWEN_NOCACHE").is_ok();
		if no_cache {
			return Arc::new(self.build_layer(il).expect("layer load"));
		}
		// Optional cap on how many layers to cache (memory testing); layers
		// beyond the cap are built fresh each call (still not per-token copied).
		let cap = std::env::var("QWEN_CACHE_MAX")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.unwrap_or(usize::MAX);

		{
			let guard = self.layer_cache.lock().unwrap();
			if let Some(arc) = &guard[il] {
				return arc.clone();
			}
		}
		let lt = Arc::new(self.build_layer(il).expect("layer load"));
		if il < cap {
			let mut guard = self.layer_cache.lock().unwrap();
			guard[il] = Some(lt.clone());
		}
		lt
	}

	/// Pre-build every layer's weights into the layer cache so the (large,
	/// one-time) GGUF read happens **here at load time** instead of being
	/// faulted in on the first forward / prefill. Without this, the first
	/// `forward_prefill` (or the first `forward_step`) pays the full 16.8 GB
	/// tensor read, which makes prefill look slower than decode even though its
	/// matmul is actually faster (it streams each weight once, reused across the
	/// whole prompt). Call once after `Model::load`; the cost is identical, it
	/// is just moved out of the request/prefill hot path.
	pub fn warm(&self) {
		let n = self.hp.n_layer_main;
		for i in 0..n {
			let _ = self.get_layer(i);
		}
	}
}
