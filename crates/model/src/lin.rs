//! Large-matmul dispatch (`lin`/`lin_batch`/`lin_seq`/`lin_seq_group`), the
//! optional Vulkan offload manager init, and token embedding.

use super::*;

/// Public form of [`gpu_prefill`], for the CLI: with weight **streaming** the
/// offload manager must exist even at `--n-gpu-layers 0`, because no layer needs
/// to be resident for prefill to use the GPU. Since prefill routing ships on,
/// this is true unless the user turned it off, so the manager is initialized on
/// every run: on a host with no Vulkan adapter that init fails cheaply and
/// everything stays on the CPU exactly as before.
pub fn gpu_prefill_requested() -> bool {
	gpu_prefill()
}

/// Whether prefill's big matmuls may be routed to the GPU.
///
/// **On by default**; `QWEN_GPU_PREFILL=0` turns the whole
/// route off, streaming included. This is only permission: *which* matmuls
/// actually go to the GPU is decided per call, against measured crossovers, in
/// `OffloadManager::matmul_batch_seq`. Read once: this sits on a path taken a
/// few hundred times per chunk.
fn gpu_prefill() -> bool {
	static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*ON.get_or_init(|| match std::env::var("QWEN_GPU_PREFILL") {
		Err(_) => true,
		Ok(v) => !matches!(
			v.trim().to_ascii_lowercase().as_str(),
			"0" | "no" | "off" | "false"
		),
	})
}

/// Whether prefill may hand `FfnDown` a q8_K activation formed straight from
/// the SwiGLU (`Model::ffn_seq`). **On by default** where the GPU prefill route
/// is; `QWEN_FUSED_SWIGLU=0` is the A/B (bit-identical output).
pub(crate) fn fused_swiglu() -> bool {
	static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	gpu_prefill()
		&& *ON.get_or_init(|| match std::env::var("QWEN_FUSED_SWIGLU") {
			Err(_) => true,
			Ok(v) => !matches!(
				v.trim().to_ascii_lowercase().as_str(),
				"0" | "no" | "off" | "false"
			),
		})
}

/// The shared input of a prefill matmul group: `f32` rows, or rows that went
/// straight to q8_K because the whole group takes the prefill row split,
/// which reads nothing else (`Model::norm_seq_in`).
pub(crate) enum SeqIn {
	F32(Vec<f32>),
	Q8(std::sync::Arc<Vec<quant::Q8K>>),
}

impl Model {
	/// `rms_norm(x[t], w)` for `N` rows of `x`, as the input of the prefill
	/// matmul group `jobs`. Where the group takes the prefill row split
	/// ([`offload::OffloadManager::prefill_q8_ok`]) each row is quantized to
	/// q8_K while in cache and the `f32` matrix is never written
	/// (`quant::par_rows_q8k`); bit-identical either way.
	/// `QWEN_FUSED_SWIGLU=0` turns this off with the SwiGLU fusion.
	pub(crate) fn norm_seq_in(
		&self,
		il: usize,
		x: &[f32],
		w: &[f32],
		n: usize,
		jobs: &[(WeightId, &LinearWeight, usize)],
	) -> SeqIn {
		let n_embd = self.hp.n_embd;
		let eps = self.hp.eps;
		let fused = n > 1
			&& fused_swiglu()
			&& self
				.offload
				.as_ref()
				.is_some_and(|o| o.prefill_q8_ok(il, jobs, n, n_embd));
		if fused {
			return SeqIn::Q8(std::sync::Arc::new(quant::par_rows_q8k(
				n,
				n_embd,
				|t, r| crate::math::rms_norm_into(&x[t * n_embd..(t + 1) * n_embd], w, eps, r),
			)));
		}
		let mut norm = quant::uninit_vec(n * n_embd);
		{
			let (np, xp) = (norm.as_mut_ptr() as usize, x.as_ptr() as usize);
			crate::forward::small_rows(n, |t| {
				// SAFETY: row `t` of each matrix, owned by this task.
				let (src, dst) = unsafe {
					(
						std::slice::from_raw_parts((xp as *const f32).add(t * n_embd), n_embd),
						std::slice::from_raw_parts_mut((np as *mut f32).add(t * n_embd), n_embd),
					)
				};
				crate::math::rms_norm_into(src, w, eps, dst);
			});
		}
		SeqIn::F32(norm)
	}

	/// [`Self::lin_seq_group`] over a [`SeqIn`] of `seq` rows.
	pub(crate) fn lin_seq_group_in(
		&self,
		il: usize,
		x: &SeqIn,
		seq: usize,
		in_dim: usize,
		jobs: &[(WeightId, &LinearWeight, usize)],
	) -> Vec<Vec<f32>> {
		match x {
			SeqIn::F32(x) => self.lin_seq_group(il, x, in_dim, jobs),
			SeqIn::Q8(q8) => self
				.offload
				.as_ref()
				.expect("a q8_K input comes from the offload manager's route")
				.matmul_seq_group_q8(il, jobs, q8, seq, in_dim),
		}
	}

	pub fn embedding(&self, tok: u32) -> Vec<f32> {
		let v = tok as usize;
		let n = self.hp.n_embd;
		let mut e = vec![0.0f32; n];
		for d in 0..n {
			e[d] = self.tok_embd[v * self.hp.n_embd + d];
		}
		e
	}

	/// Prefill (batched, `seq > 1`) variant of [`Self::lin`]. Hands the matmul
	/// to the offload manager, which decides GPU or CPU per call; without a
	/// manager (or with `QWEN_GPU_PREFILL=0`) it is the verified CPU
	/// `quant::apply_linear_batch` kernel.
	///
	/// The `is_offloaded` test this used to make is deliberately gone for
	/// `seq > 1`. It predates weight streaming and it silently capped what
	/// streaming could cover: `FfnDown`, `Wo` and `SsmOut` are the three matmuls
	/// dispatched *alone* rather than through `lin_seq_group`, so on every
	/// non-resident layer they stayed on the CPU no matter what; 43 of 65
	/// layers at the recommended `--n-gpu-layers 22`, and all 65 at the shipped
	/// default of 0. `FfnDown` is the Q6_K half of the FFN block; leaving it
	/// behind while its `FfnUp`/`FfnGate` partners streamed made the GPU do the
	/// cheap two thirds of the block. Residency is now just one of two ways the
	/// manager can reach the GPU, so asking it here was asking the wrong
	/// question. Decode (`seq == 1`) keeps the old rule.
	#[inline]
	pub(crate) fn lin_seq(
		&self,
		il: usize,
		id: WeightId,
		w: &LinearWeight,
		x: &[f32],
		out_dim: usize,
		in_dim: usize,
	) -> Vec<f32> {
		let seq = x.len() / in_dim;
		if let Some(off) = &self.offload {
			if (seq > 1 && gpu_prefill()) || (seq == 1 && off.decode_on_gpu(il)) {
				let job = BatchJobSeq {
					il,
					id,
					w,
					x,
					seq,
					out_dim,
					in_dim,
				};
				return off
					.matmul_batch_seq(std::slice::from_ref(&job))
					.pop()
					.unwrap();
			}
		}
		apply_linear_batch(w, x, seq, out_dim, in_dim)
	}

	/// Batched (prefill) variant of [`Self::lin_seq`]: several matmuls that
	/// **share the same activation `x`** (e.g. `Wq`/`Wk`/`Wv`, or `FfnUp`/
	/// `FfnGate`) are dispatched in one `matmul_batch_seq` call so the GPU
	/// offload manager uploads `x` **once** and issues a single submission
	/// (mirrors the CPU
	/// `lin_batch` grouping already used by `forward_step`). Order is preserved.
	#[inline]
	pub(crate) fn lin_seq_group(
		&self,
		il: usize,
		x: &[f32],
		in_dim: usize,
		jobs: &[(WeightId, &LinearWeight, usize)],
	) -> Vec<Vec<f32>> {
		let seq = x.len() / in_dim;
		if seq > 1 && !gpu_prefill() {
			return jobs
				.iter()
				.map(|(_id, w, od)| apply_linear_batch(w, x, seq, *od, in_dim))
				.collect();
		}
		if let Some(off) = &self.offload {
			let bjobs: Vec<BatchJobSeq> = jobs
				.iter()
				.map(|(id, w, od)| BatchJobSeq {
					il,
					id: *id,
					w,
					x,
					seq,
					out_dim: *od,
					in_dim,
				})
				.collect();
			return off.matmul_batch_seq(&bjobs);
		}
		jobs.iter()
			.map(|(_id, w, od)| apply_linear_batch(w, x, seq, *od, in_dim))
			.collect()
	}

	/// Route a large matmul through the offload manager if the layer is
	/// GPU-offloaded, otherwise use the fused CPU path (`quant::apply_linear`
	/// dispatches Q4_K fused dequant-GEMV / f16 / f32 as appropriate).
	#[inline]
	pub(crate) fn lin(
		&self,
		il: usize,
		id: WeightId,
		w: &LinearWeight,
		x: &[f32],
		out_dim: usize,
		in_dim: usize,
	) -> Vec<f32> {
		if let Some(off) = &self.offload {
			return off.matmul(il, id, w, x, out_dim, in_dim);
		}
		if phase_measure() {
			let t = Instant::now();
			let r = apply_linear(w, x, out_dim, in_dim);
			MATMUL_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
			r
		} else {
			apply_linear(w, x, out_dim, in_dim)
		}
	}

	/// Batch variant of [`Self::lin`]: dispatch several independent matmuls (that
	/// share an input at the call site) to the GPU in one submission, overlapping
	/// their H2D uploads and D2H readbacks. Falls back to the CPU per job when no
	/// GPU is present. Jobs are evaluated in order.
	#[inline]
	pub(crate) fn lin_batch(&self, jobs: &[BatchJob]) -> Vec<Vec<f32>> {
		if let Some(off) = &self.offload {
			return off.matmul_batch(jobs);
		}
		// CPU path. Jobs in a `lin_batch` call share their activation by
		// construction (Wq/Wk/Wv, Wqkv/WqkvGate, FfnUp/FfnGate, beta/alpha), so
		// `apply_linear_group` covers all of them with ONE fork-join and ONE
		// q8_K quantization instead of one of each per job. This is not the old
		// `concat_weights` fusion: no weight bytes are copied; only the output
		// row space is concatenated.
		let run = || -> Vec<Vec<f32>> {
			let shared = jobs
				.windows(2)
				.all(|w| std::ptr::eq(w[0].x, w[1].x) && w[0].in_dim == w[1].in_dim);
			let kquant = jobs
				.iter()
				.all(|j| matches!(j.w, LinearWeight::Q4K { .. } | LinearWeight::Q6K { .. }));
			if shared
				&& kquant && jobs[0].in_dim.is_multiple_of(quant::QK_K)
				&& (jobs.iter().all(|j| j.id.exact_decode()) || quant::exact_decode_forced())
			{
				// Same rule as `OffloadManager::exact_decode`.
				let g: Vec<(&LinearWeight, usize)> =
					jobs.iter().map(|j| (j.w, j.out_dim)).collect();
				return quant::apply_linear_group_exact(&g, jobs[0].x, jobs[0].in_dim);
			}
			if shared && jobs.len() > 1 {
				let g: Vec<(&LinearWeight, usize)> =
					jobs.iter().map(|j| (j.w, j.out_dim)).collect();
				quant::apply_linear_group(&g, jobs[0].x, jobs[0].in_dim)
			} else {
				jobs.iter()
					.map(|j| apply_linear(j.w, j.x, j.out_dim, j.in_dim))
					.collect()
			}
		};
		if phase_measure() {
			let t = Instant::now();
			let r = run();
			MATMUL_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
			r
		} else {
			run()
		}
	}

	/// Initialize the optional Vulkan offload manager and upload the large
	/// matmul weights for the last `n_gpu_layers` layers to the GPU.
	///
	/// The KV cache and SSM recurrent state for every layer remain CPU-side
	/// (see `RunState`); only the big matmul weights are moved to VRAM and the
	/// hidden-state vector is shipped per matmul. If no Vulkan adapter is
	/// available, the manager falls back to CPU and the existing verified path
	/// is used unchanged.
	pub fn init_offload(&mut self, n_gpu_layers: usize) {
		let n_embd = self.hp.n_embd;
		let n_head = self.hp.n_head;
		let n_head_kv = self.hp.n_head_kv;
		let head_dim = self.hp.head_dim;
		let value_dim = self.hp.ssm_inner;
		let hv = self.hp.ssm_dt_rank;
		let key_dim = self.hp.ssm_state * self.hp.ssm_n_group;
		let conv_dim = key_dim * 2 + value_dim;
		let n_ff = self.hp.n_ff;

		let mut mgr = OffloadManager::new(n_gpu_layers, self.hp.n_layer_main);
		// A speculative verify batch (`1 + n_draft` tokens) runs on the
		// decode split's persistent buffers, which are sized at upload.
		// Context drafts (`crate::suffix`) can fill a whole batch.
		if crate::suffix::suffix_enabled() {
			mgr.set_decode_seq(offload::SPEC_MAX_SEQ);
		} else if self.mtp_enabled() || self.dflash_enabled() {
			mgr.set_decode_seq(1 + self.n_draft);
		}
		// Upload the ORIGINAL quantized (Q4_K / Q6_K) bytes for the big per-layer
		// matmuls. The GPU dequantizes them on the fly (fused with the matmul)
		// and never expands them to a full fp16/fp32 matrix in VRAM. Non-K-quant
		// big matmuls (rare here) fall back to the verified CPU path.
		let up = |mgr: &mut OffloadManager,
		          il: usize,
		          id: WeightId,
		          suffix: &str,
		          out_dim: usize,
		          in_dim: usize| {
			if let Some((raw, ty, _o, _i)) =
				self.raw_weight_bytes(&format!("blk.{}.{}", il, suffix))
			{
				debug_assert_eq!(_o, out_dim);
				debug_assert_eq!(_i, in_dim);
				mgr.upload(il, id, &raw, ty, out_dim, in_dim);
			}
		};
		// Per-layer list of (id, suffix, out_dim, in_dim) for the big matmuls.
		let specs = |il: usize| -> Vec<(WeightId, &'static str, usize, usize)> {
			if self.full_attn_layers[il] {
				vec![
					(WeightId::Wq, "attn_q.weight", 2 * head_dim * n_head, n_embd),
					(WeightId::Wk, "attn_k.weight", head_dim * n_head_kv, n_embd),
					(WeightId::Wv, "attn_v.weight", head_dim * n_head_kv, n_embd),
					(
						WeightId::Wo,
						"attn_output.weight",
						n_embd,
						head_dim * n_head,
					),
				]
			} else {
				vec![
					(WeightId::Wqkv, "attn_qkv.weight", conv_dim, n_embd),
					(WeightId::WqkvGate, "attn_gate.weight", value_dim, n_embd),
					(WeightId::SsmOut, "ssm_out.weight", n_embd, value_dim),
					(WeightId::WssmBeta, "ssm_beta.weight", hv, n_embd),
					(WeightId::WssmAlpha, "ssm_alpha.weight", hv, n_embd),
				]
			}
			.into_iter()
			.chain([
				(WeightId::FfnUp, "ffn_up.weight", n_ff, n_embd),
				(WeightId::FfnGate, "ffn_gate.weight", n_ff, n_embd),
				(WeightId::FfnDown, "ffn_down.weight", n_embd, n_ff),
			])
			.collect()
		};
		// VRAM budget guard: cap offloaded layers so we never exceed the
		// adapter's memory (the old f16 path OOM'd at --n-gpu-layers 16; the
		// K-quant path is ~4x (Q4_K) / ~2.7x (Q6_K) smaller than f16 but still
		// bounded). Default 5 GiB headroom under a 6 GiB card; override with
		// QWEN_VRAM_GB. Iterate last->first so the highest (most impactful)
		// layers are kept when the budget bites.
		let vram_gb: f64 = std::env::var("QWEN_VRAM_GB")
			.ok()
			.and_then(|s| s.parse().ok())
			.unwrap_or(5.0);
		let budget: u64 = (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64;
		const QK_K: usize = 256;
		const Q4K_BLOCK: usize = 144;
		const Q6K_BLOCK: usize = 210;
		const QK8_0: usize = 32;
		const Q8_0_BLOCK: usize = 34;
		const F16_ELEM: usize = 2;
		const F32_ELEM: usize = 4;
		// Bytes a big matmul occupies on the GPU (compressed in its native
		// quant), or 0 if the type isn't offloaded.
		let kq_bytes = |ty: quant::GgmlType, od: usize, idim: usize| -> u64 {
			match ty {
				quant::GgmlType::Q4_K | quant::GgmlType::Q4_K_M => {
					(od * idim / QK_K * Q4K_BLOCK) as u64
				}
				quant::GgmlType::Q6_K => (od * idim / QK_K * Q6K_BLOCK) as u64,
				quant::GgmlType::Q8_0 => (od * idim / QK8_0 * Q8_0_BLOCK) as u64,
				quant::GgmlType::F16 => (od * idim * F16_ELEM) as u64,
				quant::GgmlType::F32 => (od * idim * F32_ELEM) as u64,
				_ => 0,
			}
		};
		let mut used: u64 = 0;
		let candidates: Vec<usize> = (0..self.hp.n_layer_main)
			.filter(|&i| mgr.is_offloaded(i))
			.collect();
		for &il in candidates.iter().rev() {
			let mut layer_bytes = 0u64;
			for (_, suffix, od, idim) in specs(il) {
				if let Some((raw, ty, _, _)) =
					self.raw_weight_bytes(&format!("blk.{}.{}", il, suffix))
				{
					if raw.len() as u64 == kq_bytes(ty, od, idim) {
						layer_bytes += raw.len() as u64;
					}
				}
			}
			if used + layer_bytes > budget {
				mgr.gpu_layers[il] = false;
				continue;
			}
			used += layer_bytes;
			for (id, suffix, od, idim) in specs(il) {
				up(&mut mgr, il, id, suffix, od, idim);
			}
		}
		// Decode row split (**default on**; `QWEN_GPU_SPLIT=0` disables): the
		// VRAM left after any
		// layer offload holds the *first rows of every matrix* of the other
		// layers, and each decode matmul runs its GPU rows concurrently with
		// the CPU's. The model is dense (every weight byte is read once per
		// token) so a byte in VRAM is worth the same wherever it sits, and a
		// uniform fraction is the whole placement policy. `auto` (the default,
		// or any non-number) fills the budget; a number in (0, 1] forces the
		// fraction.
		let split_env = std::env::var("QWEN_GPU_SPLIT").unwrap_or_else(|_| "auto".into());
		let split_on = !matches!(
			split_env.trim().to_ascii_lowercase().as_str(),
			"0" | "no" | "off" | "false"
		);
		if split_on {
			let v = split_env;
			if mgr.available {
				let mut total: u64 = 0;
				for il in 0..self.hp.n_layer_main {
					if mgr.is_offloaded(il) {
						continue;
					}
					for (_, suffix, od, idim) in specs(il) {
						if let Some((raw, ty, _, _)) =
							self.raw_weight_bytes(&format!("blk.{}.{}", il, suffix))
						{
							if matches!(
								ty,
								quant::GgmlType::Q4_K
									| quant::GgmlType::Q4_K_M | quant::GgmlType::Q6_K
							) && raw.len() as u64 == kq_bytes(ty, od, idim)
							{
								total += raw.len() as u64;
							}
						}
					}
				}
				// What may actually be allocated: the measured budget minus
				// current usage (which already includes any layer offload
				// above), less a reserve for what prefill allocates later;
				// its streaming slots and per-call x/y buffers. Without the
				// extension, fall back to the static `QWEN_VRAM_GB` budget.
				// With the MTP head 512 MB is not enough: a 4k-token prefill
				// loses the device and the run goes CPU only; 1024 survives
				// that but loses it on a 65k -> 260k prefill. 1536 held for
				// 259k tokens. The cause is unknown, so the value is
				// empirical: re-measure before lowering it.
				let reserve: u64 = std::env::var("QWEN_VRAM_RESERVE_MB")
					.ok()
					.and_then(|s| s.parse::<u64>().ok())
					.unwrap_or(if self.mtp_enabled() { 1536 } else { 512 })
					<< 20;
				let room = match mgr.vram_budget() {
					Some((b, u)) => {
						eprintln!(
							"offload: VRAM budget {:.2} GiB, in use {:.2} GiB, reserve {:.2} GiB",
							b as f64 / (1u64 << 30) as f64,
							u as f64 / (1u64 << 30) as f64,
							reserve as f64 / (1u64 << 30) as f64,
						);
						b.saturating_sub(u).saturating_sub(reserve)
					}
					None => budget.saturating_sub(used),
				};
				let budget = used + room;
				let frac = match v.trim().parse::<f64>() {
					Ok(f) if f > 0.0 => f.min(1.0),
					_ => (room as f64 / total.max(1) as f64).min(1.0),
				};
				let mut split_bytes: u64 = 0;
				for il in 0..self.hp.n_layer_main {
					if mgr.is_offloaded(il) {
						continue;
					}
					for (id, suffix, od, idim) in specs(il) {
						if let Some((raw, ty, _, _)) =
							self.raw_weight_bytes(&format!("blk.{}.{}", il, suffix))
						{
							// Multiples of 4 rows: the GEMV covers 4 rows per
							// workgroup, so a ragged tail would waste a group.
							// `ssm_beta`/`ssm_alpha` are not split: decode runs
							// them on the CPU with an f32 activation
							// (`WeightId::exact_decode`), which keeps the GPU's
							// accuracy without its round trip.
							if id.exact_decode() {
								continue;
							}
							let rows = ((od as f64 * frac) as usize) / 4 * 4;
							let bytes = raw.len() as u64 * rows as u64 / od as u64;
							if rows == 0 || used + split_bytes + bytes > budget {
								continue;
							}
							mgr.upload_split(il, id, &raw, ty, od, idim, rows);
							split_bytes += bytes;
						}
					}
				}
				eprintln!(
					"offload: row split {:.1} % of every matrix on the GPU ({} matrices, {split_bytes} bytes)",
					frac * 100.0,
					mgr.n_split(),
				);
			}
		}
		if mgr.available {
			let kept = mgr.gpu_layers.iter().filter(|&&b| b).count();
			eprintln!(
				"offload: {kept} layers dispatched to Vulkan ({used} bytes, budget {budget}); KV cache stays in CPU RAM",
			);
		}
		self.offload = Some(Arc::new(mgr));
	}

	/// Bring the GPU path back after its Vulkan device was lost (another
	/// application claiming VRAM can cause that): rebuild the device and
	/// upload its weights again, from the same bytes `init_offload` read.
	/// Called before every forward pass; one atomic check while the device is
	/// healthy, and rate-limited while it cannot be rebuilt
	/// (`OffloadManager::recover_gpu`).
	pub(crate) fn heal_offload(&self) {
		let Some(off) = &self.offload else { return };
		if !off.gpu_lost() {
			return;
		}
		off.recover_gpu(&|il, id| {
			let suffix = match id {
				WeightId::Wq => "attn_q.weight",
				WeightId::Wk => "attn_k.weight",
				WeightId::Wv => "attn_v.weight",
				WeightId::Wo => "attn_output.weight",
				WeightId::Wqkv => "attn_qkv.weight",
				WeightId::WqkvGate => "attn_gate.weight",
				WeightId::SsmOut => "ssm_out.weight",
				WeightId::WssmBeta => "ssm_beta.weight",
				WeightId::WssmAlpha => "ssm_alpha.weight",
				WeightId::FfnUp => "ffn_up.weight",
				WeightId::FfnGate => "ffn_gate.weight",
				WeightId::FfnDown => "ffn_down.weight",
				WeightId::EhProj => "nextn.eh_proj.weight",
			};
			self.raw_weight_bytes(&format!("blk.{il}.{suffix}"))
				.map(|(raw, ty, _, _)| (raw, ty))
		});
	}
}
