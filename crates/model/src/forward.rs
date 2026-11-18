//! Run-state construction, the batched-prefill passes (`prefill_chunk` /
//! `forward_prefill`) and the `forward` verification helper.
//!
//! The per-token `forward_step` lives in `vec_fwd.rs`.

// The SSM/delta-net dimension names below (`S`, `Hk`, `Hv`, `N`, `Tprev`) are
// the upper-case symbols used by the reference gated-delta-net formulation; the
// math is far easier to follow against llama.cpp when they keep that spelling.
#![allow(non_snake_case)]

use super::*;

/// Directory that [`feature_dump`] appends layer outputs to; `None` = off.
static FEATURE_DUMP: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

/// Diagnostic for drafter experiments (DFlash2 consumes the outputs of target
/// layers 5/19/33/47/61): while set, every `prefill_chunk` appends the output
/// rows of each of those layers to `dir/feat_{il}.bin` as raw `f32`.
pub fn set_feature_dump(dir: Option<std::path::PathBuf>) {
	*FEATURE_DUMP.lock().unwrap() = dir;
}

/// Rows a per-row elementwise pass runs inline on the calling thread instead
/// of across the pool: a verify batch (<= 8 rows) or a tiny chunk.
///
/// A pool dispatch costs ~7 us when the workers are spinning, but the decode
/// row split leaves them idle while the caller waits for the GPU's rows, long
/// enough to park, and then every dispatch waits 60-100 us for the slowest
/// worker to wake: for a norm or a residual add over 5 rows that takes a few
/// microseconds. Waiting longer before parking does not help (spinning, or
/// `umwait`, costs the active cores' power budget).
const SMALL_ROWS: usize = 8;

/// `par_for(n, f)`, or a plain loop when `n <= SMALL_ROWS`. Each row is
/// independent, so the result is the same either way.
pub(crate) fn small_rows<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
	if n <= SMALL_ROWS {
		(0..n).for_each(f);
	} else {
		quant::par_for(n, f);
	}
}

pub(crate) fn feature_dump(il: usize, rows: &[f32]) {
	const LAYERS: [usize; 5] = [5, 19, 33, 47, 61];
	if !LAYERS.contains(&il) {
		return;
	}
	let g = FEATURE_DUMP.lock().unwrap();
	let Some(dir) = g.as_ref() else { return };
	use std::io::Write;
	let mut f = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(dir.join(format!("feat_{il}.bin")))
		.expect("feature dump");
	let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 4) };
	f.write_all(bytes).expect("feature dump");
}

impl Model {
	/// Allocate a fresh run state (position 0, empty caches).
	pub fn new_run_state(&self) -> RunState {
		let S = self.hp.ssm_state;
		let Hk = self.hp.ssm_n_group;
		let Hv = self.hp.ssm_dt_rank;
		let value_dim = self.hp.ssm_inner;
		let key_dim = S * Hk;
		let conv_dim = key_dim * 2 + value_dim;
		// Pre-size the KV cache for full-attention layers to the full context so a
		// 200k-token prompt doesn't trigger repeated reallocations of the growing
		// cache. Only the 16 full-attn layers keep a KV cache (the SSM/delta-net
		// layers carry a fixed-size recurrent state instead). `with_capacity`
		// reserves address space only; physical pages are committed as filled.
		let kv_cap = self.ctx.max(1) * self.hp.head_dim * self.hp.n_head_kv;
		let mut kv = Vec::with_capacity(self.hp.n_layer_main);
		for il in 0..self.hp.n_layer_main {
			if self.full_attn_layers[il] {
				kv.push(Some((
					Vec::with_capacity(kv_cap),
					Vec::with_capacity(kv_cap),
				)));
			} else {
				kv.push(None);
			}
		}
		let mtp = self.mtp.as_ref().map(|_| MtpRun {
			kv: Some((Vec::with_capacity(kv_cap), Vec::with_capacity(kv_cap))),
			last_h: vec![0.0; self.hp.n_embd],
			pending: Vec::new(),
			snap: SpecSnap::default(),
		});
		RunState {
			kv,
			state: vec![vec![0.0; S * S * Hv]; self.hp.n_layer_main],
			convbuf: vec![vec![vec![0.0; conv_dim]; self.hp.conv_kernel - 1]; self.hp.n_layer_main],
			pos: 0,
			mpos: [0; 3],
			mrope: false,
			mtp,
			dflash: self
				.dflash
				.as_ref()
				.map(|d| DflashRun::new(d.layers.len(), d.kv_dim())),
		}
	}

	/// A live copy of `src` (typically a compact [`RunState::clone_compact`]
	/// snapshot) with the KV caches re-reserved for the whole context, as
	/// [`Model::new_run_state`] would have. `None` if `src` does not fit this
	/// model: a different layer layout, MTP on one side only, or no room left
	/// in `--ctx`.
	pub fn fork_run_state(&self, src: &RunState) -> Option<RunState> {
		let mut rs = self.new_run_state();
		let same_len = |a: &[Vec<f32>], b: &[Vec<f32>]| {
			a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.len() == y.len())
		};
		let fits = src.kv.len() == rs.kv.len()
			&& src
				.kv
				.iter()
				.zip(&rs.kv)
				.all(|(a, b)| a.is_some() == b.is_some())
			&& same_len(&src.state, &rs.state)
			&& src.convbuf.len() == rs.convbuf.len()
			&& src
				.convbuf
				.iter()
				.zip(&rs.convbuf)
				.all(|(a, b)| same_len(a, b))
			&& src.mtp.is_some() == rs.mtp.is_some()
			&& src.pos < self.ctx;
		if !fits {
			return None;
		}
		for (d, s) in rs.kv.iter_mut().zip(&src.kv) {
			if let (Some((dk, dv)), Some((sk, sv))) = (d, s) {
				dk.extend_from_slice(sk);
				dv.extend_from_slice(sv);
			}
		}
		rs.state.clone_from(&src.state);
		rs.convbuf.clone_from(&src.convbuf);
		rs.pos = src.pos;
		rs.mpos = src.mpos;
		rs.mrope = src.mrope;
		if let (Some(d), Some(s)) = (&mut rs.mtp, &src.mtp) {
			if let (Some((dk, dv)), Some((sk, sv))) = (&mut d.kv, &s.kv) {
				dk.extend_from_slice(sk);
				dv.extend_from_slice(sv);
			}
			d.last_h.clone_from(&s.last_h);
			d.pending = s.pending.clone();
		}
		if let (Some(d), Some(s)) = (&mut rs.dflash, &src.dflash) {
			*d = s.clone();
		}
		Some(rs)
	}

	/// Set the maximum context length (KV-cache capacity / position cap).
	/// `n == 0` keeps the model's native `context_length`; otherwise it is
	/// clamped to `context_length` (the GGUF-trained window).
	pub fn set_ctx(&mut self, n: usize) {
		self.ctx = if n == 0 {
			self.hp.context_length
		} else {
			n.min(self.hp.context_length)
		};
	}

	/// The context length `set_ctx` settled on.
	pub fn ctx(&self) -> usize {
		self.ctx
	}

	/// Set the prefill chunk size (tokens processed per batched-forward pass).
	/// Mirrors llama.cpp's `n_batch`; larger uses more transient activation
	/// memory, smaller bounds the peak (needed for very long prompts).
	pub fn set_n_batch(&mut self, n: usize) {
		self.n_batch = n.max(1);
	}

	/// Run the full forward pass over `tokens` (processed one at a time) and
	/// return the LM-head logits for the **last** token.
	///
	/// Implemented by running `forward_step` for each position on a fresh run
	/// state; used for verification against llama.cpp (argmax / cosine / layer
	/// dumps). For autoregressive generation, call `forward_step` directly on a
	/// reused `RunState` to avoid re-computing the prefix.
	pub fn forward(&self, tokens: &[u32], max_layers: Option<usize>) -> (Vec<f32>, Vec<f32>) {
		let mut rs = self.new_run_state();
		let mut last = (vec![0.0f32; self.hp.n_vocab], vec![0.0f32; self.hp.n_vocab]);
		for &tok in tokens {
			last = self.forward_step(&mut rs, tok, max_layers);
		}
		last
	}

	/// Process one sub-sequence (`tokens`, length `C`) through the full layer
	/// stack as a single batched forward, updating `rs`'s KV cache / SSM state /
	/// convbuf and writing the final hidden states into `xmat` (len `C *
	/// n_embd`). Positions covered are `rs.pos .. rs.pos + C`; `rs.pos` is
	/// advanced by `C`. `forward_prefill` calls this in `n_batch`-sized chunks
	/// so a long prompt never materializes the whole sequence's activations at
	/// once (mirrors llama.cpp's chunked prefill). Chunking is mathematically
	/// identical to a single full pass: every per-token op is unchanged, only
	/// the batched GEMM / attention grouping differs.
	///
	/// `snap` (the speculative verify pass) records every SSM layer's state
	/// and conv window after each of the first `N - 1` tokens, so the batch can
	/// be cut back to any prefix (`Model::spec_commit`).
	pub(crate) fn prefill_chunk(
		&self,
		rs: &mut RunState,
		input: PromptInput<'_>,
		max_layers: Option<usize>,
		xmat: &mut [f32],
		mut snap: Option<&mut SpecSnap>,
	) {
		self.heal_offload();
		let tokens = input.tokens;
		let measure = phase_measure();
		let t_all = if measure { Some(Instant::now()) } else { None };
		let n_embd = self.hp.n_embd;
		let N = tokens.len();
		let S = self.hp.ssm_state;
		let Hk = self.hp.ssm_n_group;
		let Hv = self.hp.ssm_dt_rank;
		let value_dim = self.hp.ssm_inner;
		let key_dim = S * Hk;
		let conv_dim = key_dim * 2 + value_dim;
		let eps = self.hp.eps;
		// Snapshots after tokens `0 .. ncap`; the last token's state is the
		// live one.
		let ncap = if snap.is_some() { N - 1 } else { 0 };
		if let Some(sn) = snap.as_deref_mut() {
			let n_layer = self.hp.n_layer_main;
			if sn.conv.len() < ncap {
				sn.conv.resize_with(ncap, || vec![Vec::new(); n_layer]);
			}
			sn.base.resize_with(n_layer, Vec::new);
			sn.inp.resize_with(n_layer, || None);
		}

		// xmat[t] = embedding(tokens[t]), seq-major [N * n_embd] (caller buffer).
		// An image contributes its projected patch embedding here instead of a
		// row of `token_embd`; the `<|image_pad|>` token it replaces is still
		// in `tokens` so the KV cache, the position bookkeeping and the
		// decoded text all stay consistent.
		for t in 0..N {
			match input.embed(t) {
				Some(e) => {
					assert_eq!(e.len(), n_embd, "image embedding width");
					xmat[t * n_embd..(t + 1) * n_embd].copy_from_slice(e);
				}
				None => {
					let e = self.embedding(tokens[t]);
					xmat[t * n_embd..(t + 1) * n_embd].copy_from_slice(&e);
				}
			}
		}

		// Batched matmul over the whole prompt (weights read once, reused ×N).
		// `lin_b` routes through the offload manager when the layer is
		// GPU-offloaded (see `lin_seq`).

		let n_layers = max_layers.unwrap_or(self.hp.n_layer_main);
		// Context length after this chunk (a full-attention layer's keys).
		let keys_after = rs.pos + N;
		let pipe = snap.is_none() && self.pipe_ok(N);
		if pipe {
			self.layers_pipe(rs, &input, n_layers, xmat, N);
			quant::progress::part(1, 1);
		}
		for il in 0..if pipe { 0 } else { n_layers } {
			let lt = self.get_layer(il);
			let is_full = self.full_attn_layers[il];

			// attn_norm per row, straight to q8_K where every matmul reading
			// it takes the prefill split (`norm_seq_in`).
			let aout: Vec<f32> = if is_full {
				let tn = if measure { Some(Instant::now()) } else { None };
				let norm = self.norm_seq_in(il, xmat, &lt.attn_norm, N, &self.attn_qkv_jobs(&lt));
				if let Some(t) = tn {
					NORM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
				}
				let base = rs.mpos;
				self.attn_seq(
					il,
					&lt,
					&mut rs.kv[il],
					&norm,
					N,
					|t| input.mpos(base, t),
					measure,
				)
			} else {
				// beta/alpha consume `norm` like Wqkv does, so they share its
				// call; the prefill split runs them on the CPU from the same
				// q8_K rows (`OffloadManager::cpu_tail_ok`). Layout is
				// [t * Hv + h].
				let jobs = [
					(WeightId::Wqkv, lt.wqkv.as_ref().unwrap(), conv_dim),
					(
						WeightId::WqkvGate,
						lt.wqkv_gate.as_ref().unwrap(),
						value_dim,
					),
					(WeightId::WssmBeta, lt.ssm_beta.as_ref().unwrap(), Hv),
					(WeightId::WssmAlpha, lt.ssm_alpha.as_ref().unwrap(), Hv),
				];
				let tn = if measure { Some(Instant::now()) } else { None };
				let norm = self.norm_seq_in(il, xmat, &lt.attn_norm, N, &jobs);
				if let Some(t) = tn {
					NORM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
				}
				// `WqkvGate` (z) is read only by pass 3, so where the input is
				// q8_K it is started on the GPU *after* the rest of the group
				// and runs there while the CPU does passes 1 and 2, which
				// otherwise leave the GPU idle (`OffloadManager::matmul_q8_defer`).
				let mut deferred = None;
				let ssm_out = [(WeightId::SsmOut, lt.ssm_out.as_ref().unwrap(), n_embd)];
				let res = match (&norm, self.offload.as_ref()) {
					(crate::lin::SeqIn::Q8(q8), Some(off))
						if off.defer_ok(il, &jobs[1], N, n_embd) =>
					{
						let rest = [jobs[0], jobs[2], jobs[3]];
						self.prefetch_hint(
							il,
							offload::NextCall::Defer(jobs[1].0, jobs[1].1, value_dim),
							N,
							n_embd,
						);
						let mut r = off.matmul_seq_group_q8(il, &rest, q8, N, n_embd);
						self.prefetch_hint(il, offload::NextCall::Split(&ssm_out), N, value_dim);
						deferred =
							off.matmul_q8_defer(il, jobs[1].0, jobs[1].1, value_dim, q8, N, n_embd);
						let gate_now = if deferred.is_none() {
							off.matmul_seq_group_q8(il, &jobs[1..2], q8, N, n_embd)
								.pop()
								.unwrap()
						} else {
							Vec::new()
						};
						r.insert(1, gate_now);
						r
					}
					_ => {
						self.prefetch_hint(il, offload::NextCall::Split(&ssm_out), N, value_dim);
						self.lin_seq_group_in(il, &norm, N, n_embd, &jobs)
					}
				};
				drop(norm);
				let qkv_block = &res[0];
				let (beta_all, alpha_all) = (&res[2], &res[3]);
				// The gated-delta-net recurrence is the ONLY part of this
				// block that must run token-by-token. Everything around it
				// (the causal conv1d, the q/k/v split + l2-norm, the beta/alpha
				// gates and the output norm) is independent per token, so it
				// is hoisted into batched passes that run across the pool. The
				// sequential middle then does nothing but the state update.
				let tssm = if measure { Some(Instant::now()) } else { None };
				let o_b = self.ssm_pass12(
					il,
					&lt,
					rs,
					qkv_block,
					beta_all,
					alpha_all,
					N,
					snap.as_deref_mut(),
				);

				let gate_fin;
				let gate_block: &[f32] = match deferred {
					Some(d) => {
						gate_fin = self.offload.as_ref().unwrap().finish_deferred(d);
						&gate_fin
					}
					None => &res[1],
				};

				// --- pass 3 (parallel over t): per-head RMS norm x silu gate.
				let mut no_block = quant::uninit_vec(N * value_dim);
				{
					let ssm_norm = lt.ssm_norm.as_ref().unwrap();
					let np = no_block.as_mut_ptr() as usize;
					let op = o_b.as_ptr() as usize;
					small_rows(N, |t| unsafe {
						let no = std::slice::from_raw_parts_mut(
							(np as *mut f32).add(t * value_dim),
							value_dim,
						);
						let z = &gate_block[t * value_dim..(t + 1) * value_dim];
						for h in 0..Hv {
							let o = std::slice::from_raw_parts(
								(op as *const f32).add((h * N + t) * S),
								S,
							);
							let n = &mut no[h * S..(h + 1) * S];
							rms_norm_into(o, ssm_norm, eps, n);
							quant::silu_mul_assign(&z[h * S..(h + 1) * S], n);
						}
					});
				}
				if let Some(t) = tssm {
					SSN_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
				}
				self.prefetch_hint(
					il,
					offload::NextCall::Split(&self.ffn_up_jobs(&lt)),
					N,
					n_embd,
				);
				self.lin_seq(
					il,
					WeightId::SsmOut,
					lt.ssm_out.as_ref().unwrap(),
					&no_block,
					n_embd,
					value_dim,
				)
			};

			let next_first = |m: &Model| m.prefetch_first(il + 1, n_layers, N, keys_after);
			self.ffn_seq(il, &lt, xmat, &aout, N, measure, Some(&next_first));
			feature_dump(il, &xmat[..N * n_embd]);
			self.dflash_capture(rs, il, &xmat[..N * n_embd], N);
			quant::progress::part(il + 1, n_layers);
		}

		rs.pos += N;
		rs.mpos = input.mpos_after(rs.mpos, N);
		rs.mrope |= input.has_image_positions();
		if let Some(t0) = t_all {
			let total = t0.elapsed().as_nanos() as u64;
			let (nm, ss, at, el) = (
				NORM_NS.with(|c| c.get()),
				SSN_NS.with(|c| c.get()),
				ATTN_NS.with(|c| c.get()),
				ELEM_NS.with(|c| c.get()),
			);
			let ms = |v: u64| v as f64 / 1e6;
			eprintln!(
				"PREFILL N={N} total_ms={:.1} gemm_ms={:.1} ssmscan_ms={:.1} attn_ms={:.1} norm_ms={:.1} elem_ms={:.1}",
				ms(total),
				ms(total.saturating_sub(nm + ss + at + el)),
				ms(ss), ms(at), ms(nm), ms(el),
			);
			for c in [&NORM_NS, &SSN_NS, &ATTN_NS, &ELEM_NS] {
				c.with(|c| c.set(0));
			}
		}
	}

	/// SSM prefill passes 1 (conv1d + silu, q/k/v split, l2-norm, beta and
	/// gate scalars, batched over `N` positions) and 2 (the state recurrence),
	/// for layer `il` on `N` rows of `Wqkv` / beta / alpha output. Updates
	/// `rs`'s conv window and state; returns the recurrence output `o`,
	/// head-major (`[(h * N + t) * S]`). `snap` as in [`Self::prefill_chunk`].
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn ssm_pass12(
		&self,
		il: usize,
		lt: &LayerTensors,
		rs: &mut RunState,
		qkv_block: &[f32],
		beta_all: &[f32],
		alpha_all: &[f32],
		N: usize,
		mut snap: Option<&mut SpecSnap>,
	) -> Vec<f32> {
		let S = self.hp.ssm_state;
		let Hk = self.hp.ssm_n_group;
		let Hv = self.hp.ssm_dt_rank;
		let value_dim = self.hp.ssm_inner;
		let key_dim = S * Hk;
		let conv_dim = key_dim * 2 + value_dim;
		let eps = self.hp.eps;
		let ncap = if snap.is_some() { N - 1 } else { 0 };
		let ksz = self.hp.conv_kernel;
		let wconv_t = lt.conv1d_t.as_ref().unwrap();

		// --- pass 1 (parallel over t): conv1d + silu, q/k/v split,
		//     l2-norm, and the per-head beta/gate scalars.
		let mut qn_b = quant::uninit_vec(N * key_dim);
		let mut kn_b = quant::uninit_vec(N * key_dim);
		let mut v_b = quant::uninit_vec(N * value_dim);
		let mut beta_b = quant::uninit_vec(N * Hv);
		let mut gate_b = quant::uninit_vec(N * Hv);
		{
			let prev = &rs.convbuf[il];
			let ssm_dt = lt.ssm_dt.as_ref().unwrap();
			let ssm_a = lt.ssm_a.as_ref().unwrap();
			let (qp, kp, vp, bp, gp) = (
				qn_b.as_mut_ptr() as usize,
				kn_b.as_mut_ptr() as usize,
				v_b.as_mut_ptr() as usize,
				beta_b.as_mut_ptr() as usize,
				gate_b.as_mut_ptr() as usize,
			);
			small_rows(N, |t| {
				// Window row `f` is token `t - (ksz-1) + f`; rows from
				// before the chunk come from the carried-in `convbuf`,
				// whose entry `f` holds token `f - (ksz-1)`.
				// Per-worker scratch row: a fresh 40 KB `Vec` per token
				// went through the locked back-end heap from every
				// worker at once (see `rms_norm_into`).
				thread_local! {
					static CONV: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
				}
				let conv_ptr = CONV.with(|cb| {
					let mut cb = cb.borrow_mut();
					cb.resize(conv_dim, 0.0);
					cb.as_mut_ptr()
				});
				// SAFETY: this worker's own buffer; nothing else on this
				// thread touches it until the task returns.
				let conv_out = unsafe { std::slice::from_raw_parts_mut(conv_ptr, conv_dim) };
				let mut rows: [&[f32]; 8] = [&[]; 8];
				assert!(ksz <= rows.len(), "conv kernel {ksz} > 8");
				for f in 0..ksz {
					let u = t as isize - (ksz as isize - 1) + f as isize;
					rows[f] = if u < 0 {
						&prev[(u + ksz as isize - 1) as usize][..]
					} else {
						&qkv_block[u as usize * conv_dim..(u as usize + 1) * conv_dim]
					};
				}
				quant::conv_silu(&rows[..ksz], wconv_t, conv_out);
				// q/k/v are stored **head-major** (`[(h * N + t) * S]`)
				// so pass 2, which walks one head over every `t`,
				// reads each of them as one sequential stream.
				unsafe {
					for h in 0..Hk {
						l2_norm_into(
							&conv_out[h * S..(h + 1) * S],
							eps,
							std::slice::from_raw_parts_mut(
								(qp as *mut f32).add((h * N + t) * S),
								S,
							),
						);
						l2_norm_into(
							&conv_out[key_dim + h * S..key_dim + (h + 1) * S],
							eps,
							std::slice::from_raw_parts_mut(
								(kp as *mut f32).add((h * N + t) * S),
								S,
							),
						);
					}
					for h in 0..Hv {
						std::ptr::copy_nonoverlapping(
							conv_out.as_ptr().add(2 * key_dim + h * S),
							(vp as *mut f32).add((h * N + t) * S),
							S,
						);
					}
					let bs = std::slice::from_raw_parts_mut((bp as *mut f32).add(t * Hv), Hv);
					let gs = std::slice::from_raw_parts_mut((gp as *mut f32).add(t * Hv), Hv);
					for h in 0..Hv {
						bs[h] = sigmoid(beta_all[t * Hv + h]);
						let ab = alpha_all[t * Hv + h] + ssm_dt[h];
						gs[h] = (softplus(ab) * ssm_a[h]).exp();
					}
				}
			});
		}
		// Conv window after token `t` (same rule as the carry below,
		// with `t + 1` tokens consumed), for rollback.
		if let Some(sn) = snap.as_deref_mut() {
			for t in 0..ncap {
				let rows: Vec<Vec<f32>> = (0..ksz - 1)
					.map(|f| {
						let u = (t + 1) as isize - (ksz as isize - 1) + f as isize;
						if u < 0 {
							rs.convbuf[il][(u + ksz as isize - 1) as usize].clone()
						} else {
							qkv_block[u as usize * conv_dim..(u as usize + 1) * conv_dim].to_vec()
						}
					})
					.collect();
				sn.conv[t][il] = rows;
			}
		}
		// carry the last `ksz-1` window rows into the next chunk
		for f in 0..ksz - 1 {
			let u = N as isize - (ksz as isize - 1) + f as isize;
			let row: Vec<f32> = if u < 0 {
				rs.convbuf[il][(u + ksz as isize - 1) as usize].clone()
			} else {
				qkv_block[u as usize * conv_dim..(u as usize + 1) * conv_dim].to_vec()
			};
			rs.convbuf[il][f] = row;
		}

		// --- pass 2 (sequential over t): the state recurrence only.
		// Zeroed per slice by its task, in cache, rather than here by
		// one thread (a ~50 MB memset per layer).
		let mut o_b = quant::uninit_vec(N * value_dim);
		{
			let mut st = std::mem::take(&mut rs.state[il]);
			let qscale = 1.0f32 / (S as f32).sqrt();
			let st_len = st.len();
			let o_ptr = o_b.as_mut_ptr() as usize;
			// Verify batches double-buffer the state: each task copies
			// its column block from the pre-batch state `st` into the
			// spare buffer and scans it there, so `st` is left intact
			// as the rollback base (`ssm_replay`) at no extra traffic;
			// the scan reads and writes every block anyway. (A
			// separate base copy wrote another ~150 MB per verify.)
			let mut work = match snap.as_deref_mut() {
				Some(sn) => {
					let mut d = std::mem::take(&mut sn.base[il]);
					if d.len() != st_len {
						d = vec![0.0; st_len];
					}
					Some(d)
				}
				None => None,
			};
			let src_ptr: usize = if work.is_some() {
				st.as_ptr() as usize
			} else {
				0
			};
			let chk: Vec<usize> = match snap.as_deref_mut() {
				Some(sn) if spec_check() => {
					sn.check.resize_with(self.hp.n_layer_main, Vec::new);
					sn.check[il] = vec![vec![0.0; st_len]; N];
					sn.check[il]
						.iter_mut()
						.map(|v| v.as_mut_ptr() as usize)
						.collect()
				}
				_ => Vec::new(),
			};
			let st_ptr = work.as_mut().unwrap_or(&mut st).as_mut_ptr() as usize;
			// Column `j` of a head's state (and `o[j]`) depends only
			// on column `j`, so one task owns a `(head, column block)`
			// and walks every token itself: one pool dispatch per
			// layer instead of one per token, and the block's
			// `S x JB` slice of state stays in L1 for the whole chunk.
			let jb = ssm_col_block(S);
			let nblk = S / jb;
			quant::par_for(Hv * nblk, |task| {
				let (h, j0) = (task / nblk, (task % nblk) * jb);
				let kh = h % Hk;
				// SAFETY: task `(h, j0)` is the only writer of columns
				// `[j0, j0 + jb)` of head `h`'s state, of the matching
				// `o` slices and of that part of every snapshot.
				let st_h = unsafe {
					std::slice::from_raw_parts_mut((st_ptr as *mut f32).add(h * S * S), S * S)
				};
				// The block is scanned in a compact `S x jb` copy: in
				// place, its rows sit 4 * S bytes apart and alias onto
				// a quarter of the L1 sets (`quant::delta_net_cols_ld`).
				// Verify batches read the pre-batch state `src` and
				// leave it intact, as before.
				thread_local! {
					static BLK: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
				}
				let blk_ptr = BLK.with(|b| {
					let mut b = b.borrow_mut();
					b.resize(S * jb, 0.0);
					b.as_mut_ptr()
				});
				// SAFETY: this worker's own buffer, untouched by
				// anything else on this thread until the task returns.
				let blk = unsafe { std::slice::from_raw_parts_mut(blk_ptr, S * jb) };
				let from: *const f32 = if src_ptr != 0 {
					unsafe { (src_ptr as *const f32).add(h * S * S) }
				} else {
					st_h.as_ptr()
				};
				for i in 0..S {
					unsafe {
						std::ptr::copy_nonoverlapping(
							from.add(i * S + j0),
							blk.as_mut_ptr().add(i * jb),
							jb,
						);
					}
				}
				for t in 0..N {
					// Head-major, see pass 1; `o_b` likewise.
					let kb = (kh * N + t) * S;
					let vb = (h * N + t) * S + j0;
					let o_blk =
						unsafe { std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(vb), jb) };
					o_blk.fill(0.0);
					quant::delta_net_cols_ld(
						blk,
						jb,
						S,
						0,
						&kn_b[kb..kb + S],
						&qn_b[kb..kb + S],
						&v_b[vb..vb + jb],
						beta_b[t * Hv + h],
						gate_b[t * Hv + h],
						qscale,
						o_blk,
					);
					if let Some(&dst) = chk.get(t) {
						for i in 0..S {
							let off = h * S * S + i * S + j0;
							unsafe {
								std::ptr::copy_nonoverlapping(
									blk.as_ptr().add(i * jb),
									(dst as *mut f32).add(off),
									jb,
								);
							}
						}
					}
				}
				for i in 0..S {
					unsafe {
						std::ptr::copy_nonoverlapping(
							blk.as_ptr().add(i * jb),
							st_h.as_mut_ptr().add(i * S + j0),
							jb,
						);
					}
				}
			});
			debug_assert_eq!(st.len(), st_len);
			match (work, snap.as_deref_mut()) {
				(Some(w), Some(sn)) => {
					rs.state[il] = w;
					sn.base[il] = st;
				}
				_ => rs.state[il] = st,
			}
		}
		if let Some(sn) = snap {
			sn.inp[il] = Some(SsmReplay {
				n: N,
				q: std::mem::take(&mut qn_b),
				k: std::mem::take(&mut kn_b),
				v: std::mem::take(&mut v_b),
				beta: std::mem::take(&mut beta_b),
				gate: std::mem::take(&mut gate_b),
			});
		}
		o_b
	}

	/// `FfnUp` and `FfnGate`, the group the FFN's first matmul runs.
	pub(crate) fn ffn_up_jobs<'a>(
		&self,
		lt: &'a LayerTensors,
	) -> [(WeightId, &'a LinearWeight, usize); 2] {
		[
			(WeightId::FfnUp, &lt.ffn_up, self.hp.n_ff),
			(WeightId::FfnGate, &lt.ffn_gate, self.hp.n_ff),
		]
	}

	/// Tell the offload manager which prefill matmul (on layer `il`'s
	/// weights) follows the one about to run, so it streams that one's
	/// weights in meanwhile (`OffloadManager::prefetch_hint`). Only for a
	/// cached layer, whose `Arc` then keeps the bytes alive for the upload.
	pub(crate) fn prefetch_hint(
		&self,
		il: usize,
		call: offload::NextCall<'_>,
		seq: usize,
		in_dim: usize,
	) {
		let Some(off) = self.offload.as_ref() else {
			return;
		};
		let Some(keep) = self.layer_cache.lock().unwrap().get(il).cloned().flatten() else {
			return;
		};
		off.prefetch_hint(il, call, seq, in_dim, keep);
	}

	/// [`Self::prefetch_hint`] for the first matmul of layer `il` (its
	/// QKV group), which follows the previous layer's `FfnDown`. `keys`:
	/// the context length after the chunk.
	pub(crate) fn prefetch_first(&self, il: usize, n_layers: usize, seq: usize, keys: usize) {
		if il >= n_layers {
			return;
		}
		let Some(lt) = self.layer_cache.lock().unwrap().get(il).cloned().flatten() else {
			return;
		};
		let Some(off) = self.offload.as_ref() else {
			return;
		};
		let n_embd = self.hp.n_embd;
		if self.full_attn_layers[il] {
			let (hd, nh, nkv) = (self.hp.head_dim, self.hp.n_head, self.hp.n_head_kv);
			let mut jobs = self.attn_qkv_jobs(&lt);
			if off.qgate_defer_ok(il, jobs[0].1, hd * nh, seq, n_embd, keys, nh, nkv, hd) {
				jobs[0].2 = hd * nh;
			}
			self.prefetch_hint(il, offload::NextCall::Split(&jobs), seq, n_embd);
		} else {
			let gate = (
				WeightId::WqkvGate,
				lt.wqkv_gate.as_ref().unwrap(),
				self.hp.ssm_inner,
			);
			let qkv = (
				WeightId::Wqkv,
				lt.wqkv.as_ref().unwrap(),
				self.hp.ssm_state * self.hp.ssm_n_group * 2 + self.hp.ssm_inner,
			);
			if off.defer_ok(il, &gate, seq, n_embd) {
				self.prefetch_hint(il, offload::NextCall::Split(&[qkv]), seq, n_embd);
			} else {
				self.prefetch_hint(il, offload::NextCall::Split(&[qkv, gate]), seq, n_embd);
			}
		}
	}

	/// The gated full-attention block over `N` rows of `norm` (the
	/// attn-normed hidden states): QKV, Q/K norms, MRoPE at `mpos(t)`, append
	/// to `kv`, causal attention over everything in `kv`, sigmoid gate, `Wo`.
	/// Shared by the main stack's full-attention layers and the MTP head.
	/// The full-attention block's QKV matmuls, which share the attn-normed
	/// input.
	pub(crate) fn attn_qkv_jobs<'a>(
		&self,
		lt: &'a LayerTensors,
	) -> [(WeightId, &'a LinearWeight, usize); 3] {
		let (head_dim, n_head, n_head_kv) = (self.hp.head_dim, self.hp.n_head, self.hp.n_head_kv);
		[
			(WeightId::Wq, lt.wq.as_ref().unwrap(), 2 * head_dim * n_head),
			(WeightId::Wk, lt.wk.as_ref().unwrap(), head_dim * n_head_kv),
			(WeightId::Wv, lt.wv.as_ref().unwrap(), head_dim * n_head_kv),
		]
	}

	pub(crate) fn attn_seq(
		&self,
		il: usize,
		lt: &LayerTensors,
		kv: &mut Option<crate::state::KvCache>,
		norm: &crate::lin::SeqIn,
		N: usize,
		mpos: impl Fn(usize) -> [i64; 3],
		measure: bool,
	) -> Vec<f32> {
		let n_embd = self.hp.n_embd;
		let head_dim = self.hp.head_dim;
		let n_head = self.hp.n_head;
		let n_head_kv = self.hp.n_head_kv;
		// `Wq`'s rows are every head's query, then every head's output gate
		// (`Model::q_gate_rows_split`). The gate is read only after the
		// attention, so where the input is q8_K the query rows run with
		// `Wk` / `Wv` and the gate rows are started on the GPU afterwards,
		// to run while the CPU does the attention, which otherwise leaves
		// the GPU idle (`OffloadManager::matmul_q8_defer_rows`).
		let n_q = head_dim * n_head;
		let wq = lt.wq.as_ref().unwrap();
		let keys = kv
			.as_ref()
			.map_or(0, |(k, _)| k.len() / (head_dim * n_head_kv))
			+ N;
		let mut deferred = None;
		let wo = [(WeightId::Wo, lt.wo.as_ref().unwrap(), n_embd)];
		let res = match (norm, self.offload.as_ref()) {
			(crate::lin::SeqIn::Q8(q8), Some(off))
				if off
					.qgate_defer_ok(il, wq, n_q, N, n_embd, keys, n_head, n_head_kv, head_dim) =>
			{
				let mut jobs = self.attn_qkv_jobs(lt);
				jobs[0].2 = n_q;
				self.prefetch_hint(
					il,
					offload::NextCall::DeferRows(WeightId::Wq, wq, n_q, n_q),
					N,
					n_embd,
				);
				let r = off.matmul_seq_group_q8(il, &jobs, q8, N, n_embd);
				self.prefetch_hint(il, offload::NextCall::Split(&wo), N, n_q);
				deferred = Some(
					off.matmul_q8_defer_rows(il, WeightId::Wq, wq, n_q, n_q, q8, N, n_embd)
						.ok_or_else(|| {
							offload::OffloadManager::gemm_q8_rows(wq, n_q, n_q, n_embd, N, q8)
						}),
				);
				r
			}
			_ => {
				self.prefetch_hint(il, offload::NextCall::Split(&wo), N, n_q);
				self.lin_seq_group_in(il, norm, N, n_embd, &self.attn_qkv_jobs(lt))
			}
		};
		// Row stride of `qfull`: the query rows alone, or query + gate.
		let qstride = if deferred.is_some() { n_q } else { 2 * n_q };
		let qfull = &res[0];
		let kblk = &res[1];
		let vblk = &res[2];

		let tatt = if measure { Some(Instant::now()) } else { None };
		let mut out_block = self.attn_core(il, lt, kv, qfull, qstride, kblk, vblk, N, &mpos);
		match deferred {
			Some(d) => {
				let gate_block = match d {
					Ok(d) => self.offload.as_ref().unwrap().finish_deferred(d),
					Err(g) => g,
				};
				for (o, g) in out_block.iter_mut().zip(&gate_block) {
					*o *= sigmoid(*g);
				}
			}
			None => {
				for (t, o) in out_block.chunks_exact_mut(n_q).enumerate() {
					for (o, g) in o
						.iter_mut()
						.zip(&qfull[t * qstride + n_q..(t + 1) * qstride])
					{
						*o *= sigmoid(*g);
					}
				}
			}
		}
		if let Some(t) = tatt {
			ATTN_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
		}
		self.prefetch_hint(
			il,
			offload::NextCall::Split(&self.ffn_up_jobs(lt)),
			N,
			n_embd,
		);
		self.lin_seq(
			il,
			WeightId::Wo,
			lt.wo.as_ref().unwrap(),
			&out_block,
			n_embd,
			head_dim * n_head,
		)
	}

	/// The attention proper of a full-attention layer, on `N` rows of its
	/// QKV output (`qfull`: query rows at stride `qstride`, gate rows ignored;
	/// `kblk` / `vblk`): q/k norms + MRoPE at `mpos(t)`, append to `kv`,
	/// causal attention over everything in `kv`. Returns the ungated
	/// `[N x n_head * head_dim]` attention output.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn attn_core(
		&self,
		il: usize,
		lt: &LayerTensors,
		kv: &mut Option<crate::state::KvCache>,
		qfull: &[f32],
		qstride: usize,
		kblk: &[f32],
		vblk: &[f32],
		N: usize,
		mpos: &dyn Fn(usize) -> [i64; 3],
	) -> Vec<f32> {
		let _ = il;
		let eps = self.hp.eps;
		let head_dim = self.hp.head_dim;
		let n_head = self.hp.n_head;
		let n_head_kv = self.hp.n_head_kv;
		let n_q = head_dim * n_head;
		// Per-position q/k norms + RoPE + gate, written straight into the
		// blocks. Independent per position, so it runs across the pool; it was
		// a serial loop with five `Vec`s per position.
		let mut qn_block = quant::uninit_vec(N * head_dim * n_head);
		let mut kn_block = quant::uninit_vec(N * head_dim * n_head_kv);
		{
			let poss: Vec<[i64; 3]> = (0..N).map(mpos).collect();
			let (qp, kp) = (
				qn_block.as_mut_ptr() as usize,
				kn_block.as_mut_ptr() as usize,
			);
			let (q_norm, k_norm) = (lt.q_norm.as_ref().unwrap(), lt.k_norm.as_ref().unwrap());
			let hp = &self.hp;
			small_rows(N, |t| {
				let qf = &qfull[t * qstride..(t + 1) * qstride];
				// SAFETY: position `t` owns row `t` of each block.
				let (qn, kn) = unsafe {
					(
						std::slice::from_raw_parts_mut((qp as *mut f32).add(t * n_q), n_q),
						std::slice::from_raw_parts_mut(
							(kp as *mut f32).add(t * head_dim * n_head_kv),
							head_dim * n_head_kv,
						),
					)
				};
				for h in 0..n_head {
					let qh = &mut qn[h * head_dim..(h + 1) * head_dim];
					rms_norm_into(&qf[h * head_dim..(h + 1) * head_dim], q_norm, eps, qh);
					rope_apply(qh, poss[t], hp);
				}
				let kr = &kblk[t * head_dim * n_head_kv..(t + 1) * head_dim * n_head_kv];
				for h in 0..n_head_kv {
					let kh = &mut kn[h * head_dim..(h + 1) * head_dim];
					rms_norm_into(&kr[h * head_dim..(h + 1) * head_dim], k_norm, eps, kh);
					rope_apply(kh, poss[t], hp);
				}
			});
		}

		// Append the whole block to the KV cache, then attend in parallel
		// over positions (each position writes a disjoint slice of
		// `out_block`; the KV cache is read-only during the scan).
		if kv.is_none() {
			*kv = Some((vec![], vec![]));
		}
		let (kcache, vcache) = kv.as_mut().unwrap();
		quant::kv_extend(kcache, &kn_block);
		quant::kv_extend(vcache, vblk);
		let Tprev = kcache.len() / (head_dim * n_head_kv) - N;
		let scale = 1.0f32 / (head_dim as f32).sqrt();
		let stride = head_dim * n_head;
		let mut out_block = vec![0.0f32; N * stride];

		// Tiled causal GQA attention: each K/V block is reused by a tile of
		// query positions x the group's heads while cache-resident (the
		// per-row scan it replaced re-streamed the cache per query row and
		// was half of a 10k-token prefill).
		// Long contexts split the KV groups with the otherwise idle GPU
		// (`OffloadManager::attn_prefill`); it declines short ones.
		let on_gpu = self.offload.as_ref().is_some_and(|o| {
			o.attn_prefill(
				&qn_block,
				kcache,
				vcache,
				N,
				Tprev,
				n_head,
				n_head_kv,
				head_dim,
				scale,
				&mut out_block,
			)
		});
		if !on_gpu {
			quant::attn_causal(
				&qn_block,
				kcache,
				vcache,
				N,
				Tprev,
				n_head,
				n_head_kv,
				head_dim,
				scale,
				&mut out_block,
			);
		}
		out_block
	}

	/// Residual add, post-attention norm, SwiGLU FFN and the second residual,
	/// in place on `xmat` (`N` rows): `x += aout; x += ffn(norm(x))`.
	pub(crate) fn ffn_seq(
		&self,
		il: usize,
		lt: &LayerTensors,
		xmat: &mut [f32],
		aout: &[f32],
		N: usize,
		measure: bool,
		next_first: Option<&dyn Fn(&Model)>,
	) {
		let n_embd = self.hp.n_embd;
		// residual add (in place: `xmat` holds the post-attention residual
		// stream from here on) + post_norm + SwiGLU FFN, all batched over the
		// block. The norm goes straight to q8_K where both FFN matmuls take
		// the prefill split (`norm_seq_in`).
		let tel = if measure { Some(Instant::now()) } else { None };
		{
			let (xp, op) = (xmat.as_mut_ptr() as usize, aout.as_ptr() as usize);
			small_rows(N, |t| unsafe {
				let x = std::slice::from_raw_parts_mut((xp as *mut f32).add(t * n_embd), n_embd);
				let o = std::slice::from_raw_parts((op as *const f32).add(t * n_embd), n_embd);
				for d in 0..n_embd {
					x[d] += o[d];
				}
			});
		}
		let jobs = self.ffn_up_jobs(lt);
		let post = self.norm_seq_in(il, xmat, &lt.post_norm, N, &jobs);
		if let Some(t) = tel {
			ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
		}
		let down = [(WeightId::FfnDown, &lt.ffn_down, n_embd)];
		self.prefetch_hint(il, offload::NextCall::Split(&down), N, self.hp.n_ff);
		let res = self.lin_seq_group_in(il, &post, N, n_embd, &jobs);
		drop(post);
		if let Some(f) = next_first {
			f(self);
		}
		let up = &res[0];
		let fg = &res[1];
		let n_ff = self.hp.n_ff;
		// Where `FfnDown` takes the prefill row split, its input goes straight
		// from `silu(gate) * up` to q8_K one row at a time, never through an
		// `f32` matrix (`quant::par_silu_mul_q8k`); bit-identical either way.
		let fused = self
			.offload
			.as_ref()
			.filter(|o| N > 1 && crate::lin::fused_swiglu() && o.prefill_q8_ok(il, &down, N, n_ff));
		let ffn_out = if let Some(off) = fused {
			let tel2 = if measure { Some(Instant::now()) } else { None };
			let q8 = std::sync::Arc::new(quant::par_silu_mul_q8k(fg, up, N, n_ff));
			if let Some(t) = tel2 {
				ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
			}
			off.matmul_seq_group_q8(il, &down, &q8, N, n_ff)
				.pop()
				.unwrap()
		} else {
			let tel2 = if measure { Some(Instant::now()) } else { None };
			let mut hb = quant::uninit_vec(N * n_ff);
			{
				let (hp, up_p, fg_p) = (
					hb.as_mut_ptr() as usize,
					up.as_ptr() as usize,
					fg.as_ptr() as usize,
				);
				small_rows(N, |t| unsafe {
					let h = std::slice::from_raw_parts_mut((hp as *mut f32).add(t * n_ff), n_ff);
					let u = std::slice::from_raw_parts((up_p as *const f32).add(t * n_ff), n_ff);
					let f = std::slice::from_raw_parts((fg_p as *const f32).add(t * n_ff), n_ff);
					quant::silu_mul(f, u, h);
				});
			}
			if let Some(t) = tel2 {
				ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
			}
			self.lin_seq(
				il,
				WeightId::FfnDown,
				&lt.ffn_down,
				&hb,
				n_embd,
				self.hp.n_ff,
			)
		};
		let tel3 = if measure { Some(Instant::now()) } else { None };
		{
			let (xp, fp) = (xmat.as_mut_ptr() as usize, ffn_out.as_ptr() as usize);
			small_rows(N, |t| unsafe {
				let x = std::slice::from_raw_parts_mut((xp as *mut f32).add(t * n_embd), n_embd);
				let f = std::slice::from_raw_parts((fp as *const f32).add(t * n_embd), n_embd);
				for d in 0..n_embd {
					x[d] += f[d];
				}
			});
		}
		if let Some(t) = tel3 {
			ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
		}
	}

	/// Batched prefill: run the prompt `[tokens]` through the layer stack in
	/// `n_batch`-sized chunks (see `prefill_chunk`), then return the LM-head
	/// logits for the **last** prompt token. Chunked so a very long prompt (e.g.
	/// a 200k context) stays within memory: each chunk re-streams the weights
	/// once but only materializes `n_batch` rows of activations, and the KV
	/// cache / SSM state carry across chunks. `rs` is advanced exactly as `N`
	/// `forward_step` calls would, so subsequent decode continues correctly.
	pub fn forward_prefill(
		&self,
		rs: &mut RunState,
		tokens: &[u32],
		max_layers: Option<usize>,
	) -> (Vec<f32>, Vec<f32>) {
		self.forward_prefill_mm(rs, PromptInput::text(tokens), max_layers)
	}

	/// Batched prefill over a prompt that may carry image embeddings and
	/// explicit MRoPE positions. [`forward_prefill`] is this with neither.
	///
	/// Chunking splits `input` by token index, and both the embedding
	/// overrides and the position triples are sliced with it, so an image that
	/// straddles an `n_batch` boundary is handled exactly as one that does not.
	pub fn forward_prefill_mm(
		&self,
		rs: &mut RunState,
		input: PromptInput<'_>,
		max_layers: Option<usize>,
	) -> (Vec<f32>, Vec<f32>) {
		let n_embd = self.hp.n_embd;
		let tokens = input.tokens;
		let N = tokens.len();
		let eps = self.hp.eps;
		// Reused activation buffer; only the last chunk's final row is needed.
		let mut xbuf = vec![0.0f32; self.n_batch * n_embd];
		let mut last_len = 0usize;
		let mut idx = 0usize;
		// The DFlash2 drafter only ever looks back `window` positions, so only
		// the prompt's tail is captured and projected into its context.
		let end_pos = rs.pos + N;
		let df_from = self
			.dflash
			.as_ref()
			.map_or(usize::MAX, |d| end_pos.saturating_sub(d.window));
		let bar = quant::progress::Prefill::start(N);
		while idx < N {
			let e = (idx + self.n_batch).min(N);
			let c = e - idx;
			quant::progress::chunk(idx, c);
			let start = rs.mpos;
			let cp = rs.pos;
			let capture = max_layers.is_none() && cp + c > df_from;
			if let (true, Some(d)) = (capture, rs.dflash.as_mut()) {
				d.capture = Some(Vec::new());
			}
			self.prefill_chunk(
				rs,
				input.slice(idx, e),
				max_layers,
				&mut xbuf[..c * n_embd],
				None,
			);
			if let Some(cap) = rs
				.dflash
				.as_mut()
				.and_then(|d| d.capture.take())
				.filter(|_| capture)
			{
				let from = df_from.max(cp);
				let f = cap.len() / c;
				self.dflash_push(rs, from, &cap[(from - cp) * f..]);
				let mut d = rs.dflash.take().unwrap();
				self.dflash_absorb(&mut d);
				rs.dflash = Some(d);
			}
			if max_layers.is_none() && rs.mtp.is_some() {
				self.mtp_absorb(rs, input.slice(idx, e), start, &xbuf[..c * n_embd]);
			}
			if max_layers.is_none() && rs.dflash.is_some() {
				self.dflash_note(input.slice(idx, e).tokens);
			}
			last_len = c;
			idx = e;
		}
		bar.done();
		let last_x = &xbuf[(last_len - 1) * n_embd..last_len * n_embd];
		let norm_last = rms_norm(last_x, &self.output_norm, eps);
		let logits = apply_linear(&self.output, &norm_last, self.hp.n_vocab, n_embd);
		// The last prompt position's logits, raw f32: how prefill-path variants
		// (GPU GEMM kernels, split fractions) are compared, since `QWEN_DUMP`
		// (`--forward`) runs the decode path.
		if let Ok(p) = std::env::var("QWEN_PREFILL_DUMP") {
			let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
			if let Err(e) = std::fs::write(&p, bytes) {
				eprintln!("QWEN_PREFILL_DUMP {p}: {e}");
			}
		}
		let dc = last_x.iter().sum::<f32>() / last_x.len() as f32;
		let mut logits_dc = vec![0.0f32; self.hp.n_vocab];
		for o in 0..self.hp.n_vocab {
			logits_dc[o] = logits[o] - dc * self.output_row_sum[o];
		}

		// The sequential reference below re-feeds plain tokens, which cannot
		// reproduce an image, so the check is only meaningful for text.
		if std::env::var("QWEN_PREFILL_CHECK").is_ok() && input.embeds.is_none() {
			let mut rs2 = self.new_run_state();
			let mut ref_last = (vec![0.0f32; self.hp.n_vocab], vec![0.0f32; self.hp.n_vocab]);
			for &tk in tokens {
				ref_last = self.forward_step(&mut rs2, tk, max_layers);
			}
			let argmax = |v: &[f32]| -> usize {
				v.iter()
					.enumerate()
					.fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
						if v > bv {
							(i, v)
						} else {
							(bi, bv)
						}
					})
					.0
			};
			let mut md = 0.0f32;
			for o in 0..self.hp.n_vocab {
				let d = (logits[o] - ref_last.0[o]).abs();
				if d > md {
					md = d;
				}
			}
			eprintln!(
				"PREFILL_CHECK max_logit_diff={:.6} argmax_prefill={} argmax_ref={}",
				md,
				argmax(&logits),
				argmax(&ref_last.0)
			);
		}
		(logits, logits_dc)
	}
}
