//! MTP (multi-token prediction) speculative decoding.
//!
//! The GGUF carries one extra decoder block past the main stack
//! (`blk.{n_layer_main}`, `qwen35.nextn_predict_layers = 1`): a full-attention
//! layer that, given the embedding of token `x_p` and the main model's final
//! output-normed hidden state `h_{p-1}`, predicts `x_{p+1}`. Its graph is
//! llama.cpp's `llama_model_qwen35::graph_mtp`:
//!
//! ```text
//! x   = eh_proj(concat(enorm(emb(x_p)), hnorm(h_{p-1})))
//! x  += attn(attn_norm(x)): own KV cache, MRoPE at position p
//! x  += ffn(post_attention_norm(x))
//! h'  = shared_head_norm(x);  logits = output(h'): the shared lm_head
//! ```
//!
//! Decode then works in steps. With `x_n` sampled but not yet fed:
//!
//! 1. **Draft.** The MTP block runs on `(x_n, h_{n-1})` and proposes `d_0`;
//!    with `n_draft > 1` it chains on its own `h'` for `d_1, …`.
//! 2. **Verify.** The main model runs `[x_n, d_0, …, d_{k-1}]` as one batch.
//!    Decode is DRAM-bound, and a 2-4 position batch reads each weight once for
//!    1.06-1.23x the cost of one token (`quant/examples/smallseq.rs`), so this
//!    yields the logits of `k + 1` positions for about the price of one.
//! 3. **Accept.** The caller samples position by position from those logits and
//!    keeps a draft only while it equals what was sampled. Every emitted token is
//!    therefore drawn from the main model's own distribution, at any temperature,
//!    exactly as without MTP; the drafts only decide how many positions one
//!    verify pass pays for.
//! 4. **Commit.** The rejected tail is cut off the main state. KV caches truncate;
//!    the SSM layers' recurrent state cannot be rolled back, so the verify pass
//!    snapshots it after each token (`SpecSnap`) and the commit swaps the right
//!    one in.
//!
//! The MTP head keeps its own KV cache over the whole context, like llama.cpp's
//! draft context: prefill runs it over the prompt, and the positions a verify
//! pass accepts are queued (`MtpRun::pending`) and folded into the next draft's
//! batch, so the head's weights are read once per step.

#![allow(non_snake_case)]

use super::*;
use crate::suffix::{suffix_enabled, suffix_min_model, suffix_min_plain, SuffixIndex};

/// Counters for one [`Decoder`] run.
#[derive(Default, Clone, Debug)]
pub struct MtpStats {
	/// Verify passes.
	pub steps: usize,
	/// Tokens drafted / accepted over those passes.
	pub drafted: usize,
	pub accepted: usize,
	pub draft_ns: u64,
	pub verify_ns: u64,
	/// The subset of the above whose drafts came from the context.
	pub suffix_steps: usize,
	pub suffix_drafted: usize,
	pub suffix_accepted: usize,
}

impl MtpStats {
	pub(crate) fn note(
		&mut self,
		drafts: &[u32],
		draft: std::time::Duration,
		verify: std::time::Duration,
		suffix: bool,
	) {
		self.steps += 1;
		self.drafted += drafts.len();
		self.draft_ns += draft.as_nanos() as u64;
		self.verify_ns += verify.as_nanos() as u64;
		if suffix {
			self.suffix_steps += 1;
			self.suffix_drafted += drafts.len();
		}
	}

	/// Add another run's counters.
	pub fn add(&mut self, o: &MtpStats) {
		self.steps += o.steps;
		self.drafted += o.drafted;
		self.accepted += o.accepted;
		self.draft_ns += o.draft_ns;
		self.verify_ns += o.verify_ns;
		self.suffix_steps += o.suffix_steps;
		self.suffix_drafted += o.suffix_drafted;
		self.suffix_accepted += o.suffix_accepted;
	}

	/// One-line summary for the `[perf]` output.
	pub fn summary(&self) -> String {
		let rate = if self.drafted > 0 {
			self.accepted as f64 / self.drafted as f64
		} else {
			0.0
		};
		let per = |ns: u64| {
			if self.steps > 0 {
				ns as f64 / 1e6 / self.steps as f64
			} else {
				0.0
			}
		};
		format!(
			"mtp: {} steps, accepted {}/{} drafts ({:.1} %), {:.2} tokens/step, draft {:.1} ms + verify {:.1} ms per step",
			self.steps,
			self.accepted,
			self.drafted,
			rate * 100.0,
			if self.steps > 0 { (self.steps + self.accepted) as f64 / self.steps as f64 } else { 0.0 },
			per(self.draft_ns),
			per(self.verify_ns),
		) + &if self.suffix_steps > 0 {
			format!(
				"; context drafts: {} steps, accepted {}/{}",
				self.suffix_steps, self.suffix_accepted, self.suffix_drafted
			)
		} else {
			String::new()
		}
	}
}

/// One draft + verify pass, waiting for the caller to decide how much of it
/// to keep ([`Model::spec_commit`]).
pub struct Spec {
	/// The drafted tokens `d_0 .. d_{k-1}`.
	pub drafts: Vec<u32>,
	/// `[(k + 1) * n_vocab]`: row `i` is the distribution of the token after
	/// verify token `i` (`x_n`, then `d_0`, …).
	pub(crate) logits: Vec<f32>,
	/// `[(k + 1) * n_embd]`: output-normed main hidden state of each row.
	pub(crate) hs: Vec<f32>,
	/// `[x_n, d_0, .., d_{k-1}]`.
	pub(crate) tokens: Vec<u32>,
	pub(crate) base_pos: usize,
	pub(crate) base_mpos: [i64; 3],
	/// MTP KV length (floats) that holds only real positions (`..= base_pos`);
	/// the chained drafts' rows past it are discarded at commit.
	pub(crate) mtp_kv_len: usize,
	/// DFlash2 only: `[(k + 1) * feat]` drafter feature rows captured by the
	/// verify pass; the kept ones become drafter context at commit.
	pub(crate) feats: Vec<f32>,
	/// Drafts came from the context ([`crate::suffix`]), not a model drafter.
	pub suffix: bool,
	/// Rollback scratch of a draft-free run (no MTP / DFlash state to own it).
	pub(crate) snap: Option<SpecSnap>,
	/// Drafter's runner-up candidate and its confidence in the draft, per
	/// depth (empty for context drafts). `QWEN_TREE_LOG` measurement only.
	pub(crate) alts: Vec<u32>,
	pub(crate) probs: Vec<f32>,
}

impl Spec {
	/// Logits row `i` (see [`Spec::logits`]).
	pub fn logits(&self, i: usize, n_vocab: usize) -> &[f32] {
		&self.logits[i * n_vocab..(i + 1) * n_vocab]
	}
}

fn argmax(v: &[f32]) -> usize {
	let mut bi = 0;
	let mut bv = f32::NEG_INFINITY;
	for (i, &x) in v.iter().enumerate() {
		if x > bv {
			bv = x;
			bi = i;
		}
	}
	bi
}

/// Softmax probability of the argmax (the draft's confidence).
fn top_prob(v: &[f32], am: usize) -> f32 {
	let m = v[am];
	let s: f32 = v.iter().map(|&x| (x - m).exp()).sum();
	1.0 / s
}

/// `QWEN_MTP_PMIN` (default 0.8, `0` disables): stop chaining drafts once the
/// MTP head's top probability falls below this. The first draft is always
/// made (verifying one extra position costs ~6 %, so even a weak guess pays
/// when it lands) but each chained one costs a pass through the head and its
/// 1 GB lm_head (~24 ms). Measured with 3 drafts (tok/s, code at temp 0 /
/// prose at temp 0.7): no cut-off 9.14 / 4.79, 0.5 9.19 / 5.11, 0.8 9.16 /
/// 5.26; the best of both, where fixed 1 draft is 7.03 / 5.32.
/// `QWEN_TREE_LOG=1`: per verify step, log where the drafts were rejected and
/// whether the drafter's runner-up at that depth was the sampled token.
fn tree_log() -> bool {
	static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*V.get_or_init(|| std::env::var("QWEN_TREE_LOG").is_ok_and(|v| v != "0"))
}

fn mtp_pmin() -> f32 {
	quant::knob::parse("QWEN_MTP_PMIN").unwrap_or(0.8)
}

impl Model {
	/// Whether this model drafts with an MTP head (`enable_mtp`).
	pub fn mtp_enabled(&self) -> bool {
		self.mtp.is_some() && self.n_draft > 0
	}

	/// The most drafts per step the drafter was enabled with (MTP's
	/// `--mtp N`, DFlash's load-time `QWEN_DFLASH_N`); 0 without one.
	pub fn draft_cap(&self) -> usize {
		if self.mtp_enabled() || self.dflash_enabled() {
			self.n_draft
		} else {
			0
		}
	}

	/// Drafts per step now: `QWEN_MTP_N` / `QWEN_DFLASH_N` as the thread's
	/// profile or the environment gives it (`quant::knob`), at most the
	/// count the drafter was enabled with (`--mtp N`, the load-time
	/// `QWEN_DFLASH_N`: the verify buffers are sized for it).
	pub(crate) fn draft_n(&self) -> usize {
		let k = if self.dflash.is_some() {
			"QWEN_DFLASH_N"
		} else {
			"QWEN_MTP_N"
		};
		quant::knob::parse::<usize>(k).map_or(self.n_draft, |n| n.clamp(1, self.n_draft.max(1)))
	}

	/// Row-wise RMS norm of `x` (`rows * n_embd`) with weight `w`.
	pub(crate) fn rows_norm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
		let n = self.hp.n_embd;
		let eps = self.hp.eps;
		let rows = x.len() / n;
		let mut out = vec![0.0f32; x.len()];
		let op = out.as_mut_ptr() as usize;
		quant::par_for(rows, |t| {
			let r = rms_norm(&x[t * n..(t + 1) * n], w, eps);
			// SAFETY: each `t` owns a disjoint row.
			unsafe { std::slice::from_raw_parts_mut((op as *mut f32).add(t * n), n) }
				.copy_from_slice(&r);
		});
		out
	}

	/// The MTP block's input row: `concat(enorm(e), hnorm(h))`.
	fn mtp_in_row(&self, e: &[f32], h: &[f32], dst: &mut Vec<f32>) {
		let mt = self.mtp.as_ref().unwrap();
		dst.extend_from_slice(&rms_norm(e, &mt.enorm, self.hp.eps));
		dst.extend_from_slice(&rms_norm(h, &mt.hnorm, self.hp.eps));
	}

	/// Run the MTP block over consecutive positions: `ein` holds their input
	/// rows (`mtp_in_row`, `2 * n_embd` each) and `mpos` their MRoPE
	/// positions. Appends one row per position to `kv`. With `want`, returns
	/// the last row's draft (argmax of its logits, over the reduced draft
	/// vocabulary when there is one), that draft's softmax probability, and
	/// the row's head-normed hidden state (the input of a chained draft).
	fn mtp_run(
		&self,
		kv: &mut Option<crate::state::KvCache>,
		ein: &[f32],
		mpos: &[[i64; 3]],
		want: bool,
	) -> Option<(u32, f32, Vec<f32>, u32)> {
		let mt = self.mtp.as_ref().expect("MTP head not loaded");
		let n = self.hp.n_embd;
		let M = mpos.len();
		if M == 0 {
			return None;
		}
		let il = self.hp.n_layer_main;
		let lt = &mt.layer;
		let mut x = self.lin_seq(il, WeightId::EhProj, &mt.eh_proj, ein, n, 2 * n);
		let norm = self.rows_norm(&x, &lt.attn_norm);
		let aout = self.attn_seq(
			il,
			lt,
			kv,
			&crate::lin::SeqIn::F32(norm),
			M,
			|t| mpos[t],
			false,
		);
		self.ffn_seq(il, lt, &mut x, &aout, M, false, None);
		if !want {
			return None;
		}
		let h = rms_norm(&x[(M - 1) * n..], &mt.head_norm, self.hp.eps);
		let (d, p, alt) = match &self.draft_vocab {
			Some(dv) => dv.draft(&self.output, &h),
			None => {
				let mut logits = apply_linear(&self.output, &h, self.hp.n_vocab, n);
				let am = argmax(&logits);
				let p = top_prob(&logits, am);
				logits[am] = f32::NEG_INFINITY;
				(am as u32, p, argmax(&logits) as u32)
			}
		};
		Some((d, p, h, alt))
	}

	/// After a prefill chunk: give the MTP head its rows for the chunk's
	/// positions (plus anything still pending before them). `x` is the chunk's
	/// final hidden states (pre output norm), `start` the MRoPE position of its
	/// first token.
	pub(crate) fn mtp_absorb(
		&self,
		rs: &mut RunState,
		input: PromptInput<'_>,
		start: [i64; 3],
		x: &[f32],
	) {
		let n = self.hp.n_embd;
		let Some(run) = rs.mtp.as_mut() else { return };
		let N = input.tokens.len();
		if let Some(dv) = &self.draft_vocab {
			dv.note(&self.output, input.tokens);
		}
		let hs = self.rows_norm(x, &self.output_norm);
		let P = run.pending.len();
		let mut ein = Vec::with_capacity((P + N) * 2 * n);
		let mut mpos = Vec::with_capacity(P + N);
		for e in run.pending.drain(..) {
			self.mtp_in_row(&e.emb, &e.h, &mut ein);
			mpos.push(e.mpos);
		}
		for t in 0..N {
			let h = if t == 0 {
				&run.last_h[..]
			} else {
				&hs[(t - 1) * n..t * n]
			};
			match input.embed(t) {
				Some(e) => self.mtp_in_row(e, h, &mut ein),
				None => {
					let e = self.embedding(input.tokens[t]);
					self.mtp_in_row(&e, h, &mut ein)
				}
			}
			mpos.push(input.mpos(start, t));
		}
		run.last_h.copy_from_slice(&hs[(N - 1) * n..N * n]);
		self.mtp_run(&mut run.kv, &ein, &mpos, false);
	}

	/// Bookkeeping for a plain `forward_step` of `token` at MRoPE position
	/// `mpos`, whose output-normed hidden state is `h`: queue its MTP row.
	/// The queue is flushed through the head once it grows, so a long run of
	/// plain decode cannot let it accumulate.
	pub(crate) fn mtp_note_step(
		&self,
		rs: &mut RunState,
		emb: Vec<f32>,
		mpos: [i64; 3],
		h: &[f32],
	) {
		let Some(run) = rs.mtp.as_mut() else { return };
		let prev = std::mem::replace(&mut run.last_h, h.to_vec());
		run.pending.push(MtpEntry { emb, h: prev, mpos });
		if run.pending.len() >= 64 {
			let mut ein = Vec::with_capacity(run.pending.len() * 2 * self.hp.n_embd);
			let mut mp = Vec::with_capacity(run.pending.len());
			for e in run.pending.drain(..) {
				self.mtp_in_row(&e.emb, &e.h, &mut ein);
				mp.push(e.mpos);
			}
			self.mtp_run(&mut run.kv, &ein, &mp, false);
		}
	}

	/// Draft with the MTP head, then verify `[token, drafts…]` with the main
	/// model in one batch. `rs` is left advanced by the whole batch; pass the
	/// result to [`Model::spec_commit`] once the caller knows how many drafts
	/// it keeps. Requires `rs.mtp` (a run state made after `enable_mtp`).
	pub fn spec_step(
		&self,
		rs: &mut RunState,
		token: u32,
		forced: Option<Vec<u32>>,
		stats: &mut MtpStats,
	) -> Spec {
		let n = self.hp.n_embd;
		let base_pos = rs.pos;
		let base_mpos = rs.mpos;
		let text = PromptInput::text(&[]);

		// --- draft
		let t0 = Instant::now();
		let run = rs
			.mtp
			.as_mut()
			.expect("spec_step needs a run state with MTP");
		let suffix = forced.is_some();
		if let Some(dv) = &self.draft_vocab {
			dv.note(&self.output, &[token]);
		}
		if let Some(drafts) = forced {
			// Context drafts: no MTP pass. The head's row for `token` joins
			// the queue and is computed with the next real draft.
			run.pending.push(MtpEntry {
				emb: self.embedding(token),
				h: run.last_h.clone(),
				mpos: base_mpos,
			});
			let mtp_kv_len = run.kv.as_ref().map_or(0, |(k, _)| k.len());
			let t1 = Instant::now();
			let (logits, hs, tokens) = self.spec_verify(rs, token, &drafts, None);
			stats.note(&drafts, t1 - t0, t1.elapsed(), true);
			return Spec {
				drafts,
				logits,
				hs,
				tokens,
				base_pos,
				base_mpos,
				mtp_kv_len,
				feats: Vec::new(),
				suffix,
				snap: None,
				alts: Vec::new(),
				probs: Vec::new(),
			};
		}
		let P = run.pending.len();
		let mut ein = Vec::with_capacity((P + 1) * 2 * n);
		let mut mpos = Vec::with_capacity(P + 1);
		for e in run.pending.drain(..) {
			self.mtp_in_row(&e.emb, &e.h, &mut ein);
			mpos.push(e.mpos);
		}
		self.mtp_in_row(&self.embedding(token), &run.last_h, &mut ein);
		mpos.push(base_mpos);
		let (d0, mut p, mut h, a0) = self.mtp_run(&mut run.kv, &ein, &mpos, true).unwrap();
		let mtp_kv_len = run.kv.as_ref().map_or(0, |(k, _)| k.len());
		let mut drafts = vec![d0];
		let mut alts = vec![a0];
		let mut probs = vec![p];
		for i in 1..self.draft_n() {
			let pmin = mtp_pmin();
			if pmin > 0.0 && p < pmin {
				break;
			}
			let mut ein = Vec::with_capacity(2 * n);
			self.mtp_in_row(&self.embedding(drafts[i - 1]), &h, &mut ein);
			let (d, a);
			(d, p, h, a) = self
				.mtp_run(&mut run.kv, &ein, &[text.mpos(base_mpos, i)], true)
				.unwrap();
			drafts.push(d);
			alts.push(a);
			probs.push(p);
		}
		let t1 = Instant::now();

		let (logits, hs, tokens) = self.spec_verify(rs, token, &drafts, None);
		stats.note(&drafts, t1 - t0, t1.elapsed(), false);
		Spec {
			drafts,
			logits,
			hs,
			tokens,
			base_pos,
			base_mpos,
			mtp_kv_len,
			feats: Vec::new(),
			suffix,
			snap: None,
			alts,
			probs,
		}
	}

	/// Run `[token, drafts…]` through the main model as one batch and return
	/// every row's logits, output-normed hidden state, and the batch. The
	/// rollback scratch is `snap` when given, else the drafter's own.
	pub(crate) fn spec_verify(
		&self,
		rs: &mut RunState,
		token: u32,
		drafts: &[u32],
		snap: Option<&mut SpecSnap>,
	) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
		let n = self.hp.n_embd;
		let nv = self.hp.n_vocab;
		let mut tokens = Vec::with_capacity(drafts.len() + 1);
		tokens.push(token);
		tokens.extend_from_slice(drafts);
		let k1 = tokens.len();
		let mut xmat = vec![0.0f32; k1 * n];
		match snap {
			Some(snap) => {
				self.prefill_chunk(rs, PromptInput::text(&tokens), None, &mut xmat, Some(snap))
			}
			None => {
				let mut snap = match (rs.mtp.as_mut(), rs.dflash.as_mut()) {
					(Some(m), _) => std::mem::take(&mut m.snap),
					(None, Some(d)) => std::mem::take(&mut d.snap),
					(None, None) => unreachable!("spec_verify without rollback scratch"),
				};
				self.prefill_chunk(
					rs,
					PromptInput::text(&tokens),
					None,
					&mut xmat,
					Some(&mut snap),
				);
				match (rs.mtp.as_mut(), rs.dflash.as_mut()) {
					(Some(m), _) => m.snap = snap,
					(None, Some(d)) => d.snap = snap,
					(None, None) => unreachable!(),
				}
			}
		}
		let hs = self.rows_norm(&xmat, &self.output_norm);
		let logits = apply_linear_batch(&self.output, &hs, k1, nv, n);
		(logits, hs, tokens)
	}

	/// Context-draft step for a run with no model drafter: verify
	/// `[token, drafts…]`, with `snap` as the rollback scratch.
	pub(crate) fn suffix_step(
		&self,
		rs: &mut RunState,
		token: u32,
		drafts: Vec<u32>,
		mut snap: SpecSnap,
		stats: &mut MtpStats,
	) -> Spec {
		let (base_pos, base_mpos) = (rs.pos, rs.mpos);
		let t0 = Instant::now();
		let (logits, hs, tokens) = self.spec_verify(rs, token, &drafts, Some(&mut snap));
		stats.note(&drafts, std::time::Duration::ZERO, t0.elapsed(), true);
		Spec {
			drafts,
			logits,
			hs,
			tokens,
			base_pos,
			base_mpos,
			mtp_kv_len: 0,
			feats: Vec::new(),
			suffix: true,
			snap: Some(snap),
			alts: Vec::new(),
			probs: Vec::new(),
		}
	}

	/// Roll every SSM layer's recurrent state back to "after the first `keep`
	/// tokens of the last verify batch": start from the pre-batch `base` and
	/// re-run the scan over those tokens with the batch's own inputs. Same
	/// kernel, same inputs, same order, so the result is bit-identical to the
	/// state the live scan passed through. Only called on a rejection; when
	/// the whole batch is kept the live state already is the answer.
	pub(crate) fn ssm_replay(&self, state: &mut [Vec<f32>], snap: &mut SpecSnap, keep: usize) {
		let S = self.hp.ssm_state;
		let Hk = self.hp.ssm_n_group;
		let Hv = self.hp.ssm_dt_rank;
		let layers: Vec<usize> = (0..snap.inp.len())
			.filter(|&il| snap.inp[il].is_some())
			.collect();
		if layers.is_empty() {
			return;
		}
		let qscale = 1.0f32 / (S as f32).sqrt();
		let jb = ssm_col_block(S);
		let nblk = S / jb;
		let per = Hv * nblk;
		let bases: Vec<usize> = layers
			.iter()
			.map(|&il| snap.base[il].as_mut_ptr() as usize)
			.collect();
		let inp = &snap.inp;
		quant::par_for(layers.len() * per, |task| {
			let (li, r) = (task / per, task % per);
			let (h, j0) = (r / nblk, (r % nblk) * jb);
			let kh = h % Hk;
			let x = inp[layers[li]].as_ref().unwrap();
			let N = x.n;
			// SAFETY: task `(layer, h, j0)` is the only writer of columns
			// `[j0, j0 + jb)` of head `h` in that layer's base state.
			let st_h = unsafe {
				std::slice::from_raw_parts_mut((bases[li] as *mut f32).add(h * S * S), S * S)
			};
			let mut o = vec![0.0f32; jb];
			for t in 0..keep.min(N) {
				let kb = (kh * N + t) * S;
				let vb = (h * N + t) * S + j0;
				o.fill(0.0);
				quant::delta_net_cols(
					st_h,
					S,
					j0,
					&x.k[kb..kb + S],
					&x.q[kb..kb + S],
					&x.v[vb..vb + jb],
					x.beta[t * Hv + h],
					x.gate[t * Hv + h],
					qscale,
					&mut o,
				);
			}
		});
		if spec_check() {
			let bad = layers
				.iter()
				.filter(|&&il| snap.check[il][keep - 1] != snap.base[il])
				.count();
			eprintln!(
				"SPEC_CHECK keep={keep} layers={} mismatched={bad}",
				layers.len()
			);
		}
		// The replayed state goes live; the post-batch one becomes the next
		// batch's `base` buffer.
		for &il in &layers {
			std::mem::swap(&mut state[il], &mut snap.base[il]);
		}
	}

	/// Keep `[token, d_0 .. d_{accepted-1}]` of a [`Model::spec_step`] and
	/// forget the rest: afterwards `rs` is exactly as if those tokens had been
	/// fed one at a time.
	pub fn spec_commit(&self, rs: &mut RunState, sp: Spec, accepted: usize) {
		self.spec_commit_snap(rs, sp, accepted);
	}

	/// [`Model::spec_commit`], handing back a draft-free run's rollback
	/// scratch for reuse.
	pub(crate) fn spec_commit_snap(
		&self,
		rs: &mut RunState,
		mut sp: Spec,
		accepted: usize,
	) -> Option<SpecSnap> {
		let n = self.hp.n_embd;
		let k = sp.drafts.len();
		assert!(accepted <= k);
		let keep = accepted + 1;
		if keep < sp.tokens.len() {
			let stride = self.hp.head_dim * self.hp.n_head_kv;
			let snap = match (sp.snap.as_mut(), rs.mtp.as_mut(), rs.dflash.as_mut()) {
				(Some(s), _, _) => s,
				(None, Some(m), _) => &mut m.snap,
				(None, None, Some(d)) => &mut d.snap,
				(None, None, None) => unreachable!("spec_commit without rollback scratch"),
			};
			self.ssm_replay(&mut rs.state, snap, keep);
			for il in 0..self.hp.n_layer_main {
				if let Some((kc, vc)) = &mut rs.kv[il] {
					kc.truncate((sp.base_pos + keep) * stride);
					vc.truncate((sp.base_pos + keep) * stride);
				} else if !self.full_attn_layers[il] {
					// The conv window after token `accepted` is the one to
					// keep; the live (post-batch) one goes back as scratch.
					std::mem::swap(&mut rs.convbuf[il], &mut snap.conv[accepted][il]);
				}
			}
			rs.pos = sp.base_pos + keep;
			rs.mpos = PromptInput::text(&[]).mpos_after(sp.base_mpos, keep);
		}
		if rs.dflash.is_some() && !sp.feats.is_empty() {
			let f = sp.feats.len() / sp.tokens.len();
			self.dflash_push(rs, sp.base_pos, &sp.feats[..keep * f]);
		}
		let Some(run) = rs.mtp.as_mut() else {
			return sp.snap;
		};
		if let Some((kc, vc)) = &mut run.kv {
			kc.truncate(sp.mtp_kv_len);
			vc.truncate(sp.mtp_kv_len);
		}
		for i in 1..keep {
			run.pending.push(MtpEntry {
				emb: self.embedding(sp.tokens[i]),
				h: sp.hs[(i - 1) * n..i * n].to_vec(),
				mpos: PromptInput::text(&[]).mpos(sp.base_mpos, i),
			});
		}
		run.last_h.copy_from_slice(&sp.hs[(keep - 1) * n..keep * n]);
		sp.snap
	}
}

/// Token-at-a-time decode driver, with MTP speculation when the model and the
/// run state have it and plain `forward_step` otherwise. Callers replace
///
/// ```text
/// next = sample(&logits); …; logits = model.forward_step(rs, next, None).0;
/// ```
///
/// with `next = sample(dec.logits()); …; dec.feed(model, rs, next);` and call
/// [`Decoder::finish`] when they stop. A fed token that matches the pending
/// draft costs nothing: its logits were already computed by the verify pass.
pub struct Decoder {
	logits: Vec<f32>,
	spec: Option<Spec>,
	/// Row of `spec` currently exposed by `logits()` = drafts accepted so far.
	row: usize,
	pub stats: MtpStats,
	/// Every token in `rs` plus the ones fed since, when context drafts are on
	/// ([`Decoder::with_context`]).
	suffix: Option<SuffixIndex>,
	/// Rollback scratch for context drafts without a model drafter.
	snap: Option<SpecSnap>,
	/// Earlier responses (the server's), searched alongside `suffix`.
	global: Option<std::sync::Arc<SuffixIndex>>,
}

impl Decoder {
	/// Start from the logits of the last prompt token.
	pub fn new(logits: Vec<f32>) -> Decoder {
		Decoder {
			logits,
			spec: None,
			row: 0,
			stats: MtpStats::default(),
			suffix: None,
			snap: None,
			global: None,
		}
	}

	/// Like [`Decoder::new`], also drafting from the context: `ctx` is every
	/// token already in the run state (prompt and earlier turns). Off with
	/// `QWEN_SUFFIX=0`.
	pub fn with_context(logits: Vec<f32>, ctx: &[u32]) -> Decoder {
		let mut d = Decoder::new(logits);
		if suffix_enabled() {
			d.suffix = Some(SuffixIndex::new(ctx));
		}
		d
	}

	/// Also draft from `global`, an index of earlier responses
	/// ([`SuffixIndex::push_doc`]). Needs [`Decoder::with_context`].
	pub fn with_global(mut self, global: std::sync::Arc<SuffixIndex>) -> Decoder {
		if self.suffix.is_some() && !global.is_empty() {
			self.global = Some(global);
		}
		self
	}

	/// Let go of the index of earlier responses, so the caller can add this
	/// response to it in place (`Arc::make_mut` copies the whole index while
	/// another `Arc` to it is alive).
	pub fn release_global(&mut self) {
		self.global = None;
	}

	/// The best context draft: this conversation's, or an earlier response's
	/// when that match is longer.
	fn ctx_draft(&self, room: usize) -> (Vec<u32>, usize) {
		let Some(ix) = &self.suffix else {
			return (Vec::new(), 0);
		};
		crate::suffix::best_draft(ix, self.global.as_deref(), room)
	}

	/// The distribution of the next token.
	pub fn logits(&self) -> &[f32] {
		match &self.spec {
			Some(sp) => {
				let nv = sp.logits.len() / sp.tokens.len();
				sp.logits(self.row, nv)
			}
			None => &self.logits,
		}
	}

	/// Feed the token the caller sampled from `logits()`.
	pub fn feed(&mut self, model: &Model, rs: &mut RunState, token: u32) {
		// Decode is DRAM-bound: past `decode_threads` more cores only stall
		// on the same bandwidth (see `quant::parallel`, "Width").
		let _width = quant::parallel::limit_width(quant::parallel::decode_threads());
		if let Some(ix) = &mut self.suffix {
			ix.push(token);
		}
		if let Some(sp) = &self.spec {
			if self.row < sp.drafts.len() && sp.drafts[self.row] == token {
				// Already in `rs`: the verify pass ran it.
				self.row += 1;
				self.stats.accepted += 1;
				if sp.suffix {
					self.stats.suffix_accepted += 1;
				}
				return;
			}
			let sp = self.spec.take().unwrap();
			if tree_log() && !sp.alts.is_empty() {
				let r = self.row;
				let rej = r < sp.drafts.len();
				eprintln!(
					"TREE n={} acc={} rej={} alt_hit={} p={:?}",
					sp.drafts.len(),
					r,
					rej as u8,
					(rej && sp.alts[r] == token) as u8,
					sp.probs
				);
			}
			if let Some(s) = model.spec_commit_snap(rs, sp, self.row) {
				self.snap = Some(s);
			}
		}
		let model_drafter = (model.dflash_enabled() && rs.dflash.is_some())
			|| (model.mtp_enabled() && rs.mtp.is_some());
		// Room left in the context and in a verify batch.
		let room = (offload::SPEC_MAX_SEQ - 1).min(model.ctx.saturating_sub(rs.pos + 1));
		let ctx_draft = {
			let (d, m) = self.ctx_draft(room);
			let min = if model_drafter {
				suffix_min_model()
			} else {
				suffix_min_plain()
			};
			(m >= min && !d.is_empty()).then_some(d)
		};
		self.row = 0;
		if model.dflash_enabled() && rs.dflash.is_some() {
			self.spec = Some(model.dflash_step(rs, token, ctx_draft, &mut self.stats));
		} else if model.mtp_enabled() && rs.mtp.is_some() {
			self.spec = Some(model.spec_step(rs, token, ctx_draft, &mut self.stats));
		} else if let Some(d) = ctx_draft {
			let snap = self.snap.take().unwrap_or_default();
			self.spec = Some(model.suffix_step(rs, token, d, snap, &mut self.stats));
		} else {
			self.logits = model.forward_step(rs, token, None).0;
		}
	}

	/// Cut `rs` back to the tokens actually fed. Call before reusing `rs`
	/// (another prefill, a checkpoint rewind) or dropping the decoder.
	pub fn finish(&mut self, model: &Model, rs: &mut RunState) {
		if let Some(sp) = self.spec.take() {
			model.spec_commit(rs, sp, self.row);
			self.row = 0;
		}
	}
}
