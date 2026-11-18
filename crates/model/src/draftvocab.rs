//! Reduced draft vocabulary for the MTP head (`QWEN_MTP_VOCAB`, default
//! 32768 base rows) and the DFlash2 drafter's head (`QWEN_DFLASH_VOCAB`).
//!
//! An MTP draft is the argmax of the head's logits, and every draft is
//! verified by the main model, so the draft logits only have to rank the
//! tokens the model is likely to emit, not all 248k of them. Over the full
//! vocabulary each draft streamed the whole ~1 GB Q6_K lm_head on the CPU
//! (~21 ms, most of the draft cost). Here the head runs over
//!
//! * rows `[0, k)`: byte-level BPE ids are in merge order, so a prefix holds
//!   the frequent tokens; a contiguous slice of the matrix, no copy;
//! * the special tokens (GGUF token types CONTROL / USER_DEFINED:
//!   `<|im_end|>`, `<tool_call>`, `<think>` ...), which sit at the end of the
//!   vocabulary but end every turn and frame every tool call;
//! * every token id `>= k` seen in a prompt or a generated token: identifiers
//!   and markdown combinations of the conversation that a prefix misses (ids
//!   151k-248k are an extension block of such tokens). Their rows are copied
//!   into a compact matrix once, when first seen; the set is shared across
//!   requests and only grows (bounded by `MAX_EXTRA` rows).
//!
//! Lossless (the main model samples every token), only acceptance can drop:
//! the prefix plus the seen tokens covered 96.9-98.5 % of the tokens of this
//! repo's sources and docs read cold, file by file (a 32k prefix alone: 92-97 %).
//! The argmax is exact within the set; the PMIN cut-off's top probability is a
//! softmax over the set, i.e. slightly higher than over the full vocabulary.

use super::*;

/// Base prefix rows (`QWEN_MTP_VOCAB`, default 32768; `0` = the full head).
pub(crate) fn draft_vocab_k() -> usize {
	static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*V.get_or_init(|| {
		std::env::var("QWEN_MTP_VOCAB")
			.ok()
			.and_then(|v| v.trim().parse().ok())
			.unwrap_or(32768)
	})
}

/// Cap on the compact extra rows (~4.2 KB each for a Q6_K head at 5120).
const MAX_EXTRA: usize = 65536;

pub(crate) struct DraftVocab {
	/// Base rows `[0, k)` of the lm_head.
	k: usize,
	/// Bytes per lm_head row.
	rb: usize,
	q6: bool,
	extra: Mutex<Extra>,
}

#[derive(Default)]
struct Extra {
	/// One bit per vocabulary id: in `ids` (or below `k`).
	seen: Vec<u64>,
	/// Vocabulary id of each compact row.
	ids: Vec<u32>,
	/// The rows' bytes, `ids.len() * rb`.
	raw: Vec<u8>,
}

impl DraftVocab {
	/// `None` when the head is not K-quant or the base set is the whole
	/// vocabulary anyway.
	pub(crate) fn new(
		output: &LinearWeight,
		n_vocab: usize,
		n_embd: usize,
		k: usize,
		specials: &[u32],
	) -> Option<Self> {
		let (raw, q6) = match output {
			LinearWeight::Q4K { raw, .. } => (raw, false),
			LinearWeight::Q6K { raw, .. } => (raw, true),
			_ => return None,
		};
		if k == 0 || k >= n_vocab || !n_embd.is_multiple_of(256) {
			return None;
		}
		let rb = n_embd / 256 * if q6 { 210 } else { 144 };
		if raw.len() != n_vocab * rb {
			return None;
		}
		let dv = DraftVocab {
			k,
			rb,
			q6,
			extra: Mutex::new(Extra {
				seen: vec![0; n_vocab.div_ceil(64)],
				..Default::default()
			}),
		};
		dv.note(output, specials);
		Some(dv)
	}

	/// Add the tokens `>= k` of `tokens` not yet in the set.
	pub(crate) fn note(&self, output: &LinearWeight, tokens: &[u32]) {
		let raw = match output {
			LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => raw,
			_ => return,
		};
		let mut e = self.extra.lock().unwrap();
		for &t in tokens {
			let t = t as usize;
			if t < self.k || t * self.rb >= raw.len() || e.ids.len() >= MAX_EXTRA {
				continue;
			}
			let (w, b) = (t / 64, 1u64 << (t % 64));
			if e.seen[w] & b != 0 {
				continue;
			}
			e.seen[w] |= b;
			e.ids.push(t as u32);
			e.raw
				.extend_from_slice(&raw[t * self.rb..(t + 1) * self.rb]);
		}
	}

	/// The draft for head-normed hidden state `h`: argmax over the set,
	/// its softmax probability within the set, and the runner-up id (logged
	/// by `QWEN_TREE_LOG`).
	pub(crate) fn draft(&self, output: &LinearWeight, h: &[f32]) -> (u32, f32, u32) {
		let raw = match output {
			LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => raw,
			_ => unreachable!("DraftVocab::new rejects other heads"),
		};
		let n = h.len();
		let gemv = |rows: usize, w: &[u8], y: &mut [f32]| {
			if self.q6 {
				quant::gemv_q6k(rows, n, h, w, y)
			} else {
				quant::gemv_q4k(rows, n, h, w, y)
			}
		};
		let mut y = quant::uninit_vec(self.k);
		gemv(self.k, &raw[..self.k * self.rb], &mut y);
		let e = self.extra.lock().unwrap();
		let mut ye = quant::uninit_vec(e.ids.len());
		if !e.ids.is_empty() {
			gemv(e.ids.len(), &e.raw, &mut ye);
		}
		let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
		let (mut si, mut sv) = (0u32, f32::NEG_INFINITY);
		let ids = (0..self.k as u32).chain(e.ids.iter().copied());
		for (id, &v) in ids.zip(y.iter().chain(&ye)) {
			if v > bv {
				(si, sv) = (bi, bv);
				(bi, bv) = (id, v);
			} else if v > sv {
				(si, sv) = (id, v);
			}
		}
		let s: f32 = y.iter().chain(&ye).map(|&x| (x - bv).exp()).sum();
		(bi, 1.0 / s, si)
	}

	/// Logits of the `seq` rows of `h` over the set, `[seq * m]` row-major with
	/// `m = k + ids.len()`: column `i < k` is vocabulary id `i`, column
	/// `k + j` is `ids[j]` (the returned copy of the extra ids). One q8_K
	/// quantization of `h` serves both matrices.
	pub(crate) fn logits_batch(
		&self,
		output: &LinearWeight,
		h: &[f32],
		seq: usize,
	) -> (Vec<f32>, Vec<u32>) {
		let raw = match output {
			LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => raw,
			_ => unreachable!("DraftVocab::new rejects other heads"),
		};
		let n = h.len() / seq;
		let q8 = std::sync::Arc::new(quant::par_quantize_q8k(h, seq, n));
		let e = self.extra.lock().unwrap();
		let m = self.k + e.ids.len();
		let mut y = quant::uninit_vec(seq * m);
		let gemm = |rows: usize, w: &[u8], y: &mut [f32], col0: usize| {
			if self.q6 {
				quant::gemm_q6k_strided_q8(rows, n, seq, w, &q8, y, m, col0)
			} else {
				quant::gemm_q4k_strided_q8(rows, n, seq, w, &q8, y, m, col0)
			}
		};
		gemm(self.k, &raw[..self.k * self.rb], &mut y, 0);
		if !e.ids.is_empty() {
			gemm(e.ids.len(), &e.raw, &mut y, self.k);
		}
		(y, e.ids.clone())
	}

	/// Vocabulary id of column `i` of [`DraftVocab::logits_batch`].
	pub(crate) fn id(&self, ids: &[u32], i: usize) -> u32 {
		if i < self.k {
			i as u32
		} else {
			ids[i - self.k]
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The draft is the full head's argmax restricted to the set: the prefix,
	/// the specials and the noted tokens, nothing else.
	#[test]
	fn draft_is_the_argmax_over_the_set() {
		let (nv, n, k) = (96usize, 256usize, 32usize);
		let mut s = 0x9e37_79b9_7f4a_7c15u64;
		let mut rnd = || {
			s = s
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((s >> 33) as f32) / (1u32 << 31) as f32 - 0.5
		};
		let w: Vec<f32> = (0..nv * n).map(|_| rnd()).collect();
		let h: Vec<f32> = (0..n).map(|_| rnd()).collect();
		for q6 in [false, true] {
			let raw = if q6 {
				quant::quantize_q6_k(&w, nv, n)
			} else {
				quant::quantize_q4_k(&w, nv, n)
			};
			let head = if q6 {
				LinearWeight::Q6K {
					raw,
					out_dim: nv,
					in_dim: n,
				}
			} else {
				LinearWeight::Q4K {
					raw,
					out_dim: nv,
					in_dim: n,
				}
			};
			let full = apply_linear(&head, &h, nv, n);
			let best = |set: &dyn Fn(usize) -> bool| {
				(0..nv)
					.filter(|&i| set(i))
					.fold((0usize, f32::NEG_INFINITY), |b, i| {
						if full[i] > b.1 {
							(i, full[i])
						} else {
							b
						}
					})
			};
			let specials = [90u32, 95];
			let dv = DraftVocab::new(&head, nv, n, k, &specials).unwrap();
			let mut set: Vec<bool> = (0..nv)
				.map(|i| i < k || specials.contains(&(i as u32)))
				.collect();
			for noted in [vec![], vec![40u32, 41, 40, 3], (0..nv as u32).collect()] {
				dv.note(&head, &noted);
				for &t in &noted {
					set[t as usize] = true;
				}
				let (id, p, _) = dv.draft(&head, &h);
				let (bi, bv) = best(&|i| set[i]);
				assert_eq!(id as usize, bi, "q6 {q6}, noted {noted:?}");
				let z: f32 = (0..nv)
					.filter(|&i| set[i])
					.map(|i| (full[i] - bv).exp())
					.sum();
				assert!((p - 1.0 / z).abs() < 1e-5);
			}
			assert_eq!(dv.extra.lock().unwrap().ids.len(), nv - k);
		}
	}

	/// `logits_batch` column `i` is the full head's logit of token `id(i)`,
	/// for every row of the batch.
	#[test]
	fn logits_batch_columns_are_the_sets_logits() {
		let (nv, n, k, seq) = (96usize, 512usize, 40usize, 3usize);
		let mut s = 0x2545_f491_4f6c_dd1du64;
		let mut rnd = || {
			s = s
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((s >> 33) as f32) / (1u32 << 31) as f32 - 0.5
		};
		let w: Vec<f32> = (0..nv * n).map(|_| rnd()).collect();
		let h: Vec<f32> = (0..seq * n).map(|_| rnd()).collect();
		for q6 in [false, true] {
			let raw = if q6 {
				quant::quantize_q6_k(&w, nv, n)
			} else {
				quant::quantize_q4_k(&w, nv, n)
			};
			let head = if q6 {
				LinearWeight::Q6K {
					raw,
					out_dim: nv,
					in_dim: n,
				}
			} else {
				LinearWeight::Q4K {
					raw,
					out_dim: nv,
					in_dim: n,
				}
			};
			let full = apply_linear_batch(&head, &h, seq, nv, n);
			let dv = DraftVocab::new(&head, nv, n, k, &[95]).unwrap();
			dv.note(&head, &[70, 41, 70, 3]);
			let (y, ids) = dv.logits_batch(&head, &h, seq);
			let m = k + 3;
			assert_eq!((y.len(), ids.as_slice()), (seq * m, &[95u32, 70, 41][..]));
			for t in 0..seq {
				for i in 0..m {
					let want = full[t * nv + dv.id(&ids, i) as usize];
					assert!(
						(y[t * m + i] - want).abs() < 1e-4,
						"q6 {q6} t {t} col {i}: {} vs {want}",
						y[t * m + i]
					);
				}
			}
		}
	}
}
