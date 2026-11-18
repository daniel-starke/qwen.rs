//! Model-free speculative drafts from the context itself (prompt-lookup /
//! SuffixDecoding-style, arXiv 2411.04975).
//!
//! Agentic and code traffic re-emits text that is already in the context: a
//! file being edited, a tool call's boilerplate, a test that mirrors the
//! function above it. When the last few generated tokens occurred earlier, the
//! tokens that followed them there are a draft that costs **no** forward pass
//! at all, only the extra verify rows. The verify/accept/rollback path is the
//! one MTP and DFlash use ([`Model::spec_commit`]), so every emitted token is
//! still sampled from the main model's own logits.
//!
//! The index maps every `GRAM`-token window to the positions that followed it.
//! A lookup takes the context's last window, extends each earlier occurrence
//! backwards to find the longest match (most recent wins a tie) and proposes
//! what followed it.

use std::collections::HashMap;

/// Window the index is keyed on, and the shortest match that drafts.
pub const GRAM: usize = 3;
/// Longest match measured backwards; longer ones draft the same.
const MAX_MATCH: usize = 64;
/// Earlier occurrences examined per lookup, newest first.
const MAX_CANDS: usize = 256;

fn key(w: &[u32]) -> u64 {
	w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
		(h ^ t as u64).wrapping_mul(0x0100_0000_01b3)
	})
}

/// `QWEN_SUFFIX` (default on, `0` disables): draft from the context when no
/// model drafter applies, and in place of one on a long match.
pub fn suffix_enabled() -> bool {
	static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		!matches!(
			std::env::var("QWEN_SUFFIX").as_deref(),
			Ok("0" | "off" | "no" | "false")
		)
	})
}

/// `QWEN_SUFFIX_MIN` (default 12): match length from which a context draft
/// replaces the MTP / DFlash draft. Below it the model drafter runs.
pub fn suffix_min_model() -> usize {
	quant::knob::parse::<usize>("QWEN_SUFFIX_MIN").unwrap_or(12)
}

/// `QWEN_SUFFIX_ENGINE_MIN` (default 6): the same for an engine that drafts
/// on its own (the NPU engine's MTP, `server::Engine::verify`). Its verify
/// step costs the same for any number of drafts and its MTP accepts less
/// than the GGUF path's: 4 / 6 / 12 = 6.39 / 6.36 / 6.29 tok/s on an
/// edit-style answer, 5.83 without context drafts.
pub fn suffix_min_engine() -> usize {
	quant::knob::parse::<usize>("QWEN_SUFFIX_ENGINE_MIN")
		.unwrap_or(6)
		.max(GRAM)
}

/// `QWEN_SUFFIX_PLAIN_MIN` (default `GRAM`): match length from which a context
/// draft is verified when there is no model drafter.
pub fn suffix_min_plain() -> usize {
	quant::knob::parse::<usize>("QWEN_SUFFIX_PLAIN_MIN")
		.unwrap_or(GRAM)
		.max(GRAM)
}

/// Separates documents in a [`SuffixIndex`] built with [`SuffixIndex::push_doc`]:
/// no window spans it and no draft runs past it.
const SEP: u32 = u32::MAX;

/// `QWEN_SUFFIX_GLOBAL_TOKENS` (default 1M, 0 disables): tokens of earlier
/// responses the server keeps for cross-request drafts.
pub fn suffix_global_cap() -> usize {
	static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		std::env::var("QWEN_SUFFIX_GLOBAL_TOKENS")
			.ok()
			.and_then(|v| v.parse().ok())
			.unwrap_or(1 << 20)
	})
}

/// Token history plus the window index over it.
#[derive(Default, Clone)]
pub struct SuffixIndex {
	toks: Vec<u32>,
	map: HashMap<u64, Vec<u32>>,
}

impl SuffixIndex {
	pub fn new(ctx: &[u32]) -> SuffixIndex {
		let mut s = SuffixIndex {
			toks: Vec::with_capacity(ctx.len() + 1024),
			map: HashMap::new(),
		};
		for &t in ctx {
			s.push(t);
		}
		s
	}

	pub fn len(&self) -> usize {
		self.toks.len()
	}

	pub fn is_empty(&self) -> bool {
		self.toks.is_empty()
	}

	/// Append one token. The window *ending before* it is indexed now, since
	/// only then does that window have a continuation to propose.
	pub fn push(&mut self, t: u32) {
		let i = self.toks.len();
		if i >= GRAM && !self.toks[i - GRAM..i].contains(&SEP) {
			self.map
				.entry(key(&self.toks[i - GRAM..i]))
				.or_default()
				.push(i as u32);
		}
		self.toks.push(t);
	}

	/// Append a whole document (an earlier response), separated from the
	/// previous one. Past `cap` tokens the oldest half is dropped.
	pub fn push_doc(&mut self, doc: &[u32], cap: usize) {
		if doc.len() < GRAM || cap == 0 {
			return;
		}
		if self.toks.len() + doc.len() + 1 > cap {
			let keep = self
				.toks
				.split_off(self.toks.len() - self.toks.len().min(cap / 2));
			let start = keep
				.iter()
				.position(|&t| t == SEP)
				.map_or(keep.len(), |i| i + 1);
			*self = SuffixIndex::default();
			for &t in &keep[start..] {
				self.push(t);
			}
		}
		if !self.toks.is_empty() {
			self.push(SEP);
		}
		for &t in doc {
			self.push(t);
		}
	}

	/// Up to `max` tokens predicted to follow the context, and the length of
	/// the match they come from (0: nothing to propose).
	pub fn draft(&self, max: usize) -> (Vec<u32>, usize) {
		self.draft_after(&self.toks, max)
	}

	/// Like [`SuffixIndex::draft`], for a context `tail` that is not this
	/// index's own history (the current conversation against earlier ones).
	pub fn draft_after(&self, tail: &[u32], max: usize) -> (Vec<u32>, usize) {
		let n = tail.len();
		if n < GRAM || max == 0 {
			return (Vec::new(), 0);
		}
		let Some(cands) = self.map.get(&key(&tail[n - GRAM..])) else {
			return (Vec::new(), 0);
		};
		let (mut best, mut best_m) = (0usize, 0usize);
		for &p in cands.iter().rev().take(MAX_CANDS) {
			let p = p as usize;
			let mut m = 0;
			while m < MAX_MATCH && m < p && m < n && self.toks[p - 1 - m] == tail[n - 1 - m] {
				m += 1;
			}
			// `m < GRAM` is a hash collision.
			if m >= GRAM && m > best_m {
				(best, best_m) = (p, m);
				if m == MAX_MATCH {
					break;
				}
			}
		}
		if best_m == 0 {
			return (Vec::new(), 0);
		}
		let end = self.toks.len().min(best + max);
		let end = self.toks[best..end]
			.iter()
			.position(|&t| t == SEP)
			.map_or(end, |i| best + i);
		(self.toks[best..end].to_vec(), best_m)
	}

	/// The last `n` tokens, the query for [`SuffixIndex::draft_after`].
	pub fn tail(&self, n: usize) -> &[u32] {
		&self.toks[self.toks.len().saturating_sub(n)..]
	}
}

/// The best context draft of up to `room` tokens: from `ix` (this
/// conversation), or from `global` (earlier responses) when its match is
/// longer. Returns the draft and its match length.
pub fn best_draft(
	ix: &SuffixIndex,
	global: Option<&SuffixIndex>,
	room: usize,
) -> (Vec<u32>, usize) {
	let (d, m, _) = best_draft_src(ix, global, room);
	(d, m)
}

/// [`best_draft`], and whether the draft came from `global`.
pub fn best_draft_src(
	ix: &SuffixIndex,
	global: Option<&SuffixIndex>,
	room: usize,
) -> (Vec<u32>, usize, bool) {
	let local = ix.draft(room);
	match global {
		Some(g) if !g.is_empty() => {
			let other = g.draft_after(ix.tail(64), room);
			if other.1 > local.1 {
				(other.0, other.1, true)
			} else {
				(local.0, local.1, false)
			}
		}
		_ => (local.0, local.1, false),
	}
}

/// Draft positions [`CtxPolicy`] keeps statistics for.
const EV_MAXD: usize = 8;
/// Match-length buckets of [`CtxPolicy`]: the lower bound of each.
const EV_BUCKETS: [usize; 11] = [3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64];
/// Prior of each position's acceptance, and its weight in observations.
const EV_PRIOR: (f32, f32) = (0.7, 3.0);
/// Observations of a bucket's first position past which its counts are
/// halved, so the statistics follow the conversation.
const EV_WINDOW: f32 = 256.0;

/// `QWEN_SUFFIX_EV` (default on with an engine that reports step costs,
/// `0` = the fixed rule of [`suffix_min_engine`]): a context draft only when
/// its expected tokens per unit of step cost beat the model drafter's
/// recent rate, cut to the length that maximizes that.
pub fn suffix_ev() -> bool {
	quant::knob::var("QWEN_SUFFIX_EV").is_none_or(|v| v != "0")
}

/// Expected-value choice of context drafts for an engine whose verify step
/// has a cost per number of positions (`costs[k - 1]`: a step of `k`).
///
/// Every context draft that could be proposed is scored against the tokens
/// that follow it, whether it was verified or not (sampling does not depend
/// on the drafts), so the per-position acceptance of each match-length
/// bucket is known without exploring. Model-drafter steps report their
/// accepted count ([`CtxPolicy::model_step`]); a context draft is used when
/// its best expected tokens / cost exceeds theirs.
pub struct CtxPolicy {
	/// per bucket and draft position: (reached, accepted)
	st: Vec<[[f32; 2]; EV_MAXD]>,
	/// candidates not yet resolved: (index of their first token, draft, bucket)
	pend: Vec<(usize, Vec<u32>, usize)>,
	/// the model drafter's steps: (tokens, cost), decayed
	model: [f32; 2],
}

impl Default for CtxPolicy {
	fn default() -> Self {
		CtxPolicy {
			st: vec![[[0.0; 2]; EV_MAXD]; EV_BUCKETS.len()],
			pend: Vec::new(),
			model: [0.0; 2],
		}
	}
}

/// The policy shared by every request of a process.
pub fn ctx_policy() -> &'static std::sync::Mutex<CtxPolicy> {
	static P: std::sync::OnceLock<std::sync::Mutex<CtxPolicy>> = std::sync::OnceLock::new();
	P.get_or_init(Default::default)
}

impl CtxPolicy {
	fn bucket(m: usize) -> usize {
		EV_BUCKETS.iter().rposition(|&b| m >= b).unwrap_or(0)
	}

	/// Forget unresolved candidates (a new request: its tokens are another
	/// stream).
	pub fn begin(&mut self) {
		self.pend.clear();
	}

	/// Score the candidates that `toks` (the whole token stream so far)
	/// now decides.
	pub fn observe(&mut self, toks: &[u32]) {
		let n = toks.len();
		let mut i = 0;
		while i < self.pend.len() {
			let (s, ref d, b) = self.pend[i];
			let known = n.saturating_sub(s).min(d.len());
			let hit = d[..known]
				.iter()
				.zip(&toks[s.min(n)..s.min(n) + known])
				.take_while(|(a, b)| a == b)
				.count();
			if hit < known || known == d.len() {
				let st = &mut self.st[b];
				for (j, e) in st.iter_mut().enumerate().take((hit + 1).min(d.len())) {
					e[0] += 1.0;
					e[1] += (j < hit) as u32 as f32;
				}
				if st[0][0] > EV_WINDOW {
					for e in st.iter_mut() {
						e[0] *= 0.5;
						e[1] *= 0.5;
					}
				}
				self.pend.swap_remove(i);
			} else {
				i += 1;
			}
		}
	}

	/// A model-drafter step of `len` drafts, `acc` of them accepted.
	pub fn model_step(&mut self, len: usize, acc: usize, costs: &[f32]) {
		let c = costs[len.min(costs.len() - 1)];
		self.model[0] = self.model[0] * 0.98 + (1 + acc) as f32;
		self.model[1] = self.model[1] * 0.98 + c;
	}

	/// Of a context draft `d` with match length `m`, proposed to follow
	/// the stream at index `at`, the prefix worth verifying (`None`: let the
	/// model drafter run). The candidate is kept for [`CtxPolicy::observe`].
	pub fn choose(&mut self, at: usize, d: &[u32], m: usize, costs: &[f32]) -> Option<usize> {
		if d.is_empty() || m < GRAM || costs.is_empty() {
			return None;
		}
		let b = Self::bucket(m);
		self.pend.push((at, d[..d.len().min(EV_MAXD)].to_vec(), b));
		let c = |k: usize| costs[k.min(costs.len() - 1)];
		// with no model step seen, a plain step's rate (1 token, `costs[0]`)
		let (w, base) = (2.0, 1.0 / c(0));
		let model = (self.model[0] + w * base * c(0)) / (self.model[1] + w * c(0));
		let (mut e, mut pall) = (1f32, 1f32);
		let mut best = (0usize, model);
		for (j, s) in self.st[b].iter().enumerate().take(d.len()) {
			let a = (s[1] + EV_PRIOR.0 * EV_PRIOR.1) / (s[0] + EV_PRIOR.1);
			pall *= a;
			e += pall;
			let r = e / c(j + 1);
			if r > best.1 {
				best = (j + 1, r);
			}
		}
		(best.0 > 0).then_some(best.0)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn drafts_what_followed_the_longest_earlier_match() {
		// 0 1 2 3 [4 5] ... 9 1 2 3 [7 8] ... 0 1 2 3 -> the 4-token match
		// wins over the newer 3-token one.
		let ctx = [0, 1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 0, 1, 2, 3];
		let s = SuffixIndex::new(&ctx);
		assert_eq!(s.draft(2), (vec![4, 5], 4));
		assert_eq!(s.draft(1), (vec![4], 4));
	}

	#[test]
	fn ties_go_to_the_most_recent_and_drafts_stop_at_the_context_end() {
		let mut s = SuffixIndex::new(&[5, 6, 7, 1, 0, 5, 6, 7, 2]);
		s.push(5);
		s.push(6);
		s.push(7);
		assert_eq!(s.draft(8), (vec![2, 5, 6, 7], 3));
	}

	#[test]
	fn documents_draft_for_another_context_and_never_across_a_separator() {
		let mut g = SuffixIndex::default();
		g.push_doc(&[1, 2, 3, 4, 5], 100);
		g.push_doc(&[7, 8, 9, 1, 2, 3], 100);
		// 1 2 3 also ends the last doc, but there it has no continuation and
		// so is not indexed: the first doc's 4 5 is the draft.
		assert_eq!(g.draft_after(&[0, 1, 2, 3], 8), (vec![4, 5], 3));
		// A window straddling the separator is never indexed.
		assert_eq!(g.draft_after(&[5, 7, 8], 8), (vec![], 0));
		// 2 3 4 is followed by 5 and then the separator: the draft stops there.
		assert_eq!(g.draft_after(&[2, 3, 4], 8), (vec![5], 3));
	}

	#[test]
	fn push_doc_drops_the_oldest_half_past_the_cap() {
		let mut g = SuffixIndex::default();
		g.push_doc(&[1, 2, 3, 4], 12);
		g.push_doc(&[5, 6, 7, 8], 12);
		g.push_doc(&[9, 10, 11, 12], 12);
		assert_eq!(g.draft_after(&[1, 2, 3], 4), (vec![], 0));
		assert_eq!(g.draft_after(&[9, 10, 11], 4), (vec![12], 3));
		assert!(g.len() <= 12);
	}

	#[test]
	fn no_draft_without_a_match() {
		let s = SuffixIndex::new(&[1, 2, 3, 4, 5]);
		assert_eq!(s.draft(4), (vec![], 0));
		let s = SuffixIndex::new(&[1, 2]);
		assert_eq!(s.draft(4), (vec![], 0));
	}

	#[test]
	fn ctx_policy_learns_each_bucket_from_every_candidate() {
		let costs = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0];
		let mut p = CtxPolicy::default();
		// drafts of match length 6 that are always right: the stream repeats them
		let toks: Vec<u32> = (0..400).map(|i| (i % 7) as u32).collect();
		for at in (10..300).step_by(8) {
			p.choose(at, &toks[at..at + 7], 6, &costs);
			p.observe(&toks[..at + 8]);
		}
		assert!(p.pend.is_empty());
		assert_eq!(p.choose(320, &toks[320..327], 6, &costs), Some(7));
		p.begin();
		// match length 3: always wrong at the first position
		for at in (10..300).step_by(8) {
			p.choose(at, &[99; 7], 3, &costs);
			p.observe(&toks[..at + 8]);
		}
		assert_eq!(p.choose(320, &[99; 7], 3, &costs), None);
		p.begin();
		// a model drafter that does well beats a mediocre bucket
		for _ in 0..50 {
			p.model_step(3, 3, &costs);
		}
		assert!(p.model[0] / p.model[1] > 0.03);
	}
}
