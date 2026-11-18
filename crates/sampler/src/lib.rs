//! Sampling for qwen.rs: temperature, top-k, top-p (nucleus), min-p and the
//! repetition / presence / frequency penalties. Standalone (no external RNG
//! dependency: uses a small xoshiro256** generator).
//!
//! One call costs ~0.2 ms on the 248k-entry vocabulary: the candidates are
//! found in one pass with a running threshold and only they are sorted. (The
//! first version sorted the whole vocabulary twice, 11 ms a token: at
//! temperature > 0 that was ~40 % of a Flash-Next MTP step.)

/// The sampling switches. `temperature` 0 is greedy; `top_k` 0, `top_p` 1 and
/// `min_p` 0 disable their cut; the penalties apply to the tokens the caller
/// passes as `prev` (the generated ones: see [`sample`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
	pub temperature: f32,
	pub top_k: usize,
	pub top_p: f32,
	/// Drop tokens whose probability is below `min_p` times the top one's.
	pub min_p: f32,
	/// llama.cpp semantics, once per distinct token in `prev`: a positive
	/// logit is divided by it, a negative one multiplied.
	pub repeat_penalty: f32,
	/// OpenAI / vLLM semantics: subtracted once from the logit of every
	/// distinct token in `prev`.
	pub presence_penalty: f32,
	/// Subtracted from a token's logit once per occurrence in `prev`.
	pub frequency_penalty: f32,
}

impl Default for SampleParams {
	/// The model card's thinking-mode set: see [`SampleParams::thinking`].
	fn default() -> Self {
		Self::thinking()
	}
}

impl SampleParams {
	/// Qwen3.8's recommended thinking-mode sampling (the 27B's and
	/// Flash-Next's cards): temperature 1.0, top-p 0.95, top-k 20, no
	/// penalties; also what the vendor's Claude Code harness evaluations
	/// ran at.
	pub const fn thinking() -> Self {
		Self {
			temperature: 1.0,
			top_k: 20,
			top_p: 0.95,
			min_p: 0.0,
			repeat_penalty: 1.0,
			presence_penalty: 0.0,
			frequency_penalty: 0.0,
		}
	}

	/// Qwen3.8's recommended non-thinking (instruct) sampling: temperature
	/// 0.7, top-p 0.8, top-k 20, presence penalty 1.5 (against endless
	/// repetition). Checked for agentic coding on both models:
	/// a 132-line verbatim rewrite and exact Edit-call `old_string`s came
	/// out identical at 0 and 1.5, 4 of 4 seeds each.
	pub const fn instruct() -> Self {
		Self {
			temperature: 0.7,
			top_k: 20,
			top_p: 0.8,
			presence_penalty: 1.5,
			..Self::thinking()
		}
	}

	/// Greedy decoding, no penalties.
	pub const fn greedy() -> Self {
		Self {
			temperature: 0.0,
			top_k: 0,
			top_p: 1.0,
			..Self::thinking()
		}
	}

	fn penalized(&self) -> bool {
		self.repeat_penalty != 1.0 || self.presence_penalty != 0.0 || self.frequency_penalty != 0.0
	}
}

/// A minimal, fast PRNG (xoshiro256** seeded via SplitMix64). Sufficient for
/// sampling; not cryptographically secure.
pub struct Rng {
	s: [u64; 4],
}

impl Rng {
	pub fn new(seed: u64) -> Rng {
		let mut sm = seed;
		let mut s = [0u64; 4];
		for i in 0..4 {
			sm = sm.wrapping_add(0x9e3779b97f4a7c15);
			let mut z = sm;
			z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
			z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
			s[i] = z ^ (z >> 31);
		}
		Rng { s }
	}

	/// Next `u64`.
	pub fn next_u64(&mut self) -> u64 {
		let result = {
			let x = self.s[1].wrapping_mul(5);
			let rot = (x >> 59) as u32;
			x.wrapping_shl(rot)
				.wrapping_add(self.s[0])
				.rotate_left(7)
				.wrapping_mul(9)
				.rotate_left(20)
				.wrapping_add(self.s[3])
				.rotate_left(20)
				.wrapping_mul(5)
				.wrapping_add(self.s[1])
				.rotate_left(7)
				.wrapping_mul(9)
		};
		let t = self.s[1] << 17;
		self.s[2] ^= self.s[0];
		self.s[3] ^= self.s[1];
		self.s[1] ^= self.s[2];
		self.s[0] ^= self.s[3];
		self.s[2] = self.s[2].rotate_left(45);
		self.s[3] = t;
		result
	}

	/// Uniform float in [0, 1).
	pub fn next_f32(&mut self) -> f32 {
		// 24-bit mantissa
		let bits = (self.next_u64() >> 40) as u32;
		f32::from_bits(0x3f80_0000 | bits) - 1.0
	}
}

/// The penalties of `p` applied to a copy of `logits`, each once per distinct
/// token of `prev` (the frequency penalty once per occurrence).
fn apply_penalties(logits: &[f32], prev: &[u32], p: &SampleParams) -> Vec<f32> {
	let mut work = logits.to_vec();
	let mut count = vec![0u32; work.len()];
	for &id in prev {
		if let Some(c) = count.get_mut(id as usize) {
			*c += 1;
		}
	}
	for &id in prev {
		let i = id as usize;
		let Some(c) = count.get_mut(i) else { continue };
		if *c == 0 {
			continue;
		}
		let v = &mut work[i];
		if p.repeat_penalty != 1.0 {
			*v = if *v > 0.0 {
				*v / p.repeat_penalty
			} else {
				*v * p.repeat_penalty
			};
		}
		*v -= p.presence_penalty + p.frequency_penalty * *c as f32;
		*c = 0;
	}
	work
}

/// The `k` largest logits (all within `spread` of the maximum when `k` is
/// 0), as (logit, id), largest first, equal logits by id; NaN counts as
/// minus infinity. One pass with a running threshold: a candidate buffer is
/// cut back to the `k` best whenever it fills, so only a few hundred entries
/// are ever sorted.
fn candidates(l: &[f32], k: usize, spread: f32) -> Vec<(f32, u32)> {
	let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
	let order = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
	let mut out: Vec<(f32, u32)> = Vec::new();
	if k == 0 || k >= l.len() {
		let m = l.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(key(v)));
		let floor = m - spread;
		out.extend(
			l.iter()
				.enumerate()
				.filter(|(_, &v)| key(v) >= floor)
				.map(|(i, &v)| (key(v), i as u32)),
		);
		out.sort_unstable_by(order);
		if k > 0 {
			out.truncate(k);
		}
		return out;
	}
	let cap = (4 * k).max(64);
	out.reserve(cap);
	// a value below `thr` cannot be among the k best; ties at `thr` may
	// (a lower id wins them)
	let mut thr = f32::NEG_INFINITY;
	for (i, &v) in l.iter().enumerate() {
		let v = key(v);
		if v >= thr {
			out.push((v, i as u32));
			if out.len() == cap {
				out.select_nth_unstable_by(k - 1, order);
				out.truncate(k);
				thr = out[k - 1].0;
			}
		}
	}
	out.sort_unstable_by(order);
	out.truncate(k);
	out
}

/// Sample a token id from `logits`. The penalties of `params` apply to the
/// tokens of `prev`: pass the tokens generated so far (vLLM's convention),
/// not the prompt; a prompt full of code would otherwise penalize the very
/// identifiers an edit has to repeat.
///
/// Order as in Hugging Face / vLLM: penalties, temperature, top-k, top-p,
/// min-p, then one uniform draw over what is left.
pub fn sample(logits: &[f32], prev: &[u32], params: &SampleParams, rng: &mut Rng) -> u32 {
	let pen;
	let l: &[f32] = if params.penalized() && !prev.is_empty() {
		pen = apply_penalties(logits, prev, params);
		&pen
	} else {
		logits
	};
	let (cand, w) = match kept(l, params) {
		Kept::Argmax(id) => return id,
		Kept::Set(c, w) => (c, w),
	};
	let sum: f32 = w.iter().sum();
	let r = rng.next_f32() * sum;
	let mut acc = 0.0f32;
	for (i, &x) in w.iter().enumerate() {
		acc += x;
		if r < acc {
			return cand[i].1;
		}
	}
	cand[w.len() - 1].1
}

/// What the truncations leave of a distribution: the argmax alone
/// (temperature 0, or nothing finite), or the candidates (logit, id),
/// largest first, with their unnormalized weights (the top token weighs 1),
/// one weight per kept candidate.
enum Kept {
	Argmax(u32),
	Set(Vec<(f32, u32)>, Vec<f32>),
}

/// Temperature, top-k, top-p and min-p of `params` over the (penalized)
/// logits `l`.
fn kept(l: &[f32], params: &SampleParams) -> Kept {
	if params.temperature <= 0.0 {
		return Kept::Argmax(argmax(l));
	}
	let t = params.temperature;
	// Without a top-k, tokens more than 40 T below the top are dropped:
	// e^-40 each, never reached by a draw.
	let cand = candidates(l, params.top_k, 40.0 * t);
	if cand.is_empty() || cand[0].0 == f32::NEG_INFINITY {
		return Kept::Argmax(argmax(l));
	}
	let m = cand[0].0;
	// unnormalized: the top token weighs 1
	let mut w: Vec<f32> = cand.iter().map(|&(v, _)| ((v - m) / t).exp()).collect();
	let total: f32 = w.iter().sum();
	// top-p: the shortest prefix whose mass reaches `top_p`
	let mut keep = w.len();
	if params.top_p < 1.0 {
		let mut cum = 0.0f32;
		for (i, &x) in w.iter().enumerate() {
			cum += x / total;
			if cum >= params.top_p {
				keep = i + 1;
				break;
			}
		}
	}
	// min-p: relative to the top token
	if params.min_p > 0.0 {
		keep = keep.min(w.iter().take_while(|&&x| x >= params.min_p).count().max(1));
	}
	w.truncate(keep);
	Kept::Set(cand, w)
}

/// Shared randomness between the sampler and a drafter (coupled
/// sampling): a Gumbel variable per (position, token id), fixed by `seed`.
/// [`sample_coupled`] draws the token at a position as the argmax of
/// `ln w + g` over the kept candidates (Gumbel-max: an exact draw from the
/// same distribution [`sample`] draws from), and a drafter that picks the
/// argmax of its own `ln q + g` with the same `g` proposes the very token
/// the sampler will draw whenever its distribution is close to the model's,
/// not only when that token is the most likely one. The noise depends on
/// the seed and the position only, never on a draft, so drafts cannot
/// change what is sampled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coupling {
	pub seed: u64,
	pub params: SampleParams,
}

impl Coupling {
	/// The Gumbel variable of token `id` at absolute position `pos`.
	pub fn gumbel(&self, pos: u64, id: u32) -> f32 {
		let mut z = self.seed
			^ pos.wrapping_mul(0x9e37_79b9_7f4a_7c15)
			^ (id as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		z ^= z >> 31;
		// uniform in (0, 1): 53 bits, never 0 or 1
		let u = ((z >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
		(-(-u.ln()).ln()) as f32
	}

	/// The token the drafter proposes at `pos` from its logits `q`:
	/// the coupled draw from `q` under this coupling's truncations (no
	/// penalties: the drafter does not see the reply), and its weight's share
	/// of the kept mass.
	pub fn pick(&self, q: &[f32], pos: u64) -> (u32, f32) {
		match kept(q, &self.params) {
			Kept::Argmax(id) => (id, 1.0),
			Kept::Set(cand, w) => {
				let i = gumbel_max(&cand, &w, |id| self.gumbel(pos, id));
				(cand[i].1, w[i] / w.iter().sum::<f32>())
			}
		}
	}
}

/// The index of the largest `ln w + g` (the first of equals).
fn gumbel_max(cand: &[(f32, u32)], w: &[f32], g: impl Fn(u32) -> f32) -> usize {
	let mut best = (0usize, f32::NEG_INFINITY);
	for (i, &x) in w.iter().enumerate() {
		let v = x.ln() + g(cand[i].1);
		if v > best.1 {
			best = (i, v);
		}
	}
	best.0
}

/// [`sample`] by the coupling's noise at absolute position `pos` instead
/// of an RNG draw: the same distribution (penalties, temperature, top-k,
/// top-p, min-p of `params`), sampled by Gumbel-max. Temperature 0 is the
/// argmax as in [`sample`].
pub fn sample_coupled(
	logits: &[f32],
	prev: &[u32],
	params: &SampleParams,
	c: &Coupling,
	pos: u64,
) -> u32 {
	let pen;
	let l: &[f32] = if params.penalized() && !prev.is_empty() {
		pen = apply_penalties(logits, prev, params);
		&pen
	} else {
		logits
	};
	match kept(l, params) {
		Kept::Argmax(id) => id,
		Kept::Set(cand, w) => cand[gumbel_max(&cand, &w, |id| c.gumbel(pos, id))].1,
	}
}

fn argmax(v: &[f32]) -> u32 {
	let mut best = 0usize;
	let mut bestv = f32::NEG_INFINITY;
	for (i, &x) in v.iter().enumerate() {
		if x > bestv {
			bestv = x;
			best = i;
		}
	}
	best as u32
}

#[cfg(test)]
mod tests {
	use super::*;

	fn params(temperature: f32, top_k: usize, top_p: f32) -> SampleParams {
		SampleParams {
			temperature,
			top_k,
			top_p,
			..SampleParams::greedy()
		}
	}

	#[test]
	fn greedy_argmax() {
		let logits = [0.1, 0.9, 0.3, -2.0];
		let mut rng = Rng::new(1);
		assert_eq!(sample(&logits, &[], &params(0.0, 0, 1.0), &mut rng), 1);
	}

	#[test]
	fn repeat_penalty_once_per_token() {
		// token 0 was generated twice; the penalty applies once
		let p = SampleParams {
			repeat_penalty: 1.5,
			..SampleParams::greedy()
		};
		let l = apply_penalties(&[5.0, 1.0], &[0, 0], &p);
		assert!((l[0] - 5.0 / 1.5).abs() < 1e-6);
		assert!((l[1] - 1.0).abs() < 1e-6);
	}

	#[test]
	fn presence_and_frequency_penalties() {
		let p = SampleParams {
			presence_penalty: 1.5,
			frequency_penalty: 0.25,
			..SampleParams::greedy()
		};
		let l = apply_penalties(&[5.0, 1.0, 2.0], &[0, 2, 0], &p);
		assert!((l[0] - (5.0 - 1.5 - 0.5)).abs() < 1e-6);
		assert!((l[1] - 1.0).abs() < 1e-6);
		assert!((l[2] - (2.0 - 1.5 - 0.25)).abs() < 1e-6);
		// greedy sampling sees the penalized logits
		let p = SampleParams {
			presence_penalty: 4.5,
			..SampleParams::greedy()
		};
		assert_eq!(sample(&[5.0, 1.0, 2.0], &[0], &p, &mut Rng::new(1)), 2);
	}

	#[test]
	fn top_p_truncates() {
		// highly peaked distribution: the nucleus keeps only the top token
		let logits = [10.0, 0.0, 0.0, 0.0];
		let mut rng = Rng::new(7);
		for _ in 0..100 {
			assert_eq!(sample(&logits, &[], &params(1.0, 0, 0.5), &mut rng), 0);
		}
	}

	#[test]
	fn min_p_truncates() {
		// p(1) / p(0) = e^-1 = 0.37: min_p 0.5 drops it
		let logits = [1.0, 0.0, -50.0];
		let mut rng = Rng::new(7);
		let p = SampleParams {
			min_p: 0.5,
			..params(1.0, 0, 1.0)
		};
		for _ in 0..100 {
			assert_eq!(sample(&logits, &[], &p, &mut rng), 0);
		}
	}

	#[test]
	fn candidates_are_the_top_k_in_order() {
		let mut rng = Rng::new(11);
		// coarse values: many ties
		let l: Vec<f32> = (0..50_000)
			.map(|_| (rng.next_f32() * 1000.0).floor())
			.collect();
		let mut want: Vec<(f32, u32)> = l.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
		want.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
		for k in [1, 2, 20, 100, 1000] {
			assert_eq!(candidates(&l, k, 0.0), want[..k].to_vec(), "k = {k}");
		}
		// no top-k: everything within the spread
		let c = candidates(&l, 0, 2.0);
		let n = want.iter().filter(|&&(v, _)| v >= want[0].0 - 2.0).count();
		assert_eq!(c, want[..n].to_vec());
	}

	#[test]
	fn nan_logits_do_not_panic() {
		let logits = [f32::NAN, 1.0, f32::NAN, 0.5];
		let mut rng = Rng::new(3);
		for _ in 0..50 {
			let t = sample(&logits, &[], &params(1.0, 2, 1.0), &mut rng);
			assert!(t == 1 || t == 3);
		}
	}

	/// The empirical distribution matches the softmax over the top-k.
	#[test]
	fn distribution_matches_softmax() {
		let logits = [2.0f32, 1.0, 0.5, -1.0, -3.0];
		let t = 0.7;
		let mut rng = Rng::new(5);
		let p = params(t, 3, 1.0);
		let mut hist = [0usize; 5];
		let n = 200_000;
		for _ in 0..n {
			hist[sample(&logits, &[], &p, &mut rng) as usize] += 1;
		}
		let w: Vec<f64> = logits[..3]
			.iter()
			.map(|&v| ((v as f64 - 2.0) / t as f64).exp())
			.collect();
		let s: f64 = w.iter().sum();
		for i in 0..3 {
			let got = hist[i] as f64 / n as f64;
			assert!(
				(got - w[i] / s).abs() < 0.005,
				"token {i}: {got} vs {}",
				w[i] / s
			);
		}
		assert_eq!(hist[3] + hist[4], 0);
	}

	/// Coupled draws (one per position) follow the same truncated softmax:
	/// chi-square over the kept tokens, with top-p cutting one of four.
	#[test]
	fn coupled_distribution_matches_softmax() {
		let logits = [2.0f32, 1.0, 0.5, -1.0, -3.0];
		let t = 0.7;
		let p = SampleParams {
			top_p: 0.97,
			..params(t, 4, 1.0)
		};
		let c = Coupling {
			seed: 99,
			params: p,
		};
		let mut hist = [0usize; 5];
		let n = 200_000;
		for pos in 0..n {
			hist[sample_coupled(&logits, &[], &p, &c, pos as u64) as usize] += 1;
		}
		// the same cut as `sample`'s (top-p keeps the first three here)
		let w: Vec<f64> = logits[..4]
			.iter()
			.map(|&v| ((v as f64 - 2.0) / t as f64).exp())
			.collect();
		let tot: f64 = w.iter().sum();
		let keep = (1..=4)
			.find(|&k| w[..k].iter().sum::<f64>() / tot >= 0.97)
			.unwrap();
		assert_eq!(keep, 3);
		let s: f64 = w[..keep].iter().sum();
		let chi2: f64 = (0..keep)
			.map(|i| {
				let e = n as f64 * w[i] / s;
				(hist[i] as f64 - e).powi(2) / e
			})
			.sum();
		// 2 degrees of freedom: p = 0.001 at 13.8
		assert!(chi2 < 13.8, "chi2 {chi2}, hist {hist:?}");
		assert_eq!(hist[3] + hist[4], 0);
		// and the plain sampler's histogram is the same distribution
		let mut rng = Rng::new(7);
		let mut h2 = [0usize; 5];
		for _ in 0..n {
			h2[sample(&logits, &[], &p, &mut rng) as usize] += 1;
		}
		for i in 0..keep {
			let (a, b) = (hist[i] as f64 / n as f64, h2[i] as f64 / n as f64);
			assert!((a - b).abs() < 0.006, "token {i}: coupled {a}, plain {b}");
		}
	}

	/// A drafter with the model's own distribution proposes exactly the
	/// sampled token; a close one most of the time, far more often than its
	/// argmax would be drawn.
	#[test]
	fn coupled_drafts_match_close_distributions() {
		let p_l = [1.0f32, 0.8, 0.6, 0.2, -0.5, -1.0];
		let q_l = [0.9f32, 0.85, 0.5, 0.3, -0.4, -1.2];
		let sp = params(1.0, 20, 0.95);
		let c = Coupling {
			seed: 5,
			params: sp,
		};
		let (mut same, mut close, mut argmax_hit) = (0, 0, 0);
		let n = 20_000;
		for pos in 0..n {
			let s = sample_coupled(&p_l, &[], &sp, &c, pos);
			same += (c.pick(&p_l, pos).0 == s) as usize;
			close += (c.pick(&q_l, pos).0 == s) as usize;
			argmax_hit += (s == 0) as usize;
		}
		assert_eq!(same, n as usize);
		assert!(close as f64 > 0.8 * n as f64, "close {close} of {n}");
		assert!(close > 2 * argmax_hit, "close {close}, argmax {argmax_hit}");
	}

	#[test]
	fn rng_deterministic() {
		let mut a = Rng::new(123);
		let mut b = Rng::new(123);
		assert_eq!(a.next_u64(), b.next_u64());
		assert_eq!(a.next_f32(), b.next_f32());
	}

	/// `cargo test --release -p sampler bench -- --ignored --nocapture`
	#[test]
	#[ignore]
	fn bench() {
		let mut rng = Rng::new(3);
		let logits: Vec<f32> = (0..248_320).map(|_| rng.next_f32() * 20.0 - 10.0).collect();
		let prev: Vec<u32> = (0..4096)
			.map(|_| (rng.next_u64() % 248_320) as u32)
			.collect();
		let presence = SampleParams {
			presence_penalty: 1.5,
			..SampleParams::instruct()
		};
		for (what, p, prev) in [
			("instruct", SampleParams::instruct(), &[][..]),
			("thinking", SampleParams::thinking(), &[][..]),
			("presence 1.5, 4096 prev", presence, &prev[..]),
			("no top-k", params(1.0, 0, 0.95), &[][..]),
		] {
			let t = std::time::Instant::now();
			for _ in 0..200 {
				std::hint::black_box(sample(&logits, prev, &p, &mut rng));
			}
			eprintln!("{what}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 200.0);
		}
	}
}
