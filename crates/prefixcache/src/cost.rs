//! What a prefill costs on this machine, learned from the prefills the
//! cache sees: the seconds per token grow with the context (attention), so
//! `t(a -> p) = (p - a) (c0 + c1 (a + p) / 2)`. Every eviction compares
//! "seconds of prefill this state saves" across entries of very different
//! lengths, and the disk tier weighs those seconds against bytes written,
//! so the absolute numbers matter as much as the ranking.
//!
//! A weighted least-squares fit of `secs / tokens` on the piece's mean
//! position, with old observations decaying and two prior pseudo-points
//! (the defaults) so a single measurement cannot tip the slope.

#[derive(Clone, Debug)]
pub struct CostModel {
	/// Seconds per token at position 0, and their growth per position.
	pub c0: f64,
	pub c1: f64,
	/// Disk read bytes per second.
	pub read_bw: f64,
	/// The priors `c0`, `c1` (the constructor's).
	p0: f64,
	p1: f64,
	// decayed sums of (w, x, y, xx, xy)
	s: [f64; 5],
}

/// Weight of each new observation relative to the decayed past.
const DECAY: f64 = 0.9;
/// Pieces shorter than this are mostly fixed overhead.
const MIN_TOKENS: usize = 64;
/// Where the prior's second point sits, and the weight of each of its two
/// points (the decayed observations weigh up to `1 / (1 - DECAY)` = 10).
const PRIOR_X: f64 = 100_000.0;
const PRIOR_W: f64 = 0.1;

impl CostModel {
	/// Priors: `tok_s` tokens per second at the start of a context, the
	/// per-token time doubling by `double_at` positions.
	pub fn new(tok_s: f64, double_at: f64) -> CostModel {
		let c0 = 1.0 / tok_s.max(1e-3);
		CostModel {
			c0,
			c1: c0 / double_at.max(1.0),
			read_bw: 1.5e9,
			p0: c0,
			p1: c0 / double_at.max(1.0),
			s: [0.0; 5],
		}
	}

	/// Seconds to prefill positions `[a, p)`.
	pub fn replay(&self, a: usize, p: usize) -> f64 {
		if p <= a {
			return 0.0;
		}
		let n = (p - a) as f64;
		n * (self.c0 + self.c1 * (a + p) as f64 / 2.0)
	}

	/// Seconds to read `bytes` back from disk.
	pub fn read(&self, bytes: usize) -> f64 {
		bytes as f64 / self.read_bw
	}

	/// A prefill of positions `[a, p)` took `secs`.
	pub fn observe(&mut self, a: usize, p: usize, secs: f64) {
		if p < a + MIN_TOKENS || !(secs > 0.0) {
			return;
		}
		let x = (a + p) as f64 / 2.0;
		let y = secs / (p - a) as f64;
		for v in &mut self.s {
			*v *= DECAY;
		}
		let o = [1.0, x, y, x * x, x * y];
		for (s, o) in self.s.iter_mut().zip(o) {
			*s += o;
		}
		self.fit();
	}

	/// A disk read of `bytes` took `secs`.
	pub fn observe_read(&mut self, bytes: usize, secs: f64) {
		if bytes >= 1 << 26 && secs > 0.0 {
			self.read_bw = 0.7 * self.read_bw + 0.3 * bytes as f64 / secs;
		}
	}

	fn fit(&mut self) {
		// prior points (0, c0) and (X, c0 + c1 X)
		let (p0, p1) = (self.p0, self.p0 + self.p1 * PRIOR_X);
		let [w, x, y, xx, xy] = self.s;
		let w = w + 2.0 * PRIOR_W;
		let x = x + PRIOR_W * PRIOR_X;
		let y = y + PRIOR_W * (p0 + p1);
		let xx = xx + PRIOR_W * PRIOR_X * PRIOR_X;
		let xy = xy + PRIOR_W * PRIOR_X * p1;
		let det = w * xx - x * x;
		if det.abs() < 1e-9 {
			return;
		}
		let c1 = ((w * xy - x * y) / det).max(0.0);
		let c0 = ((y - c1 * x) / w).max(1e-6);
		self.c0 = c0;
		self.c1 = c1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn learns_the_machine_from_its_prefills() {
		let mut c = CostModel::new(100.0, 100_000.0);
		assert!((c.replay(0, 100) - 1.0).abs() < 0.01);
		// the machine is 4x faster than the prior, with a steeper slope
		let truth = |a: usize, p: usize| (p - a) as f64 * (0.0025 + 1e-7 * (a + p) as f64 / 2.0);
		for k in 0..40 {
			let a = (k % 8) * 4096;
			c.observe(a, a + 2048, truth(a, a + 2048));
		}
		let (want, got) = (truth(10_000, 30_000), c.replay(10_000, 30_000));
		assert!((got / want - 1.0).abs() < 0.1, "{got} vs {want}");
		// short pieces and nonsense are ignored
		let before = (c.c0, c.c1);
		c.observe(0, 10, 99.0);
		c.observe(0, 1000, f64::NAN);
		assert_eq!(before, (c.c0, c.c1));
		assert_eq!(c.replay(5, 5), 0.0);
	}
}
