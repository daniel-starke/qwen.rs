//! A cache key: the token ids a state has consumed, with running hashes at
//! every [`B`] tokens so prefix tests between long keys (two sessions that
//! share a 30k-token preamble) cost a binary search and one block's
//! compare, not a 120 KB `memcmp` per pair; the eviction pass relates
//! every entry to every other.

use crate::hash;

/// Tokens per running hash.
const B: usize = 256;

#[derive(Clone, Debug)]
pub(crate) struct Key {
	pub tokens: Vec<u32>,
	/// `ph[i]`: hash of `tokens[..(i + 1) * B]`.
	ph: Vec<u64>,
}

impl Key {
	pub fn new(tokens: Vec<u32>) -> Key {
		let mut ph = Vec::with_capacity(tokens.len() / B);
		let mut h = 0u64;
		for c in tokens.chunks_exact(B) {
			h = hash::tokens(h, c);
			ph.push(h);
		}
		Key { tokens, ph }
	}

	pub fn len(&self) -> usize {
		self.tokens.len()
	}

	/// Length of the common prefix of `self` and `other`. Exact up to the
	/// running hashes' collisions (64 bits); [`Key::is_prefix_exact`]
	/// decides a hit.
	pub fn common(&self, other: &Key) -> usize {
		let n = self.ph.len().min(other.ph.len());
		// blocks are cumulative: equal at i means equal at every j < i
		let (mut lo, mut hi) = (0usize, n);
		while lo < hi {
			let mid = (lo + hi).div_ceil(2);
			if self.ph[mid - 1] == other.ph[mid - 1] {
				lo = mid;
			} else {
				hi = mid - 1;
			}
		}
		let s = lo * B;
		s + self.tokens[s..]
			.iter()
			.zip(&other.tokens[s..])
			.take_while(|(a, b)| a == b)
			.count()
	}

	/// `self` is a proper prefix of `other` (hash-fast).
	pub fn is_proper_prefix_of(&self, other: &Key) -> bool {
		self.len() < other.len() && self.common(other) == self.len()
	}

	/// `self` is a proper prefix of `ids`, compared token by token.
	pub fn is_prefix_exact(&self, ids: &[u32]) -> bool {
		self.len() < ids.len() && ids.starts_with(&self.tokens)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn common_prefix_across_block_boundaries() {
		let a: Vec<u32> = (0..1000).collect();
		for cut in [0, 1, 255, 256, 257, 511, 512, 999] {
			let mut b = a.clone();
			b[cut] = 77777;
			assert_eq!(Key::new(a.clone()).common(&Key::new(b)), cut, "{cut}");
		}
		let k = Key::new(a[..600].to_vec());
		let full = Key::new(a.clone());
		assert!(k.is_proper_prefix_of(&full) && !full.is_proper_prefix_of(&k));
		assert!(!full.is_proper_prefix_of(&full));
		assert!(k.is_prefix_exact(&a) && !full.is_prefix_exact(&a));
		assert_eq!(full.common(&k), 600);
	}
}
