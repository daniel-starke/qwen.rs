//! Content hashes: chunk identity on disk (a page written once is never
//! written again) and the token-prefix hashes of [`crate::key::Key`].
//!
//! Not cryptographic. Four multiply-rotate lanes over 8-byte words, folded
//! into 128 bits with the murmur3 finalizer; a chunk is also matched on its
//! length. Runs at memory speed, so hashing a state costs about what
//! copying it does.

const K: [u64; 4] = [
	0x9e37_79b9_7f4a_7c15,
	0xc2b2_ae3d_27d4_eb4f,
	0x1656_67b1_9e37_79f9,
	0x85eb_ca77_c2b2_ae63,
];

#[inline]
fn fmix(mut h: u64) -> u64 {
	h ^= h >> 33;
	h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
	h ^= h >> 33;
	h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
	h ^ (h >> 33)
}

#[inline]
fn lane(a: u64, w: u64, k: u64) -> u64 {
	(a ^ w).wrapping_mul(k).rotate_left(31).wrapping_add(w)
}

/// 128-bit hash of `b`.
pub fn h128(b: &[u8]) -> (u64, u64) {
	let mut a = [K[0] ^ 0x5151, K[1] ^ 0xa3a3, K[2] ^ 0x7c7c, K[3] ^ 0x0e0e];
	let mut blocks = b.chunks_exact(32);
	for c in &mut blocks {
		for i in 0..4 {
			let w = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
			a[i] = lane(a[i], w, K[i]);
		}
	}
	let rest = blocks.remainder();
	if !rest.is_empty() {
		let mut t = [0u8; 32];
		t[..rest.len()].copy_from_slice(rest);
		for i in 0..4 {
			let w = u64::from_le_bytes(t[i * 8..i * 8 + 8].try_into().unwrap());
			a[i] = lane(a[i], w, K[i]);
		}
	}
	let n = b.len() as u64;
	let x = fmix(a[0] ^ a[2].rotate_left(17) ^ n);
	let y = fmix(a[1] ^ a[3].rotate_left(23) ^ n.rotate_left(32));
	(fmix(x ^ y.rotate_left(13)), fmix(y ^ x.rotate_left(29)))
}

/// Extend a running hash of token ids by `t`.
#[inline]
pub fn tokens(mut h: u64, t: &[u32]) -> u64 {
	for &x in t {
		h = (h ^ x as u64).wrapping_mul(K[0]).rotate_left(27);
	}
	fmix(h)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn hashes_tell_contents_and_lengths_apart() {
		let a = vec![7u8; 1000];
		let mut b = a.clone();
		b[999] = 8;
		assert_eq!(h128(&a), h128(&a.clone()));
		assert_ne!(h128(&a), h128(&b));
		assert_ne!(h128(&a[..999]), h128(&a));
		// zero padding of the tail is not the same as real zeros
		assert_ne!(h128(&[0u8; 3]), h128(&[0u8; 4]));
		assert_ne!(tokens(1, &[1, 2]), tokens(1, &[2, 1]));
	}
}
