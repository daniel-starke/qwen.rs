//! The initial noise, drawn as PyTorch's CPU `torch.randn(..., generator=
//! torch.Generator("cpu").manual_seed(seed))` draws it, so a seed gives the
//! same image here as in diffusers / ComfyUI (up to the last bit of the
//! transcendental functions: torch uses SLEEF's 1-ULP `logf` / `sinf` /
//! `cosf` on AVX2).
//!
//! torch's float `normal_` on a contiguous tensor of 16+ elements: fill every
//! element with a uniform `(mt19937() & 0xFFFFFF) * 2^-24`, then transform
//! each block of 16 by Box-Muller (`u1 = 1 - d[j]`, `u2 = d[j + 8]`, `r =
//! sqrt(-2 ln u1)`, `d[j] = r cos(2 pi u2)`, `d[j + 8] = r sin(2 pi u2)`);
//! a tail of `n % 16` redoes the last 16 positions from fresh uniforms.

/// The 32-bit Mersenne Twister (`std::mt19937`).
pub struct Mt19937 {
	s: [u32; 624],
	i: usize,
}

impl Mt19937 {
	pub fn new(seed: u32) -> Mt19937 {
		let mut s = [0u32; 624];
		s[0] = seed;
		for i in 1..624 {
			s[i] = 1_812_433_253u32
				.wrapping_mul(s[i - 1] ^ (s[i - 1] >> 30))
				.wrapping_add(i as u32);
		}
		Mt19937 { s, i: 624 }
	}

	pub fn next_u32(&mut self) -> u32 {
		if self.i >= 624 {
			for k in 0..624 {
				let y = (self.s[k] & 0x8000_0000) | (self.s[(k + 1) % 624] & 0x7FFF_FFFF);
				let mut v = self.s[(k + 397) % 624] ^ (y >> 1);
				if y & 1 != 0 {
					v ^= 0x9908_B0DF;
				}
				self.s[k] = v;
			}
			self.i = 0;
		}
		let mut y = self.s[self.i];
		self.i += 1;
		y ^= y >> 11;
		y ^= (y << 7) & 0x9D2C_5680;
		y ^= (y << 15) & 0xEFC6_0000;
		y ^ (y >> 18)
	}

	fn uniform(&mut self) -> f32 {
		(self.next_u32() & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0)
	}
}

fn box_muller16(d: &mut [f32]) {
	for j in 0..8 {
		let u1 = 1.0 - d[j];
		let u2 = d[j + 8];
		let r = (-2.0 * u1.ln()).sqrt();
		let th = 2.0 * std::f32::consts::PI * u2;
		d[j] = r * th.cos();
		d[j + 8] = r * th.sin();
	}
}

/// `n` standard normal floats as `torch.randn(n)` on a CPU generator
/// seeded with `seed` (`n` >= 16).
pub fn torch_randn(seed: u64, n: usize) -> Vec<f32> {
	assert!(n >= 16);
	// torch seeds mt19937 with the low 32 bits of the 64-bit seed
	let mut g = Mt19937::new(seed as u32);
	let mut d: Vec<f32> = (0..n).map(|_| g.uniform()).collect();
	for b in (0..n - 15).step_by(16) {
		box_muller16(&mut d[b..b + 16]);
	}
	if !n.is_multiple_of(16) {
		let b = n - 16;
		for v in &mut d[b..] {
			*v = g.uniform();
		}
		box_muller16(&mut d[b..]);
	}
	d
}

#[cfg(test)]
mod tests {
	#[test]
	fn mt19937_matches_std() {
		// the C++ standard's check: the 10000th output of a default-seeded mt19937
		let mut g = super::Mt19937::new(5489);
		let mut v = 0;
		for _ in 0..10000 {
			v = g.next_u32();
		}
		assert_eq!(v, 4_123_659_995);
	}
}
