//! fp16 <-> f32 conversions.
//!
//! `f32_to_fp16` rounds to nearest (ties-to-even); subnormals flush to zero and
//! overflow saturates to inf, which matches the cache-friendly storage used
//! elsewhere in the crate. `fp16_to_f32` is the direct bit-decode (no table),
//! keeping the crate standalone.

/// Convert an `f32` to an IEEE-754 half-precision `u16` (round-to-nearest,
/// ties-to-even; subnormals flush to zero, overflow saturates to inf). Used to
/// store cached dequantized weights compactly (f16 vs f32 halves the cache
/// footprint so the whole 27B model fits in 128 GiB).
pub fn f32_to_fp16(value: f32) -> u16 {
	let x = value.to_bits();
	let sign = (x >> 16) & 0x8000;
	let exp = ((x >> 23) & 0xff) as i32;
	let mant = x & 0x7f_ffff;
	if exp == 255 {
		// inf or nan
		(sign | 0x7c00 | ((mant != 0) as u32 * 0x200)) as u16
	} else if exp == 0 {
		// zero / subnormal -> flush to zero
		sign as u16
	} else {
		let e = exp - 127 + 15;
		if e >= 31 {
			(sign | 0x7c00) as u16
		} else if e <= 0 {
			sign as u16
		} else {
			let m = mant >> 13;
			let rem = mant & 0x1fff;
			let mut h = sign | ((e as u32) << 10) | m;
			if rem > 0x1000 || (rem == 0x1000 && (m & 1) != 0) {
				h += 1;
			}
			h as u16
		}
	}
}

/// Convert an IEEE-754 half-precision value to `f32`.
///
/// This is the same conversion llama.cpp's `GGML_FP16_TO_FP32` performs via its
/// table; we keep a direct bit-decode so the crate stays standalone.
#[inline]
pub fn fp16_to_f32(h: u16) -> f32 {
	let sign = (h >> 15) as u32;
	let exp = ((h >> 10) & 0x1f) as u32;
	let mant = (h & 0x3ff) as u32;

	if exp == 0 {
		if mant == 0 {
			f32::from_bits(sign << 31)
		} else {
			// subnormal half: value = mant * 2^-24  (implicit leading bit is 0)
			let mag = (mant as f32) * f32::from_bits(0x3380_0000); // 2^-24
			if sign != 0 {
				-mag
			} else {
				mag
			}
		}
	} else if exp == 0x1f {
		f32::from_bits((sign << 31) | 0x7f80_0000 | (mant << 13))
	} else {
		f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	}
}
