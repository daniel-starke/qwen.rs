//! K-quant / f32 / f16 dequantization to plain `f32` row-major matrices.
//!
//! Reference: llama.cpp `ggml/src/ggml-quants.c` (`dequantize_row_q4_K`,
//! `get_scale_min_k4`) and `ggml/src/ggml-common.h` (`block_q4_K`,
//! `block_q8_0`, `QK_K = 256`, `K_SCALE_SIZE = 12`).
//!
//! Dequantization is dependency-free and produces plain `f32` row-major
//! matrices that the model graph and GEMM routines consume.

use super::GgmlType;
use super::{f32_to_fp16, fp16_to_f32};
use super::{K_SCALE_SIZE, QK_K};

/// Recover the 6-bit (scale, min) pair for sub-block `j` of a `Q4_K` block's
/// 12-byte `scales` array, exactly as llama.cpp `get_scale_min_k4`.
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
	if j < 4 {
		(scales[j] & 63, scales[j + 4] & 63)
	} else {
		(
			(scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
			(scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
		)
	}
}

/// Dequantize one `Q4_K` super-block (144 bytes -> 256 `f32`).
///
/// Block byte layout: `d`(2, fp16) `dmin`(2, fp16) `scales`(12) `qs`(128).
pub(crate) fn dequantize_q4_k_block(block: &[u8], out: &mut [f32]) {
	debug_assert!(block.len() >= 2 + 2 + K_SCALE_SIZE + QK_K / 2);
	debug_assert!(out.len() >= QK_K);

	let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
	let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
	let scales = &block[4..4 + K_SCALE_SIZE];
	let qs = &block[4 + K_SCALE_SIZE..];

	let mut y = 0usize;
	let mut is = 0usize;
	let mut q = qs;
	for _ in (0..QK_K).step_by(64) {
		let (sc1, m1) = get_scale_min_k4(is, scales);
		let d1 = d * sc1 as f32;
		let m1 = dmin * m1 as f32;
		let (sc2, m2) = get_scale_min_k4(is + 1, scales);
		let d2 = d * sc2 as f32;
		let m2 = dmin * m2 as f32;

		for l in 0..32 {
			out[y] = d1 * (q[l] & 0x0F) as f32 - m1;
			y += 1;
		}
		for l in 0..32 {
			out[y] = d2 * (q[l] >> 4) as f32 - m2;
			y += 1;
		}
		q = &q[32..];
		is += 2;
	}
}

/// Dequantize a full `Q4_K` tensor.
///
/// `n` is the number of *elements* (must be a multiple of `QK_K`).
pub fn dequantize_q4_k(src: &[u8], n: usize) -> Vec<f32> {
	assert!(
		n.is_multiple_of(QK_K),
		"Q4_K element count must be a multiple of 256"
	);
	let nblocks = n / QK_K;
	let block_bytes = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144
	assert_eq!(
		src.len(),
		nblocks * block_bytes,
		"Q4_K buffer size does not match element count"
	);

	let mut out = vec![0.0f32; n];
	for b in 0..nblocks {
		let start = b * block_bytes;
		dequantize_q4_k_block(&src[start..start + block_bytes], &mut out[b * QK_K..]);
		if (b == 0 || b == 7380) && std::env::var("Q4K_DBG").is_ok() {
			let d = fp16_to_f32(u16::from_le_bytes([src[start], src[start + 1]]));
			let dmin = fp16_to_f32(u16::from_le_bytes([src[start + 2], src[start + 3]]));
			let sc: Vec<u8> = src[start + 4..start + 16].to_vec();
			eprintln!("Q4KDBG b={} d={:.6} dmin={:.6} scales={:?}", b, d, dmin, sc);
			eprintln!(
				"Q4KDBG b={} out[blk]={:?}",
				b,
				&out[b * QK_K..b * QK_K + 16]
			);
		}
	}
	out
}

/// Dequantize an `F16` tensor (`n` elements, 2 bytes each).
pub fn dequantize_f16(src: &[u8], n: usize) -> Vec<f32> {
	assert_eq!(src.len(), n * 2, "F16 buffer size mismatch");
	let mut out = vec![0.0f32; n];
	for i in 0..n {
		let h = u16::from_le_bytes([src[2 * i], src[2 * i + 1]]);
		out[i] = fp16_to_f32(h);
	}
	out
}

/// Copy an `F32` tensor as-is (`n` elements, 4 bytes each).
pub fn dequantize_f32(src: &[u8], n: usize) -> Vec<f32> {
	assert_eq!(src.len(), n * 4, "F32 buffer size mismatch");
	let mut out = vec![0.0f32; n];
	for i in 0..n {
		out[i] = f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]]);
	}
	out
}

/// Dequantize one `Q8_0` block (34 bytes -> 32 `f32`): `y = d * qs` with `d`
/// an fp16 scale and `qs` signed 8-bit quants.
pub(crate) fn dequantize_q8_0_block(block: &[u8], out: &mut [f32]) {
	debug_assert!(block.len() >= 2 + 32);
	let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
	for i in 0..32 {
		out[i] = d * block[2 + i] as i8 as f32;
	}
}

/// Dequantize a full `Q8_0` tensor (`n` elements, multiple of 32).
pub fn dequantize_q8_0(src: &[u8], n: usize) -> Vec<f32> {
	assert!(
		n.is_multiple_of(32),
		"Q8_0 element count must be a multiple of 32"
	);
	let nblocks = n / 32;
	let block_bytes = 2 + 32;
	assert_eq!(src.len(), nblocks * block_bytes);

	let mut out = vec![0.0f32; n];
	for b in 0..nblocks {
		let start = b * block_bytes;
		dequantize_q8_0_block(&src[start..start + block_bytes], &mut out[b * 32..]);
	}
	out
}

/// Dequantize one `Q6_K` block (210 bytes -> 256 `f32`): `y = d * scale * q`
/// with `d` an fp16 super-scale, `scale` signed 8-bit per 16-element sub-block,
/// and `q` a signed 6-bit quant (`ql` = low 4 bits, `qh` = high 2 bits).
///
/// Byte layout matches `ggml.h` `block_q6_K`: `ql`(128) `qh`(64) `scales`(16)
/// `d`(2 fp16).
pub(crate) fn dequantize_q6_k_block(block: &[u8], out: &mut [f32]) {
	debug_assert!(block.len() >= 210);
	debug_assert!(out.len() >= QK_K);

	let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
	let ql = &block[0..128];
	let qh = &block[128..192];
	let scales = &block[192..208];

	let mut y = 0usize;
	let mut ql_off = 0usize;
	let mut qh_off = 0usize;
	let mut sc_off = 0usize;
	while y < QK_K {
		for l in 0..32 {
			let is = l / 16;
			let q1 = ((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
			let q2 =
				((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
			let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
			let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
			let s1 = scales[sc_off + is] as i8 as f32;
			let s2 = scales[sc_off + is + 2] as i8 as f32;
			let s3 = scales[sc_off + is + 4] as i8 as f32;
			let s4 = scales[sc_off + is + 6] as i8 as f32;
			out[y + l] = d * s1 * q1 as f32;
			out[y + l + 32] = d * s2 * q2 as f32;
			out[y + l + 64] = d * s3 * q3 as f32;
			out[y + l + 96] = d * s4 * q4 as f32;
		}
		y += 128;
		ql_off += 64;
		qh_off += 32;
		sc_off += 8;
	}
}

/// Dequantize a full `Q6_K` tensor (`n` elements, multiple of 256).
pub fn dequantize_q6_k(src: &[u8], n: usize) -> Vec<f32> {
	assert!(
		n.is_multiple_of(QK_K),
		"Q6_K element count must be a multiple of 256"
	);
	let nblocks = n / QK_K;
	let block_bytes = 210;
	assert_eq!(src.len(), nblocks * block_bytes);

	let mut out = vec![0.0f32; n];
	for b in 0..nblocks {
		let start = b * block_bytes;
		dequantize_q6_k_block(&src[start..start + block_bytes], &mut out[b * QK_K..]);
	}
	out
}

/// Quantize 256 `f32` into one `Q6_K` block (210 bytes), the inverse of
/// [`dequantize_q6_k_block`]. A plain per-16 absmax scale, like ggml's
/// `quantize_row_q6_K_ref` without its RMSE search: used for weights that do
/// not ship as GGUF (the DFlash2 drafter is BF16 safetensors), where 6 bits
/// with a 16-wide scale is far below the noise the consumer tolerates.
pub fn quantize_q6_k_block(x: &[f32], out: &mut [u8]) {
	assert!(x.len() >= QK_K && out.len() >= 210);
	// Per 16-group scale: the signed value of largest magnitude maps to -32,
	// which uses the whole asymmetric [-32, 31] range.
	let mut sc = [0.0f32; 16];
	let mut max_sc = 0.0f32;
	for ib in 0..16 {
		let mut vmax = 0.0f32;
		for &v in &x[ib * 16..ib * 16 + 16] {
			if v.abs() > vmax.abs() {
				vmax = v;
			}
		}
		sc[ib] = vmax / -32.0;
		if sc[ib].abs() > max_sc.abs() {
			max_sc = sc[ib];
		}
	}
	let mut l = [32u8; QK_K];
	let mut scales = [0i8; 16];
	let d = if max_sc == 0.0 {
		0.0
	} else {
		// The kernels read `d` back as f16, so requantize against that value.
		// `d` is block max / 4096: below 0.25 that is an fp16 *subnormal*,
		// which is the normal case for weights, and `f32_to_fp16` flushes
		// subnormals to zero (it was written for cached activations).
		let d = fp16_to_f32(f32_to_fp16_subnormal(max_sc / -128.0));
		for ib in 0..16 {
			scales[ib] = (sc[ib] / d).round().clamp(-128.0, 127.0) as i8;
			let ds = d * scales[ib] as f32;
			if ds == 0.0 {
				continue;
			}
			for j in ib * 16..ib * 16 + 16 {
				l[j] = ((x[j] / ds).round().clamp(-32.0, 31.0) as i32 + 32) as u8;
			}
		}
		d
	};
	let (ql, rest) = out.split_at_mut(128);
	let (qh, rest) = rest.split_at_mut(64);
	for j in 0..2 {
		let (y, qlo, qho) = (j * 128, j * 64, j * 32);
		for i in 0..32 {
			let q = |k: usize| l[y + i + 32 * k];
			ql[qlo + i] = (q(0) & 0xF) | ((q(2) & 0xF) << 4);
			ql[qlo + i + 32] = (q(1) & 0xF) | ((q(3) & 0xF) << 4);
			qh[qho + i] =
				(q(0) >> 4) | ((q(1) >> 4) << 2) | ((q(2) >> 4) << 4) | ((q(3) >> 4) << 6);
		}
	}
	for ib in 0..16 {
		rest[ib] = scales[ib] as u8;
	}
	rest[16..18].copy_from_slice(&f32_to_fp16_subnormal(d).to_le_bytes());
}

/// Quantize 256 `f32` into one `Q4_K` block (144 bytes), the inverse of
/// [`dequantize_q4_k_block`]: per 32-weight sub-block an affine
/// `x = d * sc * q - dmin * m` from that sub-block's min/max, with 6-bit `sc`/`m` against the
/// block's f16 `d`/`dmin`. No RMSE search (ggml's `make_qkx2_quants`), so it
/// is a little noisier than llama.cpp's own Q4_K; meant for drafter weights.
pub fn quantize_q4_k_block(x: &[f32], out: &mut [u8]) {
	assert!(x.len() >= QK_K && out.len() >= 144);
	let mut scale = [0.0f32; 8];
	let mut minv = [0.0f32; 8];
	for i in 0..8 {
		let s = &x[i * 32..i * 32 + 32];
		let lo = s.iter().fold(0.0f32, |m, &v| m.min(v));
		let hi = s.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
		scale[i] = (hi - lo) / 15.0;
		minv[i] = -lo;
	}
	let max_scale = scale.iter().fold(0.0f32, |m, &v| m.max(v));
	let max_min = minv.iter().fold(0.0f32, |m, &v| m.max(v));
	let d = fp16_to_f32(f32_to_fp16_subnormal(max_scale / 63.0));
	let dmin = fp16_to_f32(f32_to_fp16_subnormal(max_min / 63.0));
	let mut ls = [0u8; 8];
	let mut lm = [0u8; 8];
	for i in 0..8 {
		ls[i] = if d > 0.0 {
			(scale[i] / d).round().clamp(0.0, 63.0) as u8
		} else {
			0
		};
		lm[i] = if dmin > 0.0 {
			(minv[i] / dmin).round().clamp(0.0, 63.0) as u8
		} else {
			0
		};
	}
	let mut l = [0u8; QK_K];
	for i in 0..8 {
		let ds = d * ls[i] as f32;
		let m = dmin * lm[i] as f32;
		if ds == 0.0 {
			continue;
		}
		for j in i * 32..i * 32 + 32 {
			l[j] = ((x[j] + m) / ds).round().clamp(0.0, 15.0) as u8;
		}
	}
	out[0..2].copy_from_slice(&f32_to_fp16_subnormal(d).to_le_bytes());
	out[2..4].copy_from_slice(&f32_to_fp16_subnormal(dmin).to_le_bytes());
	let sc = &mut out[4..4 + K_SCALE_SIZE];
	sc.fill(0);
	for j in 0..8 {
		if j < 4 {
			sc[j] = ls[j];
			sc[j + 4] = lm[j];
		} else {
			sc[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
			sc[j - 4] |= (ls[j] >> 4) << 6;
			sc[j] |= (lm[j] >> 4) << 6;
		}
	}
	let qs = &mut out[4 + K_SCALE_SIZE..144];
	for j in 0..4 {
		for i in 0..32 {
			qs[j * 32 + i] = l[j * 64 + i] | (l[j * 64 + 32 + i] << 4);
		}
	}
}

/// [`quantize_q6_k`] for `Q4_K` (a [`crate::LinearWeight::Q4K`]).
pub fn quantize_q4_k(x: &[f32], rows: usize, cols: usize) -> Vec<u8> {
	assert!(cols.is_multiple_of(QK_K) && x.len() == rows * cols);
	let nb = cols / QK_K;
	let mut out = vec![0u8; rows * nb * 144];
	let op = out.as_mut_ptr() as usize;
	crate::par_for(rows, |r| {
		for b in 0..nb {
			// SAFETY: row `r` owns bytes [r*nb*144, (r+1)*nb*144).
			let dst = unsafe {
				std::slice::from_raw_parts_mut((op as *mut u8).add((r * nb + b) * 144), 144)
			};
			quantize_q4_k_block(&x[r * cols + b * QK_K..r * cols + (b + 1) * QK_K], dst);
		}
	});
	out
}

/// `f32_to_fp16`, but rounding values below the smallest normal half to the
/// nearest subnormal instead of to zero (as ggml's `GGML_FP32_TO_FP16`).
pub(crate) fn f32_to_fp16_subnormal(v: f32) -> u16 {
	let m = (v.abs() * (1u32 << 24) as f32).round();
	if m >= 1024.0 {
		return f32_to_fp16(v);
	}
	(m as u16) | if v < 0.0 { 0x8000 } else { 0 }
}

/// Quantize a row-major `[rows, cols]` matrix to `Q6_K` (`cols` a multiple of
/// 256), rows in parallel. The bytes are a [`crate::LinearWeight::Q6K`].
pub fn quantize_q6_k(x: &[f32], rows: usize, cols: usize) -> Vec<u8> {
	assert!(cols.is_multiple_of(QK_K) && x.len() == rows * cols);
	let nb = cols / QK_K;
	let mut out = vec![0u8; rows * nb * 210];
	let op = out.as_mut_ptr() as usize;
	crate::par_for(rows, |r| {
		for b in 0..nb {
			// SAFETY: row `r` owns bytes [r*nb*210, (r+1)*nb*210).
			let dst = unsafe {
				std::slice::from_raw_parts_mut((op as *mut u8).add((r * nb + b) * 210), 210)
			};
			quantize_q6_k_block(&x[r * cols + b * QK_K..r * cols + (b + 1) * QK_K], dst);
		}
	});
	out
}

/// Dispatch on the ggml quantization type. `n` is the element count (the raw
/// `src` bytes are interpreted per the type's block layout).
pub fn dequantize(ty: GgmlType, src: &[u8], n: usize) -> Vec<f32> {
	match ty {
		GgmlType::F32 => dequantize_f32(src, n),
		GgmlType::F16 => dequantize_f16(src, n),
		GgmlType::Q4_K | GgmlType::Q4_K_M => dequantize_q4_k(src, n),
		GgmlType::Q6_K => dequantize_q6_k(src, n),
		GgmlType::Q8_0 => dequantize_q8_0(src, n),
		GgmlType::BF16 => {
			// BF16: top 16 bits of f32.
			assert_eq!(src.len(), n * 2, "BF16 buffer size mismatch");
			let mut out = vec![0.0f32; n];
			for i in 0..n {
				let bits = u32::from_le_bytes([0, 0, src[2 * i], src[2 * i + 1]]);
				out[i] = f32::from_bits(bits);
			}
			out
		}
		other => panic!("dequantize: unsupported ggml type {:?}", other),
	}
}
