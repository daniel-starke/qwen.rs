//! `LinearWeight` dispatch, fused weight concatenation, transposition helpers,
//! and the batched (seq_len > 1) fused Q4_K / Q6_K dequant-GEMM kernels.

use super::{
	fp16_to_f32, gemm, gemm_f16, gemv, gemv_f16, gemv_q4k, gemv_q6k, get_scale_shuffle_k6,
	hsum_float_8, par_gemm_collect, par_range, par_range_pre, q4k_dot_q8k_scalar, q4k_rows,
	q6k_block_bytes, q6k_dot_q8k_scalar, q6k_rows, quantize_q8k, K_SCALE_SIZE, Q8K, QK_K,
};

/// A large matmul weight in the form ready for the CPU GEMV, with no per-token
/// conversion. Most big matmuls are `Q4_K_M` (kept `Q4K`) or `Q6_K` (kept `Q6K`)
/// and stay **quantized**: the GEMV dequantizes them on the fly (fused
/// dequant-GEMV) with ~4x (Q4_K) / ~2.7x (Q6_K) less memory traffic than an
/// expanded `f16`/`f32` matrix, matching llama.cpp. The few remaining non-K big
/// matmuls are kept as `f16`/`f32`.
pub enum LinearWeight {
	Q4K {
		raw: Vec<u8>,
		out_dim: usize,
		in_dim: usize,
	},
	Q6K {
		raw: Vec<u8>,
		out_dim: usize,
		in_dim: usize,
	},
	F16(Vec<u16>),
	F32(Vec<f32>),
}

impl LinearWeight {
	pub fn out_dim(&self) -> usize {
		match self {
			LinearWeight::Q4K { out_dim, .. } => *out_dim,
			LinearWeight::Q6K { out_dim, .. } => *out_dim,
			_ => panic!("out_dim only defined for quantized weights"),
		}
	}
}

/// Dispatch a single-token matmul on a [`LinearWeight`] (fused dequant for Q4_K /
/// Q6_K; f16 / f32 for the rest).
pub fn apply_linear(w: &LinearWeight, x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
	// Every kernel below writes all `out_dim` rows; see `uninit_vec`.
	let mut y = super::uninit_vec(out_dim);
	match w {
		LinearWeight::Q4K { raw, .. } => gemv_q4k(out_dim, in_dim, x, raw, &mut y),
		LinearWeight::Q6K { raw, .. } => gemv_q6k(out_dim, in_dim, x, raw, &mut y),
		LinearWeight::F16(v) => gemv_f16(out_dim, in_dim, x, v, &mut y),
		LinearWeight::F32(v) => gemv(out_dim, in_dim, x, v, &mut y),
	}
	y
}

/// Several single-token matmuls that **share the same activation**, run as one
/// pool dispatch and one `q8_K` quantization.
///
/// `Wq/Wk/Wv`, `Wqkv/WqkvGate`, `FfnUp/FfnGate` and `ssm_beta/ssm_alpha` all
/// consume the same normalized hidden state. Run separately they cost one
/// ~7 us fork-join and one activation quantization *each*; here the output rows
/// of every job are concatenated into a single global row space, so the whole
/// group is one barrier. Results are identical to calling [`apply_linear`] per
/// job: each output row is an independent dot product.
///
/// Falls back to per-job [`apply_linear`] if any weight is not `Q4_K`/`Q6_K`.
pub fn apply_linear_group(
	jobs: &[(&LinearWeight, usize)],
	x: &[f32],
	in_dim: usize,
) -> Vec<Vec<f32>> {
	let kquant = jobs
		.iter()
		.all(|(w, _)| matches!(w, LinearWeight::Q4K { .. } | LinearWeight::Q6K { .. }));
	if jobs.len() < 2 || !kquant {
		return jobs
			.iter()
			.map(|(w, od)| apply_linear(w, x, *od, in_dim))
			.collect();
	}
	let nb = in_dim / QK_K;
	let q8 = quantize_q8k(x, in_dim);
	let mut outs: Vec<Vec<f32>> = jobs.iter().map(|(_, od)| super::uninit_vec(*od)).collect();
	// Global row `g` belongs to the job whose [start, start+out_dim) covers it.
	let starts: Vec<usize> = jobs
		.iter()
		.scan(0usize, |acc, (_, od)| {
			let s = *acc;
			*acc += od;
			Some(s)
		})
		.collect();
	let total: usize = jobs.iter().map(|(_, od)| od).sum();
	let bytes: usize = jobs.iter().map(|(w, od)| row_bytes_of(w, nb) * od).sum();
	let ptrs: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
	par_range(total, bytes / total.max(1), |gs, ge| {
		let mut g = gs;
		while g < ge {
			let j = match starts.binary_search(&g) {
				Ok(j) => j,
				Err(j) => j - 1,
			};
			let (w, od) = jobs[j];
			let lo = g - starts[j];
			let hi = (ge - starts[j]).min(od);
			// SAFETY: rows [lo, hi) of job `j` are disjoint across chunks, and
			// `par_range` joins before `outs` is read.
			let out =
				unsafe { std::slice::from_raw_parts_mut((ptrs[j] as *mut f32).add(lo), hi - lo) };
			match w {
				LinearWeight::Q4K { raw, .. } => q4k_rows(raw, nb, &q8, lo, hi, out),
				LinearWeight::Q6K { raw, .. } => q6k_rows(raw, nb, &q8, lo, hi, out),
				_ => unreachable!("guarded by `kquant` above"),
			}
			g = starts[j] + hi;
		}
	});
	outs
}

/// [`apply_linear_group`] over a **row range** of each weight: job `(w, r0, r1)`
/// computes output rows `[r0, r1)` and returns `r1 - r0` values.
///
/// This is the CPU half of the decode row split: the GPU holds and computes
/// rows `[0, r0)` of every matrix, the CPU the rest, concurrently. A Q4_K/Q6_K
/// row range is a contiguous slice of the weight's bytes, so no weight data is
/// copied. Empty ranges are allowed and yield empty outputs. Each row is the
/// same `q4k_rows`/`q6k_rows` dot the unsplit group uses, so the CPU's rows are
/// bit-identical to what it would have produced without the split.
///
/// Non-K-quant weights fall back to a full [`apply_linear`] and a slice.
pub fn apply_linear_group_rows(
	jobs: &[(&LinearWeight, usize, usize)],
	x: &[f32],
	in_dim: usize,
) -> Vec<Vec<f32>> {
	apply_linear_group_rows_pre(jobs, x, in_dim, || {})
}

/// [`apply_linear_group_rows`] that runs `pre` on the calling thread once the
/// pool's workers are already computing (see `par_range_pre`). The decode row
/// split submits its GPU share there, so the submission overlaps the CPU's
/// share instead of delaying it. `pre` runs exactly once.
pub fn apply_linear_group_rows_pre<P: FnOnce() + Send>(
	jobs: &[(&LinearWeight, usize, usize)],
	x: &[f32],
	in_dim: usize,
	pre: P,
) -> Vec<Vec<f32>> {
	let kquant = jobs
		.iter()
		.all(|(w, _, _)| matches!(w, LinearWeight::Q4K { .. } | LinearWeight::Q6K { .. }));
	if !kquant {
		pre();
		return jobs
			.iter()
			.map(|(w, r0, r1)| {
				if r0 == r1 {
					Vec::new()
				} else {
					apply_linear(w, x, *r1, in_dim)[*r0..*r1].to_vec()
				}
			})
			.collect();
	}
	let nb = in_dim / QK_K;
	let total: usize = jobs.iter().map(|(_, r0, r1)| r1 - r0).sum();
	let mut outs: Vec<Vec<f32>> = jobs
		.iter()
		.map(|(_, r0, r1)| super::uninit_vec(r1 - r0))
		.collect();
	if total == 0 {
		pre();
		return outs;
	}
	let q8 = quantize_q8k(x, in_dim);
	let starts: Vec<usize> = jobs
		.iter()
		.scan(0usize, |acc, (_, r0, r1)| {
			let s = *acc;
			*acc += r1 - r0;
			Some(s)
		})
		.collect();
	let bytes: usize = jobs
		.iter()
		.map(|(w, r0, r1)| row_bytes_of(w, nb) * (r1 - r0))
		.sum();
	let ptrs: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
	par_range_pre(total, bytes / total, pre, |gs, ge| {
		let mut g = gs;
		while g < ge {
			// Last job whose start is <= g and that is non-empty.
			let mut j = match starts.binary_search(&g) {
				Ok(j) => j,
				Err(j) => j - 1,
			};
			while jobs[j].2 == jobs[j].1 {
				j += 1;
			}
			let (w, r0, r1) = jobs[j];
			let n = r1 - r0;
			let lo = g - starts[j];
			let hi = (ge - starts[j]).min(n);
			// SAFETY: rows [lo, hi) of job `j` are disjoint across chunks, and
			// `par_range` joins before `outs` is read.
			let out =
				unsafe { std::slice::from_raw_parts_mut((ptrs[j] as *mut f32).add(lo), hi - lo) };
			match w {
				LinearWeight::Q4K { raw, .. } => q4k_rows(raw, nb, &q8, r0 + lo, r0 + hi, out),
				LinearWeight::Q6K { raw, .. } => q6k_rows(raw, nb, &q8, r0 + lo, r0 + hi, out),
				_ => unreachable!("guarded by `kquant` above"),
			}
			g = starts[j] + hi;
		}
	});
	outs
}

/// [`apply_linear_group`] with the activation kept in **f32**: each output is
/// the exact `dequant(W) · x`, as the GPU's float GEMV computes it, rather than
/// a dot against `x` quantized to q8_K.
///
/// For tiny matrices only (decode's `ssm_beta` / `ssm_alpha`, 48 rows each):
/// the SSM gates they feed are sensitive to q8_K's error (computing them from
/// a q8_K activation moved `--forward 369`'s `DCR SUM` from -1035114 to
/// -1027859 (f32 reference -1035900)) but sending 48 rows to the GPU costs a
/// ~70 us round trip for ~20 us of CPU work. With an f32 activation they stay
/// on the CPU at the GPU's accuracy. It reads the weight bytes once like the
/// q8_K path; only the arithmetic is wider, which is why it is not the default
/// for the big matrices.
///
/// Q4_K takes an AVX2 kernel; Q6_K is dequantized row by row and dotted.
/// Other types are not accepted.
pub fn apply_linear_group_exact(
	jobs: &[(&LinearWeight, usize)],
	x: &[f32],
	in_dim: usize,
) -> Vec<Vec<f32>> {
	assert_eq!(x.len(), in_dim);
	let nb = in_dim / QK_K;
	let xsum: Vec<f32> = x.chunks(32).map(|c| c.iter().sum()).collect();
	let total: usize = jobs.iter().map(|(_, od)| od).sum();
	let mut outs: Vec<Vec<f32>> = jobs.iter().map(|(_, od)| super::uninit_vec(*od)).collect();
	if total == 0 {
		return outs;
	}
	let starts: Vec<usize> = jobs
		.iter()
		.scan(0usize, |acc, (_, od)| {
			let s = *acc;
			*acc += od;
			Some(s)
		})
		.collect();
	let bytes: usize = jobs.iter().map(|(w, od)| row_bytes_of(w, nb) * od).sum();
	let ptrs: Vec<usize> = outs.iter_mut().map(|v| v.as_mut_ptr() as usize).collect();
	par_range(total, bytes / total, |gs, ge| {
		for g in gs..ge {
			let j = match starts.binary_search(&g) {
				Ok(j) => (j..jobs.len()).find(|&k| jobs[k].1 > 0).unwrap(),
				Err(j) => j - 1,
			};
			let o = g - starts[j];
			let v = match jobs[j].0 {
				LinearWeight::Q4K { raw, .. } => {
					let rb = row_bytes_of(jobs[j].0, nb);
					super::gemv_q4k::q4k_dot_f32(&raw[o * rb..(o + 1) * rb], nb, x, &xsum)
				}
				w => dequant_row(w, o, in_dim)
					.iter()
					.zip(x)
					.map(|(a, b)| a * b)
					.sum(),
			};
			// SAFETY: each global row `g` maps to one (job, row) written by
			// exactly one chunk; `par_range` joins before `outs` is read.
			unsafe { *(ptrs[j] as *mut f32).add(o) = v };
		}
	});
	outs
}

/// `QWEN_EXACT_DECODE=1`: every K-quant decode matmul goes through
/// [`apply_linear_group_exact`]; an f32-activation reference for accuracy
/// A/Bs. Read once.
pub fn exact_decode_forced() -> bool {
	use std::sync::atomic::Ordering;
	match EXACT_DECODE.load(Ordering::Relaxed) {
		0 => {
			let on = std::env::var("QWEN_EXACT_DECODE").is_ok_and(|v| v != "0");
			EXACT_DECODE.store(1 + on as u8, Ordering::Relaxed);
			on
		}
		v => v == 2,
	}
}

/// 0 = not read yet, 1 = off, 2 = on.
static EXACT_DECODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Override `QWEN_EXACT_DECODE` at run time (`cli --kld-eval` uses the exact
/// path as its reference pass).
pub fn set_exact_decode(on: bool) {
	EXACT_DECODE.store(1 + on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// Row `o` of `w` (`in_dim` wide) as f32.
fn dequant_row(w: &LinearWeight, o: usize, in_dim: usize) -> Vec<f32> {
	let nb = in_dim / QK_K;
	let rb = row_bytes_of(w, nb);
	match w {
		LinearWeight::Q4K { raw, .. } => {
			super::dequant::dequantize_q4_k(&raw[o * rb..(o + 1) * rb], in_dim)
		}
		LinearWeight::Q6K { raw, .. } => {
			super::dequant::dequantize_q6_k(&raw[o * rb..(o + 1) * rb], in_dim)
		}
		_ => unreachable!("apply_linear_group_exact: only K-quant weights are routed here"),
	}
}

/// Bytes one weight row occupies for an `nb`-superblock input dimension.
fn row_bytes_of(w: &LinearWeight, nb: usize) -> usize {
	match w {
		LinearWeight::Q4K { .. } => nb * (2 + 2 + K_SCALE_SIZE + QK_K / 2),
		LinearWeight::Q6K { .. } => nb * q6k_block_bytes(),
		LinearWeight::F16(_) => nb * QK_K * 2,
		LinearWeight::F32(_) => nb * QK_K * 4,
	}
}

/// Fuse two `Q4_K`/`Q6_K` weights that share `in_dim` into a single
/// `[od_a + od_b, in_dim]` weight (row-contiguous concatenation of their
/// quantized blocks). Returns `None` when the types or `in_dim` differ, so the
/// caller can fall back to two separate matmuls. Used by the CPU decode path to
/// turn two matmuls that share an input (Wqkv+WqkvGate, FfnUp+FfnGate) into one
/// larger GEMV: the same trick llama.cpp uses to issue the ~13.5 GB weight
/// stream as fewer, bigger contiguous passes (better cache/ILP). The fused GEMV is numerically identical to two separate GEMVs (each
/// output row is independent), which `concat_matches_separate` verifies.
pub fn concat_weights(a: &LinearWeight, b: &LinearWeight) -> Option<LinearWeight> {
	use LinearWeight::*;
	match (a, b) {
		(
			Q4K {
				raw: ra,
				out_dim: oa,
				in_dim: ia,
			},
			Q4K {
				raw: rb,
				out_dim: ob,
				in_dim: ib,
			},
		) => {
			if ia != ib {
				return None;
			}
			let mut raw = ra.clone();
			raw.extend_from_slice(rb);
			Some(Q4K {
				raw,
				out_dim: oa + ob,
				in_dim: *ia,
			})
		}
		(
			Q6K {
				raw: ra,
				out_dim: oa,
				in_dim: ia,
			},
			Q6K {
				raw: rb,
				out_dim: ob,
				in_dim: ib,
			},
		) => {
			if ia != ib {
				return None;
			}
			let mut raw = ra.clone();
			raw.extend_from_slice(rb);
			Some(Q6K {
				raw,
				out_dim: oa + ob,
				in_dim: *ia,
			})
		}
		_ => None,
	}
}

/// Transpose a `[out_rows * seq_len]` buffer laid out `[o * seq + t]` into a
/// `[seq_len * out_rows]` buffer laid out `[t * out + o]`. Used by the batched
/// (seq_len > 1) matmul paths so the activation block stays seq-major and
/// per-position slices are contiguous.
#[inline]
fn transpose_seq(tmp: &[f32], y: &mut [f32], seq_len: usize, out_rows: usize) {
	for t in 0..seq_len {
		for o in 0..out_rows {
			y[t * out_rows + o] = tmp[o * seq_len + t];
		}
	}
}

/// Batched matmul dispatch over `seq_len` input rows (`x` is `[seq_len *
/// in_dim]`, row-major, one row per token). Returns `[seq_len * out_dim]` laid
/// out `[t * out_dim + o]` with `out[t * out_dim + o] = dot(w[o], x[t])`.
///
/// Each weight row is read **once** and reused across all `seq_len` positions
/// (vs `seq_len` separate GEMVs), so this is the kernel that makes prompt
/// prefill stream each big weight matrix a single time: the same win llama.cpp
/// gets from its batched prefill GEMM.
pub fn apply_linear_batch(
	w: &LinearWeight,
	x: &[f32],
	seq_len: usize,
	out_dim: usize,
	in_dim: usize,
) -> Vec<f32> {
	// The Q4_K/Q6_K arms fill every element; the F16/F32 arms below go through
	// `tmp` + `transpose_seq`, which also covers all of `y`.
	let mut y = super::uninit_vec(seq_len * out_dim);
	match w {
		LinearWeight::Q4K { raw, .. } => gemm_q4k(out_dim, in_dim, seq_len, raw, x, &mut y),
		LinearWeight::Q6K { raw, .. } => gemm_q6k(out_dim, in_dim, seq_len, raw, x, &mut y),
		LinearWeight::F16(v) => {
			let mut tmp = vec![0.0f32; seq_len * out_dim];
			gemm_f16(out_dim, in_dim, seq_len, v, x, &mut tmp);
			transpose_seq(&tmp, &mut y, seq_len, out_dim);
		}
		LinearWeight::F32(v) => {
			let mut tmp = vec![0.0f32; seq_len * out_dim];
			gemm(out_dim, in_dim, seq_len, v, x, &mut tmp);
			transpose_seq(&tmp, &mut y, seq_len, out_dim);
		}
	}
	y
}

/// Batched fused Q4_K dequant-GEMM over `seq_len` activation rows.
///
/// `input` is `[seq_len * in_dim]`; `out` is `[seq_len * out_rows]` with
/// `out[t * out_rows + o] = dot(w[o], input[t])`. Each output row's `Q4_K` bytes
/// are dequantized once and the integer dot is accumulated against every input
/// row's (independently quantized) `q8_K` activation. Parallelized over output
/// rows like [`gemm`].
// Q4_K per-block scale/min unpack (mirrors the `utmp` dance inside
// `q4k_dot_q8k_avx2`). Hoisted out of the position loop so each block's 8
// sub-scales + 8 mins are computed ONCE per weight row and reused across all
// `seq_len` positions: the single-row kernel re-does this per (row, position).
#[inline]
fn unpack_q4_scales(scales: &[u8]) -> ([i16; 8], [i16; 8]) {
	let kmask1: u32 = 0x3f3f_3f3f;
	let kmask2: u32 = 0x0f0f_0f0f;
	let kmask3: u32 = 0x0303_0303;
	let mut utmp = [0u32; 4];
	utmp[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
	utmp[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
	utmp[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
	utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
	let uaux = utmp[1] & kmask1;
	utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
	utmp[2] = uaux;
	utmp[0] &= kmask1;
	let lo0 = utmp[0].to_le_bytes();
	let lo1 = utmp[1].to_le_bytes();
	let hi0 = utmp[2].to_le_bytes();
	let hi1 = utmp[3].to_le_bytes();
	let mut s = [0i16; 8];
	let mut m = [0i16; 8];
	for k in 0..4 {
		s[k] = lo0[k] as i16;
		s[4 + k] = lo1[k] as i16;
		m[k] = hi0[k] as i16;
		m[4 + k] = hi1[k] as i16;
	}
	(s, m)
}

/// Batched Q4_K×q8_K dequant-GEMM for ONE weight row across all `seq_len`
/// positions, AVX2. Numerically identical to looping `q4k_dot_q8k_avx2` over
/// every position, but each block's sub-scale/min unpack runs once instead of
/// `seq_len` times. This is the prefill win: `gemm_q4k` otherwise re-unpacks the
/// (position-independent) weight scales for every token in the chunk.
///
/// The **q4 nibble split is hoisted the same way** (`q4v`), together with the
/// sub-scale broadcasts (`scv`): both depend only on the weight, and they were
/// being redone once per *(block, position)*; `gemm_q6k_row_all_avx2` next door
/// had already been given exactly this treatment.
///
/// `acc`/`acc_m` stay **scalar** `f32` arrays on purpose. Widening them to one
/// `__m256`+`__m128` per position (which would make this bit-identical to the
/// GEMV instead of merely equivalent, and would drop the per-block
/// `hsum_float_8`) was measured and is **slower**: at `seq = 2048` that scratch
/// is 96 KB and no longer fits L1, so every block read-modify-writes it out of
/// L2. 735 -> 624 GFLOP/s. The reduction is not the thing to fix here; the
/// accumulator working set has to stay L1-resident.
///
/// Superseded by [`gemm_q4k_rows_vacc_avx2`] once position tiling made that
/// scratch 1 KB; kept as the `QWEN_GEMM_HSUM=1` A/B.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q4k_row_all_avx2(
	row: &[u8],
	nb: usize,
	q8s: &[Q8K],
	qs8: &[*const i8],
	bsums: &[*const i16],
	acc: &mut [f32],
	acc_m: &mut [f32],
	out: &mut [f32],
) {
	use std::arch::x86_64::*;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let m4 = _mm256_set1_epi8(0x0F_i8);
	let t = q8s.len();
	// `acc`/`acc_m`/`qs8`/`bsums` are caller-owned scratch reused across every
	// row of a chunk. They used to be four `Vec`s allocated *per output row*,
	// i.e. millions of allocations per prefill chunk.
	acc[..t].fill(0.0);
	acc_m[..t].fill(0.0);
	for i in 0..nb {
		let rb = &row[i * blk..i * blk + blk];
		let ws = fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]]));
		let wmin = fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
		let (sc, mn) = unpack_q4_scales(&rb[4..4 + K_SCALE_SIZE]);
		let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);
		// weight-only: does not belong inside the position loop
		let mins128 = _mm_set_epi16(mn[7], mn[6], mn[5], mn[4], mn[3], mn[2], mn[1], mn[0]);
		let mut q4v = [_mm256_setzero_si256(); 8];
		let mut scv = [_mm256_setzero_si256(); 8];
		for j in 0..4 {
			let q4bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
			q4v[2 * j] = _mm256_and_si256(q4bits, m4);
			q4v[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
			scv[2 * j] = _mm256_set1_epi16(sc[2 * j]);
			scv[2 * j + 1] = _mm256_set1_epi16(sc[2 * j + 1]);
		}
		for p in 0..t {
			let d = q8s[p].ds[i] * ws;
			let dmin = -q8s[p].ds[i] * wmin;
			let mut sumi = _mm256_setzero_si256();
			for j in 0..4 {
				let q8base = (i * QK_K + j * 64) as isize;
				let q8l = _mm256_loadu_si256(qs8[p].offset(q8base) as *const __m256i);
				let p16l = _mm256_madd_epi16(scv[2 * j], _mm256_maddubs_epi16(q4v[2 * j], q8l));
				let q8h = _mm256_loadu_si256(qs8[p].offset(q8base + 32) as *const __m256i);
				let p16h =
					_mm256_madd_epi16(scv[2 * j + 1], _mm256_maddubs_epi16(q4v[2 * j + 1], q8h));
				sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
			}
			let q8sums = _mm256_loadu_si256(bsums[p].add(i * (QK_K / 16)) as *const __m256i);
			let q8s_h = _mm_hadd_epi16(
				_mm256_extracti128_si256(q8sums, 0),
				_mm256_extracti128_si256(q8sums, 1),
			);
			let prod = _mm_madd_epi16(mins128, q8s_h);
			let sumi_f = hsum8_tree(_mm256_cvtepi32_ps(sumi));
			let prod_f = hsum4_tree(_mm_cvtepi32_ps(prod));
			acc[p] += d * sumi_f;
			acc_m[p] += dmin * prod_f;
		}
	}
	for p in 0..t {
		out[p] = acc[p] + acc_m[p];
	}
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line64([u8; 64]);

/// One position tile of q8_K activations repacked **block-major** for the CPU
/// prefill GEMMs: block `i`, position `p` is entry `i * pn + p` of every array,
/// so the row kernels' inner position loop walks each array sequentially, and
/// `qs` is 64-byte aligned. Each `Q8K::qs` is its own heap `Vec` with 16-byte
/// alignment, so in some processes every other 32-byte load split a cache
/// line; `gemmbench` was bimodal across runs with nothing else different.
#[derive(Default)]
pub(crate) struct Q8Tile {
	qs: Vec<Line64>,
	ds: Vec<f32>,
	/// For the Q6_K GEMM.
	bsums: Vec<i16>,
	/// The f32 min operand per (block, position): `ds * (bsums[2k] +
	/// bsums[2k + 1])`, 8 of them, for [`gemm_q4k_rows_vacc_avx2`]. Both are
	/// always packed, so one packed input serves Q4_K and Q6_K matmuls alike.
	bmin: Vec<f32>,
	pn: usize,
}

impl Q8Tile {
	fn pack(&mut self, q8s: &[Q8K], nb: usize) {
		let pn = q8s.len();
		self.pn = pn;
		let lines = nb * pn * (QK_K / 64);
		if self.qs.len() < lines {
			self.qs.resize(lines, Line64([0; 64]));
		}
		self.ds.clear();
		self.bsums.clear();
		self.bmin.clear();
		let qs = self.qs.as_mut_ptr() as *mut i8;
		for i in 0..nb {
			for (p, q) in q8s.iter().enumerate() {
				unsafe {
					std::ptr::copy_nonoverlapping(
						q.qs.as_ptr().add(i * QK_K),
						qs.add((i * pn + p) * QK_K),
						QK_K,
					);
				}
				let d = q.ds[i];
				self.ds.push(d);
				let bs = &q.bsums[i * 16..i * 16 + 16];
				self.bsums.extend_from_slice(bs);
				for k in 0..8 {
					self.bmin
						.push(d * (bs[2 * k] as i32 + bs[2 * k + 1] as i32) as f32);
				}
			}
		}
	}

	fn qs(&self) -> *const i8 {
		self.qs.as_ptr() as *const i8
	}
}

type Q8Arc = std::sync::Arc<Vec<Q8K>>;
type TileKey = (Q8Arc, usize, usize);

/// Recently packed tiles, so matmuls that share an input (`FfnUp` / `FfnGate`,
/// `Wqkv` / its gate / `ssm_beta` / `ssm_alpha`, `Wq` / `Wk` / `Wv`) pack it
/// once, plus retired tiles whose buffers the next pack reuses. Keyed on the
/// input's `Arc` identity: an entry holds a clone, so the allocation cannot be
/// freed and its address reused by different data while it is cached.
struct TileCache {
	live: Vec<(TileKey, std::sync::Arc<Vec<Q8Tile>>)>,
	spare: Vec<Q8Tile>,
}

/// Inputs in flight at once: the prefill pipeline alternates two halves.
const TILE_CACHE_ENTRIES: usize = 2;

fn tile_cache() -> &'static std::sync::Mutex<TileCache> {
	static C: std::sync::OnceLock<std::sync::Mutex<TileCache>> = std::sync::OnceLock::new();
	C.get_or_init(|| {
		std::sync::Mutex::new(TileCache {
			live: Vec::new(),
			spare: Vec::new(),
		})
	})
}

/// Every `pt`-position tile of `q8s`, packed across the pool **once per
/// input** and shared by all row chunks. Packing per chunk instead cost more
/// than the layout saved where the prefill split leaves the CPU ~4000 of 17408
/// rows: 21-row chunks re-copied a 160 KB tile per 3.4 M multiply-adds.
///
/// Packed tiles are cached ([`TileCache`]): per call, a 2048-position input is
/// 35 MB of fresh tiles (FfnDown's 557 KB tiles are past the Windows heap's
/// `VirtualAlloc` threshold, so every call also page-faulted them in), and the
/// matmuls of a shared-input group each repacked the same input.
fn pack_tiles(q8s: &Q8Arc, nb: usize, pt: usize) -> std::sync::Arc<Vec<Q8Tile>> {
	let same = |k: &TileKey| std::sync::Arc::ptr_eq(&k.0, q8s) && (k.1, k.2) == (nb, pt);
	let n = q8s.len().div_ceil(pt);
	let mut tiles: Vec<Q8Tile> = {
		let mut c = tile_cache().lock().unwrap();
		if let Some((_, t)) = c.live.iter().find(|(k, _)| same(k)) {
			return t.clone();
		}
		let take = n.min(c.spare.len());
		let at = c.spare.len() - take;
		let mut v: Vec<Q8Tile> = c.spare.drain(at..).collect();
		v.extend(std::iter::repeat_with(Q8Tile::default).take(n - take));
		v
	};
	let addr = tiles.as_mut_ptr() as usize;
	let q8a = q8s.as_ptr() as usize;
	let len = q8s.len();
	super::par_for(n, move |k| {
		// SAFETY: each `k` mutates exactly one distinct slot; `q8s` is only read.
		let q8s = unsafe { std::slice::from_raw_parts(q8a as *const Q8K, len) };
		let tile = unsafe { &mut *(addr as *mut Q8Tile).add(k) };
		let p0 = k * pt;
		tile.pack(&q8s[p0..(p0 + pt).min(len)], nb);
	});
	let tiles = std::sync::Arc::new(tiles);
	let mut c = tile_cache().lock().unwrap();
	c.live.insert(0, ((q8s.clone(), nb, pt), tiles.clone()));
	while c.live.len() > TILE_CACHE_ENTRIES {
		let (_, old) = c.live.pop().unwrap();
		// Still borrowed by a running GEMM: let it go with that GEMM instead.
		if let Ok(v) = std::sync::Arc::try_unwrap(old) {
			c.spare.extend(v);
			// Two 2048-position inputs' worth (FfnDown's tiles are 557 KB).
			c.spare.truncate(128);
		}
	}
	tiles
}

/// [`gemm_q4k_row_all_avx2`] with the per-(block, position) reductions taken
/// out of the inner loop: the main term goes into one `__m256` accumulator per
/// position (`fma(ds * d_w, cvt(sumi), acc)`) and is summed horizontally once
/// per row instead of once per block. The min term becomes an f32 dot of
/// length `nb * 8`: `-d_min * m_k` per (row, block) against
/// `Q8Tile::bmin` per (position, block), one FMA per block-position into
/// the same accumulator; replacing a `bsums` load, `hadd_epi16`, `madd`, two
/// `cvt` and two horizontal sums.
///
/// The accumulators were tried as vectors once before and lost (735 -> 624
/// GFLOP/s), but that was at the untiled `seq = 2048`, where they were 96 KB;
/// a 32-position tile keeps them at 1 KB.
///
/// It runs `rows` consecutive weight rows at once, **block-outer**: block `i`
/// of every row is applied before block `i + 1` of any, so the tile's block-`i`
/// slice (`pn * 256` bytes, 8 KB at 32 positions) is read from L1 by every row
/// instead of the whole tile (160-560 KB) streaming from L2 once per row. That
/// L2 stream was the limiter (`gemmbench` FfnUp seq 2048: 2376 ->
/// 3036 GFLOP/s at 8 rows; the E-cores share one L2 per 4-core cluster). Each
/// `(row, position)` accumulator sees the same operations in the same order
/// whatever `rows` is: bit-identical. Row `r`'s results go to `out[r * ld..]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q4k_rows_vacc_avx2(
	w: &[u8],
	rows: usize,
	nb: usize,
	tile: &Q8Tile,
	accv: &mut [std::arch::x86_64::__m256],
	out: &mut [f32],
	ld: usize,
) {
	use std::arch::x86_64::*;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let row_bytes = nb * blk;
	let m4 = _mm256_set1_epi8(0x0F_i8);
	let t = tile.pn;
	let tq = tile.qs();
	for a in accv[..rows * t].iter_mut() {
		*a = _mm256_setzero_ps();
	}
	for i in 0..nb {
		let bm = tile.bmin.as_ptr().add(i * t * 8);
		let dsi = tile.ds.as_ptr().add(i * t);
		for r in 0..rows {
			let rb = &w[r * row_bytes + i * blk..r * row_bytes + i * blk + blk];
			let wsv = _mm256_set1_ps(fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]])));
			let wmin = fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
			let (sc, mn) = unpack_q4_scales(&rb[4..4 + K_SCALE_SIZE]);
			let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);
			let mnv = _mm256_mul_ps(
				_mm256_set1_ps(-wmin),
				_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_set_epi16(
					mn[7], mn[6], mn[5], mn[4], mn[3], mn[2], mn[1], mn[0],
				))),
			);
			let mut q4v = [_mm256_setzero_si256(); 8];
			let mut scv = [_mm256_setzero_si256(); 8];
			for j in 0..4 {
				let q4bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
				q4v[2 * j] = _mm256_and_si256(q4bits, m4);
				q4v[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
				scv[2 * j] = _mm256_set1_epi16(sc[2 * j]);
				scv[2 * j + 1] = _mm256_set1_epi16(sc[2 * j + 1]);
			}
			let acc = accv.as_mut_ptr().add(r * t);
			for p in 0..t {
				let q8b = tq.add((i * t + p) * QK_K);
				let mut s0 = _mm256_setzero_si256();
				let mut s1 = _mm256_setzero_si256();
				for j in 0..4 {
					let q8l = _mm256_loadu_si256(q8b.add(j * 64) as *const __m256i);
					let q8h = _mm256_loadu_si256(q8b.add(j * 64 + 32) as *const __m256i);
					s0 = _mm256_add_epi32(
						s0,
						_mm256_madd_epi16(scv[2 * j], _mm256_maddubs_epi16(q4v[2 * j], q8l)),
					);
					s1 = _mm256_add_epi32(
						s1,
						_mm256_madd_epi16(
							scv[2 * j + 1],
							_mm256_maddubs_epi16(q4v[2 * j + 1], q8h),
						),
					);
				}
				let sumf = _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1));
				let dv = _mm256_mul_ps(_mm256_broadcast_ss(&*dsi.add(p)), wsv);
				let a = _mm256_fmadd_ps(dv, sumf, *acc.add(p));
				*acc.add(p) = _mm256_fmadd_ps(mnv, _mm256_loadu_ps(bm.add(p * 8)), a);
			}
		}
	}
	for r in 0..rows {
		for p in 0..t {
			out[r * ld + p] = hsum8_tree(accv[r * t + p]);
		}
	}
}

/// [`gemm_q4k_rows_vacc_avx2`] with the integer dot on AVX-VNNI, two
/// sub-blocks per scale multiply. Per pair of 32-weight sub-blocks `a`, `b`,
/// `vpdpbusd` gives each sub-block's 4-product lane sums as i32 (|sum| <= 4 *
/// 15 * 128, so they fit an i16), `packs` narrows both into one vector (per
/// 128-bit half, 4 lanes of `a` then 4 of `b`) and one `vpdpwssd` applies
/// both 6-bit scales and accumulates. 4 uops (3 multiplies) per pair instead
/// of `maddubs` + `madd` + `add` twice (6 uops, 4 multiplies); `gemmbench`
/// FfnUp seq 2048: 3030 -> 3150-3280 GFLOP/s.
///
/// This is not the rejected `QWEN_AVXVNNI` form, which widened the scales with
/// `cvtepi16_epi32` + `mullo_epi32` and so cost more than it saved. The integer
/// total is exactly AVX2's, but `packs` spreads it over the 8 lanes
/// differently, so the f32 epilogue rounds differently: not bit-identical.
/// (A lane-preserving pairing, `(b << 16) + a`, needs `a >= 0` (a negative
/// `a` borrows from `b`) i.e. a `q8 + 128` tile and a weight-only
/// correction; it was bit-identical but no faster than AVX2.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn gemm_q4k_rows_vnni(
	w: &[u8],
	rows: usize,
	nb: usize,
	tile: &Q8Tile,
	accv: &mut [std::arch::x86_64::__m256],
	out: &mut [f32],
	ld: usize,
) {
	use std::arch::x86_64::*;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let row_bytes = nb * blk;
	let m4 = _mm256_set1_epi8(0x0F_i8);
	let t = tile.pn;
	let tq = tile.qs();
	for a in accv[..rows * t].iter_mut() {
		*a = _mm256_setzero_ps();
	}
	let z = _mm256_setzero_si256();
	for i in 0..nb {
		let bm = tile.bmin.as_ptr().add(i * t * 8);
		let dsi = tile.ds.as_ptr().add(i * t);
		for r in 0..rows {
			let rb = &w[r * row_bytes + i * blk..r * row_bytes + i * blk + blk];
			let wsv = _mm256_set1_ps(fp16_to_f32(u16::from_le_bytes([rb[0], rb[1]])));
			let wmin = fp16_to_f32(u16::from_le_bytes([rb[2], rb[3]]));
			let (sc, mn) = unpack_q4_scales(&rb[4..4 + K_SCALE_SIZE]);
			let qs = rb.as_ptr().add(4 + K_SCALE_SIZE);
			let mnv = _mm256_mul_ps(
				_mm256_set1_ps(-wmin),
				_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_set_epi16(
					mn[7], mn[6], mn[5], mn[4], mn[3], mn[2], mn[1], mn[0],
				))),
			);
			let mut q4v = [_mm256_setzero_si256(); 8];
			let mut scp = [_mm256_setzero_si256(); 4];
			for j in 0..4 {
				let q4bits = _mm256_loadu_si256(qs.add(j * 32) as *const __m256i);
				q4v[2 * j] = _mm256_and_si256(q4bits, m4);
				q4v[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
				scp[j] = _mm256_set_epi64x(
					(sc[2 * j + 1] as u16 as i64) * 0x0001_0001_0001_0001,
					(sc[2 * j] as u16 as i64) * 0x0001_0001_0001_0001,
					(sc[2 * j + 1] as u16 as i64) * 0x0001_0001_0001_0001,
					(sc[2 * j] as u16 as i64) * 0x0001_0001_0001_0001,
				);
			}
			let acc = accv.as_mut_ptr().add(r * t);
			for p in 0..t {
				let q8b = tq.add((i * t + p) * QK_K);
				let mut s0 = z;
				let mut s1 = z;
				for j in 0..4 {
					let q8l = _mm256_loadu_si256(q8b.add(j * 64) as *const __m256i);
					let q8h = _mm256_loadu_si256(q8b.add(j * 64 + 32) as *const __m256i);
					let lo = _mm256_dpbusd_avx_epi32(z, q4v[2 * j], q8l);
					let hi = _mm256_dpbusd_avx_epi32(z, q4v[2 * j + 1], q8h);
					let pr = _mm256_packs_epi32(lo, hi);
					if j & 1 == 0 {
						s0 = _mm256_dpwssd_avx_epi32(s0, pr, scp[j]);
					} else {
						s1 = _mm256_dpwssd_avx_epi32(s1, pr, scp[j]);
					}
				}
				let sumf = _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1));
				let dv = _mm256_mul_ps(_mm256_broadcast_ss(&*dsi.add(p)), wsv);
				let a = _mm256_fmadd_ps(dv, sumf, *acc.add(p));
				*acc.add(p) = _mm256_fmadd_ps(mnv, _mm256_loadu_ps(bm.add(p * 8)), a);
			}
		}
	}
	for r in 0..rows {
		for p in 0..t {
			out[r * ld + p] = hsum8_tree(accv[r * t + p]);
		}
	}
}

/// The CPU prefill GEMMs use [`gemm_q4k_rows_vnni`] / [`gemm_q6k_rows_vnni`]
/// where the CPU has AVX-VNNI; `QWEN_GEMM_VNNI=0` is the A/B.
pub(crate) fn gemm_vnni() -> bool {
	static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*V.get_or_init(|| {
		is_x86_feature_detected!("avxvnni")
			&& std::env::var("QWEN_GEMM_VNNI").map_or(true, |v| v != "0")
	})
}

/// Rows per block-outer group of the CPU Q4_K prefill GEMM
/// ([`gemm_q4k_rows_vacc_avx2`], and Q6_K's); `QWEN_GEMM_RGROUP`, 1 = the row kernel.
pub(crate) fn gemm_rgroup() -> usize {
	static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*P.get_or_init(|| {
		std::env::var("QWEN_GEMM_RGROUP")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.filter(|v| *v > 0)
			.unwrap_or(8)
	})
}

/// Batched Q6_K×q8_K dequant-GEMM for ONE weight row across all `seq_len`
/// positions, AVX2. The 6-bit weight reconstruction (`q4_0..3`) depends only on
/// the weight row, so it is built once per block and reused for every position
/// (the single-row `q6k_dot_q8k_avx2` rebuilds it per (row, position)). Scale
/// vectors and the weight scale factor are likewise hoisted. Numerically
/// identical to looping `q6k_dot_q8k_avx2` over positions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_q6k_row_all_avx2(
	w: &[u8],
	rows: usize,
	nb: usize,
	tile: &Q8Tile,
	accv: &mut [std::arch::x86_64::__m256],
	out: &mut [f32],
	ld: usize,
) {
	use std::arch::x86_64::*;
	let m3 = _mm256_set1_epi8(3);
	let m12 = _mm256_set1_epi8(12);
	let m15 = _mm256_set1_epi8(15);
	let m48 = _mm256_set1_epi8(48);
	let m192 = _mm256_set1_epi8(-64i8);
	let t = tile.pn;

	// Block-outer / position-inner, so the 6-bit weight unpack (the `q4h_*` /
	// `q4_*` mask-shift-or chain, ~20 AVX2 ops per 128 weights) runs **once per
	// block** and is reused across all `seq_len` positions. The old order was
	// position-outer, which re-decoded the same weight bytes for every token in
	// the chunk: that is why the Q6_K prefill GEMM ran at 783 GFLOP/s against
	// Q4_K's 1117 on the same shape.
	//
	// `accv` (one `__m256` per position, caller-owned scratch) preserves the
	// *vector* accumulation order of the old form, so results stay bit-identical.
	//
	// `rows` consecutive weight rows run block-outer (see
	// `gemm_q4k_rows_vacc_avx2`): the tile's block-`i` slice stays in L1
	// for every row instead of the whole tile streaming from L2 per row.
	// Row `r`'s results go to `out[r * ld..][..t]`.
	let row_bytes = nb * 210;
	for a in accv[..rows * t].iter_mut() {
		*a = _mm256_setzero_ps();
	}
	for i in 0..nb {
		for r in 0..rows {
			let accv = &mut accv[r * t..(r + 1) * t];
			let rb = &w[r * row_bytes + i * 210..r * row_bytes + i * 210 + 210];
			let scales128 = _mm_loadu_si128(rb.as_ptr().add(192) as *const __m128i);
			let scales_16 = _mm256_cvtepi8_epi16(scales128);
			let dw = fp16_to_f32(u16::from_le_bytes([rb[208], rb[209]]));

			// Decode this block's 256 six-bit weights and its 8 sub-scales once.
			let mut q4v = [_mm256_setzero_si256(); 8];
			let mut scv = [_mm256_setzero_si256(); 8];
			let mut ql = rb.as_ptr();
			let mut qh = rb.as_ptr().add(128);
			let mut is = 0usize;
			for j in 0..(QK_K / 128) {
				let q4bits1 = _mm256_loadu_si256(ql as *const __m256i);
				ql = ql.add(32);
				let q4bits2 = _mm256_loadu_si256(ql as *const __m256i);
				ql = ql.add(32);
				let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
				qh = qh.add(32);
				let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
				let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
				let q4h_2 = _mm256_and_si256(q4bits_h, m48);
				let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, m192), 2);
				q4v[j * 4] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
				q4v[j * 4 + 1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
				q4v[j * 4 + 2] =
					_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
				q4v[j * 4 + 3] =
					_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
				for m in 0..4 {
					scv[j * 4 + m] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
						scales128,
						get_scale_shuffle_k6(is + m),
					));
				}
				is += 4;
			}

			for p in 0..t {
				let d = tile.ds[i * t + p] * dw;
				let mut q8p = tile.qs().add((i * t + p) * QK_K);
				let mut sumi = _mm256_setzero_si256();
				// Same (weight chunk, q8 chunk) pairing and the same add tree as the
				// position-outer form, so `sumi` matches exactly.
				let mut n = 0usize;
				while n < 8 {
					let q8_0 = _mm256_loadu_si256(q8p as *const __m256i);
					q8p = q8p.add(32);
					let q8_1 = _mm256_loadu_si256(q8p as *const __m256i);
					q8p = q8p.add(32);
					let q8_2 = _mm256_loadu_si256(q8p as *const __m256i);
					q8p = q8p.add(32);
					let q8_3 = _mm256_loadu_si256(q8p as *const __m256i);
					q8p = q8p.add(32);
					let p16_0 = _mm256_madd_epi16(scv[n], _mm256_maddubs_epi16(q4v[n], q8_0));
					let p16_1 =
						_mm256_madd_epi16(scv[n + 1], _mm256_maddubs_epi16(q4v[n + 1], q8_1));
					let p16_2 =
						_mm256_madd_epi16(scv[n + 2], _mm256_maddubs_epi16(q4v[n + 2], q8_2));
					let p16_3 =
						_mm256_madd_epi16(scv[n + 3], _mm256_maddubs_epi16(q4v[n + 3], q8_3));
					sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
					sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
					n += 4;
				}
				// Integer, not the f32 min-dot of `gemm_q4k_rows_vacc_avx2`: that form
				// (`ds * bsums[k]` against `-32 * dw * sc_k`, two FMAs) measured
				// ~3 % slower here (2015 vs 2075 GFLOP/s).
				let q8sums =
					_mm256_loadu_si256(tile.bsums.as_ptr().add((i * t + p) * 16) as *const __m256i);
				let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
				sumi = _mm256_sub_epi32(sumi, q8sclsub);
				accv[p] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv[p]);
			}
		}
	}
	for r in 0..rows {
		for p in 0..t {
			out[r * ld + p] = hsum_float_8(accv[r * t + p]);
		}
	}
}

/// [`gemm_q6k_row_all_avx2`] on AVX-VNNI with the pairing of
/// [`gemm_q4k_rows_vnni`] (lane sums |sum| <= 4 * 63 * 128 still fit an i16).
/// The per-16 scales are the same within each 128-bit half of `scv`, which is
/// what lets one vector serve `packs`' order. `gemmbench` FfnDown seq 2048:
/// ~2580 -> 2670-2740 GFLOP/s. Not bit-identical, for the same reason.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn gemm_q6k_rows_vnni(
	w: &[u8],
	rows: usize,
	nb: usize,
	tile: &Q8Tile,
	accv: &mut [std::arch::x86_64::__m256],
	out: &mut [f32],
	ld: usize,
) {
	use std::arch::x86_64::*;
	let m3 = _mm256_set1_epi8(3);
	let m12 = _mm256_set1_epi8(12);
	let m15 = _mm256_set1_epi8(15);
	let m48 = _mm256_set1_epi8(48);
	let m192 = _mm256_set1_epi8(-64i8);
	let t = tile.pn;
	let row_bytes = nb * 210;
	let z = _mm256_setzero_si256();
	for a in accv[..rows * t].iter_mut() {
		*a = _mm256_setzero_ps();
	}
	for i in 0..nb {
		for r in 0..rows {
			let accv = &mut accv[r * t..(r + 1) * t];
			let rb = &w[r * row_bytes + i * 210..r * row_bytes + i * 210 + 210];
			let scales128 = _mm_loadu_si128(rb.as_ptr().add(192) as *const __m128i);
			let scales_16 = _mm256_cvtepi8_epi16(scales128);
			let dw = fp16_to_f32(u16::from_le_bytes([rb[208], rb[209]]));

			let mut q4v = [_mm256_setzero_si256(); 8];
			let mut scp = [_mm256_setzero_si256(); 4];
			let mut ql = rb.as_ptr();
			let mut qh = rb.as_ptr().add(128);
			let mut is = 0usize;
			for j in 0..(QK_K / 128) {
				let q4bits1 = _mm256_loadu_si256(ql as *const __m256i);
				ql = ql.add(32);
				let q4bits2 = _mm256_loadu_si256(ql as *const __m256i);
				ql = ql.add(32);
				let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
				qh = qh.add(32);
				let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
				let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
				let q4h_2 = _mm256_and_si256(q4bits_h, m48);
				let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, m192), 2);
				q4v[j * 4] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
				q4v[j * 4 + 1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
				q4v[j * 4 + 2] =
					_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
				q4v[j * 4 + 3] =
					_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
				let mut scv = [_mm256_setzero_si256(); 4];
				for m in 0..4 {
					scv[m] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
						scales128,
						get_scale_shuffle_k6(is + m),
					));
				}
				// Every i16 of a 128-bit half of `scv[m]` is the same scale, and
				// `packs(a, b)` puts 4 of `a`'s lanes, then 4 of `b`'s, in each half.
				scp[j * 2] = _mm256_unpacklo_epi64(scv[0], scv[1]);
				scp[j * 2 + 1] = _mm256_unpacklo_epi64(scv[2], scv[3]);
				is += 4;
			}
			for p in 0..t {
				let d = tile.ds[i * t + p] * dw;
				let q8p = tile.qs().add((i * t + p) * QK_K);
				let mut s0 = z;
				let mut s1 = z;
				for k in 0..4 {
					let q8a = _mm256_loadu_si256(q8p.add(k * 64) as *const __m256i);
					let q8b = _mm256_loadu_si256(q8p.add(k * 64 + 32) as *const __m256i);
					let lo = _mm256_dpbusd_avx_epi32(z, q4v[2 * k], q8a);
					let hi = _mm256_dpbusd_avx_epi32(z, q4v[2 * k + 1], q8b);
					let pr = _mm256_packs_epi32(lo, hi);
					if k & 1 == 0 {
						s0 = _mm256_dpwssd_avx_epi32(s0, pr, scp[k]);
					} else {
						s1 = _mm256_dpwssd_avx_epi32(s1, pr, scp[k]);
					}
				}
				let q8sums =
					_mm256_loadu_si256(tile.bsums.as_ptr().add((i * t + p) * 16) as *const __m256i);
				let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
				let sumi = _mm256_sub_epi32(_mm256_add_epi32(s0, s1), q8sclsub);
				accv[p] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accv[p]);
			}
		}
	}
	for r in 0..rows {
		for p in 0..t {
			out[r * ld + p] = hsum_float_8(accv[r * t + p]);
		}
	}
}

pub fn gemm_q4k(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q4k: &[u8],
	input: &[f32],
	out: &mut [f32],
) {
	assert_eq!(out.len(), seq_len * out_rows);
	gemm_q4k_strided(out_rows, in_dim, seq_len, q4k, input, out, out_rows, 0)
}

/// [`gemm_q4k`] writing row `o`, position `t` to `out[t * ld + col0 + o]`, so
/// the result can land as a column block of a wider `[seq x ld]` matrix (the
/// CPU half of the prefill row split) with no copy afterwards.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4k_strided(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q4k: &[u8],
	input: &[f32],
	out: &mut [f32],
	ld: usize,
	col0: usize,
) {
	assert_eq!(input.len(), seq_len * in_dim);
	let q8s = std::sync::Arc::new(par_quantize_q8k(input, seq_len, in_dim));
	gemm_q4k_strided_q8(out_rows, in_dim, seq_len, q4k, &q8s, out, ld, col0)
}

/// [`gemm_q4k_strided`] on an activation already quantized by
/// [`par_quantize_q8k`], so one quantization can serve several matrices that
/// read the same input (and the GPU half of the prefill split).
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4k_strided_q8(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q4k: &[u8],
	q8s: &std::sync::Arc<Vec<Q8K>>,
	out: &mut [f32],
	ld: usize,
	col0: usize,
) {
	assert_eq!(q8s.len(), seq_len);
	let nb = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	assert_eq!(q4k.len(), out_rows * nb * blk);
	let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
	let row_bytes = nb * blk;
	let q4k_addr = q4k.as_ptr() as usize;
	let q4k_len = q4k.len();
	let pt = gemm_ptile();
	let vacc = use_avx2 && q4k_vacc();
	let vnni = vacc && gemm_vnni();
	// `bmin` only: the min term no longer reads `bsums`.
	let tiles = if vacc {
		pack_tiles(q8s, nb, pt)
	} else {
		std::sync::Arc::new(Vec::new())
	};
	par_gemm_collect(out_rows, seq_len, out, ld, col0, gemm_rgroup(), {
		let q8s = q8s.clone();
		let tiles = tiles.clone();
		move |rs, re, chunk| {
			let q4k = unsafe { std::slice::from_raw_parts(q4k_addr as *const u8, q4k_len) };
			// Per-chunk, not per-row (see `gemm_q4k_row_all_avx2`).
			let qs8: Vec<*const i8> = q8s.iter().map(|q| q.qs.as_ptr()).collect();
			let bsums: Vec<*const i16> = q8s.iter().map(|q| q.bsums.as_ptr()).collect();
			let mut acc = vec![0.0f32; seq_len];
			let mut acc_m = vec![0.0f32; seq_len];
			// Position tile OUTSIDE the row loop: the activation tile is loaded
			// once and reused by every row in this range (see `gemm_ptile`).
			let rg = gemm_rgroup();
			let mut accv = Vec::new();
			if vacc {
				accv =
					vec![unsafe { std::arch::x86_64::_mm256_setzero_ps() }; pt.min(seq_len) * rg];
			}
			let mut p0 = 0usize;
			while p0 < seq_len {
				let pn = pt.min(seq_len - p0);
				let pr = p0..p0 + pn;
				if vacc {
					let mut i = 0;
					while i < re - rs {
						let rows = rg.min(re - rs - i);
						let o = rs + i;
						let kern = if vnni {
							gemm_q4k_rows_vnni
						} else {
							gemm_q4k_rows_vacc_avx2
						};
						unsafe {
							kern(
								&q4k[o * row_bytes..(o + rows) * row_bytes],
								rows,
								nb,
								&tiles[p0 / pt],
								&mut accv,
								&mut chunk[i * seq_len + p0..],
								seq_len,
							)
						}
						i += rows;
					}
					p0 += pn;
					continue;
				}
				for i in 0..(re - rs) {
					let o = rs + i;
					let wrow = &q4k[o * row_bytes..(o + 1) * row_bytes];
					let out_row = &mut chunk[i * seq_len + p0..i * seq_len + p0 + pn];
					if use_avx2 {
						unsafe {
							gemm_q4k_row_all_avx2(
								wrow,
								nb,
								&q8s[pr.clone()],
								&qs8[pr.clone()],
								&bsums[pr.clone()],
								&mut acc,
								&mut acc_m,
								out_row,
							)
						}
					} else {
						for t in 0..pn {
							out_row[t] = q4k_dot_q8k_scalar(wrow, nb, &q8s[p0 + t]);
						}
					}
				}
				p0 += pn;
			}
		}
	});
}

/// Batched fused Q6_K dequant-GEMM over `seq_len` activation rows (see
/// [`gemm_q4k`] for the contract; this is the Q6_K analogue for the `attn_qkv`
/// / `ffn_down` weights that use Q6_K in this model).
pub fn gemm_q6k(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q6k: &[u8],
	input: &[f32],
	out: &mut [f32],
) {
	assert_eq!(out.len(), seq_len * out_rows);
	gemm_q6k_strided(out_rows, in_dim, seq_len, q6k, input, out, out_rows, 0)
}

/// [`gemm_q6k`] with the strided output of [`gemm_q4k_strided`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_q6k_strided(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q6k: &[u8],
	input: &[f32],
	out: &mut [f32],
	ld: usize,
	col0: usize,
) {
	assert_eq!(input.len(), seq_len * in_dim);
	let q8s = std::sync::Arc::new(par_quantize_q8k(input, seq_len, in_dim));
	gemm_q6k_strided_q8(out_rows, in_dim, seq_len, q6k, &q8s, out, ld, col0)
}

/// [`gemm_q6k_strided`] on an already-quantized activation (see
/// [`gemm_q4k_strided_q8`]).
#[allow(clippy::too_many_arguments)]
pub fn gemm_q6k_strided_q8(
	out_rows: usize,
	in_dim: usize,
	seq_len: usize,
	q6k: &[u8],
	q8s: &std::sync::Arc<Vec<Q8K>>,
	out: &mut [f32],
	ld: usize,
	col0: usize,
) {
	assert_eq!(q8s.len(), seq_len);
	let nb = in_dim / QK_K;
	let row_bytes = nb * q6k_block_bytes();
	assert_eq!(q6k.len(), out_rows * row_bytes);
	let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
	let row_bytes = nb * q6k_block_bytes();
	let q6k_addr = q6k.as_ptr() as usize;
	let q6k_len = q6k.len();
	let ptq6 = gemm_ptile().min(seq_len.max(1));
	let vnni = use_avx2 && gemm_vnni();
	let tiles = if use_avx2 {
		pack_tiles(q8s, nb, ptq6)
	} else {
		std::sync::Arc::new(Vec::new())
	};
	par_gemm_collect(out_rows, seq_len, out, ld, col0, gemm_rgroup(), {
		let q8s = q8s.clone();
		let tiles = tiles.clone();
		move |rs, re, chunk| {
			let q6k = unsafe { std::slice::from_raw_parts(q6k_addr as *const u8, q6k_len) };
			let rg = gemm_rgroup();
			let mut accv = vec![unsafe { std::arch::x86_64::_mm256_setzero_ps() }; ptq6 * rg];
			// Position tile OUTSIDE the row loop, as in `gemm_q4k`: the
			// activation tile is loaded once and reused by every row in this
			// range (see `gemm_ptile`). Bit-exact: each output element is still
			// produced by the same instruction sequence.
			let mut p0 = 0usize;
			while p0 < seq_len {
				let pn = ptq6.min(seq_len - p0);
				if use_avx2 {
					let mut i = 0;
					while i < re - rs {
						let rows = rg.min(re - rs - i);
						let o = rs + i;
						let kern = if vnni {
							gemm_q6k_rows_vnni
						} else {
							gemm_q6k_row_all_avx2
						};
						unsafe {
							kern(
								&q6k[o * row_bytes..(o + rows) * row_bytes],
								rows,
								nb,
								&tiles[p0 / ptq6],
								&mut accv,
								&mut chunk[i * seq_len + p0..],
								seq_len,
							)
						}
						i += rows;
					}
					p0 += pn;
					continue;
				}
				for i in 0..(re - rs) {
					let o = rs + i;
					let wrow = &q6k[o * row_bytes..(o + 1) * row_bytes];
					let out_row = &mut chunk[i * seq_len + p0..i * seq_len + p0 + pn];
					{
						for t in 0..pn {
							out_row[t] = q6k_dot_q8k_scalar(wrow, nb, &q8s[p0 + t]);
						}
					}
				}
				p0 += pn;
			}
		}
	});
}

/// Position-tile width for the batched Q4_K/Q6_K GEMMs.
///
/// The row kernels sweep **all** `seq_len` positions for one weight row, so the
/// q8_K activation block is re-read once per output row: at `seq = 2048` that is
/// 10.5 MB per row against a 2.9 KB weight row, i.e. ~183 GB of activation
/// traffic for one `FfnUp`. Measured GFLOP/s peaked at `seq` 256-512 (1330) and
/// fell to 804 by 2048: the activation set outgrowing cache, nothing else.
///
/// Tiling the position dimension *outside* the row loop inverts the reuse: the
/// tile stays L2/L3-resident and is shared by every row in the thread's range,
/// and the weights are re-streamed once per tile instead (50 MB x `seq/tile`,
/// which is DRAM-cheap). Every output element is still computed by exactly the
/// same instruction sequence, so this is bit-exact.
///
/// Override with `QWEN_GEMM_PTILE` for A/B. 32 is the measured optimum after the
/// register-only reduction landed in the Q4_K kernel: Q4_K is flat over 16-32
/// (1767-1827 vs 1766-1791 across 4 repeats each) and clearly worse at 64
/// (~1650). Q6_K timings on this host are bimodal across repeats, never pick
/// this from a single run.
/// `QWEN_GEMM_HSUM=1` puts the CPU Q4_K prefill GEMM back on
/// [`gemm_q4k_row_all_avx2`] (per-block horizontal sums): the A/B for
/// [`gemm_q4k_rows_vacc_avx2`].
fn q4k_vacc() -> bool {
	static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*V.get_or_init(|| std::env::var("QWEN_GEMM_HSUM").map_or(true, |v| v == "0"))
}

pub(crate) fn gemm_ptile() -> usize {
	static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*P.get_or_init(|| {
		std::env::var("QWEN_GEMM_PTILE")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
			.filter(|v| *v > 0)
			.unwrap_or(32)
	})
}

/// Quantize all `seq_len` activation rows to `q8_K` across the worker pool.
///
/// This is the serial head of both batched GEMMs, and Amdahl applies to it: at
/// `seq = 2048` it was 7.5 ms of the Q4_K `FfnUp` (3 %) and 26.5 ms of the Q6_K
/// `FfnDown` (7 %), on one core while 23 sat idle. The rows are independent
/// (`quantize_q8k` reads one `in_dim` slice and allocates its own output) so the
/// only reason it was serial is that it was written as a `map().collect()`.
///
/// The GPU path's `pack_x_seq` was threaded for exactly this reason; this is the
/// CPU-side counterpart.
/// Quantize `seq_len` activation rows of `in_dim` to q8_K, one row per task.
pub fn par_quantize_q8k(input: &[f32], seq_len: usize, in_dim: usize) -> Vec<Q8K> {
	let mut q8s: Vec<Q8K> = std::iter::repeat_with(|| Q8K {
		ds: Vec::new(),
		qs: Vec::new(),
		bsums: Vec::new(),
	})
	.take(seq_len)
	.collect();
	let addr = q8s.as_mut_ptr() as usize;
	let inp = input.as_ptr() as usize;
	let ilen = input.len();
	super::par_for(seq_len, move |t| {
		// SAFETY: each `t` writes exactly one distinct, already-allocated slot,
		// and the placeholder it overwrites is three empty `Vec`s (no
		// allocation, nothing to drop). `input` is only read.
		let input = unsafe { std::slice::from_raw_parts(inp as *const f32, ilen) };
		let row = quantize_q8k(&input[t * in_dim..(t + 1) * in_dim], in_dim);
		unsafe { (addr as *mut Q8K).add(t).write(row) };
	});
	q8s
}

/// `par_quantize_q8k(silu_mul(gate, up))` without the `f32` intermediate: each
/// task forms one row of `silu(gate) * up` in a per-worker scratch row and
/// quantizes it from cache. Same kernels on the same values, so the bytes are
/// identical; what goes is a `seq x n` matrix written and read back through
/// DRAM. For the FFN at `seq = 2048` that is 142 MB each way per layer: the
/// SwiGLU pass and `FfnDown`'s quantization were 0.61 + 0.30 s of a 19 s
/// chunk, both memory-bound.
pub fn par_silu_mul_q8k(gate: &[f32], up: &[f32], seq_len: usize, n: usize) -> Vec<Q8K> {
	assert!(gate.len() >= seq_len * n && up.len() >= seq_len * n);
	par_rows_q8k(seq_len, n, |t, r| {
		super::silu_mul(&gate[t * n..(t + 1) * n], &up[t * n..(t + 1) * n], r)
	})
}

/// `par_quantize_q8k` of a matrix that is never materialized: `fill(t, row)`
/// writes row `t` (`n` values) into a per-worker scratch row, which is
/// quantized while still in cache. Bit-identical to filling an `f32` matrix
/// and quantizing it; what goes is that `seq x n` matrix written and read
/// back through DRAM. The producers are the activations that only ever feed
/// prefill matmuls (`par_silu_mul_q8k`, the RMS norms in `Model::prefill_chunk`).
pub fn par_rows_q8k<F>(seq_len: usize, n: usize, fill: F) -> Vec<Q8K>
where
	F: Fn(usize, &mut [f32]) + Sync + Send,
{
	let mut q8s: Vec<Q8K> = std::iter::repeat_with(|| Q8K {
		ds: Vec::new(),
		qs: Vec::new(),
		bsums: Vec::new(),
	})
	.take(seq_len)
	.collect();
	let addr = q8s.as_mut_ptr() as usize;
	super::par_for(seq_len, move |t| {
		thread_local! {
			static ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
		}
		let row = ROW.with(|r| {
			let mut r = r.borrow_mut();
			r.resize(n, 0.0);
			fill(t, &mut r);
			quantize_q8k(&r, n)
		});
		// SAFETY: as in `par_quantize_q8k`; one distinct slot per `t`.
		unsafe { (addr as *mut Q8K).add(t).write(row) };
	});
	q8s
}

/// Horizontal sum of 8 `f32`, **in registers**.
///
/// `hsum_float_8` (shared with the GEMV) spills through memory: a 32-byte store
/// followed by a scalar sum of 8, which is a store-forwarding stall. That is
/// affordable once per GEMV call; the batched GEMM runs it once per
/// *(block, position)* (20 blocks x every position x every output row) where
/// it and the min term's 16-byte-store twin were ~25 scalar ops against 24
/// vector ops of actual work.
///
/// This is a 6-op register-only tree. It is **not** bit-identical to the
/// left-to-right `tmp.iter().sum()` it replaces (pairwise vs sequential
/// summation of the same 8 values), so it lives here rather than replacing
/// `hsum_float_8`: the GEMV, and therefore decode, keeps the old order.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum8_tree(x: std::arch::x86_64::__m256) -> f32 {
	use std::arch::x86_64::*;
	let lo = _mm256_castps256_ps128(x);
	let hi = _mm256_extractf128_ps(x, 1);
	hsum4_tree(_mm_add_ps(lo, hi))
}

/// Horizontal sum of 4 `f32`, in registers. Replaces the `_mm_storeu_ps` +
/// scalar sum the min term used on the same per-(block, position) cadence.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum4_tree(x: std::arch::x86_64::__m128) -> f32 {
	use std::arch::x86_64::*;
	let s = _mm_add_ps(x, _mm_movehl_ps(x, x));
	_mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}
