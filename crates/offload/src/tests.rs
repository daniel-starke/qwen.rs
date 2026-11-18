use super::*;
#[cfg(feature = "vulkan")]
use quant::GgmlType;

#[test]
fn pack_f16_roundtrip_layout() {
	// Tiny hand-packed case: out_dim=1, in_dim=4.
	// f16 values: 1.0, 2.0, 3.0, 4.0
	let w = vec![
		quant::f32_to_fp16(1.0),
		quant::f32_to_fp16(2.0),
		quant::f32_to_fp16(3.0),
		quant::f32_to_fp16(4.0),
	];
	let packed = pack_f16_weights(&w, 1, 4);
	assert_eq!(packed.len(), 2);
	// index 0: low=1.0, high=2.0
	let p0 = unpack_u32(packed[0]);
	assert!((p0.0 - 1.0).abs() < 1e-2);
	assert!((p0.1 - 2.0).abs() < 1e-2);
	let p1 = unpack_u32(packed[1]);
	assert!((p1.0 - 3.0).abs() < 1e-2);
	assert!((p1.1 - 4.0).abs() < 1e-2);
}

#[test]
fn reference_matches_gemm_f16() {
	// Random-ish small matrix, validate the GPU-math reference equals the
	// verified CPU kernel (the shader is a faithful port of this).
	let out_dim = 8;
	let in_dim = 16;
	let mut w = vec![0u16; out_dim * in_dim];
	let mut x = vec![0.0f32; in_dim];
	for (i, v) in w.iter_mut().enumerate() {
		*v = quant::f32_to_fp16(((i % 7) as f32 - 3.0) * 0.25);
	}
	for (i, v) in x.iter_mut().enumerate() {
		*v = ((i % 5) as f32 - 2.0) * 0.5;
	}
	let ref_y = reference_matmul(&w, &x, out_dim, in_dim);
	let mut cpu_y = vec![0.0f32; out_dim];
	quant::gemm_f16(out_dim, in_dim, 1, &w, &x, &mut cpu_y);
	for i in 0..out_dim {
		assert!((ref_y[i] - cpu_y[i]).abs() < 1e-2, "mismatch at {i}");
	}
}

#[test]
fn packed_shader_math_equals_reference() {
	// Simulate exactly what the shader does: unpack pairs, dot with x.
	let out_dim = 6;
	let in_dim = 10; // even
	let mut w = vec![0u16; out_dim * in_dim];
	let mut x = vec![0.0f32; in_dim];
	for (i, v) in w.iter_mut().enumerate() {
		*v = quant::f32_to_fp16(((i % 11) as f32 - 5.0) * 0.3);
	}
	for (i, v) in x.iter_mut().enumerate() {
		*v = ((i % 4) as f32 - 1.5) * 0.7;
	}
	let packed = pack_f16_weights(&w, out_dim, in_dim);
	let half = in_dim / 2;
	let mut y = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		let mut acc = 0.0f32;
		for i in 0..half {
			let (lo, hi) = unpack_u32(packed[o * half + i]);
			acc += lo * x[2 * i] + hi * x[2 * i + 1];
		}
		y[o] = acc;
	}
	let ref_y = reference_matmul(&w, &x, out_dim, in_dim);
	for i in 0..out_dim {
		assert!((y[i] - ref_y[i]).abs() < 1e-3, "mismatch at {i}");
	}
}

// Helper: `unpack2x16float` semantics; low 16 bits = x, high 16 = y.
fn unpack_u32(v: u32) -> (f32, f32) {
	let lo = (v & 0xffff) as u16;
	let hi = (v >> 16) as u16;
	(quant::fp16_to_f32(lo), quant::fp16_to_f32(hi))
}

#[test]
fn manager_cpu_fallback_runs() {
	// Without the `vulkan` feature this is a no-op GPU path that must still
	// produce correct CPU results.
	let mgr = OffloadManager::new(2, 4);
	let out_dim = 4;
	let in_dim = 8;
	let w: Vec<u16> = (0..out_dim * in_dim)
		.map(|i| quant::f32_to_fp16(((i % 5) as f32 - 2.0) * 0.4))
		.collect();
	let lw = quant::LinearWeight::F16(w.clone());
	let x: Vec<f32> = (0..in_dim).map(|i| (i % 3) as f32 - 1.0).collect();
	let y = mgr.matmul(3, WeightId::FfnUp, &lw, &x, out_dim, in_dim);
	let mut expect = vec![0.0f32; out_dim];
	quant::gemm_f16(out_dim, in_dim, 1, &w, &x, &mut expect);
	for i in 0..out_dim {
		assert!((y[i] - expect[i]).abs() < 1e-2);
	}
}

#[test]
fn manager_cpu_fallback_is_the_model_kernel() {
	// The manager's CPU fallback must be **bit-identical** to what
	// `Model::lin` / `lin_batch` run with no manager at all, not merely
	// close to it.
	//
	// This is a regression test for a defect that shipped-on prefill routing
	// exposed: the manager is now constructed on every run, and its fallback
	// used `apply_linear_batch` for every job regardless of `seq`. At
	// `seq == 1` that is the batched GEMM standing in for the tuned GEMV, and
	// for a shared-`x` batch it is one fork-join and one q8_K quantization
	// per job instead of one for the group. Nothing asserted, no test failed:
	// `--forward 369` just quietly answered `DCR SUM -1028443.176329`
	// where every reference says `-1034584.732451`, with the argmax
	// unchanged. An approximate comparison would not have caught it, so this
	// one is exact.
	let mgr = OffloadManager::new(0, 4);
	let (out_dim, in_dim) = (8usize, 512usize);
	let mut f32w = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 11) as f32) * 0.5 - 2.0;
	}
	let mut raw = Vec::with_capacity(out_dim * in_dim / 256 * 144);
	for blk in 0..(out_dim * in_dim / 256) {
		raw.extend_from_slice(&build_q4k_block(&f32w[blk * 256..blk * 256 + 256]));
	}
	let w = quant::LinearWeight::Q4K {
		raw,
		out_dim,
		in_dim,
	};
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	// Single job, seq == 1: the GEMV, not the GEMM.
	let got = mgr.matmul(0, WeightId::FfnUp, &w, &x, out_dim, in_dim);
	assert_eq!(got, quant::apply_linear(&w, &x, out_dim, in_dim));

	// Shared-`x` batch, seq == 1: the grouped GEMV.
	let jobs = [
		BatchJob {
			il: 0,
			id: WeightId::FfnUp,
			w: &w,
			x: &x,
			out_dim,
			in_dim,
		},
		BatchJob {
			il: 0,
			id: WeightId::FfnGate,
			w: &w,
			x: &x,
			out_dim,
			in_dim,
		},
	];
	let got = mgr.matmul_batch(&jobs);
	let expect = quant::apply_linear_group(&[(&w, out_dim), (&w, out_dim)], &x, in_dim);
	assert_eq!(got, expect);

	// seq > 1 keeps the batched GEMM.
	let seq = 4usize;
	let xs: Vec<f32> = (0..seq * in_dim)
		.map(|i| ((i % 7) as f32 - 3.0) * 0.25)
		.collect();
	let job = BatchJobSeq {
		il: 0,
		id: WeightId::FfnUp,
		w: &w,
		x: &xs,
		seq,
		out_dim,
		in_dim,
	};
	let got = mgr.matmul_batch_seq(std::slice::from_ref(&job));
	assert_eq!(
		got[0],
		quant::apply_linear_batch(&w, &xs, seq, out_dim, in_dim)
	);
}

/// A chunk shorter than the routing threshold must stay on the CPU.
///
/// There is no counter to assert on, so the signal is bit-exactness: the GPU
/// GEMM sums per block where `apply_linear_batch` keeps a different order,
/// and on every shape measured here the two differ in the low bits. An
/// equality against the CPU kernel is therefore "this never reached the
/// GPU". `seq = 2048` is the control: it must *not* be bit-equal, or the
/// test is passing because nothing was dispatched at all.
#[cfg(feature = "vulkan")]
#[test]
fn manager_leaves_short_prefill_chunks_on_the_cpu() {
	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("manager_leaves_short_prefill_chunks_on_the_cpu: skipped (no Vulkan device)");
		return;
	}
	let (out_dim, in_dim) = (256usize, 1024usize);
	let mut f32w = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 11) as f32) * 0.5 - 2.0;
	}
	let mut raw = Vec::with_capacity(out_dim * in_dim / 256 * 144);
	for blk in 0..(out_dim * in_dim / 256) {
		raw.extend_from_slice(&build_q4k_block(&f32w[blk * 256..blk * 256 + 256]));
	}
	mgr.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K_M, out_dim, in_dim);
	let w = quant::LinearWeight::Q4K {
		raw,
		out_dim,
		in_dim,
	};

	let run = |seq: usize| -> (Vec<f32>, Vec<f32>) {
		let x: Vec<f32> = (0..seq * in_dim)
			.map(|i| ((i % 97) as f32 - 48.0) * 0.02)
			.collect();
		let job = BatchJobSeq {
			il: 0,
			id: WeightId::FfnUp,
			w: &w,
			x: &x,
			seq,
			out_dim,
			in_dim,
		};
		let got = mgr
			.matmul_batch_seq(std::slice::from_ref(&job))
			.pop()
			.unwrap();
		(got, quant::apply_linear_batch(&w, &x, seq, out_dim, in_dim))
	};

	for seq in [2usize, 16, 32] {
		let (got, cpu) = run(seq);
		assert_eq!(got, cpu, "seq {seq} should not have reached the GPU");
	}
	let (got, cpu) = run(2048);
	assert_ne!(got, cpu, "seq 2048 should have reached the GPU");
	let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
	for k in 0..cpu.len() {
		assert!(
			(got[k] - cpu[k]).abs() < 1e-3 * scale,
			"seq 2048 [{k}] {} vs {}",
			got[k],
			cpu[k]
		);
	}
}

#[test]
fn q4k_shader_math_matches_cpu() {
	// Mirror the shader's dequant_val exactly in Rust and compare element-wise
	// to the verified CPU `dequantize_q4_k`. Isolates shader-logic bugs.
	let mut f32w = vec![0.0f32; 256];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 11) as f32) * 0.5;
	}
	let raw = build_q4k_block(&f32w);
	let deq = quant::dequantize_q4_k(&raw, 256);
	for p in 0..256 {
		let val = rs_dequant_val(&raw, 0, p);
		assert!(
			(val - deq[p]).abs() < 1e-3,
			"p={} val={} deq={}",
			p,
			val,
			deq[p]
		);
	}
}

fn rs_f16(h: u32) -> f32 {
	let sign = (h >> 15) & 1;
	let exp = (h >> 10) & 0x1F;
	let mant = h & 0x3FF;
	if exp == 0 {
		if mant == 0 {
			if sign == 1 {
				-0.0
			} else {
				0.0
			}
		} else {
			let mag = (mant as f32) * 2f32.powi(-24);
			if sign == 1 {
				-mag
			} else {
				mag
			}
		}
	} else {
		let outbits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
		f32::from_bits(outbits)
	}
}

fn rs_scale_byte(idx: u32, s0: u32, s1: u32, s2: u32) -> u32 {
	let word = if idx >= 8 {
		s2
	} else if idx >= 4 {
		s1
	} else {
		s0
	};
	(word >> ((idx % 4) * 8)) & 0xFF
}

fn rs_get_sm(sub: u32, s0: u32, s1: u32, s2: u32) -> (f32, f32) {
	let s_sub = rs_scale_byte(sub, s0, s1, s2);
	let s_sub4 = rs_scale_byte(sub + 4, s0, s1, s2);
	let s_subm4 = rs_scale_byte(sub.wrapping_sub(4), s0, s1, s2);
	let (sc, m) = if sub < 4 {
		(s_sub & 63, s_sub4 & 63)
	} else {
		(
			(s_sub4 & 0xF) | ((s_subm4 >> 6) << 4),
			(s_sub4 >> 4) | ((s_sub >> 6) << 4),
		)
	};
	(sc as f32, m as f32)
}

fn rs_dequant_val(raw: &[u8], b: usize, p: usize) -> f32 {
	let bo = b * 144;
	let rd = |i: usize| {
		u32::from_le_bytes([
			raw[bo + i * 4],
			raw[bo + i * 4 + 1],
			raw[bo + i * 4 + 2],
			raw[bo + i * 4 + 3],
		])
	};
	let w0 = rd(0);
	let d = rs_f16(w0 & 0xFFFF);
	let dmin = rs_f16((w0 >> 16) & 0xFFFF);
	let s0 = rd(1);
	let s1 = rd(2);
	let s2 = rd(3);
	let p = p as u32;
	let g = p / 64;
	let off = p % 64;
	let byte_idx = g * 32 + (off % 32);
	let qword = rd(4 + byte_idx as usize / 4);
	let qbyte = (qword >> ((byte_idx % 4) * 8)) & 0xFF;
	let nibble = if off >= 32 { qbyte >> 4 } else { qbyte & 0xF };
	let sub = g * 2 + if off >= 32 { 1 } else { 0 };
	let (sc, m) = rs_get_sm(sub, s0, s1, s2);
	d * (sc * nibble as f32) - dmin * m
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_q4k_matmul_matches_cpu() {
	// End-to-end check of the fused Q4_K dequant+matmul shader against the
	// verified CPU `quant::dequantize_q4_k` + f32 dot. Skips when no Vulkan
	// adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_q4k_matmul_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512; // multiple of 256 (two Q4_K blocks per row)
				   // Build a Q4_K weight directly from an f32 matrix. Both sides decode the
				   // SAME bytes, so this validates the shader's fused dequant+matmul equals
				   // the verified CPU dequant + f32 dot.
	let mut f32w = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 11) as f32) * 0.5;
	}
	let mut raw = Vec::with_capacity(out_dim * in_dim / 256 * 144);
	for blk in 0..(out_dim * in_dim / 256) {
		let vals: &[f32] = &f32w[blk * 256..blk * 256 + 256];
		raw.extend_from_slice(&build_q4k_block(vals));
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K_M, out_dim, in_dim);
	let y = dev.matmul(0, WeightId::FfnUp, &x, out_dim, in_dim).unwrap();

	// Reference: exact `dequant(w)·x` for the f32-activation GEMV
	// (`q4k_f32.glsl`). The dp4a GEMV, where the device has it, reads a
	// q8_K activation, so it is compared against `gemv_q4k`, which
	// quantizes identically.
	let ref_y = q4k_gemv_ref(&dev, &raw, &x, out_dim, in_dim);
	for i in 0..out_dim {
		assert!(
			(y[i] - ref_y[i]).abs() < 1e-2,
			"gpu q4k mismatch at {i}: {} vs {}",
			y[i],
			ref_y[i]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_q4k_seq_gemv_matches_cpu() {
	// The speculative-verify GEMV (`q4k_f32_seq.glsl`, seq 2..8 through
	// the decode path) against an f64 dot of the dequantized weights, at
	// every seq and with a row count that leaves a partial row group.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_q4k_seq_gemv_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	dev.decode_seq = 8;
	let (out_dim, in_dim) = (13, 768);
	let f32w: Vec<f32> = (0..out_dim * in_dim)
		.map(|i| ((i * 7919 % 23) as f32 - 11.0) * 0.25 + (i / in_dim) as f32 * 0.1)
		.collect();
	let raw: Vec<u8> = f32w.chunks(256).flat_map(build_q4k_block).collect();
	dev.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K_M, out_dim, in_dim);
	let nb = in_dim / 256;
	for seq in 1..=8usize {
		let x: Vec<f32> = (0..seq * in_dim)
			.map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.125)
			.collect();
		let y = dev
			.matmul_batch(&[(0, WeightId::FfnUp, &x[..])])
			.unwrap()
			.pop()
			.unwrap();
		assert_eq!(y.len(), seq * out_dim);
		for t in 0..seq {
			for o in 0..out_dim {
				let r: f64 = (0..in_dim)
					.map(|p| {
						rs_dequant_val(&raw, o * nb + p / 256, p % 256) as f64
							* x[t * in_dim + p] as f64
					})
					.sum();
				let g = y[t * out_dim + o] as f64;
				assert!(
					(g - r).abs() <= 1e-4 * r.abs().max(10.0),
					"seq {seq} pos {t} row {o}: {g} vs {r}"
				);
			}
		}
	}
}

/// Build a 210-byte `Q6_K` super-block for `v` (256 f32 values). All
/// per-region scales are set to 1 (i8) and the global `d` is chosen so each
/// value maps to a valid 6-bit quant; round-trip only needs self-consistency
/// (both the shader and the verified CPU decode read the SAME bytes).
fn build_q6k_block(v: &[f32]) -> [u8; 210] {
	let mut block = [0u8; 210];
	let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
	let d_val = max_abs / 31.0;
	let d = quant::f32_to_fp16(d_val);
	block[208..210].copy_from_slice(&d.to_le_bytes());
	for i in 0..16 {
		block[192 + i] = 1u8; // as i8 == 1
	}
	for h in 0..2u32 {
		let base_off = h as usize * 128;
		let ql_off = (h * 64) as usize;
		let qh_off = (h * 32) as usize;
		for g in 0..4u32 {
			let region = (g * 32) as usize;
			for l in 0..32u32 {
				let idx = base_off + region + l as usize;
				let qf = (v[idx] / d_val).round().clamp(-32.0, 31.0);
				let qpos = (qf as i32 + 32) as u8; // 0..63
				let lo = qpos & 0xF;
				let hi = (qpos >> 4) & 0x3;
				let lus = l as usize;
				match g {
					0 => {
						block[ql_off + lus] |= lo;
						block[128 + qh_off + lus] |= hi;
					}
					1 => {
						block[ql_off + 32 + lus] |= lo;
						block[128 + qh_off + lus] |= hi << 2;
					}
					2 => {
						block[ql_off + lus] |= lo << 4;
						block[128 + qh_off + lus] |= hi << 4;
					}
					_ => {
						block[ql_off + 32 + lus] |= lo << 4;
						block[128 + qh_off + lus] |= hi << 6;
					}
				}
			}
		}
	}
	block
}

#[test]
fn q6k_shader_math_matches_cpu() {
	// Mirror the shader's `q6k_val` exactly in Rust and compare element-wise to
	// the verified CPU `dequantize_q6_k`. Isolates shader-logic bugs.
	let mut f32w = vec![0.0f32; 256];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 13) as f32) * 0.37 - 2.0;
	}
	let raw = build_q6k_block(&f32w);
	let deq = quant::dequantize_q6_k(&raw, 256);
	for p in 0..256 {
		let val = rs_q6k_val(&raw, 0, p);
		assert!(
			(val - deq[p]).abs() < 1e-3,
			"p={} val={} deq={}",
			p,
			val,
			deq[p]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_q6k_matmul_matches_cpu() {
	// End-to-end check of the fused Q6_K dequant+matmul shader against the
	// verified CPU `quant::dequantize_q6_k` + f32 dot. Skips when no Vulkan
	// adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_q6k_matmul_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512; // multiple of 256 (two Q6_K blocks per row)
	let mut f32w = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 13) as f32) * 0.37 - 2.0;
	}
	let mut raw = Vec::with_capacity(out_dim * in_dim / 256 * 210);
	for blk in 0..(out_dim * in_dim / 256) {
		let vals: &[f32] = &f32w[blk * 256..blk * 256 + 256];
		raw.extend_from_slice(&build_q6k_block(vals));
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::FfnDown, &raw, GgmlType::Q6_K, out_dim, in_dim);
	let y = dev
		.matmul(0, WeightId::FfnDown, &x, out_dim, in_dim)
		.unwrap();

	// Reference: exact `dequant(w)·x` (`q6k_f32.glsl` reads f32 activations).
	let ref_y = exact_ref(
		&quant::dequantize_q6_k(&raw, out_dim * in_dim),
		&x,
		out_dim,
		in_dim,
	);
	for i in 0..out_dim {
		assert!(
			(y[i] - ref_y[i]).abs() < 1e-2,
			"gpu q6k mismatch at {i}: {} vs {}",
			y[i],
			ref_y[i]
		);
	}
}

fn rs_q6k_val(raw: &[u8], b: usize, e: usize) -> f32 {
	let base = b * 210;
	let half = e / 128;
	let within = e % 128;
	let g = within / 32;
	let l = within % 32;
	let is = l / 16;
	let ql_off = half * 64;
	let qh_off = 128 + half * 32;
	let sc_off = half * 8;
	let (q, s_off) = match g {
		0 => (
			(raw[base + ql_off + l] as i32 & 0xF)
				| ((((raw[base + qh_off + l] as i32) & 3) << 4) - 32),
			(sc_off + is),
		),
		1 => (
			(raw[base + ql_off + l + 32] as i32 & 0xF)
				| ((((raw[base + qh_off + l] as i32 >> 2) & 3) << 4) - 32),
			sc_off + is + 2,
		),
		2 => (
			((raw[base + ql_off + l] as i32 >> 4) & 0xF)
				| ((((raw[base + qh_off + l] as i32 >> 4) & 3) << 4) - 32),
			sc_off + is + 4,
		),
		_ => (
			((raw[base + ql_off + l + 32] as i32 >> 4) & 0xF)
				| ((((raw[base + qh_off + l] as i32 >> 6) & 3) << 4) - 32),
			sc_off + is + 6,
		),
	};
	let s = raw[base + 192 + s_off] as i8 as f32;
	let d = quant::fp16_to_f32(u16::from_le_bytes([raw[base + 208], raw[base + 209]]));
	d * s * q as f32
}

/// Build a 144-byte `Q4_K` super-block for `v` (256 f32 values) using a
/// single-group encode (span/dequant round-trip only needs self-consistency).
fn build_q4k_block(v: &[f32]) -> [u8; 144] {
	let mut block = [0u8; 144];
	let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
	let span = max_abs.max(1e-6);
	let d = quant::f32_to_fp16(span / 15.0);
	let dmin = quant::f32_to_fp16(0.0);
	block[0..2].copy_from_slice(&d.to_le_bytes());
	block[2..4].copy_from_slice(&dmin.to_le_bytes());
	// scales: sub[i] = 15, min[i] = 0  (value = d*15*q)
	for i in 0..4 {
		block[4 + i] = 15;
	}
	for i in 4..8 {
		block[4 + i] = 0;
	}
	for i in 8..12 {
		block[4 + i] = 15;
	}
	let q = |x: f32| -> u8 { (x / span * 15.0).round().clamp(0.0, 15.0) as i32 as u8 };
	// Each byte `b` holds two values: low nibble = q[p_low], high = q[p_high],
	// where p_low = g*32 + b and p_high = g*32 + b + 32 (g = b/32); matching
	// the decode order in `dequantize_q4_k_block` / the shader.
	for b in 0..128u32 {
		let g = b / 32;
		let lo = q(v[(g * 32 + b) as usize]);
		let hi = q(v[(g * 32 + b + 32) as usize]);
		block[16 + b as usize] = (hi << 4) | (lo & 0x0F);
	}
	block
}

/// Build a 34-byte `Q8_0` block for `v` (32 f32 values): f16 scale + int8.
/// Round-trip only needs self-consistency (the shader and the verified CPU
/// decode read the SAME bytes).
fn build_q8_0_block(v: &[f32]) -> [u8; 34] {
	let mut block = [0u8; 34];
	let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
	let d_val = max_abs / 127.0;
	let d = quant::f32_to_fp16(d_val);
	block[0..2].copy_from_slice(&d.to_le_bytes());
	for (i, &val) in v.iter().enumerate() {
		let qf = (val / d_val).round().clamp(-128.0, 127.0);
		block[2 + i] = (qf as i32) as i8 as u8;
	}
	block
}

/// Build a length-`n*2` f16 byte buffer from random-ish magnitudes (for the
/// F16 shader mirror test).
fn build_f16_bytes(n: usize) -> Vec<u8> {
	let mut out = vec![0u8; n * 2];
	for i in 0..n {
		let h = quant::f32_to_fp16(((i % 17) as f32 - 8.0) * 0.31);
		out[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
	}
	out
}

fn rs_q8_0_val(raw: &[u8], b: usize, e: usize) -> f32 {
	let base = b * 34;
	let d = quant::fp16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
	let q = raw[base + 2 + e] as i8 as f32;
	d * q
}

fn rs_f16_val(raw: &[u8], e: usize) -> f32 {
	quant::fp16_to_f32(u16::from_le_bytes([raw[2 * e], raw[2 * e + 1]]))
}

fn rs_f32_val(raw: &[u8], e: usize) -> f32 {
	f32::from_le_bytes([raw[4 * e], raw[4 * e + 1], raw[4 * e + 2], raw[4 * e + 3]])
}

#[test]
fn q8_0_shader_math_matches_cpu() {
	// Mirror the shader's Q8_0 dequant exactly in Rust and compare element-wise
	// to the verified CPU `dequantize_q8_0`. Isolates shader-logic bugs.
	let mut f32w = vec![0.0f32; 64];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 13) as f32) * 0.37 - 2.0;
	}
	let raw = build_q8_0_block(&f32w[0..32]);
	let deq = quant::dequantize_q8_0(&raw, 32);
	for e in 0..32 {
		let val = rs_q8_0_val(&raw, 0, e);
		assert!(
			(val - deq[e]).abs() < 1e-3,
			"q8_0 e={} val={} deq={}",
			e,
			val,
			deq[e]
		);
	}
}

#[test]
fn f16_shader_math_matches_cpu() {
	// Mirror the shader's F16 dequant exactly in Rust and compare to the verified
	// CPU `dequantize_f16`.
	let n = 256;
	let raw = build_f16_bytes(n);
	let deq = quant::dequantize_f16(&raw, n);
	for e in 0..n {
		let val = rs_f16_val(&raw, e);
		assert!(
			(val - deq[e]).abs() < 1e-2,
			"f16 e={} val={} deq={}",
			e,
			val,
			deq[e]
		);
	}
}

#[test]
fn f32_shader_math_matches_cpu() {
	// Mirror the shader's F32 passthrough exactly in Rust and compare to the
	// verified CPU `dequantize_f32`.
	let n = 256;
	let mut raw = vec![0u8; n * 4];
	for i in 0..n {
		let v = ((i % 11) as f32) * 0.5;
		raw[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
	}
	let deq = quant::dequantize_f32(&raw, n);
	for e in 0..n {
		let val = rs_f32_val(&raw, e);
		assert!(
			(val - deq[e]).abs() < 1e-6,
			"f32 e={} val={} deq={}",
			e,
			val,
			deq[e]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_q8_0_matmul_matches_cpu() {
	// End-to-end check of the fused Q8_0 dequant+matmul shader against the
	// verified CPU `quant::dequantize_q8_0` + f32 dot. Skips when no Vulkan
	// adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_q8_0_matmul_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512; // multiple of 32 (two Q8_0 blocks per row)
	let mut f32w = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w.iter_mut().enumerate() {
		*v = ((i % 13) as f32) * 0.37 - 2.0;
	}
	let mut raw = Vec::with_capacity(out_dim * in_dim / 32 * 34);
	for blk in 0..(out_dim * in_dim / 32) {
		let vals: &[f32] = &f32w[blk * 32..blk * 32 + 32];
		raw.extend_from_slice(&build_q8_0_block(vals));
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::Wv, &raw, GgmlType::Q8_0, out_dim, in_dim);
	let y = dev.matmul(0, WeightId::Wv, &x, out_dim, in_dim).unwrap();

	let deq = quant::dequantize_q8_0(&raw, out_dim * in_dim);
	let mut ref_y = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		for k in 0..in_dim {
			ref_y[o] += deq[o * in_dim + k] * x[k];
		}
	}
	for i in 0..out_dim {
		assert!(
			(y[i] - ref_y[i]).abs() < 1e-2,
			"gpu q8_0 mismatch at {i}: {} vs {}",
			y[i],
			ref_y[i]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_f16_matmul_matches_cpu() {
	// End-to-end check of the fused F16 dequant+matmul shader against the
	// verified CPU `quant::dequantize_f16` + f32 dot. Skips when no Vulkan
	// adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_f16_matmul_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512;
	let mut raw = vec![0u8; out_dim * in_dim * 2];
	for i in 0..(out_dim * in_dim) {
		let h = quant::f32_to_fp16(((i % 13) as f32) * 0.37 - 2.0);
		raw[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::FfnDown, &raw, GgmlType::F16, out_dim, in_dim);
	let y = dev
		.matmul(0, WeightId::FfnDown, &x, out_dim, in_dim)
		.unwrap();

	let deq = quant::dequantize_f16(&raw, out_dim * in_dim);
	let mut ref_y = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		for k in 0..in_dim {
			ref_y[o] += deq[o * in_dim + k] * x[k];
		}
	}
	for i in 0..out_dim {
		assert!(
			(y[i] - ref_y[i]).abs() < 1e-2,
			"gpu f16 mismatch at {i}: {} vs {}",
			y[i],
			ref_y[i]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_f32_matmul_matches_cpu() {
	// End-to-end check of the fused F32 passthrough shader against the
	// verified CPU `quant::dequantize_f32` + f32 dot. Skips when no Vulkan
	// adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_f32_matmul_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512;
	let mut raw = vec![0u8; out_dim * in_dim * 4];
	for i in 0..(out_dim * in_dim) {
		let v = ((i % 13) as f32) * 0.37 - 2.0;
		raw[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::FfnDown, &raw, GgmlType::F32, out_dim, in_dim);
	let y = dev
		.matmul(0, WeightId::FfnDown, &x, out_dim, in_dim)
		.unwrap();

	let deq = quant::dequantize_f32(&raw, out_dim * in_dim);
	let mut ref_y = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		for k in 0..in_dim {
			ref_y[o] += deq[o * in_dim + k] * x[k];
		}
	}
	for i in 0..out_dim {
		assert!(
			(y[i] - ref_y[i]).abs() < 1e-3,
			"gpu f32 mismatch at {i}: {} vs {}",
			y[i],
			ref_y[i]
		);
	}
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_batch_matches_cpu() {
	// Exercises the multi-job batched dispatch that the real forward uses
	// (e.g. Wq/Wk/Wv share an input). Uploads two weights of different quant
	// types on one layer and folds them into a single submission. Skips when
	// no Vulkan adapter is present.
	let mut dev = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_batch_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let out_dim = 8;
	let in_dim = 512;

	// Q4_K weight
	let mut f32w4 = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w4.iter_mut().enumerate() {
		*v = ((i % 11) as f32) * 0.5;
	}
	let mut raw4 = Vec::with_capacity(out_dim * in_dim / 256 * 144);
	for blk in 0..(out_dim * in_dim / 256) {
		let vals: &[f32] = &f32w4[blk * 256..blk * 256 + 256];
		raw4.extend_from_slice(&build_q4k_block(vals));
	}
	// Q8_0 weight
	let mut f32w8 = vec![0.0f32; out_dim * in_dim];
	for (i, v) in f32w8.iter_mut().enumerate() {
		*v = ((i % 13) as f32) * 0.37 - 2.0;
	}
	let mut raw8 = Vec::with_capacity(out_dim * in_dim / 32 * 34);
	for blk in 0..(out_dim * in_dim / 32) {
		let vals: &[f32] = &f32w8[blk * 32..blk * 32 + 32];
		raw8.extend_from_slice(&build_q8_0_block(vals));
	}

	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();

	dev.upload(0, WeightId::Wq, &raw4, GgmlType::Q4_K_M, out_dim, in_dim);
	dev.upload(0, WeightId::Wv, &raw8, GgmlType::Q8_0, out_dim, in_dim);
	let ys = dev
		.matmul_batch(&[(0usize, WeightId::Wq, &x), (0usize, WeightId::Wv, &x)])
		.unwrap();
	assert_eq!(ys.len(), 2);

	// The q8_0 shader dots the float activation; the q4k reference follows
	// the device's decode GEMV (`q4k_gemv_ref`).
	let ref4 = q4k_gemv_ref(&dev, &raw4, &x, out_dim, in_dim);
	let deq8 = quant::dequantize_q8_0(&raw8, out_dim * in_dim);
	let mut ref8 = vec![0.0f32; out_dim];
	for o in 0..out_dim {
		for k in 0..in_dim {
			ref8[o] += deq8[o * in_dim + k] * x[k];
		}
	}
	for i in 0..out_dim {
		assert!(
			(ys[0][i] - ref4[i]).abs() < 1e-2,
			"gpu batch[0] (q4k) mismatch at {i}: {} vs {}",
			ys[0][i],
			ref4[i]
		);
		assert!(
			(ys[1][i] - ref8[i]).abs() < 1e-2,
			"gpu batch[1] (q8_0) mismatch at {i}: {} vs {}",
			ys[1][i],
			ref8[i]
		);
	}
}

/// Exact `w·x` in f64 for a dequantized row-major `[out_dim x in_dim]` weight.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn exact_ref(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
	(0..out_dim)
		.map(|o| {
			w[o * in_dim..(o + 1) * in_dim]
				.iter()
				.zip(x)
				.map(|(a, b)| *a as f64 * *b as f64)
				.sum::<f64>() as f32
		})
		.collect()
}

/// What the device's Q4_K decode GEMV should produce for `raw · x`: the exact
/// product for the f32-activation kernel (`q4k_f32.glsl`), `gemv_q4k` for the
/// dp4a one, whose activation is q8_K exactly as the CPU quantizes it.
#[cfg(feature = "vulkan")]
fn q4k_gemv_ref(
	dev: &gpu::GpuDevice,
	raw: &[u8],
	x: &[f32],
	out_dim: usize,
	in_dim: usize,
) -> Vec<f32> {
	if dev.gemv_x_f32(gpu::GpuKind::Q4K) {
		exact_ref(
			&quant::dequantize_q4_k(raw, out_dim * in_dim),
			x,
			out_dim,
			in_dim,
		)
	} else {
		let mut r = vec![0.0f32; out_dim];
		quant::gemv_q4k(out_dim, in_dim, x, raw, &mut r);
		r
	}
}

/// The decode row split through the manager: the GPU computes the first rows
/// of each split matrix while the CPU computes the rest. The CPU rows must be
/// bit-identical to the unsplit CPU group GEMV; the GPU rows (f32 activations)
/// must agree with the exact product.
#[cfg(feature = "vulkan")]
#[test]
fn manager_row_split_matches_cpu() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_row_split_matches_cpu: skipped (no Vulkan device)");
		return;
	}
	let in_dim = 512usize;
	let (o_up, o_down) = (40usize, 24usize);
	let mut st = 0x2545_F491u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let mut q4 = |od: usize| -> Vec<u8> {
		let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
		for _ in 0..od * in_dim / 256 {
			let vals: Vec<f32> = (0..256)
				.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
				.collect();
			raw.extend_from_slice(&build_q4k_block(&vals));
		}
		raw
	};
	let up = q4(o_up);
	let gate = q4(o_down);
	let x: Vec<f32> = (0..in_dim)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
		.collect();
	mgr.upload_split(
		0,
		WeightId::FfnUp,
		&up,
		quant::GgmlType::Q4_K,
		o_up,
		in_dim,
		12,
	);
	mgr.upload_split(
		0,
		WeightId::FfnGate,
		&gate,
		quant::GgmlType::Q4_K,
		o_down,
		in_dim,
		8,
	);
	assert_eq!(mgr.n_split(), 2);
	assert!(mgr.decode_on_gpu(0));
	let w_up = quant::LinearWeight::Q4K {
		raw: up.clone(),
		out_dim: o_up,
		in_dim,
	};
	let w_gate = quant::LinearWeight::Q4K {
		raw: gate.clone(),
		out_dim: o_down,
		in_dim,
	};
	let got = mgr.matmul_batch(&[
		BatchJob {
			il: 0,
			id: WeightId::FfnUp,
			w: &w_up,
			x: &x,
			out_dim: o_up,
			in_dim,
		},
		BatchJob {
			il: 0,
			id: WeightId::FfnGate,
			w: &w_gate,
			x: &x,
			out_dim: o_down,
			in_dim,
		},
	]);
	let cpu = quant::apply_linear_group(&[(&w_up, o_up), (&w_gate, o_down)], &x, in_dim);
	for (k, (raw, od, r)) in [(&up, o_up, 12usize), (&gate, o_down, 8usize)]
		.iter()
		.enumerate()
	{
		assert_eq!(got[k].len(), *od);
		let exact = q4k_gemv_ref(&mgr.gpu().unwrap(), raw, &x, *od, in_dim);
		for i in 0..*r {
			assert!(
				(got[k][i] - exact[i]).abs() < 1e-3 * exact[i].abs().max(1.0),
				"job {k} GPU row {i}: {} vs {}",
				got[k][i],
				exact[i]
			);
		}
		for i in *r..*od {
			assert_eq!(
				got[k][i].to_bits(),
				cpu[k][i].to_bits(),
				"job {k} CPU row {i}"
			);
		}
	}
}

/// A decode batch's command buffer is recorded once and resubmitted: a second
/// and third call on the same weights with a different activation must see the
/// new `x`, not the one baked in at recording time.
#[cfg(feature = "vulkan")]
#[test]
fn manager_row_split_reuses_command_buffer() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_row_split_reuses_command_buffer: skipped (no Vulkan device)");
		return;
	}
	let (od, in_dim, r) = (32usize, 512usize, 16usize);
	let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
	for b in 0..od * in_dim / 256 {
		let vals: Vec<f32> = (0..256)
			.map(|i| (((b * 31 + i * 7) % 200) as f32) / 100.0 - 1.0)
			.collect();
		raw.extend_from_slice(&build_q4k_block(&vals));
	}
	mgr.upload_split(
		0,
		WeightId::FfnUp,
		&raw,
		quant::GgmlType::Q4_K,
		od,
		in_dim,
		r,
	);
	let w = quant::LinearWeight::Q4K {
		raw: raw.clone(),
		out_dim: od,
		in_dim,
	};
	for round in 0..3 {
		let x: Vec<f32> = (0..in_dim)
			.map(|i| (((i + 17 * round) % 13) as f32 - 6.0) * 0.1)
			.collect();
		let got = mgr.matmul_batch(&[BatchJob {
			il: 0,
			id: WeightId::FfnUp,
			w: &w,
			x: &x,
			out_dim: od,
			in_dim,
		}]);
		let exact = q4k_gemv_ref(&mgr.gpu().unwrap(), &raw, &x, od, in_dim);
		for i in 0..r {
			assert!(
				(got[0][i] - exact[i]).abs() < 1e-3 * exact[i].abs().max(1.0),
				"round {round} GPU row {i}: {} vs {}",
				got[0][i],
				exact[i]
			);
		}
	}
}

/// The prefill row split: rows `[0, r)` from resident VRAM, `[r, g)` streamed,
/// `[g, out)` on the CPU, stitched into one `[seq x out]` matrix. The CPU block
/// must be bit-identical to the CPU GEMM; the GPU blocks (tiled GEMM on the
/// same q8_K activations) must agree to float tolerance.
#[cfg(feature = "vulkan")]
#[test]
fn manager_prefill_split_matches_cpu() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_prefill_split_matches_cpu: skipped (no Vulkan device)");
		return;
	}
	let (in_dim, od, seq) = (512usize, 40usize, 300usize);
	let mut st = 0x7F4A_7C15u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
	for _ in 0..od * in_dim / 256 {
		let vals: Vec<f32> = (0..256)
			.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
			.collect();
		raw.extend_from_slice(&build_q4k_block(&vals));
	}
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
		.collect();
	mgr.upload_split(
		0,
		WeightId::FfnUp,
		&raw,
		quant::GgmlType::Q4_K,
		od,
		in_dim,
		12,
	);
	let w = quant::LinearWeight::Q4K {
		raw: raw.clone(),
		out_dim: od,
		in_dim,
	};
	let got = mgr
		.matmul_batch_seq(&[BatchJobSeq {
			il: 0,
			id: WeightId::FfnUp,
			w: &w,
			x: &x,
			seq,
			out_dim: od,
			in_dim,
		}])
		.pop()
		.unwrap();
	let cpu = quant::apply_linear_batch(&w, &x, seq, od, in_dim);
	assert_eq!(got.len(), cpu.len());
	// g = floor(frac * 40 / 4) * 4 with the default Q4_K fraction: 0.77 with
	// the dp4a GEMM (28), 0.55 with the float one (20).
	let dp4a = mgr.gpu().unwrap().dp4a_gemm(gpu::GpuKind::Q4K);
	let g = if dp4a { 28usize } else { 20 };
	for t in 0..seq {
		for o in 0..od {
			let (a, b) = (got[t * od + o], cpu[t * od + o]);
			if o >= g {
				assert_eq!(
					a.to_bits(),
					b.to_bits(),
					"CPU block t={t} o={o}: {a} vs {b}"
				);
			} else {
				assert!(
					(a - b).abs() < 1e-3 * b.abs().max(1.0),
					"GPU block t={t} o={o}: {a} vs {b}"
				);
			}
		}
	}
}

/// A prefill group whose shared input arrives as q8_K
/// (`matmul_seq_group_q8`) is bit-identical to the same group on the `f32`
/// input (`matmul_batch_seq`), and a CPU-tail job in it (`ssm_beta`, no split
/// rows) is bit-identical to the plain CPU GEMM.
#[cfg(feature = "vulkan")]
#[test]
fn manager_prefill_q8_group_matches_f32() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_prefill_q8_group_matches_f32: skipped (no Vulkan device)");
		return;
	}
	let (in_dim, od, ob, seq) = (512usize, 40usize, 8usize, 300usize);
	let mut st = 0x2545_F491u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let mut q4 = |rows: usize| {
		let mut raw = Vec::with_capacity(rows * in_dim / 256 * 144);
		for _ in 0..rows * in_dim / 256 {
			let vals: Vec<f32> = (0..256)
				.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
				.collect();
			raw.extend_from_slice(&build_q4k_block(&vals));
		}
		raw
	};
	let (raw_u, raw_b) = (q4(od), q4(ob));
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|i| ((i * 41 % 103) as f32 - 51.0) * 0.02)
		.collect();
	mgr.upload_split(
		0,
		WeightId::FfnUp,
		&raw_u,
		quant::GgmlType::Q4_K,
		od,
		in_dim,
		12,
	);
	let wu = quant::LinearWeight::Q4K {
		raw: raw_u,
		out_dim: od,
		in_dim,
	};
	let wb = quant::LinearWeight::Q4K {
		raw: raw_b,
		out_dim: ob,
		in_dim,
	};
	let group = [(WeightId::FfnUp, &wu, od), (WeightId::WssmBeta, &wb, ob)];
	assert!(mgr.prefill_q8_ok(0, &group, seq, in_dim));
	let f32_out = mgr.matmul_batch_seq(&[
		BatchJobSeq {
			il: 0,
			id: WeightId::FfnUp,
			w: &wu,
			x: &x,
			seq,
			out_dim: od,
			in_dim,
		},
		BatchJobSeq {
			il: 0,
			id: WeightId::WssmBeta,
			w: &wb,
			x: &x,
			seq,
			out_dim: ob,
			in_dim,
		},
	]);
	let q8 = std::sync::Arc::new(quant::par_quantize_q8k(&x, seq, in_dim));
	let q8_out = mgr.matmul_seq_group_q8(0, &group, &q8, seq, in_dim);
	let cpu_b = quant::apply_linear_batch(&wb, &x, seq, ob, in_dim);
	for (k, (a, b)) in f32_out.iter().zip(&q8_out).enumerate() {
		assert_eq!(a.len(), b.len());
		assert!(
			a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()),
			"job {k}: q8 vs f32 route"
		);
	}
	assert!(
		q8_out[1]
			.iter()
			.zip(&cpu_b)
			.all(|(p, q)| p.to_bits() == q.to_bits()),
		"CPU tail vs CPU GEMM"
	);
}

/// A prefetched upload (`prefetch_hint`) changes when a call's streamed rows
/// reach VRAM, never what is computed: hinted calls (split, deferred, and
/// one after a hint that named a different call) are bit-identical to the
/// same calls on another layer key without hints.
#[cfg(feature = "vulkan")]
#[test]
fn manager_prefetch_is_bit_identical() {
	let mut mgr = OffloadManager::new(0, 2);
	if !mgr.available {
		eprintln!("manager_prefetch_is_bit_identical: skipped (no Vulkan device)");
		return;
	}
	let (in_dim, od, seq) = (512usize, 40usize, 300usize);
	let mut st = 0x51ED_270Bu32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let mut q4 = || {
		let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
		for _ in 0..od * in_dim / 256 {
			let vals: Vec<f32> = (0..256)
				.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
				.collect();
			raw.extend_from_slice(&build_q4k_block(&vals));
		}
		raw
	};
	let ids = [
		WeightId::FfnUp,
		WeightId::FfnDown,
		WeightId::SsmOut,
		WeightId::WqkvGate,
	];
	let raws: Vec<Vec<u8>> = ids.iter().map(|_| q4()).collect();
	for il in 0..2 {
		for (id, raw) in ids.iter().zip(&raws) {
			mgr.upload_split(il, *id, raw, quant::GgmlType::Q4_K, od, in_dim, 12);
		}
	}
	let ws: Vec<quant::LinearWeight> = raws
		.iter()
		.map(|raw| quant::LinearWeight::Q4K {
			raw: raw.clone(),
			out_dim: od,
			in_dim,
		})
		.collect();
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
		.collect();
	let q8 = std::sync::Arc::new(quant::par_quantize_q8k(&x, seq, in_dim));
	let split = |il: usize, k: usize| {
		mgr.matmul_seq_group_q8(il, &[(ids[k], &ws[k], od)], &q8, seq, in_dim)
			.pop()
			.unwrap()
	};
	let keep = || -> std::sync::Arc<dyn std::any::Any + Send + Sync> { std::sync::Arc::new(()) };
	let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
	// Layer 0, no hints: the reference.
	let want: Vec<Vec<f32>> = (0..3).map(|k| split(0, k)).collect();
	let d = mgr
		.matmul_q8_defer(0, ids[3], &ws[3], od, &q8, seq, in_dim)
		.expect("deferred");
	let want_d = mgr.finish_deferred(d);
	// Layer 1, each call hinting the next.
	let jobs: Vec<[(WeightId, &quant::LinearWeight, usize); 1]> =
		(0..3).map(|k| [(ids[k], &ws[k], od)]).collect();
	let hint = |k: usize| crate::NextCall::Split(&jobs[k]);
	mgr.prefetch_hint(1, hint(1), seq, in_dim, keep());
	assert_eq!(bits(&split(1, 0)), bits(&want[0]), "call before a prefetch");
	mgr.prefetch_hint(
		1,
		crate::NextCall::Defer(ids[3], &ws[3], od),
		seq,
		in_dim,
		keep(),
	);
	assert_eq!(bits(&split(1, 1)), bits(&want[1]), "prefetched split");
	mgr.prefetch_hint(1, hint(1), seq, in_dim, keep());
	let d = mgr
		.matmul_q8_defer(1, ids[3], &ws[3], od, &q8, seq, in_dim)
		.expect("deferred");
	assert_eq!(
		bits(&mgr.finish_deferred(d)),
		bits(&want_d),
		"prefetched deferral"
	);
	// The hint named `FfnDown`, but `SsmOut` comes.
	assert_eq!(
		bits(&split(1, 2)),
		bits(&want[2]),
		"call after a wrong hint"
	);
}

/// A deferred prefill matmul (`matmul_q8_defer` + `finish_deferred`: GPU
/// rows resident + streamed, the rest on the CPU after the caller's own
/// work) matches the plain CPU GEMM on every row.
#[cfg(feature = "vulkan")]
#[test]
fn manager_deferred_matmul_matches_cpu() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_deferred_matmul_matches_cpu: skipped (no Vulkan device)");
		return;
	}
	let (in_dim, od, seq) = (512usize, 40usize, 300usize);
	let mut st = 0x9E37_79B9u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
	for _ in 0..od * in_dim / 256 {
		let vals: Vec<f32> = (0..256)
			.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
			.collect();
		raw.extend_from_slice(&build_q4k_block(&vals));
	}
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02)
		.collect();
	mgr.upload_split(
		0,
		WeightId::WqkvGate,
		&raw,
		quant::GgmlType::Q4_K,
		od,
		in_dim,
		12,
	);
	let w = quant::LinearWeight::Q4K {
		raw,
		out_dim: od,
		in_dim,
	};
	let job = (WeightId::WqkvGate, &w, od);
	assert!(mgr.defer_ok(0, &job, seq, in_dim));
	let q8 = std::sync::Arc::new(quant::par_quantize_q8k(&x, seq, in_dim));
	let d = mgr
		.matmul_q8_defer(0, WeightId::WqkvGate, &w, od, &q8, seq, in_dim)
		.expect("deferred");
	let got = mgr.finish_deferred(d);
	let want = quant::apply_linear_batch(&w, &x, seq, od, in_dim);
	let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
	let worst = got
		.iter()
		.zip(&want)
		.fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
	assert!(
		worst <= 1e-4 * scale.max(1.0),
		"worst |deferred - cpu| {worst} (max |cpu| {scale})"
	);
}

/// The prefill attention kernel (`attn_prefill.glsl`) against the CPU's
/// `quant::attn_causal` on an f16 cache: a partial query tile, a past that
/// is not a multiple of the key block, and a group subset.
#[cfg(feature = "vulkan")]
#[test]
fn gpu_attn_prefill_matches_cpu() {
	let gpu = match gpu::GpuDevice::init() {
		Ok(d) => d,
		Err(_) => {
			eprintln!("gpu_attn_prefill_matches_cpu: skipped (no Vulkan device)");
			return;
		}
	};
	let (n_kv, hd) = (4usize, 256usize);
	let n_head = n_kv * 6;
	for &(nq, t0, g0, g1) in &[
		(37usize, 150usize, 0usize, 4usize),
		(70, 0, 1, 3),
		(9, 300, 2, 4),
	] {
		let tk = t0 + nq;
		let mut st = 0x1234_5678u32 ^ (nq as u32 * 7919);
		let mut rnd = || {
			st ^= st << 13;
			st ^= st >> 17;
			st ^= st << 5;
			(st % 2001) as f32 / 1000.0 - 1.0
		};
		let q: Vec<f32> = (0..nq * n_head * hd).map(|_| rnd()).collect();
		let kf: Vec<f32> = (0..tk * n_kv * hd).map(|_| rnd()).collect();
		let vf: Vec<f32> = (0..tk * n_kv * hd).map(|_| rnd()).collect();
		let (mut k, mut v) = (Vec::new(), Vec::new());
		quant::kv_extend(&mut k, &kf);
		quant::kv_extend(&mut v, &vf);
		let scale = 1.0 / (hd as f32).sqrt();
		let mut want = vec![0.0f32; nq * n_head * hd];
		quant::attn_causal(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, &mut want);
		let mut got = vec![f32::NAN; nq * n_head * hd];
		let p = gpu
			.attn_prefill_begin(&q, &k, &v, nq, t0, n_head, n_kv, scale, g0, g1)
			.expect("begin");
		gpu.attn_prefill_end(p, &mut got, n_head, g0).expect("end");
		for i in 0..nq {
			for h in g0 * 6..g1 * 6 {
				for d in 0..hd {
					let o = (i * n_head + h) * hd + d;
					assert!(
						(got[o] - want[o]).abs() < 1e-4,
						"nq {nq} t0 {t0} pos {i} head {h} dim {d}: {} vs {}",
						got[o],
						want[o]
					);
				}
			}
		}
	}
}

/// Kernel throughput at the real prefill shape: `cargo test --release -p
/// offload --features vulkan bench_attn_prefill -- --ignored --nocapture`.
#[cfg(feature = "vulkan")]
#[test]
#[ignore]
fn bench_attn_prefill() {
	let gpu = gpu::GpuDevice::init().expect("vulkan");
	let (n_kv, hd, nq) = (4usize, 256usize, 2048usize);
	let n_head = n_kv * 6;
	for &t0 in &[0usize, 8192, 30720] {
		let tk = t0 + nq;
		let q: Vec<f32> = (0..nq * n_head * hd)
			.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
			.collect();
		let kf: Vec<f32> = (0..tk * n_kv * hd)
			.map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.01)
			.collect();
		let (mut k, mut v) = (Vec::new(), Vec::new());
		quant::kv_extend(&mut k, &kf);
		quant::kv_extend(&mut v, &kf);
		let scale = 1.0 / (hd as f32).sqrt();
		let mut out = vec![0.0f32; nq * n_head * hd];
		// Causal FLOPs of 2 groups (12 heads): 2 * 2 * hd * sum_i (t0 + i + 1).
		let pairs = nq as f64 * t0 as f64 + (nq as f64) * (nq as f64 + 1.0) / 2.0;
		let flop = 4.0 * hd as f64 * 12.0 * pairs;
		for _ in 0..2 {
			let t = std::time::Instant::now();
			let p = gpu
				.attn_prefill_begin(&q, &k, &v, nq, t0, n_head, n_kv, scale, 0, 2)
				.unwrap();
			gpu.attn_prefill_end(p, &mut out, n_head, 0).unwrap();
			let g = t.elapsed().as_secs_f64();
			let t = std::time::Instant::now();
			quant::attn_causal_groups(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, 2..4, &mut out);
			let c = t.elapsed().as_secs_f64();
			println!(
				"t0 {t0:6}: GPU 2 groups {:8.1} ms ({:.2} TFLOP/s incl. upload)  CPU 2 groups {:8.1} ms ({:.2} TFLOP/s)",
				g * 1e3, flop / g / 1e12, c * 1e3, flop / c / 1e12
			);
		}
	}
}

/// The prefill attention's GPU/CPU query-row split: rows `[0, r)` carry
/// `frac` of the causal work, and running the two row ranges as separate
/// calls (the second offset by `r` rows, `t0 + r`) gives the one-call
/// result.
#[test]
fn attn_query_row_split_is_exact() {
	use crate::manager::attn_gpu_rows;
	assert_eq!(attn_gpu_rows(2048, 9000, 1.0), 2048);
	for &(nq, t0) in &[(2048usize, 0usize), (2048, 16384), (313, 18745)] {
		for &f in &[0.5f64, 0.88] {
			let r = attn_gpu_rows(nq, t0, f);
			assert_eq!(r % 8, 0);
			let work = |n: usize| n as f64 * (t0 as f64 + n as f64 / 2.0);
			assert!(
				work(r) <= f * work(nq) + 1.0 && work(r + 8) > f * work(nq),
				"nq {nq} t0 {t0} f {f}: r {r}"
			);
		}
	}
	let (n_kv, hd, nq, t0) = (2usize, 64usize, 40usize, 100usize);
	let n_head = n_kv * 3;
	let (qs, tk) = (n_head * hd, t0 + nq);
	let q: Vec<f32> = (0..nq * qs)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
		.collect();
	let kf: Vec<f32> = (0..tk * n_kv * hd)
		.map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.01)
		.collect();
	let (mut k, mut v) = (Vec::new(), Vec::new());
	quant::kv_extend(&mut k, &kf);
	quant::kv_extend(&mut v, &kf);
	let scale = 1.0 / (hd as f32).sqrt();
	let mut want = vec![0.0f32; nq * qs];
	quant::attn_causal(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, &mut want);
	let r = 16;
	let mut got = vec![0.0f32; nq * qs];
	quant::attn_causal_groups(
		&q[..r * qs],
		&k,
		&v,
		r,
		t0,
		n_head,
		n_kv,
		hd,
		scale,
		0..1,
		&mut got,
	);
	quant::attn_causal_groups(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, 1..2, &mut got);
	quant::attn_causal_groups(
		&q[r * qs..],
		&k,
		&v,
		nq - r,
		t0 + r,
		n_head,
		n_kv,
		hd,
		scale,
		0..1,
		&mut got[r * qs..],
	);
	let d = want
		.iter()
		.zip(&got)
		.fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
	assert!(d < 1e-5, "max diff {d}");
}

/// A lost device (another application claiming VRAM can cause one) is rebuilt
/// with its decode-split weights uploaded again, instead of leaving every
/// matmul on the CPU for the rest of the run.
#[cfg(feature = "vulkan")]
#[test]
fn manager_rebuilds_lost_device() {
	let mut mgr = OffloadManager::new(0, 1);
	if !mgr.available {
		eprintln!("manager_rebuilds_lost_device: skipped (no Vulkan device)");
		return;
	}
	let (od, in_dim, r) = (32usize, 512usize, 16usize);
	let mut raw = Vec::with_capacity(od * in_dim / 256 * 144);
	for b in 0..od * in_dim / 256 {
		let vals: Vec<f32> = (0..256)
			.map(|i| (((b * 29 + i * 11) % 200) as f32) / 100.0 - 1.0)
			.collect();
		raw.extend_from_slice(&build_q4k_block(&vals));
	}
	mgr.upload_split(
		0,
		WeightId::FfnUp,
		&raw,
		quant::GgmlType::Q4_K,
		od,
		in_dim,
		r,
	);
	let w = quant::LinearWeight::Q4K {
		raw: raw.clone(),
		out_dim: od,
		in_dim,
	};
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
	let job = [BatchJob {
		il: 0,
		id: WeightId::FfnUp,
		w: &w,
		x: &x,
		out_dim: od,
		in_dim,
	}];
	let before = mgr.matmul_batch(&job);
	assert!(!mgr.gpu_lost());

	mgr.gpu().unwrap().mark_lost();
	assert!(mgr.gpu_lost());
	// Lost: the CPU computes every row.
	let cpu = mgr.matmul_batch(&job);
	let want = quant::apply_linear_group(&[(&w, od)], &x, in_dim);
	assert_eq!(
		cpu[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
		want[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
	);

	let src = |il: usize, id: WeightId| {
		(il == 0 && id == WeightId::FfnUp).then(|| (raw.clone(), quant::GgmlType::Q4_K))
	};
	assert!(mgr.recover_gpu(&src));
	assert!(!mgr.gpu_lost());
	assert!(mgr
		.gpu()
		.unwrap()
		.has(OffloadManager::SPLIT_KEY, WeightId::FfnUp));
	// The rebuilt device computes the GPU rows again, as the first one did.
	let after = mgr.matmul_batch(&job);
	assert_eq!(
		after[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
		before[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
	);
}
