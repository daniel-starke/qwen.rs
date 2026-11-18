//! Test / benchmark module for the `quant` crate.

use super::*;

// ---- fp16 round-trip sanity ----
#[test]
fn bench_gemm_f16_local() {
	// Representative big decode matmul: ffn_up [17408 x 5120] f16, seq=1.
	// Mirrors what forward_step pays per token for every large weight.
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let mut w = vec![0u16; out_rows * in_dim];
	for (i, v) in w.iter_mut().enumerate() {
		*v = f32_to_fp16(((i % 9) as f32 - 4.0) * 0.1);
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
	let mut y = vec![0.0f32; out_rows];
	for _ in 0..3 {
		gemm_f16(out_rows, in_dim, 1, &w, &x, &mut y);
	}
	let iters = 10usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		gemm_f16(out_rows, in_dim, 1, &w, &x, &mut y);
	}
	let dt = t0.elapsed().as_secs_f64();
	let per_ms = dt / iters as f64 * 1000.0;
	let macs = out_rows as f64 * in_dim as f64;
	let gflops = macs * 2.0 / 1e9 / (per_ms / 1000.0);
	let wbytes = (out_rows * in_dim * 2) as f64;
	let bw = wbytes / 1e9 / (per_ms / 1000.0);
	println!(
		"BENCH gemm_f16 {out_rows}x{in_dim}: {per_ms:.2} ms/matmul  {gflops:.1} GFLOP/s  {bw:.1} GB/s weight-read"
	);
	// DRAM copy bandwidth of a same-sized byte buffer for comparison.
	let nbytes = w.len() * 2;
	let buf = vec![0xABu8; nbytes];
	let mut dst = vec![0u8; nbytes];
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		dst.copy_from_slice(&buf);
	}
	let dt = t0.elapsed().as_secs_f64();
	let mbw = (nbytes as f64 * iters as f64) / 1e9 / dt;
	println!("BENCH memcpy {nbytes} bytes: {mbw:.1} GB/s");
}

// ---- prefill (batched, seq>1) GEMM throughput ----
fn build_q4k(out_rows: usize, in_dim: usize) -> Vec<u8> {
	let nb = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nb * blk];
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for r in raw.chunks_mut(blk) {
		r[0] = 0x00;
		r[1] = 0x3C;
		r[2] = 0x00;
		r[3] = 0x3C;
		for b in r[4..].iter_mut() {
			*b = rnd() as u8;
		}
	}
	raw
}

#[test]
fn bench_gemm_q4k_prefill() {
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let seq_len = 24usize;
	let raw = build_q4k(out_rows, in_dim);
	let rnd = |rng: &mut u64| {
		*rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(*rng >> 33) as u32
	};
	let mut rng2 = 0xBu64;
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|_| ((rnd(&mut rng2) % 100) as f32 - 50.0) * 0.01)
		.collect();
	let macs = out_rows as f64 * in_dim as f64 * seq_len as f64 * 2.0;
	let wbytes = (raw.len()) as f64;

	// Hoisted batched path (new).
	let mut y = vec![0.0f32; seq_len * out_rows];
	for _ in 0..3 {
		gemm_q4k(out_rows, in_dim, seq_len, &raw, &input, &mut y);
	}
	let iters = 5usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		gemm_q4k(out_rows, in_dim, seq_len, &raw, &input, &mut y);
	}
	let dt = t0.elapsed().as_secs_f64();
	let gflops = macs * iters as f64 / dt / 1e9;
	let bw = wbytes * iters as f64 / dt / 1e9;
	println!(
		"BENCH gemm_q4k prefill {out_rows}x{in_dim}x{seq_len}: {gflops:.1} GFLOP/s  {bw:.1} GB/s weight-read"
	);

	// Old per-position loop (emulates the pre-hoist gemm_q4k).
	let mut y2 = vec![0.0f32; seq_len * out_rows];
	let mut row = vec![0.0f32; out_rows];
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		for t in 0..seq_len {
			gemv_q4k(
				out_rows,
				in_dim,
				&input[t * in_dim..(t + 1) * in_dim],
				&raw,
				&mut row,
			);
			for o in 0..out_rows {
				y2[t * out_rows + o] = row[o];
			}
		}
	}
	let dt2 = t0.elapsed().as_secs_f64();
	let gflops2 = macs * iters as f64 / dt2 / 1e9;
	println!(
		"BENCH gemm_q4k prefill OLD(per-row): {gflops2:.1} GFLOP/s  (new is {:.2}x)",
		gflops / gflops2
	);
	// sanity: results agree
	let mut maxd = 0.0f32;
	for i in 0..y.len() {
		maxd = maxd.max((y[i] - y2[i]).abs());
	}
	println!("BENCH gemm_q4k prefill max|new-old|={maxd:.4}");
}

#[test]
fn bench_gemm_q6k_prefill() {
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let seq_len = 24usize;
	let nb = in_dim / QK_K;
	let blk = 210usize;
	let mut raw = vec![0u8; out_rows * nb * blk];
	let mut rng = 0x123456789ABCDEF0u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for r in raw.chunks_mut(nb * blk) {
		for b in r.iter_mut() {
			*b = rnd() as u8;
		}
		for i in 0..nb {
			let n = i * blk + blk;
			r[n - 2] = 0x00;
			r[n - 1] = 0x3C; // fp16 1.0
		}
	}
	let mut rng2 = 0xBu64;
	let rnd2 = |rng: &mut u64| {
		*rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(*rng >> 33) as u32
	};
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|_| ((rnd2(&mut rng2) % 100) as f32 - 50.0) * 0.01)
		.collect();
	let macs = out_rows as f64 * in_dim as f64 * seq_len as f64 * 2.0;
	let wbytes = (raw.len()) as f64;

	// Hoisted batched path (new).
	let mut y = vec![0.0f32; seq_len * out_rows];
	for _ in 0..3 {
		gemm_q6k(out_rows, in_dim, seq_len, &raw, &input, &mut y);
	}
	let iters = 5usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		gemm_q6k(out_rows, in_dim, seq_len, &raw, &input, &mut y);
	}
	let dt = t0.elapsed().as_secs_f64();
	let gflops = macs * iters as f64 / dt / 1e9;
	let bw = wbytes * iters as f64 / dt / 1e9;
	println!(
		"BENCH gemm_q6k prefill {out_rows}x{in_dim}x{seq_len}: {gflops:.1} GFLOP/s  {bw:.1} GB/s weight-read"
	);

	// Old per-position loop (emulates the pre-hoist gemm_q6k, which called
	// q6k_dot once per position).
	let mut y2 = vec![0.0f32; seq_len * out_rows];
	let mut row = vec![0.0f32; out_rows];
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		for t in 0..seq_len {
			gemv_q6k(
				out_rows,
				in_dim,
				&input[t * in_dim..(t + 1) * in_dim],
				&raw,
				&mut row,
			);
			for o in 0..out_rows {
				y2[t * out_rows + o] = row[o];
			}
		}
	}
	let dt2 = t0.elapsed().as_secs_f64();
	let gflops2 = macs * iters as f64 / dt2 / 1e9;
	println!(
		"BENCH gemm_q6k prefill OLD(per-row): {gflops2:.1} GFLOP/s  (new is {:.2}x)",
		gflops / gflops2
	);
	let mut maxd = 0.0f32;
	for i in 0..y.len() {
		maxd = maxd.max((y[i] - y2[i]).abs());
	}
	println!("BENCH gemm_q6k prefill max|new-old|={maxd:.4}");
}

#[test]
fn fp16_known_values() {
	assert_eq!(fp16_to_f32(0x3C00), 1.0);
	assert_eq!(fp16_to_f32(0x3E00), 1.5);
	assert_eq!(fp16_to_f32(0x3400), 0.25);
	assert_eq!(fp16_to_f32(0x0000), 0.0);
	assert_eq!(fp16_to_f32(0xBC00), -1.0);
	// subnormal half values: value = mant * 2^-24
	assert_eq!(fp16_to_f32(0x0001), 5.960_464_5e-8);
	assert_eq!(fp16_to_f32(0x8001), -5.960_464_5e-8);
	assert_eq!(fp16_to_f32(0x03FF), 6.097_555e-5);
}

// ---- Q4_K against true llama.cpp reference numbers ----
// The 256 `u32` values below are the raw f32 bit patterns printed by
// llama.cpp's `dequantize_row_q4_K` (copied verbatim into a C harness) for
// the hand-built 144-byte block constructed in `q4k_reference_block()`.
const REF_Q4K: [u32; 256] = [
	1111425024, 1088946176, 1109065728, 3219128320, 1106116608, 1114963968, 1101398016, 1112604672,
	1094451200, 1110245376, 1076887552, 1107886080, 1115914240, 1103757312, 1113784320, 1099038720,
	1111425024, 1088946176, 1109065728, 3219128320, 1106116608, 1114963968, 1101398016, 1112604672,
	1094451200, 1110245376, 1076887552, 1107886080, 1115914240, 1103757312, 1113784320, 1099038720,
	3233808384, 1103101952, 1103101952, 1113063424, 1113063424, 1113063424, 1118306304, 1118306304,
	1122238464, 1122238464, 1125122048, 1125122048, 1125122048, 1127088128, 1127088128, 1129054208,
	1129054208, 1131020288, 1131020288, 1132724224, 1132724224, 1132724224, 1133707264, 1133707264,
	1134690304, 1134690304, 1135673344, 1135673344, 1135673344, 1136656384, 1136656384, 1137639424,
	1142296576, 1120501760, 1140105216, 3240361984, 1136467968, 1145024512, 1132830720, 1143205888,
	1125924864, 1141387264, 1110769664, 1138286592, 1145933824, 1134649344, 1144115200, 1129562112,
	1142296576, 1120501760, 1140105216, 3240361984, 1136467968, 1145024512, 1132830720, 1143205888,
	1125924864, 1141387264, 1110769664, 1138286592, 1145933824, 1134649344, 1144115200, 1129562112,
	1150021632, 1150685184, 1150685184, 3244818432, 3244818432, 3244818432, 1116012544, 1116012544,
	1125351424, 1125351424, 1130659840, 1130659840, 1130659840, 1134215168, 1134215168, 1136869376,
	1136869376, 1139523584, 1139523584, 1141514240, 1141514240, 1141514240, 1142841344, 1142841344,
	1144168448, 1144168448, 1145495552, 1145495552, 1145495552, 1146822656, 1146822656, 1148149760,
	1127579648, 1107558400, 1125416960, 0, 1122435072, 1130823680, 1118109696, 1128660992,
	1111883776, 1126498304, 1099169792, 1124335616, 1131905024, 1120272384, 1129742336, 1115947008,
	1127579648, 1107558400, 1125416960, 0, 1122435072, 1130823680, 1118109696, 1128660992,
	1111883776, 1126498304, 1099169792, 1124335616, 1131905024, 1120272384, 1129742336, 1115947008,
	1129824256, 1131003904, 1131003904, 1132183552, 1132183552, 1132183552, 1132912640, 1132912640,
	3196059648, 3196059648, 1099825152, 1099825152, 1099825152, 1108279296, 1108279296, 1112997888,
	1112997888, 1116700672, 1116700672, 1119059968, 1119059968, 1119059968, 1121419264, 1121419264,
	1123778560, 1123778560, 1125105664, 1125105664, 1125105664, 1126285312, 1126285312, 1127464960,
	1129709568, 1109000192, 1127153664, 3204448256, 1124597760, 1133002752, 1120010240, 1130987520,
	1114112000, 1128431616, 1100480512, 1125875712, 1133641728, 1122566144, 1132265472, 1117454336,
	1129709568, 1109000192, 1127153664, 3204448256, 1124597760, 1133002752, 1120010240, 1130987520,
	1114112000, 1128431616, 1100480512, 1125875712, 1133641728, 1122566144, 1132265472, 1117454336,
	1129398272, 1130774528, 1130774528, 1132150784, 1132150784, 1132150784, 1132994560, 1132994560,
	1133682688, 1133682688, 1134370816, 1134370816, 1134370816, 3208642560, 3208642560, 1101135872,
	1101135872, 1109721088, 1109721088, 1115226112, 1115226112, 1115226112, 1118208000, 1118208000,
	1120960512, 1120960512, 1123713024, 1123713024, 1123713024, 1125269504, 1125269504, 1126645760,
];

/// Build the exact 144-byte `Q4_K` block the C reference harness used.
fn q4k_reference_block() -> Vec<u8> {
	let mut blk = vec![0u8; 144];
	// d = 1.5 (fp16 0x3E00), dmin = 0.25 (fp16 0x3400)
	blk[0] = 0x00;
	blk[1] = 0x3E;
	blk[2] = 0x00;
	blk[3] = 0x34;
	// scales[i] = (i*17+3) & 0x3f
	for i in 0..12 {
		blk[4 + i] = ((i * 17 + 3) & 0x3f) as u8;
	}
	// qs[i] = (i*7+11) & 0xff
	for i in 0..128 {
		blk[16 + i] = ((i * 7 + 11) & 0xff) as u8;
	}
	blk
}

#[test]
fn q4k_matches_llama_cpp_reference() {
	let blk = q4k_reference_block();
	let mut out = [0.0f32; 256];
	dequantize_q4_k_block(&blk, &mut out);
	for i in 0..256 {
		assert_eq!(
			f32::to_bits(out[i]),
			REF_Q4K[i],
			"Q4_K reference mismatch at element {i}"
		);
	}
}

// ---- Q8_0 round-trip ----
#[test]
fn q8_0_matches_formula() {
	// block: d = 2.0 (fp16), qs = [-1, 2, -3, ...]
	let d = 0x4000u16; // 2.0 in fp16
	let mut blk = vec![0u8; 34];
	blk[0] = d as u8;
	blk[1] = (d >> 8) as u8;
	for i in 0..32 {
		blk[2 + i] = (i as i8).wrapping_sub(16) as u8; // signed -16..15
	}
	let mut out = [0.0f32; 32];
	dequantize_q8_0_block(&blk, &mut out);
	for i in 0..32 {
		let q = (i as i8).wrapping_sub(16) as f32;
		assert_eq!(out[i], 2.0 * q);
	}
	// full-tensor path
	let full = dequantize_q8_0(&blk, 32);
	assert_eq!(full.len(), 32);
	assert_eq!(full[0], -32.0);
}

// ---- f32 / f16 copy ----
#[test]
fn f32_copy() {
	let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
	let out = dequantize_f32(&data, 8);
	for i in 0..8 {
		assert_eq!(out[i], i as f32);
	}
}

// ---- GEMM correctness vs naive ----
#[test]
fn gemm_matches_naive() {
	let out_rows = 7;
	let in_dim = 16;
	let seq_len = 5;

	let weights: Vec<f32> = (0..out_rows * in_dim)
		.map(|i| ((i % 5) as f32 - 2.0) * 0.1)
		.collect();
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|i| ((i % 7) as f32 - 3.0) * 0.2)
		.collect();

	let mut got = vec![0.0f32; seq_len * out_rows];
	gemm(out_rows, in_dim, seq_len, &weights, &input, &mut got);

	// naive reference: out is [out_rows, seq_len] => index o*seq_len + t
	let mut exp = vec![0.0f32; seq_len * out_rows];
	for t in 0..seq_len {
		for o in 0..out_rows {
			let mut s = 0.0f32;
			for k in 0..in_dim {
				s += weights[o * in_dim + k] * input[t * in_dim + k];
			}
			exp[o * seq_len + t] = s;
		}
	}

	for i in 0..exp.len() {
		assert!(
			(got[i] - exp[i]).abs() < 1e-4,
			"gemm[{i}] got {} exp {}",
			got[i],
			exp[i]
		);
	}
}

// ---- Q6_K against true llama.cpp reference numbers ----
const REF_Q6K: [u32; 256] = [
	1145774080, 1153138688, 3303182336, 3271163904, 1141678080, 1151090688, 3297038336, 3283845120,
	1134313472, 1157234688, 3299086336, 3290185728, 1115291648, 1155186688, 3301134336, 3294281728,
	1143910400, 1151418368, 3301103616, 3268870144, 1139924992, 1149657088, 3294916608, 3282124800,
	1132879872, 1154940928, 3297581056, 3288752128, 1112997888, 1153179648, 3299342336, 3292274688,
	3272015872, 3297181696, 3299024896, 1151909888, 1137344512, 3288793088, 3258187776, 3296165888,
	1147944960, 1144258560, 1140293632, 3286302720, 3299762176, 1151172608, 1149329408, 1119092736,
	3269033984, 3294199808, 3296946176, 1149759488, 1134764032, 3285811200, 3255894016, 3293011968,
	1144934400, 1141964800, 1137139712, 3283435520, 3297540096, 1149091840, 1146122240, 1116798976,
	1146880000, 1146880000, 1146880000, 1137590272, 1137590272, 1137590272, 1136689152, 1136689152,
	3260809216, 3265593344, 3265593344, 3265593344, 3265593344, 3288956928, 3288956928, 3288956928,
	3284459520, 3284459520, 3284459520, 1140047872, 1140047872, 1140047872, 1139433472, 1139433472,
	1126744064, 1125515264, 1125515264, 1125515264, 1125515264, 3274227712, 3274227712, 3274227712,
	1130102784, 1130102784, 1130102784, 1129447424, 1129447424, 1129447424, 1128792064, 1128792064,
	1128792064, 1128136704, 1128136704, 1128136704, 1128136704, 1127481344, 1127481344, 1127481344,
	1101660160, 1101660160, 1101660160, 1101004800, 1101004800, 1101004800, 1100349440, 1100349440,
	1100349440, 1099694080, 1099694080, 1099694080, 1099694080, 1099038720, 1099038720, 1099038720,
	3267559424, 3274964992, 1129938944, 1097859072, 3263627264, 3272998912, 1124007936, 1110704128,
	3256221696, 3278897152, 1126006784, 1117126656, 3236954112, 3276931072, 1127972864, 1121058816,
	3277012992, 3284336640, 1139515392, 1107427328, 3272753152, 3282206720, 1133125632, 1120075776,
	3265429504, 3288465408, 1135255552, 1126334464, 3246522368, 3286466560, 1137385472, 1130594304,
	1117126656, 1142292480, 1144340480, 3292233728, 3277914112, 1133903872, 1103626240, 1141473280,
	3288547328, 3284664320, 3280568320, 1132068864, 1145159680, 3291414528, 3289366528, 3259498496,
	1120567296, 1145733120, 1148497920, 3296534528, 3281510400, 1137344512, 1107755008, 1144627200,
	3291557888, 3288793088, 3283722240, 1135132672, 1149421568, 3295428608, 3292663808, 3263627264,
	1126825984, 1126825984, 1126825984, 1147084800, 1147084800, 1147084800, 1147781120, 1147781120,
	3297386496, 3297038336, 3297038336, 3297038336, 3297038336, 3282698240, 3282698240, 3282698240,
	3283312640, 3283312640, 3283312640, 1140858880, 1140858880, 1140858880, 1141698560, 1141698560,
	1152186368, 1152606208, 1152606208, 1152606208, 1152606208, 3295899648, 3295899648, 3295899648,
	3270508544, 3270508544, 3270508544, 3262119936, 3262119936, 3262119936, 3295674368, 3295674368,
	0, 1114636288, 1114636288, 1114636288, 1114636288, 1123024896, 1123024896, 1123024896,
	1129201664, 1129201664, 1129201664, 1133084672, 1133084672, 1133084672, 1135337472, 1135337472,
	1135337472, 1137590272, 1137590272, 1137590272, 1137590272, 1153804288, 1153804288, 1153804288,
];

/// Build the exact 210-byte `Q6_K` block the C reference harness used
/// (real struct layout: ql@0, qh@128, scales@192, d@208).
fn q6k_reference_block() -> Vec<u8> {
	let mut blk = vec![0u8; 210];
	// d = 1.25 (fp16 0x3D00) at offset 208
	blk[208] = 0x00;
	blk[209] = 0x3D;
	// ql[i] = i*5 + 3
	for i in 0..128 {
		blk[i] = ((i * 5 + 3) & 0xff) as u8;
	}
	// qh[i] = i*3 + 9
	for i in 0..64 {
		blk[128 + i] = ((i * 3 + 9) & 0xff) as u8;
	}
	// scales[i] = i*7 - 50 (stored as int8) at offset 192
	for i in 0..16 {
		blk[192 + i] = (i as i32 * 7 - 50) as u8;
	}
	blk
}

#[test]
fn q6k_matches_llama_cpp_reference() {
	let blk = q6k_reference_block();
	let mut out = [0.0f32; 256];
	dequantize_q6_k_block(&blk, &mut out);
	for i in 0..256 {
		assert_eq!(
			f32::to_bits(out[i]),
			REF_Q6K[i],
			"Q6_K reference mismatch at element {i}"
		);
	}
}

// ---- gemm_f16 matches f32 gemm (within f16 precision) ----
#[test]
fn gemm_f16_matches_gemm() {
	let out_rows = 11;
	let in_dim = 37;
	let seq_len = 3;

	let weights_f32: Vec<f32> = (0..out_rows * in_dim)
		.map(|i| ((i % 9) as f32 - 4.0) * 0.07)
		.collect();
	let weights_f16: Vec<u16> = weights_f32.iter().map(|&x| f32_to_fp16(x)).collect();
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|i| ((i % 5) as f32 - 2.0) * 0.3)
		.collect();

	let mut got = vec![0.0f32; seq_len * out_rows];
	gemm_f16(out_rows, in_dim, seq_len, &weights_f16, &input, &mut got);

	let mut exp = vec![0.0f32; seq_len * out_rows];
	gemm(out_rows, in_dim, seq_len, &weights_f32, &input, &mut exp);

	for i in 0..exp.len() {
		// f16 weights lose ~3 decimal digits; require < 1e-2 abs and 1% rel.
		let tol = (exp[i].abs() * 0.01 + 1e-2).max(1e-2);
		assert!(
			(got[i] - exp[i]).abs() < tol,
			"gemm_f16[{i}] got {} exp {}",
			got[i],
			exp[i]
		);
	}
}

// ---- batched GEMM (Q4_K) matches per-row GEMV ----
#[test]
fn gemm_q4k_matches_per_row() {
	// 29 rows: three full 8-row groups of the block-outer kernels and a
	// ragged one.
	let out_rows = 29usize;
	let in_dim = 768usize;
	// 70 positions: two full 32-position GEMM tiles and a partial one.
	let seq_len = 70usize;
	let nb = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nb * blk];
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	// Keep the per-block `d` / `dmin` fp16 finite (1.0) so the dequant dot
	// stays finite; randomize only scales + nibbles (as real weights do).
	for r in raw.chunks_mut(blk) {
		r[0] = 0x00;
		r[1] = 0x3C;
		r[2] = 0x00;
		r[3] = 0x3C;
		for b in r[4..].iter_mut() {
			*b = rnd() as u8;
		}
	}
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|_| ((rnd() % 100) as f32 - 50.0) * 0.01)
		.collect();

	let mut got = vec![0.0f32; seq_len * out_rows];
	gemm_q4k(out_rows, in_dim, seq_len, &raw, &input, &mut got);

	let q8: Vec<Q8K> = (0..seq_len)
		.map(|t| quantize_q8k(&input[t * in_dim..(t + 1) * in_dim], in_dim))
		.collect();
	for t in 0..seq_len {
		let mut row = vec![0.0f32; out_rows];
		gemv_q4k(
			out_rows,
			in_dim,
			&input[t * in_dim..(t + 1) * in_dim],
			&raw,
			&mut row,
		);
		for o in 0..out_rows {
			let g = got[t * out_rows + o];
			if g.is_nan() && row[o].is_nan() {
				continue;
			}
			// The VNNI kernels sum in a different lane order than the GEMV,
			// and these random blocks' fp16 `d` cancel terms of up to ~1e7.
			let mag = term_mag(
				&q8[t],
				&raw[o * nb * blk..(o + 1) * nb * blk],
				nb,
				blk,
				dequantize_q4_k_block,
			);
			assert!(
				(g - row[o]).abs() < 1e-2 + 1e-5 * row[o].abs() + 1e-7 * mag,
				"gemm_q4k[{t}][{o}] got {g} exp {}",
				row[o]
			);
		}
	}
}

/// Sum of |w * x| over a row's dequantized products: the scale of the
/// cancellation a GEMM's f32 summation order can expose.
fn term_mag(q8: &Q8K, row: &[u8], nb: usize, blk: usize, deq: fn(&[u8], &mut [f32])) -> f32 {
	let mut wd = vec![0f32; QK_K];
	let mut mag = 0f64;
	for i in 0..nb {
		deq(&row[i * blk..(i + 1) * blk], &mut wd);
		for k in 0..QK_K {
			mag += (wd[k] as f64 * q8.ds[i] as f64 * q8.qs[i * QK_K + k] as f64).abs();
		}
	}
	mag as f32
}

// ---- batched GEMM (Q6_K) matches per-row GEMV ----
#[test]
fn gemm_q6k_matches_per_row() {
	// 29 rows: three full 8-row groups of the block-outer kernels and a
	// ragged one.
	let out_rows = 29usize;
	let in_dim = 768usize;
	// 70 positions: two full 32-position GEMM tiles and a partial one.
	let seq_len = 70usize;
	let nb = in_dim / QK_K;
	let row_bytes = nb * q6k_block_bytes();
	let mut raw = vec![0u8; out_rows * row_bytes];
	let mut rng = 0x123456789ABCDEF0u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	// Q6_K block = 128 ql + 64 qh + 16 scales + 2 (d, dmin) fp16. Keep the
	// two fp16 scales finite (1.0); randomize the rest.
	for r in raw.chunks_mut(row_bytes) {
		for b in r.iter_mut() {
			*b = rnd() as u8;
		}
		// d, dmin at the tail (two fp16 = 1.0)
		let n = r.len();
		r[n - 4] = 0x00;
		r[n - 3] = 0x3C;
		r[n - 2] = 0x00;
		r[n - 1] = 0x3C;
	}
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|_| ((rnd() % 100) as f32 - 50.0) * 0.01)
		.collect();

	let mut got = vec![0.0f32; seq_len * out_rows];
	gemm_q6k(out_rows, in_dim, seq_len, &raw, &input, &mut got);

	let q8: Vec<Q8K> = (0..seq_len)
		.map(|t| quantize_q8k(&input[t * in_dim..(t + 1) * in_dim], in_dim))
		.collect();
	for t in 0..seq_len {
		let mut row = vec![0.0f32; out_rows];
		gemv_q6k(
			out_rows,
			in_dim,
			&input[t * in_dim..(t + 1) * in_dim],
			&raw,
			&mut row,
		);
		for o in 0..out_rows {
			let g = got[t * out_rows + o];
			if g.is_nan() && row[o].is_nan() {
				continue;
			}
			// The VNNI kernels sum in a different lane order than the GEMV,
			// and these random blocks' fp16 `d` cancel terms of up to ~1e7.
			let mag = term_mag(
				&q8[t],
				&raw[o * row_bytes..(o + 1) * row_bytes],
				nb,
				210,
				dequantize_q6_k_block,
			);
			assert!(
				(g - row[o]).abs() < 1e-2 + 1e-5 * row[o].abs() + 1e-7 * mag,
				"gemm_q6k[{t}][{o}] got {g} exp {}",
				row[o]
			);
		}
	}
}

// ---- apply_linear_batch matches apply_linear per row (f32 / f16) ----
#[test]
fn apply_linear_batch_matches_per_row() {
	let out_rows = 13usize;
	let in_dim = 64usize;
	let seq_len = 5usize;
	let w_f32: Vec<f32> = (0..out_rows * in_dim)
		.map(|i| ((i % 9) as f32 - 4.0) * 0.07)
		.collect();
	let w_f16: Vec<u16> = w_f32.iter().map(|&x| f32_to_fp16(x)).collect();
	let input: Vec<f32> = (0..seq_len * in_dim)
		.map(|i| ((i % 5) as f32 - 2.0) * 0.3)
		.collect();

	let got_f32 = apply_linear_batch(
		&LinearWeight::F32(w_f32.clone()),
		&input,
		seq_len,
		out_rows,
		in_dim,
	);
	let got_f16 = apply_linear_batch(
		&LinearWeight::F16(w_f16.clone()),
		&input,
		seq_len,
		out_rows,
		in_dim,
	);
	for t in 0..seq_len {
		let row_in = &input[t * in_dim..(t + 1) * in_dim];
		let exp_f32 = apply_linear(&LinearWeight::F32(w_f32.clone()), row_in, out_rows, in_dim);
		let exp_f16 = apply_linear(&LinearWeight::F16(w_f16.clone()), row_in, out_rows, in_dim);
		for o in 0..out_rows {
			let g = got_f32[t * out_rows + o];
			assert!(
				(g - exp_f32[o]).abs() < 1e-4,
				"f32 batch[{t}][{o}] {g} {}",
				exp_f32[o]
			);
			let gh = got_f16[t * out_rows + o];
			let tol = (exp_f16[o].abs() * 0.01 + 1e-2).max(1e-2);
			assert!(
				(gh - exp_f16[o]).abs() < tol,
				"f16 batch[{t}][{o}] {gh} {}",
				exp_f16[o]
			);
		}
	}
}

// ---- fused Q4_K dequant-GEMV matches dequantize + dot ----
#[test]
fn gemv_q4k_matches_dequant() {
	let out_rows = 16usize;
	let in_dim = 512usize; // multiple of QK_K
	let nblocks = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for b in 0..(out_rows * nblocks) {
		let d = f32_to_fp16(((rnd() as f32) / 1e9 - 1.0) * 2.0);
		let dmin = f32_to_fp16(((rnd() as f32) / 1e9) * 0.5);
		raw[b * blk] = d as u8;
		raw[b * blk + 1] = (d >> 8) as u8;
		raw[b * blk + 2] = dmin as u8;
		raw[b * blk + 3] = (dmin >> 8) as u8;
		for i in 0..K_SCALE_SIZE {
			raw[b * blk + 4 + i] = (rnd() & 0x3f) as u8;
		}
		for i in 0..(QK_K / 2) {
			raw[b * blk + 4 + K_SCALE_SIZE + i] = (rnd() & 0xff) as u8;
		}
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
	let mut got = vec![0.0f32; out_rows];
	gemv_q4k(out_rows, in_dim, &x, &raw, &mut got);
	// The fused kernel quantizes the activation to int8 (llama.cpp's q8_K
	// intermediate). Validate the integer dot against the equivalent
	// q8-dequantized reference (dequantize the q4 row, dot with the int8
	// reconstruction of x): the integer math is then exact to ~1e-4.
	let x_q8 = quantize_q8k(&x, in_dim);
	let mut exp = vec![0.0f32; out_rows];
	for o in 0..out_rows {
		let row = dequantize_q4_k(&raw[o * nblocks * blk..(o + 1) * nblocks * blk], in_dim);
		let mut s = 0.0f32;
		for k in 0..in_dim {
			let xh = x_q8.ds[k / QK_K] * x_q8.qs[k] as f32;
			s += row[k] * xh;
		}
		exp[o] = s;
	}
	for o in 0..out_rows {
		let tol = (exp[o].abs() * 1e-3 + 1e-2).max(1e-2);
		assert!(
			(got[o] - exp[o]).abs() < tol,
			"gemv_q4k[{o}] got {} exp {} tol {}",
			got[o],
			exp[o],
			tol
		);
	}
}

// ---- gemv_f16 matches f32 gemv (within f16 precision) ----
#[test]
fn gemv_f16_matches_gemv() {
	let out_rows = 11usize;
	let in_dim = 37usize;
	let w_f32: Vec<f32> = (0..out_rows * in_dim)
		.map(|i| ((i % 9) as f32 - 4.0) * 0.07)
		.collect();
	let w_f16: Vec<u16> = w_f32.iter().map(|&x| f32_to_fp16(x)).collect();
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
	let mut got = vec![0.0f32; out_rows];
	gemv_f16(out_rows, in_dim, &x, &w_f16, &mut got);
	let mut exp = vec![0.0f32; out_rows];
	gemv(out_rows, in_dim, &x, &w_f32, &mut exp);
	for o in 0..out_rows {
		let tol = (exp[o].abs() * 0.01 + 1e-2).max(1e-2);
		assert!(
			(got[o] - exp[o]).abs() < tol,
			"gemv_f16[{o}] got {} exp {}",
			got[o],
			exp[o]
		);
	}
}

// ---- apply_linear on a Q4K weight matches dequant + dot ----
#[test]
fn apply_linear_q4k_matches_gemv() {
	let out_rows = 8usize;
	let in_dim = 256usize;
	let nblocks = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	let mut rng = 0x123456789ABCDEF0u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for b in 0..(out_rows * nblocks) {
		let d = f32_to_fp16(((rnd() as f32) / 1e9 - 1.0) * 1.5);
		let dmin = f32_to_fp16(((rnd() as f32) / 1e9) * 0.3);
		raw[b * blk] = d as u8;
		raw[b * blk + 1] = (d >> 8) as u8;
		raw[b * blk + 2] = dmin as u8;
		raw[b * blk + 3] = (dmin >> 8) as u8;
		for i in 0..K_SCALE_SIZE {
			raw[b * blk + 4 + i] = (rnd() & 0x3f) as u8;
		}
		for i in 0..(QK_K / 2) {
			raw[b * blk + 4 + K_SCALE_SIZE + i] = (rnd() & 0xff) as u8;
		}
	}
	let w = LinearWeight::Q4K {
		raw: raw.clone(),
		out_dim: out_rows,
		in_dim,
	};
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();
	let got = apply_linear(&w, &x, out_rows, in_dim);
	// q8-dequantized reference (see gemv_q4k_matches_dequant).
	let x_q8 = quantize_q8k(&x, in_dim);
	let mut exp = vec![0.0f32; out_rows];
	for o in 0..out_rows {
		let row = dequantize_q4_k(&raw[o * nblocks * blk..(o + 1) * nblocks * blk], in_dim);
		let mut s = 0.0f32;
		for k in 0..in_dim {
			let xh = x_q8.ds[k / QK_K] * x_q8.qs[k] as f32;
			s += row[k] * xh;
		}
		exp[o] = s;
	}
	for o in 0..out_rows {
		let tol = (exp[o].abs() * 1e-3 + 1e-2).max(1e-2);
		assert!(
			(got[o] - exp[o]).abs() < tol,
			"apply_linear_q4k[{o}] got {} exp {} tol {}",
			got[o],
			exp[o],
			tol
		);
	}
}

// ---- concat_weights: fused GEMV equals two separate GEMVs ----
#[test]
fn concat_matches_separate() {
	let in_dim = 5120usize;
	let q4blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let make_q4 = |out_rows: usize, seed: u64| -> LinearWeight {
		let mut r = seed;
		let mut rd = || {
			r = r
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			(r >> 33) as u32
		};
		let mut raw = vec![0u8; out_rows * (in_dim / QK_K) * q4blk];
		for b in 0..(out_rows * in_dim / QK_K) {
			let d = f32_to_fp16(((rd() as f32) / 1e9 - 1.0) * 1.5);
			let dmin = f32_to_fp16(((rd() as f32) / 1e9) * 0.3);
			raw[b * q4blk] = d as u8;
			raw[b * q4blk + 1] = (d >> 8) as u8;
			raw[b * q4blk + 2] = dmin as u8;
			raw[b * q4blk + 3] = (dmin >> 8) as u8;
			for i in 0..K_SCALE_SIZE {
				raw[b * q4blk + 4 + i] = (rd() & 0x3f) as u8;
			}
			for i in 0..(QK_K / 2) {
				raw[b * q4blk + 4 + K_SCALE_SIZE + i] = (rd() & 0xff) as u8;
			}
		}
		LinearWeight::Q4K {
			raw,
			out_dim: out_rows,
			in_dim,
		}
	};
	let make_q6 = |out_rows: usize| -> LinearWeight {
		let nb = in_dim / QK_K;
		let mut raw = vec![0u8; out_rows * nb * 210];
		for b in 0..(out_rows * nb) {
			raw[b * 210 + 208] = 0x00;
			raw[b * 210 + 209] = 0x3C;
			for i in 0..16 {
				raw[b * 210 + 192 + i] = ((i % 3) as i8) as u8;
			}
			for i in 0..128 {
				raw[b * 210 + i] = (i % 16) as u8;
			}
			for i in 0..64 {
				raw[b * 210 + 128 + i] = (i % 4) as u8;
			}
		}
		LinearWeight::Q6K {
			raw,
			out_dim: out_rows,
			in_dim,
		}
	};
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();
	for (mk_a, mk_b, tag) in [
		(make_q4(4096, 12345), make_q4(2048, 67890), "q4+q4"),
		(make_q6(4096), make_q6(2048), "q6+q6"),
	] {
		let oa = mk_a.out_dim();
		let ob = mk_b.out_dim();
		let fused = concat_weights(&mk_a, &mk_b).expect("concat");
		let ga = apply_linear(&mk_a, &x, oa, in_dim);
		let gb = apply_linear(&mk_b, &x, ob, in_dim);
		let gf = apply_linear(&fused, &x, oa + ob, in_dim);
		let tol = 1e-2;
		for o in 0..oa {
			assert!((gf[o] - ga[o]).abs() < tol, "{tag}: fused[{o}] vs a");
		}
		for o in 0..ob {
			assert!(
				(gf[oa + o] - gb[o]).abs() < tol,
				"{tag}: fused[oa+{o}] vs b"
			);
		}
	}
}

// ---- fused Q4_K GEMV throughput (synthetic, mirrors a big decode matmul) ----
#[test]
fn bench_gemv_q4k_local() {
	let avail = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(1);
	eprintln!("AVAIL_PARALLELISM={avail}");
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let nblocks = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	for (i, v) in raw.iter_mut().enumerate() {
		*v = ((i * 31 + 7) & 0xff) as u8;
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
	let mut y = vec![0.0f32; out_rows];
	for _ in 0..3 {
		gemv_q4k(out_rows, in_dim, &x, &raw, &mut y);
	}
	let iters = 10usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		gemv_q4k(out_rows, in_dim, &x, &raw, &mut y);
	}
	let dt = t0.elapsed().as_secs_f64();
	let per_ms = dt / iters as f64 * 1000.0;
	let q4k_bytes = (out_rows * in_dim / 2) as f64; // 4-bit weights
	let bw = q4k_bytes / 1e9 / (per_ms / 1000.0);
	let gflops = (out_rows as f64 * in_dim as f64 * 2.0) / 1e9 / (per_ms / 1000.0);
	let avx2 = is_x86_feature_detected!("avx2");
	let tok_s = 13.5e9 / (bw * 1e9);
	println!(
		"BENCH gemv_q4k avx2={avx2} {out_rows}x{in_dim}: {per_ms:.2} ms/matmul  {gflops:.1} GFLOP/s  {bw:.1} GB/s weight  => ~{tok_s:.2} tok/s @13.5GB/token"
	);
}

// ---- streaming throughput: many distinct Q4_K matrices (mimics the real
// model, which streams ~13.5 GB of per-layer weights per token as ~640
// separate matmuls). Isolates sustained DRAM bandwidth vs the single
// cached-matrix bench above. ----
#[test]
fn bench_gemv_q4k_stream() {
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let nblocks = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let per = out_rows * nblocks * blk; // ~50 MB
	let matrices: usize = 24; // ~1.2 GB total working set
	let mut all = vec![0u8; matrices * per];
	for (i, v) in all.iter_mut().enumerate() {
		*v = ((i * 31 + 7) & 0xff) as u8;
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
	let mut y = vec![0.0f32; out_rows];
	// warmup (let the allocator/page cache settle)
	for m in 0..matrices {
		gemv_q4k(out_rows, in_dim, &x, &all[m * per..(m + 1) * per], &mut y);
	}
	let iters = 5usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		for m in 0..matrices {
			gemv_q4k(out_rows, in_dim, &x, &all[m * per..(m + 1) * per], &mut y);
		}
	}
	let dt = t0.elapsed().as_secs_f64();
	let total_bytes = (matrices * per * iters) as f64;
	let bw = total_bytes / 1e9 / dt;
	println!(
		"BENCH gemv_q4k STREAM {}x{} x{matrices} mats: {:.2} GB/s sustained ({} ms total)",
		out_rows,
		in_dim,
		bw,
		dt * 1000.0
	);
}

// ---- fused Q6_K dequant-GEMV matches dequantize + dot ----
#[test]
fn gemv_q6k_matches_dequant() {
	let out_rows = 16usize;
	let in_dim = 512usize; // multiple of QK_K
	let nblocks = in_dim / QK_K;
	let blk = q6k_block_bytes();
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for b in 0..(out_rows * nblocks) {
		for i in 0..128 {
			raw[b * blk + i] = rnd() as u8;
		} // ql
		for i in 0..64 {
			raw[b * blk + 128 + i] = rnd() as u8;
		} // qh
		for i in 0..16 {
			raw[b * blk + 192 + i] = (rnd() & 0xff) as u8;
		} // scales (int8)
		let d = f32_to_fp16(((rnd() as f32) / 1e9 - 1.0) * 2.0);
		raw[b * blk + 208] = d as u8;
		raw[b * blk + 209] = (d >> 8) as u8;
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
	let mut got = vec![0.0f32; out_rows];
	gemv_q6k(out_rows, in_dim, &x, &raw, &mut got);
	// The fused kernel quantizes the activation to int8 (q8_K); validate the
	// integer dot against the dequantized + q8-reconstructed reference.
	let x_q8 = quantize_q8k(&x, in_dim);
	let mut exp = vec![0.0f32; out_rows];
	for o in 0..out_rows {
		let row = dequantize_q6_k(&raw[o * nblocks * blk..(o + 1) * nblocks * blk], in_dim);
		let mut s = 0.0f32;
		for k in 0..in_dim {
			let xh = x_q8.ds[k / QK_K] * x_q8.qs[k] as f32;
			s += row[k] * xh;
		}
		exp[o] = s;
	}
	for o in 0..out_rows {
		let tol = (exp[o].abs() * 1e-3 + 1e-2).max(1e-2);
		assert!(
			(got[o] - exp[o]).abs() < tol,
			"gemv_q6k[{o}] got {} exp {} tol {}",
			got[o],
			exp[o],
			tol
		);
	}
}

// ---- AVX2 q6k kernel must equal the scalar reference ----
#[test]
fn q6k_avx2_matches_scalar() {
	let in_dim = 256usize;
	let nblocks = in_dim / QK_K;
	let blk = q6k_block_bytes();
	let mut raw = vec![0u8; nblocks * blk];
	let mut rng = 0x5A5A5A5A5A5A5A5Au64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for i in 0..128 {
		raw[i] = rnd() as u8;
	}
	for i in 0..64 {
		raw[128 + i] = rnd() as u8;
	}
	for i in 0..16 {
		raw[192 + i] = (rnd() & 0xff) as u8;
	}
	let d = f32_to_fp16(1.7);
	raw[208] = d as u8;
	raw[209] = (d >> 8) as u8;
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();
	let q8 = quantize_q8k(&x, in_dim);
	let s = q6k_dot_q8k_scalar(&raw, nblocks, &q8);
	// Reference: dequantize the q6 row and dot with the int8 activation.
	let row = dequantize_q6_k(&raw, in_dim);
	let mut exp = 0.0f32;
	for k in 0..in_dim {
		let xh = q8.ds[k / QK_K] * q8.qs[k] as f32;
		exp += row[k] * xh;
	}
	eprintln!("q6k scalar={s} ref={exp}");
	assert!((s - exp).abs() < 1e-2, "q6k scalar {s} vs ref {exp}");
}

// ---- apply_linear on a Q6K weight matches dequant + dot ----
#[test]
fn apply_linear_q6k_matches_gemv() {
	let out_rows = 8usize;
	let in_dim = 256usize;
	let nblocks = in_dim / QK_K;
	let blk = q6k_block_bytes();
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	let mut rng = 0x123456789ABCDEF0u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for b in 0..(out_rows * nblocks) {
		for i in 0..128 {
			raw[b * blk + i] = rnd() as u8;
		}
		for i in 0..64 {
			raw[b * blk + 128 + i] = rnd() as u8;
		}
		for i in 0..16 {
			raw[b * blk + 192 + i] = (rnd() & 0xff) as u8;
		}
		let d = f32_to_fp16(((rnd() as f32) / 1e9 - 1.0) * 1.5);
		raw[b * blk + 208] = d as u8;
		raw[b * blk + 209] = (d >> 8) as u8;
	}
	let w = LinearWeight::Q6K {
		raw: raw.clone(),
		out_dim: out_rows,
		in_dim,
	};
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect();
	let got = apply_linear(&w, &x, out_rows, in_dim);
	let x_q8 = quantize_q8k(&x, in_dim);
	let mut exp = vec![0.0f32; out_rows];
	for o in 0..out_rows {
		let row = dequantize_q6_k(&raw[o * nblocks * blk..(o + 1) * nblocks * blk], in_dim);
		let mut s = 0.0f32;
		for k in 0..in_dim {
			let xh = x_q8.ds[k / QK_K] * x_q8.qs[k] as f32;
			s += row[k] * xh;
		}
		exp[o] = s;
	}
	for o in 0..out_rows {
		let tol = (exp[o].abs() * 1e-3 + 1e-2).max(1e-2);
		assert!(
			(got[o] - exp[o]).abs() < tol,
			"apply_linear_q6k[{o}] got {} exp {} tol {}",
			got[o],
			exp[o],
			tol
		);
	}
}

// ---- fused Q6_K GEMV throughput (synthetic, mirrors a big decode matmul) ----
#[test]
fn bench_gemv_q6k_local() {
	let avail = std::thread::available_parallelism()
		.map(|n| n.get())
		.unwrap_or(1);
	eprintln!("AVAIL_PARALLELISM={avail}");
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let nblocks = in_dim / QK_K;
	let blk = q6k_block_bytes();
	let mut raw = vec![0u8; out_rows * nblocks * blk];
	for (i, v) in raw.iter_mut().enumerate() {
		*v = ((i * 31 + 7) & 0xff) as u8;
	}
	let x: Vec<f32> = (0..in_dim).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
	let mut y = vec![0.0f32; out_rows];
	for _ in 0..3 {
		gemv_q6k(out_rows, in_dim, &x, &raw, &mut y);
	}
	let iters = 10usize;
	let t0 = std::time::Instant::now();
	for _ in 0..iters {
		gemv_q6k(out_rows, in_dim, &x, &raw, &mut y);
	}
	let dt = t0.elapsed().as_secs_f64();
	let per_ms = dt / iters as f64 * 1000.0;
	let q6k_bytes = (out_rows * in_dim / 256 * 210) as f64;
	let bw = q6k_bytes / 1e9 / (per_ms / 1000.0);
	let gflops = (out_rows as f64 * in_dim as f64 * 2.0) / 1e9 / (per_ms / 1000.0);
	let avx2 = is_x86_feature_detected!("avx2");
	println!(
		"BENCH gemv_q6k avx2={avx2} {out_rows}x{in_dim}: {per_ms:.2} ms/matmul  {gflops:.1} GFLOP/s  {bw:.1} GB/s weight"
	);
}

#[test]
fn avxvnni_kernels_match_scalar() {
	// Regression: the AVX-VNNI (vpdpbusd, VEX) micro-kernels must execute on
	// the real CPU (a CPU lacking avxvnni would #UD / crash) and produce the
	// same result as the scalar reference. Skipped on non-avxvnni hosts so the
	// suite stays green elsewhere.
	if !is_x86_feature_detected!("avxvnni") {
		eprintln!("skip avxvnni_kernels_match_scalar: avxvnni not available");
		return;
	}
	let in_dim = QK_K;
	let nb = 1usize;
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};

	// Q4_K row: d=1.0, dmin=1.0, random scales + nibbles.
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut q4 = vec![0u8; blk];
	q4[0] = 0x00;
	q4[1] = 0x3C;
	q4[2] = 0x00;
	q4[3] = 0x3C;
	for b in &mut q4[4..] {
		*b = rnd() as u8;
	}
	let x: Vec<f32> = (0..in_dim)
		.map(|_| ((rnd() % 100) as f32 - 50.0) * 0.01)
		.collect();
	let q8 = quantize_q8k(&x, in_dim);

	let got4 = unsafe { q4k_dot_q8k_avxvnni(&q4, nb, &q8) };
	let exp4 = q4k_dot_q8k_scalar(&q4, nb, &q8);
	assert!(
		(got4 - exp4).abs() < 1e-2,
		"q4k avxvnni {got4} vs scalar {exp4}"
	);

	// Q6_K row (uses two sub-scales per 32-weight vector).
	let q6blk = q6k_block_bytes();
	let mut q6 = vec![0u8; q6blk];
	for b in &mut q6 {
		*b = rnd() as u8;
	}
	let n = q6.len();
	q6[n - 4] = 0x00;
	q6[n - 3] = 0x3C;
	q6[n - 2] = 0x00;
	q6[n - 1] = 0x3C;
	let got6 = unsafe { q6k_dot_q8k_avxvnni(&q6, nb, &q8) };
	let exp6 = q6k_dot_q8k_scalar(&q6, nb, &q8);
	assert!(
		(got6 - exp6).abs() < 1e-2,
		"q6k avxvnni {got6} vs scalar {exp6}"
	);
}

// ---- BENCH: batched GEMM vs N x per-row GEMV (prefill win check) ----
#[test]
fn bench_batched_vs_perrow() {
	// Representative big prefill matmul: ffn_up [17408 x 5120] Q4_K, seq 1..32.
	let out_rows = 17408usize;
	let in_dim = 5120usize;
	let nb = in_dim / QK_K;
	let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
	let mut raw = vec![0u8; out_rows * nb * blk];
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(rng >> 33) as u32
	};
	for r in raw.chunks_mut(blk) {
		r[0] = 0x00;
		r[1] = 0x3C;
		r[2] = 0x00;
		r[3] = 0x3C;
		for b in r[4..].iter_mut() {
			*b = rnd() as u8;
		}
	}
	let iters = 3usize;
	for seq in [1usize, 4, 16, 32] {
		let input: Vec<f32> = (0..seq * in_dim)
			.map(|_| ((rnd() % 100) as f32 - 50.0) * 0.01)
			.collect();
		// batched
		let mut out_b = vec![0.0f32; seq * out_rows];
		let t0 = std::time::Instant::now();
		for _ in 0..iters {
			gemm_q4k(out_rows, in_dim, seq, &raw, &input, &mut out_b);
		}
		let dt_b = t0.elapsed().as_secs_f64() / iters as f64;
		// per-row N x gemv
		let mut out_p = vec![0.0f32; seq * out_rows];
		let t0 = std::time::Instant::now();
		for _ in 0..iters {
			for t in 0..seq {
				let mut row = vec![0.0f32; out_rows];
				gemv_q4k(
					out_rows,
					in_dim,
					&input[t * in_dim..(t + 1) * in_dim],
					&raw,
					&mut row,
				);
				out_p[t * out_rows..(t + 1) * out_rows].copy_from_slice(&row);
			}
		}
		let dt_p = t0.elapsed().as_secs_f64() / iters as f64;
		let wbytes = (out_rows * nb * blk) as f64;
		let bwb = wbytes / 1e9 / dt_b;
		let bwp = wbytes / 1e9 / dt_p;
		println!(
			"BENCH prefill seq={seq}: batched {:.1} ms ({:.1} GB/s)  perrow {:.1} ms ({:.1} GB/s)  speedup {:.2}x",
			dt_b * 1000.0, bwb, dt_p * 1000.0, bwp, dt_p / dt_b
		);
	}
}

#[test]
fn par_for_covers_all_indices() {
	// Stripe partitioning must visit every index exactly once and never
	// race on disjoint writes. The closure stays `Fn` (par_for calls it
	// from many threads) by reconstructing the slice from a raw pointer.
	let n = 1000usize;
	let mut out = vec![0u32; n];
	let ptr = out.as_mut_ptr() as usize;
	par_for(n, |i| {
		let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, n) };
		slice[i] = (i as u32) * 3 + 1;
	});
	for i in 0..n {
		assert_eq!(out[i], (i as u32) * 3 + 1, "index {i} not set");
	}
}

/// The AVX2 activation quantizer must be **bit-identical** to the scalar
/// reference: the fused Q4_K/Q6_K dots consume `qs`/`bsums`/`ds` as integers,
/// so a one-ULP disagreement in the rounding would shift every logit.
#[test]
fn q8k_avx2_matches_scalar() {
	if !is_x86_feature_detected!("avx2") {
		return;
	}
	let n = 256 * 7;
	let mut x = vec![0.0f32; n];
	let mut st = 0x1234_5678u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	for v in x.iter_mut() {
		// Mixed magnitudes plus exact .5 ties, which is where a naive
		// truncate-based round would diverge from `f32::round`.
		let r = (next() % 2001) as f32 / 100.0 - 10.0;
		*v = r;
	}
	// one all-zero super-block (the `amax == 0` early-out) and some exact ties
	for j in 0..256 {
		x[256 + j] = 0.0;
	}
	x[0] = 12.7; // makes iscale exactly -10 => .5 ties below
	for j in 1..64 {
		x[j] = (j as f32) * 0.05;
	}
	let a = super::quantize_q8k(&x, n);
	let b = super::q8k::quantize_q8k_scalar(&x, n);
	assert_eq!(a.qs, b.qs, "qs differs");
	assert_eq!(a.bsums, b.bsums, "bsums differs");
	assert_eq!(a.ds, b.ds, "ds differs");
}

/// What one pool fork-join actually costs. Decode issues ~575 of them per
/// token, so this number times 575 is the floor the whole CPU path sits on.
#[test]
fn bench_dispatch_overhead() {
	use std::time::Instant;
	let n = 20_000;
	// near-empty payload: measures the barrier, not the work
	let t = Instant::now();
	for _ in 0..n {
		super::par_for(super::parallel::threads(), |i| {
			std::hint::black_box(i);
		});
	}
	let per = t.elapsed().as_secs_f64() / n as f64;
	println!(
		"empty par_for dispatch: {:.2} us  (x575/token = {:.1} ms)",
		per * 1e6,
		per * 575.0 * 1000.0
	);

	// with a realistic per-shard payload (~64 KB touch), to show the ramp
	let mut buf = vec![0.0f32; 24 * 16384];
	let addr = buf.as_mut_ptr() as usize;
	let t = Instant::now();
	for _ in 0..2000 {
		super::par_for(super::parallel::threads(), |i| {
			let p =
				unsafe { std::slice::from_raw_parts_mut((addr as *mut f32).add(i * 16384), 16384) };
			for v in p.iter_mut() {
				*v = *v * 1.000001 + 1.0;
			}
		});
	}
	println!(
		"par_for + 64 KB/shard: {:.2} us",
		t.elapsed().as_secs_f64() / 2000.0 * 1e6
	);
	std::hint::black_box(&buf);
}

/// `apply_linear_group` must be bit-identical to calling `apply_linear` on each
/// job: it only concatenates the output row space, it does not change any dot.
#[test]
fn linear_group_matches_per_job() {
	let in_dim = 512usize;
	let mut st = 0x9E37_79B9u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let x: Vec<f32> = (0..in_dim)
		.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
		.collect();
	let nb = in_dim / super::QK_K;
	let q4_row = nb * (2 + 2 + super::K_SCALE_SIZE + super::QK_K / 2);
	let q6_row = nb * super::q6k_block_bytes();
	// mixed types and unequal out_dims, so the row->job mapping is exercised
	let (o0, o1, o2) = (33usize, 8usize, 17usize);
	let w0: Vec<u8> = (0..o0 * q4_row).map(|_| (next() % 256) as u8).collect();
	let w1: Vec<u8> = (0..o1 * q6_row).map(|_| (next() % 256) as u8).collect();
	let w2: Vec<u8> = (0..o2 * q4_row).map(|_| (next() % 256) as u8).collect();
	let lw0 = super::LinearWeight::Q4K {
		raw: w0,
		out_dim: o0,
		in_dim,
	};
	let lw1 = super::LinearWeight::Q6K {
		raw: w1,
		out_dim: o1,
		in_dim,
	};
	let lw2 = super::LinearWeight::Q4K {
		raw: w2,
		out_dim: o2,
		in_dim,
	};

	let jobs: Vec<(&super::LinearWeight, usize)> = vec![(&lw0, o0), (&lw1, o1), (&lw2, o2)];
	let got = super::apply_linear_group(&jobs, &x, in_dim);
	for (i, (w, od)) in jobs.iter().enumerate() {
		let want = super::apply_linear(w, &x, *od, in_dim);
		// Bitwise: random bytes make some f16 scales NaN, and `NaN != NaN`
		// would mask a real match. Bit-identity is the contract anyway.
		let gb: Vec<u32> = got[i].iter().map(|v| v.to_bits()).collect();
		let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
		assert_eq!(gb, wb, "job {i} differs");
	}
}

/// The decode row split's CPU half: every row range must be bit-identical to
/// the same rows of the unsplit GEMV, including empty and full ranges.
#[test]
fn linear_group_rows_matches_slices() {
	let in_dim = 512usize;
	let mut st = 0x1234_5679u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let x: Vec<f32> = (0..in_dim)
		.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
		.collect();
	let nb = in_dim / super::QK_K;
	let q4_row = nb * (2 + 2 + super::K_SCALE_SIZE + super::QK_K / 2);
	let q6_row = nb * super::q6k_block_bytes();
	let (o0, o1, o2, o3) = (33usize, 8usize, 17usize, 12usize);
	let w0: Vec<u8> = (0..o0 * q4_row).map(|_| (next() % 256) as u8).collect();
	let w1: Vec<u8> = (0..o1 * q6_row).map(|_| (next() % 256) as u8).collect();
	let w2: Vec<u8> = (0..o2 * q4_row).map(|_| (next() % 256) as u8).collect();
	let w3: Vec<u8> = (0..o3 * q6_row).map(|_| (next() % 256) as u8).collect();
	let lw0 = super::LinearWeight::Q4K {
		raw: w0,
		out_dim: o0,
		in_dim,
	};
	let lw1 = super::LinearWeight::Q6K {
		raw: w1,
		out_dim: o1,
		in_dim,
	};
	let lw2 = super::LinearWeight::Q4K {
		raw: w2,
		out_dim: o2,
		in_dim,
	};
	let lw3 = super::LinearWeight::Q6K {
		raw: w3,
		out_dim: o3,
		in_dim,
	};
	// partial, empty (at the end), full, empty-at-start-of-range
	let jobs: Vec<(&super::LinearWeight, usize, usize)> =
		vec![(&lw0, 12, 33), (&lw1, 8, 8), (&lw2, 0, 17), (&lw3, 4, 4)];
	let got = super::apply_linear_group_rows(&jobs, &x, in_dim);
	for (i, (w, r0, r1)) in jobs.iter().enumerate() {
		let od = [o0, o1, o2, o3][i];
		let want = &super::apply_linear(w, &x, od, in_dim)[*r0..*r1];
		let gb: Vec<u32> = got[i].iter().map(|v| v.to_bits()).collect();
		let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
		assert_eq!(gb, wb, "job {i} differs");
	}
}

/// `attn_scores` must agree with a per-row scalar dot (it reassociates, so this
/// is a tolerance not an equality), and must handle a strided K cache.
#[test]
fn attn_scores_matches_scalar() {
	let hd = 256usize;
	let n_kv = 4usize;
	let rows = 300usize;
	let stride = hd * n_kv;
	let mut st = 0x2468_ACE1u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let q: Vec<f32> = (0..hd)
		.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
		.collect();
	let kb: Vec<f32> = (0..rows * stride)
		.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
		.collect();
	let scale = 0.0625f32;
	for g in 0..n_kv {
		let mut got = vec![0.0f32; rows];
		super::attn_scores(&q, &kb, stride, g * hd, rows, scale, &mut got);
		for j in 0..rows {
			let k = &kb[j * stride + g * hd..j * stride + (g + 1) * hd];
			let mut want = 0.0f32;
			for d in 0..hd {
				want += q[d] * k[d];
			}
			want *= scale;
			assert!(
				(got[j] - want).abs() <= 1e-4,
				"g={g} j={j}: {} vs {}",
				got[j],
				want
			);
		}
	}
}

/// Scalar vs 4-accumulator AVX2 on the real attention shape.
#[test]
fn bench_attn_scores() {
	use std::time::Instant;
	let (hd, n_kv, rows) = (256usize, 4usize, 4096usize);
	let stride = hd * n_kv;
	let q = vec![0.7f32; hd];
	let kb = vec![0.3f32; rows * stride];
	let mut out = vec![0.0f32; rows];
	let iters = 200;
	let t = Instant::now();
	for _ in 0..iters {
		super::attn_scores(&q, &kb, stride, 0, rows, 0.0625, &mut out);
		std::hint::black_box(&out);
	}
	let e = t.elapsed().as_secs_f64() / iters as f64;
	println!(
		"BENCH attn_scores {rows}x{hd}: {:.3} ms  {:.1} GFLOP/s",
		e * 1000.0,
		2.0 * (rows * hd) as f64 / e / 1e9
	);
	let t = Instant::now();
	for _ in 0..iters {
		for j in 0..rows {
			let k = &kb[j * stride..j * stride + hd];
			let mut s = 0.0f32;
			for d in 0..hd {
				s += q[d] * k[d];
			}
			out[j] = s * 0.0625;
		}
		std::hint::black_box(&out);
	}
	let e2 = t.elapsed().as_secs_f64() / iters as f64;
	println!(
		"BENCH attn_scores SCALAR: {:.3} ms  ({:.2}x slower)",
		e2 * 1000.0,
		e2 / e
	);
}

/// The pool hands work out through a shared cursor that `dispatch` rewinds.
/// Two back-to-back dispatches must therefore each visit every index: the
/// property that broke for a one-thread pool, where the reset was skipped and
/// every dispatch after the first became a no-op.
#[test]
fn par_for_visits_every_index_on_consecutive_dispatches() {
	use std::sync::atomic::{AtomicUsize, Ordering};
	const N: usize = 733;
	for round in 0..3 {
		let hits: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
		super::par_for(N, |i| {
			hits[i].fetch_add(1, Ordering::Relaxed);
		});
		for (i, h) in hits.iter().enumerate() {
			assert_eq!(h.load(Ordering::Relaxed), 1, "round {round}, index {i}");
		}
	}
}

/// `apply_linear_group_exact` is `dequant(W) · x` in f32: checked against an
/// f64 dot over the dequantized rows, on the decode `ssm_beta`/`ssm_alpha`
/// shape (two 48 x 5120 Q4_K, large enough to go through the pool) plus a Q6_K
/// job for the fallback arm. It must also beat the q8_K path's accuracy, which
/// is its reason to exist.
#[test]
fn linear_group_exact_matches_f64_dequant() {
	let in_dim = 5120usize;
	let nb = in_dim / super::QK_K;
	let mut st = 0x2545_f491u32;
	let mut next = || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		st
	};
	let x: Vec<f32> = (0..in_dim)
		.map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
		.collect();
	let q4_blk = 2 + 2 + super::K_SCALE_SIZE + super::QK_K / 2;
	let q6_blk = super::q6k_block_bytes();
	let mut rand_blocks = |rows: usize, blk: usize, d_at: &[usize]| -> Vec<u8> {
		let mut v: Vec<u8> = (0..rows * nb * blk).map(|_| (next() % 256) as u8).collect();
		// Finite, modest fp16 scales (random bytes can be inf/NaN).
		for b in v.chunks_mut(blk) {
			for &o in d_at {
				let h = super::f32_to_fp16(0.001 + (b[o] as f32) / 25600.0);
				b[o..o + 2].copy_from_slice(&h.to_le_bytes());
			}
		}
		v
	};
	let beta = super::LinearWeight::Q4K {
		raw: rand_blocks(48, q4_blk, &[0, 2]),
		out_dim: 48,
		in_dim,
	};
	let alpha = super::LinearWeight::Q4K {
		raw: rand_blocks(48, q4_blk, &[0, 2]),
		out_dim: 48,
		in_dim,
	};
	let q6 = super::LinearWeight::Q6K {
		raw: rand_blocks(8, q6_blk, &[q6_blk - 2]),
		out_dim: 8,
		in_dim,
	};
	let jobs: Vec<(&super::LinearWeight, usize)> = vec![(&beta, 48), (&alpha, 48), (&q6, 8)];
	let got = super::apply_linear_group_exact(&jobs, &x, in_dim);
	let (mut err_exact, mut err_q8) = (0f64, 0f64);
	for (i, (w, od)) in jobs.iter().enumerate() {
		let (raw, ty) = match w {
			super::LinearWeight::Q4K { raw, .. } => (raw, super::GgmlType::Q4_K),
			super::LinearWeight::Q6K { raw, .. } => (raw, super::GgmlType::Q6_K),
			_ => unreachable!(),
		};
		let deq = super::dequant::dequantize(ty, raw, od * in_dim);
		let q8 = super::apply_linear(w, &x, *od, in_dim);
		assert_eq!(got[i].len(), *od);
		for o in 0..*od {
			let want: f64 = deq[o * in_dim..(o + 1) * in_dim]
				.iter()
				.zip(&x)
				.map(|(a, b)| *a as f64 * *b as f64)
				.sum();
			let scale = deq[o * in_dim..(o + 1) * in_dim]
				.iter()
				.map(|a| a.abs() as f64)
				.sum::<f64>()
				.max(1e-6);
			let e = (got[i][o] as f64 - want).abs() / scale;
			assert!(
				e < 1e-5,
				"job {i} row {o}: got {} want {want} (rel {e:e})",
				got[i][o]
			);
			err_exact = err_exact.max(e);
			err_q8 = err_q8.max((q8[o] as f64 - want).abs() / scale);
		}
	}
	assert!(
		err_exact * 10.0 < err_q8,
		"exact {err_exact:e} vs q8_K {err_q8:e}"
	);
}

/// `par_range_pre` runs its hook exactly once and still covers every index,
/// both when the range is dispatched to the pool and when it is small enough
/// to run inline.
#[test]
fn par_range_pre_runs_hook_once() {
	use std::sync::atomic::{AtomicUsize, Ordering};
	for (n, unit) in [(733usize, 4096usize), (5, 16), (0, 1)] {
		let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
		let pre = AtomicUsize::new(0);
		super::par_range_pre(
			n,
			unit,
			|| {
				pre.fetch_add(1, Ordering::Relaxed);
			},
			|s, e| {
				for h in &hits[s..e] {
					h.fetch_add(1, Ordering::Relaxed);
				}
			},
		);
		assert_eq!(pre.load(Ordering::Relaxed), 1, "n={n}");
		assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
	}
}

/// Two-pass reference for `attn_causal`: full score row, max, exp, normalize.
fn attn_causal_ref(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
) -> Vec<f32> {
	attn_ref(q, k, v, nq, t0, n_head, n_kv, hd, scale, 0)
}

/// Keys `< full` are visible to every query besides its causal range.
#[allow(clippy::too_many_arguments)]
fn attn_ref(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	nq: usize,
	t0: usize,
	n_head: usize,
	n_kv: usize,
	hd: usize,
	scale: f32,
	full: usize,
) -> Vec<f32> {
	let (qs, ks, hpg) = (n_head * hd, n_kv * hd, n_head / n_kv);
	let mut out = vec![0.0f32; nq * qs];
	for t in 0..nq {
		let n = (t0 + t + 1).max(full);
		for h in 0..n_head {
			let g = h / hpg;
			let qr = &q[t * qs + h * hd..t * qs + (h + 1) * hd];
			let mut s: Vec<f64> = (0..n)
				.map(|j| {
					let kr = &k[j * ks + g * hd..j * ks + (g + 1) * hd];
					qr.iter()
						.zip(kr)
						.map(|(a, b)| *a as f64 * *b as f64)
						.sum::<f64>() * scale as f64
				})
				.collect();
			let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
			let mut sum = 0.0;
			for x in &mut s {
				*x = (*x - mx).exp();
				sum += *x;
			}
			for d in 0..hd {
				let mut a = 0.0f64;
				for j in 0..n {
					a += s[j] * v[j * ks + g * hd + d] as f64;
				}
				out[t * qs + h * hd + d] = (a / sum) as f32;
			}
		}
	}
	out
}

/// Tiled online-softmax attention vs the two-pass reference, over the shapes
/// that take distinct paths: decode (split-K), short verify batches, partial
/// tiles, block-unaligned causal limits, a non-6 head group and the scalar
/// fallback (`hd` not a multiple of 16).
#[test]
fn attn_causal_matches_reference() {
	let mut st = 0x1357_9BDFu32;
	let mut next = move || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		(st % 2000) as f32 / 1000.0 - 1.0
	};
	for &(nq, t0, n_head, n_kv, hd) in &[
		(1usize, 0usize, 24usize, 4usize, 256usize),
		(1, 3000, 24, 4, 256),
		(4, 1500, 24, 4, 256),
		(37, 100, 24, 4, 256),
		(200, 0, 24, 4, 256),
		(70, 65, 6, 2, 64),
		(9, 700, 4, 4, 32),
		(13, 40, 6, 3, 24),
	] {
		let tk = t0 + nq;
		let q: Vec<f32> = (0..nq * n_head * hd).map(|_| next() * 2.0).collect();
		let k: Vec<f32> = (0..tk * n_kv * hd).map(|_| next() * 2.0).collect();
		let v: Vec<f32> = (0..tk * n_kv * hd).map(|_| next()).collect();
		let scale = 1.0 / (hd as f32).sqrt();
		let want = attn_causal_ref(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale);
		let mut got = vec![f32::NAN; nq * n_head * hd];
		super::attn_causal(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, &mut got);
		let worst = got
			.iter()
			.zip(&want)
			.map(|(a, b)| (a - b).abs())
			.fold(0.0f32, f32::max);
		assert!(
			worst <= 2e-5,
			"nq={nq} t0={t0} heads={n_head}/{n_kv} hd={hd}: max diff {worst}"
		);

		// Half-precision cache: same kernel, rows widened on load. Against the
		// reference run on the rounded values, the error budget is unchanged.
		let (mut kh, mut vh) = (Vec::new(), Vec::new());
		super::kv_extend(&mut kh, &k);
		super::kv_extend(&mut vh, &v);
		let kr: Vec<f32> = kh.iter().map(|&h| super::fp16_to_f32(h)).collect();
		let vr: Vec<f32> = vh.iter().map(|&h| super::fp16_to_f32(h)).collect();
		let want = attn_causal_ref(&q, &kr, &vr, nq, t0, n_head, n_kv, hd, scale);
		got.fill(f32::NAN);
		super::attn_causal(&q, &kh, &vh, nq, t0, n_head, n_kv, hd, scale, &mut got);
		let worst = got
			.iter()
			.zip(&want)
			.map(|(a, b)| (a - b).abs())
			.fold(0.0f32, f32::max);
		assert!(
			worst <= 2e-5,
			"f16 nq={nq} t0={t0} heads={n_head}/{n_kv} hd={hd}: max diff {worst}"
		);
	}
}

/// `attn_block`: an image block sees every key before it and the whole
/// block (a diffusion transformer's block-causal attention).
#[test]
fn attn_block_matches_reference() {
	let mut st = 0x2468_ACE1u32;
	let mut next = move || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		(st % 2000) as f32 / 1000.0 - 1.0
	};
	for &(nq, t0, n_head, n_kv, hd) in &[
		(1usize, 0usize, 4usize, 4usize, 128usize),
		(300, 0, 4, 4, 128),
		(900, 20, 2, 2, 64),
		(130, 37, 8, 8, 128),
		(9, 700, 4, 2, 32),
		(70, 65, 6, 3, 24),
	] {
		let tk = t0 + nq;
		let q: Vec<f32> = (0..nq * n_head * hd).map(|_| next() * 2.0).collect();
		let k: Vec<f32> = (0..tk * n_kv * hd).map(|_| next() * 2.0).collect();
		let v: Vec<f32> = (0..tk * n_kv * hd).map(|_| next()).collect();
		let scale = 1.0 / (hd as f32).sqrt();
		let want = attn_ref(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, tk);
		let mut got = vec![f32::NAN; nq * n_head * hd];
		super::attn_block(&q, &k, &v, nq, t0, n_head, n_kv, hd, scale, &mut got);
		let worst = got
			.iter()
			.zip(&want)
			.map(|(a, b)| (a - b).abs())
			.fold(0.0f32, f32::max);
		assert!(
			worst <= 2e-5,
			"nq={nq} t0={t0} heads={n_head}/{n_kv} hd={hd}: max diff {worst}"
		);
	}
}

/// `attn_block_rows`: a block's queries run in slices give the whole
/// block's result (the image transformer's attention per 256-row chunk).
#[test]
fn attn_block_rows_slices_match_reference() {
	let mut st = 0x1357_9BDFu32;
	let mut next = move || {
		st ^= st << 13;
		st ^= st >> 17;
		st ^= st << 5;
		(st % 2000) as f32 / 1000.0 - 1.0
	};
	for &(nq, t0, nh, hd, sl) in &[
		(1024usize, 24usize, 4usize, 128usize, 256usize),
		(300, 0, 2, 64, 128),
		(130, 37, 3, 32, 50),
	] {
		let tk = t0 + nq;
		let q: Vec<f32> = (0..nq * nh * hd).map(|_| next() * 2.0).collect();
		let k: Vec<f32> = (0..tk * nh * hd).map(|_| next() * 2.0).collect();
		let v: Vec<f32> = (0..tk * nh * hd).map(|_| next()).collect();
		let scale = 1.0 / (hd as f32).sqrt();
		let want = attn_ref(&q, &k, &v, nq, t0, nh, nh, hd, scale, tk);
		let mut got = vec![f32::NAN; nq * nh * hd];
		let w = nh * hd;
		for r0 in (0..nq).step_by(sl) {
			let r1 = (r0 + sl).min(nq);
			super::attn_block_rows(
				&q[r0 * w..r1 * w],
				&k,
				&v,
				r1 - r0,
				t0 + r0,
				nh,
				nh,
				hd,
				scale,
				tk,
				&mut got[r0 * w..r1 * w],
			);
		}
		let worst = got
			.iter()
			.zip(&want)
			.map(|(a, b)| (a - b).abs())
			.fold(0.0f32, f32::max);
		assert!(
			worst <= 2e-5,
			"nq={nq} t0={t0} slice={sl}: max diff {worst}"
		);
	}
}

/// `kv_extend` rounds like a correctly rounded f32 -> half conversion,
/// keeps subnormals, and clamps out-of-range values to the largest half.
#[test]
fn kv_extend_rounds_and_clamps() {
	let xs: Vec<f32> = (0..1000)
		.map(|i| (i as f32 * 0.7311).sin() * 10f32.powi(i % 13 - 8))
		.chain([
			1e9,
			-1e9,
			65519.0,
			3e-8,
			-6e-6,
			0.0,
			-0.0,
			1.0 + 1.0 / 2048.0,
		])
		.collect();
	let mut h = Vec::new();
	super::kv_extend(&mut h, &xs);
	assert_eq!(h.len(), xs.len());
	for (&x, &hb) in xs.iter().zip(&h) {
		let y = super::fp16_to_f32(hb);
		let c = x.clamp(-65504.0, 65504.0);
		// Exactly the nearest half: no other half is closer.
		let up = super::fp16_to_f32(hb.wrapping_add(1));
		let dn = super::fp16_to_f32(hb.wrapping_sub(1));
		assert!(y.is_finite(), "{x} -> {y}");
		assert!(
			(c - y).abs() <= (c - up).abs() || !up.is_finite() || (hb & 0x7fff) == 0x7bff,
			"{x} -> {y} vs {up}"
		);
		assert!(
			(c - y).abs() <= (c - dn).abs() || (hb & 0x7fff) == 0,
			"{x} -> {y} vs {dn}"
		);
	}
}

/// Prefill-shaped attention throughput (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn bench_attn_causal() {
	use std::time::Instant;
	let (n_head, n_kv, hd) = (24usize, 4usize, 256usize);
	fn run<T: super::KvElem>(
		tag: &str,
		k: &[T],
		v: &[T],
		nq: usize,
		t0: usize,
		n_head: usize,
		n_kv: usize,
		hd: usize,
	) {
		let q = vec![0.01f32; nq * n_head * hd];
		let mut out = vec![0.0f32; nq * n_head * hd];
		super::attn_causal(&q, k, v, nq, t0, n_head, n_kv, hd, 0.0625, &mut out);
		let reps = if nq == 1 { 50 } else { 3 };
		let t = Instant::now();
		for _ in 0..reps {
			super::attn_causal(&q, k, v, nq, t0, n_head, n_kv, hd, 0.0625, &mut out);
		}
		let s = t.elapsed().as_secs_f64() / reps as f64;
		let keys = nq as f64 * (t0 as f64 + (nq as f64 + 1.0) / 2.0);
		let flop = 4.0 * n_head as f64 * hd as f64 * keys;
		let bytes = 2.0 * (t0 + nq) as f64 * (n_kv * hd * std::mem::size_of::<T>()) as f64;
		println!(
			"attn_causal {tag} nq={nq} t0={t0}: {:.2} ms, {:.0} GFLOP/s, cache {:.1} GB/s",
			s * 1e3,
			flop / s / 1e9,
			bytes / s / 1e9
		);
	}
	for &(nq, t0) in &[
		(2048usize, 0usize),
		(2048, 8192),
		(1, 10240),
		(1, 32768),
		(8, 32768),
		(1, 131072),
	] {
		let tk = t0 + nq;
		let k: Vec<f32> = (0..tk * n_kv * hd)
			.map(|i| ((i % 97) as f32) * 1e-3)
			.collect();
		let v = k.clone();
		run("f32", &k, &v, nq, t0, n_head, n_kv, hd);
		let mut kh = Vec::new();
		super::kv_extend(&mut kh, &k);
		let vh = kh.clone();
		drop((k, v));
		run("f16", &kh, &vh, nq, t0, n_head, n_kv, hd);
	}
}

/// `delta_net_cols` over any column split (including the AVX2 widths and a
/// scalar tail) must equal the plain scalar formula **bit for bit**: the
/// prefill scan hands each worker a column block, and the decode path runs the
/// whole head through the same function.
#[test]
fn delta_net_cols_split_is_bit_exact() {
	let s = 128usize;
	let mut rng = 0x2545F4914F6CDD1Du64;
	let mut rnd = || {
		rng ^= rng << 13;
		rng ^= rng >> 7;
		rng ^= rng << 17;
		((rng >> 40) as f32) / (1u32 << 23) as f32 - 1.0
	};
	let st0: Vec<f32> = (0..s * s).map(|_| rnd()).collect();
	let (k, q, v): (Vec<f32>, Vec<f32>, Vec<f32>) = (
		(0..s).map(|_| rnd()).collect(),
		(0..s).map(|_| rnd()).collect(),
		(0..s).map(|_| rnd()).collect(),
	);
	let (beta, g, qscale) = (0.37f32, 0.91f32, 1.0 / (s as f32).sqrt());

	// Reference: the original scalar two-pass form, three steps deep so the
	// updated state is exercised too.
	let mut st_ref = st0.clone();
	let mut o_ref = vec![0.0f32; s];
	for _ in 0..3 {
		let mut sk = vec![0.0f32; s];
		for i in 0..s {
			for j in 0..s {
				let val = st_ref[i * s + j] * g;
				st_ref[i * s + j] = val;
				sk[j] += val * k[i];
			}
		}
		let d: Vec<f32> = (0..s).map(|j| (v[j] - sk[j]) * beta).collect();
		for i in 0..s {
			let qi = q[i] * qscale;
			for j in 0..s {
				let val = st_ref[i * s + j] + k[i] * d[j];
				st_ref[i * s + j] = val;
				o_ref[j] += val * qi;
			}
		}
	}

	for &blk in &[128usize, 64, 32, 8, 40, 3] {
		let mut st = st0.clone();
		let mut o = vec![0.0f32; s];
		for _ in 0..3 {
			let mut j0 = 0;
			while j0 < s {
				let jb = blk.min(s - j0);
				super::delta_net_cols(
					&mut st,
					s,
					j0,
					&k,
					&q,
					&v[j0..j0 + jb],
					beta,
					g,
					qscale,
					&mut o[j0..j0 + jb],
				);
				j0 += jb;
			}
		}
		assert!(
			st.iter()
				.zip(&st_ref)
				.all(|(a, b)| a.to_bits() == b.to_bits()),
			"state, block {blk}"
		);
		assert!(
			o.iter()
				.zip(&o_ref)
				.all(|(a, b)| a.to_bits() == b.to_bits()),
			"output, block {blk}"
		);

		// The prefill scan's form: each block copied out to a compact
		// `s x jb` buffer, scanned there (`ld = jb`), copied back.
		let mut st = st0.clone();
		let mut o = vec![0.0f32; s];
		let mut j0 = 0;
		while j0 < s {
			let jb = blk.min(s - j0);
			let mut c: Vec<f32> = (0..s)
				.flat_map(|i| st[i * s + j0..i * s + j0 + jb].to_vec())
				.collect();
			for _ in 0..3 {
				super::delta_net_cols_ld(
					&mut c,
					jb,
					s,
					0,
					&k,
					&q,
					&v[j0..j0 + jb],
					beta,
					g,
					qscale,
					&mut o[j0..j0 + jb],
				);
			}
			for i in 0..s {
				st[i * s + j0..i * s + j0 + jb].copy_from_slice(&c[i * jb..(i + 1) * jb]);
			}
			j0 += jb;
		}
		assert!(
			st.iter()
				.zip(&st_ref)
				.all(|(a, b)| a.to_bits() == b.to_bits()),
			"compact state, block {blk}"
		);
		assert!(
			o.iter()
				.zip(&o_ref)
				.all(|(a, b)| a.to_bits() == b.to_bits()),
			"compact output, block {blk}"
		);
	}
}

/// `par_silu_mul_q8k` is `par_quantize_q8k` of `silu_mul`, byte for byte.
#[test]
fn silu_mul_q8k_matches_two_pass() {
	let (seq, n) = (5usize, 3 * 256);
	let gate: Vec<f32> = (0..seq * n)
		.map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.013)
		.collect();
	let up: Vec<f32> = (0..seq * n)
		.map(|i| ((i * 53 % 997) as f32 - 498.0) * 0.007)
		.collect();
	let mut h = vec![0.0f32; seq * n];
	for t in 0..seq {
		super::silu_mul(
			&gate[t * n..(t + 1) * n],
			&up[t * n..(t + 1) * n],
			&mut h[t * n..(t + 1) * n],
		);
	}
	let a = super::par_quantize_q8k(&h, seq, n);
	let b = super::par_silu_mul_q8k(&gate, &up, seq, n);
	for (x, y) in a.iter().zip(&b) {
		assert!(x
			.ds
			.iter()
			.zip(&y.ds)
			.all(|(p, q)| p.to_bits() == q.to_bits()));
		assert_eq!(x.qs, y.qs);
		assert_eq!(x.bsums, y.bsums);
	}
}

/// `par_rows_q8k` is `par_quantize_q8k` of the rows its closure writes, byte
/// for byte, at a length that takes the pool (more than `SMALL_ROWS`).
#[test]
fn rows_q8k_matches_two_pass() {
	let (seq, n) = (37usize, 2 * 256);
	let x: Vec<f32> = (0..seq * n)
		.map(|i| ((i * 29 % 1009) as f32 - 504.0) * 0.011)
		.collect();
	let a = super::par_quantize_q8k(&x, seq, n);
	let b = super::par_rows_q8k(seq, n, |t, r| r.copy_from_slice(&x[t * n..(t + 1) * n]));
	for (p, q) in a.iter().zip(&b) {
		assert!(p
			.ds
			.iter()
			.zip(&q.ds)
			.all(|(u, v)| u.to_bits() == v.to_bits()));
		assert_eq!(p.qs, q.qs);
		assert_eq!(p.bsums, q.bsums);
	}
}

/// The recycling allocator hands a freed large block back for the same
/// layout, zeroes it for `alloc_zeroed`, and never keeps more than its bound.
#[test]
fn recycling_alloc_reuses_and_zeroes() {
	use std::alloc::{GlobalAlloc, Layout};
	let a = super::alloc::RecyclingAlloc::new();
	let big = Layout::from_size_align(4 << 20, 64).unwrap();
	unsafe {
		let p = a.alloc(big);
		std::ptr::write_bytes(p, 0xAB, big.size());
		a.dealloc(p, big);
		let z = a.alloc_zeroed(big);
		assert_eq!(z, p, "same layout reuses the block");
		assert!(std::slice::from_raw_parts(z, big.size())
			.iter()
			.all(|&b| b == 0));
		// A different size is not a hit.
		let other = Layout::from_size_align(3 << 20, 64).unwrap();
		let o = a.alloc(other);
		assert_ne!(o, z);
		a.dealloc(o, other);
		a.dealloc(z, big);
		// Small blocks bypass the cache entirely.
		let small = Layout::from_size_align(4096, 8).unwrap();
		let s = a.alloc(small);
		a.dealloc(s, small);
	}
}

#[test]
fn q6k_quantize_round_trips() {
	// Mixed magnitudes per 16-group, an all-zero group, and a sign flip of
	// the extreme so both ends of the asymmetric range are exercised.
	let (rows, cols) = (3usize, 512usize);
	let mut x = vec![0.0f32; rows * cols];
	let mut s = 0x1234_5678u32;
	for (i, v) in x.iter_mut().enumerate() {
		s = s.wrapping_mul(1664525).wrapping_add(1013904223);
		let u = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
		let g = (i / 16) % 7;
		*v = if g == 3 {
			0.0
		} else {
			u * (g as f32 + 0.1) * if i % 97 == 0 { -9.0 } else { 1.0 }
		};
	}
	let q = crate::quantize_q6_k(&x, rows, cols);
	assert_eq!(q.len(), rows * cols / 256 * 210);
	let y = crate::dequantize_q6_k(&q, rows * cols);
	let absmax = |v: &[f32]| v.iter().fold(0.0f32, |m, v| m.max(v.abs()));
	for g in 0..rows * cols / 16 {
		let xs = &x[g * 16..g * 16 + 16];
		let block_max = absmax(&x[g / 16 * 256..g / 16 * 256 + 256]);
		for j in 0..16 {
			let e = (y[g * 16 + j] - xs[j]).abs();
			// One step of the group's own scale (a value of the opposite sign
			// to the group's extreme clamps at +31), plus the 8-bit scale's
			// rounding, whose step is set by the whole block's largest value.
			assert!(
				e <= absmax(xs) / 32.0 * 1.1 + block_max / 128.0 + 1e-6,
				"group {g} elem {j}: {} vs {}",
				y[g * 16 + j],
				xs[j]
			);
		}
	}
	// Without outliers the error is at most one group step.
	let x: Vec<f32> = (0..256)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
		.collect();
	let mut q = [0u8; 210];
	crate::quantize_q6_k_block(&x, &mut q);
	let y = crate::dequantize_q6_k(&q, 256);
	let e = x
		.iter()
		.zip(&y)
		.fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
	assert!(e < 1.0 / 32.0 * 1.1, "max err {e}");
	// Typical weight magnitudes put the super-scale in fp16's subnormal range
	// (block max < 0.25); it must not flush to zero.
	let x: Vec<f32> = x.iter().map(|v| v * 0.01).collect();
	crate::quantize_q6_k_block(&x, &mut q);
	let y = crate::dequantize_q6_k(&q, 256);
	let e = x
		.iter()
		.zip(&y)
		.fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
	assert!(e < 0.01 / 32.0 * 1.1, "subnormal-scale max err {e}");
}

#[test]
fn q4k_quantize_round_trips() {
	// Weight-like magnitudes (block max < 0.25 puts d/dmin in f16's subnormal
	// range), all-positive / all-negative / mixed sub-blocks.
	let x: Vec<f32> = (0..1024)
		.map(|i| {
			let u = ((i * 37 % 101) as f32 - 50.0) / 50.0;
			match (i / 32) % 4 {
				0 => u * 0.02,
				1 => u.abs() * 0.05,
				2 => -u.abs() * 0.01,
				_ => u * 0.2,
			}
		})
		.collect();
	let q = crate::quantize_q4_k(&x, 2, 512);
	assert_eq!(q.len(), 4 * 144);
	let y = crate::dequantize_q4_k(&q, 1024);
	for s in 0..32 {
		let xs = &x[s * 32..s * 32 + 32];
		let lo = xs.iter().fold(0.0f32, |m, &v| m.min(v));
		let hi = xs.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
		let bmax = x[s / 8 * 256..s / 8 * 256 + 256]
			.iter()
			.fold(0.0f32, |m, v| m.max(v.abs()));
		for j in 0..32 {
			let e = (y[s * 32 + j] - xs[j]).abs();
			// Half a 4-bit step, plus the 6-bit scale/min codes' rounding.
			assert!(
				e <= (hi - lo) / 15.0 * 0.6 + bmax / 63.0 + 1e-6,
				"sub {s} elem {j}: {} vs {}",
				y[s * 32 + j],
				xs[j]
			);
		}
	}
}

#[test]
fn silu_mul_matches_scalar() {
	let n = 1003;
	let g: Vec<f32> = (0..n)
		.map(|i| (i as f32 - 500.0) * 0.07 + if i % 97 == 0 { 120.0 } else { 0.0 })
		.collect();
	let u: Vec<f32> = (0..n)
		.map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
		.collect();
	let mut out = vec![0.0f32; n];
	crate::silu_mul(&g, &u, &mut out);
	for i in 0..n {
		let r = (g[i] as f64 / (1.0 + (-(g[i] as f64)).exp())) * u[i] as f64;
		let err = (out[i] as f64 - r).abs();
		assert!(
			err <= 1e-6 * r.abs().max(1e-3),
			"i={i} g={} got {} want {r}",
			g[i],
			out[i]
		);
	}
}

#[test]
fn conv_silu_matches_scalar() {
	let (ksz, c_n) = (4, 203);
	let w: Vec<f32> = (0..ksz * c_n)
		.map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.05)
		.collect();
	let rows: Vec<Vec<f32>> = (0..ksz)
		.map(|f| {
			(0..c_n)
				.map(|c| ((c * 7 + f * 13) % 23) as f32 * 0.1 - 1.1)
				.collect()
		})
		.collect();
	let rr: Vec<&[f32]> = rows.iter().map(|r| &r[..]).collect();
	let mut out = vec![0.0f32; c_n];
	crate::conv_silu(&rr, &w, &mut out);
	let mut x = rows[0].clone();
	crate::silu_mul_assign(&rows[1], &mut x);
	for c in 0..c_n {
		let s: f64 = (0..ksz)
			.map(|f| w[f * c_n + c] as f64 * rows[f][c] as f64)
			.sum();
		let r = s / (1.0 + (-s).exp());
		assert!(
			(out[c] as f64 - r).abs() <= 1e-6 * r.abs().max(1e-2),
			"c={c} got {} want {r}",
			out[c]
		);
		let g = rows[1][c] as f64;
		let r2 = rows[0][c] as f64 * g / (1.0 + (-g).exp());
		assert!(
			(x[c] as f64 - r2).abs() <= 1e-6 * r2.abs().max(1e-2),
			"c={c}"
		);
	}
}

/// A width cap must still cover every index exactly once, and switching it
/// between dispatches (parked, excluded and spinning workers mixed) must not
/// lose a wake-up or run a shard twice.
#[test]
fn width_cap_covers_every_index() {
	use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
	for cap in [0, 1, 2, 5, 0, 3, 16, 0] {
		let _w = super::parallel::limit_width(cap);
		for _ in 0..300 {
			let hits: Vec<AtomicUsize> = (0..997).map(|_| AtomicUsize::new(0)).collect();
			super::par_for(hits.len(), |i| {
				hits[i].fetch_add(1, Relaxed);
			});
			assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "cap {cap}");
		}
		// Let the excluded workers reach their park.
		std::thread::sleep(std::time::Duration::from_millis(2));
	}
}

/// Multi-head (one head per KV group) block attention at the image
/// transformer's 1 MP shape (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn bench_attn_block_mha() {
	use std::time::Instant;
	let (nq, t0, nh, hd) = (4096usize, 36usize, 32usize, 128usize);
	let tk = t0 + nq;
	let mut st = 7u64;
	let mut next = || {
		st = st
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((st >> 33) % 2000) as f32 / 1000.0 - 1.0
	};
	let q: Vec<f32> = (0..nq * nh * hd).map(|_| next()).collect();
	let k: Vec<f32> = (0..tk * nh * hd).map(|_| next()).collect();
	let v: Vec<f32> = (0..tk * nh * hd).map(|_| next()).collect();
	let mut out = vec![0f32; nq * nh * hd];
	super::attn_block(&q, &k, &v, nq, t0, nh, nh, hd, 0.088, &mut out);
	let t = Instant::now();
	for _ in 0..3 {
		super::attn_block(&q, &k, &v, nq, t0, nh, nh, hd, 0.088, &mut out);
	}
	let s = t.elapsed().as_secs_f64() / 3.0;
	let flop = 4.0 * (nh * hd) as f64 * nq as f64 * tk as f64;
	println!(
		"attn_block mha {nq}x{tk}x{nh}: {:.1} ms, {:.0} GFLOP/s",
		s * 1e3,
		flop / s / 1e9
	);
}

/// Random f16 rows `[n][w]`, with channel 5 of every head offset by `bias`
/// (K's large constant channels).
fn k8_rows(n: usize, nh: usize, hd: usize, seed: u64, bias: f32) -> Vec<u16> {
	let mut st = seed;
	(0..n * nh * hd)
		.map(|i| {
			st = st
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			let x = ((st >> 33) % 2000) as f32 / 1000.0 - 1.0;
			let x = if i % hd == 5 { x + bias } else { x };
			super::f32_to_fp16(x)
		})
		.collect()
}

/// `attn_block_rows_k8` against the float kernel on the same half K / V:
/// int8 Q.K error only (a few 1e-3 of the output's RMS).
#[test]
fn attn_block_rows_k8_matches_float() {
	if !super::K8::supported(128) {
		return;
	}
	let (nq, t0, nh, hd) = (37usize, 51usize, 4usize, 128usize);
	let tk = t0 + nq + 3;
	let k = k8_rows(tk, nh, hd, 3, 20.0);
	let v = k8_rows(tk, nh, hd, 5, 3.0);
	let qh = k8_rows(nq, nh, hd, 9, 0.0);
	let q: Vec<f32> = qh.iter().map(|&h| super::fp16_to_f32(h) * 3.0).collect();
	let mut k8 = super::K8::default();
	k8.pack(&k, tk, nh, hd);
	let scale = 1.0 / (hd as f32).sqrt();
	for (full, pv) in [(t0 + nq, false), (tk, false), (t0 + nq, true), (tk, true)] {
		if pv {
			k8.pack_v(&v);
		}
		let mut want = vec![0f32; nq * nh * hd];
		let mut got = vec![0f32; nq * nh * hd];
		super::attn_block_rows(&q, &k, &v, nq, t0, nh, nh, hd, scale, full, &mut want);
		super::attn_block_rows_k8(&q, &k, &k8, &v, nq, t0, nh, hd, scale, full, &mut got);
		let rms = (want.iter().map(|x| x * x).sum::<f32>() / want.len() as f32).sqrt();
		let err = (want
			.iter()
			.zip(&got)
			.map(|(a, b)| (a - b) * (a - b))
			.sum::<f32>()
			/ want.len() as f32)
			.sqrt();
		println!("k8 full {full} int8 V {pv}: rel {:.2e}", err / rms);
		assert!(
			err / rms < if pv { 3e-2 } else { 1e-2 },
			"rel {}",
			err / rms
		);
	}
}

/// `K8::begin` + `pack_rows` / `pack_v_rows` in pieces at a foreign mean
/// (another set of keys'): the same attention as the one-shot pack within
/// the int8 error, the pieces' order and split irrelevant, `key_mean` the
/// keys' own means.
#[test]
fn k8_pack_in_pieces() {
	if !super::K8::supported(128) {
		return;
	}
	let (nq, nh, hd) = (29usize, 3usize, 128usize);
	let tk = 77usize;
	let k = k8_rows(tk, nh, hd, 3, 20.0);
	let v = k8_rows(tk, nh, hd, 5, 3.0);
	let other = k8_rows(tk, nh, hd, 7, 18.0);
	let q: Vec<f32> = k8_rows(nq, nh, hd, 9, 0.0)
		.iter()
		.map(|&h| super::fp16_to_f32(h) * 3.0)
		.collect();
	let mut one = super::K8::default();
	one.pack(&k, tk, nh, hd);
	one.pack_v(&v);
	let mut ref8 = super::K8::default();
	ref8.pack(&other, tk, nh, hd);
	let mean = ref8.key_mean();
	let pieces = |cuts: &[usize]| {
		let mut k8 = super::K8::default();
		// a stale, larger packing first: buffers are reused
		let big = k8_rows(tk + 13, nh, hd, 11, 5.0);
		k8.pack(&big, tk + 13, nh, hd);
		k8.pack_v(&big);
		k8.begin(tk, nh, hd, &mean, true);
		for w in cuts.windows(2).rev() {
			k8.pack_rows(&k, w[0], w[1]);
			k8.pack_v_rows(&v, w[0], w[1]);
		}
		k8
	};
	let a = pieces(&[0, tk]);
	let b = pieces(&[0, 5, 21, 22, 64, tk]);
	let scale = 1.0 / (hd as f32).sqrt();
	let run = |k8: &super::K8| {
		let mut o = vec![0f32; nq * nh * hd];
		super::attn_block_rows_k8(&q, &k, k8, &v, nq, 0, nh, hd, scale, tk, &mut o);
		o
	};
	let (oa, ob, o1) = (run(&a), run(&b), run(&one));
	assert_eq!(oa, ob, "the split changed the result");
	let mut want = vec![0f32; nq * nh * hd];
	super::attn_block_rows(&q, &k, &v, nq, 0, nh, nh, hd, scale, tk, &mut want);
	let rel = |x: &[f32]| {
		let rms = (want.iter().map(|x| x * x).sum::<f32>() / want.len() as f32).sqrt();
		(want
			.iter()
			.zip(x)
			.map(|(a, b)| (a - b) * (a - b))
			.sum::<f32>()
			/ want.len() as f32)
			.sqrt() / rms
	};
	println!("one-shot {:.2e}, pieces {:.2e}", rel(&o1), rel(&oa));
	assert!(rel(&oa) < 3e-2 && rel(&oa) < 2.0 * rel(&o1) + 1e-3);
	let (ma, m1) = (b.key_mean(), one.key_mean());
	for (x, y) in ma.iter().zip(&m1) {
		assert!((x - y).abs() <= 1e-5 * y.abs().max(1.0), "{x} vs {y}");
	}
}

/// [`bench_attn_block_mha`] for the int8 Q.K kernel on half K / V, against
/// the float one on the same half K / V.
#[test]
#[ignore]
fn bench_attn_k8() {
	use std::time::Instant;
	let (nq, t0, nh, hd) = (4096usize, 36usize, 32usize, 128usize);
	let tk = t0 + nq;
	let k = k8_rows(tk, nh, hd, 3, 4.0);
	let v = k8_rows(tk, nh, hd, 5, 0.0);
	let q: Vec<f32> = k8_rows(nq, nh, hd, 9, 0.0)
		.iter()
		.map(|&h| super::fp16_to_f32(h))
		.collect();
	let mut out = vec![0f32; nq * nh * hd];
	let flop = 4.0 * (nh * hd) as f64 * nq as f64 * tk as f64;
	let mut k8 = super::K8::default();
	for round in 0..3 {
		let t = Instant::now();
		super::attn_block_rows(&q, &k, &v, nq, t0, nh, nh, hd, 0.088, tk, &mut out);
		let a = t.elapsed().as_secs_f64();
		let t = Instant::now();
		k8.pack(&k, tk, nh, hd);
		let p = t.elapsed().as_secs_f64();
		let t = Instant::now();
		super::attn_block_rows_k8(&q, &k, &k8, &v, nq, t0, nh, hd, 0.088, tk, &mut out);
		let b = t.elapsed().as_secs_f64();
		let t = Instant::now();
		k8.pack_v(&v);
		let pv = t.elapsed().as_secs_f64();
		let t = Instant::now();
		super::attn_block_rows_k8(&q, &k, &k8, &v, nq, t0, nh, hd, 0.088, tk, &mut out);
		let c = t.elapsed().as_secs_f64();
		println!(
			"round {round}: f16 {:.1} ms ({:.0} GFLOP/s), k8 {:.1} ms ({:.0}) + pack {:.1} ms, kv8 {:.1} ms ({:.0}) + pack {:.1} ms",
			a * 1e3,
			flop / a / 1e9,
			b * 1e3,
			flop / b / 1e9,
			p * 1e3,
			c * 1e3,
			flop / c / 1e9,
			pv * 1e3
		);
	}
}
