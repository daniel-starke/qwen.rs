//! CPU prefill GEMM microbenchmark: the A/B harness for `gemm_q4k` / `gemm_q6k`.
//!
//! End-to-end timings cannot tell a slow kernel from a slow wrapper (this is the
//! same lesson `gpubench` exists for on the GPU side). This runs the two real
//! prefill shapes in isolation and reports GFLOP/s, and splits out the serial
//! `q8_K` activation quantization that `gemm_q4k` does before its parallel
//! section, so the two are never confused for each other.
//!
//!     cargo run --release -p quant --example gemmbench
//!     cargo run --release -p quant --example gemmbench -- 17408 5120 2048 q4k

use std::time::Instant;

fn mkbytes(n: usize, seed: u64) -> Vec<u8> {
	// xorshift; contents are irrelevant to timing but must be deterministic so
	// two runs of the harness are comparable.
	let mut s = seed | 1;
	(0..n)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			(s >> 24) as u8
		})
		.collect()
}

fn mkinput(n: usize, seed: u64) -> Vec<f32> {
	let mut s = seed | 1;
	(0..n)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			((s >> 40) as f32 / 8388608.0) - 1.0
		})
		.collect()
}

fn bench(kind: &str, out_rows: usize, in_dim: usize, seq: usize, iters: usize) {
	let nb = in_dim / quant::QK_K;
	let blk = match kind {
		"q4k" => 2 + 2 + quant::K_SCALE_SIZE + quant::QK_K / 2,
		_ => quant::q6k_block_bytes(),
	};
	let w = mkbytes(out_rows * nb * blk, 0x1234_5678);
	let x = mkinput(seq * in_dim, 0x9abc_def0);
	let mut y = vec![0.0f32; seq * out_rows];

	// The q8_K activation quantization is inside the kernel; time it separately
	// so a change to it is never mistaken for a change to the GEMM.
	let t0 = Instant::now();
	for _ in 0..iters {
		let q: Vec<quant::Q8K> = (0..seq)
			.map(|t| quant::quantize_q8k(&x[t * in_dim..(t + 1) * in_dim], in_dim))
			.collect();
		std::hint::black_box(&q);
	}
	let q8_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

	let mut best = f64::MAX;
	for _ in 0..iters {
		let t = Instant::now();
		match kind {
			"q4k" => quant::gemm_q4k(out_rows, in_dim, seq, &w, &x, &mut y),
			_ => quant::gemm_q6k(out_rows, in_dim, seq, &w, &x, &mut y),
		}
		best = best.min(t.elapsed().as_secs_f64());
		std::hint::black_box(&y);
	}
	// Bit-level checksum of the whole output: two runs that differ only in
	// QWEN_GEMM_PTILE must print the same value, or the tiling is not bit-exact.
	let ck = y
		.iter()
		.fold(0u64, |a, v| a.rotate_left(1) ^ v.to_bits() as u64);
	let fl = 2.0 * seq as f64 * out_rows as f64 * in_dim as f64;
	println!(
		"  {kind}  N={out_rows:<6} K={in_dim:<6} seq={seq:<5}  {:8.1} ms  {:7.1} GFLOP/s   (serial q8_K ref {:.1} ms)",
		best * 1e3,
		fl / best / 1e9,
		q8_ms,
	);
	println!("        checksum {ck:#018x}");
}

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let iters: usize = std::env::var("ITERS")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(3);
	println!("threads={} iters={iters}", quant::parallel::threads());
	if a.len() >= 4 {
		bench(
			&a[3],
			a[0].parse().unwrap(),
			a[1].parse().unwrap(),
			a[2].parse().unwrap(),
			iters,
		);
		return;
	}
	// The two real prefill shapes of Qwen3.8-27B: n_embd 5120, n_ff 17408.
	bench("q4k", 17408, 5120, 2048, iters); // FfnUp / FfnGate
	bench("q6k", 5120, 17408, 2048, iters); // FfnDown
}
