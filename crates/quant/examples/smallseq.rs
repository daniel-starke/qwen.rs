//! Small-batch matmul cost: the speculative-decoding (MTP) verify pass runs
//! `1 + n_draft` positions through every weight, and it only pays if that costs
//! about what one decode token does. This compares the decode GEMV at `seq = 1`
//! against the prefill GEMM at `seq = 1..=8` on the real FFN shapes, cycling
//! through enough distinct matrices that every call streams its weight from
//! DRAM, as decode does.
//!
//!     cargo run --release -p quant --example smallseq

use std::time::Instant;

fn mkbytes(n: usize, seed: u64) -> Vec<u8> {
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

fn main() {
	// (kind, out, in): FfnUp (Q4_K) and FfnDown (Q6_K).
	for (kind, out_dim, in_dim) in [("q4k", 17408usize, 5120usize), ("q6k", 5120, 17408)] {
		let nb = in_dim / quant::QK_K;
		let blk = match kind {
			"q4k" => 2 + 2 + quant::K_SCALE_SIZE + quant::QK_K / 2,
			_ => quant::q6k_block_bytes(),
		};
		// ~1 GB of distinct matrices so nothing is cache-resident.
		let bytes = out_dim * nb * blk;
		let n_mat = (1usize << 30) / bytes + 1;
		let ws: Vec<quant::LinearWeight> = (0..n_mat)
			.map(|i| {
				let raw = mkbytes(bytes, 0x1234_5678 + i as u64);
				// Clear every f16 scale's exponent top bit so no NaN/inf.
				let mut raw = raw;
				for b in raw.chunks_mut(blk) {
					if kind == "q4k" {
						b[1] &= 0x3b;
						b[3] &= 0x3b;
					} else {
						let l = b.len();
						b[l - 1] &= 0x3b;
					}
				}
				if kind == "q4k" {
					quant::LinearWeight::Q4K {
						raw,
						out_dim,
						in_dim,
					}
				} else {
					quant::LinearWeight::Q6K {
						raw,
						out_dim,
						in_dim,
					}
				}
			})
			.collect();
		let x = mkinput(8 * in_dim, 0x9abc_def0);
		let reps = 3 * n_mat;
		let gb = bytes as f64 / 1e9;
		// warm-up
		for w in &ws {
			std::hint::black_box(quant::apply_linear(w, &x[..in_dim], out_dim, in_dim));
		}
		let t = Instant::now();
		for r in 0..reps {
			std::hint::black_box(quant::apply_linear(
				&ws[r % n_mat],
				&x[..in_dim],
				out_dim,
				in_dim,
			));
		}
		let gemv = t.elapsed().as_secs_f64() / reps as f64;
		println!(
			"{kind} {out_dim}x{in_dim}: gemv seq=1 {:7.3} ms ({:5.1} GB/s)",
			gemv * 1e3,
			gb / gemv
		);
		for seq in 1..=8 {
			let t = Instant::now();
			for r in 0..reps {
				std::hint::black_box(quant::apply_linear_batch(
					&ws[r % n_mat],
					&x[..seq * in_dim],
					seq,
					out_dim,
					in_dim,
				));
			}
			let s = t.elapsed().as_secs_f64() / reps as f64;
			println!(
				"{kind} {out_dim}x{in_dim}: gemm seq={seq} {:7.3} ms ({:5.2} x gemv, {:5.2} x gemv per token)",
				s * 1e3,
				s / gemv,
				s / gemv / seq as f64
			);
		}
	}
}
