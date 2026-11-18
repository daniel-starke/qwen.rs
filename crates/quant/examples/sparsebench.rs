//! Does skipping zero-activation columns turn into DRAM time on this host?
//!
//!     cargo run --release -p quant --example sparsebench -- [ROWS] [COLS]
//!
//! Defaults are the CPU share of a decode `ffn_down` (3380 x 17408 Q4_K).
//! Cycles through enough copies of the matrix to defeat the L3, and compares
//! the row-major q8_K GEMV with the column-major sparse GEMV at several
//! activation sparsities (smallest magnitudes zeroed).

use quant::sparse_cm::CmQ4K;
use std::time::Instant;

fn main() {
	let a: Vec<usize> = std::env::args()
		.skip(1)
		.filter_map(|s| s.parse().ok())
		.collect();
	let rows = a.first().copied().unwrap_or(3380);
	let cols = a.get(1).copied().unwrap_or(17408);
	let mut seed = 0x9E3779B97F4A7C15u64;
	let mut rnd = move || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		// Sum of uniforms: roughly Gaussian, heavy enough tails for Q4_K's min.
		(0..4)
			.map(|i| ((seed >> (i * 16)) & 0xFFFF) as f32 / 65535.0 - 0.5)
			.sum::<f32>()
	};
	let w: Vec<f32> = (0..rows * cols).map(|_| rnd() * 0.05).collect();
	let raw = quant::quantize_q4_k(&w, rows, cols);
	let wq = quant::dequantize_q4_k(&raw, rows * cols);
	drop(w);
	let n_copies = std::env::var("COPIES")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or((1usize << 30) / raw.len() + 1);
	println!(
		"{rows} x {cols} Q4_K, {:.1} MB each, {n_copies} copies",
		raw.len() as f64 / 1e6
	);
	let raws: Vec<Vec<u8>> = (0..n_copies).map(|_| raw.clone()).collect();
	let t = Instant::now();
	let cms: Vec<CmQ4K> = raws
		.iter()
		.map(|r| CmQ4K::from_q4k(r, cols, 0, rows))
		.collect();
	println!(
		"repack {:.1} ms per copy, CM {:.1} MB",
		t.elapsed().as_secs_f64() * 1e3 / n_copies as f64,
		cms[0].bytes() as f64 / 1e6
	);

	let x0: Vec<f32> = (0..cols)
		.map(|_| rnd() * (1.0 + 3.0 * rnd().abs()))
		.collect();

	// Correctness: dense x, CM vs f64 dot of the dequantized rows.
	let mut y = vec![0.0f32; rows];
	cms[0].gemv(&x0, &mut y);
	let mut maxrel = 0.0f64;
	for r in (0..rows).step_by(97) {
		let e: f64 = (0..cols)
			.map(|c| wq[r * cols + c] as f64 * x0[c] as f64)
			.sum();
		maxrel = maxrel.max((y[r] as f64 - e).abs() / e.abs().max(1e-3));
	}
	println!("check: max rel err vs f64 {maxrel:.2e}");

	let iters = 4 * n_copies;
	let time = |f: &mut dyn FnMut(usize)| {
		for i in 0..n_copies {
			f(i);
		}
		let t = Instant::now();
		for i in 0..iters {
			f(i % n_copies);
		}
		t.elapsed().as_secs_f64() * 1e3 / iters as f64
	};
	let dense = time(&mut |i| quant::gemv_q4k(rows, cols, &x0, &raws[i], &mut y));
	println!(
		"row-major q8_K GEMV        {dense:6.3} ms  ({:.1} GB/s)",
		raw.len() as f64 / dense / 1e6
	);
	for sp in [0.0f32, 0.3, 0.5, 0.6, 0.7, 0.8] {
		let mut x = x0.clone();
		let k = (cols as f32 * sp) as usize;
		if k > 0 {
			let mut m: Vec<f32> = x.iter().map(|v| v.abs()).collect();
			let (_, &mut thr, _) = m.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
			for v in x.iter_mut() {
				if v.abs() <= thr {
					*v = 0.0;
				}
			}
		}
		let mut sink = 0u64;
		let tms = time(&mut |i| sink ^= cms[i].touch(&x));
		let ms = time(&mut |i| cms[i].gemv(&x, &mut y));
		print!("[touch {tms:6.3} ms {sink:x}] ");
		let read = cms[0].bytes() as f64 * (0.135 + 0.865 * (1.0 - sp as f64));
		println!(
			"column-major, sparsity {:.1}  {ms:6.3} ms  ({:.2}x dense, ~{:.1} GB/s of needed bytes)",
			sp,
			dense / ms,
			read / ms / 1e6
		);
	}
}
