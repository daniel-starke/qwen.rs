//! Check and time `Dense`'s f32 GEMM (`gemm_f32.glsl`) on random data:
//!   cargo run --release -p offload --features vulkan --example densebench [adapter]
//! Prints the worst error against f64 sums on small products (plain,
//! subtracting, transposed B, overlapping buffers) and GFLOP/s at the shapes
//! of a blocked factorization.

#[cfg(feature = "vulkan")]
fn main() -> Result<(), String> {
	use offload::Dense;
	let a: Vec<String> = std::env::args().skip(1).collect();
	let mut g = Dense::new(a.first().map(|s| s.as_str()))?;
	let mut seed = 0x1234_5678_9abc_def0u64;
	let mut rnd = move |n: usize| -> Vec<f32> {
		(0..n)
			.map(|_| {
				seed ^= seed << 13;
				seed ^= seed >> 7;
				seed ^= seed << 17;
				((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
			})
			.collect()
	};
	// correctness: regions with offsets and strides inside one buffer
	let (m, n, k, ld) = (197usize, 131usize, 77usize, 300usize);
	let data = rnd(ld * 700);
	let buf = g.alloc(data.len())?;
	let (ao, bo, co) = (5 * ld + 3, 250 * ld + 11, 420 * ld + 7);
	let mut worst = 0f64;
	for (sub, tb) in [(false, false), (true, false), (false, true), (true, true)] {
		g.upload(buf, 0, &data)?;
		g.gemm(
			(buf, ao, ld),
			m,
			(buf, bo, ld),
			n,
			k,
			(buf, co, ld),
			sub,
			tb,
			false,
		)?;
		let mut out = vec![0f32; m * n];
		g.download(buf, co, ld, m, n, &mut out)?;
		for i in 0..m {
			for j in 0..n {
				let mut s = 0f64;
				for t in 0..k {
					let bv = if tb {
						data[bo + t * ld + j]
					} else {
						data[bo + j * ld + t]
					};
					s += data[ao + i * ld + t] as f64 * bv as f64;
				}
				let want = if sub {
					data[co + i * ld + j] as f64 - s
				} else {
					s
				};
				worst = worst.max((want - out[i * n + j] as f64).abs());
			}
		}
	}
	println!("worst |gpu - f64| over 4 modes: {worst:.3e}");
	g.free(buf);
	// speed: C[m][n] -= A[m][256] B[n][256]^T (a trailing update), and a
	// wide-k product
	for (m, n, k) in [
		(8192usize, 8192usize, 256usize),
		(16384, 16384, 256),
		(256, 8192, 8192),
		(4096, 4096, 4096),
	] {
		let (ab, bb, cb) = (g.alloc(m * k)?, g.alloc(n * k)?, g.alloc(m * n)?);
		g.upload(ab, 0, &rnd(m * k))?;
		g.upload(bb, 0, &rnd(n * k))?;
		g.gemm(
			(ab, 0, k),
			m,
			(bb, 0, k),
			n,
			k,
			(cb, 0, n),
			false,
			false,
			false,
		)?;
		let t = std::time::Instant::now();
		let reps = 3;
		for _ in 0..reps {
			g.gemm(
				(ab, 0, k),
				m,
				(bb, 0, k),
				n,
				k,
				(cb, 0, n),
				true,
				false,
				false,
			)?;
		}
		let dt = t.elapsed().as_secs_f64() / reps as f64;
		println!(
			"[{m} x {n}] k {k}: {:.1} ms, {:.0} GFLOP/s",
			dt * 1e3,
			2.0 * (m * n * k) as f64 / dt / 1e9
		);
		for b in [ab, bb, cb] {
			g.free(b);
		}
	}
	Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("densebench needs --features vulkan");
}
