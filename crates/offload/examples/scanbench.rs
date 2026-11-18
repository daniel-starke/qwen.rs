//! Check and time `DeltaScan`'s prefill recurrence (`delta_prefill_npu.glsl`)
//! on random data against an f64 reference:
//!   cargo run --release -p offload --features vulkan --example scanbench [adapter] [n] [rounds]
//! Two chunks of `n` positions run back to back (the second starts from the
//! first one's state); prints the worst relative error of `o` per value row
//! over a sample of heads and the GPU time per chunk (`QWEN_NPU_SCAN_TS`).

#[cfg(feature = "vulkan")]
fn main() {
	use offload::DeltaScan;
	const S: usize = 128;
	const HV: usize = 48;
	const HK: usize = 16;
	let a: Vec<String> = std::env::args().skip(1).collect();
	let pick = a.first().map(|s| s.as_str()).unwrap_or("intel");
	let n: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
	let rounds: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(40);
	let mut seed = 0x1234_5678_9abc_def0u64;
	let mut rnd = move || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		(seed >> 11) as f64 / (1u64 << 53) as f64
	};
	// 4 KiB-aligned host buffers, as the NPU's (imported, not copied)
	let alloc = |bytes: usize| -> &'static mut [u16] {
		let l = std::alloc::Layout::from_size_align(bytes.next_multiple_of(4096), 4096).unwrap();
		// SAFETY: non-zero size; leaked on purpose (the scan imports it)
		unsafe {
			let p = std::alloc::alloc_zeroed(l) as *mut u16;
			std::slice::from_raw_parts_mut(p, bytes / 2)
		}
	};
	let f = quant::f32_to_fp16;
	let h = |x: u16| quant::fp16_to_f32(x) as f64;
	let mk = |rnd: &mut dyn FnMut() -> f64| {
		let (q, k, v, gb) = (
			alloc(n * HK * S * 2),
			alloc(n * HK * S * 2),
			alloc(n * HV * S * 2),
			alloc(n * 2 * HV * 2),
		);
		for t in 0..n {
			for hh in 0..HK {
				for x in [&mut *q, &mut *k] {
					let r: Vec<f64> = (0..S).map(|_| rnd() - 0.5).collect();
					let l = r.iter().map(|v| v * v).sum::<f64>().sqrt();
					for i in 0..S {
						x[t * HK * S + hh * S + i] = f((r[i] / l) as f32);
					}
				}
			}
			for e in 0..HV * S {
				v[t * HV * S + e] = f((rnd() * 2.0 - 1.0) as f32);
			}
			for hh in 0..HV {
				gb[t * 2 * HV + hh] = f((-0.3 * rnd()) as f32);
				gb[t * 2 * HV + HV + hh] = f((0.2 + 0.8 * rnd()) as f32);
			}
		}
		(q, k, v, gb)
	};
	let chunks: Vec<_> = (0..2).map(|_| mk(&mut rnd)).collect();
	let o = alloc(n * HV * S * 2);
	let bytes = |x: &[u16]| -> &[u8] {
		unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) }
	};
	std::env::set_var("QWEN_NPU_SCAN_TS", "1");
	let mut sc = DeltaScan::new(Some(pick), 1).expect("scan device");
	let oscale = 1.0f32;
	// reference state per sampled head, [row = key dim][col]
	let heads = [0usize, 7, 25, 47];
	let mut st: Vec<Vec<f64>> = heads.iter().map(|_| vec![0f64; S * S]).collect();
	let mut worst = 0f64;
	for (q, k, v, gb) in &chunks {
		let ob = unsafe { std::slice::from_raw_parts_mut(o.as_mut_ptr() as *mut u8, o.len() * 2) };
		sc.scan_begin(0, n, bytes(q), bytes(k), bytes(v), bytes(gb), ob, oscale)
			.unwrap();
		sc.scan_end().unwrap();
		for (hi, &hh) in heads.iter().enumerate() {
			let s = &mut st[hi];
			let kh = hh / (HV / HK);
			for t in 0..n {
				let g = h(gb[t * 2 * HV + hh]).exp();
				let beta = h(gb[t * 2 * HV + HV + hh]);
				let kv: Vec<f64> = (0..S).map(|i| h(k[t * HK * S + kh * S + i])).collect();
				let qv: Vec<f64> = (0..S)
					.map(|i| h(q[t * HK * S + kh * S + i]) / (S as f64).sqrt())
					.collect();
				s.iter_mut().for_each(|x| *x *= g);
				let mut want = vec![0f64; S];
				for j in 0..S {
					let acc: f64 = (0..S).map(|i| s[i * S + j] * kv[i]).sum();
					let d = (h(v[t * HV * S + hh * S + j]) - acc) * beta;
					for i in 0..S {
						s[i * S + j] += kv[i] * d;
					}
					want[j] = (0..S).map(|i| s[i * S + j] * qv[i]).sum();
				}
				let nrm = want.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
				let got = &o[t * HV * S + hh * S..t * HV * S + (hh + 1) * S];
				let d = want
					.iter()
					.zip(got)
					.map(|(w, &x)| (w - h(x)).powi(2))
					.sum::<f64>()
					.sqrt();
				worst = worst.max(d / nrm);
			}
		}
	}
	println!("n {n}: worst relative error of o {worst:.2e} (two chunks, heads {heads:?})");
	// timing: the [scants] lines (GPU time) go to stderr; host time here
	let (q, k, v, gb) = &chunks[0];
	let ob = unsafe { std::slice::from_raw_parts_mut(o.as_mut_ptr() as *mut u8, o.len() * 2) };
	let mut ts = Vec::new();
	for _ in 0..rounds {
		let t = std::time::Instant::now();
		sc.scan_begin(0, n, bytes(q), bytes(k), bytes(v), bytes(gb), ob, oscale)
			.unwrap();
		sc.scan_end().unwrap();
		ts.push(t.elapsed().as_secs_f64() * 1e3);
	}
	ts.sort_by(f64::total_cmp);
	println!(
		"host ms per chunk: min {:.3}, median {:.3} ({rounds} rounds)",
		ts[0],
		ts[ts.len() / 2]
	);
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("scanbench needs --features vulkan");
}
