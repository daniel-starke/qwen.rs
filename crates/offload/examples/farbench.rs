//! Check and time `DeltaScan`'s far attention kernels on random data against
//! an f64 reference, the fused pair (`attn_far_qks_npu.glsl` +
//! `attn_far_pv_npu.glsl`, the default) and the older one
//! (`QWEN_NPU_FAR_FUSED=0`: `attn_far_qk_npu.glsl` + `attn_far_npu.glsl`):
//!   cargo run --release -p offload --features vulkan --example farbench [adapter] [s]
//! Prints per pair the worst relative error of `o / l` and of `m + ln l` over
//! a sample of rows (two blocks accumulated, in one submission and in two),
//! and ms per block.

#[cfg(feature = "vulkan")]
fn main() {
	use offload::DeltaScan;
	let a: Vec<String> = std::env::args().skip(1).collect();
	let pick = a.first().map(|s| s.as_str()).unwrap_or("intel");
	let s: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
	const CAP: usize = 1024;
	const NK: usize = CAP - 2;
	let mut seed = 0x1234_5678_9abc_def0u64;
	let mut rnd = move || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		seed
	};
	// random f16 bit patterns with |x| in [2^-4, 2^-1)
	let mut f16s = |n: usize| -> Vec<u16> {
		(0..n)
			.map(|_| {
				let r = rnd();
				(((r >> 40) & 1) << 15) as u16
					| ((11 + (r >> 20) % 3) << 10) as u16
					| (r & 0x3ff) as u16
			})
			.collect()
	};
	let val = |h: u16| -> f64 {
		let m = 1.0 + (h & 0x3ff) as f64 / 1024.0;
		let e = ((h >> 10) & 31) as i32 - 15;
		m * 2f64.powi(e) * if h >> 15 == 1 { -1.0 } else { 1.0 }
	};
	let q = f16s(24 * s * 256);
	let mut blocks: Vec<(Vec<u16>, Vec<u16>)> = (0..2)
		.map(|_| (f16s(CAP * 1024), f16s(CAP * 1024)))
		.collect();
	for (k, v) in blocks.iter_mut() {
		k[NK * 1024..].fill(0);
		v[NK * 1024..].fill(0);
	}
	let bytes = |x: &[u16]| -> &[u8] {
		unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) }
	};
	let bl: Vec<(&[u8], &[u8])> = blocks.iter().map(|(k, v)| (bytes(k), bytes(v))).collect();
	// the worst errors of o / l and m + ln l against the f64 reference
	let check = |o: &[f32], ml: &[[f32; 2]]| -> (f64, f64) {
		let (mut eo, mut el) = (0f64, 0f64);
		for i in 0..48 {
			let row = (i * 7919) % (24 * s);
			let g = row / (6 * s);
			let qr: Vec<f64> = q[row * 256..(row + 1) * 256]
				.iter()
				.map(|&x| val(x))
				.collect();
			let mut sc_ = Vec::new();
			for (k, _) in &blocks {
				for t in 0..NK {
					let kr = &k[t * 1024 + g * 256..t * 1024 + (g + 1) * 256];
					sc_.push(qr.iter().zip(kr).map(|(a, &b)| a * val(b)).sum::<f64>());
				}
			}
			let m = sc_.iter().cloned().fold(f64::MIN, f64::max);
			let z: f64 = sc_.iter().map(|x| (x - m).exp()).sum();
			let mut want = vec![0f64; 256];
			for (bi, (_, v)) in blocks.iter().enumerate() {
				for t in 0..NK {
					let p = (sc_[bi * NK + t] - m).exp() / z;
					let vr = &v[t * 1024 + g * 256..t * 1024 + (g + 1) * 256];
					want.iter_mut().zip(vr).for_each(|(w, &x)| *w += p * val(x));
				}
			}
			let [gm, gl] = ml[row];
			el = el.max(((gm as f64 + (gl as f64).ln()) - (m + z.ln())).abs());
			let n: f64 = want.iter().map(|x| x * x).sum::<f64>().sqrt();
			let d: f64 = want
				.iter()
				.zip(&o[row * 256..(row + 1) * 256])
				.map(|(w, &x)| (w - x as f64 / gl as f64).powi(2))
				.sum::<f64>()
				.sqrt();
			eo = eo.max(d / n);
		}
		(eo, el)
	};
	let scan = |fused: bool| {
		std::env::set_var("QWEN_NPU_FAR_FUSED", if fused { "1" } else { "0" });
		let sc = DeltaScan::new(Some(pick), 0).expect("scan device");
		std::env::remove_var("QWEN_NPU_FAR_FUSED");
		sc
	};
	let mut pairs = [("fused", scan(true)), ("two-pass", scan(false))];
	let mut outs = Vec::new();
	for (name, sc) in pairs.iter_mut() {
		sc.far_begin(s, NK, bytes(&q), &bl, true).unwrap();
		sc.far_wait().unwrap();
		let (o, ml) = sc.far_out();
		let (eo, el) = check(o, ml);
		let one = (o[..24 * s * 256].to_vec(), ml[..24 * s].to_vec());
		// the same two blocks as two submissions (the second joins the first)
		sc.far_begin(s, NK, bytes(&q), &bl[..1], true).unwrap();
		sc.far_wait().unwrap();
		sc.far_begin(s, NK, bytes(&q), &bl[1..], false).unwrap();
		sc.far_wait().unwrap();
		let (o, ml) = sc.far_out();
		let (eo2, el2) = check(o, ml);
		println!(
			"{name}, s {s}: worst relative error of o {eo:.2e} / {eo2:.2e}, \
			 of the log sum exp {el:.2e} / {el2:.2e} (one / two submissions)"
		);
		outs.push(one);
	}
	{
		let (a, b) = (&outs[0], &outs[1]);
		let mut d = 0f64;
		for r in 0..24 * s {
			let (la, lb) = (a.1[r][1] as f64, b.1[r][1] as f64);
			for i in r * 256..(r + 1) * 256 {
				d = d.max((a.0[i] as f64 / la - b.0[i] as f64 / lb).abs());
			}
		}
		println!("fused vs two-pass: largest difference of o / l {d:.2e}");
	}
	let sc = &mut pairs[0].1;
	// eviction of imported host memory: 12 block pairs, three per call, on a
	// device that keeps at most 16 imports, against one that keeps them all
	{
		let many: Vec<(Vec<u16>, Vec<u16>)> = (0..12)
			.map(|i| (blocks[i % 2].0.clone(), blocks[(i + 1) % 2].1.clone()))
			.collect();
		let ml: Vec<(&[u8], &[u8])> = many.iter().map(|(k, v)| (bytes(k), bytes(v))).collect();
		// the same kernel pair as `sc` (the fused one), else the results differ
		// in the last bits
		std::env::set_var("QWEN_NPU_SCAN_IMPORTS", "16");
		std::env::set_var("QWEN_NPU_FAR_FUSED", "1");
		let mut small = DeltaScan::new(Some(pick), 0).expect("scan device");
		std::env::remove_var("QWEN_NPU_SCAN_IMPORTS");
		std::env::remove_var("QWEN_NPU_FAR_FUSED");
		let mut same = true;
		for round in 0..8 {
			let g = &ml[(round % 4) * 3..(round % 4) * 3 + 3];
			sc.far_begin(s, NK, bytes(&q), g, true).unwrap();
			sc.far_wait().unwrap();
			small.far_begin(s, NK, bytes(&q), g, true).unwrap();
			small.far_wait().unwrap();
			let (a, b) = (sc.far_out(), small.far_out());
			same &= a.0 == b.0 && a.1 == b.1;
		}
		println!(
			"16 imports for 26 buffers: {}",
			if same {
				"same results"
			} else {
				"RESULTS DIFFER"
			}
		);
	}
	for (name, sc) in pairs.iter_mut() {
		for n in [1usize, 2] {
			sc.far_begin(s, NK, bytes(&q), &bl[..n], true).unwrap();
			sc.far_wait().unwrap();
			let t = std::time::Instant::now();
			let reps = 8;
			for _ in 0..reps {
				sc.far_begin(s, NK, bytes(&q), &bl[..n], true).unwrap();
				sc.far_wait().unwrap();
			}
			println!(
				"{name}: {n} block(s) per submission: {:.2} ms per block",
				t.elapsed().as_secs_f64() * 1e3 / (reps * n) as f64
			);
		}
	}
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("farbench needs --features vulkan");
}
