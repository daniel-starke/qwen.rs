//! `hcbench ADAPTER WDIR [LAYER]`: checks `offload::HcPre` (a delta
//! layer's attention hyper-connection for the Flash-Next prefill) against
//! the host on that layer's weight files (`fl{L}.hca.*` in a Flash-Next
//! build's `w/`) and random streams, and times a 256-position call.

#[cfg(feature = "vulkan")]
fn main() -> Result<(), String> {
	use offload::HcPre;
	let a: Vec<String> = std::env::args().collect();
	let dev = a.get(1).cloned().unwrap_or("intel".into());
	let wdir = std::path::PathBuf::from(a.get(2).ok_or("usage: hcbench ADAPTER WDIR [LAYER]")?);
	let l: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
	let rd = |n: &str| {
		let p = wdir.join(format!("fl{l}.hca.{n}.in"));
		std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
	};
	let f16 = |b: Vec<u8>| -> Vec<f32> {
		b.chunks_exact(2)
			.map(|c| quant::fp16_to_f32(u16::from_le_bytes([c[0], c[1]])))
			.collect()
	};
	let norm = f16(rd("norm")?);
	let dw = rd("dinj.w")?;
	let ds: Vec<f32> = f16(rd("dinj.s")?).iter().map(|v| v / 128.0).collect();
	let uw = rd("up.w")?;
	let us: Vec<f32> = f16(rd("up.s")?).iter().map(|v| v / 128.0).collect();
	let mut h = HcPre::new(Some(&dev), 1)?;
	h.add_layer(0, &norm, &dw, &ds, &uw, &us)?;
	let (n, hcw, w) = (256usize, 10240usize, 2560usize);
	let mut seed = 7u64;
	let mut rnd = move || {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
	};
	let x: Vec<u16> = (0..n * hcw)
		.map(|i| quant::f32_to_fp16((rnd() * (1.0 + (i % 7) as f64)) as f32))
		.collect();
	let mut mixed = vec![0u16; n * w];
	let mut inj = vec![0u16; n * 4];
	for np in [n, 37] {
		h.begin(0, &x, np)?;
		h.end(&mut mixed, &mut inj)?;
		// host reference in f64
		let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
		let (mut em, mut ej) = (0f64, 0f64);
		// squared error and squared value of `mixed`: the rms relative error
		let (mut se, mut sy) = (0f64, 0f64);
		for p in (0..np).step_by(np / 8 + 1).chain([np - 1]) {
			let xs: Vec<f64> = x[p * hcw..][..hcw]
				.iter()
				.map(|&v| quant::fp16_to_f32(v) as f64)
				.collect();
			let mut xn = vec![0f64; hcw];
			for s in 0..4 {
				let r = &xs[s * w..][..w];
				let ms = r.iter().map(|v| v * v).sum::<f64>() / w as f64;
				let k = 1.0 / (ms + 1e-6).sqrt();
				for i in 0..w {
					xn[s * w + i] = r[i] * k * norm[s * w + i] as f64;
				}
			}
			let t: Vec<f64> = (0..324)
				.map(|o| {
					let row = &dw[o * hcw..][..hcw];
					row.iter()
						.zip(&xn)
						.map(|(&c, v)| c as i8 as f64 * v)
						.sum::<f64>() * ds[o] as f64
						* 0.25
				})
				.collect();
			let m: Vec<f64> = t[..320].iter().map(|&v| v * sig(v)).collect();
			for c in 0..4 {
				let r = 2.0 * sig(t[320 + c]);
				let g = quant::fp16_to_f32(inj[p * 4 + c]) as f64;
				ej = ej.max((g - r).abs());
			}
			for c in 0..w {
				let mut y = 0.0;
				for s in 0..4 {
					let j = s * w + c;
					let row = &uw[j * 320..][..320];
					let u = row
						.iter()
						.zip(&m)
						.map(|(&q, v)| q as i8 as f64 * v)
						.sum::<f64>() * us[j] as f64;
					y += sig(u) * xn[j];
				}
				y *= 0.25;
				let g = quant::fp16_to_f32(mixed[p * w + c]) as f64;
				em = em.max((g - y).abs() / y.abs().max(0.05));
				se += (g - y) * (g - y);
				sy += y * y;
			}
		}
		println!(
			"{np} positions: mixed max rel err {em:.2e} (rms {:.2e}), inj max abs err {ej:.2e}",
			(se / sy).sqrt()
		);
	}
	const R: usize = 40;
	for np in [256, 64] {
		let t = std::time::Instant::now();
		for _ in 0..R {
			h.begin(0, &x, np)?;
			h.end(&mut mixed, &mut inj)?;
		}
		println!(
			"{np} positions: {:.2} ms per call",
			t.elapsed().as_secs_f64() * 1e3 / R as f64
		);
	}
	Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("hcbench needs --features vulkan");
}
