//! `expbench ADAPTER WDIR [LAYER] [TOKENS]`: checks `offload::Experts`
//! (a share of the Flash-Next prefill's routed experts) against the host
//! (`npu::flash::cpuexp`'s math) on that layer's expert pools (`fl{L}.egu.*`,
//! `fl{L}.ed.*` in a Flash-Next build's `w/`), random MoE inputs and random
//! routing, and times calls over a share of the slots.

#[cfg(feature = "vulkan")]
fn main() -> Result<(), String> {
	use offload::{Experts, EXP_SLOT};
	const W: usize = 2560;
	const FE: usize = 640;
	const NE: usize = 512;
	let a: Vec<String> = std::env::args().collect();
	let dev = a.get(1).cloned().unwrap_or("intel".into());
	let wdir = std::path::PathBuf::from(
		a.get(2)
			.ok_or("usage: expbench ADAPTER WDIR [LAYER] [TOKENS]")?,
	);
	let l: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
	let ntok: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(3220);
	// page-aligned copies (imported as host memory)
	let rd = |n: &str| -> Result<&'static mut [u8], String> {
		let p = wdir.join(format!("fl{l}.{n}.in"));
		let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
		let len = b.len().div_ceil(65536) * 65536;
		let lay = std::alloc::Layout::from_size_align(len, 65536).unwrap();
		// SAFETY: non-zero size; leaked for the process
		let m = unsafe { std::slice::from_raw_parts_mut(std::alloc::alloc_zeroed(lay), len) };
		m[..b.len()].copy_from_slice(&b);
		Ok(m)
	};
	let h16 = |b: &'static [u8]| unsafe {
		std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2)
	};
	let (guw, gus, dw, ds, dpre) = (
		rd("egu.w")?,
		rd("egu.s")?,
		rd("ed.w")?,
		rd("ed.s")?,
		rd("ed.pre")?,
	);
	let (gus, ds, dpre) = (h16(gus), h16(ds), h16(dpre));
	let mut seed = 7u64;
	let mut rnd = move || {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(seed >> 33) as f64 / (1u64 << 31) as f64
	};
	// the MoE input as `post` leaves it: prescaled to the int8 grid
	let x: Vec<u16> = (0..ntok * W)
		.map(|_| quant::f32_to_fp16(((rnd() * 2.0 - 1.0) * 90.0) as f32))
		.collect();
	let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) };
	// 10 distinct experts per token, skewed (a few hot ones)
	let mut per: Vec<Vec<u32>> = vec![Vec::new(); NE];
	for t in 0..ntok {
		let mut sel: Vec<usize> = Vec::new();
		while sel.len() < 10 {
			let r = rnd();
			let e = ((r * r) * NE as f64) as usize % NE;
			if !sel.contains(&e) {
				sel.push(e);
			}
		}
		for e in sel {
			per[e].push(t as u32);
		}
	}
	// `EXPBENCH_CHUNK=N`: slots of at most N tokens (the per-size cost)
	let chunk: usize = std::env::var("EXPBENCH_CHUNK")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(EXP_SLOT)
		.clamp(1, EXP_SLOT);
	let mut slots: Vec<(usize, Vec<u32>)> = Vec::new();
	for (e, ts) in per.iter().enumerate() {
		for c in ts.chunks(chunk) {
			slots.push((e, c.to_vec()));
		}
	}
	let rows_all: usize = slots.iter().map(|s| s.1.len()).sum();
	let mut ex = Experts::new(Some(&dev), 1, ntok, rows_all)?;
	println!(
		"{} slots, {rows_all} rows; maxStorageBufferRange {} MB",
		slots.len(),
		ex.max_range() >> 20
	);
	println!(
		"dp4a accelerated (dot, accumulating): {:?}",
		ex.dp4a_accelerated()
	);
	// `EXPBENCH_LAYERS=N`: N distinct copies of the pools imported (the
	// engine imports all 48 layers' ~60 GB)
	let nl: usize = std::env::var("EXPBENCH_LAYERS")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(1);
	if nl > 1 {
		ex = Experts::new(Some(&dev), nl, ntok, rows_all)?;
		for i in 1..nl {
			let cp = |b: &[u8]| -> &'static [u8] {
				let lay = std::alloc::Layout::from_size_align(b.len(), 65536).unwrap();
				let m = unsafe { std::slice::from_raw_parts_mut(std::alloc::alloc(lay), b.len()) };
				m.copy_from_slice(b);
				m
			};
			let c16 = |h: &[u16]| {
				h16(cp(unsafe {
					std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2)
				}))
			};
			ex.add_layer(i, cp(guw), c16(gus), cp(dw), c16(ds), c16(dpre))?;
		}
		println!("{nl} layers imported; budget, usage {:?}", ex.budget());
	}
	ex.add_layer(0, guw, gus, dw, ds, dpre)?;
	let sl: Vec<(usize, &[u32])> = slots.iter().map(|(e, t)| (*e, &t[..])).collect();
	ex.begin(0, xb, &sl)?;
	let y = ex.end()?.to_vec();
	// host reference on a sample of rows (cpuexp's math, f64 sums)
	let h = |v: u16| quant::fp16_to_f32(v) as f64;
	let nib = |b: u8, hi: bool| -> i32 {
		let v = if hi { b >> 4 } else { b & 15 };
		((v << 4) as i8 >> 4) as i32
	};
	let q8 = |v: f64| v.round().clamp(-128.0, 127.0) as i32;
	let mut row = 0;
	let (mut emax, mut rmax) = (0f64, 0f64);
	let mut checked = 0;
	for (si, (e, ts)) in slots.iter().enumerate() {
		for (i, &t) in ts.iter().enumerate() {
			if (si * 7 + i) % 97 == 0 {
				let xq: Vec<i32> = (0..W).map(|k| q8(h(x[t as usize * W + k]))).collect();
				let g: Vec<f64> = (0..2 * FE)
					.map(|r| {
						let rr = e * 2 * FE + r;
						let c = &guw[rr * W / 2..][..W / 2];
						let d: i32 = (0..W).map(|k| nib(c[k / 2], k % 2 == 1) * xq[k]).sum();
						d as f64 / 256.0 * h(gus[rr])
					})
					.collect();
				let mut v: Vec<f64> = (0..FE)
					.map(|k| {
						let a = g[k];
						a / (1.0 + (-a).exp()) * g[FE + k] * h(dpre[e * FE + k])
					})
					.collect();
				for b in v.chunks_exact_mut(128) {
					let mut s = 1;
					while s < 128 {
						for i0 in (0..128).step_by(2 * s) {
							for j in i0..i0 + s {
								let (p, q) = (b[j], b[j + s]);
								b[j] = p + q;
								b[j + s] = p - q;
							}
						}
						s *= 2;
					}
					b.iter_mut().for_each(|z| *z /= 128f64.sqrt());
				}
				let hq: Vec<i32> = v.iter().map(|&z| q8(z)).collect();
				for r in 0..W {
					let rr = e * W + r;
					let c = &dw[rr * FE / 2..][..FE / 2];
					let d: i32 = (0..FE).map(|k| nib(c[k / 2], k % 2 == 1) * hq[k]).sum();
					let want = d as f64 / 256.0 * h(ds[rr]);
					let got = h(y[(row + i) * W + r]);
					emax = emax.max((got - want).abs());
					rmax = rmax.max(want.abs());
				}
				checked += 1;
			}
		}
		row += ts.len();
	}
	println!("checked {checked} rows: max |err| {emax:.3e} of max |y| {rmax:.3e}");
	for frac in [1.0, 0.5, 0.25] {
		let k = (slots.len() as f64 * frac) as usize;
		let s = &sl[slots.len() - k..];
		let rows: usize = s.iter().map(|v| v.1.len()).sum();
		let mut ts = Vec::new();
		for _ in 0..5 {
			let t = std::time::Instant::now();
			ex.begin(0, xb, s)?;
			ex.end()?;
			ts.push(t.elapsed().as_secs_f64() * 1e3);
		}
		ts.sort_by(|a, b| a.total_cmp(b));
		let ops = rows as f64 * 2.0 * (2.0 * FE as f64 * W as f64 + W as f64 * FE as f64);
		println!(
			"{k} slots, {rows} rows: min {:.1} ms, median {:.1} ms ({:.2} TOPS)",
			ts[0],
			ts[2],
			ops / ts[0] / 1e9
		);
	}
	Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("expbench needs --features vulkan");
}
