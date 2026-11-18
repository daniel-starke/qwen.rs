//! Check and time `VisAttn` (the vision tower's attention kernels,
//! `vis_rope_npu.glsl`, `vis_qk_npu.glsl`, `vis_sm_npu.glsl`,
//! `vis_pv_npu.glsl`) on random data against an f64 reference:
//!   cargo run --release -p offload --features vulkan --example visbench [adapter] [gw] [gh]
//! Prints the worst relative error over a sample of (patch, head) rows and
//! ms per block for a `gw x gh` patch grid.

fn main() {
	use offload::VisAttn;
	let a: Vec<String> = std::env::args().skip(1).collect();
	let pick = a.first().map(|s| s.as_str()).unwrap_or("intel");
	let gw: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(64);
	let gh: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(64);
	const H: usize = 16;
	const HD: usize = 72;
	let n = gw * gh;
	let mut seed = 0x1234_5678_9abc_def0u64;
	let mut rnd = move || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		seed
	};
	// random f16 bit patterns with |x| in [2^-2, 2^1)
	let x: Vec<u16> = (0..n * 3 * H * HD)
		.map(|_| {
			let r = rnd();
			(((r >> 40) & 1) << 15) as u16
				| ((13 + (r >> 20) % 3) << 10) as u16
				| (r & 0x3ff) as u16
		})
		.collect();
	let val = |h: u16| -> f64 {
		let m = 1.0 + (h & 0x3ff) as f64 / 1024.0;
		let e = ((h >> 10) & 31) as i32 - 15;
		m * 2f64.powi(e) * if h >> 15 == 1 { -1.0 } else { 1.0 }
	};
	let hw: Vec<(u32, u32)> = (0..n).map(|i| ((i / gw) as u32, (i % gw) as u32)).collect();
	let mut out = vec![0u16; n * H * HD];
	let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) };
	let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 2) };
	let mut va = VisAttn::new(Some(pick)).expect("device");
	va.attend(H, &hw, xb, ob).unwrap();
	// reference: 2-D RoPE (pairs (j, j + 36); the first 18 the row's
	// angles, the next 18 the column's), softmax(q k / sqrt(72)) v
	let rot = |t: usize, part: usize, h: usize| -> Vec<f64> {
		let r: Vec<f64> = x[((t * 3 + part) * H + h) * HD..][..HD]
			.iter()
			.map(|&v| val(v))
			.collect();
		let mut o = vec![0f64; HD];
		for j in 0..36 {
			let (pos, k) = if j < 18 {
				(hw[t].0, j)
			} else {
				(hw[t].1, j - 18)
			};
			let ang = pos as f64 / 10000f64.powf(2.0 * k as f64 / 36.0);
			o[j] = r[j] * ang.cos() - r[j + 36] * ang.sin();
			o[j + 36] = r[j + 36] * ang.cos() + r[j] * ang.sin();
		}
		o
	};
	let mut worst = 0f64;
	for i in 0..24 {
		let (t, h) = ((i * 7919) % n, (i * 5) % H);
		let q = rot(t, 0, h);
		let sc: Vec<f64> = (0..n)
			.map(|u| {
				rot(u, 1, h).iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() / (HD as f64).sqrt()
			})
			.collect();
		let m = sc.iter().cloned().fold(f64::MIN, f64::max);
		let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
		let mut want = vec![0f64; HD];
		for u in 0..n {
			let p = (sc[u] - m).exp() / z;
			for (w, &v) in want.iter_mut().zip(&x[((u * 3 + 2) * H + h) * HD..][..HD]) {
				*w += p * val(v);
			}
		}
		let got = &out[(t * H + h) * HD..][..HD];
		let nrm: f64 = want.iter().map(|v| v * v).sum::<f64>().sqrt();
		let d: f64 = want
			.iter()
			.zip(got)
			.map(|(w, &g)| (w - val(g) * if g & 0x7fff == 0 { 0.0 } else { 1.0 }).powi(2))
			.sum::<f64>()
			.sqrt();
		worst = worst.max(d / nrm);
	}
	println!("{gw} x {gh} = {n} patches: worst relative error {worst:.2e}");
	let reps = 5;
	let t = std::time::Instant::now();
	for _ in 0..reps {
		va.attend(H, &hw, xb, ob).unwrap();
	}
	let dt = t.elapsed().as_secs_f64() / reps as f64;
	println!(
		"{:.1} ms per block, {:.0} GFLOP/s",
		dt * 1e3,
		4.0 * (n * n * H * HD) as f64 / dt / 1e9
	);
}
