//! `deltabench [ADAPTER]`: checks `offload::Gemv`'s delta mixer steps
//! (`gemv_delta.glsl`: conv, recurrence + gated norm, window replay)
//! against a host form of `npu::flash::gpudec` / `FlashEngine::
//! step_recurrence` over several steps of 1-4 positions, a saving step and
//! its rollback, then times the steps of 36 layers.

#[cfg(feature = "vulkan")]
fn main() -> Result<(), String> {
	use offload::{Gemv, DELTA_STATE, DELTA_WIN};
	const H: usize = 48;
	const HK: usize = 16;
	const D: usize = 128;
	const NQKV: usize = 10240;
	const YZ: usize = NQKV;
	const YAB: usize = 16384;
	const YA: usize = 32768;
	const NL: usize = 36;
	let dev = std::env::args().nth(1).unwrap_or("geforce".into());
	let mut g = Gemv::new(Some(&dev), 1 << 30, 256 << 20)?;
	let mut seed = 1u64;
	let mut rnd = move || {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((seed >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
	};
	let f16r = |v: f32| half(v);
	let sigm = |v: f32| 1.0 / (1.0 + (-v).exp());
	// weights: conv [NQKV][4], dtb, acoef; gnw
	let conv: Vec<f32> = (0..NQKV * 4).map(|_| rnd() * 0.5).collect();
	let dtb: Vec<f32> = (0..H).map(|_| rnd()).collect();
	let acoef: Vec<f32> = (0..H).map(|_| -0.5 - rnd().abs()).collect();
	let gnw: Vec<f32> = (0..D).map(|_| 1.0 + rnd() * 0.3).collect();
	let w0: Vec<f32> = [conv.clone(), dtb.clone(), acoef.clone()].concat();
	g.delta_init(NL)?;
	// host state of layer 0: S, window (f16 values)
	let mut st: Vec<f32> = (0..H * D * D).map(|_| rnd() * 0.1).collect();
	let mut win: Vec<f32> = (0..DELTA_WIN).map(|_| f16r(rnd())).collect();
	g.delta_put(0, &st, &win)?;
	let copy = g.add_vec(0, YAB + 2 * H, 0, 0, 0, 1, 0.0, 0, 0, Some(0), None, None)?;
	let mut ops = Vec::new();
	for l in 0..NL {
		ops.push([
			g.add_delta(0, 0, l, YZ, YAB, YA, &w0)?,
			g.add_delta(1, 0, l, YZ, YAB, YA, &gnw)?,
			g.add_delta(0, 1, l, YZ, YAB, YA, &w0)?,
			g.add_delta(1, 1, l, YZ, YAB, YA, &gnw)?,
			g.add_delta(2, 0, l, YZ, YAB, YA, &[])?,
			g.add_delta(1, 2, l, YZ, YAB, YA, &gnw)?,
		]);
	}
	// the host form of one step of the positions `xs` (qkv | z | ab)
	let host = |st: &mut Vec<f32>, win: &mut Vec<f32>, xs: &[Vec<f32>]| -> Vec<Vec<f32>> {
		let mut out = Vec::new();
		for x in xs {
			let mut cv = vec![0f32; NQKV];
			for c in 0..NQKV {
				let w = &conv[c * 4..][..4];
				let mut a = w[3] * x[c];
				for j in 0..3 {
					a += w[j] * win[j * NQKV + c];
				}
				cv[c] = a * sigm(a);
				win[c] = win[NQKV + c];
				win[NQKV + c] = win[2 * NQKV + c];
				win[2 * NQKV + c] = f16r(x[c]);
			}
			for r in cv[..2 * HK * D].chunks_exact_mut(D) {
				let s = 1.0 / r.iter().map(|v| v * v).sum::<f32>().max(1e-12).sqrt();
				r.iter_mut().for_each(|v| *v *= s);
			}
			let mut a = vec![0f32; H * D];
			for h in 0..H {
				let ab = x[YAB + h] + dtb[h];
				let sp = if ab > 20.0 { ab } else { ab.exp().ln_1p() };
				let eg = (sp * acoef[h]).exp();
				let beta = sigm(x[YAB + H + h]);
				let kh = h / (H / HK);
				let q = &cv[kh * D..][..D];
				let k = &cv[HK * D + kh * D..][..D];
				let v = &cv[2 * HK * D + h * D..][..D];
				let s = &mut st[h * D * D..][..D * D];
				let mut kv = [0f32; D];
				for (ki, row) in k.iter().zip(s.chunks_exact(D)) {
					for (a, b) in kv.iter_mut().zip(row) {
						*a += ki * b;
					}
				}
				let d: Vec<f32> = (0..D).map(|j| (v[j] - eg * kv[j]) * beta).collect();
				let mut o = [0f32; D];
				for ((ki, qi), row) in k.iter().zip(q).zip(s.chunks_exact_mut(D)) {
					for ((a, &b), r) in row.iter_mut().zip(&d).zip(o.iter_mut()) {
						*a = eg * *a + ki * b;
						*r += qi * *a;
					}
				}
				let o: Vec<f32> = o.iter().map(|v| v / (D as f32).sqrt()).collect();
				let ms = o.iter().map(|v| v * v).sum::<f32>() / D as f32;
				let r = 1.0 / (ms + 1e-6).sqrt();
				for j in 0..D {
					a[h * D + j] = o[j] * r * gnw[j] * sigm(x[YZ + h * D + j]);
				}
			}
			out.push(a);
		}
		out
	};
	let mut inputs = |n: usize| -> Vec<Vec<f32>> {
		(0..n)
			.map(|_| (0..YAB + 2 * H).map(|_| rnd() * 2.0).collect())
			.collect()
	};
	let err = |a: &[f32], b: &[f32]| {
		a.iter()
			.zip(b)
			.map(|(x, y)| (x - y).abs())
			.fold(0f32, f32::max)
			/ a.iter().fold(0f32, |m, v| m.max(v.abs()))
	};
	let mut worst = 0f32;
	for (step, &n) in [1usize, 2, 4, 1, 3].iter().enumerate() {
		let xs = inputs(n);
		let want = host(&mut st, &mut win, &xs);
		let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
		g.run_chain_n(&xr, &[&[copy], &[ops[0][0]], &[ops[0][1]]])?;
		for (p, w) in want.iter().enumerate() {
			let e = err(w, &g.y_at(p)[YA..YA + H * D]);
			worst = worst.max(e);
			println!("step {step} ({n} positions) pos {p}: max err of a / max |a| {e:.2e}");
		}
	}
	let (mut gs, mut gw) = (vec![0f32; DELTA_STATE / 2], vec![0f32; DELTA_WIN]);
	g.delta_get(0, &mut gs, &mut gw)?;
	println!(
		"state after 11 positions: max err / max {:.2e}, window {:.2e}",
		err(&st, &gs),
		err(&win, &gw)
	);
	// a saving step of 4, rolled back to its first 2 positions
	let xs = inputs(4);
	let (mut s2, mut w2) = (st.clone(), win.clone());
	host(&mut s2, &mut w2, &xs[..2]);
	let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
	g.run_chain_n(&xr, &[&[copy], &[ops[0][2]], &[ops[0][3]]])?;
	let none: Vec<&[f32]> = vec![&[]; 2];
	g.run_chain_n(&none, &[&[ops[0][4], ops[0][5]]])?;
	g.delta_get(0, &mut gs, &mut gw)?;
	println!(
		"rollback to 2 of 4: state max err / max {:.2e}, window {:.2e}",
		err(&s2, &gs),
		err(&w2, &gw)
	);
	println!("worst output err {worst:.2e}");
	// timing: 36 layers, as one submission (stages conv, recur) and per layer
	for n in [1usize, 2, 4, 8] {
		let xs = inputs(n);
		let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
		let c: Vec<usize> = ops.iter().map(|o| o[0]).collect();
		let r: Vec<usize> = ops.iter().map(|o| o[1]).collect();
		let rs: Vec<usize> = ops.iter().map(|o| o[3]).collect();
		const R: usize = 30;
		let mut time = |st: &[&[usize]]| -> Result<f64, String> {
			g.run_chain_n(&xr, st)?;
			let t = std::time::Instant::now();
			for _ in 0..R {
				g.run_chain_n(&xr, st)?;
			}
			Ok(t.elapsed().as_secs_f64() * 1e3 / R as f64)
		};
		let all = time(&[&c, &r])?;
		let recur = time(&[&r])?;
		let save = time(&[&rs])?;
		let conv1 = time(&[&c])?;
		println!(
			"{n} positions, 36 layers: conv + recur {all:.2} ms (conv {conv1:.2}, recur {recur:.2}, recur saving {save:.2}); state {:.0} GB/s",
			(NL * H * D * D * 8) as f64 / recur / 1e6
		);
	}
	Ok(())
}

/// `x` rounded to f16 and back (round to nearest even).
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn half(x: f32) -> f32 {
	let b = x.to_bits();
	let e = ((b >> 23) & 0xff) as i32 - 127;
	if !(-14..16).contains(&e) {
		// subnormal: multiples of 2^-24
		let u = (-24f32).exp2();
		return if e < -14 {
			(x / u).round_ties_even() * u
		} else {
			x
		};
	}
	let sh = 13;
	let m = b & ((1 << sh) - 1);
	let mut r = b >> sh;
	if m > 1 << (sh - 1) || m == 1 << (sh - 1) && r & 1 == 1 {
		r += 1;
	}
	f32::from_bits(r << sh)
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("deltabench needs --features vulkan");
}
