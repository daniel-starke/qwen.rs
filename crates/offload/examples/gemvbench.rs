//! `gemvbench [ADAPTER]`: checks `offload::Gemv` against the host and times
//! one run per Flash-Next matrix shape, alone and as one layer's worth of
//! runs (the per-submission cost); the int8 kernels also alone (16
//! dispatches per submission) at 1-8 positions.

#[cfg(feature = "vulkan")]
fn main() -> Result<(), String> {
	use offload::{Gemv, GemvMat};
	let dev = std::env::args().nth(1).unwrap_or("geforce".into());
	let mut g = Gemv::new(Some(&dev), 1 << 30, 256 << 20)?;
	let mut seed = 1u64;
	let mut rnd = move || {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(seed >> 33) as u32
	};
	// (n, k, f16)
	let shapes = [
		(320usize, 10240usize, false),
		(10240, 320, false),
		(10240, 2560, false),
		(2560, 6144, false),
		(512, 2560, true),
		(4, 10240, true),
	];
	let mut ids = Vec::new();
	let mut refs = Vec::new();
	for &(n, k, f16) in &shapes {
		let x: Vec<f32> = (0..k).map(|_| (rnd() % 255) as f32 - 127.0).collect();
		let s: Vec<f32> = (0..n).map(|_| 1.0 + (rnd() % 100) as f32 / 100.0).collect();
		let (id, y) = if f16 {
			let w: Vec<u16> = (0..n * k)
				.map(|_| 0x3c00 | (rnd() & 0x3ff) as u16)
				.collect();
			let y: Vec<f32> = (0..n)
				.map(|r| {
					(0..k)
						.map(|i| (1.0 + (w[r * k + i] & 0x3ff) as f32 / 1024.0) * x[i])
						.sum()
				})
				.collect();
			(g.add(GemvMat::F16 { w: &w }, n, k, 0, 0)?, y)
		} else {
			let w: Vec<i8> = (0..n * k).map(|_| rnd() as i8).collect();
			let y: Vec<f32> = (0..n)
				.map(|r| s[r] * (0..k).map(|i| w[r * k + i] as f32 * x[i]).sum::<f32>())
				.collect();
			(g.add(GemvMat::I8 { w: &w, s: &s }, n, k, 0, 0)?, y)
		};
		g.run(&x, &[id])?;
		let e = y
			.iter()
			.zip(g.y())
			.map(|(a, b)| ((a - b) / a.abs().max(1.0)).abs())
			.fold(0f32, f32::max);
		const R: usize = 200;
		let t = std::time::Instant::now();
		for _ in 0..R {
			g.run(&x, &[id])?;
		}
		let us = t.elapsed().as_secs_f64() * 1e6 / R as f64;
		let bytes = (n * k * if f16 { 2 } else { 1 }) as f64;
		println!(
			"[{n}][{k}] {}: {us:.1} us per run ({:.1} GB/s), rel err {e:.1e}",
			if f16 { "f16" } else { "i8" },
			bytes / us / 1e3
		);
		if f16 {
			// 1-8 positions (verify steps): against the host, the builds
			// for 2 / 4 bitwise against the 8-position one, kernel time
			let w: Vec<u16> = (0..n * k)
				.map(|_| 0x3c00 | (rnd() & 0x3ff) as u16 | (rnd() & 0x8000) as u16)
				.collect();
			let m = g.add(GemvMat::F16 { w: &w }, n, k, 0, 0)?;
			let xs: Vec<Vec<f32>> = (0..8)
				.map(|_| (0..k).map(|_| (rnd() % 256) as f32 - 128.0).collect())
				.collect();
			let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
			let wf: Vec<f32> = w
				.iter()
				.map(|&h| {
					let v = 1.0 + (h & 0x3ff) as f32 / 1024.0;
					if h & 0x8000 != 0 {
						-v
					} else {
						v
					}
				})
				.collect();
			let mut e = 0f32;
			let mut same = true;
			for np in 1..=8 {
				g.run_chain_n(&xr[..np], &[&[m]])?;
				let got: Vec<Vec<f32>> = (0..np).map(|p| g.y_at(p)[..n].to_vec()).collect();
				for (p, y) in got.iter().enumerate() {
					for (r, v) in y.iter().enumerate() {
						let u: f32 = (0..k).map(|i| wf[r * k + i] * xs[p][i]).sum();
						e = e.max(((u - v) / u.abs().max(1.0)).abs());
					}
				}
				g.set_npb(false);
				g.run_chain_n(&xr[..np], &[&[m]])?;
				g.set_npb(true);
				same &= got.iter().enumerate().all(|(p, y)| {
					y.iter()
						.zip(&g.y_at(p)[..n])
						.all(|(a, b)| a.to_bits() == b.to_bits())
				});
			}
			let mut line = format!(
				"[{n}][{k}] f16 1-8 positions rel err {e:.1e}, builds {} the 8-position one; kernel only:",
				if same { "bitwise equal to" } else { "DIFFER from" }
			);
			for np in [1, 2, 3, 4, 8] {
				for (b, on) in [("", true), ("/8", false)] {
					if (np == 1 || np == 8) && !on {
						continue;
					}
					g.set_npb(on);
					let mut us = [0f64; 2];
					for (j, c) in [1usize, 17].into_iter().enumerate() {
						let st: Vec<&[usize]> = vec![std::slice::from_ref(&m); c];
						let t = std::time::Instant::now();
						for _ in 0..R / 4 {
							g.run_chain_n(&xr[..np], &st)?;
						}
						us[j] = t.elapsed().as_secs_f64() * 1e6 / (R / 4) as f64;
					}
					line += &format!(" x{np}{b} {:.0}", (us[1] - us[0]) / 16.0);
				}
			}
			g.set_npb(true);
			println!("{line} us");
		}
		if !f16 {
			// the dp4a form (`GemvMat::Q8`) next to the float one on another
			// matrix, at 1-8 positions (verify steps)
			let w: Vec<i8> = (0..n * k).map(|_| rnd() as i8).collect();
			let i8 = g.add(GemvMat::I8 { w: &w, s: &s }, n, k, 0, 0)?;
			let q8 = g.add(GemvMat::Q8 { w: &w, s: &s }, n, k, 0, 0)?;
			let xs: Vec<Vec<f32>> = (0..8)
				.map(|_| (0..k).map(|_| (rnd() % 256) as f32 - 128.0).collect())
				.collect();
			let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
			// against the host, after a run of another matrix wrote other
			// values to the same rows
			let href: Vec<Vec<f32>> = xs
				.iter()
				.map(|x| {
					(0..n)
						.map(|r| s[r] * (0..k).map(|i| w[r * k + i] as f32 * x[i]).sum::<f32>())
						.collect()
				})
				.collect();
			let mut e = 0f32;
			let mut same = true;
			for np in 1..=8 {
				for m in [i8, q8] {
					g.run_chain_n(&xr[..1], &[&[id]])?;
					g.run_chain_n(&xr[..np], &[&[m]])?;
					let got: Vec<Vec<f32>> = (0..np).map(|p| g.y_at(p)[..n].to_vec()).collect();
					for (p, a) in href[..np].iter().enumerate() {
						for (u, v) in a.iter().zip(&got[p]) {
							e = e.max(((u - v) / u.abs().max(1.0)).abs());
						}
					}
					g.set_npb(false);
					g.run_chain_n(&xr[..np], &[&[m]])?;
					g.set_npb(true);
					same &= got.iter().enumerate().all(|(p, y)| {
						y.iter()
							.zip(&g.y_at(p)[..n])
							.all(|(a, b)| a.to_bits() == b.to_bits())
					});
				}
			}
			let mut line =
				format!(
				"[{n}][{k}] i8 / q8 rel err {e:.1e}, 2 / 4-position builds {} the 8-position ones:",
				if same { "bitwise equal to" } else { "DIFFER from" }
			);
			for np in [1, 2, 4, 8] {
				for (name, id) in [("i8", i8), ("q8", q8)] {
					let t = std::time::Instant::now();
					for _ in 0..R {
						g.run_chain_n(&xr[..np], &[&[id]])?;
					}
					let us = t.elapsed().as_secs_f64() * 1e6 / R as f64;
					line += &format!(" {name}x{np} {us:.0}");
				}
			}
			println!("{line} us");
			// kernel time alone: 16 dispatches of the matrix in one
			// submission, less one dispatch's submission
			let mut line = format!("[{n}][{k}] kernel only:");
			for (np, on) in [
				(1, true),
				(2, true),
				(2, false),
				(3, true),
				(4, true),
				(4, false),
				(8, true),
			] {
				g.set_npb(on);
				for (name, id) in [("i8", i8), ("q8", q8)] {
					let mut us = [0f64; 2];
					for (j, c) in [1usize, 17].into_iter().enumerate() {
						let st: Vec<&[usize]> = vec![std::slice::from_ref(&id); c];
						let t = std::time::Instant::now();
						for _ in 0..R / 4 {
							g.run_chain_n(&xr[..np], &st)?;
						}
						us[j] = t.elapsed().as_secs_f64() * 1e6 / (R / 4) as f64;
					}
					line += &format!(
						" {name}x{np}{} {:.0}",
						if on { "" } else { "/8" },
						(us[1] - us[0]) / 16.0
					);
				}
			}
			g.set_npb(true);
			println!("{line} us");
		}
		ids.push(id);
		refs.push(x);
	}
	waves(&mut g, &mut rnd)?;
	decode_shapes(&mut g)?;
	// submission round trips of a decode step's length: c dispatches of
	// [10240][2560] in one chain
	let x = &refs[2];
	let mut line = "chain of [10240][2560] x c:".to_string();
	for c in [1usize, 2, 4, 8, 16] {
		let st: Vec<&[usize]> = vec![std::slice::from_ref(&ids[2]); c];
		g.run_chain(x, &st)?;
		let t = std::time::Instant::now();
		for _ in 0..100 {
			g.run_chain(x, &st)?;
		}
		line += &format!(" c{c} {:.0}", t.elapsed().as_secs_f64() * 1e4);
	}
	println!("{line} us");
	let t = std::time::Instant::now();
	for _ in 0..200 {
		g.run(&x[..16], &[])?;
	}
	println!(
		"empty run: {:.1} us",
		t.elapsed().as_secs_f64() * 1e6 / 200.0
	);
	Ok(())
}

/// The decode's GEMV shapes (int8 on float input, `dp4a` on integer
/// input, f16) at 1-8 positions: kernel time alone, and with
/// `GEMVBENCH_DUMP=FILE` every output written there, to compare two
/// binaries' kernels bitwise (inputs from a fixed seed).
#[cfg(feature = "vulkan")]
fn decode_shapes(g: &mut offload::Gemv) -> Result<(), String> {
	use offload::GemvMat;
	let mut seed = 7u64;
	let mut rnd = move || {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(seed >> 33) as u32
	};
	let mut dump = Vec::new();
	// (n, k, kind: 0 int8, 1 dp4a, 2 f16)
	for (n, k, kind) in [
		(320usize, 10240usize, 0),
		(10240, 320, 0),
		(16384, 2560, 0),
		(16480, 2560, 1),
		(2560, 6144, 1),
		(13952, 2560, 1),
		(1280, 2560, 1),
		(2560, 640, 1),
		(512, 2560, 2),
		(4, 10240, 2),
	] {
		let s: Vec<f32> = (0..n)
			.map(|_| 0.5 + (rnd() % 1000) as f32 / 997.0)
			.collect();
		let m = if kind == 2 {
			let w: Vec<u16> = (0..n * k)
				.map(|_| 0x2c00 | (rnd() & 0x3ff) as u16 | (rnd() & 0x8000) as u16)
				.collect();
			g.add(GemvMat::F16 { w: &w }, n, k, 0, 0)?
		} else {
			let w: Vec<i8> = (0..n * k).map(|_| rnd() as i8).collect();
			if kind == 0 {
				g.add(GemvMat::I8 { w: &w, s: &s }, n, k, 0, 0)?
			} else {
				g.add(GemvMat::Q8 { w: &w, s: &s }, n, k, 0, 0)?
			}
		};
		let xs: Vec<Vec<f32>> = (0..8)
			.map(|_| {
				(0..k)
					.map(|_| {
						if kind == 1 {
							(rnd() % 256) as f32 - 128.0
						} else {
							((rnd() % 200001) as f32 / 100000.0 - 1.0) * 3.7
						}
					})
					.collect()
			})
			.collect();
		let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
		for np in 1..=8 {
			g.run_chain_n(&xr[..np], &[&[m]])?;
			for p in 0..np {
				dump.extend(g.y_at(p)[..n].iter().flat_map(|f| f.to_le_bytes()));
			}
		}
		let mut line = format!(
			"shape {} [{n}][{k}] kernel only:",
			["i8", "q8", "f16"][kind]
		);
		for np in [1, 2, 3, 4, 8] {
			let mut us = [0f64; 2];
			for (j, c) in [1usize, 17].into_iter().enumerate() {
				let st: Vec<&[usize]> = vec![std::slice::from_ref(&m); c];
				let t = std::time::Instant::now();
				for _ in 0..50 {
					g.run_chain_n(&xr[..np], &st)?;
				}
				us[j] = t.elapsed().as_secs_f64() * 1e6 / 50.0;
			}
			line += &format!(" x{np} {:.0}", (us[1] - us[0]) / 16.0);
		}
		println!("{line} us");
	}
	if let Ok(f) = std::env::var("GEMVBENCH_DUMP") {
		std::fs::write(&f, &dump).map_err(|e| format!("{f}: {e}"))?;
	}
	Ok(())
}

/// Two waves ([`offload::Gemv::set_wave`]): wave 1 (slots 4-7) bitwise
/// equal to wave 0 for each kind of GEMV, both in flight at once; a delta
/// mixer step split 2 + 2 over the waves bitwise equal to one run of 4
/// (outputs, state, window), also after a rollback to 3 and to 0
/// positions; and whether the one-position builds equal the others.
#[cfg(feature = "vulkan")]
fn waves(g: &mut offload::Gemv, rnd: &mut impl FnMut() -> u32) -> Result<(), String> {
	use offload::{GemvMat, DELTA_STATE, DELTA_WIN};
	let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
	let (n, k) = (2560usize, 2560usize);
	let w8: Vec<i8> = (0..n * k).map(|_| rnd() as i8).collect();
	let s: Vec<f32> = (0..n).map(|_| 1.0 + (rnd() % 100) as f32 / 100.0).collect();
	let wh: Vec<u16> = (0..n * k)
		.map(|_| 0x2c00 | (rnd() & 0x3ff) as u16 | (rnd() & 0x8000) as u16)
		.collect();
	let mats = [
		("i8", g.add(GemvMat::I8 { w: &w8, s: &s }, n, k, 0, 0)?),
		("q8", g.add(GemvMat::Q8 { w: &w8, s: &s }, n, k, 0, 0)?),
		("f16", g.add(GemvMat::F16 { w: &wh }, n, k, 0, 0)?),
	];
	let xs: Vec<Vec<f32>> = (0..8)
		.map(|_| (0..k).map(|_| (rnd() % 256) as f32 - 128.0).collect())
		.collect();
	let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
	let mut line = "waves:".to_string();
	for (name, m) in mats {
		let mut ok = true;
		let mut one = true;
		for np in 1..=4 {
			g.set_wave(0, 0);
			g.run_chain_n(&xr[..np], &[&[m]])?;
			let a: Vec<Vec<f32>> = (0..np).map(|p| g.y_at(p)[..n].to_vec()).collect();
			// both in flight: wave 0 on other inputs first
			g.run_chain_n_begin(&xr[4..4 + np], &[&[m]])?;
			g.set_wave(1, 0);
			g.run_chain_n_begin(&xr[..np], &[&[m]])?;
			g.wait_all()?;
			ok &= (0..np).all(|p| bits(&a[p], &g.y_at(p)[..n]));
			g.set_wave(0, 0);
			if np == 2 {
				// position 0 of a 2-position run vs the 1-position build
				g.run_chain_n(&xr[..1], &[&[m]])?;
				one &= bits(&a[0], &g.y_at(0)[..n]);
			}
		}
		line += &format!(
			" {name} wave 1 {}, 1-position build {} 2-position;",
			if ok { "bitwise equal" } else { "DIFFERS" },
			if one { "equal to" } else { "differs from" }
		);
	}
	println!("{line}");

	// a delta mixer: its inputs from one f16 GEMV ([16480][16]: qkv at 0, z
	// at 10240, ab at 16384), conv + recurrence (saving), output at 32768
	const NQKV: usize = 10240;
	const H: usize = 48;
	const YA: usize = 32768;
	let kk = 16;
	let ni = 16384 + 2 * H;
	let wi: Vec<u16> = (0..ni * kk)
		.map(|_| 0x2400 | (rnd() & 0x3ff) as u16 | (rnd() & 0x8000) as u16)
		.collect();
	let mi = g.add(GemvMat::F16 { w: &wi }, ni, kk, 0, 0)?;
	g.delta_init(1)?;
	let small = |r: u32, a: f32| ((r % 2001) as f32 / 1000.0 - 1.0) * a;
	// conv weights, dt bias, A coefficients (negative: a decay)
	let mut w0 = Vec::with_capacity(NQKV * 4 + 2 * H);
	for i in 0..NQKV * 4 + 2 * H {
		let r = rnd();
		w0.push(if i < NQKV * 4 {
			small(r, 0.5)
		} else if i < NQKV * 4 + H {
			small(r, 1.0)
		} else {
			-0.5 - small(r, 0.4).abs()
		});
	}
	let gnw: Vec<f32> = (0..128).map(|_| 1.0 + small(rnd(), 0.2)).collect();
	let (yz, yab) = (NQKV, 16384);
	let op =
		|g: &mut offload::Gemv, mode, flags, w: &[f32]| g.add_delta(mode, flags, 0, yz, yab, YA, w);
	let (cv, rc) = (op(g, 0, 1, &w0)?, op(g, 1, 1, &gnw)?);
	let (win, rep) = (op(g, 2, 0, &[])?, op(g, 1, 2, &gnw)?);
	let (win0, rep0) = (op(g, 2, 4, &[])?, op(g, 1, 6, &gnw)?);
	let st0: Vec<f32> = (0..DELTA_STATE / 2).map(|_| small(rnd(), 0.05)).collect();
	let cw0: Vec<f32> = (0..DELTA_WIN).map(|_| small(rnd(), 0.05)).collect();
	let xd: Vec<Vec<f32>> = (0..4)
		.map(|_| (0..kk).map(|_| small(rnd(), 1.0)).collect())
		.collect();
	let xdr: Vec<&[f32]> = xd.iter().map(|v| v.as_slice()).collect();
	let get = |g: &mut offload::Gemv| -> Result<(Vec<f32>, Vec<f32>), String> {
		let (mut st, mut cw) = (vec![0f32; DELTA_STATE / 2], vec![0f32; DELTA_WIN]);
		g.delta_get(0, &mut st, &mut cw)?;
		Ok((st, cw))
	};
	let stages: [&[usize]; 3] = [&[mi], &[cv], &[rc]];
	let mut res = Vec::new();
	// split first: aux slots a wrong p0 would leave unwritten are still 0
	for split in [true, false] {
		g.set_wave(0, 0);
		g.delta_put(0, &st0, &cw0)?;
		let mut ys = Vec::new();
		if split {
			g.run_chain_n_begin(&xdr[..2], &stages)?;
			g.set_wave(1, 2);
			g.run_chain_n_begin(&xdr[2..], &stages)?;
			g.wait_all()?;
			g.set_wave(0, 0);
			ys.extend((0..2).map(|p| g.y_at(p)[YA..YA + 6144].to_vec()));
			g.set_wave(1, 2);
			ys.extend((0..2).map(|p| g.y_at(p)[YA..YA + 6144].to_vec()));
			g.set_wave(0, 0);
		} else {
			g.run_chain_n(&xdr, &stages)?;
			ys.extend((0..4).map(|p| g.y_at(p)[YA..YA + 6144].to_vec()));
		}
		let after = get(g)?;
		let none: Vec<&[f32]> = vec![&[]; 3];
		g.run_chain_n(&none, &[&[win, rep]])?;
		let back3 = get(g)?;
		g.run_chain_n(&none[..1], &[&[win0, rep0]])?;
		let back0 = get(g)?;
		res.push((ys, after, back3, back0));
	}
	let (a, b) = (&res[0], &res[1]);
	let same =
		|x: &(Vec<f32>, Vec<f32>), y: &(Vec<f32>, Vec<f32>)| bits(&x.0, &y.0) && bits(&x.1, &y.1);
	let fine = (0..4).all(|p| bits(&a.0[p], &b.0[p]) && a.0[p].iter().all(|v| v.is_finite()));
	println!(
		"waves: delta mixer 2 + 2 vs 4: outputs {}, state + window {}, rolled back to 3 {}, to 0 {} (0 = the state put: {})",
		if fine { "bitwise equal" } else { "DIFFER" },
		if same(&a.1, &b.1) { "bitwise equal" } else { "DIFFER" },
		if same(&a.2, &b.2) { "bitwise equal" } else { "DIFFER" },
		if same(&a.3, &b.3) { "bitwise equal" } else { "DIFFER" },
		bits(&a.3 .0, &st0) && bits(&a.3 .1, &cw0)
	);
	Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
	eprintln!("gemvbench needs --features vulkan");
}
