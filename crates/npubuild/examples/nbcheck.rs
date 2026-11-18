//! Development checks of `npubuild` against reference data.
use npubuild::{la, refm, st};

/// The float head on the final residuals `x` `[s][W]`, against truth logits.
fn kl_vs_truth(ck: &st::Checkpoint, x: &[f32], s: usize, truth: &str) -> Result<(), String> {
	let hn = refm::final_norm(ck, x)?;
	let truth = std::fs::read(truth).map_err(|e| e.to_string())?;
	let v = ck.shape("lm_head.weight")?[0];
	let (mut kl, mut top, mut kmax) = (0f64, 0usize, 0f64);
	let mut logits = vec![0f32; s * v];
	const R: usize = 8192;
	for r0 in (0..v).step_by(R) {
		let r1 = (r0 + R).min(v);
		let w = ck.rows("lm_head.weight", r0, r1)?;
		let y = la::linear(&hn, s, refm::W, &w, r1 - r0);
		for t in 0..s {
			logits[t * v + r0..t * v + r1].copy_from_slice(&y[t * (r1 - r0)..(t + 1) * (r1 - r0)]);
		}
	}
	for t in 0..s {
		let tr: Vec<f32> = truth[t * v * 4..(t + 1) * v * 4]
			.chunks_exact(4)
			.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
			.collect();
		let (k, same) = npubuild::kl_row(&tr, &logits[t * v..(t + 1) * v]);
		kl += k;
		kmax = kmax.max(k);
		top += same as usize;
	}
	println!(
		"KL(truth || ours) mean {:.6} (max {kmax:.4}), top-1 {top}/{s}",
		kl / s as f64
	);
	Ok(())
}

/// Logits of the Flash-Next head on `h` `[s][2560]` against truth logits.
fn flash_kl(ck: &st::Checkpoint, h: &[f32], s: usize, truth: &str) -> Result<(), String> {
	let w = npu::flash::W;
	let truth = std::fs::read(truth).map_err(|e| e.to_string())?;
	let v = ck.shape("lm_head.weight")?[0];
	let mut logits = vec![0f32; s * v];
	const R: usize = 8192;
	for r0 in (0..v).step_by(R) {
		let r1 = (r0 + R).min(v);
		let wr = ck.rows("lm_head.weight", r0, r1)?;
		let y = la::linear(h, s, w, &wr, r1 - r0);
		for t in 0..s {
			logits[t * v + r0..t * v + r1].copy_from_slice(&y[t * (r1 - r0)..(t + 1) * (r1 - r0)]);
		}
	}
	let (mut kl, mut top, mut kmax) = (0f64, 0usize, 0f64);
	for t in 0..s {
		let tr: Vec<f32> = truth[t * v * 4..(t + 1) * v * 4]
			.chunks_exact(4)
			.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
			.collect();
		let (k, same) = npubuild::kl_row(&tr, &logits[t * v..(t + 1) * v]);
		kl += k;
		kmax = kmax.max(k);
		top += same as usize;
	}
	println!(
		"KL(truth || ours) mean {:.6} (max {kmax:.4}), top-1 {top}/{s}",
		kl / s as f64
	);
	Ok(())
}

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	match a[0].as_str() {
		// flash-graphs DIR: the Flash-Next prefill graphs as IR files
		"flash-graphs" => {
			use npubuild::flash::graphs as fg;
			let d = std::path::Path::new(&a[1]);
			std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
			let s = fg::S;
			for (n, g) in [
				("ple", fg::ple(s)),
				("delta_pre", fg::delta_pre(s)),
				("delta_pre_m", fg::delta_pre_m(s)),
				("post_delta_a", fg::post_delta_a(s)),
				("attn_pre", fg::attn_pre(s)),
				("post_delta", fg::post(s, true, npu::flash::OSCALE_P as f64)),
				("post_attn", fg::post(s, false, 16.0)),
				("exp_g32c32", fg::experts(32, 32)),
				("pe_g32c32", fg::experts_pe(32, 32)),
				("pe_g16c64", fg::experts_pe(16, 64)),
				("pe_g32c64", fg::experts_pe(32, 64)),
				("pe_g16c128", fg::experts_pe(16, 128)),
				("hc_only", fg::hc_only(s)),
				("hc_only128", fg::hc_only(128)),
				("hc_only512", fg::hc_only(512)),
				("hc1", fg::hc_only_q(1, false)),
				("hc1_i8", fg::hc_only_q(1, true)),
				// with graphs/kern/hc (sets::write_kernel of shave/hcmix.sk)
				("hc_k", fg::hc_only_k(s, false, false)),
				("hc_km", fg::hc_only_k(s, true, false)),
				("hc_kj", fg::hc_only_k(s, false, true)),
			] {
				g.save(&d.join(format!("{n}.xml")), n)?;
				println!("{n}: {} ops", g.op_count());
			}
		}
		// flash-sim BF16_DIR OUT IDS TRUTH.f32: the Flash-Next prefill
		// pipeline (graphs on the OpenVINO CPU, OUT/w's quantized weights) on
		// the first 256 ids vs truth logits
		"flash-sim" => {
			use npubuild::flash::{refm as fr, sim};
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let out = std::path::Path::new(&a[2]);
			let ids: Vec<u32> = std::fs::read_to_string(&a[3])
				.map_err(|e| e.to_string())?
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.take(npubuild::flash::graphs::S)
				.collect();
			let table = npu::flash::ngram::Table::open(&out.join("ngram.bf16"))?;
			let mut sm = sim::Sim::new(&out.join("w"), &std::env::temp_dir().join("flashsim"))?;
			// FLASH_SIM_LAYERS=N: only N layers, compared with the float
			// reference's streams after them
			let nl: usize = std::env::var("FLASH_SIM_LAYERS")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(npu::flash::NL);
			let x = sim::run(&mut sm, &ck, &table, &ids, nl, &mut |m| eprintln!("{m}"))?;
			if nl < npu::flash::NL {
				let hash = npu::flash::ngram::Hash::new(npu::flash::VOCAB as u64, 0);
				let e = fr::ngram_embed(&hash, &table, &ids)?;
				let mut xr = fr::embed(&ck, &ids)?;
				for l in 0..nl {
					let ee = std::env::var_os("FLASH_SIM_NOPLE")
						.is_none()
						.then_some(&e[..]);
					fr::forward(&ck, &fr::load_layer(&ck, l)?, &mut xr, ee, ids.len())?;
				}
				let (mut d2, mut r2) = (0f64, 0f64);
				for (a, b) in x.iter().zip(&xr) {
					d2 += ((a - b) as f64).powi(2);
					r2 += (*b as f64).powi(2);
				}
				println!(
					"streams after {nl} layers: relative error {:.5} (rms {:.4})",
					(d2 / r2).sqrt(),
					(r2 / xr.len() as f64).sqrt()
				);
				let h = fr::final_mix(&ck, &x, ids.len())?;
				let hr = fr::final_mix(&ck, &xr, ids.len())?;
				let _ = (h, hr);
				return Ok(());
			}
			let h = fr::final_mix(&ck, &x, ids.len())?;
			flash_kl(&ck, &h, ids.len(), &a[4])?;
		}
		// flash-npucheck OUT KIND...: compiled blobs on the NPU vs their
		// graphs on the OpenVINO CPU (`sim::npu_check`)
		"flash-npucheck" => {
			let out = std::path::Path::new(&a[1]);
			npubuild::flash::sim::npu_check(&out.join("w"), &a[2..])?;
		}
		// flash-qerr BF16_DIR OUT IDS L: the quantization error of layer L's
		// quantized MatMuls on their real inputs (first 256 ids)
		"flash-qerr" => {
			use npubuild::flash::sim;
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let out = std::path::Path::new(&a[2]);
			let ids: Vec<u32> = std::fs::read_to_string(&a[3])
				.map_err(|e| e.to_string())?
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.take(npubuild::flash::graphs::S)
				.collect();
			let table = npu::flash::ngram::Table::open(&out.join("ngram.bf16"))?;
			let l: usize = a[4].parse().map_err(|_| "L")?;
			sim::qerr(&ck, &out.join("w"), &table, &ids, l, &mut |m| {
				println!("{m}")
			})?;
		}
		// flash-ref BF16_DIR IDS TRUTH.f32 NGRAM.bf16: the Flash-Next float
		// forward vs truth logits (`flash_ref.py`)
		"flash-ref" => {
			use npubuild::flash::refm as fr;
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let ids: Vec<u32> = std::fs::read_to_string(&a[2])
				.map_err(|e| e.to_string())?
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.collect();
			let s = ids.len();
			let table = npu::flash::ngram::Table::open(std::path::Path::new(&a[4]))?;
			let hash = npu::flash::ngram::Hash::new(npu::flash::VOCAB as u64, 0);
			if fr::ple_hash_of_checkpoint(&ck)? != hash {
				return Err("PLE hash differs from the checkpoint's buffers".into());
			}
			let e = fr::ngram_embed(&hash, &table, &ids)?;
			let mut x = fr::embed(&ck, &ids)?;
			let t0 = std::time::Instant::now();
			for l in 0..npu::flash::NL {
				let t = std::time::Instant::now();
				let layer = fr::load_layer(&ck, l)?;
				let tl = t.elapsed().as_secs_f64();
				let t = std::time::Instant::now();
				let taps = fr::forward(&ck, &layer, &mut x, Some(&e), s)?;
				let used = taps.edown.iter().filter(|e| !e.0.is_empty()).count();
				let rms =
					(x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
				eprintln!(
					"layer {l}: load {tl:.1} s, forward {:.1} s, {used} experts, rms {rms:.4}",
					t.elapsed().as_secs_f64()
				);
			}
			eprintln!("48 layers in {:.0} s", t0.elapsed().as_secs_f64());
			let h = fr::final_mix(&ck, &x, s)?;
			flash_kl(&ck, &h, s, &a[3])?;
		}
		// ref BF16_DIR IDS TRUTH.f32 [layers]: float forward vs truth logits
		"ref" => {
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let ids: Vec<u32> = std::fs::read_to_string(&a[2])
				.map_err(|e| e.to_string())?
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.collect();
			let s = ids.len();
			let nl: usize = a.get(4).map(|v| v.parse().unwrap()).unwrap_or(refm::NL);
			let mut x = refm::embed(&ck, &ids)?;
			let dump = |l: usize, x: &[f32]| {
				if let Ok(d) = std::env::var("NB_DUMP") {
					std::fs::write(
						format!("{d}/x{l}.f32"),
						x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
					)
					.unwrap();
				}
			};
			dump(0, &x);
			let t0 = std::time::Instant::now();
			for l in 0..nl {
				let t = std::time::Instant::now();
				let layer = refm::load_layer(&ck, l)?;
				let tl = t.elapsed().as_secs_f64();
				let t = std::time::Instant::now();
				x = refm::forward(&layer, &x, s).0;
				dump(l + 1, &x);
				let rms =
					(x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
				eprintln!(
					"layer {l}: load {tl:.1} s, forward {:.1} s, rms {rms:.4}",
					t.elapsed().as_secs_f64()
				);
			}
			eprintln!("{nl} layers in {:.0} s", t0.elapsed().as_secs_f64());
			kl_vs_truth(&ck, &x, s, &a[3])?;
		}
		// gptq DIR n k: DIR/{w,h,s}.f32 (rotated weight, rotated Hessian, scales), q.i8 (reference codes)
		"gptq" => {
			let (n, k): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
			let rd = |f: &str| -> Vec<f32> {
				std::fs::read(format!("{}/{f}", a[1]))
					.unwrap()
					.chunks_exact(4)
					.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
					.collect()
			};
			let (w, h, sref) = (rd("w.f32"), rd("h.f32"), rd("s.f32"));
			let qref: Vec<i8> = std::fs::read(format!("{}/q.i8", a[1]))
				.unwrap()
				.iter()
				.map(|&b| b as i8)
				.collect();
			let t = std::time::Instant::now();
			let sc = npubuild::gptq::row_scales(&w, n, k);
			let ds = sc
				.iter()
				.zip(&sref)
				.filter(|(a, b)| (**a - **b).abs() > 1e-6 * b.abs())
				.count();
			eprintln!(
				"scales {:.2} s, {ds} of {n} differ",
				t.elapsed().as_secs_f64()
			);
			let t = std::time::Instant::now();
			let f = npubuild::gptq::factor(h.clone(), k)?;
			eprintln!("factor {:.2} s", t.elapsed().as_secs_f64());
			#[cfg(feature = "vulkan")]
			{
				let mut gpu = npubuild::gpufac::GpuFactor::new()?;
				let p = npubuild::gptq::prepare(h.clone(), k);
				let t = std::time::Instant::now();
				let x = gpu.inv_chol(&p.hr, k)?;
				let dt = t.elapsed().as_secs_f64();
				let (mut worst, mut big) = (0f32, 0f32);
				for (a, b) in x.iter().zip(&f.x) {
					worst = worst.max((a - b).abs());
					big = big.max(b.abs());
				}
				eprintln!("gpu factor {dt:.2} s; max |gpu - cpu| {worst:.3e} of max {big:.3e}");
			}
			let t = std::time::Instant::now();
			let q = npubuild::gptq::gptq(w.clone(), n, &f, &sref);
			eprintln!("gptq {:.2} s", t.elapsed().as_secs_f64());
			let same = q.iter().zip(&qref).filter(|(a, b)| a == b).count();
			// Hessian-weighted error tr(D H D^T) of both
			let err = |q: &[i8]| -> f64 {
				let d: Vec<f32> = (0..n * k)
					.map(|i| q[i] as f32 * sref[i / k] - w[i])
					.collect();
				let dh = la::linear(&d, n, k, &h, k);
				d.iter().zip(&dh).map(|(a, b)| *a as f64 * *b as f64).sum()
			};
			let rt = npubuild::gptq::rtn(&w, n, k, &sref);
			println!(
				"codes equal {:.2} %; error ours {:.5e} ref {:.5e} rtn {:.5e}",
				100.0 * same as f64 / q.len() as f64,
				err(&q),
				err(&qref),
				err(&rt)
			);
		}
		// cal BF16_DIR OUT IDS...: OUT/cal, OUT/w (NB_LAYERS limits the layers by failing after them)
		"cal" => {
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let mut chunks = Vec::new();
			for f in &a[3..] {
				let ids: Vec<u32> = std::fs::read_to_string(f)
					.map_err(|e| e.to_string())?
					.split_whitespace()
					.map(|v| v.parse().unwrap())
					.collect();
				chunks.extend(ids.chunks_exact(256).map(|c| c.to_vec()));
			}
			eprintln!("{} chunks", chunks.len());
			let out = std::path::Path::new(&a[2]);
			let stop: usize = std::env::var("NB_LAYERS")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(64);
			let (accel, what) = npubuild::weights::Accel::new(std::env::var("NB_NOGPU").is_err());
			eprintln!("GPTQ on: {what}");
			let r = npubuild::weights::calibrate(
				&ck,
				&chunks,
				&out.join("cal"),
				&out.join("w"),
				&accel,
				&mut |m| {
					eprintln!("{m}");
					if m.starts_with(&format!("layer {}:", stop - 1)) {
						std::process::exit(0);
					}
				},
			);
			r?;
		}
		// set BF16_DIR OUT S CAP LAYERS: OUT/w/lean{L}.xml for the first LAYERS layers
		"set" => {
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let out = std::path::Path::new(&a[2]);
			let (s, cap, nl): (usize, usize, usize) = (
				a[3].parse().unwrap(),
				a[4].parse().unwrap(),
				a[5].parse().unwrap(),
			);
			for l in 0..nl {
				let g = if refm::is_attn(l) {
					let c = npubuild::weights::attn_consts(&ck, &out.join("cal"), l)?;
					npubuild::lean::attn(s, cap, npubuild::lean::Part::All, &c)
				} else {
					let f = if std::env::var("NB_EMU").is_ok() {
						npubuild::lean::delta_emu
					} else {
						npubuild::lean::delta
					};
					f(
						s,
						npubuild::lean::Cut::Whole,
						if l == 0 { 16.0 } else { 256.0 },
					)
				};
				g.save(&out.join("w").join(format!("lean{l}.xml")), "lean")?;
			}
		}
		// resid BF16_DIR RESID.f16 TRUTH.f32: an engine's final residuals (QWEN_NPU_DUMP_RESID) through the float head
		"resid" => {
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let x: Vec<f32> = std::fs::read(&a[2])
				.map_err(|e| e.to_string())?
				.chunks_exact(2)
				.map(|b| npubuild::g::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
				.collect();
			kl_vs_truth(&ck, &x, x.len() / refm::W, &a[3])?;
		}
		// dec BF16_DIR OUT LAYER MLP(0|1) FILE.xml: a verify-set layer graph
		"dec" => {
			let ck = st::Checkpoint::open(std::path::Path::new(&a[1]))?;
			let cal = std::path::Path::new(&a[2]).join("cal");
			let (l, mlp): (usize, bool) = (a[3].parse().unwrap(), a[4] == "1");
			let g = if refm::is_attn(l) {
				npubuild::dec::attn_verify(
					4,
					4096,
					l / 4,
					&npubuild::weights::attn_c(&ck, &cal, l)?,
					mlp,
				)
			} else {
				npubuild::dec::delta_verify(
					4,
					l - l / 4,
					&npubuild::weights::delta_c(&ck, &cal, l)?,
					mlp,
				)
			};
			g.save(std::path::Path::new(&a[5]), "layer")?;
			println!(
				"{} ops; inputs {:?}; outputs {:?}",
				g.op_count(),
				g.inputs().iter().map(|i| i.0.clone()).collect::<Vec<_>>(),
				g.outputs()
			);
		}
		// delta S whole|pre|post VMUL OUT.xml
		"delta" => {
			let cut = match a[2].as_str() {
				"pre" => npubuild::lean::Cut::Pre,
				"post" => npubuild::lean::Cut::Post,
				_ => npubuild::lean::Cut::Whole,
			};
			let g = npubuild::lean::delta(a[1].parse().unwrap(), cut, a[3].parse().unwrap());
			g.save(std::path::Path::new(&a[4]), "delta")?;
			println!("{} ops", g.op_count());
		}
		// core S CAP near|near0|far|farm OUT.xml
		"core" => {
			use npubuild::lean::Part;
			let part = match a[3].as_str() {
				"near" => Part::Near,
				"near0" => Part::Near0,
				"farm" => Part::Far(8),
				_ => Part::Far(1),
			};
			npubuild::lean::attn_core(a[1].parse().unwrap(), a[2].parse().unwrap(), part)
				.save(std::path::Path::new(&a[4]), "attn")?;
		}
		_ => return Err("nbcheck ref | gptq | delta | core ...".into()),
	}
	Ok(())
}
