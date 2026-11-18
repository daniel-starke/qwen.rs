//! The M1 gate: every module against `scripts/refdump.py`'s dumps of the
//! diffusers pipeline (f32, CPU), each fed the reference's own inputs so an
//! error is the module's. Needs `QWEN_TEST_IMAGE_MODEL` (the checkpoint
//! directory) and `QWEN_TEST_IMAGE_REF` (the dump directory); skipped
//! without them. Run with `--release --test-threads 1 --nocapture`.

use imagegen::config::Config;
use imagegen::dit::{layout, t2i_ids, Dit, Prefix, Rope};
use imagegen::refdump::{rel_err, Dumps};
use imagegen::sched::Schedule;
use imagegen::te::TextEncoder;
use imagegen::vae::Vae;
use std::path::PathBuf;

const TOL: f64 = 1e-3;

fn env() -> Option<(PathBuf, Dumps, Config)> {
	let m = std::env::var_os("QWEN_TEST_IMAGE_MODEL")?;
	let r = std::env::var_os("QWEN_TEST_IMAGE_REF")?;
	let m = PathBuf::from(m);
	let d = Dumps::open(&PathBuf::from(r)).unwrap();
	if d.manifest.get("images").is_some() {
		println!("QWEN_TEST_IMAGE_REF holds an edit's dumps: see tests/edit.rs");
		return None;
	}
	let c = Config::load(&m).unwrap();
	Some((m, d, c))
}

fn check(what: &str, got: &[f32], want: &[f32]) -> f64 {
	let (r, mx) = rel_err(got, want);
	println!("{what:<24} rel {r:.2e}  max abs {mx:.2e}");
	r
}

fn size(d: &Dumps) -> (usize, usize) {
	let s = d.num("size").unwrap() as usize;
	(s / 16, s / 16)
}

#[test]
fn noise_and_schedule() {
	let Some((_, d, c)) = env() else { return };
	let (h, w) = size(&d);
	let seed = d.num("seed").unwrap() as u64;
	let n = imagegen::pipeline::noise(seed, c.dit.in_ch, h, w);
	assert!(check("noise", &n, &d.get("noise").unwrap()) < 1e-6);
	let s = Schedule::new(&c.sched, c.sample_sigmas.as_deref(), 0, h * w);
	assert_eq!(s.sigmas, d.get("sched_sigmas").unwrap());
}

#[test]
fn text_encoder() {
	let Some((m, d, c)) = env() else { return };
	let te = TextEncoder::open(&m, &c.te).unwrap();
	let prompt = d.str("prompt").unwrap().to_string();
	let (ids, drop) = te.tokenize(&prompt).unwrap();
	let want: Vec<u32> = d
		.get("te_input_ids")
		.unwrap()
		.iter()
		.map(|&v| v as u32)
		.collect();
	assert_eq!(ids, want, "token ids");
	assert_eq!(
		drop,
		d.num("drop_idx").unwrap() as usize,
		"system turn length"
	);
	let mut worst = 0f64;
	let enc = te
		.encode(&prompt, |l, x| {
			if d.shape(&format!("te_layer{l}")).is_some() {
				worst = worst.max(check(
					&format!("te layer {l}"),
					x,
					&d.get(&format!("te_layer{l}")).unwrap(),
				));
			}
		})
		.unwrap();
	worst = worst.max(check("te out", &enc.hidden, &d.get("te_out").unwrap()));
	assert!(worst < TOL, "text encoder rel err {worst:.2e}");
}

#[test]
fn transformer() {
	let Some((m, d, c)) = env() else { return };
	let (h, w) = size(&d);
	let dit = Dit::open(&m, &c.dit).unwrap();
	let te = d.get("te_out").unwrap();
	let ntxt = te.len() / c.dit.ctx_dim;
	let mut worst = 0f64;
	let txt = dit.txt_in(&te).unwrap();
	worst = worst.max(check("txt_in", &txt, &d.get("dit_txt_in").unwrap()));
	let lay = layout(&vec![false; ntxt], &[], (h, w)).unwrap();
	assert_eq!(lay.ids, t2i_ids(ntxt, h, w));
	let rope = Rope::new(&c.dit, &lay.ids);
	let s = Schedule::new(&c.sched, c.sample_sigmas.as_deref(), 0, h * w);
	let mut pre = Prefix::new(&lay);
	let mut txt = Some(txt);
	let steps: Vec<usize> = match std::env::var("QWEN_TEST_IMAGE_STEPS") {
		Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
		Err(_) => (0..s.steps()).collect(),
	};
	for i in 0..s.steps() {
		if !steps.contains(&i) && i != 0 {
			continue;
		}
		let (md, temb, raw) = dit.modulation(s.t(i, c.sched.train_steps)).unwrap();
		if i == 0 {
			worst = worst.max(check("temb", &temb, &d.get("dit_temb").unwrap()));
			worst = worst.max(check("modulation", &raw, &d.get("dit_modulation").unwrap()));
		}
		let x = if i == 0 {
			d.get("noise").unwrap()
		} else {
			d.get(&format!("latents{i}")).unwrap()
		};
		let v = dit
			.forward(&x, &mut txt, &mut pre, &rope, &md, |l, p, t| {
				if i == 0 {
					let full = [p.unwrap(), t].concat();
					let e = check(
						&format!("block {l}"),
						&full,
						&d.get(&format!("dit_block{l}")).unwrap(),
					);
					worst = worst.max(e);
				}
			})
			.unwrap();
		worst = worst.max(check(
			&format!("velocity step {i}"),
			&v,
			&d.get(&format!("dit_v{i}")).unwrap(),
		));
	}
	assert!(worst < TOL, "transformer rel err {worst:.2e}");
}

#[test]
fn vae_decoder() {
	let Some((m, d, c)) = env() else { return };
	let (h, w) = size(&d);
	let vae = Vae::open(&m, &c.vae).unwrap();
	let mut worst = 0f64;
	let t0 = std::time::Instant::now();
	let img = vae
		.decode(&d.get("latents_final").unwrap(), h, w, |name, x| {
			let key = format!("vae_dec_{name}");
			if d.shape(&key).is_some() {
				// dumps are channel-major [c][h][w]
				let want = d.get(&key).unwrap();
				let mut got = vec![0f32; want.len()];
				for p in 0..x.h * x.w {
					for ch in 0..x.c {
						got[ch * x.h * x.w + p] = x.d[p * x.c + ch];
					}
				}
				worst = worst.max(check(&key, &got, &want));
			}
		})
		.unwrap();
	println!("vae decode {:.1} s", t0.elapsed().as_secs_f64());
	let want = d.get("vae_out").unwrap();
	let mut got = vec![0f32; want.len()];
	for p in 0..img.h * img.w {
		for ch in 0..4 {
			got[ch * img.h * img.w + p] = img.d[p * 4 + ch];
		}
	}
	worst = worst.max(check("vae out", &got, &want));
	assert!(worst < TOL, "vae rel err {worst:.2e}");
}
