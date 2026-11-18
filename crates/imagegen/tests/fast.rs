//! The fast CPU backend (M3) against the reference dumps: the text encoder
//! and the transformer with their matmuls quantized once (`QWEN_IMAGE_FMTS`,
//! default `q6k,q4k`), per-module error and time. Not gated at 1e-3 (the
//! weights are quantized); prints the numbers the A/B is decided on. Needs
//! `QWEN_TEST_IMAGE_MODEL` and a text-to-image `QWEN_TEST_IMAGE_REF`; run
//! with `--release --test-threads 1 --nocapture`.

use imagegen::config::Config;
use imagegen::dit::{layout, Dit, Prefix, Rope};
use imagegen::refdump::{rel_err, Dumps};
use imagegen::sched::Schedule;
use imagegen::te::TextEncoder;
use imagegen::WFmt;
use std::path::PathBuf;
use std::time::Instant;

fn env() -> Option<(PathBuf, Dumps, Config, Vec<WFmt>)> {
	let m = PathBuf::from(std::env::var_os("QWEN_TEST_IMAGE_MODEL")?);
	let d = Dumps::open(&PathBuf::from(std::env::var_os("QWEN_TEST_IMAGE_REF")?)).unwrap();
	if d.manifest.get("images").is_some() {
		return None;
	}
	let c = Config::load(&m).unwrap();
	let f = std::env::var("QWEN_IMAGE_FMTS").unwrap_or_else(|_| "q6k,q4k".into());
	let f = f.split(',').map(|s| WFmt::parse(s).unwrap()).collect();
	Some((m, d, c, f))
}

#[test]
fn quantized_modules() {
	let Some((m, d, c, fmts)) = env() else { return };
	let s = d.num("size").unwrap() as usize / 16;
	let prompt = d.str("prompt").unwrap().to_string();
	for fmt in fmts {
		let t = Instant::now();
		let te = TextEncoder::load(&m, &c.te, fmt, |_| {}).unwrap();
		let tl = t.elapsed().as_secs_f64();
		let t = Instant::now();
		let enc = te.encode(&prompt, |_, _| {}).unwrap();
		let (r, _) = rel_err(&enc.hidden, &d.get("te_out").unwrap());
		println!(
			"{fmt:?}: text encoder {:.2} GB, load {tl:.1} s, encode {:.2} s, out rel {r:.2e}",
			te.resident_bytes() as f64 / 1e9,
			t.elapsed().as_secs_f64()
		);
		drop(te);

		let t = Instant::now();
		let dit = Dit::load(&m, &c.dit, fmt, |_| {}).unwrap();
		let tl = t.elapsed().as_secs_f64();
		let tx = d.get("te_out").unwrap();
		let ntxt = tx.len() / c.dit.ctx_dim;
		let lay = layout(&vec![false; ntxt], &[], (s, s)).unwrap();
		let rope = Rope::new(&c.dit, &lay.ids);
		let sch = Schedule::new(&c.sched, c.sample_sigmas.as_deref(), 0, s * s);
		let mut pre = Prefix::new(&lay);
		let mut txt = Some(dit.txt_in(&tx).unwrap());
		let mut worst = 0f64;
		let mut step_s = Vec::new();
		for i in 0..sch.steps() {
			let (md, _, _) = dit.modulation(sch.t(i, c.sched.train_steps)).unwrap();
			let x = if i == 0 {
				d.get("noise").unwrap()
			} else {
				d.get(&format!("latents{i}")).unwrap()
			};
			let t = Instant::now();
			let v = dit
				.forward(&x, &mut txt, &mut pre, &rope, &md, |_, _, _| {})
				.unwrap();
			step_s.push(t.elapsed().as_secs_f64());
			let (r, _) = rel_err(&v, &d.get(&format!("dit_v{i}")).unwrap());
			worst = worst.max(r);
		}
		println!(
			"{fmt:?}: transformer {:.2} GB, load {tl:.1} s, velocity rel <= {worst:.2e}, step 0 {:.2} s, steps 1.. {:.2} s each ({} target tokens)",
			dit.resident_bytes() as f64 / 1e9,
			step_s[0],
			step_s[1..].iter().sum::<f64>() / (step_s.len() - 1) as f64,
			s * s
		);
	}
}
