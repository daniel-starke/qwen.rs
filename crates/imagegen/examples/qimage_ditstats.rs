//! Largest magnitudes of the transformer's residual stream per block (what
//! an f16 NPU graph has to hold): `qimage_ditstats QWEN_IMAGE_DIR [SIZE] [PROMPT]`.

use imagegen::qimage::dit::{layout, Dit, Prefix, Rope};
use imagegen::qimage::sched::Schedule;
use imagegen::qimage::Config;
use imagegen::te::TextEncoder;
use imagegen::WFmt;
use std::path::Path;

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let dir = Path::new(
		a.first()
			.ok_or("qimage_ditstats QWEN_IMAGE_DIR [SIZE] [PROMPT]")?,
	);
	let size: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
	let prompt = a.get(2).map_or(
		"A red fox sitting in a snowy forest at dawn, photograph",
		String::as_str,
	);
	let cfg = Config::load(dir)?;
	let te = TextEncoder::load(dir, &cfg.te, WFmt::Q6K, |_| {})?;
	let enc = te.encode(prompt, |_, _| {})?;
	drop(te);
	let dit = Dit::load(dir, &cfg.dit, WFmt::Q6K, |_| {})?;
	let (h, w) = (size / 16, size / 16);
	let lay = layout(&enc.image_pad, &[], (h, w))?;
	let mut txt = Some(dit.prefix_rows(&enc.hidden, &lay, &[])?);
	let rope = Rope::new(&cfg.dit, &lay.ids);
	let mut pre = Prefix::new(&lay);
	let sched = Schedule::new(&cfg.sched, cfg.sample_sigmas.as_deref(), 8, h * w);
	let mut x = imagegen::qimage::pipeline::noise(42, cfg.dit.in_ch, h, w);
	let amax = |v: &[f32]| v.iter().fold(0f32, |m, &x| m.max(x.abs()));
	let d = cfg.dit.dim();
	for i in 0..sched.steps() {
		let (m, _, _) = dit.modulation(sched.t(i, cfg.sched.train_steps))?;
		println!(
			"step {i}: |mod target| {:.3}, |mod prefix| {:.3}",
			amax(&m.target),
			amax(&m.prefix)
		);
		let v = dit.forward(&x, &mut txt, &mut pre, &rope, &m, |l, p, t| {
			let rms = (t.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / t.len() as f64).sqrt();
			// the largest channel
			let mut ch = vec![0f32; d];
			for r in t.chunks_exact(d) {
				for (c, &v) in ch.iter_mut().zip(r) {
					*c = c.max(v.abs());
				}
			}
			let top = ch
				.iter()
				.enumerate()
				.max_by(|a, b| a.1.total_cmp(b.1))
				.unwrap();
			println!(
				"  L{l:2}: target max {:9.1} rms {:7.2} (channel {} {:.1}){}",
				amax(t),
				rms,
				top.0,
				top.1,
				p.map_or(String::new(), |p| format!(", prefix max {:9.1}", amax(p)))
			);
		})?;
		sched.step(i, &mut x, &v);
	}
	Ok(())
}
