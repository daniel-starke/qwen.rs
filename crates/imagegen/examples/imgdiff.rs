//! `imgdiff A.png [B.png]`: per-channel means of A; with B, PSNR, the
//! largest difference and the share of values off by more than 8.

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let (w, h, x) = imgio::load_rgba(a.first().ok_or("imgdiff A.png [B.png]")?)?;
	let mean: Vec<f64> = (0..4)
		.map(|c| x.iter().skip(c).step_by(4).map(|&v| v as f64).sum::<f64>() / (w * h) as f64)
		.collect();
	println!("{w}x{h}, channel means {mean:.1?}");
	if let Some(b) = a.get(1) {
		let (w2, h2, y) = imgio::load_rgba(b)?;
		if (w, h) != (w2, h2) {
			return Err(format!("sizes differ: {w}x{h} vs {w2}x{h2}"));
		}
		let mut se = 0f64;
		let (mut mx, mut over) = (0u8, 0usize);
		for (&p, &q) in x.iter().zip(&y) {
			let d = p.abs_diff(q);
			se += (d as f64).powi(2);
			mx = mx.max(d);
			over += (d > 8) as usize;
		}
		let mse = se / x.len() as f64;
		println!(
			"PSNR {:.1} dB, max diff {mx}, {:.1} % of values off by more than 8",
			10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10(),
			100.0 * over as f64 / x.len() as f64
		);
	}
	Ok(())
}
