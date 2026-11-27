//! The NPU build's calibration (`npubuild::h3img`, `cal/h3dit.cal` and
//! `cal/h3adaln.bin`): the quantized CPU backend generates a few images
//! while every transformer block records its statistics
//! ([`super::dit::CAL`]) and, for GPTQ, its inputs' Hessians
//! ([`super::dit::HESS`]) over all rows of all steps, and the resident AdaLN
//! tables are written for the modulation files.

use super::config::MODALITIES;
use super::pipeline::{Model, MIN_AREA_OVERRIDE};
use crate::job::{ImageSrc, Options};
use crate::WFmt;
use npubuild::h3img::{self, Cal};
use std::path::Path;
use std::sync::atomic::Ordering;

/// Prompts and sizes of the calibration images (frozen: a calibration is
/// reproducible): a photograph, lettering, a portrait at ~1 MP, an
/// illustration at the 4 MP default, and an edit of the first image.
const PROMPTS: [(&str, usize, usize); 4] = [
	(
		"A red fox sitting in a snowy forest at dawn, soft golden light, photograph, 85 mm",
		512,
		512,
	),
	(
		"A vintage travel poster with the large title \"H3 CITY\" and a skyline at sunset, flat colours, bold lettering",
		512,
		512,
	),
	(
		"A portrait of an elderly fisherman with a weathered face and a knitted cap, harbour in the background, overcast light, photograph",
		768,
		1024,
	),
	(
		"An isometric illustration of a tiny workshop with tools, plants and a cat, pastel palette, clean lines",
		2048,
		2048,
	),
];

const EDIT: &str =
	"Turn <Picture 1> into a watercolor painting at night, with the moon behind the trees";

/// Headroom on the largest rotated inputs seen.
const HEADROOM: f32 = 1.25;

/// Write the calibration of the checkpoint `src` into `out`.
pub fn calibrate(src: &Path, out: &Path, log: &mut dyn FnMut(&str)) -> Result<(), String> {
	log("calibration: loading the quantized CPU backend");
	let m = Model::load(src, WFmt::Q6K, |_, _, _| {})?;
	let (ts, tables) = m.dit().tables().ok_or("calibration: no AdaLN tables")?;
	h3img::write_adaln(out, ts, m.cfg.dit.hidden, MODALITIES, tables)?;
	*super::dit::CAL.lock().unwrap() = Some(Cal::default());
	let shape = h3img::DShape::read(src)?;
	let hess = h3img::gptq_on();
	if hess {
		let w = h3img::hess_widths(&shape);
		*super::dit::HESS.lock().unwrap() = Some(
			(0..shape.layers)
				.map(|_| w.map(npubuild::gptq::Gram::new))
				.collect(),
		);
	}
	MIN_AREA_OVERRIDE.store(1024, Ordering::Relaxed);
	let r = (|| {
		let mut first = None;
		for (i, &(p, w, h)) in PROMPTS.iter().enumerate() {
			let t = std::time::Instant::now();
			let o = m.generate(
				&Options {
					prompt: p.to_string(),
					width: w,
					height: h,
					seed: 1 + i as u64,
					..Options::default()
				},
				|_, _, _| {},
			)?;
			if first.is_none() {
				first = Some(ImageSrc::Rgba(o.width, o.height, o.rgba));
			}
			log(&format!(
				"calibration image {}/{} ({w}x{h}) in {:.0} s",
				i + 1,
				PROMPTS.len() + 1,
				t.elapsed().as_secs_f64()
			));
		}
		let t = std::time::Instant::now();
		m.generate(
			&Options {
				prompt: EDIT.to_string(),
				width: 512,
				height: 512,
				seed: 9,
				images: vec![first.unwrap()],
				..Options::default()
			},
			|_, _, _| {},
		)?;
		log(&format!(
			"calibration image {n}/{n} (edit, 512x512) in {:.0} s",
			t.elapsed().as_secs_f64(),
			n = PROMPTS.len() + 1
		));
		Ok::<(), String>(())
	})();
	MIN_AREA_OVERRIDE.store(0, Ordering::Relaxed);
	let mut c = super::dit::CAL.lock().unwrap().take().unwrap_or_default();
	let hs = super::dit::HESS.lock().unwrap().take();
	r?;
	if let Some(hs) = hs {
		let t = std::time::Instant::now();
		for (l, g) in hs.iter().enumerate() {
			if g.iter().any(|g| g.n == 0) {
				return Err(format!("calibration: block {l} recorded no Hessian rows"));
			}
			h3img::write_hess(out, l, g)?;
		}
		log(&format!(
			"calibration: Hessians of {} rows (QKV) written in {:.0} s",
			hs[0][0].n,
			t.elapsed().as_secs_f64()
		));
	}
	if c.blocks.len() != m.cfg.dit.layers
		|| !(c.xin > 0.0)
		|| c.blocks.iter().any(|b| b[..4].iter().any(|&x| !(x > 0.0)))
	{
		return Err("calibration: a block recorded no range".into());
	}
	for b in c.blocks.iter_mut() {
		for v in &mut b[..3] {
			*v *= HEADROOM;
		}
	}
	let worst = c
		.blocks
		.iter()
		.fold([0f32; 2], |a, b| [a[0].max(b[4]), a[1].max(b[5])]);
	log(&format!(
		"calibration: residual max {:.3e} (block 49), MLP gate max {:.3e}, hidden max {:.3e}",
		c.blocks[c.blocks.len() - 1][3],
		worst[0],
		worst[1]
	));
	c.write(out)
}
