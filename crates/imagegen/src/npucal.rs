//! The NPU backend's activation ranges (`npubuild::qimage`, `cal/dit.cal`):
//! the quantized CPU backend generates a few images while every
//! transformer block records the largest magnitude of its rotated QKV,
//! out and gate / up inputs ([`crate::dit::CAL`]), over the prefix and
//! target rows of all steps.

use crate::pipeline::{Model, Options};
use crate::WFmt;
use std::path::Path;

/// Prompts and sizes of the calibration images: photograph, lettering,
/// illustration, a transparent product shot, at two aspects.
const PROMPTS: [(&str, usize, usize); 4] = [
	(
		"A red fox sitting in a snowy forest at dawn, soft golden light, photograph, 85 mm",
		512,
		512,
	),
	(
		"A vintage travel poster with the large title \"QWEN CITY\" and a skyline at sunset, flat colours, bold lettering",
		640,
		384,
	),
	(
		"An isometric illustration of a tiny workshop with tools, plants and a cat, pastel palette, clean lines",
		448,
		576,
	),
	(
		"A glass perfume bottle on a transparent background, studio lighting, product shot",
		512,
		512,
	),
];

/// Headroom on the largest magnitude seen.
const HEADROOM: f32 = 1.25;

/// `[qkv, o, gu]` per block (see the module docs).
pub fn calibrate(
	src: &Path,
	layers: usize,
	log: &mut dyn FnMut(&str),
) -> Result<Vec<[f32; 3]>, String> {
	log("calibration: loading the quantized CPU backend");
	let m = Model::load(src, WFmt::Q6K, |_, _, _| {})?;
	*crate::dit::CAL.lock().unwrap() = Some(vec![[0.0; 3]; layers]);
	let r = (|| {
		for (i, &(p, w, h)) in PROMPTS.iter().enumerate() {
			let t = std::time::Instant::now();
			m.generate(
				&Options {
					prompt: p.to_string(),
					width: w,
					height: h,
					seed: 1 + i as u64,
					..Options::default()
				},
				|_, _, _| {},
			)?;
			log(&format!(
				"calibration image {}/{} ({w}x{h}) in {:.0} s",
				i + 1,
				PROMPTS.len(),
				t.elapsed().as_secs_f64()
			));
		}
		Ok::<(), String>(())
	})();
	let v = crate::dit::CAL.lock().unwrap().take().unwrap_or_default();
	r?;
	if v.len() != layers || v.iter().flatten().any(|&x| !(x > 0.0)) {
		return Err("calibration: a block recorded no range".into());
	}
	Ok(v.iter().map(|r| r.map(|x| x * HEADROOM)).collect())
}
