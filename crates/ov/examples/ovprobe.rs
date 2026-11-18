//! `ovprobe [--lib DIR] [EXPORT_DIR]`: load the OpenVINO runtime, list its
//! devices, and (given an export directory) run the text-embedding model on a
//! few ids. A check of the runtime install and the FFI that loads no 14 GB
//! language model.

use std::path::PathBuf;

fn main() -> Result<(), String> {
	let mut lib: Option<PathBuf> = None;
	let mut dir: Option<PathBuf> = None;
	let mut it = std::env::args().skip(1);
	while let Some(a) = it.next() {
		if a == "--lib" {
			lib = it.next().map(PathBuf::from);
		} else {
			dir = Some(PathBuf::from(a));
		}
	}
	let rt = ov::Runtime::new(lib.as_deref())?;
	println!(
		"openvino {} from {}",
		rt.api.version(),
		rt.api.dir.display()
	);
	for d in rt.devices() {
		println!(
			"  {d}: {}",
			rt.property(&d, "FULL_DEVICE_NAME").unwrap_or_default()
		);
	}
	let Some(dir) = dir else { return Ok(()) };
	let m = rt.read_model(&dir.join("openvino_text_embeddings_model.xml"))?;
	let c = rt.compile(&m, "CPU", &[])?;
	let mut r = c.request()?;
	let ids: Vec<i64> = vec![248045, 846, 198, 9707];
	r.set("input", &[1, ids.len() as i64], &ids)?;
	r.infer()?;
	let (shape, e) = r.output_f32("inputs_embeds")?;
	let d = *shape.last().unwrap() as usize;
	for (i, row) in e.chunks(d).enumerate() {
		let rms = (row.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
		println!("  id {}: rms {rms:.5} first {:?}", ids[i], &row[..4]);
	}
	Ok(())
}
