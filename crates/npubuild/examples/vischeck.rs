//! The NPU vision tower against the CPU one (`vision::tower::forward`).
//!
//!   vischeck BF16_DIR OUT IMAGE [MAX_TOKENS] [MMPROJ.gguf]
//!
//! Writes the tower graphs to `OUT` (`vis{i}.xml`), compiles the missing
//! blobs with the NPU driver, runs `IMAGE` through both towers (the NPU
//! one's attention on the cores) and prints how far apart the embeddings
//! are. With an mmproj: also that its weights equal the checkpoint's.
//! `OUT` is then a tower set (`vision = OUT` in a `.npu` file), as `cli
//! BF16_DIR --npu-build` writes to `sets/vis`. `QWEN_NPU_VIS_TIMING=1`
//! splits the NPU tower's time.
use npubuild::{st, vis};

struct CpuAttn(vision::VisionConfig);

impl npu::vis::VisAttn for CpuAttn {
	fn attend(
		&mut self,
		_n: usize,
		qkv: &[u8],
		hw: &[(u32, u32)],
		out: &mut [u8],
	) -> Result<(), String> {
		vision::tower::attend_f16(&self.0, hw, qkv, out);
		Ok(())
	}
	fn name(&self) -> String {
		"cores".into()
	}
}

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	if a.len() < 3 {
		return Err("vischeck BF16_DIR OUT IMAGE [MAX_TOKENS] [MMPROJ.gguf]".into());
	}
	let ck = st::Checkpoint::open(std::path::Path::new(&a[0]))?;
	let out = std::path::Path::new(&a[1]);
	std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
	let m = vis::model(&ck)?;
	if let Some(p) = a.get(4) {
		let g = vision::VisionModel::load(p, None)?;
		let d = |x: &[f32], y: &[f32]| {
			x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(
				if x.len() == y.len() {
					0f32
				} else {
					f32::INFINITY
				},
				f32::max,
			)
		};
		let mut worst = d(&m.patch_w, &g.patch_w)
			.max(d(&m.patch_b, &g.patch_b))
			.max(d(&m.pos_w, &g.pos_w));
		worst = worst
			.max(d(&m.post_ln_w, &g.post_ln_w))
			.max(d(&m.post_ln_b, &g.post_ln_b));
		worst = worst
			.max(d(&m.mm0_w, &g.mm0_w))
			.max(d(&m.mm0_b, &g.mm0_b))
			.max(d(&m.mm2_w, &g.mm2_w))
			.max(d(&m.mm2_b, &g.mm2_b));
		for (x, y) in m.blocks.iter().zip(&g.blocks) {
			for (p, q) in [
				(&x.ln1_w, &y.ln1_w),
				(&x.ln1_b, &y.ln1_b),
				(&x.ln2_w, &y.ln2_w),
				(&x.ln2_b, &y.ln2_b),
				(&x.qkv_w, &y.qkv_w),
				(&x.qkv_b, &y.qkv_b),
			] {
				worst = worst.max(d(p, q));
			}
			for (p, q) in [
				(&x.out_w, &y.out_w),
				(&x.out_b, &y.out_b),
				(&x.up_w, &y.up_w),
				(&x.up_b, &y.up_b),
				(&x.down_w, &y.down_w),
				(&x.down_b, &y.down_b),
			] {
				worst = worst.max(d(p, q));
			}
		}
		println!("checkpoint vs mmproj: {} blocks vs {}, eps {} vs {}, mean {:?} vs {:?}, max |weight difference| {worst:e}", m.blocks.len(), g.blocks.len(), m.cfg.eps, g.cfg.eps, m.cfg.mean, g.cfg.mean);
	}
	let img = imgio::decode(&std::fs::read(&a[2]).map_err(|e| format!("{}: {e}", a[2]))?)?;
	let max_tokens: usize = a.get(3).map_or(256, |v| v.parse().unwrap());
	let c = &m.cfg;
	let grid = m.grid_for(img.width, img.height, max_tokens)?;
	let patches = vision::patchify(&img, &grid, c.patch_size, c.merge_size, c.mean, c.std);
	let t = std::time::Instant::now();
	let want = vision::tower::forward(&m, &patches, &grid);
	println!(
		"{}x{} -> {} patches, {} tokens; CPU tower {:.2} s",
		img.width,
		img.height,
		grid.patches(),
		grid.tokens(),
		t.elapsed().as_secs_f64()
	);

	let _lock = npu::lock::NpuLock::acquire()?;
	let dev: &'static npu::ze::Npu = Box::leak(Box::new(npu::ze::Npu::open()?));
	for i in 0..=c.n_layer {
		let (xml, blob) = (
			out.join(format!("vis{i}.xml")),
			out.join(format!("vis{i}.blob")),
		);
		if !blob.exists() {
			let g = vis::graph(&m, i, vis::S);
			g.save(&xml, &format!("vis{i}"))?;
			let t = std::time::Instant::now();
			let rd =
				|p: &std::path::Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
			let b = dev
				.compile(&rd(&xml)?, &rd(&xml.with_extension("bin"))?, "", 0)
				.and_then(|g| g.native())
				.map_err(|e| format!("vis{i}: {e}"))?;
			std::fs::write(&blob, b).map_err(|e| e.to_string())?;
			eprintln!("vis{i}: compiled in {:.1} s", t.elapsed().as_secs_f64());
		}
	}
	std::fs::write(out.join("vis.cfg"), vis::cfg_text(c, vis::S)).map_err(|e| e.to_string())?;
	std::fs::write(
		out.join("pos.f32"),
		m.pos_w
			.iter()
			.flat_map(|v| v.to_le_bytes())
			.collect::<Vec<u8>>(),
	)
	.map_err(|e| e.to_string())?;
	let t = std::time::Instant::now();
	let mut nv = npu::vis::NpuVision::load(dev, &a[1])?;
	eprintln!("tower loaded in {:.1} s", t.elapsed().as_secs_f64());
	let pos = vision::tower::position_embedding(c, &nv.pos_w, &grid);
	let hw = vision::patch_positions(&grid, c.merge_size);
	let mut attn = CpuAttn(c.clone());
	for round in 0..2 {
		let t = std::time::Instant::now();
		let got = nv.encode(&patches, &pos, &hw, &mut attn)?;
		let dt = t.elapsed().as_secs_f64();
		let (mut num, mut den, mut worst, mut dot, mut gg) = (0f64, 0f64, 0f32, 0f64, 0f64);
		for (x, y) in want.iter().zip(&got) {
			num += ((x - y) as f64).powi(2);
			den += (*x as f64).powi(2);
			dot += *x as f64 * *y as f64;
			gg += (*y as f64).powi(2);
			worst = worst.max((x - y).abs());
		}
		println!("round {round}: NPU tower {dt:.2} s; relative error {:.4}, cosine {:.6}, max |difference| {worst:.3} (rms of the reference {:.3})", (num / den).sqrt(), dot / (den * gg).sqrt(), (den / want.len() as f64).sqrt());
	}
	Ok(())
}
