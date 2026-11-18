//! The M2 gate: an edit's modules against `scripts/refdump.py --image`'s
//! dumps of the diffusers pipeline (f32, CPU), each fed the reference's own
//! inputs so an error is the module's. Needs `QWEN_TEST_IMAGE_MODEL` (the
//! checkpoint directory) and `QWEN_TEST_IMAGE_EDIT_REF` (the dump directory
//! of a run with `--image`); skipped without them. Run with `--release
//! --test-threads 1 --nocapture`.

use imagegen::cond::Cond;
use imagegen::config::Config;
use imagegen::dit::{layout, Dit, Layout, Prefix, Rope};
use imagegen::refdump::{rel_err, Dumps};
use imagegen::sched::Schedule;
use imagegen::te::{TextEncoder, VisionRows};
use imagegen::vae::{Map, Vae};
use npubuild::st::Json;
use std::path::PathBuf;

const TOL: f64 = 1e-3;

fn env() -> Option<(PathBuf, Dumps, Config)> {
	let m = PathBuf::from(std::env::var_os("QWEN_TEST_IMAGE_MODEL")?);
	let r = PathBuf::from(std::env::var_os("QWEN_TEST_IMAGE_EDIT_REF")?);
	let d = Dumps::open(&r).unwrap();
	assert!(
		d.manifest.get("images").is_some(),
		"QWEN_TEST_IMAGE_EDIT_REF is not an edit's dumps (no images)"
	);
	let c = Config::load(&m).unwrap();
	Some((m, d, c))
}

fn check(what: &str, got: &[f32], want: &[f32]) -> f64 {
	let (r, mx) = rel_err(got, want);
	println!("{what:<28} rel {r:.2e}  max abs {mx:.2e}");
	r
}

/// The condition images' files and the run's resolution.
fn images(d: &Dumps) -> (Vec<PathBuf>, usize) {
	let files = d
		.manifest
		.get("images")
		.map_or(&[][..], Json::arr)
		.iter()
		.map(|i| PathBuf::from(i.get("file").and_then(Json::str).unwrap()))
		.collect();
	(files, d.num("resolution").unwrap() as usize)
}

fn target(d: &Dumps) -> (usize, usize) {
	let t = d.manifest.get("target").unwrap().arr();
	let (w, h) = (t[0].num().unwrap() as usize, t[1].num().unwrap() as usize);
	(h / 16, w / 16)
}

/// Channel-major `[c][h][w]` of a pixel-major map.
fn chw(x: &Map) -> Vec<f32> {
	let mut o = vec![0f32; x.d.len()];
	for p in 0..x.h * x.w {
		for c in 0..x.c {
			o[c * x.h * x.w + p] = x.d[p * x.c + c];
		}
	}
	o
}

#[test]
fn preprocess() {
	let Some((_, d, _)) = env() else { return };
	let (files, res) = images(&d);
	for (i, f) in files.iter().enumerate() {
		let c = Cond::load(f, res).unwrap();
		let want = d.get(&format!("ref{i}_rgba")).unwrap();
		assert_eq!(d.shape(&format!("ref{i}_rgba")).unwrap(), [c.h, c.w, 4]);
		let diff = c
			.rgba
			.iter()
			.zip(&want)
			.filter(|(a, b)| **a as f32 != **b)
			.count();
		println!(
			"image {i}: {}x{}, {diff} of {} bytes differ from Pillow",
			c.w,
			c.h,
			want.len()
		);
		assert_eq!(diff, 0, "Lanczos RGBA resize differs from Pillow");
		let px = c.vae_pixels();
		assert!(
			check(
				"vae pixels",
				&chw(&px),
				&d.get(&format!("ref{i}_vae_px")).unwrap()
			) < 1e-6
		);
	}
	assert_eq!(
		imagegen::cond::target(
			&files
				.iter()
				.map(|f| Cond::load(f, res).unwrap())
				.collect::<Vec<_>>(),
			res
		)
		.map(|(w, h)| (h / 16, w / 16)),
		Some(target(&d))
	);
}

#[test]
fn vision_tower() {
	let Some((m, d, c)) = env() else { return };
	let (files, res) = images(&d);
	let conds: Vec<Cond> = files.iter().map(|f| Cond::load(f, res).unwrap()).collect();
	let te = TextEncoder::open(&m, &c.te).unwrap();
	let vm = npubuild::vis::model_deepstack(&te.ck).unwrap();
	assert_eq!(vm.deepstack.len(), 3);
	// the processor's pixels: [patch][3][2 (temporal)][16][16], frame 0
	let pv = d.get("vis_pixel_values").unwrap();
	let mut worst = 0f64;
	let mut px_all = Vec::new();
	for cd in &conds {
		let grid = vision::smart_resize(cd.w, cd.h, 16, 2, 64, 16384).unwrap();
		let p = vision::patchify(&cd.te_image(), &grid, 16, 2, vm.cfg.mean, vm.cfg.std);
		px_all.extend(p);
	}
	let plen = 3 * 16 * 16;
	let want_px: Vec<f32> = (0..px_all.len() / plen)
		.flat_map(|p| {
			let pv = &pv;
			(0..3).flat_map(move |ch| (0..256).map(move |k| pv[p * 2 * plen + (ch * 2) * 256 + k]))
		})
		.collect();
	worst = worst.max(check("vision pixels", &px_all, &want_px));
	let rows = imagegen::cond::vision_rows(&m, &te.ck, &conds, None).unwrap();
	worst = worst.max(check(
		"vision embeds",
		&rows.embeds,
		&d.get("vis_embeds").unwrap(),
	));
	for (j, f) in rows.deep.iter().enumerate() {
		worst = worst.max(check(
			&format!("deepstack {j}"),
			f,
			&d.get(&format!("vis_deep{j}")).unwrap(),
		));
	}
	assert!(worst < TOL, "vision tower rel err {worst:.2e}");
}

#[test]
fn text_encoder() {
	let Some((m, d, c)) = env() else { return };
	let te = TextEncoder::open(&m, &c.te).unwrap();
	let g = d.get("vis_grid_thw").unwrap();
	let grids: Vec<(usize, usize)> = g
		.chunks(3)
		.map(|t| (t[1] as usize / 2, t[2] as usize / 2))
		.collect();
	let vis = VisionRows {
		embeds: d.get("vis_embeds").unwrap(),
		deep: (0..3)
			.map(|j| d.get(&format!("vis_deep{j}")).unwrap())
			.collect(),
		grids,
	};
	let prompt = d.str("prompt").unwrap().to_string();
	let tokens: Vec<usize> = vis.grids.iter().map(|&(r, c)| r * c).collect();
	let (ids, drop) = te.tokenize_images(&prompt, &tokens).unwrap();
	let want: Vec<u32> = d
		.get("te_input_ids")
		.unwrap()
		.iter()
		.map(|&v| v as u32)
		.collect();
	assert_eq!(ids, want, "token ids");
	assert_eq!(drop, d.num("drop_idx").unwrap() as usize);
	let pos = te.positions(&ids, &vis.grids).unwrap();
	let wp = d.get("te_position_ids").unwrap();
	let s = ids.len();
	for (p, id) in pos.iter().enumerate() {
		for a in 0..3 {
			assert_eq!(id[a] as f32, wp[a * s + p], "MRoPE position {p} axis {a}");
		}
	}
	let mut worst = 0f64;
	let enc = te
		.encode_images(&prompt, Some(&vis), |l, x| {
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
	let mask: Vec<bool> = d
		.get("te_image_pad_mask")
		.unwrap()
		.iter()
		.map(|&v| v != 0.0)
		.collect();
	assert_eq!(enc.image_pad, mask, "image slots");
	assert!(worst < TOL, "text encoder rel err {worst:.2e}");
}

#[test]
fn vae_encoder() {
	let Some((m, d, c)) = env() else { return };
	let vae = Vae::open(&m, &c.vae).unwrap();
	let mut worst = 0f64;
	for (i, im) in d.manifest.get("images").unwrap().arr().iter().enumerate() {
		let (w, h) = (
			im.get("width").and_then(Json::num).unwrap() as usize,
			im.get("height").and_then(Json::num).unwrap() as usize,
		);
		let src = d.get(&format!("ref{i}_vae_px")).unwrap();
		let mut px = Map {
			h,
			w,
			c: 4,
			d: vec![0f32; src.len()],
		};
		for p in 0..h * w {
			for ch in 0..4 {
				px.d[p * 4 + ch] = src[ch * h * w + p];
			}
		}
		let t0 = std::time::Instant::now();
		let lat = vae
			.encode(&px, |name, x| {
				let key = format!("ref{i}_venc_{name}");
				if d.shape(&key).is_some() {
					worst = worst.max(check(&key, &chw(x), &d.get(&key).unwrap()));
				}
			})
			.unwrap();
		println!("vae encode {w}x{h}: {:.1} s", t0.elapsed().as_secs_f64());
		worst = worst.max(check(
			"latents",
			&lat,
			&d.get(&format!("ref{i}_latents")).unwrap(),
		));
	}
	assert!(worst < TOL, "vae encoder rel err {worst:.2e}");
}

fn edit_layout(d: &Dumps) -> Layout {
	let mask: Vec<bool> = d
		.get("te_image_pad_mask")
		.unwrap()
		.iter()
		.map(|&v| v != 0.0)
		.collect();
	let refs: Vec<(usize, usize)> = d
		.manifest
		.get("images")
		.unwrap()
		.arr()
		.iter()
		.map(|i| {
			(
				i.get("height").and_then(Json::num).unwrap() as usize / 16,
				i.get("width").and_then(Json::num).unwrap() as usize / 16,
			)
		})
		.collect();
	layout(&mask, &refs, target(d)).unwrap()
}

#[test]
fn transformer() {
	let Some((m, d, c)) = env() else { return };
	let (h, w) = target(&d);
	let lay = edit_layout(&d);
	let rope = Rope::new(&c.dit, &lay.ids);
	let mut worst = 0f64;
	worst = worst.max(check(
		"rope cos",
		&rope.cos,
		&d.get("dit_rope_cos").unwrap(),
	));
	worst = worst.max(check(
		"rope sin",
		&rope.sin,
		&d.get("dit_rope_sin").unwrap(),
	));
	let dit = Dit::open(&m, &c.dit).unwrap();
	let nref = d.manifest.get("images").unwrap().arr().len();
	let refs: Vec<Vec<f32>> = (0..nref)
		.map(|i| d.get(&format!("ref{i}_latents")).unwrap())
		.collect();
	let mut txt = Some(
		dit.prefix_rows(&d.get("te_out").unwrap(), &lay, &refs)
			.unwrap(),
	);
	let mut pre = Prefix::new(&lay);
	println!(
		"prefix {} rows, {} segments, target {}",
		lay.prefix,
		lay.segments.len(),
		h * w
	);
	let s = Schedule::new(&c.sched, c.sample_sigmas.as_deref(), 0, h * w);
	let steps: Vec<usize> = match std::env::var("QWEN_TEST_IMAGE_STEPS") {
		Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
		Err(_) => (0..s.steps()).collect(),
	};
	for i in 0..s.steps() {
		if !steps.contains(&i) && i != 0 {
			continue;
		}
		let (md, _, _) = dit.modulation(s.t(i, c.sched.train_steps)).unwrap();
		let x = if i == 0 {
			d.get("noise").unwrap()
		} else {
			d.get(&format!("latents{i}")).unwrap()
		};
		let t0 = std::time::Instant::now();
		let v = dit
			.forward(&x, &mut txt, &mut pre, &rope, &md, |l, p, t| {
				if i == 0 {
					let full = [p.unwrap(), t].concat();
					worst = worst.max(check(
						&format!("block {l}"),
						&full,
						&d.get(&format!("dit_block{l}")).unwrap(),
					));
				}
			})
			.unwrap();
		println!("step {i}: {:.1} s", t0.elapsed().as_secs_f64());
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
	let (h, w) = target(&d);
	let vae = Vae::open(&m, &c.vae).unwrap();
	let img = vae
		.decode(&d.get("latents_final").unwrap(), h, w, |_, _| {})
		.unwrap();
	let r = check("vae out", &chw(&img), &d.get("vae_out").unwrap());
	assert!(r < TOL, "vae rel err {r:.2e}");
}
