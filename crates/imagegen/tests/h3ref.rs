//! MiniMax-H3-Image per-layer gates: every layer against
//! `scripts/h3refdump.py`'s f32 dumps, each layer fed the reference's own
//! input so an error is that layer's. Needs `QWEN_TEST_H3_MODEL` (the
//! checkpoint directory) and `QWEN_TEST_H3_REF` (a dump directory); skipped
//! without them. `QWEN_TEST_H3_BLOCKS=0,24,49` limits the transformer blocks
//! checked (default: every block of step 0 the dump holds),
//! `QWEN_TEST_H3_TILE_BLOCKS` the VAE decoder's. An edit dump (`--image`)
//! adds the references' gates: resize, vision tower, VAE encoder, posterior
//! sample and noise. Run with `--release --test-threads 1 --nocapture`.

use imagegen::h3::config::Config;
use imagegen::h3::dit::{Dit, Rope};
use imagegen::h3::layout;
use imagegen::h3::pipeline::{reference_pixels, sample_latents, Reference};
use imagegen::h3::vae::{to_chw, Map, Vae};
use imagegen::ops::WFmt;
use imagegen::refdump::{rel_err, Dumps};
use imagegen::rng::TorchGen;
use imagegen::te::TextEncoder;
use std::path::PathBuf;

const TOL: f64 = 1e-3;

fn env() -> Option<(PathBuf, Dumps, Config)> {
	let m = PathBuf::from(std::env::var_os("QWEN_TEST_H3_MODEL")?);
	let d = Dumps::open(&PathBuf::from(std::env::var_os("QWEN_TEST_H3_REF")?)).unwrap();
	let c = Config::load(&m).unwrap();
	Some((m, d, c))
}

fn check(what: &str, got: &[f32], want: &[f32]) -> f64 {
	let (r, mx) = rel_err(got, want);
	println!("{what:<28} rel {r:.2e}  max abs {mx:.2e}");
	r
}

fn list(var: &str) -> Option<Vec<usize>> {
	std::env::var(var)
		.ok()
		.map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
}

fn ids(d: &Dumps, name: &str) -> Vec<u32> {
	d.get(name).unwrap().iter().map(|&v| v as u32).collect()
}

fn tags(d: &Dumps, name: &str) -> Vec<u8> {
	d.get(name).unwrap().iter().map(|&v| v as u8).collect()
}

/// The references of an edit dump, prepared as the pipeline prepares them.
fn references(d: &Dumps) -> Vec<Reference> {
	let Some(list) = d.manifest.get("images") else {
		return Vec::new();
	};
	list.arr()
		.iter()
		.map(|i| {
			let f = i.get("file").and_then(|f| f.str()).unwrap();
			Reference::new(&imagegen::ImageSrc::Path(PathBuf::from(f))).unwrap()
		})
		.collect()
}

/// The vision tower's rows of the dump's references (`None` without).
fn vision(m: &std::path::Path, d: &Dumps, te: &TextEncoder) -> Option<imagegen::te::VisionRows> {
	let refs = references(d);
	if refs.is_empty() {
		return None;
	}
	let conds: Vec<_> = refs.into_iter().map(|r| r.vision).collect();
	Some(imagegen::cond::vision_rows(m, &te.ck, &conds, None).unwrap())
}

/// The dump's layout, as the Rust side builds it.
fn layout_of(d: &Dumps, c: &Config) -> layout::Layout {
	let h = d.num("height").unwrap() as usize / 16;
	let w = d.num("width").unwrap() as usize / 16;
	let mut refs = Vec::new();
	let mut i = 0;
	while let Some(s) = d.shape(&format!("ref{i}_latents")) {
		refs.push((s[1], s[2]));
		i += 1;
	}
	layout::build(&c.dit, &tags(d, "te_token_tags"), &refs, (h, w))
}

#[test]
fn text_encoder_layers() {
	let Some((m, d, c)) = env() else { return };
	let te = TextEncoder::open(&m, &c.te).unwrap();
	let prompt = d.str("prompt").unwrap().to_string();
	let mut worst = 0f64;
	let vis = vision(&m, &d, &te);
	if let Some(v) = &vis {
		worst = worst.max(check(
			"vision embeds",
			&v.embeds,
			&d.get("vis_embeds").unwrap(),
		));
		for (j, f) in v.deep.iter().enumerate() {
			worst = worst.max(check(
				&format!("vision deepstack {j}"),
				f,
				&d.get(&format!("vis_deep{j}")).unwrap(),
			));
		}
	}
	let pads: Vec<usize> = vis
		.as_ref()
		.map_or_else(Vec::new, |v| v.grids.iter().map(|&(r, c)| r * c).collect());
	let (got, tg) = imagegen::h3::te::presentation(&te.tok, &prompt, &pads).unwrap();
	assert_eq!(got, ids(&d, "te_input_ids"), "token ids");
	assert_eq!(tg, tags(&d, "te_token_tags"), "token tags");
	let out = te
		.encode_ids(&got, vis.as_ref(), &mut |l, x| {
			if d.shape(&format!("te_layer{l}")).is_some() {
				worst = worst.max(check(
					&format!("te layer {l}"),
					x,
					&d.get(&format!("te_layer{l}")).unwrap(),
				));
			}
		})
		.unwrap();
	worst = worst.max(check("te out", &out, &d.get("te_out").unwrap()));
	assert!(worst < TOL, "text encoder rel. error {worst:.2e}");
}

#[test]
fn layout_and_schedule() {
	let Some((_, d, c)) = env() else { return };
	let l = layout_of(&d, &c);
	let pos: Vec<f32> = l.pos.iter().flatten().copied().collect();
	assert!(check("position ids", &pos, &d.get("dit_position_ids").unwrap()) == 0.0);
	let tg: Vec<f32> = l.tags.iter().map(|&t| t as f32).collect();
	assert_eq!(tg, d.get("dit_token_tags").unwrap(), "row tags");
	let (ts, sg) = c.schedule();
	assert_eq!(ts, d.get("sched_timesteps").unwrap(), "timesteps");
	assert_eq!(sg, d.get("sched_sigmas").unwrap(), "sigmas");
	// the generator's draws in the pipeline's order: the references'
	// posterior samples, the target's noise (randn((1, 24, 1, h, w)),
	// channel-major), the references' noise
	let (h, w) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let mut g = TorchGen::new(d.num("seed").unwrap() as u64);
	let mut lat = Vec::new();
	let mut i = 0;
	while let Some(s) = d.shape(&format!("ref{i}_moments")) {
		let m = d.get(&format!("ref{i}_moments")).unwrap();
		let (l, lh, lw) = sample_latents(&c, &m, s[1], s[2], &mut g);
		assert!(
			check(
				&format!("ref{i} latents"),
				&l,
				&d.get(&format!("ref{i}_latents")).unwrap()
			) < 1e-6
		);
		lat.push((l, lh, lw));
		i += 1;
	}
	let n = g.randn(c.dit.in_ch * h * w);
	let rows = layout::patchify(&n, c.dit.in_ch, h, w, c.dit.patch);
	assert!(check("noise", &rows, &d.get("noise").unwrap()) < 1e-6);
	if !lat.is_empty() {
		let mut r = Vec::new();
		for (l, lh, lw) in &lat {
			let e = g.randn(l.len());
			let t = imagegen::h3::config::REF_T;
			let x: Vec<f32> = l
				.iter()
				.zip(&e)
				.map(|(&a, &b)| t * a + (1.0 - t) * b)
				.collect();
			r.extend(layout::patchify(&x, c.dit.in_ch, *lh, *lw, c.dit.patch));
		}
		assert!(check("reference rows", &r, &d.get("dit_ref_rows").unwrap()) < 1e-6);
	}
}

#[test]
fn vae_encoder() {
	let Some((m, d, c)) = env() else { return };
	let refs = references(&d);
	if refs.is_empty() {
		return;
	}
	let vae = Vae::open(&m, &c.vae).unwrap();
	let mut worst = 0f64;
	for (i, r) in refs.iter().enumerate() {
		// the resized pixels, bytes [3][h][w]
		let (w, h, px) = (r.vae.0, r.vae.1, &r.vae.2);
		let want = d.get(&format!("ref{i}_rgb")).unwrap();
		let s = d.shape(&format!("ref{i}_rgb")).unwrap();
		let mut got = vec![0f32; want.len()];
		for ch in 0..3 {
			for y in 0..s[1] {
				for x in 0..s[2] {
					got[(ch * s[1] + y) * s[2] + x] = px[(y * w + x) * 3 + ch] as f32;
				}
			}
		}
		assert_eq!((h / 16 * 16, w / 16 * 16), (s[1], s[2]), "reference size");
		worst = worst.max(check(&format!("ref{i} pixels"), &got, &want));
		let (chw, ph, pw) = reference_pixels(&c, r);
		let (mo, _, _) = vae
			.encode_moments(&chw, ph, pw, |name, map| {
				if let Ok(want) = d.get(&format!("ref{i}_venc_{name}")) {
					worst = worst.max(check(&format!("ref{i} venc {name}"), &to_chw(map), &want));
				}
			})
			.unwrap();
		worst = worst.max(check(
			&format!("ref{i} moments"),
			&mo,
			&d.get(&format!("ref{i}_moments")).unwrap(),
		));
	}
	assert!(worst < TOL, "rel. error {worst:.2e}");
}

#[test]
fn transformer_text_and_embeddings() {
	let Some((m, d, c)) = env() else { return };
	let dit = Dit::open(&m, &c.dit).unwrap();
	let te = d.get("te_out").unwrap();
	let mut worst = 0f64;
	let txt = dit
		.text_rows(&te, |j, x| {
			let n = if j == usize::MAX {
				"dit_ctx".to_string()
			} else {
				format!("dit_refiner{j}")
			};
			worst = worst.max(check(&n, x, &d.get(&n).unwrap()));
		})
		.unwrap();
	worst = worst.max(check("dit_txt", &txt, &d.get("dit_txt").unwrap()));

	let l = layout_of(&d, &c);
	let (ts, _) = c.schedule();
	for (i, &t) in ts.iter().enumerate() {
		let (u, _) = l.row_timesteps(t);
		let e = dit.temb(&u).unwrap();
		worst = worst.max(check(
			&format!("temb step {i}"),
			&e,
			&d.get(&format!("dit_temb{i}")).unwrap(),
		));
	}
	let rope = Rope::new(&c.dit, &l.pos);
	let (cos, sin) = (
		d.get("dit_rope_cos").unwrap(),
		d.get("dit_rope_sin").unwrap(),
	);
	// the reference holds both halves of every head's rotary channels
	let full = |v: &[f32]| -> Vec<f32> {
		v.chunks(rope.half)
			.flat_map(|r| r.iter().chain(r.iter()).copied())
			.collect()
	};
	worst = worst.max(check("rope cos", &full(&rope.cos), &cos));
	worst = worst.max(check("rope sin", &full(&rope.sin), &sin));
	assert!(worst < TOL, "rel. error {worst:.2e}");
}

#[test]
fn transformer_blocks() {
	let Some((m, d, c)) = env() else { return };
	let dit = Dit::open(&m, &c.dit).unwrap();
	let l = layout_of(&d, &c);
	let rope = Rope::new(&c.dit, &l.pos);
	let (ts, _) = c.schedule();
	let mut worst = 0f64;
	for step in [0, ts.len() - 1] {
		let (u, tidx) = l.row_timesteps(ts[step]);
		let temb = dit.temb(&u).unwrap();
		let idx = Dit::entries(&l, &tidx);
		let blocks = list("QWEN_TEST_H3_BLOCKS").unwrap_or_else(|| (0..c.dit.layers).collect());
		for &b in &blocks {
			let want = format!("dit_s{step}_block{b}");
			let input = if b == 0 {
				if step != 0 {
					continue; // block 0's input is dumped for step 0 only
				}
				"dit_x_in".to_string()
			} else {
				format!("dit_s{step}_block{}", b - 1)
			};
			if d.shape(&want).is_none() || d.shape(&input).is_none() {
				continue;
			}
			let table = dit.adaln(b, &temb).unwrap();
			if let Some(s) = d.shape(&format!("dit_s{step}_adaln{b}")) {
				// [entries][6][hidden], the reference's own row order
				assert_eq!(s.iter().product::<usize>(), table.len());
				worst = worst.max(check(
					&format!("adaln s{step} b{b}"),
					&table,
					&d.get(&format!("dit_s{step}_adaln{b}")).unwrap(),
				));
			}
			let w = dit
				.load_block(&format!("transformer_blocks.{b}."), WFmt::F32)
				.unwrap();
			let mut x = d.get(&input).unwrap();
			let t0 = std::time::Instant::now();
			dit.block(b, &w, &mut x, &table, &idx, &rope);
			let r = check(
				&format!("block s{step} b{b} ({:.1} s)", t0.elapsed().as_secs_f64()),
				&x,
				&d.get(&want).unwrap(),
			);
			worst = worst.max(r);
		}
	}
	assert!(worst < TOL, "rel. error {worst:.2e}");
}

#[test]
fn transformer_output_heads() {
	let Some((m, d, c)) = env() else { return };
	let dit = Dit::open(&m, &c.dit).unwrap();
	let l = layout_of(&d, &c);
	let (ts, _) = c.schedule();
	let last = c.dit.layers - 1;
	let dd = c.dit.hidden;
	let mut worst = 0f64;
	for step in [0, ts.len() - 1] {
		let Ok(x) = d.get(&format!("dit_s{step}_block{last}")) else {
			continue;
		};
		let (u, tidx) = l.row_timesteps(ts[step]);
		let temb = dit.temb(&u).unwrap();
		let v = dit
			.out(&x[l.image0() * dd..], &temb, &tidx[l.image0()..], step)
			.unwrap();
		let n = l.target * c.dit.patch_dim();
		worst = worst.max(check(
			&format!("velocity step {step}"),
			&v[v.len() - n..],
			&d.get(&format!("dit_v{step}")).unwrap(),
		));
	}
	assert!(worst < TOL, "rel. error {worst:.2e}");
}

#[test]
fn vae_decoder_tile0() {
	let Some((m, d, c)) = env() else { return };
	let vae = Vae::open(&m, &c.vae).unwrap();
	let z = d.get("vae_z").unwrap();
	let (lh, lw) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let (_, th, _) = imagegen::h3::vae::split_tiles(lh * 16, 256, 64, 16);
	let (_, tw, _) = imagegen::h3::vae::split_tiles(lw * 16, 256, 64, 16);
	let (th, tw) = (th / 16, tw / 16);
	let mut t = Map {
		h: th,
		w: tw,
		c: c.vae.latent,
		d: vec![0f32; th * tw * c.vae.latent],
	};
	for y in 0..th {
		for x in 0..tw {
			for ch in 0..c.vae.latent {
				t.d[(y * tw + x) * c.vae.latent + ch] = z[(ch * lh + y) * lw + x];
			}
		}
	}
	let mut worst = 0f64;
	let only = list("QWEN_TEST_H3_TILE_BLOCKS");
	vae.decode_tile(&t, &mut |name, x| {
		let want = format!("vae_tile0_{name}");
		let keep = match (&only, name.strip_prefix("block")) {
			(Some(o), Some(b)) => o.contains(&b.parse().unwrap()),
			_ => true,
		};
		if keep {
			if let Ok(w) = d.get(&want) {
				if w.len() >= x.len() {
					worst = worst.max(check(&want, x, &w[..x.len()]));
				}
			}
		}
	})
	.unwrap();
	assert!(worst < TOL, "rel. error {worst:.2e}");
}

/// The quantized backend against the f32 dumps (`QWEN_TEST_H3_QUANT=1`;
/// format `QWEN_IMAGE_WEIGHTS`, default Q6_K): each transformer block fed
/// the reference's input (`QWEN_TEST_H3_BLOCKS`, default 0, 24, 49), the
/// whole text encoder, and step 0's velocity through every quantized block
/// (the resident model, through the weight cache when one is set).
#[test]
fn quantized() {
	let Some((m, d, c)) = env() else { return };
	if std::env::var("QWEN_TEST_H3_QUANT").map_or(true, |v| v != "1") {
		return;
	}
	let fmt = WFmt::from_env();
	let cache = imagegen::wcache::dir();
	let tag = format!("{fmt:?}").to_ascii_lowercase();
	let dit = Dit::open(&m, &c.dit).unwrap();
	let l = layout_of(&d, &c);
	let rope = Rope::new(&c.dit, &l.pos);
	let (ts, _) = c.schedule();
	let (u, tidx) = l.row_timesteps(ts[0]);
	let temb = dit.temb(&u).unwrap();
	let idx = Dit::entries(&l, &tidx);
	let mut worst_block = 0f64;
	for &b in &list("QWEN_TEST_H3_BLOCKS").unwrap_or(vec![0, 24, 49]) {
		let input = if b == 0 {
			"dit_x_in".to_string()
		} else {
			format!("dit_s0_block{}", b - 1)
		};
		let (Ok(mut x), Ok(want)) = (d.get(&input), d.get(&format!("dit_s0_block{b}"))) else {
			continue;
		};
		let w = dit
			.load_block(&format!("transformer_blocks.{b}."), fmt)
			.unwrap();
		let table = dit.adaln(b, &temb).unwrap();
		dit.block(b, &w, &mut x, &table, &idx, &rope);
		worst_block = worst_block.max(check(&format!("{tag} block s0 b{b}"), &x, &want));
	}
	drop(dit);

	let te = TextEncoder::load_cached(
		&m,
		&c.te,
		fmt,
		cache
			.as_ref()
			.map(|p| p.join(format!("h3-te-{tag}.bin")))
			.as_deref(),
		|_| {},
	)
	.unwrap();
	let vis = vision(&m, &d, &te);
	let pads: Vec<usize> = vis
		.as_ref()
		.map_or_else(Vec::new, |v| v.grids.iter().map(|&(r, c)| r * c).collect());
	let prompt = d.str("prompt").unwrap().to_string();
	let (ids, _) = imagegen::h3::te::presentation(&te.tok, &prompt, &pads).unwrap();
	let out = te.encode_ids(&ids, vis.as_ref(), &mut |_, _| {}).unwrap();
	let r_te = check(&format!("{tag} te out"), &out, &d.get("te_out").unwrap());
	drop((te, out));

	let dit = Dit::load(
		&m,
		&c.dit,
		fmt,
		&ts,
		cache
			.as_ref()
			.map(|p| p.join(format!("h3-dit-{tag}.bin")))
			.as_deref(),
		|_| {},
	)
	.unwrap();
	let txt = dit.text_rows(&d.get("te_out").unwrap(), |_, _| {}).unwrap();
	let r_txt = check(
		&format!("{tag} text rows"),
		&txt,
		&d.get("dit_txt").unwrap(),
	);
	let mut img = d.get("dit_ref_rows").unwrap_or_default();
	img.extend(d.get("noise").unwrap());
	let v = dit
		.forward(&l, &rope, &txt, &img, &u, &tidx, 0, |_, _| {})
		.unwrap();
	let n = l.target * c.dit.patch_dim();
	let r_v = check(
		&format!("{tag} velocity step 0"),
		&v[v.len() - n..],
		&d.get("dit_v0").unwrap(),
	);
	// Q6_K measures blocks 1.8e-3..2.3e-2, text encoder 7.6e-3, text rows
	// 2.3e-2, velocity 5.4e-2 (s512); the gates are about twice that
	assert!(
		worst_block < 5e-2 && r_te < 2e-2 && r_txt < 5e-2 && r_v < 1e-1,
		"quantized errors too large"
	);
}

/// PSNR of two 8-bit images.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
	let mse = a
		.iter()
		.zip(b)
		.map(|(&x, &y)| (x as f64 - y as f64).powi(2))
		.sum::<f64>()
		/ a.len() as f64;
	10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

/// The NPU backend (`QWEN_TEST_H3_NPU` = its `image.npu`) against the f32
/// dumps, the transformer fed the reference's text rows: the target rows
/// after blocks `QWEN_TEST_H3_BLOCKS` (default 0, 24, 49) of step 0, every
/// step's velocity fed the reference's latents, the free-running
/// trajectory's final latents, and the NPU VAE decoder on the reference's
/// latents and on the free-running ones (PSNR against the reference's
/// image).
#[test]
fn npu() {
	use imagegen::h3::npudit::{NpuCfg, NpuDit};
	use imagegen::h3::npuvae::NpuVae;
	let Some((m, d, c)) = env() else { return };
	let Some(np) = std::env::var_os("QWEN_TEST_H3_NPU") else {
		return;
	};
	let nc = NpuCfg::parse(std::path::Path::new(&np)).unwrap();
	let (ts, sg) = c.schedule();
	let dit = Dit::load_small(&m, &c.dit, WFmt::Q6K, &ts).unwrap();
	let mut n = NpuDit::load(&nc, &c.dit, |_| {}).unwrap();
	let l = layout_of(&d, &c);
	let rope = Rope::new(&c.dit, &l.pos);
	n.begin(&l, &rope).unwrap();
	let txt = d.get("dit_txt").unwrap();
	let refs = d.get("dit_ref_rows").unwrap_or_default();
	let x_of = |img: &[f32]| {
		let mut all = refs.clone();
		all.extend_from_slice(img);
		let mut x = txt.clone();
		x.extend(dit.image_rows(&all).unwrap());
		x
	};
	let noise = d.get("noise").unwrap();
	let hd = c.dit.hidden;
	let mut worst_block = 0f64;
	for &b in &list("QWEN_TEST_H3_BLOCKS").unwrap_or(vec![0, 24, 49]) {
		let Ok(want) = d.get(&format!("dit_s0_block{b}")) else {
			continue;
		};
		n.limit = Some(b + 1);
		let t = std::time::Instant::now();
		let mut got = n.forward(&x_of(&noise), ts[0]).unwrap();
		let s = n.scale(b);
		got.iter_mut().for_each(|v| *v *= s);
		worst_block = worst_block.max(check(
			&format!("npu block s0 b{b} ({:.1} s)", t.elapsed().as_secs_f64()),
			&got,
			&want[l.target0() * hd..],
		));
	}
	n.limit = None;
	if std::env::var_os("QWEN_TEST_H3_NPU_BLOCKS_ONLY").is_some() {
		return;
	}
	let mut worst_v = 0f64;
	for (i, &t) in ts.iter().enumerate() {
		let lat = if i == 0 {
			noise.clone()
		} else {
			d.get(&format!("latents{i}")).unwrap()
		};
		let t0 = std::time::Instant::now();
		let v = dit
			.out_at(&n.forward(&x_of(&lat), t).unwrap(), t, i)
			.unwrap();
		worst_v = worst_v.max(check(
			&format!(
				"npu velocity step {i} ({:.1} s)",
				t0.elapsed().as_secs_f64()
			),
			&v,
			&d.get(&format!("dit_v{i}")).unwrap(),
		));
	}
	let mut lat = noise.clone();
	for (i, &t) in ts.iter().enumerate() {
		let v = dit
			.out_at(&n.forward(&x_of(&lat), t).unwrap(), t, i)
			.unwrap();
		let dt = sg[i] - sg[i + 1];
		lat.iter_mut().zip(&v).for_each(|(x, &vv)| *x += dt * vv);
	}
	n.release_buffers();
	drop(n);
	let (lh, lw) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let fin = layout::unpatchify(&lat, c.dit.in_ch, lh, lw, c.dit.patch);
	let r_lat = check(
		"npu free-running latents",
		&fin,
		&d.get("latents_final").unwrap(),
	);
	let vae = Vae::open(&m, &c.vae).unwrap();
	let mut nv = NpuVae::load(nc.vae.as_ref().unwrap(), &m).unwrap();
	let t = std::time::Instant::now();
	let (img, oh, ow) = nv
		.decode(&vae, &d.get("vae_z").unwrap(), lh, lw)
		.unwrap()
		.unwrap();
	let want = d.get("vae_out").unwrap();
	let r_vae = check(
		&format!("npu vae decoder ({:.1} s)", t.elapsed().as_secs_f64()),
		&img,
		&want,
	);
	let ref8 = imagegen::h3::vae::to_rgb8(&want, oh, ow);
	let p_vae = psnr(&imagegen::h3::vae::to_rgb8(&img, oh, ow), &ref8);
	let p = lh * lw;
	let mut z = fin.clone();
	for (ch, (&s, &mn)) in c.vae.std.iter().zip(&c.vae.mean).enumerate() {
		z[ch * p..(ch + 1) * p]
			.iter_mut()
			.for_each(|v| *v = *v * s + mn);
	}
	let (img, _, _) = nv.decode(&vae, &z, lh, lw).unwrap().unwrap();
	let p_free = psnr(&imagegen::h3::vae::to_rgb8(&img, oh, ow), &ref8);
	println!("npu vae decoder PSNR {p_vae:.1} dB; free-running image PSNR {p_free:.1} dB");
	println!("npu worst: block {worst_block:.2e}, velocity {worst_v:.2e}, latents {r_lat:.2e}, vae {r_vae:.2e}");
	// Q6_K (f32 attention) measures velocity 3.5-5.3e-2 per step on s512;
	// the free-running image moves several dB with any change of rounding
	// (reported, not gated)
	assert!(worst_v < 1.1e-1 && p_vae > 40.0, "npu errors too large");
}

/// The NPU VAE decoder alone (`QWEN_TEST_H3_NPU_VAE` = its graphs
/// directory) on the reference's latents, and its time on a 4 MP canvas.
#[test]
fn npu_vae() {
	use imagegen::h3::npuvae::NpuVae;
	let Some((m, d, c)) = env() else { return };
	let Some(g) = std::env::var_os("QWEN_TEST_H3_NPU_VAE") else {
		return;
	};
	let vae = Vae::open(&m, &c.vae).unwrap();
	let t = std::time::Instant::now();
	let mut nv = NpuVae::load(std::path::Path::new(&g), &m).unwrap();
	eprintln!("npu vae loaded in {:.1} s", t.elapsed().as_secs_f64());
	let (lh, lw) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let t = std::time::Instant::now();
	let (img, oh, ow) = nv
		.decode(&vae, &d.get("vae_z").unwrap(), lh, lw)
		.unwrap()
		.unwrap();
	let want = d.get("vae_out").unwrap();
	let r = check(
		&format!("npu vae decoder ({:.1} s)", t.elapsed().as_secs_f64()),
		&img,
		&want,
	);
	let p = psnr(
		&imagegen::h3::vae::to_rgb8(&img, oh, ow),
		&imagegen::h3::vae::to_rgb8(&want, oh, ow),
	);
	println!("npu vae decoder PSNR {p:.1} dB");
	// 4 MP: 128 x 128 latents (121 tiles)
	let z: Vec<f32> = (0..c.vae.latent * 128 * 128)
		.map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0)
		.collect();
	(nv.t_wait, nv.t_cores) = (0.0, 0.0);
	let t = std::time::Instant::now();
	nv.decode(&vae, &z, 128, 128).unwrap().unwrap();
	println!(
		"npu vae decoder 2048x2048: {:.1} s (NPU wait {:.1} s, cores {:.1} s)",
		t.elapsed().as_secs_f64(),
		nv.t_wait,
		nv.t_cores
	);
	if std::env::var_os("QWEN_TEST_H3_CPU_VAE").is_some() {
		let t = std::time::Instant::now();
		vae.decode(&z, 128, 128, |_, _| {}).unwrap();
		println!(
			"cpu vae decoder 2048x2048: {:.1} s",
			t.elapsed().as_secs_f64()
		);
	}
	assert!(r < 2e-2 && p > 40.0, "npu vae decoder error too large");
}

/// The quantized CPU backend's error through the chained blocks of step 0
/// (`QWEN_TEST_H3_CHAIN=1`): the baseline of the NPU backend's block gate.
#[test]
fn quantized_chain() {
	let Some((m, d, c)) = env() else { return };
	if std::env::var("QWEN_TEST_H3_CHAIN").map_or(true, |v| v != "1") {
		return;
	}
	let fmt = WFmt::from_env();
	let cache = imagegen::wcache::dir();
	let tag = format!("{fmt:?}").to_ascii_lowercase();
	let (ts, _) = c.schedule();
	let dit = Dit::load(
		&m,
		&c.dit,
		fmt,
		&ts,
		cache
			.as_ref()
			.map(|p| p.join(format!("h3-dit-{tag}.bin")))
			.as_deref(),
		|_| {},
	)
	.unwrap();
	let l = layout_of(&d, &c);
	let rope = Rope::new(&c.dit, &l.pos);
	let (u, tidx) = l.row_timesteps(ts[0]);
	let txt = d.get("dit_txt").unwrap();
	let mut img = d.get("dit_ref_rows").unwrap_or_default();
	img.extend(d.get("noise").unwrap());
	let t0 = l.target0() * c.dit.hidden;
	let want = list("QWEN_TEST_H3_BLOCKS").unwrap_or(vec![12, 24, 30, 35, 40, 44, 49]);
	let v = dit
		.forward(&l, &rope, &txt, &img, &u, &tidx, 0, |b, x| {
			if want.contains(&b) {
				check(
					&format!("{tag} chained block s0 b{b}"),
					&x[t0..],
					&d.get(&format!("dit_s0_block{b}")).unwrap()[t0..],
				);
			}
		})
		.unwrap();
	let n = l.target * c.dit.patch_dim();
	check(
		&format!("{tag} velocity step 0"),
		&v[v.len() - n..],
		&d.get("dit_v0").unwrap(),
	);
	for i in 1..ts.len() {
		let mut img = d.get("dit_ref_rows").unwrap_or_default();
		img.extend(d.get(&format!("latents{i}")).unwrap());
		let (u, tidx) = l.row_timesteps(ts[i]);
		let v = dit
			.forward(&l, &rope, &txt, &img, &u, &tidx, i, |_, _| {})
			.unwrap();
		check(
			&format!("{tag} velocity step {i}"),
			&v[v.len() - n..],
			&d.get(&format!("dit_v{i}")).unwrap(),
		);
	}
	// free-running from the reference's noise, decoded on the cores
	let (_, sg) = c.schedule();
	let refs = d.get("dit_ref_rows").unwrap_or_default();
	let mut lat = d.get("noise").unwrap();
	for i in 0..ts.len() {
		let mut img = refs.clone();
		img.extend_from_slice(&lat);
		let (u, tidx) = l.row_timesteps(ts[i]);
		let v = dit
			.forward(&l, &rope, &txt, &img, &u, &tidx, i, |_, _| {})
			.unwrap();
		let dt = sg[i] - sg[i + 1];
		lat.iter_mut()
			.zip(&v[v.len() - n..])
			.for_each(|(x, &vv)| *x += dt * vv);
	}
	drop(dit);
	let (lh, lw) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let fin = layout::unpatchify(&lat, c.dit.in_ch, lh, lw, c.dit.patch);
	check(
		&format!("{tag} free-running latents"),
		&fin,
		&d.get("latents_final").unwrap(),
	);
	let p = lh * lw;
	let mut z = fin;
	for (ch, (&s, &mn)) in c.vae.std.iter().zip(&c.vae.mean).enumerate() {
		z[ch * p..(ch + 1) * p]
			.iter_mut()
			.for_each(|v| *v = *v * s + mn);
	}
	let vae = Vae::open(&m, &c.vae).unwrap();
	let (img, oh, ow) = vae.decode(&z, lh, lw, |_, _| {}).unwrap();
	let want = d.get("vae_out").unwrap();
	let p = psnr(
		&imagegen::h3::vae::to_rgb8(&img, oh, ow),
		&imagegen::h3::vae::to_rgb8(&want, oh, ow),
	);
	println!("{tag} free-running image PSNR {p:.1} dB");
}

/// The NPU text encoder (`QWEN_TEST_H3_NPU` = an `image.npu` with `te`)
/// against the f32 dumps: its output (`hidden_states[50]`, the vision
/// rows and deepstack of an edit dump included), the text rows the
/// refiner makes of it, step 0's velocity of the NPU transformer fed those
/// rows, and the free-running image, each beside the same from the
/// reference's text rows and from the cores' Q6_K text encoder (whose
/// time is printed too). Gated on the velocity's increase against the
/// Q6_K encoder's.
#[test]
fn npu_text_encoder() {
	use imagegen::h3::npudit::{NpuCfg, NpuDit};
	use imagegen::h3::npute::NpuTe;
	use imagegen::h3::npuvae::NpuVae;
	let Some((m, d, c)) = env() else { return };
	let Some(np) = std::env::var_os("QWEN_TEST_H3_NPU") else {
		return;
	};
	let nc = NpuCfg::parse(std::path::Path::new(&np)).unwrap();
	let (Some(tw), Some(tg)) = (nc.te.clone(), nc.te_graphs.clone()) else {
		println!("{}: no text encoder", std::path::Path::new(&np).display());
		return;
	};
	let te = TextEncoder::open(&m, &c.te).unwrap();
	let vis = vision(&m, &d, &te);
	let pads: Vec<usize> = vis
		.as_ref()
		.map_or_else(Vec::new, |v| v.grids.iter().map(|&(r, c)| r * c).collect());
	let prompt = d.str("prompt").unwrap().to_string();
	let (ids, _) = imagegen::h3::te::presentation(&te.tok, &prompt, &pads).unwrap();
	let grids: &[(usize, usize)] = vis.as_ref().map_or(&[], |v| &v.grids);
	let pos = te.positions(&ids, grids).unwrap();
	let (x, img) = te.inputs(&ids, vis.as_ref()).unwrap();
	let (cos, sin) = te.rope_tables(&pos);
	let deep: &[Vec<f32>] = vis.as_ref().map_or(&[], |v| &v.deep);
	let mut n = NpuTe::load(&tw, &tg, &c.te, |_| {}).unwrap();
	let mut out = Vec::new();
	for i in 0..3 {
		let t = std::time::Instant::now();
		out = n.encode(&x, &cos, &sin, &img, deep).unwrap();
		println!(
			"npu te: {} rows in {:.2} s (run {i}; device waits {:.2} s, attention {:.2} s, iGPU jobs {})",
			ids.len(),
			t.elapsed().as_secs_f64(),
			n.t_wait,
			n.t_attn,
			n.gpu_jobs
		);
	}
	drop(n);
	let r_te = check("npu te out", &out, &d.get("te_out").unwrap());
	// the cores' Q6_K text encoder, the CPU backend's
	let cache = imagegen::wcache::dir();
	let q6 = TextEncoder::load_cached(
		&m,
		&c.te,
		WFmt::Q6K,
		cache.as_ref().map(|p| p.join("h3-te-q6k.bin")).as_deref(),
		|_| {},
	)
	.unwrap();
	let t = std::time::Instant::now();
	let out_q6 = q6.encode_ids(&ids, vis.as_ref(), &mut |_, _| {}).unwrap();
	println!(
		"q6k te: {} rows in {:.2} s",
		ids.len(),
		t.elapsed().as_secs_f64()
	);
	drop(q6);
	check("q6k te out", &out_q6, &d.get("te_out").unwrap());
	let (ts, sg) = c.schedule();
	let dit = Dit::load_small(&m, &c.dit, WFmt::Q6K, &ts).unwrap();
	let txt = dit.text_rows(&out, |_, _| {}).unwrap();
	let txt_q6 = dit.text_rows(&out_q6, |_, _| {}).unwrap();
	let want_txt = d.get("dit_txt").unwrap();
	let r_txt = check("npu te text rows", &txt, &want_txt);
	let r_txt_q6 = check("q6k te text rows", &txt_q6, &want_txt);
	let mut nd = NpuDit::load(&nc, &c.dit, |_| {}).unwrap();
	let l = layout_of(&d, &c);
	let rope = Rope::new(&c.dit, &l.pos);
	nd.begin(&l, &rope).unwrap();
	let refs = d.get("dit_ref_rows").unwrap_or_default();
	let noise = d.get("noise").unwrap();
	let v_of = |nd: &mut NpuDit, t: &[f32], lat: &[f32], i: usize| {
		let mut img_rows = refs.clone();
		img_rows.extend_from_slice(lat);
		let mut x = t.to_vec();
		x.extend(dit.image_rows(&img_rows).unwrap());
		dit.out_at(&nd.forward(&x, ts[i]).unwrap(), ts[i], i)
			.unwrap()
	};
	let want_v = d.get("dit_v0").unwrap();
	let r_ref = check(
		"npu velocity s0, ref text",
		&v_of(&mut nd, &want_txt, &noise, 0),
		&want_v,
	);
	let r_q6 = check(
		"npu velocity s0, q6k te text",
		&v_of(&mut nd, &txt_q6, &noise, 0),
		&want_v,
	);
	let r_npu = check(
		"npu velocity s0, npu te text",
		&v_of(&mut nd, &txt, &noise, 0),
		&want_v,
	);
	// free-running from the reference's noise with each text, decoded on the NPU
	let free = |nd: &mut NpuDit, t: &[f32]| {
		let mut lat = noise.clone();
		for i in 0..ts.len() {
			let v = v_of(nd, t, &lat, i);
			let dt = sg[i] - sg[i + 1];
			lat.iter_mut().zip(&v).for_each(|(x, &vv)| *x += dt * vv);
		}
		lat
	};
	let lats: Vec<Vec<f32>> = [&want_txt, &txt_q6, &txt]
		.iter()
		.map(|t| free(&mut nd, t))
		.collect();
	nd.release_buffers();
	drop(nd);
	let (lh, lw) = (
		d.num("height").unwrap() as usize / 16,
		d.num("width").unwrap() as usize / 16,
	);
	let vae = Vae::open(&m, &c.vae).unwrap();
	let mut nv = NpuVae::load(nc.vae.as_ref().unwrap(), &m).unwrap();
	let want = d.get("vae_out").unwrap();
	let mut ref8 = None;
	let mut db = Vec::new();
	for lat in &lats {
		let mut z = layout::unpatchify(lat, c.dit.in_ch, lh, lw, c.dit.patch);
		let p = lh * lw;
		for (ch, (&s, &mn)) in c.vae.std.iter().zip(&c.vae.mean).enumerate() {
			z[ch * p..(ch + 1) * p]
				.iter_mut()
				.for_each(|v| *v = *v * s + mn);
		}
		let (img, oh, ow) = nv.decode(&vae, &z, lh, lw).unwrap().unwrap();
		let r8 = ref8.get_or_insert_with(|| imagegen::h3::vae::to_rgb8(&want, oh, ow));
		db.push(psnr(&imagegen::h3::vae::to_rgb8(&img, oh, ow), r8));
	}
	println!(
		"npu te: out {r_te:.2e}, text rows {r_txt:.2e} (q6k {r_txt_q6:.2e}), velocity step 0 {r_ref:.2e} -> {r_npu:.2e} (q6k {r_q6:.2e})"
	);
	println!(
		"free-running image PSNR: ref text {:.1} dB, q6k te {:.1} dB, npu te {:.1} dB",
		db[0], db[1], db[2]
	);
	// the velocity's increase over the reference's text rows, against the
	// cores' Q6_K encoder's
	assert!(
		r_te < 5e-2 && r_npu - r_ref < 2.0 * (r_q6 - r_ref).max(1e-2),
		"npu text encoder errors too large"
	);
}
