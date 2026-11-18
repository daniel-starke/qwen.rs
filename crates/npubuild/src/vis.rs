//! The vision tower (Qwen3-VL ViT + merger) for the NPU engine: its
//! weights from the checkpoint ([`model`]: the same `vision::VisionModel`
//! the GGUF path loads from an mmproj) and its graphs ([`graph`]).
//!
//! A static graph cannot hold the tower as it is: its attention is over
//! every patch of the image, and the patch count is the image's. So the
//! tower is cut at each attention: everything per patch (the patch
//! embedding, the norms, the projections, the MLPs, the merger) is NPU
//! graphs over [`S`] patches at a time, and the attention between two of
//! them (2-D RoPE included) runs elsewhere over all patches at once (the
//! iGPU: `offload::VisAttn`). `depth + 1` graphs:
//!
//! ```text
//!   vis0    patch, pos        -> xo, qkv     patch embedding, block 0 up to its attention
//!   vis{l}  x, a              -> xo, qkv     block l - 1 from its attention on, block l up to its
//!   vis{d}  x, a              -> emb         block d - 1 from its attention on, the merger
//! ```
//!
//! A tower with deepstack mergers (Qwen-Image's text encoder) adds to
//! graph `l + 1` of each tapped block `l` an output `deep{j}` [1, S / 4,
//! proj]: merger `j` over block `l`'s output (after the graph's other
//! outputs; `deep` in `vis.cfg` lists the taps).
//!
//! `x` / `xo` [1, S, 1152] is the residual, `qkv` [1, S, 3456] the
//! projections (q, then k, then v) before RoPE, `a` [1, S, 1152] the
//! attention's output before `attn.proj`, `emb` [1, S / 4, 5120]. Patches
//! are in merge-block order, so a graph's S patches are S / 4 whole tokens.
//! Weights are f16 constants and everything is computed in f16: against the
//! f32 tower on the cores (`vischeck`) the embeddings' relative error is
//! 0.4 - 0.8 %, their cosine 0.99997.

use crate::g::{Graph, Ty, V};
use crate::st::{parse_json, Checkpoint, Json};
use vision::{VisionBlock, VisionConfig, VisionModel};

/// Patches per NPU job.
pub const S: usize = 256;

/// True when the checkpoint has a vision tower.
pub fn has_tower(ck: &Checkpoint) -> bool {
	ck.config.get("vision_config").is_some() && ck.has("model.visual.patch_embed.proj.weight")
}

fn vnum(ck: &Checkpoint, k: &str) -> Result<usize, String> {
	let vc = ck
		.config
		.get("vision_config")
		.ok_or("config.json: no vision_config")?;
	vc.get(k)
		.and_then(Json::num)
		.map(|v| v as usize)
		.ok_or(format!("config.json: vision_config has no {k}"))
}

/// The blocks whose outputs feed deepstack mergers (`deepstack_visual_indexes`).
pub fn deepstack_layers(ck: &Checkpoint) -> Vec<usize> {
	ck.config
		.get("vision_config")
		.and_then(|v| v.get("deepstack_visual_indexes"))
		.map_or(&[][..], Json::arr)
		.iter()
		.filter_map(Json::num)
		.map(|x| x as usize)
		.collect()
}

/// The tower's configuration (`config.json`, `preprocessor_config.json`);
/// an error for a tower with deepstack mergers (the NPU graphs have none).
pub fn config(ck: &Checkpoint) -> Result<VisionConfig, String> {
	if !deepstack_layers(ck).is_empty() {
		return Err("vision: deepstack layers are not supported".into());
	}
	config_any(ck)
}

/// [`config`] with or without deepstack mergers.
pub fn config_any(ck: &Checkpoint) -> Result<VisionConfig, String> {
	let num = |k: &str| vnum(ck, k);
	let pp = std::fs::read(ck.dir.join("preprocessor_config.json"))
		.ok()
		.and_then(|b| parse_json(&b).ok());
	let triple = |k: &str| -> [f32; 3] {
		let v: Vec<f32> = pp
			.as_ref()
			.and_then(|p| p.get(k))
			.map_or(&[][..], Json::arr)
			.iter()
			.filter_map(Json::num)
			.map(|x| x as f32)
			.collect();
		if v.len() == 3 {
			[v[0], v[1], v[2]]
		} else {
			[0.5; 3]
		}
	};
	Ok(VisionConfig {
		image_size: 0,
		patch_size: num("patch_size")?,
		n_embd: num("hidden_size")?,
		n_ff: num("intermediate_size")?,
		n_layer: num("depth")?,
		n_head: num("num_heads")?,
		proj_dim: num("out_hidden_size")?,
		merge_size: num("spatial_merge_size")?,
		eps: 1e-6,
		mean: triple("image_mean"),
		std: triple("image_std"),
		n_pos: num("num_position_embeddings")?,
	})
}

/// The learned position grid, `[n_pos][embd]`.
pub fn pos_grid(ck: &Checkpoint, cfg: &VisionConfig) -> Result<Vec<f32>, String> {
	ck.mat("model.visual.pos_embed.weight", cfg.n_pos, cfg.n_embd)
}

/// The tower's weights from the checkpoint (`model.visual.*`).
pub fn model(ck: &Checkpoint) -> Result<VisionModel, String> {
	model_with(ck, config(ck)?)
}

/// The tower with its deepstack mergers (Qwen3-VL as Qwen-Image's text
/// encoder), mergers with the exact GELU as transformers runs them.
pub fn model_deepstack(ck: &Checkpoint) -> Result<VisionModel, String> {
	let mut m = model_with(ck, config_any(ck)?)?;
	let merged = m.cfg.n_embd * m.cfg.merge_size * m.cfg.merge_size;
	let p = |n: &str| format!("model.visual.{n}");
	for (j, layer) in deepstack_layers(ck).into_iter().enumerate() {
		if layer >= m.cfg.n_layer {
			return Err(format!(
				"vision: deepstack layer {layer} past the {} blocks",
				m.cfg.n_layer
			));
		}
		let d = |n: &str| p(&format!("deepstack_merger_list.{j}.{n}"));
		m.deepstack.push(vision::DeepMerger {
			layer,
			ln_w: ck.vec(&d("norm.weight"), merged)?,
			ln_b: ck.vec(&d("norm.bias"), merged)?,
			fc1_w: ck.mat(&d("linear_fc1.weight"), merged, merged)?,
			fc1_b: ck.vec(&d("linear_fc1.bias"), merged)?,
			fc2_w: ck.mat(&d("linear_fc2.weight"), m.cfg.proj_dim, merged)?,
			fc2_b: ck.vec(&d("linear_fc2.bias"), m.cfg.proj_dim)?,
		});
	}
	m.gelu_erf = true;
	Ok(m)
}

fn model_with(ck: &Checkpoint, cfg: VisionConfig) -> Result<VisionModel, String> {
	let (e, ff, depth, patch, merge) = (
		cfg.n_embd,
		cfg.n_ff,
		cfg.n_layer,
		cfg.patch_size,
		cfg.merge_size,
	);
	let tp = vnum(ck, "temporal_patch_size")?;
	let p = |n: &str| format!("model.visual.{n}");
	// the patch embedding is a Conv3d [e, 3, tp, patch, patch]; a still image
	// is the same frame `tp` times, so its temporal slices sum
	let plen = 3 * patch * patch;
	let w5 = ck.mat(&p("patch_embed.proj.weight"), e, 3 * tp * patch * patch)?;
	let mut patch_w = vec![0f32; e * plen];
	for o in 0..e {
		for c in 0..3 {
			for t in 0..tp {
				let src =
					&w5[(o * 3 + c) * tp * patch * patch + t * patch * patch..][..patch * patch];
				for (d, s) in patch_w[o * plen + c * patch * patch..][..patch * patch]
					.iter_mut()
					.zip(src)
				{
					*d += *s;
				}
			}
		}
	}
	let mut blocks = Vec::with_capacity(depth);
	for l in 0..depth {
		let b = |n: &str| p(&format!("blocks.{l}.{n}"));
		blocks.push(VisionBlock {
			ln1_w: ck.vec(&b("norm1.weight"), e)?,
			ln1_b: ck.vec(&b("norm1.bias"), e)?,
			ln2_w: ck.vec(&b("norm2.weight"), e)?,
			ln2_b: ck.vec(&b("norm2.bias"), e)?,
			qkv_w: ck.mat(&b("attn.qkv.weight"), 3 * e, e)?,
			qkv_b: ck.vec(&b("attn.qkv.bias"), 3 * e)?,
			out_w: ck.mat(&b("attn.proj.weight"), e, e)?,
			out_b: ck.vec(&b("attn.proj.bias"), e)?,
			up_w: ck.mat(&b("mlp.linear_fc1.weight"), ff, e)?,
			up_b: ck.vec(&b("mlp.linear_fc1.bias"), ff)?,
			down_w: ck.mat(&b("mlp.linear_fc2.weight"), e, ff)?,
			down_b: ck.vec(&b("mlp.linear_fc2.bias"), e)?,
		});
	}
	let merged = e * merge * merge;
	Ok(VisionModel {
		patch_b: ck.vec(&p("patch_embed.proj.bias"), e)?,
		pos_w: pos_grid(ck, &cfg)?,
		post_ln_w: ck.vec(&p("merger.norm.weight"), e)?,
		post_ln_b: ck.vec(&p("merger.norm.bias"), e)?,
		mm0_w: ck.mat(&p("merger.linear_fc1.weight"), merged, merged)?,
		mm0_b: ck.vec(&p("merger.linear_fc1.bias"), merged)?,
		mm2_w: ck.mat(&p("merger.linear_fc2.weight"), cfg.proj_dim, merged)?,
		mm2_b: ck.vec(&p("merger.linear_fc2.bias"), cfg.proj_dim)?,
		deepstack: Vec::new(),
		gelu_erf: false,
		cfg,
		blocks,
		patch_w,
	})
}

fn c16(g: &mut Graph, shape: &[usize], v: &[f32]) -> V {
	let b: Vec<u8> = v
		.iter()
		.flat_map(|&x| crate::g::f16_bits(x as f64).to_le_bytes())
		.collect();
	g.constant(Ty::F16, shape, &b)
}

/// `x` [.., k] -> `x w^T + b` [.., n].
fn linear(g: &mut Graph, x: &V, w: &[f32], b: &[f32]) -> V {
	let (n, k) = (b.len(), *x.shape.last().unwrap());
	let wc = c16(g, &[n, k], w);
	let y = g.matmul(x, &wc, false, true);
	let bc = c16(g, &[1, 1, n], b);
	g.add(&y, &bc)
}

fn layer_norm(g: &mut Graph, x: &V, w: &[f32], b: &[f32], eps: f32) -> V {
	let n = w.len();
	let y = g.mvn(x, eps as f64);
	let wc = c16(g, &[1, 1, n], w);
	let y = g.mul(&y, &wc);
	let bc = c16(g, &[1, 1, n], b);
	g.add(&y, &bc)
}

/// Block `b` up to its attention: `x` -> `qkv`.
fn pre(g: &mut Graph, x: &V, b: &VisionBlock, eps: f32) -> V {
	let h = layer_norm(g, x, &b.ln1_w, &b.ln1_b, eps);
	linear(g, &h, &b.qkv_w, &b.qkv_b)
}

/// Block `b` from its attention on: `x`, `a` -> the block's output.
fn post(g: &mut Graph, x: &V, a: &V, b: &VisionBlock, eps: f32) -> V {
	let o = linear(g, a, &b.out_w, &b.out_b);
	let x1 = g.add(x, &o);
	let h = layer_norm(g, &x1, &b.ln2_w, &b.ln2_b, eps);
	let up = linear(g, &h, &b.up_w, &b.up_b);
	let up = g.gelu_tanh(&up);
	let down = linear(g, &up, &b.down_w, &b.down_b);
	g.add(&x1, &down)
}

/// Graph `i` of the tower (0 ..= depth) over `s` patches: see the module
/// docs.
pub fn graph(m: &VisionModel, i: usize, s: usize) -> Graph {
	let c = &m.cfg;
	let (e, depth, eps) = (c.n_embd, c.n_layer, c.eps);
	let mut gr = Graph::new();
	let g = &mut gr;
	let x = if i == 0 {
		let patch = g.parameter("patch", Ty::F16, &[1, s, 3 * c.patch_size * c.patch_size]);
		let pos = g.parameter("pos", Ty::F16, &[1, s, e]);
		let x = linear(g, &patch, &m.patch_w, &m.patch_b);
		g.add(&x, &pos)
	} else {
		let x = g.parameter("x", Ty::F16, &[1, s, e]);
		let a = g.parameter("a", Ty::F16, &[1, s, e]);
		post(g, &x, &a, &m.blocks[i - 1], eps)
	};
	let mm = c.merge_size * c.merge_size;
	let act = |g: &mut Graph, y: &V| {
		if m.gelu_erf {
			g.gelu_erf(y)
		} else {
			g.gelu_tanh(y)
		}
	};
	if i < depth {
		let qkv = pre(g, &x, &m.blocks[i], eps);
		g.result(&x, "xo");
		g.result(&qkv, "qkv");
	} else {
		// the merger: norm per patch, each token's 2 x 2 patches side by
		// side (they are consecutive: merge-block order), two layers
		let h = layer_norm(g, &x, &m.post_ln_w, &m.post_ln_b, eps);
		let h = g.reshape(&h, &[1, (s / mm) as i64, (e * mm) as i64]);
		let y = linear(g, &h, &m.mm0_w, &m.mm0_b);
		let y = act(g, &y);
		let y = linear(g, &y, &m.mm2_w, &m.mm2_b);
		g.result(&y, "emb");
	}
	// deepstack mergers of block i - 1's output: norm over each token's
	// merged row (post-shuffle), two layers
	for (j, dm) in m.deepstack.iter().enumerate() {
		if i == 0 || dm.layer + 1 != i {
			continue;
		}
		let h = g.reshape(&x, &[1, (s / mm) as i64, (e * mm) as i64]);
		let h = layer_norm(g, &h, &dm.ln_w, &dm.ln_b, eps);
		let y = linear(g, &h, &dm.fc1_w, &dm.fc1_b);
		let y = act(g, &y);
		let y = linear(g, &y, &dm.fc2_w, &dm.fc2_b);
		g.result(&y, &format!("deep{j}"));
	}
	gr
}

/// `vis.cfg` of a tower set: what the engine needs besides the blobs and
/// `pos.f32` (the learned position grid, `[n_pos][embd]` f32).
pub fn cfg_text(c: &VisionConfig, s: usize) -> String {
	cfg_text_deep(c, s, &[])
}

/// [`cfg_text`] of a tower with deepstack mergers at blocks `deep`.
pub fn cfg_text_deep(c: &VisionConfig, s: usize, deep: &[usize]) -> String {
	let t = |v: [f32; 3]| format!("{},{},{}", v[0], v[1], v[2]);
	let d = if deep.is_empty() {
		String::new()
	} else {
		let v: Vec<String> = deep.iter().map(usize::to_string).collect();
		format!("deep = {}\n", v.join(","))
	};
	d + &format!(
		"patch = {}\nmerge = {}\nembd = {}\nheads = {}\nlayers = {}\nproj = {}\nnpos = {}\ns = {s}\nmean = {}\nstd = {}\n",
		c.patch_size,
		c.merge_size,
		c.n_embd,
		c.n_head,
		c.n_layer,
		c.proj_dim,
		c.n_pos,
		t(c.mean),
		t(c.std)
	)
}
