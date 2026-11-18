//! Qwen3-VL vision tower + projector (M9).
//!
//! - **Author:** Daniel Starke
//! - **Copyright:** Copyright 2026 Daniel Starke
//!
//! Loads the companion `mmproj-*.gguf` (`general.architecture = clip`,
//! `clip.projector_type = qwen3vl_merger`) and turns an image into the
//! `[n_tokens, n_embd_text]` block of embeddings that gets spliced into the
//! text model's prompt in place of its `<|image_pad|>` tokens.
//!
//! Shape of the thing, for the 27B mmproj:
//!
//! ```text
//!   image -> smart_resize -> patches [P, 3*16*16]      (merge-block order)
//!         -> patch_embd (fused Conv3d)   -> [P, 1152]
//!         -+ interpolated position embd  -> [P, 1152]
//!         -> 27 x { LN, 2-D RoPE attn, LN, MLP }
//!         -> post_ln, 2x2 merge          -> [P/4, 4608]
//!         -> mm.0, GELU, mm.2            -> [P/4, 5120]
//! ```
//!
//! Two details are easy to get wrong and produce plausible-looking nonsense
//! rather than an error, so they are called out where they happen: the
//! **merge-block patch order** (`preprocess`) and the fact that the patch
//! embedding is a **Conv3d with temporal patch size 2** whose two temporal
//! slices are both fed the same still frame (`VisionModel::load`).

use std::io::{Read, Seek, SeekFrom};

use gguf::GgufFile;

pub mod linalg;
pub mod preprocess;
pub mod prompt;
#[cfg(test)]
mod tests;
pub mod tower;

pub use preprocess::{patch_positions, patchify, smart_resize, Grid};
pub use prompt::{build_chat, build_raw, placeholder, Part, Turn, VisionTokens};

/// Default cap on how many merged tokens one image may occupy. 1024 tokens is
/// a 1024x1024-ish image; the tower is O(n^2) in patches, so this is the knob
/// that decides whether an image costs a second or a minute.
pub const DEFAULT_MAX_TOKENS: usize = 1024;
/// Floor, so a thumbnail is still upscaled to something the tower can use.
pub const DEFAULT_MIN_TOKENS: usize = 4;

/// Hyperparameters read from the mmproj's `clip.*` metadata.
#[derive(Debug, Clone)]
pub struct VisionConfig {
	pub image_size: usize,
	pub patch_size: usize,
	pub n_embd: usize,
	pub n_ff: usize,
	pub n_layer: usize,
	pub n_head: usize,
	/// Output width of the projector: the text model's `n_embd`.
	pub proj_dim: usize,
	pub merge_size: usize,
	pub eps: f32,
	pub mean: [f32; 3],
	pub std: [f32; 3],
	/// Rows in `v.position_embd.weight`; its square root is the side of the
	/// learned position grid that gets interpolated to the real one.
	pub n_pos: usize,
}

impl VisionConfig {
	pub fn head_dim(&self) -> usize {
		self.n_embd / self.n_head
	}
}

/// One transformer block of the tower. Everything is `f32`: the mmproj is BF16
/// on disk (~0.9 GB) and expands to ~1.8 GB, which is noise next to the text
/// model's 16.8 GB and saves a second dequantization path.
pub struct VisionBlock {
	pub ln1_w: Vec<f32>,
	pub ln1_b: Vec<f32>,
	pub ln2_w: Vec<f32>,
	pub ln2_b: Vec<f32>,
	/// `[3 * n_embd, n_embd]`, Q rows then K rows then V rows.
	pub qkv_w: Vec<f32>,
	pub qkv_b: Vec<f32>,
	pub out_w: Vec<f32>,
	pub out_b: Vec<f32>,
	pub up_w: Vec<f32>,
	pub up_b: Vec<f32>,
	pub down_w: Vec<f32>,
	pub down_b: Vec<f32>,
}

/// A Qwen3-VL **deepstack** merger: block `layer`'s output, 2x2-merged,
/// LayerNorm over the merged width (the "post-shuffle" norm), then `fc1`,
/// GELU, `fc2` to the text width. Its rows are added to the text stack's
/// hidden states at the image tokens after one of the first text layers.
pub struct DeepMerger {
	pub layer: usize,
	pub ln_w: Vec<f32>,
	pub ln_b: Vec<f32>,
	pub fc1_w: Vec<f32>,
	pub fc1_b: Vec<f32>,
	pub fc2_w: Vec<f32>,
	pub fc2_b: Vec<f32>,
}

pub struct VisionModel {
	pub cfg: VisionConfig,
	pub blocks: Vec<VisionBlock>,
	/// Patch embedding, `[n_embd, 3 * patch * patch]`, with both temporal
	/// slices already summed (see `load`).
	pub patch_w: Vec<f32>,
	pub patch_b: Vec<f32>,
	/// Learned position embedding, `[n_pos, n_embd]`.
	pub pos_w: Vec<f32>,
	pub post_ln_w: Vec<f32>,
	pub post_ln_b: Vec<f32>,
	pub mm0_w: Vec<f32>,
	pub mm0_b: Vec<f32>,
	pub mm2_w: Vec<f32>,
	pub mm2_b: Vec<f32>,
	/// Deepstack mergers, by block (empty for the 27B's tower).
	pub deepstack: Vec<DeepMerger>,
	/// The mergers' GELU is the exact (erf) form, as transformers'
	/// `nn.GELU()`; false keeps llama.cpp's tanh form, which the GGUF path
	/// is checked against.
	pub gelu_erf: bool,
}

/// Reads tensors from one already-open handle. `GgufFile::read_tensor_bytes`
/// reopens the file per call, and this loader pulls 334 of them.
struct TensorReader<'a> {
	file: std::fs::File,
	gguf: &'a GgufFile,
}

impl<'a> TensorReader<'a> {
	fn open(path: &str, gguf: &'a GgufFile) -> Result<Self, String> {
		let file = std::fs::File::open(path).map_err(|e| format!("mmproj: open {path}: {e}"))?;
		Ok(TensorReader { file, gguf })
	}

	/// Dequantize a tensor by name, checking its element count.
	fn get(&mut self, name: &str, want: usize) -> Result<Vec<f32>, String> {
		let t = self
			.gguf
			.tensors
			.iter()
			.find(|t| t.name == name)
			.ok_or_else(|| format!("mmproj: missing tensor {name}"))?;
		let n = t.num_elements();
		if n != want {
			return Err(format!(
				"mmproj: {name} has {n} elements, expected {want} (dims {:?})",
				t.dims
			));
		}
		let mut raw = vec![0u8; t.byte_length()];
		self.file
			.seek(SeekFrom::Start(self.gguf.data_offset + t.offset))
			.and_then(|_| self.file.read_exact(&mut raw))
			.map_err(|e| format!("mmproj: read {name}: {e}"))?;
		Ok(quant::dequantize(t.ggml_type, &raw, n))
	}

	fn has(&self, name: &str) -> bool {
		self.gguf.tensors.iter().any(|t| t.name == name)
	}
}

fn meta_usize(g: &GgufFile, key: &str) -> Option<usize> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| e.value.as_u64())
		.map(|v| v as usize)
}

fn meta_f32(g: &GgufFile, key: &str) -> Option<f32> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| e.value.as_f64())
		.map(|v| v as f32)
}

fn meta_f32x3(g: &GgufFile, key: &str, dflt: [f32; 3]) -> [f32; 3] {
	let mut out = dflt;
	if let Some(gguf::MetadataValue::Array { values, .. }) =
		g.metadata.iter().find(|e| e.key == key).map(|e| &e.value)
	{
		for (i, v) in values.iter().take(3).enumerate() {
			if let Some(f) = v.as_f64() {
				out[i] = f as f32;
			}
		}
	}
	out
}

fn meta_str<'a>(g: &'a GgufFile, key: &str) -> Option<&'a str> {
	g.metadata
		.iter()
		.find(|e| e.key == key)
		.and_then(|e| match &e.value {
			gguf::MetadataValue::String(s) => Some(s.as_str()),
			_ => None,
		})
}

impl VisionModel {
	/// Load an `mmproj-*.gguf`.
	///
	/// `text_n_embd`, when given, is checked against the projector's output
	/// width: a mismatch means the mmproj belongs to a different model, which
	/// is otherwise only visible as garbage generation.
	pub fn load(path: &str, text_n_embd: Option<usize>) -> Result<VisionModel, String> {
		let gguf = gguf::parse_file(path).map_err(|e| format!("mmproj: parse {path}: {e}"))?;

		match meta_str(&gguf, "general.architecture") {
			Some("clip") => {}
			other => {
				return Err(format!(
					"mmproj: expected general.architecture=clip, got {:?}",
					other.unwrap_or("<missing>")
				))
			}
		}
		if let Some(p) = meta_str(&gguf, "clip.projector_type") {
			if p != "qwen3vl_merger" {
				return Err(format!(
					"mmproj: projector type {p:?} is not supported (want qwen3vl_merger)"
				));
			}
		}
		let key = |k: &str| -> Result<usize, String> {
			meta_usize(&gguf, k).ok_or_else(|| format!("mmproj: missing {k}"))
		};
		let cfg = VisionConfig {
			image_size: key("clip.vision.image_size")?,
			patch_size: key("clip.vision.patch_size")?,
			n_embd: key("clip.vision.embedding_length")?,
			n_ff: key("clip.vision.feed_forward_length")?,
			n_layer: key("clip.vision.block_count")?,
			n_head: key("clip.vision.attention.head_count")?,
			proj_dim: key("clip.vision.projection_dim")?,
			merge_size: meta_usize(&gguf, "clip.vision.spatial_merge_size").unwrap_or(2),
			eps: meta_f32(&gguf, "clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6),
			mean: meta_f32x3(&gguf, "clip.vision.image_mean", [0.5, 0.5, 0.5]),
			std: meta_f32x3(&gguf, "clip.vision.image_std", [0.5, 0.5, 0.5]),
			n_pos: 0,
		};
		if !cfg.n_embd.is_multiple_of(cfg.n_head) {
			return Err(format!(
				"mmproj: embedding {} is not a multiple of {} heads",
				cfg.n_embd, cfg.n_head
			));
		}
		if let Some(n) = text_n_embd {
			if n != cfg.proj_dim {
				return Err(format!(
					"mmproj: projects to {} dims but the text model wants {n}: wrong mmproj for this model",
					cfg.proj_dim
				));
			}
		}
		// `is_deepstack_layers` marks blocks whose output is *also* injected
		// into the text stack. This mmproj has none, and honoring them would
		// need a second projector plus text-side plumbing, so refuse rather
		// than silently drop the extra features.
		if let Some(gguf::MetadataValue::Array { values, .. }) = gguf
			.metadata
			.iter()
			.find(|e| e.key == "clip.vision.is_deepstack_layers")
			.map(|e| &e.value)
		{
			if values
				.iter()
				.any(|v| matches!(v, gguf::MetadataValue::Bool(true)))
			{
				return Err("mmproj: deepstack layers are not supported by this build".to_string());
			}
		}

		let mut cfg = cfg;
		let mut r = TensorReader::open(path, &gguf)?;
		let e = cfg.n_embd;
		let plen = 3 * cfg.patch_size * cfg.patch_size;

		// Patch embedding. `v.patch_embd.weight` / `.weight.1` are the two
		// temporal slices of a Conv3d with temporal_patch_size = 2. A still
		// image is fed to the tower as the same frame twice (that is what the
		// reference preprocessor does), so the convolution collapses to a 2-D
		// one whose kernel is the *sum* of the slices. Folding it here keeps
		// the patch feature vector at 3*16*16 instead of doubling it.
		let mut patch_w = r.get("v.patch_embd.weight", e * plen)?;
		if r.has("v.patch_embd.weight.1") {
			let w1 = r.get("v.patch_embd.weight.1", e * plen)?;
			for (a, b) in patch_w.iter_mut().zip(w1.iter()) {
				*a += *b;
			}
		}
		let patch_b = r.get("v.patch_embd.bias", e)?;

		// Position embedding: rows are a square grid, interpolated per image.
		let pos_t = gguf
			.tensors
			.iter()
			.find(|t| t.name == "v.position_embd.weight")
			.ok_or("mmproj: missing v.position_embd.weight")?;
		let n_pos = pos_t.num_elements() / e;
		let side = (n_pos as f64).sqrt().round() as usize;
		if side * side != n_pos {
			return Err(format!(
				"mmproj: {n_pos} position embeddings are not a square grid"
			));
		}
		cfg.n_pos = n_pos;
		let pos_w = r.get("v.position_embd.weight", n_pos * e)?;

		let mut blocks = Vec::with_capacity(cfg.n_layer);
		for il in 0..cfg.n_layer {
			let p = |n: &str| format!("v.blk.{il}.{n}");
			blocks.push(VisionBlock {
				ln1_w: r.get(&p("ln1.weight"), e)?,
				ln1_b: r.get(&p("ln1.bias"), e)?,
				ln2_w: r.get(&p("ln2.weight"), e)?,
				ln2_b: r.get(&p("ln2.bias"), e)?,
				qkv_w: r.get(&p("attn_qkv.weight"), 3 * e * e)?,
				qkv_b: r.get(&p("attn_qkv.bias"), 3 * e)?,
				out_w: r.get(&p("attn_out.weight"), e * e)?,
				out_b: r.get(&p("attn_out.bias"), e)?,
				up_w: r.get(&p("ffn_up.weight"), cfg.n_ff * e)?,
				up_b: r.get(&p("ffn_up.bias"), cfg.n_ff)?,
				down_w: r.get(&p("ffn_down.weight"), e * cfg.n_ff)?,
				down_b: r.get(&p("ffn_down.bias"), e)?,
			});
		}

		let merged = e * cfg.merge_size * cfg.merge_size;
		Ok(VisionModel {
			post_ln_w: r.get("v.post_ln.weight", e)?,
			post_ln_b: r.get("v.post_ln.bias", e)?,
			mm0_w: r.get("mm.0.weight", merged * merged)?,
			mm0_b: r.get("mm.0.bias", merged)?,
			mm2_w: r.get("mm.2.weight", merged * cfg.proj_dim)?,
			mm2_b: r.get("mm.2.bias", cfg.proj_dim)?,
			deepstack: Vec::new(),
			gelu_erf: false,
			cfg,
			blocks,
			patch_w,
			patch_b,
			pos_w,
		})
	}

	/// Grid this model would use for an image of `width x height`.
	pub fn grid_for(&self, width: usize, height: usize, max_tokens: usize) -> Result<Grid, String> {
		smart_resize(
			width,
			height,
			self.cfg.patch_size,
			self.cfg.merge_size,
			DEFAULT_MIN_TOKENS,
			max_tokens.max(1),
		)
	}

	/// Encode one image into `[grid.tokens(), proj_dim]` embeddings, ready to
	/// be spliced over the prompt's `<|image_pad|>` positions.
	pub fn encode(
		&self,
		img: &imgio::Image,
		max_tokens: usize,
	) -> Result<(Vec<f32>, Grid), String> {
		let grid = self.grid_for(img.width, img.height, max_tokens)?;
		let patches = patchify(
			img,
			&grid,
			self.cfg.patch_size,
			self.cfg.merge_size,
			self.cfg.mean,
			self.cfg.std,
		);
		let out = tower::forward(self, &patches, &grid);
		Ok((out, grid))
	}
}
