//! Turning a chat conversation that contains images into something
//! `Model::forward_prefill_mm` can run.
//!
//! This is the one place that knows the whole chain, so it is the one place
//! the pieces can disagree. The contract, end to end:
//!
//! 1. Each image is resized to a whole number of 2x2 merge blocks and encoded
//!    by the tower into `grid.tokens()` vectors of the text model's width.
//! 2. The rendered chat text carries, in the image's place,
//!    `<|vision_start|>` + that many `<|image_pad|>` + `<|vision_end|>`.
//! 3. After BPE encoding, the i-th run of `<|image_pad|>` ids is where the
//!    i-th image's embeddings go, and its length is **re-checked** against the
//!    grid, because a template or tokenizer change that dropped or merged a
//!    placeholder would otherwise misalign every image in the prompt without
//!    any error.

use imgio::Image;
use model::{build_image_prompt, ImageSpan, MultimodalPrompt};
use tokenizer::Tokenizer;

use super::VisionModel;

/// One piece of a chat turn's content.
pub enum Part<'a> {
	Text(&'a str),
	Image(&'a Image),
}

/// One chat turn.
pub struct Turn<'a> {
	pub role: &'a str,
	pub parts: Vec<Part<'a>>,
}

impl<'a> Turn<'a> {
	pub fn text(role: &'a str, text: &'a str) -> Turn<'a> {
		Turn {
			role,
			parts: vec![Part::Text(text)],
		}
	}
}

/// Control tokens the vision splice needs, resolved once against the vocab.
pub struct VisionTokens {
	pub start: u32,
	pub end: u32,
	pub pad: u32,
}

impl VisionTokens {
	pub fn resolve(tok: &Tokenizer) -> Result<VisionTokens, String> {
		let get = |s: &str| {
			tok.token_id(s).ok_or_else(|| {
				format!("tokenizer: this model has no {s} token, so it cannot take images")
			})
		};
		Ok(VisionTokens {
			start: get("<|vision_start|>")?,
			end: get("<|vision_end|>")?,
			pad: get("<|image_pad|>")?,
		})
	}
}

/// Render `turns` as a chat prompt, encode the images, and assemble the
/// spliced prompt. The result is ready for `Model::forward_prefill_mm`.
///
/// `max_image_tokens` caps each image individually (see
/// [`crate::DEFAULT_MAX_TOKENS`]).
pub fn build_chat(
	tok: &Tokenizer,
	vm: &VisionModel,
	turns: &[Turn<'_>],
	max_image_tokens: usize,
	start_pos: i64,
) -> Result<MultimodalPrompt, String> {
	let vt = VisionTokens::resolve(tok)?;
	let mut rendered: Vec<(String, String)> = Vec::with_capacity(turns.len());
	// Per image, in prompt order: how many placeholder tokens it claims and the
	// projector output that fills them.
	let mut encoded: Vec<(usize, usize, Vec<f32>)> = Vec::new();

	for turn in turns {
		let mut content = String::new();
		for part in &turn.parts {
			match part {
				Part::Text(t) => content.push_str(t),
				Part::Image(img) => {
					let (emb, grid) = vm.encode(img, max_image_tokens)?;
					content.push_str(&placeholder(grid.tokens()));
					encoded.push((grid.mw, grid.mh, emb));
				}
			}
		}
		rendered.push((turn.role.to_string(), content));
	}

	let refs: Vec<(&str, &str)> = rendered
		.iter()
		.map(|(r, c)| (r.as_str(), c.as_str()))
		.collect();
	let text = tok.build_chat_prompt(&refs);
	let ids = tok.encode(&text);
	splice(ids, vt.pad, encoded, start_pos)
}

/// Splice already-encoded ids whose `<|image_pad|>` runs the caller placed
/// itself. This is the incremental path: a chat session renders only the turns
/// it has not fed yet, so it cannot go through [`build_chat`], which always
/// renders the whole conversation.
///
/// `images` is `(merged_grid_w, merged_grid_h, embeddings)` per image, in
/// prompt order.
pub fn build_raw(
	tok: &Tokenizer,
	ids: Vec<u32>,
	images: Vec<(usize, usize, Vec<f32>)>,
	start_pos: i64,
) -> Result<MultimodalPrompt, String> {
	let vt = VisionTokens::resolve(tok)?;
	splice(ids, vt.pad, images, start_pos)
}

/// The `<|vision_start|>` + N x `<|image_pad|>` + `<|vision_end|>` run that
/// stands in for one image in the rendered text.
pub fn placeholder(tokens: usize) -> String {
	let mut s = String::with_capacity(32 + tokens * 14);
	s.push_str("<|vision_start|>");
	for _ in 0..tokens {
		s.push_str("<|image_pad|>");
	}
	s.push_str("<|vision_end|>");
	s
}

/// Match each image to the run of `<|image_pad|>` ids reserved for it.
fn splice(
	ids: Vec<u32>,
	pad: u32,
	mut images: Vec<(usize, usize, Vec<f32>)>,
	start_pos: i64,
) -> Result<MultimodalPrompt, String> {
	let mut spans: Vec<ImageSpan> = Vec::with_capacity(images.len());
	let mut i = 0usize;
	let mut next_image = 0usize;
	while i < ids.len() {
		if ids[i] != pad {
			i += 1;
			continue;
		}
		let start = i;
		while i < ids.len() && ids[i] == pad {
			i += 1;
		}
		let run = i - start;
		let n_images = images.len();
		let slot = images.get_mut(next_image).ok_or_else(|| {
			format!(
				"prompt: found a run of {run} <|image_pad|> tokens but only {n_images} images were encoded"
			)
		})?;
		let (mw, mh) = (slot.0, slot.1);
		if run != mw * mh {
			return Err(format!(
				"prompt: image {next_image} needs {} placeholder tokens ({mw}x{mh}) but the encoded prompt reserved {run}",
				mw * mh
			));
		}
		spans.push(ImageSpan {
			start,
			grid_w: mw,
			grid_h: mh,
			embeddings: std::mem::take(&mut slot.2),
		});
		next_image += 1;
	}
	if next_image != images.len() {
		return Err(format!(
			"prompt: {} images were encoded but the prompt has {next_image} placeholder runs",
			images.len()
		));
	}
	build_image_prompt(ids, pad, spans, start_pos)
}
