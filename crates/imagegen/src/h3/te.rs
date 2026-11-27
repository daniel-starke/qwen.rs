//! The text encoder's input for MiniMax-H3-Image: the prompt verbatim, no
//! chat template, no special tokens, at most [`MAX_PROMPT_TOKENS`] tokens;
//! an edit puts `Picture i: ` (no angle brackets, unlike the `<Picture i>`
//! the prompt refers to it by) and a vision block
//! `<|vision_start|>{pads}<|vision_end|>` per reference in front. Vision
//! block rows are tagged as image, the rest as text. The conditioning is
//! every row of `hidden_states[50]` (`crate::te::TextEncoder::encode_ids`
//! with the stack cut to 50 layers), vision blocks included.

use super::config::{MAX_PROMPT_TOKENS, TAG_IMAGE, TAG_TEXT};
use tokenizer::Tokenizer;

/// Token ids and modality tags of the presentation, for references whose
/// vision tower grids hold `pads[j]` merged tokens each.
pub fn presentation(
	tok: &Tokenizer,
	prompt: &str,
	pads: &[usize],
) -> Result<(Vec<u32>, Vec<u8>), String> {
	let id = |t: &str| tok.token_id(t).ok_or_else(|| format!("tokenizer: no {t}"));
	let (vs, vp, ve) = (
		id("<|vision_start|>")?,
		id("<|image_pad|>")?,
		id("<|vision_end|>")?,
	);
	let (mut ids, mut tags) = (Vec::new(), Vec::new());
	for (j, &n) in pads.iter().enumerate() {
		let mut label = tok.encode(&format!("Picture {}: ", j + 1));
		label.truncate(MAX_PROMPT_TOKENS);
		tags.extend(std::iter::repeat_n(TAG_TEXT, label.len()));
		ids.extend(label);
		ids.push(vs);
		ids.extend(std::iter::repeat_n(vp, n));
		ids.push(ve);
		tags.extend(std::iter::repeat_n(TAG_IMAGE, n + 2));
	}
	let mut p = tok.encode(prompt);
	if p.is_empty() {
		return Err("empty prompt".into());
	}
	p.truncate(MAX_PROMPT_TOKENS);
	tags.extend(std::iter::repeat_n(TAG_TEXT, p.len()));
	ids.extend(p);
	Ok((ids, tags))
}
