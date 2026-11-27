//! Qwen-Image-2.1's text encoder prompt on the shared Qwen3-VL stack
//! ([`crate::te::TextEncoder`]).
//!
//! The prompt is the pipeline's raw template (not `apply_chat_template`):
//! `<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n
//! <|im_start|>assistant\n`. The output is the **last layer's hidden state
//! before the final RMSNorm** (transformers 4.x `hidden_states[-1]`, what the
//! transformer was trained on; 5.x would normalize it), with the system turn
//! (every token before the second `<|im_start|>`) dropped.
//!
//! With condition images (an edit) the user turn starts with
//! `<image1><|vision_start|>{pads}<|vision_end|>` (and ` <imageN>...` for
//! each further one; `<imageN>` is plain text). The pads stay in the
//! output: the transformer puts the image's latents at those slots
//! ([`Encoded::image_pad`]).

use crate::te::{TextEncoder, VisionRows};

pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";
const IM_START: &str = "<|im_start|>";

/// What [`TextEncoder::encode`] returns.
pub struct Encoded {
	/// Every token of the template, system turn included.
	pub ids: Vec<u32>,
	/// Tokens dropped from the front (the system turn).
	pub drop: usize,
	/// `[ids.len() - drop][dim]`.
	pub hidden: Vec<f32>,
	/// Per output row: an image slot (`<|image_pad|>`).
	pub image_pad: Vec<bool>,
}

impl TextEncoder {
	pub fn template(prompt: &str) -> String {
		Self::template_images(prompt, &[])
	}

	/// The template with `<imageN><|vision_start|>`, `tokens[N - 1]` pads
	/// and `<|vision_end|>` per condition image, space-separated, before the
	/// prompt.
	pub fn template_images(prompt: &str, tokens: &[usize]) -> String {
		// an empty prompt leaves the encoder nothing to read (no BOS)
		let p = if prompt.is_empty() { " " } else { prompt };
		let mut img = String::new();
		for (i, &n) in tokens.iter().enumerate() {
			if i > 0 {
				img.push(' ');
			}
			img.push_str(&format!("<image{}>", i + 1));
			img.push_str(&vision::placeholder(n));
		}
		format!(
			"{IM_START}system\n{SYSTEM_PROMPT}<|im_end|>\n{IM_START}user\n{img}{p}<|im_end|>\n{IM_START}assistant\n"
		)
	}

	pub fn tokenize(&self, prompt: &str) -> Result<(Vec<u32>, usize), String> {
		self.tokenize_images(prompt, &[])
	}

	/// Token ids of [`Self::template_images`] and the system turn's length.
	pub fn tokenize_images(
		&self,
		prompt: &str,
		tokens: &[usize],
	) -> Result<(Vec<u32>, usize), String> {
		let ids = self.tok.encode(&Self::template_images(prompt, tokens));
		let st = self
			.tok
			.token_id(IM_START)
			.ok_or("tokenizer: no <|im_start|>")?;
		let drop = ids
			.iter()
			.enumerate()
			.filter(|(_, &t)| t == st)
			.nth(1)
			.map(|(i, _)| i)
			.ok_or("template: no second <|im_start|>")?;
		Ok((ids, drop))
	}

	/// Encode `prompt`; `tap(layer, hidden)` sees every layer's output (all
	/// positions) for the reference checks.
	pub fn encode(&self, prompt: &str, tap: impl FnMut(usize, &[f32])) -> Result<Encoded, String> {
		self.encode_images(prompt, None, tap)
	}

	/// [`Self::encode`] with condition images (`vis`: the vision tower's rows).
	/// `tap` sees each layer's output before its deepstack addition.
	pub fn encode_images(
		&self,
		prompt: &str,
		vis: Option<&VisionRows>,
		mut tap: impl FnMut(usize, &[f32]),
	) -> Result<Encoded, String> {
		let grids: &[(usize, usize)] = vis.map_or(&[], |v| &v.grids);
		let tokens: Vec<usize> = grids.iter().map(|&(r, c)| r * c).collect();
		let (ids, drop) = self.tokenize_images(prompt, &tokens)?;
		let d = self.cfg.dim;
		let pad = self.image_pad_id()?;
		let x = self.encode_ids(&ids, vis, &mut tap)?;
		let hidden = x[drop * d..].to_vec();
		let image_pad = ids[drop..].iter().map(|&t| t == pad).collect();
		Ok(Encoded {
			ids,
			drop,
			hidden,
			image_pad,
		})
	}
}
