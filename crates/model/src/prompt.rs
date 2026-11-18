//! Prompt inputs for the batched prefill, including the multimodal ones.
//!
//! A text prompt is just `&[u32]`. A prompt with an image additionally needs
//! two things the token ids cannot carry:
//!
//! 1. **Embedding overrides.** The vision tower produces one `n_embd`-wide
//!    vector per merged patch. Those replace the embedding of the
//!    `<|image_pad|>` tokens that stand in for them; the placeholder tokens
//!    stay in the sequence so the KV cache, the position bookkeeping and
//!    `decode` all keep working unchanged.
//! 2. **MRoPE positions.** Text advances `(t, h, w)` together, but an image's
//!    patches share one `t` while `h`/`w` sweep the merged grid, and the text
//!    that follows resumes one past the largest coordinate the image used. See
//!    [`build_image_prompt`].

/// Borrowed view of one prompt (or one `n_batch` slice of it).
#[derive(Clone, Copy)]
pub struct PromptInput<'a> {
	pub tokens: &'a [u32],
	/// Per-token embedding override; `None` for a text-only prompt, and
	/// `Some(rows)` with `rows.len() == tokens.len()` otherwise.
	pub embeds: Option<&'a [Option<&'a [f32]>]>,
	/// Per-token MRoPE triple; `None` means "text", i.e. `[pos + t; 3]`.
	pub mpos: Option<&'a [[i64; 3]]>,
}

impl<'a> PromptInput<'a> {
	/// A plain text prompt.
	pub fn text(tokens: &'a [u32]) -> PromptInput<'a> {
		PromptInput {
			tokens,
			embeds: None,
			mpos: None,
		}
	}

	/// Embedding override for token `t` of this view, if any.
	#[inline]
	pub fn embed(&self, t: usize) -> Option<&'a [f32]> {
		self.embeds.and_then(|e| e[t])
	}

	/// MRoPE triple for token `t` of this view.
	///
	/// `base` is the run state's **MRoPE** position, not `rs.pos`. The two are
	/// equal until an image is processed and permanently different after one,
	/// so a text prefill that follows an image must continue from the MRoPE
	/// clock or it silently re-uses positions the image already occupied.
	#[inline]
	pub fn mpos(&self, base: [i64; 3], t: usize) -> [i64; 3] {
		match self.mpos {
			Some(m) => m[t],
			None => {
				let p = base[0] + t as i64;
				[p; 3]
			}
		}
	}

	/// Does any token in this view carry a non-degenerate (image) position?
	pub fn has_image_positions(&self) -> bool {
		self.mpos
			.map(|m| m.iter().any(|p| p[0] != p[1] || p[1] != p[2]))
			.unwrap_or(false)
	}

	/// MRoPE position for the token *after* a chunk of `n` tokens starting at
	/// `cur`. Text simply advances; an explicit table resumes one past the
	/// largest coordinate the chunk used, which is what the reference
	/// implementation does across an image boundary.
	pub fn mpos_after(&self, cur: [i64; 3], n: usize) -> [i64; 3] {
		match self.mpos {
			Some(m) => {
				let mut max = cur[0].max(cur[1]).max(cur[2]) - 1;
				for p in m.iter().take(n) {
					max = max.max(p[0]).max(p[1]).max(p[2]);
				}
				[max + 1; 3]
			}
			None => {
				let next = cur[0].max(cur[1]).max(cur[2]) + n as i64;
				[next; 3]
			}
		}
	}

	/// Sub-view over `tokens[a..b]`, slicing the side tables with it.
	pub fn slice(&self, a: usize, b: usize) -> PromptInput<'a> {
		PromptInput {
			tokens: &self.tokens[a..b],
			embeds: self.embeds.map(|e| &e[a..b]),
			mpos: self.mpos.map(|m| &m[a..b]),
		}
	}
}

/// An image's contribution to a prompt: where its placeholder tokens sit and
/// what to put there.
#[derive(Debug)]
pub struct ImageSpan {
	/// Index of the first `<|image_pad|>` token in `tokens`.
	pub start: usize,
	/// Merged-grid size, in tokens.
	pub grid_w: usize,
	pub grid_h: usize,
	/// `grid_w * grid_h * n_embd` projector outputs, row-major.
	pub embeddings: Vec<f32>,
}

impl ImageSpan {
	pub fn tokens(&self) -> usize {
		self.grid_w * self.grid_h
	}
}

/// Owned buffers backing a multimodal [`PromptInput`].
///
/// Kept as a separate struct because `PromptInput` borrows: the caller holds
/// one of these for as long as the prefill runs.
#[derive(Debug)]
pub struct MultimodalPrompt {
	pub tokens: Vec<u32>,
	pub mpos: Vec<[i64; 3]>,
	/// Flat rows, indexed by `rows`.
	images: Vec<ImageSpan>,
	row_of: Vec<Option<(usize, usize)>>,
}

impl MultimodalPrompt {
	/// Borrowing view for the prefill. `scratch` receives the per-token
	/// override slice; it is threaded through rather than stored so
	/// `MultimodalPrompt` stays free of self-references.
	pub fn input<'a>(
		&'a self,
		scratch: &'a mut Vec<Option<&'a [f32]>>,
		n_embd: usize,
	) -> PromptInput<'a> {
		scratch.clear();
		scratch.reserve(self.tokens.len());
		for slot in &self.row_of {
			scratch.push(
				slot.map(|(img, row)| {
					&self.images[img].embeddings[row * n_embd..(row + 1) * n_embd]
				}),
			);
		}
		PromptInput {
			tokens: &self.tokens,
			embeds: Some(&scratch[..]),
			mpos: Some(&self.mpos),
		}
	}

	/// Does this prompt actually carry an image?
	pub fn has_images(&self) -> bool {
		!self.images.is_empty()
	}
}

/// Assemble a prompt from already-tokenized text and the images that were
/// spliced into it.
///
/// `image_pad` is the id of `<|image_pad|>`; every run of exactly
/// `span.tokens()` of them starting at `span.start` is replaced by that image's
/// embeddings. `spans` must be sorted by `start` and must not overlap.
///
/// MRoPE positions follow the reference: text advances all three axes
/// together; an image's patches all share the `t` the image starts at, and
/// carry their merged-grid row and column in `h` and `w`; the text after an
/// image resumes one past the largest coordinate the image used. Note the
/// asymmetry that follows: a 1024-token image advances the sequence position
/// by 1024 but the MRoPE clock by only `max(grid_h, grid_w)`.
///
/// `start_pos` is the MRoPE coordinate the first token takes, i.e. `rs.mpos[0]`
/// for a continuing conversation and 0 for a fresh one. It is **not** `rs.pos`:
/// after an image the two differ, and using the sequence position here would
/// hand the new turn coordinates the image already occupies.
pub fn build_image_prompt(
	tokens: Vec<u32>,
	image_pad: u32,
	spans: Vec<ImageSpan>,
	start_pos: i64,
) -> Result<MultimodalPrompt, String> {
	let n = tokens.len();
	let mut row_of: Vec<Option<(usize, usize)>> = vec![None; n];
	let mut mpos = vec![[0i64; 3]; n];

	let mut next: i64 = start_pos;
	let mut t = 0usize;
	let mut si = 0usize;
	while t < n {
		if si < spans.len() && spans[si].start == t {
			let span = &spans[si];
			let count = span.tokens();
			if t + count > n {
				return Err(format!(
					"prompt: image {si} needs {count} placeholder tokens but only {} remain",
					n - t
				));
			}
			if tokens[t..t + count].iter().any(|&x| x != image_pad) {
				return Err(format!(
					"prompt: image {si} does not sit on a run of {count} <|image_pad|> tokens"
				));
			}
			let base = next;
			let mut max = base;
			for r in 0..count {
				let (gy, gx) = (r / span.grid_w, r % span.grid_w);
				let p = [base, base + gy as i64, base + gx as i64];
				max = max.max(p[1]).max(p[2]);
				mpos[t + r] = p;
				row_of[t + r] = Some((si, r));
			}
			next = max + 1;
			t += count;
			si += 1;
			continue;
		}
		mpos[t] = [next; 3];
		next += 1;
		t += 1;
	}
	if si != spans.len() {
		return Err("prompt: an image span starts past the end of the prompt".to_string());
	}
	Ok(MultimodalPrompt {
		tokens,
		mpos,
		images: spans,
		row_of,
	})
}
