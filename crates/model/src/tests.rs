//! Unit tests for the gated-delta-net recurrence.

use super::delta_net_head;
use super::prompt::{build_image_prompt, ImageSpan, PromptInput};
use quant::par_for;

// The parallel (per-head `par_for`) SSM scan must produce a bit-identical
// `st`/`o` to a serial `for h` loop. Heads own disjoint slices, so this is
// exact; if the parallelism ever aliased it would show up here.
#[test]
fn ssm_head_parallel_matches_serial() {
	let s = 16usize;
	let hv = 24usize;
	let mut rng = 0x1234_5678u64;
	let mut rnd = || {
		let mut x = rng;
		x = x
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		rng = x;
		((x >> 33) as f32) / (1u32 << 23) as f32 - 1.0
	};
	let mut state = vec![0.0f32; hv * s * s];
	let mut k = vec![0.0f32; hv * s];
	let mut q = vec![0.0f32; hv * s];
	let mut v = vec![0.0f32; hv * s];
	let mut beta = vec![0.0f32; hv];
	let mut g = vec![0.0f32; hv];
	for x in state.iter_mut() {
		*x = rnd();
	}
	for x in k.iter_mut() {
		*x = rnd();
	}
	for x in q.iter_mut() {
		*x = rnd();
	}
	for x in v.iter_mut() {
		*x = rnd();
	}
	for x in beta.iter_mut() {
		*x = rnd();
	}
	for x in g.iter_mut() {
		*x = rnd();
	}
	let qscale = 1.0 / (s as f32).sqrt();

	// Serial reference.
	let mut st_serial = state.clone();
	let mut o_serial = vec![0.0f32; hv * s];
	for h in 0..hv {
		let st_h = &mut st_serial[h * s * s..(h + 1) * s * s];
		let o_h = &mut o_serial[h * s..(h + 1) * s];
		delta_net_head(
			st_h,
			s,
			&k[h * s..(h + 1) * s],
			&q[h * s..(h + 1) * s],
			&v[h * s..(h + 1) * s],
			beta[h],
			g[h],
			qscale,
			o_h,
		);
	}

	// Parallel.
	let mut st_par = state.clone();
	let mut o_par = vec![0.0f32; hv * s];
	let st_ptr = st_par.as_mut_ptr() as usize;
	let st_len = st_par.len();
	let o_ptr = o_par.as_mut_ptr() as usize;
	let o_len = o_par.len();
	par_for(hv, |h| {
		let kvec = &k[h * s..(h + 1) * s];
		let qvec = &q[h * s..(h + 1) * s];
		let vvec = &v[h * s..(h + 1) * s];
		let st_all = unsafe { std::slice::from_raw_parts_mut(st_ptr as *mut f32, st_len) };
		let st_h = &mut st_all[h * s * s..(h + 1) * s * s];
		let o_all = unsafe { std::slice::from_raw_parts_mut(o_ptr as *mut f32, o_len) };
		let o_h = &mut o_all[h * s..(h + 1) * s];
		delta_net_head(st_h, s, kvec, qvec, vvec, beta[h], g[h], qscale, o_h);
	});

	assert_eq!(st_serial.len(), st_par.len());
	for i in 0..st_serial.len() {
		assert_eq!(st_serial[i], st_par[i], "st mismatch at {i}");
	}
	for i in 0..o_serial.len() {
		assert_eq!(o_serial[i], o_par[i], "o mismatch at {i}");
	}
}

// `delta_net_head` must match a from-scratch reference written directly in
// the matrix form (st updated in place, o = stᵀ q), independently of the
// optimized contiguous-index loops above.
#[test]
fn delta_net_head_matches_reference() {
	let s = 8usize;
	let mut rng = 0x9E3779B97F4A7C15u64;
	let mut rnd = || {
		rng = rng
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((rng >> 33) as f32) / (1u32 << 23) as f32 - 1.0
	};
	let mut st = vec![0.0f32; s * s];
	let mut k = vec![0.0f32; s];
	let mut q = vec![0.0f32; s];
	let mut v = vec![0.0f32; s];
	for x in st.iter_mut() {
		*x = rnd();
	}
	for x in k.iter_mut() {
		*x = rnd();
	}
	for x in q.iter_mut() {
		*x = rnd();
	}
	for x in v.iter_mut() {
		*x = rnd();
	}
	let beta = 0.7f32;
	let g = 0.9f32;
	let qscale = 1.0 / (s as f32).sqrt();

	let st0 = st.clone();
	let mut o = vec![0.0f32; s];
	delta_net_head(&mut st, s, &k, &q, &v, beta, g, qscale, &mut o);

	// Reference: st1 = g*st0; sk = st1ᵀ k; d = beta*(v - sk); st1 += k⊗d; o = st1ᵀ q.
	let mut st1 = vec![0.0f32; s * s];
	for i in 0..s {
		for j in 0..s {
			st1[i * s + j] = g * st0[i * s + j];
		}
	}
	let mut sk = vec![0.0f32; s];
	for j in 0..s {
		let mut acc = 0.0f32;
		for i in 0..s {
			acc += st1[i * s + j] * k[i];
		}
		sk[j] = acc;
	}
	let mut d = vec![0.0f32; s];
	for j in 0..s {
		d[j] = beta * (v[j] - sk[j]);
	}
	for i in 0..s {
		for j in 0..s {
			st1[i * s + j] += k[i] * d[j];
		}
	}
	let mut o_ref = vec![0.0f32; s];
	for j in 0..s {
		let mut acc = 0.0f32;
		for i in 0..s {
			acc += st1[i * s + j] * q[i] * qscale;
		}
		o_ref[j] = acc;
	}

	// `delta_net_head` reorders the multiply (e.g. `q[i]*qscale` then `*st`
	// vs `st*q[i]` then `*qscale`); that is only an `f32` rounding
	// difference, not a math difference, so compare with a relative tol.
	let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0);
	for i in 0..s * s {
		assert!(
			close(st[i], st1[i]),
			"st ref mismatch at {i}: {} vs {}",
			st[i],
			st1[i]
		);
	}
	for i in 0..s {
		assert!(
			close(o[i], o_ref[i]),
			"o ref mismatch at {i}: {} vs {}",
			o[i],
			o_ref[i]
		);
	}
}

// ---------------------------------------------------------------------------
// Multimodal prompt assembly (M9)
//
// The MRoPE bookkeeping around an image is the part with no visible failure
// mode: wrong coordinates do not error, they just make the model see the
// picture as a scrambled sequence. These pin the contract down.
// ---------------------------------------------------------------------------

const PAD: u32 = 9;

/// Text before and after an image; the image's patches share one `t` while
/// `h`/`w` sweep the merged grid, and the following text resumes one past the
/// largest coordinate the image used.
#[test]
fn image_positions_are_a_grid_and_text_resumes_after_it() {
	// [a, b, PAD x 6, c] with a 3x2 merged grid.
	let mut tokens = vec![1u32, 2];
	tokens.extend(std::iter::repeat_n(PAD, 6));
	tokens.push(3);
	let span = ImageSpan {
		start: 2,
		grid_w: 3,
		grid_h: 2,
		embeddings: vec![0.0; 6 * 4],
	};
	let mm = build_image_prompt(tokens, PAD, vec![span], 0).unwrap();

	assert_eq!(mm.mpos[0], [0, 0, 0]);
	assert_eq!(mm.mpos[1], [1, 1, 1]);
	// Image starts at t = 2 on every axis, then h/w walk the 3x2 grid.
	let want = [
		[2, 2, 2],
		[2, 2, 3],
		[2, 2, 4],
		[2, 3, 2],
		[2, 3, 3],
		[2, 3, 4],
	];
	for (i, w) in want.iter().enumerate() {
		assert_eq!(mm.mpos[2 + i], *w, "patch {i}");
	}
	// max coordinate used is 4, so the next text token is 5: *not* 8, which
	// is where the sequence position lands.
	assert_eq!(mm.mpos[8], [5, 5, 5]);
	assert!(mm.has_images());
}

/// A continuing conversation starts from the MRoPE clock, not from zero.
#[test]
fn image_prompt_honors_a_non_zero_start_position() {
	let tokens = vec![1u32, PAD, PAD, PAD, PAD, 2];
	let span = ImageSpan {
		start: 1,
		grid_w: 2,
		grid_h: 2,
		embeddings: vec![0.0; 4 * 4],
	};
	let mm = build_image_prompt(tokens, PAD, vec![span], 100).unwrap();
	assert_eq!(mm.mpos[0], [100, 100, 100]);
	assert_eq!(mm.mpos[1], [101, 101, 101]);
	assert_eq!(mm.mpos[4], [101, 102, 102]);
	assert_eq!(mm.mpos[5], [103, 103, 103]);
}

/// A placeholder run that does not match the grid must be an error, not a
/// silent misalignment of every image that follows.
#[test]
fn image_prompt_rejects_a_mismatched_placeholder_run() {
	let tokens = vec![PAD, PAD, 1];
	let span = ImageSpan {
		start: 0,
		grid_w: 2,
		grid_h: 2,
		embeddings: vec![0.0; 16],
	};
	let e = build_image_prompt(tokens, PAD, vec![span], 0).unwrap_err();
	assert!(e.contains("<|image_pad|>") || e.contains("remain"), "{e}");
}

#[test]
fn text_prompt_input_positions_are_degenerate_and_sequential() {
	let tokens = [1u32, 2, 3];
	let input = PromptInput::text(&tokens);
	assert_eq!(input.mpos([7; 3], 0), [7, 7, 7]);
	assert_eq!(input.mpos([7; 3], 2), [9, 9, 9]);
	assert_eq!(input.mpos_after([7; 3], 3), [10; 3]);
	assert!(!input.has_image_positions());
	assert!(input.embed(1).is_none());
}

/// `forward_prefill_mm` chunks the prompt by `n_batch`; the side tables have to
/// slice with it, or an image that straddles a chunk boundary silently gets the
/// wrong rows.
#[test]
fn prompt_input_slices_its_side_tables_with_the_tokens() {
	let tokens = vec![1u32, PAD, PAD, PAD, PAD, 2];
	let span = ImageSpan {
		start: 1,
		grid_w: 2,
		grid_h: 2,
		embeddings: (0..16).map(|i| i as f32).collect(),
	};
	let mm = build_image_prompt(tokens, PAD, vec![span], 0).unwrap();
	let mut scratch = Vec::new();
	let input = mm.input(&mut scratch, 4);
	// Rows are handed out in order, one per placeholder.
	assert_eq!(input.embed(1).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
	assert_eq!(input.embed(4).unwrap(), &[12.0, 13.0, 14.0, 15.0]);
	assert!(input.embed(0).is_none());
	assert!(input.has_image_positions());

	let tail = input.slice(3, 6);
	assert_eq!(tail.tokens, &[PAD, PAD, 2]);
	assert_eq!(tail.embed(0).unwrap(), &[8.0, 9.0, 10.0, 11.0]);
	assert_eq!(tail.mpos([0; 3], 0), [1, 2, 1]);
	assert!(tail.embed(2).is_none());
}
