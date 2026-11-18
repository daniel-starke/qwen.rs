//! The two-half prefill pipeline (`QWEN_PIPE`, default on).
//!
//! A prefill layer alternates GEMMs, which the row split runs on the GPU and
//! the CPU together, with per-position CPU work: the SSM passes or the
//! attention, the residual adds, the norms, SwiGLU. During that CPU work the
//! GPU had nothing to do: per 2048-token chunk ~3 s of a ~16 s chunk, while
//! the CPU was busy ~97 % of the time (`QWEN_GPU_TIMING=1`).
//!
//! Here the chunk's positions are split into halves A and B and each layer is
//! run as five GEMM stages, each followed by its CPU phase:
//!
//! | stage | SSM layer | full-attention layer |
//! |---|---|---|
//! | 0 | `Wqkv` (+ beta / alpha) -> passes 1-2 | query rows of `Wq`, `Wk`, `Wv` -> attention |
//! | 1 | `WqkvGate` -> pass 3, q8_K | gate rows of `Wq` -> gate, q8_K |
//! | 2 | `SsmOut` -> residual, `post_norm` | `Wo` -> residual, `post_norm` |
//! | 3 | `FfnUp` + `FfnGate` -> SwiGLU | same |
//! | 4 | `FfnDown` -> residual, next `attn_norm` | same |
//!
//! staggered so that every CPU phase has a GEMM on the GPU behind it:
//!
//! ```text
//! begin G_k(A) | phase_{k-1}(B) | finish G_k(A)
//! begin G_k(B) | phase_k(A)     | finish G_k(B)
//! ```
//!
//! A `finish` runs the call's CPU rows and then waits for its GPU rows
//! (`OffloadManager::split_begin` / `split_finish`), so each GPU share is
//! sized to cover the other half's phase plus its own CPU rows: the prefill
//! split's fraction plus a per-stage, per-half `extra`. The order keeps every
//! sequential dependency: A's SSM recurrence and KV append for a layer always
//! precede B's. A phase that needs the GPU itself (the GPU attention split
//! at long contexts) runs between batches instead.
//!
//! **Two batches in flight** (`QWEN_PIPE_OVERLAP`, default on). In the order
//! above the GPU idled from the end of each batch until the host had
//! collected it and staged the next (the q8_K copy into the staging map,
//! stream bind, recording): measured with GPU timestamps, 8.6 % of a
//! 2048-token chunk and 10.5 % of a 1218-token one. So B's stage k is begun
//! before A's is finished, and A's stage k + 1 before B's:
//!
//! ```text
//! phase_{k-1}(B) | begin G_k(B)     | finish G_k(A)
//! phase_k(A)     | begin G_{k+1}(A) | finish G_k(B)
//! ```
//!
//! The two halves use separate banks of the GPU batch machinery (fence,
//! host-pool slots, stream slot pins: `matmul_batch_seq_begin_bank`), A's
//! batches bank 0, B's bank 1. The rows each call puts on the GPU are
//! unchanged, so the output is bit-identical to `QWEN_PIPE_OVERLAP=0`.
//!
//! Per position the arithmetic is that of `prefill_chunk`; which rows run on
//! the GPU moves with the fractions, so output is not bit-identical to the
//! unpipelined path.

#![allow(non_snake_case)]

use super::*;

/// Shortest chunk the pipeline takes (`QWEN_PIPE_MIN`, default 960): each
/// half must stay long enough for the streamed split rows to pay.
fn pipe_min() -> Option<usize> {
	static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
	*V.get_or_init(|| {
		if matches!(
			std::env::var("QWEN_PIPE").as_deref().map(str::trim),
			Ok("0") | Ok("off") | Ok("no") | Ok("false")
		) {
			return None;
		}
		Some(
			std::env::var("QWEN_PIPE_MIN")
				.ok()
				.and_then(|v| v.trim().parse().ok())
				.unwrap_or(512),
		)
	})
}

/// Keep two GEMM batches in flight (`QWEN_PIPE_OVERLAP`, default on): `0`
/// collects each batch before the next is submitted, as before.
fn pipe_overlap() -> bool {
	static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*V.get_or_init(|| {
		!matches!(
			std::env::var("QWEN_PIPE_OVERLAP").as_deref().map(str::trim),
			Ok("0") | Ok("off") | Ok("no") | Ok("false")
		)
	})
}

/// Extra GPU fraction of stage `j` for half A (`b = false`, overlapping B's
/// previous phase) or B (overlapping A's phase of the same stage):
/// `QWEN_PIPE_A<j>` / `QWEN_PIPE_B<j>`, and for stage 0 of a full-attention
/// layer `QWEN_PIPE_FA0` / `QWEN_PIPE_FB0` (B's overlaps A's attention).
fn pipe_extra(j: usize, b: bool, full: bool) -> f64 {
	// A: [B's s4, B's s0, B's s1, B's s2, B's s3]; B: [A's s0 .. A's s4].
	const DA: [f64; 6] = [0.035, 0.11, 0.03, 0.0, 0.005, 0.05];
	const DB: [f64; 6] = [0.075, -0.02, 0.03, 0.005, 0.0, 0.22];
	static V: std::sync::OnceLock<[[f64; 6]; 2]> = std::sync::OnceLock::new();
	let v = V.get_or_init(|| {
		let mut v = [DA, DB];
		for (h, c) in [(0, 'A'), (1, 'B')] {
			for (j, x) in v[h].iter_mut().enumerate() {
				let k = if j == 5 {
					format!("QWEN_PIPE_F{c}0")
				} else {
					format!("QWEN_PIPE_{c}{j}")
				};
				if let Some(f) = std::env::var(k).ok().and_then(|s| s.trim().parse().ok()) {
					*x = f;
				}
			}
		}
		v
	});
	v[b as usize][if full && j == 0 { 5 } else { j }]
}

/// GPU fraction of the attention gate rows (stage 1 of a full-attention
/// layer), all streamed: `QWEN_PIPE_QGATE_A` / `_B`.
fn qgate_frac(b: bool) -> f64 {
	static V: std::sync::OnceLock<[f64; 2]> = std::sync::OnceLock::new();
	V.get_or_init(|| {
		let g = |k: &str, d: f64| {
			std::env::var(k)
				.ok()
				.and_then(|s| s.trim().parse().ok())
				.unwrap_or(d)
		};
		[g("QWEN_PIPE_QGATE_A", 1.0), g("QWEN_PIPE_QGATE_B", 0.7)]
	})[b as usize]
}

/// One position half.
struct Half {
	/// First position (chunk-relative) and length.
	t0: usize,
	n: usize,
	/// `attn_norm` of the current layer, read by stages 0 and 1.
	norm: Option<Arc<Vec<quant::Q8K>>>,
	/// Input of stages 2-4.
	q8: Option<Arc<Vec<quant::Q8K>>>,
	/// Outputs of the last finished GEMM stage.
	res: Vec<Vec<f32>>,
	/// Stage-0 phase output: the SSM recurrence output `o`, or the ungated
	/// attention output.
	mid: Vec<f32>,
}

impl Model {
	/// Whether [`Self::layers_pipe`] takes a chunk of `n` tokens.
	pub(crate) fn pipe_ok(&self, n: usize) -> bool {
		let (Some(min), Some(off)) = (pipe_min(), self.offload.as_ref()) else {
			return false;
		};
		if n < min.max(2) {
			return false;
		}
		// Every stage of the layer stack takes the split at both half lengths
		// (layer 0's FFN stands in for all: one weight set, same gates).
		let lt = self.get_layer(0);
		let jobs = [
			(WeightId::FfnUp, &lt.ffn_up, self.hp.n_ff),
			(WeightId::FfnGate, &lt.ffn_gate, self.hp.n_ff),
		];
		let na = n / 2;
		off.prefill_q8_ok(0, &jobs, na, self.hp.n_embd)
			&& off.prefill_q8_ok(0, &jobs, n - na, self.hp.n_embd)
	}

	/// The layer loop of [`Self::prefill_chunk`] as the two-half pipeline
	/// (module docs). `xmat` holds the chunk's embeddings and receives its
	/// final hidden states.
	pub(crate) fn layers_pipe(
		&self,
		rs: &mut RunState,
		input: &PromptInput<'_>,
		n_layers: usize,
		xmat: &mut [f32],
		N: usize,
	) {
		let measure = phase_measure();
		let n_embd = self.hp.n_embd;
		let na = N / 2;
		let mut h = [
			Half {
				t0: 0,
				n: na,
				norm: None,
				q8: None,
				res: Vec::new(),
				mid: Vec::new(),
			},
			Half {
				t0: na,
				n: N - na,
				norm: None,
				q8: None,
				res: Vec::new(),
				mid: Vec::new(),
			},
		];
		let lts: Vec<Arc<LayerTensors>> = (0..n_layers).map(|il| self.get_layer(il)).collect();
		let base = rs.mpos;
		let mut ph_ns = [0u64; 6];
		let mut fin_ns = [0u64; 2];
		let mut beg_ns = 0u64;
		let mut phase = |m: &Model,
		                 k: usize,
		                 hh: &mut Half,
		                 rs: &mut RunState,
		                 xmat: &mut [f32],
		                 last: bool| {
			let t = if measure { Some(Instant::now()) } else { None };
			match k {
				usize::MAX => hh.norm = Some(m.pipe_norm(&lts[0].attn_norm, xmat, hh)),
				k => m.pipe_phase(&lts, k / 5, k % 5, hh, rs, xmat, input, base, last),
			}
			if let Some(t) = t {
				ph_ns[if k == usize::MAX { 5 } else { k % 5 }] += t.elapsed().as_nanos() as u64;
			}
		};
		let off = self
			.offload
			.as_ref()
			.expect("pipe_ok requires the offload manager");
		let mut finish = |d: offload::Deferred, b: usize| {
			let t = if measure { Some(Instant::now()) } else { None };
			let r = off.split_finish(d);
			if let Some(t) = t {
				fin_ns[b] += t.elapsed().as_nanos() as u64;
			}
			r
		};
		// Stage-0 phase of a full-attention layer runs between batches when
		// the GPU takes part of the attention.
		let excl = |m: &Model, k: usize, hh: &Half, rs: &RunState| -> bool {
			let il = k / 5;
			k != usize::MAX && k.is_multiple_of(5) && m.full_attn_layers[il] && {
				let (hd, nh, nkv) = (m.hp.head_dim, m.hp.n_head, m.hp.n_head_kv);
				let keys = rs.kv[il]
					.as_ref()
					.map_or(0, |(kc, _)| kc.len() / (hd * nkv))
					+ hh.n;
				off.attn_prefill_on(hh.n, keys, nh, nkv, hd)
			}
		};

		let [a, b] = &mut h;
		phase(self, usize::MAX, a, rs, xmat, false);
		let nk = n_layers * 5;
		if pipe_overlap() {
			// Two batches in flight: B's stage k is queued behind A's before
			// A's is collected, and A's stage k + 1 behind B's, so the GPU
			// goes from one to the next without waiting for the host. A phase
			// that needs the GPU itself runs with nothing in flight.
			let mut begin = |m: &Model, k: usize, hh: &Half, bh: bool| {
				let t = if measure { Some(Instant::now()) } else { None };
				let d = m.pipe_begin(&lts, k / 5, k % 5, hh, bh);
				if let Some(t) = t {
					beg_ns += t.elapsed().as_nanos() as u64;
				}
				d
			};
			let mut da = Some(begin(self, 0, a, false));
			for k in 0..nk {
				quant::progress::part(k, nk);
				let prev = if k == 0 { usize::MAX } else { k - 1 };
				let db = if excl(self, prev, b, rs) {
					a.res = finish(da.take().unwrap(), 0);
					phase(self, prev, b, rs, xmat, true);
					begin(self, k, b, true)
				} else {
					phase(self, prev, b, rs, xmat, true);
					let db = begin(self, k, b, true);
					a.res = finish(da.take().unwrap(), 0);
					db
				};
				if excl(self, k, a, rs) {
					b.res = finish(db, 1);
					phase(self, k, a, rs, xmat, false);
					if k + 1 < nk {
						da = Some(begin(self, k + 1, a, false));
					}
				} else {
					phase(self, k, a, rs, xmat, false);
					if k + 1 < nk {
						da = Some(begin(self, k + 1, a, false));
					}
					b.res = finish(db, 1);
				}
			}
		} else {
			for k in 0..nk {
				quant::progress::part(k, nk);
				let prev = if k == 0 { usize::MAX } else { k - 1 };
				let tbg = if measure { Some(Instant::now()) } else { None };
				let da = self.pipe_begin(&lts, k / 5, k % 5, a, false);
				if let Some(t) = tbg {
					beg_ns += t.elapsed().as_nanos() as u64;
				}
				let xb = excl(self, prev, b, rs);
				if !xb {
					phase(self, prev, b, rs, xmat, true);
				}
				a.res = finish(da, 0);
				if xb {
					phase(self, prev, b, rs, xmat, true);
				}
				let tbg = if measure { Some(Instant::now()) } else { None };
				let db = self.pipe_begin(&lts, k / 5, k % 5, b, true);
				if let Some(t) = tbg {
					beg_ns += t.elapsed().as_nanos() as u64;
				}
				let xa = excl(self, k, a, rs);
				if !xa {
					phase(self, k, a, rs, xmat, false);
				}
				b.res = finish(db, 1);
				if xa {
					phase(self, k, a, rs, xmat, false);
				}
			}
		}
		phase(self, nk - 1, b, rs, xmat, true);
		if measure {
			let ms = |v: u64| v as f64 / 1e6;
			eprintln!(
				"PIPE N={N} phase_ms s0={:.1} s1={:.1} s2={:.1} s3={:.1} s4={:.1} init={:.1}  finish_ms A={:.1} B={:.1}  begin_ms {:.1}",
				ms(ph_ns[0]), ms(ph_ns[1]), ms(ph_ns[2]), ms(ph_ns[3]), ms(ph_ns[4]), ms(ph_ns[5]),
				ms(fin_ns[0]), ms(fin_ns[1]), ms(beg_ns),
			);
		}
		let _ = n_embd;
	}

	/// RMS norm of a half's rows of `xmat`, straight to q8_K.
	fn pipe_norm(&self, w: &[f32], xmat: &[f32], hh: &Half) -> Arc<Vec<quant::Q8K>> {
		let (n_embd, eps) = (self.hp.n_embd, self.hp.eps);
		let x = &xmat[hh.t0 * n_embd..(hh.t0 + hh.n) * n_embd];
		Arc::new(quant::par_rows_q8k(hh.n, n_embd, |t, r| {
			rms_norm_into(&x[t * n_embd..(t + 1) * n_embd], w, eps, r)
		}))
	}

	/// Start stage `j` of layer `il` for half `hh` (`b`: it is half B).
	fn pipe_begin<'a>(
		&self,
		lts: &'a [Arc<LayerTensors>],
		il: usize,
		j: usize,
		hh: &Half,
		b: bool,
	) -> offload::Deferred<'a> {
		let off = self.offload.as_ref().unwrap();
		let lt = &*lts[il];
		let (n_embd, n_ff) = (self.hp.n_embd, self.hp.n_ff);
		let extra = pipe_extra(j, b, self.full_attn_layers[il]);
		// Half A's batches are bank 0, B's bank 1: one of each in flight.
		let bank = b as usize;
		let n = hh.n;
		let norm = || hh.norm.as_ref().expect("attn_norm before stages 0-1");
		let q8 = || hh.q8.as_ref().expect("phase output before stages 2-4");
		if self.full_attn_layers[il] {
			let n_q = self.hp.head_dim * self.hp.n_head;
			let wq = lt.wq.as_ref().unwrap();
			match j {
				0 => {
					let mut jobs = self.attn_qkv_jobs(lt);
					jobs[0].2 = n_q;
					off.split_begin(il, &jobs, norm(), n, n_embd, extra, bank)
				}
				1 => off.split_begin_rows(
					il,
					WeightId::Wq,
					wq,
					n_q,
					n_q,
					norm(),
					n,
					n_embd,
					qgate_frac(b),
					bank,
				),
				2 => off.split_begin(
					il,
					&[(WeightId::Wo, lt.wo.as_ref().unwrap(), n_embd)],
					q8(),
					n,
					n_q,
					extra,
					bank,
				),
				3 => off.split_begin(
					il,
					&[
						(WeightId::FfnUp, &lt.ffn_up, n_ff),
						(WeightId::FfnGate, &lt.ffn_gate, n_ff),
					],
					q8(),
					n,
					n_embd,
					extra,
					bank,
				),
				_ => off.split_begin(
					il,
					&[(WeightId::FfnDown, &lt.ffn_down, n_embd)],
					q8(),
					n,
					n_ff,
					extra,
					bank,
				),
			}
		} else {
			let (Hv, vd) = (self.hp.ssm_dt_rank, self.hp.ssm_inner);
			let conv_dim = self.hp.ssm_state * self.hp.ssm_n_group * 2 + vd;
			match j {
				0 => off.split_begin(
					il,
					&[
						(WeightId::Wqkv, lt.wqkv.as_ref().unwrap(), conv_dim),
						(WeightId::WssmBeta, lt.ssm_beta.as_ref().unwrap(), Hv),
						(WeightId::WssmAlpha, lt.ssm_alpha.as_ref().unwrap(), Hv),
					],
					norm(),
					n,
					n_embd,
					extra,
					bank,
				),
				1 => off.split_begin(
					il,
					&[(WeightId::WqkvGate, lt.wqkv_gate.as_ref().unwrap(), vd)],
					norm(),
					n,
					n_embd,
					extra,
					bank,
				),
				2 => off.split_begin(
					il,
					&[(WeightId::SsmOut, lt.ssm_out.as_ref().unwrap(), n_embd)],
					q8(),
					n,
					vd,
					extra,
					bank,
				),
				3 => off.split_begin(
					il,
					&[
						(WeightId::FfnUp, &lt.ffn_up, n_ff),
						(WeightId::FfnGate, &lt.ffn_gate, n_ff),
					],
					q8(),
					n,
					n_embd,
					extra,
					bank,
				),
				_ => off.split_begin(
					il,
					&[(WeightId::FfnDown, &lt.ffn_down, n_embd)],
					q8(),
					n,
					n_ff,
					extra,
					bank,
				),
			}
		}
	}

	/// The CPU phase after stage `j` of layer `il` for half `hh`, on the
	/// outputs in `hh.res`. `last`: `hh` is half B, so the layer's output is
	/// complete after its stage-4 phase.
	#[allow(clippy::too_many_arguments)]
	fn pipe_phase(
		&self,
		lts: &[Arc<LayerTensors>],
		il: usize,
		j: usize,
		hh: &mut Half,
		rs: &mut RunState,
		xmat: &mut [f32],
		input: &PromptInput<'_>,
		base: [i64; 3],
		last: bool,
	) {
		let lt = &*lts[il];
		let (n_embd, eps) = (self.hp.n_embd, self.hp.eps);
		let n = hh.n;
		let res = std::mem::take(&mut hh.res);
		let residual = |xmat: &mut [f32], add: &[f32]| {
			let x = &mut xmat[hh.t0 * n_embd..(hh.t0 + n) * n_embd];
			let (xp, ap) = (x.as_mut_ptr() as usize, add.as_ptr() as usize);
			quant::par_for(n, |t| unsafe {
				let x = std::slice::from_raw_parts_mut((xp as *mut f32).add(t * n_embd), n_embd);
				let a = std::slice::from_raw_parts((ap as *const f32).add(t * n_embd), n_embd);
				for d in 0..n_embd {
					x[d] += a[d];
				}
			});
		};
		let full = self.full_attn_layers[il];
		match j {
			0 if full => {
				let n_q = self.hp.head_dim * self.hp.n_head;
				let t0 = hh.t0;
				let mpos = |t: usize| input.mpos(base, t0 + t);
				hh.mid = self.attn_core(
					il,
					lt,
					&mut rs.kv[il],
					&res[0],
					n_q,
					&res[1],
					&res[2],
					n,
					&mpos,
				);
			}
			0 => {
				hh.mid = self.ssm_pass12(il, lt, rs, &res[0], &res[1], &res[2], n, None);
			}
			1 if full => {
				let n_q = self.hp.head_dim * self.hp.n_head;
				let (mid, gate) = (&hh.mid, &res[0]);
				hh.q8 = Some(Arc::new(quant::par_rows_q8k(n, n_q, |t, r| {
					for ((r, o), g) in r
						.iter_mut()
						.zip(&mid[t * n_q..(t + 1) * n_q])
						.zip(&gate[t * n_q..(t + 1) * n_q])
					{
						*r = *o * sigmoid(*g);
					}
				})));
				hh.mid = Vec::new();
			}
			1 => {
				let (S, Hv, vd) = (self.hp.ssm_state, self.hp.ssm_dt_rank, self.hp.ssm_inner);
				let ssm_norm = lt.ssm_norm.as_ref().unwrap();
				let (o_b, z) = (&hh.mid, &res[0]);
				hh.q8 = Some(Arc::new(quant::par_rows_q8k(n, vd, |t, r| {
					let z = &z[t * vd..(t + 1) * vd];
					for h in 0..Hv {
						let o = &o_b[(h * n + t) * S..(h * n + t + 1) * S];
						let r = &mut r[h * S..(h + 1) * S];
						rms_norm_into(o, ssm_norm, eps, r);
						quant::silu_mul_assign(&z[h * S..(h + 1) * S], r);
					}
				})));
				hh.mid = Vec::new();
			}
			2 => {
				residual(xmat, &res[0]);
				hh.q8 = Some(self.pipe_norm(&lt.post_norm, xmat, hh));
			}
			3 => {
				hh.q8 = Some(Arc::new(quant::par_silu_mul_q8k(
					&res[1],
					&res[0],
					n,
					self.hp.n_ff,
				)));
			}
			_ => {
				residual(xmat, &res[0]);
				hh.q8 = None;
				if last {
					let rows = hh.t0 + n;
					crate::forward::feature_dump(il, &xmat[..rows * n_embd]);
					self.dflash_capture(rs, il, &xmat[..rows * n_embd], rows);
				}
				hh.norm =
					(il + 1 < lts.len()).then(|| self.pipe_norm(&lts[il + 1].attn_norm, xmat, hh));
			}
		}
	}
}
