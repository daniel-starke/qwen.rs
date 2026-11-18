//! `Vec<f32>` CPU forward path (symmetric with the `Vulkan` GPU offload path).
//!
//! Numerically identical to the batched `prefill_chunk` path: every
//! scalar op (norms, RoPE, attention, SSM scan, silu/sigmoid/softplus) and
//! every matmul reuses the exact same helpers / `quant::gemm_f16` kernel. The
//! activation is a plain `Vec<f32>`; the two residual connections are plain
//! element-wise `Vec` adds.

// The SSM/delta-net dimension names below (`S`, `Hk`, `Hv`, `N`, `Tprev`) are
// the upper-case symbols used by the reference gated-delta-net formulation; the
// math is far easier to follow against llama.cpp when they keep that spelling.
#![allow(non_snake_case)]

use super::*;

impl Model {
	pub fn forward_step(
		&self,
		rs: &mut RunState,
		token: u32,
		max_layers: Option<usize>,
	) -> (Vec<f32>, Vec<f32>) {
		self.heal_offload();
		let dbg = std::env::var("QWEN_DEBUG").is_ok();
		let measure = phase_measure();
		if measure {
			MATMUL_NS.with(|c| c.set(0));
			LMHEAD_NS.with(|c| c.set(0));
			SSN_NS.with(|c| c.set(0));
			LAYER_LOOP_NS.with(|c| c.set(0));
			ELEM_NS.with(|c| c.set(0));
			PREP_NS.with(|c| c.set(0));
		}
		let t0 = if measure { Some(Instant::now()) } else { None };
		let S = self.hp.ssm_state;
		let Hk = self.hp.ssm_n_group;
		let Hv = self.hp.ssm_dt_rank;
		let value_dim = self.hp.ssm_inner;
		let key_dim = S * Hk;
		let conv_dim = key_dim * 2 + value_dim;

		let emb = self.embedding(token);
		let mut x: Vec<f32> = emb.clone();
		let dbg_norm = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
		if dbg {
			eprintln!(
				"DBG embed step={} token={} norm={:.4}",
				rs.pos,
				token,
				dbg_norm(&emb)
			);
		}

		let n_layers = max_layers.unwrap_or(self.hp.n_layer_main);
		let t_loop = if measure { Some(Instant::now()) } else { None };
		let dump_inp = std::env::var("QWEN_LAYERINP").is_ok();
		for il in 0..n_layers {
			let lt = self.get_layer(il);
			let t_flex = if measure { Some(Instant::now()) } else { None };
			let xs = x.clone();
			if let Some(t_flex) = t_flex {
				FLEXOP_NS.with(|c| c.set(c.get() + t_flex.elapsed().as_nanos() as u64));
			}
			if dump_inp && rs.pos == 0 {
				let p = format!("/tmp/kilo/qwen_layinp_{}.bin", il);
				if let Ok(mut f) = std::fs::File::create(&p) {
					use std::io::Write;
					let _ = f.write_all(unsafe {
						std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4)
					});
				}
			}
			let is_full = self.full_attn_layers[il];
			let mut gpu_layer_out: Option<Vec<f32>> = None;
			let eps = self.hp.eps;
			let head_dim = self.hp.head_dim;
			let n_head = self.hp.n_head;
			let n_head_kv = self.hp.n_head_kv;

			// L2/L3: route the whole layer to the GPU-resident fused block (keeps
			// `x` in VRAM across the layer). With QWEN_GPU_LAYER every offloaded
			// layer (full-attn or SSM) runs entirely on the GPU, so the
			// contiguous trailing block stays GPU-resident (one H2D + one D2H per
			// layer instead of one per matmul). Falls through to the verified CPU
			// path if the device is unavailable/lost.
			// The fused block's shader takes a single scalar position, so it
			// can only do the degenerate (text) MRoPE. Once an image is in the
			// context that is wrong, and silently so: fall back to the CPU
			// path rather than rotate the wrong channels.
			if std::env::var("QWEN_GPU_LAYER").is_ok()
				&& !rs.mrope && self.offload.as_ref().is_some_and(|o| o.is_offloaded(il))
			{
				let half_nrot = self.hp.n_rot / 2;
				let sect_dims = (self.hp.rope_sections[0]
					+ self.hp.rope_sections[1]
					+ self.hp.rope_sections[2]
					+ self.hp.rope_sections[3]) as f32;
				let info = GpuBlockInfo {
					n_embd: self.hp.n_embd,
					n_head: self.hp.n_head,
					n_head_kv: self.hp.n_head_kv,
					head_dim: self.hp.head_dim,
					n_ff: self.hp.n_ff,
					eps: self.hp.eps,
					pos: rs.pos,
					half_nrot,
					rope_freq_base: self.hp.rope_freq_base as f32,
					rope_freq_scale: self.hp.rope_freq_scale as f32,
					rope_attn_factor: self.hp.rope_attn_factor as f32,
					sect_dims,
					attn_norm: &lt.attn_norm,
					post_norm: &lt.post_norm,
					q_norm: lt.q_norm.as_ref().map(|v| &v[..]).unwrap_or(&[]),
					k_norm: lt.k_norm.as_ref().map(|v| &v[..]).unwrap_or(&[]),
					ssm_state: self.hp.ssm_state,
					ssm_n_group: self.hp.ssm_n_group,
					ssm_dt_rank: self.hp.ssm_dt_rank,
					conv_kernel: self.hp.conv_kernel,
					ssm_conv1d: lt.conv1d.as_ref().map(|v| &v[..]).unwrap_or(&[]),
					ssm_dt: lt.ssm_dt.as_ref().map(|v| &v[..]).unwrap_or(&[]),
					ssm_a: lt.ssm_a.as_ref().map(|v| &v[..]).unwrap_or(&[]),
					ssm_norm: lt.ssm_norm.as_ref().map(|v| &v[..]).unwrap_or(&[]),
				};
				let out = if is_full {
					self.offload
						.as_ref()
						.unwrap()
						.full_attn_block(il, &x, &info)
				} else {
					self.offload.as_ref().unwrap().ssm_block(il, &x, &info)
				};
				if let Some(out) = out {
					// QWEN_GPU_LAYERCMP: keep the fused output aside, run the CPU
					// path for the same layer anyway and report the divergence
					// (the CPU value is what propagates, so every layer is
					// measured against the reference rather than against drift).
					if std::env::var("QWEN_GPU_LAYERCMP").is_ok() {
						gpu_layer_out = Some(out);
					} else {
						x = out;
						continue;
					}
				}
				// GPU unavailable/failed: fall through to the verified CPU path.
			}

			let xin = &xs;
			let mut norm = rms_norm(xin, &lt.attn_norm, eps);
			sparse::sparsify(sparse::Site::AttnIn, &mut norm);

			let t_aout = if measure { Some(Instant::now()) } else { None };
			let aout = if is_full {
				// Wq/Wk/Wv consume the same `norm` and are independent: batch
				// them (one GPU submission, overlapped H2D/D2H) when offloaded.
				let attn = self.lin_batch(&[
					BatchJob {
						il,
						id: WeightId::Wq,
						w: lt.wq.as_ref().unwrap(),
						x: &norm,
						out_dim: 2 * head_dim * n_head,
						in_dim: self.hp.n_embd,
					},
					BatchJob {
						il,
						id: WeightId::Wk,
						w: lt.wk.as_ref().unwrap(),
						x: &norm,
						out_dim: head_dim * n_head_kv,
						in_dim: self.hp.n_embd,
					},
					BatchJob {
						il,
						id: WeightId::Wv,
						w: lt.wv.as_ref().unwrap(),
						x: &norm,
						out_dim: head_dim * n_head_kv,
						in_dim: self.hp.n_embd,
					},
				]);
				let qfull = &attn[0];
				// Every head's query, then every head's gate (`q_gate_rows_split`).
				let (qcur, gate) = qfull.split_at(head_dim * n_head);
				let mut qn = vec![0.0f32; head_dim * n_head];
				for h in 0..n_head {
					let slice = &qcur[h * head_dim..(h + 1) * head_dim];
					let n = rms_norm(slice, lt.q_norm.as_ref().unwrap(), eps);
					qn[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
				}
				let k = &attn[1];
				let v = &attn[2];
				let mut kn = vec![0.0f32; head_dim * n_head_kv];
				for h in 0..n_head_kv {
					let slice = &k[h * head_dim..(h + 1) * head_dim];
					let n = rms_norm(slice, lt.k_norm.as_ref().unwrap(), eps);
					kn[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
				}
				let p = rs.mpos;
				for h in 0..n_head {
					rope_apply(&mut qn[h * head_dim..(h + 1) * head_dim], p, &self.hp);
				}
				for h in 0..n_head_kv {
					rope_apply(&mut kn[h * head_dim..(h + 1) * head_dim], p, &self.hp);
				}

				let (kcache, vcache) = match &mut rs.kv[il] {
					Some(c) => c,
					None => {
						rs.kv[il] = Some((vec![], vec![]));
						rs.kv[il].as_mut().unwrap()
					}
				};
				quant::kv_extend(kcache, &kn);
				quant::kv_extend(vcache, v);

				let tseen = kcache.len() / (head_dim * n_head_kv);
				let scale = 1.0f32 / (head_dim as f32).sqrt();

				let mut out = vec![0.0f32; head_dim * n_head];
				// One query row per head; the kernel splits the keys across
				// the pool and reads each K/V row once per KV group.
				quant::attn_causal(
					&qn,
					kcache,
					vcache,
					1,
					tseen - 1,
					n_head,
					n_head_kv,
					head_dim,
					scale,
					&mut out,
				);
				for i in 0..head_dim * n_head {
					out[i] *= sigmoid(gate[i]);
				}
				sparse::sparsify(sparse::Site::AttnOut, &mut out);
				self.lin(
					il,
					WeightId::Wo,
					lt.wo.as_ref().unwrap(),
					&out,
					self.hp.n_embd,
					head_dim * n_head,
				)
			} else {
				// Wqkv and WqkvGate both consume `norm`; batch when offloaded.
				let qg = self.lin_batch(&[
					BatchJob {
						il,
						id: WeightId::Wqkv,
						w: lt.wqkv.as_ref().unwrap(),
						x: &norm,
						out_dim: conv_dim,
						in_dim: self.hp.n_embd,
					},
					BatchJob {
						il,
						id: WeightId::WqkvGate,
						w: lt.wqkv_gate.as_ref().unwrap(),
						x: &norm,
						out_dim: value_dim,
						in_dim: self.hp.n_embd,
					},
				]);
				let qkv = &qg[0];
				let mut input = vec![0.0f32; self.hp.conv_kernel * conv_dim];
				let prev = &rs.convbuf[il];
				for f in 0..self.hp.conv_kernel - 1 {
					input[f * conv_dim..(f + 1) * conv_dim].copy_from_slice(&prev[f]);
				}
				input[(self.hp.conv_kernel - 1) * conv_dim..].copy_from_slice(qkv);
				let mut conv_out = vec![0.0f32; conv_dim];
				let rows: Vec<&[f32]> = input.chunks_exact(conv_dim).collect();
				quant::conv_silu(&rows, lt.conv1d_t.as_ref().unwrap(), &mut conv_out);
				for f in 0..self.hp.conv_kernel - 1 {
					rs.convbuf[il][f]
						.copy_from_slice(&input[(f + 1) * conv_dim..(f + 2) * conv_dim]);
				}
				let t_prep = if measure { Some(Instant::now()) } else { None };
				let mut q = vec![0.0f32; key_dim];
				let mut k = vec![0.0f32; key_dim];
				let mut v = vec![0.0f32; value_dim];
				q.copy_from_slice(&conv_out[0..key_dim]);
				k.copy_from_slice(&conv_out[key_dim..2 * key_dim]);
				v.copy_from_slice(&conv_out[2 * key_dim..2 * key_dim + value_dim]);
				let mut qn = vec![0.0f32; key_dim];
				let mut kn = vec![0.0f32; key_dim];
				for h in 0..Hk {
					let qs = l2_norm(&q[h * S..(h + 1) * S], eps);
					let ks = l2_norm(&k[h * S..(h + 1) * S], eps);
					qn[h * S..(h + 1) * S].copy_from_slice(&qs);
					kn[h * S..(h + 1) * S].copy_from_slice(&ks);
				}
				if let Some(t) = t_prep {
					PREP_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
				}

				// Both consume `norm` and are the same shape: one dispatch.
				let ba = self.lin_batch(&[
					BatchJob {
						il,
						id: WeightId::WssmBeta,
						w: lt.ssm_beta.as_ref().unwrap(),
						x: &norm,
						out_dim: Hv,
						in_dim: self.hp.n_embd,
					},
					BatchJob {
						il,
						id: WeightId::WssmAlpha,
						w: lt.ssm_alpha.as_ref().unwrap(),
						x: &norm,
						out_dim: Hv,
						in_dim: self.hp.n_embd,
					},
				]);
				let (beta, alpha) = (&ba[0], &ba[1]);
				let mut beta_s = vec![0.0f32; Hv];
				for h in 0..Hv {
					beta_s[h] = sigmoid(beta[h]);
				}
				let ssm_dt = &lt.ssm_dt.as_ref().unwrap();
				let ssm_a = &lt.ssm_a.as_ref().unwrap();
				let mut gate_h = vec![0.0f32; Hv];
				for h in 0..Hv {
					let ab = alpha[h] + ssm_dt[h];
					let sp = softplus(ab);
					gate_h[h] = (sp * ssm_a[h]).exp();
				}

				let t_ssn = if measure { Some(Instant::now()) } else { None };
				let mut st = std::mem::take(&mut rs.state[il]);
				let mut o = vec![0.0f32; value_dim];
				let qscale = 1.0f32 / (S as f32).sqrt();
				// Independent SSM heads: each owns a disjoint `S*S` slice of
				// `st` and `S` slice of `o`, so they run concurrently across
				// the thread pool. Result is
				// bit-identical to the serial `for h` below. Pointers (not
				// `&mut`) are captured so the closure stays `Fn` (par_for
				// calls it from many threads); head slices are disjoint.
				let st_ptr = st.as_mut_ptr() as usize;
				let st_len = st.len();
				let o_ptr = o.as_mut_ptr() as usize;
				let o_len = o.len();
				quant::par_for(Hv, |h| {
					let kh = h % Hk;
					let g = gate_h[h];
					let beta = beta_s[h];
					let kvec = &kn[kh * S..(kh + 1) * S];
					let qvec = &qn[kh * S..(kh + 1) * S];
					let vvec = &v[h * S..(h + 1) * S];
					let st_all =
						unsafe { std::slice::from_raw_parts_mut(st_ptr as *mut f32, st_len) };
					let st_h = &mut st_all[h * S * S..(h + 1) * S * S];
					let o_all = unsafe { std::slice::from_raw_parts_mut(o_ptr as *mut f32, o_len) };
					let o_h = &mut o_all[h * S..(h + 1) * S];
					delta_net_head(st_h, S, kvec, qvec, vvec, beta, g, qscale, o_h);
				});
				rs.state[il] = st;
				if let Some(t_ssn) = t_ssn {
					SSN_NS.with(|c| c.set(c.get() + t_ssn.elapsed().as_nanos() as u64));
				}

				let z = &qg[1];
				let ssm_norm = &lt.ssm_norm.as_ref().unwrap();
				let mut no = vec![0.0f32; value_dim];
				let t_el2 = if measure { Some(Instant::now()) } else { None };
				for h in 0..Hv {
					let slice = &o[h * S..(h + 1) * S];
					let n = rms_norm(slice, ssm_norm, eps);
					quant::silu_mul(&z[h * S..(h + 1) * S], &n, &mut no[h * S..(h + 1) * S]);
				}
				if let Some(t) = t_el2 {
					ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
				}
				sparse::sparsify(sparse::Site::AttnOut, &mut no);
				self.lin(
					il,
					WeightId::SsmOut,
					lt.ssm_out.as_ref().unwrap(),
					&no,
					self.hp.n_embd,
					value_dim,
				)
			};

			let after_attn: Vec<f32> = xin.iter().zip(&aout).map(|(a, b)| a + b).collect();
			if let Some(t_aout) = t_aout {
				AOUT_NS.with(|c| c.set(c.get() + t_aout.elapsed().as_nanos() as u64));
			}
			let t_flex2 = if measure { Some(Instant::now()) } else { None };
			x = x.iter().zip(aout.iter()).map(|(a, b)| a + b).collect();
			if measure {
				FLEXOP_NS.with(|c| c.set(c.get() + t_flex2.unwrap().elapsed().as_nanos() as u64));
			}
			let mut post = rms_norm(&after_attn, &lt.post_norm, eps);
			sparse::sparsify(sparse::Site::FfnIn, &mut post);
			// FfnUp/FfnGate both consume `post`; batch when offloaded.
			let up_fg = self.lin_batch(&[
				BatchJob {
					il,
					id: WeightId::FfnUp,
					w: &lt.ffn_up,
					x: &post,
					out_dim: self.hp.n_ff,
					in_dim: self.hp.n_embd,
				},
				BatchJob {
					il,
					id: WeightId::FfnGate,
					w: &lt.ffn_gate,
					x: &post,
					out_dim: self.hp.n_ff,
					in_dim: self.hp.n_embd,
				},
			]);
			let up = &up_fg[0];
			let fg = &up_fg[1];
			let t_el = if measure { Some(Instant::now()) } else { None };
			let mut h = quant::uninit_vec(self.hp.n_ff);
			quant::silu_mul(fg, up, &mut h);
			if let Some(t) = t_el {
				ELEM_NS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
			}
			sparse::set_layer(il);
			sparse::sparsify(sparse::Site::FfnDown, &mut h);
			let ffn_out = self.lin(
				il,
				WeightId::FfnDown,
				&lt.ffn_down,
				&h,
				self.hp.n_embd,
				self.hp.n_ff,
			);
			let t_flex3 = if measure { Some(Instant::now()) } else { None };
			x = x.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
			if measure {
				FLEXOP_NS.with(|c| c.set(c.get() + t_flex3.unwrap().elapsed().as_nanos() as u64));
			}

			if let Some(g) = gpu_layer_out.take() {
				let maxdiff = g
					.iter()
					.zip(x.iter())
					.map(|(a, b)| (a - b).abs())
					.fold(0.0f32, f32::max);
				eprintln!(
					"LAYERCMP il={} {} maxabsdiff={:.6} |cpu|={:.4} |gpu|={:.4}",
					il,
					if is_full { "full" } else { "ssm " },
					maxdiff,
					dbg_norm(&x),
					dbg_norm(&g)
				);
			}
			if dbg && (il <= 16 || il % 8 == 0 || il == self.hp.n_layer_main - 1) {
				let pmin = post.iter().cloned().fold(f32::INFINITY, f32::min);
				let pmax = post.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let xnow = x.clone();
				let umin = up.iter().cloned().fold(f32::INFINITY, f32::min);
				let umax = up.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let xmean = xnow.iter().sum::<f32>() / xnow.len() as f32;
				let amean = aout.iter().sum::<f32>() / aout.len() as f32;
				let pmean = post.iter().sum::<f32>() / post.len() as f32;
				let fmean = ffn_out.iter().sum::<f32>() / ffn_out.len() as f32;
				eprintln!(
					"DBG layer {} x0={:.2}(μ={:.3}) aout={:.2}(μ={:.3}) after={:.2} post={:.2}[{},{}](μ={:.3}) up={:.2}[{},{}] fg={:.2} h={:.2} ffn={:.2}(μ={:.3})",
					il, dbg_norm(&xnow), xmean,
					dbg_norm(&aout), amean,
					dbg_norm(&after_attn),
					dbg_norm(&post), pmin, pmax, pmean, dbg_norm(up), umin, umax,
					dbg_norm(fg), dbg_norm(&h), dbg_norm(&ffn_out), fmean
				);
			}
		}

		let last = x.clone();
		let nan_last = last.iter().filter(|v| v.is_nan()).count();
		let norm = rms_norm(&last, &self.output_norm, self.hp.eps);
		if let Ok(p) = std::env::var("QWEN_HIDDEN") {
			if let Ok(mut f) = std::fs::File::create(&p) {
				use std::io::Write;
				let _ = f.write_all(unsafe {
					std::slice::from_raw_parts(norm.as_ptr() as *const u8, norm.len() * 4)
				});
			}
		}
		if let Ok(p) = Self::stub_env("QWEN_RAWEMB") {
			let e = self.embedding(token);
			if let Ok(mut f) = std::fs::File::create(&p) {
				use std::io::Write;
				let _ = f.write_all(unsafe {
					std::slice::from_raw_parts(e.as_ptr() as *const u8, e.len() * 4)
				});
			}
			if let Ok(rp) = Self::stub_env("QWEN_REFRAM") {
				if let Ok(rb) = std::fs::read(&rp) {
					let r = unsafe {
						std::slice::from_raw_parts(rb.as_ptr() as *const f32, rb.len() / 4)
					};
					let n = self.hp.n_embd;
					let v = token as usize;
					let col: Vec<f32> = (0..n)
						.map(|d| self.tok_embd[d * self.hp.n_vocab + v])
						.collect();
					let corr = |a: &[f32], b: &[f32]| {
						let m = |x: &[f32]| x.iter().sum::<f32>() / x.len() as f32;
						let ma = m(a);
						let mb = m(b);
						let mut num = 0.0f64;
						let mut da = 0.0f64;
						let mut db = 0.0f64;
						for i in 0..a.len() {
							let x = (a[i] - ma) as f64;
							let y = (b[i] - mb) as f64;
							num += x * y;
							da += x * x;
							db += y * y;
						}
						num / (da.sqrt() * db.sqrt())
					};
					eprintln!(
						"DBG embcheck row corr={:.4} norm={:.4} | col corr={:.4} norm={:.4} | ref norm={:.4}",
						corr(&e, r), e.iter().map(|v| v * v).sum::<f32>().sqrt(),
						corr(&col, r), col.iter().map(|v| v * v).sum::<f32>().sqrt(),
						r.iter().map(|v| v * v).sum::<f32>().sqrt()
					);
				}
			}
		}
		if let Ok(p) = Self::stub_env("QWEN_NORMW") {
			let w = self.output_norm.clone();
			if let Ok(mut f) = std::fs::File::create(&p) {
				use std::io::Write;
				let _ = f.write_all(unsafe {
					std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
				});
			}
		}
		if let Some(t_loop) = t_loop {
			LAYER_LOOP_NS.with(|c| c.set(c.get() + t_loop.elapsed().as_nanos() as u64));
		}
		let loop_end_inst = Instant::now();
		// NOTE: these are whole-matrix scans of the lm_head (5.1 GB as f32 on
		// this model) and must stay inside the `dbg` guard; computing them
		// unconditionally cost ~216 ms/token (36 % of decode).
		if dbg {
			let nan_norm = norm.iter().filter(|v| v.is_nan()).count();
			let (nan_out, out0) = match &self.output {
				LinearWeight::F32(v) => (v.iter().filter(|x| x.is_nan()).count(), v[0]),
				LinearWeight::F16(v) => (0, quant::fp16_to_f32(v[0])),
				LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => {
					(0, raw.first().copied().unwrap_or(0) as f32)
				}
			};
			eprintln!(
				"DBG logits step={} last_nan={} norm_nan={} out_nan={} last0={:.3} norm0={:.3} out0={:.3}",
				rs.pos, nan_last, nan_norm, nan_out, last[0], norm[0], out0
			);
		}
		let t_lm = if measure { Some(Instant::now()) } else { None };
		let logits = apply_linear(&self.output, &norm, self.hp.n_vocab, self.hp.n_embd);

		let dc = last.iter().sum::<f32>() / last.len() as f32;
		// DC-removed logits via precomputed row sum: logits_dc[o] = logits[o]
		// - dc*Σ_d output[o,d]. Avoids a second full read of the output matrix.
		let mut logits_dc = vec![0.0f32; self.hp.n_vocab];
		for o in 0..self.hp.n_vocab {
			logits_dc[o] = logits[o] - dc * self.output_row_sum[o];
		}
		if let Some(t_lm) = t_lm {
			LMHEAD_NS.with(|c| c.set(c.get() + t_lm.elapsed().as_nanos() as u64));
		}

		if rs.mtp.is_some() && max_layers.is_none() {
			self.mtp_note_step(rs, emb, rs.mpos, &norm);
		}
		rs.pos += 1;
		rs.advance_text_mpos();
		let pre_ms = if let (Some(a), Some(b)) = (t0, t_loop) {
			b.duration_since(a).as_nanos() as f64 / 1e6
		} else {
			0.0
		};
		let post_ms = if let (Some(a), Some(b)) = (Some(loop_end_inst), t_lm) {
			b.duration_since(a).as_nanos() as f64 / 1e6
		} else {
			0.0
		};
		if let Some(t0) = t0 {
			let total = t0.elapsed().as_nanos() as u64;
			let mm = MATMUL_NS.with(|c| c.get());
			let non = total.saturating_sub(mm);
			let lm = LMHEAD_NS.with(|c| c.get());
			let non_ex_lm = non.saturating_sub(lm);
			let ssm = SSN_NS.with(|c| c.get());
			let loop_ns = LAYER_LOOP_NS.with(|c| c.get());
			let norm = NORM_NS.with(|c| c.get());
			let q8 = quant::q8_prof_ns();
			let flexop = FLEXOP_NS.with(|c| c.get());
			let aout_t = AOUT_NS.with(|c| c.get());
			let glue = total
				.saturating_sub(mm)
				.saturating_sub(lm)
				.saturating_sub(ssm)
				.saturating_sub(norm)
				.saturating_sub(flexop)
				.saturating_sub(aout_t);
			eprintln!(
				"PHASE step={} total_ms={:.2} matmul_ms={:.2} lmhead_ms={:.2} ssm_ms={:.2} norm_ms={:.2} q8_ms={:.2} elem_ms={:.2} prep_ms={:.2} flexop_ms={:.2} aout_ms={:.2} glue_ms={:.2} layerloop_ms={:.2} pre_ms={:.2} post_ms={:.2} nonmatmul_ex_lm_ms={:.2} matmul_frac={:.1}%",
				rs.pos - 1,
				total as f64 / 1e6,
				mm as f64 / 1e6,
				lm as f64 / 1e6,
				ssm as f64 / 1e6,
				norm as f64 / 1e6,
				q8 as f64 / 1e6,
				ELEM_NS.with(|c| c.get()) as f64 / 1e6,
				PREP_NS.with(|c| c.get()) as f64 / 1e6,
				flexop as f64 / 1e6,
				aout_t as f64 / 1e6,
				glue as f64 / 1e6,
				loop_ns as f64 / 1e6,
				pre_ms,
				post_ms,
				non_ex_lm as f64 / 1e6,
				100.0 * mm as f64 / total as f64
			);
		}
		(logits, logits_dc)
	}

	/// `std::env::var` wrapper kept local so the `vec_fwd` module doesn't need the
	/// `std::env` import at the top of the file (mirrors the default path).
	fn stub_env(k: &str) -> Result<String, std::env::VarError> {
		std::env::var(k)
	}
}
