//! `cli MODEL --tune-split`: measure the prefill row split on this host and
//! recommend its GPU fraction.
//!
//! The split fractions (`QWEN_PREFILL_GPU_FRAC_Q4K` / `_Q6K`, the
//! `QWEN_PREFILL_DFRAC_<ID>` corrections) were set by end-to-end sweeps, which
//! the host's run-to-run drift makes slow and blunt, and they go stale every
//! time a CPU or GPU GEMM changes. This times the real split path
//! (`OffloadManager::matmul_seq_group_q8`, the call prefill makes) on the
//! loaded model's own weights and resident rows, at a grid of forced
//! fractions, and picks the fastest per matmul group.
//!
//! What it does not see: the deferred `WqkvGate` / output-gate matmuls that
//! overlap the CPU's SSM passes and attention, and anything else prefill does
//! between matmuls. The recommendation balances each matmul in isolation;
//! confirm it end to end (interleaved runs, never single ones).

use super::*;

/// One prefill matmul group as the model dispatches it (shared input).
struct Group {
	/// `QWEN_PREFILL_DFRAC_<key>` entry this group's fraction maps to.
	key: &'static str,
	il: usize,
	/// Q6_K by bytes (else Q4_K): which per-quant default the group follows.
	q6k: bool,
	in_dim: usize,
	lt: Arc<LayerTensors>,
	jobs: Vec<(WeightId, usize)>,
	/// Layers of the model with the same key and quant type.
	n_layers: usize,
}

impl Group {
	fn weight(&self, id: WeightId) -> &LinearWeight {
		let lt = &*self.lt;
		match id {
			WeightId::Wq => lt.wq.as_ref(),
			WeightId::Wk => lt.wk.as_ref(),
			WeightId::Wv => lt.wv.as_ref(),
			WeightId::Wo => lt.wo.as_ref(),
			WeightId::Wqkv => lt.wqkv.as_ref(),
			WeightId::SsmOut => lt.ssm_out.as_ref(),
			WeightId::FfnUp => Some(&lt.ffn_up),
			WeightId::FfnGate => Some(&lt.ffn_gate),
			WeightId::FfnDown => Some(&lt.ffn_down),
			_ => None,
		}
		.expect("tune_groups only builds groups of present weights")
	}

	/// `(id, weight, out_dim)` per job, as the model's group calls take them.
	fn job_refs(&self) -> Vec<(WeightId, &LinearWeight, usize)> {
		self.jobs
			.iter()
			.map(|&(id, od)| (id, self.weight(id), od))
			.collect()
	}

	fn flops_per_tok(&self) -> f64 {
		self.jobs
			.iter()
			.map(|(_, od)| 2.0 * (*od * self.in_dim) as f64)
			.sum()
	}
}

/// Measured result of one group at one chunk length.
struct Fit {
	key: &'static str,
	q6k: bool,
	n_layers: usize,
	flops: f64,
	best_f: f64,
	best_ms: f64,
	dflt_f: f64,
	dflt_ms: f64,
}

/// Fraction of a group's rows on the GPU under the current settings
/// (row-weighted over its jobs).
fn group_frac(off: &OffloadManager, g: &Group, seq: usize) -> Option<f64> {
	let (mut gpu, mut all) = (0usize, 0usize);
	for (id, w, od) in g.job_refs() {
		let j = BatchJobSeq {
			il: g.il,
			id,
			w,
			x: &[],
			seq,
			out_dim: od,
			in_dim: g.in_dim,
		};
		let (_, rows) = off.prefill_split_plan(&j)?;
		gpu += rows;
		all += od;
	}
	Some(gpu as f64 / all as f64)
}

impl Model {
	/// The distinct (group, quant type) pairs of the main stack, each with the
	/// first layer that has it and a count of the layers that do.
	fn tune_groups(&self, off: &OffloadManager) -> Vec<Group> {
		let hp = &self.hp;
		let (n_embd, n_ff) = (hp.n_embd, hp.n_ff);
		let (head_dim, n_head, n_head_kv) = (hp.head_dim, hp.n_head, hp.n_head_kv);
		let value_dim = hp.ssm_inner;
		let conv_dim = hp.ssm_state * hp.ssm_n_group * 2 + value_dim;
		let mut out: Vec<Group> = Vec::new();
		for il in 0..hp.n_layer_main {
			let lt = self.get_layer(il);
			let mut groups: Vec<(&'static str, usize, Vec<(WeightId, &LinearWeight, usize)>)> =
				Vec::new();
			if self.full_attn_layers[il] {
				groups.push((
					"WQ",
					n_embd,
					vec![
						(WeightId::Wq, lt.wq.as_ref().unwrap(), 2 * head_dim * n_head),
						(WeightId::Wk, lt.wk.as_ref().unwrap(), head_dim * n_head_kv),
						(WeightId::Wv, lt.wv.as_ref().unwrap(), head_dim * n_head_kv),
					],
				));
				groups.push((
					"WO",
					head_dim * n_head,
					vec![(WeightId::Wo, lt.wo.as_ref().unwrap(), n_embd)],
				));
			} else {
				groups.push((
					"WQKV",
					n_embd,
					vec![(WeightId::Wqkv, lt.wqkv.as_ref().unwrap(), conv_dim)],
				));
				groups.push((
					"SSMOUT",
					value_dim,
					vec![(WeightId::SsmOut, lt.ssm_out.as_ref().unwrap(), n_embd)],
				));
			}
			groups.push((
				"FFNUP",
				n_embd,
				vec![
					(WeightId::FfnUp, &lt.ffn_up, n_ff),
					(WeightId::FfnGate, &lt.ffn_gate, n_ff),
				],
			));
			groups.push((
				"FFNDOWN",
				n_ff,
				vec![(WeightId::FfnDown, &lt.ffn_down, n_embd)],
			));
			for (key, in_dim, jobs) in groups {
				// Only groups whose every matrix is K-quant and split.
				let mut q6 = 0usize;
				let mut all = 0usize;
				let mut ok = true;
				for (id, w, _) in &jobs {
					match w {
						LinearWeight::Q4K { raw, .. } => all += raw.len(),
						LinearWeight::Q6K { raw, .. } => {
							q6 += raw.len();
							all += raw.len();
						}
						_ => ok = false,
					}
					ok &= off.split_rows(il, *id).is_some();
				}
				if !ok {
					continue;
				}
				let q6k = 2 * q6 > all;
				if let Some(g) = out.iter_mut().find(|g| g.key == key && g.q6k == q6k) {
					g.n_layers += 1;
					continue;
				}
				out.push(Group {
					key,
					il,
					q6k,
					in_dim,
					lt: lt.clone(),
					jobs: jobs.into_iter().map(|(id, _, od)| (id, od)).collect(),
					n_layers: 1,
				});
			}
		}
		out
	}

	/// Sweep the forced prefill split fraction for every matmul group at each
	/// chunk length in `seqs`, `rounds` interleaved passes per point (best
	/// kept), and print the fastest fraction with the settings that give it.
	pub fn tune_split(&self, seqs: &[usize], rounds: usize) -> Result<(), String> {
		let off = self
			.offload
			.as_ref()
			.filter(|o| o.available && o.n_split() > 0)
			.ok_or("--tune-split needs the Vulkan build, a GPU and the decode row split (QWEN_GPU_SPLIT)")?;
		let rounds = rounds.max(1);
		let groups = self.tune_groups(off);
		if groups.is_empty() {
			return Err("no split K-quant matmul groups to tune".into());
		}
		println!(
			"== tune-split: {} matmul groups, {rounds} rounds per point",
			groups.len()
		);
		for g in &groups {
			println!(
				"  {:<8} {}  layer {:>2} (x{:>2} layers)  {:.2} GFLOP/token",
				g.key,
				if g.q6k { "Q6_K" } else { "Q4_K" },
				g.il,
				g.n_layers,
				g.flops_per_tok() / 1e9,
			);
		}

		let mut by_seq: Vec<(usize, Vec<Fit>)> = Vec::new();
		for &seq in seqs {
			if seq < 64 {
				println!("\n-- seq {seq}: below 64 tokens prefill uses the short-chunk split (resident rows only); skipped");
				continue;
			}
			println!("\n-- seq {seq} --");
			println!("  group     quant | default        | best           | gain");
			let mut fits = Vec::new();
			// A short chunk's call is a few ms, where the host's jitter is
			// the size of the effect: give it more passes (cheap there).
			let rounds = rounds * (1024 / seq).clamp(1, 8);
			for g in &groups {
				if let Some(f) = self.tune_group(off, g, seq, rounds) {
					println!(
						"  {:<8}  {} | {:.2} {:8.2} ms | {:.2} {:8.2} ms | {:+5.1} %",
						f.key,
						if f.q6k { "Q6_K" } else { "Q4_K" },
						f.dflt_f,
						f.dflt_ms,
						f.best_f,
						f.best_ms,
						100.0 * (f.dflt_ms / f.best_ms - 1.0),
					);
					fits.push(f);
				} else {
					println!(
						"  {:<8}  {} | no streamed rows at this length",
						g.key,
						if g.q6k { "Q6_K" } else { "Q4_K" }
					);
				}
			}
			let (d, b): (f64, f64) = fits
				.iter()
				.map(|f| (f.dflt_ms * f.n_layers as f64, f.best_ms * f.n_layers as f64))
				.fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
			if b > 0.0 {
				println!(
					"  split matmuls per {seq}-token chunk: {:.0} ms default -> {:.0} ms tuned ({:+.1} %)",
					d,
					b,
					100.0 * (d / b - 1.0)
				);
			}
			by_seq.push((seq, fits));
		}
		off.set_prefill_frac_override(None);
		self.print_recommendation(&by_seq);
		Ok(())
	}

	/// Time `g` at a coarse then a fine grid of forced fractions (plus the
	/// current default), interleaved over `rounds` passes, keeping each
	/// point's best time. `None` if this length streams no extra rows.
	fn tune_group(
		&self,
		off: &OffloadManager,
		g: &Group,
		seq: usize,
		rounds: usize,
	) -> Option<Fit> {
		// The forced fraction is clamped to the resident rows from below, and
		// under the streaming gate it can move nothing at all.
		off.set_prefill_frac_override(Some(0.95));
		let hi = group_frac(off, g, seq)?;
		off.set_prefill_frac_override(Some(0.0));
		let lo = group_frac(off, g, seq)?;
		off.set_prefill_frac_override(None);
		let dflt_f = group_frac(off, g, seq)?;
		if hi <= lo + 1e-3 {
			return None;
		}

		// Deterministic input; the values do not affect timing.
		let x: Vec<f32> = (0..seq * g.in_dim)
			.map(|j| ((j * 7919 % 997) as f32 - 498.0) * 0.004)
			.collect();
		let q8 = std::sync::Arc::new(quant::par_quantize_q8k(&x, seq, g.in_dim));
		let jobs = g.job_refs();
		let run = |f: Option<f64>| -> f64 {
			off.set_prefill_frac_override(f);
			let t = Instant::now();
			let r = if off.prefill_q8_ok(g.il, &jobs, seq, g.in_dim) {
				off.matmul_seq_group_q8(g.il, &jobs, &q8, seq, g.in_dim)
			} else {
				let b: Vec<BatchJobSeq> = jobs
					.iter()
					.map(|(id, w, od)| BatchJobSeq {
						il: g.il,
						id: *id,
						w,
						x: &x,
						seq,
						out_dim: *od,
						in_dim: g.in_dim,
					})
					.collect();
				off.matmul_batch_seq(&b)
			};
			let ms = t.elapsed().as_secs_f64() * 1e3;
			std::hint::black_box(r);
			ms
		};
		// Warm the stream slots and buffer pools at the widest share.
		run(Some(0.95));
		run(None);

		// Points are forced fractions; `None` is the shipped default.
		let sweep = |pts: &[Option<f64>]| -> Vec<f64> {
			let mut best = vec![f64::MAX; pts.len()];
			for r in 0..rounds {
				// Alternate the direction so a drift over the pass does not
				// favor one end of the grid.
				let order: Vec<usize> = if r % 2 == 0 {
					(0..pts.len()).collect()
				} else {
					(0..pts.len()).rev().collect()
				};
				for i in order {
					best[i] = best[i].min(run(pts[i]));
				}
			}
			best
		};
		let lo_f = (lo * 100.0).ceil() / 100.0;
		let hi_f = 0.95f64.min((hi * 100.0).floor() / 100.0);
		let mut coarse: Vec<Option<f64>> = vec![None];
		let mut f = lo_f;
		while f <= hi_f + 1e-9 {
			coarse.push(Some(f));
			f += 0.04;
		}
		let t = sweep(&coarse);
		let dflt_ms = t[0];
		let (ci, _) = t
			.iter()
			.enumerate()
			.skip(1)
			.min_by(|a, b| a.1.total_cmp(b.1))
			.unwrap();
		let c = coarse[ci].unwrap();
		let mut fine: Vec<Option<f64>> = vec![None];
		for k in -3i32..=3 {
			let f = ((c + k as f64 * 0.01) * 100.0).round() / 100.0;
			if f >= lo_f - 1e-9 && f <= hi_f + 1e-9 {
				fine.push(Some(f));
			}
		}
		let t2 = sweep(&fine);
		let (fi, _) = t2
			.iter()
			.enumerate()
			.skip(1)
			.min_by(|a, b| a.1.total_cmp(b.1))
			.unwrap();
		off.set_prefill_frac_override(None);
		// The fine pass re-measures the default too; keep both passes' best
		// so the two columns are like for like.
		let dflt_ms = dflt_ms.min(t2[0]);
		let (mut best_f, mut best_ms) = (fine[fi].unwrap(), t2[fi].min(t[ci]));
		// A sweep that cannot beat the default is noise, not a finding.
		if dflt_ms <= best_ms {
			(best_f, best_ms) = (dflt_f, dflt_ms);
		}
		Some(Fit {
			key: g.key,
			q6k: g.q6k,
			n_layers: g.n_layers,
			flops: g.flops_per_tok() * seq as f64,
			best_f: (best_f * 100.0).round() / 100.0,
			best_ms,
			dflt_f,
			dflt_ms,
		})
	}

	/// Turn the per-group optima into the settings that reproduce them: one
	/// base fraction per quant type (time-weighted over the groups that
	/// follow it) and per-group corrections, as environment overrides and as
	/// the code defaults in `offload::manager`.
	fn print_recommendation(&self, by_seq: &[(usize, Vec<Fit>)]) {
		let Some((top_seq, top)) = by_seq.iter().max_by_key(|(s, _)| *s) else {
			return;
		};
		if top.is_empty() {
			return;
		}
		// Weight by the group's share of a chunk's matmul time.
		let base = |fits: &[Fit], q6k: bool| -> Option<f64> {
			let (s, w) = fits
				.iter()
				.filter(|f| f.q6k == q6k)
				.map(|f| {
					(
						f.best_f * f.best_ms * f.n_layers as f64,
						f.best_ms * f.n_layers as f64,
					)
				})
				.fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
			(w > 0.0).then(|| (s / w * 100.0).round() / 100.0)
		};
		let (b4, b6) = (base(top, false), base(top, true));
		println!("\n== recommendation (from seq {top_seq}) ==");
		let mut env = Vec::new();
		if let Some(b) = b4 {
			env.push(format!("QWEN_PREFILL_GPU_FRAC_Q4K={b:.2}"));
		}
		if let Some(b) = b6 {
			env.push(format!("QWEN_PREFILL_GPU_FRAC_Q6K={b:.2}"));
		}
		// Per-key correction: time-weighted mean of (optimum - its base).
		let mut keys: Vec<&'static str> = top.iter().map(|f| f.key).collect();
		keys.dedup();
		keys.sort();
		keys.dedup();
		let mut deltas = Vec::new();
		for k in keys {
			let (s, w) = top
				.iter()
				.filter(|f| f.key == k)
				.filter_map(|f| {
					let b = if f.q6k { b6? } else { b4? };
					let wt = f.best_ms * f.n_layers as f64;
					Some(((f.best_f - b) * wt, wt))
				})
				.fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
			let d = if w > 0.0 {
				(s / w * 100.0).round() / 100.0
			} else {
				0.0
			};
			deltas.push((k, d));
			env.push(format!("QWEN_PREFILL_DFRAC_{k}={d:.2}"));
		}
		println!("  environment (a QWEN_PREFILL_GPU_FRAC_* override applies at every chunk length >= 64,");
		println!("  replacing the code's lower bases below 192 and 128 tokens):");
		println!("    set {}", env.join(" & set "));
		println!("  per-group optima at seq {top_seq}:");
		for f in top {
			println!(
				"    {:<8} {}  {:.2}  ({:.0} GFLOP/s combined)",
				f.key,
				if f.q6k { "Q6_K" } else { "Q4_K" },
				f.best_f,
				f.flops / f.best_ms * 1e-6,
			);
		}
		// The code keeps the base at >= 192 tokens and lowers it below 192
		// and below 128; report what the shorter lengths measured.
		let short: Vec<String> = by_seq
			.iter()
			.filter(|(s, _)| s != top_seq)
			.filter_map(|(s, fits)| {
				let o4 = base(fits, false)
					.zip(b4)
					.map(|(x, b)| format!("Q4_K {:+.2}", x - b));
				let o6 = base(fits, true)
					.zip(b6)
					.map(|(x, b)| format!("Q6_K {:+.2}", x - b));
				let v: Vec<String> = o4.into_iter().chain(o6).collect();
				(!v.is_empty()).then(|| format!("seq {s}: {}", v.join(", ")))
			})
			.collect();
		if !short.is_empty() {
			println!("  base offset at shorter chunks (the code default is -0.07 below 192, -0.11 below 128):");
			for s in short {
				println!("    {s}");
			}
		}
		println!(
			"  code defaults: `prefill_split_rows` f_dp4a {} / {}, `prefill_frac_delta` {}",
			b4.map_or("-".into(), |b| format!("{b:.2}")),
			b6.map_or("-".into(), |b| format!("{b:.2}")),
			deltas
				.iter()
				.map(|(k, d)| format!("{k} {d:+.2}"))
				.collect::<Vec<_>>()
				.join(", "),
		);
		println!("  Confirm end to end with interleaved prefill runs before changing defaults.");
	}
}
