//! `--kld-eval`: teacher-forced accuracy of a decode-path variant against the
//! unmodified model, over the same text.
//!
//! The baseline pass stores every position's log-softmax; each variant then
//! reports mean / p99 / max KL(base || variant), top-1 agreement and the
//! perplexity of the text under both. Built for the activation-sparsity
//! experiment (`model::sparse`), where `DCR SUM` on one prompt cannot rank
//! anything. Runs `forward_step`, i.e. the decode path, token by token.
//!
//! `QWEN_KLD_BASE=FILE` makes the baseline cross-binary: if `FILE` exists the
//! baseline pass is skipped and its log-softmax rows are read from it,
//! otherwise they are computed and written there. That is how a change that
//! cannot be toggled at runtime (the KV cache's storage type) is measured:
//! save the baseline with the old build, evaluate with the new one.

use std::time::Instant;

use tokenizer::Tokenizer;

fn log_softmax(l: &[f32]) -> Vec<f32> {
	let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let lse = m as f64 + l.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
	l.iter().map(|&v| (v as f64 - lse) as f32).collect()
}

fn argmax(v: &[f32]) -> usize {
	let mut b = 0;
	for i in 1..v.len() {
		if v[i] > v[b] {
			b = i;
		}
	}
	b
}

/// `QWEN_KLD_BASE` rows: `rows * n_vocab` little-endian f32, nothing else.
/// `None` when the file is missing or has another shape.
fn read_base(f: &str, rows: usize, n_vocab: usize) -> Option<Vec<Vec<f32>>> {
	let b = std::fs::read(f).ok()?;
	if b.len() != rows * n_vocab * 4 {
		eprintln!(
			"kld-eval: {f} has {} bytes, expected {}; recomputing",
			b.len(),
			rows * n_vocab * 4
		);
		return None;
	}
	Some(
		b.chunks_exact(n_vocab * 4)
			.map(|r| {
				r.chunks_exact(4)
					.map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
					.collect()
			})
			.collect(),
	)
}

fn write_base(f: &str, rows: &[Vec<f32>]) -> std::io::Result<()> {
	let mut w = std::io::BufWriter::new(std::fs::File::create(f)?);
	for r in rows {
		for x in r {
			std::io::Write::write_all(&mut w, &x.to_le_bytes())?;
		}
	}
	std::io::Write::flush(&mut w)
}

/// Run `tokens` through `forward_step`, calling `f(pos, log_softmax)`.
fn pass(model: &model::Model, tokens: &[u32], mut f: impl FnMut(usize, Vec<f32>)) -> f64 {
	let mut rs = model.new_run_state();
	let t = Instant::now();
	for (i, &tok) in tokens.iter().enumerate() {
		let (logits, _) = model.forward_step(&mut rs, tok, None);
		f(i, log_softmax(&logits));
	}
	tokens.len() as f64 / t.elapsed().as_secs_f64()
}

pub fn kld_eval(
	path: &str,
	text_file: &str,
	n_tokens: usize,
	sweep: &str,
	n_gpu_layers: usize,
	n_batch: usize,
) -> Result<(), Box<dyn std::error::Error>> {
	let cfgs: Vec<model::sparse::SparseCfg> = sweep
		.split(';')
		.filter(|s| !s.trim().is_empty())
		.map(model::sparse::SparseCfg::parse)
		.collect::<Result<_, _>>()?;
	let text = std::fs::read_to_string(text_file)?;
	let gguf = gguf::parse_file(path)?;
	let tk = Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
	drop(gguf);
	let mut toks = tk.encode(&text);
	toks.truncate(n_tokens + 1);
	if toks.len() < 2 {
		return Err("kld-eval: text too short".into());
	}
	let (input, target) = (&toks[..toks.len() - 1], &toks[1..]);
	println!("== kld-eval: {} positions from {text_file}", input.len());

	let mut model = model::Model::load(path)?;
	model.set_ctx(toks.len() + 16);
	model.set_n_batch(n_batch);
	model.warm();
	if n_gpu_layers > 0 || model::gpu_prefill_requested() {
		model.init_offload(n_gpu_layers);
	}

	// `QWEN_KLD_REF=exact`: the reference pass runs every decode matmul with an
	// f32 activation (`QWEN_EXACT_DECODE`), so an unsparsified variant measures
	// the shipped q8_K / GPU-split decode's own error; the noise floor.
	let exact_ref = std::env::var("QWEN_KLD_REF").is_ok_and(|v| v == "exact");
	quant::set_exact_decode(exact_ref);
	model::sparse::set_sparsity(model::sparse::SparseCfg::default());
	let base_file = std::env::var("QWEN_KLD_BASE").ok();
	let mut base: Vec<Vec<f32>> = Vec::with_capacity(input.len());
	let mut nll = 0.0f64;
	if let Some(rows) = base_file
		.as_deref()
		.and_then(|f| read_base(f, input.len(), model.hp.n_vocab))
	{
		base = rows;
		for (i, ls) in base.iter().enumerate() {
			nll -= ls[target[i] as usize] as f64;
		}
		let ppl0 = (nll / input.len() as f64).exp();
		println!(
			"BASE (from {}) ppl {ppl0:.4}",
			base_file.as_deref().unwrap()
		);
	} else {
		let tps = pass(&model, input, |i, ls| {
			nll -= ls[target[i] as usize] as f64;
			base.push(ls);
		});
		let ppl0 = (nll / input.len() as f64).exp();
		println!(
			"BASE{} ppl {ppl0:.4}  ({tps:.2} tok/s)",
			if exact_ref { " (exact)" } else { "" }
		);
		if let Some(f) = &base_file {
			write_base(f, &base)?;
			println!("BASE saved to {f}");
		}
	}
	let ppl0 = (nll / input.len() as f64).exp();
	quant::set_exact_decode(false);

	let freq = std::env::var("QWEN_SPARSE_FREQ").ok();
	if freq.is_some() {
		model::sparse::freq_enable();
	}
	for (ci, c) in cfgs.into_iter().enumerate() {
		model::sparse::set_sparsity(c);
		let mut kl: Vec<f64> = Vec::with_capacity(input.len());
		let mut agree = 0usize;
		let mut nll = 0.0f64;
		let tps = pass(&model, input, |i, ls| {
			let b = &base[i];
			let mut s = 0.0f64;
			for (pb, ps) in b.iter().zip(&ls) {
				let p = (*pb as f64).exp();
				if p > 0.0 {
					s += p * (*pb as f64 - *ps as f64);
				}
			}
			kl.push(s.max(0.0));
			agree += (argmax(b) == argmax(&ls)) as usize;
			nll -= ls[target[i] as usize] as f64;
		});
		let n = kl.len() as f64;
		let mean = kl.iter().sum::<f64>() / n;
		let mut sorted = kl.clone();
		sorted.sort_by(|a, b| a.total_cmp(b));
		let p99 = sorted[((n * 0.99) as usize).min(sorted.len() - 1)];
		let max = *sorted.last().unwrap();
		let ppl = (nll / n).exp();
		println!(
			"CFG {c}  KLD mean {mean:.5} p99 {p99:.4} max {max:.4}  top1 {:.2}%  ppl {ppl:.4} ({:+.2}%)  ({tps:.2} tok/s)",
			agree as f64 * 100.0 / n,
			(ppl / ppl0 - 1.0) * 100.0,
		);
		println!("    {}", model::sparse::sparsity_report());
		if let Some(p) = &freq {
			model::sparse::freq_dump(&format!("{p}.{ci}"));
		}
	}
	Ok(())
}
