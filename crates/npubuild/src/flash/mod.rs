//! Qwen3.8-Flash-Next (`qwen4_exp`): its float reference ([`refm`], the
//! counterpart of the 27B's [`crate::refm`]) and its calibration and
//! quantization ([`weights`]).

pub mod graphs;
pub mod mtp;
pub mod refm;
pub mod sim;
pub mod weights;

use crate::st::Checkpoint;
use crate::{calib_chunks, flash_calib, mkdir, say, step_emb, step_tokenizer, Options};

/// True when the checkpoint is Qwen3.8-Flash-Next (`model_type` `qwen4_exp`).
pub fn is_flash(ck: &Checkpoint) -> bool {
	ck.config.get("model_type").and_then(|v| v.str()) == Some("qwen4_exp")
}

/// `OUT/ngram.bf16`: the PLE layer's n-gram table, its checkpoint shards
/// concatenated (rows of 160 BF16; ~102 GB).
fn step_ngram(o: &Options, ck: &Checkpoint) -> Result<std::path::PathBuf, String> {
	use std::io::Write;
	let p = o.out.join("ngram.bf16");
	let pre = format!(
		"model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.shard_",
		npu::flash::PLE_LAYER
	);
	let mut names = Vec::new();
	while ck.has(&format!("{pre}{}.weight", names.len())) {
		names.push(format!("{pre}{}.weight", names.len()));
	}
	let mut total = 0u64;
	for n in &names {
		if ck.dtype(n)? != "BF16" || ck.shape(n)?[1] != npu::flash::ngram::DIM {
			return Err(format!("{n}: not BF16 rows of {}", npu::flash::ngram::DIM));
		}
		total += (ck.shape(n)?[0] * npu::flash::ngram::DIM * 2) as u64;
	}
	if p.metadata().map(|m| m.len() == total).unwrap_or(false) {
		return Ok(p);
	}
	say(&format!(
		"n-gram table: {} shards, {:.1} GB",
		names.len(),
		total as f64 / 1e9
	));
	let tmp = p.with_extension("tmp");
	let mut f = std::io::BufWriter::with_capacity(
		1 << 20,
		std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
	);
	for n in &names {
		ck.copy_raw(n, &mut f)?;
	}
	f.flush().map_err(|e| e.to_string())?;
	drop(f);
	std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))?;
	Ok(p)
}

/// `OUT/lm_head.f16`: `[vocab][2560]` f16 (the host head of the v1 engine).
fn step_head(o: &Options, ck: &Checkpoint) -> Result<(), String> {
	use std::io::Write;
	let p = o.out.join("lm_head.f16");
	let sh = ck.shape("lm_head.weight")?;
	let (v, w) = (sh[0], sh[1]);
	if p.metadata()
		.map(|m| m.len() == (v * w * 2) as u64)
		.unwrap_or(false)
	{
		return Ok(());
	}
	say("lm_head (f16)");
	let tmp = p.with_extension("tmp");
	let mut f = std::io::BufWriter::new(
		std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
	);
	for r0 in (0..v).step_by(4096) {
		let rows = ck.rows("lm_head.weight", r0, (r0 + 4096).min(v))?;
		let b: Vec<u8> = rows
			.iter()
			.flat_map(|&x| crate::g::f16_bits(x as f64).to_le_bytes())
			.collect();
		f.write_all(&b).map_err(|e| e.to_string())?;
	}
	f.flush().map_err(|e| e.to_string())?;
	drop(f);
	std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
}

/// The engine's graph kinds and their builders.
pub fn graph_kinds() -> Vec<(&'static str, crate::g::Graph)> {
	use graphs as fg;
	let s = fg::S;
	vec![
		("delta_pre", fg::delta_pre(s)),
		// the same after its hyper-connection (on the iGPU: `offload::HcPre`)
		("delta_pre_m", fg::delta_pre_m(s)),
		// the same up to its projections: the conv, Swish and q / k norms
		// on the iGPU ahead of the recurrence (`offload::DeltaScan::conv_next`)
		("delta_pre_c", fg::delta_pre_c(s)),
		("attn_pre", fg::attn_pre(s)),
		("post_delta", fg::post(s, true, npu::flash::OSCALE_P as f64)),
		("post_attn", fg::post(s, false, 16.0)),
		// `post_delta` in two around its MoE hyper-connection (on the
		// iGPU: `offload::HcPre`)
		("post_delta_a", fg::post_delta_a(s)),
		("post_delta_b", fg::post_delta_b(s)),
		// `post_delta_a` after the gated norm (on the iGPU, with the scan)
		("post_delta_t", fg::post_delta_t(s)),
		("ple", fg::ple(s)),
		// a prefill attention layer's past keys on the NPU, a block per job
		("attn_far", fg::attn_far(s, false)),
		("attn_farm", fg::attn_far(s, true)),
		// the block indexer's scores of the blocks before the sub-chunk
		("idx_score", fg::idx_score(s)),
		("experts", fg::experts(32, 32)),
		// prefill: experts with more than 32 tokens, 64 per slot
		("experts_c64", fg::experts(16, 64)),
		// the same slots, per-expert ops and outputs, 32 per job
		("experts_w", fg::experts_pe(32, 64)),
		// `experts_w` with fewer SHAVE Multiplies (`d.s` in the engine)
		("experts_ws", fg::experts_pe_s(32, 64, true)),
		("experts_ns", fg::experts_pe_s(32, 32, true)),
		// short expert passes: experts with at most 8 / 16 tokens
		("experts_ns8", fg::experts_pe_s(32, 8, true)),
		("experts_ns16", fg::experts_pe_s(32, 16, true)),
		// decode steps: one position, its 10 experts in one job
		("delta_pre_d1", fg::delta_pre(1)),
		("attn_pre_d1", fg::attn_pre(1)),
		("post_delta_d1", fg::post(1, true, 16.0)),
		("post_attn_d1", fg::post(1, false, 16.0)),
		("ple_d1", fg::ple(1)),
		("experts_d1", fg::experts(10, 1)),
	]
}

/// The hyper-connection matrices' int8 files (`weights::hc_i8`) of a
/// weight directory written before they existed, from its f16 files.
fn hc_i8_files(o: &Options) -> Result<(), String> {
	let w = o.out.join("w");
	for l in 0..npu::flash::NL {
		for p in ["hca", "hcm"] {
			for (m, k) in [
				("down", npu::flash::HC * npu::flash::W),
				("up", npu::flash::HC_RANK),
			] {
				let f = |x: &str| w.join(format!("fl{l}.{p}.{m}{x}.in"));
				if f(".s").exists() {
					continue;
				}
				let b = std::fs::read(f("")).map_err(|e| format!("{}: {e}", f("").display()))?;
				let v: Vec<f32> = b
					.chunks_exact(2)
					.map(|c| crate::g::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
					.collect();
				let (c, s) = weights::hc_i8(&v, k).map_err(|e| format!("fl{l}.{p}.{m}: {e}"))?;
				std::fs::write(f(".w"), c).map_err(|e| e.to_string())?;
				std::fs::write(f(".s"), s).map_err(|e| e.to_string())?;
			}
			if graphs::HC_FOLD_INJ {
				let f = |x: &str| w.join(format!("fl{l}.{p}.{x}.in"));
				if f("dinj.s").exists() {
					continue;
				}
				let rd =
					|x: &str| std::fs::read(f(x)).map_err(|e| format!("{}: {e}", f(x).display()));
				let (c, s) = weights::dinj(&rd("down.w")?, &rd("down.s")?, &rd("inj")?)
					.map_err(|e| format!("fl{l}.{p}.inj: {e}"))?;
				std::fs::write(f("dinj.w"), c).map_err(|e| e.to_string())?;
				std::fs::write(f("dinj.s"), s).map_err(|e| e.to_string())?;
			}
		}
	}
	Ok(())
}

/// `OUT/graphs/flash/<kind>.xml` + `.args` (the input names in order).
fn step_graphs(o: &Options) -> Result<(), String> {
	if graphs::HC_I8 {
		hc_i8_files(o)?;
	}
	let d = o.out.join("graphs").join("flash");
	mkdir(&d)?;
	for (n, g) in graph_kinds() {
		g.save(&d.join(format!("{n}.xml")), n)?;
		let args: Vec<String> = g.inputs().into_iter().map(|i| i.0).collect();
		std::fs::write(
			d.join(format!("{n}.args")),
			args.join(
				"
",
			) + "
",
		)
		.map_err(|e| e.to_string())?;
	}
	say(&format!("Flash-Next graphs in {}", d.display()));
	Ok(())
}

/// The SHAVE kernel directory of the Flash-Next graphs, under `OUT/graphs`.
const HC_KERN: &str = "kern/hc";
/// The same for the far attention graphs (`softmax_lse`, `-DNONORM`).
const FAR_KERN: &str = "kern/fattn";

/// Whether graph kind `n` is compiled to read its weight inputs swizzled
/// (`NPU_RS_SWZIN`; the engine swizzles the files the `.swz` names after
/// reading them): the dense layer graphs, whose weights only the NPU reads
/// from these buffers. Not the routed experts (their pools are shared with
/// the cores and the iGPU), nor the far attention and indexer graphs (their
/// MatMul operands are activations). Delta layer pre / post at 256
/// positions: 3.33 -> 2.72 / 2.57 -> 2.44 / 1.55 -> 1.48 ms
/// (`delta_pre_c` / `post_delta_t` / `post_delta_b`, npubench minima).
pub fn swizzled(n: &str) -> bool {
	!(n.starts_with("experts") || n.starts_with("attn_far") || n.starts_with("idx_score"))
}

/// Compile the graphs whose blob is missing.
fn step_compile(o: &Options) -> Result<(), String> {
	let t = crate::sets::Tools::find(o.compiler.as_deref(), o.jobs)?;
	let gdir = o.out.join("graphs");
	// the hyper-connections' SHAVE kernel (carrier Minimum: graphs::hc_in_k)
	crate::sets::write_kernel(
		&gdir.join(HC_KERN),
		include_str!("../../../npu/shave/hcmix.sk"),
		"eltwise_min",
		&[("GS".to_string(), npu::flash::OSCALE_P as i64)],
	)?;
	// the delta layers' Swish + l2 norm of q / k (carrier Mish:
	// graphs::L2_KERNEL)
	crate::sets::write_kernel(
		&gdir.join(HC_KERN),
		include_str!("../../../npu/shave/l2n.sk"),
		"activation_mish",
		&[],
	)?;
	// the far attention blocks' Softmax (max and sum in the spare slots)
	crate::sets::write_kernel(
		&gdir.join(FAR_KERN),
		include_str!("../../../npu/shave/softmax_lse.sk"),
		"softmax",
		&[
			("N".to_string(), graphs::FAR_CAP as i64),
			("NONORM".to_string(), 1),
			("KL".to_string(), 4),
		],
	)?;
	for (n, _) in graph_kinds() {
		let stem = format!("flash/{n}");
		if gdir.join(format!("{stem}.blob")).exists() {
			continue;
		}
		say(&format!("compiling {stem}"));
		let kern = if n.starts_with("attn_far") {
			FAR_KERN
		} else {
			HC_KERN
		};
		crate::sets::compile_graph_kern(&t, &gdir, &stem, Some(kern), swizzled(n))?;
	}
	Ok(())
}

/// The engine descriptions (paths relative to the file), as the 27B's
/// [`crate::sets`] step: `OUT/model.npu` (NPU + iGPU + GTX 1060 + cores:
/// the full mode, with the MTP head when the `mtp` step made it; the head
/// runs only with `dense`), `OUT/model_igpu.npu` (NPU + iGPU, the decode
/// experts on the cores) and `OUT/model_npu.npu` (NPU only, plus the
/// cores for the decode experts and the delta recurrence).
fn step_sets(o: &Options) -> Result<(), String> {
	let common = "arch      = flash
graphs    = graphs/flash
weights   = w
ngram     = ngram.bf16
emb       = emb_table.f16
head      = lm_head.f16
tokenizer = tokenizer
";
	// images, when the `vis` step made the tower's set
	let vis = if o.out.join("sets/vis/vis.cfg").exists() {
		"# images: the tower's per-patch work on the NPU, its attention on the `scan` adapter (else the cores)
vision    = sets/vis
"
	} else {
		""
	};
	// the MTP drafter, when the `mtp` step made its files
	let mtp = if o.out.join("mtp/done").exists() {
		"# MTP drafts (on the cores, needs `dense`)
mtp       = mtp
"
	} else {
		""
	};
	let scan = "# helper: delta recurrence and prefill attention on the iGPU
scan      = intel
";
	let write =
		|n: &str, t: String| std::fs::write(o.out.join(n), t).map_err(|e| format!("{n}: {e}"));
	write(
		"model.npu",
		format!(
			"# Flash-Next, NPU + CPU + GPU: decode's dense part and lm_head on the GTX 1060, routed experts on the cores
{common}{vis}{scan}dense     = geforce
{mtp}"
		),
	)?;
	write(
		"model_igpu.npu",
		format!(
			"# Flash-Next, NPU + iGPU
{common}{vis}{scan}"
		),
	)?;
	write(
		"model_npu.npu",
		format!(
			"# Flash-Next, NPU only
{common}{vis}"
		),
	)?;
	say(&format!(
		"done: {} (also model_igpu.npu, model_npu.npu)",
		o.out.join("model.npu").display()
	));
	Ok(())
}

/// The Flash-Next build: `emb`, `ngram`, `weights`, `head`, `graphs`,
/// `compile`, `vis` (the vision tower's set: [`crate::sets::vis_set`]),
/// `mtp` (the MTP drafter's files: [`mtp::step_mtp`]), `sets`.
pub fn build(o: &Options, ck: &Checkpoint) -> Result<(), String> {
	let tok = step_tokenizer(o)?;
	if o.wants("emb") {
		step_emb(o, ck)?;
	}
	let table = if o.wants("ngram") || o.wants("weights") {
		Some(step_ngram(o, ck)?)
	} else {
		None
	};
	if o.wants("weights") {
		let chunks = calib_chunks(o, &tok, &flash_calib())?;
		say(&format!(
			"Flash-Next calibration on {} chunks of {} tokens",
			chunks.len(),
			weights::CHUNK
		));
		let (cal, w) = (o.out.join("cal"), o.out.join("w"));
		mkdir(&cal)?;
		let t = npu::flash::ngram::Table::open(table.as_ref().unwrap())?;
		weights::calibrate(ck, &t, &chunks, &cal, &w, &mut |m| say(m))?;
	}
	if o.wants("head") {
		step_head(o, ck)?;
	}
	if o.wants("graphs") {
		step_graphs(o)?;
	}
	if o.wants("compile") {
		step_compile(o)?;
	}
	if o.wants("vis") && crate::sets::vis_set(ck, &o.out, &mut |m| say(m))? {
		say(&format!(
			"vision tower: {}",
			o.out.join("sets").join("vis").display()
		));
	}
	if o.wants("mtp") {
		mtp::step_mtp(&o.out, ck, &mut |m| say(m))?;
	}
	if o.wants("sets") {
		step_sets(o)?;
	}
	Ok(())
}
