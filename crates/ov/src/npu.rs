//! The NPU engine: the export's language model made stateless
//! ([`crate::stateless`]), reshaped to static shapes and compiled through
//! NPUW **without** its LLM mode (which fails on this architecture), with the
//! state kept on the host.
//!
//! Two static graphs share one weights bank:
//! - **decode**: 1 position per `infer`;
//! - **prefill** (optional): `chunk` positions per `infer`. A remainder
//!   shorter than a chunk is right-padded ([`stateless::pad_inputs`]: the
//!   filler positions leave the conv / SSM state exactly as the real ones
//!   left it); one of fewer than `pad_min` positions, where a padded call
//!   costs more than single steps, goes through the decode graph.
//!
//! State layout, all `f16` on the host (NPUW LLM mode keeps its KV in `f16`
//! too; an `f32` state port fails with `f32 != f16` on infer-request
//! creation):
//! - KV (16 layers x key/value): a fixed `[1, 4, P, 256]` past buffer, the
//!   model's `present` output is `[1, 4, P + S, 256]` = past ++ new, so each
//!   step copies its `S` new rows into the past buffer at the conversation's
//!   position. `attention_mask` marks the filled past slots and the new
//!   positions; unfilled slots are masked out.
//! - conv / SSM (48 layers each): two buffers per state, the step's input and
//!   output, swapped after every `infer`; no copy.

use std::path::Path;
use std::time::Instant;

use crate::rt::{Compiled, HostTensor, Request, Runtime};
use crate::stateless::{self, State};

pub const HIDDEN: i64 = 5120;

struct Stage {
	// Field order is drop order: the request goes before the tensors it binds.
	req: Request,
	_c: Compiled,
	s: usize,
	x: HostTensor,
	mask: HostTensor,
	pos: HostTensor,
	beam: Option<HostTensor>,
	/// Host-computed rotary `(cos, sin)` inputs, `[1, s, 64]` f32.
	rope: Option<(HostTensor, HostTensor)>,
	/// Padding inputs ([`stateless::VALID`], [`stateless::CONV_KEEP`]) and
	/// the conv state width, when the graph takes padded input.
	pad: Option<(HostTensor, HostTensor, usize)>,
	/// [`stateless::LAST_ONEHOT`], when the graph computes one position's
	/// logits.
	last: Option<HostTensor>,
	/// `present` KV outputs, one per KV state (same order as `NpuLlm::kv`).
	kv_out: Vec<HostTensor>,
}

struct Recur {
	st: State,
	cur: HostTensor,
	next: HostTensor,
}

struct Kv {
	st: State,
	past: HostTensor,
	/// Heads x head dim of one position (`dims[1]`, `dims[3]`).
	heads: usize,
	hd: usize,
}

/// The rotary tables are computed on the host (`stateless::rope_inputs`).
struct Rope {
	inv_freq: Vec<f32>,
	/// `QWEN_OV_ROPE=f16sim`: with the angle rounded as in f16.
	f16: bool,
}

pub struct NpuLlm {
	dec: Stage,
	pre: Option<Stage>,
	kv: Vec<Kv>,
	recur: Vec<Recur>,
	rope: Option<Rope>,
	/// `QWEN_OV_PROBES`: extra outputs dumped per step (`crate::dbg`).
	probes: Vec<String>,
	/// KV capacity (static past length).
	pub cap: usize,
	pub pos: usize,
	/// Shortest remainder run through the padded prefill graph (shorter
	/// ones go through the decode graph one position at a time).
	pad_min: usize,
	// Last: released after the requests and compiled models are gone.
	_lock: crate::lock::NpuLock,
}

pub struct NpuOptions<'a> {
	pub cap: usize,
	pub chunk: usize,
	pub cache_dir: Option<&'a str>,
	pub props: &'a [(String, String)],
}

impl NpuLlm {
	/// `xml` / `bin`: the export's stateful language model.
	pub fn load(rt: &Runtime, xml: &Path, bin: &Path, o: &NpuOptions) -> Result<NpuLlm, String> {
		// `QWEN_OV_STATIC_DEVICE=CPU`: the same static stateless graphs on
		// another device, to check the host-side state handling against the
		// stateful reference without an NPU compile.
		let device = std::env::var("QWEN_OV_STATIC_DEVICE").unwrap_or_else(|_| "NPU".into());
		let t = Instant::now();
		let text = std::fs::read_to_string(xml).map_err(|e| format!("{}: {e}", xml.display()))?;
		let (mut sx, states) = stateless::rewrite(&text)?;
		// Delta-net scaling (`stateless::scale_delta_q`), per layer:
		// `QWEN_OV_DSCALE=kq,kv,ko;kq,kv,ko` gives the exponents for layer 0
		// and for the others (`0` = off). The NPU flushes subnormal products
		// inside the recurrence, and the deeper layers' states are
		// 1e-3-small; the default `8,4,4;6,8,6` lifts them (layer 0's state
		// reaches ~50, hence its smaller value scale): decode-graph KL vs the
		// exact CPU 0.0034 -> 0.0001 over 516 positions. It needs the
		// blocked inverse in the prefill graph (`stateless::chunk_delta`):
		// the Neumann form's intermediates overflow f16 under it (KL 0.43 at
		// 2490 tokens), so with `QWEN_OV_DELTA_INV=neumann` the default is
		// the old `6,0,6` (query x64). The old uniform form:
		// `QWEN_OV_QSCALE=C` (a power of two, the norm-side scale) with
		// `QWEN_OV_VSCALE=V` (the part of it put on the value), used when
		// either is set.
		let pow2 = |name: &str, d: u32| -> Result<u32, String> {
			let v: u32 = std::env::var(name)
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(d)
				.max(1);
			if v.is_power_of_two() {
				Ok(v.trailing_zeros())
			} else {
				Err(format!("{name}={v}: not a power of two"))
			}
		};
		let legacy = std::env::var_os("QWEN_OV_QSCALE").is_some()
			|| std::env::var_os("QWEN_OV_VSCALE").is_some();
		let spec: Option<[stateless::DeltaScale; 2]> = if legacy {
			let (kc, kv) = (pow2("QWEN_OV_QSCALE", 64)?, pow2("QWEN_OV_VSCALE", 1)?);
			let d = stateless::DeltaScale {
				kq: kc as i32 - kv as i32,
				kv,
				ko: kc as i32,
			};
			(kc > 0 || kv > 0).then_some([d, d])
		} else {
			let neumann = std::env::var("QWEN_OV_DELTA_INV").is_ok_and(|v| v == "neumann");
			let v = std::env::var("QWEN_OV_DSCALE")
				.unwrap_or_else(|_| if neumann { "6,0,6" } else { "8,4,4;6,8,6" }.into());
			if v.trim() == "0" {
				None
			} else {
				let parse = |t: &str| -> Option<stateless::DeltaScale> {
					let n: Vec<i32> = t
						.split(',')
						.map(|x| x.trim().parse().ok())
						.collect::<Option<_>>()?;
					match n[..] {
						[kq, kv, ko] if kv >= 0 => Some(stateless::DeltaScale {
							kq,
							kv: kv as u32,
							ko,
						}),
						_ => None,
					}
				};
				let p: Vec<_> = v
					.split(';')
					.map(parse)
					.collect::<Option<_>>()
					.ok_or_else(|| format!("QWEN_OV_DSCALE={v}: expected kq,kv,ko;kq,kv,ko"))?;
				match p[..] {
					[a] => Some([a, a]),
					[a, b] => Some([a, b]),
					_ => return Err(format!("QWEN_OV_DSCALE={v}: expected one or two kq,kv,ko")),
				}
			}
		};
		if let Some(sp) = spec {
			let (x, n) = stateless::scale_delta_q(&sx, bin, |l| sp[(l > 0) as usize])?;
			let f = |d: stateless::DeltaScale| {
				format!(
					"query x2^{}, value / state x2^{}, output x2^{}",
					d.kq, d.kv, d.ko
				)
			};
			eprintln!(
				"  npu: delta-net scaling in {n} layers: layer 0 {}; others {}",
				f(sp[0]),
				f(sp[1])
			);
			sx = x;
		}
		// Conv weight scales out of f16's subnormal range
		// (`stateless::scale_conv_weights`): NPUW's folded layers get them
		// as function inputs, which the NPU flushes. `QWEN_OV_CONVSCALE=0`
		// is the A/B.
		let mut conv_k = 0;
		// `QWEN_OV_CONVSCALE=K` forces the exponent.
		let force_k = std::env::var("QWEN_OV_CONVSCALE")
			.ok()
			.and_then(|v| v.parse::<u32>().ok());
		if force_k != Some(0) {
			let (x, n, k) = stateless::scale_conv_weights(&sx, bin, force_k)?;
			eprintln!("  npu: conv weight scales x2^{k} (out of f16 subnormals) in {n} layers");
			sx = x;
			conv_k = k;
		}
		// `QWEN_OV_ROPE`: `host` computes the rotary tables on the
		// host (see `stateless::rope_inputs`), `graph` keeps the export's
		// in-graph `Cos` / `Sin` (default), `f16sim` feeds host tables with f16-rounded
		// angles (the NPU's error, on another device).
		// `QWEN_OV_PROBES=8,8a,...`: the residual after layer 8 (`8`: after
		// its FFN, `8a`: after its mixer), as extra outputs written to
		// `QWEN_OV_PROBE_DUMP` every step (all positions, f32 x 5120 each,
		// probes in the given order).
		let mut probes = Vec::new();
		if let Ok(p) = std::env::var("QWEN_OV_PROBES") {
			let names: Vec<String> = p
				.split(',')
				.map(str::trim)
				.filter(|s| !s.is_empty())
				.map(|s| {
					// `L:SUFFIX`: any op of layer L by its IR name, e.g.
					// `0:.mlp.down_proj/ov_ext::linear/MatMul`.
					if let Some((l, suf)) = s.split_once(':') {
						return format!("__module.model.model.language_model.layers.{l}{suf}");
					}
					let (l, op) = match s.strip_suffix('a') {
						Some(l) => (l, "Add"),
						None => (s, "Add_1"),
					};
					format!("__module.model.model.language_model.layers.{l}/aten::add/{op}")
				})
				.collect();
			let (x, outs) = stateless::add_probes(&sx, &names)?;
			eprintln!("  npu: {} probes: {p}", outs.len());
			sx = x;
			probes = outs;
		}
		let rope_mode = std::env::var("QWEN_OV_ROPE").unwrap_or_else(|_| "graph".into());
		let rope = match rope_mode.as_str() {
			"graph" => None,
			"host" | "f16sim" => {
				let (x, inv) = stateless::rope_inputs(&sx, bin)?;
				sx = x;
				eprintln!("  npu: rotary cos / sin computed on the host ({rope_mode})");
				Some(Rope {
					inv_freq: inv,
					f16: rope_mode == "f16sim",
				})
			}
			m => return Err(format!("QWEN_OV_ROPE={m}: expected host, graph or f16sim")),
		};
		// Per process: concurrent loads would otherwise overwrite each
		// other's rewritten graph between write and read.
		let tmp = std::env::temp_dir()
			.join("qwen-rs-ov")
			.join(std::process::id().to_string());
		std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
		let sxml = tmp.join("openvino_language_model.stateless.xml");
		std::fs::write(&sxml, &sx).map_err(|e| format!("{}: {e}", sxml.display()))?;
		// The prefill graph: the NPU compiler rejects more than one position
		// through the delta-net Loops, so they are replaced; by the chunked
		// (matmul) form by default, `QWEN_OV_DELTA=unroll` for one step per
		// position, `QWEN_OV_DELTA=loop` keeps them (e.g. for the CPU).
		// `QWEN_OV_DELTA_C`: positions per chunked sub-chunk (default the
		// largest of 32, 16, 8, ... dividing the chunk; with
		// `QWEN_OV_DELTA_INV=neumann` at most 16: 32 gave NaN logits on the
		// NPU there, f16 overflow in the `A^16` power, see `chunk_delta`).
		let mode = std::env::var("QWEN_OV_DELTA").unwrap_or_else(|_| "chunk".into());
		let mut sub = None;
		// `QWEN_OV_PAD=0`: the old remainder handling (decode graph only), as
		// the A/B for right-padded remainders.
		let pad = o.chunk > 1 && std::env::var("QWEN_OV_PAD").map_or(true, |v| v != "0");
		let mut conv_w = None;
		let pad_x;
		let sx_pre: &str = if pad {
			let t = Instant::now();
			let (x, w) = stateless::pad_inputs(&sx, bin, o.chunk, &states)?;
			eprintln!("  [perf] npu: {}-position graph takes right-padded input (conv state width {w}) in {:.1} s", o.chunk, t.elapsed().as_secs_f64());
			conv_w = Some(w);
			pad_x = x;
			&pad_x
		} else {
			pad_x = String::new();
			&sx
		};
		let pxml = if o.chunk > 1 && (mode != "loop" || pad) {
			let t = Instant::now();
			let (px, n, how) = if mode == "loop" {
				(sx_pre.to_string(), 0, "kept".to_string())
			} else if mode == "unroll" {
				let (px, n) = stateless::unroll_delta(sx_pre, bin, o.chunk)?;
				(px, n, format!("unrolled to {} steps", o.chunk))
			} else {
				// `QWEN_OV_DELTA_INV=neumann`: the old `(I + A)^-1` product
				// form, whose intermediates overflow f16 under a value scale.
				let blocked = std::env::var("QWEN_OV_DELTA_INV").map_or(true, |v| v != "neumann");
				let c = match std::env::var("QWEN_OV_DELTA_C")
					.ok()
					.and_then(|v| v.parse::<usize>().ok())
				{
					Some(c) => c,
					None => [32, 16, 8, 4, 2, 1]
						.into_iter()
						.filter(|&c| blocked || c <= 16)
						.find(|c| o.chunk.is_multiple_of(*c))
						.unwrap(),
				};
				let (px, n) = stateless::chunk_delta(sx_pre, bin, o.chunk, c, blocked)?;
				sub = Some((c, blocked));
				(
					px,
					n,
					format!(
						"chunked, {c} positions per sub-chunk, {} inverse",
						if blocked { "blocked" } else { "Neumann" }
					),
				)
			};
			// Logits of one position only (the last real one, all a chunk's
			// caller reads): keeps the head subgraph's NPU blob under 2 GiB,
			// so the graph caches. `QWEN_OV_LAST_LOGITS=0` is the A/B.
			let px = if std::env::var("QWEN_OV_LAST_LOGITS").map_or(true, |v| v != "0") {
				stateless::last_position_logits(&px, bin, o.chunk)?
			} else {
				px
			};
			let p = tmp.join(format!(
				"openvino_language_model.stateless.s{}.{mode}{}{}.xml",
				o.chunk,
				sub.map_or(String::new(), |(c, b)| format!(
					"{c}{}",
					if b { "b" } else { "" }
				)),
				if pad { ".pad" } else { "" }
			));
			std::fs::write(&p, px).map_err(|e| format!("{}: {e}", p.display()))?;
			eprintln!(
				"  [perf] npu: {n} delta-net Loops {how} in {:.1} s",
				t.elapsed().as_secs_f64()
			);
			p
		} else {
			sxml.clone()
		};
		drop(pad_x);
		drop(sx);
		let n_kv = states.iter().filter(|s| s.is_kv()).count();
		eprintln!(
			"  [perf] npu: stateless rewrite in {:.1} s: {} states ({n_kv} KV, {} conv/SSM) -> {}",
			t.elapsed().as_secs_f64(),
			states.len(),
			states.len() - n_kv,
			sxml.display()
		);

		let cap = o.cap;
		let mut kv = Vec::new();
		let mut recur = Vec::new();
		for st in &states {
			let mut d = st.dims.clone();
			d[0] = 1;
			if st.is_kv() {
				d[2] = cap as i64;
				let past = HostTensor::zeros::<u16>(rt, &d)?;
				kv.push(Kv {
					heads: d[1] as usize,
					hd: d[3] as usize,
					st: st.clone(),
					past,
				});
			} else {
				if d.iter().any(|&x| x < 0) {
					return Err(format!(
						"state {} has a dynamic dim besides batch: {:?}",
						st.input, st.dims
					));
				}
				recur.push(Recur {
					cur: HostTensor::zeros::<u16>(rt, &d)?,
					next: HostTensor::zeros::<u16>(rt, &d)?,
					st: st.clone(),
				});
			}
		}

		let mut props: Vec<(&str, &str)> = if device == "NPU" {
			vec![
				("NPU_USE_NPUW", "YES"),
				("NPUW_ONLINE_PIPELINE", "REP"),
				("NPUW_FOLD", "YES"),
				("NPUW_DQ", "YES"),
				("NPUW_FUNCALL_ASYNC", "YES"),
				("NPUW_WEIGHTS_BANK", "shared"),
			]
		} else {
			Vec::new()
		};
		if let Some(c) = o.cache_dir {
			props.push(("CACHE_DIR", c));
		}
		for (k, v) in o.props {
			props.push((k, v));
		}

		// Compiling runs on the host CPU and peaks at 30-47 GB, so it takes a
		// compile slot (at most two at once machine-wide) rather than the NPU
		// lock; the NPU lock is taken once both graphs are compiled, before
		// the first infer request, and held for the engine's lifetime.
		// `QWEN_OV_XML_ONLY=DIR`: write the rewritten graphs there and stop
		// before compiling (the input of the low-level track's layer cuts,
		// `crates/npu/scripts`). Debug only.
		if let Some(d) = std::env::var_os("QWEN_OV_XML_ONLY") {
			let d = Path::new(&d);
			let _ = std::fs::create_dir_all(d);
			for p in [&sxml, &pxml] {
				if let Some(f) = p.file_name() {
					std::fs::copy(p, d.join(f)).map_err(|e| format!("{}: {e}", p.display()))?;
				}
			}
			let _ = std::fs::remove_dir_all(&tmp);
			return Err(format!(
				"QWEN_OV_XML_ONLY: graphs written to {}",
				d.display()
			));
		}
		let npu = device == "NPU";
		let r = rope.is_some();
		let slot = if npu {
			Some(crate::lock::CompileSlot::acquire()?)
		} else {
			None
		};
		// NPUW's f16 interconnect breaks both graphs: the prefill graph
		// fails at infer-request creation with `f32 != f16` on a partition
		// output, the 1-position graph with the conv rescale input
		// (`CONV_UNSCALE`) fails over to the CPU mid-infer with the same
		// mismatch. f32 interconnect runs them at the same speed. Placed
		// before the user's `--ov-prop`s so one can override it.
		let mut props = props;
		if npu {
			props.insert(props.len() - o.props.len(), ("NPUW_F16IC", "NO"));
		}
		let dec = Self::compile(rt, &device, &sxml, bin, &states, 1, cap, r, &props)?;
		let pre = if o.chunk > 1 {
			Some(Self::compile(
				rt, &device, &pxml, bin, &states, o.chunk, cap, r, &props,
			)?)
		} else {
			None
		};
		// `QWEN_OV_KEEP_XML=DIR`: copy the rewritten graphs there first, for
		// the micro-graph scripts (`scripts/npu_layer.py`). Debug only.
		if let Some(d) = std::env::var_os("QWEN_OV_KEEP_XML") {
			let d = Path::new(&d);
			let _ = std::fs::create_dir_all(d);
			for p in [&sxml, &pxml] {
				if let Some(f) = p.file_name() {
					let _ = std::fs::copy(p, d.join(f));
				}
			}
		}
		let _ = std::fs::remove_dir_all(&tmp);
		drop(slot);
		let lock = if npu {
			crate::lock::NpuLock::acquire()?
		} else {
			crate::lock::NpuLock::none()
		};
		let mut dec = Self::stage(rt, dec, &kv, 1, cap, None, None, r)?;
		let mut pre = match pre {
			Some(c) => Some(Self::stage(rt, c, &kv, o.chunk, cap, sub, conv_w, r)?),
			None => None,
		};
		if conv_k > 0 {
			for st in std::iter::once(&mut dec).chain(pre.as_mut()) {
				st.req.set(
					stateless::CONV_UNSCALE,
					&[1, 1, 1],
					&[2f32.powi(-(conv_k as i32))],
				)?;
			}
		}
		// A padded call costs a whole chunk: on the NPU 2.8 s for 32
		// positions, 1.7 s for 16, against 0.63 s per decode step, so
		// padding pays from ~chunk / 7 positions. `QWEN_OV_PAD_MIN` overrides
		// (`1` also sends every decode step through the padded graph: a
		// test of the padding, not a mode).
		let pad_min = match std::env::var("QWEN_OV_PAD_MIN")
			.ok()
			.and_then(|v| v.parse::<usize>().ok())
		{
			Some(m) => m.max(1),
			None => o.chunk.div_ceil(7).max(2),
		};
		let pad_min = if conv_w.is_some() {
			pad_min
		} else {
			usize::MAX
		};
		Ok(NpuLlm {
			dec,
			pre,
			kv,
			recur,
			rope,
			probes,
			cap,
			pos: 0,
			pad_min,
			_lock: lock,
		})
	}

	/// Read, reshape to `s` positions / KV capacity `cap`, and compile.
	#[allow(clippy::too_many_arguments)]
	fn compile(
		rt: &Runtime,
		device: &str,
		xml: &Path,
		bin: &Path,
		states: &[State],
		s: usize,
		cap: usize,
		rope: bool,
		props: &[(&str, &str)],
	) -> Result<Compiled, String> {
		let t = Instant::now();
		let m = rt.read_model_bin(xml, Some(bin))?;
		let (si, p) = (s as i64, cap as i64);
		let mut shapes: Vec<(&str, Vec<i64>)> = vec![
			("inputs_embeds", vec![1, si, HIDDEN]),
			("attention_mask", vec![1, p + si]),
			("position_ids", vec![4, 1, si]),
			("beam_idx", vec![1]),
		];
		if rope {
			shapes.push((stateless::ROPE_COS, vec![1, si, 64]));
			shapes.push((stateless::ROPE_SIN, vec![1, si, 64]));
		}
		for st in states {
			let mut d = st.dims.clone();
			d[0] = 1;
			if st.is_kv() {
				d[2] = p;
			}
			shapes.push((&st.input, d));
		}
		m.reshape(&shapes)?;
		let ins: Vec<&str> = states.iter().map(|s| s.input.as_str()).collect();
		let outs: Vec<&str> = states.iter().map(|s| s.output.as_str()).collect();
		let m = m.with_f16_io(&ins, &outs)?;
		eprintln!("  npu: compiling the {s}-position graph for {device} (KV capacity {cap}); a first NPU compile takes ~5 min");
		let c = rt.compile(&m, device, props)?;
		eprintln!(
			"  [perf] npu: {s}-position graph compiled in {:.1} s",
			t.elapsed().as_secs_f64()
		);
		Ok(c)
	}

	/// The infer request of a compiled graph, with its host tensors bound.
	fn stage(
		rt: &Runtime,
		c: Compiled,
		kv: &[Kv],
		s: usize,
		cap: usize,
		sub: Option<(usize, bool)>,
		conv_w: Option<usize>,
		rope: bool,
	) -> Result<Stage, String> {
		let t = Instant::now();
		let (si, p) = (s as i64, cap as i64);
		let req = c.request()?;
		eprintln!(
			"  [perf] npu: {s}-position infer request in {:.1} s",
			t.elapsed().as_secs_f64()
		);
		let mut kv_out = Vec::new();
		for k in kv {
			let mut d = k.past.dims.clone();
			d[2] = p + si;
			kv_out.push(HostTensor::zeros::<u16>(rt, &d)?);
		}
		let mut st = Stage {
			req,
			_c: c,
			s,
			x: HostTensor::zeros::<f32>(rt, &[1, si, HIDDEN])?,
			mask: HostTensor::zeros::<i64>(rt, &[1, p + si])?,
			pos: HostTensor::zeros::<i64>(rt, &[4, 1, si])?,
			beam: None,
			rope: if rope {
				Some((
					HostTensor::zeros::<f32>(rt, &[1, si, 64])?,
					HostTensor::zeros::<f32>(rt, &[1, si, 64])?,
				))
			} else {
				None
			},
			pad: None,
			last: None,
			kv_out,
		};
		if let Some((c, sn)) = &st.rope {
			st.req.bind(stateless::ROPE_COS, c)?;
			st.req.bind(stateless::ROPE_SIN, sn)?;
		}
		st.req.bind("inputs_embeds", &st.x)?;
		st.req.bind("attention_mask", &st.mask)?;
		st.req.bind("position_ids", &st.pos)?;
		let beam = HostTensor::zeros::<i32>(rt, &[1])?;
		if st.req.bind("beam_idx", &beam).is_ok() {
			st.beam = Some(beam);
		}
		// The chunked delta-net's constant masks.
		if let Some((c, blocked)) = sub {
			let (strict, incl) = stateless::tril_masks(c);
			let d = [c as i64, c as i64];
			st.req.set(stateless::TRIL_STRICT, &d, &strict)?;
			st.req.set(stateless::TRIL_INCL, &d, &incl)?;
			if blocked {
				for (name, m) in stateless::blk_masks(c) {
					st.req.set(&name, &d, &m)?;
				}
			}
		}
		let mut last = HostTensor::zeros::<f32>(rt, &[1, si, 1])?;
		last.as_mut::<f32>()[s - 1] = 1.0;
		if st.req.bind(stateless::LAST_ONEHOT, &last).is_ok() {
			st.last = Some(last);
		}
		if let Some(w) = conv_w {
			let mut valid = HostTensor::zeros::<f32>(rt, &[1, 1, si])?;
			let mut keep = HostTensor::zeros::<i32>(rt, &[w as i64])?;
			let (v, k) = stateless::pad_values(s, s, w);
			valid.as_mut::<f32>().copy_from_slice(&v);
			keep.as_mut::<i32>().copy_from_slice(&k);
			st.req.bind(stateless::VALID, &valid)?;
			st.req.bind(stateless::CONV_KEEP, &keep)?;
			st.pad = Some((valid, keep, w));
		}
		for (k, out) in kv.iter().zip(&st.kv_out) {
			st.req.bind(&k.st.input, &k.past)?;
			st.req.bind(&k.st.output, out)?;
		}
		Ok(st)
	}

	/// Start a new conversation.
	pub fn reset(&mut self) {
		self.pos = 0;
		for r in &mut self.recur {
			r.cur.as_mut::<u16>().fill(0);
		}
		// Past KV slots beyond `pos` are masked, so they need no clearing.
	}

	/// Feed `emb` (`n` x 5120 embeddings) after everything fed so far;
	/// returns the logits of the last position.
	pub fn forward(&mut self, emb: &[f32]) -> Result<Vec<f32>, String> {
		let d = HIDDEN as usize;
		let n = emb.len() / d;
		if n == 0 {
			return Err("forward: no positions".into());
		}
		if self.pos + n > self.cap {
			return Err(format!(
				"context full: {} + {n} tokens > {} (--ov-ctx)",
				self.pos, self.cap
			));
		}
		let chunk = self.pre.as_ref().map_or(usize::MAX, |p| p.s);
		let mut i = 0;
		// `QWEN_OV_STATE_LOAD=DIR@POS` with `QWEN_OV_STATE_LOAD_SKIP=1`: the
		// first `POS` positions of a new conversation are not run at all;
		// the loaded states stand in for them (see `crate::dbg`).
		if self.pos == 0 && std::env::var("QWEN_OV_STATE_LOAD_SKIP").is_ok_and(|v| v == "1") {
			if let Some((_, at)) = crate::dbg::dir_at("QWEN_OV_STATE_LOAD") {
				if at < n {
					eprintln!("  npu: skipping the first {at} positions (their states are loaded)");
					i = at;
					self.pos = at;
				}
			}
		}
		let mut last = Vec::new();
		// (calls, seconds) through the prefill graph (whole, padded) and the
		// decode graph
		let mut t = [(0, 0.0); 3];
		let mut padded = 0;
		while i < n {
			let rem = n - i;
			let (take, k) = if rem >= chunk {
				(chunk, 0)
			} else if rem >= self.pad_min {
				padded += chunk - rem;
				(rem, 1)
			} else {
				(1, 2)
			};
			let t0 = Instant::now();
			last = self.step(&emb[i * d..(i + take) * d], k < 2)?;
			t[k] = (t[k].0 + 1, t[k].1 + t0.elapsed().as_secs_f64());
			i += take;
		}
		if t[0].0 + t[1].0 > 0 {
			eprintln!(
				"  [perf] npu prefill: {} x {chunk} positions in {:.2} s ({:.3} s each), {} padded ({padded} filler) in {:.2} s, {} single in {:.2} s",
				t[0].0,
				t[0].1,
				t[0].1 / t[0].0.max(1) as f64,
				t[1].0,
				t[1].1,
				t[2].0,
				t[2].1
			);
		}
		Ok(last)
	}

	/// Run `emb` (`n` positions) through the prefill graph (`pre`; `n` up to
	/// its size, right-padded) or the decode graph (`n` = 1).
	fn step(&mut self, emb: &[f32], pre: bool) -> Result<Vec<f32>, String> {
		let stage = if pre {
			self.pre.as_mut().unwrap()
		} else {
			&mut self.dec
		};
		let s = stage.s;
		let d = HIDDEN as usize;
		let n = emb.len() / d;
		let (cap, pos) = (self.cap, self.pos);
		if let Some((valid, keep, w)) = stage.pad.as_mut() {
			// Rewritten only when the real length changes.
			if valid.as_ref::<f32>().iter().filter(|&&v| v != 0.0).count() != n {
				let (v, k) = stateless::pad_values(s, n, *w);
				valid.as_mut::<f32>().copy_from_slice(&v);
				keep.as_mut::<i32>().copy_from_slice(&k);
			}
		} else if n != s {
			return Err(format!(
				"{n} positions into the {s}-position graph, which takes no padding"
			));
		}
		// Filler: the last real position again (finite, ordinary values;
		// its outputs are dropped).
		let x = stage.x.as_mut::<f32>();
		x[..n * d].copy_from_slice(emb);
		for j in n..s {
			x.copy_within((n - 1) * d..n * d, j * d);
		}
		if let Some(t) = stage.last.as_mut() {
			let t = t.as_mut::<f32>();
			if t[n - 1] != 1.0 {
				t.fill(0.0);
				t[n - 1] = 1.0;
			}
		}
		let m = stage.mask.as_mut::<i64>();
		m[..pos].fill(1);
		m[pos..cap].fill(0);
		m[cap..].fill(1);
		let pids = stage.pos.as_mut::<i64>();
		for r in 0..4 {
			for j in 0..s {
				pids[r * s + j] = (pos + j) as i64;
			}
		}
		if let (Some(r), Some((c, sn))) = (&self.rope, stage.rope.as_mut()) {
			let (cv, sv) = stateless::rope_tables(&r.inv_freq, pos, s, r.f16);
			c.as_mut::<f32>().copy_from_slice(&cv);
			sn.as_mut::<f32>().copy_from_slice(&sv);
		}
		if let Some((dir, at)) = crate::dbg::dir_at("QWEN_OV_STATE_LOAD") {
			if at == pos {
				let mut n = 0;
				let all = self
					.kv
					.iter_mut()
					.map(|k| (&k.st, &mut k.past))
					.chain(self.recur.iter_mut().map(|r| (&r.st, &mut r.cur)));
				for (st, t) in all {
					if !crate::dbg::load_selected(&st.input) {
						continue;
					}
					let f = dir.join(format!("{}.f16", st.input));
					let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
					let dst = t.as_mut::<u16>();
					if b.len() != dst.len() * 2 {
						return Err(format!(
							"{}: {} bytes, expected {}",
							f.display(),
							b.len(),
							dst.len() * 2
						));
					}
					for (d, c) in dst.iter_mut().zip(b.chunks_exact(2)) {
						*d = u16::from_le_bytes([c[0], c[1]]);
					}
					n += 1;
				}
				eprintln!(
					"  npu: loaded {n} states from {} at position {pos}",
					dir.display()
				);
			}
		}
		for r in &self.recur {
			stage.req.bind(&r.st.input, &r.cur)?;
			stage.req.bind(&r.st.output, &r.next)?;
		}
		stage.req.infer()?;
		// KV: append the real positions' rows (at index `cap..cap + n` of
		// `present`).
		for (k, out) in self.kv.iter_mut().zip(&stage.kv_out) {
			let (h, hd) = (k.heads, k.hd);
			let src = out.as_ref::<u16>();
			let dst = k.past.as_mut::<u16>();
			for head in 0..h {
				let so = (head * (cap + s) + cap) * hd;
				let dof = (head * cap + pos) * hd;
				dst[dof..dof + n * hd].copy_from_slice(&src[so..so + n * hd]);
			}
		}
		for r in &mut self.recur {
			std::mem::swap(&mut r.cur, &mut r.next);
		}
		self.pos += n;
		if let Some((dir, at)) = crate::dbg::dir_at("QWEN_OV_STATE_SAVE") {
			if at == self.pos {
				std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
				let all = self
					.kv
					.iter()
					.map(|k| (&k.st, &k.past))
					.chain(self.recur.iter().map(|r| (&r.st, &r.cur)));
				for (st, t) in all {
					let x = t.as_ref::<u16>();
					let f = dir.join(format!("{}.f16", st.input));
					let b: Vec<u8> = x.iter().flat_map(|h| h.to_le_bytes()).collect();
					std::fs::write(&f, b).map_err(|e| format!("{}: {e}", f.display()))?;
					if !st.is_kv() {
						let (rms, mx, z, sub, nf) = crate::dbg::f16_stats(x);
						eprintln!("  STATE {} rms={rms:.4e} max={mx:.4e} zero={z} subnormal={sub} nonfinite={nf} of {}", st.input, x.len());
					}
				}
				eprintln!(
					"  npu: saved the states at position {} to {}",
					self.pos,
					dir.display()
				);
			}
		}
		let (shape, l) = stage.req.output_f32("logits")?;
		let v = *shape.last().unwrap_or(&0) as usize;
		if crate::dbg::tf_enabled() && v > 0 {
			// The real positions' rows (one row when the graph computes
			// only the last position's logits).
			crate::dbg::tf_dump(&l[..l.len().min(n * v)]);
		}
		for p in &self.probes {
			let (shape, x) = stage.req.output_f32(p)?;
			if self.pos == n {
				eprintln!("  npu: probe {p} shape {shape:?}");
			}
			crate::dbg::probe_dump(x);
		}
		if stage.last.is_some() && v > 0 && l.len() == v {
			return Ok(l.to_vec());
		}
		if v == 0 || l.len() != s * v {
			return Err(format!("logits: unexpected shape {shape:?}"));
		}
		Ok(l[(n - 1) * v..n * v].to_vec())
	}
}
