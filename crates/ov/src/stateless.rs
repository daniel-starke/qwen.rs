//! Stateful -> stateless rewrite of an IR's XML, so the NPU can run it.
//!
//! The export keeps its KV cache and its conv / SSM state in 128 state
//! variables (`ReadValue` -> … -> `Assign`). The NPU compiler only takes
//! static shapes, and a variable's shape is not an input that can be
//! reshaped. OpenVINO's NPUW LLM mode does this conversion itself, and fails
//! on this architecture (see the crate docs). The C API has no
//! `stateful_to_stateless` pass, so it is done here on the XML:
//!
//! - each top-level `ReadValue` becomes a `Parameter` named after its output
//!   (`cache_params.past.ssm.0`, …), and the edges into it (its initializer
//!   subgraph) are dropped;
//! - each `Assign` becomes a `Result` named after the variable's "present"
//!   half (`cache_params.present.ssm.0`).
//!
//! The rewritten XML is written beside nothing: it goes to a temp file and is
//! read with the export's **original** `.bin`, since no constant moved.

use std::collections::HashMap;

/// One state variable as a pair of model ports.
#[derive(Clone, Debug)]
pub struct State {
	/// Input carrying the state in (`cache_params.past.*`).
	pub input: String,
	/// Output carrying the state out (`cache_params.present.*`).
	pub output: String,
	/// Dims with `-1` for dynamic (batch, and the KV sequence length).
	pub dims: Vec<i64>,
}

impl State {
	pub fn is_kv(&self) -> bool {
		self.input.contains(".key.") || self.input.contains(".value.")
	}
}

fn attr<'a>(line: &'a str, name: &str) -> Option<&'a str> {
	let k = format!(" {name}=\"");
	let s = line.find(&k)? + k.len();
	let e = line[s..].find('"')? + s;
	Some(&line[s..e])
}

/// Rewrite `xml`; returns the new XML and the states in file order.
pub fn rewrite(xml: &str) -> Result<(String, Vec<State>), String> {
	let lines: Vec<&str> = xml.split_inclusive('\n').collect();
	let mut out = String::with_capacity(xml.len());
	let mut states = Vec::new();
	// layer id -> "rv" | "as"
	let mut kind: HashMap<String, &str> = HashMap::new();
	let mut i = 0;
	// Top-level layers and edges sit at exactly two tabs; Loop bodies nest
	// deeper and are left alone.
	let top = |l: &str, tag: &str| {
		l.starts_with(&format!("\t\t<{tag}")) && !l.starts_with(&"\t\t\t".to_string())
	};
	while i < lines.len() {
		let l = lines[i];
		if top(l, "layer ") && (l.contains("type=\"ReadValue\"") || l.contains("type=\"Assign\"")) {
			let id = attr(l, "id").ok_or("layer without id")?.to_string();
			let is_rv = l.contains("type=\"ReadValue\"");
			let mut j = i + 1;
			while j < lines.len() && !lines[j].starts_with("\t\t</layer>") {
				j += 1;
			}
			let block = &lines[i..=j];
			let var = block
				.iter()
				.find_map(|b| attr(b, "variable_id"))
				.ok_or_else(|| format!("state layer {id} without variable_id"))?;
			// Output-port dims (ReadValue) / input-port dims (Assign).
			let section = if is_rv { "<output>" } else { "<input>" };
			let s = block
				.iter()
				.position(|b| b.contains(section))
				.ok_or("state layer without ports")?;
			let mut dims = Vec::new();
			let mut prec = "FP32".to_string();
			for b in &block[s..] {
				if let Some(p) = attr(b, "precision") {
					prec = p.to_string();
				}
				if let Some(d) = b
					.trim()
					.strip_prefix("<dim>")
					.and_then(|r| r.strip_suffix("</dim>"))
				{
					dims.push(d.parse::<i64>().map_err(|e| format!("dim {d}: {e}"))?);
				}
				if b.contains("</port>") {
					break;
				}
			}
			let dims_xml: String = dims
				.iter()
				.map(|d| format!("\t\t\t\t\t<dim>{d}</dim>\n"))
				.collect();
			let et = match prec.as_str() {
				"FP32" => "f32",
				"FP16" => "f16",
				p => return Err(format!("state {var}: unsupported precision {p}")),
			};
			// `variable_id` is "<past name><present name>".
			let (past, pres) = match var.find("cache_params.present.") {
				Some(p) if p > 0 => (&var[..p], &var[p..]),
				_ => return Err(format!("cannot split variable id {var}")),
			};
			if is_rv {
				let shape: Vec<String> = dims
					.iter()
					.map(|d| if *d < 0 { "?".into() } else { d.to_string() })
					.collect();
				out.push_str(&format!(
					"\t\t<layer id=\"{id}\" name=\"{past}\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"{}\" element_type=\"{et}\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"{prec}\" names=\"{past}\">\n{dims_xml}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
					shape.join(",")
				));
				kind.insert(id, "rv");
				states.push(State {
					input: past.to_string(),
					output: pres.to_string(),
					dims,
				});
			} else {
				out.push_str(&format!(
					"\t\t<layer id=\"{id}\" name=\"{pres}/result\" type=\"Result\" version=\"opset1\" output_names=\"{pres}\">\n\t\t\t<input>\n\t\t\t\t<port id=\"0\" precision=\"{prec}\">\n{dims_xml}\t\t\t\t</port>\n\t\t\t</input>\n\t\t</layer>\n"
				));
				kind.insert(id, "as");
			}
			i = j + 1;
			continue;
		}
		if top(l, "edge ") {
			let (Some(fl), Some(tl)) = (attr(l, "from-layer"), attr(l, "to-layer")) else {
				return Err(format!("bad edge: {l}"));
			};
			match (kind.get(fl).copied(), kind.get(tl).copied()) {
				// The initializer feeding a ReadValue, or an Assign's
				// (unused) output: gone with the variable.
				(_, Some("rv")) | (Some("as"), _) => {}
				(Some("rv"), _) => {
					out.push_str(&l.replacen(" from-port=\"1\"", " from-port=\"0\"", 1))
				}
				_ => out.push_str(l),
			}
			i += 1;
			continue;
		}
		out.push_str(l);
		i += 1;
	}
	// Pair every ReadValue with an Assign.
	let n_as = kind.values().filter(|k| **k == "as").count();
	if n_as != states.len() {
		return Err(format!(
			"{} ReadValue but {n_as} Assign layers",
			states.len()
		));
	}
	Ok((out, states))
}

/// A top-level layer of the IR.
struct Layer {
	/// Lines `[start, end]` of the layer's block.
	start: usize,
	end: usize,
	id: String,
	name: String,
	ty: String,
	/// `(element_type, offset, size)` of a `Const`.
	data: Option<(String, u64, u64)>,
	/// Dims of the first output port.
	out_dims: Vec<i64>,
}

struct Edge {
	line: usize,
	from: (String, String),
	to: (String, String),
}

fn dims_xml(d: &[i64]) -> String {
	d.iter()
		.map(|d| format!("\t\t\t\t\t<dim>{d}</dim>\n"))
		.collect()
}

fn mul_layer(id: usize, name: &str, a: &[i64], b: &[i64], out: &[i64]) -> String {
	format!(
		"\t\t<layer id=\"{id}\" name=\"{name}\" type=\"Multiply\" version=\"opset1\">\n\t\t\t<data auto_broadcast=\"numpy\" />\n\t\t\t<input>\n\t\t\t\t<port id=\"0\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t\t<port id=\"1\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t</input>\n\t\t\t<output>\n\t\t\t\t<port id=\"2\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
		dims_xml(a),
		dims_xml(b),
		dims_xml(out)
	)
}

fn edge(from: (&str, &str), to: (&str, &str)) -> String {
	format!(
		"\t\t<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{}\" to-port=\"{}\" />\n",
		from.0, from.1, to.0, to.1
	)
}

fn read_f32(bin: &std::path::Path, offset: u64) -> Result<f32, String> {
	use std::io::{Read, Seek, SeekFrom};
	let mut f = std::fs::File::open(bin).map_err(|e| format!("{}: {e}", bin.display()))?;
	f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
	let mut b = [0u8; 4];
	f.read_exact(&mut b).map_err(|e| e.to_string())?;
	Ok(f32::from_le_bytes(b))
}

/// The top level of an IR: its layers and edges, and the lines of
/// `</layers>` / `</edges>`, where new ones are inserted.
struct Ir<'a> {
	lines: Vec<&'a str>,
	layers: Vec<Layer>,
	edges: Vec<Edge>,
	layers_end: usize,
	edges_end: usize,
}

fn parse(xml: &str) -> Result<Ir<'_>, String> {
	let lines: Vec<&str> = xml.split_inclusive('\n').collect();
	let top =
		|l: &str, tag: &str| l.starts_with(&format!("\t\t<{tag}")) && !l.starts_with("\t\t\t");
	let mut layers: Vec<Layer> = Vec::new();
	let mut edges: Vec<Edge> = Vec::new();
	let (mut layers_end, mut edges_end) = (None, None);
	let mut i = 0;
	while i < lines.len() {
		let l = lines[i];
		if l.starts_with("\t</layers>") {
			layers_end = Some(i);
		} else if l.starts_with("\t</edges>") {
			edges_end = Some(i);
		} else if top(l, "layer ") {
			let id = attr(l, "id").ok_or("layer without id")?.to_string();
			let name = attr(l, "name").unwrap_or("").to_string();
			let ty = attr(l, "type").unwrap_or("").to_string();
			let (mut data, mut out_dims, mut in_out, mut seen_out) =
				(None, Vec::new(), false, false);
			let mut j = i;
			if !l.trim_end().ends_with("/>") {
				j += 1;
				while j < lines.len() && !lines[j].starts_with("\t\t</layer>") {
					let b = lines[j];
					if b.contains("<data ") && ty == "Const" {
						let n = |k: &str| attr(b, k).and_then(|v| v.parse::<u64>().ok());
						if let (Some(et), Some(o), Some(s)) =
							(attr(b, "element_type"), n("offset"), n("size"))
						{
							data = Some((et.to_string(), o, s));
						}
					}
					// The layer's own first output port (a Loop's body
					// nests deeper and has outputs of its own).
					if b.starts_with("			<output>") && !seen_out {
						(in_out, seen_out) = (true, true);
					} else if in_out && b.contains("</port>") {
						in_out = false;
					} else if in_out {
						if let Some(d) = b
							.trim()
							.strip_prefix("<dim>")
							.and_then(|r| r.strip_suffix("</dim>"))
						{
							out_dims.push(d.parse::<i64>().map_err(|e| format!("dim {d}: {e}"))?);
						}
					}
					j += 1;
				}
			}
			layers.push(Layer {
				start: i,
				end: j,
				id,
				name,
				ty,
				data,
				out_dims,
			});
			i = j + 1;
			continue;
		} else if top(l, "edge ") {
			let g = |k: &str| {
				attr(l, k)
					.map(str::to_string)
					.ok_or_else(|| format!("bad edge: {l}"))
			};
			edges.push(Edge {
				line: i,
				from: (g("from-layer")?, g("from-port")?),
				to: (g("to-layer")?, g("to-port")?),
			});
		}
		i += 1;
	}
	let (Some(layers_end), Some(edges_end)) = (layers_end, edges_end) else {
		return Err("IR without top-level </layers> / </edges>".into());
	};
	Ok(Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	})
}

/// Per-layer exponents of [`scale_delta_q`]: query `2^kq`, value (and so
/// state) `2^kv`, recurrence output as seen by the norm `2^ko`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaScale {
	pub kq: i32,
	pub kv: u32,
	pub ko: i32,
}

/// Lift the gated delta-net output out of f16's subnormal range.
///
/// Each linear-attention layer's recurrence output is linear in its query,
/// and its only reader is a gated RMSNorm over the 128-wide head. On the
/// first tokens of a conversation some heads put out values of ~1e-5, which
/// an f16 device (the NPU) flushes to zero before the norm scales them back
/// up to O(1): layer 0 alone came out 34 % off (relative RMS) against the
/// CPU. RMSNorm is scale-invariant, so the query is multiplied by `c = 2^k`
/// and the norm's eps by `c^2`: the same function in exact arithmetic,
/// computed where f16 is normal. (Layer 0's error: 34 % -> 0.4 %.)
///
/// The rewrite reuses the export's `.bin`, which has no `c`: a new `Const`
/// points at the bytes of the norm's existing `2.0` exponent (checked by
/// reading them), and `c` and `c^2` are products of it that constant folding
/// removes. Returns the XML and the number of layers rescaled.
///
/// `kv > 0` moves part of the scale onto the **value** instead (the query
/// gets `2^kq`, the value `2^kv`, eps `2^(2 (kq + kv))`): the recurrence is
/// linear in (state, value), so the state (and the f16 state ports)
/// then carry `2^kv S`. The state of the deeper layers has an RMS of only
/// ~3e-3 to 3e-2 after a few hundred positions, with 5-40 % of its entries
/// in f16's subnormal range, which the NPU flushes to zero.
///
/// The scales are per layer ([`DeltaScale`], from `scale(layer index)`), and
/// `ko` puts an output scale on the Loop: the recurrence runs on `2^kq q` and
/// `2^kv v` (so on `2^kv S`), its output is multiplied by `2^(ko - kq - kv)`
/// and the norm's eps by `2^(2 ko)`. That separates two needs: the NPU
/// flushes subnormal **products** inside the recurrence's reductions
/// (`q_k S_kv`, `k_k S_kv`, `k_k d_v`), which wants `q` and `S` large, while
/// the gated norm squares its input in f16, which wants the output small.
pub fn scale_delta_q(
	xml: &str,
	bin: &std::path::Path,
	scale: impl Fn(usize) -> DeltaScale,
) -> Result<(String, usize), String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let by_id: HashMap<&str, &Layer> = layers.iter().map(|l| (l.id.as_str(), l)).collect();
	let by_name: HashMap<&str, &Layer> = layers.iter().map(|l| (l.name.as_str(), l)).collect();
	let input_from = |id: &str, port: &str| edges.iter().find(|e| e.to.0 == id && e.to.1 == port);
	// An f32 `Const` feeding `id`'s `port`, and its value.
	let const_in = |id: &str, port: &str| -> Result<(&Layer, f32), String> {
		let e = input_from(id, port).ok_or_else(|| format!("layer {id} has no input {port}"))?;
		let c = by_id
			.get(e.from.0.as_str())
			.ok_or("edge from an unknown layer")?;
		match &c.data {
			Some((et, off, 4)) if c.ty == "Const" && et == "f32" => Ok((c, read_f32(bin, *off)?)),
			_ => Err(format!(
				"{}: input {port} is not an f32 scalar Const",
				by_id[id].name
			)),
		}
	};
	let mut next_id = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let mut new_layers = String::new();
	let mut new_edges = String::new();
	let mut replaced: HashMap<usize, String> = HashMap::new();
	let mut two: Option<()> = None;
	// pows[e - 1]: the source port of 2^e.
	let mut pows: Vec<(String, String)> = Vec::new();
	let mut n = 0;
	let s11 = [1i64, 1];
	for q in layers
		.iter()
		.filter(|l| l.ty == "Divide" && l.name.ends_with("linear_attn/aten::mul/Multiply_3"))
	{
		let base = &q.name[..q.name.len() - "/aten::mul/Multiply_3".len()];
		let pow = by_name
			.get(format!("{base}.norm/aten::pow/Power").as_str())
			.ok_or_else(|| format!("{base}: no norm Power"))?;
		let add = by_name
			.get(format!("{base}.norm/aten::add/Add").as_str())
			.ok_or_else(|| format!("{base}: no norm Add"))?;
		let (eps_c, eps) = const_in(&add.id, "1")?;
		if !(eps > 0.0 && eps < 1e-3) {
			return Err(format!("{base}: norm eps {eps} does not look like an eps"));
		}
		let to_loop: Vec<&Edge> = edges
			.iter()
			.filter(|e| {
				e.from.0 == q.id && by_id.get(e.to.0.as_str()).is_some_and(|t| t.ty == "Loop")
			})
			.collect();
		if to_loop.len() != 1 {
			return Err(format!(
				"{base}: query feeds {} Loops, expected 1",
				to_loop.len()
			));
		}
		if two.is_none() {
			let (pc, v) = const_in(&pow.id, "1")?;
			if v != 2.0 {
				return Err(format!("{base}: norm exponent is {v}, expected 2"));
			}
			let Some((_, off, _)) = &pc.data else {
				unreachable!()
			};
			let tid = next_id;
			next_id += 1;
			new_layers.push_str(&format!(
				"\t\t<layer id=\"{tid}\" name=\"qwen_rs.qscale.two\" type=\"Const\" version=\"opset1\">\n\t\t\t<data element_type=\"f32\" shape=\"1, 1\" offset=\"{off}\" size=\"4\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
				dims_xml(&s11)
			));
			// 2^1 .. 2^max as a chain of products.
			let max = (0..64)
				.map(&scale)
				.map(|d| {
					d.kq.unsigned_abs()
						.max(d.kv)
						.max(2 * d.ko.max(0) as u32)
						.max((d.ko - d.kq - d.kv as i32).unsigned_abs())
				})
				.max()
				.unwrap_or(1)
				.max(1);
			pows.push((tid.to_string(), "0".to_string()));
			for step in 2..=max {
				let id = next_id;
				next_id += 1;
				new_layers.push_str(&mul_layer(
					id,
					&format!("qwen_rs.qscale.pow{step}"),
					&s11,
					&s11,
					&s11,
				));
				let (pid, pport) = pows.last().unwrap().clone();
				new_edges.push_str(&edge((&pid, &pport), (&id.to_string(), "0")));
				new_edges.push_str(&edge((&tid.to_string(), "0"), (&id.to_string(), "1")));
				pows.push((id.to_string(), "2".to_string()));
			}
			two = Some(());
		}
		// 2^e (e >= 1) as a source port.
		let p2 = |e: u32| -> (&str, &str) {
			let (a, b) = &pows[e as usize - 1];
			(a.as_str(), b.as_str())
		};
		let e = to_loop[0];
		let li: usize = base
			.rsplit_once(".layers.")
			.and_then(|(_, r)| r.split('.').next())
			.and_then(|i| i.parse().ok())
			.ok_or_else(|| format!("{base}: no layer index"))?;
		let DeltaScale { kq, kv, ko } = scale(li);
		// query * 2^kq (or / 2^-kq), into the Loop in place of the query.
		if kq != 0 {
			let mq = next_id;
			next_id += 1;
			let mut x = mul_layer(
				mq,
				&format!("{base}/qwen_rs.qscale"),
				&q.out_dims,
				&s11,
				&q.out_dims,
			);
			if kq < 0 {
				x = x.replacen("type=\"Multiply\"", "type=\"Divide\"", 1);
			}
			new_layers.push_str(&x);
			new_edges.push_str(&edge((&q.id, &e.from.1), (&mq.to_string(), "0")));
			new_edges.push_str(&edge(p2(kq.unsigned_abs()), (&mq.to_string(), "1")));
			replaced.insert(e.line, edge((&mq.to_string(), "2"), (&e.to.0, &e.to.1)));
		}
		// value * 2^kv, into the Loop's input 4: the recurrent state (and
		// so the state ports) carry `2^kv S`.
		if kv > 0 {
			let ve =
				input_from(&e.to.0, "4").ok_or_else(|| format!("{base}: Loop without input 4"))?;
			let vl = by_id
				.get(ve.from.0.as_str())
				.ok_or("edge from an unknown layer")?;
			let mv = next_id;
			next_id += 1;
			new_layers.push_str(&mul_layer(
				mv,
				&format!("{base}/qwen_rs.vscale"),
				&vl.out_dims,
				&s11,
				&vl.out_dims,
			));
			new_edges.push_str(&edge((&ve.from.0, &ve.from.1), (&mv.to_string(), "0")));
			new_edges.push_str(&edge(p2(kv), (&mv.to_string(), "1")));
			replaced.insert(ve.line, edge((&mv.to_string(), "2"), (&ve.to.0, &ve.to.1)));
		}
		// Loop output * 2^(ko - kq - kv), in front of every reader of port 9.
		let kd = ko - kq - kv as i32;
		if kd != 0 {
			let lp = by_id.get(e.to.0.as_str()).unwrap();
			let md = next_id;
			next_id += 1;
			let mut x = mul_layer(
				md,
				&format!("{base}/qwen_rs.oscale"),
				&lp.out_dims,
				&s11,
				&lp.out_dims,
			);
			if kd < 0 {
				x = x.replacen("type=\"Multiply\"", "type=\"Divide\"", 1);
			}
			new_layers.push_str(&x);
			new_edges.push_str(&edge((&lp.id, "9"), (&md.to_string(), "0")));
			new_edges.push_str(&edge(p2(kd.unsigned_abs()), (&md.to_string(), "1")));
			for r in edges
				.iter()
				.filter(|r| r.from.0 == lp.id && r.from.1 == "9")
			{
				replaced.insert(r.line, edge((&md.to_string(), "2"), (&r.to.0, &r.to.1)));
			}
		}
		// eps * 2^(2 ko), into the norm's Add in place of eps.
		let ke = 2 * ko;
		if ke <= 0 {
			n += 1;
			continue;
		}
		let me = next_id;
		next_id += 1;
		new_layers.push_str(&mul_layer(
			me,
			&format!("{base}.norm/qwen_rs.epsscale"),
			&eps_c.out_dims,
			&s11,
			&eps_c.out_dims,
		));
		new_edges.push_str(&edge((&eps_c.id, "0"), (&me.to_string(), "0")));
		new_edges.push_str(&edge(p2(ke as u32), (&me.to_string(), "1")));
		let ae = input_from(&add.id, "1").unwrap();
		replaced.insert(ae.line, edge((&me.to_string(), "2"), (&add.id, "1")));
		n += 1;
	}

	let mut out = String::with_capacity(xml.len() + new_layers.len() + new_edges.len());
	for (i, l) in lines.iter().enumerate() {
		if i == layers_end {
			out.push_str(&new_layers);
		}
		if i == edges_end {
			out.push_str(&new_edges);
		}
		match replaced.get(&i) {
			Some(r) => out.push_str(r),
			None => out.push_str(l),
		}
	}
	Ok((out, n))
}

/// The `2^-k` input of [`scale_conv_weights`], `[1, 1, 1]` f32.
pub const CONV_UNSCALE: &str = "qwen_rs.conv_unscale";

/// Lift the conv1d weight scales out of f16's subnormal range.
///
/// Each linear-attention layer's depthwise conv weight is int8 asymmetric
/// with an f16 scale per channel, and ~1.5 % of those scales are below
/// 2^-14 (down to 3e-6). NPUW folds layers 7-63 into one repeated function
/// whose per-layer constants become function inputs ("closures"), and there
/// those channels' weights come out zero: on the CPU, flushing the
/// subnormal scales reproduces the NPU's per-layer error to three digits
/// (layer 8: 19 % of its residual increment). Layers 0-7 (not folded) are
/// unaffected. The conv is linear and per channel, so the scales are
/// multiplied by `2^k` and the conv output by `2^-k`: exact. The `2^-k`
/// must be a runtime input ([`CONV_UNSCALE`]): as a constant (a Divide by
/// `2^k`) the NPU compiler folds it back into the weight scales and nothing
/// changes. `k` is the
/// smallest that makes every layer's nonzero scales normal, capped at 5
/// (scales from 1.9e-6 up; this export goes down to 1.2e-7, in a few
/// channels of a few layers), the same in every layer (NPUW asserts on
/// repeated blocks that differ); layer 0 is left alone. `2^k` is built from the norm's `2.0`
/// exponent as in [`scale_delta_q`]. Returns the XML, the number of layers
/// rescaled and `k` (0: nothing to do, no input added); `force_k` overrides
/// the computed `k`.
pub fn scale_conv_weights(
	xml: &str,
	bin: &std::path::Path,
	force_k: Option<u32>,
) -> Result<(String, usize, u32), String> {
	use std::io::{Read, Seek, SeekFrom};
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let by_id: HashMap<&str, &Layer> = layers.iter().map(|l| (l.id.as_str(), l)).collect();
	let by_name: HashMap<&str, &Layer> = layers.iter().map(|l| (l.name.as_str(), l)).collect();
	let input_from = |id: &str, port: &str| edges.iter().find(|e| e.to.0 == id && e.to.1 == port);
	let mut f = std::fs::File::open(bin).map_err(|e| format!("{}: {e}", bin.display()))?;
	let mut next_id = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let (mut new_layers, mut new_edges) = (String::new(), String::new());
	let mut replaced: HashMap<usize, String> = HashMap::new();
	// pows[e - 1]: the source port of 2^e (f32, [1, 1]).
	let mut pows: Vec<(String, String)> = Vec::new();
	let s11 = [1i64, 1];
	let mut n = 0;
	// The conv layers with their scale Multiply and scale Const, and the
	// smallest nonzero |scale| over all of them.
	let mut convs = Vec::new();
	let mut min = f32::INFINITY;
	for conv in layers.iter().filter(|l| {
		l.ty == "GroupConvolution"
			&& l.name
				.ends_with("linear_attn/aten::_convolution/GroupConvolution")
	}) {
		let base = &conv.name[..conv.name.len() - "/aten::_convolution/GroupConvolution".len()];
		// Layer 0 is never folded, and its conv input is the largest (up to
		// ~70, so ~90 out): x2^9 would come close to f16's 65504.
		if base.ends_with(".layers.0.linear_attn") {
			continue;
		}
		let mul = by_name
			.get(format!("{base}/aten::_convolution/Reshape/fq_weights_1").as_str())
			.ok_or_else(|| format!("{base}: no conv weight scale Multiply"))?;
		let se = input_from(&mul.id, "1")
			.ok_or_else(|| format!("{base}: scale Multiply without input 1"))?;
		let sc = by_id
			.get(se.from.0.as_str())
			.ok_or("edge from an unknown layer")?;
		let Some((et, off, size)) = sc
			.data
			.as_ref()
			.filter(|(et, _, _)| sc.ty == "Const" && et == "f16")
		else {
			return Err(format!("{base}: conv weight scale is not an f16 Const"));
		};
		let _ = et;
		let mut b = vec![0u8; *size as usize];
		f.seek(SeekFrom::Start(*off)).map_err(|e| e.to_string())?;
		f.read_exact(&mut b).map_err(|e| e.to_string())?;
		// Smallest nonzero |scale|, from the f16 bits (exponent field 0 =
		// subnormal: 2^-24 * mantissa).
		min = b
			.chunks_exact(2)
			.map(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7fff)
			.filter(|&h| h != 0)
			.map(|h| {
				if h < 0x400 {
					h as f32 * 2f32.powi(-24)
				} else {
					f32::from_bits(((h as u32 >> 10) + 112) << 23 | (h as u32 & 0x3ff) << 13)
				}
			})
			.fold(min, f32::min);
		convs.push((conv, base, *mul, se, *sc));
	}
	// One k for every layer: NPUW folds the layers into one repeated
	// function and asserts if the blocks differ in structure.
	let mut k = 0u32;
	while min * 2f32.powi(k as i32) < 2f32.powi(-14) {
		k += 1;
	}
	// Capped at 5: x2^9 (what scales down to 1.2e-7 would need) broke the
	// 32-position prefill graph at a KV capacity of 4096 (KL ~1, fine at
	// 1024), x2^5 does not, and the decode error is the same at 5 and 9.
	let k = force_k.unwrap_or(k.min(5));
	if k == 0 {
		return Ok((xml.to_string(), 0, 0));
	}
	let uid = next_id;
	next_id += 1;
	new_layers.push_str(&format!(
		"		<layer id=\"{uid}\" name=\"{CONV_UNSCALE}\" type=\"Parameter\" version=\"opset1\">
			<data shape=\"1,1,1\" element_type=\"f32\" />
			<output>
				<port id=\"0\" precision=\"FP32\" names=\"{CONV_UNSCALE}\">
{}				</port>
			</output>
		</layer>
",
		dims_xml(&[1, 1, 1])
	));
	for (conv, base, mul, se, sc) in convs {
		if pows.is_empty() {
			let pow = by_name
				.get(format!("{base}.norm/aten::pow/Power").as_str())
				.ok_or_else(|| format!("{base}: no norm Power"))?;
			let pe = input_from(&pow.id, "1").ok_or("norm Power without input 1")?;
			let pc = by_id
				.get(pe.from.0.as_str())
				.ok_or("edge from an unknown layer")?;
			let Some(("f32", poff, 4)) = pc.data.as_ref().map(|(e, o, s)| (e.as_str(), *o, *s))
			else {
				return Err(format!("{base}: norm exponent is not an f32 scalar Const"));
			};
			if read_f32(bin, poff)? != 2.0 {
				return Err(format!("{base}: norm exponent is not 2"));
			}
			let tid = next_id;
			next_id += 1;
			new_layers.push_str(&format!(
				"\t\t<layer id=\"{tid}\" name=\"qwen_rs.convscale.two\" type=\"Const\" version=\"opset1\">\n\t\t\t<data element_type=\"f32\" shape=\"1, 1\" offset=\"{poff}\" size=\"4\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
				dims_xml(&s11)
			));
			pows.push((tid.to_string(), "0".to_string()));
		}
		while pows.len() < k as usize {
			let id = next_id;
			next_id += 1;
			new_layers.push_str(&mul_layer(
				id,
				&format!("qwen_rs.convscale.pow{}", pows.len() + 1),
				&s11,
				&s11,
				&s11,
			));
			let (pid, pport) = pows.last().unwrap().clone();
			new_edges.push_str(&edge((&pid, &pport), (&id.to_string(), "0")));
			new_edges.push_str(&edge((&pows[0].0, "0"), (&id.to_string(), "1")));
			pows.push((id.to_string(), "2".to_string()));
		}
		let (pid, pport) = pows[k as usize - 1].clone();
		// scale * f16(2^k), into the weight Multiply in place of the scale.
		let (cv, ms, dv) = (next_id, next_id + 1, next_id + 2);
		next_id += 3;
		new_layers.push_str(&layer_xml(
			cv,
			&format!("{base}/qwen_rs.convscale.f16"),
			"Convert",
			"destination_type=\"f16\"",
			&[("FP32", &s11)],
			&[("FP16", &s11)],
		));
		new_edges.push_str(&edge((&pid, &pport), (&cv.to_string(), "0")));
		new_layers.push_str(&layer_xml(
			ms,
			&format!("{base}/qwen_rs.convscale"),
			"Multiply",
			"auto_broadcast=\"numpy\"",
			&[("FP16", &sc.out_dims), ("FP16", &s11)],
			&[("FP16", &sc.out_dims)],
		));
		new_edges.push_str(&edge((&se.from.0, &se.from.1), (&ms.to_string(), "0")));
		new_edges.push_str(&edge((&cv.to_string(), "1"), (&ms.to_string(), "1")));
		replaced.insert(se.line, edge((&ms.to_string(), "2"), (&mul.id, "1")));
		// conv output * the runtime input 2^-k, in front of every reader.
		new_layers.push_str(&mul_layer(
			dv,
			&format!("{base}/qwen_rs.convunscale"),
			&conv.out_dims,
			&[1, 1, 1],
			&conv.out_dims,
		));
		new_edges.push_str(&edge((&conv.id, "2"), (&dv.to_string(), "0")));
		new_edges.push_str(&edge((&uid.to_string(), "0"), (&dv.to_string(), "1")));
		for r in edges
			.iter()
			.filter(|r| r.from.0 == conv.id && r.from.1 == "2")
		{
			replaced.insert(r.line, edge((&dv.to_string(), "2"), (&r.to.0, &r.to.1)));
		}
		n += 1;
	}

	let mut out = String::with_capacity(xml.len() + new_layers.len() + new_edges.len());
	for (i, l) in lines.iter().enumerate() {
		if i == layers_end {
			out.push_str(&new_layers);
		}
		if i == edges_end {
			out.push_str(&new_edges);
		}
		match replaced.get(&i) {
			Some(r) => out.push_str(r),
			None => out.push_str(l),
		}
	}
	Ok((out, n, k))
}

/// Add a `Result` at the first output of each named layer, as extra model
/// outputs `qwen_rs.probe.<k>` (`k` = index in `names`): an accuracy
/// probe. Returns the XML and the probes' output names.
pub fn add_probes(xml: &str, names: &[String]) -> Result<(String, Vec<String>), String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let mut next_id = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let (mut nl, mut ne, mut outs) = (String::new(), String::new(), Vec::new());
	for (k, n) in names.iter().enumerate() {
		let l = layers
			.iter()
			.find(|l| &l.name == n)
			.ok_or_else(|| format!("probe: no layer {n}"))?;
		let port = edges
			.iter()
			.find(|e| e.from.0 == l.id)
			.map(|e| e.from.1.clone())
			.ok_or_else(|| format!("probe: {n} has no reader"))?;
		let on = format!("qwen_rs.probe.{k}");
		let id = next_id;
		next_id += 1;
		nl.push_str(&format!(
			"\t\t<layer id=\"{id}\" name=\"{on}/result\" type=\"Result\" version=\"opset1\" output_names=\"{on}\">\n\t\t\t<input>\n\t\t\t\t<port id=\"0\" precision=\"FP32\">\n{}\t\t\t\t</port>\n\t\t\t</input>\n\t\t</layer>\n",
			dims_xml(&l.out_dims)
		));
		ne.push_str(&edge((&l.id, &port), (&id.to_string(), "0")));
		outs.push(on);
	}
	Ok((
		splice(
			xml,
			&lines,
			layers_end,
			edges_end,
			&nl,
			&ne,
			&[],
			&HashMap::new(),
		),
		outs,
	))
}

/// Input names of the host-computed rotary tables ([`rope_inputs`]).
pub const ROPE_COS: &str = "qwen_rs.rope_cos";
pub const ROPE_SIN: &str = "qwen_rs.rope_sin";

/// Make the rotary embedding's `cos` / `sin` tables model inputs.
///
/// The export computes them in the graph: `emb = position_ids * inv_freq`
/// (f32 `[.., S, 64]`, the 64 rotary channels of each 256-wide head), then
/// `Cos` / `Sin`. The NPU computes in f16, where a position-sized angle is
/// coarse (the f16 spacing is 0.25 between 256 and 512, 0.5 up to 1024)
/// and the `Cos` / `Sin` range reduction of such an argument loses more.
/// The attention layers then rotate queries and keys by the wrong angles,
/// an error that grows with the position. Here the two ops become
/// `Parameter`s ([`ROPE_COS`], [`ROPE_SIN`], f32 `[1, S, 64]`) that the
/// host fills from [`rope_tables`]; the rest of the rotary subgraph has no
/// reader left and is dropped when the model is read. Returns the XML and
/// the 32 inverse frequencies (the graph's own f32 constant).
pub fn rope_inputs(xml: &str, bin: &std::path::Path) -> Result<(String, Vec<f32>), String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let find = |suffix: &str| -> Result<&Layer, String> {
		let v: Vec<&Layer> = layers.iter().filter(|l| l.name.ends_with(suffix)).collect();
		match v[..] {
			[l] => Ok(l),
			_ => Err(format!("rope_inputs: {} layers named *{suffix}", v.len())),
		}
	};
	let cos = find("rotary_emb/aten::cos/Cos")?;
	let sin = find("rotary_emb/aten::sin/Sin")?;
	let inv = find("rotary_emb/aten::to/Convert")?;
	let inv_freq = match &inv.data {
		Some((et, off, 128)) if inv.ty == "Const" && et == "f32" => (0..32)
			.map(|i| read_f32(bin, off + 4 * i))
			.collect::<Result<Vec<f32>, String>>()?,
		_ => {
			return Err(format!(
				"rope_inputs: {} is not a 32-entry f32 Const",
				inv.name
			))
		}
	};
	if cos.out_dims.last() != Some(&64) || sin.out_dims.last() != Some(&64) {
		return Err(format!(
			"rope_inputs: cos / sin dims {:?} / {:?}, expected [.., 64]",
			cos.out_dims, sin.out_dims
		));
	}
	let mut replaced: HashMap<usize, String> = HashMap::new();
	let mut dropped = Vec::new();
	for (l, name) in [(cos, ROPE_COS), (sin, ROPE_SIN)] {
		dropped.push((l.start + 1, l.end));
		replaced.insert(
			l.start,
			format!(
				"\t\t<layer id=\"{}\" name=\"{name}\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"?,?,64\" element_type=\"f32\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"FP32\" names=\"{name}\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
				l.id,
				dims_xml(&[-1, -1, 64])
			),
		);
		for e in &edges {
			if e.to.0 == l.id {
				dropped.push((e.line, e.line));
			} else if e.from.0 == l.id {
				replaced.insert(e.line, edge((&l.id, "0"), (&e.to.0, &e.to.1)));
			}
		}
	}
	dropped.sort();
	Ok((
		splice(
			xml, &lines, layers_end, edges_end, "", "", &dropped, &replaced,
		),
		inv_freq,
	))
}

/// `cos` and `sin` rows of positions `pos..pos + s` for [`rope_inputs`]:
/// `[s, 64]` each, channel `j` at angle `p * inv_freq[j % 32]` (the export's
/// `cat(freqs, freqs)`; for text all three MRoPE rows hold the same position,
/// so the interleave picks the same angle). Computed in f64.
///
/// `f16_angle`: round the angle as an f16 device would (`f16(f16(p) *
/// f16(inv_freq))`): the NPU's error, reproduced on an f32 device.
pub fn rope_tables(
	inv_freq: &[f32],
	pos: usize,
	s: usize,
	f16_angle: bool,
) -> (Vec<f32>, Vec<f32>) {
	let (mut c, mut si) = (Vec::with_capacity(s * 64), Vec::with_capacity(s * 64));
	for p in pos..pos + s {
		for j in 0..64 {
			let f = inv_freq[j % 32];
			let a = if f16_angle {
				round_f16(round_f16(p as f32) * round_f16(f)) as f64
			} else {
				p as f64 * f as f64
			};
			c.push(a.cos() as f32);
			si.push(a.sin() as f32);
		}
	}
	(c, si)
}

/// `x` rounded to the nearest f16 (normal range; ties to even).
fn round_f16(x: f32) -> f32 {
	if x == 0.0 || !x.is_finite() {
		return x;
	}
	let e = x.abs().log2().floor().max(-14.0);
	let ulp = (e - 10.0).exp2();
	let q = x / ulp;
	let mut r = q.round();
	if (q - q.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
		r -= q.signum();
	}
	r * ulp
}

fn read_i64(bin: &std::path::Path, offset: u64) -> Result<i64, String> {
	use std::io::{Read, Seek, SeekFrom};
	let mut f = std::fs::File::open(bin).map_err(|e| format!("{}: {e}", bin.display()))?;
	f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
	let mut b = [0u8; 8];
	f.read_exact(&mut b).map_err(|e| e.to_string())?;
	Ok(i64::from_le_bytes(b))
}

/// One port of a generated layer: precision and dims (empty = scalar).
type Port<'a> = (&'a str, &'a [i64]);

fn port_xml(id: usize, (prec, dims): Port) -> String {
	if dims.is_empty() {
		format!("\t\t\t\t<port id=\"{id}\" precision=\"{prec}\" />\n")
	} else {
		format!(
			"\t\t\t\t<port id=\"{id}\" precision=\"{prec}\">\n{}\t\t\t\t</port>\n",
			dims_xml(dims)
		)
	}
}

/// A layer with input ports `0..n` and output ports `n..`.
fn layer_xml(id: usize, name: &str, ty: &str, data: &str, ins: &[Port], outs: &[Port]) -> String {
	let mut s =
		format!("\t\t<layer id=\"{id}\" name=\"{name}\" type=\"{ty}\" version=\"opset1\">\n");
	if !data.is_empty() {
		s.push_str(&format!("\t\t\t<data {data} />\n"));
	}
	s.push_str("\t\t\t<input>\n");
	for (k, p) in ins.iter().enumerate() {
		s.push_str(&port_xml(k, *p));
	}
	s.push_str("\t\t\t</input>\n\t\t\t<output>\n");
	for (k, p) in outs.iter().enumerate() {
		s.push_str(&port_xml(ins.len() + k, *p));
	}
	s.push_str("\t\t\t</output>\n\t\t</layer>\n");
	s
}

/// Replace every delta-net `Loop` with `s` explicit recurrent steps.
///
/// The NPU compiler rejects any graph with more than one position that still
/// holds the Loops (`SetupLocationVerifier Pass failed : Found 24*S
/// duplicated names`; one position compiles). Each Loop is the gated delta
/// rule, one position per iteration, so for a static `s` it is
///
/// ```text
/// S <- S * exp(g_t);  d = (v_t - k_t S) * beta_t;  S <- S + k_t^T d;  o_t = q_t S
/// ```
///
/// written out `s` times: q / k / v / g / beta are `Split` along the position
/// axis, the outputs `Concat`ed. Only `Split` / `Unsqueeze` need constants
/// (the axes 2 and 3), and the export already has `i64` scalars of both, so
/// like [`scale_delta_q`] this reuses the original `.bin`. The resulting graph
/// only accepts exactly `s` positions. Returns the XML and the Loop count.
pub fn unroll_delta(xml: &str, bin: &std::path::Path, s: usize) -> Result<(String, usize), String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let (ax2, ax3) = axes23(&layers, bin)?;

	let mut next_id = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let mut id = || {
		next_id += 1;
		next_id - 1
	};
	let mut new_layers = String::new();
	let mut new_edges = String::new();
	let mut dropped_lines: Vec<(usize, usize)> = Vec::new();
	let mut replaced: HashMap<usize, String> = HashMap::new();
	let mut n = 0;
	for lp in layers.iter().filter(|l| l.ty == "Loop") {
		// Output port 9 is `[B, H, S, D]`.
		let (h, d) = match lp.out_dims.as_slice() {
			[_, h, _, d] => (*h, *d),
			o => return Err(format!("{}: unexpected output dims {o:?}", lp.name)),
		};
		let src = |p: &str| -> Result<(String, String), String> {
			edges
				.iter()
				.find(|e| e.to.0 == lp.id && e.to.1 == p)
				.map(|e| e.from.clone())
				.ok_or_else(|| format!("{}: no input {p}", lp.name))
		};
		// Inputs 2..6 are q, k, v, g, beta (sliced on axis 2), 7 the state.
		let (q, k, v, g, beta, st0) = (
			src("2")?,
			src("3")?,
			src("4")?,
			src("5")?,
			src("6")?,
			src("7")?,
		);
		let nm = |what: &str| format!("{}/qwen_rs.unroll.{what}", lp.name);
		let (b4, b4u, bh3) = ([-1, h, -1, d], [-1, h, -1, 1], [-1, h, -1]);
		let (slice, slice1, state) = (&[-1, h, 1, d][..], &[-1, h, 1, 1][..], [-1, h, d, d]);
		let split_outs = |dims| -> Vec<Port> { vec![("FP32", dims); s] };
		let ns = format!("num_splits=\"{s}\"");
		// q, k, v: split into `s` positions.
		let mut parts = Vec::new();
		for (x, w) in [(&q, "q"), (&k, "k"), (&v, "v")] {
			let sid = id();
			new_layers.push_str(&layer_xml(
				sid,
				&nm(&format!("split_{w}")),
				"Split",
				&ns,
				&[("FP32", &b4), ("I64", &[])],
				&split_outs(slice),
			));
			new_edges.push_str(&edge((&x.0, &x.1), (&sid.to_string(), "0")));
			new_edges.push_str(&edge((ax2, "0"), (&sid.to_string(), "1")));
			parts.push(sid);
		}
		// g, beta: [B, H, S] -> [B, H, S, 1], then split.
		for (x, w) in [(&g, "g"), (&beta, "beta")] {
			let uid = id();
			new_layers.push_str(&layer_xml(
				uid,
				&nm(&format!("unsqueeze_{w}")),
				"Unsqueeze",
				"",
				&[("FP32", &bh3), ("I64", &[])],
				&[("FP32", &b4u)],
			));
			new_edges.push_str(&edge((&x.0, &x.1), (&uid.to_string(), "0")));
			new_edges.push_str(&edge((ax3, "0"), (&uid.to_string(), "1")));
			let sid = id();
			new_layers.push_str(&layer_xml(
				sid,
				&nm(&format!("split_{w}")),
				"Split",
				&ns,
				&[("FP32", &b4u), ("I64", &[])],
				&split_outs(slice1),
			));
			new_edges.push_str(&edge((&uid.to_string(), "2"), (&sid.to_string(), "0")));
			new_edges.push_str(&edge((ax2, "0"), (&sid.to_string(), "1")));
			parts.push(sid);
		}
		let [sq, sk, sv, sg, sb] = parts[..] else {
			unreachable!()
		};
		let part = |sid: usize, t: usize| (sid.to_string(), (2 + t).to_string());
		let mut st = st0.clone();
		let mut outs = Vec::new();
		let ew = |ty: &str, a: &[i64], b: &[i64], o: &[i64], id: usize, name: String| {
			layer_xml(
				id,
				&name,
				ty,
				"auto_broadcast=\"numpy\"",
				&[("FP32", a), ("FP32", b)],
				&[("FP32", o)],
			)
		};
		let mm = |a: &[i64], b: &[i64], o: &[i64], ta: bool, id: usize, name: String| {
			layer_xml(
				id,
				&name,
				"MatMul",
				&format!("transpose_a=\"{ta}\" transpose_b=\"false\""),
				&[("FP32", a), ("FP32", b)],
				&[("FP32", o)],
			)
		};
		for t in 0..s {
			let ts = |w: &str| nm(&format!("{w}{t}"));
			// decay = exp(g_t); S *= decay
			let e = id();
			new_layers.push_str(&layer_xml(
				e,
				&ts("exp"),
				"Exp",
				"",
				&[("FP32", slice1)],
				&[("FP32", slice1)],
			));
			let (gp0, gp1) = part(sg, t);
			new_edges.push_str(&edge((&gp0, &gp1), (&e.to_string(), "0")));
			let m1 = id();
			new_layers.push_str(&ew("Multiply", &state, slice1, &state, m1, ts("decay")));
			new_edges.push_str(&edge((&st.0, &st.1), (&m1.to_string(), "0")));
			new_edges.push_str(&edge((&e.to_string(), "1"), (&m1.to_string(), "1")));
			// kv = k_t S
			let (kp0, kp1) = part(sk, t);
			let kv = id();
			new_layers.push_str(&mm(slice, &state, slice, false, kv, ts("kv")));
			new_edges.push_str(&edge((&kp0, &kp1), (&kv.to_string(), "0")));
			new_edges.push_str(&edge((&m1.to_string(), "2"), (&kv.to_string(), "1")));
			// d = (v_t - kv) * beta_t
			let (vp0, vp1) = part(sv, t);
			let sub = id();
			new_layers.push_str(&ew("Subtract", slice, slice, slice, sub, ts("vmkv")));
			new_edges.push_str(&edge((&vp0, &vp1), (&sub.to_string(), "0")));
			new_edges.push_str(&edge((&kv.to_string(), "2"), (&sub.to_string(), "1")));
			let (bp0, bp1) = part(sb, t);
			let dd = id();
			new_layers.push_str(&ew("Multiply", slice, slice1, slice, dd, ts("delta")));
			new_edges.push_str(&edge((&sub.to_string(), "2"), (&dd.to_string(), "0")));
			new_edges.push_str(&edge((&bp0, &bp1), (&dd.to_string(), "1")));
			// S += k_t^T d
			let kd = id();
			new_layers.push_str(&mm(slice, slice, &state, true, kd, ts("kd")));
			new_edges.push_str(&edge((&kp0, &kp1), (&kd.to_string(), "0")));
			new_edges.push_str(&edge((&dd.to_string(), "2"), (&kd.to_string(), "1")));
			let add = id();
			new_layers.push_str(&ew("Add", &state, &state, &state, add, ts("state")));
			new_edges.push_str(&edge((&m1.to_string(), "2"), (&add.to_string(), "0")));
			new_edges.push_str(&edge((&kd.to_string(), "2"), (&add.to_string(), "1")));
			st = (add.to_string(), "2".to_string());
			// o_t = q_t S
			let (qp0, qp1) = part(sq, t);
			let o = id();
			new_layers.push_str(&mm(slice, &state, slice, false, o, ts("out")));
			new_edges.push_str(&edge((&qp0, &qp1), (&o.to_string(), "0")));
			new_edges.push_str(&edge((&st.0, &st.1), (&o.to_string(), "1")));
			outs.push(o);
		}
		let cat = id();
		let ins: Vec<Port> = vec![("FP32", slice); s];
		new_layers.push_str(&layer_xml(
			cat,
			&nm("concat"),
			"Concat",
			"axis=\"2\"",
			&ins,
			&[("FP32", &b4)],
		));
		for (t, o) in outs.iter().enumerate() {
			new_edges.push_str(&edge(
				(&o.to_string(), "2"),
				(&cat.to_string(), &t.to_string()),
			));
		}
		// The Loop's readers: port 9 (outputs) and 10 (final state).
		for e in edges.iter().filter(|e| e.from.0 == lp.id) {
			let from = match e.from.1.as_str() {
				"9" => (cat.to_string(), s.to_string()),
				"10" => st.clone(),
				p => return Err(format!("{}: read from unexpected port {p}", lp.name)),
			};
			replaced.insert(e.line, edge((&from.0, &from.1), (&e.to.0, &e.to.1)));
		}
		for e in edges.iter().filter(|e| e.to.0 == lp.id) {
			replaced.insert(e.line, String::new());
		}
		dropped_lines.push((lp.start, lp.end));
		n += 1;
	}
	Ok((
		splice(
			xml,
			&lines,
			layers_end,
			edges_end,
			&new_layers,
			&new_edges,
			&dropped_lines,
			&replaced,
		),
		n,
	))
}

/// Input name of a [`last_position_logits`] graph: `[1, s, 1]` f32, one-hot
/// at the position whose logits are wanted (the last real one).
pub const LAST_ONEHOT: &str = "qwen_rs.last_onehot";

/// Feed only **one** of `s` positions to the lm_head, so the graph's
/// `logits` are `[1, 1, vocab]` instead of `[1, s, vocab]`.
///
/// Only one position's logits are ever read after a prefill chunk (the last
/// real one: with right-padded input, [`pad_inputs`], not the last row), and
/// the full head is what kept the 16 / 32 / 64-position graphs out of the
/// compile cache: NPUW compiles the last layer + lm_head as its own NPU
/// subgraph, and from 16 positions on that subgraph's blob passes 2 GiB
/// (1.47 GB at 1 and 8 positions); the NPU plugin's blob export then
/// fails (`Write blob to stream failed. Blob is broken!`; the cached file
/// records that subgraph at exactly `0x7fffffff` bytes and never imports).
/// The head input is multiplied by the [`LAST_ONEHOT`] input and summed over
/// positions (exact: one term, the rest are `0 * finite`); the axis is an
/// `i64` scalar Const 1 already in the `.bin`.
pub fn last_position_logits(xml: &str, bin: &std::path::Path, s: usize) -> Result<String, String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let head = layers
		.iter()
		.find(|l| {
			l.ty == "MatMul"
				&& lines[l.start..=l.end]
					.iter()
					.any(|b| b.contains("names=\"logits\""))
		})
		.ok_or("no MatMul producing `logits`")?;
	let e = edges
		.iter()
		.find(|e| e.to.0 == head.id && e.to.1 == "0")
		.ok_or("lm_head MatMul without an input edge")?;
	let mut ax1 = None;
	for l in layers
		.iter()
		.filter(|l| l.ty == "Const" && l.out_dims.is_empty())
	{
		if let Some((et, off, 8)) = &l.data {
			if et == "i64" && read_i64(bin, *off)? == 1 {
				ax1 = Some(l.id.as_str());
				break;
			}
		}
	}
	let ax1 = ax1.ok_or("no i64 scalar Const 1 in the IR")?;
	let next = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let mut g = Gen {
		next,
		layers: String::new(),
		edges: String::new(),
	};
	let oh_dims = [1, s as i64, 1];
	let id = g.next;
	g.next += 1;
	g.layers.push_str(&format!(
		"\t\t<layer id=\"{id}\" name=\"{LAST_ONEHOT}\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"1,{s},1\" element_type=\"f32\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"FP32\" names=\"{LAST_ONEHOT}\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
		dims_xml(&oh_dims)
	));
	let oh: Src = (id.to_string(), "0".to_string());
	let src: Src = (e.from.0.clone(), e.from.1.clone());
	let d = [-1, -1, HEAD_IN];
	let m = g.ew("Multiply", (&src, &d), (&oh, &oh_dims), &d);
	let r = g.op_axis(
		"ReduceSum",
		"keep_dims=\"true\"",
		(&m, &d),
		ax1,
		&[&[-1, 1, HEAD_IN]],
	);
	let mut replaced = HashMap::new();
	replaced.insert(e.line, edge((&r[0].0, &r[0].1), (&head.id, "0")));
	Ok(splice(
		xml,
		&lines,
		layers_end,
		edges_end,
		&g.layers,
		&g.edges,
		&[],
		&replaced,
	))
}

/// Hidden size, the lm_head's input width.
const HEAD_IN: i64 = 5120;

/// The ids of `i64` scalar Consts holding 2 and 3, reused as axes.
fn axes23<'a>(layers: &'a [Layer], bin: &std::path::Path) -> Result<(&'a str, &'a str), String> {
	let mut axis: [Option<&str>; 2] = [None, None];
	for l in layers.iter().filter(|l| l.ty == "Const") {
		if let Some((et, off, 8)) = &l.data {
			if et == "i64" && l.out_dims.is_empty() {
				let v = read_i64(bin, *off)?;
				if (2..=3).contains(&v) && axis[v as usize - 2].is_none() {
					axis[v as usize - 2] = Some(&l.id);
				}
			}
		}
		if axis.iter().all(Option::is_some) {
			break;
		}
	}
	match axis {
		[Some(a), Some(b)] => Ok((a, b)),
		_ => Err("no i64 scalar Consts 2 and 3 in the IR".into()),
	}
}

/// The id of an `i64` scalar Const holding `v`, if the IR has one.
fn scalar_i64<'a>(
	layers: &'a [Layer],
	bin: &std::path::Path,
	v: i64,
) -> Result<Option<&'a str>, String> {
	for l in layers.iter().filter(|l| l.ty == "Const") {
		if let Some((et, off, 8)) = &l.data {
			if et == "i64" && l.out_dims.is_empty() && read_i64(bin, *off)? == v {
				return Ok(Some(&l.id));
			}
		}
	}
	Ok(None)
}

/// `lines` without the `dropped` line ranges, with `replaced` lines swapped
/// and the new layers / edges inserted before `</layers>` / `</edges>`.
#[allow(clippy::too_many_arguments)]
fn splice(
	xml: &str,
	lines: &[&str],
	layers_end: usize,
	edges_end: usize,
	new_layers: &str,
	new_edges: &str,
	dropped_lines: &[(usize, usize)],
	replaced: &HashMap<usize, String>,
) -> String {
	let mut out = String::with_capacity(xml.len() + new_layers.len() + new_edges.len());
	let mut drop = dropped_lines.iter().peekable();
	let mut i = 0;
	while i < lines.len() {
		if let Some(&&(a, b)) = drop.peek() {
			if i == a {
				i = b + 1;
				drop.next();
				continue;
			}
		}
		if i == layers_end {
			out.push_str(new_layers);
		}
		if i == edges_end {
			out.push_str(new_edges);
		}
		match replaced.get(&i) {
			Some(r) => out.push_str(r),
			None => out.push_str(lines[i]),
		}
		i += 1;
	}
	out
}

/// Input names of the two `[c, c]` f32 masks a [`chunk_delta`] graph reads:
/// strictly lower triangular, and lower triangular with the diagonal.
pub const TRIL_STRICT: &str = "qwen_rs.delta_tril_strict";
pub const TRIL_INCL: &str = "qwen_rs.delta_tril_incl";

/// The two mask inputs of a [`chunk_delta`] graph with sub-chunk `c`, row
/// major: `(strict, incl)`.
pub fn tril_masks(c: usize) -> (Vec<f32>, Vec<f32>) {
	let m = |incl: bool| {
		(0..c * c)
			.map(|i| {
				if i % c < i / c || (incl && i % c == i / c) {
					1.0
				} else {
					0.0
				}
			})
			.collect()
	};
	(m(false), m(true))
}

/// The block masks of a blocked [`chunk_delta`] graph with sub-chunk `c`:
/// `(input name, [c, c] row major)` for `b = 1, 2, 4, ... < c`, 1 where row
/// `i` and column `j` are in the same `2b` group, `i` in its second half
/// and `j` in its first.
pub fn blk_masks(c: usize) -> Vec<(String, Vec<f32>)> {
	let mut v = Vec::new();
	let mut b = 1;
	while b < c {
		let m = (0..c * c)
			.map(|x| (x / c, x % c))
			.map(|(i, j)| {
				if i / (2 * b) == j / (2 * b) && i / b > j / b {
					1.0
				} else {
					0.0
				}
			})
			.collect();
		v.push((format!("qwen_rs.delta_blk{b}"), m));
		b *= 2;
	}
	v
}

/// Generated layers and edges, with fresh ids.
struct Gen {
	next: usize,
	layers: String,
	edges: String,
}

type Src = (String, String);

impl Gen {
	/// Add a layer reading `ins` (source port + declared dims) with one
	/// output of `out` dims; returns its output port.
	fn op(&mut self, ty: &str, ver: &str, data: &str, ins: &[(&Src, &[i64])], out: &[i64]) -> Src {
		let id = self.next;
		self.next += 1;
		let ports: Vec<Port> = ins.iter().map(|(_, d)| ("FP32", *d)).collect();
		let mut x = layer_xml(
			id,
			&format!("qwen_rs.chunk.{id}"),
			ty,
			data,
			&ports,
			&[("FP32", out)],
		);
		if ver != "opset1" {
			x = x.replacen("version=\"opset1\"", &format!("version=\"{ver}\""), 1);
		}
		self.layers.push_str(&x);
		for (k, (s, _)) in ins.iter().enumerate() {
			self.edges
				.push_str(&edge((&s.0, &s.1), (&id.to_string(), &k.to_string())));
		}
		(id.to_string(), ins.len().to_string())
	}

	/// Like [`Gen::op`] with an `i64` scalar axis Const as the last input.
	fn op_axis(
		&mut self,
		ty: &str,
		data: &str,
		x: (&Src, &[i64]),
		axis: &str,
		outs: &[&[i64]],
	) -> Vec<Src> {
		let id = self.next;
		self.next += 1;
		let o: Vec<Port> = outs.iter().map(|d| ("FP32", *d)).collect();
		self.layers.push_str(&layer_xml(
			id,
			&format!("qwen_rs.chunk.{id}"),
			ty,
			data,
			&[("FP32", x.1), ("I64", &[])],
			&o,
		));
		self.edges
			.push_str(&edge((&x.0 .0, &x.0 .1), (&id.to_string(), "0")));
		self.edges
			.push_str(&edge((axis, "0"), (&id.to_string(), "1")));
		(0..outs.len())
			.map(|k| (id.to_string(), (2 + k).to_string()))
			.collect()
	}

	fn ew(&mut self, ty: &str, a: (&Src, &[i64]), b: (&Src, &[i64]), out: &[i64]) -> Src {
		self.op(ty, "opset1", "auto_broadcast=\"numpy\"", &[a, b], out)
	}

	fn mm(&mut self, a: (&Src, &[i64]), b: (&Src, &[i64]), ta: bool, tb: bool, out: &[i64]) -> Src {
		self.op(
			"MatMul",
			"opset1",
			&format!("transpose_a=\"{ta}\" transpose_b=\"{tb}\""),
			&[a, b],
			out,
		)
	}
}

/// The batched body of [`chunk_delta`] for one Loop: everything that does not
/// read the state (`A`, `T = (I + A)^-1`, `W`, `U`, the masked `Q K^T`, the
/// decays) is computed once for all `parts` sub-chunks as 5-D tensors
/// `[b, h, parts, c, ..]`; only the state chain `D_p = U_p - W_p S_p`,
/// `S_{p+1} = exp(G_c) S_p + (exp(G_c - G) K)_p^T D_p` (two small matmuls
/// per sub-chunk) runs in sequence, and the outputs
/// `O = exp(G) Q S_p + (tril_incl . exp(G_t - G_i) . Q K^T) D_p` are again
/// one batched pair. On the NPU each tiny op costs ~0.3-0.5 ms of latency
/// whatever its size, so the per-sub-chunk form cost ~4.5 ms per sub-chunk
/// and layer (36 ms of an 87 ms 256-position layer). Returns `(out, state)`.
#[allow(clippy::too_many_arguments)]
fn chunk_batched(
	g: &mut Gen,
	split: &[Vec<Src>],
	st0: &Src,
	h: i64,
	d: i64,
	parts: usize,
	c: i64,
	(ax2, ax3, neg1): (&Src, &Src, &Src),
	(tril, trili, eye, blks): (&Src, &Src, &Src, &[Src]),
) -> (Src, Src) {
	let pn = parts as i64;
	let cc = [c, c];
	let (x4, x3) = ([-1, h, c, d], [-1, h, c]);
	let (x5, x5g, x5c1, x51c, x5cc, x52d) = (
		[-1, h, pn, c, d],
		[-1, h, pn, c],
		[-1, h, pn, c, 1],
		[-1, h, pn, 1, c],
		[-1, h, pn, c, c],
		[-1, h, pn, c, 2 * d],
	);
	let (u4, u3) = ([-1, h, 1, c, d], [-1, h, 1, c]);
	let (state, s5, h5, g51) = (
		[-1, h, d, d],
		[-1, h, pn, d, d],
		[-1, h, pn, 1, 1],
		[-1, h, pn, 1],
	);
	// parts x [b, h, c, ..] -> [b, h, parts, c, ..]
	let stack = |g: &mut Gen, xs: &[Src], part: &[i64], un: &[i64], to: &[i64]| -> Src {
		let us: Vec<Src> = xs
			.iter()
			.map(|x| g.op("Unsqueeze", "opset1", "", &[(x, part), (ax2, &[])], un))
			.collect();
		if us.len() == 1 {
			return us[0].clone();
		}
		let r: Vec<(&Src, &[i64])> = us.iter().map(|u| (u, un)).collect();
		g.op("Concat", "opset1", "axis=\"2\"", &r, to)
	};
	// [b, h, parts, ..] -> parts x [b, h, ..]
	let unstack = |g: &mut Gen, x: &Src, from: &[i64], one: &[i64], to: &[i64]| -> Vec<Src> {
		let sp = if parts == 1 {
			vec![x.clone()]
		} else {
			g.op_axis(
				"Split",
				&format!("num_splits=\"{parts}\""),
				(x, from),
				&ax2.0,
				&vec![one; parts],
			)
		};
		sp.iter()
			.map(|p| g.op("Squeeze", "opset1", "", &[(p, one), (ax2, &[])], to))
			.collect()
	};
	let q = stack(g, &split[0], &x4, &u4, &x5);
	let k = stack(g, &split[1], &x4, &u4, &x5);
	let v = stack(g, &split[2], &x4, &u4, &x5);
	let gg = stack(g, &split[3], &x3, &u3, &x5g);
	let beta = stack(g, &split[4], &x3, &u3, &x5g);
	let cum = g.op(
		"CumSum",
		"opset3",
		"exclusive=\"false\" reverse=\"false\"",
		&[(&gg, &x5g), (ax3, &[])],
		&x5g,
	);
	let gc = g.op(
		"Unsqueeze",
		"opset1",
		"",
		&[(&cum, &x5g), (neg1, &[])],
		&x5c1,
	);
	let gr = g.op(
		"Unsqueeze",
		"opset1",
		"",
		&[(&cum, &x5g), (ax3, &[])],
		&x51c,
	);
	let diff = g.ew("Subtract", (&gc, &x5c1), (&gr, &x51c), &x5cc);
	let diffm = g.ew("Multiply", (&diff, &x5cc), (trili, &cc), &x5cc);
	let e = g.op("Exp", "opset1", "", &[(&diffm, &x5cc)], &x5cc);
	let bc = g.op(
		"Unsqueeze",
		"opset1",
		"",
		&[(&beta, &x5g), (neg1, &[])],
		&x5c1,
	);
	let eg = g.op("Exp", "opset1", "", &[(&gc, &x5c1)], &x5c1);
	// A = tril_strict . beta_t . exp(G_t - G_i) . k_t.k_i
	let kk = g.mm((&k, &x5), (&k, &x5), false, true, &x5cc);
	let a1 = g.ew("Multiply", (&kk, &x5cc), (&e, &x5cc), &x5cc);
	let a2 = g.ew("Multiply", (&a1, &x5cc), (&bc, &x5c1), &x5cc);
	let a = g.ew("Multiply", (&a2, &x5cc), (tril, &cc), &x5cc);
	// [W | U] = (I + A)^-1 [beta exp(G) K | beta V], blocked inverse.
	let kb = g.ew("Multiply", (&k, &x5), (&bc, &x5c1), &x5);
	let kbe = g.ew("Multiply", (&kb, &x5), (&eg, &x5c1), &x5);
	let vb = g.ew("Multiply", (&v, &x5), (&bc, &x5c1), &x5);
	let rhs = g.op(
		"Concat",
		"opset1",
		"axis=\"4\"",
		&[(&kbe, &x5), (&vb, &x5)],
		&x52d,
	);
	let mut t: Option<Src> = None;
	for bm in blks {
		let l = g.ew("Multiply", (&a, &x5cc), (bm, &cc), &x5cc);
		t = Some(match &t {
			None => g.ew("Subtract", (eye, &cc), (&l, &x5cc), &x5cc),
			Some(t) => {
				let tl = g.mm((t, &x5cc), (&l, &x5cc), false, false, &x5cc);
				let tlt = g.mm((&tl, &x5cc), (t, &x5cc), false, false, &x5cc);
				g.ew("Subtract", (t, &x5cc), (&tlt, &x5cc), &x5cc)
			}
		});
	}
	let x = match &t {
		Some(t) => g.mm((t, &x5cc), (&rhs, &x52d), false, false, &x52d),
		None => rhs.clone(),
	};
	let wu = g.op_axis(
		"Split",
		"num_splits=\"2\"",
		(&x, &x52d),
		&neg1.0,
		&[&x5, &x5],
	);
	let qk = g.mm((&q, &x5), (&k, &x5), false, true, &x5cc);
	let qke = g.ew("Multiply", (&qk, &x5cc), (&e, &x5cc), &x5cc);
	let qkm = g.ew("Multiply", (&qke, &x5cc), (trili, &cc), &x5cc);
	let qe = g.ew("Multiply", (&q, &x5), (&eg, &x5c1), &x5);
	let gsum = g.op(
		"ReduceSum",
		"opset1",
		"keep_dims=\"true\"",
		&[(&gg, &x5g), (ax3, &[])],
		&g51,
	);
	let gcu = g.op(
		"Unsqueeze",
		"opset1",
		"",
		&[(&gsum, &g51), (neg1, &[])],
		&h5,
	);
	let dg = g.ew("Subtract", (&gcu, &h5), (&gc, &x5c1), &x5c1);
	let edg = g.op("Exp", "opset1", "", &[(&dg, &x5c1)], &x5c1);
	let kd = g.ew("Multiply", (&k, &x5), (&edg, &x5c1), &x5);
	let egc = g.op("Exp", "opset1", "", &[(&gcu, &h5)], &h5);
	let one5 = [-1, h, 1, c, d];
	let ws_p = unstack(g, &wu[0], &x5, &one5, &x4);
	let us_p = unstack(g, &wu[1], &x5, &one5, &x4);
	let kd_p = unstack(g, &kd, &x5, &one5, &x4);
	let eg_p = unstack(g, &egc, &h5, &[-1, h, 1, 1, 1], &[-1, h, 1, 1]);
	// The state chain.
	let mut st = st0.clone();
	let (mut ss, mut ds) = (Vec::new(), Vec::new());
	for p in 0..parts {
		ss.push(st.clone());
		let ws = g.mm((&ws_p[p], &x4), (&st, &state), false, false, &x4);
		let dm = g.ew("Subtract", (&us_p[p], &x4), (&ws, &x4), &x4);
		let sd = g.ew(
			"Multiply",
			(&st, &state),
			(&eg_p[p], &[-1, h, 1, 1]),
			&state,
		);
		let kdd = g.mm((&kd_p[p], &x4), (&dm, &x4), true, false, &state);
		st = g.ew("Add", (&sd, &state), (&kdd, &state), &state);
		ds.push(dm);
	}
	let sstk = stack(g, &ss, &state, &[-1, h, 1, d, d], &s5);
	let dstk = stack(g, &ds, &x4, &u4, &x5);
	let o1 = g.mm((&qe, &x5), (&sstk, &s5), false, false, &x5);
	let o2 = g.mm((&qkm, &x5cc), (&dstk, &x5), false, false, &x5);
	let o = g.ew("Add", (&o1, &x5), (&o2, &x5), &x5);
	let os = unstack(g, &o, &x5, &one5, &x4);
	let out = if parts == 1 {
		os[0].clone()
	} else {
		let r: Vec<(&Src, &[i64])> = os.iter().map(|o| (o, &x4[..])).collect();
		g.op("Concat", "opset1", "axis=\"2\"", &r, &[-1, h, pn * c, d])
	};
	(out, st)
}

/// Replace every delta-net `Loop` with the **chunked** gated delta rule
/// (the WY form of DeltaNet / GatedDeltaNet), `c` positions per sub-chunk.
///
/// [`unroll_delta`] writes the recurrence out one position at a time: 8 ops
/// per position and layer, each reading or writing the whole `48 x 128 x 128`
/// state, and on the NPU that was ~1.0 s of a 2.3 s 16-position chunk. Here a
/// sub-chunk of `c` positions is a fixed set of matmuls. Per head, with
/// `G = cumsum(g)`, `A[t,i] = beta_t exp(G_t - G_i) k_t.k_i` (`i < t`):
///
/// ```text
/// (I + A) D = beta (V - exp(G) K S0)   ->  D = U - W S0,
///     U = (I + A)^-1 beta V,  W = (I + A)^-1 beta exp(G) K
/// O = exp(G) Q S0 + (tril_incl . exp(G_t - G_i) . Q K^T) D
/// S = exp(G_c) S0 + (exp(G_c - G) K)^T D
/// ```
///
/// `T = (I + A)^-1` is computed **blocked** (`blocked`, the default): with
/// `T_1 = I` and `L_b` the part of `A` coupling the two halves of each `2b`
/// group (mask [`blk_masks`]), `T_2b = T_b - T_b L_b T_b` (the inverse of a
/// block lower-triangular matrix), `log2 c` levels of two small matmuls,
/// then `[W | U] = T [beta exp(G) K | beta V]`. Every intermediate is an
/// entry of `T` (|T| <= 1 measured), so nothing exceeds the result. The old
/// form (`blocked = false`, `QWEN_OV_DELTA_INV=neumann`): `A` is nilpotent,
/// so `(I + A)^-1 = (I - A)(I + A^2)(I + A^4)...` applied to the right-hand
/// side factor by factor; its partial products are truncated Neumann
/// series that reach 1e3-1e4 times the result (`A^8` entries ~2e3 at 2.5k
/// tokens), which overflows f16 under a delta-net value scale and costs
/// precision without one. The exponent `G_t - G_i` is masked to 0
/// above the diagonal before `exp` (it is positive there and would
/// overflow); the masks are model inputs ([`TRIL_STRICT`], [`TRIL_INCL`],
/// from [`tril_masks`]) because the export's `.bin` holds no such constant.
/// (A `Minimum(x, 0)` clamp instead came out all zeros on the NPU.)
///
/// `s` positions must be a multiple of `c`; sub-chunks run in sequence. On
/// the NPU (f16) the Neumann form gave NaN logits at `c = 32` (the
/// explicit powers up to `A^16` overflow) while `c = 16` matches the
/// unrolled graph's accuracy; two 16-position sub-chunks cost ~4 % more
/// than one of 32 would.
/// Returns the XML and the Loop count.
pub fn chunk_delta(
	xml: &str,
	bin: &std::path::Path,
	s: usize,
	c: usize,
	blocked: bool,
) -> Result<(String, usize), String> {
	if c == 0 || !s.is_multiple_of(c) {
		return Err(format!(
			"chunk_delta: {s} positions are not a multiple of the sub-chunk {c}"
		));
	}
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let (ax2, ax3) = axes23(&layers, bin)?;
	let (ax2, ax3) = (
		(ax2.to_string(), "0".to_string()),
		(ax3.to_string(), "0".to_string()),
	);
	let next = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let mut g = Gen {
		next,
		layers: String::new(),
		edges: String::new(),
	};
	let ci = c as i64;
	let cc = [ci, ci];
	// The two mask inputs.
	let mut mask = |name: &str| -> Src {
		let id = g.next;
		g.next += 1;
		g.layers.push_str(&format!(
			"\t\t<layer id=\"{id}\" name=\"{name}\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"{c},{c}\" element_type=\"f32\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"FP32\" names=\"{name}\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
			dims_xml(&cc)
		));
		(id.to_string(), "0".to_string())
	};
	let (tril, trili) = (mask(TRIL_STRICT), mask(TRIL_INCL));
	let blks: Vec<Src> = if blocked {
		blk_masks(c).iter().map(|(n, _)| mask(n)).collect()
	} else {
		Vec::new()
	};
	// The identity, for `T`'s first level.
	let eye = if blocked {
		Some(g.ew("Subtract", (&trili, &cc), (&tril, &cc), &cc))
	} else {
		None
	};
	// `QWEN_OV_DELTA_BATCH=0`: one full set of ops per sub-chunk in
	// sequence (the A/B); by default only the state chain is sequential.
	let batched: Option<Src> =
		if blocked && std::env::var("QWEN_OV_DELTA_BATCH").map_or(true, |v| v != "0") {
			let id = scalar_i64(&layers, bin, -1)?
				.ok_or("chunk_delta: no i64 scalar Const -1 in the IR")?;
			Some((id.to_string(), "0".to_string()))
		} else {
			None
		};
	let mut dropped_lines: Vec<(usize, usize)> = Vec::new();
	let mut replaced: HashMap<usize, String> = HashMap::new();
	let mut n = 0;
	let parts = s / c;
	for lp in layers.iter().filter(|l| l.ty == "Loop") {
		let (h, d) = match lp.out_dims.as_slice() {
			[_, h, _, d] => (*h, *d),
			o => return Err(format!("{}: unexpected output dims {o:?}", lp.name)),
		};
		let src = |p: &str| -> Result<Src, String> {
			edges
				.iter()
				.find(|e| e.to.0 == lp.id && e.to.1 == p)
				.map(|e| e.from.clone())
				.ok_or_else(|| format!("{}: no input {p}", lp.name))
		};
		// Inputs 2..6 are q, k, v, g, beta (sliced on axis 2), 7 the state.
		let ins = [src("2")?, src("3")?, src("4")?, src("5")?, src("6")?];
		let mut st = src("7")?;
		let (bs, bh) = ([-1, h, s as i64, d], [-1, h, s as i64]);
		let (x4, x3, x41, x14, xcc) = (
			[-1, h, ci, d],
			[-1, h, ci],
			[-1, h, ci, 1],
			[-1, h, 1, ci],
			[-1, h, ci, ci],
		);
		let (state, x2d, h11, h1) = ([-1, h, d, d], [-1, h, ci, 2 * d], [-1, h, 1, 1], [-1, h, 1]);
		// [q, k, v, g, beta][part]
		let split: Vec<Vec<Src>> = ins
			.iter()
			.enumerate()
			.map(|(j, x)| {
				if parts == 1 {
					return vec![x.clone()];
				}
				let (from, to): (&[i64], &[i64]) = if j < 3 { (&bs, &x4) } else { (&bh, &x3) };
				g.op_axis(
					"Split",
					&format!("num_splits=\"{parts}\""),
					(x, from),
					&ax2.0,
					&vec![to; parts],
				)
			})
			.collect();
		if let Some(neg1) = &batched {
			let out;
			(out, st) = chunk_batched(
				&mut g,
				&split,
				&st,
				h,
				d,
				parts,
				ci,
				(&ax2, &ax3, neg1),
				(&tril, &trili, eye.as_ref().unwrap(), &blks),
			);
			for e in edges.iter().filter(|e| e.from.0 == lp.id) {
				let from = match e.from.1.as_str() {
					"9" => &out,
					"10" => &st,
					p => return Err(format!("{}: read from unexpected port {p}", lp.name)),
				};
				replaced.insert(e.line, edge((&from.0, &from.1), (&e.to.0, &e.to.1)));
			}
			for e in edges.iter().filter(|e| e.to.0 == lp.id) {
				replaced.insert(e.line, String::new());
			}
			dropped_lines.push((lp.start, lp.end));
			n += 1;
			continue;
		}
		let mut outs = Vec::new();
		for p in 0..parts {
			let (q, k, v, gg, beta) = (
				&split[0][p],
				&split[1][p],
				&split[2][p],
				&split[3][p],
				&split[4][p],
			);
			let cum = g.op(
				"CumSum",
				"opset3",
				"exclusive=\"false\" reverse=\"false\"",
				&[(gg, &x3), (&ax2, &[])],
				&x3,
			);
			let gc = g.op("Unsqueeze", "opset1", "", &[(&cum, &x3), (&ax3, &[])], &x41);
			let gr = g.op("Unsqueeze", "opset1", "", &[(&cum, &x3), (&ax2, &[])], &x14);
			let diff = g.ew("Subtract", (&gc, &x41), (&gr, &x14), &xcc);
			let diffm = g.ew("Multiply", (&diff, &xcc), (&trili, &cc), &xcc);
			let e = g.op("Exp", "opset1", "", &[(&diffm, &xcc)], &xcc);
			let bc = g.op("Unsqueeze", "opset1", "", &[(beta, &x3), (&ax3, &[])], &x41);
			let eg = g.op("Exp", "opset1", "", &[(&gc, &x41)], &x41);
			// A = tril_strict . beta_t . exp(G_t - G_i) . k_t.k_i
			let kk = g.mm((k, &x4), (k, &x4), false, true, &xcc);
			let a1 = g.ew("Multiply", (&kk, &xcc), (&e, &xcc), &xcc);
			let a2 = g.ew("Multiply", (&a1, &xcc), (&bc, &x41), &xcc);
			let a = g.ew("Multiply", (&a2, &xcc), (&tril, &cc), &xcc);
			// [W | U] = (I + A)^-1 [beta exp(G) K | beta V]
			let kb = g.ew("Multiply", (k, &x4), (&bc, &x41), &x4);
			let kbe = g.ew("Multiply", (&kb, &x4), (&eg, &x41), &x4);
			let vb = g.ew("Multiply", (v, &x4), (&bc, &x41), &x4);
			let rhs = g.op(
				"Concat",
				"opset1",
				"axis=\"3\"",
				&[(&kbe, &x4), (&vb, &x4)],
				&x2d,
			);
			let x = if let Some(eye) = &eye {
				// T_2 = I - L_1, T_2b = T_b - T_b L_b T_b.
				let mut t: Option<Src> = None;
				for bm in &blks {
					let l = g.ew("Multiply", (&a, &xcc), (bm, &cc), &xcc);
					t = Some(match &t {
						None => g.ew("Subtract", (eye, &cc), (&l, &xcc), &xcc),
						Some(t) => {
							let tl = g.mm((t, &xcc), (&l, &xcc), false, false, &xcc);
							let tlt = g.mm((&tl, &xcc), (t, &xcc), false, false, &xcc);
							g.ew("Subtract", (t, &xcc), (&tlt, &xcc), &xcc)
						}
					});
				}
				match &t {
					Some(t) => g.mm((t, &xcc), (&rhs, &x2d), false, false, &x2d),
					None => rhs.clone(),
				}
			} else {
				let ar = g.mm((&a, &xcc), (&rhs, &x2d), false, false, &x2d);
				let mut x = g.ew("Subtract", (&rhs, &x2d), (&ar, &x2d), &x2d);
				let mut pw = a.clone();
				let mut m = 2;
				while m < c {
					pw = g.mm((&pw, &xcc), (&pw, &xcc), false, false, &xcc);
					let px = g.mm((&pw, &xcc), (&x, &x2d), false, false, &x2d);
					x = g.ew("Add", (&x, &x2d), (&px, &x2d), &x2d);
					m *= 2;
				}
				x
			};
			let wu = g.op_axis("Split", "num_splits=\"2\"", (&x, &x2d), &ax3.0, &[&x4, &x4]);
			let ws = g.mm((&wu[0], &x4), (&st, &state), false, false, &x4);
			let dm = g.ew("Subtract", (&wu[1], &x4), (&ws, &x4), &x4);
			// O = exp(G) Q S0 + (tril_incl . exp(G_t - G_i) . Q K^T) D
			let qk = g.mm((q, &x4), (k, &x4), false, true, &xcc);
			let qke = g.ew("Multiply", (&qk, &xcc), (&e, &xcc), &xcc);
			let qkm = g.ew("Multiply", (&qke, &xcc), (&trili, &cc), &xcc);
			let qe = g.ew("Multiply", (q, &x4), (&eg, &x41), &x4);
			let o1 = g.mm((&qe, &x4), (&st, &state), false, false, &x4);
			let o2 = g.mm((&qkm, &xcc), (&dm, &x4), false, false, &x4);
			outs.push(g.ew("Add", (&o1, &x4), (&o2, &x4), &x4));
			// S = exp(G_c) S0 + (exp(G_c - G) K)^T D
			let gsum = g.op(
				"ReduceSum",
				"opset1",
				"keep_dims=\"true\"",
				&[(gg, &x3), (&ax2, &[])],
				&h1,
			);
			let gcu = g.op(
				"Unsqueeze",
				"opset1",
				"",
				&[(&gsum, &h1), (&ax3, &[])],
				&h11,
			);
			let dg = g.ew("Subtract", (&gcu, &h11), (&gc, &x41), &x41);
			let edg = g.op("Exp", "opset1", "", &[(&dg, &x41)], &x41);
			let kd = g.ew("Multiply", (k, &x4), (&edg, &x41), &x4);
			let egc = g.op("Exp", "opset1", "", &[(&gcu, &h11)], &h11);
			let sd = g.ew("Multiply", (&st, &state), (&egc, &h11), &state);
			let kdd = g.mm((&kd, &x4), (&dm, &x4), true, false, &state);
			st = g.ew("Add", (&sd, &state), (&kdd, &state), &state);
		}
		let out = if parts == 1 {
			outs.pop().unwrap()
		} else {
			let o: Vec<(&Src, &[i64])> = outs.iter().map(|o| (o, &x4[..])).collect();
			g.op("Concat", "opset1", "axis=\"2\"", &o, &bs)
		};
		// The Loop's readers: port 9 (outputs) and 10 (final state).
		for e in edges.iter().filter(|e| e.from.0 == lp.id) {
			let from = match e.from.1.as_str() {
				"9" => &out,
				"10" => &st,
				p => return Err(format!("{}: read from unexpected port {p}", lp.name)),
			};
			replaced.insert(e.line, edge((&from.0, &from.1), (&e.to.0, &e.to.1)));
		}
		for e in edges.iter().filter(|e| e.to.0 == lp.id) {
			replaced.insert(e.line, String::new());
		}
		dropped_lines.push((lp.start, lp.end));
		n += 1;
	}
	Ok((
		splice(
			xml,
			&lines,
			layers_end,
			edges_end,
			&g.layers,
			&g.edges,
			&dropped_lines,
			&replaced,
		),
		n,
	))
}

/// Input names of a [`pad_inputs`] graph: `[1, 1, s]` f32, 1 for a real
/// position and 0 for padding; and `[w]` i32, the indices (into the
/// `w + s` conv window) of the `w` columns that become the next conv state.
pub const VALID: &str = "qwen_rs.valid";
pub const CONV_KEEP: &str = "qwen_rs.conv_keep";

/// Make an `s`-position graph accept **right-padded** input: the first `n`
/// positions are real, the rest are filler whose outputs are ignored, and
/// the state outputs are exactly those of the `n` real positions.
///
/// Attention needs nothing: the causal mask keeps real positions from
/// reading the filler after them, and the host copies only the real KV rows.
/// The two recurrences do:
/// - **delta net**: `g` (log decay) and `beta` are multiplied by [`VALID`]
///   before the Loop, so a filler position has decay `exp(0) = 1` and update
///   `beta (v - k S) = 0`; the state passes through unchanged, exactly
///   (and in the chunked form its rows of `A`, `W`, `U` and `D` are zero);
/// - **conv**: the next conv state was `Slice(concat(state, x), -w:)`, the
///   last `w` columns of the window, which would be filler; it becomes a
///   `Gather` at [`CONV_KEEP`] = `n .. n + w`.
///
/// With [`VALID`] all ones and [`CONV_KEEP`] = `s .. s + w` the graph
/// computes what it did before (a multiply by 1, the same columns). Run
/// before [`chunk_delta`] / [`unroll_delta`], which read the Loop's `g` /
/// `beta` through whatever feeds them. `states` are [`rewrite`]'s. Returns the
/// XML and the conv state width `w`.
pub fn pad_inputs(
	xml: &str,
	bin: &std::path::Path,
	s: usize,
	states: &[State],
) -> Result<(String, usize), String> {
	let Ir {
		lines,
		layers,
		edges,
		layers_end,
		edges_end,
	} = parse(xml)?;
	let (ax2, _) = axes23(&layers, bin)?;
	let by_id: HashMap<&str, &Layer> = layers.iter().map(|l| (l.id.as_str(), l)).collect();
	let next = layers
		.iter()
		.filter_map(|l| l.id.parse::<usize>().ok())
		.max()
		.unwrap_or(0)
		+ 1;
	let mut g = Gen {
		next,
		layers: String::new(),
		edges: String::new(),
	};
	let mut replaced: HashMap<usize, String> = HashMap::new();

	let conv: Vec<&State> = states
		.iter()
		.filter(|st| st.output.starts_with("cache_params.present.conv."))
		.collect();
	let w = match conv.first().map(|st| st.dims.as_slice()) {
		Some([_, _, w]) if *w > 0 => *w as usize,
		_ => return Err("pad_inputs: no conv state of shape [B, C, w]".into()),
	};
	let param = |g: &mut Gen, name: &str, et: &str, prec: &str, dims: &[i64]| -> Src {
		let id = g.next;
		g.next += 1;
		let shape: Vec<String> = dims.iter().map(i64::to_string).collect();
		g.layers.push_str(&format!(
			"\t\t<layer id=\"{id}\" name=\"{name}\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"{}\" element_type=\"{et}\" />\n\t\t\t<output>\n\t\t\t\t<port id=\"0\" precision=\"{prec}\" names=\"{name}\">\n{}\t\t\t\t</port>\n\t\t\t</output>\n\t\t</layer>\n",
			shape.join(","),
			dims_xml(dims)
		));
		(id.to_string(), "0".to_string())
	};
	let vdims = [1, 1, s as i64];
	let valid = param(&mut g, VALID, "f32", "FP32", &vdims);
	let keep = param(&mut g, CONV_KEEP, "i32", "I32", &[w as i64]);

	// Delta net: g (port 5) and beta (port 6) of every Loop, times `valid`.
	let mut n_loops = 0;
	for lp in layers.iter().filter(|l| l.ty == "Loop") {
		let h = match lp.out_dims.as_slice() {
			[_, h, _, _] => *h,
			o => return Err(format!("{}: unexpected output dims {o:?}", lp.name)),
		};
		let bh = [-1, h, -1];
		for p in ["5", "6"] {
			let e = edges
				.iter()
				.find(|e| e.to.0 == lp.id && e.to.1 == p)
				.ok_or_else(|| format!("{}: no input {p}", lp.name))?;
			let m = g.ew("Multiply", (&e.from, &bh), (&valid, &vdims), &bh);
			replaced.insert(e.line, edge((&m.0, &m.1), (&e.to.0, &e.to.1)));
		}
		n_loops += 1;
	}
	if n_loops == 0 {
		return Err("pad_inputs: no delta-net Loops".into());
	}

	// Conv: each conv state Result reads a Slice of the window; read a
	// Gather of the same window at `keep` instead.
	for st in &conv {
		let ch = match st.dims.as_slice() {
			[_, c, ww] if *ww as usize == w => *c,
			d => {
				return Err(format!(
					"{}: conv state dims {d:?}, expected [B, C, {w}]",
					st.output
				))
			}
		};
		let rname = format!("{}/result", st.output);
		let r = layers
			.iter()
			.find(|l| l.ty == "Result" && l.name == rname)
			.ok_or_else(|| format!("no Result {rname}"))?;
		let e = edges
			.iter()
			.find(|e| e.to.0 == r.id)
			.ok_or_else(|| format!("{rname} has no input"))?;
		let sl = by_id
			.get(e.from.0.as_str())
			.ok_or("edge from an unknown layer")?;
		if sl.ty != "Slice" {
			return Err(format!(
				"{rname} reads a {} ({}), expected a Slice",
				sl.ty, sl.name
			));
		}
		let win = edges
			.iter()
			.find(|x| x.to.0 == sl.id && x.to.1 == "0")
			.ok_or_else(|| format!("{}: no data input", sl.name))?;
		let id = g.next;
		g.next += 1;
		let (wd, od) = ([-1, ch, -1], [-1, ch, w as i64]);
		let x = layer_xml(
			id,
			&format!("{}/qwen_rs.conv_keep", sl.name),
			"Gather",
			"batch_dims=\"0\"",
			&[("FP32", &wd), ("I32", &[w as i64]), ("I64", &[])],
			&[("FP32", &od)],
		)
		.replacen("version=\"opset1\"", "version=\"opset8\"", 1);
		g.layers.push_str(&x);
		g.edges
			.push_str(&edge((&win.from.0, &win.from.1), (&id.to_string(), "0")));
		g.edges
			.push_str(&edge((&keep.0, &keep.1), (&id.to_string(), "1")));
		g.edges.push_str(&edge((ax2, "0"), (&id.to_string(), "2")));
		replaced.insert(e.line, edge((&id.to_string(), "3"), (&e.to.0, &e.to.1)));
	}
	Ok((
		splice(
			xml,
			&lines,
			layers_end,
			edges_end,
			&g.layers,
			&g.edges,
			&[],
			&replaced,
		),
		w,
	))
}

/// [`pad_inputs`]' two inputs for `n` real positions: `(valid, conv_keep)`.
pub fn pad_values(s: usize, n: usize, w: usize) -> (Vec<f32>, Vec<i32>) {
	(
		(0..s).map(|j| if j < n { 1.0 } else { 0.0 }).collect(),
		(n..n + w).map(|j| j as i32).collect(),
	)
}

#[cfg(test)]
mod tests {
	/// The OpenVINO export these tests read: `QWEN_TEST_OV_EXPORT`, a
	/// directory with `openvino_language_model.xml`. `None` (the test is
	/// skipped) when it is not set or the file cannot be read.
	fn read_export(test: &str) -> Option<(std::path::PathBuf, String)> {
		let Some(d) = std::env::var_os("QWEN_TEST_OV_EXPORT") else {
			eprintln!("{test}: skipped (QWEN_TEST_OV_EXPORT not set)");
			return None;
		};
		let xml = std::path::Path::new(&d).join("openvino_language_model.xml");
		match std::fs::read_to_string(&xml) {
			Ok(x) => Some((xml, x)),
			Err(e) => {
				eprintln!("{test}: skipped ({}: {e})", xml.display());
				None
			}
		}
	}

	/// The export's 128 variables become 128 parameter / result pairs: 16
	/// KV pairs with a dynamic sequence dim, 48 conv and 48 SSM states with
	/// static ones. Needs the export, so it is skipped when it is absent.
	#[test]
	fn rewrite_turns_every_variable_into_a_port_pair() {
		let Some((_, x)) = read_export("rewrite_turns_every_variable_into_a_port_pair") else {
			return;
		};
		let (out, st) = super::rewrite(&x).unwrap();
		assert_eq!(st.len(), 64 + 64);
		assert_eq!(st.iter().filter(|s| s.is_kv()).count(), 32);
		assert!(!out.contains("type=\"ReadValue\"") && !out.contains("type=\"Assign\""));
		for s in &st {
			assert!(
				s.input.starts_with("cache_params.past.")
					&& s.output.starts_with("cache_params.present."),
				"{s:?}"
			);
			assert_eq!(
				s.input["cache_params.past.".len()..],
				s.output["cache_params.present.".len()..]
			);
			assert_eq!(s.dims[0], -1);
			assert_eq!(
				s.dims.iter().skip(1).filter(|&&d| d < 0).count(),
				usize::from(s.is_kv()),
				"{s:?}"
			);
		}
	}

	/// Every delta-net Loop is replaced, its readers rewired, and nothing
	/// dangles at the Loop's old id. (The numerics are checked end to end:
	/// on the CPU the unrolled prefill graph's logits equal the Loop graph's
	/// bit for bit.)
	#[test]
	fn unroll_replaces_every_loop() {
		let Some((xml, x)) = read_export("unroll_replaces_every_loop") else {
			return;
		};
		let (sx, _) = super::rewrite(&x).unwrap();
		let ir = super::parse(&sx).unwrap();
		let loops: Vec<String> = ir
			.layers
			.iter()
			.filter(|l| l.ty == "Loop")
			.map(|l| l.id.clone())
			.collect();
		assert_eq!(loops.len(), 48);
		let (ux, n) = super::unroll_delta(&sx, &xml.with_extension("bin"), 4).unwrap();
		assert_eq!(n, 48);
		let u = super::parse(&ux).unwrap();
		assert!(u.layers.iter().all(|l| l.ty != "Loop"));
		assert_eq!(
			u.layers
				.iter()
				.filter(|l| l.name.ends_with("qwen_rs.unroll.concat"))
				.count(),
			48
		);
		for e in &u.edges {
			assert!(
				!loops.contains(&e.from.0) && !loops.contains(&e.to.0),
				"edge still touches a Loop: {:?} -> {:?}",
				e.from,
				e.to
			);
		}
	}

	/// The chunked form replaces every Loop too, adds exactly the two mask
	/// inputs, and every new edge starts at a layer that exists.
	#[test]
	fn chunk_replaces_every_loop() {
		let Some((xml, x)) = read_export("chunk_replaces_every_loop") else {
			return;
		};
		let (sx, _) = super::rewrite(&x).unwrap();
		let loops: Vec<String> = super::parse(&sx)
			.unwrap()
			.layers
			.iter()
			.filter(|l| l.ty == "Loop")
			.map(|l| l.id.clone())
			.collect();
		let bin = xml.with_extension("bin");
		assert!(super::chunk_delta(&sx, &bin, 24, 16, true).is_err());
		for blocked in [false, true] {
			let (cx, n) = super::chunk_delta(&sx, &bin, 32, 16, blocked).unwrap();
			assert_eq!(n, 48);
			let u = super::parse(&cx).unwrap();
			assert!(u.layers.iter().all(|l| l.ty != "Loop"));
			let params: Vec<&str> = u
				.layers
				.iter()
				.filter(|l| l.ty == "Parameter" && l.name.starts_with("qwen_rs."))
				.map(|l| l.name.as_str())
				.collect();
			let mut want = vec![super::TRIL_STRICT.to_string(), super::TRIL_INCL.to_string()];
			if blocked {
				want.extend(super::blk_masks(16).into_iter().map(|(n, _)| n));
			}
			assert_eq!(params, want);
			let ids: std::collections::HashSet<&str> =
				u.layers.iter().map(|l| l.id.as_str()).collect();
			for e in &u.edges {
				assert!(
					!loops.contains(&e.from.0) && !loops.contains(&e.to.0),
					"edge still touches a Loop: {:?} -> {:?}",
					e.from,
					e.to
				);
				assert!(
					ids.contains(e.from.0.as_str()) && ids.contains(e.to.0.as_str()),
					"dangling edge {:?} -> {:?}",
					e.from,
					e.to
				);
			}
		}
	}

	/// The rotary `Cos` / `Sin` become the two inputs, read by what read
	/// them before, and the inverse frequencies are the export's.
	#[test]
	fn rope_inputs_replace_cos_and_sin() {
		let Some((xml, x)) = read_export("rope_inputs_replace_cos_and_sin") else {
			return;
		};
		let (sx, _) = super::rewrite(&x).unwrap();
		let (rx, inv) = super::rope_inputs(&sx, &xml.with_extension("bin")).unwrap();
		assert_eq!(inv.len(), 32);
		assert_eq!(inv[0], 1.0);
		// theta 1e7 over 64 rotary channels
		assert!(
			(inv[1] as f64 - 1e7f64.powf(-2.0 / 64.0)).abs() < 1e-6,
			"{}",
			inv[1]
		);
		let u = super::parse(&rx).unwrap();
		assert!(u.layers.iter().all(|l| l.ty != "Cos" && l.ty != "Sin"));
		for n in [super::ROPE_COS, super::ROPE_SIN] {
			let p = u.layers.iter().find(|l| l.name == n).unwrap();
			assert_eq!(p.ty, "Parameter");
			assert_eq!(u.edges.iter().filter(|e| e.from.0 == p.id).count(), 1);
			assert!(u.edges.iter().all(|e| e.to.0 != p.id));
		}
	}

	/// Layers 1-63's conv weight scales get `x 2^k` and their conv outputs
	/// the one `CONV_UNSCALE` input, which every reader of a conv now reads
	/// through; layer 0 is untouched. (Exactness is checked end to end: KL
	/// 0.0000 against the unrewritten graph on the CPU.)
	#[test]
	fn conv_scales_rewire_every_conv_but_layer_0() {
		let Some((xml, x)) = read_export("conv_scales_rewire_every_conv_but_layer_0") else {
			return;
		};
		let (sx, _) = super::rewrite(&x).unwrap();
		let (cx, n, k) = super::scale_conv_weights(&sx, &xml.with_extension("bin"), None).unwrap();
		assert_eq!((n, k), (47, 5));
		let u = super::parse(&cx).unwrap();
		let p = u
			.layers
			.iter()
			.find(|l| l.name == super::CONV_UNSCALE)
			.unwrap();
		assert_eq!(p.ty, "Parameter");
		let unscales: Vec<&str> = u
			.edges
			.iter()
			.filter(|e| e.from.0 == p.id)
			.map(|e| e.to.0.as_str())
			.collect();
		assert_eq!(unscales.len(), 47);
		for conv in u.layers.iter().filter(|l| l.ty == "GroupConvolution") {
			let readers: Vec<&str> = u
				.edges
				.iter()
				.filter(|e| e.from.0 == conv.id)
				.map(|e| e.to.0.as_str())
				.collect();
			if conv.name.contains(".layers.0.") {
				assert!(readers.iter().all(|r| !unscales.contains(r)));
			} else {
				assert_eq!(readers.len(), 1, "{}", conv.name);
				assert!(unscales.contains(&readers[0]), "{}", conv.name);
			}
		}
	}

	#[test]
	fn round_f16_matches_the_format() {
		assert_eq!(super::round_f16(500.3), 500.25);
		assert_eq!(super::round_f16(1000.3), 1000.5);
		assert_eq!(super::round_f16(1.0), 1.0);
		assert_eq!(super::round_f16(0.1), 0.099_975_586);
	}

	/// The head reads the one-hot reduction, and the graph gains exactly
	/// that one input.
	#[test]
	fn last_position_feeds_the_head() {
		let Some((xml, x)) = read_export("last_position_feeds_the_head") else {
			return;
		};
		let (sx, _) = super::rewrite(&x).unwrap();
		let lx = super::last_position_logits(&sx, &xml.with_extension("bin"), 32).unwrap();
		let u = super::parse(&lx).unwrap();
		let head = u
			.layers
			.iter()
			.find(|l| l.name.ends_with("lm_head/ov_ext::linear/MatMul"))
			.unwrap();
		let into: Vec<_> = u
			.edges
			.iter()
			.filter(|e| e.to.0 == head.id && e.to.1 == "0")
			.collect();
		assert_eq!(into.len(), 1);
		let red = u.layers.iter().find(|l| l.id == into[0].from.0).unwrap();
		assert_eq!(red.ty, "ReduceSum");
		let params: Vec<&str> = u
			.layers
			.iter()
			.filter(|l| l.ty == "Parameter" && l.name.starts_with("qwen_rs."))
			.map(|l| l.name.as_str())
			.collect();
		assert_eq!(params, [super::LAST_ONEHOT]);
	}

	/// Padding adds its two inputs, masks every Loop's g and beta, and
	/// re-reads all 48 conv states through a Gather; the chunked form still
	/// applies on top and nothing dangles.
	#[test]
	fn pad_masks_every_recurrence() {
		let Some((xml, x)) = read_export("pad_masks_every_recurrence") else {
			return;
		};
		let bin = xml.with_extension("bin");
		let (sx, st) = super::rewrite(&x).unwrap();
		let (px, w) = super::pad_inputs(&sx, &bin, 32, &st).unwrap();
		assert_eq!(w, 4);
		let u = super::parse(&px).unwrap();
		assert_eq!(
			u.layers
				.iter()
				.filter(|l| l.ty == "Gather" && l.name.ends_with("qwen_rs.conv_keep"))
				.count(),
			48
		);
		let valid = u.layers.iter().find(|l| l.name == super::VALID).unwrap();
		assert_eq!(u.edges.iter().filter(|e| e.from.0 == valid.id).count(), 96);
		let (cx, n) = super::chunk_delta(&px, &bin, 32, 16, true).unwrap();
		assert_eq!(n, 48);
		let c = super::parse(&cx).unwrap();
		let ids: std::collections::HashSet<&str> = c.layers.iter().map(|l| l.id.as_str()).collect();
		for e in &c.edges {
			assert!(
				ids.contains(e.from.0.as_str()) && ids.contains(e.to.0.as_str()),
				"dangling edge {:?} -> {:?}",
				e.from,
				e.to
			);
		}
		assert_eq!(
			super::pad_values(4, 3, 4),
			(vec![1., 1., 1., 0.], vec![3, 4, 5, 6])
		);
	}

	#[test]
	fn tril_masks_are_lower_triangular() {
		let (s, i) = super::tril_masks(3);
		assert_eq!(s, [0., 0., 0., 1., 0., 0., 1., 1., 0.]);
		assert_eq!(i, [1., 0., 0., 1., 1., 0., 1., 1., 1.]);
	}
}
