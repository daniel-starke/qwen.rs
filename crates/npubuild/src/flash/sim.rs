//! The Flash-Next prefill pipeline of one chunk from zero state with the NPU
//! graphs ([`super::graphs`]) on OpenVINO's CPU plugin and the quantized
//! weight files: what the engine will do, with the host doing the iGPU's
//! part (recurrence, attention) and the routing / combine. Checks the graphs
//! and the quantization together against the float reference
//! (`nbcheck flash-sim`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::graphs::{self, S};
use super::refm::{self as fr, D, DA, FE, H, HCW, HKV, HQ, NE, NQKV, TOPK};
use crate::g::{f16_bits, f16_to_f32, Graph, Ty};
use npu::flash::{ngram, HC, W};

/// One graph kind compiled for the CPU.
struct Kind {
	g: Graph,
	c: ov::Compiled,
	/// The compiled blob on the NPU instead (`FLASH_SIM_NPU`).
	ng: Option<npu::ze::Graph<'static>>,
}

pub struct Sim {
	rt: ov::Runtime,
	tmp: PathBuf,
	wdir: PathBuf,
	kinds: HashMap<String, Kind>,
	/// `FLASH_SIM_NPU=kind,...`: these kinds run their blob from
	/// `OUT/graphs/flash` on the NPU (the engine's compile), the rest on the
	/// CPU: finds a graph the NPU compile gets wrong.
	npu_kinds: Vec<String>,
	npu: Option<(&'static npu::ze::Npu, npu::lock::NpuLock)>,
}

/// A runtime input.
pub enum In {
	F16(Vec<f32>),
	I32(Vec<i32>),
	Raw(Vec<u8>),
}

fn et(t: Ty) -> u32 {
	match t {
		Ty::F16 => ov::ET_F16,
		Ty::F32 => ov::ET_F32,
		Ty::I32 => ov::ET_I32,
		Ty::I64 => ov::ET_I64,
		Ty::I8 => ov::ET_I8,
		Ty::I4 => ov::ET_I4,
		Ty::U8 => ov::ET_I8,
	}
}

fn f16b(v: &[f32]) -> Vec<u8> {
	v.iter()
		.flat_map(|&x| f16_bits(x as f64).to_le_bytes())
		.collect()
}

fn unf16(b: &[u8]) -> Vec<f32> {
	b.chunks_exact(2)
		.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect()
}

impl Sim {
	pub fn new(wdir: &Path, tmp: &Path) -> Result<Sim, String> {
		std::fs::create_dir_all(tmp).map_err(|e| e.to_string())?;
		Ok(Sim {
			rt: ov::Runtime::new(None)?,
			tmp: tmp.to_path_buf(),
			wdir: wdir.to_path_buf(),
			kinds: HashMap::new(),
			npu_kinds: std::env::var("FLASH_SIM_NPU")
				.unwrap_or_default()
				.split(',')
				.filter(|k| !k.is_empty())
				.map(String::from)
				.collect(),
			npu: None,
		})
	}

	/// Run graph `kind` (built by `mk` on first use): inputs from `rt` by
	/// name, every other input from layer `l`'s file; the named f16 outputs.
	fn run(
		&mut self,
		kind: &str,
		mk: impl FnOnce() -> Graph,
		l: Option<usize>,
		rt: Vec<(&str, In)>,
		outs: &[&str],
	) -> Result<Vec<Vec<f32>>, String> {
		if !self.kinds.contains_key(kind) {
			let g = mk();
			let x = self.tmp.join(format!("{kind}.xml"));
			g.save(&x, kind)?;
			let m = self.rt.read_model_bin(&x, Some(&x.with_extension("bin")))?;
			let c = self.rt.compile(&m, "CPU", &[])?;
			let ng = if self.npu_kinds.iter().any(|k| k == kind) {
				if self.npu.is_none() {
					let lock = npu::lock::NpuLock::acquire()?;
					let n: &'static npu::ze::Npu = Box::leak(Box::new(npu::ze::Npu::open()?));
					self.npu = Some((n, lock));
				}
				let n = self.npu.as_ref().unwrap().0;
				let p = self.wdir.join(format!("../graphs/flash/{kind}.blob"));
				let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
				let mut ng = n.load(&b, 0)?;
				ng.initialize()?;
				Some(ng)
			} else {
				None
			};
			self.kinds.insert(kind.to_string(), Kind { g, c, ng });
		}
		let mut rt: HashMap<&str, In> = rt.into_iter().collect();
		if self.kinds[kind].ng.is_some() {
			let n = self.npu.as_ref().unwrap().0;
			let k = self.kinds.get_mut(kind).unwrap();
			let onames = k.g.outputs();
			let ng = k.ng.as_mut().unwrap();
			let mut bufs = Vec::with_capacity(ng.args.len());
			let mut oidx = Vec::new();
			for i in 0..ng.args.len() {
				let a = ng.args[i].clone();
				let mut hb = n.alloc(a.bytes)?;
				if a.is_output {
					oidx.push(i);
				} else {
					let bytes = match rt.remove(a.name.as_str()) {
						Some(In::F16(v)) => f16b(&v),
						Some(In::I32(v)) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
						Some(In::Raw(b)) => b,
						None => {
							let l = l.ok_or(format!("{kind}: no input {}", a.name))?;
							let p = self.wdir.join(format!("fl{l}.{}.in", a.name));
							std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?
						}
					};
					if bytes.len() != a.bytes {
						return Err(format!(
							"{kind} (npu): input {} has {} bytes, wants {}",
							a.name,
							bytes.len(),
							a.bytes
						));
					}
					hb.as_mut_slice().copy_from_slice(&bytes);
				}
				ng.bind(i, &hb)?;
				bufs.push(hb);
			}
			n.batch(&mut [ng])?.run()?;
			return outs
				.iter()
				.map(|o| {
					let j = onames
						.iter()
						.position(|x| x == o)
						.ok_or(format!("{kind}: no output {o}"))?;
					Ok(unf16(bufs[oidx[j]].as_slice()))
				})
				.collect();
		}
		let k = &self.kinds[kind];
		let mut req = k.c.request()?;
		for (name, ty, shape) in k.g.inputs() {
			let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
			let bytes = match rt.remove(name.as_str()) {
				Some(In::F16(v)) => f16b(&v),
				Some(In::I32(v)) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
				Some(In::Raw(b)) => b,
				None => {
					let l = l.ok_or(format!("{kind}: no input {name}"))?;
					let p = self.wdir.join(format!("fl{l}.{name}.in"));
					std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?
				}
			};
			let want = ty.bytes(shape.iter().product());
			if bytes.len() != want {
				return Err(format!(
					"{kind}: input {name} has {} bytes, wants {want}",
					bytes.len()
				));
			}
			req.set_raw(&name, et(ty), &dims, &bytes)?;
		}
		req.infer()?;
		outs.iter()
			.map(|o| Ok(unf16(req.output_raw(o, 2)?.1)))
			.collect()
	}
}

/// The recurrence on the pre graph's outputs (`qn`, `kn` [S][2048], `v`
/// [S][6144], `gb` [S][96]) from zero state: `o` [S][6144] x `oscale`.
fn recurrence(qn: &[f32], kn: &[f32], v: &[f32], gb: &[f32], s: usize, oscale: f32) -> Vec<f32> {
	let mut o = vec![0f32; s * H * D];
	let qs = 1.0 / (D as f32).sqrt();
	for h in 0..H {
		let kh = h / (H / fr::HK);
		let mut st = vec![0f32; D * D];
		for t in 0..s {
			let q = &qn[t * fr::HK * D + kh * D..][..D];
			let k = &kn[t * fr::HK * D + kh * D..][..D];
			let vv = &v[t * H * D + h * D..][..D];
			let (g, beta) = (gb[t * 2 * H + h], gb[t * 2 * H + H + h]);
			let eg = g.exp();
			st.iter_mut().for_each(|x| *x *= eg);
			let mut kv = [0f32; D];
			for (ki, row) in k.iter().zip(st.chunks_exact(D)) {
				for (a, b) in kv.iter_mut().zip(row) {
					*a += ki * b;
				}
			}
			for (a, b) in kv.iter_mut().zip(vv) {
				*a = (b - *a) * beta;
			}
			for (ki, row) in k.iter().zip(st.chunks_exact_mut(D)) {
				for (a, b) in row.iter_mut().zip(&kv) {
					*a += ki * b;
				}
			}
			let ot = &mut o[t * H * D + h * D..][..D];
			for (qi, row) in q.iter().zip(st.chunks_exact(D)) {
				for (a, b) in ot.iter_mut().zip(row) {
					*a += qi * qs * b * oscale;
				}
			}
		}
	}
	o
}

/// Dense causal attention of `q` [S][24][256] over `k` / `v` [S][2][256]:
/// `o` [S][6144].
fn attention(q: &[f32], k: &[f32], v: &[f32], s: usize) -> Vec<f32> {
	let mut o = vec![0f32; s * HQ * DA];
	let sc = 1.0 / (DA as f32).sqrt();
	for t in 0..s {
		for h in 0..HQ {
			let kh = h / (HQ / HKV);
			let qr = &q[(t * HQ + h) * DA..][..DA];
			let p: Vec<f32> = (0..=t)
				.map(|u| {
					qr.iter()
						.zip(&k[(u * HKV + kh) * DA..][..DA])
						.map(|(a, b)| a * b)
						.sum::<f32>() * sc
				})
				.collect();
			let m = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
			let e: Vec<f32> = p.iter().map(|x| (x - m).exp()).collect();
			let z: f32 = e.iter().sum();
			let ot = &mut o[(t * HQ + h) * DA..][..DA];
			for (u, w) in e.iter().enumerate() {
				for (a, b) in ot.iter_mut().zip(&v[(u * HKV + kh) * DA..][..DA]) {
					*a += w / z * b;
				}
			}
		}
	}
	o
}

/// `rotate(x * pre)` over `[.., W]` rows in Hadamard blocks of `b` (the
/// graphs' `prerot`).
fn rotate_host(x: &[f32], pre: &[f32], b: usize) -> Vec<f32> {
	let s = 1.0 / (b as f32).sqrt();
	let k = pre.len();
	let mut y = vec![0f32; x.len()];
	for (xr, yr) in x.chunks_exact(k).zip(y.chunks_exact_mut(k)) {
		let t: Vec<f32> = xr.iter().zip(pre).map(|(a, p)| a * p).collect();
		for (tb, yb) in t.chunks_exact(b).zip(yr.chunks_exact_mut(b)) {
			for (j, yv) in yb.iter_mut().enumerate() {
				*yv = tb
					.iter()
					.enumerate()
					.map(|(i, &v)| if (i & j).count_ones() & 1 == 1 { -v } else { v })
					.sum::<f32>() * s;
			}
		}
	}
	y
}

/// The streams after the first `nl` layers for the `S` tokens `ids` (from
/// zero state).
pub fn run(
	sim: &mut Sim,
	ck: &crate::st::Checkpoint,
	table: &ngram::Table,
	ids: &[u32],
	nl: usize,
	log: &mut dyn FnMut(&str),
) -> Result<Vec<f32>, String> {
	assert_eq!(ids.len(), S);
	let s = S;
	let oscale = npu::flash::OSCALE_P as f32;
	let mut x = fr::embed(ck, ids)?;
	let hash = ngram::Hash::new(npu::flash::VOCAB as u64, 0);
	let e = fr::ngram_embed(&hash, table, ids)?;
	let ones = vec![1f32; s];
	// FLASH_SIM_TF=1: teacher forcing: every layer starts from the float
	// reference's streams; logs each layer's error relative to the float
	// layer's update (`|x_q - x_f| / |x_f - x_in|`)
	let tf = std::env::var_os("FLASH_SIM_TF").is_some();
	let mut xr = x.clone();
	for l in 0..nl {
		let t0 = std::time::Instant::now();
		let xin = xr.clone();
		if tf {
			x = xr.clone();
			let ee = std::env::var_os("FLASH_SIM_NOPLE")
				.is_none()
				.then_some(&e[..]);
			fr::forward(ck, &fr::load_layer(ck, l)?, &mut xr, ee, s)?;
		}
		let fple = std::env::var("FLASH_SIM_F").is_ok_and(|v| v.split(',').any(|q| q == "ple"));
		if l == npu::flash::PLE_LAYER && fple && std::env::var_os("FLASH_SIM_NOPLE").is_none() {
			let layer = fr::load_layer(ck, l)?;
			let add = fr::ple(layer.ple.as_ref().unwrap(), &x, &e, s);
			x.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
		} else if l == npu::flash::PLE_LAYER && std::env::var_os("FLASH_SIM_NOPLE").is_none() {
			let back = 3 * ngram::N;
			let keep: Vec<i32> = (s..s + back).map(|i| i as i32).collect();
			let r = sim.run(
				"ple",
				|| graphs::ple(s),
				Some(l),
				vec![
					("x", In::F16(x.clone())),
					("e", In::F16(e.clone())),
					("cstate", In::F16(vec![0.0; back * HCW])),
					("keep", In::I32(keep)),
				],
				&["xo"],
			)?;
			x = r.into_iter().next().unwrap();
		}
		let (o, zg, inj) = if fr::is_attn(l) {
			let (mut cs, mut sn) = (vec![0f32; s * fr::RD], vec![0f32; s * fr::RD]);
			for t in 0..s {
				for i in 0..fr::RD / 2 {
					let w = 1e7f64.powf(-(i as f64) / (fr::RD / 2) as f64);
					let (a, b) = (t as f64 * w).sin_cos();
					for j in [i, i + fr::RD / 2] {
						cs[t * fr::RD + j] = b as f32;
						sn[t * fr::RD + j] = a as f32;
					}
				}
			}
			let r = sim.run(
				"attn_pre",
				|| graphs::attn_pre(s),
				Some(l),
				vec![
					("x", In::F16(x.clone())),
					("cos", In::F16(cs)),
					("sin", In::F16(sn)),
				],
				&["q", "gate", "k", "v", "inj"],
			)?;
			let o = attention(&r[0], &r[2], &r[3], s);
			(o, r[1].clone(), r[4].clone())
		} else {
			let keep: Vec<i32> = (s..s + 3).map(|i| i as i32).collect();
			let r = sim.run(
				"delta_pre",
				|| graphs::delta_pre(s),
				Some(l),
				vec![
					("x", In::F16(x.clone())),
					("cpast", In::F16(vec![0.0; 3 * NQKV])),
					("valid", In::F16(ones.clone())),
					("keep", In::I32(keep)),
				],
				&["qn", "kn", "v", "gb", "z", "inj"],
			)?;
			let o = recurrence(&r[0], &r[1], &r[2], &r[3], s, oscale);
			(o, r[4].clone(), r[5].clone())
		};
		let delta = !fr::is_attn(l);
		let r = sim.run(
			if delta { "post_delta" } else { "post_attn" },
			|| graphs::post(s, delta, oscale as f64),
			Some(l),
			vec![
				("x", In::F16(x.clone())),
				("o", In::F16(o)),
				("zg", In::F16(zg)),
				("inj", In::F16(inj)),
			],
			&["x1", "moe_q", "inj_m", "logits", "shared"],
		)?;
		let logits = &r[3];
		let (mut x1, mut moe_in) = (r[0].clone(), r[1].clone());
		let (mut inj_m, mut shared) = (r[2].clone(), r[4].clone());
		// routing
		let mut routes: Vec<[(u16, f32); TOPK]> = (0..s)
			.map(|t| {
				let lr = &logits[t * NE..][..NE];
				let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
				let p: Vec<f32> = lr.iter().map(|v| (v - m).exp()).collect();
				let mut idx: Vec<usize> = (0..NE).collect();
				idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
				let sum: f32 = idx[..TOPK].iter().map(|&i| p[i]).sum();
				std::array::from_fn(|j| (idx[j] as u16, p[idx[j]] / sum))
			})
			.collect();
		// FLASH_SIM_F=hc,route,exp,shared: those parts of the MoE in f32
		// from the checkpoint, on the float hyper-connection of `x1` (A/B of
		// the quantization error by part)
		// (`mix`: the attention / delta sublayer too, which implies hc,
		// route and shared; the routed experts then read the float MoE input
		// rotated on the host)
		let fparts = std::env::var("FLASH_SIM_F").unwrap_or_default();
		let mix = fparts.split(',').any(|q| q == "mix");
		let fpart = |p: &str| (mix && p != "exp") || fparts.split(',').any(|q| q == p);
		let mut fexp = None;
		if !fparts.is_empty() {
			let layer = fr::load_layer(ck, l)?;
			if mix {
				x1 = x.clone();
				let h = layer.hc_a.apply(&x1, s);
				let (o, wo) = match &layer.mixer {
					fr::Mixer::Delta(d) => (fr::delta_mixer(d, &h.mixed, s), &d.wo),
					fr::Mixer::Attn(a) => (fr::attn_mixer(a, &h.mixed, s), &a.wo),
				};
				let y = crate::la::linear(&o, s, H * D, wo, W);
				fr::inject(&mut x1, &y, &h.inj, s);
			}
			let hm = layer.hc_m.apply(&x1, s);
			if mix {
				let pre = unf16(
					&std::fs::read(sim.wdir.join(format!("fl{l}.moe.pre.in")))
						.map_err(|e| e.to_string())?,
				);
				moe_in = rotate_host(&hm.mixed, &pre, super::weights::rot_block(W));
			}
			let fr_routes = fr::route(&layer.moe, &hm.mixed, s);
			// routing differences: chosen experts not in the float top-k,
			// and the routing weights' L1 difference
			let (mut miss, mut l1) = (0usize, 0f64);
			for (a, b) in routes.iter().zip(&fr_routes) {
				for &(e, w) in a {
					match b.iter().find(|p| p.0 == e) {
						Some(p) => l1 += (w - p.1).abs() as f64,
						None => {
							miss += 1;
							l1 += w as f64;
						}
					}
				}
				l1 += b
					.iter()
					.filter(|p| !a.iter().any(|q| q.0 == p.0))
					.map(|p| p.1 as f64)
					.sum::<f64>();
			}
			log(&format!(
				"layer {l}: routing vs float: {miss} of {} choices differ, weight L1 {:.4} per token",
				s * TOPK,
				l1 / s as f64
			));
			if fpart("hc") {
				inj_m = hm.inj.clone();
			}
			if fpart("route") {
				routes = fr_routes;
			}
			if fpart("shared") {
				shared = fr::shared(&layer.moe, &hm.mixed, s).0;
			}
			if fpart("exp") {
				fexp = Some(fr::routed(ck, &layer.moe, &hm.mixed, s, &routes)?.0);
			} else if fpart("edq") {
				fexp = Some(host_experts(&sim.wdir, l, &moe_in, s, &routes)?);
			}
		}
		let mut slots: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
		let mut per: Vec<Vec<(usize, f32)>> = vec![Vec::new(); NE];
		for (t, r) in routes.iter().enumerate() {
			for &(e, w) in r {
				per[e as usize].push((t, w));
			}
		}
		const C: usize = 32;
		const G: usize = 32;
		for (ei, toks) in per.iter().enumerate() {
			for part in toks.chunks(C) {
				slots.push((ei, part.to_vec()));
			}
		}
		let rd = |n: &str| {
			let p = sim.wdir.join(format!("fl{l}.{n}.in"));
			std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
		};
		let (egu_w, egu_s, ed_w, ed_s) = (rd("egu.w")?, rd("egu.s")?, rd("ed.w")?, rd("ed.s")?);
		let ed_pre = rd("ed.pre")?;
		let (guw, gus, dw, ds) = (2 * FE * W / 2, 2 * FE * 2, W * FE / 2, W * 2);
		let mut y = vec![0f32; s * W];
		if let Some(f) = fexp {
			y = f;
			slots.clear();
		}
		for job in slots.chunks(G) {
			let mut ins: Vec<(String, In)> = Vec::new();
			for j in 0..G {
				let (ei, toks) = job.get(j).cloned().unwrap_or((0, Vec::new()));
				// the expert's tokens' rows of `moe_q` (row 0 for empty slots)
				let mut xe = Vec::with_capacity(C * W);
				for i in 0..C {
					let t = toks.get(i).map_or(0, |p| p.0);
					xe.extend_from_slice(&moe_in[t * W..][..W]);
				}
				ins.push((format!("x{j}"), In::F16(xe)));
				ins.push((
					format!("gu{j}.w"),
					In::Raw(egu_w[ei * guw..][..guw].to_vec()),
				));
				ins.push((
					format!("gu{j}.s"),
					In::Raw(egu_s[ei * gus..][..gus].to_vec()),
				));
				ins.push((format!("d{j}.w"), In::Raw(ed_w[ei * dw..][..dw].to_vec())));
				ins.push((format!("d{j}.s"), In::Raw(ed_s[ei * ds..][..ds].to_vec())));
				ins.push((
					format!("d{j}.pre"),
					In::Raw(ed_pre[ei * FE * 2..][..FE * 2].to_vec()),
				));
			}
			let names: Vec<String> = ins.iter().map(|p| p.0.clone()).collect();
			let rt: Vec<(&str, In)> = names
				.iter()
				.map(|n| n.as_str())
				.zip(ins.into_iter().map(|p| p.1))
				.collect();
			let r = sim.run("experts", || graphs::experts(G, C), Some(l), rt, &["y"])?;
			for (j, (_, toks)) in job.iter().enumerate() {
				for (i, &(t, w)) in toks.iter().enumerate() {
					let yr = &r[0][(j * C + i) * W..][..W];
					for (a, b) in y[t * W..][..W].iter_mut().zip(yr) {
						*a += w * b;
					}
				}
			}
		}
		// combine: x = x1 + inj_m * (shared + experts) per stream
		x = x1;
		for t in 0..s {
			for c in 0..HC {
				let j = inj_m[t * HC + c];
				for i in 0..W {
					x[t * HCW + c * W + i] += j * (shared[t * W + i] + y[t * W + i]);
				}
			}
		}
		if tf {
			let (mut d2, mut u2) = (0f64, 0f64);
			for ((a, b), c) in x.iter().zip(&xr).zip(&xin) {
				d2 += ((a - b) as f64).powi(2);
				u2 += ((b - c) as f64).powi(2);
			}
			log(&format!(
				"TF layer {l}: error / update {:.4}",
				(d2 / u2).sqrt()
			));
		}
		log(&format!(
			"layer {l}: {} expert slots, {:.1} s",
			slots.len(),
			t0.elapsed().as_secs_f64()
		));
	}
	Ok(x)
}

/// The routed experts on the host from the quantized files, `moe_q` as the
/// graphs get it, but each expert's down_proj input on the int8 grid of its
/// own largest value (`FLASH_SIM_F=edq`: what per-expert down ranges would
/// give; optimistic, the range is taken on the same tokens).
fn host_experts(
	wdir: &Path,
	l: usize,
	moe_q: &[f32],
	s: usize,
	routes: &[[(u16, f32); TOPK]],
) -> Result<Vec<f32>, String> {
	use super::weights::{rot_block, sg};
	use crate::gptq;
	use crate::la::linear;
	let rd = |n: &str| {
		let p = wdir.join(format!("fl{l}.{n}.in"));
		std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
	};
	let (egu_w, egu_s, ed_w, ed_s) = (rd("egu.w")?, rd("egu.s")?, rd("ed.w")?, rd("ed.s")?);
	let dpre = unf16(&rd("ed.pre")?);
	let (guw, gus, dw, ds) = (2 * FE * W / 2, 2 * FE * 2, W * FE / 2, W * 2);
	let deq = |w: &[u8], sc: &[u8], k: usize, mul: f32| -> Vec<f32> {
		let sc = unf16(sc);
		gptq::unpack4(w)
			.iter()
			.enumerate()
			.map(|(i, &c)| c as f32 * sc[i / k] / 256.0 * mul)
			.collect()
	};
	let mut tok_of: Vec<Vec<(usize, f32)>> = vec![Vec::new(); NE];
	for (t, r) in routes.iter().enumerate() {
		for &(e, w) in r {
			tok_of[e as usize].push((t, w));
		}
	}
	let mut y = vec![0f32; s * W];
	for (e, toks) in tok_of.iter().enumerate() {
		if toks.is_empty() {
			continue;
		}
		let n = toks.len();
		let mut xq = Vec::with_capacity(n * W);
		for &(t, _) in toks {
			xq.extend(
				moe_q[t * W..][..W]
					.iter()
					.map(|v| v.round().clamp(-128.0, 127.0)),
			);
		}
		let h = linear(
			&xq,
			n,
			W,
			&deq(&egu_w[e * guw..][..guw], &egu_s[e * gus..][..gus], W, 1.0),
			2 * FE,
		);
		let mut a = vec![0f32; n * FE];
		for i in 0..n {
			for j in 0..FE {
				a[i * FE + j] = crate::refm::silu(h[i * 2 * FE + j]) * h[i * 2 * FE + FE + j];
			}
		}
		gptq::rotate_rows_b(&mut a, FE, &sg(FE), rot_block(FE));
		let re = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
		a.iter_mut()
			.for_each(|v| *v = (*v * 127.0 / re).round().clamp(-128.0, 127.0));
		// the file's scales assume its range `rl`: x * re/127 * 127/rl
		let rl = 127.0 / dpre[e * FE].abs();
		let o = linear(
			&a,
			n,
			FE,
			&deq(&ed_w[e * dw..][..dw], &ed_s[e * ds..][..ds], FE, re / rl),
			W,
		);
		for (i, &(t, w)) in toks.iter().enumerate() {
			for (yv, ov) in y[t * W..][..W].iter_mut().zip(&o[i * W..][..W]) {
				*yv += w * ov;
			}
		}
	}
	Ok(y)
}

/// The quantization error of each quantized MatMul of layer `l` on its real
/// inputs (the float reference's, for `ids` from zero state): relative
/// error of `x W^T` with (a) the input on the static int8 grid only, (b)
/// the int4 weights only, (c) both (what the graphs do), (d) the input on a
/// per-token int8 grid instead, with the int4 weights. Lines for `log`.
pub fn qerr(
	ck: &crate::st::Checkpoint,
	wdir: &Path,
	table: &ngram::Table,
	ids: &[u32],
	l: usize,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	use super::weights::{rot_block, sg};
	use crate::gptq;
	use crate::la::linear;
	let s = ids.len();
	let hash = ngram::Hash::new(npu::flash::VOCAB as u64, 0);
	let e = fr::ngram_embed(&hash, table, ids)?;
	let mut x = fr::embed(ck, ids)?;
	for i in 0..l {
		fr::forward(ck, &fr::load_layer(ck, i)?, &mut x, Some(&e), s)?;
	}
	let layer = fr::load_layer(ck, l)?;
	let taps = fr::forward(ck, &layer, &mut x, Some(&e), s)?;
	let rd = |n: &str| {
		let p = wdir.join(format!("fl{l}.{n}.in"));
		std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
	};
	let rel = |a: &[f32], b: &[f32]| {
		let (mut d, mut r) = (0f64, 0f64);
		for (x, y) in a.iter().zip(b) {
			d += ((x - y) as f64).powi(2);
			r += (*y as f64).powi(2);
		}
		(d / r).sqrt()
	};
	let mut check = |name: &str,
	                 pre: (&str, usize),
	                 xin: &[f32],
	                 k: usize,
	                 w: &[f32],
	                 n: usize,
	                 file: Option<(Vec<u8>, Vec<u8>)>|
	 -> Result<(), String> {
		let s = xin.len() / k;
		let r = 127.0 / unf16(&rd(&format!("{}.pre", pre.0))?)[pre.1 * k].abs();
		let mut xr = xin.to_vec();
		gptq::rotate_rows_b(&mut xr, k, &sg(k), rot_block(k));
		let fq = |v: f32, r: f32| (v * 127.0 / r).round().clamp(-128.0, 127.0) * r / 127.0;
		let xs: Vec<f32> = xr.iter().map(|&v| fq(v, r)).collect();
		let mut xd = xr.clone();
		let mut clip = 0usize;
		for row in xd.chunks_exact_mut(k) {
			let m = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
			clip += (m > r) as usize;
			row.iter_mut().for_each(|v| *v = fq(*v, m));
		}
		let (cw, cs) = match file {
			Some(f) => f,
			None => (rd(&format!("{name}.w"))?, rd(&format!("{name}.s"))?),
		};
		let codes = if cw.len() == n * k {
			cw.iter().map(|&v| v as i8).collect()
		} else {
			gptq::unpack4(&cw)
		};
		let sc = unf16(&cs);
		let wq: Vec<f32> = codes
			.iter()
			.enumerate()
			.map(|(i, &c)| c as f32 * sc[i / k] / 256.0 * 127.0 / r)
			.collect();
		let mut wr = w.to_vec();
		gptq::rotate_rows_b(&mut wr, k, &sg(k), rot_block(k));
		let yf = linear(xin, s, k, w, n);
		let rtn = |g: usize| -> Vec<f32> {
			let sc = gptq::row_scales(&wr, n * k / g, g);
			wr.iter()
				.enumerate()
				.map(|(i, &v)| {
					let s = sc[i / g];
					(v / s).round_ties_even().clamp(-8.0, 7.0) * s
				})
				.collect()
		};
		let (q8, s8) = gptq::quant_i8(&wr, n, k);
		let w8: Vec<f32> = q8
			.iter()
			.enumerate()
			.map(|(i, &c)| c as f32 * s8[i / k])
			.collect();
		log(&format!(
			"layer {l} {name:>4}: weights alone: RTN per row {:.4}, RTN g128 {:.4}, RTN g64 {:.4}, int8 per row {:.4}",
			rel(&linear(&xr, s, k, &rtn(k), n), &yf),
			rel(&linear(&xr, s, k, &rtn(128.min(k)), n), &yf),
			rel(&linear(&xr, s, k, &rtn(64), n), &yf),
			rel(&linear(&xr, s, k, &w8, n), &yf),
		));
		log(&format!(
			"layer {l} {name:>4}: range {r:.2} ({clip} of {s} tokens past it): int8 in {:.4}, file w {:.4}, both {:.4}, per-token int8 + file w {:.4}",
			rel(&linear(&xs, s, k, &wr, n), &yf),
			rel(&linear(&xr, s, k, &wq, n), &yf),
			rel(&linear(&xs, s, k, &wq, n), &yf),
			rel(&linear(&xd, s, k, &wq, n), &yf),
		));
		Ok(())
	};
	match &layer.mixer {
		fr::Mixer::Delta(d) => {
			check("qkv", ("proj", 0), &taps.proj, W, &d.wqkv, NQKV, None)?;
			check("z", ("proj", 0), &taps.proj, W, &d.wz, H * D, None)?;
			check("o", ("out", 0), &taps.out, H * D, &d.wo, W, None)?;
		}
		fr::Mixer::Attn(a) => {
			check("q", ("proj", 0), &taps.proj, W, &a.wq, 2 * HQ * DA, None)?;
			check("k", ("proj", 0), &taps.proj, W, &a.wk, HKV * DA, None)?;
			check("v", ("proj", 0), &taps.proj, W, &a.wv, HKV * DA, None)?;
			check("o", ("out", 0), &taps.out, HQ * DA, &a.wo, W, None)?;
		}
	}
	let mut sgu = layer.moe.sg.clone();
	sgu.extend_from_slice(&layer.moe.su);
	check("sgu", ("moe", 0), &taps.moe, W, &sgu, 2 * FE, None)?;
	check("sd", ("sd", 0), &taps.sdown, FE, &layer.moe.sd, W, None)?;
	// the three routed experts with the most tokens
	let mut n_of = vec![0usize; NE];
	for r in &taps.routes {
		for &(e, _) in r {
			n_of[e as usize] += 1;
		}
	}
	let mut top: Vec<usize> = (0..NE).collect();
	top.sort_by_key(|&e| std::cmp::Reverse(n_of[e]));
	let (egu_w, egu_s, ed_w, ed_s) = (rd("egu.w")?, rd("egu.s")?, rd("ed.w")?, rd("ed.s")?);
	let (guw, gus, dw, ds) = (2 * FE * W / 2, 2 * FE * 2, W * FE / 2, W * 2);
	for &e in &top[..3] {
		let (gu, dn) = fr::expert(ck, &layer.moe, e)?;
		let (toks, din) = &taps.edown[e];
		let mut xe = Vec::new();
		for &t in toks {
			xe.extend_from_slice(&taps.moe[t as usize * W..][..W]);
		}
		check(
			&format!("egu{e}:{}", toks.len()),
			("moe", 0),
			&xe,
			W,
			&gu,
			2 * FE,
			Some((
				egu_w[e * guw..][..guw].to_vec(),
				egu_s[e * gus..][..gus].to_vec(),
			)),
		)?;
		check(
			&format!("ed{e}"),
			("ed", e),
			din,
			FE,
			&dn,
			W,
			Some((ed_w[e * dw..][..dw].to_vec(), ed_s[e * ds..][..ds].to_vec())),
		)?;
	}
	Ok(())
}

/// `nbcheck flash-npucheck OUT KIND...`: each graph kind's compiled blob
/// on the NPU against its graph on the OpenVINO CPU, on random runtime
/// inputs (a position past 0, so RoPE and the conv windows matter) and a
/// real layer's files: finds a blob the NPU compile gets wrong.
pub fn npu_check(wdir: &Path, kinds: &[String]) -> Result<(), String> {
	let tmp = std::env::temp_dir().join("flashnpucheck");
	let mut cpu = Sim::new(wdir, &tmp)?;
	let mut dev = Sim::new(wdir, &tmp.join("npu"))?;
	dev.npu_kinds = kinds.to_vec();
	let all = super::graph_kinds();
	let mut seed = 0x9e37_79b9_7f4a_7c15u64;
	let mut rnd = move || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		(seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
	};
	for kind in kinds {
		let base = kind.strip_suffix("_d1").unwrap_or(kind);
		let g = &all
			.iter()
			.find(|k| k.0 == kind)
			.ok_or(format!("no kind {kind}"))?
			.1;
		let l = match base {
			"attn_pre" | "post_attn" => 3,
			"ple" => 1,
			_ => 0,
		};
		let mut rt: Vec<(String, In)> = Vec::new();
		for (name, ty, shape) in g.inputs() {
			let n: usize = shape.iter().product();
			let s = shape.get(1).copied().unwrap_or(1);
			let v = match name.as_str() {
				"keep" => In::I32((0..n).map(|j| (s + j) as i32).collect()),
				"valid" => In::F16(vec![1.0; n]),
				"cos" | "sin" => In::F16(
					(0..n)
						.map(|i| {
							let (t, j) = (i / 64, i % 32);
							let a = (7 + t) as f64 * 1e7f64.powf(-(j as f64) / 32.0);
							(if name == "cos" { a.cos() } else { a.sin() }) as f32
						})
						.collect(),
				),
				"inj" => In::F16((0..n).map(|_| (1.0 + 0.5 * rnd()) as f32).collect()),
				"x" | "cpast" | "o" | "zg" | "e" | "cstate" | "mixed" | "hinj" | "m" => {
					In::F16((0..n).map(|_| rnd() as f32).collect())
				}
				_ => continue,
			};
			let _ = ty;
			rt.push((name, v));
		}
		let outs = g.outputs();
		let os: Vec<&str> = outs.iter().map(|s| s.as_str()).collect();
		let mk = || {
			super::graph_kinds()
				.into_iter()
				.find(|k| k.0 == kind)
				.unwrap()
				.1
		};
		let ins = || rt.iter().map(|(n, v)| (n.as_str(), dup_in(v))).collect();
		let a = cpu.run(kind, mk, Some(l), ins(), &os)?;
		let b = dev.run(kind, mk, Some(l), ins(), &os)?;
		for ((o, x), y) in outs.iter().zip(&a).zip(&b) {
			let (mut d2, mut r2) = (0f64, 0f64);
			for (p, q) in x.iter().zip(y) {
				d2 += ((p - q) as f64).powi(2);
				r2 += (*p as f64).powi(2);
			}
			println!(
				"{kind} {o}: relative error {:.5} (rms {:.4})",
				(d2 / r2.max(1e-30)).sqrt(),
				(r2 / x.len() as f64).sqrt()
			);
		}
	}
	Ok(())
}

fn dup_in(v: &In) -> In {
	match v {
		In::F16(v) => In::F16(v.clone()),
		In::I32(v) => In::I32(v.clone()),
		In::Raw(v) => In::Raw(v.clone()),
	}
}
