//! Qwen3.8-Flash-Next in f32 from the BF16 checkpoint, one layer at a time:
//! the reference the Flash-Next graph sets are calibrated against and checked
//! with (as [`crate::refm`] is for the 27B). A sequence starts from zero state
//! at position 0; text positions only.
//!
//! The model: the residual is `HC`
//! streams of `W`; every sublayer reads a mix of the normed streams
//! (hyper-connection) and adds its output to each stream with a weight of
//! its own. Layer 1 first adds the PLE term (hashed n-gram embeddings). The
//! mixers are the 27B's delta-net (sigmoid output gate) and a sparse
//! attention whose indexer picks 512 blocks of 4 keys per query; every layer
//! ends in a 512-expert MoE (top 10 + a shared expert).

use crate::la::linear;
use crate::refm::{rmsnorm, sigmoid, silu, DeltaCore};
use crate::st::Checkpoint;
use npu::flash::ngram;
use npu::flash::{HC, HC_RANK, W};
use quant::parallel::par_for;

pub const HCW: usize = HC * W;
/// Delta-net layers (the 27B's shape).
pub const H: usize = 48;
pub const HK: usize = 16;
pub const D: usize = 128;
pub const NQKV: usize = 2 * HK * D + H * D;
/// Attention layers: query heads, key / value heads, head width, rotary dims.
pub const HQ: usize = 24;
pub const HKV: usize = 2;
pub const DA: usize = 256;
pub const RD: usize = 64;
/// Indexer: query heads, head width, keys per block, blocks kept per query.
pub const IH: usize = 4;
pub const ID: usize = 128;
pub const IBLK: usize = 4;
pub const ITOP: usize = 512;
/// MoE: experts, chosen per token, expert / shared-expert width.
pub const NE: usize = 512;
pub const TOPK: usize = 10;
pub const FE: usize = 640;
pub const EPS: f32 = 1e-6;

pub fn is_attn(l: usize) -> bool {
	l % 4 == 3
}

const P: &str = "model.language_model.";

/// A hyper-connection: the per-stream norm, the low-rank input mix and (not
/// for the final mixer) the four injection weights.
pub struct Hc {
	/// `1 + w`, `[HCW]`
	pub norm: Vec<f32>,
	/// `[HC_RANK][HCW]`
	pub down: Vec<f32>,
	/// `[HCW][HC_RANK]`
	pub up: Vec<f32>,
	/// `[HC][HCW]`
	pub inject: Option<Vec<f32>>,
}

/// What a hyper-connection hands its sublayer: the mixed input `[s][W]`, the
/// injection weights `[s][HC]` and the normed streams (the input of the
/// mix's MatMuls, `[s][HCW]`).
pub struct HcOut {
	pub mixed: Vec<f32>,
	pub inj: Vec<f32>,
	pub xn: Vec<f32>,
}

impl Hc {
	pub fn load(ck: &Checkpoint, pre: &str, inject: bool) -> Result<Hc, String> {
		Ok(Hc {
			norm: norm_w(ck, &format!("{pre}hc_norm.weight"), HCW)?,
			down: ck.mat(&format!("{pre}input_mix_weight_down.weight"), HC_RANK, HCW)?,
			up: ck.mat(&format!("{pre}input_mix_weight_up.weight"), HCW, HC_RANK)?,
			inject: if inject {
				Some(ck.mat(&format!("{pre}block_inject_weight.weight"), HC, HCW)?)
			} else {
				None
			},
		})
	}

	/// The final mixer (`hyper_connection_mixer`, no injection).
	pub fn load_final(ck: &Checkpoint) -> Result<Hc, String> {
		Hc::load(ck, &format!("{P}hyper_connection_mixer."), false)
	}

	pub fn apply(&self, x: &[f32], s: usize) -> HcOut {
		let xn = rmsnorm_g(x, &self.norm);
		let mut m = linear(&xn, s, HCW, &self.down, HC_RANK);
		m.iter_mut().for_each(|v| *v = silu(*v / HC as f32));
		let mut m = linear(&m, s, HC_RANK, &self.up, HCW);
		m.iter_mut().for_each(|v| *v = sigmoid(*v));
		let mut mixed = vec![0f32; s * W];
		for t in 0..s {
			let o = &mut mixed[t * W..][..W];
			for c in 0..HC {
				let mr = &m[t * HCW + c * W..][..W];
				let xr = &xn[t * HCW + c * W..][..W];
				for ((a, b), y) in mr.iter().zip(xr).zip(o.iter_mut()) {
					*y += a * b / HC as f32;
				}
			}
		}
		let inj = match &self.inject {
			Some(w) => {
				let mut j = linear(&xn, s, HCW, w, HC);
				j.iter_mut()
					.for_each(|v| *v = 2.0 * sigmoid(*v / HC as f32));
				j
			}
			None => Vec::new(),
		};
		HcOut { mixed, inj, xn }
	}
}

/// `x[t][c] += y[t] * inj[t][c]` for every stream `c`.
pub fn inject(x: &mut [f32], y: &[f32], inj: &[f32], s: usize) {
	for t in 0..s {
		for c in 0..HC {
			let j = inj[t * HC + c];
			for (a, b) in x[t * HCW + c * W..][..W].iter_mut().zip(&y[t * W..][..W]) {
				*a += b * j;
			}
		}
	}
}

/// An RMSNorm of each `W`-wide stream of `[s][HCW]` rows, weight `[HCW]`.
pub fn rmsnorm_g(x: &[f32], w: &[f32]) -> Vec<f32> {
	let mut y = rmsnorm(x, W, &vec![1.0; W]);
	for r in y.chunks_exact_mut(HCW) {
		r.iter_mut().zip(w).for_each(|(a, b)| *a *= b);
	}
	y
}

/// RMSNorm weights are stored less one (`1 + w`), as in the 27B.
pub fn norm_w(ck: &Checkpoint, name: &str, n: usize) -> Result<Vec<f32>, String> {
	crate::refm::norm_w(ck, name, n)
}

pub struct Delta {
	pub wqkv: Vec<f32>,
	pub wz: Vec<f32>,
	pub wa: Vec<f32>,
	pub wb: Vec<f32>,
	pub conv: Vec<f32>,
	pub dtb: Vec<f32>,
	pub acoef: Vec<f32>,
	pub gnw: Vec<f32>,
	pub wo: Vec<f32>,
}

pub struct Attn {
	/// `[24][2][256]` rows: per head the query rows, then the gate rows
	pub wq: Vec<f32>,
	pub wk: Vec<f32>,
	pub wv: Vec<f32>,
	pub qnw: Vec<f32>,
	pub knw: Vec<f32>,
	pub wo: Vec<f32>,
	/// Indexer: `[IH * ID + ID][W]` (queries, then the one key head)
	pub wi: Vec<f32>,
	pub iqn: Vec<f32>,
	pub ikn: Vec<f32>,
}

pub enum Mixer {
	Delta(Delta),
	Attn(Attn),
}

/// The MoE block without its routed experts, which are read per use
/// ([`expert`]): 512 x 4.9M parameters per layer.
pub struct Moe {
	/// router `[NE][W]`
	pub gate: Vec<f32>,
	pub sg: Vec<f32>,
	pub su: Vec<f32>,
	pub sd: Vec<f32>,
	/// shared-expert gate `[W]`
	pub sgate: Vec<f32>,
	pub pre: String,
}

pub struct Ple {
	pub hash: ngram::Hash,
	/// `[HCW][W]`
	pub key: Vec<f32>,
	/// `[W][W]`
	pub value: Vec<f32>,
	pub nkey: Vec<f32>,
	pub nquery: Vec<f32>,
	pub nconv: Vec<f32>,
	/// `[HCW][4]`, dilation `ngram::N`
	pub conv: Vec<f32>,
}

pub struct Layer {
	pub ple: Option<Ple>,
	pub hc_a: Hc,
	pub mixer: Mixer,
	pub hc_m: Hc,
	pub moe: Moe,
}

pub fn load_layer(ck: &Checkpoint, l: usize) -> Result<Layer, String> {
	let p = format!("{P}layers.{l}.");
	let mixer = if is_attn(l) {
		let a = format!("{p}self_attn.");
		Mixer::Attn(Attn {
			wq: ck.mat(&format!("{a}q_proj.weight"), 2 * HQ * DA, W)?,
			wk: ck.mat(&format!("{a}k_proj.weight"), HKV * DA, W)?,
			wv: ck.mat(&format!("{a}v_proj.weight"), HKV * DA, W)?,
			qnw: norm_w(ck, &format!("{a}q_norm.weight"), DA)?,
			knw: norm_w(ck, &format!("{a}k_norm.weight"), DA)?,
			wo: ck.mat(&format!("{a}o_proj.weight"), W, HQ * DA)?,
			wi: ck.mat(
				&format!("{a}indexer.index_qk_proj.weight"),
				(IH + 1) * ID,
				W,
			)?,
			iqn: norm_w(ck, &format!("{a}indexer.q_layernorm.weight"), ID)?,
			ikn: norm_w(ck, &format!("{a}indexer.k_layernorm.weight"), ID)?,
		})
	} else {
		let q = format!("{p}linear_attn.");
		Mixer::Delta(Delta {
			wqkv: ck.mat(&format!("{q}in_proj_qkv.weight"), NQKV, W)?,
			wz: ck.mat(&format!("{q}in_proj_z.weight"), H * D, W)?,
			wa: ck.mat(&format!("{q}in_proj_a.weight"), H, W)?,
			wb: ck.mat(&format!("{q}in_proj_b.weight"), H, W)?,
			conv: ck.mat(&format!("{q}conv1d.weight"), NQKV, 4)?,
			dtb: ck.vec(&format!("{q}dt_bias"), H)?,
			acoef: ck
				.vec(&format!("{q}A_log"), H)?
				.iter()
				.map(|a| -a.exp())
				.collect(),
			gnw: ck.vec(&format!("{q}norm.weight"), D)?,
			wo: ck.mat(&format!("{q}out_proj.weight"), W, H * D)?,
		})
	};
	let m = format!("{p}mlp.");
	let moe = Moe {
		gate: ck.mat(&format!("{m}gate.weight"), NE, W)?,
		sg: ck.mat(&format!("{m}shared_expert.gate_proj.weight"), FE, W)?,
		su: ck.mat(&format!("{m}shared_expert.up_proj.weight"), FE, W)?,
		sd: ck.mat(&format!("{m}shared_expert.down_proj.weight"), W, FE)?,
		sgate: ck.vec(&format!("{m}shared_expert_gate.weight"), W)?,
		pre: m,
	};
	let ple = if l == npu::flash::PLE_LAYER {
		let q = format!("{p}ple.");
		Some(Ple {
			hash: ngram::Hash::new(npu::flash::VOCAB as u64, 0),
			key: ck.mat(&format!("{q}key_proj.weight"), HCW, W)?,
			value: ck.mat(&format!("{q}value_proj.weight"), W, W)?,
			nkey: norm_w(ck, &format!("{q}norm_key.weight"), HCW)?,
			nquery: norm_w(ck, &format!("{q}norm_query.weight"), HCW)?,
			nconv: norm_w(ck, &format!("{q}norm_conv.weight"), HCW)?,
			conv: ck.mat(&format!("{q}conv1d.weight"), HCW, 4)?,
		})
	} else {
		None
	};
	Ok(Layer {
		ple,
		hc_a: Hc::load(ck, &format!("{p}attn_hyper_connection."), true)?,
		mixer,
		hc_m: Hc::load(ck, &format!("{p}mlp_hyper_connection."), true)?,
		moe,
	})
}

/// Routed expert `e` of the MoE `m`: (`gate_up` `[2 FE][W]`, `down` `[W][FE]`).
pub fn expert(ck: &Checkpoint, m: &Moe, e: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
	Ok((
		ck.rows(&format!("{}experts.gate_up_proj", m.pre), e, e + 1)?,
		ck.rows(&format!("{}experts.down_proj", m.pre), e, e + 1)?,
	))
}

/// The checkpoint's own hash buffers of the PLE layer (to check
/// [`ngram::Hash::new`] against).
pub fn ple_hash_of_checkpoint(ck: &Checkpoint) -> Result<ngram::Hash, String> {
	let q = format!("{P}layers.{}.ple.ple_embedding.", npu::flash::PLE_LAYER);
	let m = ck.i64s(&format!("{q}layer_multipliers"))?;
	let sz = ck.i64s(&format!("{q}ngram_heads_vocab_sizes"))?;
	let of = ck.i64s(&format!("{q}ngram_heads_offsets"))?;
	if m.len() != ngram::N || sz.len() != ngram::HEADS || of.len() != ngram::HEADS {
		return Err("PLE hash buffers: unexpected sizes".into());
	}
	Ok(ngram::Hash {
		mult: m.try_into().unwrap(),
		sizes: sz.try_into().unwrap(),
		offs: of.try_into().unwrap(),
	})
}

fn rope_cs(p: usize) -> [(f32, f32); RD / 2] {
	std::array::from_fn(|i| {
		let w = 1e7f64.powf(-(i as f64) / (RD / 2) as f64);
		let (s, c) = (p as f64 * w).sin_cos();
		(c as f32, s as f32)
	})
}

/// Rotate the first [`RD`] channels of a row (rotate_half pairs `i`, `i + 32`).
fn rope_row(r: &mut [f32], cs: &[(f32, f32); RD / 2]) {
	for i in 0..RD / 2 {
		let (a, b) = (r[i], r[i + RD / 2]);
		r[i] = a * cs[i].0 - b * cs[i].1;
		r[i + RD / 2] = b * cs[i].0 + a * cs[i].1;
	}
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: every use hands disjoint ranges to the tasks.
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// For each query `t` the key positions it attends to (sorted): the tokens of
/// its top-[`ITOP`] complete blocks by indexer score, then the incomplete
/// tail. All of `0..=t` while it has no more than `ITOP` complete blocks.
pub fn indexer(a: &Attn, x: &[f32], s: usize) -> Vec<Vec<u32>> {
	let qk = linear(x, s, W, &a.wi, (IH + 1) * ID);
	let mut q = vec![0f32; s * IH * ID];
	let mut raw = vec![0f32; s * ID];
	for t in 0..s {
		q[t * IH * ID..][..IH * ID].copy_from_slice(&qk[t * (IH + 1) * ID..][..IH * ID]);
		raw[t * ID..][..ID].copy_from_slice(&qk[t * (IH + 1) * ID + IH * ID..][..ID]);
	}
	let mut q = rmsnorm(&q, ID, &a.iqn);
	for t in 0..s {
		let cs = rope_cs(t);
		for h in 0..IH {
			rope_row(&mut q[(t * IH + h) * ID..][..ID], &cs);
		}
	}
	let nb = s / IBLK;
	let mut bk = vec![0f32; nb * ID];
	for b in 0..nb {
		let o = &mut bk[b * ID..][..ID];
		for j in 0..IBLK {
			for (a, v) in o.iter_mut().zip(&raw[(b * IBLK + j) * ID..][..ID]) {
				*a += v / IBLK as f32;
			}
		}
	}
	let mut bk = rmsnorm(&bk, ID, &a.ikn);
	for b in 0..nb {
		rope_row(&mut bk[b * ID..][..ID], &rope_cs(b * IBLK));
	}
	let mut out = vec![Vec::new(); s];
	let op = std::sync::Mutex::new(&mut out);
	par_for(s, |t| {
		let n = (t + 1) / IBLK;
		let mut sel: Vec<u32> = if n <= ITOP {
			(0..(n * IBLK) as u32).collect()
		} else {
			let qr = &q[t * IH * ID..][..IH * ID];
			let mut sc: Vec<(f32, u32)> = (0..n)
				.map(|b| {
					let kr = &bk[b * ID..][..ID];
					let v: f32 = (0..IH)
						.map(|h| {
							let d: f32 =
								qr[h * ID..][..ID].iter().zip(kr).map(|(a, b)| a * b).sum();
							d.max(0.0)
						})
						.sum();
					(v / (ID as f32).sqrt(), b as u32)
				})
				.collect();
			sc.select_nth_unstable_by(ITOP - 1, |a, b| b.0.total_cmp(&a.0));
			let mut bs: Vec<u32> = sc[..ITOP].iter().map(|p| p.1).collect();
			bs.sort_unstable();
			bs.iter()
				.flat_map(|&b| b * IBLK as u32..(b + 1) * IBLK as u32)
				.collect()
		};
		sel.extend((n * IBLK) as u32..=t as u32);
		op.lock().unwrap()[t] = sel;
	});
	out
}

/// The attention mixer: (the gated output `[s][6144]` = the input of
/// o_proj).
pub fn attn_mixer(a: &Attn, x: &[f32], s: usize) -> Vec<f32> {
	let keys = indexer(a, x, s);
	let qg = linear(x, s, W, &a.wq, 2 * HQ * DA);
	let k = linear(x, s, W, &a.wk, HKV * DA);
	let v = linear(x, s, W, &a.wv, HKV * DA);
	let mut q = vec![0f32; s * HQ * DA];
	let mut gate = vec![0f32; s * HQ * DA];
	for t in 0..s {
		for h in 0..HQ {
			q[(t * HQ + h) * DA..][..DA].copy_from_slice(&qg[(t * HQ + h) * 2 * DA..][..DA]);
			gate[(t * HQ + h) * DA..][..DA]
				.copy_from_slice(&qg[(t * HQ + h) * 2 * DA + DA..][..DA]);
		}
	}
	let mut q = rmsnorm(&q, DA, &a.qnw);
	let mut k = rmsnorm(&k, DA, &a.knw);
	for t in 0..s {
		let cs = rope_cs(t);
		for h in 0..HQ {
			rope_row(&mut q[(t * HQ + h) * DA..][..DA], &cs);
		}
		for h in 0..HKV {
			rope_row(&mut k[(t * HKV + h) * DA..][..DA], &cs);
		}
	}
	let scale = 1.0 / (DA as f32).sqrt();
	let mut o = vec![0f32; s * HQ * DA];
	let op = Ptr(o.as_mut_ptr());
	par_for(s * HQ, |i| {
		let op = op;
		let (t, h) = (i / HQ, i % HQ);
		let kh = h / (HQ / HKV);
		let qr = &q[(t * HQ + h) * DA..][..DA];
		let sc: Vec<f32> = keys[t]
			.iter()
			.map(|&u| {
				qr.iter()
					.zip(&k[(u as usize * HKV + kh) * DA..][..DA])
					.map(|(a, b)| a * b)
					.sum::<f32>() * scale
			})
			.collect();
		let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let p: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
		let sum: f32 = p.iter().sum();
		// SAFETY: task i owns o[t][h][..]
		let ot = unsafe { std::slice::from_raw_parts_mut(op.0.add((t * HQ + h) * DA), DA) };
		ot.fill(0.0);
		for (&u, pu) in keys[t].iter().zip(&p) {
			let w = pu / sum;
			for (a, b) in ot.iter_mut().zip(&v[(u as usize * HKV + kh) * DA..][..DA]) {
				*a += w * b;
			}
		}
	});
	for (ov, g) in o.iter_mut().zip(&gate) {
		*ov *= sigmoid(*g);
	}
	o
}

pub fn delta_mixer(d: &Delta, x: &[f32], s: usize) -> Vec<f32> {
	let qkv = linear(x, s, W, &d.wqkv, NQKV);
	let z = linear(x, s, W, &d.wz, H * D);
	let pa = linear(x, s, W, &d.wa, H);
	let pb = linear(x, s, W, &d.wb, H);
	DeltaCore {
		conv: &d.conv,
		dtb: &d.dtb,
		acoef: &d.acoef,
		gnw: &d.gnw,
		sigmoid_gate: true,
	}
	.run(&qkv, &z, &pa, &pb, s)
}

/// The routing of `[s][W]` inputs: per token the top-[`TOPK`] experts and
/// their weights (softmax over all experts, renormalized over the chosen).
pub fn route(m: &Moe, x: &[f32], s: usize) -> Vec<[(u16, f32); TOPK]> {
	let lg = linear(x, s, W, &m.gate, NE);
	lg.chunks_exact(NE)
		.map(|r| {
			let mx = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
			let p: Vec<f32> = r.iter().map(|v| (v - mx).exp()).collect();
			let mut idx: Vec<u16> = (0..NE as u16).collect();
			idx.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
			let sum: f32 = idx[..TOPK].iter().map(|&e| p[e as usize]).sum();
			std::array::from_fn(|j| (idx[j], p[idx[j] as usize] / sum))
		})
		.collect()
}

/// What one layer's MatMuls read (calibration taps), `[s][k]` each.
#[derive(Default)]
pub struct Taps {
	/// the normed streams the two hyper-connections' MatMuls read
	pub hc_a: Vec<f32>,
	pub hc_m: Vec<f32>,
	/// the mixer's input (its projections') and o_proj / out_proj's input
	pub proj: Vec<f32>,
	pub out: Vec<f32>,
	/// the MoE input (router, shared expert, every routed expert's gate_up)
	pub moe: Vec<f32>,
	/// the shared expert's down_proj input
	pub sdown: Vec<f32>,
	/// per routed expert: the tokens it saw and its down_proj inputs `[n][FE]`
	pub edown: Vec<(Vec<u32>, Vec<f32>)>,
	pub routes: Vec<[(u16, f32); TOPK]>,
}

/// The MoE over `x` `[s][W]`: (output `[s][W]`, shared down input, routes,
/// per-expert down inputs).
#[allow(clippy::type_complexity)]
fn moe(
	ck: &Checkpoint,
	m: &Moe,
	x: &[f32],
	s: usize,
) -> Result<
	(
		Vec<f32>,
		Vec<f32>,
		Vec<[(u16, f32); TOPK]>,
		Vec<(Vec<u32>, Vec<f32>)>,
	),
	String,
> {
	let routes = route(m, x, s);
	let (mut y, edown) = routed(ck, m, x, s, &routes)?;
	let (so, sd) = shared(m, x, s);
	y.iter_mut().zip(&so).for_each(|(a, b)| *a += b);
	Ok((y, sd, routes, edown))
}

/// The routed experts' weighted sum `[s][W]` over `x` `[s][W]` for the given
/// routes, and per expert the tokens it saw and its down_proj inputs.
#[allow(clippy::type_complexity)]
pub fn routed(
	ck: &Checkpoint,
	m: &Moe,
	x: &[f32],
	s: usize,
	routes: &[[(u16, f32); TOPK]],
) -> Result<(Vec<f32>, Vec<(Vec<u32>, Vec<f32>)>), String> {
	let mut tok_of: Vec<Vec<(u32, f32)>> = vec![Vec::new(); NE];
	for (t, r) in routes.iter().enumerate() {
		for &(e, w) in r {
			tok_of[e as usize].push((t as u32, w));
		}
	}
	let mut y = vec![0f32; s * W];
	let mut edown = vec![(Vec::new(), Vec::new()); NE];
	for e in 0..NE {
		if tok_of[e].is_empty() {
			continue;
		}
		let (gu, dn) = expert(ck, m, e)?;
		let n = tok_of[e].len();
		let mut xe = Vec::with_capacity(n * W);
		for &(t, _) in &tok_of[e] {
			xe.extend_from_slice(&x[t as usize * W..][..W]);
		}
		let h = linear(&xe, n, W, &gu, 2 * FE);
		let mut a = vec![0f32; n * FE];
		for i in 0..n {
			for j in 0..FE {
				a[i * FE + j] = silu(h[i * 2 * FE + j]) * h[i * 2 * FE + FE + j];
			}
		}
		let o = linear(&a, n, FE, &dn, W);
		for (i, &(t, w)) in tok_of[e].iter().enumerate() {
			for (yv, ov) in y[t as usize * W..][..W].iter_mut().zip(&o[i * W..][..W]) {
				*yv += w * ov;
			}
		}
		edown[e] = (tok_of[e].iter().map(|p| p.0).collect(), a);
	}
	Ok((y, edown))
}

/// The gated shared expert over `x` `[s][W]`: (output `[s][W]`, its
/// down_proj input `[s][FE]`).
pub fn shared(m: &Moe, x: &[f32], s: usize) -> (Vec<f32>, Vec<f32>) {
	let mut y = vec![0f32; s * W];
	let g = linear(x, s, W, &m.sg, FE);
	let mut sd = linear(x, s, W, &m.su, FE);
	sd.iter_mut().zip(&g).for_each(|(u, g)| *u *= silu(*g));
	let so = linear(&sd, s, FE, &m.sd, W);
	for t in 0..s {
		let gt: f32 = x[t * W..][..W]
			.iter()
			.zip(&m.sgate)
			.map(|(a, b)| a * b)
			.sum();
		let gt = sigmoid(gt);
		for (yv, ov) in y[t * W..][..W].iter_mut().zip(&so[t * W..][..W]) {
			*yv = gt * ov;
		}
	}
	(y, sd)
}

/// The PLE term added to the streams `x` `[s][HCW]` before layer 1, from the
/// tokens' n-gram embeddings `e` `[s][W]`.
pub fn ple(p: &Ple, x: &[f32], e: &[f32], s: usize) -> Vec<f32> {
	let key = rmsnorm_g(&linear(e, s, W, &p.key, HCW), &p.nkey);
	let val = linear(e, s, W, &p.value, W);
	let q = rmsnorm_g(x, &p.nquery);
	let mut gv = vec![0f32; s * HCW];
	for t in 0..s {
		for c in 0..HC {
			let o = t * HCW + c * W;
			let g: f32 = key[o..][..W]
				.iter()
				.zip(&q[o..][..W])
				.map(|(a, b)| a * b)
				.sum::<f32>()
				/ (W as f32).sqrt();
			let g = g.abs().max(1e-6).sqrt() * g.signum();
			let g = sigmoid(g);
			for (a, b) in gv[o..][..W].iter_mut().zip(&val[t * W..][..W]) {
				*a = g * b;
			}
		}
	}
	let gn = rmsnorm_g(&gv, &p.nconv);
	// causal depthwise conv, kernel 4, dilation N: out[t] = sum_j w[j] in[t - (3 - j) N]
	let mut out = gv;
	for t in 0..s {
		for ch in 0..HCW {
			let mut a = 0f32;
			for j in 0..4 {
				let back = (3 - j) * ngram::N;
				if t >= back {
					a += p.conv[ch * 4 + j] * gn[(t - back) * HCW + ch];
				}
			}
			out[t * HCW + ch] += silu(a);
		}
	}
	out
}

/// The n-gram embeddings `[s][W]` of `ids` from the table file.
pub fn ngram_embed(h: &ngram::Hash, table: &ngram::Table, ids: &[u32]) -> Result<Vec<f32>, String> {
	let eos = npu::flash::EOS;
	let mut c = ngram::Ctx::new(eos);
	let rows: Vec<[u64; ngram::HEADS]> = ids.iter().map(|&t| h.ids(c.push(t, eos))).collect();
	let b = table.gather(&rows, 16)?;
	Ok(b.iter()
		.map(|&v| f32::from_bits((v as u32) << 16))
		.collect())
}

/// One layer over the streams `x` `[s][HCW]` (in place); `e`: the n-gram
/// embeddings (layer 1 only).
pub fn forward(
	ck: &Checkpoint,
	layer: &Layer,
	x: &mut [f32],
	e: Option<&[f32]>,
	s: usize,
) -> Result<Taps, String> {
	if let (Some(p), Some(e)) = (&layer.ple, e) {
		let add = ple(p, x, e, s);
		x.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
	}
	let h = layer.hc_a.apply(x, s);
	let (mix, wo) = match &layer.mixer {
		Mixer::Delta(d) => (delta_mixer(d, &h.mixed, s), &d.wo),
		Mixer::Attn(a) => (attn_mixer(a, &h.mixed, s), &a.wo),
	};
	let y = linear(&mix, s, HQ * DA, wo, W);
	inject(x, &y, &h.inj, s);
	let hm = layer.hc_m.apply(x, s);
	let (y, sdown, routes, edown) = moe(ck, &layer.moe, &hm.mixed, s)?;
	inject(x, &y, &hm.inj, s);
	Ok(Taps {
		hc_a: h.xn,
		hc_m: hm.xn,
		proj: h.mixed,
		out: mix,
		moe: hm.mixed,
		sdown,
		edown,
		routes,
	})
}

/// The token embeddings of `ids` repeated into the `HC` streams, `[s][HCW]`.
pub fn embed(ck: &Checkpoint, ids: &[u32]) -> Result<Vec<f32>, String> {
	let mut x = Vec::with_capacity(ids.len() * HCW);
	for &id in ids {
		let r = ck.rows(
			&format!("{P}embed_tokens.weight"),
			id as usize,
			id as usize + 1,
		)?;
		for _ in 0..HC {
			x.extend_from_slice(&r);
		}
	}
	Ok(x)
}

/// The final mixer's output `[s][W]` (the lm_head's input; no final norm).
pub fn final_mix(ck: &Checkpoint, x: &[f32], s: usize) -> Result<Vec<f32>, String> {
	Ok(Hc::load(ck, &format!("{P}hyper_connection_mixer."), false)?
		.apply(x, s)
		.mixed)
}
