//! The model in f32 from the BF16 checkpoint, one layer at a time: the
//! reference the graph sets are calibrated against (activation ranges, GPTQ
//! Hessians) and checked with. A chunk starts from zero state at position 0.

use crate::la::linear;
use crate::st::Checkpoint;
use quant::parallel::par_for;

pub const W: usize = 5120;
pub const FF: usize = 17408;
pub const NL: usize = 64;
/// Delta-net layers: value heads, key heads, head width.
pub const H: usize = 48;
pub const HK: usize = 16;
pub const D: usize = 128;
/// Attention layers: query heads, key / value heads, head width, rotary dims.
pub const HQ: usize = 24;
pub const HKV: usize = 4;
pub const DA: usize = 256;
pub const RD: usize = 64;
pub const EPS: f32 = 1e-6;

pub fn is_attn(l: usize) -> bool {
	l % 4 == 3
}

pub struct Mlp {
	pub ln2: Vec<f32>,
	pub wg: Vec<f32>,
	pub wu: Vec<f32>,
	pub wd: Vec<f32>,
}

pub struct Delta {
	pub ln1: Vec<f32>,
	pub wqkv: Vec<f32>,
	pub wz: Vec<f32>,
	pub wa: Vec<f32>,
	pub wb: Vec<f32>,
	/// `[10240][4]`
	pub conv: Vec<f32>,
	pub dtb: Vec<f32>,
	/// `-exp(A_log)`
	pub acoef: Vec<f32>,
	pub gnw: Vec<f32>,
	pub wo: Vec<f32>,
}

pub struct Attn {
	pub ln1: Vec<f32>,
	/// `[24][2][256]` rows: per head the query rows, then the gate rows
	pub wq: Vec<f32>,
	pub wk: Vec<f32>,
	pub wv: Vec<f32>,
	pub qnw: Vec<f32>,
	pub knw: Vec<f32>,
	pub wo: Vec<f32>,
}

pub enum Mixer {
	Delta(Delta),
	Attn(Attn),
}

pub struct Layer {
	pub mixer: Mixer,
	pub mlp: Mlp,
}

/// The inputs of a layer's four quantized MatMul groups, `[s][k]` each:
/// the projection (qkv / z / a / b, or q / k / v), the output projection,
/// gate / up and down.
pub struct Taps {
	pub proj: Vec<f32>,
	pub out: Vec<f32>,
	pub gu: Vec<f32>,
	pub down: Vec<f32>,
}

/// The weight of one of the model's RMSNorms: the checkpoint stores it less
/// one (`y = x / rms(x) * (1 + weight)`). The delta-net's gated norm
/// (`linear_attn.norm`) stores its weight as it is used.
pub fn norm_w(ck: &Checkpoint, name: &str, n: usize) -> Result<Vec<f32>, String> {
	Ok(ck.vec(name, n)?.iter().map(|w| 1.0 + w).collect())
}

fn mlp_of(ck: &Checkpoint, p: &str) -> Result<Mlp, String> {
	Ok(Mlp {
		ln2: norm_w(ck, &format!("{p}.post_attention_layernorm.weight"), W)?,
		wg: ck.mat(&format!("{p}.mlp.gate_proj.weight"), FF, W)?,
		wu: ck.mat(&format!("{p}.mlp.up_proj.weight"), FF, W)?,
		wd: ck.mat(&format!("{p}.mlp.down_proj.weight"), W, FF)?,
	})
}

fn attn_of(ck: &Checkpoint, p: &str) -> Result<Attn, String> {
	Ok(Attn {
		ln1: norm_w(ck, &format!("{p}.input_layernorm.weight"), W)?,
		wq: ck.mat(&format!("{p}.self_attn.q_proj.weight"), 2 * HQ * DA, W)?,
		wk: ck.mat(&format!("{p}.self_attn.k_proj.weight"), HKV * DA, W)?,
		wv: ck.mat(&format!("{p}.self_attn.v_proj.weight"), HKV * DA, W)?,
		qnw: norm_w(ck, &format!("{p}.self_attn.q_norm.weight"), DA)?,
		knw: norm_w(ck, &format!("{p}.self_attn.k_norm.weight"), DA)?,
		wo: ck.mat(&format!("{p}.self_attn.o_proj.weight"), W, HQ * DA)?,
	})
}

/// Layer `l` of the language model.
pub fn load_layer(ck: &Checkpoint, l: usize) -> Result<Layer, String> {
	let p = format!("model.language_model.layers.{l}");
	let mixer = if is_attn(l) {
		Mixer::Attn(attn_of(ck, &p)?)
	} else {
		let q = format!("{p}.linear_attn");
		Mixer::Delta(Delta {
			ln1: norm_w(ck, &format!("{p}.input_layernorm.weight"), W)?,
			wqkv: ck.mat(&format!("{q}.in_proj_qkv.weight"), 2 * HK * D + H * D, W)?,
			wz: ck.mat(&format!("{q}.in_proj_z.weight"), H * D, W)?,
			wa: ck.mat(&format!("{q}.in_proj_a.weight"), H, W)?,
			wb: ck.mat(&format!("{q}.in_proj_b.weight"), H, W)?,
			conv: ck.mat(&format!("{q}.conv1d.weight"), 2 * HK * D + H * D, 4)?,
			dtb: ck.vec(&format!("{q}.dt_bias"), H)?,
			acoef: ck
				.vec(&format!("{q}.A_log"), H)?
				.iter()
				.map(|a| -a.exp())
				.collect(),
			gnw: ck.vec(&format!("{q}.norm.weight"), D)?,
			wo: ck.mat(&format!("{q}.out_proj.weight"), W, H * D)?,
		})
	};
	Ok(Layer {
		mixer,
		mlp: mlp_of(ck, &p)?,
	})
}

/// The MTP block (`mtp.layers.0`, a full-attention layer).
pub fn load_mtp_layer(ck: &Checkpoint) -> Result<Layer, String> {
	Ok(Layer {
		mixer: Mixer::Attn(attn_of(ck, "mtp.layers.0")?),
		mlp: mlp_of(ck, "mtp.layers.0")?,
	})
}

/// `x / sqrt(mean(x^2) + eps) * w` per row of `[s][n]`; `w` of `n` values or
/// of a divisor of `n` (repeated).
pub fn rmsnorm(x: &[f32], n: usize, w: &[f32]) -> Vec<f32> {
	let mut y = vec![0f32; x.len()];
	for (xr, yr) in x.chunks_exact(n).zip(y.chunks_exact_mut(n)) {
		let ms = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
		let r = (1.0 / (ms + EPS as f64).sqrt()) as f32;
		for (i, (yv, xv)) in yr.iter_mut().zip(xr).enumerate() {
			*yv = xv * r * w[i % w.len()];
		}
	}
	y
}

pub fn silu(x: f32) -> f32 {
	x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
	1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
	if x > 20.0 {
		x
	} else {
		x.exp().ln_1p()
	}
}

/// The rotary angles' frequencies (`rope_theta` = 1e7, 32 pairs).
pub fn inv_freq() -> [f64; RD / 2] {
	std::array::from_fn(|i| 1e7f64.powf(-(i as f64) / (RD / 2) as f64))
}

/// Rotate the first [`RD`] channels of every `[DA]` row of `t` (`[s][heads][DA]`)
/// by the angles of position `p0 + row`.
fn rope(t: &mut [f32], s: usize, heads: usize, p0: usize) {
	let f = inv_freq();
	for p in 0..s {
		let cs: Vec<(f32, f32)> = f
			.iter()
			.map(|&w| ((p0 + p) as f64 * w).sin_cos())
			.map(|(s, c)| (c as f32, s as f32))
			.collect();
		for h in 0..heads {
			let r = &mut t[(p * heads + h) * DA..][..RD];
			for i in 0..RD / 2 {
				let (a, b) = (r[i], r[i + RD / 2]);
				r[i] = a * cs[i].0 - b * cs[i].1;
				r[i + RD / 2] = b * cs[i].0 + a * cs[i].1;
			}
		}
	}
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
// SAFETY: every use hands disjoint ranges to the tasks.
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// The delta-net mixer over `s` positions from zero state: (the gated-norm
/// output `[s][6144]` = the input of out_proj, the projection's input).
fn delta_mixer(d: &Delta, x: &[f32], s: usize) -> (Vec<f32>, Vec<f32>) {
	let xn = rmsnorm(x, W, &d.ln1);
	let nq = 2 * HK * D + H * D;
	let qkv = linear(&xn, s, W, &d.wqkv, nq);
	let z = linear(&xn, s, W, &d.wz, H * D);
	let pa = linear(&xn, s, W, &d.wa, H);
	let pb = linear(&xn, s, W, &d.wb, H);
	let core = DeltaCore {
		conv: &d.conv,
		dtb: &d.dtb,
		acoef: &d.acoef,
		gnw: &d.gnw,
		sigmoid_gate: false,
	};
	(core.run(&qkv, &z, &pa, &pb, s), xn)
}

/// The delta-net's per-layer constants, independent of the model width
/// (shared with Qwen3.8-Flash-Next, whose output gate is a sigmoid).
pub struct DeltaCore<'a> {
	/// `[10240][4]`
	pub conv: &'a [f32],
	pub dtb: &'a [f32],
	/// `-exp(A_log)`
	pub acoef: &'a [f32],
	pub gnw: &'a [f32],
	pub sigmoid_gate: bool,
}

impl DeltaCore<'_> {
	/// From the projections `qkv` `[s][10240]`, `z` `[s][6144]`, `pa` / `pb`
	/// `[s][48]` over `s` positions from zero state to the gated-norm output
	/// `[s][6144]` (the input of out_proj).
	pub fn run(&self, qkv: &[f32], z: &[f32], pa: &[f32], pb: &[f32], s: usize) -> Vec<f32> {
		let d = self;
		let nq = 2 * HK * D + H * D;
		// causal depthwise conv over time (kernel 4, three zero positions before), SiLU
		let mut cv = vec![0f32; s * nq];
		for t in 0..s {
			for j in 0..4 {
				if t + j < 3 {
					continue;
				}
				let src = &qkv[(t + j - 3) * nq..][..nq];
				for (c, o) in cv[t * nq..][..nq].iter_mut().enumerate() {
					*o += d.conv[c * 4 + j] * src[c];
				}
			}
		}
		cv.iter_mut().for_each(|v| *v = silu(*v));
		// q / k: l2-normalized per key head; q also / sqrt(D)
		let l2 = |t: &mut [f32], sc: f32| {
			for r in t.chunks_exact_mut(D) {
				let ss: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum();
				let f = (1.0 / (ss + EPS as f64).sqrt()) as f32 * sc;
				r.iter_mut().for_each(|v| *v *= f);
			}
		};
		let mut o = vec![0f32; s * H * D];
		let mut cvm = cv;
		for t in 0..s {
			let row = &mut cvm[t * nq..][..nq];
			l2(&mut row[..HK * D], 1.0 / (D as f32).sqrt());
			l2(&mut row[HK * D..2 * HK * D], 1.0);
		}
		let cv = cvm;
		let op = Ptr(o.as_mut_ptr());
		par_for(H, |h| {
			let op = op;
			let hk = h / (H / HK);
			// S[k][v]
			let mut st = vec![0f32; D * D];
			let mut kv = [0f32; D];
			for t in 0..s {
				let row = &cv[t * nq..][..nq];
				let q = &row[hk * D..][..D];
				let k = &row[HK * D + hk * D..][..D];
				let v = &row[2 * HK * D + h * D..][..D];
				let g = d.acoef[h] * softplus(pa[t * H + h] + d.dtb[h]);
				let beta = sigmoid(pb[t * H + h]);
				let eg = g.exp();
				st.iter_mut().for_each(|x| *x *= eg);
				kv.fill(0.0);
				for (ki, srow) in k.iter().zip(st.chunks_exact(D)) {
					for (a, b) in kv.iter_mut().zip(srow) {
						*a += ki * b;
					}
				}
				for (a, b) in kv.iter_mut().zip(v) {
					*a = (b - *a) * beta;
				}
				for (ki, srow) in k.iter().zip(st.chunks_exact_mut(D)) {
					for (a, b) in srow.iter_mut().zip(&kv) {
						*a += ki * b;
					}
				}
				// SAFETY: head h owns o[t][h][..]
				let ot = unsafe { std::slice::from_raw_parts_mut(op.0.add((t * H + h) * D), D) };
				ot.fill(0.0);
				for (qi, srow) in q.iter().zip(st.chunks_exact(D)) {
					for (a, b) in ot.iter_mut().zip(srow) {
						*a += qi * b;
					}
				}
			}
		});
		// gated RMSNorm per head: o / rms(o) * gw * act(z)
		let mut a = rmsnorm(&o, D, d.gnw);
		for (av, zv) in a.iter_mut().zip(z) {
			*av *= if d.sigmoid_gate {
				sigmoid(*zv)
			} else {
				silu(*zv)
			};
		}
		a
	}
}

/// The attention mixer over `s` positions from an empty cache: (the gated
/// attention output `[s][6144]`, the projection's input).
fn attn_mixer(a: &Attn, x: &[f32], s: usize) -> (Vec<f32>, Vec<f32>) {
	let xn = rmsnorm(x, W, &a.ln1);
	let qg = linear(&xn, s, W, &a.wq, 2 * HQ * DA);
	let k = linear(&xn, s, W, &a.wk, HKV * DA);
	let v = linear(&xn, s, W, &a.wv, HKV * DA);
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
	rope(&mut q, s, HQ, 0);
	rope(&mut k, s, HKV, 0);
	let scale = 1.0 / (DA as f32).sqrt();
	let mut o = vec![0f32; s * HQ * DA];
	let op = Ptr(o.as_mut_ptr());
	par_for(s * HQ, |i| {
		let op = op;
		let (t, h) = (i / HQ, i % HQ);
		let kh = h / (HQ / HKV);
		let qr = &q[(t * HQ + h) * DA..][..DA];
		let sc: Vec<f32> = (0..=t)
			.map(|u| {
				qr.iter()
					.zip(&k[(u * HKV + kh) * DA..][..DA])
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
		for (u, pu) in p.iter().enumerate() {
			let w = pu / sum;
			for (a, b) in ot.iter_mut().zip(&v[(u * HKV + kh) * DA..][..DA]) {
				*a += w * b;
			}
		}
	});
	for (ov, g) in o.iter_mut().zip(&gate) {
		*ov *= sigmoid(*g);
	}
	(o, xn)
}

/// One layer over the chunk `x` `[s][W]`: (the residual after it, the taps).
pub fn forward(layer: &Layer, x: &[f32], s: usize) -> (Vec<f32>, Taps) {
	let ((mix, proj), wo) = match &layer.mixer {
		Mixer::Delta(d) => (delta_mixer(d, x, s), &d.wo),
		Mixer::Attn(a) => (attn_mixer(a, x, s), &a.wo),
	};
	let mut x1 = linear(&mix, s, H * D, wo, W);
	x1.iter_mut().zip(x).for_each(|(a, b)| *a += b);
	let gu = rmsnorm(&x1, W, &layer.mlp.ln2);
	let g = linear(&gu, s, W, &layer.mlp.wg, FF);
	let mut down = linear(&gu, s, W, &layer.mlp.wu, FF);
	down.iter_mut().zip(&g).for_each(|(u, g)| *u *= silu(*g));
	let mut y = linear(&down, s, FF, &layer.mlp.wd, W);
	y.iter_mut().zip(&x1).for_each(|(a, b)| *a += b);
	(
		y,
		Taps {
			proj,
			out: mix,
			gu,
			down,
		},
	)
}

/// The embeddings of `ids`, `[n][W]`.
pub fn embed(ck: &Checkpoint, ids: &[u32]) -> Result<Vec<f32>, String> {
	let mut x = Vec::with_capacity(ids.len() * W);
	for &id in ids {
		x.extend(ck.rows(
			"model.language_model.embed_tokens.weight",
			id as usize,
			id as usize + 1,
		)?);
	}
	Ok(x)
}

/// The output-normed hidden states of `x` (`[s][W]` after layer 63).
pub fn final_norm(ck: &Checkpoint, x: &[f32]) -> Result<Vec<f32>, String> {
	Ok(rmsnorm(
		x,
		W,
		&norm_w(ck, "model.language_model.norm.weight", W)?,
	))
}
