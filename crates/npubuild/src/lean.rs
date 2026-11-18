//! The lean prefill graphs: one delta-net layer or one full-attention layer
//! over `S` positions, f16 activations, int4 weights as graph inputs.
//!
//! Delta-net layers take every per-layer constant as an input (weights, row
//! scales, norm / conv / gate vectors), so one compiled blob serves every
//! layer of a kind. Attention layers keep their small constants (norm
//! weights, row scales, ranges) in the graph: one blob per layer.
//!
//! How each construct maps onto the NPU (and why it has this form) is in
//! `crates/npu/NPU.md`; the per-layer values the inputs carry are written by
//! [`crate::weights`].

use crate::g::{Graph, Ty, V};
use crate::gptq::ROTB;
use crate::refm::{D, DA, FF, H, HK, HKV, HQ, RD, W};

/// Positions per sub-chunk of the delta rule's chunked form.
const C: usize = 64;
/// Head groups the recurrence is built in.
const RGROUPS: usize = 6;
/// The scale the queries (and so the recurrence's output) run at.
pub const QS: f64 = 16.0;
/// Block size the SHAVE kernel `trinv` inverts; the doubling goes on from it.
const SHINV: usize = 16;
/// Row blocks of the projection (q k | v | z rows) and of gate / up.
pub const PROJSPLIT: [usize; 3] = [5120, 5120, 6144];
pub const GUSPLIT: usize = 4;
/// Query heads per key / value head.
const G: usize = HQ / HKV;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cut {
	/// The whole layer.
	Whole,
	/// Up to the recurrence: `qn`, `kn`, `v`, `gb`, `z`, `conv_out`.
	Pre,
	/// After it: inputs `x`, `o`, `z`.
	Post,
}

fn hadamard(n: usize) -> Vec<f64> {
	let s = 1.0 / (n as f64).sqrt();
	(0..n * n)
		.map(|i| {
			if ((i / n) & (i % n)).count_ones() & 1 == 1 {
				-s
			} else {
				s
			}
		})
		.collect()
}

/// `x R` without the sign multiply (the signs are folded upstream).
fn rotate(g: &mut Graph, x: &V) -> V {
	let t = g.reshape(x, &[-1, ROTB as i64]);
	let hb = g.cf(x.ty, &[ROTB, ROTB], &hadamard(ROTB));
	let t = g.matmul(&t, &hb, false, false);
	let shape: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
	g.reshape(&t, &shape)
}

/// `x / sqrt(mean(x^2) + eps)` over the last axis, in the form the
/// compiler's FuseRMSNorm turns into one SHAVE op.
fn rms_fused(g: &mut Graph, x: &V, eps: f64) -> V {
	let two = g.sf(x.ty, 2.0);
	let sq = g.power(x, &two);
	let ms = g.reduce_mean(&sq, -1);
	let e = g.sf(x.ty, eps);
	let t = g.add(&ms, &e);
	let t = g.sqrt(&t);
	let one = g.sf(x.ty, 1.0);
	let r = g.div(&one, &t);
	g.mul(x, &r)
}

/// Static int8 quantization of an input pre-scaled to +-127.
fn fq_static(g: &mut Graph, x: &V) -> V {
	g.fake_quantize(x, -128.0, 127.0, 256)
}

struct DeltaB {
	g: Graph,
	s: usize,
	/// plain ops in place of the two SHAVE kernels (see [`delta_emu`])
	emu: bool,
	/// index of the next weight / scale input (`w{n}`, `s{n}`)
	nw: usize,
}

impl DeltaB {
	/// A per-layer f16 input.
	fn pconst(&mut self, name: &str, shape: &[usize]) -> V {
		self.nw += 1;
		self.g.parameter(name, Ty::F16, shape)
	}

	/// A weight input `w{n}` [n, k] and its row-scale input `s{n}`:
	/// (w * 2^-8, s). The fixed 2^-8 keeps the MatMul's output in f16 range.
	fn weight(&mut self, ty: Ty, n: usize, k: usize) -> (V, V) {
		let i = self.nw;
		let prm = self.g.parameter(&format!("w{i}"), ty, &[n, k]);
		self.nw += 1;
		let c = self.g.convert(&prm, Ty::F16);
		let sc = self.g.s16(2f64.powi(-8));
		let w = self.g.mul(&c, &sc);
		let s = self.pconst(&format!("s{i}"), &[1, 1, n]);
		(w, s)
	}

	/// `fq(x) @ w^T * s` for each (weight, scale) of `ws`.
	fn fcs(&mut self, xq: &V, ws: &[(V, V)]) -> Vec<V> {
		ws.iter()
			.map(|(w, s)| {
				let y = self.g.matmul(xq, w, false, true);
				self.g.mul(&y, s)
			})
			.collect()
	}

	/// A rotated, statically quantized linear layer on a sign-folded input.
	fn linear(&mut self, x: &V, n: usize, k: usize, parts: usize) -> Vec<V> {
		let xr = rotate(&mut self.g, x);
		let ws: Vec<(V, V)> = (0..parts)
			.map(|_| self.weight(Ty::I4, n / parts, k))
			.collect();
		let xq = fq_static(&mut self.g, &xr);
		self.fcs(&xq, &ws)
	}

	/// l2-normalize the `D`-rows of `t` [1, S, 2048]: [1, S, 16, D]. A
	/// per-row max-abs pre-scale keeps the squares out of f16's subnormals.
	fn l2n(&mut self, t: &V) -> V {
		let g = &mut self.g;
		let t = g.reshape(t, &[1, self.s as i64, HK as i64, D as i64]);
		let a = g.abs(&t);
		let m = g.reduce_max(&a, -1);
		let lo = g.s16(1e-4);
		let m = g.maximum(&m, &lo);
		let one = g.s16(1.0);
		let r = g.div(&one, &m);
		let ts = g.mul(&t, &r);
		let two = g.s16(2.0);
		let sq = g.power(&ts, &two);
		let ones = g.c16(&[D, 1], &vec![1.0; D]);
		let ss = g.matmul(&sq, &ones, false, false);
		let fl = g.s16(6.2e-5);
		let ss = g.maximum(&ss, &fl);
		let one = g.s16(1.0);
		let rt = g.sqrt(&ss);
		let r = g.div(&one, &rt);
		g.mul(&ts, &r)
	}

	/// [1, S, 16, D] -> [1, 48, S, D]: value head h reads key head h / 3.
	fn heads48(&mut self, t: &V) -> V {
		let g = &mut self.g;
		let s = self.s;
		let t = g.reshape(t, &[1, s as i64, HK as i64, 1, D as i64]);
		let t = g.broadcast_to(&t, &[1, s, HK, H / HK, D]);
		let t = g.reshape(&t, &[1, s as i64, H as i64, D as i64]);
		g.transpose(&t, &[0, 2, 1, 3])
	}

	/// The chunked WY recurrence over `hn` heads: (o [1, hn, S, D], state).
	#[allow(clippy::too_many_arguments)]
	fn rec(&mut self, qh: &V, kh: &V, vh: &V, gate: &V, beta: &V, spast: &V, hn: usize) -> (V, V) {
		let g = &mut self.g;
		let p = self.s / C;
		let x5 = [1, hn as i64, p as i64, C as i64, D as i64];
		let (q5, k5, v5) = (g.reshape(qh, &x5), g.reshape(kh, &x5), g.reshape(vh, &x5));
		let g5 = g.reshape(gate, &[1, hn as i64, p as i64, C as i64]);
		let b5 = g.reshape(beta, &[1, hn as i64, p as i64, C as i64]);
		// Gate sums as one masked MatMul (fp32 accumulation, one f16
		// rounding): differences G_i - G_j, running sums, sums of the rest
		// and the total.
		let g4 = g.unsqueeze(&g5, 3);
		let cols = C * C + 2 * C + 1;
		let mut m = vec![0f64; C * cols];
		for sr in 0..C {
			for i in 0..C {
				for j in 0..C {
					m[sr * cols + i * C + j] = (j < sr && sr <= i) as u8 as f64;
				}
			}
			for j in 0..C {
				m[sr * cols + C * C + j] = (sr <= j) as u8 as f64;
				m[sr * cols + C * C + C + j] = (sr > j) as u8 as f64;
			}
			m[sr * cols + C * C + 2 * C] = 1.0;
		}
		let mc = g.c16(&[C, cols], &m);
		let allm = g.matmul(&g4, &mc, false, false);
		let sp = g.variadic_split(&allm, 4, &[C * C, C, C, 1]);
		let cum = g.squeeze(&sp[1], 3);
		let diff = g.reshape(&sp[0], &[1, hn as i64, p as i64, C as i64, C as i64]);
		let e = g.exp(&diff);
		let rs = g.squeeze(&sp[2], 3);
		let rsum = g.unsqueeze(&rs, -1);
		let gtot = sp[3].clone();
		let gc = g.unsqueeze(&cum, -1);
		let bc = g.unsqueeze(&b5, -1);
		let eg = g.exp(&gc);
		// beta applied to k once (the rows of A and the right-hand side share it)
		let kb = g.mul(&k5, &bc);
		let kk = g.matmul(&kb, &k5, false, true);
		let a = g.mul(&kk, &e);
		let r1 = g.mul(&kb, &eg);
		let r2 = g.mul(&v5, &bc);
		let rhs = g.concat(&[r1, r2], 4);
		// T = (I + tril(A, -1))^-1: the SHINV diagonal blocks by the SHAVE
		// kernel `trinv` (riding on a LogSoftmax over the flattened matrix),
		// then block doubling T_2b = T_b - T_b L_b T_b
		let (mut t, mut b) = if self.emu {
			let eye: Vec<f64> = (0..C * C).map(|x| (x / C == x % C) as u8 as f64).collect();
			let eye = g.c16(&[C, C], &eye);
			let low: Vec<f64> = (0..C * C)
				.map(|x| ((x / C) / 2 == (x % C) / 2 && x / C > x % C) as u8 as f64)
				.collect();
			let low = g.c16(&[C, C], &low);
			let l = g.mul(&a, &low);
			(g.sub(&eye, &l), 2)
		} else {
			let af = g.reshape(&a, &[1, (hn * p) as i64, 1, (C * C) as i64]);
			let ls = g.log_softmax(&af, 3);
			(
				g.reshape(&ls, &[1, hn as i64, p as i64, C as i64, C as i64]),
				SHINV,
			)
		};
		while b < C {
			let msk: Vec<f64> = (0..C * C)
				.map(|x| {
					((x / C) / (2 * b) == (x % C) / (2 * b) && (x / C) / b > (x % C) / b) as u8
						as f64
				})
				.collect();
			let mk = g.c16(&[C, C], &msk);
			let l = g.mul(&a, &mk);
			let tl = g.matmul(&t, &l, false, false);
			let tlt = g.matmul(&tl, &t, false, false);
			t = g.sub(&t, &tlt);
			b *= 2;
		}
		let x = g.matmul(&t, &rhs, false, false);
		let wu = g.variadic_split(&x, 4, &[D, D]);
		let (wm, um) = (wu[0].clone(), wu[1].clone());
		let qk = g.matmul(&q5, &k5, false, true);
		let qke = g.mul(&qk, &e);
		let tri: Vec<f64> = (0..C * C).map(|x| (x % C <= x / C) as u8 as f64).collect();
		let tri = g.c16(&[C, C], &tri);
		let qkm = g.mul(&qke, &tri);
		let qe = g.mul(&q5, &eg);
		let ers = g.exp(&rsum);
		let kd = g.mul(&k5, &ers);
		let egc = g.exp(&gtot);
		let part = |g: &mut Graph, t: &V, i: usize| {
			let sl = g.slice(t, i, i + 1, 1, 2);
			g.squeeze(&sl, 2)
		};
		let mut st = spast.clone();
		let (mut ss, mut ds) = (Vec::new(), Vec::new());
		for i in 0..p {
			ss.push(st.clone());
			let wp = part(g, &wm, i);
			let ws = g.matmul(&wp, &st, false, false);
			let up = part(g, &um, i);
			let dm = g.sub(&up, &ws);
			let ep = part(g, &egc, i);
			let dec = g.mul(&st, &ep);
			let kp = part(g, &kd, i);
			let upd = g.matmul(&kp, &dm, true, false);
			st = g.add(&dec, &upd);
			ds.push(dm);
		}
		let os: Vec<V> = (0..p)
			.map(|i| {
				let qp = part(g, &qe, i);
				let a = g.matmul(&qp, &ss[i], false, false);
				let mp = part(g, &qkm, i);
				let b = g.matmul(&mp, &ds[i], false, false);
				g.add(&a, &b)
			})
			.collect();
		let o = if p > 1 {
			g.concat(&os, 2)
		} else {
			os[0].clone()
		};
		(o, st)
	}
}

/// A delta-net layer over `s` positions (a multiple of 64).
///
/// Inputs: `x` [1, s, 5120], `conv` [1, 3, 10240] (the last three positions'
/// qkv), `ssm` [1, 48, 128, 128] (the state S[h][k][v] times `vmul`), `valid`
/// [1, s, 1] (0 at right padding), `keep` [3] i32 (rows of concat(conv, qkv)
/// that are the next window), then the per-layer inputs. Outputs: `y`,
/// `conv_out`, `ssm_out`.
pub fn delta(s: usize, cut: Cut, vmul: f64) -> Graph {
	delta_build(s, cut, vmul, false)
}

/// [`delta`] with the two SHAVE kernels (the block inverse on a LogSoftmax,
/// the gated norm on a Minimum) spelled out in plain ops: the same function
/// on any device, for checks on the CPU. Not for the NPU (f16 overflows).
pub fn delta_emu(s: usize, cut: Cut, vmul: f64) -> Graph {
	delta_build(s, cut, vmul, true)
}

fn delta_build(s: usize, cut: Cut, vmul: f64, emu: bool) -> Graph {
	assert!(s.is_multiple_of(C));
	let mut b = DeltaB {
		g: Graph::new(),
		s,
		emu,
		nw: 0,
	};
	let x = b.g.parameter("x", Ty::F16, &[1, s, W]);
	if cut == Cut::Post {
		let o_in = b.g.parameter("o", Ty::F16, &[1, s, H * D]);
		let z_in = b.g.parameter("z", Ty::F16, &[1, s, H * D]);
		// the inputs of the layer's first half, unused here: the argument
		// list (and the names) of the whole layer's graph
		b.pconst("ln1", &[W]);
		for n in PROJSPLIT {
			b.weight(Ty::I4, n, W);
		}
		b.weight(Ty::I8, 2 * H, W);
		b.pconst("cw", &[2 * HK * D + H * D, 1, 1, 4]);
		b.pconst("dtb", &[H]);
		b.pconst("acoef", &[H]);
		let o = b.g.reshape(&o_in, &[1, s as i64, H as i64, D as i64]);
		let y = delta_tail(&mut b, &x, &o, &z_in);
		b.g.result(&y, "y");
		return b.g;
	}
	let cpast = b.g.parameter("conv", Ty::F16, &[1, 3, 2 * HK * D + H * D]);
	let spast = b.g.parameter("ssm", Ty::F16, &[1, H, D, D]);
	let valid = b.g.parameter("valid", Ty::F16, &[1, s, 1]);
	let keep = b.g.parameter("keep", Ty::I32, &[3]);
	// projection: one rotation and one static int8 quantization for the
	// int4 q k v z rows and the int8 gate rows
	let ln1 = b.pconst("ln1", &[W]);
	let xn = rms_fused(&mut b.g, &x, 1e-6);
	let xn1 = b.g.mul(&xn, &ln1);
	let xr = rotate(&mut b.g, &xn1);
	let xq = fq_static(&mut b.g, &xr);
	let mut ws: Vec<(V, V)> = PROJSPLIT.iter().map(|&n| b.weight(Ty::I4, n, W)).collect();
	let main = b.fcs(&xq, &ws);
	ws = vec![b.weight(Ty::I8, 2 * H, W)];
	let ab = b.fcs(&xq, &ws).remove(0);
	let qkv = b.g.concat(&main[..2], 2);
	let z = main[2].clone();
	let pab = b.g.variadic_split(&ab, 2, &[H, H]);
	// conv1d, kernel 4, causal, depthwise (a GroupConvolution runs on the
	// DPU also with its filter as an input)
	let ext = b.g.concat(&[cpast.clone(), qkv.clone()], 1);
	let xt = b.g.transpose(&ext, &[0, 2, 1]);
	let cw = b.pconst("cw", &[2 * HK * D + H * D, 1, 1, 4]);
	let cv = b.g.group_conv1d(&xt, &cw);
	let cv = b.g.transpose(&cv, &[0, 2, 1]);
	let cv = b.g.swish1(&cv);
	let qkvs = b.g.variadic_split(&cv, 2, &[HK * D, HK * D, H * D]);
	let (q, k, v) = (qkvs[0].clone(), qkvs[1].clone(), qkvs[2].clone());
	// gates: g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)
	let dtb = b.pconst("dtb", &[H]);
	let acoef = b.pconst("acoef", &[H]);
	if cut == Cut::Pre {
		// (the whole graph's gate inputs come first; these are the ones read)
		let dtb = b.pconst("dtb", &[H]);
		let acoef = b.pconst("acoef", &[H]);
		let g = &mut b.g;
		let t = g.add(&pab[0], &dtb);
		let t = g.softplus(&t);
		let t = g.mul(&t, &acoef);
		let g_tm = g.mul(&t, &valid);
		let t = g.sigmoid(&pab[1]);
		let b_tm = g.mul(&t, &valid);
		let flat = [1, s as i64, (HK * D) as i64];
		let qn = b.l2n(&q);
		let qn = b.g.reshape(&qn, &flat);
		let kn = b.l2n(&k);
		let kn = b.g.reshape(&kn, &flat);
		let gb = b.g.concat(&[g_tm, b_tm], 2);
		let ext = b.g.concat(&[cpast, qkv], 1);
		let cnext = b.g.gather(&ext, &keep, 1);
		for (t, n) in [
			(&qn, "qn"),
			(&kn, "kn"),
			(&v, "v"),
			(&gb, "gb"),
			(&z, "z"),
			(&cnext, "conv_out"),
		] {
			b.g.result(t, n);
		}
		let _ = spast;
		return b.g;
	}
	let qn = b.l2n(&q);
	let qsc = b.g.s16(QS / (D as f64).sqrt());
	let qn = b.g.mul(&qn, &qsc);
	let qh = b.heads48(&qn);
	let kn = b.l2n(&k);
	let kh = b.heads48(&kn);
	let vr = b.g.reshape(&v, &[1, s as i64, H as i64, D as i64]);
	let one = b.g.s16(1.0);
	let vr = b.g.mul(&vr, &one);
	let vh = b.g.transpose(&vr, &[0, 2, 1, 3]);
	let vt = b.g.transpose(&valid, &[0, 2, 1]);
	let t = b.g.add(&pab[0], &dtb);
	let t = b.g.softplus(&t);
	let t = b.g.mul(&t, &acoef);
	let t = b.g.transpose(&t, &[0, 2, 1]);
	let gate = b.g.mul(&t, &vt);
	let t = b.g.sigmoid(&pab[1]);
	let t = b.g.transpose(&t, &[0, 2, 1]);
	let beta = b.g.mul(&t, &vt);
	// the state arrives at the export's scale (x vmul); the recurrence runs
	// at v x 1
	let sin = b.g.s16(1.0 / vmul);
	let s_in = b.g.mul(&spast, &sin);
	// the recurrence in RGROUPS independent head groups
	let hg = H / RGROUPS;
	let mut res = Vec::new();
	for i in 0..RGROUPS {
		let sl = |g: &mut Graph, t: &V| g.slice(t, i * hg, (i + 1) * hg, 1, 1);
		let (a, bb, c, d, e, f) = (
			sl(&mut b.g, &qh),
			sl(&mut b.g, &kh),
			sl(&mut b.g, &vh),
			sl(&mut b.g, &gate),
			sl(&mut b.g, &beta),
			sl(&mut b.g, &s_in),
		);
		res.push(b.rec(&a, &bb, &c, &d, &e, &f, hg));
	}
	let os: Vec<V> = res.iter().map(|r| r.0.clone()).collect();
	let sts: Vec<V> = res.iter().map(|r| r.1.clone()).collect();
	let o = b.g.concat(&os, 1);
	let st = b.g.concat(&sts, 1);
	let o = b.g.transpose(&o, &[0, 2, 1, 3]);
	let y = delta_tail(&mut b, &x, &o, &z);
	let ext = b.g.concat(&[cpast, qkv], 1);
	let cnext = b.g.gather(&ext, &keep, 1);
	let so = b.g.s16(vmul);
	let st = b.g.mul(&st, &so);
	b.g.result(&y, "y");
	b.g.result(&cnext, "conv_out");
	b.g.result(&st, "ssm_out");
	b.g
}

/// From the recurrence's output `o` [1, S, 48, D] (at scale [`QS`]) and the
/// gate `z` to the layer's output: gated RMSNorm, out_proj, residual, MLP.
fn delta_tail(b: &mut DeltaB, x: &V, o: &V, z: &V) -> V {
	let s = b.s;
	// the gated RMSNorm (o / rms(o) * gw * swish(z), per head) as one SHAVE
	// kernel of ours (`gnorm`) riding on a Minimum: A = [1, S, H, 2, D] with
	// rows (h, 0) = o, (h, 1) = z; B = `gnb` with rows (h, 0) = gw and eps
	// (f32) in row (0, 1)
	let zz = b.g.reshape(z, &[1, s as i64, H as i64, D as i64]);
	let ou = b.g.unsqueeze(o, 3);
	let zu = b.g.unsqueeze(&zz, 3);
	let a = b.g.concat(&[ou, zu], 3);
	let gnb = b.pconst("gnb", &[1, 1, H, 2, D]);
	let y = if b.emu {
		let g = &mut b.g;
		let gw = g.slice(&gnb, 0, 1, 1, 3);
		let ou = g.slice(&a, 0, 1, 1, 3);
		let zu = g.slice(&a, 1, 2, 1, 3);
		let n = rms_fused(g, &ou, 1e-6 * QS * QS);
		let n = g.mul(&n, &gw);
		let sw = g.swish1(&zu);
		g.mul(&n, &sw)
	} else {
		let y = b.g.minimum(&a, &gnb);
		b.g.slice(&y, 0, 1, 1, 3)
	};
	let attn = b.g.reshape(&y, &[1, s as i64, (H * D) as i64]);
	let op = b.linear(&attn, W, H * D, 1).remove(0);
	let x1 = b.g.add(x, &op);
	let ln2 = b.pconst("ln2", &[W]);
	let xn = rms_fused(&mut b.g, &x1, 1e-6);
	let xn2 = b.g.mul(&xn, &ln2);
	let gu = b.linear(&xn2, 2 * FF, W, GUSPLIT);
	let gg = b.g.concat(&gu[..GUSPLIT / 2], 2);
	let uu = b.g.concat(&gu[GUSPLIT / 2..], 2);
	let sw = b.g.swish1(&gg);
	let h = b.g.mul(&sw, &uu);
	let dn = b.linear(&h, W, FF, 1).remove(0);
	b.g.add(&x1, &dn)
}

/// Which part of an attention layer a graph holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
	/// The whole layer with a window of `cap` past keys.
	All,
	/// x, cos, sin -> qg, gate, knew, vnew.
	Pre,
	/// x, o, gate -> y.
	Post,
	/// The chunk's keys and one partly filled block (no weights).
	Near,
	/// `Near` for a decode step: the new key is in the block already.
	Near0,
	/// `nblk` full blocks of past keys (no weights, no mask).
	Far(usize),
}

/// The per-layer constants of an attention layer's graph.
pub struct AttnConsts {
	/// RMSNorm weights `[5120]`.
	pub ln1: Vec<f32>,
	pub ln2: Vec<f32>,
	/// Head norm weights `[256]`.
	pub qnw: Vec<f32>,
	pub knw: Vec<f32>,
	/// Row scales of the int4 matrices: projection `[14336]`, o_proj
	/// `[5120]`, gate / up `[34816]`, down `[5120]`.
	pub s_proj: Vec<f32>,
	pub s_out: Vec<f32>,
	pub s_gu: Vec<f32>,
	pub s_down: Vec<f32>,
	/// Static ranges of the four quantized inputs.
	pub r_proj: f32,
	pub r_out: f32,
	pub r_gu: f32,
	pub r_down: f32,
}

struct AttnB<'a> {
	g: Graph,
	s: usize,
	c: &'a AttnConsts,
	sg: Vec<f32>,
	nw: usize,
}

impl AttnB<'_> {
	fn f64s(v: &[f32]) -> Vec<f64> {
		v.iter().map(|&x| x as f64).collect()
	}

	/// `fq(x R) @ (w_i * s_i)^T` for `parts` row blocks of an [n, k] matrix.
	fn linear(
		&mut self,
		x: &V,
		n: usize,
		k: usize,
		parts: usize,
		sc: &[f32],
		range: f32,
	) -> Vec<V> {
		let g = &mut self.g;
		let sgc = g.c16(&[k], &Self::f64s(&self.sg[..k]));
		let t = g.mul(x, &sgc);
		let xr = rotate(g, &t);
		let m = n / parts;
		let ws: Vec<V> = (0..parts)
			.map(|i| {
				let prm = g.parameter(&format!("w{}", self.nw), Ty::I4, &[m, k]);
				self.nw += 1;
				let c = g.convert(&prm, Ty::F16);
				let s = g.c16(&[m, 1], &Self::f64s(&sc[i * m..(i + 1) * m]));
				g.mul(&c, &s)
			})
			.collect();
		let (lo, hi) = (g.s16(-range as f64), g.s16(range as f64));
		let xq = g.fq(&xr, &lo, &hi, 255);
		ws.iter().map(|w| g.matmul(&xq, w, false, true)).collect()
	}

	fn rmsnorm(&mut self, x: &V, w: &[f32]) -> V {
		let y = rms_fused(&mut self.g, x, 1e-6);
		let wc = self.g.c16(&[w.len()], &Self::f64s(w));
		self.g.mul(&y, &wc)
	}

	/// Rotate the first [`RD`] channels of `t` [1, S, n, 256].
	fn rope(&mut self, t: &V, cos: &V, sin: &V) -> V {
		let g = &mut self.g;
		let rp = g.variadic_split(t, 3, &[RD, DA - RD]);
		let r12 = g.variadic_split(&rp[0], 3, &[RD / 2, RD / 2]);
		let neg = g.negative(&r12[1]);
		let rot = g.concat(&[neg, r12[0].clone()], 3);
		let (mut cos, mut sin) = (cos.clone(), sin.clone());
		if self.s == 1 {
			// one position: the compiler broadcasts [1, 1, 1, 64] over the
			// heads wrongly; spell the copies out
			let n = t.shape[2];
			cos = g.concat(&vec![cos; n], 2);
			sin = g.concat(&vec![sin; n], 2);
		}
		let a = g.mul(&rp[0], &cos);
		let b = g.mul(&rot, &sin);
		let ab = g.add(&a, &b);
		g.concat(&[ab, rp[1].clone()], 3)
	}

	/// (qg [1, 4, 6 S, 256], sigmoid(gate) [1, S, 6144], knew, vnew [1, S, 4, 256])
	fn pre(&mut self, x: &V, cos: &V, sin: &V) -> (V, V, V, V) {
		let s = self.s as i64;
		let c = self.c;
		let xn = self.rmsnorm(x, &c.ln1);
		let (nq, nk) = (HQ * DA, HKV * DA);
		let proj = self
			.linear(&xn, 2 * nq + 2 * nk, W, 1, &c.s_proj, c.r_proj)
			.remove(0);
		let sp = self.g.variadic_split(&proj, 2, &[nq, nq, nk, nk]);
		let cs = self.g.unsqueeze(cos, 2);
		let sn = self.g.unsqueeze(sin, 2);
		let scale = 1.0 / (DA as f32).sqrt();
		let qw: Vec<f32> = c.qnw.iter().map(|w| w * scale).collect();
		let q = self.g.reshape(&sp[0], &[1, s, HQ as i64, DA as i64]);
		let q = self.headnorm(&q, &qw);
		let q = self.rope(&q, &cs, &sn);
		let k = self.g.reshape(&sp[2], &[1, s, HKV as i64, DA as i64]);
		let k = self.headnorm(&k, &c.knw);
		let knew = self.rope(&k, &cs, &sn);
		let vnew = self.g.reshape(&sp[3], &[1, s, HKV as i64, DA as i64]);
		// query head h reads KV head h / 6: its 6 query heads as rows
		let g = &mut self.g;
		let t = g.reshape(&q, &[1, s, HKV as i64, G as i64, DA as i64]);
		let t = g.transpose(&t, &[0, 2, 3, 1, 4]);
		let qg = g.reshape(&t, &[1, HKV as i64, G as i64 * s, DA as i64]);
		let gate = g.sigmoid(&sp[1]);
		(qg, gate, knew, vnew)
	}

	fn headnorm(&mut self, t: &V, w: &[f32]) -> V {
		let y = rms_fused(&mut self.g, t, 1e-6);
		let wc = self.g.c16(&[w.len()], &Self::f64s(w));
		self.g.mul(&y, &wc)
	}

	fn post(&mut self, x: &V, o: &V, sgate: &V) -> V {
		let c = self.c;
		let og = self.g.mul(o, sgate);
		let op = self.linear(&og, W, HQ * DA, 1, &c.s_out, c.r_out).remove(0);
		let x1 = self.g.add(x, &op);
		let xn = self.rmsnorm(&x1, &c.ln2);
		let gu = self.linear(&xn, 2 * FF, W, GUSPLIT, &c.s_gu, c.r_gu);
		let gg = self.g.concat(&gu[..GUSPLIT / 2], 2);
		let uu = self.g.concat(&gu[GUSPLIT / 2..], 2);
		let sw = self.g.swish1(&gg);
		let h = self.g.mul(&sw, &uu);
		let dn = self.linear(&h, W, FF, 1, &c.s_down, c.r_down).remove(0);
		self.g.add(&x1, &dn)
	}
}

/// [1, n, 4, D] -> [1, 4, n, D]
fn heads(g: &mut Graph, t: &V) -> V {
	g.transpose(t, &[0, 2, 1, 3])
}

/// The additive mask [1, 1, S, n] repeated for the 6 query heads of a KV head.
fn mask6(g: &mut Graph, mask: &V, s: usize, n: usize) -> V {
	let t = g.unsqueeze(mask, 2);
	let t = g.broadcast_to(&t, &[1, 1, G, s, n]);
	g.reshape(&t, &[1, 1, (G * s) as i64, n as i64])
}

/// The last two columns of the attention weights: with `softmax_lse` as the
/// Softmax kernel they come back as (max score, sum exp(score - max)).
fn lse(g: &mut Graph, p: &V) -> V {
	let n = p.shape[3];
	g.variadic_split(p, 3, &[n - 2, 2]).remove(1)
}

/// A weight-free attention graph ([`Part::Near`], [`Part::Near0`],
/// [`Part::Far`]) over `s` positions and blocks of `cap` key slots (the last
/// two of a block are not keys).
pub fn attn_core(s: usize, cap: usize, part: Part) -> Graph {
	let mut g = Graph::new();
	let qg = g.parameter("qg", Ty::F16, &[1, HKV, G * s, DA]);
	let blk = [1, cap, HKV, DA];
	match part {
		Part::Near => {
			let kpast = g.parameter("kpast", Ty::F16, &blk);
			let vpast = g.parameter("vpast", Ty::F16, &blk);
			let kn = g.parameter("knew", Ty::F16, &[1, s, HKV, DA]);
			let vn = g.parameter("vnew", Ty::F16, &[1, s, HKV, DA]);
			let mask = g.parameter("mask", Ty::F16, &[1, 1, s, cap + s]);
			let mk = mask6(&mut g, &mask, s, cap + s);
			let k = g.concat(&[kn, kpast], 1);
			let k = heads(&mut g, &k);
			let sc = g.matmul(&qg, &k, false, true);
			let sc = g.add(&sc, &mk);
			let p = g.softmax(&sc, 3);
			let v = g.concat(&[vn, vpast], 1);
			let v = heads(&mut g, &v);
			let o = g.matmul(&p, &v, false, false);
			g.result(&o, "o");
			let l = lse(&mut g, &p);
			g.result(&l, "lse");
		}
		Part::Near0 => {
			let kpast = g.parameter("kpast", Ty::F16, &blk);
			let vpast = g.parameter("vpast", Ty::F16, &blk);
			let mask = g.parameter("mask", Ty::F16, &[1, 1, s, cap]);
			let mk = mask6(&mut g, &mask, s, cap);
			let k = heads(&mut g, &kpast);
			let sc = g.matmul(&qg, &k, false, true);
			let sc = g.add(&sc, &mk);
			let p = g.softmax(&sc, 3);
			let v = heads(&mut g, &vpast);
			let o = g.matmul(&p, &v, false, false);
			g.result(&o, "o");
			let l = lse(&mut g, &p);
			g.result(&l, "lse");
		}
		Part::Far(1) => {
			let kb = g.parameter("kblk", Ty::F16, &blk);
			let vb = g.parameter("vblk", Ty::F16, &blk);
			let k = heads(&mut g, &kb);
			let sc = g.matmul(&qg, &k, false, true);
			let p = g.softmax(&sc, 3);
			let v = heads(&mut g, &vb);
			let o = g.matmul(&p, &v, false, false);
			g.result(&o, "o");
			let l = lse(&mut g, &p);
			g.result(&l, "lse");
		}
		Part::Far(n) => {
			// several blocks per job (decode steps): o / lse block-major
			let kbs: Vec<V> = (0..n)
				.map(|i| g.parameter(&format!("kblk{i}"), Ty::F16, &blk))
				.collect();
			let vbs: Vec<V> = (0..n)
				.map(|i| g.parameter(&format!("vblk{i}"), Ty::F16, &blk))
				.collect();
			let ps: Vec<V> = kbs
				.iter()
				.map(|k| {
					let k = heads(&mut g, k);
					let sc = g.matmul(&qg, &k, false, true);
					g.softmax(&sc, 3)
				})
				.collect();
			let os: Vec<V> = ps
				.iter()
				.zip(&vbs)
				.map(|(p, v)| {
					let v = heads(&mut g, v);
					g.matmul(p, &v, false, false)
				})
				.collect();
			let o = g.concat(&os, 1);
			g.result(&o, "o");
			let ls: Vec<V> = ps.iter().map(|p| lse(&mut g, p)).collect();
			let l = g.concat(&ls, 1);
			g.result(&l, "lse");
		}
		_ => panic!("attn_core: {part:?}"),
	}
	g
}

/// An attention layer over `s` positions ([`Part::All`] with a window of
/// `cap` past keys, [`Part::Pre`], [`Part::Post`]).
///
/// `All`: `x`, `kpast` / `vpast` [1, cap, 4, 256] (token-major), `mask`
/// [1, 1, s, cap + s] (additive), `cos` / `sin` [1, s, 64], weights `w0..w6`
/// -> `y`, `knew`, `vnew` [1, s, 4, 256].
pub fn attn(s: usize, cap: usize, part: Part, c: &AttnConsts) -> Graph {
	let mut b = AttnB {
		g: Graph::new(),
		s,
		c,
		sg: crate::gptq::signs(FF),
		nw: 0,
	};
	let x = b.g.parameter("x", Ty::F16, &[1, s, W]);
	match part {
		Part::All => {
			let kpast = b.g.parameter("kpast", Ty::F16, &[1, cap, HKV, DA]);
			let vpast = b.g.parameter("vpast", Ty::F16, &[1, cap, HKV, DA]);
			let mask = b.g.parameter("mask", Ty::F16, &[1, 1, s, cap + s]);
			let cos = b.g.parameter("cos", Ty::F16, &[1, s, RD]);
			let sin = b.g.parameter("sin", Ty::F16, &[1, s, RD]);
			let (qg, gate, knew, vnew) = b.pre(&x, &cos, &sin);
			let g = &mut b.g;
			let mk = mask6(g, &mask, s, cap + s);
			let k = g.concat(&[kpast, knew.clone()], 1);
			let k = heads(g, &k);
			let v = g.concat(&[vpast, vnew.clone()], 1);
			let v = heads(g, &v);
			let sc = g.matmul(&qg, &k, false, true);
			let sc = g.add(&sc, &mk);
			let p = g.softmax(&sc, 3);
			let o = g.matmul(&p, &v, false, false);
			let o = g.reshape(&o, &[1, HKV as i64, G as i64, s as i64, DA as i64]);
			let o = g.transpose(&o, &[0, 3, 1, 2, 4]);
			let o = g.reshape(&o, &[1, s as i64, (HQ * DA) as i64]);
			let y = b.post(&x, &o, &gate);
			b.g.result(&y, "y");
			b.g.result(&knew, "knew");
			b.g.result(&vnew, "vnew");
		}
		Part::Pre => {
			let cos = b.g.parameter("cos", Ty::F16, &[1, s, RD]);
			let sin = b.g.parameter("sin", Ty::F16, &[1, s, RD]);
			let (qg, gate, knew, vnew) = b.pre(&x, &cos, &sin);
			for (t, n) in [
				(&qg, "qg"),
				(&gate, "gate"),
				(&knew, "knew"),
				(&vnew, "vnew"),
			] {
				b.g.result(t, n);
			}
		}
		Part::Post => {
			// weights named as in the whole layer's graph (w0 is the projection)
			b.nw = 1;
			let o = b.g.parameter("o", Ty::F16, &[1, s, HQ * DA]);
			let gate = b.g.parameter("gate", Ty::F16, &[1, s, HQ * DA]);
			let y = b.post(&x, &o, &gate);
			b.g.result(&y, "y");
		}
		_ => panic!("attn: {part:?}"),
	}
	b.g
}
