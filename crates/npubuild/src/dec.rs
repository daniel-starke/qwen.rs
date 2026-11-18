//! The decode-side graphs: the verify set's layers (`s` positions per step,
//! deferred-rollback WY form of the delta rule), the head, the MTP block and
//! its draft head. Activations are f32 in the IR (the NPU runs them as f16),
//! the int4 codes are the prefill sets' bound buffers (inputs named like
//! their files), each with its row scales on the weight in front of the
//! MatMul: the pattern the compiler turns into an int4 fully-connected
//! layer at these sizes.

use crate::g::{Graph, Ty, V};
use crate::gptq::{signs, ROTB};
use crate::lean::{GUSPLIT, PROJSPLIT};
use crate::refm::{D, DA, FF, H, HK, HKV, HQ, RD, W};

/// The scale the decode graphs' queries run at: `2^6 / sqrt(D)` (the
/// recurrence's output and the gated norm's eps scale with it).
const QDEC: f64 = 64.0;
const G: usize = HQ / HKV;
/// Additive mask value (finite in f16).
const NEG: f64 = -30000.0;

fn f64s(v: &[f32]) -> Vec<f64> {
	v.iter().map(|&x| x as f64).collect()
}

fn hadamard() -> Vec<f64> {
	let n = ROTB;
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

/// `x / sqrt(mean(x^2) + eps)` over the last axis.
fn rms(g: &mut Graph, x: &V, eps: f64) -> V {
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

/// `x R` (f32): the rotation's signs, then the block Hadamard MatMul.
fn rotate(g: &mut Graph, x: &V, sg: &[f32]) -> V {
	let k = *x.shape.last().unwrap();
	let sc = g.cf(x.ty, &[k], &f64s(&sg[..k]));
	let t = g.mul(x, &sc);
	let t = g.reshape(&t, &[-1, ROTB as i64]);
	let hb = g.cf(x.ty, &[ROTB, ROTB], &hadamard());
	let t = g.matmul(&t, &hb, false, false);
	let shape: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
	g.reshape(&t, &shape)
}

/// The int4 weight input `name` [n, 1, k] with its row scales: f32 [n, k].
fn weight(g: &mut Graph, name: &str, n: usize, k: usize, sc: &[f32]) -> V {
	assert_eq!(sc.len(), n, "{name}");
	let prm = g.parameter(name, Ty::I4, &[n, 1, k]);
	let c = g.convert(&prm, Ty::F16);
	let s = g.c16(&[n, 1, 1], &f64s(sc));
	let w = g.mul(&c, &s);
	let w = g.reshape(&w, &[n as i64, k as i64]);
	g.convert(&w, Ty::F32)
}

/// `rot(x) @ W^T` for a matrix stored as row blocks: the named int4 inputs
/// of `rows` rows each, concatenated.
fn fc(g: &mut Graph, xr: &V, parts: &[(String, usize)], k: usize, sc: &[f32]) -> V {
	let mut r0 = 0;
	let ys: Vec<V> = parts
		.iter()
		.map(|(name, n)| {
			let w = weight(g, name, *n, k, &sc[r0..r0 + n]);
			r0 += n;
			g.matmul(xr, &w, false, true)
		})
		.collect();
	assert_eq!(r0, sc.len());
	if ys.len() == 1 {
		ys[0].clone()
	} else {
		g.concat(&ys, -1)
	}
}

/// The names of a layer's MLP weight inputs: (gate blocks, up blocks, down).
pub fn mlp_names(attn: bool) -> (Vec<String>, Vec<String>, String) {
	let w = |i: usize| format!("w{i}");
	let h = GUSPLIT / 2;
	if attn {
		(
			(0..h).map(|i| w(2 + i)).collect(),
			(0..h).map(|i| w(2 + h + i)).collect(),
			w(2 + GUSPLIT),
		)
	} else {
		(
			(0..h).map(|i| w(16 + 2 * i)).collect(),
			(0..h).map(|i| w(16 + 2 * (h + i))).collect(),
			w(24),
		)
	}
}

/// What a layer's MLP needs: the norm weight and the row scales of gate / up
/// (`[2 FF]`) and down.
pub struct MlpC {
	pub ln2: Vec<f32>,
	pub s_gu: Vec<f32>,
	pub s_down: Vec<f32>,
}

/// `x1 + W_down rot(silu(W_gate rot(n)) * W_up rot(n))`, `n` the normed `x1`.
fn mlp(g: &mut Graph, x1: &V, c: &MlpC, attn: bool, sg: &[f32]) -> V {
	let (gate, up, down) = mlp_names(attn);
	let n = FF / gate.len();
	let xn = rms(g, x1, 1e-6);
	let lw = g.c32(&[1, 1, W], &c.ln2);
	let xn = g.mul(&xn, &lw);
	let xr = rotate(g, &xn, sg);
	let parts = |v: &[String]| v.iter().map(|s| (s.clone(), n)).collect::<Vec<_>>();
	let gg = fc(g, &xr, &parts(&gate), W, &c.s_gu[..FF]);
	let sw = g.swish(&gg);
	let xr2 = rotate(g, &xn, sg);
	let uu = fc(g, &xr2, &parts(&up), W, &c.s_gu[FF..]);
	let h = g.mul(&sw, &uu);
	let hr = rotate(g, &h, sg);
	let dn = fc(g, &hr, &[(down, W)], FF, &c.s_down);
	g.add(x1, &dn)
}

/// The per-layer constants of a delta-net layer's decode graph.
pub struct DeltaC {
	pub ln1: Vec<f32>,
	/// Row scales of the q k v z rows `[16384]` and of out_proj.
	pub s_proj: Vec<f32>,
	pub s_out: Vec<f32>,
	/// The int8 gate rows `[96][5120]` (a, then b; rotated) and their scales.
	pub ab: Vec<i8>,
	pub s_ab: Vec<f32>,
	/// `[10240][4]`
	pub conv: Vec<f32>,
	pub dtb: Vec<f32>,
	pub acoef: Vec<f32>,
	pub gnw: Vec<f32>,
	pub mlp: MlpC,
}

/// A delta-net layer's verify graph over `s` positions.
///
/// Inputs: `x` [1, s, 5120]; `cache_params.past.conv.{idx}` [1, 10240, 4]
/// (the last four positions' qkv, channel-major); `cache_params.past.ssm.{idx}`
/// [1, 48, 128, 128] = `sbase` of the previous step; `qwen_rs.kp` / `.dp`
/// [1, 48, s, 128], `qwen_rs.w` [1, 48, s, 1], `qwen_rs.alpha`
/// [1, 48, 1, 1]: the previous step's rollback, applied here as
/// `S0 = alpha sbase + (kp w)^T dp`. Outputs, in this order: `y`,
/// `conv_full` [1, 10240, 4 + s], `sbase` = S0, `kcur`, `dnew`
/// [1, 48, s, 128], `gcum` [1, 48, s, 1] (f32). With `with_mlp` false `y` is
/// the residual before the MLP (the engine runs it).
pub fn delta_verify(s: usize, idx: usize, c: &DeltaC, with_mlp: bool) -> Graph {
	let mut gr = Graph::new();
	let g = &mut gr;
	let sg = signs(FF);
	let nq = 2 * HK * D + H * D;
	let x = g.parameter("x", Ty::F16, &[1, s, W]);
	let cpast = g.parameter(
		&format!("cache_params.past.conv.{idx}"),
		Ty::F16,
		&[1, nq, 4],
	);
	let spast = g.parameter(
		&format!("cache_params.past.ssm.{idx}"),
		Ty::F16,
		&[1, H, D, D],
	);
	let kp = g.parameter("qwen_rs.kp", Ty::F16, &[1, H, s, D]);
	let dp = g.parameter("qwen_rs.dp", Ty::F16, &[1, H, s, D]);
	let wr = g.parameter("qwen_rs.w", Ty::F16, &[1, H, s, 1]);
	let al = g.parameter("qwen_rs.alpha", Ty::F16, &[1, H, 1, 1]);
	let x32 = g.convert(&x, Ty::F32);
	let xn = rms(g, &x32, 1e-6);
	let lw = g.c32(&[1, 1, W], &c.ln1);
	let xn = g.mul(&xn, &lw);
	let xr = rotate(g, &xn, &sg);
	// q k v rows (two blocks), z, and the int8 gate rows
	let qkv = fc(
		g,
		&xr,
		&[("w1".into(), PROJSPLIT[0]), ("w3".into(), PROJSPLIT[1])],
		W,
		&c.s_proj[..nq],
	);
	let z = fc(g, &xr, &[("w5".into(), PROJSPLIT[2])], W, &c.s_proj[nq..]);
	let mut ab = Vec::new();
	for i in 0..2 {
		let q: Vec<u8> = c.ab[i * H * W..(i + 1) * H * W]
			.iter()
			.map(|&v| v as u8)
			.collect();
		let w = g.constant(Ty::I8, &[H, W], &q);
		let w = g.convert(&w, Ty::F16);
		let sc = g.c16(&[H, 1], &f64s(&c.s_ab[i * H..(i + 1) * H]));
		let w = g.mul(&w, &sc);
		let w = g.convert(&w, Ty::F32);
		ab.push(g.matmul(&xr, &w, false, true));
	}
	// gates [1, 48, s, 1]: log decay g = -exp(A_log) softplus(a + dt_bias), beta
	let dtb = g.c32(&[1, 1, H], &c.dtb);
	let t = g.add(&ab[0], &dtb);
	let t = g.softplus(&t);
	let ac = g.c32(&[1, 1, H], &c.acoef);
	let t = g.mul(&t, &ac);
	let t = g.transpose(&t, &[0, 2, 1]);
	let gate = g.reshape(&t, &[1, H as i64, s as i64, 1]);
	let t = g.sigmoid(&ab[1]);
	let t = g.transpose(&t, &[0, 2, 1]);
	let beta = g.reshape(&t, &[1, H as i64, s as i64, 1]);
	// conv over the window and the new positions
	let cp = g.convert(&cpast, Ty::F32);
	let ct = g.transpose(&qkv, &[0, 2, 1]);
	let cfull = g.concat(&[cp, ct], -1);
	let cw = g.c32(&[nq, 1, 1, 4], &c.conv);
	let cv = g.group_conv1d(&cfull, &cw);
	let cv = g.slice(&cv, 1, s + 1, 1, 2);
	let cv = g.swish(&cv);
	let cv = g.transpose(&cv, &[0, 2, 1]);
	let sp = g.variadic_split(&cv, -1, &[HK * D, HK * D, H * D]);
	let v = g.reshape(&sp[2], &[1, s as i64, H as i64, D as i64]);
	let v = g.transpose(&v, &[0, 2, 1, 3]);
	let idx48: Vec<i32> = (0..H).map(|h| (h / (H / HK)) as i32).collect();
	let l2 = |g: &mut Graph, t: &V| {
		let t = g.reshape(t, &[1, s as i64, HK as i64, D as i64]);
		let ix = g.ci32(&idx48);
		let t = g.gather(&t, &ix, 2);
		let sq = g.mul(&t, &t);
		let ss = g.reduce_sum(&sq, -1);
		let e = g.s32(1e-6);
		let ss = g.add(&ss, &e);
		let rt = g.sqrt(&ss);
		let one = g.s32(1.0);
		let r = g.div(&one, &rt);
		let t = g.mul(&t, &r);
		g.transpose(&t, &[0, 2, 1, 3])
	};
	let k = l2(g, &sp[1]);
	let q = l2(g, &sp[0]);
	let qs = g.s32((QDEC / (D as f64).sqrt()) as f32);
	let q = g.mul(&q, &qs);
	// S0 = alpha sbase + (kp w)^T dp
	let s32 = g.convert(&spast, Ty::F32);
	let al = g.convert(&al, Ty::F32);
	let sa = g.mul(&s32, &al);
	let kp = g.convert(&kp, Ty::F32);
	let wr = g.convert(&wr, Ty::F32);
	let kw = g.mul(&kp, &wr);
	let dp = g.convert(&dp, Ty::F32);
	let roll = g.matmul(&kw, &dp, true, false);
	let s0 = g.add(&sa, &roll);
	// G = cumsum(g); A = exp(G); M_jm = exp(G_j - G_m), m <= j (the exponent
	// masked first: above the diagonal it can overflow)
	let tri: Vec<f32> = (0..s * s).map(|i| (i % s <= i / s) as u8 as f32).collect();
	let tric = g.c32(&[s, s], &tri);
	let gc = g.matmul(&tric, &gate, false, false);
	let a = g.exp(&gc);
	let m1 = g.s32(-1.0);
	let gn = g.mul(&gc, &m1);
	let gt = g.transpose(&gn, &[0, 1, 3, 2]);
	let dg = g.add(&gc, &gt);
	let tric2 = g.c32(&[s, s], &tri);
	let dg = g.mul(&dg, &tric2);
	let e = g.exp(&dg);
	let tric3 = g.c32(&[s, s], &tri);
	let m = g.mul(&e, &tric3);
	let ks = g.matmul(&k, &s0, false, false);
	let qsm = g.matmul(&q, &s0, false, false);
	// D = (I - Bn)^-1 rhs with Bn = -beta (K K^T * M) strictly lower: exact
	// after s - 1 terms of rhs + Bn (rhs + Bn (...))
	let kk = g.matmul(&k, &k, false, true);
	let kkm = g.mul(&kk, &m);
	let low: Vec<f32> = (0..s * s)
		.map(|i| if i % s < i / s { -1.0 } else { 0.0 })
		.collect();
	let lowc = g.c32(&[s, s], &low);
	let bn = g.mul(&kkm, &lowc);
	let bn = g.mul(&bn, &beta);
	let m1 = g.s32(-1.0);
	let an = g.mul(&a, &m1);
	let t = g.mul(&an, &ks);
	let t = g.add(&v, &t);
	let rhs = g.mul(&t, &beta);
	let mut dn = rhs.clone();
	for _ in 1..s {
		let t = g.matmul(&bn, &dn, false, false);
		dn = g.add(&rhs, &t);
	}
	let qk = g.matmul(&q, &k, false, true);
	let qkm = g.mul(&qk, &m);
	let t1 = g.mul(&a, &qsm);
	let t2 = g.matmul(&qkm, &dn, false, false);
	let o = g.add(&t1, &t2);
	// gated RMSNorm per head, out_proj, residual
	let o = g.transpose(&o, &[0, 2, 1, 3]);
	let o = g.reshape(&o, &[-1, D as i64]);
	let on = rms(g, &o, 1e-6 * QDEC * QDEC);
	let gw = g.c32(&[1, D], &c.gnw);
	let on = g.mul(&gw, &on);
	let zz = g.reshape(&z, &[(s * H) as i64, D as i64]);
	let zz = g.swish(&zz);
	let att = g.mul(&on, &zz);
	let att = g.reshape(&att, &[1, s as i64, (H * D) as i64]);
	let ar = rotate(g, &att, &sg);
	let op = fc(g, &ar, &[("w13".into(), W)], H * D, &c.s_out);
	let x1 = g.add(&x32, &op);
	let y = if with_mlp {
		mlp(g, &x1, &c.mlp, false, &sg)
	} else {
		x1
	};
	let y = g.convert(&y, Ty::F16);
	g.result(&y, "y");
	let t = g.convert(&cfull, Ty::F16);
	g.result(&t, "conv_full");
	let t = g.convert(&s0, Ty::F16);
	g.result(&t, "sbase");
	let t = g.convert(&k, Ty::F16);
	g.result(&t, "kcur");
	let t = g.convert(&dn, Ty::F16);
	g.result(&t, "dnew");
	g.result(&gc, "gcum");
	gr
}

/// The result names of [`delta_verify`], in order (a line of `outs.txt`).
pub const DELTA_OUTS: &str = "y conv_full sbase kcur dnew gcum";

/// The per-layer constants of an attention layer's decode graph.
pub struct AttnC {
	pub ln1: Vec<f32>,
	pub qnw: Vec<f32>,
	pub knw: Vec<f32>,
	/// Row scales of the projection `[14336]` (q rows, gate rows, k, v) and
	/// of o_proj.
	pub s_proj: Vec<f32>,
	pub s_out: Vec<f32>,
	pub mlp: MlpC,
}

/// How an attention graph is wired to its caller.
enum Io<'a> {
	/// A layer of the verify set: ring-buffer window inputs
	/// `cache_params.past.key.{i}` [1, 4, cap, 256], `.value.{i}`
	/// [1, 4, 256, cap] (transposed), `attn_bias` [1, 1, 1, cap] (additive),
	/// `position_ids` [4, 1, s]; outputs `y`, the new value rows, the new
	/// key rows [1, 4, s, 256].
	Layer(usize),
	/// The MTP block: `past_key` / `past_value` [1, 4, cap, 256],
	/// `attention_mask` [1, cap + s] i64 (1 = a key), `position_ids` [1, s];
	/// outputs the whole windows [1, 4, cap + s, 256]. `x` is computed by the
	/// caller.
	Mtp(&'a V),
}

/// Rotate the first [`RD`] channels of `t` [1, n, s, 256] by `cos` / `sin`
/// [1, 1, s, 64].
fn rope(g: &mut Graph, t: &V, cos: &V, sin: &V) -> V {
	let rp = g.variadic_split(t, 3, &[RD, DA - RD]);
	let a = g.mul(&rp[0], cos);
	let r12 = g.variadic_split(&rp[0], 3, &[RD / 2, RD / 2]);
	let m1 = g.s32(-1.0);
	let neg = g.mul(&r12[1], &m1);
	let rot = g.concat(&[neg, r12[0].clone()], -1);
	let b = g.mul(&rot, sin);
	let ab = g.add(&a, &b);
	g.concat(&[ab, rp[1].clone()], -1)
}

/// The attention mixer and (optionally) the MLP of one layer: see [`Io`].
/// Returns the layer's output (f32) after declaring the window results.
#[allow(clippy::too_many_arguments)]
fn attn_body(
	g: &mut Graph,
	s: usize,
	cap: usize,
	io: Io,
	c: &AttnC,
	wname: &dyn Fn(usize) -> String,
	with_mlp: bool,
	sg: &[f32],
) -> (V, Vec<(V, &'static str)>) {
	let mtp = matches!(io, Io::Mtp(_));
	let (x32, pk, pv, bias, pos) = match io {
		Io::Layer(i) => {
			let x = g.parameter("x", Ty::F16, &[1, s, W]);
			let pk = g.parameter(
				&format!("cache_params.past.key.{i}"),
				Ty::F16,
				&[1, HKV, cap, DA],
			);
			let pv = g.parameter(
				&format!("cache_params.past.value.{i}"),
				Ty::F16,
				&[1, HKV, DA, cap],
			);
			let pos = g.parameter("position_ids", Ty::I64, &[4, 1, s]);
			let bias = g.parameter("attn_bias", Ty::F16, &[1, 1, 1, cap]);
			// the t row of the (t, h, w) positions: text positions are equal in all
			let p = g.slice(&pos, 1, 2, 1, 0);
			let p = g.reshape(&p, &[1, s as i64, 1]);
			(g.convert(&x, Ty::F32), pk, pv, bias, p)
		}
		Io::Mtp(x) => {
			let pk = g.parameter("past_key", Ty::F16, &[1, HKV, cap, DA]);
			let pv = g.parameter("past_value", Ty::F16, &[1, HKV, cap, DA]);
			let pos = g.parameter("position_ids", Ty::I64, &[1, s]);
			let am = g.parameter("attention_mask", Ty::I64, &[1, cap + s]);
			// (1 - m) * NEG, f16
			let m = g.reshape(&am, &[1, 1, 1, (cap + s) as i64]);
			let m = g.convert(&m, Ty::F16);
			let one = g.s16(1.0);
			let t = g.sub(&one, &m);
			let neg = g.s16(NEG);
			let bias = g.mul(&t, &neg);
			let p = g.reshape(&pos, &[1, s as i64, 1]);
			(x.clone(), pk, pv, bias, p)
		}
	};
	let xn = rms(g, &x32, 1e-6);
	let lw = g.c32(&[1, 1, W], &c.ln1);
	let xn = g.mul(&xn, &lw);
	let xr = rotate(g, &xn, sg);
	let (nq, nk) = (HQ * DA, HKV * DA);
	let proj = fc(g, &xr, &[(wname(0), 2 * nq + 2 * nk)], W, &c.s_proj);
	// rotary angles of the s positions: [1, 1, s, 64]
	let pf = g.convert(&pos, Ty::F32);
	let inv: Vec<f32> = crate::refm::inv_freq().iter().map(|&v| v as f32).collect();
	let invc = g.c32(&[1, RD / 2], &inv);
	let ang = g.matmul(&pf, &invc, false, false);
	let ang = g.concat(&[ang.clone(), ang], -1);
	let cos = g.cos(&ang);
	let cos = g.reshape(&cos, &[1, 1, s as i64, RD as i64]);
	let sin = g.sin(&ang);
	let sin = g.reshape(&sin, &[1, 1, s as i64, RD as i64]);
	let headnorm = |g: &mut Graph, t: &V, n: usize, w: &[f32]| {
		let t = g.reshape(t, &[1, s as i64, n as i64, DA as i64]);
		let t = rms(g, &t, 1e-6);
		let wc = g.c32(&[1, 1, 1, DA], w);
		let t = g.mul(&t, &wc);
		g.transpose(&t, &[0, 2, 1, 3])
	};
	let kraw = g.slice(&proj, 2 * nq, 2 * nq + nk, 1, 2);
	let k = headnorm(g, &kraw, HKV, &c.knw);
	let k = rope(g, &k, &cos, &sin);
	let knew = g.convert(&k, Ty::F16);
	let vraw = g.slice(&proj, 2 * nq + nk, 2 * nq + 2 * nk, 1, 2);
	let v = g.reshape(&vraw, &[1, s as i64, HKV as i64, DA as i64]);
	let v = g.transpose(&v, &[0, 2, 1, 3]);
	let vnew = g.convert(&v, Ty::F16);
	let qraw = g.slice(&proj, 0, nq, 1, 2);
	let q = headnorm(g, &qraw, HQ, &c.qnw);
	let q = rope(g, &q, &cos, &sin);
	let q = g.convert(&q, Ty::F16);
	// [1, 24, s, 256] -> [1, 4, 6 s, 256]: query head h reads KV head h / 6
	let qg = g.reshape(&q, &[1, HKV as i64, (G * s) as i64, DA as i64]);
	let scale = g.s16(1.0 / (DA as f64).sqrt());
	// the causal mask among the new rows, per (query head of the group, position)
	let tri: Vec<f64> = (0..G * s * s)
		.map(|i| if i % s <= (i / s) % s { 0.0 } else { NEG })
		.collect();
	let rep = |g: &mut Graph, b: &V, n: usize| {
		let t = g.unsqueeze(b, 2);
		let t = g.broadcast_to(&t, &[1, 1, G, s, n]);
		g.reshape(&t, &[1, 1, (G * s) as i64, n as i64])
	};
	let mut outs = Vec::new();
	let o = if mtp {
		let kall = g.concat(&[pk, knew], 2);
		let vall = g.concat(&[pv, vnew], 2);
		let sc = g.matmul(&qg, &kall, false, true);
		let sc = g.mul(&sc, &scale);
		// past keys: the mask; new keys: the mask and the causal order
		let mut cm = vec![0f64; s * (cap + s)];
		for i in 0..s {
			for j in i + 1..s {
				cm[i * (cap + s) + cap + j] = NEG;
			}
		}
		let cmc = g.c16(&[1, 1, s, cap + s], &cm);
		let b = g.add(&cmc, &bias);
		let b = rep(g, &b, cap + s);
		let sc = g.add(&sc, &b);
		let p = g.softmax(&sc, 3);
		let o = g.matmul(&p, &vall, false, false);
		outs.push((kall, "present_key"));
		outs.push((vall, "present_value"));
		o
	} else {
		let sp = g.matmul(&qg, &pk, false, true);
		let sp = g.mul(&sp, &scale);
		let b = rep(g, &bias, cap);
		let sp = g.add(&sp, &b);
		let sn = g.matmul(&qg, &knew, false, true);
		let mut sn = g.mul(&sn, &scale);
		if s > 1 {
			let tc = g.c16(&[1, 1, G * s, s], &tri);
			sn = g.add(&sn, &tc);
		}
		let all = g.concat(&[sp, sn], 3);
		let p = g.softmax(&all, 3);
		let pp = g.variadic_split(&p, 3, &[cap, s]);
		let op = g.matmul(&pp[0], &pv, false, true);
		let on = g.matmul(&pp[1], &vnew, false, false);
		outs.push((vnew, "value"));
		outs.push((knew, "key"));
		g.add(&op, &on)
	};
	// [1, 4, 6 s, 256] -> [1, s, 24 * 256], gate, o_proj, residual
	let o = g.reshape(&o, &[1, HQ as i64, s as i64, DA as i64]);
	let o = g.convert(&o, Ty::F32);
	let o = g.transpose(&o, &[0, 2, 1, 3]);
	let o = g.reshape(&o, &[1, s as i64, nq as i64]);
	let graw = g.slice(&proj, nq, 2 * nq, 1, 2);
	let gate = g.sigmoid(&graw);
	let og = g.mul(&o, &gate);
	let ogr = rotate(g, &og, sg);
	let op = fc(g, &ogr, &[(wname(1), W)], nq, &c.s_out);
	let x1 = g.add(&x32, &op);
	let y = if with_mlp {
		if mtp {
			mlp_named(g, &x1, &c.mlp, wname, sg)
		} else {
			mlp(g, &x1, &c.mlp, true, sg)
		}
	} else {
		x1
	};
	(y, outs)
}

/// [`mlp`] with the weight inputs named by `wname(2 ..= 2 + GUSPLIT)`.
fn mlp_named(g: &mut Graph, x1: &V, c: &MlpC, wname: &dyn Fn(usize) -> String, sg: &[f32]) -> V {
	let n = 2 * FF / GUSPLIT;
	let xn = rms(g, x1, 1e-6);
	let lw = g.c32(&[1, 1, W], &c.ln2);
	let xn = g.mul(&xn, &lw);
	let xr = rotate(g, &xn, sg);
	let h = GUSPLIT / 2;
	let gg = fc(
		g,
		&xr,
		&(0..h).map(|i| (wname(2 + i), n)).collect::<Vec<_>>(),
		W,
		&c.s_gu[..FF],
	);
	let sw = g.swish(&gg);
	let uu = fc(
		g,
		&xr,
		&(h..GUSPLIT).map(|i| (wname(2 + i), n)).collect::<Vec<_>>(),
		W,
		&c.s_gu[FF..],
	);
	let hh = g.mul(&sw, &uu);
	let hr = rotate(g, &hh, sg);
	let dn = fc(g, &hr, &[(wname(2 + GUSPLIT), W)], FF, &c.s_down);
	g.add(x1, &dn)
}

/// An attention layer's verify graph over `s` positions with a ring window
/// of `cap` keys (see [`Io::Layer`]); `idx` numbers the attention layers.
/// Outputs in this order: `y`, `cache_params.present.value.{idx}`,
/// `cache_params.present.key.{idx}`.
pub fn attn_verify(s: usize, cap: usize, idx: usize, c: &AttnC, with_mlp: bool) -> Graph {
	let mut g = Graph::new();
	let sg = signs(FF);
	let (y, outs) = attn_body(
		&mut g,
		s,
		cap,
		Io::Layer(idx),
		c,
		&|i| format!("w{i}"),
		with_mlp,
		&sg,
	);
	let y = g.convert(&y, Ty::F16);
	g.result(&y, "y");
	for (t, n) in outs {
		g.result(&t, &format!("cache_params.present.{n}.{idx}"));
	}
	g
}

/// The line of `outs.txt` of [`attn_verify`].
pub fn attn_outs(idx: usize) -> String {
	format!("y cache_params.present.value.{idx} cache_params.present.key.{idx}")
}

/// Name of the lm_head's bound weight input (and file, `<name>.in`): int8
/// codes `[vocab][5120]`.
pub const LM_HEAD: &str = "w.lm_head.i8";

/// The head over `s` positions: `x` [1, s, 5120] -> `logits` (f32
/// [1, s, vocab]) and `hn` (the output-normed hidden state, f16). `sc`: the
/// row scales of the int8 lm_head codes. The first f32 [1, 1, 5120] constant
/// of the IR is the final norm's weight (the engine reads it from there).
pub fn head(s: usize, norm: &[f32], sc: &[f32]) -> Graph {
	let mut gr = Graph::new();
	let g = &mut gr;
	let v = sc.len();
	let x = g.parameter("x", Ty::F16, &[1, s, W]);
	let x32 = g.convert(&x, Ty::F32);
	let xn = rms(g, &x32, 1e-6);
	let nw = g.c32(&[1, 1, W], norm);
	let hn = g.mul(&xn, &nw);
	let w = g.parameter(LM_HEAD, Ty::I8, &[v, W]);
	let w = g.convert(&w, Ty::F16);
	let c = g.s16(2f64.powi(-8));
	let w = g.mul(&w, &c);
	let w = g.convert(&w, Ty::F32);
	let lg = g.matmul(&hn, &w, false, true);
	let s256: Vec<f32> = sc.iter().map(|s| s * 256.0).collect();
	let scc = g.c32(&[1, 1, v], &s256);
	let lg = g.mul(&lg, &scc);
	g.result(&lg, "logits");
	let h16 = g.convert(&hn, Ty::F16);
	g.result(&h16, "hn");
	gr
}

/// The draft head: lm_head rows `[0, sc.len())` on an already normed `x`
/// [1, 1, 5120] -> `logits` (f32). Binds a prefix of the lm_head's buffer.
pub fn mtp_head(sc: &[f32]) -> Graph {
	let mut gr = Graph::new();
	let g = &mut gr;
	let v = sc.len();
	let x = g.parameter("x", Ty::F16, &[1, 1, W]);
	let w = g.parameter(LM_HEAD, Ty::I8, &[v, W]);
	let w = g.convert(&w, Ty::F16);
	let c = g.s16(2f64.powi(-8));
	let w = g.mul(&w, &c);
	let lg = g.matmul(&x, &w, false, true);
	let s256: Vec<f64> = sc.iter().map(|s| *s as f64 * 256.0).collect();
	let scc = g.c16(&[1, 1, v], &s256);
	let lg = g.mul(&lg, &scc);
	let lg = g.convert(&lg, Ty::F32);
	g.result(&lg, "logits");
	gr
}

/// The constants of the MTP block's graph.
pub struct MtpC {
	/// The norms of the embedding and of the hidden state before `fc`, and
	/// the block's output norm.
	pub ne: Vec<f32>,
	pub nh: Vec<f32>,
	pub norm: Vec<f32>,
	/// Row scales of `fc` `[5120]` (its input: embedding, then hidden).
	pub s_fc: Vec<f32>,
	pub layer: AttnC,
}

/// Names of the MTP block's weight inputs (and files): `fc`, then the
/// attention layer's projection, o_proj, the gate / up blocks, down.
pub fn mtp_weight(i: usize) -> String {
	match i {
		0 => "w.mtp.proj".into(),
		1 => "w.mtp.out".into(),
		i if i < 2 + GUSPLIT => format!("w.mtp.gu{}", i - 2),
		_ => "w.mtp.down".into(),
	}
}
pub const MTP_FC: &str = "w.mtp.fc";

/// The MTP block over `s` positions with a window of `cap` keys: `hidden`
/// (the main model's output-normed hidden state at the previous position),
/// `emb` (the embedding of the token) [1, s, 5120], `attention_mask`
/// [1, cap + s] i64, `position_ids` [1, s] i64, `past_key` / `past_value`
/// [1, 4, cap, 256] -> `h` [1, s, 5120] (normed: the draft head's input),
/// `present_key`, `present_value` [1, 4, cap + s, 256].
pub fn mtp(s: usize, cap: usize, c: &MtpC) -> Graph {
	let mut gr = Graph::new();
	let g = &mut gr;
	let sg = signs(FF);
	let hid = g.parameter("hidden", Ty::F16, &[1, s, W]);
	let emb = g.parameter("emb", Ty::F16, &[1, s, W]);
	let pre = |g: &mut Graph, t: &V, w: &[f32]| {
		let t = g.convert(t, Ty::F32);
		let t = rms(g, &t, 1e-6);
		let wc = g.c32(&[1, 1, W], w);
		g.mul(&t, &wc)
	};
	let e = pre(g, &emb, &c.ne);
	let h = pre(g, &hid, &c.nh);
	let cat = g.concat(&[e, h], -1);
	let cr = rotate(g, &cat, &sg);
	let x0 = fc(g, &cr, &[(MTP_FC.into(), W)], 2 * W, &c.s_fc);
	let (y, outs) = attn_body(g, s, cap, Io::Mtp(&x0), &c.layer, &mtp_weight, true, &sg);
	let y = rms(g, &y, 1e-6);
	let nw = g.c32(&[1, 1, W], &c.norm);
	let y = g.mul(&y, &nw);
	let y = g.convert(&y, Ty::F16);
	g.result(&y, "h");
	for (t, n) in outs {
		g.result(&t, n);
	}
	gr
}
