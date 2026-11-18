//! Qwen3.8-Flash-Next's NPU prefill graphs over `s` positions. Every per-layer value is an input named after its
//! file (`fl{L}.<name>.in`, [`super::weights`]), so one blob serves every
//! layer of a kind:
//!
//! - [`delta_pre`]: hyper-connection, projections, conv, l2 norms, gates ->
//!   the recurrence's inputs (the 27B's `Cut::Pre` outputs, which
//!   `offload::DeltaScan` reads), `z`, the injection weights;
//! - [`attn_pre`]: hyper-connection, projections, head norms, RoPE, the
//!   indexer's queries / keys;
//! - [`post`]: gated norm (delta) or gate (attention), output projection,
//!   injection, the MoE's hyper-connection, router logits, shared expert;
//! - [`experts`]: `g` routed experts of `c` tokens gathered from the
//!   super-chunk's MoE input;
//! - [`ple`]: the PLE term of layer 1.

use crate::g::{Graph, Ty, V};
use npu::flash::{HC, HC_RANK, W};

use super::refm::{D, DA, FE, H, HCW, HK, HKV, HQ, ID, IH, NE, NQKV, RD};
use super::weights::rot_block;

/// Positions per prefill graph.
pub const S: usize = 256;

/// The code type of every quantized matrix but the routed experts' (int8
/// per row: int4 cost KL 0.122 of the 0.154 of v1 alone; the experts stay
/// int4 for RAM).
pub const DENSE: Ty = Ty::I8;

/// The layers' hyper-connection down / up matrices int8 per row, weight
/// only (half the bytes a decode step reads of them, against f16).
pub const HC_I8: bool = true;

/// The per-stream norm of the hyper-connections in our SHAVE kernel
/// (`shave/hcmix.sk` mode N, [`hc_in_k`]) on graphs of at least
/// [`HC_KERNEL_MIN`] positions: one pass where the compiler's fused RMS
/// norm and the weight's Multiply took two (HC 4.47 -> 3.92 ms at 256
/// positions, same output). Its mode M (the gated stream mean) is slower
/// than the compiler's ops and stays off.
pub const HC_KERNEL: bool = true;

/// The delta layers' gated norm in `post` in the same kernel (mode G) on
/// graphs of at least [`HC_KERNEL_MIN`] positions.
pub const GN_KERNEL: bool = true;

/// [`GN_KERNEL`] with two per-channel factors folded into the kernel's
/// `B` (`[48, 2, 128]` per head: `gnw * out.pre`, then `z.s`): `z` leaves
/// [`delta_pre`] without its row scale (the kernel takes `sigmoid(z *
/// z.s)`), and the out-projection's input prescale is no Multiply of its
/// own. Same graphs' predicate on both sides: [`gn_fold`].
pub const GN_FOLD: bool = true;

/// Whether the `s`-position delta graphs take [`GN_FOLD`] (`delta_pre`'s
/// `z` raw, `post`'s prescale in the gated norm).
pub fn gn_fold(s: usize) -> bool {
	GN_FOLD && GN_KERNEL && PRE_FOLD && s >= HC_KERNEL_MIN
}

/// Swish and the l2 norms of `q` / `k` in [`delta_pre`] /
/// [`delta_pre_m`] in our SHAVE kernel (`shave/l2n.sk`, carrier Mish), one
/// pass with an f32 sum of squares, where the stock ops took a Swish over
/// all of `qkv` and ~10 serial SHAVE / DPU ops per norm (`delta_pre_m`
/// 6.56 -> 5.46 ms at 256 positions; ids_long KL 0.0547 -> 0.0539).
pub const L2_KERNEL: bool = true;

/// Whether the `s`-position delta graphs take [`L2_KERNEL`] (the compiler
/// tiles the carrier over the positions).
pub fn l2_kernel(s: usize) -> bool {
	L2_KERNEL && s >= HC_KERNEL_MIN
}

/// Shortest graph that takes [`HC_KERNEL`] (the compiler tiles the carrier
/// over the positions; one-position graphs keep the stock ops).
pub const HC_KERNEL_MIN: usize = 64;

/// The injection matrix folded into the down matrix (int8 rows of
/// `{p}.dinj`) in the [`HC_KERNEL`] form.
pub const HC_FOLD_INJ: bool = true;

pub(crate) fn hadamard(n: usize) -> Vec<f64> {
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

/// `x R` over the last axis in Hadamard blocks of `b` (signs applied before).
pub(crate) fn rotate(g: &mut Graph, x: &V, b: usize) -> V {
	let t = g.reshape(x, &[-1, b as i64]);
	let hb = g.cf(x.ty, &[b, b], &hadamard(b));
	let t = g.matmul(&t, &hb, false, false);
	let shape: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
	g.reshape(&t, &shape)
}

/// `x / sqrt(mean(x^2) + eps)` over the last axis (the compiler's
/// FuseRMSNorm form).
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

fn p16(g: &mut Graph, name: &str, shape: &[usize]) -> V {
	g.parameter(name, Ty::F16, shape)
}

/// `rotate(x * pre)`: a quantized linear's input on the int8 grid's scale.
fn prerot(g: &mut Graph, x: &V, pre: &str, k: usize) -> V {
	let pre = p16(g, &format!("{pre}.pre"), &[k]);
	let t = g.mul(x, &pre);
	rotate(g, &t, rot_block(k))
}

/// A quantized linear group on `x` (`[.., k]`): `fq(rotate(x * pre))`
/// against each `(matrix, rows, type)`: `x W^T` per matrix.
fn qlin(g: &mut Graph, x: &V, pre: &str, k: usize, mats: &[(&str, usize, Ty)]) -> Vec<V> {
	let t = prerot(g, x, pre, k);
	qmats(g, &t, k, mats)
}

/// [`qlin`] on an input already rotated and prescaled.
fn qmats(g: &mut Graph, t: &V, k: usize, mats: &[(&str, usize, Ty)]) -> Vec<V> {
	let xq = g.fake_quantize(t, -128.0, 127.0, 256);
	mats.iter()
		.map(|&(m, n, ty)| {
			let y = qmat_raw(g, &xq, k, m, n, ty);
			let s = p16(g, &format!("{m}.s"), &[n]);
			g.mul(&y, &s)
		})
		.collect()
}

/// [`qmats`]' MatMul of one matrix on the quantized input `xq`, without
/// its row scale `{m}.s`.
fn qmat_raw(g: &mut Graph, xq: &V, k: usize, m: &str, n: usize, ty: Ty) -> V {
	let w = g.parameter(&format!("{m}.w"), ty, &[n, k]);
	// int8 codes x 2^-16, their row scales x 2^16 in the files
	// (`weights::Out::mat`): at int4's 2^-8 most int8 scales of small
	// matrices (the PLE key projection: 98 %) are f16 subnormals, which
	// the NPU flushes to zero.
	let c = g.convert(&w, Ty::F16);
	let sc = g.s16(2f64.powi(if ty == Ty::I8 { -16 } else { -8 }));
	let c = g.mul(&c, &sc);
	g.matmul(xq, &c, false, true)
}

/// The `HC` streams `x` `[1, s, HCW]` normed per stream with weight
/// `{p}.norm`.
fn norm_g(g: &mut Graph, x: &V, w: &str, s: usize) -> V {
	let t = g.reshape(x, &[1, s as i64, HC as i64, W as i64]);
	let t = rms(g, &t, 1e-6);
	let t = g.reshape(&t, &[1, s as i64, HCW as i64]);
	let w = p16(g, w, &[HCW]);
	g.mul(&t, &w)
}

/// A hyper-connection `{p}.*` on the streams: (the mixed input `[1, s, W]`,
/// the injection weights `[1, s, HC]`).
fn hc_in(g: &mut Graph, x: &V, p: &str, s: usize) -> (V, V) {
	if HC_KERNEL && HC_I8 && s >= HC_KERNEL_MIN {
		return hc_in_k(g, x, p, s, false, HC_FOLD_INJ);
	}
	hc_in_q(g, x, p, s, HC_I8)
}

/// `x W^T` of an f16 activation and the weight `{w}`: f16 `[n][k]`, or
/// with `i8` int8 codes `{w}.w` (x 2^-7 in the graph) and f16 row scales
/// `{w}.s` (weight only: the activation stays f16).
fn wlin(g: &mut Graph, x: &V, w: &str, n: usize, k: usize, i8: bool) -> V {
	if !i8 {
		let m = p16(g, w, &[n, k]);
		return g.matmul(x, &m, false, true);
	}
	let c = g.parameter(&format!("{w}.w"), Ty::I8, &[n, k]);
	let c = g.convert(&c, Ty::F16);
	let sc = g.s16(2f64.powi(-7));
	let c = g.mul(&c, &sc);
	let y = g.matmul(x, &c, false, true);
	let sr = p16(g, &format!("{w}.s"), &[n]);
	g.mul(&y, &sr)
}

/// [`hc_in`], the down / up matrices int8 with `i8`.
fn hc_in_q(g: &mut Graph, x: &V, p: &str, s: usize, i8: bool) -> (V, V) {
	let xn = norm_g(g, x, &format!("{p}.norm"), s);
	let inj = p16(g, &format!("{p}.inj"), &[HC, HCW]);
	let q = g.s16(1.0 / HC as f64);
	let m = wlin(g, &xn, &format!("{p}.down"), HC_RANK, HCW, i8);
	let m = g.mul(&m, &q);
	let m = g.swish1(&m);
	let m = wlin(g, &m, &format!("{p}.up"), HCW, HC_RANK, i8);
	let m = g.sigmoid(&m);
	let p = g.mul(&m, &xn);
	let p = g.reshape(&p, &[1, s as i64, HC as i64, W as i64]);
	let mixed = g.reduce_mean(&p, 2);
	let mixed = g.reshape(&mixed, &[1, s as i64, W as i64]);
	let j = g.matmul(&xn, &inj, false, true);
	let j = g.mul(&j, &q);
	let j = g.sigmoid(&j);
	let two = g.s16(2.0);
	(mixed, g.mul(&j, &two))
}

/// [`hc_in_q`] (int8) with its eltwise work in our SHAVE kernel
/// `shave/hcmix.sk` (carrier `Minimum`): mode N the per-stream norm, mode M
/// the gated mean of the streams with the up matrix's row scale.
/// With `fold` the injection matrix is int8 rows `[HC_RANK, HC_RANK + HC)`
/// of `{p}.dinj` (the down matrix's first `HC_RANK`): one MatMul instead of
/// the down one and a 4-row one.
fn hc_in_k(g: &mut Graph, x: &V, p: &str, s: usize, mk: bool, fold: bool) -> (V, V) {
	let si = s as i64;
	let sh5 = [1, si, HC as i64, (W / 128) as i64, 128];
	let x5 = g.reshape(x, &sh5);
	let w = p16(g, &format!("{p}.norm"), &[HCW]);
	let w5 = g.reshape(&w, &[1, 1, HC as i64, (W / 128) as i64, 128]);
	let xn5 = g.minimum(&x5, &w5);
	let xn = g.reshape(&xn5, &[1, si, HCW as i64]);
	let q = g.s16(1.0 / HC as f64);
	let (m, jf) = if fold {
		let t = wlin(g, &xn, &format!("{p}.dinj"), HC_RANK + HC, HCW, true);
		let t = g.mul(&t, &q);
		(
			g.slice(&t, 0, HC_RANK, 1, 2),
			Some(g.slice(&t, HC_RANK, HC_RANK + HC, 1, 2)),
		)
	} else {
		let m = wlin(g, &xn, &format!("{p}.down"), HC_RANK, HCW, true);
		(g.mul(&m, &q), None)
	};
	let m = g.swish1(&m);
	let mixed = if !mk {
		let m = wlin(g, &m, &format!("{p}.up"), HCW, HC_RANK, true);
		let m = g.sigmoid(&m);
		let pm = g.mul(&m, &xn);
		let pm = g.reshape(&pm, &[1, si, HC as i64, W as i64]);
		let t = g.reduce_mean(&pm, 2);
		g.reshape(&t, &[1, si, W as i64])
	} else {
		// the up matrix without its row scale (the kernel applies it)
		let c = g.parameter(&format!("{p}.up.w"), Ty::I8, &[HCW, HC_RANK]);
		let c = g.convert(&c, Ty::F16);
		let sc = g.s16(2f64.powi(-7));
		let c = g.mul(&c, &sc);
		let u = g.matmul(&m, &c, false, true);
		// [S, stream, block, pair, 128]: the pair inside the block, so the
		// compiler cannot merge the block with the 128 lanes (positions stay
		// the unique largest dimension it tiles)
		let sh6 = [1, si, HC as i64, (W / 128) as i64, 1, 128];
		let u6 = g.reshape(&u, &sh6);
		let xn6 = g.reshape(&xn, &sh6);
		let a = g.concat(&[u6, xn6], 4);
		let su = p16(g, &format!("{p}.up.s"), &[HCW]);
		let su6 = g.reshape(&su, &[1, 1, HC as i64, (W / 128) as i64, 1, 128]);
		let y = g.minimum(&a, &su6);
		let y = g.slice(&y, 0, 1, 1, 2);
		let y = g.slice(&y, 0, 1, 1, 4);
		g.reshape(&y, &[1, si, W as i64])
	};
	let j = match jf {
		Some(j) => j,
		None => {
			let inj = p16(g, &format!("{p}.inj"), &[HC, HCW]);
			let j = g.matmul(&xn, &inj, false, true);
			g.mul(&j, &q)
		}
	};
	let j = g.sigmoid(&j);
	let two = g.s16(2.0);
	(mixed, g.mul(&j, &two))
}

/// `x + y * inj` per stream: `x` `[1, s, HCW]`, `y` `[1, s, W]`, `inj`
/// `[1, s, HC]`.
fn inject(g: &mut Graph, x: &V, y: &V, inj: &V, s: usize) -> V {
	let xs = g.reshape(x, &[1, s as i64, HC as i64, W as i64]);
	if INJ_STREAMS && s >= HC_KERNEL_MIN {
		let si = s as i64;
		let y4 = g.reshape(y, &[1, si, 1, W as i64]);
		let parts: Vec<V> = (0..HC)
			.map(|st| {
				let j = g.slice(inj, st, st + 1, 1, 2);
				let j = g.reshape(&j, &[1, si, 1, 1]);
				let t = g.mul(&y4, &j);
				let xp = g.slice(&xs, st, st + 1, 1, 2);
				g.add(&xp, &t)
			})
			.collect();
		let t = g.concat(&parts, 2);
		return g.reshape(&t, &[1, si, HCW as i64]);
	}
	let yy = g.unsqueeze(y, 2);
	let jj = g.unsqueeze(inj, 3);
	let t = g.mul(&yy, &jj);
	let t = g.add(&xs, &t);
	g.reshape(&t, &[1, s as i64, HCW as i64])
}

/// l2-normalize the `D`-rows of `t` `[1, s, HK * D]` with a per-row max-abs
/// pre-scale (keeps the squares out of f16's subnormals; as the 27B's).
fn l2n(g: &mut Graph, t: &V, s: usize) -> V {
	let t = g.reshape(t, &[1, s as i64, HK as i64, D as i64]);
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
	let rt = g.sqrt(&ss);
	let one = g.s16(1.0);
	let r = g.div(&one, &rt);
	let y = g.mul(&ts, &r);
	g.reshape(&y, &[1, s as i64, (HK * D) as i64])
}

/// The prefill form of [`delta_pre`] on graphs of at least
/// [`HC_KERNEL_MIN`] positions: `qkv`'s row scale folded into the conv
/// weights (the conv is per channel; `cpast` is divided by it and the next
/// window multiplied back, both 3 rows): its Multiply was ~0.5 ms of serial
/// SHAVE work (`delta_pre_m` 6.7 -> 6.2 ms). The l2 norms of `q` and `k`
/// as one chain over their 32 heads was slower (6.5): those ops are bound
/// by their bytes, not their count.
pub const PRE_FOLD: bool = true;

/// [`inject`] on graphs of at least [`HC_KERNEL_MIN`] positions per
/// stream: `x_s + y * inj[s]` as a Multiply by one value per position (a
/// DPU scale-shift) and an Add per stream, concatenated, where the
/// broadcast `[s, 1, W] x [s, HC, 1]` Multiply ran on the SHAVEs
/// (`post_delta_a` 4.6 -> 4.05 ms; a K = 1 MatMul as the outer product:
/// 4.85).
pub const INJ_STREAMS: bool = true;

/// A delta-net layer up to its recurrence.
///
/// Inputs: `x` [1, s, 10240] (the streams), `cpast` [1, 3, 10240] (the last
/// three positions' qkv), `valid` [1, s, 1] (0 at right padding), `keep`
/// [3] i32 (rows of concat(cpast, qkv) that are the next window), then the
/// layer's files. Outputs: `qn`, `kn` [1, s, 2048], `v` [1, s, 6144], `gb`
/// [1, s, 96] (log decay, then beta), `z` [1, s, 6144], `conv_out`, `inj`.
pub fn delta_pre(s: usize) -> Graph {
	delta_pre_g(s, false)
}

/// [`delta_pre`] after its hyper-connection, which the iGPU runs
/// (`offload::HcPre`): inputs `mixed` [1, s, 2560] and `hinj` [1, s, 4] in
/// place of `x`; `hinj` comes back as the output `inj` (through the NPU's
/// queue, so it is written after the previous `post_delta` on the bank read
/// it).
pub fn delta_pre_m(s: usize) -> Graph {
	delta_pre_g(s, true)
}

/// [`delta_pre_m`] up to its projections (a prefill graph of at least
/// [`HC_KERNEL_MIN`] positions): the scan device runs the short conv,
/// Swish and the l2 norms of `q` / `k` before its recurrence
/// (`offload::DeltaScan::set_conv`), where they were ~1 ms of serial
/// SHAVE / DMA work of the NPU's ~5.5. Inputs `mixed` [1, s, 2560], `hinj`
/// [1, s, 4], `valid` [1, s, 1], then the layer's files (`proj.pre`,
/// `qkv.w` / `z.w` / `ab.w`, `ab.s`, `dtb`, `acoef`). Outputs: `qkv` [1, s,
/// 10240] the raw MatMul output (the conv applies `qkv.s`), `gb`, `z` (raw:
/// [`gn_fold`]), `inj`.
pub fn delta_pre_c(s: usize) -> Graph {
	assert!(PRE_FOLD && gn_fold(s), "delta_pre_c: a prefill graph");
	let mut g = Graph::new();
	let mixed = p16(&mut g, "mixed", &[1, s, W]);
	let hinj = p16(&mut g, "hinj", &[1, s, HC]);
	let valid = p16(&mut g, "valid", &[1, s, 1]);
	let one = g.s16(1.0);
	let inj = g.mul(&hinj, &one);
	let t = prerot(&mut g, &mixed, "proj", W);
	let xq = g.fake_quantize(&t, -128.0, 127.0, 256);
	let qkv = qmat_raw(&mut g, &xq, W, "qkv", NQKV, DENSE);
	let z = qmat_raw(&mut g, &xq, W, "z", H * D, DENSE);
	let ab = qmat_raw(&mut g, &xq, W, "ab", 2 * H, Ty::I8);
	let sc = p16(&mut g, "ab.s", &[2 * H]);
	let ab = g.mul(&ab, &sc);
	let pab = g.variadic_split(&ab, 2, &[H, H]);
	let dtb = p16(&mut g, "dtb", &[H]);
	let acoef = p16(&mut g, "acoef", &[H]);
	let t = g.add(&pab[0], &dtb);
	let t = g.softplus(&t);
	let t = g.mul(&t, &acoef);
	let g_tm = g.mul(&t, &valid);
	let t = g.sigmoid(&pab[1]);
	let b_tm = g.mul(&t, &valid);
	let gb = g.concat(&[g_tm, b_tm], 2);
	for (t, n) in [(&qkv, "qkv"), (&gb, "gb"), (&z, "z"), (&inj, "inj")] {
		g.result(t, n);
	}
	g
}

fn delta_pre_g(s: usize, m: bool) -> Graph {
	let mut g = Graph::new();
	let x = if m {
		(
			p16(&mut g, "mixed", &[1, s, W]),
			Some(p16(&mut g, "hinj", &[1, s, HC])),
		)
	} else {
		(p16(&mut g, "x", &[1, s, HCW]), None)
	};
	let cpast = p16(&mut g, "cpast", &[1, 3, NQKV]);
	let valid = p16(&mut g, "valid", &[1, s, 1]);
	let keep = g.parameter("keep", Ty::I32, &[3]);
	let (mixed, inj) = if let Some(hinj) = x.1 {
		let one = g.s16(1.0);
		(x.0, g.mul(&hinj, &one))
	} else {
		hc_in(&mut g, &x.0, "hca", s)
	};
	let fold = PRE_FOLD && s >= HC_KERNEL_MIN;
	let pr = if fold {
		let t = prerot(&mut g, &mixed, "proj", W);
		let xq = g.fake_quantize(&t, -128.0, 127.0, 256);
		let mut pr = vec![qmat_raw(&mut g, &xq, W, "qkv", NQKV, DENSE)];
		for (m, n, ty) in [("z", H * D, DENSE), ("ab", 2 * H, Ty::I8)] {
			let y = qmat_raw(&mut g, &xq, W, m, n, ty);
			if m == "z" && gn_fold(s) {
				// `post`'s gated norm applies `z.s`
				pr.push(y);
				continue;
			}
			let sc = p16(&mut g, &format!("{m}.s"), &[n]);
			pr.push(g.mul(&y, &sc));
		}
		pr
	} else {
		qlin(
			&mut g,
			&mixed,
			"proj",
			W,
			&[
				("qkv", NQKV, DENSE),
				("z", H * D, DENSE),
				("ab", 2 * H, Ty::I8),
			],
		)
	};
	let (qkv, z, ab) = (pr[0].clone(), pr[1].clone(), pr[2].clone());
	let pab = g.variadic_split(&ab, 2, &[H, H]);
	let cw = p16(&mut g, "conv", &[NQKV, 1, 1, 4]);
	let (ext, cw, qs) = if fold {
		// `qkv` is the raw MatMul output: `cpast` (scaled rows) on its scale,
		// the row scale in the conv weights
		let qs = p16(&mut g, "qkv.s", &[NQKV]);
		// all-zero rows have scale 0 (13 in layer 1): any finite divisor,
		// their conv weights are 0
		let lo = g.s16(2f64.powi(-14));
		let qd = g.maximum(&qs, &lo);
		let cp = g.div(&cpast, &qd);
		let qs4 = g.reshape(&qs, &[NQKV as i64, 1, 1, 1]);
		let cw = g.mul(&cw, &qs4);
		(g.concat(&[cp, qkv], 1), cw, Some(qs))
	} else {
		(g.concat(&[cpast, qkv], 1), cw, None)
	};
	let xt = g.transpose(&ext, &[0, 2, 1]);
	let cv = g.group_conv1d(&xt, &cw);
	let cv = g.transpose(&cv, &[0, 2, 1]);
	let l2k = l2_kernel(s);
	let qkvs = if l2k {
		// `shave/l2n.sk` (carrier Mish): Swish and the l2 norm of q and k
		// per head in one pass, A [1, s, 32, 128]; Swish of v alone
		let p = g.variadic_split(&cv, 2, &[2 * HK * D, H * D]);
		let v = g.swish1(&p[1]);
		let a = g.reshape(&p[0], &[1, s as i64, 2 * HK as i64, D as i64]);
		let y = g.mish(&a);
		let y = g.reshape(&y, &[1, s as i64, (2 * HK * D) as i64]);
		let qk = g.variadic_split(&y, 2, &[HK * D, HK * D]);
		vec![qk[0].clone(), qk[1].clone(), v]
	} else {
		let cv = g.swish1(&cv);
		g.variadic_split(&cv, 2, &[HK * D, HK * D, H * D])
	};
	let dtb = p16(&mut g, "dtb", &[H]);
	let acoef = p16(&mut g, "acoef", &[H]);
	let t = g.add(&pab[0], &dtb);
	let t = g.softplus(&t);
	let t = g.mul(&t, &acoef);
	let g_tm = g.mul(&t, &valid);
	let t = g.sigmoid(&pab[1]);
	let b_tm = g.mul(&t, &valid);
	let (qn, kn) = if l2k {
		(qkvs[0].clone(), qkvs[1].clone())
	} else {
		(l2n(&mut g, &qkvs[0], s), l2n(&mut g, &qkvs[1], s))
	};
	let gb = g.concat(&[g_tm, b_tm], 2);
	let cnext = g.gather(&ext, &keep, 1);
	let cnext = match &qs {
		Some(qs) => g.mul(&cnext, qs),
		None => cnext,
	};
	for (t, n) in [
		(&qn, "qn"),
		(&kn, "kn"),
		(&qkvs[2], "v"),
		(&gb, "gb"),
		(&z, "z"),
		(&cnext, "conv_out"),
		(&inj, "inj"),
	] {
		g.result(t, n);
	}
	g
}

/// Rotate the first [`RD`] channels of the last axis of `t` (`[1, s, n,
/// d]`, or `[1, n, d]` at one position; rotate_half pairs) with `cos` /
/// `sin` (`[1, s, 1, RD]` / `[1, 1, RD]`).
fn rope(g: &mut Graph, t: &V, cos: &V, sin: &V) -> V {
	let ax = t.shape.len() - 1;
	let d = t.shape[ax];
	let rp = g.variadic_split(t, ax as i64, &[RD, d - RD]);
	let r12 = g.variadic_split(&rp[0], ax as i64, &[RD / 2, RD / 2]);
	let neg = g.negative(&r12[1]);
	let rot = g.concat(&[neg, r12[0].clone()], ax as i64);
	let a = g.mul(&rp[0], cos);
	let b = g.mul(&rot, sin);
	let ab = g.add(&a, &b);
	g.concat(&[ab, rp[1].clone()], ax as i64)
}

/// Head norm (weight `w`) and RoPE of `t` (`[1, s, n, d]` or `[1, s, n *
/// d]`) with `cs` / `sn` from [`attn_pre`]. At one position on `[1, n, d]`: the NPU compile of the 4-D form
/// at s = 1 gets the 24- and 4-head tensors wrong (relative error ~1; the
/// 2-head keys right; `nbcheck flash-npucheck`).
fn norm_rope(g: &mut Graph, t: &V, w: &str, s: usize, n: usize, d: usize, cs: &V, sn: &V) -> V {
	let (si, ni, di) = (s as i64, n as i64, d as i64);
	if s == 1 {
		let t = g.reshape(t, &[1, ni, di]);
		let t = headnorm(g, &t, w, d);
		let t = rope(g, &t, cs, sn);
		g.reshape(&t, &[1, 1, ni, di])
	} else {
		let t = if t.shape.len() == 4 {
			t.clone()
		} else {
			g.reshape(t, &[1, si, ni, di])
		};
		let t = headnorm(g, &t, w, d);
		rope(g, &t, cs, sn)
	}
}

fn headnorm(g: &mut Graph, t: &V, w: &str, d: usize) -> V {
	let y = rms(g, t, 1e-6);
	let w = p16(g, w, &[d]);
	g.mul(&y, &w)
}

/// An attention layer up to its attention.
///
/// Inputs: `x` [1, s, 10240], `cos` / `sin` [1, s, 64] (per position, from
/// the host), then the layer's files. Outputs: `q` [1, s, 24, 256] (normed,
/// roped), `gate` [1, s, 6144] (sigmoid applied), `k` / `v` [1, s, 2, 256]
/// (k normed, roped), the indexer's `iq` [1, s, 4, 128] (normed, roped) and
/// raw keys `ik` [1, s, 128], `inj`.
pub fn attn_pre(s: usize) -> Graph {
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, s, HCW]);
	let cos = p16(&mut g, "cos", &[1, s, RD]);
	let sin = p16(&mut g, "sin", &[1, s, RD]);
	let (mixed, inj) = hc_in(&mut g, &x, "hca", s);
	let pr = qlin(
		&mut g,
		&mixed,
		"proj",
		W,
		&[
			("q", 2 * HQ * DA, DENSE),
			("k", HKV * DA, DENSE),
			("v", HKV * DA, DENSE),
			("iqk", (IH + 1) * ID, DENSE),
		],
	);
	let (cs, sn) = if s == 1 {
		(cos, sin)
	} else {
		(g.unsqueeze(&cos, 2), g.unsqueeze(&sin, 2))
	};
	let si = s as i64;
	let qg = g.reshape(&pr[0], &[1, si, HQ as i64, 2 * DA as i64]);
	let qg = g.variadic_split(&qg, 3, &[DA, DA]);
	let q = norm_rope(&mut g, &qg[0], "qn", s, HQ, DA, &cs, &sn);
	let gate = g.reshape(&qg[1], &[1, si, (HQ * DA) as i64]);
	let gate = g.sigmoid(&gate);
	let k = norm_rope(&mut g, &pr[1], "kn", s, HKV, DA, &cs, &sn);
	let v = g.reshape(&pr[2], &[1, si, HKV as i64, DA as i64]);
	let iqk = g.variadic_split(&pr[3], 2, &[IH * ID, ID]);
	let iq = norm_rope(&mut g, &iqk[0], "iqn", s, IH, ID, &cs, &sn);
	for (t, n) in [
		(&q, "q"),
		(&gate, "gate"),
		(&k, "k"),
		(&v, "v"),
		(&iq, "iq"),
		(&iqk[1], "ik"),
		(&inj, "inj"),
	] {
		g.result(t, n);
	}
	g
}

/// After the mixer: the output projection, its injection, the MoE's
/// hyper-connection, router logits and shared expert.
///
/// Inputs: `x` [1, s, 10240], `o` [1, s, 6144], `zg` [1, s, 6144] (delta:
/// the gate input `z`, attention: the sigmoid gate), `inj` [1, s, 4], then
/// the layer's files. Delta layers: `o` is the recurrence's output times
/// `oscale` (the gated norm absorbs it). Outputs: `x1` (the streams after
/// the mixer), `moe_q` [1, s, 2560] (the MoE input prescaled and rotated:
/// what every expert's gate_up quantizes), `inj_m` [1, s, 4], `logits` [1,
/// s, 512], `shared` [1, s, 2560] (gated).
pub fn post(s: usize, delta: bool, oscale: f64) -> Graph {
	let mut g = Graph::new();
	let x1 = post_x1(&mut g, s, delta, oscale);
	let (m, inj_m) = hc_in(&mut g, &x1, "hcm", s);
	let [mq, logits, shared] = post_moe(&mut g, &m);
	for (t, n) in [
		(&x1, "x1"),
		(&mq, "moe_q"),
		(&inj_m, "inj_m"),
		(&logits, "logits"),
		(&shared, "shared"),
	] {
		g.result(t, n);
	}
	g
}

/// A delta layer's [`post`] up to the injection of the mixer's output
/// (output `x1`): the MoE's hyper-connection runs on the iGPU
/// (`offload::HcPre`) and [`post_delta_b`] after it.
pub fn post_delta_a(s: usize) -> Graph {
	let mut g = Graph::new();
	let x1 = post_x1(&mut g, s, true, npu::flash::OSCALE_P as f64);
	g.result(&x1, "x1");
	g
}

/// [`post_delta_a`] from the gated norm's output: `o` [1, s, 6144] is
/// already normed, gated, prescaled and rotated (the scan device's
/// `gn_npu.glsl` right after the recurrence, in place of its `o`), so the
/// graph is the out-projection and the injection (output `x1`).
pub fn post_delta_t(s: usize) -> Graph {
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, s, HCW]);
	let t = p16(&mut g, "o", &[1, s, H * D]);
	let inj = p16(&mut g, "inj", &[1, s, HC]);
	let x1 = inject_out(&mut g, &x, &t, &inj, s);
	g.result(&x1, "x1");
	g
}

/// The rest of a delta layer's [`post`] from the MoE hyper-connection's
/// mixed input `m` [1, s, 2560]: outputs `moe_q`, `logits`, `shared`.
pub fn post_delta_b(s: usize) -> Graph {
	let mut g = Graph::new();
	let m = p16(&mut g, "m", &[1, s, W]);
	let outs = post_moe(&mut g, &m);
	for (t, n) in outs.iter().zip(["moe_q", "logits", "shared"]) {
		g.result(t, n);
	}
	g
}

/// [`post`]'s router logits, MoE input and shared expert from the mixed
/// input `m`: `[moe_q, logits, shared]`.
fn post_moe(g: &mut Graph, m: &V) -> [V; 3] {
	let router = p16(g, "router", &[NE, W]);
	let logits = g.matmul(m, &router, false, true);
	let mq = prerot(g, m, "moe", W);
	let gu = qmats(g, &mq, W, &[("sgu", 2 * FE, DENSE)]).remove(0);
	let gu = g.variadic_split(&gu, 2, &[FE, FE]);
	let sw = g.swish1(&gu[0]);
	let h = g.mul(&sw, &gu[1]);
	let sd = qlin(g, &h, "sd", FE, &[("sd", W, DENSE)]).remove(0);
	let sgate = p16(g, "sgate", &[1, W]);
	let gt = g.matmul(m, &sgate, false, true);
	let gt = g.sigmoid(&gt);
	let shared = g.mul(&sd, &gt);
	[mq, logits, shared]
}

/// [`post`] up to `x1`: gated norm (delta) or gate (attention), output
/// projection, injection into the streams `x`.
fn post_x1(g: &mut Graph, s: usize, delta: bool, oscale: f64) -> V {
	let si = s as i64;
	let x = p16(g, "x", &[1, s, HCW]);
	let o = p16(g, "o", &[1, s, H * D]);
	let zg = p16(g, "zg", &[1, s, H * D]);
	let inj = p16(g, "inj", &[1, s, HC]);
	let a = if delta && GN_KERNEL && s >= HC_KERNEL_MIN {
		// `shave/hcmix.sk` mode G: A [1, s, 48, 2, 128] = (o, z) per head,
		// B = gw; its eps is 1e-6 x GS^2 (the kernel's define)
		assert_eq!(
			oscale,
			npu::flash::OSCALE_P as f64,
			"hcmix.sk is built for -DGS=OSCALE_P"
		);
		let o4 = g.reshape(&o, &[1, si, H as i64, 1, D as i64]);
		let z4 = g.reshape(&zg, &[1, si, H as i64, 1, D as i64]);
		let a = g.concat(&[o4, z4], 3);
		let gw = p16(g, "gnw", &[D]);
		let gw = g.reshape(&gw, &[1, 1, 1, 1, D as i64]);
		let b = if gn_fold(s) {
			// B [1, 1, 48, 2, 128]: per head gnw * out.pre, then z.s
			let hd = [1, 1, H as i64, 1, D as i64];
			let op = p16(g, "out.pre", &[H * D]);
			let op = g.reshape(&op, &hd);
			let wp = g.mul(&op, &gw);
			let zs = p16(g, "z.s", &[H * D]);
			let zs = g.reshape(&zs, &hd);
			g.concat(&[wp, zs], 3)
		} else {
			gw
		};
		let y = g.minimum(&a, &b);
		let y = g.slice(&y, 0, 1, 1, 3);
		let y = g.reshape(&y, &[1, si, (H * D) as i64]);
		if gn_fold(s) {
			let t = rotate(g, &y, rot_block(H * D));
			return inject_out(g, &x, &t, &inj, s);
		}
		y
	} else if delta {
		let o4 = g.reshape(&o, &[1, si, H as i64, D as i64]);
		let n = rms(g, &o4, 1e-6 * oscale * oscale);
		let gw = p16(g, "gnw", &[D]);
		let n = g.mul(&n, &gw);
		let z4 = g.reshape(&zg, &[1, si, H as i64, D as i64]);
		let sz = g.sigmoid(&z4);
		let n = g.mul(&n, &sz);
		g.reshape(&n, &[1, si, (H * D) as i64])
	} else {
		g.mul(&o, &zg)
	};
	let y = qlin(g, &a, "out", H * D, &[("o", W, DENSE)]).remove(0);
	inject(g, &x, &y, &inj, s)
}

/// [`post_x1`]'s out-projection on its input already prescaled and
/// rotated (`t`), then the injection into the streams `x`.
fn inject_out(g: &mut Graph, x: &V, t: &V, inj: &V, s: usize) -> V {
	let y = qmats(g, t, H * D, &[("o", W, DENSE)]).remove(0);
	inject(g, x, &y, inj, s)
}

/// `ng` routed experts of `c` tokens each: per expert its tokens' MoE
/// inputs `x{e}` [c, 2560] (as `post` leaves them in `moe_q`: prescaled,
/// rotated; gathered on the host, so the graph does not depend on how many
/// positions an expert pass covers; an in-graph gather from 2048 rows
/// failed at append with 0x70000002).
///
/// Inputs: per expert `x{e}`, `gu{e}.w` int4 [1280, 2560] (gate
/// rows, then up rows), `gu{e}.s` [1280], `d{e}.pre` [640] (the down input's
/// signs and range are the expert's own), `d{e}.w` int4 [2560, 640],
/// `d{e}.s` [2560]. Output `y` [ng, c, 2560].
pub fn experts(ng: usize, c: usize) -> Graph {
	let mut g = Graph::new();
	// the inputs, in the engine's order: per expert x, gu.w, gu.s, d.pre,
	// d.w, d.s
	let mut ins = Vec::with_capacity(ng);
	for e in 0..ng {
		let idx = p16(&mut g, &format!("x{e}"), &[c, W]);
		let guw = g.parameter(&format!("gu{e}.w"), Ty::I4, &[2 * FE, W]);
		let gus = p16(&mut g, &format!("gu{e}.s"), &[2 * FE]);
		let dpre = p16(&mut g, &format!("d{e}.pre"), &[FE]);
		let dw = g.parameter(&format!("d{e}.w"), Ty::I4, &[W, FE]);
		let ds = p16(&mut g, &format!("d{e}.s"), &[W]);
		ins.push([idx, guw, gus, dpre, dw, ds]);
	}
	// the experts' element-wise work runs once over all `ng c` rows (a
	// SHAVE / DMA task per op, not per op and expert); only the matmuls
	// are per expert
	let cat = |g: &mut Graph, k: usize, sh: &[i64]| -> V {
		let vs: Vec<V> = ins.iter().map(|i| g.reshape(&i[k], sh)).collect();
		if ng > 1 {
			g.concat(&vs, 0)
		} else {
			vs[0].clone()
		}
	};
	// fake-quantized per expert (one quantize over all rows, then a split,
	// fails in the compiler: the quantize moves through the split and its
	// u8 / i8 types disagree)
	let xs: Vec<V> = ins
		.iter()
		.map(|i| g.fake_quantize(&i[0], -128.0, 127.0, 256))
		.collect();
	let mm = |g: &mut Graph, xq: &V, wp: &V| {
		let cv = g.convert(wp, Ty::F16);
		let sc = g.s16(2f64.powi(-8));
		let cv = g.mul(&cv, &sc);
		g.matmul(xq, &cv, false, true)
	};
	let hs: Vec<V> = (0..ng)
		.map(|e| {
			let h = mm(&mut g, &xs[e], &ins[e][1]);
			g.unsqueeze(&h, 0)
		})
		.collect();
	let h = if ng > 1 {
		g.concat(&hs, 0)
	} else {
		hs[0].clone()
	};
	let gus = cat(&mut g, 2, &[1, 1, (2 * FE) as i64]);
	let h = g.mul(&h, &gus);
	let h = g.variadic_split(&h, 2, &[FE, FE]);
	let sw = g.swish1(&h[0]);
	let h = g.mul(&sw, &h[1]);
	let dpre = cat(&mut g, 3, &[1, 1, FE as i64]);
	let tt = g.mul(&h, &dpre);
	// rotated and fake-quantized per expert (as the input: a quantize
	// before the split fails to compile)
	let hqs = if ng > 1 {
		g.variadic_split(&tt, 0, &vec![1; ng])
	} else {
		vec![tt]
	};
	let ys: Vec<V> = (0..ng)
		.map(|e| {
			let he = g.reshape(&hqs[e], &[c as i64, FE as i64]);
			let he = rotate(&mut g, &he, rot_block(FE));
			let he = g.fake_quantize(&he, -128.0, 127.0, 256);
			let y = mm(&mut g, &he, &ins[e][4]);
			g.unsqueeze(&y, 0)
		})
		.collect();
	let y = if ng > 1 {
		g.concat(&ys, 0)
	} else {
		ys[0].clone()
	};
	let ds = cat(&mut g, 5, &[1, 1, W as i64]);
	let y = g.mul(&y, &ds);
	g.result(&y, "y");
	g
}

/// [`experts`] with every op per expert and one output `y{e}` [c, 2560]
/// per expert (the engine binds them at `e c` rows of the job's output):
/// no concat / split over the job's rows.
pub fn experts_pe(ng: usize, c: usize) -> Graph {
	experts_pe_s(ng, c, false)
}

/// [`experts_pe`]; `ws`: fewer SHAVE Multiplies per expert: the down
/// input's range folded into the up rows' scale (input `gu{e}.s2`: gate
/// rows `gu.s`, up rows `gu.s d.pre`, made by the engine; no `d{e}.pre`)
/// and the output without its row scale `d.s` (the engine applies it in
/// its gather). Row scales on the weights instead (`codes 2^-8 s`) run
/// on the SHAVEs over the whole weight when `s` is an input: 220 ms a job.
pub fn experts_pe_s(ng: usize, c: usize, ws: bool) -> Graph {
	if ws {
		return experts_ws(ng, c);
	}
	experts_pe0(ng, c)
}

fn experts_ws(ng: usize, c: usize) -> Graph {
	let mut g = Graph::new();
	for e in 0..ng {
		let x = p16(&mut g, &format!("x{e}"), &[c, W]);
		let guw = g.parameter(&format!("gu{e}.w"), Ty::I4, &[2 * FE, W]);
		let gus = p16(&mut g, &format!("gu{e}.s2"), &[2 * FE]);
		let dw = g.parameter(&format!("d{e}.w"), Ty::I4, &[W, FE]);
		let mm = |g: &mut Graph, xq: &V, wp: &V| {
			let cv = g.convert(wp, Ty::F16);
			let sc = g.s16(2f64.powi(-8));
			let cv = g.mul(&cv, &sc);
			g.matmul(xq, &cv, false, true)
		};
		let xq = g.fake_quantize(&x, -128.0, 127.0, 256);
		let h = mm(&mut g, &xq, &guw);
		let h = g.mul(&h, &gus);
		let h = g.variadic_split(&h, 1, &[FE, FE]);
		let sw = g.swish1(&h[0]);
		let h = g.mul(&sw, &h[1]);
		let h = rotate(&mut g, &h, rot_block(FE));
		let h = g.fake_quantize(&h, -128.0, 127.0, 256);
		let y = mm(&mut g, &h, &dw);
		g.result(&y, &format!("y{e}"));
	}
	g
}

fn experts_pe0(ng: usize, c: usize) -> Graph {
	let mut g = Graph::new();
	for e in 0..ng {
		let x = p16(&mut g, &format!("x{e}"), &[c, W]);
		let guw = g.parameter(&format!("gu{e}.w"), Ty::I4, &[2 * FE, W]);
		let gus = p16(&mut g, &format!("gu{e}.s"), &[2 * FE]);
		let dpre = p16(&mut g, &format!("d{e}.pre"), &[FE]);
		let dw = g.parameter(&format!("d{e}.w"), Ty::I4, &[W, FE]);
		let ds = p16(&mut g, &format!("d{e}.s"), &[W]);
		let mm = |g: &mut Graph, xq: &V, wp: &V| {
			let cv = g.convert(wp, Ty::F16);
			let sc = g.s16(2f64.powi(-8));
			let cv = g.mul(&cv, &sc);
			g.matmul(xq, &cv, false, true)
		};
		let xq = g.fake_quantize(&x, -128.0, 127.0, 256);
		let h = mm(&mut g, &xq, &guw);
		let h = g.mul(&h, &gus);
		let h = g.variadic_split(&h, 1, &[FE, FE]);
		let sw = g.swish1(&h[0]);
		let h = g.mul(&sw, &h[1]);
		let h = g.mul(&h, &dpre);
		let h = rotate(&mut g, &h, rot_block(FE));
		let h = g.fake_quantize(&h, -128.0, 127.0, 256);
		let y = mm(&mut g, &h, &dw);
		let y = g.mul(&y, &ds);
		g.result(&y, &format!("y{e}"));
	}
	g
}

/// The PLE term of layer 1, added to the streams.
///
/// Inputs: `x` [1, s, 10240], `e` [1, s, 2560] (the tokens' n-gram
/// embeddings), `cstate` [1, 9, 10240] (the conv's last nine normed rows),
/// `keep` [9] i32 (rows of concat(cstate, rows) that are the next window),
/// then the layer's files. Outputs: `xo` (`x` with the term added),
/// `cstate_out`.
pub fn ple(s: usize) -> Graph {
	let mut g = Graph::new();
	let si = s as i64;
	let back = 3 * npu::flash::ngram::N;
	let x = p16(&mut g, "x", &[1, s, HCW]);
	let e = p16(&mut g, "e", &[1, s, W]);
	let cst = p16(&mut g, "cstate", &[1, back, HCW]);
	let keep = g.parameter("keep", Ty::I32, &[back]);
	let kv = qlin(
		&mut g,
		&e,
		"ple",
		W,
		&[("pk", HCW, DENSE), ("pv", W, DENSE)],
	);
	let key = norm_g(&mut g, &kv[0], "nkey", s);
	let q = norm_g(&mut g, &x, "nquery", s);
	let k4 = g.reshape(&key, &[1, si, HC as i64, W as i64]);
	let q4 = g.reshape(&q, &[1, si, HC as i64, W as i64]);
	let kq = g.mul(&k4, &q4);
	let gt = g.reduce_sum(&kq, -1);
	let sc = g.s16(1.0 / (W as f64).sqrt());
	let gt = g.mul(&gt, &sc);
	// signed square root: g / sqrt(max(|g|, 1e-6))
	let a = g.abs(&gt);
	let lo = g.s16(1e-6);
	let a = g.maximum(&a, &lo);
	let a = g.sqrt(&a);
	let gt = g.div(&gt, &a);
	let gt = g.sigmoid(&gt);
	let vv = g.unsqueeze(&kv[1], 2);
	let gv = g.mul(&gt, &vv);
	let gv = g.reshape(&gv, &[1, si, HCW as i64]);
	let gn = norm_g(&mut g, &gv, "nconv", s);
	// causal depthwise conv, kernel 4, dilation N: out[t] = sum_j w_j
	// gn[t - (3 - j) N] over concat(cstate, gn)
	let ext = g.concat(&[cst, gn], 1);
	let w = p16(&mut g, "pconv", &[HCW, 4]);
	let n = npu::flash::ngram::N;
	let mut acc: Option<V> = None;
	for j in 0..4 {
		let tap = g.slice(&ext, j * n, j * n + s, 1, 1);
		let wj = g.slice(&w, j, j + 1, 1, 1);
		let wj = g.reshape(&wj, &[HCW as i64]);
		let t = g.mul(&tap, &wj);
		acc = Some(match acc {
			None => t,
			Some(a) => g.add(&a, &t),
		});
	}
	let cv = g.swish1(&acc.unwrap());
	let t = g.add(&gv, &cv);
	let y = g.add(&x, &t);
	let cnext = g.gather(&ext, &keep, 1);
	g.result(&y, "xo");
	g.result(&cnext, "cstate_out");
	g
}

/// One hyper-connection alone (`x` -> `mixed`, `inj`): a benchmark of its
/// share of the layer graphs.
pub fn hc_only(s: usize) -> Graph {
	hc_only_q(s, false)
}

/// [`hc_only`], the matrices int8 with `i8`.
pub fn hc_only_q(s: usize, i8: bool) -> Graph {
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, s, HCW]);
	let (m, j) = hc_in_q(&mut g, &x, "hca", s, i8);
	g.result(&m, "mixed");
	g.result(&j, "inj");
	g
}

/// [`hc_only_q`] (int8) in the [`HC_KERNEL`] form; `mk` also puts the
/// gated stream mean in the kernel (mode M: slower than the compiler's
/// ops, 3.7 vs 3.2 ms of a profiled 256-position graph).
pub fn hc_only_k(s: usize, mk: bool, fold: bool) -> Graph {
	let mut g = Graph::new();
	let x = p16(&mut g, "x", &[1, s, HCW]);
	let (m, j) = hc_in_k(&mut g, &x, "hca", s, mk, fold);
	g.result(&m, "mixed");
	g.result(&j, "inj");
	g
}

/// Pooled indexer keys per [`idx_score`] job.
pub const IDX_T: usize = 4096;

/// The block indexer's scores of `s` queries against [`IDX_T`] pooled
/// keys (no weights): a prefill attention layer's blocks before its
/// sub-chunk, on the NPU right after `attn_pre` (whose `iq` it reads),
/// where the host's scoring grew to ~8.5 ms per sub-chunk at 72k
/// positions.
///
/// Inputs: `iq` [1, 4 s, 128] (`attn_pre`'s output, per position its 4
/// heads), `kidx` [1, IDX_T, 128] (the pooled, normed, roped block keys).
/// Output `sc` [1, s, IDX_T] f32: per position and block the sum over the
/// heads of `max(q . k, 0)`.
pub fn idx_score(s: usize) -> Graph {
	let mut g = Graph::new();
	let q = p16(&mut g, "iq", &[1, IH * s, ID]);
	let k = p16(&mut g, "kidx", &[1, IDX_T, ID]);
	let sc = g.matmul(&q, &k, false, true);
	let z = g.s16(0.0);
	let sc = g.maximum(&sc, &z);
	let sc = g.convert(&sc, Ty::F32);
	let sc = g.reshape(&sc, &[1, s as i64, IH as i64, IDX_T as i64]);
	let sc = g.reduce_sum(&sc, 2);
	let sc = g.reshape(&sc, &[1, s as i64, IDX_T as i64]);
	g.result(&sc, "sc");
	g
}

/// Key slots per [`attn_far`] block: [`FAR_KEYS`] keys, two empty slots
/// and the two the `softmax_lse` kernel (`-DKL=4`) leaves the row's max and
/// sum in (their K / V rows zero): 255 whole blocks of 4 keys.
pub const FAR_CAP: usize = 1024;
/// Keys per [`attn_far`] block.
pub const FAR_KEYS: usize = FAR_CAP - 4;

/// The attention of `s` queries over one block of [`FAR_CAP`] past key
/// slots (no weights): a prefill attention layer's keys before its
/// sub-chunk, on the NPU while the iGPU runs the rest. The Softmax is `softmax_lse`
/// with `-DNONORM`: `o` is P V undivided, `lse` [.., 2] the row's (max,
/// sum); the host merges the blocks by log sum exp.
///
/// Inputs: `qg` [1, 2, 12 s, 256] (per KV head, per position its 12 query
/// heads: token major, prescaled by 1 / 16), `kblk` / `vblk` [1, FAR_CAP,
/// 2, 256] (token major), with `mask` the additive [1, 1, s, 1, FAR_CAP]
/// (the indexer's selection, broadcast over the 12 heads; the dense form
/// has none). Outputs `o` [1, 2, 12 s,
/// 256], `lse` [1, 2, 12 s, 2].
pub fn attn_far(s: usize, mask: bool) -> Graph {
	let mut g = Graph::new();
	let gq = HQ / HKV;
	let qg = p16(&mut g, "qg", &[1, HKV, gq * s, DA]);
	let blk = [1, FAR_CAP, HKV, DA];
	let kb = p16(&mut g, "kblk", &blk);
	let vb = p16(&mut g, "vblk", &blk);
	let k = g.transpose(&kb, &[0, 2, 1, 3]);
	let mut sc = g.matmul(&qg, &k, false, true);
	if mask {
		let m = p16(&mut g, "mask", &[1, 1, s, 1, FAR_CAP]);
		let t = g.reshape(&sc, &[1, HKV as i64, s as i64, gq as i64, FAR_CAP as i64]);
		let t = g.add(&t, &m);
		sc = g.reshape(&t, &[1, HKV as i64, (gq * s) as i64, FAR_CAP as i64]);
	}
	let p = g.softmax(&sc, 3);
	let v = g.transpose(&vb, &[0, 2, 1, 3]);
	let o = g.matmul(&p, &v, false, false);
	g.result(&o, "o");
	let l = g.variadic_split(&p, 3, &[FAR_CAP - 2, 2]).remove(1);
	g.result(&l, "lse");
	g
}
