//! Row-wise elementwise pieces of the three models, f32, on the pool.

use quant::par_for;

#[derive(Clone, Copy)]
pub(crate) struct Ptr(pub *mut f32);
// SAFETY: every user writes disjoint rows / chunks of the buffer.
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// `f(row)` for each `w`-wide row of `x`, in parallel.
pub fn rows_mut(x: &mut [f32], w: usize, f: impl Fn(&mut [f32]) + Sync + Send) {
	let n = x.len() / w;
	let p = Ptr(x.as_mut_ptr());
	par_for(n, move |r| {
		let p = p;
		// SAFETY: task r owns row r
		f(unsafe { std::slice::from_raw_parts_mut(p.0.add(r * w), w) })
	});
}

/// `f(i, chunk)` over chunks of `c` elements, in parallel.
pub fn chunks_mut(x: &mut [f32], c: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
	let n = x.len();
	let p = Ptr(x.as_mut_ptr());
	par_for(n.div_ceil(c), move |i| {
		let p = p;
		let (a, b) = (i * c, ((i + 1) * c).min(n));
		// SAFETY: task i owns [a, b)
		f(a, unsafe {
			std::slice::from_raw_parts_mut(p.0.add(a), b - a)
		})
	});
}

/// RMSNorm of one row: `x / rms(x) * w` (`w + 1` when `plus_one`, the
/// zero-centered form). Sums in f64 like an f32 reference summing pairwise.
pub fn rms_row(x: &mut [f32], w: &[f32], eps: f32, plus_one: bool) {
	let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
	let r = 1.0 / (ms as f32 + eps).sqrt();
	let o = if plus_one { 1.0 } else { 0.0 };
	for (v, &g) in x.iter_mut().zip(w) {
		*v = *v * r * (g + o);
	}
}

/// RMSNorm over every row of `x`.
pub fn rms(x: &mut [f32], w: &[f32], eps: f32, plus_one: bool) {
	let n = w.len();
	rows_mut(x, n, |r| rms_row(r, w, eps, plus_one));
}

/// LayerNorm without affine of one row, then `* (1 + scale)`.
pub fn ln_scale_row(x: &[f32], scale: &[f32], eps: f32, out: &mut [f32]) {
	let n = x.len() as f64;
	let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
	let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
	let r = 1.0 / (var as f32 + eps).sqrt();
	let m = mean as f32;
	for ((o, &v), &s) in out.iter_mut().zip(x).zip(scale) {
		*o = (v - m) * r * (1.0 + s);
	}
}

/// [`ln_scale_row`] over every row.
pub fn ln_scale(x: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
	let n = scale.len();
	let mut out = quant::uninit_vec(x.len());
	let p = Ptr(out.as_mut_ptr());
	par_for(x.len() / n, move |r| {
		let p = p;
		// SAFETY: task r owns row r of out
		let o = unsafe { std::slice::from_raw_parts_mut(p.0.add(r * n), n) };
		ln_scale_row(&x[r * n..(r + 1) * n], scale, eps, o);
	});
	out
}

#[inline]
pub fn silu(v: f32) -> f32 {
	v / (1.0 + (-v).exp())
}

#[inline]
pub fn gelu_tanh(v: f32) -> f32 {
	const C: f32 = 0.797_884_6; // sqrt(2 / pi)
	0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh())
}

/// `x[i] = act(x[i])` in parallel.
pub fn map(x: &mut [f32], act: fn(f32) -> f32) {
	chunks_mut(x, 1 << 16, |_, c| c.iter_mut().for_each(|v| *v = act(*v)));
}

/// `x += g * y` per row (`g` one row of gates).
pub fn add_gated(x: &mut [f32], y: &[f32], g: &[f32]) {
	let n = g.len();
	assert_eq!(x.len(), y.len());
	let p = Ptr(x.as_mut_ptr());
	par_for(y.len() / n, move |r| {
		let p = p;
		// SAFETY: task r owns row r of x
		let xr = unsafe { std::slice::from_raw_parts_mut(p.0.add(r * n), n) };
		for ((a, &b), &gg) in xr.iter_mut().zip(&y[r * n..(r + 1) * n]).zip(g) {
			*a += gg * b;
		}
	});
}

/// `y[s][n] = x[s][k] w[n][k]^T` (f32, the cores).
pub fn linear(x: &[f32], s: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
	npubuild::la::linear(x, s, k, w, n)
}

/// Weight format of the resident (fast) backend: `QWEN_IMAGE_WEIGHTS` =
/// `q6k` (default), `q4k`, or `f32` (the M1 reference: f32 weights re-read
/// from the checkpoint per use, nothing resident).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WFmt {
	F32,
	Q4K,
	Q6K,
}

impl WFmt {
	pub fn parse(s: &str) -> Result<WFmt, String> {
		match s.to_ascii_lowercase().as_str() {
			"f32" | "ref" => Ok(WFmt::F32),
			"q4k" | "q4_k" => Ok(WFmt::Q4K),
			"q6k" | "q6_k" | "" => Ok(WFmt::Q6K),
			_ => Err(format!("weight format {s:?}: want q6k, q4k or f32")),
		}
	}

	pub fn from_env() -> WFmt {
		std::env::var("QWEN_IMAGE_WEIGHTS")
			.ok()
			.and_then(|v| WFmt::parse(&v).ok())
			.unwrap_or(WFmt::Q6K)
	}
}

/// A matmul weight `[n][k]`: f32, or a K-quant run by `quant`'s GEMMs on
/// q8_K activations (needs `k` a multiple of 256; other shapes stay f32).
///
/// A K-quant weight is stored **rotated**: each 256-wide block of a row is
/// `H (s * w)` (`H` the normalized Hadamard matrix, `s` fixed random signs,
/// `npubuild::gptq::signs`), and every activation row is rotated the same
/// way before its q8_K quantization, so the product is unchanged (`H` is
/// orthonormal). Qwen's hidden states carry a few channels hundreds of
/// times the rest; q8_K scales each 256-block by its largest value, and
/// unrotated such a block loses the other 255 (text encoder output rel.
/// error 1.1e-1 with Q6_K; rotated, the outlier is spread over the block).
/// `QWEN_IMAGE_ROT=0` is the A/B.
pub enum Lin {
	F32 { w: Vec<f32>, n: usize, k: usize },
	Q { q: quant::LinearWeight, rot: bool },
}

const RB: usize = 256;

fn rot_signs() -> &'static [f32] {
	static S: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
	S.get_or_init(|| npubuild::gptq::signs(RB))
}

fn rot_on() -> bool {
	std::env::var("QWEN_IMAGE_ROT").map_or(true, |v| v != "0")
}

/// Rotate one row in place (`H (s * x)` per 256-block).
fn rotate_row(x: &mut [f32]) {
	let s = rot_signs();
	for b in x.chunks_exact_mut(RB) {
		b.iter_mut().zip(s).for_each(|(v, &g)| *v *= g);
	}
	npubuild::la::fwht_blocks(x, RB);
}

impl Lin {
	pub fn new(w: Vec<f32>, n: usize, k: usize, fmt: WFmt) -> Lin {
		assert_eq!(w.len(), n * k);
		if !k.is_multiple_of(256) {
			return Lin::F32 { w, n, k };
		}
		if fmt == WFmt::F32 {
			return Lin::F32 { w, n, k };
		}
		let rot = rot_on();
		let mut w = w;
		if rot {
			rows_mut(&mut w, k, rotate_row);
		}
		let q = match fmt {
			WFmt::Q4K => quant::LinearWeight::Q4K {
				raw: quant::quantize_q4_k(&w, n, k),
				out_dim: n,
				in_dim: k,
			},
			_ => quant::LinearWeight::Q6K {
				raw: quant::quantize_q6_k(&w, n, k),
				out_dim: n,
				in_dim: k,
			},
		};
		Lin::Q { q, rot }
	}

	fn rot(&self) -> bool {
		matches!(self, Lin::Q { rot: true, .. })
	}

	pub fn n(&self) -> usize {
		match self {
			Lin::F32 { n, .. } => *n,
			Lin::Q { q, .. } => q.out_dim(),
		}
	}

	pub fn bytes(&self) -> usize {
		match self {
			Lin::F32 { w, .. } => w.len() * 4,
			Lin::Q {
				q: quant::LinearWeight::Q4K { raw, .. } | quant::LinearWeight::Q6K { raw, .. },
				..
			} => raw.len(),
			Lin::Q { .. } => 0,
		}
	}

	fn q8(&self, q8: &std::sync::Arc<Vec<quant::Q8K>>, s: usize) -> Vec<f32> {
		let n = self.n();
		let mut y = quant::uninit_vec(s * n);
		match self {
			Lin::Q {
				q: quant::LinearWeight::Q4K { raw, in_dim, .. },
				..
			} => quant::gemm_q4k_strided_q8(n, *in_dim, s, raw, q8, &mut y, n, 0),
			Lin::Q {
				q: quant::LinearWeight::Q6K { raw, in_dim, .. },
				..
			} => quant::gemm_q6k_strided_q8(n, *in_dim, s, raw, q8, &mut y, n, 0),
			_ => unreachable!("q8 on an f32 weight"),
		}
		y
	}

	/// `y[s][n] = x[s][k] w^T`.
	pub fn apply(&self, x: &[f32], s: usize) -> Vec<f32> {
		match self {
			Lin::F32 { w, n, k } => linear(x, s, *k, w, *n),
			Lin::Q { rot, .. } => self.q8(&quantize(x, s, *rot), s),
		}
	}
}

/// q8_K rows of `x` (`s` rows), rotated first when `rot`.
fn quantize(x: &[f32], s: usize, rot: bool) -> std::sync::Arc<Vec<quant::Q8K>> {
	let k = x.len() / s;
	std::sync::Arc::new(if rot {
		quant::par_rows_q8k(s, k, |t, r| {
			r.copy_from_slice(&x[t * k..(t + 1) * k]);
			rotate_row(r);
		})
	} else {
		quant::par_quantize_q8k(x, s, k)
	})
}

/// Several weights on one activation `x` (`s` rows), quantized once when
/// they are all K-quants.
pub fn multi(ws: &[&Lin], x: &[f32], s: usize) -> Vec<Vec<f32>> {
	let rot = ws[0].rot();
	if ws
		.iter()
		.all(|w| matches!(w, Lin::Q { .. }) && w.rot() == rot)
	{
		let q8 = quantize(x, s, rot);
		ws.iter().map(|w| w.q8(&q8, s)).collect()
	} else {
		ws.iter().map(|w| w.apply(x, s)).collect()
	}
}

/// `down(silu(gate) * up)`; a K-quant `down` reads q8_K rows formed
/// straight from the product (no f32 matrix in between).
pub fn swiglu_down(down: &Lin, gate: &[f32], mut up: Vec<f32>, s: usize) -> Vec<f32> {
	match down {
		Lin::Q { rot, .. } => {
			let n = up.len() / s;
			let q8 = std::sync::Arc::new(quant::par_rows_q8k(s, n, |t, r| {
				quant::silu_mul(&gate[t * n..(t + 1) * n], &up[t * n..(t + 1) * n], r);
				if *rot {
					rotate_row(r);
				}
			}));
			down.q8(&q8, s)
		}
		Lin::F32 { .. } => {
			quant::silu_mul_assign(gate, &mut up);
			down.apply(&up, s)
		}
	}
}

/// Rows per chunk of the MLPs (the hidden state of a chunk, not of the
/// whole sequence, is materialized).
pub const MLP_CHUNK: usize = 256;
