//! A decode layer's MLP on the CPU cores (`scripts/cutmlp.py` cuts it off
//! the NPU graph). Decode is bound by reading weights from DRAM, and the
//! cores read them faster than the NPU's DMA does (~58 against ~36 GB/s);
//! the MLP's gate / up / down matrices are 70 % of a layer's bytes.
//!
//! The codes are the prefill set's plain int4 files (per-row scale, GPTQ of
//! the block-Hadamard-rotated weights); the activation of each matrix is
//! quantized to int8 per position (it is Hadamard-rotated, so close to
//! Gaussian), and a row is `maddubs` over (nibble + 8) x int8.

use core::arch::x86_64::*;
use std::path::Path;

const W: usize = 5120;
const F: usize = 17408;
const HB: usize = 256;
/// Positions one call takes at most (the verify set's rows).
pub const MAXS: usize = 8;

/// Nanoseconds since the last [`take_split`]: before the cores' share
/// (norm, rotations, GPU submit), in it, waiting for the GPU after it.
static SPLIT: [std::sync::atomic::AtomicU64; 3] =
	[const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The three phase times of [`Mlp::run`] since the last call, in ms.
pub fn take_split() -> [f64; 3] {
	std::array::from_fn(|i| SPLIT[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-6)
}

pub struct Mlp {
	eps: f32,
	nw: Vec<f32>,
	d: [Vec<f32>; 3],
	/// Dense rotations, when they are not the Sylvester Hadamard / 16.
	h: Option<[Vec<f32>; 3]>,
	sg: Vec<f32>,
	su: Vec<f32>,
	sd: Vec<f32>,
	/// `[F][W / 2]`, nibbles biased by 8.
	gate: Vec<u8>,
	up: Vec<u8>,
	/// `[W][F / 2]`.
	down: Vec<u8>,
}

fn codes(dir: &str, l: usize, names: &str, bytes: usize) -> Result<Vec<u8>, String> {
	let mut v = Vec::with_capacity(bytes);
	for nm in names.split(',') {
		let f = Path::new(dir).join(format!("lean{l}.{nm}.in"));
		v.extend_from_slice(&std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?);
	}
	if v.len() != bytes {
		return Err(format!(
			"{dir}: lean{l}.{names}.in hold {} bytes, the MLP wants {bytes}",
			v.len()
		));
	}
	for b in v.iter_mut() {
		*b ^= 0x88;
	}
	Ok(v)
}

impl Mlp {
	/// `layer{l}.mlp` of a cut decode set.
	pub fn load(file: &str, l: usize) -> Result<Mlp, String> {
		let raw = std::fs::read(file).map_err(|e| format!("{file}: {e}"))?;
		let nl = raw
			.iter()
			.position(|&b| b == b'\n')
			.ok_or(format!("{file}: no header"))?;
		let head = String::from_utf8_lossy(&raw[..nl]).to_string();
		let get = |k: &str| {
			head.split(' ')
				.find_map(|t| t.strip_prefix(k))
				.map(str::to_string)
				.ok_or(format!("{file}: no {k}"))
		};
		if !head.starts_with("mlp1 ") {
			return Err(format!("{file}: not an mlp1 file"));
		}
		let fl: Vec<f32> = raw[nl + 1..]
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect();
		let sizes = [W, W, HB * HB, W, HB * HB, F, HB * HB, F, F, W];
		if fl.len() != sizes.iter().sum::<usize>() {
			return Err(format!("{file}: {} floats", fl.len()));
		}
		let mut o = 0;
		let mut part = |i: usize| {
			let v = fl[o..o + sizes[i]].to_vec();
			o += sizes[i];
			v
		};
		let (nw, d1, h1, d2, h2, d3, h3, sg, su, sd) = (
			part(0),
			part(1),
			part(2),
			part(3),
			part(4),
			part(5),
			part(6),
			part(7),
			part(8),
			part(9),
		);
		// (a relative directory is relative to the file's)
		let mut dir = get("wdir=")?;
		if Path::new(&dir).is_relative() {
			if let Some(d) = Path::new(file).parent() {
				dir = d.join(&dir).to_string_lossy().into_owned();
			}
		}
		Ok(Mlp {
			eps: get("eps=")?.parse().map_err(|_| format!("{file}: eps"))?,
			nw,
			d: [d1, d2, d3],
			h: if get("had=")? == "1" {
				None
			} else {
				Some([h1, h2, h3])
			},
			sg,
			su,
			sd,
			gate: codes(&dir, l, &get("gate=")?, F * W / 2)?,
			up: codes(&dir, l, &get("up=")?, F * W / 2)?,
			down: codes(&dir, l, &get("down=")?, W * F / 2)?,
		})
	}

	/// `v` as `[blocks][256] @ H`, in place.
	fn rot(&self, which: usize, v: &mut [f32]) {
		match &self.h {
			None => {
				for b in v.chunks_exact_mut(HB) {
					let mut len = 1;
					while len < HB {
						for i in (0..HB).step_by(2 * len) {
							for j in i..i + len {
								let (a, c) = (b[j], b[j + len]);
								b[j] = a + c;
								b[j + len] = a - c;
							}
						}
						len *= 2;
					}
					for x in b.iter_mut() {
						*x *= 1.0 / 16.0;
					}
				}
			}
			Some(h) => {
				let h = &h[which];
				let mut t = [0f32; HB];
				for b in v.chunks_exact_mut(HB) {
					t.fill(0.0);
					for (i, &x) in b.iter().enumerate() {
						for (o, &hv) in t.iter_mut().zip(&h[i * HB..(i + 1) * HB]) {
							*o += x * hv;
						}
					}
					b.copy_from_slice(&t);
				}
			}
		}
	}

	/// What a GPU share needs of this layer: gate / up / down codes
	/// (nibbles biased by 8), their row scales, the diagonal before the
	/// last rotation.
	#[allow(clippy::type_complexity)]
	pub fn gpu_layer(&self) -> (&[u8], &[u8], &[u8], &[f32], &[f32], &[f32], &[f32]) {
		(
			&self.gate, &self.up, &self.down, &self.sg, &self.su, &self.d[2], &self.sd,
		)
	}

	/// Whether a GPU share can take part (it computes the rotations as
	/// Hadamard transforms).
	pub fn hadamard(&self) -> bool {
		self.h.is_none()
	}

	/// Gate / up rows `[lo, hi)` and the down columns `[lo, hi)` (multiples
	/// of 256) of the activations `a` / `b` `[s][W]`, added to `y` `[s][W]`.
	unsafe fn part(&self, s: usize, qa: &Quant, qb: &Quant, lo: usize, hi: usize, y: &mut [f32]) {
		if lo >= hi {
			return;
		}
		let n = hi - lo;
		// gate / up rows -> silu(g) * u * d3
		let mut prod = vec![0f32; s * n];
		let pp = prod.as_mut_ptr() as usize;
		const RG: usize = 32;
		quant::parallel::par_for(n / RG, |c| {
			// SAFETY: disjoint rows per claim; features as the caller's
			unsafe {
				let p = pp as *mut f32;
				let (mut gb, mut ub) = ([[0f32; RG]; MAXS], [[0f32; RG]; MAXS]);
				for i in 0..RG {
					let r = lo + c * RG + i;
					let g = dot(self.gate.as_ptr().add(r * (W / 2)), W / 2, qa);
					let u = dot(self.up.as_ptr().add(r * (W / 2)), W / 2, qb);
					for j in 0..s {
						gb[j][i] = self.sg[r] * (g[j] - qa.bias[j]) as f32 * qa.inv[j];
						ub[j][i] = self.su[r] * (u[j] - qb.bias[j]) as f32 * qb.inv[j];
					}
				}
				for j in 0..s {
					silu_mul(
						&gb[j],
						&ub[j],
						self.d[2].as_ptr().add(lo + c * RG),
						p.add(j * n + c * RG),
					);
				}
			}
		});
		let mut qc = Quant::new(s, n);
		for j in 0..s {
			let r = &mut prod[j * n..(j + 1) * n];
			self.rot(2, r);
			qc.set(j, r);
		}
		let yp = y.as_mut_ptr() as usize;
		const RD: usize = 16;
		quant::parallel::par_for(W / RD, |c| {
			// SAFETY: as above
			unsafe {
				let p = yp as *mut f32;
				for r in c * RD..(c + 1) * RD {
					let d = dot(self.down.as_ptr().add(r * (F / 2) + lo / 2), n / 2, &qc);
					for j in 0..s {
						*p.add(j * W + r) += self.sd[r] * (d[j] - qc.bias[j]) as f32 * qc.inv[j];
					}
				}
			}
		});
	}

	/// `y = x + MLP(x)` in place on `s` rows of f16 `[W]`. With `gpu`, its
	/// rows run there while the cores do the rest; when it fails, the cores
	/// do its rows too and the error comes back (the result is complete
	/// either way). `Ok`: the seconds the cores then waited for the GPU.
	///
	/// # Safety
	/// The CPU has AVX2, F16C and FMA.
	pub unsafe fn run(
		&self,
		l: usize,
		xy: &mut [u8],
		s: usize,
		gpu: Option<&mut (dyn MlpGpu + '_)>,
	) -> Result<f64, String> {
		assert!((1..=MAXS).contains(&s) && xy.len() >= s * W * 2);
		let t0 = std::time::Instant::now();
		let mut x = vec![0f32; s * W];
		cvt_in(xy.as_ptr(), &mut x);
		let (mut a, mut b) = (Quant::new(s, W), Quant::new(s, W));
		let mut t = vec![0f32; W];
		for j in 0..s {
			let r = &x[j * W..(j + 1) * W];
			let ms = r.iter().map(|v| v * v).sum::<f32>() / W as f32;
			let k = 1.0 / (ms + self.eps).sqrt();
			for (which, q) in [(0, &mut a), (1, &mut b)] {
				for i in 0..W {
					t[i] = r[i] * k * self.nw[i] * self.d[which][i];
				}
				self.rot(which, &mut t);
				q.set(j, &t);
			}
		}
		let mut res = Ok(0.0);
		let mut lo = 0;
		let gpu = match gpu {
			Some(g) if self.h.is_none() && s <= g.max_s() => {
				match g.begin(l, s, [a.view(), b.view()]) {
					Ok(()) => {
						lo = g.rows(s);
						Some(g)
					}
					Err(e) => {
						res = Err(e);
						None
					}
				}
			}
			_ => None,
		};
		let t1 = std::time::Instant::now();
		// `QWEN_NPU_MLP_GPU_SYNC=1`: wait for the GPU before the cores start
		// (its time alone, under `gpu wait` of `QWEN_NPU_MLP_TIMING`)
		static SYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
		let mut gpu = gpu;
		if *SYNC.get_or_init(|| std::env::var("QWEN_NPU_MLP_GPU_SYNC").is_ok()) {
			if let Some(g) = gpu.take() {
				if let Err(e) = g.finish(s, &mut x) {
					res = Err(e);
					lo = 0;
				}
				SPLIT[2].fetch_add(
					t1.elapsed().as_nanos() as u64,
					std::sync::atomic::Ordering::Relaxed,
				);
			}
		}
		let t1 = std::time::Instant::now();
		self.part(s, &a, &b, lo, F, &mut x);
		let t2 = std::time::Instant::now();
		for (i, d) in [t1 - t0, t2 - t1].into_iter().enumerate() {
			SPLIT[i].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
		}
		if let Some(g) = gpu {
			if let Err(e) = g.finish(s, &mut x) {
				res = Err(e);
				self.part(s, &a, &b, 0, lo, &mut x);
			}
			let w = t2.elapsed();
			SPLIT[2].fetch_add(w.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
			res = res.map(|_| w.as_secs_f64());
		}
		cvt_out(&x, xy.as_mut_ptr());
		res
	}
}

/// A share of the MLPs on a GPU with its own memory (`offload::MlpSplit`,
/// a discrete card: its weight bytes leave the DRAM bus that bounds
/// decode): gate / up rows `[0, rows)` and the down columns `[0, rows)` of
/// every layer.
pub trait MlpGpu: Send {
	/// Rows per layer it computes for `s` positions, a multiple of 256.
	fn rows(&self, s: usize) -> usize;
	/// Positions one call takes at most.
	fn max_s(&self) -> usize;
	/// Start layer `l` on the rotated, quantized activations of the gate
	/// and the up matrix.
	fn begin(&mut self, l: usize, s: usize, ab: [Act<'_>; 2]) -> Result<(), String>;
	/// Wait for it and add its `[s][W]` to `y`.
	fn finish(&mut self, s: usize, y: &mut [f32]) -> Result<(), String>;
}

/// An int8 activation of `s` positions of `W` elements: per position the
/// even elements, the odd elements (`[s][W / 2]` each), the scale back to
/// f32 and 8 x the sum of its values (the nibble bias).
#[derive(Clone, Copy)]
pub struct Act<'a> {
	pub even: &'a [i8],
	pub odd: &'a [i8],
	pub inv: &'a [f32],
	pub bias: &'a [i32],
}

/// Int8 activations of up to [`MAXS`] positions, even and odd elements
/// apart (a weight byte's low nibble is the even element).
struct Quant {
	n2: usize,
	s: usize,
	vnni: bool,
	even: Vec<i8>,
	odd: Vec<i8>,
	/// 8 x the sum of a position's int8 values (the nibble bias).
	bias: [i32; MAXS],
	inv: [f32; MAXS],
}

impl Quant {
	fn view(&self) -> Act<'_> {
		Act {
			even: &self.even,
			odd: &self.odd,
			inv: &self.inv[..self.s],
			bias: &self.bias[..self.s],
		}
	}

	fn new(s: usize, n: usize) -> Quant {
		Quant {
			n2: n / 2,
			s,
			vnni: std::env::var("QWEN_NPU_MLP_VNNI")
				.map(|v| v != "0")
				.unwrap_or(true)
				&& std::is_x86_feature_detected!("avxvnni"),
			even: vec![0; s * n / 2],
			odd: vec![0; s * n / 2],
			bias: [0; MAXS],
			inv: [0.0; MAXS],
		}
	}

	/// # Safety
	/// AVX2; `v.len()` is a multiple of 8.
	#[target_feature(enable = "avx2")]
	unsafe fn set(&mut self, j: usize, v: &[f32]) {
		let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
		let mut m = _mm256_setzero_ps();
		for i in (0..v.len()).step_by(8) {
			m = _mm256_max_ps(m, _mm256_and_ps(abs, _mm256_loadu_ps(v.as_ptr().add(i))));
		}
		let mut t = [0f32; 8];
		_mm256_storeu_ps(t.as_mut_ptr(), m);
		let q = 127.0 / t.iter().fold(1e-30f32, |a, &b| a.max(b));
		let (e, o) = (
			self.even.as_mut_ptr().add(j * self.n2),
			self.odd.as_mut_ptr().add(j * self.n2),
		);
		let qv = _mm256_set1_ps(q);
		// [e0 o0 e1 o1 ...] as i32 -> the even and the odd bytes
		let pick = _mm256_setr_epi8(
			0, 8, -1, -1, -1, -1, -1, -1, 4, 12, -1, -1, -1, -1, -1, -1, 0, 8, -1, -1, -1, -1, -1,
			-1, 4, 12, -1, -1, -1, -1, -1, -1,
		);
		let mut sum = _mm256_setzero_si256();
		for i in (0..v.len()).step_by(8) {
			let x = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v.as_ptr().add(i)), qv));
			sum = _mm256_add_epi32(sum, x);
			let y = _mm256_shuffle_epi8(x, pick);
			let (lo, hi) = (_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1));
			let ev = (_mm_cvtsi128_si32(lo) as u32 & 0xffff)
				| ((_mm_cvtsi128_si32(hi) as u32 & 0xffff) << 16);
			let od = (_mm_extract_epi32(lo, 2) as u32 & 0xffff)
				| ((_mm_extract_epi32(hi, 2) as u32 & 0xffff) << 16);
			(e.add(i / 2) as *mut u32).write_unaligned(ev);
			(o.add(i / 2) as *mut u32).write_unaligned(od);
		}
		let mut t = [0i32; 8];
		_mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, sum);
		self.bias[j] = 8 * t.iter().sum::<i32>();
		self.inv[j] = 1.0 / q;
	}
}

/// `out = silu(g) * u * d` over a multiple of 8 elements.
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul(g: &[f32], u: &[f32], d: *const f32, out: *mut f32) {
	let one = _mm256_set1_ps(1.0);
	for i in (0..g.len()).step_by(8) {
		let gv = _mm256_loadu_ps(g.as_ptr().add(i));
		// exp(-g): n = round(x / ln 2), 2^n * poly(x - n ln 2)
		let x = _mm256_max_ps(
			_mm256_min_ps(_mm256_sub_ps(_mm256_setzero_ps(), gv), _mm256_set1_ps(87.0)),
			_mm256_set1_ps(-87.0),
		);
		let n = _mm256_round_ps(
			_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
			_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
		);
		let r = _mm256_fnmadd_ps(
			n,
			_mm256_set1_ps(1.428_606_8e-6),
			_mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_145_75), x),
		);
		let mut p = _mm256_set1_ps(1.0 / 720.0);
		for c in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
			p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
		}
		let e = _mm256_mul_ps(
			p,
			_mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)),
				23,
			)),
		);
		let sv = _mm256_div_ps(gv, _mm256_add_ps(one, e));
		_mm256_storeu_ps(
			out.add(i),
			_mm256_mul_ps(
				_mm256_mul_ps(sv, _mm256_loadu_ps(u.as_ptr().add(i))),
				_mm256_loadu_ps(d.add(i)),
			),
		);
	}
}

/// One weight row (`n` bytes, a multiple of 128) against every position.
#[inline]
unsafe fn dot(w: *const u8, n: usize, q: &Quant) -> [i32; MAXS] {
	if q.vnni {
		dot_vnni(w, n, q)
	} else {
		dot_avx2(w, n, q)
	}
}

macro_rules! finish {
	($acc:ident, $out:ident, $s:expr) => {
		for j in 0..$s {
			let x = _mm_add_epi32(
				_mm256_castsi256_si128($acc[j]),
				_mm256_extracti128_si256($acc[j], 1),
			);
			let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4e));
			let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));
			$out[j] = _mm_cvtsi128_si32(x);
		}
	};
}

macro_rules! by_s {
	($q:ident, $go:ident) => {
		match $q.s {
			1 => $go!(1),
			2 => $go!(2),
			3 => $go!(3),
			4 => $go!(4),
			5 => $go!(5),
			6 => $go!(6),
			7 => $go!(7),
			_ => $go!(8),
		}
	};
}

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot_vnni(w: *const u8, n: usize, q: &Quant) -> [i32; MAXS] {
	let m4 = _mm256_set1_epi8(15);
	let mut out = [0i32; MAXS];
	macro_rules! go {
		($s:expr) => {{
			let mut acc = [_mm256_setzero_si256(); $s];
			let mut i = 0;
			while i < n {
				let v = _mm256_loadu_si256(w.add(i) as *const __m256i);
				let lo = _mm256_and_si256(v, m4);
				let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
				for j in 0..$s {
					let e = _mm256_loadu_si256(q.even.as_ptr().add(j * q.n2 + i) as *const __m256i);
					let o = _mm256_loadu_si256(q.odd.as_ptr().add(j * q.n2 + i) as *const __m256i);
					acc[j] = _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(acc[j], lo, e), hi, o);
				}
				i += 32;
			}
			finish!(acc, out, $s);
		}};
	}
	by_s!(q, go);
	out
}

#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(w: *const u8, n: usize, q: &Quant) -> [i32; MAXS] {
	let m4 = _mm256_set1_epi8(15);
	let one = _mm256_set1_epi16(1);
	let mut out = [0i32; MAXS];
	macro_rules! go {
		($s:expr) => {{
			let mut acc = [_mm256_setzero_si256(); $s];
			let mut i = 0;
			while i < n {
				let mut a16 = [_mm256_setzero_si256(); $s];
				// 4 x (2 x 15 x 127 x 2) fits i16
				for k in 0..4 {
					let v = _mm256_loadu_si256(w.add(i + k * 32) as *const __m256i);
					let lo = _mm256_and_si256(v, m4);
					let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
					for j in 0..$s {
						let e = _mm256_loadu_si256(
							q.even.as_ptr().add(j * q.n2 + i + k * 32) as *const __m256i
						);
						let o = _mm256_loadu_si256(
							q.odd.as_ptr().add(j * q.n2 + i + k * 32) as *const __m256i
						);
						a16[j] = _mm256_add_epi16(
							a16[j],
							_mm256_add_epi16(
								_mm256_maddubs_epi16(lo, e),
								_mm256_maddubs_epi16(hi, o),
							),
						);
					}
				}
				for j in 0..$s {
					acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(a16[j], one));
				}
				i += 128;
			}
			finish!(acc, out, $s);
		}};
	}
	by_s!(q, go);
	out
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn cvt_in(src: *const u8, dst: &mut [f32]) {
	for i in (0..dst.len()).step_by(8) {
		_mm256_storeu_ps(
			dst.as_mut_ptr().add(i),
			_mm256_cvtph_ps(_mm_loadu_si128(src.add(i * 2) as *const __m128i)),
		);
	}
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn cvt_out(src: &[f32], dst: *mut u8) {
	for i in (0..src.len()).step_by(8) {
		_mm_storeu_si128(
			dst.add(i * 2) as *mut __m128i,
			_mm256_cvtps_ph(
				_mm256_loadu_ps(src.as_ptr().add(i)),
				_MM_FROUND_TO_NEAREST_INT,
			),
		);
	}
}
