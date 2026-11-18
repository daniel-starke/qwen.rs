//! GPTQ's factorization on a Vulkan device: the Cholesky factor of a large
//! Hessian and its inverse, blocked so that every large step is one product
//! `C (-)= A B^T` in VRAM (`offload::Dense`, `gemm_f32.glsl`); only the
//! 256 x 256 diagonal blocks are factored and inverted on the host.
//!
//! The 17408-wide input of `down_proj` is most of a layer's quantization
//! time; with it on the GPU the cores quantize the layer's other matrices
//! meanwhile ([`crate::weights`]).

use offload::Dense;

/// Columns per block.
const JB: usize = 256;

pub struct GpuFactor {
	g: Dense,
}

/// Cholesky factor of the `[n][n]` block `a` (its lower triangle) and the
/// factor's inverse, both lower triangular, in f64.
fn block(a: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
	let mut l = vec![0f64; n * n];
	for c in 0..n {
		let mut d = a[c * n + c] as f64;
		for x in 0..c {
			d -= l[c * n + x] * l[c * n + x];
		}
		if d <= 0.0 || !d.is_finite() {
			return Err(format!("cholesky: a pivot is {d:e}"));
		}
		let d = d.sqrt();
		l[c * n + c] = d;
		for r in c + 1..n {
			let mut s = a[r * n + c] as f64;
			for x in 0..c {
				s -= l[r * n + x] * l[c * n + x];
			}
			l[r * n + c] = s / d;
		}
	}
	let mut x = vec![0f64; n * n];
	for i in 0..n {
		let d = l[i * n + i];
		x[i * n + i] = 1.0 / d;
		for j in 0..i {
			let mut s = 0f64;
			for t in j..i {
				s += l[i * n + t] * x[t * n + j];
			}
			x[i * n + j] = -s / d;
		}
	}
	Ok((
		l.iter().map(|&v| v as f32).collect(),
		x.iter().map(|&v| v as f32).collect(),
	))
}

impl GpuFactor {
	/// On the default Vulkan adapter (the discrete GPU).
	pub fn new() -> Result<GpuFactor, String> {
		Ok(GpuFactor {
			g: Dense::new(None)?,
		})
	}

	/// True when the device has room for a `k`-wide factorization.
	pub fn fits(&self, k: usize) -> bool {
		let need = (3 * k * k + JB * k + JB * JB) as u64 * 4 + (256 << 20);
		self.g.free_bytes().is_none_or(|f| f >= need)
	}

	/// The inverse of the Cholesky factor of `hr` (`[k][k]`, lower triangle
	/// read): `[k][k]`, zero above the diagonal.
	pub fn inv_chol(&mut self, hr: &[f32], k: usize) -> Result<Vec<f32>, String> {
		if !self.fits(k) {
			return Err("not enough free VRAM".into());
		}
		let g = &mut self.g;
		let mut bufs = Vec::new();
		let r = (|| {
			let mut alloc = |g: &mut Dense, n: usize| -> Result<usize, String> {
				let b = g.alloc(n)?;
				bufs.push(b);
				Ok(b)
			};
			// the working matrix, the factor, the inverse, a block row and a block
			let (hb, lb, xb, tb, sb) = (
				alloc(g, k * k)?,
				alloc(g, k * k)?,
				alloc(g, k * k)?,
				alloc(g, JB * k)?,
				alloc(g, JB * JB)?,
			);
			g.upload(hb, 0, hr)?;
			g.zero(xb)?;
			let mut invs = Vec::new();
			let mut d = vec![0f32; JB * JB];
			for j in (0..k).step_by(JB) {
				let jb = JB.min(k - j);
				let j2 = j + jb;
				g.download(hb, j * k + j, k, jb, jb, &mut d[..jb * jb])?;
				let (l11, inv) = block(&d[..jb * jb], jb)?;
				g.upload_rows(lb, j * k + j, k, jb, jb, &l11)?;
				if j2 < k {
					// the panel below: L21 = A21 L11^-T, then the trailing
					// lower triangle -= L21 L21^T
					g.upload(sb, 0, &inv)?;
					let m = k - j2;
					g.gemm(
						(hb, j2 * k + j, k),
						m,
						(sb, 0, jb),
						jb,
						jb,
						(lb, j2 * k + j, k),
						false,
						false,
						false,
					)?;
					g.gemm(
						(lb, j2 * k + j, k),
						m,
						(lb, j2 * k + j, k),
						m,
						jb,
						(hb, j2 * k + j2, k),
						true,
						false,
						true,
					)?;
				}
				invs.push(inv);
			}
			// X = L^-1 block row by block row: X[d][d] = L[d][d]^-1,
			// X[d][< d] = -X[d][d] (L[d][< d] X[< d][< d])
			for (bi, inv) in invs.iter().enumerate() {
				let j = bi * JB;
				let jb = JB.min(k - j);
				g.upload_rows(xb, j * k + j, k, jb, jb, inv)?;
				if j > 0 {
					g.gemm(
						(lb, j * k, k),
						jb,
						(xb, 0, k),
						j,
						j,
						(tb, 0, k),
						false,
						true,
						false,
					)?;
					g.upload(sb, 0, inv)?;
					g.gemm(
						(sb, 0, jb),
						jb,
						(tb, 0, k),
						j,
						jb,
						(xb, j * k, k),
						true,
						true,
						false,
					)?;
				}
			}
			let mut x = vec![0f32; k * k];
			g.download(xb, 0, k, k, k, &mut x)?;
			Ok(x)
		})();
		for b in bufs {
			g.free(b);
		}
		r
	}
}
