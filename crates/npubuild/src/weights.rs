//! Calibration and weight quantization: the float model runs the
//! calibration chunks layer by layer ([`crate::refm`]); each layer's four
//! quantized MatMul groups get the static int8 range of their (rotated)
//! input and int4 GPTQ codes against that input's Hessian; the results are
//! written as the prefill graphs' bound input files (`lean{L}.<name>.in`)
//! and a per-layer record ([`LayerCal`]) the graph builders read.

use std::path::{Path, PathBuf};

use crate::g::f16_bits;
use crate::gptq::{self, signs};
use crate::la::{self, Acc};
use crate::lean::{GUSPLIT, PROJSPLIT, QS};
use crate::refm::{self, Layer, Mixer, D, DA, FF, H, HKV, HQ, NL, W};
use crate::st::Checkpoint;

/// Positions per calibration chunk.
pub const CHUNK: usize = 256;

/// What calibration leaves of a layer besides its weight files: the static
/// ranges of the four quantized inputs (projection, output projection, gate
/// / up, down) and the unfolded row scales of the four int4 matrices.
pub struct LayerCal {
	pub r: [f32; 4],
	/// Row scales: projection, output projection, gate / up, down (the MTP
	/// block's record: `fc` first, then these).
	pub s: Vec<Vec<f32>>,
}

const MAGIC: &[u8; 8] = b"NPUCAL1\n";

impl LayerCal {
	pub fn path(dir: &Path, l: usize) -> PathBuf {
		dir.join(format!("L{l}.cal"))
	}

	pub fn save(&self, dir: &Path, l: usize) -> Result<(), String> {
		self.save_as(&Self::path(dir, l))
	}

	pub fn load(dir: &Path, l: usize) -> Result<LayerCal, String> {
		Self::load_from(&Self::path(dir, l))
	}

	pub fn save_as(&self, p: &Path) -> Result<(), String> {
		let mut b = MAGIC.to_vec();
		self.r.iter().for_each(|v| b.extend(v.to_le_bytes()));
		for s in &self.s {
			b.extend((s.len() as u32).to_le_bytes());
			s.iter().for_each(|v| b.extend(v.to_le_bytes()));
		}
		write(p, &b)
	}

	pub fn load_from(p: &Path) -> Result<LayerCal, String> {
		let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
		if b.len() < 24 || &b[..8] != MAGIC {
			return Err(format!("{}: not a calibration record", p.display()));
		}
		let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
		let r = [f(8), f(12), f(16), f(20)];
		let mut o = 24;
		let mut s = Vec::new();
		while o + 4 <= b.len() {
			let n = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
			if o + 4 + 4 * n > b.len() {
				return Err(format!("{}: truncated", p.display()));
			}
			s.push((0..n).map(|i| f(o + 4 + 4 * i)).collect());
			o += 4 + 4 * n;
		}
		Ok(LayerCal { r, s })
	}
}

pub fn write(p: &Path, b: &[u8]) -> Result<(), String> {
	// a set's files are hard links: never write through one
	let _ = std::fs::remove_file(p);
	std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn f16s(v: impl Iterator<Item = f64>) -> Vec<u8> {
	v.flat_map(|x| f16_bits(x).to_le_bytes()).collect()
}

/// Running calibration statistics of one quantized input.
struct Stat {
	k: usize,
	h: Vec<f32>,
	r: f32,
}

impl Stat {
	fn new(k: usize) -> Stat {
		Stat {
			k,
			h: vec![0f32; k * k],
			r: 0.0,
		}
	}

	/// Add the chunk `x` `[s][k]`.
	fn add(&mut self, x: &[f32], s: usize, sg: &[f32]) {
		let k = self.k;
		let xt = la::transpose(x, s, k);
		// SAFETY: xt is [k][s], h [k][k]
		unsafe {
			la::gemm_nt_raw(
				xt.as_ptr(),
				s,
				k,
				xt.as_ptr(),
				s,
				k,
				s,
				self.h.as_mut_ptr(),
				k,
				Acc::Add,
			)
		};
		let mut xr = x.to_vec();
		gptq::rotate_rows(&mut xr, k, &sg[..k]);
		self.r = xr.iter().fold(self.r, |m, v| m.max(v.abs()));
	}
}

/// Where GPTQ's largest factorization runs: on a Vulkan device beside the
/// cores (a `vulkan` build with a device that has the VRAM), else on the
/// cores like the others.
pub struct Accel {
	#[cfg(feature = "vulkan")]
	gpu: Option<std::sync::Mutex<crate::gpufac::GpuFactor>>,
	#[cfg(feature = "vulkan")]
	failed: std::sync::atomic::AtomicBool,
}

impl Accel {
	/// With `gpu` false, or without a usable device, everything stays on
	/// the cores. The second value says which it is.
	pub fn new(gpu: bool) -> (Accel, String) {
		#[cfg(feature = "vulkan")]
		{
			let (g, what) = match gpu.then(crate::gpufac::GpuFactor::new) {
				None => (None, "CPU (the GPU was not asked for)".to_string()),
				Some(Ok(g)) if g.fits(FF) => (
					Some(std::sync::Mutex::new(g)),
					"CPU + Vulkan device".to_string(),
				),
				Some(Ok(_)) => (None, "CPU (the Vulkan device lacks free VRAM)".to_string()),
				Some(Err(e)) => (None, format!("CPU (no Vulkan device: {e})")),
			};
			(
				Accel {
					gpu: g,
					failed: std::sync::atomic::AtomicBool::new(false),
				},
				what,
			)
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = gpu;
			(
				Accel {},
				"CPU (built without the vulkan feature)".to_string(),
			)
		}
	}

	/// The inverse Cholesky factor of `hr` on the device; `None` when there
	/// is none or it failed (the caller then uses the cores).
	pub(crate) fn inv_chol(&self, hr: &[f32], k: usize) -> Option<Vec<f32>> {
		#[cfg(feature = "vulkan")]
		{
			use std::sync::atomic::Ordering;
			let g = self.gpu.as_ref()?;
			if self.failed.load(Ordering::Relaxed) {
				return None;
			}
			match g.lock().unwrap().inv_chol(hr, k) {
				Ok(x) => Some(x),
				Err(e) => {
					eprintln!(
						"npu-build: the Vulkan factorization failed ({e}); continuing on the CPU"
					);
					self.failed.store(true, Ordering::Relaxed);
					None
				}
			}
		}
		#[cfg(not(feature = "vulkan"))]
		{
			let _ = (hr, k);
			None
		}
	}
}

/// A matrix on its way through GPTQ: rotated, with its row scales and its
/// input's Hessian ready to factor.
struct Half {
	m: Vec<f32>,
	n: usize,
	sc: Vec<f32>,
	p: gptq::Prepared,
}

fn half(mut m: Vec<f32>, n: usize, st: Stat, sg: &[f32]) -> Half {
	let k = st.k;
	gptq::rotate_rows(&mut m, k, &sg[..k]);
	let sc = gptq::row_scales(&m, n, k);
	let mut h = st.h;
	gptq::rotate_sym(&mut h, k, &sg[..k]);
	Half {
		m,
		n,
		sc,
		p: gptq::prepare(h, k),
	}
}

impl Half {
	/// With `x` the inverse Cholesky factor of the prepared Hessian (`None`:
	/// computed here, on the cores): (packed int4 codes, row scales).
	fn finish(mut self, x: Option<Vec<f32>>) -> Result<(Vec<u8>, Vec<f32>), String> {
		let x = match x {
			Some(x) => x,
			None => gptq::inv_chol(std::mem::take(&mut self.p.hr), self.p.k)?,
		};
		let f = self.p.finish(x);
		let q = gptq::gptq(self.m, self.n, &f, &self.sc);
		Ok((gptq::pack4(&q), self.sc))
	}
}

/// `m R` quantized against the input statistics `st`: (packed int4 codes
/// `[n][k / 2]`, row scales).
fn quantize(m: Vec<f32>, n: usize, st: Stat, sg: &[f32]) -> Result<(Vec<u8>, Vec<f32>), String> {
	half(m, n, st, sg).finish(None)
}

fn cat(a: Vec<f32>, b: &[f32]) -> Vec<f32> {
	let mut a = a;
	a.extend_from_slice(b);
	a
}

/// The attention projection's rows in the graphs' order: the query rows of
/// every head, the gate rows of every head, k, v.
pub fn attn_proj(a: &refm::Attn) -> Vec<f32> {
	let mut m = Vec::with_capacity((2 * HQ * DA + 2 * HKV * DA) * W);
	for part in 0..2 {
		for h in 0..HQ {
			m.extend_from_slice(&a.wq[(h * 2 + part) * DA * W..][..DA * W]);
		}
	}
	m.extend_from_slice(&a.wk);
	m.extend_from_slice(&a.wv);
	m
}

/// Quantize layer `l` from its float weights and its inputs' statistics,
/// write its bound input files into `wdir` and return its record.
fn layer_files(
	layer: Layer,
	st: [Stat; 4],
	name: &dyn Fn(&str) -> PathBuf,
	accel: &Accel,
) -> Result<LayerCal, String> {
	let sg = signs(FF);
	let [sp, so, sgu, sd] = st;
	let r = [sp.r, so.r, sgu.r, sd.r];
	let Layer { mixer, mlp } = layer;
	// down_proj's 17408-wide input is most of the work: its factorization
	// goes to the Vulkan device (if any) while the cores do the rest
	let mut hd = half(mlp.wd, W, sd, &sg);
	let hr = std::mem::take(&mut hd.p.hr);
	let is_delta = matches!(mixer, Mixer::Delta(_));
	std::thread::scope(|scope| {
		let job = scope.spawn(move || (accel.inv_chol(&hr, FF), hr));
		let (qgu, sgu_) = quantize(cat(mlp.wg, &mlp.wu), 2 * FF, sgu, &sg)?;
		let (r_proj, r_out, r_gu, r_down) = (r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64);
		let (s_proj, s_out);
		match mixer {
			Mixer::Delta(d) => {
				let (qp, sp_) = quantize(cat(d.wqkv, &d.wz), PROJSPLIT.iter().sum(), sp, &sg)?;
				let (qo, so_) = quantize(d.wo, W, so, &sg)?;
				// the gate rows (in_proj_a / b) stay int8, round to nearest
				let mut ab = cat(d.wa, &d.wb);
				gptq::rotate_rows(&mut ab, W, &sg[..W]);
				let (qab, sab) = gptq::quant_i8(&ab, 2 * H, W);
				// Input names: `w{i}` / `s{i}` are numbered by their position
				// among the per-layer inputs (ln1 is 0).
				write(
					&name("ln1"),
					&f16s((0..W).map(|i| d.ln1[i] as f64 * sg[i] as f64 * 127.0 / r_proj)),
				)?;
				let mut r0 = 0;
				for (i, n) in PROJSPLIT.iter().enumerate() {
					write(
						&name(&format!("w{}", 1 + 2 * i)),
						&qp[r0 * W / 2..(r0 + n) * W / 2],
					)?;
					write(
						&name(&format!("s{}", 1 + 2 * i)),
						&f16s(
							sp_[r0..r0 + n]
								.iter()
								.map(|&s| s as f64 * r_proj / 127.0 * 256.0),
						),
					)?;
					r0 += n;
				}
				write(
					&name("w7"),
					&qab.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
				)?;
				write(
					&name("s7"),
					&f16s(sab.iter().map(|&s| s as f64 * r_proj / 127.0 * 256.0)),
				)?;
				write(&name("cw"), &f16s(d.conv.iter().map(|&v| v as f64)))?;
				write(&name("dtb"), &f16s(d.dtb.iter().map(|&v| v as f64)))?;
				write(&name("acoef"), &f16s(d.acoef.iter().map(|&v| v as f64)))?;
				// gnb [1, 1, H, 2, D]: rows (h, 0) the gated norm's weight (with
				// the next rotation's signs and range folded in), the norm's eps
				// (f32, at the recurrence's scale) in the first two slots of row (0, 1)
				let mut gnb = vec![0u16; H * 2 * D];
				for h in 0..H {
					for i in 0..D {
						gnb[h * 2 * D + i] =
							f16_bits(d.gnw[i] as f64 * sg[h * D + i] as f64 * 127.0 / r_out);
					}
				}
				let eps = ((refm::EPS as f64 * QS * QS) as f32).to_bits();
				gnb[D] = eps as u16;
				gnb[D + 1] = (eps >> 16) as u16;
				write(
					&name("gnb"),
					&gnb.iter()
						.flat_map(|v| v.to_le_bytes())
						.collect::<Vec<u8>>(),
				)?;
				write(&name("w13"), &qo)?;
				write(
					&name("s13"),
					&f16s(so_.iter().map(|&s| s as f64 * r_out / 127.0 * 256.0)),
				)?;
				write(
					&name("ln2"),
					&f16s((0..W).map(|i| mlp.ln2[i] as f64 * sg[i] as f64 * 127.0 / r_gu)),
				)?;
				let n = 2 * FF / GUSPLIT;
				for i in 0..GUSPLIT {
					write(
						&name(&format!("w{}", 16 + 2 * i)),
						&qgu[i * n * W / 2..(i + 1) * n * W / 2],
					)?;
					// the up rows carry the down rotation's signs and range
					let sc = (i * n..(i + 1) * n).map(|j| {
						sgu_[j] as f64 * r_gu / 127.0
							* 256.0 * if j < FF {
							1.0
						} else {
							sg[j - FF] as f64 * 127.0 / r_down
						}
					});
					write(&name(&format!("s{}", 16 + 2 * i)), &f16s(sc))?;
				}
				(s_proj, s_out) = (sp_, so_);
			}
			Mixer::Attn(a) => {
				let (qp, sp_) = quantize(attn_proj(&a), 2 * HQ * DA + 2 * HKV * DA, sp, &sg)?;
				let (qo, so_) = quantize(a.wo, W, so, &sg)?;
				write(&name("w0"), &qp)?;
				write(&name("w1"), &qo)?;
				let n = 2 * FF / GUSPLIT * W / 2;
				for i in 0..GUSPLIT {
					write(&name(&format!("w{}", 2 + i)), &qgu[i * n..(i + 1) * n])?;
				}
				(s_proj, s_out) = (sp_, so_);
			}
		}
		let (x, hr) = job
			.join()
			.map_err(|_| "the factorization thread panicked".to_string())?;
		hd.p.hr = hr;
		let (qd, sdn) = hd.finish(x)?;
		if is_delta {
			write(&name("w24"), &qd)?;
			write(
				&name("s24"),
				&f16s(sdn.iter().map(|&s| s as f64 * r_down / 127.0 * 256.0)),
			)?;
		} else {
			write(&name("w6"), &qd)?;
		}
		Ok(LayerCal {
			r,
			s: vec![s_proj, s_out, sgu_, sdn],
		})
	})
}

/// Calibrate and quantize every layer on the chunks `chunks` (each
/// [`CHUNK`] token ids): `wdir/lean{L}.*.in` and `cal/L{L}.cal`. Layers whose
/// record exists are skipped (the residuals after the last finished layer
/// are kept in `cal/resid.f32`). `log` gets a line per layer.
pub fn calibrate(
	ck: &Checkpoint,
	chunks: &[Vec<u32>],
	cal: &Path,
	wdir: &Path,
	accel: &Accel,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	for d in [cal, wdir] {
		std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
	}
	let done = (0..NL)
		.take_while(|&l| LayerCal::path(cal, l).exists())
		.count();
	if done == NL {
		return Ok(());
	}
	let resid = cal.join("resid.f32");
	let n = chunks.len();
	let mut xs: Vec<Vec<f32>> = Vec::new();
	let mut start = 0;
	if done > 0 {
		// "L <n chunks>" then the residuals
		if let Ok(b) = std::fs::read(&resid) {
			if b.len() == 16 + n * CHUNK * W * 4
				&& u64::from_le_bytes(b[..8].try_into().unwrap()) == done as u64
				&& u64::from_le_bytes(b[8..16].try_into().unwrap()) == n as u64
			{
				xs = b[16..]
					.chunks_exact(CHUNK * W * 4)
					.map(|c| {
						c.chunks_exact(4)
							.map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
							.collect()
					})
					.collect();
				start = done;
			}
		}
	}
	if xs.is_empty() {
		for c in chunks {
			assert_eq!(c.len(), CHUNK);
			xs.push(refm::embed(ck, c)?);
		}
	}
	let sg = signs(FF);
	for l in start..NL {
		let t0 = std::time::Instant::now();
		let layer = refm::load_layer(ck, l)?;
		let mut st = [Stat::new(W), Stat::new(H * D), Stat::new(W), Stat::new(FF)];
		for x in xs.iter_mut() {
			let (y, taps) = refm::forward(&layer, x, CHUNK);
			if l >= done {
				for (s, t) in st
					.iter_mut()
					.zip([&taps.proj, &taps.out, &taps.gu, &taps.down])
				{
					s.add(t, CHUNK, &sg);
				}
			}
			*x = y;
		}
		let tf = t0.elapsed().as_secs_f64();
		if l < done {
			// only the residuals were missing
			continue;
		}
		let c = layer_files(layer, st, &|n| wdir.join(format!("lean{l}.{n}.in")), accel)?;
		// residuals first, then the record that marks the layer done
		let mut b = Vec::with_capacity(16 + n * CHUNK * W * 4);
		b.extend((l as u64 + 1).to_le_bytes());
		b.extend((n as u64).to_le_bytes());
		for x in &xs {
			x.iter().for_each(|v| b.extend(v.to_le_bytes()));
		}
		write(&resid, &b)?;
		c.save(cal, l)?;
		log(&format!(
			"layer {l}: forward {tf:.1} s, quantized in {:.1} s; ranges proj {:.2} out {:.2} gate/up {:.2} down {:.2}",
			t0.elapsed().as_secs_f64() - tf,
			c.r[0],
			c.r[1],
			c.r[2],
			c.r[3]
		));
	}
	Ok(())
}

/// The residuals after the last layer of every calibration chunk: kept by
/// [`calibrate`], else computed again.
fn final_residuals(
	ck: &Checkpoint,
	chunks: &[Vec<u32>],
	cal: &Path,
) -> Result<Vec<Vec<f32>>, String> {
	let n = chunks.len();
	if let Ok(b) = std::fs::read(cal.join("resid.f32")) {
		if b.len() == 16 + n * CHUNK * W * 4
			&& u64::from_le_bytes(b[..8].try_into().unwrap()) == NL as u64
			&& u64::from_le_bytes(b[8..16].try_into().unwrap()) == n as u64
		{
			return Ok(b[16..]
				.chunks_exact(CHUNK * W * 4)
				.map(|c| {
					c.chunks_exact(4)
						.map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
						.collect()
				})
				.collect());
		}
	}
	let mut xs = Vec::new();
	for c in chunks {
		xs.push(refm::embed(ck, c)?);
	}
	for l in 0..NL {
		let layer = refm::load_layer(ck, l)?;
		for x in xs.iter_mut() {
			*x = refm::forward(&layer, x, CHUNK).0;
		}
	}
	Ok(xs)
}

/// The file of a bound weight input `w.*` in `wdir`.
pub fn bound(wdir: &Path, name: &str) -> PathBuf {
	wdir.join(format!("{name}.in"))
}

/// Calibrate and quantize the MTP block (`mtp.*`: `fc` on the normed
/// embedding of a token and the normed hidden state before it, then a
/// full-attention layer) on the same chunks: `wdir/w.mtp.*.in`,
/// `cal/mtp.cal` (row scales: fc, projection, o_proj, gate / up, down).
pub fn calibrate_mtp(
	ck: &Checkpoint,
	chunks: &[Vec<u32>],
	cal: &Path,
	wdir: &Path,
	accel: &Accel,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	let rec = cal.join("mtp.cal");
	if rec.exists() {
		return Ok(());
	}
	let t0 = std::time::Instant::now();
	let xs = final_residuals(ck, chunks, cal)?;
	let layer = refm::load_mtp_layer(ck)?;
	let fc = ck.mat("mtp.fc.weight", W, 2 * W)?;
	let ne = refm::norm_w(ck, "mtp.pre_fc_norm_embedding.weight", W)?;
	let nh = refm::norm_w(ck, "mtp.pre_fc_norm_hidden.weight", W)?;
	let sg = signs(FF);
	let s = CHUNK - 1;
	let mut sfc = Stat::new(2 * W);
	let mut st = [
		Stat::new(W),
		Stat::new(HQ * DA),
		Stat::new(W),
		Stat::new(FF),
	];
	for (x, ids) in xs.iter().zip(chunks) {
		// position t: the hidden state after token t, the embedding of token t + 1
		let hn = refm::final_norm(ck, &x[..s * W])?;
		let h = refm::rmsnorm(&hn, W, &nh);
		let e = refm::rmsnorm(&refm::embed(ck, &ids[1..])?, W, &ne);
		let mut cat = Vec::with_capacity(s * 2 * W);
		for t in 0..s {
			cat.extend_from_slice(&e[t * W..(t + 1) * W]);
			cat.extend_from_slice(&h[t * W..(t + 1) * W]);
		}
		sfc.add(&cat, s, &sg);
		let x0 = la::linear(&cat, s, 2 * W, &fc, W);
		let (_, taps) = refm::forward(&layer, &x0, s);
		for (a, t) in st
			.iter_mut()
			.zip([&taps.proj, &taps.out, &taps.gu, &taps.down])
		{
			a.add(t, s, &sg);
		}
	}
	let (qfc, s_fc) = quantize(fc, W, sfc, &sg)?;
	write(&bound(wdir, crate::dec::MTP_FC), &qfc)?;
	let mut c = layer_files(
		layer,
		st,
		&|n| bound(wdir, &crate::dec::mtp_weight(n[1..].parse().unwrap())),
		accel,
	)?;
	c.s.insert(0, s_fc);
	c.save_as(&rec)?;
	log(&format!(
		"MTP block: calibrated and quantized in {:.0} s",
		t0.elapsed().as_secs_f64()
	));
	Ok(())
}

/// The lm_head as int8 per row (round to nearest): `wdir/w.lm_head.i8.in`
/// and `cal/head.cal` (the row scales).
pub fn quantize_head(
	ck: &Checkpoint,
	cal: &Path,
	wdir: &Path,
	log: &mut dyn FnMut(&str),
) -> Result<(), String> {
	let rec = cal.join("head.cal");
	if rec.exists() {
		return Ok(());
	}
	use std::io::Write;
	let v = ck.shape("lm_head.weight")?[0];
	let p = bound(wdir, crate::dec::LM_HEAD);
	let _ = std::fs::remove_file(&p);
	let mut f = std::io::BufWriter::new(
		std::fs::File::create(&p).map_err(|e| format!("{}: {e}", p.display()))?,
	);
	let mut sc = Vec::with_capacity(v);
	for r0 in (0..v).step_by(8192) {
		let r1 = (r0 + 8192).min(v);
		let (q, s) = gptq::quant_i8(&ck.rows("lm_head.weight", r0, r1)?, r1 - r0, W);
		f.write_all(&q.iter().map(|&x| x as u8).collect::<Vec<u8>>())
			.map_err(|e| e.to_string())?;
		sc.extend(s);
	}
	f.flush().map_err(|e| e.to_string())?;
	LayerCal {
		r: [0.0; 4],
		s: vec![sc],
	}
	.save_as(&rec)?;
	log("lm_head: int8");
	Ok(())
}

/// The gate rows of a delta-net layer as the graphs hold them: in_proj_a,
/// then in_proj_b, rotated, int8 per row with their scales.
fn gate_rows(ck: &Checkpoint, l: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
	let q = format!("model.language_model.layers.{l}.linear_attn");
	let mut ab = cat(
		ck.mat(&format!("{q}.in_proj_a.weight"), H, W)?,
		&ck.mat(&format!("{q}.in_proj_b.weight"), H, W)?,
	);
	gptq::rotate_rows(&mut ab, W, &signs(W));
	Ok(gptq::quant_i8(&ab, 2 * H, W))
}

fn mlp_c(
	ck: &Checkpoint,
	p: &str,
	s_gu: Vec<f32>,
	s_down: Vec<f32>,
) -> Result<crate::dec::MlpC, String> {
	Ok(crate::dec::MlpC {
		ln2: refm::norm_w(ck, &format!("{p}.post_attention_layernorm.weight"), W)?,
		s_gu,
		s_down,
	})
}

fn four(c: LayerCal, what: &str) -> Result<[Vec<f32>; 4], String> {
	<[Vec<f32>; 4]>::try_from(c.s).map_err(|_| format!("{what}: not a layer's calibration record"))
}

/// The constants of delta-net layer `l`'s decode graph.
pub fn delta_c(ck: &Checkpoint, cal: &Path, l: usize) -> Result<crate::dec::DeltaC, String> {
	let [s_proj, s_out, s_gu, s_down] = four(LayerCal::load(cal, l)?, "delta layer")?;
	let p = format!("model.language_model.layers.{l}");
	let q = format!("{p}.linear_attn");
	let (ab, s_ab) = gate_rows(ck, l)?;
	Ok(crate::dec::DeltaC {
		ln1: refm::norm_w(ck, &format!("{p}.input_layernorm.weight"), W)?,
		s_proj,
		s_out,
		ab,
		s_ab,
		conv: ck.mat(&format!("{q}.conv1d.weight"), 2 * refm::HK * D + H * D, 4)?,
		dtb: ck.vec(&format!("{q}.dt_bias"), H)?,
		acoef: ck
			.vec(&format!("{q}.A_log"), H)?
			.iter()
			.map(|a| -a.exp())
			.collect(),
		gnw: ck.vec(&format!("{q}.norm.weight"), D)?,
		mlp: mlp_c(ck, &p, s_gu, s_down)?,
	})
}

fn attn_c_of(ck: &Checkpoint, p: &str, s: [Vec<f32>; 4]) -> Result<crate::dec::AttnC, String> {
	let [s_proj, s_out, s_gu, s_down] = s;
	Ok(crate::dec::AttnC {
		ln1: refm::norm_w(ck, &format!("{p}.input_layernorm.weight"), W)?,
		qnw: refm::norm_w(ck, &format!("{p}.self_attn.q_norm.weight"), DA)?,
		knw: refm::norm_w(ck, &format!("{p}.self_attn.k_norm.weight"), DA)?,
		s_proj,
		s_out,
		mlp: mlp_c(ck, p, s_gu, s_down)?,
	})
}

/// The constants of attention layer `l`'s decode graph.
pub fn attn_c(ck: &Checkpoint, cal: &Path, l: usize) -> Result<crate::dec::AttnC, String> {
	attn_c_of(
		ck,
		&format!("model.language_model.layers.{l}"),
		four(LayerCal::load(cal, l)?, "attention layer")?,
	)
}

/// The constants of the MTP block's graph.
pub fn mtp_c(ck: &Checkpoint, cal: &Path) -> Result<crate::dec::MtpC, String> {
	let mut c = LayerCal::load_from(&cal.join("mtp.cal"))?;
	if c.s.len() != 5 {
		return Err("mtp.cal: not the MTP block's record".into());
	}
	let s_fc = c.s.remove(0);
	Ok(crate::dec::MtpC {
		ne: refm::norm_w(ck, "mtp.pre_fc_norm_embedding.weight", W)?,
		nh: refm::norm_w(ck, "mtp.pre_fc_norm_hidden.weight", W)?,
		norm: refm::norm_w(ck, "mtp.norm.weight", W)?,
		s_fc,
		layer: attn_c_of(ck, "mtp.layers.0", four(c, "mtp")?)?,
	})
}

/// The row scales of the int8 lm_head.
pub fn head_scales(cal: &Path) -> Result<Vec<f32>, String> {
	let mut c = LayerCal::load_from(&cal.join("head.cal"))?;
	if c.s.len() != 1 {
		return Err("head.cal: not the lm_head's record".into());
	}
	Ok(c.s.remove(0))
}

/// The constants of attention layer `l`'s prefill graphs.
pub fn attn_consts(
	ck: &Checkpoint,
	cal: &Path,
	l: usize,
) -> Result<crate::lean::AttnConsts, String> {
	let c = LayerCal::load(cal, l)?;
	let p = format!("model.language_model.layers.{l}");
	let r = c.r;
	let [s_proj, s_out, s_gu, s_down] = four(c, "attention layer")?;
	Ok(crate::lean::AttnConsts {
		ln1: refm::norm_w(ck, &format!("{p}.input_layernorm.weight"), W)?,
		ln2: refm::norm_w(ck, &format!("{p}.post_attention_layernorm.weight"), W)?,
		qnw: refm::norm_w(ck, &format!("{p}.self_attn.q_norm.weight"), DA)?,
		knw: refm::norm_w(ck, &format!("{p}.self_attn.k_norm.weight"), DA)?,
		s_proj,
		s_out,
		s_gu,
		s_down,
		r_proj: r[0],
		r_out: r[1],
		r_gu: r[2],
		r_down: r[3],
	})
}
