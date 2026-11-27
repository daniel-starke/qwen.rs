//! The VAE's 3x3 stride-1 convolutions on the NPU (`image.npu`'s `vae`
//! graphs, `npubuild::qvae`): f16 over tiles of `T x T` output pixels.
//!
//! A convolution runs by bands of jobs, two in flight: the cores build band
//! `b + 1`'s halo tiles (zero padding, upsampling, RMS norm + SiLU) and
//! write band `b - 1`'s outputs back (bias added on the NPU, a residual on
//! the cores) while the NPU runs band `b`. The weights are f16 copies in NPU
//! host memory, made at load from the checkpoint (~0.4 GB, decoder and
//! encoder).

use super::vae::Map;
use crate::ops::Ptr;
use npu::lock::{NpuLock, SharedNpuLock};
use npu::ze::{Batch, Graph, HostBuf, Npu};
use npubuild::qvae::{self, T};
use npubuild::st::Checkpoint;
use std::collections::HashMap;
use std::path::Path;

/// What a convolution's input tiles are made from.
pub enum Src<'a> {
	/// The map as it is.
	Map(&'a Map),
	/// `silu(rms_ch(map) * gamma)` (`gamma` already `x sqrt(c)`).
	NormSilu(&'a Map, &'a [f32]),
	/// The nearest-neighbor 2x upsampling of the map.
	Up(&'a Map),
}

impl Src<'_> {
	fn map(&self) -> &Map {
		match self {
			Src::Map(m) | Src::NormSilu(m, _) | Src::Up(m) => m,
		}
	}
}

struct Kind {
	g: Graph<'static>,
	/// Tiles per job.
	n: usize,
	/// Argument indices of `x`, `w`, `b`, `y`.
	ix: usize,
	iw: usize,
	ib: usize,
	iy: usize,
}

pub struct NpuVae {
	npu: &'static Npu,
	kinds: HashMap<(usize, usize), Kind>,
	/// Per convolution: (weight [co][ci][3][3], bias [co]) as f16.
	w: HashMap<String, (HostBuf<'static>, HostBuf<'static>)>,
	/// Two bands' input tiles and outputs (grown on demand).
	xb: [Option<HostBuf<'static>>; 2],
	yb: [Option<HostBuf<'static>>; 2],
	/// Seconds the NPU was waited for, and on the cores, since the last
	/// report.
	pub t_wait: f64,
	pub t_cores: f64,
	_lock: SharedNpuLock,
}

// SAFETY: as `NpuDit`: used from one thread at a time.
unsafe impl Send for NpuVae {}
unsafe impl Sync for NpuVae {}

/// Bytes of a band's input tiles (two bands are in flight).
const BAND_BYTES: usize = 48 << 20;

impl NpuVae {
	/// The graphs in `graphs` (`c{ci}_{co}.blob`) and the decoder's weights
	/// from the checkpoint directory `dir`.
	pub fn load(graphs: &Path, dir: &Path) -> Result<NpuVae, String> {
		let lock = NpuLock::shared()?;
		let npu = crate::npuhost::npu()?;
		let ck = Checkpoint::open_dir(&dir.join("vae"))?;
		let mut kinds = HashMap::new();
		let mut w = HashMap::new();
		let mut names: Vec<String> = ck
			.names()
			.filter(|n| ck.shape(n).is_ok_and(|s| qvae::on_npu(n, s)))
			.map(str::to_string)
			.collect();
		names.sort();
		for n in names {
			let s = ck.shape(&n)?.to_vec();
			let (co, ci) = (s[0], s[1]);
			if let std::collections::hash_map::Entry::Vacant(e) = kinds.entry((ci, co)) {
				let f = graphs.join(format!("{}.blob", qvae::kind(ci, co)));
				if !f.is_file() {
					// (a set built before this kind was: the cores run it)
					continue;
				}
				let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
				let mut g = npu.load(&b, 0)?;
				g.initialize()?;
				let at = |name: &str, out: bool| {
					g.args
						.iter()
						.position(|a| a.is_output == out && (out || a.name == name))
						.ok_or(format!("{}: no argument {name}", f.display()))
				};
				let (ix, iw, ib, iy) = (
					at("x", false)?,
					at("w", false)?,
					at("b", false)?,
					at("y", true)?,
				);
				let n = g.args[ix].bytes / ((T + 2) * (T + 2) * ci * 2);
				if n * (T + 2) * (T + 2) * ci * 2 != g.args[ix].bytes
					|| g.args[iy].bytes != n * T * T * co * 2
				{
					return Err(format!("{}: unexpected shapes", f.display()));
				}
				e.insert(Kind {
					g,
					n,
					ix,
					iw,
					ib,
					iy,
				});
			}
			let stem = n.trim_end_matches(".weight");
			let to_buf = |v: &[f32]| -> Result<HostBuf<'static>, String> {
				let mut hb = npu.alloc(v.len() * 2)?;
				let s = hb.as_mut_slice();
				// SAFETY: host buffers are 4 KiB aligned; len = 2 v.len()
				let d =
					unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u16, v.len()) };
				quant::f32_to_f16_slice(v, d);
				Ok(hb)
			};
			let wb = to_buf(&ck.f32(&n)?)?;
			let bb = to_buf(&ck.f32(&format!("{stem}.bias"))?)?;
			w.insert(stem.to_string(), (wb, bb));
		}
		Ok(NpuVae {
			npu,
			kinds,
			w,
			xb: [None, None],
			yb: [None, None],
			t_wait: 0.0,
			t_cores: 0.0,
			_lock: lock,
		})
	}

	/// Whether conv `name` (its prefix, without `.weight`) runs here.
	pub fn has(&self, name: &str) -> bool {
		self.w.contains_key(name)
	}

	/// Bytes of the weights in NPU host memory.
	pub fn resident_bytes(&self) -> usize {
		self.w.values().map(|(a, b)| a.len() + b.len()).sum()
	}

	/// Free the band buffers (the weights stay).
	pub fn release_buffers(&mut self) {
		self.xb = [None, None];
		self.yb = [None, None];
	}

	/// 3x3 conv `name` (padding 1) of `src` (an `Up` source's output is
	/// twice its map's size), plus `add` (a map of the output's shape).
	pub fn conv(&mut self, name: &str, src: &Src, add: Option<&Map>) -> Result<Map, String> {
		let m = src.map();
		let (ho, wo) = match src {
			Src::Up(_) => (2 * m.h, 2 * m.w),
			_ => (m.h, m.w),
		};
		let ci = m.c;
		let (wb, bb) = self.w.get(name).ok_or(format!("npu vae: no conv {name}"))?;
		let co = bb.len() / 2;
		if wb.len() != co * ci * 9 * 2 {
			return Err(format!(
				"npu vae: {name} takes {} channels, not {ci}",
				wb.len() / 18 / co
			));
		}
		if let Some(a) = add {
			assert_eq!((a.h, a.w, a.c), (ho, wo, co));
		}
		let kd = self
			.kinds
			.get_mut(&(ci, co))
			.ok_or(format!("npu vae: no graph for {ci} -> {co}"))?;
		let (tx, ty) = (wo.div_ceil(T), ho.div_ceil(T));
		let tiles = tx * ty;
		let n = kd.n;
		let xt = (T + 2) * (T + 2) * ci; // halves per input tile
		let yt = T * T * co;
		let jobs_band = (BAND_BYTES / (n * xt * 2)).max(1);
		let tiles_band = jobs_band * n;
		let bands = tiles.div_ceil(tiles_band);
		let need_x = tiles_band.min(tiles.next_multiple_of(n)) * xt * 2;
		let need_y = tiles_band.min(tiles.next_multiple_of(n)) * yt * 2;
		for s in 0..2 {
			if self.xb[s].as_ref().is_none_or(|b| b.len() < need_x) {
				self.xb[s] = None;
				self.xb[s] = Some(self.npu.alloc(need_x)?);
			}
			if self.yb[s].as_ref().is_none_or(|b| b.len() < need_y) {
				self.yb[s] = None;
				self.yb[s] = Some(self.npu.alloc(need_y)?);
			}
		}
		let mut out = Map {
			h: ho,
			w: wo,
			c: co,
			d: quant::uninit_vec(ho * wo * co),
		};
		let gamma: Option<&[f32]> = match src {
			Src::NormSilu(_, g) => Some(g),
			_ => None,
		};
		let up = matches!(src, Src::Up(_));
		let npu = self.npu;
		let mut pend: [Option<Batch<'static>>; 2] = [None, None];
		let (mut t_wait, mut t_cores) = (0.0, 0.0);
		for b in 0..=bands {
			// build band b (slot b % 2), submit it, then finish band b - 1
			if b < bands {
				let t = std::time::Instant::now();
				let t0 = b * tiles_band;
				let nt = tiles_band.min(tiles - t0);
				let jobs = nt.div_ceil(n);
				let xp = self.xb[b % 2].as_ref().unwrap().ptr() as usize;
				quant::par_for(jobs * n, |k| {
					// SAFETY: task k owns input tile k of the slot
					let dst =
						unsafe { std::slice::from_raw_parts_mut((xp as *mut u16).add(k * xt), xt) };
					if k >= nt {
						dst.fill(0);
						return;
					}
					let tile = t0 + k;
					build_tile(dst, m, gamma, up, ho, wo, (tile / tx) * T, (tile % tx) * T);
				});
				t_cores += t.elapsed().as_secs_f64();
				let (xs, ys) = (
					self.xb[b % 2].as_ref().unwrap(),
					self.yb[b % 2].as_ref().unwrap(),
				);
				let (ix, iw, ib, iy) = (kd.ix, kd.iw, kd.ib, kd.iy);
				let t = std::time::Instant::now();
				let mut bt = npu.batch_each(&mut kd.g, jobs, &mut |g, j| {
					g.bind_at(ix, xs, j * n * xt * 2)?;
					g.bind_at(iy, ys, j * n * yt * 2)?;
					g.bind(iw, wb)?;
					g.bind(ib, bb)?;
					Ok(())
				})?;
				bt.submit()?;
				t_wait += t.elapsed().as_secs_f64();
				pend[b % 2] = Some(bt);
			}
			if b > 0 {
				let p = b - 1;
				let t = std::time::Instant::now();
				if let Some(mut bt) = pend[p % 2].take() {
					bt.wait()?;
				}
				t_wait += t.elapsed().as_secs_f64();
				let t = std::time::Instant::now();
				let t0 = p * tiles_band;
				let nt = tiles_band.min(tiles - t0);
				let ys = self.yb[p % 2].as_ref().unwrap().ptr() as usize;
				let op = Ptr(out.d.as_mut_ptr());
				quant::par_for(nt, |k| {
					let op = op;
					let tile = t0 + k;
					let (y0, x0) = ((tile / tx) * T, (tile % tx) * T);
					// SAFETY: task k owns output tile k of the slot and its
					// pixels of `out`
					let src =
						unsafe { std::slice::from_raw_parts((ys as *const u16).add(k * yt), yt) };
					for r in 0..T.min(ho - y0) {
						let wn = T.min(wo - x0);
						let o0 = ((y0 + r) * wo + x0) * co;
						let dst = unsafe { std::slice::from_raw_parts_mut(op.0.add(o0), wn * co) };
						quant::f16_to_f32_slice(&src[r * T * co..(r * T + wn) * co], dst);
						if let Some(a) = add {
							dst.iter_mut()
								.zip(&a.d[o0..o0 + wn * co])
								.for_each(|(d, &s)| *d += s);
						}
					}
				});
				t_cores += t.elapsed().as_secs_f64();
			}
		}
		self.t_wait += t_wait;
		self.t_cores += t_cores;
		Ok(out)
	}
}

/// One input tile (`(T + 2)^2` pixels of `m.c` halves): output pixels `(y0
/// .., x0 ..)`'s taps, zero outside the `ho x wo` output map.
#[allow(clippy::too_many_arguments)]
fn build_tile(
	dst: &mut [u16],
	m: &Map,
	gamma: Option<&[f32]>,
	up: bool,
	ho: usize,
	wo: usize,
	y0: usize,
	x0: usize,
) {
	let c = m.c;
	let mut row = vec![0f32; c];
	for r in 0..T + 2 {
		for q in 0..T + 2 {
			let d = &mut dst[(r * (T + 2) + q) * c..(r * (T + 2) + q + 1) * c];
			let (y, x) = ((y0 + r).wrapping_sub(1), (x0 + q).wrapping_sub(1));
			if y >= ho || x >= wo {
				d.fill(0);
				continue;
			}
			let (sy, sx) = if up { (y / 2, x / 2) } else { (y, x) };
			let s = &m.d[(sy * m.w + sx) * c..(sy * m.w + sx + 1) * c];
			match gamma {
				None => quant::f32_to_f16_slice(s, d),
				Some(g) => {
					let nrm = s.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt() as f32;
					let f = 1.0 / nrm.max(1e-12);
					for ((o, &v), &gg) in row.iter_mut().zip(s).zip(g) {
						*o = crate::ops::silu(v * f * gg);
					}
					quant::f32_to_f16_slice(&row, d);
				}
			}
		}
	}
}
