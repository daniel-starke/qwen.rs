//! The vision tower on the NPU (`vision = DIR` of a `.npu` file; the set
//! `cli BF16_DIR --npu-build` writes as `sets/vis`).
//!
//! The tower is cut at each block's attention (`npubuild::vis`): `layers +
//! 1` graphs over `s` patches each do everything that is per patch, and the
//! attention over all of an image's patches runs between two of them on
//! another device ([`VisAttn`]: the iGPU's in `cli`, else the cores). The
//! residual, the projections and the attention's output of the whole image
//! live in host buffers ([`NpuVision::encode`] copies a job's rows in and
//! out of them), so the attention device reads and writes them in place.

use std::path::Path;

use crate::ir::{f16_bits, f16_to_f32};
use crate::ze::{Graph, HostBuf, Npu};

/// `vis.cfg` of a tower set.
#[derive(Clone, Debug, Default)]
pub struct VisCfg {
	pub patch: usize,
	pub merge: usize,
	pub embd: usize,
	pub heads: usize,
	pub layers: usize,
	/// Width of an output token (the text model's).
	pub proj: usize,
	/// Rows of the learned position grid ([`NpuVision::pos_w`]).
	pub n_pos: usize,
	/// Patches per NPU job.
	pub s: usize,
	pub mean: [f32; 3],
	pub std: [f32; 3],
	/// The blocks whose outputs feed deepstack mergers (graph `l + 1` of
	/// each has an output `deep{j}`).
	pub deep: Vec<usize>,
}

impl VisCfg {
	pub fn parse(path: &Path) -> Result<VisCfg, String> {
		let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
		let mut c = VisCfg::default();
		for line in t
			.lines()
			.map(str::trim)
			.filter(|l| !l.is_empty() && !l.starts_with('#'))
		{
			let (k, v) = line
				.split_once('=')
				.ok_or(format!("{}: bad line {line}", path.display()))?;
			let (k, v) = (k.trim(), v.trim());
			let n = || {
				v.parse::<usize>()
					.map_err(|_| format!("{}: bad value in {line}", path.display()))
			};
			let t3 = || -> Result<[f32; 3], String> {
				let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
				<[f32; 3]>::try_from(f)
					.map_err(|_| format!("{}: bad value in {line}", path.display()))
			};
			match k {
				"patch" => c.patch = n()?,
				"merge" => c.merge = n()?,
				"embd" => c.embd = n()?,
				"heads" => c.heads = n()?,
				"layers" => c.layers = n()?,
				"proj" => c.proj = n()?,
				"npos" => c.n_pos = n()?,
				"s" => c.s = n()?,
				"mean" => c.mean = t3()?,
				"std" => c.std = t3()?,
				"deep" => {
					c.deep = v
						.split(',')
						.map(|x| x.trim().parse::<usize>())
						.collect::<Result<_, _>>()
						.map_err(|_| format!("{}: bad value in {line}", path.display()))?
				}
				_ => return Err(format!("{}: unknown key {k}", path.display())),
			}
		}
		let mm = c.merge * c.merge;
		if c.patch == 0
			|| mm == 0
			|| c.embd == 0
			|| c.heads == 0
			|| c.layers == 0
			|| c.proj == 0
			|| c.s == 0
			|| c.s % mm != 0
			|| c.embd % c.heads != 0
			|| c.deep.iter().any(|&l| l >= c.layers)
		{
			return Err(format!("{}: incomplete", path.display()));
		}
		Ok(c)
	}

	/// Features of one patch (`3 * patch * patch`).
	pub fn plen(&self) -> usize {
		3 * self.patch * self.patch
	}
}

/// The attention of one tower block over all `n` patches of an image.
///
/// `qkv` is `[n][3 * embd]` f16 (per patch q, then k, then v, each
/// `[heads][embd / heads]`, before RoPE), `hw` each patch's (row, column)
/// (the 2-D RoPE positions), `out` `[n][embd]` f16. Both buffers are page
/// aligned host memory that stays allocated at its address as long as the
/// engine lives (an implementation may map it once).
pub trait VisAttn {
	fn attend(
		&mut self,
		n: usize,
		qkv: &[u8],
		hw: &[(u32, u32)],
		out: &mut [u8],
	) -> Result<(), String>;
	/// What runs it, for the log.
	fn name(&self) -> String;
}

/// See the module docs.
pub struct NpuVision {
	pub cfg: VisCfg,
	/// The learned position grid, `[n_pos][embd]`.
	pub pos_w: Vec<f32>,
	npu: &'static Npu,
	/// the set's directory (the blobs are read again after [`Self::unload`])
	dir: std::path::PathBuf,
	/// the `layers + 1` graphs, bound to `job`; empty while unloaded
	graphs: Vec<Graph<'static>>,
	/// the graphs' blobs: the driver reads a blob after `pfnCreate2`
	/// returns, so each lives as long as its graph (dropped after them)
	blobs: Vec<Vec<u8>>,
	/// one job's rows: patch, pos, x, a, xo, qkv, emb, then each deepstack
	/// merger's
	job: Vec<HostBuf<'static>>,
	/// the image's rows: x `[n][embd]`, qkv `[n][3 embd]`, a `[n][embd]`
	img: Option<[HostBuf<'static>; 3]>,
	/// earlier, smaller ones (an attention device may still map them)
	old: Vec<HostBuf<'static>>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Io {
	Patch,
	Pos,
	X,
	A,
	Xo,
	Qkv,
	Emb,
	Deep(usize),
}

impl Io {
	/// Index into [`NpuVision::job`].
	fn at(self) -> usize {
		match self {
			Io::Patch => 0,
			Io::Pos => 1,
			Io::X => 2,
			Io::A => 3,
			Io::Xo => 4,
			Io::Qkv => 5,
			Io::Emb => 6,
			Io::Deep(j) => 7 + j,
		}
	}
}

fn f16_row(dst: &mut [u8], src: &[f32]) {
	for (d, &v) in dst.chunks_exact_mut(2).zip(src) {
		d.copy_from_slice(&f16_bits(v).to_le_bytes());
	}
}

impl NpuVision {
	/// The set in `dir` (`vis.cfg`, `pos.f32`, `vis{0..=layers}.blob`).
	pub fn load(npu: &'static Npu, dir: &str) -> Result<NpuVision, String> {
		let d = Path::new(dir);
		let cfg = VisCfg::parse(&d.join("vis.cfg"))?;
		let pf = d.join("pos.f32");
		let pos_w: Vec<f32> = std::fs::read(&pf)
			.map_err(|e| format!("{}: {e}", pf.display()))?
			.chunks_exact(4)
			.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
			.collect();
		if pos_w.len() != cfg.n_pos * cfg.embd {
			return Err(format!(
				"{}: {} values, expected {} x {}",
				pf.display(),
				pos_w.len(),
				cfg.n_pos,
				cfg.embd
			));
		}
		let (s, e) = (cfg.s, cfg.embd);
		let sizes = [
			s * cfg.plen(),
			s * e,
			s * e,
			s * e,
			s * e,
			s * 3 * e,
			s / (cfg.merge * cfg.merge) * cfg.proj,
		];
		let mut job = Vec::new();
		let deep = vec![s / (cfg.merge * cfg.merge) * cfg.proj; cfg.deep.len()];
		for n in sizes.into_iter().chain(deep) {
			let mut b = npu.alloc(n * 2)?;
			b.as_mut_slice().fill(0);
			job.push(b);
		}
		let mut v = NpuVision {
			cfg,
			pos_w,
			npu,
			dir: d.to_path_buf(),
			graphs: Vec::new(),
			blobs: Vec::new(),
			job,
			img: None,
			old: Vec::new(),
		};
		v.load_graphs()?;
		Ok(v)
	}

	/// Load, bind and initialize the graphs (no-op when loaded).
	fn load_graphs(&mut self) -> Result<(), String> {
		if !self.graphs.is_empty() {
			return Ok(());
		}
		let (cfg, job, d) = (&self.cfg, &self.job, &self.dir);
		let dir = d.display();
		let mut graphs = Vec::with_capacity(cfg.layers + 1);
		let mut blobs = Vec::with_capacity(cfg.layers + 1);
		for i in 0..=cfg.layers {
			let f = d.join(format!("vis{i}.blob"));
			let b = std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?;
			graphs.push(self.npu.load(&b, 0)?);
			blobs.push(b);
		}
		for (i, g) in graphs.iter_mut().enumerate() {
			// (the driver names an output after the op that computes it:
			// outputs go by their order, `xo` then `qkv`, or `emb`)
			let mut outs = if i < cfg.layers {
				vec![Io::Xo, Io::Qkv]
			} else {
				vec![Io::Emb]
			};
			outs.extend(
				cfg.deep
					.iter()
					.enumerate()
					.filter(|&(_, &l)| l + 1 == i)
					.map(|(j, _)| Io::Deep(j)),
			);
			let mut outs = outs.into_iter();
			for (ai, a) in g.args.clone().iter().enumerate() {
				let io = match a.name.as_str() {
					_ if a.is_output => outs
						.next()
						.ok_or(format!("{dir}/vis{i}.blob: more outputs than expected"))?,
					"patch" => Io::Patch,
					"pos" => Io::Pos,
					"x" => Io::X,
					"a" => Io::A,
					n => return Err(format!("{dir}/vis{i}.blob: unknown argument {n}")),
				};
				if a.bytes != job[io.at()].len() {
					return Err(format!(
						"{dir}/vis{i}.blob: argument {} has {} bytes, expected {}",
						a.name,
						a.bytes,
						job[io.at()].len()
					));
				}
				g.bind(ai, &job[io.at()])?;
			}
			g.initialize()?;
		}
		self.graphs = graphs;
		self.blobs = blobs;
		Ok(())
	}

	/// Destroy the graphs and keep the buffers: an initialized graph holds
	/// driver memory every other NPU user of the process shares
	/// (`0x70000002` when it runs out). The next encode loads them again.
	pub fn unload(&mut self) {
		self.graphs.clear();
		self.blobs.clear();
	}

	fn buf(&mut self, io: Io) -> &mut [u8] {
		self.job[io.at()].as_mut_slice()
	}

	/// [`Self::encode`], plus each deepstack merger's rows (`cfg.deep`
	/// order, `[n / merge^2][proj]` each).
	pub fn encode_deep(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
		attn: &mut dyn VisAttn,
	) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
		let mut deep = vec![Vec::new(); self.cfg.deep.len()];
		let out = self.run(patches, pos, hw, attn, &mut deep)?;
		Ok((out, deep))
	}

	/// One image through the tower: `patches` `[n][3 * patch * patch]`
	/// (normalized pixels, merge-block order), `pos` `[n][embd]` (the
	/// position embedding of each patch), `hw` each patch's (row, column)
	/// -> `[n / merge^2][proj]`.
	pub fn encode(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
		attn: &mut dyn VisAttn,
	) -> Result<Vec<f32>, String> {
		let mut deep = vec![Vec::new(); self.cfg.deep.len()];
		self.run(patches, pos, hw, attn, &mut deep)
	}

	fn run(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
		attn: &mut dyn VisAttn,
		deep: &mut [Vec<f32>],
	) -> Result<Vec<f32>, String> {
		let c = self.cfg.clone();
		let (s, e, pl, mm) = (c.s, c.embd, c.plen(), c.merge * c.merge);
		let n = hw.len();
		if n == 0 || !n.is_multiple_of(mm) || patches.len() != n * pl || pos.len() != n * e {
			return Err(format!(
				"vision: {n} patches with {} features and {} position values",
				patches.len(),
				pos.len()
			));
		}
		let timing = std::env::var_os("QWEN_NPU_VIS_TIMING").is_some();
		let t0 = std::time::Instant::now();
		if self.graphs.is_empty() {
			self.load_graphs()?;
			if timing {
				eprintln!(
					"  [npu] vision: graphs loaded in {:.0} ms",
					t0.elapsed().as_secs_f64() * 1e3
				);
			}
		}
		let chunks = n.div_ceil(s);
		if self
			.img
			.as_ref()
			.is_none_or(|b| b[0].len() < chunks * s * e * 2)
		{
			// whole jobs (a job's rows are copied without a tail case), and
			// room to spare: the buffers stay for the engine's life
			let rows = (chunks * s).next_power_of_two().max(1024);
			let a = |k: usize| -> Result<HostBuf<'static>, String> {
				let mut b = self.npu.alloc(rows * k * 2)?;
				b.as_mut_slice().fill(0);
				Ok(b)
			};
			let new = [a(e)?, a(3 * e)?, a(e)?];
			self.old.extend(self.img.replace(new).into_iter().flatten());
		}
		let (mut t_npu, mut t_att) = (0.0, 0.0);
		let mut out = vec![0f32; n / mm * c.proj];
		for d in deep.iter_mut() {
			*d = vec![0f32; n / mm * c.proj];
		}
		for l in 0..=c.layers {
			if l > 0 {
				let t = std::time::Instant::now();
				let [_, qkv, a] = self.img.as_mut().unwrap();
				attn.attend(
					n,
					&qkv.as_slice()[..n * 3 * e * 2],
					hw,
					&mut a.as_mut_slice()[..n * e * 2],
				)?;
				t_att += t.elapsed().as_secs_f64();
			}
			let t = std::time::Instant::now();
			for ci in 0..chunks {
				let (r0, real) = (ci * s, (n - ci * s).min(s));
				if l == 0 {
					for (io, src, k) in [(Io::Patch, patches, pl), (Io::Pos, pos, e)] {
						let b = self.buf(io);
						b.fill(0);
						f16_row(&mut b[..real * k * 2], &src[r0 * k..(r0 + real) * k]);
					}
				} else {
					for (io, k) in [(Io::X, 0usize), (Io::A, 2)] {
						let src: *const [u8] =
							&self.img.as_ref().unwrap()[k].as_slice()[r0 * e * 2..(r0 + s) * e * 2];
						// SAFETY: the image's rows and a job's are distinct allocations
						self.buf(io).copy_from_slice(unsafe { &*src });
					}
				}
				self.graphs[l].run()?;
				if l < c.layers {
					for (io, k, w) in [(Io::Xo, 0usize, e), (Io::Qkv, 1, 3 * e)] {
						let src: *const [u8] = &*self.buf(io);
						// SAFETY: as above
						self.img.as_mut().unwrap()[k].as_mut_slice()[r0 * w * 2..(r0 + s) * w * 2]
							.copy_from_slice(unsafe { &*src });
					}
				} else {
					let (t0, tn) = (r0 / mm, real / mm);
					let b = &self.job[Io::Emb.at()].as_slice()[..tn * c.proj * 2];
					for (d, h) in out[t0 * c.proj..(t0 + tn) * c.proj]
						.iter_mut()
						.zip(b.chunks_exact(2))
					{
						*d = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
					}
				}
				for (j, &dl) in c.deep.iter().enumerate() {
					if dl + 1 != l {
						continue;
					}
					let (t0, tn) = (r0 / mm, real / mm);
					let b = &self.job[Io::Deep(j).at()].as_slice()[..tn * c.proj * 2];
					for (d, h) in deep[j][t0 * c.proj..(t0 + tn) * c.proj]
						.iter_mut()
						.zip(b.chunks_exact(2))
					{
						*d = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
					}
				}
			}
			// (one list at a time: the engine's own lists leave the driver's
			// descriptor heap no room for one per graph)
			self.graphs[l].release();
			t_npu += t.elapsed().as_secs_f64();
		}
		if timing {
			eprintln!(
				"  [npu] vision: {n} patches ({} tokens) in {:.0} ms: {} NPU jobs {:.0} ms, attention ({}) {:.0} ms",
				n / mm,
				t0.elapsed().as_secs_f64() * 1e3,
				chunks * (c.layers + 1),
				t_npu * 1e3,
				attn.name(),
				t_att * 1e3
			);
		}
		Ok(out)
	}
}

#[cfg(test)]
mod tests {
	#[test]
	fn tower_config_parses() {
		let d = std::env::temp_dir().join(format!("qwen-rs-viscfg-{}", std::process::id()));
		std::fs::create_dir_all(&d).unwrap();
		let f = d.join("vis.cfg");
		std::fs::write(&f, "# a tower\npatch = 16\nmerge = 2\nembd = 1152\nheads = 16\nlayers = 27\nproj = 5120\nnpos = 2304\ns = 256\nmean = 0.5,0.5,0.5\nstd = 0.5, 0.25, 0.5\n").unwrap();
		let c = super::VisCfg::parse(&f).unwrap();
		assert_eq!(
			(c.patch, c.merge, c.embd, c.heads, c.layers, c.proj, c.n_pos, c.s),
			(16, 2, 1152, 16, 27, 5120, 2304, 256)
		);
		assert_eq!((c.std, c.plen()), ([0.5, 0.25, 0.5], 768));
		// a job must hold whole tokens; an unknown key is an error
		std::fs::write(
			&f,
			"patch = 16\nmerge = 2\nembd = 8\nheads = 2\nlayers = 1\nproj = 8\nnpos = 4\ns = 6\n",
		)
		.unwrap();
		assert!(super::VisCfg::parse(&f).is_err());
		std::fs::write(&f, "patch = 16\nwidth = 3\n").unwrap();
		assert!(super::VisCfg::parse(&f).is_err());
		let _ = std::fs::remove_dir_all(&d);
	}
}
