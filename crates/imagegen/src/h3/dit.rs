//! The transformer (`MiniMaxH3ImageTransformer2DModel`, 33.1B), f32 on the
//! cores, weights converted from BF16 one layer at a time.
//!
//! - Text rows: `context_embedder` (Linear + bias) of the text encoder's
//!   rows, then the token refiner (plain pre-norm blocks, full attention
//!   over the text, no RoPE, no modulation) and its final RMSNorm. None of
//!   it depends on the step, so [`Dit::text_rows`] runs once per job.
//! - Image rows: `proj_in` (Linear + bias) of the packed latents.
//! - 50 blocks over the whole packed sequence with **full** attention:
//!   `x += gate_msa * attn(RMSNorm(x) * (1 + scale_msa) + shift_msa)`, the
//!   same for the SwiGLU MLP. The six modulation rows come per row from a
//!   `(timestep, modality)` table: each block's own `adaln_proj` (Linear
//!   2688 -> 3 x 6 x 5376 + bias) of `SiLU(temb)`; the gates are not
//!   squashed. q / k get a per-head RMSNorm and a rotate-half RoPE over the
//!   first `6 * rope_freq` channels of each head (axes t, h, w, one shared
//!   frequency list), the rest pass through.
//! - Output: RMSNorm * (1 + scale) + shift per timestep (`norm_out`), then
//!   the step's own head (`pdd_proj_out[i]`, Linear + bias) on the image
//!   rows.
//!
//! The timestep embedding is `[cos, sin]` of the **unscaled** `t` in [0, 1]
//! (`t = 1 - sigma`, 1 = clean) through `linear_1 -> SiLU -> linear_2`, with
//! biases.
//!
//! Resident ([`Dit::load`]): the matmuls quantized once, and the
//! modulation as tables: it depends only on the timestep, and a job uses
//! only the schedule's timesteps and the references' [`REF_T`], so the 13B
//! parameters of the AdaLN projections become 7 x 50 x 3 x 6 x 5376 f32
//! (~135 MB), `norm_out`'s likewise. A timestep outside the tables falls
//! back to the projections, read from the checkpoint.

use super::config::{DitCfg, MODALITIES, REF_T};
use super::layout::Layout;
use crate::ops::{self, Lin, WFmt};
use crate::wcache;
use npubuild::st::Checkpoint;
use std::path::Path;

/// The NPU build's calibration (`npubuild::h3img::Cal`): while set, the
/// blocks record the largest magnitude of their input (block 0), of their
/// rotated QKV, out and gate / up inputs, of their output, and of the MLP's
/// gate and hidden state.
pub static CAL: std::sync::Mutex<Option<npubuild::h3img::Cal>> = std::sync::Mutex::new(None);

/// The NPU build's GPTQ calibration: while set, every block adds its
/// QKV, out, gate / up and down inputs to its Hessians
/// (`npubuild::h3img::HNAMES`).
pub static HESS: std::sync::Mutex<Option<Vec<[npubuild::gptq::Gram; 4]>>> =
	std::sync::Mutex::new(None);

/// Add the rows `x` to block `l`'s Hessian `i`.
fn hess_add(l: usize, i: usize, x: &[f32]) {
	if let Some(h) = HESS.lock().unwrap().as_mut() {
		h[l][i].add(x);
	}
}

fn cal_on() -> bool {
	CAL.lock().unwrap().is_some()
}

/// Record `v` as block `l`'s statistic `i` (`l` = `None`: the input).
fn cal_max(l: Option<usize>, i: usize, v: f32) {
	if let Some(c) = CAL.lock().unwrap().as_mut() {
		match l {
			None => c.xin = c.xin.max(v),
			Some(l) => {
				if c.blocks.len() <= l {
					c.blocks.resize(l + 1, [0.0; 6]);
				}
				c.blocks[l][i] = c.blocks[l][i].max(v);
			}
		}
	}
}

fn max_abs(x: &[f32]) -> f32 {
	let m = std::sync::Mutex::new(0f32);
	const C: usize = 1 << 16;
	quant::par_for(x.len().div_ceil(C), |i| {
		let v = x[i * C..((i + 1) * C).min(x.len())]
			.iter()
			.fold(0f32, |a, &b| a.max(b.abs()));
		let mut g = m.lock().unwrap();
		*g = g.max(v);
	});
	m.into_inner().unwrap()
}

/// The largest magnitude of the rows of `x` (`k` wide) after the NPU
/// graphs' rotation.
fn rot_max(x: &[f32], k: usize) -> f32 {
	let mut r = x.to_vec();
	npubuild::gptq::rotate_rows_b(&mut r, k, &npubuild::gptq::signs(k), npubuild::qimage::ROT);
	max_abs(&r)
}

/// The attention on the cores in int8 (`quant::K8`: Q.K and P V on
/// AVX-VNNI) when `QWEN_H3_ATTN_I8=1`; f32 by default: on the s512
/// reference int8 doubled step 2's velocity error on the NPU backend
/// (6.6e-2 -> 1.1e-1) and moved the CPU backend's free-running image from
/// 23.4 to 18.7 dB, for 5 % of a 4 MP job on the NPU backend.
pub fn attn_i8() -> bool {
	std::env::var("QWEN_H3_ATTN_I8").is_ok_and(|v| v == "1")
}

/// Full (bidirectional) attention of `n` rows of q / k / v (`[n][heads *
/// hd]`) into `out`, in int8 when `int8` (and [`attn_i8`]).
pub fn attention_rows(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	n: usize,
	heads: usize,
	hd: usize,
	int8: bool,
	out: &mut [f32],
) {
	let scale = 1.0 / (hd as f32).sqrt();
	if int8 && attn_i8() && quant::K8::supported(hd) {
		let mut kh = vec![0u16; k.len()];
		let mut vh = vec![0u16; v.len()];
		quant::f32_to_f16_slice(k, &mut kh);
		quant::f32_to_f16_slice(v, &mut vh);
		let mut k8 = quant::K8::default();
		k8.pack(&kh, n, heads, hd);
		k8.pack_v(&vh);
		quant::attn_block_rows_k8(q, &kh, &k8, &vh, n, 0, heads, hd, scale, n, out);
	} else {
		quant::attn_block(q, k, v, n, 0, heads, heads, hd, scale, out);
	}
}

pub struct Dit {
	pub cfg: DitCfg,
	pub ck: Checkpoint,
	/// The quantized weights and tables ([`Self::load`]); `None` = the
	/// reference, which converts each block from BF16 per use.
	res: Option<Resident>,
}

/// What [`Dit::load`] keeps.
struct Resident {
	blocks: Vec<BlockW>,
	refiner: Vec<BlockW>,
	ctx: Lin,
	ctx_b: Vec<f32>,
	proj_in: Lin,
	proj_in_b: Vec<f32>,
	refiner_norm: Vec<f32>,
	out_norm: Vec<f32>,
	/// `pdd_proj_out.i` weight and bias per step.
	heads: Vec<(Vec<f32>, Vec<f32>)>,
	/// The timesteps the tables hold.
	ts: Vec<f32>,
	/// Per block `[ts][MODALITIES][6][hidden]`.
	adaln: Vec<Vec<f32>>,
	/// `norm_out`'s `[ts][2][hidden]` (shift, scale).
	out_ss: Vec<f32>,
}

/// A block's weights: resident, or loaded for one use.
enum BlockRef<'a> {
	Res(&'a BlockW),
	Own(Box<BlockW>),
}

impl std::ops::Deref for BlockRef<'_> {
	type Target = BlockW;
	fn deref(&self) -> &BlockW {
		match self {
			BlockRef::Res(b) => b,
			BlockRef::Own(b) => b,
		}
	}
}

/// `(cos, sin)` per row over the `3 * rope_freq` rotary frequencies
/// (t, h, w); channel `i < 3 * rope_freq` and `i + 3 * rope_freq` share
/// angle `i`.
pub struct Rope {
	pub cos: Vec<f32>,
	pub sin: Vec<f32>,
	pub half: usize,
}

impl Rope {
	pub fn new(cfg: &DitCfg, pos: &[[f32; 3]]) -> Rope {
		let f = cfg.rope_freq;
		let theta = cfg.rope_theta as f32;
		// torch: 1 / theta ** (arange(0, 2f, 2) / 2f), in f32
		let inv: Vec<f32> = (0..f)
			.map(|i| 1.0 / theta.powf((2 * i) as f32 / (2 * f) as f32))
			.collect();
		let half = 3 * f;
		let mut cos = vec![0f32; pos.len() * half];
		let mut sin = vec![0f32; pos.len() * half];
		for (r, p) in pos.iter().enumerate() {
			for ax in 0..3 {
				for (i, &w) in inv.iter().enumerate() {
					let a = p[ax] * w;
					cos[r * half + ax * f + i] = a.cos();
					sin[r * half + ax * f + i] = a.sin();
				}
			}
		}
		Rope { cos, sin, half }
	}

	/// Rotate every head of rows `[n][heads * hd]`.
	pub fn apply(&self, x: &mut [f32], heads: usize, hd: usize) {
		let half = self.half;
		ops::rows_mut_idx(x, heads * hd, |r, row| {
			let (c, s) = (
				&self.cos[r * half..(r + 1) * half],
				&self.sin[r * half..(r + 1) * half],
			);
			for h in 0..heads {
				let v = &mut row[h * hd..h * hd + 2 * half];
				for i in 0..half {
					let (a, b) = (v[i], v[i + half]);
					v[i] = a * c[i] - b * s[i];
					v[i + half] = b * c[i] + a * s[i];
				}
			}
		});
	}
}

/// One block's weights.
pub struct BlockW {
	pub n1: Vec<f32>,
	pub n2: Vec<f32>,
	pub q: Lin,
	pub k: Lin,
	pub v: Lin,
	pub nq: Vec<f32>,
	pub nk: Vec<f32>,
	pub o: Lin,
	pub up: Lin,
	pub gate: Lin,
	pub down: Lin,
}

impl BlockW {
	fn lins(&self) -> [&Lin; 7] {
		[
			&self.q, &self.k, &self.v, &self.o, &self.up, &self.gate, &self.down,
		]
	}

	fn bytes(&self) -> usize {
		self.lins().iter().map(|l| l.bytes()).sum()
	}

	fn save(&self, w: &mut wcache::Writer) -> Result<(), String> {
		for v in [&self.n1, &self.n2, &self.nq, &self.nk] {
			w.f32s(v)?;
		}
		self.lins().iter().try_for_each(|l| w.lin(l))
	}

	fn read(r: &mut wcache::Reader) -> Result<BlockW, String> {
		let (n1, n2, nq, nk) = (r.f32s()?, r.f32s()?, r.f32s()?, r.f32s()?);
		Ok(BlockW {
			n1,
			n2,
			nq,
			nk,
			q: r.lin()?,
			k: r.lin()?,
			v: r.lin()?,
			o: r.lin()?,
			up: r.lin()?,
			gate: r.lin()?,
			down: r.lin()?,
		})
	}
}

/// `x[s][k] w^T + b`.
pub fn linear_b(x: &[f32], s: usize, k: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
	let n = b.len();
	let mut y = ops::linear(x, s, k, w, n);
	ops::rows_mut(&mut y, n, |r| {
		r.iter_mut().zip(b).for_each(|(v, &bb)| *v += bb)
	});
	y
}

/// RMSNorm of every row of `x`, then `* (1 + scale) + shift` with the row's
/// modulation: table entry `idx[r]` (`[entries][6][hidden]`), parts `js`
/// (scale) and `js - 1` (shift).
fn rms_mod(x: &[f32], w: &[f32], eps: f32, table: &[f32], idx: &[u16], js: usize) -> Vec<f32> {
	let d = w.len();
	let mut h = x.to_vec();
	ops::rows_mut_idx(&mut h, d, |r, row| {
		ops::rms_row(row, w, eps, false);
		let e = &table[idx[r] as usize * 6 * d..];
		let (sh, sc) = (&e[(js - 1) * d..js * d], &e[js * d..(js + 1) * d]);
		for ((v, &a), &b) in row.iter_mut().zip(sc).zip(sh) {
			*v = *v * (1.0 + a) + b;
		}
	});
	h
}

/// `x += gate * y` with the row's gate (table part `j`).
fn add_gate(x: &mut [f32], y: &[f32], table: &[f32], idx: &[u16], j: usize, d: usize) {
	ops::rows_mut_idx(x, d, |r, row| {
		let g = &table[idx[r] as usize * 6 * d + j * d..][..d];
		for ((a, &b), &gg) in row.iter_mut().zip(&y[r * d..(r + 1) * d]).zip(g) {
			*a += gg * b;
		}
	});
}

impl Dit {
	pub fn open(dir: &Path, cfg: &DitCfg) -> Result<Dit, String> {
		let ck = Checkpoint::open_dir(&dir.join("transformer_diffusers"))?;
		for n in [
			"transformer_blocks.0.adaln_proj.linear.weight",
			"token_refiner.final_norm.weight",
			"pdd_proj_out.0.weight",
		] {
			if !ck.has(n) {
				return Err(format!("transformer: unexpected tensor layout (no {n})"));
			}
		}
		Ok(Dit {
			cfg: cfg.clone(),
			ck,
			res: None,
		})
	}

	/// [`Self::open`] with every matmul converted to `fmt` once and kept,
	/// and the modulation tables of timesteps `ts` (and [`REF_T`]); through
	/// the weight cache file `file` when given (read when it was made from
	/// this checkpoint, `fmt` and `ts`, else written after the load).
	/// `progress(block)` after each transformer block. `WFmt::F32` keeps
	/// nothing (the reference).
	pub fn load(
		dir: &Path,
		cfg: &DitCfg,
		fmt: WFmt,
		ts: &[f32],
		file: Option<&Path>,
		mut progress: impl FnMut(usize),
	) -> Result<Dit, String> {
		let mut d = Self::open(dir, cfg)?;
		if fmt == WFmt::F32 {
			return Ok(d);
		}
		let all = with_ref(ts);
		let key = match file {
			Some(_) => Some(wcache::key(
				&dir.join("transformer_diffusers"),
				&format!("h3dit;{fmt:?};rot {};ts {all:?}", ops::rot_on()),
			)?),
			None => None,
		};
		if let (Some(f), Some(k)) = (file, &key) {
			if let Some(mut r) = wcache::Reader::open(f, k) {
				d.res = Some(Resident::read(&mut r, cfg, &mut progress)?);
				return Ok(d);
			}
		}
		let res = d.make_resident(fmt, all, true, &mut progress)?;
		if let (Some(f), Some(k)) = (file, &key) {
			let save = || -> Result<(), String> {
				let mut w = wcache::Writer::create(f, k)?;
				res.save(&mut w)?;
				w.finish()
			};
			if let Err(e) = save() {
				eprintln!("image: transformer weights not cached: {e}");
			}
		}
		d.res = Some(res);
		Ok(d)
	}

	/// [`Self::open`] with what the NPU backend leaves to the cores: the
	/// text rows' embedder and refiner (in `fmt`), `proj_in`, `norm_out`'s
	/// table of timesteps `ts` (and [`REF_T`]) and the output heads; no
	/// blocks, no AdaLN tables.
	pub fn load_small(dir: &Path, cfg: &DitCfg, fmt: WFmt, ts: &[f32]) -> Result<Dit, String> {
		let mut d = Self::open(dir, cfg)?;
		d.res = Some(d.make_resident(fmt, with_ref(ts), false, &mut |_| {})?);
		Ok(d)
	}

	/// The resident AdaLN tables: (timesteps, per block
	/// `[ts][MODALITIES][6][hidden]`).
	pub fn tables(&self) -> Option<(&[f32], &[Vec<f32>])> {
		self.res
			.as_ref()
			.filter(|r| !r.adaln.is_empty())
			.map(|r| (&r.ts[..], &r.adaln[..]))
	}

	fn make_resident(
		&self,
		fmt: WFmt,
		ts: Vec<f32>,
		with_blocks: bool,
		progress: &mut dyn FnMut(usize),
	) -> Result<Resident, String> {
		let c = &self.cfg;
		let temb = self.temb(&ts)?;
		let mut blocks = Vec::with_capacity(c.layers);
		let mut adaln = Vec::with_capacity(c.layers);
		for l in (0..c.layers).filter(|_| with_blocks) {
			blocks.push(self.load_block(&format!("transformer_blocks.{l}."), fmt)?);
			adaln.push(self.adaln(l, &temb)?);
			progress(l);
		}
		let refiner = (0..c.refiner_layers)
			.map(|j| self.load_block(&format!("token_refiner.refiner_blocks.{j}."), fmt))
			.collect::<Result<_, _>>()?;
		let (k, d, pd) = (c.text_dim, c.hidden, c.patch_dim());
		Ok(Resident {
			blocks,
			refiner,
			ctx: Lin::new(self.w("context_embedder.weight")?, d, k, fmt),
			ctx_b: self.w("context_embedder.bias")?,
			proj_in: Lin::new(self.w("proj_in.weight")?, d, pd, WFmt::F32),
			proj_in_b: self.w("proj_in.bias")?,
			refiner_norm: self.w("token_refiner.final_norm.weight")?,
			out_norm: self.w("norm_out.norm.weight")?,
			heads: (0..c.pdd_heads)
				.map(|i| {
					let p = format!("pdd_proj_out.{i}.");
					Ok((self.w(&format!("{p}weight"))?, self.w(&format!("{p}bias"))?))
				})
				.collect::<Result<_, String>>()?,
			out_ss: self.out_table(&temb)?,
			ts,
			adaln,
		})
	}

	/// Bytes of the resident weights and tables.
	pub fn resident_bytes(&self) -> usize {
		self.res.as_ref().map_or(0, |r| {
			r.blocks
				.iter()
				.chain(&r.refiner)
				.map(BlockW::bytes)
				.sum::<usize>()
				+ r.ctx.bytes()
				+ r.adaln.iter().map(|t| t.len() * 4).sum::<usize>()
				+ r.out_ss.len() * 4
		})
	}

	fn block_w(&self, l: usize) -> Result<BlockRef<'_>, String> {
		Ok(match &self.res {
			Some(r) => BlockRef::Res(&r.blocks[l]),
			None => BlockRef::Own(Box::new(
				self.load_block(&format!("transformer_blocks.{l}."), WFmt::F32)?,
			)),
		})
	}

	fn refiner_w(&self, j: usize) -> Result<BlockRef<'_>, String> {
		Ok(match &self.res {
			Some(r) => BlockRef::Res(&r.refiner[j]),
			None => BlockRef::Own(Box::new(
				self.load_block(&format!("token_refiner.refiner_blocks.{j}."), WFmt::F32)?,
			)),
		})
	}

	/// The resident tables' row of each of `ts`, when they hold them all.
	fn table_rows(&self, ts: &[f32]) -> Option<Vec<usize>> {
		let r = self.res.as_ref()?;
		ts.iter()
			.map(|t| r.ts.iter().position(|x| x == t))
			.collect()
	}

	fn w(&self, name: &str) -> Result<Vec<f32>, String> {
		self.ck.f32(name)
	}

	/// The weights of block `prefix` (`transformer_blocks.L.` or
	/// `token_refiner.refiner_blocks.L.`), matmuls in `fmt`.
	pub fn load_block(&self, prefix: &str, fmt: WFmt) -> Result<BlockW, String> {
		let c = &self.cfg;
		let (d, inner, f) = (c.hidden, c.inner(), c.ffn);
		let p = |n: &str| format!("{prefix}{n}");
		let lin = |n: &str, rows: usize, cols: usize| -> Result<Lin, String> {
			Ok(Lin::new(self.ck.mat(&p(n), rows, cols)?, rows, cols, fmt))
		};
		let ff0 = p("ff.net.0.proj.weight");
		if self.ck.shape(&ff0)? != [2 * f, d] {
			return Err(format!("{ff0}: expected [{}, {d}]", 2 * f));
		}
		Ok(BlockW {
			n1: self.w(&p("norm1.weight"))?,
			n2: self.w(&p("norm2.weight"))?,
			q: lin("attn.to_q.weight", inner, d)?,
			k: lin("attn.to_k.weight", inner, d)?,
			v: lin("attn.to_v.weight", inner, d)?,
			nq: self.w(&p("attn.norm_q.weight"))?,
			nk: self.w(&p("attn.norm_k.weight"))?,
			o: lin("attn.to_out.0.weight", d, inner)?,
			// SwiGLU: the first half of `net.0.proj` is the value, the second the gate
			up: Lin::new(self.ck.rows(&ff0, 0, f)?, f, d, fmt),
			gate: Lin::new(self.ck.rows(&ff0, f, 2 * f)?, f, d, fmt),
			down: lin("ff.net.2.weight", d, f)?,
		})
	}

	/// Full attention of rows `h` (`[n][hidden]`, already normalized and
	/// modulated) through `w`: q / k head norms, RoPE (`rope`, `None` in the
	/// refiner), SDPA over all rows, `to_out`.
	fn attention(
		&self,
		w: &BlockW,
		h: &[f32],
		rope: Option<&Rope>,
		cal: Option<usize>,
	) -> Vec<f32> {
		let c = &self.cfg;
		let (hd, nh) = (c.head_dim, c.heads);
		let n = h.len() / c.hidden;
		let mut qkv = ops::multi(&[&w.q, &w.k, &w.v], h, n).into_iter();
		let (mut q, mut k, v) = (
			qkv.next().unwrap(),
			qkv.next().unwrap(),
			qkv.next().unwrap(),
		);
		ops::rms(&mut q, &w.nq, c.qk_eps, false);
		ops::rms(&mut k, &w.nk, c.qk_eps, false);
		if let Some(r) = rope {
			r.apply(&mut q, nh, hd);
			r.apply(&mut k, nh, hd);
		}
		let mut a = vec![0f32; n * nh * hd];
		// int8 in the quantized backend, f32 in the reference
		attention_rows(&q, &k, &v, n, nh, hd, self.res.is_some(), &mut a);
		drop((q, k, v));
		if let Some(l) = cal {
			cal_max(Some(l), 1, rot_max(&a, nh * hd));
			hess_add(l, 1, &a);
		}
		w.o.apply(&a, n)
	}

	/// The SwiGLU MLP of rows `h` by row chunks, added to `x` through `add`.
	fn mlp(&self, w: &BlockW, h: &[f32], cal: Option<usize>, mut add: impl FnMut(usize, &[f32])) {
		let d = self.cfg.hidden;
		let n = h.len() / d;
		for c0 in (0..n).step_by(ops::MLP_CHUNK) {
			let r = ops::MLP_CHUNK.min(n - c0);
			let hc = &h[c0 * d..(c0 + r) * d];
			let mut gu = ops::multi(&[&w.gate, &w.up], hc, r).into_iter();
			let (g, u) = (gu.next().unwrap(), gu.next().unwrap());
			if let Some(l) = cal {
				cal_max(Some(l), 4, max_abs(&g));
				let hid: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| ops::silu(a) * b).collect();
				cal_max(Some(l), 5, max_abs(&hid));
				hess_add(l, 3, &hid);
			}
			add(c0, &ops::swiglu_down(&w.down, &g, u, r));
		}
	}

	/// The text rows: `context_embedder` of the text encoder's output
	/// (`[n][text_dim]`), the token refiner, its final norm. `tap(j, rows)`
	/// sees each refiner block's output (`j` = refiner block) and the
	/// embedder's (`j = usize::MAX`).
	pub fn text_rows(
		&self,
		te: &[f32],
		mut tap: impl FnMut(usize, &[f32]),
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let (k, d) = (c.text_dim, c.hidden);
		let n = te.len() / k;
		let mut x = match &self.res {
			Some(r) => {
				let mut y = r.ctx.apply(te, n);
				ops::rows_mut(&mut y, d, |row| {
					row.iter_mut().zip(&r.ctx_b).for_each(|(v, &b)| *v += b)
				});
				y
			}
			None => linear_b(
				te,
				n,
				k,
				&self.w("context_embedder.weight")?,
				&self.w("context_embedder.bias")?,
			),
		};
		tap(usize::MAX, &x);
		for j in 0..c.refiner_layers {
			let w = self.refiner_w(j)?;
			let mut h = x.clone();
			ops::rms(&mut h, &w.n1, c.eps, false);
			let o = self.attention(&w, &h, None, None);
			x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
			let mut h = x.clone();
			ops::rms(&mut h, &w.n2, c.eps, false);
			self.mlp(&w, &h, None, |c0, o| {
				x[c0 * d..c0 * d + o.len()]
					.iter_mut()
					.zip(o)
					.for_each(|(a, b)| *a += b);
			});
			tap(j, &x);
		}
		match &self.res {
			Some(r) => ops::rms(&mut x, &r.refiner_norm, c.final_eps, false),
			None => ops::rms(
				&mut x,
				&self.w("token_refiner.final_norm.weight")?,
				c.final_eps,
				false,
			),
		}
		Ok(x)
	}

	/// `proj_in` of packed latent rows `[n][patch_dim]`.
	pub fn image_rows(&self, rows: &[f32]) -> Result<Vec<f32>, String> {
		let k = self.cfg.patch_dim();
		Ok(match &self.res {
			Some(r) => {
				let mut y = r.proj_in.apply(rows, rows.len() / k);
				ops::rows_mut(&mut y, self.cfg.hidden, |row| {
					row.iter_mut().zip(&r.proj_in_b).for_each(|(v, &b)| *v += b)
				});
				y
			}
			None => linear_b(
				rows,
				rows.len() / k,
				k,
				&self.w("proj_in.weight")?,
				&self.w("proj_in.bias")?,
			),
		})
	}

	/// Timestep embedding rows `[ts.len()][temb_dim]`.
	pub fn temb(&self, ts: &[f32]) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let (fd, half) = (c.freq_dim, c.freq_dim / 2);
		let mut e = vec![0f32; ts.len() * fd];
		for (r, &t) in ts.iter().enumerate() {
			for i in 0..half {
				// diffusers get_timestep_embedding: exp(-ln(1e4) i / half), flip to [cos, sin]
				let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
				let a = t * f;
				e[r * fd + i] = a.cos();
				e[r * fd + half + i] = a.sin();
			}
		}
		let mut h = linear_b(
			&e,
			ts.len(),
			fd,
			&self.w("time_embedder.linear_1.weight")?,
			&self.w("time_embedder.linear_1.bias")?,
		);
		ops::map(&mut h, ops::silu);
		Ok(linear_b(
			&h,
			ts.len(),
			c.temb_hidden,
			&self.w("time_embedder.linear_2.weight")?,
			&self.w("time_embedder.linear_2.bias")?,
		))
	}

	/// Block `l`'s modulation table for timestep embedding rows `temb`:
	/// `[temb rows][MODALITIES][6][hidden]` (`shift_msa, scale_msa,
	/// gate_msa, shift_mlp, scale_mlp, gate_mlp`), addressed by
	/// `t_index * MODALITIES + tag`.
	pub fn adaln(&self, l: usize, temb: &[f32]) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let p = format!("transformer_blocks.{l}.adaln_proj.linear.");
		let mut a = temb.to_vec();
		ops::map(&mut a, ops::silu);
		let n = a.len() / c.temb_dim;
		let b = self.w(&format!("{p}bias"))?;
		if b.len() != MODALITIES * 6 * c.hidden {
			return Err(format!("{p}bias: {} values", b.len()));
		}
		Ok(linear_b(
			&a,
			n,
			c.temb_dim,
			&self.w(&format!("{p}weight"))?,
			&b,
		))
	}

	/// Block `l` over the packed rows `x`; `table` from [`Self::adaln`],
	/// `idx[r]` = row `r`'s table entry.
	pub fn block(
		&self,
		l: usize,
		w: &BlockW,
		x: &mut [f32],
		table: &[f32],
		idx: &[u16],
		rope: &Rope,
	) {
		let c = &self.cfg;
		let d = c.hidden;
		let cal = cal_on().then_some(l);
		if cal == Some(0) {
			cal_max(None, 0, max_abs(x));
		}
		let h = rms_mod(x, &w.n1, c.eps, table, idx, 1);
		if let Some(l) = cal {
			cal_max(Some(l), 0, rot_max(&h, d));
			hess_add(l, 0, &h);
		}
		let o = self.attention(w, &h, Some(rope), cal);
		drop(h);
		add_gate(x, &o, table, idx, 2, d);
		drop(o);
		let h = rms_mod(x, &w.n2, c.eps, table, idx, 4);
		if let Some(l) = cal {
			cal_max(Some(l), 2, rot_max(&h, d));
			hess_add(l, 2, &h);
		}
		let mut ff = vec![0f32; x.len()];
		self.mlp(w, &h, cal, |c0, o| {
			ff[c0 * d..c0 * d + o.len()].copy_from_slice(o)
		});
		add_gate(x, &ff, table, idx, 5, d);
		if let Some(l) = cal {
			cal_max(Some(l), 3, max_abs(x));
		}
	}

	/// Table entry of every row: `t_index * MODALITIES + tag`.
	pub fn entries(layout: &Layout, tidx: &[u8]) -> Vec<u16> {
		layout
			.tags
			.iter()
			.zip(tidx)
			.map(|(&g, &t)| t as u16 * MODALITIES as u16 + g as u16)
			.collect()
	}

	/// One step's velocity of the image rows (references and target):
	/// `txt` from [`Self::text_rows`], `img` the packed image rows
	/// (`[refs + target][patch_dim]`), `ts` / `tidx` from
	/// [`Layout::row_timesteps`], `step` the head. `tap(l, x)` sees every
	/// block's output (all rows).
	#[allow(clippy::too_many_arguments)]
	pub fn forward(
		&self,
		layout: &Layout,
		rope: &Rope,
		txt: &[f32],
		img: &[f32],
		ts: &[f32],
		tidx: &[u8],
		step: usize,
		mut tap: impl FnMut(usize, &[f32]),
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let d = c.hidden;
		let mut x = Vec::with_capacity(layout.len() * d);
		x.extend_from_slice(txt);
		x.extend(self.image_rows(img)?);
		if x.len() != layout.len() * d {
			return Err(format!(
				"dit: {} rows for a layout of {}",
				x.len() / d,
				layout.len()
			));
		}
		match (self.table_rows(ts), &self.res) {
			(Some(rows), Some(r)) => {
				// the resident tables, indexed by their own timestep rows
				let g: Vec<u8> = tidx.iter().map(|&t| rows[t as usize] as u8).collect();
				let idx = Self::entries(layout, &g);
				for l in 0..c.layers {
					self.block(l, &r.blocks[l], &mut x, &r.adaln[l], &idx, rope);
					tap(l, &x);
				}
				self.out_with(
					&x[layout.image0() * d..],
					&r.out_ss,
					&g[layout.image0()..],
					step,
				)
			}
			_ => {
				let temb = self.temb(ts)?;
				let idx = Self::entries(layout, tidx);
				for l in 0..c.layers {
					let w = self.block_w(l)?;
					let table = self.adaln(l, &temb)?;
					self.block(l, &w, &mut x, &table, &idx, rope);
					tap(l, &x);
				}
				self.out(
					&x[layout.image0() * d..],
					&temb,
					&tidx[layout.image0()..],
					step,
				)
			}
		}
	}

	/// `norm_out` at timestep `t` and output head `step` of final rows `x`
	/// (`[n][hidden]`, all at `t`).
	pub fn out_at(&self, x: &[f32], t: f32, step: usize) -> Result<Vec<f32>, String> {
		let n = x.len() / self.cfg.hidden;
		match (self.table_rows(&[t]), &self.res) {
			(Some(r), Some(res)) => self.out_with(x, &res.out_ss, &vec![r[0] as u8; n], step),
			_ => self.out(x, &self.temb(&[t])?, &vec![0u8; n], step),
		}
	}

	/// `norm_out`'s `[temb rows][2][hidden]` (shift, scale).
	fn out_table(&self, temb: &[f32]) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let mut a = temb.to_vec();
		ops::map(&mut a, ops::silu);
		Ok(linear_b(
			&a,
			a.len() / c.temb_dim,
			c.temb_dim,
			&self.w("norm_out.linear.weight")?,
			&self.w("norm_out.linear.bias")?,
		))
	}

	/// `norm_out` (per row's timestep) and output head `step` of the final
	/// rows `x` (`[n][hidden]`, `tidx` theirs).
	pub fn out(
		&self,
		x: &[f32],
		temb: &[f32],
		tidx: &[u8],
		step: usize,
	) -> Result<Vec<f32>, String> {
		let ss = self.out_table(temb)?;
		self.out_with(x, &ss, tidx, step)
	}

	/// [`Self::out`] with `norm_out`'s table `ss` (`[ts][2][hidden]`).
	fn out_with(
		&self,
		x: &[f32],
		ss: &[f32],
		tidx: &[u8],
		step: usize,
	) -> Result<Vec<f32>, String> {
		let c = &self.cfg;
		let d = c.hidden;
		let own;
		let nw = match &self.res {
			Some(r) => &r.out_norm,
			None => {
				own = self.w("norm_out.norm.weight")?;
				&own
			}
		};
		let mut h = x.to_vec();
		ops::rows_mut_idx(&mut h, d, |r, row| {
			ops::rms_row(row, nw, c.final_eps, false);
			let e = &ss[tidx[r] as usize * 2 * d..];
			let (sh, sc) = (&e[..d], &e[d..2 * d]);
			for ((v, &a), &b) in row.iter_mut().zip(sc).zip(sh) {
				*v = *v * (1.0 + a) + b;
			}
		});
		let n = x.len() / d;
		match self.res.as_ref().and_then(|r| r.heads.get(step)) {
			Some((w, b)) => Ok(linear_b(&h, n, d, w, b)),
			None => {
				let p = format!("pdd_proj_out.{step}.");
				Ok(linear_b(
					&h,
					n,
					d,
					&self.w(&format!("{p}weight"))?,
					&self.w(&format!("{p}bias"))?,
				))
			}
		}
	}
}

/// `ts` with [`REF_T`], sorted, distinct: the timesteps of the tables.
pub fn with_ref(ts: &[f32]) -> Vec<f32> {
	let mut all: Vec<f32> = ts.to_vec();
	all.push(REF_T);
	all.sort_by(f32::total_cmp);
	all.dedup();
	all
}

impl Resident {
	fn save(&self, w: &mut wcache::Writer) -> Result<(), String> {
		w.f32s(&self.ts)?;
		for (b, t) in self.blocks.iter().zip(&self.adaln) {
			b.save(w)?;
			w.f32s(t)?;
		}
		for b in &self.refiner {
			b.save(w)?;
		}
		w.lin(&self.ctx)?;
		w.f32s(&self.ctx_b)?;
		w.lin(&self.proj_in)?;
		for v in [
			&self.proj_in_b,
			&self.refiner_norm,
			&self.out_norm,
			&self.out_ss,
		] {
			w.f32s(v)?;
		}
		for (hw, hb) in &self.heads {
			w.f32s(hw)?;
			w.f32s(hb)?;
		}
		Ok(())
	}

	fn read(
		r: &mut wcache::Reader,
		c: &DitCfg,
		progress: &mut dyn FnMut(usize),
	) -> Result<Resident, String> {
		let ts = r.f32s()?;
		let (mut blocks, mut adaln) = (Vec::new(), Vec::new());
		for l in 0..c.layers {
			blocks.push(BlockW::read(r)?);
			adaln.push(r.f32s()?);
			progress(l);
		}
		let refiner = (0..c.refiner_layers)
			.map(|_| BlockW::read(r))
			.collect::<Result<_, _>>()?;
		let (ctx, ctx_b, proj_in) = (r.lin()?, r.f32s()?, r.lin()?);
		let (proj_in_b, refiner_norm, out_norm, out_ss) =
			(r.f32s()?, r.f32s()?, r.f32s()?, r.f32s()?);
		let heads = (0..c.pdd_heads)
			.map(|_| Ok((r.f32s()?, r.f32s()?)))
			.collect::<Result<_, String>>()?;
		Ok(Resident {
			blocks,
			refiner,
			ctx,
			ctx_b,
			proj_in,
			proj_in_b,
			refiner_norm,
			out_norm,
			heads,
			ts,
			adaln,
			out_ss,
		})
	}
}
