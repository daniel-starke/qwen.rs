//! A one-position decode step's dense layers on a discrete GPU (F8, `.npu`
//! key `dense = ADAPTER`): every dense matrix of every layer lives in VRAM
//! ([`DenseGpu`], `offload::Gemv` in the CLI) and the host does the
//! elementwise work between the GEMVs in f32; norms, the hyper-connection
//! mixes, conv, gates, RoPE, rotations and int8 quantization of the
//! quantized matrices' inputs. The graphs' math (`npubuild::flash::graphs`)
//! on the same weight files; the attention and the routed experts stay where
//! the engine runs them for a one-position step (host, cores). A delta
//! layer's whole mixer (conv, l2 norms, gates, recurrence, gated norm) runs
//! on the GPU in the submission of its projections, on states kept in VRAM
//! (`shaders/gemv_delta.glsl`, [`GpuDec::delta_mix_n`]; the engine moves the
//! states to the host only for a prefill chunk or a snapshot, and a verify
//! step's rollback replays on the GPU, [`GpuDec::delta_rollback`]);
//! `QWEN_FLASH_MIX_GPU=0` runs it on the host between two submissions.
//!
//! A layer is 2 submissions: the attention hyper-connection (down -> SiLU
//! -> up -> mix) -> pre-scale, FWHT, rounding -> mixer projections; and
//! after the mixer, output projection -> injection into the streams -> norm
//! -> the MoE's hyper-connection -> router + shared gate_up -> SwiGLU ->
//! shared down (`offload::Gemv::add_vec` steps between the GEMVs; 1 more at
//! the PLE layer); a delta layer with its mixer on the GPU runs both as one
//! ([`GpuDec::delta_post_n`], `QWEN_FLASH_FUSE=0`: two). The shared expert (gate_up -> SwiGLU -> down) is a third
//! submission, left running while the cores do the routed experts
//! (`QWEN_FLASH_SHARED_ASYNC=0`: in the second, as before). In the chained
//! form the small steps between the GEMVs are merged where they follow
//! each other ([`Merged`], `QWEN_FLASH_MERGE=0`: separate): a
//! hyper-connection's SiLU is the down GEMV's epilogue, its mix one
//! dispatch with the rotation after it, the injection one with the MoE
//! norm, SwiGLU one with the shared down's rotation; bit-identical, 4-6
//! barriers less per layer. `QWEN_FLASH_CHAIN=0` runs the session-10 form (6 per
//! layer, the host's elementwise work between). The
//! lm_head (int8) runs there too (`QWEN_FLASH_HEAD_GPU=0` keeps it on the
//! cores), and with `mtp` the MTP head's dense part ([`GpuDec::add_mtp`]).

use std::path::Path;

use super::{HC, HC_RANK, NL, PLE_LAYER, W};

const HCW: usize = HC * W;
const H: usize = 48;
const HK: usize = 16;
const D: usize = 128;
const NQKV: usize = 2 * HK * D + H * D;
const HQ: usize = 24;
const HKV: usize = 2;
const DA: usize = 256;
const RD: usize = 64;
const IH: usize = 4;
const ID: usize = 128;
const NE: usize = 512;
const FE: usize = 640;
/// The two input regions of a GEMV run.
const X0: usize = 0;
const X1: usize = 16384;
/// Output offsets of a hyper-connection's injection weights and mix.
const YINJ: usize = 20480;
const YMIX: usize = 24576;
/// Input offsets of the streams and the injection weights of a chained
/// `post`; output offsets of the streams after injection, the MoE input
/// rotated, the shared expert's down projection.
const XS: usize = 24576;
const XINJ: usize = 36864;
const YX1: usize = 32768;
const YMQ: usize = 45056;
const YSD: usize = 4096;
/// The most positions of one wave ([`GpuDec::set_wave`]; `offload::GEMV_WAVE`).
pub const WAVE_MAX: usize = 4;
/// lm_head rows per GPU matrix (one 256 MB VRAM block holds one).
pub const HEAD_ROWS: usize = 16384;

/// A matrix for [`DenseGpu::add`]: int8 codes with row scales, or f16.
pub enum Mat<'a> {
	I8(&'a [i8], &'a [f32]),
	/// `I8` on an input of int8 values (`offload::GemvMat::Q8`)
	Q8(&'a [i8], &'a [f32]),
	F16(&'a [u16]),
}

/// Resident GEMVs on a GPU (`offload::Gemv`).
pub trait DenseGpu: Send {
	/// Upload `[n][k]`, read at input float `xoff`, written to output `yoff`.
	fn add(
		&mut self,
		m: Mat<'_>,
		n: usize,
		k: usize,
		xoff: usize,
		yoff: usize,
	) -> Result<usize, String>;
	/// Run `ids` on the input `x`.
	fn run(&mut self, x: &[f32], ids: &[usize]) -> Result<(), String>;
	/// An elementwise step (`offload::Gemv::add_act`).
	#[allow(clippy::too_many_arguments)]
	fn act(
		&mut self,
		mode: u32,
		n: usize,
		src: usize,
		dst: usize,
		xsrc: usize,
		hc: usize,
		a: f32,
	) -> Result<usize, String>;
	/// A vector step (`offload::Gemv::add_vec`).
	#[allow(clippy::too_many_arguments)]
	fn vec(
		&mut self,
		mode: u32,
		n: usize,
		src: usize,
		dst: usize,
		aux: usize,
		hc: usize,
		a: f32,
		blk: usize,
		flags: u32,
		ycopy: Option<usize>,
		extra: Option<usize>,
		w: Option<&[f32]>,
	) -> Result<usize, String>;
	/// Let int8 GEMV `id` also write `silu(a y)` of its outputs to input
	/// float `xdst` (`offload::Gemv::set_silu`); false when it cannot, and
	/// the elementwise step stays.
	fn set_silu(&mut self, _id: usize, _a: f32, _xdst: usize) -> Result<bool, String> {
		Ok(false)
	}
	/// Run stages of ops in one submission (`offload::Gemv::run_chain`).
	fn run_chain(&mut self, x: &[f32], stages: &[&[usize]]) -> Result<(), String>;
	/// The output region of the last run.
	fn y(&self) -> &[f32];
	/// [`Self::run_chain`] on `xs.len()` positions at once
	/// (`offload::Gemv::run_chain_n`).
	fn run_chain_n(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String>;
	/// Position `p`'s output region of the last run.
	fn y_at(&self, p: usize) -> &[f32];
	/// The most positions a run takes.
	fn max_np(&self) -> usize;
	/// [`Self::run_chain_n`] left in flight (`offload::Gemv::run_chain_n_begin`):
	/// the outputs after [`Self::run_wait`].
	fn run_chain_n_begin(&mut self, xs: &[&[f32]], stages: &[&[usize]]) -> Result<(), String> {
		self.run_chain_n(xs, stages)
	}
	/// Wait for the run [`Self::run_chain_n_begin`] left in flight.
	fn run_wait(&mut self) -> Result<(), String> {
		Ok(())
	}
	/// True when it has two waves ([`Self::set_wave`]).
	fn has_waves(&self) -> bool {
		false
	}
	/// Make wave `w` (0 or 1) the one runs, waits and outputs refer to
	/// (`offload::Gemv::set_wave`): each wave has its own position slots
	/// and fence, so a run of each can be in flight; wave 1 takes at most
	/// [`Self::max_np`] / 2 positions, and its delta mixer steps continue
	/// at aux slot `p0` (wave 0's positions; 0 for wave 0).
	fn set_wave(&mut self, _w: usize, _p0: usize) {}
	/// The next run moves only these input / output float ranges of each
	/// position (`offload::Gemv::io_next`); the rest of [`Self::y_at`] is
	/// stale after it. A hint: copying everything is correct too.
	fn io_next(&mut self, _xr: &[std::ops::Range<usize>], _yr: &[std::ops::Range<usize>]) {}
	/// VRAM for `layers` delta mixers' states (`offload::Gemv::delta_init`).
	fn delta_init(&mut self, _layers: usize) -> Result<(), String> {
		Err("no delta mixer steps".into())
	}
	/// A delta mixer step (`offload::Gemv::add_delta`).
	#[allow(clippy::too_many_arguments)]
	fn delta(
		&mut self,
		_mode: u32,
		_flags: u32,
		_layer: usize,
		_yz: usize,
		_yab: usize,
		_ya: usize,
		_w: &[f32],
	) -> Result<usize, String> {
		Err("no delta mixer steps".into())
	}
	/// Write a delta mixer's state `[H][D][D]` and conv window `[3][NQKV]`.
	fn delta_put(&mut self, _layer: usize, _st: &[f32], _win: &[f32]) -> Result<(), String> {
		Err("no delta mixer steps".into())
	}
	/// Read them back.
	fn delta_get(
		&mut self,
		_layer: usize,
		_st: &mut [f32],
		_win: &mut [f32],
	) -> Result<(), String> {
		Err("no delta mixer steps".into())
	}
}

/// Output offset of the delta mixer's gated output on the GPU.
const YA: usize = 32768;
/// VRAM of the delta mixers' states on the GPU (`offload::Gemv::delta_init`:
/// state + backup, window, saved windows, recurrence inputs).
const DMIX_BYTES: u64 = ((2 * H * D * D + 30 * NQKV + 8 * (NQKV + 2 * H)) * 4) as u64;

/// Builds the [`DenseGpu`] for `need` bytes of matrices.
pub type DenseGpuNew = Box<dyn Fn(u64) -> Result<Box<dyn DenseGpu>, String> + Send + Sync>;

fn f16s(b: &[u8]) -> Vec<u16> {
	b.chunks_exact(2)
		.map(|c| u16::from_le_bytes([c[0], c[1]]))
		.collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
	b.chunks_exact(2)
		.map(|c| crate::ir::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect()
}

fn sigmoid(x: f32) -> f32 {
	1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
	x * sigmoid(x)
}

/// `x / sqrt(mean(x^2) + eps) * w` over `x`.
fn rmsw(x: &mut [f32], w: &[f32], eps: f32) {
	let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
	let r = 1.0 / (ms + eps).sqrt();
	x.iter_mut().zip(w).for_each(|(a, w)| *a *= r * w);
}

/// The streams normed per stream with `w` `[HCW]`.
fn norm_g(x: &[f32], w: &[f32]) -> Vec<f32> {
	let mut y = x.to_vec();
	for (r, w) in y.chunks_exact_mut(W).zip(w.chunks_exact(W)) {
		rmsw(r, w, 1e-6);
	}
	y
}

/// `fq(rotate(x * pre))`: a quantized matrix's input as int8 values.
fn qin(x: &[f32], pre: &[f32]) -> Vec<f32> {
	let mut t = rot(x, pre);
	t.iter_mut()
		.for_each(|v| *v = v.round_ties_even().clamp(-128.0, 127.0));
	t
}

/// The block of `weights::rot_block` for `k` inputs.
fn rot_blk(k: usize) -> usize {
	[256, 128, 64]
		.into_iter()
		.find(|b| k.is_multiple_of(*b))
		.unwrap()
}

/// `rotate(x * pre)` in the blocks of `weights::rot_block`.
fn rot(x: &[f32], pre: &[f32]) -> Vec<f32> {
	let b = rot_blk(x.len());
	let mut t: Vec<f32> = x.iter().zip(pre).map(|(a, b)| a * b).collect();
	super::cpuexp::fwht(&mut t, b);
	t
}

/// RoPE (rotate_half pairs) of the first [`RD`] values of `t` at `cs` / `sn`.
fn rope(t: &mut [f32], cs: &[f32], sn: &[f32]) {
	for i in 0..RD / 2 {
		let (a, b) = (t[i], t[i + RD / 2]);
		t[i] = a * cs[i] - b * sn[i];
		t[i + RD / 2] = b * cs[i + RD / 2] + a * sn[i + RD / 2];
	}
}

/// A hyper-connection's injection weights from its run's output region.
fn inj_w(y: &[f32]) -> [f32; HC] {
	let q = 1.0 / HC as f32;
	let mut j = [0f32; HC];
	for (c, o) in j.iter_mut().enumerate() {
		*o = 2.0 * sigmoid(y[YINJ + c] * q);
	}
	j
}

struct Hcg {
	norm: Vec<f32>,
	down: usize,
	up: usize,
	inj: usize,
	/// `down` writes the SiLU of its rows into `up`'s input itself (the
	/// chained form with [`Merged`] steps): no SiLU step between
	silu: bool,
}

enum Mix {
	Delta {
		proj: [usize; 3],
		conv: Vec<f32>,
		dtb: Vec<f32>,
		acoef: Vec<f32>,
		gnw: Vec<f32>,
	},
	Attn {
		proj: [usize; 4],
		qn: Vec<f32>,
		kn: Vec<f32>,
		iqn: Vec<f32>,
	},
}

struct Layer {
	hca: Hcg,
	hcm: Hcg,
	ppre: Vec<f32>,
	opre: Vec<f32>,
	mix: Mix,
	o: usize,
	router: usize,
	sgate: usize,
	sgu: usize,
	sd: usize,
	mpre: Vec<f32>,
	sdpre: Vec<f32>,
	/// the chained form's vector steps: qin of the mixer input, of the
	/// output projection's input, the MoE norm, the MoE input's rotation,
	/// qin of the shared down's input
	vp: usize,
	vo: usize,
	vn: usize,
	vm: usize,
	vsd: usize,
	/// the merged steps (`QWEN_FLASH_MERGE`, the chained form): mix + qin
	/// of the mixer input, injection + the MoE norm, mix + copy + the MoE
	/// input's rotation, SwiGLU + qin of the shared down's input
	merged: Option<Merged>,
}

/// A layer's merged vector steps (`shaders/gemv_vec.glsl` modes 5, 1 and 0
/// with flags & 8): each one dispatch where the separate steps were two or
/// three, so one or two barriers less; every value computed as there.
#[derive(Clone, Copy)]
struct Merged {
	vp: usize,
	vn: usize,
	vm: usize,
	vsd: usize,
}

/// `QWEN_FLASH_MERGE` (default on): the chained form's merged vector steps
/// ([`Merged`]); `0` runs the separate steps (the A/B).
fn merge() -> bool {
	std::env::var("QWEN_FLASH_MERGE").as_deref() != Ok("0")
}

struct Ple {
	pre: Vec<f32>,
	pk: usize,
	pv: usize,
	nkey: Vec<f32>,
	nquery: Vec<f32>,
	nconv: Vec<f32>,
	pconv: Vec<f32>,
}

/// A post run's outputs per position ([`GpuDec::post_n`]): (MoE input
/// prescaled and rotated `[W]`, inj_m, router logits `[NE]`, shared `[W]`).
pub type Post = Vec<(Vec<f32>, [f32; HC], Vec<f32>, Vec<f32>)>;

/// The dense layers on the GPU plus their host-side vectors.
pub struct GpuDec {
	g: Box<dyn DenseGpu>,
	layers: Vec<Layer>,
	ple: Ple,
	/// the hyper-connections' SiLU and mix steps
	silu: usize,
	mix: usize,
	/// the chained form's shared steps: injection, the MoE input as is,
	/// SwiGLU (`None`: the unchained form)
	chain: Option<[usize; 3]>,
	/// the lm_head's matrices, if uploaded ([`Self::add_head`])
	head: Vec<usize>,
	/// the shared experts' gates of the positions whose down projection
	/// [`GpuDec::post_n`] left running ([`GpuDec::shared_n`]), per wave
	sgates: [Vec<f32>; 2],
	/// A post run's input vectors, one per position, kept between runs: a
	/// fresh zeroed one is 140 KB per position and layer for ~50 KB of
	/// values (the run copies only the ranges it names).
	xscr: Vec<Vec<f32>>,
	/// the wave the runs refer to ([`GpuDec::set_wave`])
	wave: usize,
	/// `QWEN_FLASH_SHARED_ASYNC` (default on)
	shared_async: bool,
	/// per delta layer: the mixer's steps on the GPU (conv, recurrence,
	/// both saving, window replay, recurrence replay, both of none), empty
	/// when the host runs the mixer (`QWEN_FLASH_MIX_GPU=0`)
	dmix: Vec<[usize; 8]>,
	/// the fused form ([`GpuDec::delta_post_n`], `QWEN_FLASH_FUSE`): per
	/// delta layer qin of the output projection's input read from the
	/// mixer's output `YA`, and the step forming the injection weights
	/// from `YINJ` into the post's input `XINJ` (`inj_w` on the GPU)
	dvo: Vec<usize>,
	cinj: Option<usize>,
	/// the MTP head's dense part, if uploaded ([`Self::add_mtp`])
	mtp: Option<MtpG>,
	pub t_gpu: f64,
	/// `t_gpu` by call site (`QWEN_NPU_TIMING`'s `gpu split` line): mixer
	/// projections, post, shared expert's wait, lm_head, other
	pub t_site: [f64; 5],
}

/// `QWEN_FLASH_MIX_GPU` (default on): the delta mixers on the GPU.
fn mix_gpu() -> bool {
	std::env::var("QWEN_FLASH_MIX_GPU").as_deref() != Ok("0")
}

/// `QWEN_FLASH_FUSE` (default on): a delta layer's mixer and post in one
/// submission ([`GpuDec::delta_post_n`]).
fn fuse() -> bool {
	std::env::var("QWEN_FLASH_FUSE").as_deref() != Ok("0")
}

/// The matrices a [`GpuDec`] uploads: (layer, name) of each int8 one.
fn i8_names(l: usize) -> Vec<&'static str> {
	let mut v = vec!["hca.down", "hca.up", "hcm.down", "hcm.up", "o", "sgu", "sd"];
	if l % 4 == 3 {
		v.extend(["q", "k", "v", "iqk"]);
	} else {
		v.extend(["qkv", "z", "ab"]);
	}
	if l == PLE_LAYER {
		v.extend(["pk", "pv"]);
	}
	v
}

impl GpuDec {
	/// VRAM bytes the weights in `wdir` will take, with `head_rows` rows
	/// of the int8 lm_head.
	pub fn need(wdir: &Path, head_rows: usize) -> Result<u64, String> {
		let len = |p: std::path::PathBuf| {
			std::fs::metadata(&p)
				.map(|m| m.len())
				.map_err(|e| format!("{}: {e}", p.display()))
		};
		let mut n = 0;
		for l in 0..NL {
			for m in i8_names(l) {
				n += len(wdir.join(format!("fl{l}.{m}.w.in")))?;
				n += 2 * len(wdir.join(format!("fl{l}.{m}.s.in")))?;
			}
			for m in ["hca.inj", "hcm.inj", "router", "sgate"] {
				n += len(wdir.join(format!("fl{l}.{m}.in")))?;
			}
		}
		n += (head_rows * (W + 4)) as u64;
		if mix_gpu() {
			n += (NL - NL / 4) as u64 * DMIX_BYTES;
		}
		Ok(n + (NL * 24 * 512) as u64)
	}

	pub fn load(wdir: &Path, mut g: Box<dyn DenseGpu>) -> Result<GpuDec, String> {
		let t0 = std::time::Instant::now();
		let mut layers = Vec::with_capacity(NL);
		let mut ple = None;
		for l in 0..NL {
			let rd = |n: &str| -> Result<Vec<u8>, String> {
				let p = wdir.join(format!("fl{l}.{n}.in"));
				std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
			};
			let v = |n: &str| -> Result<Vec<f32>, String> { Ok(f32s(&rd(n)?)) };
			// int8 `{m}.w` / `{m}.s`: the quantized matrices' scales x 2^16,
			// the hyper-connections' x 2^7 (`graphs::qmats` / `wlin`)
			let i8m = |g: &mut Box<dyn DenseGpu>,
			           m: &str,
			           n: usize,
			           k: usize,
			           xo: usize,
			           yo: usize|
			 -> Result<usize, String> {
				let w = rd(&format!("{m}.w"))?;
				let f = if m.starts_with("hc") { 128.0 } else { 65536.0 };
				let s: Vec<f32> = f32s(&rd(&format!("{m}.s"))?)
					.iter()
					.map(|v| v / f)
					.collect();
				// SAFETY: u8 and i8 have the same layout
				let w = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const i8, w.len()) };
				// the quantized matrices' inputs are rounded to int8 values
				let m = if m.starts_with("hc") {
					Mat::I8(w, &s)
				} else {
					Mat::Q8(w, &s)
				};
				g.add(m, n, k, xo, yo)
			};
			let hc = |g: &mut Box<dyn DenseGpu>, p: &str| -> Result<Hcg, String> {
				let down = i8m(g, &format!("{p}.down"), HC_RANK, HCW, X0, 0)?;
				let up = i8m(g, &format!("{p}.up"), HCW, HC_RANK, X1, 0)?;
				let inj = g.add(
					Mat::F16(&f16s(&rd(&format!("{p}.inj"))?)),
					HC,
					HCW,
					X0,
					YINJ,
				)?;
				Ok(Hcg {
					norm: v(&format!("{p}.norm"))?,
					down,
					up,
					inj,
					silu: false,
				})
			};
			let (mut hca, mut hcm) = (hc(&mut g, "hca")?, hc(&mut g, "hcm")?);
			let mix = if l % 4 == 3 {
				Mix::Attn {
					proj: [
						i8m(&mut g, "q", 2 * HQ * DA, W, X0, 0)?,
						i8m(&mut g, "k", HKV * DA, W, X0, 12288)?,
						i8m(&mut g, "v", HKV * DA, W, X0, 12800)?,
						i8m(&mut g, "iqk", (IH + 1) * ID, W, X0, 13312)?,
					],
					qn: v("qn")?,
					kn: v("kn")?,
					iqn: v("iqn")?,
				}
			} else {
				Mix::Delta {
					proj: [
						i8m(&mut g, "qkv", NQKV, W, X0, 0)?,
						i8m(&mut g, "z", H * D, W, X0, NQKV)?,
						i8m(&mut g, "ab", 2 * H, W, X0, 16384)?,
					],
					conv: v("conv")?,
					dtb: v("dtb")?,
					acoef: v("acoef")?,
					gnw: v("gnw")?,
				}
			};
			let o = i8m(&mut g, "o", W, H * D, X0, 0)?;
			let sgu = i8m(&mut g, "sgu", 2 * FE, W, X1, 1024)?;
			let sd = i8m(&mut g, "sd", W, FE, X0, YSD)?;
			if l == PLE_LAYER {
				ple = Some(Ple {
					pre: v("ple.pre")?,
					pk: i8m(&mut g, "pk", HCW, W, X0, 0)?,
					pv: i8m(&mut g, "pv", W, W, X0, HCW)?,
					nkey: v("nkey")?,
					nquery: v("nquery")?,
					nconv: v("nconv")?,
					pconv: v("pconv")?,
				});
			}
			let router = g.add(Mat::F16(&f16s(&rd("router")?)), NE, W, X0, 0)?;
			let sgate = g.add(Mat::F16(&f16s(&rd("sgate")?)), 1, W, X0, 512)?;
			let (ppre, opre) = (v("proj.pre")?, v("out.pre")?);
			let (mpre, sdpre) = (v("moe.pre")?, v("sd.pre")?);
			// qin: pre-scale (4), FWHT, round (2); in place on x unless from y (1)
			let qv = |g: &mut Box<dyn DenseGpu>, n: usize, src, fl: u32, pre: &[f32]| {
				g.vec(
					0,
					n,
					src,
					X0,
					0,
					1,
					0.0,
					rot_blk(n),
					fl | 4,
					None,
					None,
					Some(pre),
				)
			};
			let vp = qv(&mut g, W, YMIX, 3, &ppre)?;
			let vo = qv(&mut g, H * D, X0, 2, &opre)?;
			let vn = g.vec(1, W, XS, X0, 0, HC, 1e-6, 0, 0, None, None, Some(&hcm.norm))?;
			let vm = g.vec(
				0,
				W,
				YMIX,
				X1,
				0,
				1,
				0.0,
				rot_blk(W),
				7,
				Some(YMQ),
				None,
				Some(&mpre),
			)?;
			let vsd = qv(&mut g, FE, X0, 2, &sdpre)?;
			let merged = if merge() && std::env::var("QWEN_FLASH_CHAIN").as_deref() != Ok("0") {
				let q = 1.0 / HC as f32;
				hca.silu = g.set_silu(hca.down, q, X1)?;
				hcm.silu = g.set_silu(hcm.down, q, X1)?;
				Some(Merged {
					vp: g.vec(
						5,
						W,
						0,
						X0,
						0,
						HC,
						q,
						rot_blk(W),
						6,
						None,
						None,
						Some(&ppre),
					)?,
					vn: g.vec(
						1,
						W,
						XS,
						X0,
						XINJ,
						HC,
						1e-6,
						0,
						8,
						Some(YX1),
						Some(0),
						Some(&hcm.norm),
					)?,
					vm: g.vec(
						5,
						W,
						0,
						X1,
						0,
						HC,
						q,
						rot_blk(W),
						6,
						Some(YMQ),
						Some(X0),
						Some(&mpre),
					)?,
					vsd: g.vec(
						0,
						FE,
						1024,
						X0,
						0,
						1,
						0.0,
						rot_blk(FE),
						14,
						None,
						None,
						Some(&sdpre),
					)?,
				})
			} else {
				None
			};
			layers.push(Layer {
				hca,
				hcm,
				ppre,
				opre,
				mix,
				o,
				router,
				sgate,
				sgu,
				sd,
				mpre,
				sdpre,
				vp,
				vo,
				vn,
				vm,
				vsd,
				merged,
			});
		}
		eprintln!(
			"flash: dense layers on the GPU in {:.1} s",
			t0.elapsed().as_secs_f64()
		);
		let q = 1.0 / HC as f32;
		let silu = g.act(0, HC_RANK, 0, X1, 0, 0, q)?;
		let mix = g.act(1, W, 0, YMIX, 0, HC, q)?;
		let chain = if std::env::var("QWEN_FLASH_CHAIN").as_deref() == Ok("0") {
			None
		} else {
			Some([
				g.vec(2, W, 0, XS, XINJ, HC, 0.0, 0, 0, Some(YX1), None, None)?,
				g.vec(0, W, YMIX, X0, 0, 1, 0.0, 0, 1, None, None, None)?,
				g.vec(3, FE, 1024, X0, 0, 1, 0.0, 0, 0, None, None, None)?,
			])
		};
		let mut dmix = Vec::new();
		let mut dvo = Vec::new();
		let mut cinj = None;
		if chain.is_some() && mix_gpu() {
			if fuse() {
				cinj = Some(g.vec(
					4,
					HC,
					YINJ,
					XINJ,
					0,
					1,
					1.0 / HC as f32,
					0,
					1,
					None,
					None,
					None,
				)?);
			}
			let nd = NL - NL / 4;
			g.delta_init(nd)?;
			for (di, ly) in layers
				.iter()
				.filter(|l| matches!(l.mix, Mix::Delta { .. }))
				.enumerate()
			{
				let Mix::Delta {
					conv,
					dtb,
					acoef,
					gnw,
					..
				} = &ly.mix
				else {
					unreachable!()
				};
				if cinj.is_some() {
					let n = H * D;
					dvo.push(g.vec(
						0,
						n,
						YA,
						X0,
						0,
						1,
						0.0,
						rot_blk(n),
						7,
						None,
						None,
						Some(&ly.opre),
					)?);
				}
				let w0 = [conv.as_slice(), dtb, acoef].concat();
				let (yz, yab) = (NQKV, 16384);
				let mut op = |mode, flags, w: &[f32]| g.delta(mode, flags, di, yz, yab, YA, w);
				dmix.push([
					op(0, 0, &w0)?,
					op(1, 0, gnw)?,
					op(0, 1, &w0)?,
					op(1, 1, gnw)?,
					op(2, 0, &[])?,
					op(1, 2, gnw)?,
					op(2, 4, &[])?,
					op(1, 6, gnw)?,
				]);
			}
		}
		Ok(GpuDec {
			dmix,
			dvo,
			cinj,
			chain,
			g,
			layers,
			ple: ple.ok_or("flash: no PLE layer")?,
			silu,
			mix,
			head: Vec::new(),
			sgates: [Vec::new(), Vec::new()],
			xscr: Vec::new(),
			wave: 0,
			shared_async: std::env::var("QWEN_FLASH_SHARED_ASYNC").as_deref() != Ok("0"),
			mtp: None,
			t_gpu: 0.0,
			t_site: [0.0; 5],
		})
	}

	/// Upload the int8 lm_head's first `rows` rows (codes `[VOCAB][W]`,
	/// row scales).
	pub fn add_head(&mut self, c: &[i8], s: &[f32], rows: usize) -> Result<(), String> {
		let rows = rows.min(super::VOCAB);
		for r0 in (0..rows).step_by(HEAD_ROWS) {
			let n = HEAD_ROWS.min(rows - r0);
			let m = Mat::I8(&c[r0 * W..(r0 + n) * W], &s[r0..r0 + n]);
			self.head.push(self.g.add(m, n, W, X0, r0)?);
		}
		Ok(())
	}

	/// The logits of the final mixer's output `m` `[W]`, if the lm_head is
	/// on the GPU.
	pub fn head(&mut self, m: &[f32]) -> Result<Option<Vec<f32>>, String> {
		Ok(self.head_n(&[m])?.map(|mut v| v.remove(0)))
	}

	/// [`Self::head`] of several positions' mixer outputs in one pass over
	/// the lm_head.
	pub fn head_n(&mut self, ms: &[&[f32]]) -> Result<Option<Vec<Vec<f32>>>, String> {
		if self.head.is_empty() {
			return Ok(None);
		}
		if self.head.len() * HEAD_ROWS < super::VOCAB {
			return Err("gpudec: the GPU holds only part of the lm_head".into());
		}
		let ids = self.head.clone();
		self.run_n(ms, &ids)?;
		Ok(Some(
			(0..ms.len())
				.map(|p| self.g.y_at(p)[..super::VOCAB].to_vec())
				.collect(),
		))
	}

	/// The lm_head's row blocks on the GPU ([`HEAD_ROWS`] rows each).
	pub fn head_blocks(&self) -> usize {
		self.head.len()
	}

	/// [`Self::head_n`] over the first `k` row blocks only, left in flight
	/// while the cores take the rest: rows `[0, k HEAD_ROWS)` of each
	/// position's logits after [`Self::head_wait`].
	pub fn head_begin(&mut self, ms: &[&[f32]], k: usize) -> Result<(), String> {
		let t = std::time::Instant::now();
		let ids = self.head[..k].to_vec();
		self.g.run_chain_n_begin(ms, &[&ids])?;
		self.tick(t, 3);
		Ok(())
	}

	/// Wait for [`Self::head_begin`] and copy rows `[0, rows)` of
	/// position `p`'s logits into `outs[p]`.
	pub fn head_wait(&mut self, outs: &mut [Vec<f32>], rows: usize) -> Result<(), String> {
		let t = std::time::Instant::now();
		self.g.run_wait()?;
		self.tick(t, 3);
		for (p, o) in outs.iter_mut().enumerate() {
			o[..rows].copy_from_slice(&self.g.y_at(p)[..rows]);
		}
		Ok(())
	}

	/// The most positions one step takes ([`DenseGpu::max_np`]); 1 in the
	/// unchained form.
	pub fn max_np(&self) -> usize {
		if self.chain.is_none() {
			1
		} else {
			self.g.max_np()
		}
	}

	/// True when a step can run as two waves ([`Self::set_wave`]): the GPU
	/// has them and the delta layers run fused on it.
	pub fn waves(&self) -> bool {
		self.g.has_waves() && self.chain.is_some() && self.cinj.is_some() && self.mix_gpu()
	}

	/// Make wave `w` the one the next runs and their outputs refer to
	/// ([`DenseGpu::set_wave`]; `p0` = wave 0's positions for wave 1, else
	/// 0). A step of two waves runs each wave's positions through every
	/// method here as a step of its own, wave 0's before wave 1's in each
	/// layer: a run of each can be in flight (the shared expert of one
	/// beside the other's layer).
	pub fn set_wave(&mut self, w: usize, p0: usize) {
		self.wave = w;
		self.g.set_wave(w, p0);
	}

	/// Set whether a post run leaves the shared expert running on its own
	/// ([`Self::shared_n`]) or computes it in the post's submission (same
	/// kernels, same values); returns the previous setting.
	pub fn set_shared_async(&mut self, on: bool) -> bool {
		std::mem::replace(&mut self.shared_async, on)
	}

	/// `n` input vectors of a post run's largest layout taken out of
	/// `xscr` (put back after the run is begun); their gaps hold whatever
	/// earlier runs left, which no stage reads.
	fn scratch(&mut self, n: usize) -> Vec<Vec<f32>> {
		let mut v = std::mem::take(&mut self.xscr);
		Self::scratch_fit(&mut v, n);
		v
	}

	fn scratch_fit(v: &mut Vec<Vec<f32>>, n: usize) {
		while v.len() < n {
			v.push(vec![0f32; XINJ + HC]);
		}
	}

	/// The streams `xs` normed per stream with `w` (as [`norm_g`]) into
	/// `v[p][..HCW]` of scratch vectors fitted to `xs.len()`.
	fn norm_scr(v: &mut Vec<Vec<f32>>, xs: &[Vec<f32>], w: &[f32]) {
		Self::scratch_fit(v, xs.len());
		for (v, x) in v.iter_mut().zip(xs) {
			v[..HCW].copy_from_slice(x);
			for (r, w) in v[..HCW].chunks_exact_mut(W).zip(w.chunks_exact(W)) {
				rmsw(r, w, 1e-6);
			}
		}
	}

	fn tick(&mut self, t: std::time::Instant, site: usize) {
		let e = t.elapsed().as_secs_f64();
		self.t_gpu += e;
		self.t_site[site] += e;
	}

	fn run(&mut self, x: &[f32], ids: &[usize]) -> Result<&[f32], String> {
		let t = std::time::Instant::now();
		self.g.run(x, ids)?;
		self.tick(t, 4);
		Ok(self.g.y())
	}

	/// [`Self::run`] on several positions: the outputs in `g.y_at(p)`.
	fn run_n(&mut self, xs: &[&[f32]], ids: &[usize]) -> Result<(), String> {
		let t = std::time::Instant::now();
		self.g.run_chain_n(xs, &[ids])?;
		self.tick(t, 4);
		Ok(())
	}

	/// A hyper-connection on the streams `x`: (mixed input `[W]`, injection
	/// weights `[HC]`).
	fn hc_in(&mut self, x: &[f32], l: usize, m: bool) -> Result<(Vec<f32>, [f32; HC]), String> {
		let h = if m {
			&self.layers[l].hcm
		} else {
			&self.layers[l].hca
		};
		let (down, up, inj) = (h.down, h.up, h.inj);
		let xn = norm_g(x, &h.norm);
		let t = std::time::Instant::now();
		self.g
			.run_chain(&xn, &[&[down, inj], &[self.silu], &[up], &[self.mix]])?;
		self.tick(t, 4);
		let y = self.g.y();
		Ok((y[YMIX..YMIX + W].to_vec(), inj_w(y)))
	}

	/// A mixer's projections `proj` on the streams `xs[p]` of each position,
	/// the attention hyper-connection chained in front when the form is
	/// chained: the output regions in `g.y_at(p)`; the injection weights.
	fn proj_n(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
		proj: &[usize],
	) -> Result<Vec<[f32; HC]>, String> {
		if self.chain.is_none() {
			if xs.len() != 1 {
				return Err("flash: several positions need the chained form".into());
			}
			let (mixed, inj) = self.hc_in(&xs[0], l, false)?;
			let xq = qin(&mixed, &self.layers[l].ppre);
			self.run(&xq, proj)?;
			return Ok(vec![inj]);
		}
		self.proj_begin(xs, l, proj)?;
		let t = std::time::Instant::now();
		self.g.run_wait()?;
		self.tick(t, 0);
		Ok((0..xs.len()).map(|p| inj_w(self.g.y_at(p))).collect())
	}

	/// [`Self::proj_n`]'s run (the chained form) left in flight.
	fn proj_begin(&mut self, xs: &[Vec<f32>], l: usize, proj: &[usize]) -> Result<(), String> {
		let ly = &self.layers[l];
		let h = &ly.hca;
		let mut xin = std::mem::take(&mut self.xscr);
		Self::norm_scr(&mut xin, xs, &h.norm);
		let xr: Vec<&[f32]> = xin[..xs.len()].iter().map(|v| &v[..HCW]).collect();
		let t = std::time::Instant::now();
		let (hd, silu, up, mix, vp) = ([h.down, h.inj], [self.silu], [h.up], [self.mix], [ly.vp]);
		let mut st: Vec<&[usize]> = vec![&hd];
		if !h.silu {
			st.push(&silu);
		}
		st.push(&up);
		let vpm;
		if let Some(m) = ly.merged {
			vpm = [m.vp];
			st.push(&vpm);
		} else {
			st.extend([&mix[..], &vp]);
		}
		st.push(proj);
		let r = self.g.run_chain_n_begin(&xr, &st);
		self.xscr = xin;
		r?;
		self.tick(t, 0);
		Ok(())
	}

	/// The PLE term of a step on the streams `xs[t]` of each position, in
	/// order: `es[t]` the token's n-gram embedding `[W]`, `cst` the conv
	/// window `[9][HCW]` f16 (updated per position; with `save`, a copy
	/// after each position is pushed there).
	pub fn ple_n(
		&mut self,
		xs: &mut [Vec<f32>],
		es: &[Vec<f32>],
		cst: &mut [u16],
		mut save: Option<&mut Vec<Vec<u16>>>,
	) -> Result<(), String> {
		let xq: Vec<Vec<f32>> = es.iter().map(|e| qin(e, &self.ple.pre)).collect();
		let xr: Vec<&[f32]> = xq.iter().map(|v| v.as_slice()).collect();
		let (pk, pv) = (self.ple.pk, self.ple.pv);
		self.run_n(&xr, &[pk, pv])?;
		for (t, x) in xs.iter_mut().enumerate() {
			let y = self.g.y_at(t);
			let (kr, val) = (y[..HCW].to_vec(), y[HCW..HCW + W].to_vec());
			self.ple_host(x, &kr, &val, cst);
			if let Some(v) = save.as_mut() {
				v.push(cst.to_vec());
			}
		}
		Ok(())
	}

	/// [`Self::ple_n`]'s host part of one position: `kr` / `val` the key
	/// and value projections.
	fn ple_host(&self, x: &mut [f32], kr: &[f32], val: &[f32], cst: &mut [u16]) {
		let p = &self.ple;
		let key = norm_g(kr, &p.nkey);
		let q = norm_g(x, &p.nquery);
		let mut gv = vec![0f32; HCW];
		for c in 0..HC {
			let g: f32 = key[c * W..][..W]
				.iter()
				.zip(&q[c * W..][..W])
				.map(|(a, b)| a * b)
				.sum::<f32>()
				/ (W as f32).sqrt();
			let g = sigmoid(g / g.abs().max(1e-6).sqrt());
			for i in 0..W {
				gv[c * W + i] = g * val[i];
			}
		}
		let gn = norm_g(&gv, &p.nconv);
		let n = super::ngram::N;
		let back = 3 * n;
		let un = crate::ir::f16_to_f32;
		for i in 0..HCW {
			let w = &p.pconv[i * 4..][..4];
			let mut a = w[3] * gn[i];
			for j in 0..3 {
				a += w[j] * un(cst[j * n * HCW + i]);
			}
			x[i] += gv[i] + silu(a);
		}
		cst.copy_within(HCW.., 0);
		for (o, &v) in cst[(back - 1) * HCW..].iter_mut().zip(&gn) {
			*o = crate::ir::f16_bits(v);
		}
	}

	/// A delta layer's projections of each position: (qkv `[NQKV]`, z `[H
	/// D]`, ab `[2 H]`), and the injection weights.
	#[allow(clippy::type_complexity)]
	pub fn delta_pre_n(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
	) -> Result<Vec<(Vec<f32>, Vec<f32>, Vec<f32>, [f32; HC])>, String> {
		let Mix::Delta { proj, .. } = self.layers[l].mix else {
			return Err("flash: not a delta layer".into());
		};
		let inj = self.proj_n(xs, l, &proj)?;
		Ok(inj
			.into_iter()
			.enumerate()
			.map(|(p, inj)| {
				let y = self.g.y_at(p);
				(
					y[..NQKV].to_vec(),
					y[NQKV..NQKV + H * D].to_vec(),
					y[16384..16384 + 2 * H].to_vec(),
					inj,
				)
			})
			.collect())
	}

	/// The delta mixers run on the GPU ([`Self::delta_mix_n`]), their
	/// states in VRAM.
	pub fn mix_gpu(&self) -> bool {
		!self.dmix.is_empty()
	}

	/// A delta layer's whole mixer on the GPU in the submission of its
	/// projections ([`Self::delta_pre_n`] + `delta_conv_n` + the recurrence +
	/// [`Self::delta_gate_n`]), `di` its delta index: the gated outputs (the
	/// output projection's inputs) and the injection weights. `save`: keep
	/// what [`Self::delta_rollback`] needs.
	#[allow(clippy::type_complexity)]
	pub fn delta_mix_n(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
		di: usize,
		save: bool,
	) -> Result<(Vec<Vec<f32>>, Vec<[f32; HC]>), String> {
		let Mix::Delta { proj, .. } = self.layers[l].mix else {
			return Err("flash: not a delta layer".into());
		};
		let ops = self.dmix[di];
		let (cv, rc) = if save {
			(ops[2], ops[3])
		} else {
			(ops[0], ops[1])
		};
		let ly = &self.layers[l];
		let h = &ly.hca;
		let mut xin = std::mem::take(&mut self.xscr);
		Self::norm_scr(&mut xin, xs, &h.norm);
		let xr: Vec<&[f32]> = xin[..xs.len()].iter().map(|v| &v[..HCW]).collect();
		let t = std::time::Instant::now();
		// the host reads the gated outputs and the injection weights only
		self.g.io_next(&[], &[YINJ..YINJ + HC, YA..YA + H * D]);
		let r = self.g.run_chain_n(
			&xr,
			&[
				&[h.down, h.inj],
				&[self.silu],
				&[h.up],
				&[self.mix],
				&[ly.vp],
				&proj,
				&[cv],
				&[rc],
			],
		);
		self.xscr = xin;
		r?;
		self.tick(t, 0);
		Ok((0..xs.len())
			.map(|p| {
				let y = self.g.y_at(p);
				(y[YA..YA + H * D].to_vec(), inj_w(y))
			})
			.unzip())
	}

	/// Keep the first `n` positions of the last saving
	/// [`Self::delta_mix_n`] step of every delta layer: the states replayed
	/// over them from their backups, the windows from the saved ones.
	pub fn delta_rollback(&mut self, n: usize) -> Result<(), String> {
		// (a step of two waves saved slots 0..n of both)
		self.set_wave(0, 0);
		let ids: Vec<usize> = self
			.dmix
			.iter()
			.flat_map(|o| if n == 0 { [o[6], o[7]] } else { [o[4], o[5]] })
			.collect();
		let none: Vec<&[f32]> = vec![&[]; n.max(1)];
		self.g.run_chain_n(&none, &[&ids])
	}

	/// Write delta layer `di`'s state and conv window (f16) to the GPU.
	pub fn delta_put(&mut self, di: usize, st: &[f32], win: &[u16]) -> Result<(), String> {
		let w: Vec<f32> = win.iter().map(|&v| crate::ir::f16_to_f32(v)).collect();
		self.g.delta_put(di, st, &w)
	}

	/// Read delta layer `di`'s state and conv window back.
	pub fn delta_get(&mut self, di: usize, st: &mut [f32], win: &mut [u16]) -> Result<(), String> {
		let mut w = vec![0f32; win.len()];
		self.g.delta_get(di, st, &mut w)?;
		for (o, &v) in win.iter_mut().zip(&w) {
			*o = crate::ir::f16_bits(v);
		}
		Ok(())
	}

	/// The recurrence's inputs of a delta layer's step, each position of
	/// `pre[t]` = (qkv, ab) in order: conv over the window `cpast`
	/// `[3][NQKV]` f16 (updated) + `qkv`, SiLU, l2 norms, gates: (qn, kn
	/// `[HK D]`, v `[H D]`, gb `[2 H]`). With `save`, the window after each
	/// position is pushed there. Over the pool by blocks of channels (one
	/// thread's scalar SiLU and f16 conversions cost ~65 us per layer and
	/// position); each channel's arithmetic is the one-thread form's.
	#[allow(clippy::type_complexity)]
	pub fn delta_conv_n(
		&self,
		l: usize,
		pre: &[(&[f32], &[f32])],
		cpast: &mut [u16],
		save: Option<&mut Vec<Vec<u16>>>,
	) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
		let Mix::Delta {
			conv, dtb, acoef, ..
		} = &self.layers[l].mix
		else {
			unreachable!()
		};
		let n = pre.len();
		let keep = save.is_some();
		let mut cv = vec![0f32; n * NQKV];
		let mut snap = vec![0u16; if keep { n * 3 * NQKV } else { 0 }];
		#[derive(Clone, Copy)]
		struct P(*mut f32, *mut u16, *mut u16);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let p = P(cv.as_mut_ptr(), cpast.as_mut_ptr(), snap.as_mut_ptr());
		// a block holds whole q / k heads (their l2 norms)
		const CB: usize = 2 * D;
		quant::parallel::par_for(NQKV / CB, |bi| {
			let p = p;
			let un = crate::ir::f16_to_f32;
			for c in bi * CB..(bi + 1) * CB {
				let w = &conv[c * 4..][..4];
				// SAFETY: channel c's window entries belong to this block
				let at = |j: usize| unsafe { p.1.add(j * NQKV + c) };
				let mut pb = [0u16; 3];
				for (j, v) in pb.iter_mut().enumerate() {
					*v = unsafe { *at(j) };
				}
				for (t, (qkv, _)) in pre.iter().enumerate() {
					let mut a = w[3] * qkv[c];
					for j in 0..3 {
						a += w[j] * un(pb[j]);
					}
					// SAFETY: row t's channel c, this block's
					unsafe { *p.0.add(t * NQKV + c) = silu(a) };
					pb = [pb[1], pb[2], crate::ir::f16_bits(qkv[c])];
					if keep {
						for (j, &v) in pb.iter().enumerate() {
							unsafe { *p.2.add(t * 3 * NQKV + j * NQKV + c) = v };
						}
					}
				}
				for (j, &v) in pb.iter().enumerate() {
					unsafe { *at(j) = v };
				}
			}
			if bi * CB < 2 * HK * D {
				for t in 0..n {
					// SAFETY: this block's channels of row t
					let r =
						unsafe { std::slice::from_raw_parts_mut(p.0.add(t * NQKV + bi * CB), CB) };
					for r in r.chunks_exact_mut(D) {
						let s = 1.0 / r.iter().map(|v| v * v).sum::<f32>().max(1e-12).sqrt();
						r.iter_mut().for_each(|v| *v *= s);
					}
				}
			}
		});
		if let Some(sv) = save {
			sv.extend(snap.chunks_exact(3 * NQKV).map(|c| c.to_vec()));
		}
		pre.iter()
			.zip(cv.chunks_exact(NQKV))
			.map(|((_, ab), cv)| {
				let mut gb = vec![0f32; 2 * H];
				for h in 0..H {
					let a = ab[h] + dtb[h];
					let sp = if a > 20.0 { a } else { a.exp().ln_1p() };
					gb[h] = sp * acoef[h];
					gb[H + h] = sigmoid(ab[H + h]);
				}
				(
					cv[..HK * D].to_vec(),
					cv[HK * D..2 * HK * D].to_vec(),
					cv[2 * HK * D..].to_vec(),
					gb,
				)
			})
			.collect()
	}

	/// The gated norm of a delta layer's recurrence outputs `os[t]` (x
	/// `oscale`) with the gate inputs `zs[t]`: the output projection's
	/// inputs. Over the pool by heads.
	pub fn delta_gate_n(
		&self,
		l: usize,
		os: &[Vec<f32>],
		zs: &[&[f32]],
		oscale: f32,
	) -> Vec<Vec<f32>> {
		let Mix::Delta { gnw, .. } = &self.layers[l].mix else {
			unreachable!()
		};
		let mut a: Vec<Vec<f32>> = os.to_vec();
		#[derive(Clone, Copy)]
		struct P(*const *mut f32);
		unsafe impl Send for P {}
		unsafe impl Sync for P {}
		let ptrs: Vec<*mut f32> = a.iter_mut().map(|v| v.as_mut_ptr()).collect();
		let p = P(ptrs.as_ptr());
		const HB: usize = 8;
		quant::parallel::par_for(os.len() * (H / HB), |i| {
			let p = p;
			let (t, h0) = (i / (H / HB), i % (H / HB) * HB);
			// SAFETY: heads [h0, h0 + HB) of row t belong to this task
			let r = unsafe { std::slice::from_raw_parts_mut((*p.0.add(t)).add(h0 * D), HB * D) };
			let zr = &zs[t][h0 * D..][..HB * D];
			for (r, zr) in r.chunks_exact_mut(D).zip(zr.chunks_exact(D)) {
				rmsw(r, gnw, 1e-6 * oscale * oscale);
				r.iter_mut().zip(zr).for_each(|(v, &z)| *v *= sigmoid(z));
			}
		});
		a
	}

	/// An attention layer's projections of each position at its RoPE
	/// `cs[p]` / `sn[p]` `[RD]`: (q `[HQ][DA]` normed and roped, gate `[HQ
	/// DA]` (sigmoid), k `[HKV][DA]` normed and roped, v, iq `[IH][ID]`
	/// normed and roped, ik `[ID]`), and the injection weights.
	#[allow(clippy::type_complexity)]
	pub fn attn_pre_n(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
		cs: &[[f32; RD]],
		sn: &[[f32; RD]],
	) -> Result<Vec<([Vec<f32>; 6], [f32; HC])>, String> {
		if self.chain.is_none() {
			let Mix::Attn { proj, .. } = self.layers[l].mix else {
				return Err("flash: not an attention layer".into());
			};
			let injs = self.proj_n(xs, l, &proj)?;
			return Ok(self.attn_pre_out(injs, l, cs, sn));
		}
		self.attn_pre_begin(xs, l)?;
		self.attn_pre_end(xs.len(), l, cs, sn)
	}

	/// [`Self::attn_pre_n`]'s run left in flight (the chained form);
	/// [`Self::attn_pre_end`] collects it.
	pub fn attn_pre_begin(&mut self, xs: &[Vec<f32>], l: usize) -> Result<(), String> {
		let Mix::Attn { proj, .. } = self.layers[l].mix else {
			return Err("flash: not an attention layer".into());
		};
		self.proj_begin(xs, l, &proj)
	}

	/// Wait for [`Self::attn_pre_begin`]'s run of `n` positions: what
	/// [`Self::attn_pre_n`] returns.
	#[allow(clippy::type_complexity)]
	pub fn attn_pre_end(
		&mut self,
		n: usize,
		l: usize,
		cs: &[[f32; RD]],
		sn: &[[f32; RD]],
	) -> Result<Vec<([Vec<f32>; 6], [f32; HC])>, String> {
		let t = std::time::Instant::now();
		self.g.run_wait()?;
		self.tick(t, 0);
		let injs = (0..n).map(|p| inj_w(self.g.y_at(p))).collect();
		Ok(self.attn_pre_out(injs, l, cs, sn))
	}

	#[allow(clippy::type_complexity)]
	fn attn_pre_out(
		&self,
		injs: Vec<[f32; HC]>,
		l: usize,
		cs: &[[f32; RD]],
		sn: &[[f32; RD]],
	) -> Vec<([Vec<f32>; 6], [f32; HC])> {
		let mut out = Vec::with_capacity(injs.len());
		for (p, inj) in injs.into_iter().enumerate() {
			let (cs, sn) = (&cs[p], &sn[p]);
			// (read in place: the region is GEMV_Y floats, 1 MB)
			let y = self.g.y_at(p);
			let Mix::Attn { qn, kn, iqn, .. } = &self.layers[l].mix else {
				unreachable!()
			};
			let (mut q, mut gate) = (vec![0f32; HQ * DA], vec![0f32; HQ * DA]);
			// over the pool by heads (one thread's scalar sigmoids: ~30 us)
			#[derive(Clone, Copy)]
			struct P(*mut f32, *mut f32);
			unsafe impl Send for P {}
			unsafe impl Sync for P {}
			let pp = P(q.as_mut_ptr(), gate.as_mut_ptr());
			quant::parallel::par_for(HQ, |h| {
				let pp = pp;
				// SAFETY: head h's rows of q and gate belong to this task
				let qr = unsafe { std::slice::from_raw_parts_mut(pp.0.add(h * DA), DA) };
				qr.copy_from_slice(&y[h * 2 * DA..][..DA]);
				rmsw(qr, qn, 1e-6);
				rope(qr, cs, sn);
				let gr = unsafe { std::slice::from_raw_parts_mut(pp.1.add(h * DA), DA) };
				for (o, &v) in gr.iter_mut().zip(&y[h * 2 * DA + DA..][..DA]) {
					*o = sigmoid(v);
				}
			});
			let mut k = y[12288..12288 + HKV * DA].to_vec();
			for r in k.chunks_exact_mut(DA) {
				rmsw(r, kn, 1e-6);
				rope(r, cs, sn);
			}
			let v = y[12800..12800 + HKV * DA].to_vec();
			let mut iq = y[13312..13312 + IH * ID].to_vec();
			for r in iq.chunks_exact_mut(ID) {
				rmsw(r, iqn, 1e-6);
				rope(r, cs, sn);
			}
			let ik = y[13312 + IH * ID..13312 + (IH + 1) * ID].to_vec();
			out.push(([q, gate, k, v, iq, ik], inj));
		}
		out
	}

	/// After the mixer: output projection of `a` (the mixer's gated output),
	/// injection into `x` (the streams become `x1`), the MoE's
	/// hyper-connection, router logits, the shared expert: (moe input
	/// prescaled and rotated `[W]`, inj_m, logits `[NE]`, shared `[W]`).
	/// [`Self::post`] of each position (`xs[p]`, `a[p]`, `inj[p]`) in one
	/// submission; several positions need the chained form. In the chained
	/// form the shared expert (gate_up, SwiGLU, down) is left running on
	/// the GPU while the cores do the routed experts: its `shared` is empty
	/// until [`Self::shared_n`] (`QWEN_FLASH_SHARED_ASYNC=0`: one
	/// submission, as before).
	#[allow(clippy::type_complexity)]
	pub fn post_n(
		&mut self,
		xs: &mut [Vec<f32>],
		l: usize,
		a: &[Vec<f32>],
		inj: &[[f32; HC]],
	) -> Result<Vec<(Vec<f32>, [f32; HC], Vec<f32>, Vec<f32>)>, String> {
		if self.chain.is_none() {
			if xs.len() != 1 {
				return Err("flash: several positions need the chained form".into());
			}
			return Ok(vec![self.post(&mut xs[0], l, &a[0], &inj[0])?]);
		}
		self.post_begin(xs, l, a, inj)?;
		self.post_end(xs, l, 1)
	}

	/// [`Self::post_n`]'s run (the chained form) left in flight;
	/// [`Self::post_end`] collects it.
	pub fn post_begin(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
		a: &[Vec<f32>],
		inj: &[[f32; HC]],
	) -> Result<(), String> {
		let mut xin = self.scratch(xs.len());
		for (p, v) in xin.iter_mut().enumerate().take(xs.len()) {
			v[..a[p].len()].copy_from_slice(&a[p]);
			v[XS..XS + HCW].copy_from_slice(&xs[p]);
			v[XINJ..XINJ + HC].copy_from_slice(&inj[p]);
		}
		let xr: Vec<&[f32]> = xin[..xs.len()].iter().map(|v| &v[..XINJ + HC]).collect();
		let t = std::time::Instant::now();
		// what the host writes of this run (the gaps of `xin` are never read)
		let al = a.first().map_or(0, |v| v.len());
		let vo = self.layers[l].vo;
		let r = self.post_run_begin(l, &xr, &[], &[vo], &[0..al, XS..XS + HCW, XINJ..XINJ + HC]);
		self.xscr = xin;
		r?;
		self.tick(t, 1);
		Ok(())
	}

	/// The fused form of a delta layer ([`Self::delta_mix_n`] then
	/// [`Self::post_n`]) in one submission: the post reads the mixer's
	/// gated output and injection weights where the mixer left them on the
	/// GPU, the streams are uploaded with the mixer's input. Bit-identical
	/// to the two runs. `None` when the fused form is off.
	#[allow(clippy::type_complexity)]
	pub fn delta_post_n(
		&mut self,
		xs: &mut [Vec<f32>],
		l: usize,
		di: usize,
		save: bool,
	) -> Result<Option<Post>, String> {
		if !self.delta_post_begin(xs, l, di, save)? {
			return Ok(None);
		}
		Ok(Some(self.post_end(xs, l, 0)?))
	}

	/// [`Self::delta_post_n`]'s run left in flight (false when the fused
	/// form is off); [`Self::post_end`] collects it.
	pub fn delta_post_begin(
		&mut self,
		xs: &[Vec<f32>],
		l: usize,
		di: usize,
		save: bool,
	) -> Result<bool, String> {
		let Some(cinj) = self.cinj else {
			return Ok(false);
		};
		let Mix::Delta { proj, .. } = self.layers[l].mix else {
			return Err("flash: not a delta layer".into());
		};
		let ops = self.dmix[di];
		let (cv, rc) = if save {
			(ops[2], ops[3])
		} else {
			(ops[0], ops[1])
		};
		let ly = &self.layers[l];
		let h = &ly.hca;
		let mut xin = std::mem::take(&mut self.xscr);
		Self::scratch_fit(&mut xin, xs.len());
		for (v, x) in xin.iter_mut().zip(xs) {
			// ([`norm_g`] in place)
			v[..HCW].copy_from_slice(x);
			for (r, w) in v[..HCW].chunks_exact_mut(W).zip(h.norm.chunks_exact(W)) {
				rmsw(r, w, 1e-6);
			}
			v[XS..XS + HCW].copy_from_slice(x);
		}
		let xr: Vec<&[f32]> = xin[..xs.len()].iter().map(|v| &v[..XS + HCW]).collect();
		let (hd, silu, up, mix, vp) = ([h.down, h.inj], [self.silu], [h.up], [self.mix], [ly.vp]);
		let mut pre: Vec<&[usize]> = vec![&hd];
		if !h.silu {
			pre.push(&silu);
		}
		pre.push(&up);
		let vpm;
		if let Some(m) = ly.merged {
			vpm = [m.vp];
			pre.push(&vpm);
		} else {
			pre.extend([&mix[..], &vp]);
		}
		let (cv, rc) = ([cv], [rc]);
		pre.extend([&proj[..], &cv, &rc]);
		let first = [self.dvo[di], cinj];
		let t = std::time::Instant::now();
		let r = self.post_run_begin(l, &xr, &pre, &first, &[0..HCW, XS..XS + HCW]);
		self.xscr = xin;
		r?;
		self.tick(t, 0);
		Ok(true)
	}

	/// Wait for the post run [`Self::post_begin`] / [`Self::delta_post_begin`]
	/// left in flight on the current wave and read its outputs (`site`: the
	/// `t_site` the wait counts to); the streams `xs` become `x1`.
	pub fn post_end(&mut self, xs: &mut [Vec<f32>], l: usize, site: usize) -> Result<Post, String> {
		let t = std::time::Instant::now();
		self.g.run_wait()?;
		self.tick(t, site);
		self.post_run_end(xs, l)
	}

	/// [`Self::post_n`]'s run: the stages `pre`, then the post with `first`
	/// as its first stage (qin of the output projection's input), on the
	/// inputs `xr` of which the host writes `xio`; reads the outputs back.
	fn post_run_begin(
		&mut self,
		l: usize,
		xr: &[&[f32]],
		pre: &[&[usize]],
		first: &[usize],
		xio: &[std::ops::Range<usize>],
	) -> Result<(), String> {
		let Some([vinj, vcopy, vsw]) = self.chain else {
			unreachable!()
		};
		let ly = &self.layers[l];
		let h = &ly.hcm;
		let split = self.shared_async;
		// what the host reads of this run (the shared expert's output only
		// without `split`)
		let ysd = if split { 0..0 } else { YSD..YSD + W };
		self.g.io_next(
			xio,
			&[
				0..NE + 1,
				ysd,
				YINJ..YINJ + HC,
				YX1..YX1 + HCW,
				YMQ..YMQ + W,
			],
		);
		let (o, hd, silu, up) = ([ly.o], [h.down, h.inj], [self.silu], [h.up]);
		let (vinj, vn, mix, vm) = ([vinj], [ly.vn], [self.mix], [vcopy, ly.vm]);
		let (sgu, vsw, vsd, sd) = ([ly.sgu], [vsw], [ly.vsd], [ly.sd]);
		let (mn, mm, msd);
		let (hc, sh): (Vec<&[usize]>, Vec<&[usize]>) = match ly.merged {
			Some(m) => {
				(mn, mm, msd) = ([m.vn], [m.vm], [m.vsd]);
				let mut hc = vec![first, &o, &mn, &hd];
				if !h.silu {
					hc.push(&silu);
				}
				hc.extend([&up[..], &mm]);
				(hc, vec![&sgu, &msd, &sd])
			}
			None => (
				vec![first, &o, &vinj, &vn, &hd, &silu, &up, &mix, &vm],
				vec![&sgu, &vsw, &vsd, &sd],
			),
		};
		let (r2, r3) = ([ly.router, ly.sgate], [ly.router, ly.sgate, ly.sgu]);
		let mut st = pre.to_vec();
		st.extend_from_slice(&hc);
		if split {
			st.push(&r2);
		} else {
			st.push(&r3);
			st.extend_from_slice(&sh[1..]);
		}
		self.g.run_chain_n_begin(xr, &st)
	}

	/// [`Self::post_run_begin`]'s outputs once its run is done.
	fn post_run_end(&mut self, xs: &mut [Vec<f32>], l: usize) -> Result<Post, String> {
		let split = self.shared_async;
		let mut out = Vec::with_capacity(xs.len());
		self.sgates[self.wave].clear();
		for (p, x) in xs.iter_mut().enumerate() {
			let y = self.g.y_at(p);
			x.copy_from_slice(&y[YX1..YX1 + HCW]);
			let gt = sigmoid(y[512]);
			let shared: Vec<f32> = if split {
				self.sgates[self.wave].push(gt);
				Vec::new()
			} else {
				y[YSD..YSD + W].iter().map(|v| v * gt).collect()
			};
			out.push((y[YMQ..YMQ + W].to_vec(), inj_w(y), y[..NE].to_vec(), shared));
		}
		if split {
			// reads the MoE input (int8, x region) the run above left on
			// the device; this run's own input upload touches only floats
			// [0, 1) of each position, which SwiGLU overwrites first
			let ly = &self.layers[l];
			let Some([_, _, vsw]) = self.chain else {
				unreachable!()
			};
			let (sgu, vsw, vsd, sd) = ([ly.sgu], [vsw], [ly.vsd], [ly.sd]);
			let msd;
			let sh: Vec<&[usize]> = match ly.merged {
				Some(m) => {
					msd = [m.vsd];
					vec![&sgu, &msd, &sd]
				}
				None => vec![&sgu, &vsw, &vsd, &sd],
			};
			let none: Vec<&[f32]> = vec![&[]; xs.len()];
			self.g.run_chain_n_begin(&none, &sh)?;
		}
		Ok(out)
	}

	/// The shared experts' outputs of the positions [`Self::post_n`] left
	/// running, into `post[p].3`; nothing when it did not.
	#[allow(clippy::type_complexity)]
	pub fn shared_n(
		&mut self,
		post: &mut [(Vec<f32>, [f32; HC], Vec<f32>, Vec<f32>)],
	) -> Result<(), String> {
		if self.sgates[self.wave].is_empty() {
			return Ok(());
		}
		let t = std::time::Instant::now();
		self.g.run_wait()?;
		self.tick(t, 2);
		for (p, (o, &gt)) in post.iter_mut().zip(&self.sgates[self.wave]).enumerate() {
			o.3 = self.g.y_at(p)[YSD..YSD + W]
				.iter()
				.map(|v| v * gt)
				.collect();
		}
		self.sgates[self.wave].clear();
		Ok(())
	}

	/// One position of [`Self::post_n`] in the unchained form.
	#[allow(clippy::type_complexity)]
	fn post(
		&mut self,
		x: &mut [f32],
		l: usize,
		a: &[f32],
		inj: &[f32; HC],
	) -> Result<(Vec<f32>, [f32; HC], Vec<f32>, Vec<f32>), String> {
		let xq = qin(a, &self.layers[l].opre);
		let o = self.layers[l].o;
		let y = self.run(&xq, &[o])?;
		for c in 0..HC {
			for i in 0..W {
				x[c * W + i] += y[i] * inj[c];
			}
		}
		let (m, inj_m) = self.hc_in(x, l, true)?;
		let ly = &self.layers[l];
		let mq = rot(&m, &ly.mpre);
		let mut xin = vec![0f32; X1 + W];
		xin[..W].copy_from_slice(&m);
		for (o, &v) in xin[X1..].iter_mut().zip(&mq) {
			*o = v.round_ties_even().clamp(-128.0, 127.0);
		}
		let ids = [ly.router, ly.sgate, ly.sgu];
		let y = self.run(&xin, &ids)?;
		let logits = y[..NE].to_vec();
		let gt = sigmoid(y[512]);
		let gu = &y[1024..1024 + 2 * FE];
		let h: Vec<f32> = (0..FE).map(|i| silu(gu[i]) * gu[FE + i]).collect();
		let xq = qin(&h, &self.layers[l].sdpre);
		let sd = self.layers[l].sd;
		let y = self.run(&xq, &[sd])?;
		let shared: Vec<f32> = y[YSD..YSD + W].iter().map(|v| v * gt).collect();
		Ok((mq, inj_m, logits, shared))
	}
}

/// The MTP head's dense part on the GPU ([`GpuDec::add_mtp`]): two chained
/// runs per row, the head's attention and routed experts on the host
/// between and after them ([`super::mtp::Mtp::run`]). Drafts only, so its
/// sums need not be the host's bit for bit.
///
/// Run 1 ([`GpuDec::mtp_pre`]): `fc_embedding` and `fc_hidden` per stream
/// -> the streams `e + h[c]` -> the attention hyper-connection (norm,
/// down -> SiLU -> up, mix, injection weights) -> q, k, v.
/// Run 2 ([`GpuDec::mtp_post`]): output projection -> injection into the
/// streams -> the MoE hyper-connection -> router, shared gate / up ->
/// SwiGLU -> shared down.
struct MtpG {
	pre: Vec<Vec<usize>>,
	post: Vec<Vec<usize>>,
}

// run 1's input / output floats
const MXE: usize = 0;
const MXH: usize = 4096;
const MXONE: usize = 16384;
const MXS: usize = 20480;
const MXN: usize = 32768;
const MXU: usize = 45056;
const MXA: usize = 49152;
const MYE: usize = 0;
const MYH: usize = 4096;
const MYX: usize = 16384;
// (the downs write at 0: `Gemv::set_silu` bounds its epilogue by the
// output's end; fce's / o's outputs there are read before)
const MYD: usize = 0;
const MYJ: usize = 29184;
const MYU: usize = 32768;
const MYQ: usize = 45056;
const MYK: usize = MYQ + 2 * HQ * DA;
const MYV: usize = MYK + HKV * DA;
// run 2's
const NXATT: usize = 0;
const NXS: usize = 8192;
const NXINJ: usize = 18432;
const NXN: usize = 20480;
const NXU: usize = 32768;
const NXM: usize = 36864;
const NXSW: usize = 40960;
const NYO: usize = 0;
const NYX: usize = 4096;
const NYD: usize = 0;
const NYJ: usize = 16896;
const NYU: usize = 20480;
const NYM: usize = 32768;
const NYR: usize = 36864;
const NYG: usize = 38912;
const NYSD: usize = 40960;

/// What run 1 gives per row: the streams before the attention `[HCW]`, q
/// (with its gate) `[2 HQ DA]`, k and v `[HKV DA]`, the attention
/// hyper-connection's injection weights.
pub struct MtpPre {
	pub x: Vec<f32>,
	pub q: Vec<f32>,
	pub k: Vec<f32>,
	pub v: Vec<f32>,
	pub inj: [f32; HC],
}

/// What run 2 gives: the streams after the attention's injection `[HCW]`,
/// the MoE input `[W]`, the router logits `[NE]`, the shared expert's down
/// projection before its gate `[W]`, the MoE hyper-connection's injection
/// weights.
pub struct MtpPost {
	pub x: Vec<f32>,
	pub m: Vec<f32>,
	pub logits: Vec<f32>,
	pub shared: Vec<f32>,
	pub inj: [f32; HC],
}

/// One of the MTP head's int8 matrices in `d` (the build's `mtp` step):
/// codes `[n][k]`, f32 row scales.
fn mtp_mat(d: &Path, name: &str, n: usize, k: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
	let rd = |x: &str| {
		let p = d.join(format!("{name}.{x}"));
		std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
	};
	let c = rd("i8")?;
	let s: Vec<f32> = rd("s")?
		.chunks_exact(4)
		.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
		.collect();
	if c.len() != n * k || s.len() != n {
		return Err(format!("mtp/{name}: not [{n}][{k}]"));
	}
	// SAFETY: u8 and i8 have the same layout
	let c = unsafe { std::mem::transmute::<Vec<u8>, Vec<i8>>(c) };
	Ok((c, s))
}

fn mtp_vec(d: &Path, name: &str, n: usize) -> Result<Vec<f32>, String> {
	let p = d.join(format!("{name}.f32"));
	let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
	if b.len() != 4 * n {
		return Err(format!("mtp/{name}: not {n} values"));
	}
	Ok(b.chunks_exact(4)
		.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
		.collect())
}

/// The MTP head's matrices [`GpuDec::add_mtp`] uploads: (rows, inputs,
/// copies).
const MTP_MATS: [(usize, usize, usize); 11] = [
	(W, W, 1),           // fce
	(W, W, HC),          // fch, once per stream
	(HC_RANK, HCW, 2),   // hca.down, hcm.down
	(HCW, HC_RANK, 2),   // hca.up, hcm.up
	(HC, HCW, 2),        // hca.inj, hcm.inj
	(2 * HQ * DA, W, 1), // q
	(HKV * DA, W, 2),    // k, v
	(W, HQ * DA, 1),     // o
	(NE, W, 1),          // router
	(FE, W, 2),          // sg, su
	(W, FE, 1),          // sd
];

/// `QWEN_FLASH_MTP_GPU` (default on): the MTP head's dense part on the
/// dense GPU; `0` keeps the whole head on the cores (the A/B).
pub fn mtp_gpu() -> bool {
	std::env::var("QWEN_FLASH_MTP_GPU").as_deref() != Ok("0")
}

impl GpuDec {
	/// VRAM bytes [`Self::add_mtp`] takes: the matrices (codes and f32
	/// scales), the two norm vectors, slack for alignment.
	pub fn mtp_need() -> u64 {
		let mats: usize = MTP_MATS.iter().map(|&(n, k, c)| c * n * (k + 4)).sum();
		(mats + 2 * 4 * HCW + (1 << 20)) as u64
	}

	/// Upload the MTP head's dense part from `d` (the build's `mtp`).
	pub fn add_mtp(&mut self, d: &Path) -> Result<(), String> {
		let t0 = std::time::Instant::now();
		let g = &mut self.g;
		let q = 1.0 / HC as f32;
		let add = |g: &mut Box<dyn DenseGpu>,
		           name: &str,
		           n: usize,
		           k: usize,
		           xo: usize,
		           yo: usize|
		 -> Result<usize, String> {
			let (c, s) = mtp_mat(d, name, n, k)?;
			g.add(Mat::I8(&c, &s), n, k, xo, yo)
		};
		// a hyper-connection on the normed streams at `xn`: [down + inj],
		// (SiLU), [up], [mix into `xa`, copied to `ym`]
		#[allow(clippy::too_many_arguments)]
		let hc = |g: &mut Box<dyn DenseGpu>,
		    p: &str,
		    xn: usize,
		    xu: usize,
		    xa: usize,
		    yd: usize,
		    yj: usize,
		    yu: usize,
		    ym: Option<usize>|
		 -> Result<Vec<Vec<usize>>, String> {
			let down = add(g, &format!("{p}.down"), HC_RANK, HCW, xn, yd)?;
			let inj = add(g, &format!("{p}.inj"), HC, HCW, xn, yj)?;
			let up = add(g, &format!("{p}.up"), HCW, HC_RANK, xu, yu)?;
			let mut st = vec![vec![down, inj]];
			if !g.set_silu(down, q, xu)? {
				st.push(vec![g.act(0, HC_RANK, yd, xu, 0, 0, q)?]);
			}
			st.push(vec![up]);
			st.push(vec![g.vec(5, W, yu, xa, xn, HC, q, 0, 0, ym, None, None)?]);
			Ok(st)
		};
		// run 1
		let mut st0 = vec![add(g, "fce", W, W, MXE, MYE)?];
		{
			let (c, s) = mtp_mat(d, "fch", W, W)?;
			for i in 0..HC {
				st0.push(g.add(Mat::I8(&c, &s), W, W, MXH + i * W, MYH + i * W)?);
			}
		}
		let mut pre = vec![st0];
		pre.push(vec![
			g.vec(0, HCW, MYH, MXS, 0, 1, 0.0, 0, 1, None, None, None)?
		]);
		let hca_norm = mtp_vec(d, "hca.norm", HCW)?;
		pre.push(vec![g.vec(
			1,
			W,
			MXS,
			MXN,
			MXONE,
			HC,
			1e-6,
			0,
			8,
			Some(MYX),
			Some(MYE),
			Some(&hca_norm),
		)?]);
		pre.extend(hc(g, "hca", MXN, MXU, MXA, MYD, MYJ, MYU, None)?);
		pre.push(vec![
			add(g, "q", 2 * HQ * DA, W, MXA, MYQ)?,
			add(g, "k", HKV * DA, W, MXA, MYK)?,
			add(g, "v", HKV * DA, W, MXA, MYV)?,
		]);
		// run 2
		let mut post = vec![vec![add(g, "o", W, HQ * DA, NXATT, NYO)?]];
		let hcm_norm = mtp_vec(d, "hcm.norm", HCW)?;
		post.push(vec![g.vec(
			1,
			W,
			NXS,
			NXN,
			NXINJ,
			HC,
			1e-6,
			0,
			8,
			Some(NYX),
			Some(NYO),
			Some(&hcm_norm),
		)?]);
		post.extend(hc(g, "hcm", NXN, NXU, NXM, NYD, NYJ, NYU, Some(NYM))?);
		post.push(vec![
			add(g, "router", NE, W, NXM, NYR)?,
			add(g, "sg", FE, W, NXM, NYG)?,
			add(g, "su", FE, W, NXM, NYG + FE)?,
		]);
		post.push(vec![
			g.vec(3, FE, NYG, NXSW, 0, 1, 0.0, 0, 0, None, None, None)?
		]);
		post.push(vec![add(g, "sd", W, FE, NXSW, NYSD)?]);
		self.mtp = Some(MtpG { pre, post });
		eprintln!(
			"flash: MTP head's dense part on the GPU in {:.1} s",
			t0.elapsed().as_secs_f64()
		);
		Ok(())
	}

	/// True when [`Self::add_mtp`] uploaded the head.
	pub fn has_mtp(&self) -> bool {
		self.mtp.is_some()
	}

	/// Run 1 of [`MtpG`] for rows of normed embeddings `en[p]` `[W]` and
	/// normed streams `hn[p]` `[HCW]`.
	pub fn mtp_pre(&mut self, en: &[Vec<f32>], hn: &[Vec<f32>]) -> Result<Vec<MtpPre>, String> {
		let st = self
			.mtp
			.as_ref()
			.ok_or("flash: no MTP head on the GPU")?
			.pre
			.clone();
		let sr: Vec<&[usize]> = st.iter().map(|v| v.as_slice()).collect();
		let mut out = Vec::with_capacity(en.len());
		let step = self.g.max_np().min(WAVE_MAX);
		for c0 in (0..en.len()).step_by(step) {
			let c1 = (c0 + step).min(en.len());
			let xs: Vec<Vec<f32>> = (c0..c1)
				.map(|p| {
					let mut x = vec![0f32; MXONE + HC];
					x[MXE..MXE + W].copy_from_slice(&en[p]);
					x[MXH..MXH + HCW].copy_from_slice(&hn[p]);
					x[MXONE..MXONE + HC].fill(1.0);
					x
				})
				.collect();
			let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
			self.g.io_next(
				&[MXE..MXE + W, MXH..MXH + HCW, MXONE..MXONE + HC],
				&[MYX..MYX + HCW, MYJ..MYJ + HC, MYQ..MYV + HKV * DA],
			);
			let t = std::time::Instant::now();
			self.g.run_chain_n(&xr, &sr)?;
			self.tick(t, 4);
			for p in 0..c1 - c0 {
				let y = self.g.y_at(p);
				let mut inj = [0f32; HC];
				for (c, o) in inj.iter_mut().enumerate() {
					*o = 2.0 * sigmoid(y[MYJ + c] / HC as f32);
				}
				out.push(MtpPre {
					x: y[MYX..MYX + HCW].to_vec(),
					q: y[MYQ..MYK].to_vec(),
					k: y[MYK..MYV].to_vec(),
					v: y[MYV..MYV + HKV * DA].to_vec(),
					inj,
				});
			}
		}
		Ok(out)
	}

	/// Run 2 of [`MtpG`] on the attention's gated output `att` `[HQ DA]`,
	/// the streams `x` before it and run 1's injection weights `inj`.
	pub fn mtp_post(&mut self, att: &[f32], x: &[f32], inj: &[f32; HC]) -> Result<MtpPost, String> {
		let st = self
			.mtp
			.as_ref()
			.ok_or("flash: no MTP head on the GPU")?
			.post
			.clone();
		let sr: Vec<&[usize]> = st.iter().map(|v| v.as_slice()).collect();
		let mut xin = vec![0f32; NXINJ + HC];
		xin[NXATT..NXATT + HQ * DA].copy_from_slice(att);
		xin[NXS..NXS + HCW].copy_from_slice(x);
		xin[NXINJ..NXINJ + HC].copy_from_slice(inj);
		self.g.io_next(
			&[NXATT..NXATT + HQ * DA, NXS..NXS + HCW, NXINJ..NXINJ + HC],
			&[
				NYX..NYX + HCW,
				NYJ..NYJ + HC,
				NYM..NYM + W,
				NYR..NYR + NE,
				NYSD..NYSD + W,
			],
		);
		let t = std::time::Instant::now();
		self.g.run_chain_n(&[&xin], &sr)?;
		self.tick(t, 4);
		let y = self.g.y_at(0);
		let mut inj = [0f32; HC];
		for (c, o) in inj.iter_mut().enumerate() {
			*o = 2.0 * sigmoid(y[NYJ + c] / HC as f32);
		}
		Ok(MtpPost {
			x: y[NYX..NYX + HCW].to_vec(),
			m: y[NYM..NYM + W].to_vec(),
			logits: y[NYR..NYR + NE].to_vec(),
			shared: y[NYSD..NYSD + W].to_vec(),
			inj,
		})
	}
}
