//! The NPU-only model as a forward-pass engine: prefill through the lean
//! per-layer graphs (`build_all.sh` / `mkset.sh` sets of 256 or 64 positions
//! and some attention window; each chunk takes the cheapest set whose window
//! reaches its start), decode one token at a time through
//! the s = 1 decode set (`drec.py` + `attn_lean.py --vt`: delta state stored
//! transposed, attention K / V as a ring), or speculatively through an s = K
//! verify set (`wyrec.py` deferred rollback) with the MTP drafter
//! (`verify` / `commit` / `drafts`): `npubench pf`, `gen` and `spec` behind
//! one interface.
//!
//! `forward(ids)` prefills `ids[..n - 1]` (from the current state; through
//! decode steps when that is cheaper) and then decodes `ids[n - 1]`, whose
//! logits it returns.
//! The prefill's state is converted to the decode layout on the host:
//! delta ssm `S x vmul` -> `S^T` (vmul 16 in layer 0, 256 elsewhere: the
//! export's `QWEN_OV_DSCALE` convention the prefill graphs keep), the last
//! three conv rows -> columns 1..3 of the 4-column window, attention K rows
//! -> ring slots, V rows -> columns of the transposed value window.
//!
//! A set may hold **cut** delta layers (`build_delta.py --cut pre|post`,
//! `lean{L}a` / `lean{L}b`): the layer runs as two NPU jobs with its
//! recurrence in between on another device ([`Scan`], the iGPU's in `cli`),
//! which keeps the recurrent state (f32) between chunks. The host waits for
//! the pre job, runs the scan on the pre job's output buffers, then submits
//! the post job.

use crate::ReadAt;
use std::path::Path;

use crate::ir::{f16_bits, f16_to_f32};
use crate::lock::NpuLock;
use crate::ze::{Batch, Graph, HostBuf, Npu};

/// Where the graph sets live (a `.npu` file: `key = path` lines; a relative
/// path that exists beside the file is taken from there, so a build's
/// directory can be moved as a whole).
#[derive(Clone, Debug, Default)]
pub struct Config {
	/// Prefill set, 256 positions (lean{L}.blob, .use, per-layer .in files).
	pub prefill: String,
	/// Optional 64-position prefill set (the same weights).
	pub small: Option<String>,
	/// Optional 4096-window prefill set, for chunks past the main window.
	pub long: Option<String>,
	/// Optional 64-position 4096-window set: short tails past the main
	/// window (a continued conversation's new turn).
	pub long_small: Option<String>,
	/// Optional decode form of the far attention (`DEC=1 mkfar.sh`, one
	/// position): decode steps past the decode set's window.
	pub fardec: Option<String>,
	/// Further prefill sets (`set = DIR`, repeatable): any positions and
	/// window; each chunk takes the cheapest set whose window reaches it.
	pub sets: Vec<String>,
	/// Decode set (layer{L}.blob, head.blob, outs.txt): s = 1 (`drec.py` +
	/// `attn_lean.py --vt`), or the s = K verify set of speculative decoding
	/// (`wyrec.py` delta layers + lean attention), which `mtp` drafts for.
	pub decode: String,
	/// MTP drafter for a verify set (`cut_mtp.py` / `heads.py`: head.blob
	/// with the `hn` output, which replaces the verify set's head,
	/// mtp_s{K}.blob, mtp_s1.blob, mtphead.blob).
	pub mtp: Option<String>,
	/// Embedding table, [vocab, 5120] f16.
	pub emb: String,
	/// Directory with tokenizer.json and tokenizer_config.json (the
	/// checkpoint's, or an export's).
	pub tokenizer: String,
	/// Vulkan adapter for the recurrence of cut delta layers (`scan =
	/// intel`: the iGPU); sets with cut layers need it.
	pub scan: Option<String>,
	/// Vulkan adapter (name part) that takes a share of a cut decode set's MLPs.
	pub mlpgpu: Option<String>,
	/// The vision tower's set (`vis.cfg`, `pos.f32`, `vis{i}.blob`:
	/// [`crate::vis`]); with it the engine takes images.
	pub vision: Option<String>,
}

/// An image in the tokens of [`NpuEngine::feed_mm`]: its run of placeholder
/// tokens and the tower's embeddings that take their place.
pub struct ImageSpan<'a> {
	/// Index of the run's first token.
	pub start: usize,
	/// The merged grid: the run is `grid_w * grid_h` tokens, row by row.
	pub grid_w: usize,
	pub grid_h: usize,
	/// `[grid_w * grid_h][width]` (the text model's: 5120, Flash-Next 2560).
	pub emb: &'a [f32],
}

/// What a prefill with images needs besides the token ids (set for the
/// duration of [`NpuEngine::feed_mm`]'s prefill).
struct Mm {
	/// position of `pos[0]`
	t0: usize,
	/// MRoPE coordinates (t, h, w) per position
	pos: Vec<[i64; 3]>,
	/// (first position, f16 rows) per image
	emb: Vec<(usize, Vec<u8>)>,
	/// first position after the last image
	text_from: usize,
}

/// The rotary angle's coordinate of frequency `i` (0..32): this model's
/// MRoPE is the interleaved one, t, h, w in turn (`model::math::rope_apply`;
/// sections 11 / 11 / 10 cover all 32). For text the three are equal.
fn mrope_axis(i: usize) -> usize {
	i % 3
}

/// The MRoPE coordinates (t, h, w) of `n` tokens whose first has the text
/// position `first`, with images `(start, grid_w, grid_h)` (runs of
/// `grid_w * grid_h` tokens, row by row), and the text position that
/// follows them. Text advances all three together; an image's tokens share
/// one `t`, `h` / `w` are that plus the token's row / column, and the text
/// after it resumes one past the largest coordinate it used
/// (`model::prompt::build_image_prompt`).
pub(crate) fn mm_positions(
	first: i64,
	n: usize,
	spans: &[(usize, usize, usize)],
) -> Result<(Vec<[i64; 3]>, i64), String> {
	let mut pos = Vec::with_capacity(n);
	let mut next = first;
	for &(start, gw, gh) in spans {
		if start < pos.len() || gw * gh == 0 || start + gw * gh > n {
			return Err(format!(
				"an image of {gw} x {gh} tokens at {start} of {n} tokens"
			));
		}
		while pos.len() < start {
			pos.push([next; 3]);
			next += 1;
		}
		pos.extend((0..gw * gh).map(|i| [next, next + (i / gw) as i64, next + (i % gw) as i64]));
		next += gw.max(gh) as i64;
	}
	while pos.len() < n {
		pos.push([next; 3]);
		next += 1;
	}
	Ok((pos, next))
}

/// The recurrence of a cut delta layer, run outside the NPU. `di` numbers
/// the delta layers (0..48); states are `[48][128][128]` f32, row = key dim,
/// true scale.
pub trait Scan {
	/// Delta layer `di`'s state: `None` = zero.
	fn set_state(&mut self, di: usize, st: Option<&[f32]>) -> Result<(), String>;
	fn state(&mut self, di: usize) -> Result<Vec<f32>, String>;
	/// Layers `0..n`'s states (`len` values each) at once: `fill(di, dst)`
	/// writes layer `di`'s into `dst`, or returns false for a zero state. A
	/// device implements it with one transfer (the default: per layer).
	fn set_states(
		&mut self,
		n: usize,
		len: usize,
		fill: &mut dyn FnMut(usize, &mut [f32]) -> bool,
	) -> Result<(), String> {
		let mut v = vec![0f32; len];
		for di in 0..n {
			let st = fill(di, &mut v).then_some(&v[..]);
			self.set_state(di, st)?;
		}
		Ok(())
	}
	/// Layers `0..n`'s states at once, each handed to `take(di, state)`.
	fn get_states(&mut self, n: usize, take: &mut dyn FnMut(usize, &[f32])) -> Result<(), String> {
		for di in 0..n {
			take(di, &self.state(di)?);
		}
		Ok(())
	}
	/// Positions `0..n` of the pre job's `qn`, `kn` [n][2048], `v` [n][6144],
	/// `gb` [n][96] (f16) -> `o` [n][6144] f16 x `oscale`. The buffers are
	/// the engine's (fixed addresses for its lifetime).
	#[allow(clippy::too_many_arguments)]
	fn scan(
		&mut self,
		di: usize,
		n: usize,
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		o: &mut [u8],
		oscale: f32,
	) -> Result<(), String>;
	/// [`Self::scan`] started without waiting for it: `o` holds the result
	/// once [`Self::scan_end`] returned. A device that cannot run it apart
	/// keeps the default (the whole scan here).
	#[allow(clippy::too_many_arguments)]
	fn scan_begin(
		&mut self,
		di: usize,
		n: usize,
		qn: &[u8],
		kn: &[u8],
		v: &[u8],
		gb: &[u8],
		o: &mut [u8],
		oscale: f32,
	) -> Result<(), String> {
		self.scan(di, n, qn, kn, v, gb, o, oscale)
	}
	/// Wait for [`Self::scan_begin`]'s scan.
	fn scan_end(&mut self) -> Result<(), String> {
		Ok(())
	}
	/// Delta layer `di`'s gated-norm factors for [`Self::scan_begin_gn`]:
	/// `w` [2][6144] = per channel the norm weight times the out-projection's
	/// input prescale, then `z`'s row scale. `Ok(false)`: this device has
	/// no gated norm.
	fn set_gnorm(&mut self, _di: usize, _w: &[f32]) -> Result<bool, String> {
		Ok(false)
	}
	/// [`Self::scan_begin`], then in place of `o` per (position, head)
	/// `o / sqrt(mean(o^2) + 1e-6 oscale^2) * w0 * sigmoid(z * w1)`
	/// rotated in Hadamard blocks of 256 (Flash-Next's `post_delta_t`
	/// input). `z` [n][6144] f16 is the pre graph's gate input.
	#[allow(clippy::too_many_arguments)]
	fn scan_begin_gn(
		&mut self,
		_di: usize,
		_n: usize,
		_qn: &[u8],
		_kn: &[u8],
		_v: &[u8],
		_gb: &[u8],
		_z: &[u8],
		_o: &mut [u8],
		_oscale: f32,
	) -> Result<(), String> {
		Err("this scan device has no gated norm".into())
	}
	/// Delta layer `di`'s short conv for [`Self::conv_next`]: `w` = the
	/// taps `[10240][4]`, then `qkv`'s row scale `[10240]` (the conv's
	/// input is `qkv * qs`). `Ok(false)`: this device has no conv.
	fn set_conv(&mut self, _di: usize, _w: &[f32]) -> Result<bool, String> {
		Ok(false)
	}
	/// Run delta layer `di`'s short conv, Swish and q / k l2 norms of `n`
	/// positions first in the next scan submission (over the same `n`):
	/// from `cpast` [3][10240] (scaled) and the raw `qkv` [n][10240] (f16,
	/// Flash-Next's `delta_pre_c`) into `qn`, `kn`, `v`, the buffers that
	/// scan reads.
	#[allow(clippy::too_many_arguments)]
	fn conv_next(
		&mut self,
		_di: usize,
		_n: usize,
		_cpast: &[u8],
		_qkv: &[u8],
		_qn: &mut [u8],
		_kn: &mut [u8],
		_v: &mut [u8],
	) -> Result<(), String> {
		Err("this scan device has no conv".into())
	}
	/// Called every few tens of ms while the engine does not need the scan
	/// device for long (far attention): an idle iGPU drops to a power state
	/// the next scans then run 10x slower from.
	fn poke(&mut self) -> Result<(), String> {
		Ok(())
	}
	/// Far attention on the scan device, next to the NPU's far jobs: blocks
	/// `(k, v)` (`[cap][4][256]` f16, keys `[0, nk)`) for the pre graph's
	/// `qg` over `s` positions, submitted without waiting; `fresh` starts a
	/// new result, else the blocks join the running one. `Ok(false)`: not
	/// available (nothing was started).
	fn far_begin(
		&mut self,
		_s: usize,
		_nk: usize,
		_qg: &[u8],
		_blocks: &[(&[u8], &[u8])],
		_fresh: bool,
	) -> Result<bool, String> {
		Ok(false)
	}
	/// True once no `far_begin` submission is in flight; never blocks.
	fn far_poll(&mut self) -> Result<bool, String> {
		Ok(true)
	}
	fn far_wait(&mut self) -> Result<(), String> {
		Ok(())
	}
	/// The running result: `o` `[24 s][256]`, the sum of exp(score - m) v,
	/// and `[m, l]` per row (max score, sum of exp(score - m)).
	fn far_out(&self) -> (&[f32], &[[f32; 2]]) {
		(&[], &[])
	}
}

impl Config {
	pub fn parse(path: &str) -> Result<Config, String> {
		let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
		let mut c = Config::default();
		for line in t.lines() {
			let line = line.trim();
			if line.is_empty() || line.starts_with('#') {
				continue;
			}
			let (k, v) = line
				.split_once('=')
				.ok_or(format!("{path}: bad line {line}"))?;
			let mut v = v.trim().to_string();
			if !matches!(k.trim(), "scan" | "mlpgpu") && Path::new(&v).is_relative() {
				if let Some(beside) = Path::new(path)
					.parent()
					.map(|d| d.join(&v))
					.filter(|p| p.exists())
				{
					v = beside.to_string_lossy().into_owned();
				}
			}
			match k.trim() {
				"prefill" => c.prefill = v,
				"small" => c.small = Some(v),
				"long" => c.long = Some(v),
				"long_small" => c.long_small = Some(v),
				"fardec" => c.fardec = Some(v),
				"set" => c.sets.push(v),
				"decode" => c.decode = v,
				"mtp" => c.mtp = Some(v),
				"emb" => c.emb = v,
				"tokenizer" => c.tokenizer = v,
				"scan" => c.scan = Some(v),
				"mlpgpu" => c.mlpgpu = Some(v),
				"vision" => c.vision = Some(v),
				k => return Err(format!("{path}: unknown key {k}")),
			}
		}
		if c.prefill.is_empty() || c.decode.is_empty() || c.emb.is_empty() {
			return Err(format!("{path}: needs prefill, decode and emb"));
		}
		Ok(c)
	}
}

const W: usize = 5120;
const NL: usize = 64;
/// Rows of one attention position (4 KV heads x 256, f16).
const ROWB: usize = 4 * 256 * 2;

struct Dl {
	args: [usize; 4],
	cb: [usize; 2],
	sb: [usize; 2],
}

/// One prefill graph set (positions `s`, attention window `cap`).
///
/// Everything a chunk binds exists twice (slots 0 and 1: its embedding
/// input, the residual rows between layers, attention K / V outputs, scan
/// buffers, mask / cos / sin / valid / keep), so two chunks can be in flight
/// layer by layer (`NpuEngine::prefill`); weights and the delta layers'
/// state buffers are shared (slot 1 runs at the other parity).
struct PSet {
	s: usize,
	cap: usize,
	/// NPU jobs in order, per slot: one per layer, two per cut delta layer
	jobs: [Vec<Job>; 2],
	dls: Vec<Option<Dl>>,
	/// cut delta layers' scan buffers qn, kn, v, gb, o (z goes pre -> post)
	scan_bufs: [Option<[usize; 5]>; 2],
	atts: [Vec<Option<[usize; 4]>>; 2],
	shared: [BufMap; 2],
	lists: Vec<Vec<Batch<'static>>>,
	busy: Vec<Vec<bool>>,
	/// (slot, job, parity) a list is bound to
	cur: Vec<Vec<(usize, usize, usize)>>,
	/// next ring index per blob
	next: Vec<usize>,
}

/// One NPU job of a prefill set: layer, blob, its argument buffers, and
/// whether it is the pre half of a cut layer (the scan follows it).
struct Job {
	l: usize,
	gi: usize,
	set: Vec<usize>,
	pre: bool,
	post: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
	Full,
	Pre,
	Post,
	/// an attention layer of a far set: `lean{L}a` / `lean{L}b` (`Far`)
	APre,
	APost,
}

type BufMap = std::collections::HashMap<String, usize>;

/// What every prefill set's binding shares: residual rows between layers,
/// the per-layer state buffers, the weights.
struct Bind<'b> {
	npu: &'static Npu,
	bufs: &'b mut Vec<HostBuf<'static>>,
	weights: std::collections::HashMap<(usize, String), usize>,
	st_in: std::collections::HashMap<(usize, String), usize>,
	x0: [usize; 2],
	presid: [Vec<usize>; 2],
	slot: usize,
	smax: usize,
	maxcap: usize,
}

impl Bind<'_> {
	fn state_buf(&mut self, key: (usize, String), bytes: usize) -> Result<usize, String> {
		match self.st_in.get(&key) {
			Some(&b) => Ok(b),
			None => {
				let b = alloc(self.npu, self.bufs, bytes, None)?;
				self.st_in.insert(key, b);
				Ok(b)
			}
		}
	}

	fn shared(&mut self, shared: &mut BufMap, name: &str, bytes: usize) -> Result<usize, String> {
		match shared.get(name) {
			Some(&b) => Ok(b),
			None => {
				let b = alloc(self.npu, self.bufs, bytes, None)?;
				shared.insert(name.to_string(), b);
				Ok(b)
			}
		}
	}

	/// The buffers of one job's arguments (layer `l`, set `dir` of `s`
	/// positions), filling the layer's state slots in `dl` / `at`.
	#[allow(clippy::too_many_arguments)]
	fn job(
		&mut self,
		args: &[crate::ze::Arg],
		l: usize,
		role: Role,
		dir: &str,
		s: usize,
		shared: &mut BufMap,
		dl: &mut Dl,
		at: &mut [usize; 4],
	) -> Result<Vec<usize>, String> {
		let mut set = Vec::new();
		let mut outs_k = 0;
		for (i, arg) in args.iter().enumerate() {
			let bi = if arg.is_output {
				if role == Role::APre {
					// qg, gate, knew, vnew (`build_attn.py --part pre`)
					let k = outs_k;
					outs_k += 1;
					match k {
						0 => self.shared(shared, "att.qg", arg.bytes)?,
						1 => self.shared(shared, "att.gate", arg.bytes)?,
						_ => {
							let b = self.state_buf(
								(l, format!("out{}.{}", k - 2, self.slot)),
								arg.bytes / s * self.smax,
							)?;
							at[k] = b;
							b
						}
					}
				} else if role == Role::Pre {
					// qn, kn, v, gb, z, conv_out (`build_delta.py --cut pre`)
					const NAMES: [&str; 5] = ["scan.qn", "scan.kn", "scan.v", "scan.gb", "scan.z"];
					let k = outs_k;
					outs_k += 1;
					if k < 5 {
						self.shared(shared, NAMES[k], arg.bytes)?
					} else {
						let b = self.state_buf((l, "out0".into()), arg.bytes)?;
						dl.args[2] = i;
						dl.cb[1] = b;
						b
					}
				} else if arg.bytes == s * W * 2 {
					self.presid[self.slot][l]
				} else {
					let sz = if l % 4 == 3 {
						arg.bytes / s * self.smax
					} else {
						arg.bytes
					};
					// attention K / V rows are per slot, delta states shared
					let key = if l % 4 == 3 {
						format!("out{outs_k}.{}", self.slot)
					} else {
						format!("out{outs_k}")
					};
					let b = self.state_buf((l, key), sz)?;
					if l % 4 == 3 {
						at[2 + outs_k] = b;
					} else if outs_k == 0 {
						dl.args[2] = i;
						dl.cb[1] = b;
					} else {
						dl.args[3] = i;
						dl.sb[1] = b;
					}
					outs_k += 1;
					b
				}
			} else {
				let nm = arg.name.clone();
				match nm.as_str() {
					"x" => {
						if l == 0 {
							self.x0[self.slot]
						} else {
							self.presid[self.slot][l - 1]
						}
					}
					"o" if role == Role::APost => self.shared(shared, "att.o", arg.bytes)?,
					"gate" if role == Role::APost => self.shared(shared, "att.gate", arg.bytes)?,
					"o" if role == Role::Post => self.shared(shared, "scan.o", arg.bytes)?,
					"z" if role == Role::Post => self.shared(shared, "scan.z", arg.bytes)?,
					"conv" | "ssm" | "kpast" | "vpast" => {
						let sz = if nm == "kpast" || nm == "vpast" {
							arg.bytes / arg.dims[1] as usize * self.maxcap
						} else {
							arg.bytes
						};
						let b = self.state_buf((l, nm.clone()), sz)?;
						match nm.as_str() {
							"conv" => {
								dl.args[0] = i;
								dl.cb[0] = b;
							}
							"ssm" => {
								// a cut layer's pre graph keeps the (unused)
								// input: the buffer, but no parity flips
								if role != Role::Pre {
									dl.args[1] = i;
								}
								dl.sb[0] = b;
							}
							"kpast" => at[0] = b,
							_ => at[1] = b,
						}
						b
					}
					_ => {
						let own = Path::new(dir).join(format!("lean{l}.{nm}.in"));
						if own.exists() {
							match self.weights.get(&(l, nm.clone())) {
								Some(&b) => b,
								None => {
									let (b, _) = read_pinned(self.npu, self.bufs, &own, arg.bytes)?;
									self.weights.insert((l, nm.clone()), b);
									b
								}
							}
						} else {
							self.shared(shared, &nm, arg.bytes)?
						}
					}
				}
			};
			set.push(bi);
		}
		if role == Role::Pre {
			// the state the hand-over reads at parity 1 (written from the scan)
			let n = self.bufs[dl.sb[0]].len();
			dl.sb[1] = self.state_buf((l, "out1".into()), n)?;
		}
		Ok(set)
	}
}

/// One decode delta layer: append index, its state args and buffers
/// (ping-pong by step parity). In a verify set (`wy`) `ssm` is `sbase` and
/// `conv` one window buffer (both entries), `conv_out` the `conv_full` arg.
struct DDelta {
	g: usize,
	conv_in: usize,
	ssm_in: usize,
	conv_out: usize,
	ssm_out: usize,
	conv: [usize; 2],
	ssm: [usize; 2],
	wy: Option<Wy>,
}

/// The deferred-rollback factors of a verify-set delta layer (`wyrec.py`):
/// the next step forms its start state `alpha * sbase + (kp * w)^T dp`.
#[derive(Clone, Copy)]
struct Wy {
	k_in: usize,
	k_out: usize,
	d_in: usize,
	d_out: usize,
	kb: [usize; 2],
	db: [usize; 2],
	w: usize,
	alpha: usize,
	gcum: usize,
	conv_full: usize,
}

/// The MTP drafter (`npubench spec --mtp`): a K-position catch-up over the
/// positions a verify step accepted (its last row gives the first draft),
/// chained 1-position steps for the rest, a reduced-vocabulary draft head.
struct Mtp {
	/// m4 (catch-up), m1 (chained draft), mh (draft head)
	lists: Vec<Batch<'static>>,
	mb: std::collections::HashMap<String, usize>,
	m1_past: (usize, usize),
	cap: usize,
	vocab: usize,
	m4_f32: bool,
	m1_f32: bool,
	/// last position the MTP window holds, rows of it in use
	last: usize,
	count: usize,
	/// the main model's final norm (the `hn` output's), for prefill rows
	norm: Vec<f32>,
	/// (position, normed hidden) a prefill left for the next catch-up
	saved: Option<(usize, Vec<u8>)>,
}

/// Builds the GPU share of a cut decode set's MLPs (`mlpgpu = ADAPTER`).
pub type MlpGpuNew =
	Box<dyn Fn(&[crate::mlp::Mlp]) -> Result<Box<dyn crate::mlp::MlpGpu>, String> + Send>;

pub struct NpuEngine {
	npu: &'static Npu,
	_lock: NpuLock,
	bufs: Vec<HostBuf<'static>>,
	table: std::fs::File,
	// prefill
	psets: Vec<PSet>,
	x0: [usize; 2],
	maxcap: usize,
	// decode
	batch: Batch<'static>,
	/// A cut decode set: a list per layer (`batch` is then the head alone),
	/// the layers' CPU MLPs and their residual buffers.
	parts: Vec<Batch<'static>>,
	mlp: Vec<crate::mlp::Mlp>,
	dresid: Vec<usize>,
	/// The MLPs' GPU share, what builds it (again, after a failure) and
	/// when to try next.
	mlp_gpu: Option<Box<dyn crate::mlp::MlpGpu>>,
	mlp_gpu_new: Option<MlpGpuNew>,
	mlp_gpu_retry: Option<std::time::Instant>,
	mlp_gpu_slow: usize,
	/// Far passes left that keep the MLPs on the cores (see `run_far`).
	mlp_gpu_far_off: usize,
	mlp_gpu_far_said: bool,
	/// Least seconds a pass waited for the GPU share, by rows (the watch's baseline).
	mlp_gpu_base: [f64; 5],
	emb: usize,
	pos_i: usize,
	bias_i: usize,
	logits_i: usize,
	vocab: usize,
	cap: usize,
	deltas: Vec<DDelta>,
	/// (present, past, is value) per attention window, and its layer
	kv: Vec<(usize, usize, bool, usize)>,
	parity: usize,
	/// positions fed since the last reset
	t: usize,
	/// positions per decode run (1, or K of a verify set)
	k: usize,
	wy: bool,
	/// the head's output-normed hidden (verify set with MTP), f32 or f16
	hn: Option<(usize, bool)>,
	mtp: Option<Mtp>,
	/// ids of a verify step not yet committed
	pending: Option<Vec<u32>>,
	/// the last committed verify step (start position, accepted ids), whose
	/// `hn` rows the MTP catch-up has not consumed yet
	last: Option<(usize, Vec<u32>)>,
	/// MTP catch-up row that gives the next first draft
	mrow: Option<usize>,
	/// the prefill's last-layer output (final residual rows of a chunk)
	presid_last: [usize; 2],
	/// slot of the last prefill chunk (whose rows `presid_last` holds)
	last_slot: usize,
	/// recurrence of cut delta layers, and whether it holds the current
	/// states (else the prefill buffers do)
	scan: Option<Box<dyn Scan>>,
	scan_live: bool,
	far: Option<Far>,
	/// Sequence position less MRoPE position of the text after the images
	/// fed so far (an image of `w x h` tokens advances the one by `w * h`,
	/// the other by `max(w, h)`); 0 without images.
	mshift: usize,
	mm: Option<Mm>,
	vis: Option<crate::vis::NpuVision>,
	vis_attn: Option<Box<dyn crate::vis::VisAttn>>,
}

/// Attention past every compiled window (a set built by `mkfar.sh`): the
/// K / V rows of all positions live in host blocks of `cap - 2` keys per
/// attention layer. A chunk's attention layer runs as its `pre` graph, the
/// `far` graph once per full block (no weights, one blob for every layer;
/// two in flight), the `near` graph over the chunk's own keys and the
/// partly filled block, a host merge of the blocks' outputs by their log
/// sum exp (`softmax_lse.c` leaves max and sum in the two spare key slots),
/// and the `post` graph.
struct Far {
	/// rows of a block buffer (`cap - 2` keys)
	cap: usize,
	/// per attention layer, per block: K and V buffer
	blocks: Vec<Vec<[usize; 2]>>,
	/// the blocks hold every position fed (else the window buffers do)
	live: bool,
	/// the far sets (one per chunk size), all on the same blocks
	sets: Vec<FarSet>,
	dec: Option<FarDec>,
	t_att: f64,
	/// far blocks on the scan device: its ms per block at 256 positions
	/// (running estimate, 0 = none yet), the blocks it took since the last
	/// timing line, the NPU's ms per block (as the first); then the CPU
	/// cores' ms per block and blocks (as the scan device's)
	gpu: [f64; 5],
}

/// `QWEN_NPU_FAR_GPU=0`: no far blocks on the scan device (the A/B);
/// `QWEN_NPU_FAR_GPU_LEAD=X`: blocks' worth of NPU time (the near job, the
/// merges) the device may finish after the NPU's last far block (default 2).
/// Threads of the CPU MLPs of a cut decode set: `QWEN_NPU_MLP_THREADS`,
/// default 12 (DRAM-bound: 8 / 12 / 16 / 24 threads = 160-165 / 157 / 156 /
/// 156 ms of MLP per token; 4 verify rows 171 / - / 163).
fn mlp_threads() -> usize {
	static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*N.get_or_init(|| {
		std::env::var("QWEN_NPU_MLP_THREADS")
			.ok()
			.and_then(|v| v.parse().ok())
			.filter(|&n| n > 0)
			.unwrap_or(12)
	})
}

/// `QWEN_GPU_RECOVER_SECS`: seconds between attempts to build the MLPs' GPU
/// share again (default 15, 0 = never).
fn mlp_gpu_secs() -> u64 {
	std::env::var("QWEN_GPU_RECOVER_SECS")
		.ok()
		.and_then(|v| v.parse().ok())
		.unwrap_or(15)
}

/// `QWEN_NPU_FAR_CPU=1`: far blocks on the CPU cores too (off by default:
/// prefill stays on the NPU and the scan device, the cores free for other
/// applications; needs AVX2, FMA and F16C). Prefill chunks only (64+
/// positions): a decode step's blocks are bound by memory, not arithmetic.
fn far_cpu_on() -> bool {
	static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		std::env::var("QWEN_NPU_FAR_CPU").is_ok_and(|v| v == "1") && crate::farcpu::supported()
	})
}

fn far_gpu_cfg() -> (bool, f64) {
	static C: std::sync::OnceLock<(bool, f64)> = std::sync::OnceLock::new();
	*C.get_or_init(|| {
		(
			std::env::var("QWEN_NPU_FAR_GPU")
				.map(|v| v != "0")
				.unwrap_or(true),
			std::env::var("QWEN_NPU_FAR_GPU_LEAD")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(2.0),
		)
	})
}

/// Decode steps on the far blocks (`fardec = DIR`): the decode set's delta
/// layers three at a time (`seg`), each attention layer as the `pre` graph
/// (the decode set's positions: 1, or the K of a verify set), the blocks,
/// and `post`, then the head.
struct FarDec {
	fs: FarSet,
	pre: Vec<Batch<'static>>,
	post: Vec<Batch<'static>>,
	seg: Vec<Batch<'static>>,
	head: Batch<'static>,
	/// `farm.blob`: several full blocks per far job
	multi: Option<FarMulti>,
	/// pre outputs: qg, knew, vnew; the merged `o`; RoPE tables
	qg: usize,
	kv: [usize; 2],
	o: usize,
	cos: usize,
	sin: usize,
}

/// A far graph over `n` blocks (`build_attn.py --part far --nblk N`, the
/// fardec dir's `farm.blob`): a decode step's far blocks are bound by the
/// memory they read (8 MB of K / V each, ~0.27 ms), and a job per block
/// adds ~0.08 ms to each.
struct FarMulti {
	list: Vec<Batch<'static>>,
	n: usize,
	/// argument positions: qg, the K blocks, the V blocks, o, lse
	qg: usize,
	k: Vec<usize>,
	v: Vec<usize>,
	oi: usize,
	li: usize,
	ob: [usize; 2],
	lb: [usize; 2],
}

impl FarMulti {
	fn load(
		npu: &'static Npu,
		bufs: &mut Vec<HostBuf<'static>>,
		file: &str,
	) -> Result<FarMulti, String> {
		let g = &mut load_blobs(npu, &[file.to_string()])?[0];
		let pos = |n: &str| g.args.iter().position(|a| a.name == n && !a.is_output);
		let qg = pos("qg").ok_or(format!("{file}: no qg"))?;
		let k: Vec<usize> = (0..).map_while(|i| pos(&format!("kblk{i}"))).collect();
		let v: Vec<usize> = (0..).map_while(|i| pos(&format!("vblk{i}"))).collect();
		let mut o: Vec<usize> = (0..g.args.len()).filter(|&i| g.args[i].is_output).collect();
		o.sort_by_key(|&i| std::cmp::Reverse(g.args[i].bytes));
		if k.len() < 2 || v.len() != k.len() || o.len() != 2 {
			return Err(format!("{file}: unexpected arguments"));
		}
		let (oi, li) = (o[0], o[1]);
		let ab: Vec<usize> = g
			.args
			.iter()
			.map(|a| alloc(npu, bufs, a.bytes, None))
			.collect::<Result<_, _>>()?;
		let (o2, l2) = (
			alloc(npu, bufs, g.args[oi].bytes, None)?,
			alloc(npu, bufs, g.args[li].bytes, None)?,
		);
		let mut list = Vec::new();
		for _ in 0..2 {
			let refs: Vec<&HostBuf<'_>> = ab.iter().map(|&i| &bufs[i]).collect();
			list.push(npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, refs)])?);
		}
		Ok(FarMulti {
			list,
			n: k.len(),
			qg,
			k,
			v,
			oi,
			li,
			ob: [ab[oi], o2],
			lb: [ab[li], l2],
		})
	}

	/// The first full blocks of `blocks[..nfull]` in jobs of `n` (a last
	/// job of at least `n - 1`, its spare inputs any block, ignored),
	/// merged into `acc` / `lacc`; the blocks done.
	#[allow(clippy::too_many_arguments)]
	fn run(
		&mut self,
		blocks: &[[usize; 2]],
		nfull: usize,
		bufs: &[HostBuf<'static>],
		qg: usize,
		acc: &mut [f32],
		lacc: &mut [f32],
		unnorm: bool,
	) -> Result<usize, String> {
		let rows = lacc.len();
		let mut first = true;
		let mut pend = [0usize; 2];
		let mut collect = |m: &mut FarMulti, r: usize, used: usize| -> Result<(), String> {
			m.list[r].wait()?;
			let (o, l) = (raw(bufs, m.ob[r]), raw(bufs, m.lb[r]));
			for j in 0..used {
				// SAFETY: as `attend`'s merges
				unsafe {
					merge_rows(
						acc,
						lacc,
						&o[j * rows * 512..],
						&l[j * rows * 4..],
						std::mem::replace(&mut first, false),
						unnorm,
					)
				};
			}
			Ok(())
		};
		let (mut b, mut j) = (0, 0);
		while nfull - b >= self.n - 1 {
			let (used, r) = ((nfull - b).min(self.n), j % 2);
			if pend[r] > 0 {
				collect(self, r, pend[r])?;
			}
			let mut upd: Vec<(usize, usize, &HostBuf<'_>)> = vec![
				(0, self.qg, &bufs[qg]),
				(0, self.oi, &bufs[self.ob[r]]),
				(0, self.li, &bufs[self.lb[r]]),
			];
			for i in 0..self.n {
				let [k, v] = blocks[b + if i < used { i } else { 0 }];
				upd.push((0, self.k[i], &bufs[k]));
				upd.push((0, self.v[i], &bufs[v]));
			}
			self.list[r].update(&upd)?;
			self.list[r].submit()?;
			pend[r] = used;
			b += used;
			j += 1;
		}
		for r in [j % 2, (j + 1) % 2] {
			if pend[r] > 0 {
				collect(self, r, pend[r])?;
			}
		}
		Ok(b)
	}
}

/// One far set's near / far graphs and their buffers.
struct FarSet {
	si: usize,
	near: Batch<'static>,
	far: Vec<Batch<'static>>,
	/// argument positions: near qg, kpast, vpast, knew, vnew, mask, o, lse;
	/// far qg, kblk, vblk, o, lse
	nix: Vec<usize>,
	fix: Vec<usize>,
	fo: [usize; 2],
	fl: [usize; 2],
	no: usize,
	nl: usize,
	mask: [usize; 2],
	acc: Vec<f32>,
	lacc: Vec<f32>,
	/// the far graphs leave P V undivided (`far.nonorm` in the set dir)
	unnorm: bool,
}

fn raw(bufs: &[HostBuf<'static>], i: usize) -> &'static mut [u8] {
	// SAFETY: a live host allocation of the engine; the callers touch it
	// only while no job that uses it runs.
	unsafe { std::slice::from_raw_parts_mut(bufs[i].ptr() as *mut u8, bufs[i].len()) }
}

/// acc = wa * acc + wb * o per row of 256 (`o` f16), the block merge.
/// `unnorm`: `o` is P V with P = exp(score - max), not yet divided by the
/// block's sum (a far graph built with the `-DNONORM` Softmax kernel).
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn merge_rows(
	acc: &mut [f32],
	lacc: &mut [f32],
	o: &[u8],
	lse: &[u8],
	first: bool,
	unnorm: bool,
) {
	use std::arch::x86_64::*;
	if far_stats().is_some() {
		let lb = |r: usize| {
			f16_to_f32(u16::from_le_bytes([lse[r * 4], lse[r * 4 + 1]]))
				+ f16_to_f32(u16::from_le_bytes([lse[r * 4 + 2], lse[r * 4 + 3]])).ln()
		};
		FAR_LB.with(|c| c.borrow_mut().push((0..lacc.len()).map(lb).collect()));
	}
	for r in 0..lacc.len() {
		let m = f16_to_f32(u16::from_le_bytes([lse[r * 4], lse[r * 4 + 1]]));
		let z = f16_to_f32(u16::from_le_bytes([lse[r * 4 + 2], lse[r * 4 + 3]]));
		let lb = m + z.ln();
		let (wa, wb) = if first {
			lacc[r] = lb;
			(0.0f32, 1.0f32)
		} else {
			let la = lacc[r];
			if lb < la - 30.0 {
				continue;
			}
			let hi = la.max(lb);
			let ln = hi + ((la - hi).exp() + (lb - hi).exp()).ln();
			lacc[r] = ln;
			((la - ln).exp(), (lb - ln).exp())
		};
		let (va, vb) = (
			_mm256_set1_ps(wa),
			_mm256_set1_ps(if unnorm { wb / z } else { wb }),
		);
		let a = acc.as_mut_ptr().add(r * 256);
		let b = o.as_ptr().add(r * 512);
		for j in (0..256).step_by(8) {
			let x = _mm256_cvtph_ps(_mm_loadu_si128(b.add(j * 2) as *const __m128i));
			_mm256_storeu_ps(
				a.add(j),
				_mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), va, _mm256_mul_ps(x, vb)),
			);
		}
	}
}

/// `QWEN_NPU_FAR_STATS=FILE` (diagnostic; run it with `QWEN_NPU_FAR_GPU=0`,
/// the scan device's blocks come back merged): per attention call with full
/// blocks, how many of them carry less than `eps` of the softmax mass.
fn far_stats() -> Option<&'static std::path::Path> {
	static F: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
	F.get_or_init(|| std::env::var_os("QWEN_NPU_FAR_STATS").map(Into::into))
		.as_deref()
}

/// the attention layer of the call in progress (for `far_stats_line`)
static FAR_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
	/// log mass per row of every block merged since the last `far_stats_line`
	static FAR_LB: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Appends one line for the attention call whose merges are in `FAR_LB`
/// (the far blocks, then the near one) and whose total is `lacc` (rows
/// `hg * s + i`): `layer s nfar`, then per eps `1e-2 .. 1e-6`
/// `union:worst query head`: blocks no row needs, the largest mass a row
/// loses to them, and the mean blocks one query (all its heads) / one row
/// does not need.
fn far_stats_line(file: &std::path::Path, lacc: &[f32], s: usize) {
	use std::io::Write;
	let lbs = FAR_LB.with(|c| std::mem::take(&mut *c.borrow_mut()));
	let nfar = lbs.len().saturating_sub(1);
	if nfar == 0 {
		return;
	}
	let (rows, hgs) = (lacc.len(), lacc.len() / s);
	let share = |b: usize, r: usize| (lbs[b][r] - lacc[r]).exp();
	let mut line = format!(
		"{} {s} {nfar}",
		FAR_LAYER.load(std::sync::atomic::Ordering::Relaxed)
	);
	for eps in [1e-2f32, 1e-3, 1e-4, 1e-5, 1e-6] {
		let (mut lost, mut union, mut query, mut head) =
			(vec![0.0f32; rows], 0usize, 0usize, 0usize);
		for b in 0..nfar {
			let mut all = true;
			for i in 0..s {
				let n = (0..hgs).filter(|hg| share(b, hg * s + i) < eps).count();
				head += n;
				query += (n == hgs) as usize;
				all &= n == hgs;
			}
			if all {
				union += 1;
				(0..rows).for_each(|r| lost[r] += share(b, r));
			}
		}
		let worst = lost.iter().fold(0.0f32, |a, &b| a.max(b));
		line += &format!(
			" | {eps:e} {union}:{worst:.1e} {:.1} {:.1}",
			query as f64 / s as f64,
			head as f64 / rows as f64
		);
	}
	if let Ok(mut f) = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(file)
	{
		let _ = writeln!(f, "{line}");
	}
}

/// `merge_rows` for the scan device's result (`Scan::far_out`).
#[target_feature(enable = "avx2,fma")]
unsafe fn merge_rows_f32(
	acc: &mut [f32],
	lacc: &mut [f32],
	o: &[f32],
	ml: &[[f32; 2]],
	first: bool,
) {
	use std::arch::x86_64::*;
	for r in 0..lacc.len() {
		let [m, z] = ml[r];
		let lb = m + z.ln();
		let (wa, wb) = if first {
			lacc[r] = lb;
			(0.0f32, 1.0f32)
		} else {
			let la = lacc[r];
			if lb < la - 30.0 {
				continue;
			}
			let hi = la.max(lb);
			let ln = hi + ((la - hi).exp() + (lb - hi).exp()).ln();
			lacc[r] = ln;
			((la - ln).exp(), (lb - ln).exp())
		};
		let (va, vb) = (_mm256_set1_ps(wa), _mm256_set1_ps(wb / z));
		let a = acc.as_mut_ptr().add(r * 256);
		let b = o.as_ptr().add(r * 256);
		for j in (0..256).step_by(8) {
			_mm256_storeu_ps(
				a.add(j),
				_mm256_fmadd_ps(
					_mm256_loadu_ps(a.add(j)),
					va,
					_mm256_mul_ps(_mm256_loadu_ps(b.add(j)), vb),
				),
			);
		}
	}
}

/// The merged attention output `[4][6][s][256]` f32 as the post graph's
/// `o` `[s][4][6][256]` f16.
#[target_feature(enable = "avx2,f16c")]
unsafe fn store_o(acc: &[f32], dst: &mut [u8], s: usize) {
	use std::arch::x86_64::*;
	for hg in 0..24 {
		for i in 0..s {
			let a = acc.as_ptr().add((hg * s + i) * 256);
			let d = dst.as_mut_ptr().add((i * 24 + hg) * 512);
			for j in (0..256).step_by(8) {
				_mm_storeu_si128(
					d.add(j * 2) as *mut __m128i,
					_mm256_cvtps_ph(_mm256_loadu_ps(a.add(j)), _MM_FROUND_TO_NEAREST_INT),
				);
			}
		}
	}
}

/// `QWEN_NPU_KQ` / `QWEN_NPU_QQ` = `BITS[:RANGE]` (an accuracy simulation of
/// integer far keys / queries, not a speed-up): the keys of a full block /
/// the queries of an attention call are rounded to BITS-bit integers, per
/// row of 256 with the row's own scale (max |x|) or the fixed RANGE, and
/// stored back as f16.
fn sim_quant(var: &str) -> Option<(u32, f32)> {
	let v = std::env::var(var).ok()?;
	let mut it = v.split(':');
	let bits: u32 = it.next()?.parse().ok()?;
	(bits > 1).then(|| (bits, it.next().and_then(|r| r.parse().ok()).unwrap_or(0.0)))
}

/// Reports each new largest |x| of the simulated keys / queries.
fn sim_top(what: &str, top: f32) {
	use std::sync::atomic::{AtomicU32, Ordering};
	static TOP: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
	let t = &TOP[(what == "q") as usize];
	if top > f32::from_bits(t.load(Ordering::Relaxed)) * 1.05 {
		t.store(top.to_bits(), Ordering::Relaxed);
		eprintln!("npu engine: simulated int {what}: max |{what}| {top:.3}");
	}
}

/// Rows of 256 f16 rounded as `sim_quant` describes; the largest |x| seen.
#[target_feature(enable = "avx2,f16c")]
unsafe fn quant_rows(buf: &mut [u8], (bits, range): (u32, f32)) -> f32 {
	use std::arch::x86_64::*;
	let qmax = ((1u32 << (bits - 1)) - 1) as f32;
	let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
	let mut top = 0.0f32;
	for row in buf.chunks_exact_mut(512) {
		let p = row.as_mut_ptr();
		let mut m = _mm256_setzero_ps();
		for j in (0..512).step_by(16) {
			m = _mm256_max_ps(
				m,
				_mm256_and_ps(
					abs,
					_mm256_cvtph_ps(_mm_loadu_si128(p.add(j) as *const __m128i)),
				),
			);
		}
		let mut t = [0.0f32; 8];
		_mm256_storeu_ps(t.as_mut_ptr(), m);
		let mx = t.iter().fold(0.0f32, |a, &b| a.max(b));
		top = top.max(mx);
		let r = if range > 0.0 { range } else { mx };
		if r == 0.0 {
			continue;
		}
		let (sc, inv, lim) = (
			_mm256_set1_ps(qmax / r),
			_mm256_set1_ps(r / qmax),
			_mm256_set1_ps(qmax),
		);
		for j in (0..512).step_by(16) {
			let x = _mm256_mul_ps(
				_mm256_cvtph_ps(_mm_loadu_si128(p.add(j) as *const __m128i)),
				sc,
			);
			let q = _mm256_round_ps(
				_mm256_max_ps(
					_mm256_min_ps(x, lim),
					_mm256_sub_ps(_mm256_setzero_ps(), lim),
				),
				_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
			);
			_mm_storeu_si128(
				p.add(j) as *mut __m128i,
				_mm256_cvtps_ph(_mm256_mul_ps(q, inv), _MM_FROUND_TO_NEAREST_INT),
			);
		}
	}
	top
}

impl Far {
	fn kb(&self) -> usize {
		self.cap - 2
	}

	fn set(&self, si: usize) -> Option<usize> {
		self.sets.iter().position(|x| x.si == si)
	}

	/// Blocks for positions `0..end` (and the block position `end` opens).
	fn ensure(
		&mut self,
		npu: &'static Npu,
		bufs: &mut Vec<HostBuf<'static>>,
		end: usize,
	) -> Result<(), String> {
		let n = end / self.kb() + 1;
		let bytes = self.cap * ROWB;
		for bl in self.blocks.iter_mut() {
			while bl.len() < n {
				bl.push([
					alloc(npu, bufs, bytes, None)?,
					alloc(npu, bufs, bytes, None)?,
				]);
			}
		}
		Ok(())
	}

	/// Rows `[pos][4][256]` of positions `p0..p0 + n` between a layer's
	/// blocks and `k` / `v` buffers whose row 0 is position `base`.
	#[allow(clippy::too_many_arguments)]
	fn copy(
		&self,
		bufs: &[HostBuf<'static>],
		ai: usize,
		p0: usize,
		n: usize,
		kv: [usize; 2],
		base: usize,
		to_blocks: bool,
	) {
		let kb = self.kb();
		for w in 0..2 {
			let lin = raw(bufs, kv[w]);
			for p in p0..p0 + n {
				let blk = raw(bufs, self.blocks[ai][p / kb][w]);
				let (a, b) = (
					&mut blk[(p % kb) * ROWB..(p % kb + 1) * ROWB],
					&mut lin[(p - base) * ROWB..(p - base + 1) * ROWB],
				);
				if to_blocks {
					a.copy_from_slice(b);
					if w == 0 && p % kb == kb - 1 {
						if let Some(q) = sim_quant("QWEN_NPU_KQ") {
							// SAFETY: avx2 / f16c checked at load
							let top = unsafe { quant_rows(&mut blk[..kb * ROWB], q) };
							sim_top("k", top);
						}
					}
				} else {
					b.copy_from_slice(a);
				}
			}
		}
	}

	/// One attention layer's core for a chunk at `q0` (set `g` of
	/// `self.sets`) whose pre graph is done: far graph per full block, near
	/// graph, merge -> `o`.
	#[allow(clippy::too_many_arguments)]
	fn attend(
		&mut self,
		bufs: &[HostBuf<'static>],
		g: usize,
		ai: usize,
		q0: usize,
		s: usize,
		qg: usize,
		knew: usize,
		vnew: usize,
		slot: usize,
		o: usize,
		sc: &mut Option<Box<dyn Scan>>,
		poked: &mut std::time::Instant,
	) -> Result<(), String> {
		let t = std::time::Instant::now();
		let nfull = q0 / self.kb();
		FAR_LAYER.store(ai, std::sync::atomic::Ordering::Relaxed);
		attend(
			&self.blocks[ai],
			0,
			nfull,
			&mut self.sets[g],
			bufs,
			s,
			qg,
			Some([knew, vnew]),
			slot,
			o,
			sc,
			poked,
			&mut self.gpu,
		)?;
		self.t_att += t.elapsed().as_secs_f64();
		Ok(())
	}
}

/// Blocks `0..nfull` through the far graph (two in flight), block `nfull`
/// (and the chunk's own keys `kv`, if the near graph takes them) through
/// the near graph, the merge of all of them -> `o`.
///
/// With a scan device that does far attention (`Scan::far_begin`), the NPU
/// takes the blocks from the front and the device the last ones, all in one
/// submission made first: the device idles during attention otherwise. The
/// split follows the two rates measured over the calls before (`gpu`), so
/// that both finish together.
///
/// Blocks `0..from` are merged already (`FarMulti::run`).
#[allow(clippy::too_many_arguments)]
fn attend(
	blocks: &[[usize; 2]],
	from: usize,
	nfull: usize,
	fs: &mut FarSet,
	bufs: &[HostBuf<'static>],
	s: usize,
	qg: usize,
	kv: Option<[usize; 2]>,
	slot: usize,
	o: usize,
	sc: &mut Option<Box<dyn Scan>>,
	poked: &mut std::time::Instant,
	gpu: &mut [f64; 5],
) -> Result<(), String> {
	use std::time::Instant;
	let first = std::cell::Cell::new(from == 0);
	let mut pend = [false; 2];
	if nfull > 0 {
		if let Some(q) = sim_quant("QWEN_NPU_QQ") {
			// SAFETY: avx2 / f16c checked at load; the pre job is done
			let top = unsafe { quant_rows(raw(bufs, qg), q) };
			sim_top("q", top);
		}
	}
	// `QWEN_NPU_FAR_DUMP=DIR:N` (diagnostic): the queries and the K / V
	// blocks of every attention call with exactly N full blocks, as raw f16
	// (`L{layer}.qg`, `L{layer}.k{b}`, `L{layer}.v{b}`), for offline checks
	if let Some((d, n)) = std::env::var("QWEN_NPU_FAR_DUMP")
		.ok()
		.as_deref()
		.and_then(|v| v.rsplit_once(':'))
	{
		if n.parse() == Ok(nfull) && from == 0 {
			let l = FAR_LAYER.load(std::sync::atomic::Ordering::Relaxed);
			let _ = std::fs::write(format!("{d}/L{l}.qg"), &*raw(bufs, qg));
			for (b, &[k, v]) in blocks[..nfull].iter().enumerate() {
				let _ = std::fs::write(format!("{d}/L{l}.k{b}"), &*raw(bufs, k));
				let _ = std::fs::write(format!("{d}/L{l}.v{b}"), &*raw(bufs, v));
			}
		}
	}
	// SAFETY (merge_rows / store_o): avx2 / f16c / fma checked at load;
	// the job that wrote the buffers is done.
	let merge = |acc: &mut [f32], lacc: &mut [f32], o: usize, l: usize, unnorm: bool| {
		unsafe {
			merge_rows(
				acc,
				lacc,
				raw(bufs, o),
				raw(bufs, l),
				first.replace(false),
				unnorm,
			)
		};
	};
	let unnorm = fs.unnorm;
	let (gpu_on, lead) = far_gpu_cfg();
	let rel = 256.0 / s as f64;
	let ema = |x: &mut f64, ms: f64| *x = if *x == 0.0 { ms } else { 0.7 * *x + 0.3 * ms };
	// blocks [hi, nfull) are the scan device's (its submission in flight),
	// the cores' below them; each device's share by its ms per block (4, 9
	// and 5 until measured), the NPU keeps one at least
	let mut hi = nfull;
	let mut busy: Option<Instant> = None;
	let mut ngpu = 0usize;
	let gpu_ok = gpu_on && sc.is_some() && s.is_multiple_of(4) && nfull >= 2;
	let cpu_ok = far_cpu_on() && s >= 64 && nfull - from >= 2;
	let tn = if gpu[2] > 0.0 { gpu[2] } else { 4.0 };
	let tg = if gpu[0] > 0.0 { gpu[0] } else { 9.0 };
	let tc = if gpu[3] > 0.0 { gpu[3] } else { 5.0 };
	let rsum = 1.0 / tn + if gpu_ok { 1.0 / tg } else { 0.0 } + if cpu_ok { 1.0 / tc } else { 0.0 };
	if gpu_ok {
		let n = (((nfull as f64 + lead) / tg / rsum).round() as usize).min(nfull - 1);
		if n > 0 {
			let nk = bufs[blocks[0][0]].len() / ROWB - 2;
			let bl: Vec<(&[u8], &[u8])> = blocks[nfull - n..nfull]
				.iter()
				.map(|&[k, v]| (&*raw(bufs, k), &*raw(bufs, v)))
				.collect();
			// an error starts nothing: the NPU then takes every block
			let started = sc
				.as_mut()
				.unwrap()
				.far_begin(s, nk, raw(bufs, qg), &bl, true)
				.unwrap_or_else(|e| {
					static ONCE: std::sync::Once = std::sync::Once::new();
					ONCE.call_once(|| eprintln!("npu: far blocks on the NPU only: {e}"));
					false
				});
			if started {
				hi -= n;
				ngpu = n;
				busy = Some(Instant::now());
				gpu[1] += n as f64;
			}
		}
	}
	let mut cpu: Option<(crate::farcpu::Job, usize)> = None;
	if cpu_ok {
		let n = (((nfull - from) as f64 + lead) / tc / rsum).round() as usize;
		let n = n.min((hi - from).saturating_sub(1));
		if n > 0 {
			let nk = bufs[blocks[0][0]].len() / ROWB - 2;
			let bl: Vec<(&'static [u8], &'static [u8])> = blocks[hi - n..hi]
				.iter()
				.map(|&[k, v]| (&*raw(bufs, k), &*raw(bufs, v)))
				.collect();
			cpu = Some((crate::farcpu::Job::start(raw(bufs, qg), s, nk, bl), n));
			hi -= n;
			gpu[4] += n as f64;
		}
	}
	let on_gpu = busy.is_some();
	let t0 = Instant::now();
	for b in from..hi {
		if let Some(t) = busy {
			// only to time it: the result is collected after the near job
			if sc.as_mut().unwrap().far_poll()? {
				ema(
					&mut gpu[0],
					t.elapsed().as_secs_f64() * 1e3 / ngpu as f64 * rel,
				);
				busy = None;
			}
		} else if !on_gpu && poked.elapsed().as_millis() >= 30 {
			// keep an unused scan device awake (`Scan::poke`)
			if let Some(g) = sc.as_mut() {
				g.poke()?;
			}
			*poked = Instant::now();
		}
		let [k, v] = blocks[b];
		let r = b % 2;
		if pend[r] {
			fs.far[r].wait()?;
			merge(&mut fs.acc, &mut fs.lacc, fs.fo[r], fs.fl[r], unnorm);
		}
		let mut set = [0usize; 5];
		for (i, bi) in [qg, k, v, fs.fo[r], fs.fl[r]].into_iter().enumerate() {
			set[fs.fix[i]] = bi;
		}
		let upd: Vec<(usize, usize, &HostBuf<'_>)> = set
			.iter()
			.enumerate()
			.map(|(i, &b)| (0, i, &bufs[b]))
			.collect();
		fs.far[r].update(&upd)?;
		fs.far[r].submit()?;
		pend[r] = true;
	}
	let [k, v] = blocks[nfull];
	let [kn, vn] = kv.unwrap_or([usize::MAX; 2]);
	let mut upd: Vec<(usize, usize, &HostBuf<'_>)> = Vec::new();
	for (i, bi) in [qg, k, v, kn, vn, fs.mask[slot], fs.no, fs.nl]
		.into_iter()
		.enumerate()
	{
		if fs.nix[i] != usize::MAX {
			upd.push((0, fs.nix[i], &bufs[bi]));
		}
	}
	fs.near.update(&upd)?;
	fs.near.submit()?;
	for r in [hi % 2, (hi + 1) % 2] {
		if pend[r] {
			fs.far[r].wait()?;
			merge(&mut fs.acc, &mut fs.lacc, fs.fo[r], fs.fl[r], unnorm);
		}
	}
	if (on_gpu || cpu.is_some()) && hi - from >= 2 {
		ema(
			&mut gpu[2],
			t0.elapsed().as_secs_f64() * 1e3 / (hi - from) as f64 * rel,
		);
	}
	if let Some(t) = busy {
		// before the near job's wait: the time taken here is the device's
		sc.as_mut().unwrap().far_wait()?;
		ema(
			&mut gpu[0],
			t.elapsed().as_secs_f64() * 1e3 / ngpu as f64 * rel,
		);
	}
	fs.near.wait()?;
	merge(&mut fs.acc, &mut fs.lacc, fs.no, fs.nl, false);
	if on_gpu {
		let g = sc.as_mut().unwrap();
		let (go, gml) = g.far_out();
		let rows = fs.lacc.len();
		// SAFETY: as above; the device's submission is done
		unsafe {
			merge_rows_f32(
				&mut fs.acc,
				&mut fs.lacc,
				&go[..rows * 256],
				&gml[..rows],
				false,
			)
		};
		*poked = Instant::now();
	}
	if let Some((job, n)) = cpu {
		let (co, cml, ms) = job.finish()?;
		ema(&mut gpu[3], ms / n as f64 * rel);
		let rows = fs.lacc.len();
		if co.len() < rows * 256 || cml.len() < rows {
			return Err(format!(
				"far attention on the cores: {} rows for {rows}",
				cml.len()
			));
		}
		// SAFETY: as above
		unsafe {
			merge_rows_f32(
				&mut fs.acc,
				&mut fs.lacc,
				&co[..rows * 256],
				&cml[..rows],
				false,
			)
		};
	}
	if let Some(f) = far_stats() {
		far_stats_line(f, &fs.lacc, s);
	}
	// `o` is read by the post job submitted next
	unsafe { store_o(&fs.acc, raw(bufs, o), s) };
	Ok(())
}

/// The near / far graphs of a far set dir as a `FarSet` (set index `si`),
/// and the rows of a block buffer.
fn far_set(
	npu: &'static Npu,
	bufs: &mut Vec<HostBuf<'static>>,
	d: &str,
	si: usize,
) -> Result<(FarSet, usize), String> {
	let gs = load_blobs(npu, &[format!("{d}/near.blob"), format!("{d}/far.blob")])?;
	let (gn, gf) = gs.split_at_mut(1);
	let (gn, gf) = (&mut gn[0], &mut gf[0]);
	let ix = |g: &Graph<'_>, names: &[&str]| -> Vec<usize> {
		// a near graph for decode steps has no knew / vnew
		let mut v: Vec<usize> = names
			.iter()
			.map(|n| {
				g.args
					.iter()
					.position(|a| a.name == *n && !a.is_output)
					.unwrap_or(usize::MAX)
			})
			.collect();
		// outputs: o (the large one), then lse
		let mut o: Vec<usize> = (0..g.args.len()).filter(|&i| g.args[i].is_output).collect();
		o.sort_by_key(|&i| std::cmp::Reverse(g.args[i].bytes));
		v.extend(o);
		v
	};
	let nix = ix(gn, &["qg", "kpast", "vpast", "knew", "vnew", "mask"]);
	let fix = ix(gf, &["qg", "kblk", "vblk"]);
	if nix.len() != 8
		|| fix.len() != 5
		|| fix.contains(&usize::MAX)
		|| [0, 1, 2, 5].iter().any(|&i| nix[i] == usize::MAX)
	{
		return Err(format!("{d}: near / far graphs: unexpected arguments"));
	}
	let cap = gf.args[fix[1]].dims[1] as usize;
	if gn.args[nix[1]].dims[1] as usize != cap {
		return Err(format!("{d}: near / far graphs with different block sizes"));
	}
	let nb: Vec<usize> = gn
		.args
		.iter()
		.map(|a| alloc(npu, bufs, a.bytes, None))
		.collect::<Result<_, _>>()?;
	let fb: Vec<usize> = gf
		.args
		.iter()
		.map(|a| alloc(npu, bufs, a.bytes, None))
		.collect::<Result<_, _>>()?;
	let fb2 = [
		alloc(npu, bufs, gf.args[fix[3]].bytes, None)?,
		alloc(npu, bufs, gf.args[fix[4]].bytes, None)?,
	];
	let mask2 = alloc(npu, bufs, gn.args[nix[5]].bytes, None)?;
	let near = {
		let refs: Vec<&HostBuf<'_>> = nb.iter().map(|&i| &bufs[i]).collect();
		npu.batch_plan_mutable(std::slice::from_mut(gn), &[(0, refs)])?
	};
	let mut fl = Vec::new();
	for _ in 0..2 {
		let refs: Vec<&HostBuf<'_>> = fb.iter().map(|&i| &bufs[i]).collect();
		fl.push(npu.batch_plan_mutable(std::slice::from_mut(gf), &[(0, refs)])?);
	}
	let rows = gf.args[fix[4]].bytes / 4;
	Ok((
		FarSet {
			si,
			near,
			far: fl,
			fo: [fb[fix[3]], fb2[0]],
			fl: [fb[fix[4]], fb2[1]],
			no: nb[nix[6]],
			nl: nb[nix[7]],
			mask: [nb[nix[5]], mask2],
			acc: vec![0.0; rows * 256],
			lacc: vec![0.0; rows],
			unnorm: std::path::Path::new(&format!("{d}/far.nonorm")).exists(),
			nix,
			fix,
		},
		cap,
	))
}

// SAFETY: the engine owns its NPU handles and host buffers; it is used from
// one thread at a time (the server holds it behind a Mutex).
unsafe impl Send for NpuEngine {}

/// Row `r` of a hidden-state output as f16 bytes (the compiled blobs keep
/// the MTP's `h` / the head's `hn` f32 although the IR converts to f16).
fn h_row(src: &[u8], r: usize, f32out: bool) -> Vec<u8> {
	if f32out {
		src[r * W * 4..(r + 1) * W * 4]
			.chunks_exact(4)
			.flat_map(|c| f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
			.collect()
	} else {
		src[r * W * 2..(r + 1) * W * 2].to_vec()
	}
}

/// The `[5120]` f32 weight of the final norm in an MTP head IR (`heads.py`:
/// the `hn` output is `x / rms(x) * w`, eps 1e-6).
fn final_norm(xml: &str, bin: &str) -> Result<Vec<f32>, String> {
	let x = std::fs::read_to_string(xml).map_err(|e| format!("{xml}: {e}"))?;
	let tag = "element_type=\"f32\" shape=\"1, 1, 5120\" offset=\"";
	let i = x.find(tag).ok_or(format!(
		"{xml}: no [1, 1, 5120] f32 constant (the final norm)"
	))? + tag.len();
	let off: u64 = x[i..]
		.split('"')
		.next()
		.and_then(|v| v.parse().ok())
		.ok_or(format!("{xml}: bad offset"))?;
	let mut b = vec![0u8; W * 4];
	std::fs::File::open(bin)
		.and_then(|f| f.seek_read(&mut b, off).map(|_| ()))
		.map_err(|e| format!("{bin}: {e}"))?;
	Ok(b.chunks_exact(4)
		.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect())
}

/// Run `jobs` on up to 16 threads.
fn run_parallel(jobs: Vec<Box<dyn FnOnce() + Send>>) {
	let n = jobs.len().clamp(1, 16);
	let queue = std::sync::Mutex::new(jobs);
	std::thread::scope(|sc| {
		for _ in 0..n {
			sc.spawn(|| loop {
				let Some(j) = queue.lock().unwrap().pop() else {
					break;
				};
				j();
			});
		}
	});
}

/// Prefill -> decode hand-over of one delta layer: the prefill's state
/// `S[h][k][v] x vmul` (f16) into the decode state (`St[h][v][k]` when
/// `transposed`, else `S` as is), unscaled by `inv`; the conv rows (oldest
/// -> newest, `[3][C]`) into columns 1..3 of the `[C][4]` window.
fn handover_delta(
	ssm: &[u8],
	conv: &[u8],
	st: &mut [u8],
	cw: &mut [u8],
	inv: f32,
	transposed: bool,
) {
	let hd = 128;
	let n = ssm.len() / 2;
	let mut t = vec![0f32; n];
	crate::ir::f16s_to_f32(ssm, &mut t, inv);
	if transposed {
		let mut u = vec![0f32; n];
		for h in 0..n / (hd * hd) {
			for kk in 0..hd {
				for v in 0..hd {
					u[(h * hd + v) * hd + kk] = t[(h * hd + kk) * hd + v];
				}
			}
		}
		t = u;
	}
	crate::ir::f32s_to_f16(&t, st, 1.0);
	let ch = conv.len() / 2 / 3;
	for c in 0..ch {
		cw[c * 8..c * 8 + 2].fill(0);
		for r in 0..3 {
			let o = (c * 4 + 1 + r) * 2;
			cw[o..o + 2].copy_from_slice(&conv[(r * ch + c) * 2..(r * ch + c) * 2 + 2]);
		}
	}
}

/// Prefill -> decode hand-over of one attention layer's K or V window:
/// positions `lo..hi` of the prefill's token-major rows `[pos][4][256]` into
/// the decode ring (`K [4][cap][256]`, `V` transposed `[4][256][cap]`).
fn handover_kv(src: &[u8], ps: &mut [u8], is_v: bool, cap: usize, lo: usize, hi: usize) {
	for h in 0..4 {
		if is_v {
			for d in 0..256 {
				let row = &mut ps[(h * 256 + d) * cap * 2..(h * 256 + d + 1) * cap * 2];
				for pos in lo..hi {
					let o = ((pos * 4 + h) * 256 + d) * 2;
					row[pos * 2..pos * 2 + 2].copy_from_slice(&src[o..o + 2]);
				}
			}
		} else {
			for pos in lo..hi {
				let o = (h * cap + pos) * 512;
				ps[o..o + 512].copy_from_slice(&src[(pos * 4 + h) * 512..(pos * 4 + h + 1) * 512]);
			}
		}
	}
}

fn leak<T>(v: T) -> &'static mut T {
	Box::leak(Box::new(v))
}

fn alloc(
	npu: &'static Npu,
	bufs: &mut Vec<HostBuf<'static>>,
	bytes: usize,
	data: Option<Vec<u8>>,
) -> Result<usize, String> {
	let mut b = npu.alloc(bytes)?;
	match data {
		Some(d) => b.as_mut_slice()[..d.len()].copy_from_slice(&d),
		None => b.as_mut_slice().fill(0),
	}
	bufs.push(b);
	Ok(bufs.len() - 1)
}

/// A pinned buffer of at least `min` bytes holding file `f`, read straight
/// into it without the file cache (`FILE_FLAG_NO_BUFFERING`; the length is
/// rounded up to 4 KiB): a `Vec` read through the cache and then copied
/// cost a copy, and the cached copies pushed other pages out. The buffer's
/// index and the file's length.
pub(crate) fn read_pinned(
	npu: &'static Npu,
	bufs: &mut Vec<HostBuf<'static>>,
	f: &Path,
	min: usize,
) -> Result<(usize, usize), String> {
	use std::io::Read;
	let err = |e: std::io::Error| format!("{}: {e}", f.display());
	let mut o = std::fs::OpenOptions::new();
	o.read(true);
	#[cfg(windows)]
	std::os::windows::fs::OpenOptionsExt::custom_flags(&mut o, 0x2000_0000);
	let mut file = o.open(f).map_err(err)?;
	let n = file.metadata().map_err(err)?.len() as usize;
	let mut b = npu.alloc(n.max(min).div_ceil(4096) * 4096)?;
	let dst = &mut b.as_mut_slice()[..n.div_ceil(4096) * 4096];
	let mut at = 0;
	while at < n {
		let k = file.read(&mut dst[at..]).map_err(err)?;
		if k == 0 {
			return Err(format!("{}: short read", f.display()));
		}
		at += k;
	}
	bufs.push(b);
	Ok((bufs.len() - 1, n))
}

/// A `bindc.py` weight input (`w.*`): the bytes of `DIR/NAME.in`, one buffer
/// per name (names are content-addressed, so graphs naming the same file
/// share it; an input may read a prefix, e.g. the draft head's lm_head rows).
fn bound_weight(
	npu: &'static Npu,
	bufs: &mut Vec<HostBuf<'static>>,
	cache: &mut std::collections::HashMap<String, usize>,
	dir: &str,
	name: &str,
	bytes: usize,
) -> Result<usize, String> {
	let f = Path::new(dir).join(format!("{name}.in"));
	let (b, n) = match cache.get(name) {
		// (the name may come from another set's directory)
		Some(&b) => (
			b,
			std::fs::metadata(&f).map_or(bufs[b].len(), |m| m.len() as usize),
		),
		None => {
			let (b, n) = read_pinned(npu, bufs, &f, 0)?;
			cache.insert(name.to_string(), b);
			(b, n)
		}
	};
	if n < bytes {
		return Err(format!(
			"{dir}: input {name} wants {bytes} bytes, its file has {n}"
		));
	}
	Ok(b)
}

/// The distinct blobs of a set and, per layer, its blob (the pre half of a
/// cut layer) and a cut layer's post blob. `lean{L}a` (with `lean{L}b`)
/// replaces `lean{L}`; a `.use` file names the blob a duplicate shares.
/// Without a scan device (`scan` false) a delta layer takes its uncut blob
/// where the set has one (`mkcut.sh` keeps them): the recurrence then runs
/// on the NPU.
fn blob_names(dir: &str, scan: bool) -> Result<(Vec<String>, Vec<(usize, Option<usize>)>), String> {
	let resolve = |base: &str| -> Result<Option<String>, String> {
		let u = Path::new(dir).join(format!("{base}.use"));
		if u.exists() {
			return Ok(Some(
				std::fs::read_to_string(&u)
					.map_err(|e| e.to_string())?
					.trim()
					.to_string(),
			));
		}
		Ok(Path::new(dir)
			.join(format!("{base}.blob"))
			.exists()
			.then(|| base.to_string()))
	};
	let mut names: Vec<String> = Vec::new();
	let mut index = |nm: String| match names.iter().position(|x| *x == nm) {
		Some(i) => i,
		None => {
			names.push(nm);
			names.len() - 1
		}
	};
	let mut lb = Vec::new();
	for l in 0..NL {
		let uncut = if scan || l % 4 == 3 {
			None
		} else {
			resolve(&format!("lean{l}"))?
		};
		match resolve(&format!("lean{l}a"))?.filter(|_| uncut.is_none()) {
			Some(a) => {
				let b = resolve(&format!("lean{l}b"))?
					.ok_or(format!("{dir}: lean{l}a without lean{l}b"))?;
				let ga = index(a);
				lb.push((ga, Some(index(b))));
			}
			None => {
				let nm = match uncut {
					Some(nm) => nm,
					None => resolve(&format!("lean{l}"))?.unwrap_or(format!("lean{l}")),
				};
				lb.push((index(nm), None));
			}
		}
	}
	Ok((names, lb))
}

fn load_blobs(
	npu: &'static Npu,
	files: &[String],
) -> Result<&'static mut Vec<Graph<'static>>, String> {
	let raws: &'static Vec<Vec<u8>> = leak(
		files
			.iter()
			.map(|f| std::fs::read(f).map_err(|e| format!("{f}: {e}")))
			.collect::<Result<Vec<_>, _>>()?,
	);
	Ok(leak(
		raws.iter()
			.map(|b| npu.load(b, 0))
			.collect::<Result<Vec<_>, _>>()?,
	))
}

impl PSet {
	/// Submit job `ji` of `slot` at state parity `par` on the next list of
	/// its blob's ring (waiting for that list first if it still runs); the
	/// list it took.
	fn submit(
		&mut self,
		bufs: &[HostBuf<'static>],
		slot: usize,
		ji: usize,
		par: usize,
	) -> Result<(usize, usize), String> {
		let j = &self.jobs[slot][ji];
		let gi = j.gi;
		let r = self.next[gi] % self.lists[gi].len();
		self.next[gi] += 1;
		if self.busy[gi][r] {
			self.lists[gi][r].wait()?;
			self.busy[gi][r] = false;
		}
		if self.cur[gi][r] != (slot, ji, par) {
			let mut set = j.set.clone();
			if let Some(d) = self.dls[j.l].as_ref().filter(|_| !j.post) {
				for (a, b) in [
					(0, d.cb[par]),
					(1, d.sb[par]),
					(2, d.cb[par ^ 1]),
					(3, d.sb[par ^ 1]),
				] {
					if d.args[a] != usize::MAX {
						set[d.args[a]] = b;
					}
				}
			}
			let upd: Vec<(usize, usize, &HostBuf<'_>)> = set
				.iter()
				.enumerate()
				.map(|(i, &b)| (0, i, &bufs[b]))
				.collect();
			self.lists[gi][r].update(&upd)?;
			self.cur[gi][r] = (slot, ji, par);
		}
		self.lists[gi][r].submit()?;
		self.busy[gi][r] = true;
		Ok((gi, r))
	}

	fn wait(&mut self, (gi, r): (usize, usize)) -> Result<(), String> {
		if self.busy[gi][r] {
			self.lists[gi][r].wait()?;
			self.busy[gi][r] = false;
		}
		Ok(())
	}
}

impl NpuEngine {
	pub fn load(cfg: &Config) -> Result<NpuEngine, String> {
		Self::load_with_scan(cfg, None)
	}

	/// [`Self::load`] with the device that runs cut delta layers' recurrence
	/// (needed when a set has `lean{L}a` / `lean{L}b` blobs).
	pub fn load_with_scan(cfg: &Config, scan: Option<Box<dyn Scan>>) -> Result<NpuEngine, String> {
		let t0 = std::time::Instant::now();
		let lock = NpuLock::acquire()?;
		let npu: &'static Npu = leak(Npu::open()?);
		let mut bufs: Vec<HostBuf<'static>> = Vec::new();
		let table = std::fs::File::open(&cfg.emb).map_err(|e| format!("{}: {e}", cfg.emb))?;

		// ---- prefill sets ----
		let dirs: Vec<String> = std::iter::once(cfg.prefill.clone())
			.chain(cfg.small.clone())
			.chain(cfg.long.clone())
			.chain(cfg.long_small.clone())
			.chain(cfg.sets.iter().cloned())
			.collect();
		let mut pgraphs: Vec<&'static mut Vec<Graph<'static>>> = Vec::new();
		let mut lbs = Vec::new();
		for d in &dirs {
			let (names, lb) = blob_names(d, scan.is_some())?;
			pgraphs.push(load_blobs(
				npu,
				&names
					.iter()
					.map(|n| format!("{d}/{n}.blob"))
					.collect::<Vec<_>>(),
			)?);
			lbs.push(lb);
		}
		let svals: Vec<usize> = pgraphs
			.iter()
			.map(|g| {
				g[0].args
					.iter()
					.find(|x| x.name == "x")
					.map(|x| x.bytes / (W * 2))
					.unwrap_or(0)
			})
			.collect();
		let caps: Vec<usize> = pgraphs
			.iter()
			.map(|g| {
				g.iter()
					.flat_map(|x| x.args.iter())
					.find(|x| x.name == "kpast")
					.map(|x| x.dims[1] as usize)
					.unwrap_or(0)
			})
			.collect();
		let far_sis: Vec<usize> = (0..dirs.len())
			.filter(|&i| Path::new(&dirs[i]).join("far.blob").exists())
			.collect();
		let smax = *svals.iter().max().unwrap();
		// rows of the K / V window buffers: with far sets, also what the
		// decode set's window takes at the hand-over
		let maxcap = if far_sis.is_empty() {
			*caps.iter().max().unwrap()
		} else {
			(*caps.iter().max().unwrap()).max(4096)
		};
		let caps: Vec<usize> = caps
			.iter()
			.enumerate()
			.map(|(i, &c)| if far_sis.contains(&i) { usize::MAX } else { c })
			.collect();
		let x0 = [
			alloc(npu, &mut bufs, smax * W * 2, None)?,
			alloc(npu, &mut bufs, smax * W * 2, None)?,
		];
		let mut presid: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
		for p in presid.iter_mut() {
			*p = (0..NL)
				.map(|_| alloc(npu, &mut bufs, smax * W * 2, None))
				.collect::<Result<_, _>>()?;
		}
		let mut bind = Bind {
			npu,
			bufs: &mut bufs,
			weights: Default::default(),
			st_in: Default::default(),
			x0: [x0[0], x0[1]],
			presid: [presid[0].clone(), presid[1].clone()],
			slot: 0,
			smax,
			maxcap,
		};
		let mut psets: Vec<PSet> = Vec::new();
		for (si, gs) in pgraphs.into_iter().enumerate() {
			let mut ps = PSet {
				s: svals[si],
				cap: caps[si],
				jobs: [Vec::new(), Vec::new()],
				dls: Vec::new(),
				scan_bufs: [None, None],
				atts: [Vec::new(), Vec::new()],
				shared: Default::default(),
				lists: Vec::new(),
				busy: Vec::new(),
				cur: Vec::new(),
				next: Vec::new(),
			};
			// slot 1: a second chunk in flight (`PSet` docs)
			for slot in 0..2 {
				bind.slot = slot;
				for l in 0..NL {
					let (gi, post) = lbs[si][l];
					let mut dl = Dl {
						args: [usize::MAX; 4],
						cb: [0; 2],
						sb: [0; 2],
					};
					let mut at = [usize::MAX; 4];
					let att = l % 4 == 3;
					let role = match (post.is_some(), att) {
						(false, _) => Role::Full,
						(true, false) => Role::Pre,
						(true, true) => Role::APre,
					};
					let set = bind.job(
						&gs[gi].args,
						l,
						role,
						&dirs[si],
						ps.s,
						&mut ps.shared[slot],
						&mut dl,
						&mut at,
					)?;
					ps.jobs[slot].push(Job {
						l,
						gi,
						set,
						pre: post.is_some(),
						post: false,
					});
					if let Some(gp) = post {
						let set = bind.job(
							&gs[gp].args,
							l,
							if att { Role::APost } else { Role::Post },
							&dirs[si],
							ps.s,
							&mut ps.shared[slot],
							&mut dl,
							&mut at,
						)?;
						ps.jobs[slot].push(Job {
							l,
							gi: gp,
							set,
							pre: false,
							post: true,
						});
					}
					if post.is_some() && !att {
						let sh = &ps.shared[slot];
						let sb = |n: &str| {
							sh.get(n)
								.copied()
								.ok_or(format!("{}: cut layer {l} has no {n}", dirs[si]))
						};
						ps.scan_bufs[slot] = Some([
							sb("scan.qn")?,
							sb("scan.kn")?,
							sb("scan.v")?,
							sb("scan.gb")?,
							sb("scan.o")?,
						]);
					}
					if l % 4 == 3 {
						ps.atts[slot].push(Some(at));
					} else {
						ps.atts[slot].push(None);
						if slot == 0 {
							ps.dls.push(Some(dl));
						}
					}
					if slot == 0 && l % 4 == 3 {
						ps.dls.push(None);
					}
				}
			}
			// mutable lists per blob, a ring of up to 3 (a blob one layer
			// uses has one: a pair's second job on it waits for the first,
			// which costs one wait in layer 0; more lists exhaust the
			// device's memory, 0x70000002)
			let mut nuse = vec![0usize; gs.len()];
			for j in &ps.jobs[0] {
				nuse[j.gi] += 1;
			}
			let mut lists: Vec<Vec<Batch<'static>>> = (0..gs.len()).map(|_| Vec::new()).collect();
			let mut gref: Vec<Option<&'static mut Graph<'static>>> =
				gs.iter_mut().map(Some).collect();
			for j in &ps.jobs[0] {
				if lists[j.gi].len() < 3.min(nuse[j.gi]) {
					let refs: Vec<&HostBuf<'_>> = j.set.iter().map(|&i| &bind.bufs[i]).collect();
					let g = gref[j.gi].as_deref_mut().unwrap();
					lists[j.gi]
						.push(npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, refs)])?);
				}
			}
			ps.busy = lists.iter().map(|v| vec![false; v.len()]).collect();
			ps.cur = lists
				.iter()
				.map(|v| vec![(usize::MAX, usize::MAX, usize::MAX); v.len()])
				.collect();
			ps.next = vec![0; lists.len()];
			ps.lists = lists;
			psets.push(ps);
		}
		// ---- far attention (`mkfar.sh` sets): near / far graphs, K / V blocks ----
		let mut far: Option<Far> = None;
		for &si in &far_sis {
			let (fs, cap) = far_set(npu, bind.bufs, &dirs[si], si)?;
			let f = far.get_or_insert_with(|| Far {
				cap,
				blocks: vec![Vec::new(); NL / 4],
				live: false,
				sets: Vec::new(),
				dec: None,
				t_att: 0.0,
				gpu: [0.0; 5],
			});
			if f.cap != cap {
				return Err(format!("{}: far sets with different block sizes", dirs[si]));
			}
			f.sets.push(fs);
		}
		if let Some(f) = far.as_mut() {
			// `QWEN_NPU_FAR_PREALLOC=N` (a test): the blocks of N positions now
			if let Some(n) = std::env::var("QWEN_NPU_FAR_PREALLOC")
				.ok()
				.and_then(|v| v.parse::<usize>().ok())
			{
				let t = std::time::Instant::now();
				f.ensure(npu, bind.bufs, n)?;
				eprintln!("npu engine: K / V blocks for {n} positions ({} buffers of {} bytes) in {:.1} s", f.blocks.iter().map(|b| b.len() * 2).sum::<usize>(), f.cap * ROWB, t.elapsed().as_secs_f64());
			}
		}
		let weights = bind.weights;

		// ---- decode set ----
		let outs: Vec<Vec<String>> = {
			let f = format!("{}/outs.txt", cfg.decode);
			let t = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
			t.lines()
				.map(|l| {
					l.split_whitespace()
						.skip(1)
						.map(|s| s.to_string())
						.collect()
				})
				.collect()
		};
		let hdir = cfg.mtp.clone().unwrap_or_else(|| cfg.decode.clone());
		let head = format!("{hdir}/head.blob");
		// `bindc.py` weights (`w.*` inputs), one buffer per name
		let mut cw: std::collections::HashMap<String, usize> = Default::default();
		let files: Vec<String> = (0..NL)
			.map(|l| format!("{}/layer{l}.blob", cfg.decode))
			.chain(Some(head))
			.collect();
		let dgraphs = load_blobs(npu, &files)?;
		let xa = dgraphs[0]
			.args
			.iter()
			.find(|a| a.name == "x")
			.ok_or("decode layer 0: no x")?
			.clone();
		let k = xa.dims[1] as usize;
		let emb = alloc(npu, &mut bufs, k * W * 2, None)?;
		let resid: Vec<usize> = (0..NL)
			.map(|_| alloc(npu, &mut bufs, k * W * 2, None))
			.collect::<Result<_, _>>()?;
		let (mut bias_i, mut pos_i, mut logits_i, mut vocab, mut cap) =
			(usize::MAX, usize::MAX, usize::MAX, 0usize, 0usize);
		let mut hn: Option<(usize, bool)> = None;
		let mut plan: Vec<Vec<usize>> = Vec::new();
		let mut deltas: Vec<DDelta> = Vec::new();
		let mut kv: Vec<(usize, usize, bool, usize)> = Vec::new();
		for gi in 0..=NL {
			let args = dgraphs[gi].args.clone();
			let is_head = gi == NL;
			let onames: Vec<String> = if is_head {
				vec!["logits".into(), "hn".into()]
			} else {
				outs.get(gi).cloned().ok_or("outs.txt too short")?
			};
			let mut set = Vec::new();
			let mut ko = 0;
			let mut d = DDelta {
				g: gi,
				conv_in: usize::MAX,
				ssm_in: usize::MAX,
				conv_out: 0,
				ssm_out: 0,
				conv: [0; 2],
				ssm: [0; 2],
				wy: None,
			};
			let mut wy = Wy {
				k_in: 0,
				k_out: 0,
				d_in: 0,
				d_out: 0,
				kb: [0; 2],
				db: [0; 2],
				w: 0,
				alpha: 0,
				gcum: 0,
				conv_full: 0,
			};
			let mut pasts: Vec<(String, usize)> = Vec::new();
			for (i, arg) in args.iter().enumerate() {
				let b = if arg.is_output {
					let nm = onames.get(ko).cloned().unwrap_or_default();
					ko += 1;
					if is_head && nm == "hn" {
						let b = alloc(npu, &mut bufs, arg.bytes, None)?;
						hn = Some((b, arg.bytes == k * W * 4));
						b
					} else if is_head {
						let b = alloc(npu, &mut bufs, arg.bytes, None)?;
						logits_i = b;
						vocab = arg.bytes / 4 / k;
						b
					} else if nm == "y" {
						resid[gi]
					} else if nm == "conv_full" {
						d.conv_out = i;
						wy.conv_full = alloc(npu, &mut bufs, arg.bytes, None)?;
						wy.conv_full
					} else if nm == "sbase" {
						d.ssm_out = i;
						d.ssm[1]
					} else if nm == "kcur" {
						wy.k_out = i;
						wy.kb[1] = alloc(npu, &mut bufs, arg.bytes, None)?;
						wy.kb[1]
					} else if nm == "dnew" {
						wy.d_out = i;
						wy.db[1] = alloc(npu, &mut bufs, arg.bytes, None)?;
						wy.db[1]
					} else if nm == "gcum" {
						wy.gcum = alloc(npu, &mut bufs, arg.bytes, None)?;
						wy.gcum
					} else if let Some(st) = nm.strip_prefix("cache_params.present.") {
						if st.starts_with("key") || st.starts_with("value") {
							let p = pasts
								.iter()
								.find(|(n, _)| n == st)
								.map(|x| x.1)
								.ok_or(format!("layer {gi}: no past for {nm}"))?;
							let q = alloc(npu, &mut bufs, arg.bytes, None)?;
							kv.push((q, p, st.starts_with("value"), gi));
							q
						} else if st.starts_with("ssm") {
							d.ssm_out = i;
							d.ssm[1]
						} else {
							d.conv_out = i;
							d.conv[1]
						}
					} else {
						return Err(format!("decode layer {gi}: unknown output {nm}"));
					}
				} else {
					match arg.name.as_str() {
						"x" => {
							if gi == 0 {
								emb
							} else {
								resid[gi - 1]
							}
						}
						"attn_bias" => {
							if bias_i == usize::MAX {
								bias_i = alloc(npu, &mut bufs, arg.bytes, None)?;
								cap = arg.bytes / 2;
							}
							bias_i
						}
						"position_ids" => {
							if pos_i == usize::MAX {
								pos_i = alloc(npu, &mut bufs, arg.bytes, None)?;
							}
							pos_i
						}
						"qwen_rs.kp" => {
							wy.k_in = i;
							wy.kb[0] = alloc(npu, &mut bufs, arg.bytes, None)?;
							wy.kb[0]
						}
						"qwen_rs.dp" => {
							wy.d_in = i;
							wy.db[0] = alloc(npu, &mut bufs, arg.bytes, None)?;
							wy.db[0]
						}
						"qwen_rs.w" => {
							wy.w = alloc(npu, &mut bufs, arg.bytes, None)?;
							wy.w
						}
						"qwen_rs.alpha" => {
							wy.alpha = alloc(npu, &mut bufs, arg.bytes, None)?;
							wy.alpha
						}
						nm if nm.starts_with("cache_params.past.") => {
							let st = nm.trim_start_matches("cache_params.past.").to_string();
							let b = alloc(npu, &mut bufs, arg.bytes, None)?;
							if st.starts_with("ssm") {
								d.ssm_in = i;
								d.ssm = [b, alloc(npu, &mut bufs, arg.bytes, None)?];
							} else if st.starts_with("conv") {
								d.conv_in = i;
								d.conv = [b, if k == 1 { alloc(npu, &mut bufs, arg.bytes, None)? } else { b }];
							}
							pasts.push((st, b));
							b
						}
						// `bindw.py`: int4 codes as inputs named after the
						// prefill's weight files, bound to the same buffers
						nm if weights.contains_key(&(gi, nm.to_string())) => {
							let b = weights[&(gi, nm.to_string())];
							if bufs[b].len() < arg.bytes {
								return Err(format!("decode layer {gi}: input {nm} wants {} bytes, the prefill's buffer has {}", arg.bytes, bufs[b].len()));
							}
							b
						}
						nm if nm.starts_with("w.") => {
							let dir = if is_head { &hdir } else { &cfg.decode };
							bound_weight(npu, &mut bufs, &mut cw, dir, nm, arg.bytes)?
						}
						nm => return Err(format!("decode layer {gi}: unknown input {nm} (needs the lean attention graphs)")),
					}
				};
				set.push(b);
			}
			plan.push(set);
			if wy.alpha != 0 {
				if wy.gcum == 0 || wy.conv_full == 0 || wy.kb[1] == 0 || wy.db[1] == 0 {
					return Err(format!("decode layer {gi}: incomplete deferred-rollback outputs (outs.txt: y conv_full sbase kcur dnew gcum)"));
				}
				d.wy = Some(wy);
			}
			if d.ssm_in != usize::MAX {
				deltas.push(d);
			}
		}
		if bias_i == usize::MAX {
			return Err("decode set: no attn_bias input (build it with attn_lean.py --vt)".into());
		}
		let wy = deltas.iter().all(|d| d.wy.is_some());
		if k > 1 && !wy {
			return Err(format!(
				"decode set of {k} positions: needs the wyrec.py delta layers"
			));
		}
		let refs: Vec<(usize, Vec<&HostBuf<'_>>)> = plan
			.iter()
			.enumerate()
			.map(|(g, s)| (g, s.iter().map(|&i| &bufs[i]).collect()))
			.collect();
		// a cut set (`cutmlp.py`): one list per layer, the CPU runs each
		// layer's MLP in between (`mlp.rs`); `batch` is the head alone
		let mut parts = Vec::new();
		let mut mlp: Vec<crate::mlp::Mlp> = Vec::new();
		let batch = if Path::new(&format!("{}/layer0.mlp", cfg.decode)).exists() {
			if !(std::is_x86_feature_detected!("avx2")
				&& std::is_x86_feature_detected!("f16c")
				&& std::is_x86_feature_detected!("fma"))
			{
				return Err(
					"a cut decode set (layer{L}.mlp) needs a CPU with AVX2, F16C and FMA".into(),
				);
			}
			for (g, set) in &refs[..NL] {
				parts.push(npu.batch_plan_mutable(&mut dgraphs[*g..*g + 1], &[(0, set.clone())])?);
			}
			let mut slots: Vec<Option<Result<crate::mlp::Mlp, String>>> =
				(0..NL).map(|_| None).collect();
			std::thread::scope(|sc| {
				for (c, ch) in slots.chunks_mut(8).enumerate() {
					let dir = &cfg.decode;
					sc.spawn(move || {
						for (j, o) in ch.iter_mut().enumerate() {
							let l = c * 8 + j;
							*o = Some(crate::mlp::Mlp::load(&format!("{dir}/layer{l}.mlp"), l));
						}
					});
				}
			});
			for o in slots {
				mlp.push(o.unwrap()?);
			}
			npu.batch_plan_mutable(&mut dgraphs[NL..NL + 1], &[(0, refs[NL].1.clone())])?
		} else {
			npu.batch_plan_mutable(dgraphs.as_mut_slice(), &refs)?
		};
		drop(refs);
		if let (Some(d), Some(f)) = (cfg.fardec.as_ref(), far.as_mut()) {
			let (fs, cap) = far_set(npu, &mut bufs, d, usize::MAX)?;
			if cap != f.cap {
				return Err(format!(
					"{d}: block size differs from the far prefill sets'"
				));
			}
			if bufs[fs.mask[0]].len() != k * cap * 2 {
				return Err(format!("{d}: not built for the decode set's {k} position(s) (mkfar.sh with DEC=1 and S = {k})"));
			}
			let names: Vec<String> = (0..NL / 4)
				.flat_map(|i| {
					[
						format!("{d}/lean{}a.blob", 4 * i + 3),
						format!("{d}/lean{}b.blob", 4 * i + 3),
					]
				})
				.collect();
			let gs = load_blobs(npu, &names)?;
			let mut sh: BufMap = Default::default();
			let (mut pre, mut post) = (Vec::new(), Vec::new());
			for (gi, g) in gs.iter_mut().enumerate() {
				let l = 4 * (gi / 2) + 3;
				let mut set = Vec::new();
				let mut ko = 0;
				for a in g.args.clone() {
					let mut shared =
						|n: &str, bufs: &mut Vec<HostBuf<'static>>| -> Result<usize, String> {
							match sh.get(n) {
								Some(&b) => Ok(b),
								None => {
									let b = alloc(npu, bufs, a.bytes, None)?;
									sh.insert(n.to_string(), b);
									Ok(b)
								}
							}
						};
					let b = if a.is_output {
						ko += 1;
						match (gi % 2, ko) {
							// pre: qg, gate, knew, vnew; post: y
							(0, 1) => shared("qg", &mut bufs)?,
							(0, 2) => shared("gate", &mut bufs)?,
							(0, 3) => shared("knew", &mut bufs)?,
							(0, 4) => shared("vnew", &mut bufs)?,
							(1, 1) => resid[l],
							_ => return Err(format!("{d}: layer {l}: unexpected outputs")),
						}
					} else {
						match a.name.as_str() {
							"x" => resid[l - 1],
							"o" | "gate" | "cos" | "sin" => shared(&a.name, &mut bufs)?,
							nm => *weights.get(&(l, nm.to_string())).ok_or(format!(
								"{d}: layer {l}: input {nm} is not a prefill weight"
							))?,
						}
					};
					set.push(b);
				}
				let refs: Vec<&HostBuf<'_>> = set.iter().map(|&i| &bufs[i]).collect();
				let list = npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, refs)])?;
				if gi % 2 == 0 {
					pre.push(list)
				} else {
					post.push(list)
				}
			}
			let mut seg = Vec::new();
			for i in 0..if mlp.is_empty() { NL / 4 } else { 0 } {
				let refs: Vec<(usize, Vec<&HostBuf<'_>>)> = (0..3)
					.map(|j| (j, plan[4 * i + j].iter().map(|&b| &bufs[b]).collect()))
					.collect();
				seg.push(npu.batch_plan_mutable(&mut dgraphs[4 * i..4 * i + 3], &refs)?);
			}
			let head = {
				let refs: Vec<&HostBuf<'_>> = plan[NL].iter().map(|&b| &bufs[b]).collect();
				npu.batch_plan_mutable(&mut dgraphs[NL..NL + 1], &[(0, refs)])?
			};
			let g = |n: &str| sh.get(n).copied().ok_or(format!("{d}: no {n}"));
			// `QWEN_NPU_FARM=0`: one block per far job (the A/B)
			let fm = format!("{d}/farm.blob");
			let multi = if std::path::Path::new(&fm).exists()
				&& std::env::var("QWEN_NPU_FARM")
					.map(|v| v != "0")
					.unwrap_or(true)
			{
				Some(FarMulti::load(npu, &mut bufs, &fm)?)
			} else {
				None
			};
			f.dec = Some(FarDec {
				fs,
				pre,
				post,
				seg,
				head,
				multi,
				qg: g("qg")?,
				kv: [g("knew")?, g("vnew")?],
				o: g("o")?,
				cos: g("cos")?,
				sin: g("sin")?,
			});
		}
		let mtp = match &cfg.mtp {
			None => None,
			Some(m) => {
				if !wy || hn.is_none() {
					return Err(format!("mtp = {m}: needs a verify decode set and {m}/head.blob with the `hn` output"));
				}
				let mgraphs = load_blobs(
					npu,
					&[
						format!("{m}/mtp_s{k}.blob"),
						format!("{m}/mtp_s1.blob"),
						format!("{m}/mtphead.blob"),
					],
				)?;
				let mut mb: std::collections::HashMap<String, usize> = Default::default();
				let mut msets: Vec<Vec<usize>> = Vec::new();
				for (gi, g) in mgraphs.iter().enumerate() {
					let tag = ["m4", "m1", "mh"][gi];
					let onames: &[&str] = if gi == 2 {
						&["logits"]
					} else {
						&["h", "present_key", "present_value"]
					};
					let mut ko = 0;
					let mut set = Vec::new();
					for arg in &g.args {
						let key = if arg.is_output {
							ko += 1;
							format!("{tag}.{}", onames[ko - 1])
						} else {
							format!("{tag}.{}", arg.name)
						};
						let b = if !arg.is_output && arg.name.starts_with("w.") {
							bound_weight(npu, &mut bufs, &mut cw, m, &arg.name, arg.bytes)?
						} else {
							alloc(npu, &mut bufs, arg.bytes, None)?
						};
						mb.insert(key, b);
						set.push(b);
					}
					msets.push(set);
				}
				let pk = mgraphs[1]
					.args
					.iter()
					.position(|x| x.name == "past_key")
					.ok_or("mtp_s1: no past_key")?;
				let pv = mgraphs[1]
					.args
					.iter()
					.position(|x| x.name == "past_value")
					.ok_or("mtp_s1: no past_value")?;
				let mcap = bufs[*mb
					.get("m4.attention_mask")
					.ok_or("MTP catch-up graph: no attention_mask")?]
				.len() / 8 - k;
				let mvocab = bufs[mb["mh.logits"]].len() / 4;
				let m4_f32 = bufs[mb["m4.h"]].len() == k * W * 4;
				let m1_f32 = bufs[mb["m1.h"]].len() == W * 4;
				let mut lists = Vec::new();
				for (gi, g) in mgraphs.iter_mut().enumerate() {
					let refs: Vec<&HostBuf<'_>> = msets[gi].iter().map(|&i| &bufs[i]).collect();
					lists.push(npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, refs)])?);
				}
				let norm = final_norm(&format!("{m}/head.xml"), &format!("{m}/head.bin"))?;
				Some(Mtp {
					lists,
					mb,
					m1_past: (pk, pv),
					cap: mcap,
					vocab: mvocab,
					m4_f32,
					m1_f32,
					last: 0,
					count: 0,
					norm,
					saved: None,
				})
			}
		};
		let mut e = NpuEngine {
			npu,
			_lock: lock,
			bufs,
			table,
			psets,
			x0,
			maxcap,
			batch,
			parts,
			mlp,
			dresid: resid.clone(),
			mlp_gpu: None,
			mlp_gpu_new: None,
			mlp_gpu_retry: None,
			mlp_gpu_slow: 0,
			mlp_gpu_far_off: 0,
			mlp_gpu_far_said: false,
			mlp_gpu_base: [f64::INFINITY; 5],
			emb,
			pos_i,
			bias_i,
			logits_i,
			vocab,
			cap,
			deltas,
			kv,
			parity: 1, // reset() re-points every state argument
			t: 0,
			k,
			wy,
			hn,
			mtp,
			pending: None,
			last: None,
			mrow: None,
			presid_last: [presid[0][NL - 1], presid[1][NL - 1]],
			last_slot: 0,
			scan,
			scan_live: false,
			far,
			mshift: 0,
			mm: None,
			vis: None,
			vis_attn: None,
		};
		if let Some(d) = &cfg.vision {
			let t = std::time::Instant::now();
			e.vis = Some(crate::vis::NpuVision::load(npu, d)?);
			eprintln!(
				"npu engine: vision tower ({} graphs of {} patches) loaded in {:.1} s",
				e.vis.as_ref().unwrap().cfg.layers + 1,
				e.vis.as_ref().unwrap().cfg.s,
				t.elapsed().as_secs_f64()
			);
		}
		if e.far.is_some()
			&& !(std::is_x86_feature_detected!("avx2")
				&& std::is_x86_feature_detected!("f16c")
				&& std::is_x86_feature_detected!("fma"))
		{
			return Err("a far prefill set needs a CPU with AVX2, F16C and FMA".into());
		}
		if e.psets.iter().any(|p| p.scan_bufs[0].is_some()) && e.scan.is_none() {
			return Err("a prefill set has cut delta layers (lean{L}a / lean{L}b) without their uncut graphs (lean{L}) but no scan device (config: scan = intel)".into());
		}
		e.reset()?;
		eprintln!(
			"npu engine: prefill sets {:?} positions (windows {:?}), decode {k} position(s){}, window {cap}, loaded in {:.1} s",
			e.psets.iter().map(|p| p.s).collect::<Vec<_>>(),
			e.psets.iter().map(|p| p.cap).collect::<Vec<_>>(),
			match &e.mtp {
				Some(m) => format!(" + MTP drafter ({}-row draft head, window {})", m.vocab, m.cap),
				None => String::new(),
			},
			t0.elapsed().as_secs_f64()
		);
		let _ = e.npu;
		Ok(e)
	}

	pub fn vocab(&self) -> usize {
		self.vocab
	}

	pub fn context(&self) -> usize {
		if self.far.as_ref().is_some_and(|f| f.dec.is_some()) {
			Self::FAR_CTX
		} else {
			self.cap
		}
	}

	fn sl(&self, i: usize) -> &'static mut [u8] {
		let b = &self.bufs[i];
		// SAFETY: a live host allocation owned by the engine; touched by the
		// host only while no list runs (every run is waited for).
		unsafe { std::slice::from_raw_parts_mut(b.ptr() as *mut u8, b.len()) }
	}

	/// Forget everything fed: zero the decode states, close the KV ring.
	pub fn reset(&mut self) -> Result<(), String> {
		if let Some(f) = self.far.as_mut() {
			f.live = false;
		}
		let one = f16_bits(1.0).to_le_bytes();
		for d in &self.deltas {
			for b in d.conv.iter().chain(d.ssm.iter()) {
				self.sl(*b).fill(0);
			}
			if let Some(w) = &d.wy {
				for b in w.kb.iter().chain(w.db.iter()).chain(Some(&w.w)) {
					self.sl(*b).fill(0);
				}
				// alpha = 1, w = 0: the start state is sbase as is
				for c in self.sl(w.alpha).chunks_exact_mut(2) {
					c.copy_from_slice(&one);
				}
			}
		}
		let neg = f16_bits(-30000.0).to_le_bytes();
		for c in self.sl(self.bias_i).chunks_exact_mut(2) {
			c.copy_from_slice(&neg);
		}
		// parity 0: states read set 0
		if self.parity != 0 {
			self.flip(0)?;
		}
		self.t = 0;
		self.mshift = 0;
		self.pending = None;
		self.last = None;
		self.mrow = None;
		if let Some(m) = self.mtp.as_mut() {
			m.last = 0;
			m.count = 0;
			m.saved = None;
		}
		Ok(())
	}

	/// The RoPE position of text at sequence position `p` (after every
	/// image fed so far).
	fn rp(&self, p: usize) -> i64 {
		p as i64 - self.mshift as i64
	}

	/// The vision tower's configuration and its learned position grid
	/// (`[n_pos][embd]`), when the config names one (`vision = DIR`).
	pub fn vision(&self) -> Option<(&crate::vis::VisCfg, &[f32])> {
		self.vis.as_ref().map(|v| (&v.cfg, &v.pos_w[..]))
	}

	/// What runs the tower's attention ([`Self::encode_image`] needs one).
	pub fn set_vis_attn(&mut self, a: Box<dyn crate::vis::VisAttn>) {
		self.vis_attn = Some(a);
	}

	/// One image through the vision tower (`crate::vis::NpuVision::encode`):
	/// the embeddings of its tokens, `[n / merge^2][5120]`. Independent of
	/// the text state.
	pub fn encode_image(
		&mut self,
		patches: &[f32],
		pos: &[f32],
		hw: &[(u32, u32)],
	) -> Result<Vec<f32>, String> {
		let v = self
			.vis
			.as_mut()
			.ok_or("this engine has no vision tower (config: vision = DIR)")?;
		let a = self
			.vis_attn
			.as_deref_mut()
			.ok_or("the vision tower has no attention device")?;
		if v.cfg.proj != W {
			return Err(format!(
				"the vision tower projects to {} dims, the model has {W}",
				v.cfg.proj
			));
		}
		v.encode(patches, pos, hw, a)
	}

	/// Point the state inputs at set `p` and the state outputs at `p ^ 1`.
	fn flip(&mut self, p: usize) -> Result<(), String> {
		let b = &self.bufs;
		let upd: Vec<(usize, usize, &HostBuf<'_>)> = self
			.deltas
			.iter()
			.flat_map(|d| match &d.wy {
				None => vec![
					(d.g, d.conv_in, &b[d.conv[p]]),
					(d.g, d.ssm_in, &b[d.ssm[p]]),
					(d.g, d.conv_out, &b[d.conv[p ^ 1]]),
					(d.g, d.ssm_out, &b[d.ssm[p ^ 1]]),
				],
				Some(w) => vec![
					(d.g, d.ssm_in, &b[d.ssm[p]]),
					(d.g, d.ssm_out, &b[d.ssm[p ^ 1]]),
					(d.g, w.k_in, &b[w.kb[p]]),
					(d.g, w.k_out, &b[w.kb[p ^ 1]]),
					(d.g, w.d_in, &b[w.db[p]]),
					(d.g, w.d_out, &b[w.db[p ^ 1]]),
				],
			})
			.collect();
		if self.parts.is_empty() {
			self.batch.update(&upd)?;
		} else {
			for c in upd.chunk_by(|a, b| a.0 == b.0) {
				let one: Vec<(usize, usize, &HostBuf<'_>)> =
					c.iter().map(|&(_, i, b)| (0, i, b)).collect();
				self.parts[c[0].0].update(&one)?;
			}
		}
		self.parity = p;
		Ok(())
	}

	/// Positions a verify step takes when the engine drafts (a verify set
	/// with MTP), else 0.
	pub fn spec_k(&self) -> usize {
		if self.mtp.is_some() {
			self.k
		} else {
			0
		}
	}

	/// Commit a verify step left pending (every fed token accepted).
	fn settle(&mut self) -> Result<(), String> {
		if let Some(n) = self.pending.as_ref().map(|p| p.len()) {
			self.commit(n)?;
		}
		Ok(())
	}

	/// Feed `ids` (1..=K tokens: the next token, then drafts) at positions
	/// `t..` through the verify set, tentatively, and return each position's
	/// logits. [`NpuEngine::commit`] keeps a prefix of them; anything else
	/// that feeds first keeps them all.
	pub fn verify(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
		if !self.wy {
			return Err("verify: needs a verify decode set".into());
		}
		self.settle()?;
		if ids.is_empty() || ids.len() > self.k {
			return Err(format!("verify: {} tokens (1..={})", ids.len(), self.k));
		}
		// on the far blocks a step stays inside the open block: it takes
		// fewer tokens (and returns fewer rows) at a block's end
		let far = self.far_next(ids.len());
		let ids = if far {
			let kb = self.far.as_ref().unwrap().kb();
			&ids[..ids
				.len()
				.min(kb - self.t % kb)
				.min(Self::FAR_CTX.saturating_sub(self.t))]
		} else {
			ids
		};
		if ids.is_empty() || (!far && self.t + ids.len() > self.cap) {
			return Err(format!(
				"context full: {} + {} > {}",
				self.t,
				ids.len(),
				self.context()
			));
		}
		// the MTP consumes the last step's `hn` rows before they are overwritten
		self.catch_up(ids[0])?;
		let k = self.k;
		for j in 0..k {
			let id = ids.get(j).copied().unwrap_or(ids[0]);
			self.table
				.seek_read(
					&mut self.sl(self.emb)[j * W * 2..(j + 1) * W * 2],
					(id as usize * W * 2) as u64,
				)
				.map_err(|e| e.to_string())?;
		}
		let p = self.sl(self.pos_i);
		for r in 0..p.len() / 8 / k {
			for j in 0..k {
				let o = (r * k + j) * 8;
				p[o..o + 8].copy_from_slice(&self.rp(self.t + j).to_le_bytes());
			}
		}
		if far {
			self.run_far(ids.len())?;
		} else {
			self.dec_run(ids.len())?;
		}
		self.pending = Some(ids.to_vec());
		let lg = self.sl(self.logits_i);
		Ok((0..ids.len())
			.map(|j| {
				lg[j * self.vocab * 4..(j + 1) * self.vocab * 4]
					.chunks_exact(4)
					.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
					.collect()
			})
			.collect())
	}

	/// Keep the first `n` positions of the pending verify step: the
	/// deferred rollback's `alpha` / `w`, the conv windows, the accepted
	/// K / V rows into the ring.
	pub fn commit(&mut self, n: usize) -> Result<(), String> {
		let ids = self
			.pending
			.take()
			.ok_or("commit: no verify step pending")?;
		if n == 0 || n > ids.len() {
			return Err(format!("commit: {n} of {} positions", ids.len()));
		}
		let (k, t_old, j) = (self.k, self.t, n - 1);
		for d in &self.deltas {
			let w = d.wy.as_ref().unwrap();
			let gb = self.sl(w.gcum);
			let heads = self.sl(w.alpha).len() / 2;
			let f32g = gb.len() == heads * k * 4;
			let gv = |h: usize, m: usize| -> f32 {
				if f32g {
					let o = (h * k + m) * 4;
					f32::from_le_bytes([gb[o], gb[o + 1], gb[o + 2], gb[o + 3]])
				} else {
					let o = (h * k + m) * 2;
					f16_to_f32(u16::from_le_bytes([gb[o], gb[o + 1]]))
				}
			};
			let (al, ww) = (self.sl(w.alpha), self.sl(w.w));
			for h in 0..heads {
				let gj = gv(h, j);
				al[h * 2..h * 2 + 2].copy_from_slice(&f16_bits(gj.exp()).to_le_bytes());
				for m in 0..k {
					let v = if m <= j { (gj - gv(h, m)).exp() } else { 0.0 };
					ww[(h * k + m) * 2..(h * k + m) * 2 + 2]
						.copy_from_slice(&f16_bits(v).to_le_bytes());
				}
			}
			// conv: the 4 columns ending at the last accepted position
			let (f, cp) = (self.sl(w.conv_full), self.sl(d.conv[0]));
			let cc = f.len() / (cp.len() / 4);
			for c in 0..cp.len() / 8 {
				cp[c * 8..c * 8 + 8].copy_from_slice(&f[(c * cc + n) * 2..(c * cc + n + 4) * 2]);
			}
		}
		let cap = self.cap;
		// (on the far blocks the step's K / V rows are in the open block)
		let live = self.far.as_ref().is_some_and(|f| f.live);
		for &(q, p, is_v, _) in self.kv.iter().filter(|_| !live) {
			let (qs, ps) = (self.sl(q), self.sl(p));
			for jj in 0..n {
				let slot = (t_old + jj) % cap;
				for h in 0..qs.len() / (k * 512) {
					let src = &qs[(h * k + jj) * 512..(h * k + jj + 1) * 512];
					if is_v {
						for d in 0..256 {
							let o = ((h * 256 + d) * cap + slot) * 2;
							ps[o..o + 2].copy_from_slice(&src[d * 2..d * 2 + 2]);
						}
					} else {
						let o = (h * cap + slot) * 512;
						ps[o..o + 512].copy_from_slice(src);
					}
				}
			}
		}
		let b = self.sl(self.bias_i);
		for jj in (0..n).filter(|_| !live) {
			let slot = (t_old + jj) % cap;
			b[slot * 2..slot * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		}
		self.flip(self.parity ^ 1)?;
		self.t += n;
		if self.mtp.is_some() {
			self.last = Some((t_old, ids[..n].to_vec()));
		}
		Ok(())
	}

	/// MTP catch-up over the positions the last committed verify step added,
	/// `next` being the token at position `t` (not fed yet): position `p`
	/// pairs the main model's normed hidden at `p - 1` with token `p`. The
	/// hidden before the step's first position is the one a prefill saved.
	fn catch_up(&mut self, next: u32) -> Result<(), String> {
		let Some((t_old, ids)) = self.last.take() else {
			return Ok(());
		};
		let (k, t) = (self.k, self.t);
		let (hn, hn_f32) = self.hn.unwrap();
		let (saved, mlast) = {
			let m = self.mtp.as_mut().unwrap();
			(m.saved.take().filter(|s| s.0 + 1 == t_old), m.last)
		};
		// the first position whose hidden is known; before it a gap (the
		// window keeps its older rows)
		let lo = if saved.is_some() { t_old } else { t_old + 1 };
		let start = (mlast + 1).clamp(lo, t).max((t + 1).saturating_sub(k));
		let hnb = self.sl(hn);
		let rows: Vec<(u32, Vec<u8>)> = (start..=t)
			.map(|p| {
				let tok = if p < t { ids[p - t_old] } else { next };
				let h = if p == t_old {
					saved.as_ref().unwrap().1.clone()
				} else {
					h_row(hnb, p - 1 - t_old, hn_f32)
				};
				(tok, h)
			})
			.collect();
		self.mtp_run(start, &rows)
	}

	/// One MTP catch-up run: `rows` (token, normed hidden of the position
	/// before, f16) for positions `start..`, at most K of them.
	fn mtp_run(&mut self, start: usize, rows: &[(u32, Vec<u8>)]) -> Result<(), String> {
		let (k, c) = (self.k, rows.len());
		if c == 0 || c > k {
			return Err(format!("MTP catch-up of {c} rows"));
		}
		let m = self.mtp.as_mut().unwrap();
		let (mcap, mcount) = (m.cap, m.count);
		let mb = &m.mb;
		let bufs = &self.bufs;
		let mshift = self.mshift as i64;
		// SAFETY: live host allocations owned by the engine, no list running.
		let sl = |i: usize| -> &'static mut [u8] {
			unsafe { std::slice::from_raw_parts_mut(bufs[i].ptr() as *mut u8, bufs[i].len()) }
		};
		let (he, eb) = (sl(mb["m4.hidden"]), sl(mb["m4.emb"]));
		for j in 0..k {
			let (tok, h) = &rows[j.min(c - 1)];
			self.table
				.seek_read(
					&mut eb[j * W * 2..(j + 1) * W * 2],
					(*tok as usize * W * 2) as u64,
				)
				.map_err(|e| e.to_string())?;
			he[j * W * 2..(j + 1) * W * 2].copy_from_slice(h);
		}
		let mk = sl(mb["m4.attention_mask"]);
		mk.fill(0);
		for s in mcap.saturating_sub(mcount)..mcap + k {
			mk[s * 8..s * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
		}
		let pp = sl(mb["m4.position_ids"]);
		for j in 0..k {
			pp[j * 8..j * 8 + 8].copy_from_slice(&((start + j) as i64 - mshift).to_le_bytes());
		}
		m.lists[0].run()?;
		// the committed window: present rows [c, c + cap)
		for (q, p) in [
			("m4.present_key", "m4.past_key"),
			("m4.present_value", "m4.past_value"),
		] {
			let (qs, ps) = (sl(mb[q]), sl(mb[p]));
			for h in 0..ps.len() / (mcap * 512) {
				let s = (h * (mcap + k) + c) * 512;
				ps[h * mcap * 512..(h + 1) * mcap * 512].copy_from_slice(&qs[s..s + mcap * 512]);
			}
		}
		m.count = (m.count + c).min(mcap);
		m.last = start + c - 1;
		self.mrow = Some(c - 1);
		Ok(())
	}

	/// After a prefill of positions `..n` whose last chunk started at `p0`
	/// (its final residual rows still in the prefill's last-layer output):
	/// the MTP catch-up over the last `QWEN_NPU_MTP_CTX` (default 64)
	/// positions whose token is known, the normed hidden of position
	/// `n - 1` saved for the verify step that feeds token `n - 1`... `n`.
	fn mtp_prefill_context(
		&mut self,
		ids: &[u32],
		t0: usize,
		p0: usize,
		n: usize,
	) -> Result<(), String> {
		let Some(m) = self.mtp.as_ref() else {
			return Ok(());
		};
		let ctx: usize = quant::knob::parse("QWEN_NPU_MTP_CTX").unwrap_or(64);
		let res = self.sl(self.presid_last[self.last_slot]);
		let norm = &m.norm;
		let hidden = |q: usize| -> Vec<u8> {
			let r = &res[(q - p0) * W * 2..(q - p0 + 1) * W * 2];
			let x: Vec<f32> = r
				.chunks_exact(2)
				.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
				.collect();
			let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / W as f32 + 1e-6).sqrt();
			x.iter()
				.zip(norm)
				.flat_map(|(v, w)| f16_bits(v * inv * w).to_le_bytes())
				.collect()
		};
		let saved = (n - 1, hidden(n - 1));
		// positions p (token ids[p - t0]) with the hidden of p - 1 in the
		// chunk; text after the last image only (an image's tokens are not
		// what stands in the ids, and their positions are not the text's)
		let lo = (p0 + 1)
			.max(t0 + 1)
			.max(n.saturating_sub(ctx))
			.max(self.mm.as_ref().map_or(0, |m| m.text_from));
		let rows: Vec<(usize, (u32, Vec<u8>))> =
			(lo..n).map(|p| (p, (ids[p - t0], hidden(p - 1)))).collect();
		for c in rows.chunks(self.k) {
			let r: Vec<(u32, Vec<u8>)> = c.iter().map(|x| x.1.clone()).collect();
			self.mtp_run(c[0].0, &r)?;
		}
		self.mrow = None;
		self.mtp.as_mut().unwrap().saved = Some(saved);
		Ok(())
	}

	/// After [`NpuEngine::commit`]: up to K - 1 MTP drafts to follow `next`
	/// (the token sampled from the last accepted position). Empty when the
	/// engine has no drafter or nothing was committed since the last call.
	pub fn drafts(&mut self, next: u32) -> Result<Vec<u32>, String> {
		if self.mtp.is_none() {
			return Ok(Vec::new());
		}
		self.catch_up(next)?;
		let Some(row) = self.mrow.take() else {
			return Ok(Vec::new());
		};
		let (k, t) = (self.k, self.t);
		let mshift = self.mshift as i64;
		let m = self.mtp.as_mut().unwrap();
		let (mcap, mvocab) = (m.cap, m.vocab);
		let bufs = &self.bufs;
		// SAFETY: live host allocations owned by the engine, no list running.
		let sl = |i: usize| -> &'static mut [u8] {
			unsafe { std::slice::from_raw_parts_mut(bufs[i].ptr() as *mut u8, bufs[i].len()) }
		};
		let mb = &m.mb;
		let argmax = |lg: &[u8]| -> u32 {
			let mut best = (0usize, f32::MIN);
			for (i, c) in lg[..mvocab * 4].chunks_exact(4).enumerate() {
				let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
				if v > best.1 {
					best = (i, v);
				}
			}
			best.0 as u32
		};
		let mut out = Vec::new();
		let h4 = h_row(sl(mb["m4.h"]), row, m.m4_f32);
		sl(mb["mh.x"])[..W * 2].copy_from_slice(&h4);
		m.lists[2].run()?;
		out.push(argmax(sl(mb["mh.logits"])));
		sl(mb["m1.hidden"])[..W * 2].copy_from_slice(&h4);
		for i in 1..k - 1 {
			self.table
				.seek_read(sl(mb["m1.emb"]), (out[i - 1] as usize * W * 2) as u64)
				.map_err(|e| e.to_string())?;
			if i > 1 {
				let h1 = h_row(sl(mb["m1.h"]), 0, m.m1_f32);
				sl(mb["m1.hidden"])[..W * 2].copy_from_slice(&h1);
			}
			let (pk, pv) = if i == 1 {
				("m4.past_key", "m4.past_value")
			} else {
				("m1.past_key", "m1.past_value")
			};
			m.lists[1].update(&[
				(0, m.m1_past.0, &bufs[mb[pk]]),
				(0, m.m1_past.1, &bufs[mb[pv]]),
			])?;
			let valid = (m.count + i - 1).min(mcap);
			let mk = sl(mb["m1.attention_mask"]);
			mk.fill(0);
			for s in mcap - valid..=mcap {
				mk[s * 8..s * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
			}
			sl(mb["m1.position_ids"])[..8]
				.copy_from_slice(&((t + i) as i64 - mshift).to_le_bytes());
			m.lists[1].run()?;
			let h1 = h_row(sl(mb["m1.h"]), 0, m.m1_f32);
			sl(mb["mh.x"])[..W * 2].copy_from_slice(&h1);
			m.lists[2].run()?;
			out.push(argmax(sl(mb["mh.logits"])));
			if i + 1 < k - 1 {
				// the next chained draft sees this one's KV row
				for (q, p) in [
					("m1.present_key", "m1.past_key"),
					("m1.present_value", "m1.past_value"),
				] {
					let (qs, ps) = (sl(mb[q]), sl(mb[p]));
					for h in 0..ps.len() / (mcap * 512) {
						let s = (h * (mcap + 1) + 1) * 512;
						ps[h * mcap * 512..(h + 1) * mcap * 512]
							.copy_from_slice(&qs[s..s + mcap * 512]);
					}
				}
			}
		}
		// the drafts and the token before them must fit the window
		out.truncate(self.context().saturating_sub(t + 1));
		Ok(out)
	}

	/// Delta layer `di`'s recurrent state, unscaled, as `S[h][k][v]` (f32).
	fn ssm_state(&self, di: usize) -> Vec<f32> {
		let d = &self.deltas[di];
		let p = self.parity;
		let hd = 128;
		let f32s = |b: &[u8], n: usize| {
			let mut v = vec![0f32; n];
			crate::ir::f16s_to_f32(b, &mut v, 1.0);
			v
		};
		let st = self.sl(d.ssm[p]);
		let n = st.len() / 2;
		let nh = n / (hd * hd);
		let mut s = vec![0f32; n];
		match &d.wy {
			None => {
				// St[h][v][k]
				let t = f32s(st, n);
				for h in 0..nh {
					for kk in 0..hd {
						for v in 0..hd {
							s[(h * hd + kk) * hd + v] = t[(h * hd + v) * hd + kk];
						}
					}
				}
			}
			Some(w) => {
				// alpha * sbase + sum_m w_m kp_m^T dp_m
				let k = self.k;
				let (al, ww) = (f32s(self.sl(w.alpha), nh), f32s(self.sl(w.w), nh * k));
				let kp = f32s(self.sl(w.kb[p]), nh * k * hd);
				let dp = f32s(self.sl(w.db[p]), nh * k * hd);
				for h in 0..nh {
					let sh = &mut s[h * hd * hd..(h + 1) * hd * hd];
					crate::ir::f16s_to_f32(&st[h * hd * hd * 2..], sh, al[h]);
					for m in 0..k {
						let wm = ww[h * k + m];
						if wm == 0.0 {
							continue;
						}
						let dr = &dp[(h * k + m) * hd..(h * k + m + 1) * hd];
						for kk in 0..hd {
							let kv = wm * kp[(h * k + m) * hd + kk];
							for (x, &y) in sh[kk * hd..(kk + 1) * hd].iter_mut().zip(dr) {
								*x += kv * y;
							}
						}
					}
				}
			}
		}
		s
	}

	/// One decode step: feed `id` at position `t`, return the logits.
	fn step(&mut self, id: u32) -> Result<Vec<f32>, String> {
		if self.far_next(1) {
			return self.step_far(id);
		}
		if self.t >= self.cap {
			return Err(format!("context full ({} positions)", self.cap));
		}
		self.table
			.seek_read(self.sl(self.emb), (id as usize * W * 2) as u64)
			.map_err(|e| e.to_string())?;
		for c in self.sl(self.pos_i).chunks_exact_mut(8) {
			c.copy_from_slice(&self.rp(self.t).to_le_bytes());
		}
		self.dec_run(1)?;
		// the new K / V rows into the ring, open the slot
		let slot = self.t % self.cap;
		for &(q, p, is_v, _) in &self.kv {
			let (qs, ps) = (self.sl(q), self.sl(p));
			for h in 0..qs.len() / 512 {
				let src = &qs[h * 512..(h + 1) * 512];
				if is_v {
					for d in 0..256 {
						let o = ((h * 256 + d) * self.cap + slot) * 2;
						ps[o..o + 2].copy_from_slice(&src[d * 2..d * 2 + 2]);
					}
				} else {
					let o = (h * self.cap + slot) * 512;
					ps[o..o + 512].copy_from_slice(src);
				}
			}
		}
		self.sl(self.bias_i)[slot * 2..slot * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		self.flip(self.parity ^ 1)?;
		self.t += 1;
		let lg = self.sl(self.logits_i);
		Ok(lg[..self.vocab * 4]
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect())
	}

	/// The K / V window buffers `[pos][4][256]` of the attention layers.
	fn windows(&self) -> Vec<[usize; 2]> {
		self.psets
			.iter()
			.find_map(|p| {
				let v: Vec<[usize; 2]> = p.atts[0]
					.iter()
					.flatten()
					.filter(|a| a[0] != usize::MAX)
					.map(|a| [a[0], a[1]])
					.collect();
				(!v.is_empty()).then_some(v)
			})
			.unwrap_or_default()
	}

	/// A decode step on the far blocks (`FarDec`): position `t` may be past
	/// the decode set's window.
	fn step_far(&mut self, id: u32) -> Result<Vec<f32>, String> {
		if self.t >= Self::FAR_CTX {
			return Err(format!("context full ({} positions)", Self::FAR_CTX));
		}
		self.table
			.seek_read(self.sl(self.emb), (id as usize * W * 2) as u64)
			.map_err(|e| e.to_string())?;
		self.run_far(1)?;
		self.flip(self.parity ^ 1)?;
		self.t += 1;
		let lg = self.sl(self.logits_i);
		Ok(lg[..self.vocab * 4]
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect())
	}

	/// Whether the next decode run goes through the far blocks: they are the
	/// K / V store already, or `n` more positions pass the decode window.
	fn far_next(&self, n: usize) -> bool {
		// `QWEN_NPU_FARDEC=1` (a test): every step on the far blocks
		let force = std::env::var_os("QWEN_NPU_FARDEC").is_some();
		self.far
			.as_ref()
			.is_some_and(|f| f.dec.is_some() && (f.live || force || self.t + n > self.cap))
	}

	/// One run of the decode set on the far blocks, the embeddings of its
	/// K positions in place: the delta layers three at a time, each attention
	/// layer as the `pre` graph, the blocks, and `post`, then the head. The
	/// first `m` positions (`t..t + m`, all in one block) are real: their
	/// K / V rows go into the open block, where a verify step's rejected
	/// ones are simply overwritten later. The states are not flipped.
	/// Set what builds the MLPs' GPU share and build it. A failure (now or
	/// later: device lost, VRAM taken by another application) leaves the
	/// MLPs on the cores; a new share is built every `QWEN_GPU_RECOVER_SECS`
	/// (default 15, 0 = never) until one holds.
	pub fn set_mlp_gpu(&mut self, new: MlpGpuNew) {
		if self.mlp.is_empty() {
			eprintln!("npu engine: mlpgpu needs a cut decode set (layer{{L}}.mlp); ignored");
			return;
		}
		self.mlp_gpu_new = Some(new);
		self.mlp_gpu_retry = Some(std::time::Instant::now());
		self.mlp_gpu_heal();
	}

	fn mlp_gpu_heal(&mut self) {
		let (Some(new), Some(at)) = (&self.mlp_gpu_new, self.mlp_gpu_retry) else {
			return;
		};
		if self.mlp_gpu.is_some() || std::time::Instant::now() < at {
			return;
		}
		let t = std::time::Instant::now();
		match new(&self.mlp) {
			Ok(g) => {
				eprintln!("npu engine: MLP share on the GPU: {} of 17408 rows per layer ({} at {} positions), loaded in {:.1} s", g.rows(1), g.rows(g.max_s()), g.max_s(), t.elapsed().as_secs_f64());
				self.mlp_gpu = Some(g);
				self.mlp_gpu_retry = None;
				self.mlp_gpu_slow = 0;
				self.mlp_gpu_base = [f64::INFINITY; 5];
			}
			Err(e) => {
				let secs = mlp_gpu_secs();
				eprintln!(
					"npu engine: no MLP share on the GPU ({e}){}",
					if secs > 0 {
						format!("; next attempt in {secs} s")
					} else {
						String::new()
					}
				);
				self.mlp_gpu_retry = (secs > 0)
					.then(|| std::time::Instant::now() + std::time::Duration::from_secs(secs));
			}
		}
	}

	fn mlp_gpu_drop(&mut self, why: &str) {
		eprintln!("npu engine: MLP share on the GPU dropped ({why}); the cores take it");
		self.mlp_gpu = None;
		self.mlp_gpu_base = [f64::INFINITY; 5];
		self.mlp_gpu_slow = 0;
		let secs = mlp_gpu_secs();
		self.mlp_gpu_retry =
			(secs > 0).then(|| std::time::Instant::now() + std::time::Duration::from_secs(secs));
	}

	/// Layer `l`'s MLP on `m` rows of its residual; the seconds the cores
	/// waited for the GPU share. Drops a share that failed or stalled:
	/// when another application takes the VRAM, WDDM does not fail, it
	/// pages (seconds per layer, seen with `vramhog`).
	fn mlp_run(&mut self, l: usize, m: usize) -> f64 {
		// SAFETY: avx2 / f16c / fma checked at load; the layer's job is done
		match unsafe {
			self.mlp[l].run(
				l,
				raw(&self.bufs, self.dresid[l]),
				m,
				self.mlp_gpu.as_deref_mut(),
			)
		} {
			Err(e) => {
				self.mlp_gpu_drop(&e);
				0.0
			}
			Ok(w) if w > 0.25 => {
				self.mlp_gpu_drop(&format!(
					"a layer waited {:.0} ms for it: its memory is being paged?",
					w * 1e3
				));
				0.0
			}
			Ok(w) => w,
		}
	}

	/// After a pass over `m` rows that waited `wait` seconds for the GPU
	/// share in all: drop a share that stays slower than it was (part of
	/// its VRAM demoted to system memory; a new one is uploaded afresh).
	fn mlp_gpu_watch(&mut self, m: usize, wait: f64) {
		if self.mlp_gpu.is_none() {
			return;
		}
		let b = &mut self.mlp_gpu_base[m.min(4)];
		*b = b.min(wait);
		// (a pass of MLPs on the cores alone takes ~50 ms longer than a
		// balanced split: only a share later than that is worse than none)
		if wait > *b + 0.06 {
			self.mlp_gpu_slow += 1;
			if self.mlp_gpu_slow >= 8 {
				let why = format!(
					"8 passes waited {:.0} ms for it, {:.0} ms when it was loaded: VRAM demoted?",
					wait * 1e3,
					*b * 1e3
				);
				self.mlp_gpu_drop(&why);
			}
		} else {
			self.mlp_gpu_slow = 0;
		}
	}

	/// One pass of the decode set over `m` real positions.
	fn dec_run(&mut self, m: usize) -> Result<(), String> {
		if self.parts.is_empty() {
			self.batch.run()?;
			return Ok(());
		}
		let _w = quant::parallel::limit_width(mlp_threads());
		let (mut tn, mut tc, mut ta, mut tw) = (0.0, 0.0, 0.0, 0.0);
		self.mlp_gpu_heal();
		for l in 0..NL {
			let dt = self.parts[l].run()?;
			tn += dt;
			if l % 4 == 3 {
				ta += dt;
			}
			let t = std::time::Instant::now();
			tw += self.mlp_run(l, m);
			tc += t.elapsed().as_secs_f64();
		}
		self.mlp_gpu_watch(m, tw);
		tn += self.batch.run()?;
		if std::env::var("QWEN_NPU_MLP_TIMING").is_ok() {
			let sp = crate::mlp::take_split();
			eprintln!(
				"[MLP] npu {:.1} ms (attention layers {:.1}), mlp {:.1} ms ({m} rows: prep {:.1}, cores {:.1}, gpu wait {:.1})",
				tn * 1e3,
				ta * 1e3,
				tc * 1e3,
				sp[0],
				sp[1],
				sp[2]
			);
		}
		Ok(())
	}

	fn run_far(&mut self, m: usize) -> Result<(), String> {
		let (t, k) = (self.t, self.k);
		if !self.far.as_ref().unwrap().live {
			// the blocks take over from the decode ring (via the windows)
			if t > self.maxcap.min(self.cap) {
				return Err(format!(
					"far decode at {t}: the positions before it are not in the decode window"
				));
			}
			self.decode_to_prefill();
			let wins = self.windows();
			let far = self.far.as_mut().unwrap();
			far.ensure(self.npu, &mut self.bufs, t)?;
			for (ai, w) in wins.iter().enumerate() {
				far.copy(&self.bufs, ai, 0, t, *w, 0, true);
			}
			far.live = true;
		}
		self.far
			.as_mut()
			.unwrap()
			.ensure(self.npu, &mut self.bufs, t + m)?;
		let p = self.parity;
		let rp0 = self.rp(t);
		let bufs = &self.bufs;
		let far = self.far.as_mut().unwrap();
		let (kb, cap) = (far.kb(), far.cap);
		let dec = far.dec.as_mut().unwrap();
		// RoPE of positions t..; per position the open block's keys up to
		// its own (put there before the near graph runs)
		for (b, f) in [(dec.cos, 0usize), (dec.sin, 1)] {
			let tb = raw(bufs, b);
			for j in 0..k {
				for d in 0..64 {
					let ang =
						(rp0 + j as i64) as f64 * 1e7f64.powf(-((2 * (d % 32)) as f64) / 64.0);
					let v = if f == 0 { ang.cos() } else { ang.sin() };
					tb[(j * 64 + d) * 2..(j * 64 + d) * 2 + 2]
						.copy_from_slice(&f16_bits(v as f32).to_le_bytes());
				}
			}
		}
		let mk = raw(bufs, dec.fs.mask[0]);
		let (blk, row) = (t / kb, t % kb);
		for j in 0..k {
			for c in 0..cap {
				let o = (j * cap + c) * 2;
				mk[o..o + 2].copy_from_slice(
					&f16_bits(if c <= row + j.min(m - 1) {
						0.0
					} else {
						-30000.0
					})
					.to_le_bytes(),
				);
			}
		}
		let mut far_wait = 0.0;
		for i in 0..NL / 4 {
			let upd: Vec<(usize, usize, &HostBuf<'_>)> = self
				.deltas
				.iter()
				.filter(|d| d.g / 4 == i)
				.flat_map(|d| {
					let g = d.g - 4 * i;
					match &d.wy {
						None => vec![
							(g, d.conv_in, &bufs[d.conv[p]]),
							(g, d.ssm_in, &bufs[d.ssm[p]]),
							(g, d.conv_out, &bufs[d.conv[p ^ 1]]),
							(g, d.ssm_out, &bufs[d.ssm[p ^ 1]]),
						],
						Some(w) => vec![
							(g, d.ssm_in, &bufs[d.ssm[p]]),
							(g, d.ssm_out, &bufs[d.ssm[p ^ 1]]),
							(g, w.k_in, &bufs[w.kb[p]]),
							(g, w.k_out, &bufs[w.kb[p ^ 1]]),
							(g, w.d_in, &bufs[w.db[p]]),
							(g, w.d_out, &bufs[w.db[p ^ 1]]),
						],
					}
				})
				.collect();
			if self.parts.is_empty() {
				dec.seg[i].update(&upd)?;
				dec.seg[i].run()?;
			} else {
				// a cut set: its per-layer lists (pointed by `flip`), the MLPs between
				let _w = quant::parallel::limit_width(mlp_threads());
				for l in 4 * i..4 * i + 3 {
					self.parts[l].run()?;
					let gpu = if self.mlp_gpu_far_off == 0 {
						self.mlp_gpu.as_deref_mut()
					} else {
						None
					};
					// SAFETY: as `mlp_run` (not through it: `far` is borrowed)
					match unsafe { self.mlp[l].run(l, raw(bufs, self.dresid[l]), m, gpu) } {
						Ok(w) if w < 0.25 => far_wait += w,
						r => {
							eprintln!(
								"npu engine: MLP share on the GPU dropped ({}); the cores take it",
								r.err().unwrap_or(
									"a layer waited for it: its memory is being paged?".into()
								)
							);
							self.mlp_gpu = None;
							self.mlp_gpu_retry = Some(
								std::time::Instant::now()
									+ std::time::Duration::from_secs(mlp_gpu_secs().max(1)),
							);
						}
					}
				}
			}
			dec.pre[i].run()?;
			for w in 0..2 {
				raw(bufs, far.blocks[i][blk][w])[row * ROWB..(row + m) * ROWB]
					.copy_from_slice(&raw(bufs, dec.kv[w])[..m * ROWB]);
			}
			FAR_LAYER.store(i, std::sync::atomic::Ordering::Relaxed);
			let from = match dec.multi.as_mut() {
				Some(mu) => mu.run(
					&far.blocks[i],
					blk,
					bufs,
					dec.qg,
					&mut dec.fs.acc,
					&mut dec.fs.lacc,
					dec.fs.unnorm,
				)?,
				None => 0,
			};
			attend(
				&far.blocks[i],
				from,
				blk,
				&mut dec.fs,
				bufs,
				k,
				dec.qg,
				None,
				0,
				dec.o,
				&mut None,
				&mut std::time::Instant::now(),
				&mut [0.0; 5],
			)?;
			// (queueing this job, the next segment and the next pre job
			// together: +2.6 %, but the output varied from run to run)
			dec.post[i].run()?;
		}
		dec.head.run()?;
		// On the far blocks the GPU share is late (4 ms per layer against
		// 0.1: the card idles ~15 ms per attention layer and its 3 MLP
		// bursts in between run slowly). Later than the ~40 ms a pass of
		// these MLPs on the cores alone costs more: the cores take the next
		// 200 far passes, then one tries the share again.
		if self.mlp_gpu_far_off > 0 {
			self.mlp_gpu_far_off -= 1;
		} else if far_wait > 0.04 {
			if !self.mlp_gpu_far_said {
				eprintln!("npu engine: on the far blocks the MLP share on the GPU is late ({:.0} ms per pass); the cores take it there", far_wait * 1e3);
				self.mlp_gpu_far_said = true;
			}
			self.mlp_gpu_far_off = 200;
		}
		if !self.parts.is_empty() && std::env::var("QWEN_NPU_MLP_TIMING").is_ok() {
			let sp = crate::mlp::take_split();
			eprintln!("[MLP] far pass, {m} rows, 48 delta layers' MLPs: prep {:.1}, cores {:.1}, gpu wait {:.1} ms", sp[0], sp[1], sp[2]);
		}
		Ok(())
	}

	/// The chunks `(set, start, positions)` that prefill positions `t0..n`;
	/// an error when a chunk would start past every set's window. Each chunk
	/// takes, among the sets whose window reaches its start, the one with
	/// the fewest positions when the rest fits two of its chunks, else the
	/// one with the most; ties go to the smaller window (cheaper attention).
	fn plan(&self, t0: usize, n: usize) -> Result<Vec<(usize, usize, usize)>, String> {
		let mut plan = Vec::new();
		let mut p0 = t0;
		while p0 < n {
			let r = n - p0;
			// with a far set, a windowed set takes only chunks whose keys all
			// fit the window buffers (the far blocks are filled from them);
			// `QWEN_NPU_FAR_MIN=N` (a test): the far set from position N on
			let is_far = |i: usize| self.far.as_ref().is_some_and(|f| f.set(i).is_some());
			let fmin: usize = std::env::var("QWEN_NPU_FAR_MIN")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(usize::MAX);
			let ok: Vec<usize> = (0..self.psets.len())
				.filter(|&i| {
					let ps = &self.psets[i];
					is_far(i)
						|| (p0 <= ps.cap
							&& (self.far.is_none()
								|| (p0 < fmin && p0 + r.min(ps.s) <= self.maxcap)))
				})
				.collect();
			let lo = ok
				.iter()
				.copied()
				.min_by_key(|&i| (self.psets[i].s, self.psets[i].cap));
			let hi = ok
				.iter()
				.copied()
				.max_by_key(|&i| (self.psets[i].s, std::cmp::Reverse(self.psets[i].cap)));
			let (Some(lo), Some(hi)) = (lo, hi) else {
				let caps: Vec<usize> = self.psets.iter().map(|p| p.cap).collect();
				return Err(format!(
					"prompt of {n} tokens: chunk at {p0} past every prefill window {caps:?}"
				));
			};
			let si = if r <= 2 * self.psets[lo].s { lo } else { hi };
			let real = r.min(self.psets[si].s);
			plan.push((si, p0, real));
			p0 += real;
		}
		Ok(plan)
	}

	/// Positions a far set's prefill reaches (the model's context).
	const FAR_CTX: usize = 262144;
	/// Last part of a snapshot whose K / V parts are far-block rows.
	const FAR_SNAP: &'static [u8] = b"far-blocks";

	/// Estimated NPU time of a prefill chunk of `s` positions, and of one
	/// decode step, in ms (measured: 64 positions ~1.1 s, 256
	/// ~3.4 s, a decode step ~0.38 s), for `feed`'s choice between them.
	fn chunk_ms(s: usize) -> f64 {
		330.0 + 12.0 * s as f64
	}
	const STEP_MS: f64 = 380.0;
	/// A verify step (K = 4, deferred rollback: ~455 ms NPU).
	const VERIFY_MS: f64 = 470.0;

	/// The decode-layout state (positions `0..t`) into the prefill set's
	/// buffers, so a prefill can continue from it: the inverse of the
	/// hand-over at the end of `prefill`.
	/// Move the delta layers' recurrent states to the scan device
	/// (`to_scan`: from the prefill buffers at parity `par`, or zero) or back
	/// (f16 x vmul, the layout the full graphs and the hand-over read).
	fn scan_sync(&mut self, to_scan: bool, par: usize, zero: bool) -> Result<(), String> {
		let Some(sc) = self.scan.as_mut() else {
			return Err(
				"a set has cut delta layers but no scan device (config: scan = intel)".into(),
			);
		};
		// delta layer di = l - l / 4: its prefill state buffer (f16 x vmul)
		let bufs: Vec<(&'static mut [u8], f32)> = (0..NL)
			.filter(|l| l % 4 != 3)
			.map(|l| {
				let dl = self.psets[0].dls[l].as_ref().unwrap();
				let b = &self.bufs[dl.sb[par]];
				// SAFETY: as `sl`: the engine's host buffer, no job running
				let buf = unsafe { std::slice::from_raw_parts_mut(b.ptr() as *mut u8, b.len()) };
				(buf, if l == 0 { 16.0f32 } else { 256.0 })
			})
			.collect();
		let len = bufs[0].0.len() / 2;
		if to_scan {
			sc.set_states(bufs.len(), len, &mut |di, dst| {
				if !zero {
					crate::ir::f16s_to_f32(bufs[di].0, dst, 1.0 / bufs[di].1);
				}
				!zero
			})?;
		} else {
			let mut bufs = bufs;
			sc.get_states(bufs.len(), &mut |di, st| {
				let (buf, vmul) = &mut bufs[di];
				crate::ir::f32s_to_f16(st, buf, *vmul);
			})?;
		}
		self.scan_live = to_scan;
		Ok(())
	}

	fn decode_to_prefill(&mut self) {
		let t = self.t;
		let p = self.parity;
		let mut di = 0;
		for l in 0..NL {
			if let Some(dl) = &self.psets[0].dls[l] {
				// ssm: S[h][k][v] x vmul
				let mul = if l == 0 { 16.0 } else { 256.0 };
				let s = self.ssm_state(di);
				crate::ir::f32s_to_f16(&s, self.sl(dl.sb[0]), mul);
				let d = &self.deltas[di];
				di += 1;
				// conv: window [C][4], columns 1..3 -> rows (oldest -> newest) [3][C]
				let (cw, conv) = (self.sl(d.conv[p]), self.sl(dl.cb[0]));
				let ch = conv.len() / 2 / 3;
				for c in 0..ch {
					for r in 0..3 {
						let o = (c * 4 + 1 + r) * 2;
						conv[(r * ch + c) * 2..(r * ch + c) * 2 + 2].copy_from_slice(&cw[o..o + 2]);
					}
				}
			} else if let Some(at) =
				self.psets[0].atts[0][l].filter(|_| !self.far.as_ref().is_some_and(|f| f.live))
			{
				for &(_, past, is_v, _) in self.kv.iter().filter(|x| x.3 == l) {
					let dst = self.sl(if is_v { at[1] } else { at[0] });
					let ps = self.sl(past);
					for pos in 0..t {
						for h in 0..4 {
							let r = &mut dst[(pos * 4 + h) * 512..(pos * 4 + h + 1) * 512];
							if is_v {
								for d in 0..256 {
									let o = ((h * 256 + d) * self.cap + pos) * 2;
									r[d * 2..d * 2 + 2].copy_from_slice(&ps[o..o + 2]);
								}
							} else {
								let o = (h * self.cap + pos) * 512;
								r.copy_from_slice(&ps[o..o + 512]);
							}
						}
					}
				}
			}
		}
	}

	/// A chunk's inputs in slot `slot` of set `si`: the embeddings of its
	/// `real` tokens (`ids[p0 - t0..]`), the attention mask and RoPE tables
	/// for positions `p0..`, `valid`, `keep`.
	fn prep_chunk(
		&mut self,
		si: usize,
		slot: usize,
		ids: &[u32],
		t0: usize,
		p0: usize,
		real: usize,
	) -> Result<(), String> {
		let inv: Vec<f64> = (0..32)
			.map(|i| 1e7f64.powf(-((2 * i) as f64) / 64.0))
			.collect();
		let f16b = |v: f32| f16_bits(v).to_le_bytes();
		let s = self.psets[si].s;
		let cap = self.psets[si].cap;
		{
			let x = self.sl(self.x0[slot]);
			x[..s * W * 2].fill(0);
			for j in 0..real {
				self.table
					.seek_read(
						&mut x[j * W * 2..(j + 1) * W * 2],
						(ids[p0 - t0 + j] as usize * W * 2) as u64,
					)
					.map_err(|e| e.to_string())?;
			}
			// an image's tokens: the tower's rows
			for (start, rows) in self.mm.iter().flat_map(|m| m.emb.iter()) {
				let (a, b) = (
					(*start).max(p0),
					(start + rows.len() / (W * 2)).min(p0 + real),
				);
				if a < b {
					x[(a - p0) * W * 2..(b - p0) * W * 2]
						.copy_from_slice(&rows[(a - start) * W * 2..(b - start) * W * 2]);
				}
			}
			let shared = &self.psets[si].shared[slot];
			if let Some(&m) = shared.get("mask") {
				let mk = self.sl(m);
				let row = cap + s;
				let (zero, neg) = (f16b(0.0), f16b(-30000.0));
				for i in 0..s {
					for k in 0..row {
						let open = if k < cap { k < p0 } else { k - cap <= i };
						let o = (i * row + k) * 2;
						mk[o..o + 2].copy_from_slice(if open { &zero } else { &neg });
					}
				}
			}
			if let Some((f, g)) = self.far.as_ref().and_then(|f| f.set(si).map(|g| (f, g))) {
				// near graph: the chunk's keys (causal), then the open block
				let mk = self.sl(f.sets[g].mask[slot]);
				let (row, fill) = (s + f.cap, p0 % f.kb());
				let (zero, neg) = (f16b(0.0), f16b(-30000.0));
				for i in 0..s {
					for k in 0..row {
						let open = if k < s { k <= i } else { k - s < fill };
						let o = (i * row + k) * 2;
						mk[o..o + 2].copy_from_slice(if open { &zero } else { &neg });
					}
				}
			}
			// (t, h, w) of the chunk's positions: equal for text
			let pos3: Vec<[i64; 3]> = (p0..p0 + s)
				.map(|q| {
					match self
						.mm
						.as_ref()
						.and_then(|m| m.pos.get(q.wrapping_sub(m.t0)))
					{
						Some(&x) if q >= t0 => x,
						_ => [self.rp(q); 3],
					}
				})
				.collect();
			for (nm, f) in [("cos", 0usize), ("sin", 1)] {
				if let Some(&b) = shared.get(nm) {
					let t = self.sl(b);
					for i in 0..s {
						for d in 0..64 {
							let ang = pos3[i][mrope_axis(d % 32)] as f64 * inv[d % 32];
							let v = if f == 0 { ang.cos() } else { ang.sin() };
							t[(i * 64 + d) * 2..(i * 64 + d) * 2 + 2]
								.copy_from_slice(&f16b(v as f32));
						}
					}
				}
			}
			if let Some(&b) = shared.get("valid") {
				let t = self.sl(b);
				for i in 0..s {
					t[i * 2..i * 2 + 2].copy_from_slice(&f16b(if i < real { 1.0 } else { 0.0 }));
				}
			}
			if let Some(&b) = shared.get("keep") {
				let t = self.sl(b);
				for j in 0..3 {
					t[j * 4..j * 4 + 4].copy_from_slice(&((real + j) as i32).to_le_bytes());
				}
			}
		}
		Ok(())
	}

	/// Prefill `ids` after the `t` positions fed so far and move the result
	/// into the decode states (positions 0..t + ids.len()).
	fn prefill(&mut self, ids: &[u32]) -> Result<(), String> {
		let t0 = self.t;
		let n = t0 + ids.len();
		let plan = self.plan(t0, n)?;
		let bar = quant::progress::Prefill::start(ids.len());
		let timing = std::env::var_os("QWEN_NPU_TIMING").is_some();
		let tm = std::time::Instant::now();
		if t0 == 0 {
			// zero start state (set 0 of every delta layer)
			for d in self.psets[0].dls.iter().flatten() {
				self.sl(d.cb[0]).fill(0);
				self.sl(d.sb[0]).fill(0);
			}
		} else {
			self.decode_to_prefill();
			if timing {
				eprintln!(
					"  [npu] decode -> prefill state at {t0}: {:.0} ms",
					tm.elapsed().as_secs_f64() * 1e3
				);
			}
		}
		let mut par = 0usize;
		let maxcap = self.maxcap;
		// `QWEN_NPU_PAIR=0`: one chunk at a time (the A/B)
		let pairing = std::env::var("QWEN_NPU_PAIR")
			.map(|v| v != "0")
			.unwrap_or(true);
		let mut ci = 0;
		while ci < plan.len() {
			let (si, p0, real) = plan[ci];
			// two consecutive chunks of one set run as a pair, layer by layer
			// (pairing across window sets in the larger one measured worse:
			// with an odd count the unpaired chunk lands in a costlier window)
			let pair = pairing && ci + 1 < plan.len() && plan[ci + 1].0 == si;
			let chunks: Vec<(usize, usize)> = if pair {
				vec![(p0, real), (plan[ci + 1].1, plan[ci + 1].2)]
			} else {
				vec![(p0, real)]
			};
			ci += chunks.len();
			quant::progress::chunk(p0 - t0, chunks.iter().map(|c| c.1).sum());
			let s = self.psets[si].s;
			let cap = self.psets[si].cap;
			let tc = std::time::Instant::now();
			for (slot, &(q0, qr)) in chunks.iter().enumerate() {
				self.prep_chunk(si, slot, ids, t0, q0, qr)?;
			}
			let t_prep = tc.elapsed().as_secs_f64();
			let cut = self.psets[si].scan_bufs[0].is_some();
			if cut != self.scan_live {
				// the recurrent state moves to (or back from) the scan device
				let ts = std::time::Instant::now();
				self.scan_sync(cut, par, t0 == 0 && p0 == t0)?;
				if timing {
					eprintln!(
						"  [npu] scan sync at {p0} ({}): {:.0} ms",
						if cut { "to the scan device" } else { "back" },
						ts.elapsed().as_secs_f64() * 1e3
					);
				}
			}
			let far_g = self.far.as_ref().and_then(|f| f.set(si));
			let is_far = far_g.is_some();
			if is_far {
				let tf = std::time::Instant::now();
				let (nb0, live0) = self.far.as_ref().map_or((0, false), |f| {
					(f.blocks.first().map_or(0, |b| b.len()), f.live)
				});
				let wins: Vec<[usize; 2]> = self
					.psets
					.iter()
					.find_map(|p| {
						let v: Vec<[usize; 2]> = p.atts[0]
							.iter()
							.flatten()
							.filter(|a| a[0] != usize::MAX)
							.map(|a| [a[0], a[1]])
							.collect();
						(!v.is_empty()).then_some(v)
					})
					.unwrap_or_default();
				let far = self.far.as_mut().unwrap();
				let end = chunks.last().map(|c| c.0 + c.1).unwrap();
				far.ensure(self.npu, &mut self.bufs, end)?;
				if !far.live {
					// the blocks take over from the window buffers
					if p0 > maxcap || (p0 > 0 && wins.is_empty()) {
						return Err(format!("far prefill at {p0}: the positions before it are not in the window buffers"));
					}
					for (ai, w) in wins.iter().enumerate() {
						far.copy(&self.bufs, ai, 0, p0, *w, 0, true);
					}
					far.live = true;
				}
				far.t_att = 0.0;
				far.gpu[1] = 0.0;
				far.gpu[4] = 0.0;
				let nb1 = far.blocks.first().map_or(0, |b| b.len());
				if timing && (nb1 != nb0 || !live0) {
					eprintln!(
						"  [npu] far blocks at {p0}: {nb0} -> {nb1} per layer{}: {:.0} ms",
						if live0 { "" } else { ", window rows copied in" },
						tf.elapsed().as_secs_f64() * 1e3
					);
				}
			}
			let sbufs: [Option<[&'static mut [u8]; 5]>; 2] =
				[0, 1].map(|k| self.psets[si].scan_bufs[k].map(|b| b.map(|i| self.sl(i))));
			let mut t_scan = 0.0;
			let bufs = &self.bufs;
			let ps = &mut self.psets[si];
			ps.next.iter_mut().for_each(|x| *x = 0);
			let nj = ps.jobs[0].len();
			let mut ji = 0;
			let mut poked = std::time::Instant::now();
			while ji < nj {
				quant::progress::part(ji, nj);
				let l = ps.jobs[0][ji].l;
				let cut_l = ps.jobs[0][ji].pre;
				let at_l = ps.atts[0][l].is_some();
				let mut scan =
					|slot: usize, sc: &mut Option<Box<dyn Scan>>| -> Result<(), String> {
						let ts = std::time::Instant::now();
						let [qn, kn, v, gb, o] = sbufs[slot]
							.as_ref()
							.map(|b| b.each_ref().map(|x| &**x as *const [u8]))
							.unwrap();
						let sc = sc.as_mut().ok_or(
							"a set has cut delta layers but no scan device (config: scan = intel)",
						)?;
						// SAFETY: the engine's own buffers; the NPU does not touch
						// this slot's scan buffers until its post job is submitted.
						unsafe {
							sc.scan(
								l - l / 4,
								chunks[slot].1,
								&*qn,
								&*kn,
								&*v,
								&*gb,
								&mut *(o as *mut [u8]),
								16.0,
							)?
						};
						t_scan += ts.elapsed().as_secs_f64();
						Ok(())
					};
				if at_l && cut_l {
					// far set: pre, blocks, merge, post, per chunk; a chunk's
					// K / V rows join the blocks before the next one's turn
					let far = self
						.far
						.as_mut()
						.ok_or("cut attention layers without near.blob / far.blob")?;
					for (slot, &(q0, qr)) in chunks.iter().enumerate() {
						let a = ps.submit(bufs, slot, ji, par ^ slot)?;
						ps.wait(a)?;
						let at = ps.atts[slot][l].unwrap();
						let sh = &ps.shared[slot];
						far.attend(
							bufs,
							far_g.unwrap(),
							l / 4,
							q0,
							s,
							sh["att.qg"],
							at[2],
							at[3],
							slot,
							sh["att.o"],
							&mut self.scan,
							&mut poked,
						)?;
						ps.submit(bufs, slot, ji + 1, par ^ slot)?;
						far.copy(bufs, l / 4, q0, qr, [at[2], at[3]], q0, true);
					}
				} else if !pair {
					let a = ps.submit(bufs, 0, ji, par)?;
					if cut_l {
						ps.wait(a)?;
						scan(0, &mut self.scan)?;
						ps.submit(bufs, 0, ji + 1, par)?;
					}
				} else if at_l {
					// B's attention reads A's new K / V rows from the window
					let a = ps.submit(bufs, 0, ji, par)?;
					ps.wait(a)?;
					let at = ps.atts[0][l].unwrap();
					let m = chunks[0].1.min(maxcap.saturating_sub(chunks[0].0));
					for (src, dst) in [(at[2], at[0]), (at[3], at[1])] {
						// SAFETY: distinct host buffers of the engine, not in
						// use by a running job (A's attention job is done)
						let (s_, d_) = unsafe {
							(
								std::slice::from_raw_parts(
									bufs[src].ptr() as *const u8,
									bufs[src].len(),
								),
								std::slice::from_raw_parts_mut(
									bufs[dst].ptr() as *mut u8,
									bufs[dst].len(),
								),
							)
						};
						d_[chunks[0].0 * ROWB..(chunks[0].0 + m) * ROWB]
							.copy_from_slice(&s_[..m * ROWB]);
					}
					ps.submit(bufs, 1, ji, par ^ 1)?;
				} else if cut_l {
					let a = ps.submit(bufs, 0, ji, par)?;
					let b = ps.submit(bufs, 1, ji, par ^ 1)?;
					ps.wait(a)?;
					scan(0, &mut self.scan)?;
					ps.submit(bufs, 0, ji + 1, par)?;
					ps.wait(b)?;
					scan(1, &mut self.scan)?;
					ps.submit(bufs, 1, ji + 1, par ^ 1)?;
				} else {
					ps.submit(bufs, 0, ji, par)?;
					ps.submit(bufs, 1, ji, par ^ 1)?;
				}
				ji += if cut_l { 2 } else { 1 };
			}
			for (gi, v) in ps.lists.iter_mut().enumerate() {
				for (r, b) in v.iter_mut().enumerate() {
					if ps.busy[gi][r] {
						b.wait()?;
						ps.busy[gi][r] = false;
					}
				}
			}
			let t_npu = tc.elapsed().as_secs_f64();
			quant::progress::part(1, 1);
			for (slot, &(q0, qr)) in chunks.iter().enumerate() {
				// `QWEN_NPU_DUMP_RESID=FILE` (diagnostic): append the chunk's
				// final residual rows (f16 [real][5120]; `lean_chain.py --resid`)
				if let Some(f) = std::env::var_os("QWEN_NPU_DUMP_RESID") {
					use std::io::Write;
					let rows = &self.sl(self.presid_last[slot])[..qr * W * 2];
					std::fs::OpenOptions::new()
						.create(true)
						.append(true)
						.open(&f)
						.and_then(|mut h| h.write_all(rows))
						.map_err(|e| e.to_string())?;
				}
				// the chunk's K / V rows into the windows (a pair's first
				// chunk did that before the second one's attention)
				if slot + 1 == chunks.len() {
					let atts: Vec<[usize; 4]> = self.psets[si].atts[slot]
						.iter()
						.flatten()
						.filter(|a| a[0] != usize::MAX)
						.copied()
						.collect();
					for at in atts {
						for (src, dst) in [(at[2], at[0]), (at[3], at[1])] {
							let m = qr.min(maxcap.saturating_sub(q0));
							let (s_, d_) = (self.sl(src), self.sl(dst));
							d_[q0 * ROWB..(q0 + m) * ROWB].copy_from_slice(&s_[..m * ROWB]);
						}
					}
				}
				self.last_slot = slot;
				par ^= 1;
			}
			if timing {
				let e = tc.elapsed().as_secs_f64();
				eprintln!(
					"  [npu] chunk{} {:?} (s {s}, cap {}): prep {:.1} ms, npu {:.1} ms (scan {:.1}, far attention {:.1}{}), kv {:.1} ms",
					if pair { "s" } else { "" },
					chunks.iter().map(|c| format!("{}+{}", c.0, c.1)).collect::<Vec<_>>(),
					if is_far { "far".to_string() } else { cap.to_string() },
					t_prep * 1e3,
					(t_npu - t_prep) * 1e3,
					t_scan * 1e3,
					self.far.as_ref().filter(|_| is_far).map_or(0.0, |f| f.t_att * 1e3),
					self.far.as_ref().filter(|f| is_far && f.gpu[1] + f.gpu[4] > 0.0).map_or(String::new(), |f| format!(": {} blocks on the scan device, {:.1} ms each, {} on the cores, {:.1} ms each, NPU {:.1}", f.gpu[1], f.gpu[0], f.gpu[4], f.gpu[3], f.gpu[2])),
					(e - t_npu) * 1e3
				);
			}
		}
		let t_chunks = tm.elapsed().as_secs_f64();
		if self.scan_live {
			// the hand-over reads the states from the prefill buffers
			self.scan_sync(false, par, false)?;
		}
		let t_sync = tm.elapsed().as_secs_f64();
		// far blocks: back into the window buffers the hand-over reads, or
		// (past the decode window, with `fardec`) they stay the K / V store
		let keep = self
			.far
			.as_ref()
			.is_some_and(|f| f.live && f.dec.is_some() && n >= self.cap);
		if self.far.as_ref().is_some_and(|f| f.live) && !keep {
			let wins = self.windows();
			let far = self.far.as_mut().unwrap();
			for (ai, w) in wins.iter().enumerate() {
				far.copy(&self.bufs, ai, 0, n.min(maxcap), *w, 0, false);
			}
			far.live = false;
		}
		let t_run = tm.elapsed().as_secs_f64();
		// hand over: prefill set 0's buffers (shared by every set) -> decode;
		// the MTP keeps its window (the prefilled positions are a gap in it)
		let mkeep = self.mtp.as_ref().map(|m| (m.last, m.count));
		let mshift = self.mshift;
		self.reset()?;
		self.mshift = mshift;
		if let Some(f) = self.far.as_mut() {
			f.live = keep;
		}
		if let (Some(m), Some((l, c))) = (self.mtp.as_mut(), mkeep) {
			(m.last, m.count) = (l, c);
		}
		let t_reset = tm.elapsed().as_secs_f64();
		if let Some(&(_, p0, _)) = plan.last() {
			self.mtp_prefill_context(ids, t0, p0, n)?;
		}
		let t_mtp = tm.elapsed().as_secs_f64();
		// one job per delta layer and per K / V window, run in parallel
		let p = self.parity;
		let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
		let mut di = 0;
		for l in 0..NL {
			if let Some(dl) = &self.psets[0].dls[l] {
				let (conv, ssm) = (self.sl(dl.cb[par]), self.sl(dl.sb[par]));
				let d = &self.deltas[di];
				di += 1;
				let (st, cw) = (self.sl(d.ssm[p]), self.sl(d.conv[p]));
				let transposed = d.wy.is_none();
				// ssm: S[h][k][v] x vmul -> unscaled
				let inv = 1.0 / if l == 0 { 16.0 } else { 256.0 };
				jobs.push(Box::new(move || {
					handover_delta(ssm, conv, st, cw, inv, transposed)
				}));
			} else if let Some(at) = self.psets[0].atts[0][l] {
				for &(_, past, is_v, _) in self.kv.iter().filter(|x| x.3 == l) {
					let src: &'static [u8] = self.sl(if is_v { at[1] } else { at[0] });
					let ps = self.sl(past);
					let (cap, hi) = (self.cap, n.min(self.cap));
					jobs.push(Box::new(move || handover_kv(src, ps, is_v, cap, t0, hi)));
				}
			}
		}
		run_parallel(jobs);
		let b = self.sl(self.bias_i);
		for pos in 0..n.min(self.cap) {
			b[pos * 2..pos * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		}
		self.t = n;
		if timing {
			let e = tm.elapsed().as_secs_f64();
			eprintln!(
				"  [npu] prefill {t0}..{n}: {} chunk(s) {:?}, {:.0} ms = chunks {:.0} + scan sync back {:.0} + far copy-back {:.0} + reset {:.0} + MTP context {:.0} + hand-over {:.0}; {:.1} tok/s",
				plan.len(),
				plan.iter().map(|p| self.psets[p.0].s).collect::<Vec<_>>(),
				e * 1e3,
				t_chunks * 1e3,
				(t_sync - t_chunks) * 1e3,
				(t_run - t_sync) * 1e3,
				(t_reset - t_run) * 1e3,
				(t_mtp - t_reset) * 1e3,
				(e - t_mtp) * 1e3,
				ids.len() as f64 / e
			);
		}
		bar.done();
		Ok(())
	}

	/// Feed `ids` after everything fed since the last reset; the last
	/// position's logits (`vocab()` values). All but the last token go
	/// through the prefill graphs when there are enough of them, also after
	/// earlier calls (the state is moved back to the prefill layout first).
	pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
		if ids.is_empty() {
			return Err("forward: no tokens".into());
		}
		if self.far.is_none() && self.t + ids.len() > self.cap {
			return Err(format!(
				"context full: {} + {} > {}",
				self.t,
				ids.len(),
				self.cap
			));
		}
		let n = ids.len();
		self.feed(&ids[..n - 1])?;
		if self.t >= self.cap && self.far.as_ref().is_none_or(|f| f.dec.is_none()) {
			return Err(format!("prefilled {} positions; decoding past the decode set's {}-position window needs `fardec` in the config", self.t, self.cap));
		}
		if self.wy {
			let l = self.verify(&ids[n - 1..])?;
			self.commit(1)?;
			Ok(l.into_iter().next().unwrap())
		} else {
			self.step(ids[n - 1])
		}
	}

	/// Feed `ids` without computing logits: through the prefill graphs, or
	/// as decode steps when that is estimated cheaper (a few tokens) or the
	/// prefill windows do not reach.
	pub fn feed(&mut self, ids: &[u32]) -> Result<(), String> {
		// with a far set the prefill reaches past the decode window
		if ids.is_empty() {
			return Ok(());
		}
		let lim = if self.far.is_some() {
			Self::FAR_CTX
		} else {
			self.cap
		};
		if self.t + ids.len() > lim {
			return Err(format!("context full: {} + {} > {lim}", self.t, ids.len()));
		}
		self.settle()?;
		// `QWEN_NPU_NOPREFILL=1` (diagnostic): every token through decode
		// or verify steps, to tell prefill error from decode error
		let noprefill = std::env::var_os("QWEN_NPU_NOPREFILL").is_some();
		if let Some(plan) = self
			.plan(self.t, self.t + ids.len())
			.ok()
			.filter(|_| !noprefill)
		{
			let est: f64 = plan.iter().map(|c| Self::chunk_ms(self.psets[c.0].s)).sum();
			let steps = if self.wy {
				ids.len().div_ceil(self.k) as f64 * Self::VERIFY_MS
			} else {
				ids.len() as f64 * Self::STEP_MS
			};
			if est < steps || self.t + ids.len() > self.cap {
				return self.prefill(ids);
			}
		}
		if std::env::var_os("QWEN_NPU_TIMING").is_some() {
			eprintln!(
				"  [npu] feed {}..{}: {} tokens as {} steps",
				self.t,
				self.t + ids.len(),
				ids.len(),
				if self.wy { "verify" } else { "decode" }
			);
		}
		if self.wy {
			let mut rest = ids;
			while !rest.is_empty() {
				// (a step on the far blocks may take fewer)
				let n = self.verify(&rest[..rest.len().min(self.k)])?.len();
				self.commit(n)?;
				rest = &rest[n..];
			}
			return Ok(());
		}
		for &id in ids {
			self.step(id)?;
		}
		Ok(())
	}

	/// [`Self::feed`] for tokens with images: each span's run of placeholder
	/// tokens is fed as the tower's embeddings, at the image's MRoPE
	/// coordinates (one `t` for the image, `h` / `w` its token's row and
	/// column; the text after it resumes one past the largest coordinate,
	/// `model::prompt`). Everything goes through the prefill graphs: only
	/// they take three coordinates per position.
	pub fn feed_mm(&mut self, ids: &[u32], spans: &[ImageSpan<'_>]) -> Result<(), String> {
		if spans.is_empty() {
			return self.feed(ids);
		}
		let lim = if self.far.is_some() {
			Self::FAR_CTX
		} else {
			self.cap
		};
		if self.t + ids.len() > lim {
			return Err(format!("context full: {} + {} > {lim}", self.t, ids.len()));
		}
		self.settle()?;
		let t0 = self.t;
		let geom: Vec<(usize, usize, usize)> = spans
			.iter()
			.map(|s| (s.start, s.grid_w, s.grid_h))
			.collect();
		let (pos, next) =
			mm_positions(self.rp(t0), ids.len(), &geom).map_err(|e| format!("feed_mm: {e}"))?;
		if let Some(sp) = spans
			.iter()
			.find(|s| s.emb.len() != s.grid_w * s.grid_h * W)
		{
			return Err(format!(
				"feed_mm: {} values for an image of {} x {} tokens",
				sp.emb.len(),
				sp.grid_w,
				sp.grid_h
			));
		}
		let emb = spans
			.iter()
			.map(|sp| {
				(
					t0 + sp.start,
					sp.emb
						.iter()
						.flat_map(|&v| f16_bits(v).to_le_bytes())
						.collect(),
				)
			})
			.collect();
		let at = spans.last().map_or(0, |s| s.start + s.grid_w * s.grid_h);
		// the shift of the text after the last image (what the prefill's
		// hand-over and every later step use)
		let shift = (t0 + ids.len()) as i64 - next;
		self.mm = Some(Mm {
			t0,
			pos,
			emb,
			text_from: t0 + at,
		});
		let old = std::mem::replace(&mut self.mshift, shift as usize);
		let r = self.prefill(ids);
		self.mm = None;
		if r.is_err() {
			self.mshift = old;
		}
		r
	}

	/// Positions fed since the last reset.
	pub fn pos(&self) -> usize {
		self.t
	}

	/// The buffers that hold a delta layer's state at parity `p`: the conv
	/// window and the ssm state (transposed in an s = 1 set), plus a verify
	/// set's rollback factors (`kp`, `dp`, `alpha`, `w`).
	fn state_bufs(d: &DDelta, p: usize) -> Vec<usize> {
		match &d.wy {
			None => vec![d.conv[p], d.ssm[p]],
			Some(w) => vec![d.conv[0], d.ssm[p], w.kb[p], w.db[p], w.alpha, w.w],
		}
	}

	/// A copy of the state after the `pos()` positions fed so far, compact
	/// (only the KV rows in use): per delta layer its state buffers
	/// (`state_bufs`), then per attention window its rows `0..t`
	/// (K: `[head][t][256]`, V: `[head][256][t]`). A pending verify step is
	/// committed first. While the far blocks are the K / V store (past the
	/// decode window), the attention parts are instead the blocks' rows
	/// `0..t` per layer (K, then V, both `[t][4][256]`) and a last part
	/// `FAR_SNAP` marks the layout. With an MTP drafter its state follows
	/// (`mtp_snap`), so a restored state drafts as the original would.
	pub fn snapshot(&mut self) -> Result<Vec<Vec<u8>>, String> {
		self.snapshot_rows().map(|r| r.0)
	}

	/// [`NpuEngine::snapshot`] with, per part, its KV row layout when it is
	/// one: `(streams, rows)`, `streams` blocks of `rows` positions each
	/// (`None`: any other part). The prefix cache pages such parts and
	/// shares the pages between checkpoints.
	#[allow(clippy::type_complexity)]
	pub fn snapshot_rows(&mut self) -> Result<(Vec<Vec<u8>>, Vec<Option<(usize, usize)>>), String> {
		let ts = std::time::Instant::now();
		let r = self.snapshot_parts();
		if std::env::var_os("QWEN_NPU_TIMING").is_some() {
			if let Ok((p, _)) = &r {
				eprintln!(
					"  [npu] snapshot at {}: {} MB in {:.0} ms",
					self.t,
					p.iter().map(|x| x.len()).sum::<usize>() >> 20,
					ts.elapsed().as_secs_f64() * 1e3
				);
			}
		}
		r
	}

	#[allow(clippy::type_complexity)]
	fn snapshot_parts(&mut self) -> Result<(Vec<Vec<u8>>, Vec<Option<(usize, usize)>>), String> {
		self.settle()?;
		let (t, p, cap) = (self.t, self.parity, self.cap);
		let mut parts = Vec::new();
		let mut rows_of: Vec<Option<(usize, usize)>> = Vec::new();
		for d in &self.deltas {
			for b in Self::state_bufs(d, p) {
				parts.push(self.sl(b).to_vec());
			}
		}
		if let Some(far) = self.far.as_ref().filter(|f| f.live) {
			let kb = far.kb();
			for bl in &far.blocks {
				for w in 0..2 {
					let mut v = Vec::with_capacity(t * ROWB);
					for (b, blk) in bl.iter().enumerate().take(t.div_ceil(kb)) {
						v.extend_from_slice(
							&raw(&self.bufs, blk[w])[..(t - b * kb).min(kb) * ROWB],
						);
					}
					parts.push(v);
				}
			}
			// (every part so far: the delta states, then far rows)
			rows_of.resize(parts.len(), None);
			let far_parts = far.blocks.len() * 2;
			let n = rows_of.len();
			for r in &mut rows_of[n - far_parts..] {
				*r = Some((1, t));
			}
			parts.push(Self::FAR_SNAP.to_vec());
			self.mtp_snap(&mut parts);
			self.mrope_snap(&mut parts);
			rows_of.resize(parts.len(), None);
			return Ok((parts, rows_of));
		}
		rows_of.resize(parts.len(), None);
		for &(_, past, is_v, _) in &self.kv {
			let ps = self.sl(past);
			let rows = if is_v {
				ps.len() / 2 / cap
			} else {
				ps.len() / 512 / cap
			};
			let w = if is_v { 2 } else { 512 };
			let mut v = Vec::with_capacity(rows * t * w);
			for r in 0..rows {
				v.extend_from_slice(&ps[r * cap * w..(r * cap + t) * w]);
			}
			parts.push(v);
			rows_of.push(Some((rows, t)));
		}
		self.mtp_snap(&mut parts);
		self.mrope_snap(&mut parts);
		rows_of.resize(parts.len(), None);
		Ok((parts, rows_of))
	}

	/// Last part of a snapshot of a state with images (`mrope_snap`).
	const MROPE_SNAP: &'static [u8] = b"mrope-shift";

	/// After images: the text's position shift (u64) and `MROPE_SNAP` as a
	/// snapshot's last two parts. A snapshot of text alone has neither.
	fn mrope_snap(&self, parts: &mut Vec<Vec<u8>>) {
		if self.mshift != 0 {
			parts.push((self.mshift as u64).to_le_bytes().to_vec());
			parts.push(Self::MROPE_SNAP.to_vec());
		}
	}

	/// Last part of a snapshot that ends in the MTP's state (`mtp_snap`).
	const MTP_SNAP: &'static [u8] = b"mtp-window";

	/// The MTP drafter's state as a snapshot's last parts: counters (u64:
	/// rows in the window, its last position, the position of the saved
	/// hidden, the start of the verify step not caught up over, then that
	/// step's ids; `u64::MAX` = none), the window's rows in use (K, V), the
	/// saved hidden, the step's `hn` rows, `MTP_SNAP`.
	fn mtp_snap(&self, parts: &mut Vec<Vec<u8>>) {
		let Some(m) = self.mtp.as_ref() else { return };
		let mut meta = vec![
			m.count as u64,
			m.last as u64,
			m.saved.as_ref().map_or(u64::MAX, |s| s.0 as u64),
			self.last.as_ref().map_or(u64::MAX, |l| l.0 as u64),
		];
		meta.extend(self.last.iter().flat_map(|l| l.1.iter().map(|&x| x as u64)));
		parts.push(meta.iter().flat_map(|x| x.to_le_bytes()).collect());
		for name in ["m4.past_key", "m4.past_value"] {
			let ps = self.sl(m.mb[name]);
			parts.push(
				ps.chunks_exact(m.cap * 512)
					.flat_map(|h| &h[(m.cap - m.count) * 512..])
					.copied()
					.collect(),
			);
		}
		parts.push(m.saved.as_ref().map_or(Vec::new(), |s| s.1.clone()));
		parts.push(match (&self.last, self.hn) {
			(Some(_), Some((hn, _))) => self.sl(hn).to_vec(),
			_ => Vec::new(),
		});
		parts.push(Self::MTP_SNAP.to_vec());
	}

	/// The MTP's state from a snapshot's `mtp_snap` parts (after a reset).
	fn mtp_restore(&mut self, pos: usize, parts: &[&[u8]]) -> Result<(), String> {
		let Some(m) = self.mtp.as_ref() else {
			return Ok(());
		};
		let bad = || "restore: MTP state size mismatch".to_string();
		let meta: Vec<u64> = parts[0]
			.chunks_exact(8)
			.map(|c| u64::from_le_bytes(c.try_into().unwrap()))
			.collect();
		if meta.len() < 4 || meta[0] as usize > m.cap {
			return Err(bad());
		}
		let (count, mcap) = (meta[0] as usize, m.cap);
		for (name, src) in [("m4.past_key", &parts[1]), ("m4.past_value", &parts[2])] {
			let ps = self.sl(m.mb[name]);
			if src.len() != ps.len() / mcap * count {
				return Err(bad());
			}
			for (h, c) in src.chunks_exact((count * 512).max(1)).enumerate() {
				ps[(h * mcap + mcap - count) * 512..(h + 1) * mcap * 512].copy_from_slice(c);
			}
		}
		let saved = if meta[2] == u64::MAX {
			None
		} else if parts[3].len() == W * 2 {
			Some((meta[2] as usize, parts[3].to_vec()))
		} else {
			return Err(bad());
		};
		let last = if meta[3] == u64::MAX {
			None
		} else {
			let (t_old, ids) = (meta[3] as usize, &meta[4..]);
			let hnb = self.sl(self.hn.ok_or_else(bad)?.0);
			if ids.is_empty()
				|| ids.len() > self.k
				|| t_old + ids.len() != pos
				|| parts[4].len() != hnb.len()
			{
				return Err(bad());
			}
			hnb.copy_from_slice(parts[4]);
			Some((t_old, ids.iter().map(|&x| x as u32).collect()))
		};
		let m = self.mtp.as_mut().unwrap();
		(m.count, m.last, m.saved) = (count, meta[1] as usize, saved);
		self.last = last;
		Ok(())
	}

	/// Replace the state with a `snapshot()` taken after `pos` positions.
	/// A snapshot without the MTP's state (an older one, or of an engine
	/// without a drafter) leaves the MTP window empty.
	pub fn restore(&mut self, pos: usize, parts: &[&[u8]]) -> Result<(), String> {
		let ts = std::time::Instant::now();
		let r = self.restore_parts(pos, parts);
		if r.is_ok() && std::env::var_os("QWEN_NPU_TIMING").is_some() {
			eprintln!(
				"  [npu] restore at {pos}{}: {:.0} ms",
				if self.far.as_ref().is_some_and(|f| f.live) {
					" (far blocks)"
				} else {
					""
				},
				ts.elapsed().as_secs_f64() * 1e3
			);
		}
		r
	}

	fn restore_parts(&mut self, pos: usize, parts: &[&[u8]]) -> Result<(), String> {
		let (parts, mshift) = match parts {
			[rest @ .., sh, m] if m[..] == Self::MROPE_SNAP[..] && sh.len() == 8 => (
				rest,
				u64::from_le_bytes(sh[..].try_into().unwrap()) as usize,
			),
			_ => (parts, 0),
		};
		if mshift > pos {
			return Err("restore: bad position shift".into());
		}
		let (parts, mtp) = match parts.split_last() {
			Some((m, rest)) if m[..] == Self::MTP_SNAP[..] && rest.len() >= 5 => {
				rest.split_at(rest.len() - 5)
			}
			_ => (parts, &parts[..0]),
		};
		let nd: usize = self
			.deltas
			.iter()
			.map(|d| Self::state_bufs(d, 0).len())
			.sum();
		let in_blocks = parts.last().is_some_and(|m| m[..] == Self::FAR_SNAP[..]);
		let fits = if in_blocks {
			self.far
				.as_ref()
				.is_some_and(|f| f.dec.is_some() && parts.len() == nd + 2 * f.blocks.len() + 1)
				&& pos <= Self::FAR_CTX
		} else {
			pos <= self.cap && parts.len() == nd + self.kv.len()
		};
		if !fits {
			return Err(format!(
				"restore: snapshot of {} parts at {pos} does not fit this engine",
				parts.len()
			));
		}
		self.reset()?;
		self.mshift = mshift;
		if !mtp.is_empty() {
			self.mtp_restore(pos, mtp)?;
		}
		let (p, cap) = (self.parity, self.cap);
		let mut it = parts.iter();
		for d in &self.deltas {
			for b in Self::state_bufs(d, p) {
				let (src, dst) = (it.next().unwrap(), self.sl(b));
				if src.len() != dst.len() {
					return Err("restore: delta state size mismatch".into());
				}
				dst.copy_from_slice(src);
			}
		}
		if in_blocks {
			let far = self.far.as_mut().unwrap();
			far.ensure(self.npu, &mut self.bufs, pos)?;
			let kb = far.kb();
			for bl in &far.blocks {
				for w in 0..2 {
					let src = it.next().unwrap();
					if src.len() != pos * ROWB {
						return Err("restore: KV size mismatch".into());
					}
					for (b, c) in src.chunks(kb * ROWB).enumerate() {
						raw(&self.bufs, bl[b][w])[..c.len()].copy_from_slice(c);
					}
				}
			}
			far.live = true;
			self.t = pos;
			return Ok(());
		}
		for &(_, past, is_v, _) in &self.kv {
			let (src, ps) = (it.next().unwrap(), self.sl(past));
			let w = if is_v { 2 } else { 512 };
			let rows = ps.len() / w / cap;
			if src.len() != rows * pos * w {
				return Err("restore: KV size mismatch".into());
			}
			for r in 0..rows {
				ps[r * cap * w..(r * cap + pos) * w]
					.copy_from_slice(&src[r * pos * w..(r + 1) * pos * w]);
			}
		}
		let b = self.sl(self.bias_i);
		for q in 0..pos {
			b[q * 2..q * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		}
		self.t = pos;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::Config;

	/// The values `model::tests` pins for the GGUF path's
	/// `build_image_prompt`: one text token, a 2 x 2 image, then text.
	#[test]
	fn image_positions_follow_the_models_mrope() {
		let (p, next) = super::mm_positions(100, 6, &[(1, 2, 2)]).unwrap();
		assert_eq!(p[0], [100; 3]);
		assert_eq!(
			&p[1..5],
			&[
				[101, 101, 101],
				[101, 101, 102],
				[101, 102, 101],
				[101, 102, 102]
			]
		);
		assert_eq!(p[5], [103; 3]);
		assert_eq!(next, 104);
		// a 3 x 2 image advances the text by 3, whatever its 6 tokens; two
		// images in a row
		let (p, next) = super::mm_positions(0, 12, &[(2, 3, 2), (9, 1, 2)]).unwrap();
		assert_eq!(p[2], [2, 2, 2]);
		assert_eq!(p[7], [2, 3, 4]);
		assert_eq!(p[8], [5; 3]);
		assert_eq!(&p[9..11], &[[6, 6, 6], [6, 7, 6]]);
		assert_eq!((p[11], next), ([8; 3], 9));
		// text alone; a run past the end; overlapping runs
		assert_eq!(
			super::mm_positions(7, 2, &[]).unwrap(),
			(vec![[7; 3], [8; 3]], 9)
		);
		assert!(super::mm_positions(0, 4, &[(1, 2, 2)]).is_err());
		assert!(super::mm_positions(0, 9, &[(0, 2, 2), (3, 2, 2)]).is_err());
		// every rotary frequency reads one of the three coordinates
		assert_eq!(
			(0..32).map(super::mrope_axis).filter(|&a| a == 2).count(),
			10
		);
	}

	/// A relative path of a `.npu` file is taken from beside the file when
	/// it exists there; device names and other paths stay as written.
	#[test]
	fn npu_file_paths_resolve_beside_the_file() {
		let d = std::env::temp_dir().join(format!("qwen-rs-npucfg-{}", std::process::id()));
		std::fs::create_dir_all(d.join("sets/g1c")).unwrap();
		std::fs::write(d.join("emb_table.f16"), b"").unwrap();
		let f = d.join("model.npu");
		std::fs::write(&f, "# a comment\nprefill = sets/g1c\ndecode = elsewhere/dec\nemb = emb_table.f16\ntokenizer = tokenizer\nscan = intel\n").unwrap();
		let c = Config::parse(f.to_str().unwrap()).unwrap();
		assert_eq!(std::path::Path::new(&c.prefill), d.join("sets/g1c"));
		assert_eq!(std::path::Path::new(&c.emb), d.join("emb_table.f16"));
		assert_eq!(c.decode, "elsewhere/dec");
		assert_eq!(c.tokenizer, "tokenizer");
		assert_eq!(c.scan.as_deref(), Some("intel"));
		std::fs::remove_dir_all(&d).unwrap();
	}
}
