//! Activation sparsity (TEAL-style, arXiv 2408.14690): **measurement only**.
//!
//! Decode is DRAM-bound, so a matmul input that is exactly zero is a weight
//! column that need not be read. This module zeroes the smallest-magnitude
//! fraction of each matmul input in `forward_step`, per token, so the accuracy
//! cost can be measured (`cli --kld-eval`) *before* any weight re-layout that
//! would turn the zeros into skipped bytes. Nothing here reads fewer bytes yet.
//!
//! Four sites, each with its own fraction:
//! * `a`: attention input (`attn_norm` output → Wq/Wk/Wv, Wqkv/gate/beta/alpha)
//! * `o`: attention output (→ Wo, ssm_out)
//! * `f`: FFN input (`post_norm` output → FfnUp/FfnGate)
//! * `d`: FfnDown input (`silu(gate) * up`)
//!
//! Off (all zero) unless `QWEN_SPARSE` or [`set_sparsity`] says otherwise.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Site {
	AttnIn = 0,
	AttnOut = 1,
	FfnIn = 2,
	FfnDown = 3,
}

const N_SITES: usize = 4;
const NAMES: [&str; N_SITES] = ["a", "o", "f", "d"];

/// Fraction of each site's input zeroed, as `f32` bits.
static FRAC: [AtomicU32; N_SITES] = [const { AtomicU32::new(0) }; N_SITES];
/// Per site: vectors seen, elements, elements zeroed, 32-blocks / 256-blocks
/// that ended up all zero (what a block-granular skip could use).
static STATS: [[AtomicU64; 5]; N_SITES] = [const { [const { AtomicU64::new(0) }; 5] }; N_SITES];

/// Sparsity fractions per site, parsed from `a=0.4,o=0.4,f=0.4,d=0.5` (also
/// `all=0.4`; unnamed sites stay 0).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseCfg(pub [f32; N_SITES]);

impl SparseCfg {
	pub fn parse(s: &str) -> Result<SparseCfg, String> {
		let mut c = [0.0f32; N_SITES];
		for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
			let (k, v) = part
				.split_once('=')
				.ok_or_else(|| format!("sparsity spec `{part}`: want key=fraction"))?;
			let v: f32 = v
				.trim()
				.parse()
				.map_err(|_| format!("sparsity spec `{part}`: bad fraction"))?;
			if !(0.0..1.0).contains(&v) {
				return Err(format!(
					"sparsity spec `{part}`: fraction must be in [0, 1)"
				));
			}
			match k.trim() {
				"all" => c = [v; N_SITES],
				k => {
					let i = NAMES
						.iter()
						.position(|n| *n == k)
						.ok_or_else(|| format!("sparsity site `{k}`: one of a, o, f, d, all"))?;
					c[i] = v;
				}
			}
		}
		Ok(SparseCfg(c))
	}
}

impl std::fmt::Display for SparseCfg {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let parts: Vec<String> = NAMES
			.iter()
			.zip(self.0)
			.map(|(n, v)| format!("{n}={v:.2}"))
			.collect();
		write!(f, "{}", parts.join(","))
	}
}

fn init_from_env() {
	static INIT: OnceLock<()> = OnceLock::new();
	INIT.get_or_init(|| {
		if let Ok(s) = std::env::var("QWEN_SPARSE") {
			match SparseCfg::parse(&s) {
				Ok(c) => store(c),
				Err(e) => eprintln!("QWEN_SPARSE ignored: {e}"),
			}
		}
	});
}

fn store(c: SparseCfg) {
	for (a, v) in FRAC.iter().zip(c.0) {
		a.store(v.to_bits(), Ordering::Relaxed);
	}
}

/// Replace the active configuration (overrides `QWEN_SPARSE`) and reset the stats.
pub fn set_sparsity(c: SparseCfg) {
	init_from_env();
	store(c);
	for s in &STATS {
		for a in s {
			a.store(0, Ordering::Relaxed);
		}
	}
}

/// Zero the smallest-magnitude `frac` of `v` for this site (no-op at 0).
#[inline]
pub(crate) fn sparsify(site: Site, v: &mut [f32]) {
	init_from_env();
	let frac = f32::from_bits(FRAC[site as usize].load(Ordering::Relaxed));
	if frac <= 0.0 || v.is_empty() {
		return;
	}
	let k = ((v.len() as f32) * frac) as usize;
	if k == 0 {
		return;
	}
	let mut mag: Vec<f32> = v.iter().map(|x| x.abs()).collect();
	let (_, &mut thr, _) = mag.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
	// Ties at the threshold can push the count past k; only matters for exact
	// repeated magnitudes, which do not occur in practice.
	let mut zeroed = 0u64;
	for x in v.iter_mut() {
		if x.abs() <= thr {
			*x = 0.0;
			zeroed += 1;
		}
	}
	if site == Site::FfnDown {
		if let Some(f) = FREQ.get() {
			let mut f = f.lock().unwrap();
			let l = LAYER.load(Ordering::Relaxed) as usize;
			let n = v.len();
			if f.len() < (l + 1) * n {
				f.resize((l + 1) * n, 0);
			}
			for (i, x) in v.iter().enumerate() {
				f[l * n + i] += (*x != 0.0) as u32;
			}
		}
	}
	let blocks = |w: usize| v.chunks(w).filter(|b| b.iter().all(|x| *x == 0.0)).count() as u64;
	let st = &STATS[site as usize];
	st[0].fetch_add(1, Ordering::Relaxed);
	st[1].fetch_add(v.len() as u64, Ordering::Relaxed);
	st[2].fetch_add(zeroed, Ordering::Relaxed);
	st[3].fetch_add(blocks(32), Ordering::Relaxed);
	st[4].fetch_add(blocks(256), Ordering::Relaxed);
}

/// One line per active site: achieved element sparsity and the share of
/// 32- / 256-wide blocks that are entirely zero.
pub fn sparsity_report() -> String {
	let mut out = Vec::new();
	for (i, st) in STATS.iter().enumerate() {
		let n = st[1].load(Ordering::Relaxed);
		if n == 0 {
			continue;
		}
		let z = st[2].load(Ordering::Relaxed) as f64 / n as f64;
		let b32 = st[3].load(Ordering::Relaxed) as f64 / (n as f64 / 32.0);
		let b256 = st[4].load(Ordering::Relaxed) as f64 / (n as f64 / 256.0);
		out.push(format!(
			"{}: zero {:.1}% blk32 {:.2}% blk256 {:.3}%",
			NAMES[i],
			z * 100.0,
			b32 * 100.0,
			b256 * 100.0
		));
	}
	out.join(" | ")
}

/// `QWEN_SPARSE_FREQ=FILE`: count how often each FfnDown input neuron
/// survives, per layer, and write the counts (`u32`, layer-major) to FILE at
/// [`sparsity_report`]. Measures how skewed activation frequency is: whether
/// a hot/cold column order would make the skipped bytes contiguous.
static FREQ: OnceLock<std::sync::Mutex<Vec<u32>>> = OnceLock::new();
static LAYER: AtomicU32 = AtomicU32::new(0);

pub(crate) fn set_layer(il: usize) {
	LAYER.store(il as u32, Ordering::Relaxed);
}

pub fn freq_enable() {
	let _ = FREQ.set(std::sync::Mutex::new(Vec::new()));
}

pub fn freq_dump(path: &str) {
	if let Some(f) = FREQ.get() {
		let mut f = f.lock().unwrap();
		let b: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
		let _ = std::fs::write(path, b);
		f.clear();
	}
}
