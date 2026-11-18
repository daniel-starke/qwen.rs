//! From the calibrated weights to what the engine loads: every graph as an
//! IR ([`graphs`]), compiled to an NPU blob ([`compile`]: our build of the
//! NPU compiler for the layer graphs, the driver's compiler for the heads
//! and the MTP block), and the graph sets with their `.npu` files
//! ([`assemble`]).
//!
//! `OUT/graphs/<group>/<name>.{xml,bin,blob,swz}`; groups: `p256` / `p64`
//! (prefill, 256 / 64 positions), `d4` / `d4m` (the verify set with the MLP
//! on the NPU / cut off), `f4` (the verify set's far attention), `mtp`,
//! `vis` (the vision tower, when the checkpoint has one: [`crate::vis`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use crate::dec;
use crate::g::Graph;
use crate::lean::{self, Cut, Part};
use crate::refm::{is_attn, NL};
use crate::st::Checkpoint;
use crate::weights;

/// Positions of the verify set (the sampled token and three drafts).
pub const K: usize = 4;
/// Key slots of an attention window / block (the last two of a far block
/// are not keys).
pub const CAP: usize = 1024;
/// Window of the verify set's and the MTP block's attention.
pub const DCAP: usize = 4096;
/// lm_head rows of the draft head.
pub const DRAFT_VOCAB: usize = 32768;
/// Key of the weight swizzling (compiler patch 0009, `NPU_RS_SWZKEY`).
const SWZ_KEY: usize = npu::ze::SWZ_KEY;

fn mkdir(d: &Path) -> Result<(), String> {
	std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))
}

/// How a graph is compiled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum How {
	/// Our compiler with the prefill options (grouped MatMul, the CMX view
	/// folding) and the given SHAVE kernel directory (`""`: none).
	Lean(&'static str),
	/// Our compiler, weight swizzling only (decode attention layers: the
	/// grouped-MatMul options make them slower).
	Plain,
	/// The driver's compiler (heads, MTP: needs the NPU).
	Driver,
}

struct Job {
	/// `<group>/<name>` under `OUT/graphs`.
	stem: String,
	how: How,
	/// weight inputs may be read swizzled
	swz: bool,
}

/// The kernel directories: (name, kernel source, kernel it replaces, defines).
fn kernels(
	s: usize,
	group: &str,
) -> Vec<(
	String,
	Vec<(&'static str, &'static str, Vec<(String, i64)>)>,
)> {
	let gn = include_str!("../../npu/shave/gnorm.sk");
	let ti = include_str!("../../npu/shave/trinv.sk");
	let sm = include_str!("../../npu/shave/softmax_lse.sk");
	let n = |v: usize| vec![("N".to_string(), v as i64)];
	let mut far = n(CAP);
	far.push(("NONORM".to_string(), 1));
	let near = if group == "f4" { CAP } else { CAP + s };
	vec![
		(
			"kern/delta".to_string(),
			vec![(gn, "eltwise_min", vec![]), (ti, "log_softmax", vec![])],
		),
		(format!("kern/near{near}"), vec![(sm, "softmax", n(near))]),
		("kern/far".to_string(), vec![(sm, "softmax", far)]),
	]
}

/// Build one SHAVE kernel `src` as the carrier `name`'s ELF in `d` (both
/// file names the compiler looks for), unless it is there.
pub fn write_kernel(d: &Path, src: &str, name: &str, defs: &[(String, i64)]) -> Result<(), String> {
	mkdir(d)?;
	// built every time (milliseconds) and written only when it changed: a
	// kernel kept because its file existed outlived edits of its source
	let k = npu::shave::dsl::compile(src, defs).map_err(|e| format!("kernel {name}: {e}"))?;
	let obj = npu::shave::assemble(&k.text()).map_err(|e| format!("kernel {name}: {e}"))?;
	let elf = npu::shave::link::link_with(&[obj], "controlled_act", npu::shave::link::COMMENT)?;
	for f in [
		format!("{name}_3720xx_lsu0_wo.elf"),
		format!("{name}_3720xx.elf"),
	] {
		let p = d.join(&f);
		if std::fs::read(&p).is_ok_and(|old| old == elf) {
			continue;
		}
		std::fs::write(&p, &elf).map_err(|e| format!("{}: {e}", p.display()))?;
	}
	Ok(())
}

/// Build the SHAVE kernels with our kernel compiler into `gdir/kern/*`.
fn build_kernels(gdir: &Path) -> Result<(), String> {
	for (s, group) in [(256, "p256"), (64, "p64"), (K, "f4")] {
		for (dir, ks) in kernels(s, group) {
			let d = gdir.join(&dir);
			mkdir(&d)?;
			for (src, name, defs) in ks {
				write_kernel(&d, src, name, &defs)?;
			}
		}
	}
	Ok(())
}

pub(crate) fn save(gdir: &Path, stem: &str, g: &Graph) -> Result<(), String> {
	let p = gdir.join(format!("{stem}.xml"));
	mkdir(p.parent().unwrap())?;
	g.save(&p, stem.rsplit('/').next().unwrap())?;
	// the argument names, in order (the compiler logs swizzled inputs by index)
	// and the ones the graph reads (a cut graph keeps the whole layer's list)
	let names: Vec<String> = g.inputs().into_iter().map(|i| i.0).collect();
	std::fs::write(
		gdir.join(format!("{stem}.uses")),
		g.used_inputs().join("\n") + "\n",
	)
	.map_err(|e| e.to_string())?;
	std::fs::write(gdir.join(format!("{stem}.args")), names.join("\n") + "\n")
		.map_err(|e| e.to_string())
}

/// Graphs of the vision tower in `OUT/graphs/vis` (0 without one): what
/// the steps after `graphs`, which do not read the checkpoint's weights,
/// plan from.
fn vis_graphs(out: &Path) -> usize {
	(0..)
		.take_while(|i| out.join(format!("graphs/vis/vis{i}.xml")).exists())
		.count()
}

/// The vision tower's set alone (Flash-Next's build; the 27B's makes it
/// with its other sets): the IRs in `OUT/graphs/vis`, the driver's blobs
/// of the missing ones, and `OUT/sets/vis` (`vis{i}.blob`, `vis.cfg`,
/// `pos.f32`). False when the checkpoint has no tower.
pub fn vis_set(ck: &Checkpoint, out: &Path, log: &mut dyn FnMut(&str)) -> Result<bool, String> {
	if !crate::vis::has_tower(ck) {
		return Ok(false);
	}
	let gdir = out.join("graphs");
	let deep = crate::vis::deepstack_layers(ck);
	let c = crate::vis::config_any(ck)?;
	let n = c.n_layer + 1;
	let stem = |i: usize| format!("vis/vis{i}");
	let missing: Vec<usize> = (0..n)
		.filter(|&i| !gdir.join(format!("{}.xml", stem(i))).exists())
		.collect();
	if !missing.is_empty() {
		log(&format!("vision tower: {} graphs", missing.len()));
		let m = if deep.is_empty() {
			crate::vis::model(ck)?
		} else {
			crate::vis::model_deepstack(ck)?
		};
		for i in missing {
			save(&gdir, &stem(i), &crate::vis::graph(&m, i, crate::vis::S))?;
		}
	}
	let stale: Vec<usize> = (0..n)
		.filter(|&i| !gdir.join(format!("{}.blob", stem(i))).exists())
		.collect();
	if !stale.is_empty() {
		log(&format!(
			"compiling {} vision graphs with the NPU driver",
			stale.len()
		));
		let _lock = npu::lock::NpuLock::acquire()?;
		let dev = npu::ze::Npu::open()?;
		for i in stale {
			let rd = |e: &str| {
				std::fs::read(gdir.join(format!("{}.{e}", stem(i))))
					.map_err(|x| format!("{}.{e}: {x}", stem(i)))
			};
			let slot = npu::lock::CompileSlot::acquire()?;
			let blob = dev
				.compile(&rd("xml")?, &rd("bin")?, "", 0)
				.and_then(|g| g.native())
				.map_err(|e| format!("{}: {e}", stem(i)))?;
			drop(slot);
			let p = gdir.join(format!("{}.blob", stem(i)));
			std::fs::write(&p, blob).map_err(|e| format!("{}: {e}", p.display()))?;
		}
	}
	let d = out.join("sets").join("vis");
	mkdir(&d)?;
	for i in 0..n {
		link(
			&gdir.join(format!("{}.blob", stem(i))),
			&d.join(format!("vis{i}.blob")),
		)?;
	}
	std::fs::write(
		d.join("vis.cfg"),
		crate::vis::cfg_text_deep(&c, crate::vis::S, &deep),
	)
	.map_err(|e| e.to_string())?;
	let pos: Vec<u8> = crate::vis::pos_grid(ck, &c)?
		.iter()
		.flat_map(|v| v.to_le_bytes())
		.collect();
	weights::write(&d.join("pos.f32"), &pos)?;
	Ok(true)
}

/// The jobs of every graph; with `ck` the IRs that are missing are written.
fn plan(ck: Option<&Checkpoint>, out: &Path) -> Result<Vec<Job>, String> {
	let (gdir, cal) = (out.join("graphs"), out.join("cal"));
	let mut jobs = Vec::new();
	// the vision tower: f16 graphs, the driver's compiler
	let nvis = match ck.filter(|ck| crate::vis::has_tower(ck)) {
		Some(ck) => {
			let n = crate::vis::config(ck)?.n_layer + 1;
			let missing: Vec<usize> = (0..n)
				.filter(|i| {
					!gdir.join(format!("vis/vis{i}.xml")).exists()
						|| !gdir.join(format!("vis/vis{i}.args")).exists()
				})
				.collect();
			if !missing.is_empty() {
				let m = crate::vis::model(ck)?;
				for i in missing {
					save(
						&gdir,
						&format!("vis/vis{i}"),
						&crate::vis::graph(&m, i, crate::vis::S),
					)?;
				}
			}
			n
		}
		None => vis_graphs(out),
	};
	jobs.extend((0..nvis).map(|i| Job {
		stem: format!("vis/vis{i}"),
		how: How::Driver,
		swz: false,
	}));
	let mut add = |stem: String,
	               how: How,
	               swz: bool,
	               make: &dyn Fn(&Checkpoint) -> Result<Graph, String>|
	 -> Result<(), String> {
		if let Some(ck) = ck {
			if !gdir.join(format!("{stem}.xml")).exists()
				|| !gdir.join(format!("{stem}.args")).exists()
			{
				save(&gdir, &stem, &make(ck)?)?;
			}
		}
		jobs.push(Job { stem, how, swz });
		Ok(())
	};
	let near_dir = |s: usize, dec: bool| -> &'static str {
		match (s, dec) {
			(_, true) => "kern/near1024",
			(256, _) => "kern/near1280",
			_ => "kern/near1088",
		}
	};
	// prefill, 256 and 64 positions
	for s in [256usize, 64] {
		let gp = format!("p{s}");
		for (l, vmul) in [(0usize, 16.0), (1, 256.0)] {
			for (sfx, cut) in [("", Cut::Whole), ("a", Cut::Pre), ("b", Cut::Post)] {
				add(
					format!("{gp}/lean{l}{sfx}"),
					How::Lean("kern/delta"),
					true,
					&|_| Ok(lean::delta(s, cut, vmul)),
				)?;
			}
		}
		for l in (0..NL).filter(|&l| is_attn(l)) {
			let c = |ck: &Checkpoint| weights::attn_consts(ck, &cal, l);
			if s == 256 {
				add(format!("{gp}/lean{l}"), How::Lean(""), true, &|ck| {
					Ok(lean::attn(s, CAP, Part::All, &c(ck)?))
				})?;
			}
			add(format!("{gp}/lean{l}a"), How::Lean(""), true, &|ck| {
				Ok(lean::attn(s, CAP, Part::Pre, &c(ck)?))
			})?;
			add(format!("{gp}/lean{l}b"), How::Lean(""), true, &|ck| {
				Ok(lean::attn(s, CAP, Part::Post, &c(ck)?))
			})?;
		}
		add(
			format!("{gp}/near"),
			How::Lean(near_dir(s, false)),
			false,
			&|_| Ok(lean::attn_core(s, CAP, Part::Near)),
		)?;
		add(format!("{gp}/far"), How::Lean("kern/far"), false, &|_| {
			Ok(lean::attn_core(s, CAP, Part::Far(1)))
		})?;
	}
	// the verify set, with the MLP on the NPU and cut off
	for (gp, mlp) in [("d4", true), ("d4m", false)] {
		for l in 0..NL {
			if is_attn(l) {
				add(format!("{gp}/layer{l}"), How::Plain, true, &|ck| {
					Ok(dec::attn_verify(
						K,
						DCAP,
						l / 4,
						&weights::attn_c(ck, &cal, l)?,
						mlp,
					))
				})?;
			} else {
				add(
					format!("{gp}/layer{l}"),
					How::Lean("kern/delta"),
					true,
					&|ck| {
						Ok(dec::delta_verify(
							K,
							l - l / 4,
							&weights::delta_c(ck, &cal, l)?,
							mlp,
						))
					},
				)?;
			}
		}
	}
	// its attention past the window: the far form
	for l in (0..NL).filter(|&l| is_attn(l)) {
		let c = |ck: &Checkpoint| weights::attn_consts(ck, &cal, l);
		add(format!("f4/lean{l}a"), How::Lean(""), true, &|ck| {
			Ok(lean::attn(K, CAP, Part::Pre, &c(ck)?))
		})?;
		add(format!("f4/lean{l}b"), How::Lean(""), true, &|ck| {
			Ok(lean::attn(K, CAP, Part::Post, &c(ck)?))
		})?;
	}
	add(
		"f4/near".into(),
		How::Lean(near_dir(K, true)),
		false,
		&|_| Ok(lean::attn_core(K, CAP, Part::Near0)),
	)?;
	add("f4/far".into(), How::Lean("kern/far"), false, &|_| {
		Ok(lean::attn_core(K, CAP, Part::Far(1)))
	})?;
	add("f4/farm".into(), How::Lean("kern/far"), false, &|_| {
		Ok(lean::attn_core(K, CAP, Part::Far(8)))
	})?;
	// head, MTP block and draft head
	add("mtp/head".into(), How::Driver, false, &|ck| {
		let sc = weights::head_scales(&cal)?;
		Ok(dec::head(
			K,
			&crate::refm::norm_w(ck, "model.language_model.norm.weight", crate::refm::W)?,
			&sc,
		))
	})?;
	add("mtp/mtphead".into(), How::Driver, false, &|_| {
		Ok(dec::mtp_head(&weights::head_scales(&cal)?[..DRAFT_VOCAB]))
	})?;
	for s in [K, 1] {
		add(format!("mtp/mtp_s{s}"), How::Driver, false, &|ck| {
			Ok(dec::mtp(s, DCAP, &weights::mtp_c(ck, &cal)?))
		})?;
	}
	Ok(jobs)
}

/// Write every IR that is missing.
pub fn graphs(ck: &Checkpoint, out: &Path) -> Result<usize, String> {
	Ok(plan(Some(ck), out)?.len())
}

/// Where our build of the NPU compiler is.
pub struct Tools {
	/// The directory of `vpux-translate.exe` and `vpux-opt.exe`.
	pub bin: PathBuf,
	/// The directory of its TBB DLLs.
	pub tbb: PathBuf,
	/// Compiles at once.
	pub jobs: usize,
}

impl Tools {
	/// `bin` (or `NPUC_BIN`, or the `npu-compiler` directory bundled beside
	/// the executable), with TBB from `NPUC_TBB`, the compiler's own directory
	/// when it holds `tbb12.dll`, or the build tree beside it
	/// (`<build>/temp/Windows_AMD64/tbb/bin`).
	pub fn find(bin: Option<&Path>, jobs: usize) -> Result<Tools, String> {
		let bundled = || {
			let d = std::env::current_exe().ok()?.parent()?.join("npu-compiler");
			d.join("vpux-opt.exe").is_file().then_some(d)
		};
		let bin = bin.map(Path::to_path_buf).or_else(|| std::env::var_os("NPUC_BIN").map(PathBuf::from)).or_else(bundled).ok_or(
			"the NPU compiler is needed for this step: give its directory (vpux-translate.exe, vpux-opt.exe) with --npu-compiler DIR or NPUC_BIN (see crates/npu/compiler-patches/README.md)",
		)?;
		for exe in ["vpux-translate.exe", "vpux-opt.exe"] {
			if !bin.join(exe).is_file() {
				return Err(format!("{}: no {exe}", bin.display()));
			}
		}
		let tbb = std::env::var_os("NPUC_TBB")
			.map(PathBuf::from)
			.unwrap_or_else(|| {
				if bin.join("tbb12.dll").is_file() {
					bin.clone()
				} else {
					bin.join("../../../temp/Windows_AMD64/tbb/bin")
				}
			});
		Ok(Tools {
			bin,
			tbb,
			jobs: jobs.max(1),
		})
	}
}

fn newer(a: &Path, b: &Path) -> bool {
	match (
		a.metadata().and_then(|m| m.modified()),
		b.metadata().and_then(|m| m.modified()),
	) {
		(Ok(x), Ok(y)) => x >= y,
		_ => false,
	}
}

/// One graph through our compiler: import, `--default-hw-mode`, ELF. The
/// swizzled inputs' names go to `<stem>.swz`.
fn compile_one(t: &Tools, gdir: &Path, j: &Job) -> Result<(), String> {
	let stem = gdir.join(&j.stem);
	let ext = |e: &str| PathBuf::from(format!("{}.{e}", stem.display()));
	let (xml, blob, log) = (ext("xml"), ext("blob"), ext("log"));
	let (ie, vpuip, elf, tmp, idx) = (
		ext("ie.mlir"),
		ext("vpuip.mlir"),
		ext("elf.mlir"),
		ext("tmp.blob"),
		ext("idx"),
	);
	let mut path = std::ffi::OsString::from(&t.tbb);
	path.push(";");
	path.push(&t.bin);
	path.push(";");
	path.push(std::env::var_os("PATH").unwrap_or_default());
	let mut opts = "verify-locations=off".to_string();
	let mut env: Vec<(&str, String)> = Vec::new();
	if let How::Lean(k) = j.how {
		opts += " enable-grouped-matmul=true";
		for v in ["NPU_RS_DPU1LIST", "NPU_RS_VIEWFOLD", "NPU_RS_DISTVIEW"] {
			env.push((v, "1".into()));
		}
		env.push(("NPU_RS_GMM_MAXG", "96".into()));
		if !k.is_empty() {
			env.push((
				"NPU_RS_KERNEL_DIR",
				gdir.join(k).to_string_lossy().replace('\\', "/"),
			));
		}
	}
	if j.swz {
		env.push(("NPU_RS_SWZIN", "1".into()));
		env.push(("NPU_RS_SWZKEY", SWZ_KEY.to_string()));
		env.push(("NPU_RS_SWZIN_LOG", idx.to_string_lossy().into_owned()));
	}
	let _ = std::fs::remove_file(&idx);
	let arch = "--vpu-arch=NPU37XX";
	let hw = format!("--default-hw-mode={opts}");
	let s = |p: &Path| p.to_string_lossy().into_owned();
	let stages: [(&str, &str, Vec<String>); 4] = [
		(
			"import-IE",
			"vpux-translate.exe",
			vec![
				arch.into(),
				"--import-IE".into(),
				s(&xml),
				"-o".into(),
				s(&ie),
			],
		),
		(
			"default-hw",
			"vpux-opt.exe",
			vec![arch.into(), hw, s(&ie), "-o".into(), s(&vpuip)],
		),
		(
			"to-ELF",
			"vpux-opt.exe",
			vec![
				arch.into(),
				"--lower-VPUIP-to-ELF".into(),
				s(&vpuip),
				"-o".into(),
				s(&elf),
			],
		),
		(
			"export-ELF",
			"vpux-translate.exe",
			vec![
				arch.into(),
				"--export-ELF".into(),
				s(&elf),
				"-o".into(),
				s(&tmp),
			],
		),
	];
	let mut text = String::new();
	let mut fail = None;
	for (what, exe, args) in stages {
		let mut c = Command::new(t.bin.join(exe));
		c.args(&args).env("PATH", &path);
		for v in [
			"NPU_RS_DPU1LIST",
			"NPU_RS_VIEWFOLD",
			"NPU_RS_DISTVIEW",
			"NPU_RS_GMM_MAXG",
			"NPU_RS_KERNEL_DIR",
			"NPU_RS_SWZIN",
			"NPU_RS_SWZKEY",
			"NPU_RS_SWZIN_LOG",
		] {
			c.env_remove(v);
		}
		c.envs(env.iter().map(|(k, v)| (*k, v.as_str())));
		let o = c
			.output()
			.map_err(|e| format!("{}: {e}", t.bin.join(exe).display()))?;
		text += &format!(
			"== {what}: {}\n{}{}\n",
			o.status,
			String::from_utf8_lossy(&o.stdout),
			String::from_utf8_lossy(&o.stderr)
		);
		if !o.status.success() {
			fail = Some(what);
			break;
		}
	}
	for f in [&ie, &vpuip, &elf] {
		let _ = std::fs::remove_file(f);
	}
	let _ = std::fs::write(&log, &text);
	if let Some(what) = fail {
		return Err(format!("{}: {what} failed (see {})", j.stem, log.display()));
	}
	// swizzled inputs: indices -> names
	let args = std::fs::read_to_string(ext("args"))
		.map_err(|e| format!("{}: {e}", ext("args").display()))?;
	let args: Vec<&str> = args.lines().collect();
	let mut names = Vec::new();
	if j.swz {
		for i in std::fs::read_to_string(&idx)
			.unwrap_or_default()
			.split_whitespace()
		{
			let i: usize = i
				.parse()
				.map_err(|_| format!("{}: bad swizzle log", j.stem))?;
			names.push(
				*args
					.get(i)
					.ok_or(format!("{}: swizzle log names input {i}", j.stem))?,
			);
		}
	}
	let _ = std::fs::remove_file(&idx);
	std::fs::write(ext("swz"), names.join("\n") + "\n").map_err(|e| e.to_string())?;
	std::fs::rename(&tmp, &blob).map_err(|e| format!("{}: {e}", blob.display()))
}

/// Compile `gdir/<stem>.xml` (with its `.args`) with our compiler: the
/// prefill options without SHAVE kernels when `lean`, weight swizzling when
/// `swz` (Qwen3.8-Flash-Next's graphs).
pub fn compile_graph(
	t: &Tools,
	gdir: &Path,
	stem: &str,
	lean: bool,
	swz: bool,
) -> Result<(), String> {
	compile_graph_kern(t, gdir, stem, if lean { Some("") } else { None }, swz)
}

/// [`compile_graph`] with the prefill options and the SHAVE kernels of
/// `gdir/<kern>` (`Some("")`: none) or without them (`None`).
pub fn compile_graph_kern(
	t: &Tools,
	gdir: &Path,
	stem: &str,
	kern: Option<&'static str>,
	swz: bool,
) -> Result<(), String> {
	compile_one(
		t,
		gdir,
		&Job {
			stem: stem.to_string(),
			how: kern.map_or(How::Plain, How::Lean),
			swz,
		},
	)
}

/// Compile every graph whose blob is missing or older than its IR.
pub fn compile(
	out: &Path,
	tools: Option<&Tools>,
	log: &(dyn Fn(&str) + Sync),
) -> Result<(), String> {
	let gdir = out.join("graphs");
	let jobs = plan(None, out)?;
	let stale = |j: &Job| {
		let (x, b) = (
			gdir.join(format!("{}.xml", j.stem)),
			gdir.join(format!("{}.blob", j.stem)),
		);
		!newer(&b, &x) || (j.how != How::Driver && !gdir.join(format!("{}.swz", j.stem)).exists())
	};
	for j in &jobs {
		if !gdir.join(format!("{}.xml", j.stem)).exists() {
			return Err(format!(
				"{}.xml is missing: run the graphs step first",
				j.stem
			));
		}
	}
	let ours: Vec<&Job> = jobs
		.iter()
		.filter(|j| j.how != How::Driver && stale(j))
		.collect();
	if !ours.is_empty() {
		let t = tools
			.ok_or("the NPU compiler is needed for this step (--npu-compiler DIR or NPUC_BIN)")?;
		build_kernels(&gdir)?;
		log(&format!(
			"compiling {} graphs, {} at a time",
			ours.len(),
			t.jobs
		));
		let queue = Mutex::new(ours.iter());
		let errs = Mutex::new(Vec::new());
		let done = std::sync::atomic::AtomicUsize::new(0);
		std::thread::scope(|sc| {
			for _ in 0..t.jobs {
				sc.spawn(|| loop {
					let Some(j) = queue.lock().unwrap().next() else {
						break;
					};
					if let Err(e) = compile_one(t, &gdir, j) {
						errs.lock().unwrap().push(e);
					}
					let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
					if n.is_multiple_of(20) {
						log(&format!("  {n} of {}", ours.len()));
					}
				});
			}
		});
		let errs = errs.into_inner().unwrap();
		if !errs.is_empty() {
			return Err(format!(
				"{} graphs failed to compile; the first: {}",
				errs.len(),
				errs[0]
			));
		}
	}
	let drv: Vec<&Job> = jobs
		.iter()
		.filter(|j| j.how == How::Driver && stale(j))
		.collect();
	if !drv.is_empty() {
		log(&format!(
			"compiling {} graphs with the NPU driver",
			drv.len()
		));
		let _lock = npu::lock::NpuLock::acquire()?;
		let dev = npu::ze::Npu::open()?;
		for j in drv {
			let rd = |e: &str| {
				std::fs::read(gdir.join(format!("{}.{e}", j.stem)))
					.map_err(|x| format!("{}.{e}: {x}", j.stem))
			};
			let slot = npu::lock::CompileSlot::acquire()?;
			let blob = dev
				.compile(&rd("xml")?, &rd("bin")?, "", 0)
				.and_then(|g| g.native())
				.map_err(|e| format!("{}: {e}", j.stem))?;
			drop(slot);
			let p = gdir.join(format!("{}.blob", j.stem));
			std::fs::write(&p, blob).map_err(|e| format!("{}: {e}", p.display()))?;
		}
	}
	Ok(())
}

/// A swizzled copy of a weight buffer ([`npu::ze::swizzle`]).
fn swizzle(b: &[u8], key: usize) -> Result<Vec<u8>, String> {
	let mut o = b.to_vec();
	npu::ze::swizzle(&mut o, key)?;
	Ok(o)
}

fn link(src: &Path, dst: &Path) -> Result<(), String> {
	let _ = std::fs::remove_file(dst);
	std::fs::hard_link(src, dst)
		.or_else(|_| std::fs::copy(src, dst).map(|_| ()))
		.map_err(|e| format!("{} -> {}: {e}", src.display(), dst.display()))
}

/// Path `to` relative to the directory `from` (both under one root), with `/`.
fn rel(from: &Path, to: &Path) -> String {
	let (a, b): (Vec<_>, Vec<_>) = (from.components().collect(), to.components().collect());
	let n = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
	let mut p: Vec<String> = vec!["..".to_string(); a.len() - n];
	p.extend(
		b[n..]
			.iter()
			.map(|c| c.as_os_str().to_string_lossy().into_owned()),
	);
	p.join("/")
}

/// The `.mlp` file of a cut layer (what the engine's CPU MLP needs besides
/// the codes): a text line, then f32 arrays; the norm weight, and per
/// rotation its signs and matrix, then the gate, up and down row scales.
fn mlp_file(c: &dec::MlpC, attn: bool, wdir: &str) -> Vec<u8> {
	let (gate, up, down) = dec::mlp_names(attn);
	let mut b = format!(
		"mlp1 eps=1e-6 gate={} up={} down={down} wdir={wdir} had=1\n",
		gate.join(","),
		up.join(",")
	)
	.into_bytes();
	let sg = crate::gptq::signs(crate::refm::FF);
	let n = crate::gptq::ROTB;
	let had: Vec<f32> = (0..n * n)
		.map(|i| {
			if ((i / n) & (i % n)).count_ones() & 1 == 1 {
				-1.0 / 16.0
			} else {
				1.0 / 16.0
			}
		})
		.collect();
	let w = crate::refm::W;
	let mut put = |v: &[f32]| v.iter().for_each(|x| b.extend(x.to_le_bytes()));
	put(&c.ln2);
	for k in [w, w, crate::refm::FF] {
		put(&sg[..k]);
		put(&had);
	}
	put(&c.s_gu[..crate::refm::FF]);
	put(&c.s_gu[crate::refm::FF..]);
	put(&c.s_down);
	b
}

/// The graph sets and the `.npu` files, from the compiled graphs.
pub fn assemble(ck: &Checkpoint, out: &Path, log: &dyn Fn(&str)) -> Result<(), String> {
	let (gdir, cal, wdir, wz, sdir) = (
		out.join("graphs"),
		out.join("cal"),
		out.join("w"),
		out.join("wz"),
		out.join("sets"),
	);
	let jobs = plan(None, out)?;
	for j in &jobs {
		if !gdir.join(format!("{}.blob", j.stem)).exists() {
			return Err(format!(
				"{}.blob is missing: run the compile step first",
				j.stem
			));
		}
	}
	let blob = |stem: &str| gdir.join(format!("{stem}.blob"));
	// which layer weight inputs are read swizzled: every graph that binds
	// an input must read it in the same form
	let mut form: BTreeMap<(usize, String), BTreeSet<bool>> = BTreeMap::new();
	for j in jobs.iter().filter(|j| j.swz) {
		let name = j.stem.rsplit('/').next().unwrap();
		let digits: String = name
			.chars()
			.skip_while(|c| !c.is_ascii_digit())
			.take_while(|c| c.is_ascii_digit())
			.collect();
		let l: usize = digits
			.parse()
			.map_err(|_| format!("{}: no layer number", j.stem))?;
		let rd = |e: &str| {
			std::fs::read_to_string(gdir.join(format!("{}.{e}", j.stem)))
				.map_err(|x| format!("{}.{e}: {x}", j.stem))
		};
		let swz: BTreeSet<String> = rd("swz")?.lines().map(str::to_string).collect();
		// the delta-net prefill graphs are shared by every delta layer
		let layers: Vec<usize> = if name.starts_with("lean") && !is_attn(l) {
			(0..NL)
				.filter(|&x| !is_attn(x) && (x == 0) == (l == 0))
				.collect()
		} else {
			vec![l]
		};
		for a in rd("uses")?
			.lines()
			.filter(|a| a.starts_with('w') && !a.starts_with("w."))
		{
			for &x in &layers {
				if wdir.join(format!("lean{x}.{a}.in")).exists() {
					form.entry((x, a.to_string()))
						.or_default()
						.insert(swz.contains(a));
				}
			}
		}
	}
	let bad: Vec<String> = form
		.iter()
		.filter(|(_, v)| v.len() > 1)
		.map(|((l, a), _)| format!("lean{l}.{a}"))
		.collect();
	if !bad.is_empty() {
		return Err(format!(
			"{} weight inputs are read swizzled by some graphs and plain by others (first: {})",
			bad.len(),
			bad[0]
		));
	}
	mkdir(&wz)?;
	let nswz = form.values().filter(|v| v.contains(&true)).count();
	log(&format!(
		"{} of {} layer weight inputs are read swizzled",
		nswz,
		form.len()
	));
	for ((l, a), v) in &form {
		let f = format!("lean{l}.{a}.in");
		if v.contains(&true) && !newer(&wz.join(&f), &wdir.join(&f)) {
			let d = std::fs::read(wdir.join(&f)).map_err(|e| format!("{f}: {e}"))?;
			weights::write(&wz.join(&f), &swizzle(&d, SWZ_KEY)?)?;
		}
	}
	// a prefill set's bound files: every per-layer input file, in its form
	let link_weights = |d: &Path| -> Result<(), String> {
		for e in std::fs::read_dir(&wdir).map_err(|e| e.to_string())? {
			let name = e
				.map_err(|e| e.to_string())?
				.file_name()
				.to_string_lossy()
				.into_owned();
			let Some(rest) = name
				.strip_prefix("lean")
				.and_then(|r| r.strip_suffix(".in"))
			else {
				continue;
			};
			let Some((l, a)) = rest.split_once('.') else {
				continue;
			};
			let swz = l
				.parse()
				.ok()
				.and_then(|l: usize| form.get(&(l, a.to_string())))
				.is_some_and(|v| v.contains(&true));
			link(&(if swz { &wz } else { &wdir }).join(&name), &d.join(&name))?;
		}
		Ok(())
	};
	let write = |p: PathBuf, text: &str| {
		std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))
	};
	// prefill sets: g1c (256 positions, window of CAP keys), f1c / f1h (far)
	for (set, gp, far) in [
		("g1c", "p256", false),
		("f1c", "p256", true),
		("f1h", "p64", true),
	] {
		let d = sdir.join(set);
		mkdir(&d)?;
		for l in 0..NL {
			if is_attn(l) {
				if far {
					for c in ["a", "b"] {
						link(
							&blob(&format!("{gp}/lean{l}{c}")),
							&d.join(format!("lean{l}{c}.blob")),
						)?;
					}
				} else {
					link(
						&blob(&format!("{gp}/lean{l}")),
						&d.join(format!("lean{l}.blob")),
					)?;
				}
			} else {
				// the whole layer (run without a scan device) and its cut
				// around the recurrence; layers 1.. share layer 1's graphs
				for c in ["", "a", "b"] {
					if l <= 1 {
						link(
							&blob(&format!("{gp}/lean{l}{c}")),
							&d.join(format!("lean{l}{c}.blob")),
						)?;
					} else {
						write(d.join(format!("lean{l}{c}.use")), &format!("lean1{c}\n"))?;
					}
				}
			}
		}
		if far {
			link(&blob(&format!("{gp}/near")), &d.join("near.blob"))?;
			link(&blob(&format!("{gp}/far")), &d.join("far.blob"))?;
			// the far graphs leave P undivided: the engine divides in its merge
			write(d.join("far.nonorm"), "")?;
		}
		link_weights(&d)?;
	}
	// the verify set's far attention
	let d = sdir.join("f4d");
	mkdir(&d)?;
	for l in (0..NL).filter(|&l| is_attn(l)) {
		for c in ["a", "b"] {
			link(
				&blob(&format!("f4/lean{l}{c}")),
				&d.join(format!("lean{l}{c}.blob")),
			)?;
		}
	}
	for n in ["near", "far", "farm"] {
		link(&blob(&format!("f4/{n}")), &d.join(format!("{n}.blob")))?;
	}
	write(d.join("far.nonorm"), "")?;
	// the verify sets
	let outs: String = (0..NL)
		.map(|l| {
			format!(
				"{l} {}\n",
				if is_attn(l) {
					dec::attn_outs(l / 4)
				} else {
					dec::DELTA_OUTS.to_string()
				}
			)
		})
		.collect();
	let head = weights::bound(&wdir, dec::LM_HEAD);
	for (set, gp, mlp) in [("dec4vww", "d4", true), ("dec4vm", "d4m", false)] {
		let d = sdir.join(set);
		mkdir(&d)?;
		for l in 0..NL {
			link(
				&blob(&format!("{gp}/layer{l}")),
				&d.join(format!("layer{l}.blob")),
			)?;
			if !mlp {
				let c = if is_attn(l) {
					weights::attn_c(ck, &cal, l)?.mlp
				} else {
					weights::delta_c(ck, &cal, l)?.mlp
				};
				weights::write(
					&d.join(format!("layer{l}.mlp")),
					&mlp_file(&c, is_attn(l), &rel(&d, &wdir)),
				)?;
			}
		}
		link(&blob("mtp/head"), &d.join("head.blob"))?;
		link(&head, &d.join(head.file_name().unwrap()))?;
		write(d.join("outs.txt"), &outs)?;
	}
	// head, MTP block and draft head
	let d = sdir.join("mtp");
	mkdir(&d)?;
	for n in ["head", "mtphead", "mtp_s1", &format!("mtp_s{K}")] {
		link(&blob(&format!("mtp/{n}")), &d.join(format!("{n}.blob")))?;
	}
	for e in ["xml", "bin"] {
		link(
			&gdir.join(format!("mtp/head.{e}")),
			&d.join(format!("head.{e}")),
		)?;
	}
	for e in std::fs::read_dir(&wdir).map_err(|e| e.to_string())? {
		let name = e.map_err(|e| e.to_string())?.file_name();
		if name.to_string_lossy().starts_with("w.") {
			link(&wdir.join(&name), &d.join(&name))?;
		}
	}
	// the vision tower
	let nvis = vis_graphs(out);
	if nvis > 0 {
		let d = sdir.join("vis");
		mkdir(&d)?;
		for i in 0..nvis {
			link(
				&blob(&format!("vis/vis{i}")),
				&d.join(format!("vis{i}.blob")),
			)?;
		}
		let c = crate::vis::config(ck)?;
		write(d.join("vis.cfg"), &crate::vis::cfg_text(&c, crate::vis::S))?;
		let pos: Vec<u8> = crate::vis::pos_grid(ck, &c)?
			.iter()
			.flat_map(|v| v.to_le_bytes())
			.collect();
		weights::write(&d.join("pos.f32"), &pos)?;
	}
	// the engine descriptions (paths relative to the file)
	let vision = if nvis > 0 {
		"# images: the tower's per-patch work on the NPU, its attention on the `scan` adapter (else the cores)\nvision    = sets/vis\n"
	} else {
		""
	};
	let common = format!("emb       = emb_table.f16\ntokenizer = tokenizer\n{vision}");
	let prefill = "# prefill: 256 positions, then the far sets for positions past the window\nprefill   = sets/g1c\nset       = sets/f1c\nset       = sets/f1h\n";
	let spec = |dec: &str| {
		format!("# decode: {K}-position verify set + MTP drafts, far form past its {DCAP} window\ndecode    = sets/{dec}\nmtp       = sets/mtp\nfardec    = sets/f4d\n")
	};
	write(
		out.join("model.npu"),
		&format!("# NPU + CPU + GPU: the MLPs of the decode step on the cores and the GTX 1060\n{prefill}{}{common}# helpers: delta recurrence and far blocks on the iGPU, MLP share on the discrete GPU\nscan      = intel\nmlpgpu    = geforce\n", spec("dec4vm")),
	)?;
	write(
		out.join("model_igpu.npu"),
		&format!(
			"# NPU + iGPU\n{prefill}{}{common}scan      = intel\n",
			spec("dec4vww")
		),
	)?;
	write(
		out.join("model_npu.npu"),
		&format!("# NPU only\n{prefill}{}{common}", spec("dec4vww")),
	)?;
	log(&format!(
		"sets: g1c f1c f1h f4d dec4vm dec4vww mtp{}; model.npu, model_igpu.npu, model_npu.npu",
		if nvis > 0 { " vis" } else { "" }
	));
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn swizzle_rotates_the_blocks_of_each_row() {
		let b: Vec<u8> = (0..4096u32).map(|i| (i / 16) as u8).collect();
		let s = swizzle(&b, 2).unwrap();
		// row 0 unchanged; row 1 rotated by 8 blocks, row 2 by 16, row 3 by 24
		assert_eq!(&s[..512], &b[..512]);
		for r in 1..4usize {
			for blk in 0..32usize {
				let to = (blk + 8 * r) % 32;
				assert_eq!(
					s[r * 512 + to * 16],
					b[r * 512 + blk * 16],
					"row {r} block {blk}"
				);
			}
		}
		// the pattern repeats every 4 rows
		assert_eq!(&s[2048..2560], &b[2048..2560]);
		assert!(swizzle(&b[..1000], 2).is_err());
	}

	#[test]
	fn swizzle_over_threads_matches_one_pass() {
		// past the threaded split (2 MiB pieces at key 2), odd piece count
		let n = 2048 * 1024 * 5 + 2048 * 3;
		let b: Vec<u8> = (0..n).map(|i| (i * 7 + i / 512) as u8).collect();
		let s = swizzle(&b, 2).unwrap();
		for (r, (x, y)) in b.chunks_exact(512).zip(s.chunks_exact(512)).enumerate() {
			let mut want = x.to_vec();
			want.rotate_right((r % 4) * 8 * 16);
			assert_eq!(y, &want[..], "row {r}");
		}
	}

	#[test]
	fn relative_paths() {
		assert_eq!(rel(Path::new("o/sets/dec4vm"), Path::new("o/w")), "../../w");
	}
}
