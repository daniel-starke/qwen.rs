//! The kernel compiler: a small language for ACT-SHAVE kernels
//! (`crates/npu/SHAVE-DSL.md`) to bundles.
//!
//! A kernel is a C function in a subset of C with the SHAVE's vector types
//! (`half8`, `float4`, `int4`): every operator and intrinsic is one
//! instruction. The compiler does what is tedious and easy to get wrong by
//! hand on this processor:
//!
//! - [`parse`]: the preprocessor (`#define`, `#if`) and the C subset;
//! - [`lower`]: constants folded, constant `for` loops unrolled, each
//!   operator, intrinsic and memory access to its instruction(s);
//! - [`live`]: a register of its own for every value a variable takes in a
//!   block (SSA form within the block), and which values are alive where;
//! - [`sched`]: packs every straight-line block into bundles (one operation
//!   per unit and cycle), keeps each consumer behind its producer's latency,
//!   fills a branch's six delay slots with the end of the block, and assigns
//!   the machine's registers.
//!
//! The result is a [`Program`]: assembly text for [`super::asm`], and what
//! [`super::mach`] runs in the host tests.

pub mod live;
pub mod lower;
pub mod parse;
pub mod sched;

use super::prog::Program;

/// A virtual register.
pub type Vr = usize;

#[derive(Clone, Debug)]
pub enum Arg {
	/// the result register (and an element index to print)
	Def(Vr, Option<u32>),
	Use(Vr, Option<u32>),
	Int(i64),
}

/// A memory access: `size` bytes at `base + off`.
#[derive(Clone, Debug)]
pub struct Mem {
	pub store: bool,
	/// the base variable's `@tag`: accesses under different tags never overlap
	pub tag: Option<usize>,
	pub base: Vr,
	/// which definition of `base` (same register and version = same address)
	pub ver: u32,
	pub off: i64,
	pub size: i64,
}

/// One instruction on virtual registers.
#[derive(Clone, Debug)]
pub struct AOp {
	/// the mnemonic on each unit that can run it
	pub alts: Vec<String>,
	pub args: Vec<Arg>,
	pub lat: u32,
	/// a result that replaces only these bytes (offset, length) of its register
	pub part: Option<(u8, u8)>,
	/// with `part`: the first write of a new value (the other bytes follow)
	pub begin: bool,
	pub mem: Option<Mem>,
	pub line: usize,
}

#[derive(Clone, Debug)]
pub enum Term {
	None,
	/// branch to `label` when the compare `ops[cmp]` gives `cc`
	Branch {
		cmp: usize,
		cc: &'static str,
		label: usize,
	},
	/// branch to `label`
	Jump(usize),
	Ret,
}

#[derive(Clone, Debug)]
pub struct Block {
	pub ops: Vec<AOp>,
	pub term: Term,
}

#[derive(Clone, Debug)]
pub enum Item {
	Label(usize),
	/// a loop's first cycle (its source line) and last
	LoopBegin(usize),
	LoopEnd,
	Block(Block),
}

#[derive(Clone, Debug)]
pub struct VregInfo {
	pub vector: bool,
	pub name: String,
	pub tag: Option<usize>,
	/// a fixed machine register (`params` = i18, the stack pointer = i19)
	pub pin: Option<u32>,
}

pub struct Kernel {
	pub program: Program,
	/// one line per loop: cycles per round
	pub report: Vec<String>,
}

impl Kernel {
	/// The assembly source ([`super::asm`]).
	pub fn text(&self) -> String {
		self.program.text(ENTRY)
	}

	/// The linked kernel ELF.
	pub fn elf(&self) -> Result<Vec<u8>, String> {
		super::build(&self.text(), ENTRY)
	}
}

/// The entry symbol of every kernel.
pub const ENTRY: &str = "controlled_act";

/// Compile a kernel source; `defs` are the command line's `-DNAME=VALUE`.
pub fn compile(src: &str, defs: &[(String, i64)]) -> Result<Kernel, String> {
	// a -D the source never looks at is a typing error
	let names: Vec<&str> = src
		.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
		.collect();
	if let Some(d) = defs.iter().find(|d| !names.contains(&d.0.as_str())) {
		return Err(format!("-D{}: the kernel does not use {}", d.0, d.0));
	}
	let file = parse::parse(src, defs)?;
	let low = lower::lower(&file)?;
	let mut values = low.clone();
	let flow = live::flow(&low.vregs, &low.items)?;
	live::rename(&mut values.vregs, &mut values.items, &flow);
	// before giving up: the schedules with the fewest values alive, then the
	// variables in place (an update in place can need a register less)
	use sched::{schedule, Goal};
	schedule(&values.vregs, &values.items, Goal::Short)
		.or_else(|e| schedule(&values.vregs, &values.items, Goal::Narrow).map_err(|_| e))
		.or_else(|e| schedule(&low.vregs, &low.items, Goal::Narrow).map_err(|_| e))
}
