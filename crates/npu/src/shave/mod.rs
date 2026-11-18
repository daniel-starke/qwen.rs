//! Our own toolchain for ACT-SHAVE kernels (the NPU's software cores): a
//! kernel compiler, an assembler and a linker, so that writing and building
//! a kernel needs no vendor tool.
//!
//! - [`dsl`]: the kernel language (`crates/npu/SHAVE-DSL.md`) to bundles.
//! - [`prog`]: a program as bundles; [`mach`]: the processor's timing rules
//!   and an emulator that runs a program on the host.
//! - [`isa`]: the instruction encodings (`isa.tbl`).
//! - [`asm`]: assembly source to a relocatable object.
//! - [`elf`]: that object, as an ELF32 file.
//! - [`link`]: objects to the kernel's executable ELF.
//! - [`conflicts`]: instructions that cannot be issued together
//!   (`conflicts.tbl`).
//!
//! `examples/shave.rs` is the command line. `tests/shave.rs` builds the
//! kernels of `crates/npu/shave` end to end; the compiler's kernels are
//! checked in the emulator (`tests/shave_dsl.rs`) and on the NPU
//! (`scripts/sktest.py`).

pub mod asm;
pub mod conflicts;
pub mod dsl;
pub mod elf;
pub mod isa;
pub mod link;
pub mod mach;
pub mod prog;

/// Assemble one source file. Two instructions that share a resource of the
/// processor ([`conflicts`]) are refused.
pub fn assemble(source: &str) -> Result<elf::Object, String> {
	if let Some(c) = prog::Program::parse(source)
		.ok()
		.and_then(|p| conflicts::check(&p).into_iter().next())
	{
		return Err(c);
	}
	asm::assemble(source)
}

/// Assemble one source file and link it into a kernel ELF.
pub fn build(source: &str, entry: &str) -> Result<Vec<u8>, String> {
	link::link(&[assemble(source)?], entry)
}
