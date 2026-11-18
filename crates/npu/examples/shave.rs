//! ACT-SHAVE kernel compiler, assembler and linker (`npu::shave`):
//!
//! ```text
//! shave build KERNEL.sk OUT.elf [-DNAME=VALUE ...] [--asm OUT.s]
//!                                                  compile (SHAVE-DSL.md), assemble and link
//! shave build KERNEL.s OUT.elf [--entry NAME]      assemble and link
//! shave as    KERNEL.s OUT.o                       assemble only
//! shave ld    OUT.elf IN.o [IN.o ...] [--entry NAME]
//! ```
//!
//! The entry symbol defaults to `controlled_act`. `--comment TEXT` sets the
//! linked file's `.comment`. `OUT.elf` goes to
//! `NPU_RS_KERNEL_DIR` of our compiler under the name of the kernel it
//! replaces (compiler patch 0008).

use npu::shave::{dsl, elf::Object, link};

fn run(mut a: Vec<String>) -> Result<(), String> {
	let mut entry = "controlled_act".to_string();
	if let Some(i) = a.iter().position(|x| x == "--entry") {
		if i + 1 >= a.len() {
			return Err("--entry needs a name".into());
		}
		entry = a.remove(i + 1);
		a.remove(i);
	}
	// the `.comment` text
	let mut comment = link::COMMENT.to_string();
	if let Some(i) = a.iter().position(|x| x == "--comment") {
		if i + 1 >= a.len() {
			return Err("--comment needs a text".into());
		}
		comment = a.remove(i + 1);
		a.remove(i);
	}
	// a kernel-language source: its parameters and where to leave the assembly
	let mut defs = Vec::new();
	while let Some(i) = a.iter().position(|x| x.starts_with("-D")) {
		let d = a.remove(i);
		let (name, value) = d[2..].split_once('=').unwrap_or((&d[2..], "1"));
		defs.push((
			name.to_string(),
			value
				.parse::<i64>()
				.map_err(|_| format!("{d}: the value must be an integer"))?,
		));
	}
	let mut asm_out = None;
	if let Some(i) = a.iter().position(|x| x == "--asm") {
		if i + 1 >= a.len() {
			return Err("--asm needs a file".into());
		}
		asm_out = Some(a.remove(i + 1));
		a.remove(i);
	}
	let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
	let source = |p: &str| -> Result<Object, String> {
		let mut text = String::from_utf8(read(p)?).map_err(|_| format!("{p}: not UTF-8"))?;
		if p.ends_with(".sk") {
			let k = dsl::compile(&text, &defs).map_err(|e| format!("{p}: {e}"))?;
			for line in &k.report {
				println!("{p}: {line}");
			}
			text = k.text();
			if let Some(out) = &asm_out {
				std::fs::write(out, &text).map_err(|e| format!("{out}: {e}"))?;
			}
		} else if !defs.is_empty() || asm_out.is_some() {
			return Err("-D and --asm go with a .sk source".into());
		}
		npu::shave::assemble(&text).map_err(|e| format!("{p}: {e}"))
	};
	let (out, bytes) = match a.iter().map(String::as_str).collect::<Vec<_>>()[..] {
		["build", src, out] => (out, link::link_with(&[source(src)?], &entry, &comment)?),
		["as", src, out] => (out, source(src)?.write()),
		["ld", out, ref objs @ ..] if !objs.is_empty() => {
			let objs = objs.iter().map(|p| Object::read(&read(p)?).map_err(|e| format!("{p}: {e}"))).collect::<Result<Vec<_>, _>>()?;
			(out, link::link_with(&objs, &entry, &comment)?)
		}
		_ => return Err("usage: shave build IN.sk|IN.s OUT.elf [-DNAME=VALUE ...] [--asm OUT.s] | as IN.s OUT.o | ld OUT.elf IN.o ... [--entry NAME]".into()),
	};
	std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
	println!("{out}: {} bytes", bytes.len());
	Ok(())
}

fn main() {
	if let Err(e) = run(std::env::args().skip(1).collect()) {
		eprintln!("shave: {e}");
		std::process::exit(1);
	}
}
