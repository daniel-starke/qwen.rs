//! SHAVE linker: relocatable objects to the executable ELF an ACT-SHAVE
//! kernel is shipped as.
//!
//! The layout is fixed, the one the NPU's kernel loader expects:
//!
//! ```text
//! 0x1e000000  .arg.data : .arg.data sections, ALIGN(16), .data*
//! 0x1d000000  .text     : .text.entry, .text*, ALIGN(16), .gnu.linkonce.text.*,
//!                         ALIGN(16), .rodata*, ALIGN(16), .uuid.rodata*
//! ```
//!
//! Sections no relocation reaches from the entry symbol are dropped
//! (`--gc-sections`; `.text.entry`, `.arg.data` and `.uuid.rodata*` are
//! always kept).
//!
//! No archives: a kernel that calls into a library does not link here (the
//! symbol is reported as undefined).

use super::elf::{self, put32, Object, Shdr, Strings, Target};

pub const TEXT_ADDR: u32 = 0x1d00_0000;
pub const DATA_ADDR: u32 = 0x1e00_0000;
/// The `.comment` text.
pub const COMMENT: &str = "Linker: qwen.rs shave linker";
/// `-z max-page-size`: file offsets of loaded sections are congruent to
/// their addresses modulo this.
const PAGE: u32 = 16;

fn align(x: u32, a: u32) -> u32 {
	x.div_ceil(a.max(1)) * a.max(1)
}

/// Where the script puts an input section: (output section, rank in it).
fn place(name: &str) -> Option<(usize, u32)> {
	Some(match name {
		".arg.data" => (0, 0),
		n if n.starts_with(".data") => (0, 1),
		".text.entry" => (1, 0),
		n if n.starts_with(".text") => (1, 1),
		n if n.starts_with(".gnu.linkonce.text.") => (1, 2),
		n if n.starts_with(".rodata") => (1, 3),
		n if n.starts_with(".uuid.rodata") => (1, 4),
		_ => return None,
	})
}

struct Output {
	name: &'static str,
	addr: u32,
	flags: u32,
	align: u32,
	data: Vec<u8>,
}

/// Link `objects` into an executable whose entry point is the symbol `entry`.
pub fn link(objects: &[Object], entry: &str) -> Result<Vec<u8>, String> {
	link_with(objects, entry, COMMENT)
}

/// [`link`] with another `.comment` text.
pub fn link_with(objects: &[Object], entry: &str, comment: &str) -> Result<Vec<u8>, String> {
	let comment = [comment.as_bytes(), &[0]].concat();
	// global definitions: name -> (object, symbol)
	let mut defs: Vec<(&str, usize, usize)> = Vec::new();
	for (oi, o) in objects.iter().enumerate() {
		for (si, s) in o
			.symbols
			.iter()
			.enumerate()
			.filter(|(_, s)| s.section.is_some())
		{
			if defs.iter().any(|d| d.0 == s.name) {
				return Err(format!("symbol {} is defined twice", s.name));
			}
			defs.push((&s.name, oi, si));
		}
	}
	let find = |name: &str| defs.iter().find(|d| d.0 == name).map(|d| (d.1, d.2));
	// the section a relocation target lives in
	let resolve = |oi: usize, t: Target| -> Result<(usize, usize, u32), String> {
		match t {
			Target::Section(s) => Ok((oi, s, 0)),
			Target::Symbol(s) => {
				let sym = &objects[oi].symbols[s];
				let (o, i) = match sym.section {
					Some(_) => (oi, s),
					None => {
						find(&sym.name).ok_or_else(|| format!("undefined symbol {}", sym.name))?
					}
				};
				let d = &objects[o].symbols[i];
				Ok((o, d.section.unwrap(), d.value))
			}
		}
	};
	// keep what the entry reaches
	let (eo, es) = find(entry).ok_or_else(|| format!("undefined entry symbol {entry}"))?;
	let mut live: Vec<Vec<bool>> = objects
		.iter()
		.map(|o| vec![false; o.sections.len()])
		.collect();
	let mut todo = vec![(eo, objects[eo].symbols[es].section.unwrap())];
	for (oi, o) in objects.iter().enumerate() {
		for (si, s) in o.sections.iter().enumerate() {
			if matches!(s.name.as_str(), ".text.entry" | ".arg.data")
				|| s.name.starts_with(".uuid.rodata")
			{
				todo.push((oi, si));
			}
		}
	}
	while let Some((oi, si)) = todo.pop() {
		if std::mem::replace(&mut live[oi][si], true) {
			continue;
		}
		for r in &objects[oi].sections[si].relocs {
			let (o, s, _) = resolve(oi, r.target)?;
			todo.push((o, s));
		}
	}
	// layout
	let mut outs = [
		Output {
			name: ".arg.data",
			addr: DATA_ADDR,
			flags: elf::SHF_ALLOC,
			align: 1,
			data: Vec::new(),
		},
		Output {
			name: ".text",
			addr: TEXT_ADDR,
			flags: elf::SHF_ALLOC | elf::SHF_EXEC,
			align: 1,
			data: Vec::new(),
		},
	];
	let mut inputs: Vec<(usize, u32, usize, usize)> = Vec::new();
	for (oi, o) in objects.iter().enumerate() {
		for (si, s) in o
			.sections
			.iter()
			.enumerate()
			.filter(|&(si, _)| live[oi][si])
		{
			let (out, rank) = place(&s.name)
				.ok_or_else(|| format!("section {} has no place in a kernel", s.name))?;
			inputs.push((out, rank, oi, si));
		}
	}
	inputs.sort_by_key(|&(out, rank, _, _)| (out, rank));
	// address of every live input section
	let mut addr: Vec<Vec<u32>> = objects.iter().map(|o| vec![0; o.sections.len()]).collect();
	for (k, out) in outs.iter_mut().enumerate() {
		let mut rank = 0;
		for &(_, r, oi, si) in inputs.iter().filter(|i| i.0 == k) {
			let s = &objects[oi].sections[si];
			if r != rank {
				// the script's `. = ALIGN(16)` between the groups
				let n = align(out.data.len() as u32, 16);
				out.data.resize(n as usize, 0);
				rank = r;
			}
			let at = align(out.data.len() as u32, s.align);
			out.data.resize(at as usize, 0);
			addr[oi][si] = out.addr + at;
			out.data.extend_from_slice(&s.data);
			out.align = out.align.max(s.align);
			out.flags |= s.flags;
		}
		if k == 1 {
			// three `ALIGN(16)` follow the code
			let n = align(out.data.len() as u32, 16);
			out.data.resize(n as usize, 0);
		}
	}
	// relocations
	for &(k, _, oi, si) in &inputs {
		let s = &objects[oi].sections[si];
		for r in &s.relocs {
			let (o, t, value) = resolve(oi, r.target)?;
			let target = addr[o][t].wrapping_add(value).wrapping_add(r.addend as u32);
			let place = addr[oi][si] + r.offset;
			let at = (place - outs[k].addr) as usize;
			let nibble = matches!(r.kind, elf::R_LO16_N | elf::R_HI16_N | elf::R_BRANCH_N);
			let d = &mut outs[k].data;
			if at + 2 + nibble as usize > d.len() {
				return Err(format!(
					"relocation at {}+{:#x} is outside its section",
					s.name, r.offset
				));
			}
			let shift = if nibble { 4 } else { 0 };
			let old = (d[at] as u32
				| (d[at + 1] as u32) << 8
				| if nibble { (d[at + 2] as u32) << 16 } else { 0 })
				>> shift & 0xffff;
			let new = match r.kind {
				elf::R_LO16 | elf::R_LO16_N => target & 0xffff,
				elf::R_HI16 | elf::R_HI16_N => target >> 16,
				elf::R_BRANCH | elf::R_BRANCH_N => {
					let v = target as i64 - place as i64 + old as i16 as i64;
					if !(-0x8000..=0x7fff).contains(&v) {
						return Err(format!(
							"branch at {}+{:#x} is out of range ({v} bytes)",
							s.name, r.offset
						));
					}
					v as u32 & 0xffff
				}
				k => {
					return Err(format!(
						"unknown relocation type {k:#x} at {}+{:#x}",
						s.name, r.offset
					))
				}
			};
			let mask = 0xffffu32 << shift;
			let word = (d[at] as u32
				| (d[at + 1] as u32) << 8
				| if nibble { (d[at + 2] as u32) << 16 } else { 0 })
				& !mask | new << shift;
			d[at] = word as u8;
			d[at + 1] = (word >> 8) as u8;
			if nibble {
				d[at + 2] = (word >> 16) as u8;
			}
		}
	}
	// symbols of live sections, in input order
	let mut strtab = Strings::new();
	let mut symtab = Vec::new();
	elf::sym(&mut symtab, 0, 0, 0, 0, 0);
	let mut entry_addr = 0;
	for (oi, o) in objects.iter().enumerate() {
		for s in &o.symbols {
			let Some(si) = s.section.filter(|&si| live[oi][si]) else {
				continue;
			};
			let out = place(&o.sections[si].name).unwrap().0;
			let value = addr[oi][si] + s.value;
			if s.name == entry {
				entry_addr = value;
			}
			elf::sym(
				&mut symtab,
				strtab.add(&s.name),
				value,
				s.size,
				0x10 | s.kind as u8,
				out as u16 + 1,
			);
		}
	}
	// file: headers, room for two more program headers than are used, the
	// sections, the section headers
	let loads = outs.iter().filter(|o| !o.data.is_empty()).count() as u32;
	let phnum = loads + 1;
	let mut file = elf::ehdr(2, entry_addr, 52, 0, phnum as u16, 7, 5);
	let mut ph = Vec::new();
	let mut shstr = Strings::new();
	let mut sh = vec![Shdr {
		name: 0,
		kind: 0,
		flags: 0,
		addr: 0,
		offset: 0,
		size: 0,
		link: 0,
		info: 0,
		align: 0,
		entsize: 0,
	}];
	let mut at = 52 + 32 * (phnum + 2);
	let mut body: Vec<(u32, &[u8])> = Vec::new();
	for o in &outs {
		at = if o.data.is_empty() {
			align(at, o.align)
		} else {
			at + (o.addr.wrapping_sub(at)) % PAGE
		};
		if !o.data.is_empty() {
			for x in [
				1,
				at,
				o.addr,
				o.addr,
				o.data.len() as u32,
				o.data.len() as u32,
				if o.flags & elf::SHF_EXEC != 0 { 5 } else { 6 },
				PAGE,
			] {
				put32(&mut ph, x);
			}
		}
		sh.push(Shdr {
			name: shstr.add(o.name),
			kind: elf::SHT_PROGBITS,
			flags: o.flags,
			addr: o.addr,
			offset: at,
			size: o.data.len() as u32,
			link: 0,
			info: 0,
			align: o.align,
			entsize: 0,
		});
		body.push((at, &o.data));
		at += o.data.len() as u32;
	}
	for x in [0x6474_e551, 0, 0, 0, 0, 0, 6, 0] {
		put32(&mut ph, x); // PT_GNU_STACK
	}
	sh.push(Shdr {
		name: shstr.add(".comment"),
		kind: elf::SHT_PROGBITS,
		flags: 0x30,
		addr: 0,
		offset: at,
		size: comment.len() as u32,
		link: 0,
		info: 0,
		align: 1,
		entsize: 1,
	});
	body.push((at, &comment));
	at = align(at + comment.len() as u32, 4);
	sh.push(Shdr {
		name: shstr.add(".symtab"),
		kind: elf::SHT_SYMTAB,
		flags: 0,
		addr: 0,
		offset: at,
		size: symtab.len() as u32,
		link: 6,
		info: 1,
		align: 4,
		entsize: 16,
	});
	body.push((at, &symtab));
	at += symtab.len() as u32;
	let name = shstr.add(".shstrtab");
	let strname = shstr.add(".strtab");
	sh.push(Shdr {
		name,
		kind: elf::SHT_STRTAB,
		flags: 0,
		addr: 0,
		offset: at,
		size: shstr.0.len() as u32,
		link: 0,
		info: 0,
		align: 1,
		entsize: 0,
	});
	body.push((at, &shstr.0));
	at += shstr.0.len() as u32;
	sh.push(Shdr {
		name: strname,
		kind: elf::SHT_STRTAB,
		flags: 0,
		addr: 0,
		offset: at,
		size: strtab.0.len() as u32,
		link: 0,
		info: 0,
		align: 1,
		entsize: 0,
	});
	body.push((at, &strtab.0));
	let shoff = align(at + strtab.0.len() as u32, 4);
	file[32..36].copy_from_slice(&shoff.to_le_bytes());
	file.extend_from_slice(&ph);
	for (offset, bytes) in body {
		file.resize(offset as usize, 0);
		file.extend_from_slice(bytes);
	}
	file.resize(shoff as usize, 0);
	for s in &sh {
		s.write(&mut file);
	}
	Ok(file)
}

#[cfg(test)]
mod tests {
	use super::super::asm::assemble;
	use super::*;

	#[test]
	fn relocations_and_layout() {
		let src = ".code .text.entry\n.align 16\ncontrolled_act:\nNOP 6\nx:\nLSU0.LDIL i1 x\nLSU0.LDIH i1 x\nCMU.ALIGNVEC v5 v5 v5 0 || LSU0.LDIL i1 x\n\
                   CMU.ALIGNVEC v5 v5 v5 0 || LSU0.LDIH i1 x\nIAU.AND i1 i2 i3 || BRU.BRA x\nNOP 6\nBRU.BRA x\nNOP 6\nLSU0.LDIL i1 dat\nLSU0.LDIH i1 dat\n\
                   BRU.JMP i30\nNOP 6\n.data .data.foo\n.align 4\ndat:\n.int 0x11223344\n.short 7\n.byte 1, 2\n.float32 1.5\n";
		let e = link(&[assemble(src).unwrap()], "controlled_act").unwrap();
		// the expected bytes of the linked code
		let text = [
			0x50, 0x82, 0x0a, 0x01, 0x00, 0x82, 0x0b, 0x00, 0x1d, 0xb4, 0x58, 0x4a, 0x01, 0xa1,
			0x10, 0x00, 0x00, 0xb4, 0x58, 0x4a, 0x01, 0xb1, 0x00, 0xd0, 0x01, 0xae, 0x38, 0x50,
			0x84, 0x81, 0xfe, 0x1f, 0x00, 0x50, 0xc7, 0xdf, 0xff, 0x01, 0x50, 0x82, 0x0a, 0x00,
			0x00, 0x82, 0x0b, 0x00, 0x1e, 0x47, 0xf1, 0x01, 0x50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0,
		];
		assert_eq!(
			e[0xe0..0xec],
			[0x44, 0x33, 0x22, 0x11, 7, 0, 1, 2, 0, 0, 0xc0, 0x3f]
		);
		assert_eq!(e[0xf0..0x130], text);
		assert_eq!(e[24..28], TEXT_ADDR.to_le_bytes());
	}

	#[test]
	fn errors() {
		let o = assemble(".code .text.entry\nf:\nBRU.BRA g\nNOP 6\n").unwrap();
		assert!(link(std::slice::from_ref(&o), "f")
			.unwrap_err()
			.contains("undefined symbol g"));
		assert!(link(&[o], "main").unwrap_err().contains("entry"));
	}
}
