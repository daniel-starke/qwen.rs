//! ELF32 little-endian relocatable objects (`EM_SPARC`, the machine SHAVE
//! objects carry): the model the assembler fills and the linker reads.
//!
//! The writer lays an object out as sections packed without padding,
//! `.symtab`, `.strtab`, the `.rela` sections, `.shstrtab`, then the section
//! headers.

pub const EM_SPARC: u16 = 2;
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_RELA: u32 = 4;
pub const SHF_WRITE: u32 = 1;
pub const SHF_ALLOC: u32 = 2;
pub const SHF_EXEC: u32 = 4;

/// SHAVE relocation types. Each is a 16-bit field; the odd member of a pair
/// is the same field starting at bit 4 of its first byte (bundles are packed
/// by bits, and payloads are a multiple of 4 bits long).
pub const R_LO16: u8 = 0xc8;
pub const R_LO16_N: u8 = 0xc9;
pub const R_HI16: u8 = 0xca;
pub const R_HI16_N: u8 = 0xcb;
/// `S + A - P` added to the field (which holds the field's offset in its
/// bundle, so the result is the distance from the start of the bundle)
pub const R_BRANCH: u8 = 0xd1;
pub const R_BRANCH_N: u8 = 0xd2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymKind {
	NoType = 0,
	Object = 1,
	Func = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
	/// the start of a section of this object
	Section(usize),
	/// an entry of `Object::symbols`
	Symbol(usize),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reloc {
	pub offset: u32,
	pub kind: u8,
	pub target: Target,
	pub addend: i32,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Section {
	pub name: String,
	pub flags: u32,
	pub align: u32,
	pub data: Vec<u8>,
	pub relocs: Vec<Reloc>,
}

/// A global symbol; `section` is `None` while it is undefined.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Symbol {
	pub name: String,
	pub section: Option<usize>,
	pub value: u32,
	pub size: u32,
	pub kind: SymKind,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Object {
	pub sections: Vec<Section>,
	pub symbols: Vec<Symbol>,
}

pub(crate) fn put16(v: &mut Vec<u8>, x: u16) {
	v.extend_from_slice(&x.to_le_bytes());
}

pub(crate) fn put32(v: &mut Vec<u8>, x: u32) {
	v.extend_from_slice(&x.to_le_bytes());
}

/// A string table: `add` returns the offset of the name.
#[derive(Default)]
pub(crate) struct Strings(pub Vec<u8>);

impl Strings {
	pub fn new() -> Strings {
		Strings(vec![0])
	}

	pub fn add(&mut self, s: &str) -> u32 {
		let at = self.0.len() as u32;
		self.0.extend_from_slice(s.as_bytes());
		self.0.push(0);
		at
	}
}

pub(crate) struct Shdr {
	pub name: u32,
	pub kind: u32,
	pub flags: u32,
	pub addr: u32,
	pub offset: u32,
	pub size: u32,
	pub link: u32,
	pub info: u32,
	pub align: u32,
	pub entsize: u32,
}

impl Shdr {
	pub fn write(&self, v: &mut Vec<u8>) {
		for x in [
			self.name,
			self.kind,
			self.flags,
			self.addr,
			self.offset,
			self.size,
			self.link,
			self.info,
			self.align,
			self.entsize,
		] {
			put32(v, x);
		}
	}
}

pub(crate) fn ehdr(
	kind: u16,
	entry: u32,
	phoff: u32,
	shoff: u32,
	phnum: u16,
	shnum: u16,
	shstrndx: u16,
) -> Vec<u8> {
	let mut v = vec![0x7f, b'E', b'L', b'F', 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	put16(&mut v, kind);
	put16(&mut v, EM_SPARC);
	put32(&mut v, 1);
	put32(&mut v, entry);
	put32(&mut v, phoff);
	put32(&mut v, shoff);
	put32(&mut v, 0);
	put16(&mut v, 52);
	put16(&mut v, if phnum > 0 { 32 } else { 0 });
	put16(&mut v, phnum);
	put16(&mut v, 40);
	put16(&mut v, shnum);
	put16(&mut v, shstrndx);
	v
}

pub(crate) fn sym(v: &mut Vec<u8>, name: u32, value: u32, size: u32, info: u8, shndx: u16) {
	put32(v, name);
	put32(v, value);
	put32(v, size);
	v.push(info);
	v.push(0);
	put16(v, shndx);
}

impl Object {
	/// The relocatable ELF file.
	pub fn write(&self) -> Vec<u8> {
		let ns = self.sections.len();
		let mut shstr = Strings::new();
		let mut strtab = Strings::new();
		let mut body = Vec::new();
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
		let at = |body: &Vec<u8>| 52 + body.len() as u32;
		for s in &self.sections {
			sh.push(Shdr {
				name: shstr.add(&s.name),
				kind: SHT_PROGBITS,
				flags: s.flags,
				addr: 0,
				offset: at(&body),
				size: s.data.len() as u32,
				link: 0,
				info: 0,
				align: s.align,
				entsize: 0,
			});
			body.extend_from_slice(&s.data);
		}
		// null, one symbol per section, the globals
		let mut symtab = Vec::new();
		sym(&mut symtab, 0, 0, 0, 0, 0);
		for i in 0..ns {
			sym(&mut symtab, 0, 0, 0, 3, i as u16 + 1);
		}
		for s in &self.symbols {
			sym(
				&mut symtab,
				strtab.add(&s.name),
				s.value,
				s.size,
				0x10 | s.kind as u8,
				s.section.map_or(0, |i| i as u16 + 1),
			);
		}
		let (symndx, strndx) = (ns as u32 + 1, ns as u32 + 2);
		sh.push(Shdr {
			name: shstr.add(".symtab"),
			kind: SHT_SYMTAB,
			flags: 0,
			addr: 0,
			offset: at(&body),
			size: symtab.len() as u32,
			link: strndx,
			info: ns as u32 + 1,
			align: 8,
			entsize: 16,
		});
		body.extend_from_slice(&symtab);
		sh.push(Shdr {
			name: shstr.add(".strtab"),
			kind: SHT_STRTAB,
			flags: 0,
			addr: 0,
			offset: at(&body),
			size: strtab.0.len() as u32,
			link: 0,
			info: 0,
			align: 8,
			entsize: 0,
		});
		body.extend_from_slice(&strtab.0);
		for (i, s) in self
			.sections
			.iter()
			.enumerate()
			.filter(|(_, s)| !s.relocs.is_empty())
		{
			let mut rela = Vec::new();
			for r in &s.relocs {
				let index = match r.target {
					Target::Section(t) => t + 1,
					Target::Symbol(t) => ns + 1 + t,
				};
				put32(&mut rela, r.offset);
				put32(&mut rela, (index as u32) << 8 | r.kind as u32);
				put32(&mut rela, r.addend as u32);
			}
			sh.push(Shdr {
				name: shstr.add(&format!(".rela{}", s.name)),
				kind: SHT_RELA,
				flags: 0,
				addr: 0,
				offset: at(&body),
				size: rela.len() as u32,
				link: symndx,
				info: i as u32 + 1,
				align: 4,
				entsize: 12,
			});
			body.extend_from_slice(&rela);
		}
		let name = shstr.add(".shstrtab");
		sh.push(Shdr {
			name,
			kind: SHT_STRTAB,
			flags: 0,
			addr: 0,
			offset: at(&body),
			size: shstr.0.len() as u32,
			link: 0,
			info: 0,
			align: 8,
			entsize: 0,
		});
		body.extend_from_slice(&shstr.0);
		let mut out = ehdr(1, 0, 0, at(&body), 0, sh.len() as u16, sh.len() as u16 - 1);
		out.extend_from_slice(&body);
		for s in &sh {
			s.write(&mut out);
		}
		out
	}

	/// Read a relocatable object. Local symbols other
	/// than section symbols are resolved into their relocations' addends.
	pub fn read(file: &[u8]) -> Result<Object, String> {
		let u16at = |o: usize| {
			file.get(o..o + 2)
				.map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
				.ok_or("truncated ELF")
		};
		let u32at = |o: usize| {
			file.get(o..o + 4)
				.map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
				.ok_or("truncated ELF")
		};
		if file.len() < 52 || file[..7] != [0x7f, b'E', b'L', b'F', 1, 1, 1] {
			return Err("not an ELF32 little-endian file".into());
		}
		if u16at(16)? != 1 || u16at(18)? != EM_SPARC as usize {
			return Err("not a relocatable SHAVE (EM_SPARC) object".into());
		}
		let (shoff, shnum, shstrndx) = (u32at(32)? as usize, u16at(48)?, u16at(50)?);
		let hdr = |i: usize| -> Result<[u32; 10], String> {
			let mut h = [0u32; 10];
			for (k, x) in h.iter_mut().enumerate() {
				*x = u32at(shoff + 40 * i + 4 * k)?;
			}
			Ok(h)
		};
		let bytes = |h: &[u32; 10]| {
			file.get(h[4] as usize..(h[4] + h[5]) as usize)
				.ok_or("section outside the file")
		};
		let cstr = |tab: &[u8], at: usize| -> String {
			let end = tab[at.min(tab.len())..]
				.iter()
				.position(|&b| b == 0)
				.map_or(tab.len(), |p| at + p);
			String::from_utf8_lossy(&tab[at.min(tab.len())..end]).into_owned()
		};
		let headers = (0..shnum).map(hdr).collect::<Result<Vec<_>, _>>()?;
		let names = bytes(headers.get(shstrndx).ok_or("no section name table")?)?;
		let mut obj = Object::default();
		// ELF section index -> index in `obj.sections`
		let mut index = vec![None; shnum];
		for (i, h) in headers.iter().enumerate() {
			if h[1] == SHT_PROGBITS {
				index[i] = Some(obj.sections.len());
				obj.sections.push(Section {
					name: cstr(names, h[0] as usize),
					flags: h[2],
					align: h[8].max(1),
					data: bytes(h)?.to_vec(),
					relocs: Vec::new(),
				});
			} else if h[1] == 8 && h[5] > 0 {
				return Err(format!(
					"section {} (NOBITS) is not supported",
					cstr(names, h[0] as usize)
				));
			}
		}
		let Some(st) = headers.iter().find(|h| h[1] == SHT_SYMTAB) else {
			return Ok(obj);
		};
		let (symbytes, strs) = (
			bytes(st)?,
			bytes(headers.get(st[6] as usize).ok_or("no symbol names")?)?,
		);
		// ELF symbol index -> where it points
		enum Loc {
			None,
			At(usize, u32),
			Global(usize),
		}
		let mut locs = Vec::new();
		for e in symbytes.chunks_exact(16) {
			let (name, value, size) = (u32at_s(e, 0), u32at_s(e, 4), u32at_s(e, 8));
			let (info, shndx) = (e[12], u16::from_le_bytes([e[14], e[15]]) as usize);
			let section = if shndx == 0 {
				None
			} else {
				Some(
					index
						.get(shndx)
						.copied()
						.flatten()
						.ok_or("symbol in an unsupported section")?,
				)
			};
			locs.push(if info >> 4 == 0 {
				section.map_or(Loc::None, |s| Loc::At(s, value))
			} else {
				let kind = match info & 15 {
					1 => SymKind::Object,
					2 => SymKind::Func,
					_ => SymKind::NoType,
				};
				obj.symbols.push(Symbol {
					name: cstr(strs, name as usize),
					section,
					value,
					size,
					kind,
				});
				Loc::Global(obj.symbols.len() - 1)
			});
		}
		for h in headers.iter().filter(|h| h[1] == SHT_RELA) {
			let target = index
				.get(h[7] as usize)
				.copied()
				.flatten()
				.ok_or("relocations for an unsupported section")?;
			for e in bytes(h)?.chunks_exact(12) {
				let (offset, info, addend) = (u32at_s(e, 0), u32at_s(e, 4), u32at_s(e, 8) as i32);
				let (target_sym, addend) = match locs
					.get((info >> 8) as usize)
					.ok_or("relocation against a missing symbol")?
				{
					Loc::None => return Err("relocation against the null symbol".into()),
					Loc::At(s, v) => (Target::Section(*s), addend.wrapping_add(*v as i32)),
					Loc::Global(g) => (Target::Symbol(*g), addend),
				};
				obj.sections[target].relocs.push(Reloc {
					offset,
					kind: info as u8,
					target: target_sym,
					addend,
				});
			}
		}
		Ok(obj)
	}
}

fn u32at_s(b: &[u8], o: usize) -> u32 {
	u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
