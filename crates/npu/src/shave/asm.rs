//! SHAVE assembler: assembly (the kernel compiler's, [`super::dsl`], or
//! hand-written) to a relocatable object.
//!
//! ```text
//! .code .text.entry            // section (.data NAME for data)
//! .align 16
//! controlled_act:              // a label, on its own line
//!     LSU0.LDO.32 i9 i18 8     // one operation = one bundle = one cycle
//!     PEU.PC1C EQ              // a bundle of two: `||` joins operations
//!         || BRU.BRA .LBB0_13
//!     NOP 5
//! ```
//!
//! A bundle holds at most one operation per unit (`VAU`, `CMU`, `LSU0`,
//! `LSU1`, `SAU`, `IAU`, `BRU`, `PEU`) and is stored as a header naming each
//! unit's size class, then the operations' payloads packed by bits in that
//! unit order (see [`bundle`]). Labels that start with `.L` are local;
//! every other label becomes a global symbol.
//!
//! Not checked: that the code is valid for the processor (result latencies,
//! write-port conflicts between cycles, which operations may share a
//! bundle); [`super::conflicts::check`] finds conflicting pairs, and the
//! kernel compiler never emits them.

use super::elf::{self, Object, Reloc, Section, SymKind, Symbol, Target};
use super::isa::{Encoded, FixKind, Isa, Operand, Unit};
use std::collections::HashMap;

/// A symbol reference inside an encoded bundle.
#[derive(Debug, PartialEq)]
pub struct BundleFix {
	pub kind: FixKind,
	/// bit of the 16-bit field, from the start of the bundle
	pub bit: u32,
	pub symbol: String,
}

#[derive(Debug, PartialEq)]
pub struct Bundle {
	pub bytes: Vec<u8>,
	pub fixes: Vec<BundleFix>,
	/// bit of the delay-slot length byte, from the start of the bundle
	pub delay: Option<u32>,
}

/// One-byte headers: the low nibble names two units, bits 4-5 and 6-7 hold
/// their size classes (0 = absent). A single operation uses the nibble of
/// `SINGLE`.
const PAIRS: [(u8, Unit, Unit); 13] = [
	(0x1, Unit::Vau, Unit::Cmu),
	(0x2, Unit::Vau, Unit::Lsu0),
	(0x3, Unit::Vau, Unit::Peu),
	(0x4, Unit::Cmu, Unit::Lsu0),
	(0x5, Unit::Bru, Unit::Peu),
	(0x6, Unit::Cmu, Unit::Iau),
	(0x7, Unit::Cmu, Unit::Bru),
	(0x8, Unit::Cmu, Unit::Peu),
	(0x9, Unit::Lsu0, Unit::Lsu1),
	(0xa, Unit::Lsu0, Unit::Iau),
	(0xb, Unit::Lsu0, Unit::Peu),
	(0xc, Unit::Sau, Unit::Iau),
	(0xd, Unit::Iau, Unit::Peu),
];
const SINGLE: [(Unit, u8); 8] = [
	(Unit::Vau, 0x3),
	(Unit::Cmu, 0x1),
	(Unit::Lsu0, 0x2),
	(Unit::Lsu1, 0x9),
	(Unit::Sau, 0xc),
	(Unit::Iau, 0x6),
	(Unit::Bru, 0x7),
	(Unit::Peu, 0x5),
];
/// Two-byte headers `(n << 4) | 0xe`, then the classes of four units.
const QUADS: [[Unit; 4]; 15] = [
	[Unit::Vau, Unit::Cmu, Unit::Lsu0, Unit::Lsu1],
	[Unit::Vau, Unit::Cmu, Unit::Lsu0, Unit::Peu],
	[Unit::Vau, Unit::Cmu, Unit::Lsu0, Unit::Iau],
	[Unit::Vau, Unit::Cmu, Unit::Lsu0, Unit::Sau],
	[Unit::Vau, Unit::Cmu, Unit::Sau, Unit::Iau],
	[Unit::Vau, Unit::Lsu0, Unit::Lsu1, Unit::Iau],
	[Unit::Cmu, Unit::Lsu0, Unit::Lsu1, Unit::Iau],
	[Unit::Cmu, Unit::Lsu0, Unit::Lsu1, Unit::Peu],
	[Unit::Cmu, Unit::Lsu0, Unit::Lsu1, Unit::Sau],
	[Unit::Cmu, Unit::Lsu0, Unit::Sau, Unit::Iau],
	[Unit::Cmu, Unit::Iau, Unit::Bru, Unit::Peu],
	[Unit::Lsu0, Unit::Lsu1, Unit::Sau, Unit::Iau],
	[Unit::Lsu0, Unit::Lsu1, Unit::Iau, Unit::Bru],
	[Unit::Lsu0, Unit::Lsu1, Unit::Iau, Unit::Peu],
	[Unit::Lsu0, Unit::Sau, Unit::Iau, Unit::Peu],
];

/// The header for these size classes (indexed by `Unit as usize`, 0 =
/// absent): the shortest form that can name every unit present.
fn header(class: &[u8; 8]) -> Vec<u8> {
	let c = |u: Unit| class[u as usize];
	let present: Vec<Unit> = Unit::ALL.into_iter().filter(|&u| c(u) != 0).collect();
	let pair = match present[..] {
		[u] => SINGLE
			.iter()
			.find(|s| s.0 == u)
			.and_then(|s| PAIRS.iter().find(|p| p.0 == s.1)),
		[a, b] => PAIRS
			.iter()
			.find(|p| (p.1 == a && p.2 == b) || (p.1 == b && p.2 == a)),
		_ => None,
	};
	if let Some(&(nibble, a, b)) = pair {
		return vec![nibble | c(a) << 4 | c(b) << 6];
	}
	if let Some((n, q)) = QUADS
		.iter()
		.enumerate()
		.find(|(_, q)| present.iter().all(|u| q.contains(u)))
	{
		return vec![
			(n as u8) << 4 | 0xe,
			c(q[0]) | c(q[1]) << 2 | c(q[2]) << 4 | c(q[3]) << 6,
		];
	}
	vec![
		0x0f,
		c(Unit::Vau) | c(Unit::Cmu) << 2 | c(Unit::Lsu0) << 4 | c(Unit::Lsu1) << 6,
		c(Unit::Sau) | c(Unit::Iau) << 2 | c(Unit::Bru) << 4 | c(Unit::Peu) << 6,
	]
}

/// Encode one bundle from its operations.
pub fn bundle(ops: &[Encoded]) -> Result<Bundle, String> {
	let mut class = [0u8; 8];
	for op in ops {
		if class[op.unit as usize] != 0 {
			return Err(format!("{:?} is used twice in one bundle", op.unit));
		}
		class[op.unit as usize] = op.class;
	}
	let mut bytes = header(&class);
	let mut bit = 8 * bytes.len() as u32;
	let mut out = Bundle {
		bytes: Vec::new(),
		fixes: Vec::new(),
		delay: None,
	};
	for unit in Unit::ALL {
		let Some(op) = ops.iter().find(|op| op.unit == unit) else {
			continue;
		};
		let mut payload = op.payload;
		if let Some(f) = &op.fix {
			let at = bit + f.bit;
			if f.kind == FixKind::Branch {
				// the linker adds `target - field address`: with the field's own
				// offset stored, the result counts from the start of the bundle
				payload |= ((at / 8) as u64) << f.bit;
			}
			out.fixes.push(BundleFix {
				kind: f.kind,
				bit: at,
				symbol: f.symbol.clone(),
			});
		}
		if let Some(d) = op.delay {
			out.delay = Some(bit + d);
		}
		for k in 0..op.bits {
			if bit.is_multiple_of(8) {
				bytes.push(0);
			}
			*bytes.last_mut().unwrap() |= ((payload >> k) as u8 & 1) << (bit % 8);
			bit += 1;
		}
	}
	out.bytes = bytes;
	Ok(out)
}

/// `NOP` (no count) is the one-byte 0x07; `NOP n` is one byte per 16 cycles,
/// `(cycles - 1) << 4`. Returns (byte, cycles) pairs.
fn nop(count: Option<i64>) -> Result<Vec<(u8, u32)>, String> {
	let Some(mut n) = count else {
		return Ok(vec![(0x07, 1)]);
	};
	if !(1..=4096).contains(&n) {
		return Err(format!("NOP {n}: the count must be 1..4096"));
	}
	let mut v = Vec::new();
	while n > 0 {
		let k = n.min(16);
		v.push((((k - 1) as u8) << 4, k as u32));
		n -= k;
	}
	Ok(v)
}

pub(crate) fn operand(t: &str) -> Operand {
	let b = t.as_bytes();
	if let Some(p) = t.strip_prefix('[').and_then(|p| p.strip_suffix(']')) {
		return Operand::Pattern(p.to_string());
	}
	if b.len() > 1 && matches!(b[0], b'i' | b'v' | b'I' | b'V') {
		let (num, elem) = t[1..]
			.split_once('.')
			.map_or((&t[1..], None), |(n, e)| (n, Some(e)));
		let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
		if digits(num) && elem.is_none_or(digits) {
			if let (Ok(n), Ok(elem)) = (num.parse(), elem.map(|e| e.parse()).transpose()) {
				return Operand::Reg {
					vector: matches!(b[0], b'v' | b'V'),
					n,
					elem,
				};
			}
		}
	}
	if let Some(v) = number(t) {
		return Operand::Int(v);
	}
	if t.contains('.') && t.parse::<f64>().is_ok() {
		return Operand::Float(t.to_string());
	}
	Operand::Ident(t.to_string())
}

fn number(t: &str) -> Option<i64> {
	let (neg, body) = t.strip_prefix('-').map_or((false, t), |b| (true, b));
	let v = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
		Some(h) => i64::from_str_radix(h, 16).ok()?,
		None => body.parse::<i64>().ok()?,
	};
	Some(if neg { -v } else { v })
}

fn is_label_name(s: &str) -> bool {
	!s.is_empty()
		&& !s.as_bytes()[0].is_ascii_digit()
		&& s.bytes()
			.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'$'))
}

/// One cycle's worth of code in a section, for the delay-slot lengths.
struct Step {
	offset: u32,
	len: u32,
	cycles: u32,
	/// bit of the delay-length byte, from the start of the section
	delay: Option<u32>,
}

struct Pending {
	section: usize,
	offset: u32,
	kind: u8,
	symbol: String,
	line: usize,
}

#[derive(Default)]
struct State {
	sections: Vec<Section>,
	steps: Vec<Vec<Step>>,
	current: Option<usize>,
	labels: HashMap<String, (usize, u32)>,
	/// global names in the order they first appear
	globals: Vec<String>,
	types: HashMap<String, SymKind>,
	sizes: Vec<(String, String, usize)>,
	pending: Vec<Pending>,
}

impl State {
	fn section(&mut self, name: &str, flags: u32) -> usize {
		let i = self
			.sections
			.iter()
			.position(|s| s.name == name)
			.unwrap_or_else(|| {
				self.sections.push(Section {
					name: name.to_string(),
					flags,
					align: 1,
					data: Vec::new(),
					relocs: Vec::new(),
				});
				self.steps.push(Vec::new());
				self.sections.len() - 1
			});
		self.current = Some(i);
		i
	}

	fn here(&mut self) -> usize {
		match self.current {
			Some(i) => i,
			None => self.section(".text", elf::SHF_ALLOC | elf::SHF_EXEC),
		}
	}

	fn global(&mut self, name: &str) {
		if !name.starts_with(".L") && !self.globals.iter().any(|g| g == name) {
			self.globals.push(name.to_string());
		}
	}

	/// The value of `A`, `A-B` or a number (labels of one section).
	fn value(&self, expr: &str) -> Result<u32, String> {
		let term = |t: &str| -> Result<(Option<usize>, i64), String> {
			let t = t.trim();
			match number(t) {
				Some(v) => Ok((None, v)),
				None => self
					.labels
					.get(t)
					.map(|&(s, o)| (Some(s), o as i64))
					.ok_or_else(|| format!("undefined label {t}")),
			}
		};
		// a leading '-' belongs to a number, not to a difference
		match expr.trim().char_indices().skip(1).find(|&(_, c)| c == '-') {
			Some((at, _)) => {
				let (a, b) = (term(&expr.trim()[..at])?, term(&expr.trim()[at + 1..])?);
				if a.0 != b.0 {
					return Err(format!("{expr}: the labels are in different sections"));
				}
				Ok((a.1 - b.1) as u32)
			}
			None => Ok(term(expr)?.1 as u32),
		}
	}
}

fn data_values(args: &str, size: usize, out: &mut Vec<u8>) -> Result<(), String> {
	for a in args.split(',').map(str::trim).filter(|a| !a.is_empty()) {
		let v = number(a).ok_or_else(|| format!("{a:?} is not a number"))?;
		let (lo, hi) = (-(1i64 << (8 * size - 1)), (1i64 << (8 * size)) - 1);
		if v < lo || v > hi {
			return Err(format!("{v} does not fit {size} byte(s)"));
		}
		out.extend_from_slice(&v.to_le_bytes()[..size]);
	}
	Ok(())
}

/// Remove `/* */` and `//` comments, keeping the line structure.
pub(crate) fn strip_comments(src: &str) -> String {
	let mut out = String::with_capacity(src.len());
	let (mut rest, mut block) = (src, false);
	while !rest.is_empty() {
		if block {
			match rest.find("*/") {
				Some(e) => {
					out.extend(rest[..e].chars().filter(|&c| c == '\n'));
					rest = &rest[e + 2..];
					block = false;
				}
				None => break,
			}
		} else {
			let (a, b) = (rest.find("/*"), rest.find("//"));
			match (a, b) {
				(Some(a), b) if b.is_none_or(|b| a < b) => {
					out.push_str(&rest[..a]);
					rest = &rest[a + 2..];
					block = true;
				}
				(_, Some(b)) => {
					out.push_str(&rest[..b]);
					rest = rest[b..].find('\n').map_or("", |n| &rest[b + n..]);
				}
				_ => {
					out.push_str(rest);
					break;
				}
			}
		}
	}
	out
}

/// Assemble a source file into a relocatable object.
pub fn assemble(src: &str) -> Result<Object, String> {
	assemble_with(&Isa::new(), src)
}

pub fn assemble_with(isa: &Isa, src: &str) -> Result<Object, String> {
	let text = strip_comments(src);
	// statements: a bundle's `||` lines are joined to its first line
	let mut lines: Vec<(usize, String)> = Vec::new();
	for (n, l) in text.lines().enumerate() {
		let l = l.trim();
		if l.is_empty() {
			continue;
		}
		match (l.strip_prefix("||"), lines.last_mut()) {
			(Some(more), Some(last)) if !last.1.starts_with('.') && !last.1.ends_with(':') => {
				last.1.push_str(" || ");
				last.1.push_str(more.trim());
			}
			(Some(_), _) => {
				return Err(format!("line {}: `||` does not follow an operation", n + 1))
			}
			_ => lines.push((n + 1, l.to_string())),
		}
	}
	let mut st = State::default();
	for (line, l) in &lines {
		let at = |e: String| format!("line {line}: {e}");
		if let Some(name) = l
			.strip_suffix(':')
			.map(str::trim)
			.filter(|n| is_label_name(n))
		{
			let s = st.here();
			let offset = st.sections[s].data.len() as u32;
			if st.labels.insert(name.to_string(), (s, offset)).is_some() {
				return Err(at(format!("label {name} is defined twice")));
			}
			st.global(name);
			continue;
		}
		if l.starts_with('.') {
			let (dir, args) = l
				.split_once(char::is_whitespace)
				.map_or((l.as_str(), ""), |(d, a)| (d, a.trim()));
			match dir {
				".nowarn" | ".nowarnend" | ".file" | ".version" => {}
				".end" => break,
				".code" | ".data" => {
					if !is_label_name(args) {
						return Err(at(format!("{dir} needs a section name")));
					}
					st.section(
						args,
						elf::SHF_ALLOC
							| if dir == ".code" {
								elf::SHF_EXEC
							} else {
								elf::SHF_WRITE
							},
					);
				}
				".align" => {
					let n = number(args)
						.filter(|n| (1..=4096).contains(n) && (n & (n - 1)) == 0)
						.ok_or_else(|| at(format!(".align {args}: not a power of two")))?
						as u32;
					let s = st.here();
					let sec = &mut st.sections[s];
					sec.align = sec.align.max(n);
					while !(sec.data.len() as u32).is_multiple_of(n) {
						sec.data.push(0);
					}
				}
				".type" => {
					let (name, kind) = args
						.split_once(',')
						.map(|(a, b)| (a.trim(), b.trim()))
						.ok_or_else(|| at(".type needs `name,@kind`".into()))?;
					let kind = match kind {
						"@function" => SymKind::Func,
						"@object" => SymKind::Object,
						k => return Err(at(format!(".type {k}: not @function or @object"))),
					};
					st.types.insert(name.to_string(), kind);
				}
				".size" => {
					let (name, expr) = args
						.split_once(',')
						.map(|(a, b)| (a.trim(), b.trim()))
						.ok_or_else(|| at(".size needs `name, expression`".into()))?;
					st.sizes.push((name.to_string(), expr.to_string(), *line));
				}
				".byte" | ".short" | ".int" | ".long" => {
					let s = st.here();
					let size = [1, 2, 4, 4][[".byte", ".short", ".int", ".long"]
						.iter()
						.position(|d| *d == dir)
						.unwrap()];
					data_values(args, size, &mut st.sections[s].data).map_err(at)?;
				}
				".float32" => {
					let s = st.here();
					for a in args.split(',').map(str::trim).filter(|a| !a.is_empty()) {
						let v: f32 = a.parse().map_err(|_| at(format!("{a:?} is not a float")))?;
						st.sections[s].data.extend_from_slice(&v.to_le_bytes());
					}
				}
				d => return Err(at(format!("unsupported directive {d}"))),
			}
			continue;
		}
		let s = st.here();
		if st.sections[s].flags & elf::SHF_EXEC == 0 {
			return Err(at("an operation in a data section".into()));
		}
		let mut ops = Vec::new();
		let mut nops = None;
		for part in l.split("||").map(str::trim) {
			let mut tok = part
				.split(|c: char| c.is_whitespace() || c == ',')
				.filter(|t| !t.is_empty());
			let mnemonic = tok.next().ok_or_else(|| at("empty operation".into()))?;
			let operands: Vec<Operand> = tok.map(operand).collect();
			if ["NOP", "BRU.NOP", "BRU.INC"]
				.iter()
				.any(|n| mnemonic.eq_ignore_ascii_case(n))
			{
				let count = match operands[..] {
					[] => None,
					[Operand::Int(n)] => Some(n),
					_ => return Err(at("NOP takes one count".into())),
				};
				nops = Some(nop(count).map_err(at)?);
				continue;
			}
			ops.push(isa.encode(mnemonic, &operands).map_err(at)?);
		}
		let start = st.sections[s].data.len() as u32;
		if let Some(nops) = nops {
			if !ops.is_empty() {
				return Err(at("NOP cannot share a bundle".into()));
			}
			for (k, (byte, cycles)) in nops.into_iter().enumerate() {
				st.steps[s].push(Step {
					offset: start + k as u32,
					len: 1,
					cycles,
					delay: None,
				});
				st.sections[s].data.push(byte);
			}
			continue;
		}
		let b = bundle(&ops).map_err(at)?;
		for f in b.fixes {
			let nibble = match f.bit % 8 {
				0 => 0,
				4 => 1,
				_ => return Err(at("a symbol field is not on a 4-bit boundary".into())),
			};
			let kind = match f.kind {
				FixKind::Lo16 => elf::R_LO16,
				FixKind::Hi16 => elf::R_HI16,
				FixKind::Branch => elf::R_BRANCH,
			} + nibble;
			st.global(&f.symbol);
			st.pending.push(Pending {
				section: s,
				offset: start + f.bit / 8,
				kind,
				symbol: f.symbol,
				line: *line,
			});
		}
		st.steps[s].push(Step {
			offset: start,
			len: b.bytes.len() as u32,
			cycles: 1,
			delay: b.delay.map(|d| 8 * start + d),
		});
		st.sections[s].data.extend_from_slice(&b.bytes);
	}
	delay_slots(&mut st);
	finish(st)
}

/// A branch takes effect after 6 more cycles; its encoding holds the byte
/// length of the code in them. The field is 0 for a branch that is itself
/// in the delay slots of an earlier one, and when fewer than 6 cycles
/// follow in the section.
fn delay_slots(st: &mut State) {
	for (steps, sec) in st.steps.iter().zip(st.sections.iter_mut()) {
		let mut covered = 0; // steps before this index are in an open branch's delay slots
		for (i, step) in steps.iter().enumerate() {
			let Some(bit) = step.delay else { continue };
			if i < covered {
				continue;
			}
			let (mut cycles, mut len, mut j) = (0, 0, i + 1);
			while cycles < 6 && j < steps.len() {
				cycles += steps[j].cycles;
				len += steps[j].len;
				j += 1;
			}
			if cycles < 6 {
				continue;
			}
			covered = j;
			debug_assert!(step.offset * 8 <= bit && len < 256);
			for k in 0..8 {
				let at = (bit + k) as usize;
				sec.data[at / 8] |= ((len >> k) as u8 & 1) << (at % 8);
			}
		}
	}
}

fn finish(st: State) -> Result<Object, String> {
	let mut obj = Object {
		sections: st.sections,
		symbols: Vec::new(),
	};
	for name in &st.globals {
		let def = st.labels.get(name);
		let section = def.map(|d| d.0);
		let kind = st.types.get(name).copied().unwrap_or(match section {
			Some(s) if obj.sections[s].flags & elf::SHF_EXEC == 0 => SymKind::Object,
			_ => SymKind::NoType,
		});
		obj.symbols.push(Symbol {
			name: name.clone(),
			section,
			value: def.map_or(0, |d| d.1),
			size: 0,
			kind,
		});
	}
	let labels = &st.labels;
	let value = |expr: &str| {
		State {
			labels: labels.clone(),
			..State::default()
		}
		.value(expr)
	};
	for (name, expr, line) in &st.sizes {
		let size = value(expr).map_err(|e| format!("line {line}: .size {name}: {e}"))?;
		obj.symbols
			.iter_mut()
			.find(|s| s.name == *name)
			.ok_or_else(|| format!("line {line}: .size of the unknown symbol {name}"))?
			.size = size;
	}
	for p in st.pending {
		let (target, addend) = if p.symbol.starts_with(".L") {
			let &(s, o) = st
				.labels
				.get(&p.symbol)
				.ok_or_else(|| format!("line {}: undefined label {}", p.line, p.symbol))?;
			(Target::Section(s), o as i32)
		} else {
			(
				Target::Symbol(st.globals.iter().position(|g| *g == p.symbol).unwrap()),
				0,
			)
		};
		obj.sections[p.section].relocs.push(Reloc {
			offset: p.offset,
			kind: p.kind,
			target,
			addend,
		});
	}
	Ok(obj)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn text(src: &str) -> Vec<u8> {
		assemble(&format!(".code .text.entry\n{src}\n"))
			.unwrap()
			.sections[0]
			.data
			.clone()
	}

	#[test]
	fn single_operations() {
		assert_eq!(text("IAU.ADD i1 i2 i3"), [0xc6, 0x40, 0x84, 0x01]);
		assert_eq!(
			text("LSU0.LDO.64 v1.0 i8 32"),
			[0xc2, 0x12, 0x50, 0x40, 0x00]
		);
		assert_eq!(text("CMU.CMZ.i32 i9"), [0x41, 0x90, 0x00]);
		assert_eq!(
			text("LSU0.SWZC8 [32107654]"),
			[0xc2, 0x2f, 0xac, 0x8f, 0x68]
		);
		assert_eq!(
			text("VAU.MUL.f32 v2 v2 0.5"),
			text("vau.mul.F32 v2,v2,0.50")
		);
	}

	/// `isa.samples`: two lines of every form of the table. Each assembles
	/// (followed by `NOP 12`), and the encodings, hashed per unit, stay what
	/// they are now. A regression check: the ground truth is the device
	/// (`scripts/sktest.py`, `scripts/shavedev.py`).
	#[test]
	fn every_form_is_stable() {
		let isa = Isa::new();
		let mut hash: std::collections::BTreeMap<String, (u32, u64)> = Default::default();
		for src in include_str!("isa.samples")
			.lines()
			.filter(|l| !l.starts_with('#') && !l.trim().is_empty())
		{
			let obj = assemble_with(&isa, &format!(".code .text.entry\n{src}\nNOP 12\n"))
				.unwrap_or_else(|e| panic!("{src}: {e}"));
			let got = &obj.sections[0].data;
			let unit = src.split('.').next().unwrap().to_string();
			let h = hash.entry(unit).or_insert((0, 0xcbf2_9ce4_8422_2325));
			h.0 += 1;
			for &b in &got[..got.len() - 1] {
				h.1 = (h.1 ^ b as u64).wrapping_mul(0x100_0000_01b3);
			}
		}
		let got: Vec<(&str, u32, u64)> =
			hash.iter().map(|(u, &(n, h))| (u.as_str(), n, h)).collect();
		assert_eq!(got, STABLE, "{got:#x?}");
	}

	/// (unit, sample lines, FNV-1a of their encodings)
	const STABLE: &[(&str, u32, u64)] = &[
		("BRU", 38, 0xdfcf_e7ad_4aaa_cd7f),
		("CMU", 1368, 0xd573_9dee_de78_e314),
		("IAU", 126, 0x800c_cda7_c1ee_84cb),
		("LSU0", 1169, 0x6a5a_a737_10c5_8af4),
		("LSU1", 1169, 0x0eed_c814_5ab7_1bb4),
		("PEU", 179, 0x36ed_de8a_a6b2_e80c),
		("SAU", 628, 0x8634_af00_a6bb_735f),
		("VAU", 480, 0x2e3f_437a_19b6_81a7),
	];

	#[test]
	fn nops() {
		assert_eq!(
			text("BRU.NOP\nBRU.INC 2\nbru.nop 17"),
			text("NOP\nNOP 2\nNOP 17")
		);
		assert_eq!(
			text("NOP\nNOP 2\nNOP 6\nNOP 17\nNOP 40"),
			[0x07, 0x10, 0x50, 0xf0, 0x00, 0xf0, 0xf0, 0x70]
		);
	}

	#[test]
	fn bundles_pack_by_bits() {
		// two byte-sized payloads behind a two-byte header
		assert_eq!(
			text("VAU.ADD.f16 v1 v2 v3 || IAU.ADD i1 i2 i3"),
			[0x2e, 0xc3, 0x40, 0x84, 0x61, 0x40, 0x84, 0x01]
		);
		assert_eq!(
			text("IAU.ADD i1 i2 i3\n || VAU.ADD.f16 v1 v2 v3"),
			text("VAU.ADD.f16 v1 v2 v3 || IAU.ADD i1 i2 i3")
		);
		// a 28-bit CMU payload: the LSU0 payload starts in the middle of a byte
		assert_eq!(
			text("LSU0.SWZC8 [32107654] || CMU.ALIGNVEC v5 v5 v5 0"),
			[0xf4, 0x58, 0x4a, 0x01, 0xf1, 0xc2, 0xfa, 0x88, 0x06]
		);
		assert_eq!(
			text("PEU.PC1C EQ || BRU.JMP i30\nNOP 6"),
			[0x95, 0xf1, 0x01, 0x1b, 0x50]
		);
		// the short classes: PEU 4 bits, SAU / IAU 16, BRU 20
		assert_eq!(
			text("LSU0.LD.128.F16.F32 v1 i2 || PEU.ANDACC"),
			[0x5b, 0x10, 0x04, 0x24]
		);
		assert_eq!(
			text("SAU.NOT i31 i31 || IAU.NOT i1 i2"),
			[0x5c, 0xf8, 0x7f, 0x50, 0x04]
		);
		assert_eq!(
			text("BRU.RPL i31 i31 || PEU.PC1C EQ"),
			[0xa5, 0xff, 0x0f, 0xb0, 0x01]
		);
		assert_eq!(
			text("IAU.NOT i31 i31 || BRU.JMP i30\nNOP 12")[..6],
			[0xae, 0x14, 0xf0, 0x7f, 0xf1, 0x01]
		);
		assert!(assemble(".code .text\nIAU.ADD i1 i2 i3 || IAU.SUB i1 i2 i3").is_err());
	}

	#[test]
	fn delay_slot_lengths() {
		let a = "IAU.ADD i1 i2 i3";
		// six cycles follow: a NOP 5 (one byte) and one operation
		assert_eq!(
			text(&format!("BRU.JMP i30\nNOP 5\n{a}"))[..3],
			[0x47, 0xf1, 0x05]
		);
		// a NOP 20 is two bytes; the first covers the six cycles
		assert_eq!(text("BRU.JMP i30\nNOP 20")[..3], [0x47, 0xf1, 0x01]);
		// the second branch is in the first one's delay slots; the third is not
		let t = text(&format!("BRU.JMP i30\n{a}\n{a}\n{a}\n{a}\n{a}\nBRU.JMP i30\n{a}\nBRU.JMP i30\n{a}\n{a}\n{a}\n{a}\n{a}\n{a}"));
		assert_eq!((t[2], t[25], t[32]), (23, 0, 24));
		// fewer than six cycles left in the section
		assert_eq!(text("BRU.JMP i30\nNOP\nNOP")[..3], [0x47, 0xf1, 0x00]);
	}

	#[test]
	fn symbols_and_relocations() {
		let src = ".code .text.entry\n.align 16\ncontrolled_act:\nNOP 6\nx:\nLSU0.LDIL i1 x\nCMU.ALIGNVEC v5 v5 v5 0 || LSU0.LDIH i1 x\n\
                   IAU.AND i1 i2 i3 || BRU.BRA .Lend\nNOP 6\n.Lend:\nLSU0.LDIL i1 dat\n.data .data.foo\n.align 4\ndat:\n.int 0x11223344\n.short 7\n.byte 1, 2\n.float32 1.5\n";
		let o = assemble(src).unwrap();
		assert_eq!(
			o.sections[1].data,
			[0x44, 0x33, 0x22, 0x11, 7, 0, 1, 2, 0, 0, 0xc0, 0x3f]
		);
		assert_eq!((o.sections[0].align, o.sections[1].align), (16, 4));
		let names: Vec<_> = o
			.symbols
			.iter()
			.map(|s| (s.name.as_str(), s.section, s.value, s.kind))
			.collect();
		assert_eq!(
			names,
			[
				("controlled_act", Some(0), 0, SymKind::NoType),
				("x", Some(0), 1, SymKind::NoType),
				("dat", Some(1), 0, SymKind::Object)
			]
		);
		let r: Vec<_> = o.sections[0]
			.relocs
			.iter()
			.map(|r| (r.offset, r.kind, r.target, r.addend))
			.collect();
		assert_eq!(
			r,
			[
				(3, elf::R_LO16, Target::Symbol(1), 0),
				(10, elf::R_HI16_N, Target::Symbol(1), 0),
				(17, elf::R_BRANCH_N, Target::Section(0), 22),
				(24, elf::R_LO16, Target::Symbol(2), 0),
			]
		);
		assert_eq!(Object::read(&o.write()).unwrap(), o);
	}
}
