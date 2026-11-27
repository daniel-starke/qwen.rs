//! The SHAVE (NPU3720 ACT-SHAVE) instruction encodings, from `isa.tbl`.
//!
//! One table line per instruction form (mnemonic + operand kinds):
//!
//! ```text
//! MNEMONIC kind ... | UNIT class nbits base | slot ; slot ; ...
//! ```
//!
//! `class` is the unit's 2-bit size class in the bundle header (it fixes the
//! payload's length in bits, [`Unit::bits`]; `nbits` is only the length of
//! the bytes the probe saw). `base` is the form's payload with every integer
//! operand 0 and every enumerated operand at its zero entry; each slot XORs
//! bits into it. An
//! integer slot is `<operand><kind> lo..hi @pos` (a plain field) or `lo..hi
//! m0,m1,...` (one mask per value bit, two's complement); an enumerated slot
//! is `<operand><kind> symbol=mask ...`. Kinds: `r` register number, `e`
//! element index (`v3.1`), `#` immediate, `#l` / `#h` the low / high half of
//! a 32-bit value or symbol, `C` condition, `N` named register (`V_ACC0`,
//! `C_CMU0`, `T18`), `F` float constant, `pN` character N of a `[....]`
//! pattern, `L` branch target. `D @pos` marks the byte that holds the length
//! of the branch's delay slots. Several lines can share a mnemonic and its
//! operand kinds (a short and a long immediate): the first that takes the
//! operands is used.
//!
//! The table is a statement of functional facts of the processor's
//! instruction set (1495 mnemonics), kept for interoperability: which
//! operations exist, their operands, and where each operand's bits go
//! (`crates/npu/NPU.md` "Provenance"). `isa.samples` holds two lines per
//! form for the regression test of [`super::asm`]. Not in it: `NOP`, the
//! label operands other than `BRU.BRA`'s and `LDIL` / `LDIH`'s.

/// The eight units of a bundle, in the order their payloads are stored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unit {
	Vau,
	Cmu,
	Lsu0,
	Lsu1,
	Sau,
	Iau,
	Bru,
	Peu,
}

impl Unit {
	pub const ALL: [Unit; 8] = [
		Unit::Vau,
		Unit::Cmu,
		Unit::Lsu0,
		Unit::Lsu1,
		Unit::Sau,
		Unit::Iau,
		Unit::Bru,
		Unit::Peu,
	];

	fn parse(s: &str) -> Option<Unit> {
		Some(match s {
			"VAU" => Unit::Vau,
			"CMU" => Unit::Cmu,
			"LSU0" => Unit::Lsu0,
			"LSU1" => Unit::Lsu1,
			"SAU" => Unit::Sau,
			"IAU" => Unit::Iau,
			"BRU" => Unit::Bru,
			"PEU" => Unit::Peu,
			_ => return None,
		})
	}

	/// Payload bits of this unit's size class 1..=3 (the 2-bit field of the
	/// bundle header).
	pub fn bits(self, class: u8) -> u32 {
		let t: [u32; 3] = match self {
			Unit::Vau => [16, 20, 24],
			Unit::Cmu => [16, 24, 28],
			Unit::Lsu0 | Unit::Lsu1 => [20, 24, 32],
			Unit::Sau | Unit::Iau => [16, 20, 24],
			Unit::Bru => [16, 20, 24],
			Unit::Peu => [4, 8, 12],
		};
		t[class as usize - 1]
	}
}

/// One operand as written in the source.
#[derive(Clone, PartialEq, Debug)]
pub enum Operand {
	/// `i5`, `v3`, `v3.1`
	Reg {
		vector: bool,
		n: u32,
		elem: Option<u32>,
	},
	Int(i64),
	/// `0.5`: kept as text, compared by value
	Float(String),
	/// `[32107654]` without the brackets
	Pattern(String),
	/// a condition, a named register or a symbol
	Ident(String),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
	Reg,
	Elem,
	Imm,
	ImmLo,
	ImmHi,
	Cond,
	Name,
	Float,
	Pat(usize),
	Label,
}

#[derive(Debug)]
enum Map {
	Int {
		lo: i64,
		hi: i64,
		masks: Vec<u64>,
	},
	Enum(Vec<(String, u64)>),
	/// a 16-bit field at this payload bit
	Field(u32),
}

#[derive(Debug)]
struct Slot {
	operand: usize,
	kind: Kind,
	map: Map,
}

/// What a symbol operand needs from the linker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FixKind {
	/// 16-bit distance from the start of the bundle
	Branch,
	/// low / high 16 bits of the address
	Lo16,
	Hi16,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Fix {
	pub kind: FixKind,
	/// payload bit of the 16-bit field
	pub bit: u32,
	pub symbol: String,
}

/// One encoded operation of a bundle.
#[derive(Clone, Debug)]
pub struct Encoded {
	pub unit: Unit,
	pub class: u8,
	pub bits: u32,
	pub payload: u64,
	pub fix: Option<Fix>,
	/// payload bit of the delay-slot length byte (branches)
	pub delay: Option<u32>,
}

#[derive(Debug)]
pub struct Form {
	mnemonic: String,
	/// operand kinds as the table names them (`i`, `v.e`, `#`, `[8]`, ...)
	kinds: Vec<String>,
	unit: Unit,
	class: u8,
	base: u64,
	slots: Vec<Slot>,
	delay: Option<u32>,
}

pub struct Isa {
	forms: Vec<Form>,
}

fn int(s: &str) -> Result<i64, String> {
	s.parse::<i64>().map_err(|_| format!("bad number {s:?}"))
}

fn hex(s: &str) -> Result<u64, String> {
	u64::from_str_radix(s, 16).map_err(|_| format!("bad hex {s:?}"))
}

/// Bits a value in `lo..=hi` needs (two's complement when `lo < 0`).
fn width(lo: i64, hi: i64) -> u32 {
	let pos = 64 - (hi.max(0) as u64).leading_zeros();
	if lo < 0 {
		pos.max(64 - ((-lo - 1) as u64).leading_zeros()) + 1
	} else {
		pos
	}
}

impl Isa {
	/// The built-in table.
	pub fn new() -> Isa {
		Isa::parse(quant::packed_str!("isa.tbl")).expect("isa.tbl")
	}

	pub fn parse(text: &str) -> Result<Isa, String> {
		let mut forms = Vec::new();
		for (n, line) in text.lines().enumerate() {
			let line = line.trim();
			if line.is_empty() || line.starts_with('#') {
				continue;
			}
			forms.push(Self::parse_line(line).map_err(|e| format!("isa.tbl line {}: {e}", n + 1))?);
		}
		Ok(Isa { forms })
	}

	fn parse_line(line: &str) -> Result<Form, String> {
		let mut parts = line.split('|');
		let (head, enc, slots) = (
			parts.next().unwrap_or(""),
			parts.next().ok_or("no encoding")?,
			parts.next().unwrap_or(""),
		);
		let mut h = head.split_whitespace();
		let mnemonic = h.next().ok_or("no mnemonic")?.to_ascii_uppercase();
		let kinds: Vec<String> = h.map(str::to_string).collect();
		let e: Vec<&str> = enc.split_whitespace().collect();
		if e.len() != 4 {
			return Err("encoding is not `UNIT class nbits base`".into());
		}
		let unit = Unit::parse(e[0]).ok_or("unknown unit")?;
		let class = int(e[1])? as u8;
		if !(1..=3).contains(&class) {
			return Err(format!("{} has no size class {class}", e[0]));
		}
		let mut form = Form {
			mnemonic,
			kinds,
			unit,
			class,
			base: hex(e[3])?,
			slots: Vec::new(),
			delay: None,
		};
		for s in slots.split(';').map(str::trim).filter(|s| !s.is_empty()) {
			let mut t = s.split_whitespace();
			let id = t.next().unwrap();
			if id == "D" {
				form.delay = Some(int(t
					.next()
					.ok_or("D without a position")?
					.trim_start_matches('@'))? as u32);
				continue;
			}
			let split = id
				.find(|c: char| !c.is_ascii_digit())
				.ok_or("slot without a kind")?;
			let operand = int(&id[..split])? as usize;
			let kind = match &id[split..] {
				"r" => Kind::Reg,
				"e" => Kind::Elem,
				"#" => Kind::Imm,
				"#l" => Kind::ImmLo,
				"#h" => Kind::ImmHi,
				"C" => Kind::Cond,
				"N" => Kind::Name,
				"F" => Kind::Float,
				"L" => Kind::Label,
				k if k.starts_with('p') => Kind::Pat(int(&k[1..])? as usize),
				k => return Err(format!("unknown slot kind {k:?}")),
			};
			let rest: Vec<&str> = t.collect();
			let map = if kind == Kind::Label {
				Map::Field(int(rest
					.first()
					.ok_or("L without a position")?
					.trim_start_matches('@'))? as u32)
			} else if let Some((lo, hi)) = rest.first().and_then(|r| r.split_once("..")) {
				let (lo, hi) = (int(lo)?, int(hi)?);
				let w = width(lo, hi);
				let masks = match rest.get(1) {
					None => vec![0; w as usize],
					Some(p) if p.starts_with('@') => {
						let pos = int(&p[1..])? as u32;
						(0..w).map(|j| 1u64 << (pos + j)).collect()
					}
					Some(list) => list.split(',').map(hex).collect::<Result<Vec<_>, _>>()?,
				};
				if masks.len() < w as usize {
					return Err(format!(
						"slot {id}: {} masks for {w} value bits",
						masks.len()
					));
				}
				Map::Int { lo, hi, masks }
			} else {
				let mut tab = Vec::new();
				for r in rest {
					let (k, m) = r
						.split_once('=')
						.ok_or_else(|| format!("slot {id}: {r:?} is not symbol=mask"))?;
					tab.push((k.to_string(), hex(m)?));
				}
				Map::Enum(tab)
			};
			form.slots.push(Slot { operand, kind, map });
		}
		Ok(form)
	}

	/// Encode one operation. `mnemonic` is matched without regard to case.
	pub fn encode(&self, mnemonic: &str, ops: &[Operand]) -> Result<Encoded, String> {
		let up = mnemonic.to_ascii_uppercase();
		let mut known = false;
		let mut err = String::new();
		for f in self.forms.iter().filter(|f| f.mnemonic == up) {
			known = true;
			match f.encode(ops) {
				Ok(e) => return Ok(e),
				Err(e) => {
					if err.is_empty() || f.kinds.len() == ops.len() {
						err = format!("{} {}: {e}", mnemonic, f.kinds.join(" "));
					}
				}
			}
		}
		Err(if known {
			err
		} else {
			format!("unknown instruction {mnemonic} (not in isa.tbl)")
		})
	}
}

impl Default for Isa {
	fn default() -> Self {
		Isa::new()
	}
}

impl Form {
	fn kind_matches(kind: &str, op: &Operand) -> bool {
		match (kind, op) {
			(
				"i",
				Operand::Reg {
					vector: false,
					elem: None,
					..
				},
			) => true,
			(
				"v",
				Operand::Reg {
					vector: true,
					elem: None,
					..
				},
			) => true,
			(
				"i.e",
				Operand::Reg {
					vector: false,
					elem: Some(_),
					..
				},
			) => true,
			(
				"v.e",
				Operand::Reg {
					vector: true,
					elem: Some(_),
					..
				},
			) => true,
			("#", Operand::Int(_) | Operand::Ident(_)) => true,
			("F", Operand::Float(_)) => true,
			("C" | "N", Operand::Ident(_)) => true,
			("L", Operand::Ident(_) | Operand::Int(_)) => true,
			(k, Operand::Pattern(p)) => {
				k.strip_prefix('[')
					.and_then(|k| k.strip_suffix(']'))
					.and_then(|n| n.parse().ok())
					== Some(p.len())
			}
			_ => false,
		}
	}

	fn encode(&self, ops: &[Operand]) -> Result<Encoded, String> {
		if ops.len() != self.kinds.len() {
			return Err(format!("{} operands given", ops.len()));
		}
		for (n, (k, op)) in self.kinds.iter().zip(ops).enumerate() {
			if !Self::kind_matches(k, op) {
				return Err(format!("operand {} does not fit", n + 1));
			}
		}
		let mut payload = self.base;
		let mut fix = None;
		for s in &self.slots {
			let op = &ops[s.operand];
			let name = s.operand + 1;
			// the slot's value: an integer, or a symbol of an enumeration
			let (mut value, mut symbol) = (None, None);
			match (s.kind, op) {
				(Kind::Reg, Operand::Reg { n, .. }) => value = Some(*n as i64),
				(Kind::Elem, Operand::Reg { elem: Some(e), .. }) => value = Some(*e as i64),
				(Kind::Imm | Kind::ImmLo | Kind::ImmHi | Kind::Label, Operand::Int(v)) => {
					value = Some(*v)
				}
				(Kind::ImmLo | Kind::ImmHi | Kind::Label, Operand::Ident(sym)) => {
					let (kind, bit) = match (&s.map, s.kind) {
						(Map::Field(bit), _) => (FixKind::Branch, *bit),
						(Map::Int { masks, .. }, k) => (
							if k == Kind::ImmLo {
								FixKind::Lo16
							} else {
								FixKind::Hi16
							},
							masks.first().map_or(0, |m| m.trailing_zeros()),
						),
						_ => return Err(format!("operand {name}: no field for a symbol")),
					};
					fix = Some(Fix {
						kind,
						bit,
						symbol: sym.clone(),
					});
					continue;
				}
				(Kind::Imm, Operand::Ident(sym)) => {
					return Err(format!("operand {name}: {sym} is not a number"))
				}
				(Kind::Cond | Kind::Name, Operand::Ident(c)) => {
					symbol = Some(c.to_ascii_uppercase())
				}
				(Kind::Float, Operand::Float(f)) => symbol = Some(f.clone()),
				(Kind::Pat(p), Operand::Pattern(pat)) => {
					symbol = pat.get(p..p + 1).map(str::to_string)
				}
				_ => return Err(format!("operand {name} does not fit")),
			}
			match &s.map {
				Map::Field(bit) => {
					let v = value.unwrap();
					if !(-0x8000..=0xffff).contains(&v) {
						return Err(format!("operand {name}: {v} does not fit 16 bits"));
					}
					payload ^= ((v as u64) & 0xffff) << bit;
				}
				Map::Int { lo, hi, masks } => {
					let mut v = value.ok_or_else(|| format!("operand {name} does not fit"))?;
					if s.kind == Kind::ImmHi && !(0..=0xffff).contains(&v) {
						// the assembler takes the high half of a value wider than 16 bits
						if !(-(1i64 << 31)..(1i64 << 32)).contains(&v) {
							return Err(format!("operand {name}: {v} does not fit 32 bits"));
						}
						v = ((v as u32) >> 16) as i64;
					}
					if v > *hi && (1i64 << 31..1i64 << 32).contains(&v) {
						v -= 1i64 << 32; // 0xffffffff is accepted for -1
					}
					if v < *lo || v > *hi {
						return Err(format!("operand {name}: {v} is outside {lo}..{hi}"));
					}
					for (j, m) in masks.iter().enumerate() {
						if (v >> j) & 1 != 0 {
							payload ^= m;
						}
					}
				}
				Map::Enum(tab) => {
					let key = symbol
						.or_else(|| value.map(|v| v.to_string()))
						.unwrap_or_default();
					let hit = tab.iter().find(|(k, _)| {
						*k == key
							|| (s.kind == Kind::Float
								&& k.parse::<f64>()
									.ok()
									.zip(key.parse::<f64>().ok())
									.is_some_and(|(a, b)| a == b))
					});
					payload ^= hit
						.ok_or_else(|| {
							format!(
								"operand {name}: {key:?} is not one of {}",
								tab.iter()
									.map(|(k, _)| k.as_str())
									.collect::<Vec<_>>()
									.join(" ")
							)
						})?
						.1;
				}
			}
		}
		let bits = self.unit.bits(self.class);
		debug_assert!(
			bits == 64 || payload >> bits == 0,
			"{}: payload wider than its class",
			self.mnemonic
		);
		Ok(Encoded {
			unit: self.unit,
			class: self.class,
			bits,
			payload,
			fix,
			delay: self.delay,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn r(n: u32) -> Operand {
		Operand::Reg {
			vector: false,
			n,
			elem: None,
		}
	}

	#[test]
	fn table_parses() {
		let isa = Isa::new();
		assert!(isa.forms.len() > 2500);
		// no payload is wider than its class
		for f in &isa.forms {
			let all = f.slots.iter().fold(f.base, |a, s| match &s.map {
				Map::Int { masks, .. } => masks.iter().fold(a, |a, m| a | m),
				Map::Enum(tab) => tab.iter().fold(a, |a, t| a | t.1),
				Map::Field(bit) => a | 0xffff << bit,
			});
			assert_eq!(
				all >> f.unit.bits(f.class),
				0,
				"{} {}",
				f.mnemonic,
				f.kinds.join(" ")
			);
		}
	}

	#[test]
	fn named_registers() {
		let isa = Isa::new();
		let (i, v, name) = (
			r(1),
			|n| Operand::Reg {
				vector: true,
				n,
				elem: None,
			},
			|s: &str| Operand::Ident(s.into()),
		);
		// CMU.CPTI i1 C_CMU0 = 41 2d 03; a number is another form: CMU.CPTI i1 4 = 41 4d 02
		assert_eq!(
			isa.encode("CMU.CPTI", &[i.clone(), name("c_cmu0")])
				.unwrap()
				.payload,
			0x32d
		);
		assert_eq!(
			isa.encode("CMU.CPTI", &[i.clone(), Operand::Int(4)])
				.unwrap()
				.payload,
			0x24d
		);
		// VAU.DP4A v1 v2 v3 V_ACC1 = 33 5f 84 31
		assert_eq!(
			isa.encode("VAU.DP4A", &[v(1), v(2), v(3), name("V_ACC1")])
				.unwrap()
				.payload,
			0x31845f
		);
		assert!(isa
			.encode("VAU.ACCPW.F16", &[v(1), v(2), name("S_ACC0")])
			.is_err());
	}

	#[test]
	fn immediates() {
		let isa = Isa::new();
		// IAU.INCS i3 -2 = c6 63 f8 ff
		let e = isa.encode("IAU.INCS", &[r(3), Operand::Int(-2)]).unwrap();
		assert_eq!((e.class, e.bits, e.payload), (3, 24, 0xfff863));
		assert!(isa.encode("IAU.INCS", &[r(3), Operand::Int(8192)]).is_err());
		// LDIH takes the high half of a wide value, LDIL the low half
		let h = isa
			.encode("LSU0.LDIH", &[r(0), Operand::Int(0x12345678)])
			.unwrap();
		assert_eq!(h.payload, 0x123403);
		let l = isa
			.encode("LSU0.LDIL", &[r(0), Operand::Int(0x12345678)])
			.unwrap();
		assert_eq!(l.payload, 0x567802);
		assert_eq!(
			isa.encode("lsu0.ldil", &[r(0), Operand::Int(0xffff_ffff)])
				.unwrap()
				.payload,
			0xffff02
		);
	}

	#[test]
	fn symbols() {
		let isa = Isa::new();
		let e = isa
			.encode("LSU1.LDIH", &[r(1), Operand::Ident("x".into())])
			.unwrap();
		assert_eq!(
			e.fix,
			Some(Fix {
				kind: FixKind::Hi16,
				bit: 8,
				symbol: "x".into()
			})
		);
		let b = isa
			.encode("BRU.BRA", &[Operand::Ident(".LBB0_2".into())])
			.unwrap();
		assert_eq!((b.fix.unwrap().kind, b.delay), (FixKind::Branch, Some(16)));
		assert!(isa
			.encode("IAU.ADD", &[r(1), r(2), Operand::Ident("x".into())])
			.is_err());
	}
}
