//! The kernel language's statements to straight-line blocks of operations
//! on virtual registers: constants folded, constant loops unrolled, every
//! operator and intrinsic an instruction or a short sequence of them.

use super::parse::{fold_num, Base, Declarator, Expr, File, Node, Num, Stmt, Type};
use super::{AOp, Arg, Block, Item, Mem, Term, Vr, VregInfo};
use crate::shave::mach::{d2h, latency};
use std::collections::HashMap;

/// The intrinsics: (name, arguments, result, instruction, memory access,
/// bytes). Types: `H` half8, `F` float4, `I` int4, `V` any vector (as a
/// result: the first argument's type), `h` half, `f` float, `i` u32 / int,
/// `p` a pointer, `#` a constant. Instruction: `$d` the result, `$0`.. the
/// arguments, `.N` / `.$K` an element index (a constant argument's value);
/// the unit `LSU` stands for either load / store unit. Memory: `L` a load
/// through argument 0.
pub const INTRINSICS: &[(&str, &str, char, &str, char, i64)] = &[
	("exp", "H", 'H', "VAU.EXP $d $0", ' ', 0),
	("sigm", "H", 'H', "VAU.SIGM $d $0", ' ', 0),
	("tanh", "H", 'H', "VAU.TANH $d $0", ' ', 0),
	("log", "H", 'H', "VAU.LOG $d $0", ' ', 0),
	("sqrt", "H", 'H', "VAU.SQT $d $0", ' ', 0),
	("sqrt", "h", 'h', "SAU.SQT $d $0.0", ' ', 0),
	("rcp", "h", 'h', "SAU.RCP $d $0.0", ' ', 0),
	("rsqrt", "h", 'h', "SAU.RQT $d $0.0", ' ', 0),
	("max", "HH", 'H', "CMU.MAX.F16 $d $0 $1", ' ', 0),
	("min", "HH", 'H', "CMU.MIN.F16 $d $0 $1", ' ', 0),
	("max", "FF", 'F', "CMU.MAX.F32 $d $0 $1", ' ', 0),
	("min", "FF", 'F', "CMU.MIN.F32 $d $0 $1", ' ', 0),
	("rot", "V#", 'V', "CMU.ALIGNVEC $d $0 $0 $1", ' ', 0),
	("align", "VV#", 'V', "CMU.ALIGNVEC $d $0 $1 $2", ' ', 0),
	("splat", "h", 'H', "CMU.CP.128.16.R $d $0.0", ' ', 0),
	("splat", "f", 'F', "CMU.CP.128.32.R $d $0", ' ', 0),
	("splat", "i", 'I', "CMU.CP.128.32.R $d $0", ' ', 0),
	("sum", "F", 'f', "SAU.SUMX.F32 $d $0", ' ', 0),
	(
		"f16_to_f32",
		"H#",
		'F',
		"CMU.CP.128.F16.F32 $d $0.$1",
		' ',
		0,
	),
	("i32_to_f32", "I", 'F', "CMU.CP.128.I32.F32 $d $0", ' ', 0),
	("f32_to_i32", "F", 'I', "CMU.CP.128.F32.I32S $d $0", ' ', 0),
	(
		"f32_to_f16",
		"F#",
		'h',
		"CMU.CP.16.F32.F16 $d.0 $0.$1",
		' ',
		0,
	),
	("f16_to_u8f", "H", 'I', "CMU.CP.64.F16.U8FS $d.0 $0", ' ', 0),
	("load_splat", "p", 'H', "LSU.LD.128.16.R $d $0", 'L', 2),
	("load_f32", "p", 'F', "LSU.LD.128.F16.F32 $d $0", 'L', 8),
];

/// The infix operators on registers: (operator, operand type, instruction).
/// `i` covers u32, int and (for `+` / `-`) pointers.
pub const OPERATORS: &[(&str, char, &str)] = &[
	("+", 'H', "VAU.ADD.F16"),
	("-", 'H', "VAU.SUB.F16"),
	("*", 'H', "VAU.MUL.F16"),
	("+", 'F', "VAU.ADD.F32"),
	("-", 'F', "VAU.SUB.F32"),
	("*", 'F', "VAU.MUL.F32"),
	("+", 'I', "VAU.ADD.I32"),
	("-", 'I', "VAU.SUB.I32"),
	(">>", 'I', "VAU.SHR.I32"),
	("&", 'V', "VAU.AND"),
	("|", 'V', "VAU.OR"),
	("^", 'V', "VAU.XOR"),
	("+", 'i', "IAU.ADD"),
	("-", 'i', "IAU.SUB"),
	("*", 'i', "IAU.MUL"),
	("&", 'i', "IAU.AND"),
	("|", 'i', "IAU.OR"),
	("^", 'i', "IAU.XOR"),
	("<<", 'i', "IAU.SHL"),
	(">>", 'i', "IAU.SHR.U32"),
];

#[derive(Clone)]
enum Bind {
	Const(i64),
	/// `const float NAME = ...;` in front of the function
	Flt(f64),
	Reg(Vr, Type),
	/// `register T name[n]`
	Arr(Vec<Vr>, Type),
	/// `T name[n]`: memory on the stack (offset, element, alias tag)
	Stack(i64, Base, usize),
}

/// What an expression gives: a constant not yet in a register, or a register.
#[derive(Clone, Copy, Debug)]
enum Val {
	Int(i64),
	Flt(f64),
	Reg(Vr, Type),
}

#[derive(Clone)]
pub struct Lowered {
	pub vregs: Vec<VregInfo>,
	pub items: Vec<Item>,
}

pub const PARAMS: Vr = 0;
const SP: Vr = 1;
const UNROLL_MAX: usize = 4096;

#[derive(Clone)]
struct Lower {
	vregs: Vec<VregInfo>,
	/// definitions of each register so far (a memory operand's base version)
	version: Vec<u32>,
	items: Vec<Item>,
	cur: Vec<AOp>,
	scopes: Vec<HashMap<String, Bind>>,
	tags: usize,
	labels: usize,
	stack: i64,
	line: usize,
}

fn alts(mnem: &str) -> Vec<String> {
	match mnem.strip_prefix("LSU.") {
		Some(rest) => vec![format!("LSU0.{rest}"), format!("LSU1.{rest}")],
		None => vec![mnem.to_string()],
	}
}

fn inverse(cc: &str) -> &'static str {
	match cc {
		"EQ" => "NEQ",
		"NEQ" => "EQ",
		"LT" => "GTE",
		"GTE" => "LT",
		"GT" => "LTE",
		_ => "GT",
	}
}

fn int_like(t: Type) -> bool {
	!t.ptr && matches!(t.base, Base::U32 | Base::Int)
}

fn letter(t: Type) -> char {
	match (t.ptr, t.base) {
		(true, _) => 'p',
		(_, Base::Half8) => 'H',
		(_, Base::Float4) => 'F',
		(_, Base::Int4) => 'I',
		(_, Base::Half) => 'h',
		(_, Base::Float) => 'f',
		(_, Base::U32 | Base::Int) => 'i',
		_ => '?',
	}
}

fn of_letter(c: char) -> Type {
	let base = match c {
		'H' => Base::Half8,
		'F' => Base::Float4,
		'I' => Base::Int4,
		'h' => Base::Half,
		'f' => Base::Float,
		_ => Base::U32,
	};
	Type { base, ptr: false }
}

/// Does any expression of `nodes` name `name`?
fn mentions(nodes: &[Node], name: &str) -> bool {
	fn ex(e: &Expr, name: &str) -> bool {
		match e {
			Expr::Name(n) => n == name,
			Expr::Int(_) | Expr::Float(_) => false,
			Expr::Index(a, b) | Expr::Binary(_, a, b) => ex(a, name) || ex(b, name),
			Expr::Call(_, args) => args.iter().any(|a| ex(a, name)),
			Expr::Unary(_, a) | Expr::Cast(_, a) => ex(a, name),
		}
	}
	nodes.iter().any(|n| match &n.stmt {
		Stmt::Decl { items, .. } => items.iter().any(|d| {
			d.len.as_ref().is_some_and(|e| ex(e, name))
				|| d.init.as_ref().is_some_and(|e| ex(e, name))
		}),
		Stmt::Assign(a, _, b) => ex(a, name) || ex(b, name),
		Stmt::Call(e) | Stmt::Assert(e, _) => ex(e, name),
		Stmt::If(c, a, b) => ex(c, name) || mentions(a, name) || mentions(b, name),
		Stmt::DoWhile(a, c) | Stmt::While(c, a) => ex(c, name) || mentions(a, name),
		Stmt::For(i, c, s, b, _) => {
			mentions(i, name)
				|| c.as_ref().is_some_and(|c| ex(c, name))
				|| mentions(s, name)
				|| mentions(b, name)
		}
		Stmt::Block(b) => mentions(b, name),
	})
}

impl Lower {
	fn err<T>(&self, text: impl AsRef<str>) -> Result<T, String> {
		Err(format!("line {}: {}", self.line, text.as_ref()))
	}

	fn vreg(&mut self, vector: bool, name: &str, tag: Option<usize>) -> Vr {
		self.vregs.push(VregInfo {
			vector,
			name: name.to_string(),
			tag,
			pin: None,
		});
		self.version.push(0);
		self.vregs.len() - 1
	}

	fn temp(&mut self, vector: bool) -> Vr {
		self.vreg(vector, &format!("(line {})", self.line), None)
	}

	fn tag(&mut self) -> usize {
		self.tags += 1;
		self.tags
	}

	fn lookup(&self, name: &str) -> Option<&Bind> {
		self.scopes.iter().rev().find_map(|s| s.get(name))
	}

	fn bind(&mut self, name: &str, b: Bind) -> Result<(), String> {
		if self.scopes.last().unwrap().contains_key(name) {
			return self.err(format!("{name} is already declared in this block"));
		}
		self.scopes.last_mut().unwrap().insert(name.to_string(), b);
		Ok(())
	}

	/// A compile-time number, if `e` is one.
	fn number(&self, e: &Expr) -> Option<Num> {
		fold_num(e, &|n| match self.lookup(n) {
			Some(Bind::Const(v)) => Some(Num::Int(*v)),
			Some(Bind::Flt(x)) => Some(Num::Flt(*x)),
			_ => None,
		})
		.ok()
	}

	/// A compile-time integer, if `e` is one.
	fn constant(&self, e: &Expr) -> Option<i64> {
		match self.number(e) {
			Some(Num::Int(v)) => Some(v),
			_ => None,
		}
	}

	fn ceval(&self, e: &Expr, what: &str) -> Result<i64, String> {
		match self.constant(e) {
			Some(v) => Ok(v),
			None => self.err(format!("{what} must be a compile-time constant")),
		}
	}

	/// The register variable `e` names: `name` or `name[constant]` of a register array.
	fn var(&self, e: &Expr) -> Result<Option<(Vr, Type)>, String> {
		match e {
			Expr::Name(n) => match self.lookup(n) {
				Some(Bind::Reg(r, t)) => Ok(Some((*r, *t))),
				Some(Bind::Arr(..)) => self.err(format!("{n} is an array: {n}[index]")),
				Some(_) => Ok(None),
				None => self.err(format!("{n} is not declared")),
			},
			Expr::Index(a, ix) => match &**a {
				Expr::Name(n) => match self.lookup(n) {
					Some(Bind::Arr(regs, t)) => {
						let k = self.ceval(ix, &format!("the index of the register array {n}"))?;
						match regs.get(k as usize).filter(|_| k >= 0) {
							Some(r) => Ok(Some((*r, *t))),
							None => self.err(format!("{n}[{k}] is outside {n}[{}]", regs.len())),
						}
					}
					_ => Ok(None),
				},
				_ => Ok(None),
			},
			_ => Ok(None),
		}
	}

	/// The `restrict` pointer (or stack array) the address `e` is based on.
	fn based_on(&self, e: &Expr) -> Option<usize> {
		match e {
			Expr::Name(n) => match self.lookup(n) {
				Some(Bind::Reg(r, t)) if t.ptr => self.vregs[*r].tag,
				Some(Bind::Stack(_, _, tag)) => Some(*tag),
				_ => None,
			},
			Expr::Index(..) => self
				.var(e)
				.ok()
				.flatten()
				.filter(|v| v.1.ptr)
				.and_then(|v| self.vregs[v.0].tag),
			Expr::Cast(_, a) | Expr::Unary("&", a) => self.based_on(a),
			Expr::Binary("+" | "-", a, b) => self.based_on(a).or_else(|| self.based_on(b)),
			_ => None,
		}
	}

	fn push(
		&mut self,
		alts: Vec<String>,
		args: Vec<Arg>,
		part: Option<(u8, u8)>,
		mem: Option<Mem>,
	) -> usize {
		let lat = latency(&alts[0]).unwrap_or(1);
		for a in &args {
			if let Arg::Def(r, _) = a {
				self.version[*r] += 1;
			}
		}
		self.cur.push(AOp {
			alts,
			args,
			lat,
			part,
			begin: false,
			mem,
			line: self.line,
		});
		self.cur.len() - 1
	}

	fn mem(&self, store: bool, base: Vr, tag: Option<usize>, off: i64, size: i64) -> Option<Mem> {
		Some(Mem {
			store,
			tag: tag.or(self.vregs[base].tag),
			base,
			ver: self.version[base],
			off,
			size,
		})
	}

	/// `dst = value`
	fn li(&mut self, dst: Vr, value: i64) -> Result<(), String> {
		if !(-(1i64 << 31)..1i64 << 32).contains(&value) {
			return self.err(format!("{value} does not fit 32 bits"));
		}
		let v = value as u32;
		self.push(
			alts("LSU.LDIL"),
			vec![Arg::Def(dst, None), Arg::Int((v & 0xffff) as i64)],
			None,
			None,
		);
		// whether LDIL extends the sign is not known: set the high half unless both ways give it
		if v > 0x7fff {
			self.push(
				alts("LSU.LDIH"),
				vec![Arg::Def(dst, None), Arg::Int((v >> 16) as i64)],
				Some((2, 2)),
				None,
			);
		}
		Ok(())
	}

	fn mov(&mut self, dst: Vr, src: Vr) {
		if dst == src {
			return;
		}
		let w = if self.vregs[dst].vector { "128" } else { "32" };
		let a = vec![
			format!("LSU0.CP.{w}"),
			format!("LSU1.CP.{w}"),
			format!("CMU.CP.{w}"),
		];
		self.push(
			a,
			vec![Arg::Def(dst, None), Arg::Use(src, None)],
			None,
			None,
		);
	}

	/// `dst = a + k`
	fn add_const(&mut self, dst: Vr, a: Vr, k: i64) -> Result<(), String> {
		match k {
			0 => self.mov(dst, a),
			1..=255 => {
				self.push(
					alts("IAU.ADD"),
					vec![Arg::Def(dst, None), Arg::Use(a, None), Arg::Int(k)],
					None,
					None,
				);
			}
			-255..=-1 => {
				self.push(
					alts("IAU.SUB"),
					vec![Arg::Def(dst, None), Arg::Use(a, None), Arg::Int(-k)],
					None,
					None,
				);
			}
			_ => {
				let t = self.temp(false);
				self.li(t, k)?;
				self.push(
					alts("IAU.ADD"),
					vec![Arg::Def(dst, None), Arg::Use(a, None), Arg::Use(t, None)],
					None,
					None,
				);
			}
		}
		Ok(())
	}

	fn dst(&mut self, dst: Option<Vr>, vector: bool) -> Vr {
		match dst {
			Some(d) if self.vregs[d].vector == vector => d,
			_ => self.temp(vector),
		}
	}

	/// `v` in a register of type `want` (which a constant takes on).
	fn reg(&mut self, v: Val, want: Type, dst: Option<Vr>) -> Result<(Vr, Type), String> {
		let (bits, splat) = match (v, want.ptr, want.base) {
			(Val::Reg(r, t), ..) => return Ok((r, t)),
			(Val::Int(k), true, _) | (Val::Int(k), _, Base::U32 | Base::Int) => (k, None),
			(Val::Int(k), _, Base::Int4) => (k, Some("CMU.CP.128.32.R")),
			(Val::Int(k), _, Base::Half) => (d2h(k as f64) as i64, None),
			(Val::Flt(x), false, Base::Half) => (d2h(x) as i64, None),
			(Val::Int(k), _, Base::Float) => ((k as f32).to_bits() as i64, None),
			(Val::Flt(x), false, Base::Float) => ((x as f32).to_bits() as i64, None),
			(Val::Int(k), _, Base::Half8) => (d2h(k as f64) as i64, Some("CMU.CP.128.16.R")),
			(Val::Flt(x), false, Base::Half8) => (d2h(x) as i64, Some("CMU.CP.128.16.R")),
			(Val::Int(k), _, Base::Float4) => {
				((k as f32).to_bits() as i64, Some("CMU.CP.128.32.R"))
			}
			(Val::Flt(x), false, Base::Float4) => {
				((x as f32).to_bits() as i64, Some("CMU.CP.128.32.R"))
			}
			(Val::Flt(_), ..) => {
				return self.err(format!(
					"a floating constant where {} is needed",
					want.show()
				))
			}
			(Val::Int(_), ..) => {
				return self.err(format!("a constant where {} is needed", want.show()))
			}
		};
		match splat {
			None => {
				let d = self.dst(dst, false);
				self.li(d, bits)?;
				Ok((d, want))
			}
			Some(m) => {
				let (t, d) = (self.temp(false), self.dst(dst, true));
				self.li(t, bits)?;
				let elem = (m == "CMU.CP.128.16.R").then_some(0);
				self.push(
					alts(m),
					vec![Arg::Def(d, None), Arg::Use(t, elem)],
					None,
					None,
				);
				Ok((d, want))
			}
		}
	}

	fn load(
		&mut self,
		elem: Base,
		base: Vr,
		tag: Option<usize>,
		off: i64,
		dst: Option<Vr>,
	) -> Result<Val, String> {
		if !(-16384..16384).contains(&off) || (elem.vector() && off + 8 >= 16384) {
			return self.err(format!(
				"the offset {off} does not fit a load (-16384..16383 bytes)"
			));
		}
		let t = Type {
			base: elem,
			ptr: false,
		};
		match elem {
			Base::U32 | Base::Int | Base::Float => {
				let d = self.dst(dst, false);
				let m = self.mem(false, base, tag, off, 4);
				self.push(
					alts("LSU.LDO.32"),
					vec![Arg::Def(d, None), Arg::Use(base, None), Arg::Int(off)],
					None,
					m,
				);
				Ok(Val::Reg(d, t))
			}
			Base::Half8 | Base::Float4 | Base::Int4 => {
				let d = self.dst(dst, true);
				if off == 0 {
					let m = self.mem(false, base, tag, 0, 16);
					self.push(
						alts("LSU.LD.128"),
						vec![Arg::Def(d, None), Arg::Use(base, None)],
						None,
						m,
					);
				} else {
					for h in 0..2i64 {
						let m = self.mem(false, base, tag, off + 8 * h, 8);
						let j = self.push(
							alts("LSU.LDO.64"),
							vec![
								Arg::Def(d, Some(h as u32)),
								Arg::Use(base, None),
								Arg::Int(off + 8 * h),
							],
							Some((8 * h as u8, 8)),
							m,
						);
						self.cur[j].begin = h == 0;
					}
				}
				Ok(Val::Reg(d, t))
			}
			Base::Half => {
				self.err("a half is not loaded on its own: load_splat(p) gives it in all 8 lanes")
			}
			_ => self.err(format!(
				"a {} cannot be loaded: cast the pointer",
				elem.name()
			)),
		}
	}

	fn store(
		&mut self,
		elem: Base,
		base: Vr,
		tag: Option<usize>,
		off: i64,
		value: &Expr,
	) -> Result<(), String> {
		if !(-16384..16384).contains(&off) || (elem.vector() && off + 8 >= 16384) {
			return self.err(format!(
				"the offset {off} does not fit a store (-16384..16383 bytes)"
			));
		}
		// through a `u8 *`: the low 8 bytes of an int4 (`f16_to_u8f`)
		let want = Type {
			base: if elem == Base::U8 { Base::Int4 } else { elem },
			ptr: false,
		};
		let v = self.rv(value, None, Some(want))?;
		let (v, t) = self.reg(v, want, None)?;
		self.assignable(want, t)?;
		match elem {
			Base::U8 => {
				let m = self.mem(true, base, tag, off, 8);
				self.push(
					alts("LSU.STO.64"),
					vec![Arg::Use(v, Some(0)), Arg::Use(base, None), Arg::Int(off)],
					None,
					m,
				);
			}
			Base::U32 | Base::Int | Base::Float => {
				let m = self.mem(true, base, tag, off, 4);
				self.push(
					alts("LSU.STO.32"),
					vec![Arg::Use(v, None), Arg::Use(base, None), Arg::Int(off)],
					None,
					m,
				);
			}
			Base::Half => {
				let m = self.mem(true, base, tag, off, 2);
				self.push(
					alts("LSU.STO.16"),
					vec![Arg::Use(v, Some(0)), Arg::Use(base, None), Arg::Int(off)],
					None,
					m,
				);
			}
			Base::Half8 | Base::Float4 | Base::Int4 => {
				if off == 0 {
					let m = self.mem(true, base, tag, 0, 16);
					self.push(
						alts("LSU.ST.128"),
						vec![Arg::Use(v, None), Arg::Use(base, None)],
						None,
						m,
					);
				} else {
					for h in 0..2i64 {
						let m = self.mem(true, base, tag, off + 8 * h, 8);
						self.push(
							alts("LSU.STO.64"),
							vec![
								Arg::Use(v, Some(h as u32)),
								Arg::Use(base, None),
								Arg::Int(off + 8 * h),
							],
							None,
							m,
						);
					}
				}
			}
			_ => {
				return self.err(format!(
					"a {} cannot be stored: cast the pointer",
					elem.name()
				))
			}
		}
		Ok(())
	}

	fn assignable(&self, to: Type, from: Type) -> Result<(), String> {
		let ok = to == from
			|| (int_like(to) && int_like(from))
			|| (to.ptr && from.ptr && (to.base == Base::Void || from.base == Base::Void));
		if ok {
			Ok(())
		} else {
			self.err(format!(
				"{} where {} is needed{}",
				from.show(),
				to.show(),
				if to.ptr || from.ptr { " (cast it)" } else { "" }
			))
		}
	}

	/// The memory `e[index]` names: (element, base register, tag, byte offset).
	fn place(
		&mut self,
		e: &Expr,
		index: &Expr,
	) -> Result<Option<(Base, Vr, Option<usize>, i64)>, String> {
		if let Expr::Name(n) = e {
			if let Some(Bind::Stack(off, elem, tag)) = self.lookup(n).cloned() {
				let k = self.ceval(
					index,
					"an index into a stack array (use a pointer for a run-time one)",
				)?;
				return Ok(Some((elem, SP, Some(tag), off + k * elem.size())));
			}
		}
		if self
			.var(&Expr::Index(Box::new(e.clone()), Box::new(index.clone())))?
			.is_some()
		{
			return Ok(None);
		}
		let b = self.rv(e, None, None)?;
		match b {
			Val::Reg(r, t) if t.ptr => {
				let k = self.ceval(
					index,
					"an index (for a run-time one add it to the pointer: (p + i)[0])",
				)?;
				Ok(Some((t.base, r, None, k * t.base.size())))
			}
			_ => Ok(None),
		}
	}

	/// Lower an expression. `dst`: the register its last instruction may
	/// write; `want`: the type a constant operand takes.
	fn rv(&mut self, e: &Expr, dst: Option<Vr>, want: Option<Type>) -> Result<Val, String> {
		match self.number(e) {
			Some(Num::Int(k)) => return Ok(Val::Int(k)),
			Some(Num::Flt(x)) => return Ok(Val::Flt(x)),
			None => {}
		}
		match e {
			Expr::Int(k) => Ok(Val::Int(*k)),
			Expr::Float(x) => Ok(Val::Flt(*x)),
			Expr::Name(n) => match self.lookup(n).cloned() {
				Some(Bind::Reg(r, t)) => Ok(Val::Reg(r, t)),
				Some(Bind::Stack(off, elem, tag)) => {
					let d = self.dst(dst, false);
					self.add_const(d, SP, off)?;
					if dst != Some(d) {
						self.vregs[d].tag = Some(tag);
					}
					Ok(Val::Reg(
						d,
						Type {
							base: elem,
							ptr: true,
						},
					))
				}
				Some(Bind::Arr(..)) => self.err(format!("{n} is an array: {n}[index]")),
				Some(Bind::Const(_) | Bind::Flt(_)) => unreachable!(),
				None => self.err(format!("{n} is not declared")),
			},
			Expr::Unary("&", a) => match &**a {
				Expr::Name(n) if matches!(self.lookup(n), Some(Bind::Stack(..))) => {
					self.rv(a, dst, want)
				}
				Expr::Index(p, ix) => self.rv(&Expr::Binary("+", p.clone(), ix.clone()), dst, want),
				_ => self.err("& takes a stack array or p[i]"),
			},
			Expr::Unary(op, _) => self.err(format!(
				"unary {op} works on constants only{}",
				if *op == "-" { " (write 0 - x)" } else { "" }
			)),
			Expr::Cast(t, a) => {
				let v = self.rv(a, dst, Some(*t))?;
				match v {
					Val::Reg(r, from) => {
						let ints = |t: Type| t.ptr || matches!(t.base, Base::U32 | Base::Int);
						if (ints(*t) && ints(from))
							|| (!t.ptr && !from.ptr && t.base.vector() && from.base.vector())
							|| *t == from
						{
							Ok(Val::Reg(r, *t))
						} else {
							self.err(format!("a cast from {} to {}: only between pointers and integers, or between vector types (the bits stay)", from.show(), t.show()))
						}
					}
					c => {
						let (r, _) = self.reg(c, *t, dst)?;
						Ok(Val::Reg(r, *t))
					}
				}
			}
			Expr::Index(a, ix) => {
				if let Some((r, t)) = self.var(e)? {
					return Ok(Val::Reg(r, t));
				}
				if let Some((elem, base, tag, off)) = self.place(a, ix)? {
					return self.load(elem, base, tag, off, dst);
				}
				// a lane of a vector
				let v = self.rv(a, None, None)?;
				let Val::Reg(r, t) = v else {
					return self.err("[ ] on a constant");
				};
				let lane = self.ceval(ix, "a vector's lane")?;
				let (m, elem, lanes, part) = match t.base {
					Base::Half8 => ("CMU.CP.16", Base::Half, 8, Some(0)),
					Base::Float4 => ("CMU.CP.32", Base::Float, 4, None),
					Base::Int4 => ("CMU.CP.32", Base::Int, 4, None),
					_ => return self.err(format!("[ ] on {}", t.show())),
				};
				if !(0..lanes).contains(&lane) {
					return self.err(format!("lane {lane} of {}", t.show()));
				}
				let d = self.dst(dst, false);
				self.push(
					alts(m),
					vec![Arg::Def(d, part), Arg::Use(r, Some(lane as u32))],
					None,
					None,
				);
				Ok(Val::Reg(
					d,
					Type {
						base: elem,
						ptr: false,
					},
				))
			}
			Expr::Binary(op, a, b) => self.binary(op, a, b, dst, want),
			Expr::Call(name, args) => self.call(name, args, dst),
		}
	}

	fn binary(
		&mut self,
		op: &'static str,
		a: &Expr,
		b: &Expr,
		dst: Option<Vr>,
		want: Option<Type>,
	) -> Result<Val, String> {
		if matches!(op, "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||") {
			return self.err(format!(
				"{op} on registers is allowed only as the condition of if / while / for"
			));
		}
		if matches!(op, "/" | "%") {
			return self.err(format!(
				"{op} works on constants only (the SHAVE's divide is not in the language)"
			));
		}
		let (va, vb) = (self.rv(a, None, None)?, self.rv(b, None, None)?);
		// the type of the operation: a register operand's, else what is wanted
		let ty = match (va, vb) {
			(Val::Reg(_, t), _) if t.ptr => t,
			(_, Val::Reg(_, t)) if t.ptr => t,
			(Val::Reg(_, t), _) | (_, Val::Reg(_, t)) => t,
			_ => want.unwrap_or(Type {
				base: Base::U32,
				ptr: false,
			}),
		};
		if ty.ptr {
			// pointer arithmetic, in elements
			let (p, k, size) = match (va, vb) {
				(Val::Reg(p, t), k) if t.ptr && matches!(op, "+" | "-") => (p, k, t.base.size()),
				(k, Val::Reg(p, t)) if t.ptr && op == "+" => (p, k, t.base.size()),
				_ => {
					return self.err(format!(
						"{op} on a pointer (only pointer + integer, pointer - integer)"
					))
				}
			};
			let tag = self.vregs[p].tag;
			let d = self.dst(dst, false);
			match k {
				Val::Int(k) => self.add_const(d, p, if op == "-" { -k } else { k } * size)?,
				Val::Reg(r, t) if int_like(t) => {
					let mut r = r;
					if size > 1 {
						let s = self.temp(false);
						self.push(
							alts("IAU.SHL"),
							vec![
								Arg::Def(s, None),
								Arg::Use(r, None),
								Arg::Int(size.trailing_zeros() as i64),
							],
							None,
							None,
						);
						r = s;
					}
					self.push(
						alts(if op == "-" { "IAU.SUB" } else { "IAU.ADD" }),
						vec![Arg::Def(d, None), Arg::Use(p, None), Arg::Use(r, None)],
						None,
						None,
					);
				}
				_ => return self.err("a pointer moves by an integer"),
			}
			if dst != Some(d) {
				self.vregs[d].tag = tag;
			}
			return Ok(Val::Reg(d, ty));
		}
		let class = letter(ty);
		let Some(row) = OPERATORS
			.iter()
			.find(|r| r.0 == op && (r.1 == class || (r.1 == 'V' && ty.base.vector())))
		else {
			return self.err(format!("{op} is not defined for {}", ty.show()));
		};
		if class == 'i' {
			// constants: in the instruction where it has room for them
			match (op, va, vb) {
				("+", Val::Reg(r, _), Val::Int(k)) | ("+", Val::Int(k), Val::Reg(r, _)) => {
					let d = self.dst(dst, false);
					self.add_const(d, r, k)?;
					return Ok(Val::Reg(d, ty));
				}
				("-", Val::Reg(r, _), Val::Int(k)) => {
					let d = self.dst(dst, false);
					self.add_const(d, r, -k)?;
					return Ok(Val::Reg(d, ty));
				}
				("<<" | ">>", Val::Reg(r, t), Val::Int(k)) => {
					if !(0..32).contains(&k) {
						return self.err(format!("a shift by {k}"));
					}
					if op == ">>" && t.base != Base::U32 {
						return self.err(">> on an int (only the logical shift is there: use u32)");
					}
					let d = self.dst(dst, false);
					self.push(
						alts(row.2),
						vec![Arg::Def(d, None), Arg::Use(r, None), Arg::Int(k)],
						None,
						None,
					);
					return Ok(Val::Reg(d, ty));
				}
				("<<" | ">>", ..) => return self.err(format!("{op} takes a constant count")),
				_ => {}
			}
		}
		if op == ">>" {
			// int4 >> constant
			let (Val::Reg(r, _), Val::Int(k)) = (va, vb) else {
				return self.err(">> takes a constant count");
			};
			let d = self.dst(dst, true);
			self.push(
				alts(row.2),
				vec![Arg::Def(d, None), Arg::Use(r, None), Arg::Int(k & 31)],
				None,
				None,
			);
			return Ok(Val::Reg(d, ty));
		}
		let (ra, ta) = self.reg(va, ty, None)?;
		let (rb, tb) = self.reg(vb, ty, None)?;
		if class == 'i' {
			if !int_like(ta) || !int_like(tb) {
				return self.err(format!("{op} on {} and {}", ta.show(), tb.show()));
			}
		} else if ta != tb {
			return self.err(format!(
				"{op} on {} and {} (cast one: the bits stay)",
				ta.show(),
				tb.show()
			));
		}
		let unsigned = ta.base == Base::U32 || tb.base == Base::U32;
		let d = self.dst(dst, ty.base.vector());
		self.push(
			alts(row.2),
			vec![Arg::Def(d, None), Arg::Use(ra, None), Arg::Use(rb, None)],
			None,
			None,
		);
		Ok(Val::Reg(
			d,
			if class == 'i' {
				Type {
					base: if unsigned { Base::U32 } else { Base::Int },
					ptr: false,
				}
			} else {
				ty
			},
		))
	}

	fn call(&mut self, name: &str, args: &[Expr], dst: Option<Vr>) -> Result<Val, String> {
		if !INTRINSICS.iter().any(|r| r.0 == name) {
			return self.err(format!("unknown function {name}"));
		}
		let mut vals = Vec::new();
		for a in args {
			vals.push(self.rv(a, None, None)?);
		}
		let fits = |sig: &str| {
			sig.len() == vals.len()
				&& sig.chars().zip(&vals).all(|(w, v)| match v {
					Val::Reg(_, t) => letter(*t) == w || (w == 'V' && t.base.vector() && !t.ptr),
					Val::Int(_) => w != 'p' && w != 'V',
					Val::Flt(_) => matches!(w, 'h' | 'f' | 'H' | 'F'),
				})
		};
		let Some(row) = INTRINSICS.iter().find(|r| r.0 == name && fits(r.1)) else {
			let forms: Vec<String> = INTRINSICS
				.iter()
				.filter(|r| r.0 == name)
				.map(|r| {
					format!(
						"{name}({})",
						r.1.chars()
							.map(|c| if c == '#' {
								"constant".into()
							} else if c == 'V' {
								"vector".into()
							} else if c == 'p' {
								"half *".into()
							} else {
								of_letter(c).show()
							})
							.collect::<Vec<_>>()
							.join(", ")
					)
				})
				.collect();
			return self.err(format!(
				"{name}: the arguments do not fit {}",
				forms.join(" / ")
			));
		};
		// each argument: a register, or a constant's value
		let mut regs: Vec<(Option<Vr>, i64)> = Vec::new();
		let mut first = None;
		for (w, v) in row.1.chars().zip(&vals) {
			regs.push(match (w, v) {
				('#', Val::Int(k)) => (None, *k),
				('#', _) => return self.err(format!("{name}: a constant is needed")),
				(w, v) => {
					let (r, t) = self.reg(*v, of_letter(w), None)?;
					first.get_or_insert(t);
					(Some(r), 0)
				}
			});
		}
		if row.1.starts_with("VV") {
			if let (Val::Reg(_, a), Val::Reg(_, b)) = (vals[0], vals[1]) {
				if a != b {
					return self.err(format!("{name} on {} and {}", a.show(), b.show()));
				}
			}
		}
		let ret = if row.2 == 'V' {
			first.unwrap()
		} else {
			of_letter(row.2)
		};
		let d = self.dst(dst, ret.base.vector());
		let mut tok = row.3.split_whitespace();
		let mnem = tok.next().unwrap();
		let mut out = Vec::new();
		for t in tok {
			let (who, elem) = t[1..]
				.split_once('.')
				.map_or((&t[1..], None), |(w, e)| (w, Some(e)));
			let elem = elem.map(|e| match e.strip_prefix('$') {
				Some(k) => regs[k.parse::<usize>().unwrap()].1,
				None => e.parse::<i64>().unwrap(),
			});
			if let Some(e) = elem.filter(|e| !(0..8).contains(e)) {
				return self.err(format!("{name}: no element {e}"));
			}
			let elem = elem.map(|e| e as u32);
			out.push(match who {
				"d" => Arg::Def(d, elem),
				k => match regs[k.parse::<usize>().unwrap()] {
					(Some(r), _) => Arg::Use(r, elem),
					(None, v) => Arg::Int(v),
				},
			});
		}
		let mem = if row.4 == 'L' {
			self.mem(false, regs[0].0.unwrap(), None, 0, row.5)
		} else {
			None
		};
		self.push(alts(mnem), out, None, mem);
		Ok(Val::Reg(d, ret))
	}

	/// `d = e` for the register variable `d` of type `t`.
	fn assign(&mut self, d: Vr, t: Type, e: &Expr) -> Result<(), String> {
		let v = self.rv(e, Some(d), Some(t))?;
		let (r, from) = self.reg(v, t, Some(d))?;
		self.assignable(t, from)?;
		self.mov(d, r);
		Ok(())
	}

	fn close(&mut self, term: Term) {
		let ops = std::mem::take(&mut self.cur);
		if !ops.is_empty() || !matches!(term, Term::None) {
			self.items.push(Item::Block(Block { ops, term }));
		}
	}

	/// The compare of a run-time condition, into the current block: (its index, the condition name).
	fn compare(&mut self, c: &Expr) -> Result<(usize, &'static str), String> {
		let (op, a, b) = match c {
			Expr::Binary(op @ ("==" | "!=" | "<" | "<=" | ">" | ">="), a, b) => (*op, &**a, &**b),
			Expr::Binary("&&" | "||", ..) | Expr::Unary("!", _) => {
				return self.err("&&, || and ! work on constants only: nest the ifs")
			}
			e => ("!=", e, &Expr::Int(0)),
		};
		let cc = match op {
			"==" => "EQ",
			"!=" => "NEQ",
			"<" => "LT",
			"<=" => "LTE",
			">" => "GT",
			_ => "GTE",
		};
		let u32t = Type {
			base: Base::U32,
			ptr: false,
		};
		let (va, vb) = (self.rv(a, None, None)?, self.rv(b, None, None)?);
		let scalar = |t: Type| t.ptr || matches!(t.base, Base::U32 | Base::Int);
		for v in [va, vb] {
			if let Val::Reg(_, t) = v {
				if !scalar(t) {
					return self.err(format!(
						"a condition compares integers or pointers, not {}",
						t.show()
					));
				}
			}
		}
		// as in C: unsigned when either side is
		let unsigned = [va, vb]
			.iter()
			.any(|v| matches!(v, Val::Reg(_, t) if t.ptr || t.base == Base::U32));
		let (ra, _) = self.reg(va, u32t, None)?;
		if matches!(op, "==" | "!=") && matches!(vb, Val::Int(0)) {
			return Ok((
				self.push(alts("CMU.CMZ.I32"), vec![Arg::Use(ra, None)], None, None),
				cc,
			));
		}
		let (rb, _) = self.reg(vb, u32t, None)?;
		let mnem = if unsigned {
			"CMU.CMII.U32"
		} else {
			"CMU.CMII.I32"
		};
		Ok((
			self.push(
				alts(mnem),
				vec![Arg::Use(ra, None), Arg::Use(rb, None)],
				None,
				None,
			),
			cc,
		))
	}

	fn label(&mut self) -> usize {
		self.labels += 1;
		self.labels
	}

	fn block(&mut self, body: &[Node]) -> Result<(), String> {
		self.scopes.push(HashMap::new());
		let r = body.iter().try_for_each(|n| self.stmt(n));
		self.scopes.pop();
		r
	}

	/// `body` `count` times (a register or a constant of at least 1), counted down.
	fn repeat(
		&mut self,
		count: Val,
		body: &[Node],
		step: &[Node],
		line: usize,
	) -> Result<(), String> {
		let cnt = self.temp(false);
		match count {
			Val::Int(k) => self.li(cnt, k)?,
			Val::Reg(r, _) => self.mov(cnt, r),
			Val::Flt(_) => unreachable!(),
		}
		self.close(Term::None);
		let top = self.label();
		self.items.push(Item::Label(top));
		self.items.push(Item::LoopBegin(line));
		self.scopes.push(HashMap::new());
		let r = body.iter().chain(step).try_for_each(|n| self.stmt(n));
		self.scopes.pop();
		r?;
		self.line = line;
		self.push(
			alts("IAU.SUB"),
			vec![Arg::Def(cnt, None), Arg::Use(cnt, None), Arg::Int(1)],
			None,
			None,
		);
		let cmp = self.push(alts("CMU.CMZ.I32"), vec![Arg::Use(cnt, None)], None, None);
		self.close(Term::Branch {
			cmp,
			cc: "NEQ",
			label: top,
		});
		self.items.push(Item::LoopEnd);
		Ok(())
	}

	/// `do body while (c)`; `step` runs after the body.
	fn do_while(
		&mut self,
		body: &[Node],
		step: &[Node],
		c: &Expr,
		line: usize,
	) -> Result<(), String> {
		self.close(Term::None);
		let top = self.label();
		self.items.push(Item::Label(top));
		self.items.push(Item::LoopBegin(line));
		// the condition sees the body's variables
		self.scopes.push(HashMap::new());
		let r = body
			.iter()
			.chain(step)
			.try_for_each(|n| self.stmt(n))
			.and_then(|_| {
				self.line = line;
				self.compare(c)
			});
		self.scopes.pop();
		let (cmp, cc) = r?;
		self.close(Term::Branch {
			cmp,
			cc,
			label: top,
		});
		self.items.push(Item::LoopEnd);
		Ok(())
	}

	fn if_else(&mut self, c: &Expr, then: &[Node], other: &[Node]) -> Result<(), String> {
		if let Some(k) = self.constant(c) {
			return self.block(if k != 0 { then } else { other });
		}
		let (cmp, cc) = self.compare(c)?;
		let skip = self.label();
		self.close(Term::Branch {
			cmp,
			cc: inverse(cc),
			label: skip,
		});
		self.block(then)?;
		if other.is_empty() {
			self.close(Term::None);
			self.items.push(Item::Label(skip));
		} else {
			let end = self.label();
			self.close(Term::Jump(end));
			self.items.push(Item::Label(skip));
			self.block(other)?;
			self.close(Term::None);
			self.items.push(Item::Label(end));
		}
		Ok(())
	}

	/// A `for` whose variable is a compile-time constant in every round.
	fn unroll(
		&mut self,
		name: &str,
		start: i64,
		cond: &Expr,
		step: &[Node],
		body: &[Node],
	) -> Result<(), String> {
		let mut i = start;
		for round in 0.. {
			if round > UNROLL_MAX {
				return self.err(format!("a loop of more than {UNROLL_MAX} rounds is not unrolled (#pragma nounroll, or a variable the body does not need as a constant)"));
			}
			self.scopes
				.push(HashMap::from([(name.to_string(), Bind::Const(i))]));
			let go = self.ceval(cond, "the condition of an unrolled loop");
			let r = go
				.and_then(|go| {
					if go != 0 {
						self.block(body).map(|_| true)
					} else {
						Ok(false)
					}
				})
				.and_then(|go| {
					if !go {
						return Ok(None);
					}
					// the step: assignments to the variable
					let mut next = i;
					for s in step {
						self.scopes
							.last_mut()
							.unwrap()
							.insert(name.to_string(), Bind::Const(next));
						next = match &s.stmt {
							Stmt::Assign(Expr::Name(n), None, e) if n == name => {
								self.ceval(e, "the step of an unrolled loop")?
							}
							Stmt::Assign(Expr::Name(n), Some(op), e) if n == name => self.ceval(
								&Expr::Binary(op, Box::new(Expr::Int(next)), Box::new(e.clone())),
								"the step of an unrolled loop",
							)?,
							_ => {
								return self
									.err("the step of an unrolled loop changes only its variable")
							}
						};
					}
					Ok(Some(next))
				});
			self.scopes.pop();
			match r? {
				Some(next) => i = next,
				None => break,
			}
		}
		Ok(())
	}

	/// A `for` at run time.
	fn for_runtime(
		&mut self,
		init: &[Node],
		cond: Option<&Expr>,
		step: &[Node],
		body: &[Node],
		line: usize,
	) -> Result<(), String> {
		let Some(cond) = cond else {
			return self.err("a for without a condition never ends");
		};
		// `for (T i = A; i < B; i++)` whose body does not use i: count down
		if let (
			[Node {
				stmt: Stmt::Decl {
					base: Base::U32 | Base::Int,
					items,
					..
				},
				..
			}],
			Expr::Binary("<", l, bound),
			[Node {
				stmt: Stmt::Assign(Expr::Name(s), Some("+"), Expr::Int(1)),
				..
			}],
		) = (init, cond, step)
		{
			if let [Declarator {
				ptr: false,
				name,
				len: None,
				init: Some(from),
				..
			}] = &items[..]
			{
				// (the bound must not change in the loop)
				let invariant = match &**bound {
					Expr::Name(b) if self.constant(bound).is_none() => !assigned(body, b),
					b => self.constant(b).is_some(),
				};
				if matches!(&**l, Expr::Name(n) if n == name)
					&& s == name && !mentions(body, name)
					&& invariant
				{
					return match (self.constant(from), self.constant(bound)) {
						(Some(a), Some(b)) if b <= a => Ok(()),
						// (one round: no loop)
						(Some(a), Some(b)) if b - a == 1 => self.block(body),
						(Some(a), Some(b)) => self.repeat(Val::Int(b - a), body, &[], line),
						_ => {
							// no round unless from < bound, compared as the variable's type
							let Stmt::Decl { base, .. } = &init[0].stmt else {
								unreachable!()
							};
							let from_typed = Expr::Cast(
								Type {
									base: *base,
									ptr: false,
								},
								Box::new(from.clone()),
							);
							let (cmp, cc) = self.compare(&Expr::Binary(
								"<",
								Box::new(from_typed),
								bound.clone(),
							))?;
							let skip = self.label();
							self.close(Term::Branch {
								cmp,
								cc: inverse(cc),
								label: skip,
							});
							let n = self.rv(
								&Expr::Binary("-", bound.clone(), Box::new(from.clone())),
								None,
								None,
							)?;
							self.repeat(n, body, &[], line)?;
							self.close(Term::None);
							self.items.push(Item::Label(skip));
							Ok(())
						}
					};
				}
			}
		}
		// in general: init; if (cond) do { body; step; } while (cond);
		self.scopes.push(HashMap::new());
		let r = init.iter().try_for_each(|n| self.stmt(n)).and_then(|_| {
			self.line = line;
			if let Some(k) = self.constant(cond) {
				return if k == 0 {
					Ok(())
				} else {
					self.err("the condition is a constant that is not 0: the loop never ends")
				};
			}
			let (cmp, cc) = self.compare(cond)?;
			let skip = self.label();
			self.close(Term::Branch {
				cmp,
				cc: inverse(cc),
				label: skip,
			});
			self.do_while(body, step, cond, line)?;
			self.close(Term::None);
			self.items.push(Item::Label(skip));
			Ok(())
		});
		self.scopes.pop();
		r
	}

	fn declare(
		&mut self,
		base: Base,
		register: bool,
		konst: bool,
		d: &Declarator,
	) -> Result<(), String> {
		let t = Type { base, ptr: d.ptr };
		if d.restrict && !d.ptr {
			return self.err(format!("{}: restrict goes with a pointer", d.name));
		}
		if let Some(len) = &d.len {
			let n = self.ceval(len, &format!("the length of {}", d.name))?;
			if d.init.is_some() {
				return self.err(format!("{}: an array has no initial value", d.name));
			}
			if register {
				if !(1..=64).contains(&n) {
					return self.err(format!("{}: a register array has 1..64 elements", d.name));
				}
				// (restrict: every element points to memory of its own)
				let regs = (0..n)
					.map(|k| {
						let tag = d.restrict.then(|| self.tag());
						self.vreg(base.vector() && !d.ptr, &format!("{}[{k}]", d.name), tag)
					})
					.collect();
				return self.bind(&d.name, Bind::Arr(regs, t));
			}
			if d.ptr || matches!(base, Base::Void | Base::U8) {
				return self.err(format!("{}: a stack array holds u32, int, float, half or vectors (`register` for an array of registers)", d.name));
			}
			let bytes = n * base.size();
			if !(1..=4096).contains(&bytes) {
				return self.err(format!(
					"{}: a stack array of 1..4096 bytes (a SHAVE's stack is 7 KB)",
					d.name
				));
			}
			let tag = self.tag();
			self.bind(&d.name, Bind::Stack(self.stack, base, tag))?;
			self.stack += (bytes + 15) / 16 * 16;
			return Ok(());
		}
		if !d.ptr && matches!(base, Base::Void | Base::U8) {
			return self.err(format!(
				"{}: {} is only a pointer's target",
				d.name,
				base.name()
			));
		}
		// `const u32 N = 4;` is a constant, not a register
		if konst && int_like(t) {
			if let Some(v) = d.init.as_ref().and_then(|e| self.constant(e)) {
				return self.bind(&d.name, Bind::Const(v));
			}
		}
		// in front of the function: `const float X = 1.0 / 64;` names a number
		if konst && !d.ptr && matches!(base, Base::Float | Base::Half) && self.scopes.len() == 1 {
			match d.init.as_ref().and_then(|e| self.number(e)) {
				Some(Num::Int(v)) => return self.bind(&d.name, Bind::Flt(v as f64)),
				Some(Num::Flt(x)) => return self.bind(&d.name, Bind::Flt(x)),
				None => {}
			}
		}
		let r = self.vreg(base.vector() && !d.ptr, &d.name, None);
		if let Some(e) = &d.init {
			// (the initial value may name an outer variable of the same name)
			let based = if d.ptr { self.based_on(e) } else { None };
			self.assign(r, t, e)?;
			self.vregs[r].tag = based;
		}
		if d.restrict {
			self.vregs[r].tag = Some(self.tag());
		}
		self.bind(&d.name, Bind::Reg(r, t))
	}

	fn stmt(&mut self, node: &Node) -> Result<(), String> {
		self.line = node.line;
		match &node.stmt {
			Stmt::Decl {
				base,
				register,
				konst,
				items,
			} => {
				for d in items {
					self.declare(*base, *register, *konst, d)?;
				}
			}
			Stmt::Assign(lhs, op, rhs) => {
				let value = match op {
					Some(op) => Expr::Binary(op, Box::new(lhs.clone()), Box::new(rhs.clone())),
					None => rhs.clone(),
				};
				if let Some((d, t)) = self.var(lhs)? {
					// (a pointer's first value may be any address)
					if t.ptr && self.version[d] > 0 {
						if let Some(tag) = self.vregs[d].tag {
							if self.based_on(&value) != Some(tag) {
								return self.err("a pointer based on a restrict pointer takes only addresses based on the same one: declare another pointer");
							}
						}
					}
					return self.assign(d, t, &value);
				}
				match lhs {
					Expr::Index(a, ix) => match self.place(a, ix)? {
						Some((elem, base, tag, off)) => self.store(elem, base, tag, off, &value)?,
						None => return self.err("a vector's lane cannot be assigned"),
					},
					Expr::Name(n) => return self.err(format!("{n} cannot be assigned")),
					_ => unreachable!(),
				}
			}
			Stmt::Call(Expr::Call(name, args)) if name == "barrier" && args.is_empty() => {
				self.close(Term::None)
			}
			Stmt::Call(_) => return self.err("a call whose result is not used"),
			Stmt::Assert(c, text) => {
				if self.ceval(c, "a static assertion")? == 0 {
					return self.err(format!(
						"static assertion failed{}{text}",
						if text.is_empty() { "" } else { ": " }
					));
				}
			}
			Stmt::Block(body) => self.block(body)?,
			Stmt::If(c, then, other) => self.if_else(c, then, other)?,
			Stmt::DoWhile(body, c) => match self.constant(c) {
				Some(0) => self.block(body)?,
				Some(_) => {
					return self
						.err("the condition is a constant that is not 0: the loop never ends")
				}
				None => self.do_while(body, &[], c, node.line)?,
			},
			Stmt::While(c, body) => {
				let inner = [Node {
					stmt: Stmt::DoWhile(body.clone(), c.clone()),
					line: node.line,
				}];
				match self.constant(c) {
					Some(0) => {}
					Some(_) => {
						return self
							.err("the condition is a constant that is not 0: the loop never ends")
					}
					None => self.if_else(c, &inner, &[])?,
				}
			}
			Stmt::For(init, cond, step, body, pragma) => {
				// `for (int i = constant; ...)`: i may be a constant of each round
				let var = match &init[..] {
					[Node {
						stmt:
							Stmt::Decl {
								base: Base::U32 | Base::Int,
								items,
								..
							},
						..
					}] => match &items[..] {
						[Declarator {
							ptr: false,
							name,
							len: None,
							init: Some(e),
							..
						}] => self.constant(e).map(|v| (name.clone(), v)),
						_ => None,
					},
					_ => None,
				};
				let unrolled = |l: &mut Lower| {
					match (&var, cond) {
					(Some((name, start)), Some(c)) => l.unroll(name, *start, c, step, body),
					_ => l.err("#pragma unroll: the loop must be `for (int i = constant; condition on i; step of i)`"),
				}
				};
				match pragma {
					Some(true) => unrolled(self)?,
					Some(false) => self.for_runtime(init, cond.as_ref(), step, body, node.line)?,
					None => {
						// at run time where the body allows it, else unrolled
						let mut trial = self.clone();
						match trial.for_runtime(init, cond.as_ref(), step, body, node.line) {
							Ok(()) => *self = trial,
							Err(first) => {
								self.line = node.line;
								if var.is_none() {
									return Err(first);
								}
								let mut trial = self.clone();
								match unrolled(&mut trial) {
									Ok(()) => *self = trial,
									// the more telling of the two errors
									Err(second) => {
										return Err(if second.contains("unrolled loop") {
											first
										} else {
											second
										})
									}
								}
							}
						}
					}
				}
			}
		}
		Ok(())
	}
}

/// Is `name` assigned anywhere in `nodes`?
fn assigned(nodes: &[Node], name: &str) -> bool {
	nodes.iter().any(|n| match &n.stmt {
		Stmt::Assign(Expr::Name(l), ..) => l == name,
		Stmt::If(_, a, b) => assigned(a, name) || assigned(b, name),
		Stmt::DoWhile(a, _) | Stmt::While(_, a) | Stmt::Block(a) => assigned(a, name),
		Stmt::For(i, _, s, b, _) => assigned(i, name) || assigned(s, name) || assigned(b, name),
		_ => false,
	})
}

pub fn lower(file: &File) -> Result<Lowered, String> {
	let mut l = Lower {
		vregs: Vec::new(),
		version: Vec::new(),
		items: Vec::new(),
		cur: Vec::new(),
		scopes: vec![HashMap::new()],
		tags: 0,
		labels: 0,
		stack: 0,
		line: 0,
	};
	l.vreg(false, &file.param.1, None);
	l.vreg(false, "(stack pointer)", None);
	l.vregs[PARAMS].pin = Some(18);
	l.vregs[SP].pin = Some(19);
	// nothing a kernel stores to overlaps its parameter block
	l.vregs[PARAMS].tag = Some(l.tag());
	for (name, v) in &file.defines {
		l.scopes[0].insert(name.clone(), Bind::Const(*v));
	}
	for g in &file.globals {
		match &g.stmt {
			Stmt::Decl { konst: true, .. } | Stmt::Assert(..) => l.stmt(g)?,
			_ => {
				return Err(format!(
				"line {}: only constants (const u32 NAME = ...;) are declared outside the function",
				g.line
			))
			}
		}
		if !l.cur.is_empty() || l.vregs.len() > 2 {
			return Err(format!(
				"line {}: a constant's value must be known at compile time",
				g.line
			));
		}
	}
	if !file.param.0.ptr {
		return Err(format!(
			"the kernel's parameter is a pointer: void controlled_act(u32 *{})",
			file.param.1
		));
	}
	l.bind(&file.param.1, Bind::Reg(PARAMS, file.param.0))?;
	l.block(&file.body)?;
	l.line = 0;
	if l.stack > 0 {
		// the frame: taken in front of the first operation, given back before the return
		let (body, size) = (std::mem::take(&mut l.cur), l.stack);
		l.add_const(SP, SP, -size)?;
		let enter = std::mem::replace(&mut l.cur, body);
		l.items.insert(
			0,
			Item::Block(Block {
				ops: enter,
				term: Term::None,
			}),
		);
		l.add_const(SP, SP, size)?;
	}
	l.close(Term::Ret);
	Ok(Lowered {
		vregs: l.vregs,
		items: l.items,
	})
}
