//! Preprocessor, lexer and parser of the kernel language
//! (`crates/npu/SHAVE-DSL.md`): a subset of C.

use crate::shave::asm::strip_comments;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
	Ident(String),
	Int(i64),
	Float(f64),
	Str(String),
	P(&'static str),
	End,
}

const PUNCT: [&str; 39] = [
	"<<=", ">>=", "++", "--", "+=", "-=", "*=", "&=", "|=", "^=", "==", "!=", "<=", ">=", "<<",
	">>", "&&", "||", "(", ")", "{", "}", "[", "]", ",", ";", "=", "<", ">", "+", "-", "*", "/",
	"%", "&", "|", "^", "!", "~",
];

pub fn lex(src: &str) -> Result<Vec<(Tok, usize)>, String> {
	let b = src.as_bytes();
	let (mut at, mut line) = (0, 1);
	let mut out = Vec::new();
	let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
	while at < b.len() {
		let c = b[at];
		if c == b'\n' {
			line += 1;
			at += 1;
		} else if c.is_ascii_whitespace() {
			at += 1;
		} else if c == b'"' {
			let end = src[at + 1..]
				.find('"')
				.ok_or(format!("line {line}: string without an end"))?;
			out.push((Tok::Str(src[at + 1..at + 1 + end].to_string()), line));
			at += end + 2;
		} else if c.is_ascii_digit()
			|| (c == b'.' && b.get(at + 1).is_some_and(|d| d.is_ascii_digit()))
		{
			let start = at;
			if src[at..].starts_with("0x") || src[at..].starts_with("0X") {
				at += 2;
				while at < b.len() && b[at].is_ascii_hexdigit() {
					at += 1;
				}
				out.push((
					Tok::Int(
						i64::from_str_radix(&src[start + 2..at], 16)
							.map_err(|_| format!("line {line}: bad number"))?,
					),
					line,
				));
			} else {
				while at < b.len() && b[at].is_ascii_digit() {
					at += 1;
				}
				let mut float = false;
				if at < b.len() && b[at] == b'.' {
					float = true;
					at += 1;
					while at < b.len() && b[at].is_ascii_digit() {
						at += 1;
					}
				}
				if at < b.len() && (b[at] == b'e' || b[at] == b'E') {
					let mut k = at + 1;
					if k < b.len() && (b[k] == b'-' || b[k] == b'+') {
						k += 1;
					}
					if k < b.len() && b[k].is_ascii_digit() {
						float = true;
						while k < b.len() && b[k].is_ascii_digit() {
							k += 1;
						}
						at = k;
					}
				}
				let t = &src[start..at];
				let bad = format!("line {line}: bad number {t}");
				out.push((
					if float {
						Tok::Float(t.parse().map_err(|_| bad)?)
					} else {
						Tok::Int(t.parse().map_err(|_| bad)?)
					},
					line,
				));
				if float && at < b.len() && (b[at] == b'f' || b[at] == b'F') {
					at += 1;
				}
			}
			// 1u, 1U, 1ul
			while at < b.len() && matches!(b[at], b'u' | b'U' | b'l' | b'L') {
				at += 1;
			}
			if at < b.len() && ident(b[at]) {
				return Err(format!("line {line}: bad number {}", &src[start..=at]));
			}
		} else if c.is_ascii_alphabetic() || c == b'_' {
			let start = at;
			while at < b.len() && ident(b[at]) {
				at += 1;
			}
			out.push((Tok::Ident(src[start..at].to_string()), line));
		} else {
			let p = PUNCT
				.iter()
				.find(|p| src[at..].starts_with(**p))
				.ok_or_else(|| {
					format!(
						"line {line}: unexpected `{}`",
						src[at..].chars().next().unwrap()
					)
				})?;
			out.push((Tok::P(p), line));
			at += p.len();
		}
	}
	out.push((Tok::End, line));
	Ok(out)
}

/// What a value is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
	Void,
	U8,
	U32,
	Int,
	/// an f16 in the low half of an int register
	Half,
	/// an f32 in an int register
	Float,
	Half8,
	Float4,
	Int4,
}

impl Base {
	fn parse(s: &str) -> Option<Base> {
		Some(match s {
			"void" => Base::Void,
			"u8" | "char" => Base::U8,
			"u32" | "unsigned" => Base::U32,
			"int" => Base::Int,
			"half" => Base::Half,
			"float" => Base::Float,
			"half8" => Base::Half8,
			"float4" => Base::Float4,
			"int4" => Base::Int4,
			_ => return None,
		})
	}

	pub fn name(self) -> &'static str {
		match self {
			Base::Void => "void",
			Base::U8 => "u8",
			Base::U32 => "u32",
			Base::Int => "int",
			Base::Half => "half",
			Base::Float => "float",
			Base::Half8 => "half8",
			Base::Float4 => "float4",
			Base::Int4 => "int4",
		}
	}

	/// Bytes of one element in memory.
	pub fn size(self) -> i64 {
		match self {
			Base::Void | Base::U8 => 1,
			Base::Half => 2,
			Base::U32 | Base::Int | Base::Float => 4,
			Base::Half8 | Base::Float4 | Base::Int4 => 16,
		}
	}

	pub fn vector(self) -> bool {
		matches!(self, Base::Half8 | Base::Float4 | Base::Int4)
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Type {
	pub base: Base,
	pub ptr: bool,
}

impl Type {
	pub fn show(self) -> String {
		format!("{}{}", self.base.name(), if self.ptr { " *" } else { "" })
	}
}

#[derive(Clone, Debug)]
pub enum Expr {
	Int(i64),
	Float(f64),
	Name(String),
	/// `a[i]`; `*a` is `a[0]`
	Index(Box<Expr>, Box<Expr>),
	Call(String, Vec<Expr>),
	Unary(&'static str, Box<Expr>),
	Binary(&'static str, Box<Expr>, Box<Expr>),
	Cast(Type, Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct Declarator {
	pub ptr: bool,
	pub restrict: bool,
	pub name: String,
	pub len: Option<Expr>,
	pub init: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
	Decl {
		base: Base,
		register: bool,
		konst: bool,
		items: Vec<Declarator>,
	},
	/// `lhs = rhs`, `lhs op= rhs` (`lhs++` is `lhs += 1`)
	Assign(Expr, Option<&'static str>, Expr),
	Call(Expr),
	If(Expr, Vec<Node>, Vec<Node>),
	DoWhile(Vec<Node>, Expr),
	While(Expr, Vec<Node>),
	/// init, condition, step, body, `#pragma unroll` (true) / `nounroll` (false)
	For(Vec<Node>, Option<Expr>, Vec<Node>, Vec<Node>, Option<bool>),
	Block(Vec<Node>),
	Assert(Expr, String),
}

/// A statement and its source line.
#[derive(Clone, Debug)]
pub struct Node {
	pub stmt: Stmt,
	pub line: usize,
}

#[derive(Debug)]
pub struct File {
	/// `#define`d and `-D` constants
	pub defines: Vec<(String, i64)>,
	/// declarations in front of the function (constants)
	pub globals: Vec<Node>,
	/// the kernel function's parameter: its type and name
	pub param: (Type, String),
	pub body: Vec<Node>,
}

/// A compile-time number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
	Int(i64),
	Flt(f64),
}

/// A compile-time integer: literals, names `lookup` knows, C's operators.
pub fn fold(e: &Expr, lookup: &dyn Fn(&str) -> Option<i64>) -> Result<i64, String> {
	match fold_num(e, &|n| lookup(n).map(Num::Int))? {
		Num::Int(v) => Ok(v),
		Num::Flt(_) => Err("a float where an integer constant is needed".into()),
	}
}

/// A compile-time number: as [`fold`], and floating arithmetic (`+ - * /`,
/// the comparisons, `&& || !`; an integer beside a float converts, as in C).
pub fn fold_num(e: &Expr, lookup: &dyn Fn(&str) -> Option<Num>) -> Result<Num, String> {
	use Num::{Flt, Int};
	let truth = |v: bool| Int(v as i64);
	Ok(match e {
		Expr::Int(v) => Int(*v),
		Expr::Float(x) => Flt(*x),
		Expr::Name(n) => lookup(n).ok_or_else(|| format!("{n} is not a constant"))?,
		Expr::Index(..) | Expr::Call(..) => return Err("not a constant".into()),
		Expr::Cast(t, a) if !t.ptr && matches!(t.base, Base::U32 | Base::Int) => {
			match fold_num(a, lookup)? {
				Int(v) => Int(v),
				Flt(_) => return Err("a float where an integer constant is needed".into()),
			}
		}
		Expr::Cast(..) => return Err("not a constant".into()),
		Expr::Unary(op, a) => match (*op, fold_num(a, lookup)?) {
			("-", Int(a)) => Int(a.wrapping_neg()),
			("-", Flt(a)) => Flt(-a),
			("!", Int(a)) => truth(a == 0),
			("!", Flt(a)) => truth(a == 0.0),
			("~", Int(a)) => Int(!a),
			("~", Flt(_)) => return Err("~ on a floating constant".into()),
			_ => return Err("not a constant".into()),
		},
		Expr::Binary(op, a, b) => match (fold_num(a, lookup)?, fold_num(b, lookup)?) {
			(Int(a), Int(b)) => Int(match *op {
				"+" => a.wrapping_add(b),
				"-" => a.wrapping_sub(b),
				"*" => a.wrapping_mul(b),
				"/" | "%" if b == 0 => return Err("division by zero".into()),
				"/" => a.wrapping_div(b),
				"%" => a.wrapping_rem(b),
				"<<" => a.wrapping_shl(b as u32),
				">>" => a.wrapping_shr(b as u32),
				"&" => a & b,
				"|" => a | b,
				"^" => a ^ b,
				"==" => (a == b) as i64,
				"!=" => (a != b) as i64,
				"<" => (a < b) as i64,
				"<=" => (a <= b) as i64,
				">" => (a > b) as i64,
				">=" => (a >= b) as i64,
				"&&" => (a != 0 && b != 0) as i64,
				_ => (a != 0 || b != 0) as i64,
			}),
			(a, b) => {
				let f = |v| match v {
					Int(v) => v as f64,
					Flt(x) => x,
				};
				let (a, b) = (f(a), f(b));
				match *op {
					"+" => Flt(a + b),
					"-" => Flt(a - b),
					"*" => Flt(a * b),
					"/" if b == 0.0 => return Err("division by zero".into()),
					"/" => Flt(a / b),
					"==" => truth(a == b),
					"!=" => truth(a != b),
					"<" => truth(a < b),
					"<=" => truth(a <= b),
					">" => truth(a > b),
					">=" => truth(a >= b),
					"&&" => truth(a != 0.0 && b != 0.0),
					"||" => truth(a != 0.0 || b != 0.0),
					_ => return Err(format!("{op} on a floating constant")),
				}
			}
		},
	})
}

/// The preprocessor: `#define NAME value`, `#if` / `#ifdef` / `#ifndef` /
/// `#elif` / `#else` / `#endif`, `#error`, `#pragma unroll` / `nounroll`.
/// Returns the source with directives and inactive lines blanked, and the
/// constants (`defs` first: the command line's `-DNAME=VALUE`).
pub fn preprocess(
	src: &str,
	defs: &[(String, i64)],
) -> Result<(String, Vec<(String, i64)>), String> {
	let mut consts: Vec<(String, i64)> = defs.to_vec();
	let mut out = String::new();
	// per open #if: (the enclosing code is active, a branch was taken, this branch is active)
	let mut stack: Vec<(bool, bool, bool)> = Vec::new();
	for (n, raw) in strip_comments(src).lines().enumerate() {
		let line = n + 1;
		let active = stack.last().is_none_or(|s| s.2);
		let Some(d) = raw.trim_start().strip_prefix('#') else {
			if active {
				out.push_str(raw);
			}
			out.push('\n');
			continue;
		};
		let d = d.trim();
		let (word, rest) = d
			.split_once(char::is_whitespace)
			.map_or((d, ""), |(w, r)| (w, r.trim()));
		let value = |text: &str, consts: &[(String, i64)]| -> Result<i64, String> {
			let at = |e: String| e.replacen("line 1", &format!("line {line}"), 1);
			let mut p = Parser {
				toks: lex(text).map_err(at)?,
				at: 0,
			};
			let e = p.expr(0).map_err(at)?;
			if *p.peek() != Tok::End {
				return Err(format!("line {line}: #{word}: more than one expression"));
			}
			fold(&e, &|n| consts.iter().rev().find(|c| c.0 == n).map(|c| c.1))
				.map_err(|e| format!("line {line}: #{word}: {e}"))
		};
		let defined = |name: &str, consts: &[(String, i64)]| consts.iter().any(|c| c.0 == name);
		match word {
			"if" | "ifdef" | "ifndef" => {
				let take = active
					&& match word {
						"if" => value(rest, &consts)? != 0,
						"ifdef" => defined(rest, &consts),
						_ => !defined(rest, &consts),
					};
				stack.push((active, take, take));
			}
			"elif" | "else" => {
				let s = stack
					.last()
					.copied()
					.ok_or(format!("line {line}: #{word} without #if"))?;
				let take = s.0 && !s.1 && (word == "else" || value(rest, &consts)? != 0);
				*stack.last_mut().unwrap() = (s.0, s.1 || take, take);
			}
			"endif" => {
				stack
					.pop()
					.ok_or(format!("line {line}: #endif without #if"))?;
			}
			_ if !active => {}
			"define" => {
				let (name, text) = rest
					.split_once(char::is_whitespace)
					.map_or((rest, ""), |(a, b)| (a, b.trim()));
				if name.is_empty() || name.contains('(') {
					return Err(format!(
						"line {line}: #define NAME value (macros with arguments are not supported)"
					));
				}
				let v = if text.is_empty() {
					1
				} else {
					value(text, &consts)?
				};
				consts.push((name.to_string(), v));
			}
			"error" => return Err(format!("line {line}: #error {rest}")),
			"pragma" if rest == "unroll" => out.push_str("__unroll"),
			"pragma" if rest == "nounroll" => out.push_str("__nounroll"),
			"pragma" => {}
			w => return Err(format!("line {line}: #{w} is not supported")),
		}
		out.push('\n');
	}
	if !stack.is_empty() {
		return Err("#if without #endif".into());
	}
	Ok((out, consts))
}

struct Parser {
	toks: Vec<(Tok, usize)>,
	at: usize,
}

/// Binary operators by binding strength.
const LEVELS: [&[&str]; 10] = [
	&["||"],
	&["&&"],
	&["|"],
	&["^"],
	&["&"],
	&["==", "!="],
	&["<", "<=", ">", ">="],
	&["<<", ">>"],
	&["+", "-"],
	&["*", "/", "%"],
];
const ASSIGN: [(&str, &str); 8] = [
	("+=", "+"),
	("-=", "-"),
	("*=", "*"),
	("&=", "&"),
	("|=", "|"),
	("^=", "^"),
	("<<=", "<<"),
	(">>=", ">>"),
];

impl Parser {
	fn peek(&self) -> &Tok {
		&self.toks[self.at].0
	}

	fn peek2(&self) -> &Tok {
		&self.toks[(self.at + 1).min(self.toks.len() - 1)].0
	}

	fn line(&self) -> usize {
		self.toks[self.at].1
	}

	fn err<T>(&self, what: &str) -> Result<T, String> {
		let got = match self.peek() {
			Tok::Ident(s) => format!("`{s}`"),
			Tok::Int(v) => v.to_string(),
			Tok::Float(v) => v.to_string(),
			Tok::Str(s) => format!("\"{s}\""),
			Tok::P(p) => format!("`{p}`"),
			Tok::End => "the end of the file".into(),
		};
		Err(format!(
			"line {}: expected {what}, found {got}",
			self.line()
		))
	}

	fn next(&mut self) -> Tok {
		let t = self.toks[self.at].0.clone();
		if self.at + 1 < self.toks.len() {
			self.at += 1;
		}
		t
	}

	fn eat(&mut self, p: &str) -> bool {
		let hit = matches!(self.peek(), Tok::P(q) if *q == p);
		if hit {
			self.at += 1;
		}
		hit
	}

	fn word(&mut self, w: &str) -> bool {
		let hit = matches!(self.peek(), Tok::Ident(s) if s == w);
		if hit {
			self.at += 1;
		}
		hit
	}

	fn expect(&mut self, p: &str) -> Result<(), String> {
		if self.eat(p) {
			Ok(())
		} else {
			self.err(&format!("`{p}`"))
		}
	}

	fn ident(&mut self) -> Result<String, String> {
		match self.peek().clone() {
			Tok::Ident(s) => {
				self.at += 1;
				Ok(s)
			}
			_ => self.err("a name"),
		}
	}

	fn type_ahead(&self) -> bool {
		matches!(self.peek(), Tok::Ident(s) if Base::parse(s).is_some() || matches!(s.as_str(), "const" | "register" | "static"))
	}

	/// `[const] [register] base [int] [const]`: (base, register, const)
	fn base(&mut self) -> Result<(Base, bool, bool), String> {
		let (mut register, mut konst) = (false, false);
		loop {
			if self.word("const") {
				konst = true;
			} else if self.word("register") {
				register = true;
			} else if !self.word("static") {
				break;
			}
		}
		let name = match self.peek() {
			Tok::Ident(s) if Base::parse(s).is_some() => s.clone(),
			_ => return self.err("a type (void u8 u32 int half float half8 float4 int4)"),
		};
		self.at += 1;
		if name == "unsigned" {
			self.word("int");
		}
		if self.word("const") {
			konst = true;
		}
		Ok((Base::parse(&name).unwrap(), register, konst))
	}

	/// `[*] [const] [restrict]`: (pointer, restrict)
	fn stars(&mut self) -> (bool, bool) {
		let ptr = self.eat("*");
		let mut restrict = false;
		loop {
			if self.word("restrict") || self.word("__restrict") {
				restrict = true;
			} else if !self.word("const") {
				break;
			}
		}
		(ptr, restrict)
	}

	fn primary(&mut self) -> Result<Expr, String> {
		match self.next() {
			Tok::Int(v) => Ok(Expr::Int(v)),
			Tok::Float(v) => Ok(Expr::Float(v)),
			Tok::P("(") => {
				if self.type_ahead() {
					let (base, ..) = self.base()?;
					let (ptr, _) = self.stars();
					self.expect(")")?;
					return Ok(Expr::Cast(Type { base, ptr }, Box::new(self.unary()?)));
				}
				let e = self.expr(0)?;
				self.expect(")")?;
				Ok(e)
			}
			Tok::Ident(name) => {
				if self.eat("(") {
					let mut args = Vec::new();
					if !self.eat(")") {
						loop {
							args.push(self.expr(0)?);
							if self.eat(")") {
								break;
							}
							self.expect(",")?;
						}
					}
					Ok(Expr::Call(name, args))
				} else {
					Ok(Expr::Name(name))
				}
			}
			_ => {
				self.at -= 1;
				self.err("a value")
			}
		}
	}

	fn unary(&mut self) -> Result<Expr, String> {
		for op in ["-", "!", "~", "&"] {
			if self.eat(op) {
				return Ok(Expr::Unary(op, Box::new(self.unary()?)));
			}
		}
		if self.eat("*") {
			return Ok(Expr::Index(Box::new(self.unary()?), Box::new(Expr::Int(0))));
		}
		if self.eat("+") {
			return self.unary();
		}
		let mut e = self.primary()?;
		while self.eat("[") {
			let index = self.expr(0)?;
			self.expect("]")?;
			e = Expr::Index(Box::new(e), Box::new(index));
		}
		Ok(e)
	}

	fn expr(&mut self, level: usize) -> Result<Expr, String> {
		if level == LEVELS.len() {
			return self.unary();
		}
		let mut left = self.expr(level + 1)?;
		loop {
			let Some(op) = LEVELS[level]
				.iter()
				.find(|op| matches!(self.peek(), Tok::P(p) if p == *op))
			else {
				return Ok(left);
			};
			self.at += 1;
			left = Expr::Binary(op, Box::new(left), Box::new(self.expr(level + 1)?));
		}
	}

	/// A declaration without its `;`.
	fn decl(&mut self) -> Result<Stmt, String> {
		let (base, register, konst) = self.base()?;
		let mut items = Vec::new();
		loop {
			let (ptr, restrict) = self.stars();
			let name = self.ident()?;
			let len = if self.eat("[") {
				let e = self.expr(0)?;
				self.expect("]")?;
				Some(e)
			} else {
				None
			};
			let init = if self.eat("=") {
				Some(self.expr(0)?)
			} else {
				None
			};
			items.push(Declarator {
				ptr,
				restrict,
				name,
				len,
				init,
			});
			if !self.eat(",") {
				break;
			}
		}
		Ok(Stmt::Decl {
			base,
			register,
			konst,
			items,
		})
	}

	/// An assignment, increment or call, without its `;`.
	fn simple(&mut self) -> Result<Stmt, String> {
		let line = self.line();
		for (op, by) in [("++", "+"), ("--", "-")] {
			if self.eat(op) {
				return Ok(Stmt::Assign(self.unary()?, Some(by), Expr::Int(1)));
			}
		}
		let e = self.expr(0)?;
		let lvalue = matches!(e, Expr::Name(_) | Expr::Index(..));
		let stmt = if self.eat("=") {
			Stmt::Assign(e, None, self.expr(0)?)
		} else if let Some((_, op)) = ASSIGN
			.iter()
			.find(|(a, _)| matches!(self.peek(), Tok::P(p) if p == a))
		{
			self.at += 1;
			Stmt::Assign(e, Some(op), self.expr(0)?)
		} else if self.eat("++") {
			Stmt::Assign(e, Some("+"), Expr::Int(1))
		} else if self.eat("--") {
			Stmt::Assign(e, Some("-"), Expr::Int(1))
		} else if matches!(e, Expr::Call(..)) {
			return Ok(Stmt::Call(e));
		} else {
			return self.err("`=`");
		};
		if !lvalue {
			return Err(format!(
				"line {line}: only a variable, `p[i]` or `*p` can be assigned"
			));
		}
		Ok(stmt)
	}

	fn list(&mut self, end: &str) -> Result<Vec<Node>, String> {
		let mut v = Vec::new();
		while !matches!(self.peek(), Tok::P(p) if *p == end) {
			let line = self.line();
			v.push(Node {
				stmt: self.simple()?,
				line,
			});
			if !self.eat(",") {
				break;
			}
		}
		Ok(v)
	}

	fn block(&mut self) -> Result<Vec<Node>, String> {
		self.expect("{")?;
		let mut body = Vec::new();
		while !self.eat("}") {
			if *self.peek() == Tok::End {
				return self.err("`}`");
			}
			body.extend(self.stmt()?);
		}
		Ok(body)
	}

	/// The statement after `if (..)`, `for (..)` ...: a block or one statement.
	fn body(&mut self) -> Result<Vec<Node>, String> {
		if matches!(self.peek(), Tok::P("{")) {
			self.block()
		} else {
			Ok(self.stmt()?.into_iter().collect())
		}
	}

	fn paren(&mut self) -> Result<Expr, String> {
		self.expect("(")?;
		let e = self.expr(0)?;
		self.expect(")")?;
		Ok(e)
	}

	fn stmt(&mut self) -> Result<Option<Node>, String> {
		let line = self.line();
		let mut pragma = None;
		loop {
			if self.word("__unroll") {
				pragma = Some(true);
			} else if self.word("__nounroll") {
				pragma = Some(false);
			} else {
				break;
			}
		}
		if pragma.is_some() && !matches!(self.peek(), Tok::Ident(s) if s == "for") {
			return Err(format!(
				"line {line}: #pragma unroll / nounroll goes in front of a `for`"
			));
		}
		let stmt = if self.eat(";") {
			return Ok(None);
		} else if matches!(self.peek(), Tok::P("{")) {
			Stmt::Block(self.block()?)
		} else if self.word("if") {
			let c = self.paren()?;
			let then = self.body()?;
			let other = if self.word("else") {
				self.body()?
			} else {
				Vec::new()
			};
			Stmt::If(c, then, other)
		} else if self.word("do") {
			let body = self.body()?;
			if !self.word("while") {
				return self.err("`while`");
			}
			let c = self.paren()?;
			self.expect(";")?;
			Stmt::DoWhile(body, c)
		} else if self.word("while") {
			let c = self.paren()?;
			Stmt::While(c, self.body()?)
		} else if self.word("for") {
			self.expect("(")?;
			let init = if self.type_ahead() {
				let l = self.line();
				vec![Node {
					stmt: self.decl()?,
					line: l,
				}]
			} else {
				self.list(";")?
			};
			self.expect(";")?;
			let cond = if matches!(self.peek(), Tok::P(";")) {
				None
			} else {
				Some(self.expr(0)?)
			};
			self.expect(";")?;
			let step = self.list(")")?;
			self.expect(")")?;
			Stmt::For(init, cond, step, self.body()?, pragma)
		} else if self.word("_Static_assert") || self.word("static_assert") {
			self.expect("(")?;
			let c = self.expr(0)?;
			let text = if self.eat(",") {
				match self.next() {
					Tok::Str(s) => s,
					_ => {
						self.at -= 1;
						return self.err("a message in quotes");
					}
				}
			} else {
				String::new()
			};
			self.expect(")")?;
			self.expect(";")?;
			Stmt::Assert(c, text)
		} else if self.type_ahead() {
			let d = self.decl()?;
			self.expect(";")?;
			d
		} else {
			let s = self.simple()?;
			self.expect(";")?;
			s
		};
		Ok(Some(Node { stmt, line }))
	}
}

/// Parse a source: constants, then `void NAME(TYPE *PARAM) { ... }`.
pub fn parse(src: &str, defs: &[(String, i64)]) -> Result<File, String> {
	let (text, defines) = preprocess(src, defs)?;
	let mut p = Parser {
		toks: lex(&text)?,
		at: 0,
	};
	let mut globals = Vec::new();
	loop {
		if *p.peek() == Tok::End {
			return Err("no kernel function: void controlled_act(u32 *params) { ... }".into());
		}
		// a function: `[static] [inline] void NAME (`
		let start = p.at;
		while p.word("static") || p.word("inline") {}
		if p.word("void") && matches!(p.peek(), Tok::Ident(_)) && matches!(p.peek2(), Tok::P("(")) {
			p.ident()?;
			p.expect("(")?;
			let (base, ..) = p.base()?;
			let (ptr, _) = p.stars();
			let name = p.ident()?;
			p.expect(")")?;
			let body = p.block()?;
			if *p.peek() != Tok::End {
				return p.err("the end of the file (one function per kernel)");
			}
			return Ok(File {
				defines,
				globals,
				param: (Type { base, ptr }, name),
				body,
			});
		}
		p.at = start;
		if !p.type_ahead()
			&& !matches!(p.peek(), Tok::Ident(s) if s == "_Static_assert" || s == "static_assert")
		{
			return p.err("a constant's declaration or the kernel function");
		}
		globals.extend(p.stmt()?);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tokens() {
		let t: Vec<Tok> = lex("a += m[0] * 1.5f; p++; x <<= 2; 0x10u 2e-3 \"hi\"")
			.unwrap()
			.into_iter()
			.map(|t| t.0)
			.collect();
		assert!(
			t.contains(&Tok::P("+=")) && t.contains(&Tok::P("++")) && t.contains(&Tok::P("<<="))
		);
		assert!(
			t.contains(&Tok::Float(1.5))
				&& t.contains(&Tok::Int(16))
				&& t.contains(&Tok::Float(2e-3))
				&& t.contains(&Tok::Str("hi".into()))
		);
	}

	#[test]
	fn preprocessor() {
		let src = "#ifndef N\n#define N 8\n#endif\n#define H (N / 2)\n#if H > 2\nint a;\n#else\nint b;\n#endif\n#pragma unroll\n";
		let (text, c) = preprocess(src, &[]).unwrap();
		assert_eq!(c, [("N".to_string(), 8), ("H".to_string(), 4)]);
		assert_eq!(text, "\n\n\n\n\nint a;\n\n\n\n__unroll\n");
		let (text, c) = preprocess(src, &[("N".into(), 4)]).unwrap();
		assert_eq!((c[0].1, c[1].1), (4, 2));
		assert!(text.contains("int b;") && !text.contains("int a;"));
		assert!(preprocess("#if 1\n", &[]).is_err());
		assert!(preprocess("#if 0\n#error no\n#endif\n#error yes", &[])
			.unwrap_err()
			.contains("line 4: #error yes"));
	}

	#[test]
	fn statements() {
		let f = parse(
			"#define N 8\nconst u32 H = N / 2;\nvoid controlled_act(const u32 *params) {\n const half8 *restrict x = (const half8 *)params[0];\n register half8 m[H];\n \
             for (int u = 0; u < H; ++u) m[u] = x[u];\n u32 n = 3;\n do { n--; x += 2; } while (n != 0);\n if (n) *x = m[0] + m[1]; else { n = 1; }\n}",
			&[],
		)
		.unwrap();
		assert_eq!(
			(
				f.defines.len(),
				f.globals.len(),
				f.body.len(),
				f.param.1.as_str()
			),
			(1, 1, 6, "params")
		);
		assert!(parse("void k(u32 *p) { a + 1; }", &[])
			.unwrap_err()
			.contains("line 1"));
		assert!(parse("void k(u32 *p) { 3 = a; }", &[]).is_err());
	}
}
