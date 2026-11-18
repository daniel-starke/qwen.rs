//! A SHAVE program as bundles: what the kernel compiler ([`super::dsl`])
//! emits, what the emulator ([`super::mach`]) runs and what the assembler
//! reads as text.

use super::asm;
use super::isa::Operand;

/// One operation: `VAU.ADD.f16 v1 v2 v3`.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
	pub mnem: String,
	pub args: Vec<Operand>,
}

impl Op {
	pub fn new(mnem: &str, args: Vec<Operand>) -> Op {
		Op {
			mnem: mnem.to_string(),
			args,
		}
	}

	/// `VAU`, `LSU0`, ...: the text before the first dot.
	pub fn unit(&self) -> &str {
		self.mnem.split('.').next().unwrap_or("")
	}
}

#[derive(Clone, Debug, PartialEq)]
pub enum Line {
	Label(String),
	/// `NOP n`: n cycles
	Nop(u32),
	/// the operations of one cycle
	Bundle(Vec<Op>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
	pub lines: Vec<Line>,
}

pub fn vreg(n: u32) -> Operand {
	Operand::Reg {
		vector: true,
		n,
		elem: None,
	}
}

fn show(op: &Operand) -> String {
	match op {
		Operand::Reg { vector, n, elem } => {
			let r = format!("{}{n}", if *vector { 'v' } else { 'i' });
			elem.map_or(r.clone(), |e| format!("{r}.{e}"))
		}
		Operand::Int(v) => v.to_string(),
		Operand::Float(f) => f.clone(),
		Operand::Pattern(p) => format!("[{p}]"),
		Operand::Ident(s) => s.clone(),
	}
}

impl Program {
	/// The bundles of an assembly source; directives are skipped.
	pub fn parse(src: &str) -> Result<Program, String> {
		let text = asm::strip_comments(src);
		let mut joined: Vec<(usize, String)> = Vec::new();
		for (n, l) in text.lines().enumerate() {
			let l = l.trim();
			if l.is_empty() {
				continue;
			}
			match (l.strip_prefix("||"), joined.last_mut()) {
				(Some(more), Some(last)) => {
					last.1.push_str(" || ");
					last.1.push_str(more.trim());
				}
				(Some(_), None) => {
					return Err(format!("line {}: `||` does not follow an operation", n + 1))
				}
				_ => joined.push((n + 1, l.to_string())),
			}
		}
		let mut lines = Vec::new();
		for (n, l) in joined {
			if let Some(name) = l.strip_suffix(':') {
				lines.push(Line::Label(name.trim().to_string()));
				continue;
			}
			if l.starts_with('.') {
				if l == ".end" {
					break;
				}
				continue;
			}
			let mut ops = Vec::new();
			for part in l.split("||").map(str::trim) {
				let mut tok = part
					.split(|c: char| c.is_whitespace() || c == ',')
					.filter(|t| !t.is_empty());
				let mnem = tok
					.next()
					.ok_or_else(|| format!("line {n}: empty operation"))?;
				let args: Vec<Operand> = tok.map(asm::operand).collect();
				if ["NOP", "BRU.NOP", "BRU.INC"]
					.iter()
					.any(|n| mnem.eq_ignore_ascii_case(n))
				{
					let count = match args[..] {
						[] => 1,
						[Operand::Int(c)] if c >= 1 => c as u32,
						_ => return Err(format!("line {n}: NOP takes one count")),
					};
					lines.push(Line::Nop(count));
					continue;
				}
				ops.push(Op {
					mnem: mnem.to_ascii_uppercase(),
					args,
				});
			}
			if !ops.is_empty() {
				lines.push(Line::Bundle(ops));
			}
		}
		Ok(Program { lines })
	}

	/// Assembly source of a kernel whose entry is the first line.
	pub fn text(&self, entry: &str) -> String {
		let mut s = format!(
			".nowarn\n.code .text.entry\n.align 16\n\t.type\t{entry},@function\n{entry}:\n"
		);
		for l in &self.lines {
			match l {
				Line::Label(name) => s.push_str(&format!("{name}:\n")),
				Line::Nop(1) => s.push_str("        NOP\n"),
				Line::Nop(n) => s.push_str(&format!("        NOP {n}\n")),
				Line::Bundle(ops) => {
					for (k, op) in ops.iter().enumerate() {
						s.push_str(if k == 0 {
							"        "
						} else {
							"            || "
						});
						s.push_str(&op.mnem);
						for a in &op.args {
							s.push(' ');
							s.push_str(&show(a));
						}
						s.push('\n');
					}
				}
			}
		}
		s.push_str(&format!(
			".Lend_{entry}:\n\t.size\t{entry}, .Lend_{entry}-{entry}\n.nowarnend\n.end\n"
		));
		s
	}

	/// Cycles of straight-line code (every bundle and NOP once).
	pub fn cycles(&self) -> u32 {
		self.lines
			.iter()
			.map(|l| match l {
				Line::Label(_) => 0,
				Line::Nop(n) => *n,
				Line::Bundle(_) => 1,
			})
			.sum()
	}
}
