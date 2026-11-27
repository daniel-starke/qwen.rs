//! Instructions that cannot be issued together: the processor's units share
//! resources their encoding does not show (read and write ports of the
//! register files, the flags, a datapath). `conflicts.tbl` lists the pairs
//! of instruction forms that share one, the second issued 0 to [`WINDOW`]
//! cycles after the first (functional facts, `NPU.md` "Provenance"). One of
//! them was first met on the device: `CMU.CMII` reads its second operand
//! through LSU1's read port, and beside an `LSU1` load one of the two reads
//! garbage. The kernel compiler's scheduler keeps such pairs apart
//! ([`super::dsl::sched`]), the emulator counts them ([`super::mach`]) and
//! [`check`] finds them in a program.

use super::isa::Operand;
use super::prog::{Line, Op, Program};
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

/// Cycles over which two instructions can conflict.
pub const WINDOW: u32 = 8;

pub struct Conflicts {
	/// (form of the first, form of the second) -> (cycles apart, resource)
	pairs: HashMap<(String, String), Vec<(u32, String)>>,
}

/// An instruction's form as the table names it, in lower case: the mnemonic
/// and its operands' kinds (`cmu.cp.128.16.r v i.e`).
pub fn form(op: &Op) -> String {
	let mut f = op.mnem.to_ascii_lowercase();
	for a in &op.args {
		f += match a {
			Operand::Reg {
				vector: false,
				elem: None,
				..
			} => " i",
			Operand::Reg { vector: false, .. } => " i.e",
			Operand::Reg {
				vector: true,
				elem: None,
				..
			} => " v",
			Operand::Reg { vector: true, .. } => " v.e",
			Operand::Int(_) => " #",
			_ => " ?",
		};
	}
	f
}

pub fn table() -> &'static Conflicts {
	static TABLE: OnceLock<Conflicts> = OnceLock::new();
	TABLE.get_or_init(|| {
		let mut pairs: HashMap<(String, String), Vec<(u32, String)>> = HashMap::new();
		for line in quant::packed_str!("conflicts.tbl")
			.lines()
			.filter(|l| !l.starts_with('#') && !l.trim().is_empty())
		{
			let f: Vec<&str> = line.split('|').map(str::trim).collect();
			let list = f[2]
				.split_whitespace()
				.map(|t| t.split_once(':').expect("conflicts.tbl"))
				.map(|(d, r)| (d.parse().expect("conflicts.tbl"), r.to_string()));
			pairs
				.entry((f[0].to_ascii_lowercase(), f[1].to_ascii_lowercase()))
				.or_default()
				.extend(list);
		}
		Conflicts { pairs }
	})
}

impl Conflicts {
	/// The resource two instructions collide on when `second` is issued
	/// `apart` cycles after `first` (0: in one bundle), by their [`form`]s.
	pub fn hit(&self, first: &str, second: &str, apart: u32) -> Option<&str> {
		let find = |a: &str, b: &str| {
			self.pairs
				.get(&(a.to_string(), b.to_string()))
				.and_then(|l| l.iter().find(|c| c.0 == apart))
				.map(|c| c.1.as_str())
		};
		find(first, second).or_else(|| {
			if apart == 0 {
				find(second, first)
			} else {
				None
			}
		})
	}
}

/// The instructions of the last cycles, to hold a new bundle against.
#[derive(Default)]
pub struct Recent {
	cycles: VecDeque<(u64, Vec<String>)>,
}

impl Recent {
	/// The conflicts of a bundle issued in `cycle` with itself and with what
	/// was issued before: (earlier instruction, later instruction, resource).
	pub fn issue(&mut self, cycle: u64, ops: &[Op]) -> Vec<(String, String, String)> {
		let (t, forms) = (table(), ops.iter().map(form).collect::<Vec<_>>());
		while self
			.cycles
			.front()
			.is_some_and(|c| c.0 + (WINDOW as u64) < cycle)
		{
			self.cycles.pop_front();
		}
		let mut found = Vec::new();
		for (k, f) in forms.iter().enumerate() {
			for (at, earlier) in &self.cycles {
				found.extend(earlier.iter().filter_map(|e| {
					t.hit(e, f, (cycle - at) as u32)
						.map(|r| (e.clone(), f.clone(), r.to_string()))
				}));
			}
			found.extend(forms[..k].iter().filter_map(|e| {
				t.hit(e, f, 0)
					.map(|r| (e.clone(), f.clone(), r.to_string()))
			}));
		}
		self.cycles.push_back((cycle, forms));
		found
	}
}

/// Every conflict of a program, its cycles counted in the order of the text.
pub fn check(prog: &Program) -> Vec<String> {
	let (mut recent, mut cycle, mut out) = (Recent::default(), 0u64, Vec::new());
	for line in &prog.lines {
		match line {
			Line::Label(_) => {}
			Line::Nop(n) => cycle += *n as u64,
			Line::Bundle(ops) => {
				out.extend(
					recent
						.issue(cycle, ops)
						.into_iter()
						.map(|(a, b, r)| format!("cycle {cycle}: {b} shares {r} with {a}")),
				);
				cycle += 1;
			}
		}
	}
	out
}
