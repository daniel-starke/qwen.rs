//! A model of the ACT-SHAVE for host tests: the result latencies of
//! `NPU.md` §3.2 and an emulator that runs a [`Program`] cycle by cycle.
//!
//! The emulator is as strict about time as the processor: an operation reads
//! its operands in the cycle it is issued, and its result is in the
//! destination register `latency` cycles later, not before. A consumer
//! issued too early reads the old value, exactly the bug a wrong schedule
//! has. A taken branch takes effect after its six delay slots.
//!
//! It knows the instructions the kernel compiler emits; its timing model is
//! the measured one of `NPU.md` §3 (`scripts/shavedev.py`). Transcendentals
//! are exact here and
//! within 6e-4 on the device, so results are compared with a tolerance.
//!
//! [`Stats`] also counts what the processor may not tolerate and nothing
//! measured rules out: an SAU operation beside a VAU one, and two results
//! of one unit for the same register file (`i` or `v`) arriving in the same
//! cycle. The second is fatal on the device: an `LSU0.CP.32` whose result
//! arrives with an `LSU0.LDO.32`'s is lost, while a
//! copy to an `i` register beside a load to a `v` register is what compiled
//! code does. The kernel compiler avoids both.

use super::isa::Operand;
use super::prog::{Line, Op, Program};
use std::collections::HashMap;

/// Where `BRU.JMP i30` returns to when the kernel is done.
pub const RETURN: u32 = 0x1c00_0000;
pub const UNITS: [&str; 8] = ["VAU", "CMU", "LSU0", "LSU1", "SAU", "IAU", "BRU", "PEU"];

/// Cycles from issue until the result can be read (`NPU.md` §3.2); 0 for
/// an operation without a register result. Mnemonics in upper case.
pub fn latency(mnem: &str) -> Option<u32> {
	let (unit, name) = mnem.split_once('.')?;
	let has = |p: &str| name == p || name.starts_with(&format!("{p}."));
	Some(match unit {
		"IAU" => match name {
			"MUL" | "LMULL" => 2,
			"LMULH" => 3,
			_ => 1,
		},
		"SAU" => match name {
			"RCP" | "RQT" | "SQT" | "SUMX.F32" => 5,
			"MUL.I32" => 3,
			"AND" | "OR" | "XOR" => 1,
			_ => 2,
		},
		"CMU" => {
			if has("MAX")
				|| has("MIN")
				|| matches!(
					name,
					"CP.128.I32.F32" | "CP.128.F32.I32S" | "CP.64.F16.U8FS"
				) {
				2
			} else if name.starts_with("CM") {
				0
			} else {
				1
			}
		}
		"LSU0" | "LSU1" => {
			if has("ST") || has("STO") || name.starts_with("SWZ") {
				0
			} else if name == "LD.128" {
				8
			} else if has("LD") || has("LDO") {
				7
			} else {
				1
			}
		}
		"VAU" => match name {
			"EXP" | "SIGM" | "TANH" | "LOG" | "SQT" => 5,
			"AND" | "OR" | "XOR" | "NOT" => 1,
			"ADD.I32" | "SUB.I32" => 2,
			_ => 3,
		},
		"BRU" | "PEU" => 0,
		_ => return None,
	})
}

/// A 128-bit load or store moves two 64-bit halves in two cycles: its LSU
/// can take nothing in the next cycle, and a load's low half arrives one
/// cycle before its high half (measured on the device: `LSU0.LD.128`
/// followed by any load or store on `LSU0` corrupts both; on `LSU1`, or in
/// the same bundle on the other LSU, both are right).
pub fn two_beats(mnem: &str) -> bool {
	matches!(mnem.split_once('.').map(|x| x.1), Some("LD.128" | "ST.128"))
}

/// Instructions other than the compares that rewrite the condition flags
/// (measured on the device: of 27 instructions tried between a compare and
/// its `PEU.PC1C`, only `CMU.MAX` changed the outcome).
pub fn sets_flags(mnem: &str) -> bool {
	mnem.starts_with("CMU.MAX") || mnem.starts_with("CMU.MIN")
}

pub fn h2d(h: u16) -> f64 {
	let (s, e, m) = (h >> 15, (h >> 10) & 31, (h & 1023) as f64);
	let v = match e {
		0 => m * 2f64.powi(-24),
		31 => {
			if m == 0.0 {
				f64::INFINITY
			} else {
				f64::NAN
			}
		}
		_ => (1024.0 + m) * 2f64.powi(e as i32 - 25),
	};
	if s == 1 {
		-v
	} else {
		v
	}
}

/// Round to nearest even.
pub fn d2h(x: f64) -> u16 {
	let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
	if x.is_nan() {
		return 0x7e00;
	}
	let a = x.abs();
	if a >= 65520.0 {
		return sign | 0x7c00;
	}
	if a < 2f64.powi(-14) {
		return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
	}
	let mut e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
	let mut r = (a * 2f64.powi(10 - e)).round_ties_even() as u32;
	if r == 2048 {
		r = 1024;
		e += 1;
	}
	sign | (((e + 15) as u16) << 10) | (r - 1024) as u16
}

/// A register write on its way.
#[derive(Clone, Debug)]
struct Write {
	land: u64,
	vector: bool,
	n: usize,
	at: usize,
	bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
	pub cycles: u64,
	pub bundles: u64,
	/// per unit ([`UNITS`]): results that arrived with another one for the same
	/// register file, and operations on an LSU busy with a 128-bit access
	pub landings: [u64; 8],
	/// bundles with an SAU operation beside a VAU one
	pub sau_vau: u64,
	/// conditions read from flags a CMU.MAX / MIN rewrote after the compare
	pub flags: u64,
	/// reads of a register nothing has written
	pub undefined: u64,
	/// instructions issued together that share a resource ([`super::conflicts`])
	pub conflicts: u64,
	/// the first of each kind, for a message
	pub notes: Vec<String>,
}

pub struct Machine {
	pub i: [u32; 32],
	pub v: [[u8; 16]; 32],
	mem: Vec<(u32, Vec<u8>)>,
	/// the last compare's operands; None once something else rewrote the flags
	flags: Option<(i64, i64)>,
	/// last cycle in which a CMU.MAX / MIN on its way may write the flags
	flag_noise: u64,
	pending: Vec<Write>,
	recent: super::conflicts::Recent,
	/// (unit, landing cycle) of results on their way
	ports: Vec<(usize, u64)>,
	/// per LSU: the cycle a 128-bit access keeps it busy
	lsu_busy: [u64; 2],
	idef: u32,
	vdef: u32,
	pub stats: Stats,
}

fn reg(op: &Op, k: usize) -> Result<(bool, usize, Option<usize>), String> {
	match op.args.get(k) {
		Some(Operand::Reg { vector, n, elem }) if *n < 32 => {
			Ok((*vector, *n as usize, elem.map(|e| e as usize)))
		}
		_ => Err(format!("{}: operand {} is not a register", op.mnem, k + 1)),
	}
}

fn imm(op: &Op, k: usize) -> Result<i64, String> {
	match op.args.get(k) {
		Some(Operand::Int(v)) => Ok(*v),
		_ => Err(format!("{}: operand {} is not a number", op.mnem, k + 1)),
	}
}

fn f16s(v: &[u8; 16]) -> [f64; 8] {
	std::array::from_fn(|k| h2d(u16::from_le_bytes([v[2 * k], v[2 * k + 1]])))
}

fn f32s(v: &[u8; 16]) -> [f32; 4] {
	std::array::from_fn(|k| f32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
}

fn i32s(v: &[u8; 16]) -> [i32; 4] {
	std::array::from_fn(|k| i32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
}

fn pack16(x: [f64; 8]) -> Vec<u8> {
	x.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect()
}

fn pack32(x: [f32; 4]) -> Vec<u8> {
	x.iter().flat_map(|e| e.to_le_bytes()).collect()
}

/// The SHAVE flushes f16 subnormal inputs of a multiply to zero.
fn flush(x: f64) -> f64 {
	if x.abs() < 2f64.powi(-14) {
		0.0 * x
	} else {
		x
	}
}

impl Default for Machine {
	fn default() -> Self {
		Machine::new()
	}
}

impl Machine {
	pub fn new() -> Machine {
		let mut m = Machine {
			i: [0; 32],
			v: [[0; 16]; 32],
			mem: Vec::new(),
			flags: Some((0, 0)),
			flag_noise: 0,
			pending: Vec::new(),
			recent: Default::default(),
			ports: Vec::new(),
			lsu_busy: [0; 2],
			idef: 1 << 18 | 1 << 19 | 1 << 30,
			vdef: 0,
			stats: Stats::default(),
		};
		m.i[30] = RETURN;
		m
	}

	/// Map `bytes` at `addr`.
	pub fn map(&mut self, addr: u32, bytes: Vec<u8>) {
		self.mem.push((addr, bytes));
	}

	pub fn read(&self, addr: u32, len: usize) -> Result<&[u8], String> {
		self.mem
			.iter()
			.find(|(a, b)| addr >= *a && (addr - a) as usize + len <= b.len())
			.map(|(a, b)| &b[(addr - a) as usize..(addr - a) as usize + len])
			.ok_or_else(|| format!("read of {len} bytes at {addr:#x}: not mapped"))
	}

	fn write(&mut self, addr: u32, bytes: &[u8]) -> Result<(), String> {
		let (a, b) = self
			.mem
			.iter_mut()
			.find(|(a, b)| addr >= *a && (addr - a) as usize + bytes.len() <= b.len())
			.ok_or_else(|| format!("write of {} bytes at {addr:#x}: not mapped", bytes.len()))?;
		let at = (addr - *a) as usize;
		b[at..at + bytes.len()].copy_from_slice(bytes);
		Ok(())
	}

	fn note(&mut self, text: String) {
		if self.stats.notes.len() < 8 {
			self.stats.notes.push(text);
		}
	}

	fn ri(&mut self, op: &Op, k: usize) -> Result<u32, String> {
		let (vector, n, _) = reg(op, k)?;
		if vector {
			return Err(format!(
				"{}: operand {} must be an i register",
				op.mnem,
				k + 1
			));
		}
		if self.idef >> n & 1 == 0 {
			self.stats.undefined += 1;
			self.note(format!(
				"cycle {}: {} reads i{n}, which nothing wrote",
				self.stats.cycles, op.mnem
			));
		}
		Ok(self.i[n])
	}

	fn rv(&mut self, op: &Op, k: usize) -> Result<[u8; 16], String> {
		let (vector, n, _) = reg(op, k)?;
		if !vector {
			return Err(format!(
				"{}: operand {} must be a v register",
				op.mnem,
				k + 1
			));
		}
		if self.vdef >> n & 1 == 0 {
			self.stats.undefined += 1;
			self.note(format!(
				"cycle {}: {} reads v{n}, which nothing wrote",
				self.stats.cycles, op.mnem
			));
		}
		Ok(self.v[n])
	}

	/// Register or immediate.
	fn rim(&mut self, op: &Op, k: usize) -> Result<u32, String> {
		match op.args.get(k) {
			Some(Operand::Int(v)) => Ok(*v as u32),
			_ => self.ri(op, k),
		}
	}

	/// Queue the result of `op` (unit `u`): `bytes` at byte `at` of its
	/// destination, operand `k`.
	fn put(
		&mut self,
		op: &Op,
		u: usize,
		k: usize,
		at: usize,
		bytes: Vec<u8>,
	) -> Result<(), String> {
		let lat = latency(&op.mnem).ok_or_else(|| format!("{}: no latency", op.mnem))?;
		self.put_at(op, u, k, at, bytes, lat)
	}

	fn put_at(
		&mut self,
		op: &Op,
		u: usize,
		k: usize,
		at: usize,
		bytes: Vec<u8>,
		lat: u32,
	) -> Result<(), String> {
		let (vector, n, _) = reg(op, k)?;
		let land = self.stats.cycles + lat as u64;
		// one result per unit, register file and cycle
		let port = u + if vector { 8 } else { 0 };
		if self.ports.contains(&(port, land)) {
			self.stats.landings[u] += 1;
			self.note(format!(
				"cycle {}: {} is the second {} result for {} register of cycle {land}",
				self.stats.cycles,
				op.mnem,
				UNITS[u],
				if vector { "a v" } else { "an i" }
			));
		}
		self.ports.push((port, land));
		self.pending.push(Write {
			land,
			vector,
			n,
			at,
			bytes,
		});
		Ok(())
	}

	fn land(&mut self) {
		let now = self.stats.cycles;
		// whole-register writes first: LDIL || LDIH, the halves of a vector
		let mut due: Vec<Write> = Vec::new();
		self.pending.retain(|w| {
			if w.land <= now {
				due.push(w.clone());
			}
			w.land > now
		});
		due.sort_by_key(|w| (w.land, std::cmp::Reverse(w.bytes.len())));
		for w in due {
			if w.vector {
				self.v[w.n][w.at..w.at + w.bytes.len()].copy_from_slice(&w.bytes);
				self.vdef |= 1 << w.n;
			} else {
				let mut b = self.i[w.n].to_le_bytes();
				b[w.at..w.at + w.bytes.len()].copy_from_slice(&w.bytes);
				self.i[w.n] = u32::from_le_bytes(b);
				self.idef |= 1 << w.n;
			}
		}
		self.ports.retain(|p| p.1 > now);
	}

	fn cond(&mut self, name: &str) -> Result<bool, String> {
		let Some((a, b)) = self.flags else {
			self.stats.flags += 1;
			self.note(format!(
				"cycle {}: a condition is read after CMU.MAX / MIN rewrote the flags",
				self.stats.cycles
			));
			return Ok(false);
		};
		Ok(match name {
			"EQ" => a == b,
			"NEQ" => a != b,
			"LT" => a < b,
			"LTE" => a <= b,
			"GT" => a > b,
			"GTE" => a >= b,
			c => return Err(format!("condition {c} is not modeled")),
		})
	}

	/// Run from `entry` (a label, or the first line) until the kernel
	/// returns. `i18` (the argument) and the stack pointer `i19` are the
	/// caller's to set.
	pub fn run(
		&mut self,
		prog: &Program,
		entry: Option<&str>,
		max_cycles: u64,
	) -> Result<(), String> {
		let labels: HashMap<&str, usize> = prog
			.lines
			.iter()
			.enumerate()
			.filter_map(|(k, l)| {
				if let Line::Label(n) = l {
					Some((n.as_str(), k))
				} else {
					None
				}
			})
			.collect();
		let mut pc = match entry {
			Some(e) => *labels.get(e).ok_or_else(|| format!("no label {e}"))?,
			None => 0,
		};
		// taken branches: (cycles left, target line; None = return)
		let mut jumps: Vec<(u32, Option<usize>)> = Vec::new();
		let mut nops = 0u32;
		loop {
			if self.stats.cycles >= max_cycles {
				return Err(format!("no return after {max_cycles} cycles"));
			}
			self.land();
			if nops == 0 {
				while let Some(Line::Label(_)) = prog.lines.get(pc) {
					pc += 1;
				}
				match prog.lines.get(pc) {
					None => return Err("ran off the end of the code".into()),
					Some(Line::Nop(n)) => nops = *n,
					Some(Line::Bundle(ops)) => {
						self.stats.bundles += 1;
						if let Some(target) = self.bundle(ops, &labels)? {
							jumps.push((7, target));
						}
					}
					Some(Line::Label(_)) => unreachable!(),
				}
			}
			nops = nops.saturating_sub(1);
			if nops == 0 {
				pc += 1;
			}
			self.stats.cycles += 1;
			for j in jumps.iter_mut() {
				j.0 -= 1;
			}
			if let Some(k) = jumps.iter().position(|j| j.0 == 0) {
				match jumps.remove(k).1 {
					None => {
						// results still on their way arrive
						while !self.pending.is_empty() {
							self.stats.cycles += 1;
							self.land();
						}
						return Ok(());
					}
					Some(t) => {
						pc = t;
						nops = 0;
					}
				}
			}
		}
	}

	/// One cycle's operations; the target of a taken branch.
	fn bundle(
		&mut self,
		ops: &[Op],
		labels: &HashMap<&str, usize>,
	) -> Result<Option<Option<usize>>, String> {
		let unit = |op: &Op| {
			UNITS
				.iter()
				.position(|u| *u == op.unit())
				.ok_or_else(|| format!("{}: unknown unit", op.mnem))
		};
		let has = |u: &str| ops.iter().any(|o| o.unit() == u);
		if has("SAU") && has("VAU") {
			self.stats.sau_vau += 1;
			self.note(format!("cycle {}: SAU beside VAU", self.stats.cycles));
		}
		for (first, second, res) in self.recent.issue(self.stats.cycles, ops) {
			self.stats.conflicts += 1;
			self.note(format!(
				"cycle {}: {second} shares {res} with {first}",
				self.stats.cycles
			));
		}
		// predication: PC1C gates the bundle, PCCX the units of its mask
		// (bit 0 IAU, 1 SAU, 2 CMU, 3 VAU, 4 LSU0, 5 LSU1) and the branch
		let mut gated = [false; 8];
		if let Some(p) = ops.iter().find(|o| o.unit() == "PEU") {
			let name = match p.args.first() {
				Some(Operand::Ident(c)) => c.to_ascii_uppercase(),
				_ => return Err(format!("{}: no condition", p.mnem)),
			};
			if !self.cond(&name)? {
				match p.mnem.as_str() {
					"PEU.PC1C" => gated = [true; 8],
					"PEU.PCCX" => {
						let mask = imm(p, 1)?;
						for (bit, u) in ["IAU", "SAU", "CMU", "VAU", "LSU0", "LSU1"]
							.iter()
							.enumerate()
						{
							gated[UNITS.iter().position(|x| x == u).unwrap()] =
								mask >> bit & 1 != 0;
						}
						gated[6] = true;
					}
					m => return Err(format!("{m} is not modeled")),
				}
			}
		}
		// operand swizzle of the CMU operation's first source, 16-bit elements
		let swz: Option<Vec<usize>> = match ops
			.iter()
			.find(|o| o.mnem == "LSU0.SWZC8" || o.mnem == "LSU1.SWZC8")
		{
			Some(Op { args, .. }) => match args.first() {
				Some(Operand::Pattern(p)) if p.len() == 8 => {
					Some(p.bytes().rev().map(|c| (c - b'0') as usize).collect())
				}
				_ => return Err("SWZC8 needs an 8-entry pattern".into()),
			},
			None => None,
		};
		let mut flags = None;
		let mut jump = None;
		for op in ops {
			let u = unit(op)?;
			if (u == 2 || u == 3)
				&& self.lsu_busy[u - 2] == self.stats.cycles
				&& self.stats.cycles > 0
			{
				self.stats.landings[u] += 1;
				self.note(format!(
					"cycle {}: {} while {} moves the second half of a 128-bit access",
					self.stats.cycles, op.mnem, UNITS[u]
				));
			}
			if two_beats(&op.mnem) && !gated[u] {
				self.lsu_busy[u - 2] = self.stats.cycles + 1;
			}
			if gated[u] || u == 7 {
				continue;
			}
			let name = op.mnem.split_once('.').map_or("", |x| x.1);
			let lsu = u == 2 || u == 3;
			match (UNITS[u], name) {
				("IAU", "ADD") | ("SAU", "ADD.I32") => {
					let r = self.ri(op, 1)?.wrapping_add(self.rim(op, 2)?);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU", "SUB") => {
					let r = self.ri(op, 1)?.wrapping_sub(self.rim(op, 2)?);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU" | "SAU", "AND" | "OR" | "XOR") => {
					let (a, b) = (self.ri(op, 1)?, self.ri(op, 2)?);
					let r = match name {
						"AND" => a & b,
						"OR" => a | b,
						_ => a ^ b,
					};
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU", "SHL") | ("SAU", "SHL.X32") => {
					let r = self.ri(op, 1)? << (imm(op, 2)? & 31);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU" | "SAU", "SHR.U32") => {
					let r = self.ri(op, 1)? >> (imm(op, 2)? & 31);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU", "INCS") => {
					let r = self.ri(op, 0)?.wrapping_add(imm(op, 1)? as u32);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("IAU", "MUL" | "LMULL") | ("SAU", "MUL.I32") => {
					let r = self.ri(op, 1)?.wrapping_mul(self.ri(op, 2)?);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("SAU", "RCP" | "RQT" | "SQT") => {
					let e = reg(op, 1)?.2.unwrap_or(0);
					let x = h2d((self.ri(op, 1)? >> (16 * e)) as u16);
					let r = match name {
						"RCP" => 1.0 / x,
						"RQT" => 1.0 / x.sqrt(),
						_ => x.sqrt(),
					};
					self.put(op, u, 0, 0, (d2h(r) as u32).to_le_bytes().to_vec())?
				}
				("SAU", "SUMX.F32") => {
					let x = f32s(&self.rv(op, 1)?);
					let r = (x[0] + x[1]) + (x[2] + x[3]);
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				("CMU", "CMZ.I32") => flags = Some((self.ri(op, 0)? as i32 as i64, 0)),
				("CMU", "CMII.I32") => {
					flags = Some((self.ri(op, 0)? as i32 as i64, self.ri(op, 1)? as i32 as i64))
				}
				("CMU", "CMII.U32") => {
					flags = Some((self.ri(op, 0)? as i64, self.ri(op, 1)? as i64))
				}
				("CMU", "CMPI") => flags = Some((self.ri(op, 0)? as i32 as i64, imm(op, 1)?)),
				("CMU" | "LSU0" | "LSU1", "CP.32") => match (reg(op, 0)?, reg(op, 1)?) {
					((false, _, None), (false, _, None)) => {
						let r = self.ri(op, 1)?;
						self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
					}
					((false, _, None), (true, _, Some(e))) => {
						let r = self.rv(op, 1)?[4 * e..4 * e + 4].to_vec();
						self.put(op, u, 0, 0, r)?
					}
					((true, _, Some(e)), (false, _, None)) => {
						let r = self.ri(op, 1)?;
						self.put(op, u, 0, 4 * e, r.to_le_bytes().to_vec())?
					}
					_ => return Err(format!("{}: this operand form is not modeled", op.mnem)),
				},
				("CMU" | "LSU0" | "LSU1", "CP.128") => {
					let r = self.rv(op, 1)?;
					self.put(op, u, 0, 0, r.to_vec())?
				}
				("CMU", "CP.128.32.R") => {
					let r = self.ri(op, 1)?.to_le_bytes();
					self.put(op, u, 0, 0, r.repeat(4))?
				}
				("CMU", "CP.128.16.R") => {
					let e = reg(op, 1)?.2.unwrap_or(0);
					let r = ((self.ri(op, 1)? >> (16 * e)) as u16).to_le_bytes();
					self.put(op, u, 0, 0, r.repeat(8))?
				}
				("CMU", "CP.128.F16.F32") => {
					let e = reg(op, 1)?.2.unwrap_or(0);
					let x = f16s(&self.rv(op, 1)?);
					self.put(
						op,
						u,
						0,
						0,
						pack32(std::array::from_fn(|k| flush(x[4 * e + k]) as f32)),
					)?
				}
				("CMU", "CP.64.F16.U8FS") => {
					// 8 x f16 as u8 fractions (x * 255, to nearest, saturating) in a 64-bit half
					let e = reg(op, 0)?.2.unwrap_or(0);
					let x = f16s(&self.rv(op, 1)?);
					self.put(
						op,
						u,
						0,
						8 * e,
						x.iter()
							.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
							.collect(),
					)?
				}
				("CMU", "CP.128.I32.F32") => {
					let x = i32s(&self.rv(op, 1)?);
					self.put(op, u, 0, 0, pack32(x.map(|e| e as f32)))?
				}
				("CMU", "CP.128.F32.I32S") => {
					let x = f32s(&self.rv(op, 1)?);
					self.put(
						op,
						u,
						0,
						0,
						x.iter().flat_map(|e| (*e as i32).to_le_bytes()).collect(),
					)?
				}
				("CMU", "CP.16") => {
					let (d, s) = (reg(op, 0)?.2.unwrap_or(0), reg(op, 1)?.2.unwrap_or(0));
					let r = self.rv(op, 1)?[2 * s..2 * s + 2].to_vec();
					self.put(op, u, 0, 2 * d, r)?
				}
				("CMU", "CP.16.F32.F16") => {
					let (d, s) = (reg(op, 0)?.2.unwrap_or(0), reg(op, 1)?.2.unwrap_or(0));
					let r = d2h(f32s(&self.rv(op, 1)?)[s] as f64);
					self.put(op, u, 0, 2 * d, r.to_le_bytes().to_vec())?
				}
				("CMU", "MAX.F16" | "MIN.F16") => {
					let (a, b) = (f16s(&self.rv(op, 1)?), f16s(&self.rv(op, 2)?));
					self.put(
						op,
						u,
						0,
						0,
						pack16(std::array::from_fn(|k| {
							if name == "MAX.F16" {
								a[k].max(b[k])
							} else {
								a[k].min(b[k])
							}
						})),
					)?
				}
				("CMU", "MAX.F32" | "MIN.F32") => {
					let (a, b) = (f32s(&self.rv(op, 1)?), f32s(&self.rv(op, 2)?));
					self.put(
						op,
						u,
						0,
						0,
						pack32(std::array::from_fn(|k| {
							if name == "MAX.F32" {
								a[k].max(b[k])
							} else {
								a[k].min(b[k])
							}
						})),
					)?
				}
				("CMU", "ALIGNVEC") => {
					let mut a = self.rv(op, 1)?;
					let b = self.rv(op, 2)?;
					if let Some(p) = &swz {
						let s = a;
						for (k, &from) in p.iter().enumerate() {
							a[2 * k..2 * k + 2].copy_from_slice(&s[2 * from..2 * from + 2]);
						}
					}
					let n = imm(op, 3)? as usize & 15;
					let r: Vec<u8> = a[n..].iter().chain(&b[..n]).copied().collect();
					self.put(op, u, 0, 0, r)?
				}
				(_, "SWZC8") if lsu => {}
				(_, "LDIL") if lsu => {
					let r = (imm(op, 1)? as u32) & 0xffff;
					self.put(op, u, 0, 0, r.to_le_bytes().to_vec())?
				}
				(_, "LDIH") if lsu => {
					let v = imm(op, 1)?;
					let r = if (0..=0xffff).contains(&v) {
						v as u16
					} else {
						((v as u32) >> 16) as u16
					};
					self.put(op, u, 0, 2, r.to_le_bytes().to_vec())?
				}
				(
					_,
					"LD.32" | "LDO.32" | "LD.64" | "LDO.64" | "LD.128" | "LD.128.16.R"
					| "LD.128.F16.F32",
				) if lsu => {
					let off = if name.starts_with("LDO") {
						imm(op, 2)? as u32
					} else {
						0
					};
					let addr = self.ri(op, 1)?.wrapping_add(off);
					let (_, _, e) = reg(op, 0)?;
					match name {
						"LD.32" | "LDO.32" => {
							let b = self.read(addr, 4)?.to_vec();
							self.put(op, u, 0, 4 * e.unwrap_or(0), b)?
						}
						"LD.64" | "LDO.64" => {
							let b = self.read(addr, 8)?.to_vec();
							self.put(op, u, 0, 8 * e.ok_or("a 64-bit load needs v.0 or v.1")?, b)?
						}
						"LD.128" => {
							let b = self.read(addr, 16)?.to_vec();
							self.put_at(op, u, 0, 0, b[..8].to_vec(), 7)?;
							self.put_at(op, u, 0, 8, b[8..].to_vec(), 8)?
						}
						"LD.128.16.R" => {
							let b = self.read(addr, 2)?.repeat(8);
							self.put(op, u, 0, 0, b)?
						}
						_ => {
							let b = self.read(addr, 8)?.to_vec();
							let x: [f32; 4] = std::array::from_fn(|k| {
								flush(h2d(u16::from_le_bytes([b[2 * k], b[2 * k + 1]]))) as f32
							});
							self.put(op, u, 0, 0, pack32(x))?
						}
					}
				}
				(_, "ST.16" | "STO.16" | "ST.32" | "STO.32" | "ST.64" | "STO.64" | "ST.128")
					if lsu =>
				{
					let off = if name.starts_with("STO") {
						imm(op, 2)? as u32
					} else {
						0
					};
					let addr = self.ri(op, 1)?.wrapping_add(off);
					let (vector, _, e) = reg(op, 0)?;
					let bytes: Vec<u8> = match (name, vector) {
						("ST.16" | "STO.16", false) => {
							let e = e.unwrap_or(0);
							self.ri(op, 0)?.to_le_bytes()[2 * e..2 * e + 2].to_vec()
						}
						("ST.32" | "STO.32", false) => self.ri(op, 0)?.to_le_bytes().to_vec(),
						("ST.32" | "STO.32", true) => {
							let e = e.unwrap_or(0);
							self.rv(op, 0)?[4 * e..4 * e + 4].to_vec()
						}
						("ST.64" | "STO.64", true) => {
							let e = e.ok_or("a 64-bit store needs v.0 or v.1")?;
							self.rv(op, 0)?[8 * e..8 * e + 8].to_vec()
						}
						("ST.128", true) => self.rv(op, 0)?.to_vec(),
						_ => return Err(format!("{}: this operand form is not modeled", op.mnem)),
					};
					self.write(addr, &bytes)?
				}
				("VAU", "ADD.F16" | "SUB.F16" | "MUL.F16") => {
					let (a, b) = (f16s(&self.rv(op, 1)?), f16s(&self.rv(op, 2)?));
					let r: [f64; 8] = std::array::from_fn(|k| match name {
						"ADD.F16" => a[k] + b[k],
						"SUB.F16" => a[k] - b[k],
						_ => flush(a[k]) * flush(b[k]),
					});
					self.put(op, u, 0, 0, pack16(r))?
				}
				("VAU", "ADD.F32" | "SUB.F32" | "MUL.F32") => {
					let a = f32s(&self.rv(op, 1)?);
					let b = match op.args.get(2) {
						Some(Operand::Float(f)) => {
							[f.parse::<f32>().map_err(|_| format!("bad constant {f}"))?; 4]
						}
						_ => f32s(&self.rv(op, 2)?),
					};
					let r: [f32; 4] = std::array::from_fn(|k| match name {
						"ADD.F32" => a[k] + b[k],
						"SUB.F32" => a[k] - b[k],
						_ => a[k] * b[k],
					});
					self.put(op, u, 0, 0, pack32(r))?
				}
				("VAU", "ADD.I32" | "SUB.I32") => {
					let (a, b) = (i32s(&self.rv(op, 1)?), i32s(&self.rv(op, 2)?));
					let r: Vec<u8> = (0..4)
						.flat_map(|k| {
							if name == "ADD.I32" {
								a[k].wrapping_add(b[k])
							} else {
								a[k].wrapping_sub(b[k])
							}
							.to_le_bytes()
						})
						.collect();
					self.put(op, u, 0, 0, r)?
				}
				("VAU", "SHR.I32") => {
					let (a, n) = (i32s(&self.rv(op, 1)?), imm(op, 2)? & 31);
					self.put(
						op,
						u,
						0,
						0,
						a.iter().flat_map(|e| (e >> n).to_le_bytes()).collect(),
					)?
				}
				("VAU", "AND" | "OR" | "XOR") => {
					let (a, b) = (self.rv(op, 1)?, self.rv(op, 2)?);
					let r: Vec<u8> = (0..16)
						.map(|k| match name {
							"AND" => a[k] & b[k],
							"OR" => a[k] | b[k],
							_ => a[k] ^ b[k],
						})
						.collect();
					self.put(op, u, 0, 0, r)?
				}
				("VAU", "EXP" | "SIGM" | "TANH" | "LOG" | "SQT") => {
					let a = f16s(&self.rv(op, 1)?);
					let r = a.map(|x| match name {
						"EXP" => x.exp(),
						"SIGM" => 1.0 / (1.0 + (-x).exp()),
						"TANH" => x.tanh(),
						"LOG" => x.ln(),
						_ => x.sqrt(),
					});
					self.put(op, u, 0, 0, pack16(r))?
				}
				("BRU", "BRA") => {
					let target = match op.args.first() {
						Some(Operand::Ident(l)) => *labels
							.get(l.as_str())
							.ok_or_else(|| format!("no label {l}"))?,
						_ => return Err("BRU.BRA needs a label".into()),
					};
					jump = Some(Some(target));
				}
				("BRU", "JMP") => {
					let to = self.ri(op, 0)?;
					if to != RETURN {
						return Err(format!("BRU.JMP to {to:#x}: only the return is modeled"));
					}
					jump = Some(None);
				}
				_ => return Err(format!("{} is not modeled", op.mnem)),
			}
		}
		if ops.iter().any(|o| sets_flags(&o.mnem) && !gated[1]) {
			self.flags = None;
			self.flag_noise = self.stats.cycles + 2;
		}
		if let Some(f) = flags {
			// compiled code compares two cycles behind a MAX
			self.flags = (self.stats.cycles >= self.flag_noise).then_some(f);
		}
		Ok(jump)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn half_floats() {
		for h in [
			0u16, 1, 0x3ff, 0x400, 0x3c00, 0x3c01, 0x7bff, 0x8001, 0xbc00, 0xfbff,
		] {
			assert_eq!(d2h(h2d(h)), h, "{h:#x}");
		}
		assert_eq!(d2h(1.0 + 2f64.powi(-11)), 0x3c00); // a tie rounds to even
		assert_eq!(d2h(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
		assert_eq!(d2h(65520.0), 0x7c00);
		assert_eq!(h2d(0xfbff), -65504.0);
	}

	fn run(src: &str) -> Machine {
		let mut m = Machine::new();
		m.map(0x1000, (0..64).collect());
		m.i[18] = 0x1000;
		m.run(&Program::parse(src).unwrap(), None, 1000).unwrap();
		m
	}

	#[test]
	fn a_result_arrives_after_its_latency() {
		// the add reads i1 one cycle early (old value 0), the second in time
		let m = run(
			"LSU0.LDO.32 i1 i18 4\nNOP 5\nIAU.ADD i2 i1 1\nIAU.ADD i3 i1 1\nBRU.JMP i30\nNOP 6",
		);
		assert_eq!((m.i[2], m.i[3]), (1, 0x07060505));
		assert_eq!(m.stats.undefined, 1);
	}

	#[test]
	fn delay_slots_and_predication() {
		let m = run("LSU0.LDIL i1 3\nLSU0.LDIL i2 0\n.Ltop:\nIAU.SUB i1 i1 1\nCMU.CMZ.I32 i1\nPEU.PC1C NEQ || BRU.BRA .Ltop\nIAU.ADD i2 i2 1\nNOP 5\nBRU.JMP i30\nNOP 6");
		assert_eq!((m.i[1], m.i[2]), (0, 3));
	}

	#[test]
	fn two_results_of_a_unit_in_one_cycle_are_counted() {
		let m = run("CMU.MAX.F16 v1 v0 v0\nCMU.CP.128 v2 v0\nBRU.JMP i30\nNOP 6");
		assert_eq!(m.stats.landings[1], 1);
	}
}
