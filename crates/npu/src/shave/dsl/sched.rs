//! Bundles and machine registers for the lowered kernel.
//!
//! **Schedule.** Every straight-line block is scheduled on its own, cycle by
//! cycle, twice: most urgent operation first (longest chain of latencies
//! behind it), and furthest along first (a chain that was begun is carried
//! on before another one is started, so that few intermediate values are
//! alive at once). The shorter schedule counts, of two equally long ones
//! the one with fewer values alive. An operation goes into a cycle when
//!
//! - every value it reads has arrived (producer's cycle + its latency), a
//!   register it overwrites is not read later by an earlier operation, and
//!   two writes of one register arrive in source order;
//! - a unit that can run it is free, no SAU operation shares the bundle
//!   with a VAU one, and no other result of that unit for the same register
//!   file (`i` / `v`) arrives in the same cycle: the processor loses one of
//!   them (`mach`);
//! - memory accesses that may overlap keep their order: a load 8 cycles
//!   after a store, a store after a load's 7;
//! - a 128-bit load or store keeps its LSU for two cycles;
//! - it shares no resource with an operation issued in the same or a nearby
//!   cycle ([`crate::shave::conflicts`]: `CMU.CMII` and an `LSU1` load in
//!   one bundle read through the same port);
//! - a block's compare goes into the cycle before its branch, as in
//!   compiled code, and no `CMU.MAX` / `MIN` (they rewrite the flags) is on
//!   its way then or issued up to the branch.
//!
//! A block ends when all its results have arrived, so blocks do not depend
//! on each other's timing. A block that ends in a branch puts it seven
//! cycles before its end: the six delay slots are the block's last cycles.
//! Operations in the branch's own bundle stay unconditional (`PEU.PCCX`).
//!
//! **Registers.** A value holds its machine register from the cycle its
//! first result arrives to its last read, and through every block it is
//! alive across ([`super::live`]); a variable between its last read and its
//! next value holds none. Results still on their way do not occupy the
//! register, as on the processor. Pool: `i0`-`i17` and `v0`-`v23`, which a
//! kernel may use without saving.

use super::live::{def, flow, starts};
use super::{AOp, Arg, Block, Item, Kernel, Term, Vr, VregInfo};
use crate::shave::conflicts::{self, WINDOW};
use crate::shave::isa::Operand;
use crate::shave::mach::{sets_flags, two_beats, UNITS};
use crate::shave::prog::{Line, Op, Program};
use std::collections::{BTreeMap, HashMap, HashSet};

const IREGS: u32 = 18;
const VREGS: u32 = 24;
/// Cycles between a store and a load that may read it / a branch and its target.
const STORE_LOAD: u32 = 8;
const DELAY: u32 = 6;

struct Plan {
	time: Vec<u32>,
	alt: Vec<usize>,
	len: u32,
	/// cycle of the block's branch
	branch: Option<u32>,
	/// most values alive at once: vectors, integers (what the block computes
	/// and what only passes through it)
	alive: (u32, u32),
}

/// What a schedule is chosen by.
#[derive(Clone, Copy, PartialEq)]
pub enum Goal {
	/// the shortest
	Short,
	/// the one with the fewest values alive
	Narrow,
}

fn unit(mnem: &str) -> usize {
	let u = mnem.split('.').next().unwrap();
	UNITS.iter().position(|x| *x == u).expect("unit")
}

fn overlap(a: &AOp, b: &AOp) -> bool {
	match (a.part, b.part) {
		(Some((x, n)), Some((y, m))) => x < y + m && y < x + n,
		_ => true,
	}
}

fn alias(a: &AOp, b: &AOp) -> bool {
	let (Some(x), Some(y)) = (&a.mem, &b.mem) else {
		return false;
	};
	if let (Some(s), Some(t)) = (x.tag, y.tag) {
		if s != t {
			return false;
		}
	}
	if x.base == y.base && x.ver == y.ver {
		return x.off < y.off + y.size && y.off < x.off + x.size;
	}
	true
}

#[derive(Default)]
struct RegState {
	full: Option<usize>,
	parts: Vec<usize>,
	readers: Vec<usize>,
}

fn plan(b: &Block, vregs: &[VregInfo], live_out: &HashSet<Vr>, goal: Goal) -> Result<Plan, String> {
	let (wide, deep) = (
		plan_as(b, vregs, live_out, false)?,
		plan_as(b, vregs, live_out, true)?,
	);
	// (a schedule that cannot get its registers is no shorter)
	let fits = |p: &Plan| p.alive.0 <= VREGS && p.alive.1 <= IREGS;
	let key = |p: &Plan| {
		if goal == Goal::Short {
			(!fits(p), p.len, p.alive.0, p.alive.1)
		} else {
			(false, p.alive.0, p.alive.1, p.len)
		}
	};
	Ok(if key(&deep) < key(&wide) { deep } else { wide })
}

/// Where an operation's operand comes from: operations of the block, or the
/// register's value at the block's entry.
enum Source {
	Ops(Vec<usize>),
	Entry(Vr),
}

fn plan_as(
	b: &Block,
	vregs: &[VregInfo],
	live_out: &HashSet<Vr>,
	deep: bool,
) -> Result<Plan, String> {
	let ops = &b.ops;
	let n = ops.len();
	// (earlier operation, cycles this one must be behind it)
	let mut preds: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
	let mut regs: HashMap<Vr, RegState> = HashMap::new();
	let mut mems: Vec<usize> = Vec::new();
	// the operands' sources, and the readers of each operation's result / of each entry value
	let mut sources: Vec<Vec<Source>> = Vec::new();
	let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
	let mut entry_users: HashMap<Vr, Vec<usize>> = HashMap::new();
	for (j, op) in ops.iter().enumerate() {
		sources.push(Vec::new());
		for a in &op.args {
			if let Arg::Use(r, _) = a {
				let s = regs.entry(*r).or_default();
				preds[j].extend(s.full.iter().chain(&s.parts).map(|&d| (d, ops[d].lat)));
				let from: Vec<usize> = s.full.iter().chain(&s.parts).copied().collect();
				from.iter().for_each(|&d| users[d].push(j));
				if from.is_empty() {
					entry_users.entry(*r).or_default().push(j);
				}
				sources[j].push(if from.is_empty() {
					Source::Entry(*r)
				} else {
					Source::Ops(from)
				});
			}
		}
		if let Some(r) = def(op) {
			let s = regs.entry(r).or_default();
			preds[j].extend(s.readers.iter().filter(|&&u| u != j).map(|&u| (u, 0)));
			preds[j].extend(
				s.full
					.iter()
					.chain(&s.parts)
					.filter(|&&d| overlap(&ops[d], op))
					.map(|&d| (d, (ops[d].lat + 1).saturating_sub(op.lat))),
			);
			if op.part.is_none() {
				*s = RegState {
					full: Some(j),
					..Default::default()
				};
			} else {
				s.parts.push(j);
			}
		}
		for a in &op.args {
			if let Arg::Use(r, _) = a {
				if def(op) != Some(*r) || op.part.is_some() {
					regs.entry(*r).or_default().readers.push(j);
				}
			}
		}
		if let Some(m) = &op.mem {
			for &k in &mems {
				if alias(&ops[k], op) {
					match (ops[k].mem.as_ref().unwrap().store, m.store) {
						(true, false) => preds[j].push((k, STORE_LOAD)),
						(false, true) => preds[j].push((k, ops[k].lat)),
						(true, true) => preds[j].push((k, 1)),
						(false, false) => {}
					}
				}
			}
			mems.push(j);
		}
	}
	// urgency: the longest chain behind an operation; the loop's compare
	// (and what it needs) first, so that the branch can sit early
	let mut prio: Vec<u32> = ops.iter().map(|o| o.lat.max(1)).collect();
	let compare = if let Term::Branch { cmp, .. } = b.term {
		Some(cmp)
	} else {
		None
	};
	if let Some(cmp) = compare {
		if cmp + 1 != n {
			return Err(format!(
				"line {}: internal: the compare is not the block's last operation",
				ops[cmp].line
			));
		}
		prio[cmp] += 1 << 20;
	}
	for j in (0..n).rev() {
		for &(i, d) in &preds[j] {
			prio[i] = prio[i].max(d + prio[j]);
		}
	}
	// what leaves the block in its register
	let kept: HashSet<usize> = live_out
		.iter()
		.filter_map(|r| regs.get(r))
		.flat_map(|s| s.full.iter().chain(&s.parts).copied())
		.collect();
	let entry_kept = |r: &Vr| {
		live_out.contains(r)
			&& regs
				.get(r)
				.is_none_or(|s| s.full.is_none() && s.parts.is_empty())
	};
	let mut order: Vec<usize> = (0..n).collect();
	if deep {
		// how far along its chain an operation is
		let mut depth = vec![0u32; n];
		for j in 0..n {
			depth[j] = sources[j]
				.iter()
				.filter_map(|s| {
					if let Source::Ops(from) = s {
						from.iter().map(|&d| depth[d] + 1).max()
					} else {
						None
					}
				})
				.max()
				.unwrap_or(0);
		}
		order.sort_by_key(|&j| (prio[j] < 1 << 20, std::cmp::Reverse(depth[j]), j));
	} else {
		order.sort_by_key(|&j| (std::cmp::Reverse(prio[j]), j));
	}
	let mut time = vec![u32::MAX; n];
	let mut alt = vec![0; n];
	// each operation's form on each of its units, and what is issued in each cycle
	let forms: Vec<Vec<String>> = ops
		.iter()
		.map(|o| {
			let args: Vec<Operand> = o
				.args
				.iter()
				.map(|a| match a {
					Arg::Def(r, elem) | Arg::Use(r, elem) => Operand::Reg {
						vector: vregs[*r].vector,
						n: 0,
						elem: *elem,
					},
					Arg::Int(v) => Operand::Int(*v),
				})
				.collect();
			o.alts
				.iter()
				.map(|m| {
					conflicts::form(&Op {
						mnem: m.clone(),
						args: args.clone(),
					})
				})
				.collect()
		})
		.collect();
	let mut issued: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
	let table = conflicts::table();
	// would operation `j` on its unit `k` in cycle `t` share a resource with one issued?
	let clash = |issued: &HashMap<u32, Vec<(usize, usize)>>, j: usize, k: usize, t: u32| {
		let f = &forms[j][k];
		(0..=WINDOW).any(|d| {
			let before = t
				.checked_sub(d)
				.and_then(|c| issued.get(&c))
				.is_some_and(|l| {
					l.iter()
						.any(|&(i, a)| i != j && table.hit(&forms[i][a], f, d).is_some())
				});
			before
				|| (d > 0
					&& issued.get(&(t + d)).is_some_and(|l| {
						l.iter()
							.any(|&(i, a)| i != j && table.hit(f, &forms[i][a], d).is_some())
					}))
		})
	};
	let mut ports: HashSet<(usize, u32)> = HashSet::new();
	// the compare is placed afterwards, in the cycle before the branch
	let (mut left, mut t) = (n - compare.is_some() as usize, 0u32);
	let mut held = [false; 8];
	// (cycle, unit) taken
	let mut busy: HashSet<(u32, usize)> = HashSet::new();
	while left > 0 {
		// a 128-bit access keeps its LSU for a second cycle
		let mut used = std::mem::take(&mut held);
		loop {
			let mut progress = false;
			for &j in &order {
				if Some(j) == compare
					|| time[j] != u32::MAX
					|| !preds[j]
						.iter()
						.all(|&(i, d)| time[i] != u32::MAX && time[i] + d <= t)
				{
					continue;
				}
				let op = &ops[j];
				for (k, m) in op.alts.iter().enumerate() {
					let u = unit(m);
					// VAU = 0, SAU = 4 share their datapath
					if used[u] || (u == 0 && used[4]) || (u == 4 && used[0]) {
						continue;
					}
					// one result per unit, register file and cycle
					let port =
						def(op).map(|r| (u + if vregs[r].vector { 8 } else { 0 }, t + op.lat));
					let wide = two_beats(m);
					// (a wide load's low half arrives a cycle before the high one)
					if port.is_some_and(|p| {
						ports.contains(&p) || (wide && ports.contains(&(p.0, p.1 - 1)))
					}) || clash(&issued, j, k, t)
					{
						continue;
					}
					issued.entry(t).or_default().push((j, k));
					ports.extend(port);
					if wide {
						ports.extend(port.map(|p| (p.0, p.1 - 1)));
						held[u] = true;
						busy.insert((t + 1, u));
					}
					busy.insert((t, u));
					used[u] = true;
					time[j] = t;
					alt[j] = k;
					left -= 1;
					progress = true;
					break;
				}
			}
			if !progress {
				break;
			}
		}
		t += 1;
		if t > 1 << 20 {
			return Err(format!(
				"line {}: the block cannot be scheduled",
				ops[0].line
			));
		}
	}
	// all results in (and the second cycle of a 128-bit store in the last one)
	let t = t + held.iter().any(|h| *h) as u32;
	let len = ops
		.iter()
		.zip(&time)
		.filter(|(o, t)| def(o).is_some() && **t != u32::MAX)
		.map(|(o, t)| t + o.lat)
		.max()
		.unwrap_or(0)
		.max(t);
	// Everything was issued as early as possible, which is earlier than the
	// readers need most results: an operation moves to the last cycle that
	// changes nothing else (a unit and its result port free, its readers and
	// the block's end not delayed), so that its value holds a register only
	// from shortly before it is read. Loads and immediates always; another
	// operation as far as its operands are alive anyway (read by something
	// else that late, or leaving the block), or it would trade one register
	// for several. What the compare waits for stays.
	let mut succs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
	for (j, p) in preds.iter().enumerate() {
		for &(i, d) in p {
			succs[i].push((j, d));
		}
	}
	let mut late: Vec<usize> = (0..n).filter(|&j| time[j] != u32::MAX).collect();
	late.sort_by_key(|&j| std::cmp::Reverse((time[j], j)));
	let mut moved = vec![false; n];
	for j in late {
		moved[j] = true;
		let op = &ops[j];
		let load = op.mem.as_ref().is_some_and(|m| !m.store)
			|| !op.args.iter().any(|a| matches!(a, Arg::Use(..)));
		let Some(r) = def(op) else { continue };
		if succs[j].iter().any(|&(k, _)| time[k] == u32::MAX)
			|| sets_flags(&op.alts[0])
			|| vregs[r].pin.is_some()
		{
			continue;
		}
		let mut last = succs[j]
			.iter()
			.map(|&(k, d)| time[k].saturating_sub(d))
			.min()
			.unwrap_or(u32::MAX)
			.min(len.saturating_sub(op.lat));
		if !load {
			// when another reader is issued: where it is, or (not moved yet) where it may go
			let when = |k: usize| {
				let may = succs[k]
					.iter()
					.map(|&(s, d)| {
						if time[s] == u32::MAX {
							0
						} else {
							time[s].saturating_sub(d)
						}
					})
					.min();
				if moved[k] || def(&ops[k]).is_none() {
					time[k]
				} else {
					may.unwrap_or(time[k]).max(time[k])
				}
			};
			let others = |l: &[usize]| {
				l.iter()
					.filter(|&&k| k != j && time[k] != u32::MAX)
					.map(|&k| when(k))
					.max()
					.unwrap_or(0)
			};
			for s in &sources[j] {
				last = last.min(match s {
					Source::Ops(from) if from.iter().any(|d| kept.contains(d)) => u32::MAX,
					Source::Ops(from) => from.iter().map(|&d| others(&users[d])).min().unwrap_or(0),
					Source::Entry(r) if entry_kept(r) => u32::MAX,
					Source::Entry(r) => others(&entry_users[r]),
				});
			}
		}
		let (wide, file) = (two_beats(&op.alts[0]), if vregs[r].vector { 8 } else { 0 });
		let beats = |t: u32| t..=t + wide as u32;
		let lands = |t: u32| t + op.lat - wide as u32..=t + op.lat;
		let (t0, u0) = (time[j], unit(&op.alts[alt[j]]));
		busy.retain(|&(c, u)| u != u0 || !beats(t0).contains(&c));
		issued.get_mut(&t0).unwrap().retain(|e| e.0 != j);
		ports.retain(|&(p, c)| p != u0 + file || !lands(t0).contains(&c));
		// (the CMU stays free where the compare goes)
		let tail = if compare.is_some() {
			len.saturating_sub(DELAY + 3)
		} else {
			u32::MAX
		};
		let fits =
			|t: u32, u: usize, busy: &HashSet<(u32, usize)>, ports: &HashSet<(usize, u32)>| {
				let shared = (u == 0 && busy.contains(&(t, 4)))
					|| (u == 4 && busy.contains(&(t, 0)))
					|| (u == 1 && t >= tail);
				!shared
					&& beats(t).all(|c| !busy.contains(&(c, u)))
					&& lands(t).all(|c| !ports.contains(&(u + file, c)))
			};
		// (on its unit if that is free: the halves of a vector stay LSU0 low, LSU1 high,
		// which the memory serves faster than the other way round)
		let units = || std::iter::once(alt[j]).chain((0..op.alts.len()).filter(|&k| k != alt[j]));
		let found = (t0 + 1..=last.max(t0)).rev().find_map(|t| {
			units()
				.find(|&k| fits(t, unit(&op.alts[k]), &busy, &ports) && !clash(&issued, j, k, t))
				.map(|k| (t, k))
		});
		let (t1, k1) = found.unwrap_or((t0, alt[j]));
		let u1 = unit(&op.alts[k1]);
		busy.extend(beats(t1).map(|c| (c, u1)));
		ports.extend(lands(t1).map(|c| (u1 + file, c)));
		issued.entry(t1).or_default().push((j, k1));
		(time[j], alt[j]) = (t1, k1);
	}
	let cmu_busy: HashSet<u32> = busy.iter().filter(|b| b.1 == 1).map(|b| b.0).collect();
	// values alive at once: from a result's arrival to its last reader (the
	// block's end if it leaves the block)
	let alive = |time: &[u32], len: u32| {
		let mut most = [0u32; 2];
		for class in [false, true] {
			let mut events: Vec<(u32, i32)> = Vec::new();
			for (j, op) in ops.iter().enumerate() {
				if !def(op).is_some_and(|r| {
					vregs[r].vector == class
						&& vregs[r].pin.is_none()
						&& (op.part.is_none() || op.begin)
				}) {
					continue;
				}
				let from = time[j] + op.lat;
				let to = if kept.contains(&j) {
					len
				} else {
					users[j]
						.iter()
						.map(|&k| time[k])
						.max()
						.unwrap_or(from)
						.max(from)
				};
				events.extend([(from, 1), (to + 1, -1)]);
			}
			events.sort();
			let mut now = live_out
				.iter()
				.filter(|r| vregs[**r].vector == class && vregs[**r].pin.is_none() && entry_kept(r))
				.count() as i32;
			most[class as usize] = now as u32;
			for (_, d) in events {
				now += d;
				most[class as usize] = most[class as usize].max(now as u32);
			}
		}
		(most[1], most[0])
	};
	Ok(match b.term {
		Term::None => Plan {
			alive: alive(&time, len),
			time,
			alt,
			len,
			branch: None,
		},
		Term::Branch { cmp, .. } => {
			// As in compiled code the compare sits right in front of its branch:
			// CMU.MAX / MIN rewrite the flags (measured), and a compare 28
			// cycles ahead of its branch lost the device. So: the CMU free in
			// the compare's cycle, and no MAX / MIN on its way then or issued
			// up to the branch.
			let ready = preds[cmp]
				.iter()
				.map(|&(i, d)| time[i] + d)
				.max()
				.unwrap_or(0);
			let noisy: Vec<(u32, u32)> = ops
				.iter()
				.zip(&time)
				.filter(|(o, _)| sets_flags(&o.alts[0]))
				.map(|(o, t)| (*t, o.lat))
				.collect();
			let mut at = len.saturating_sub(DELAY + 1).max(ready + 1);
			while cmu_busy.contains(&(at - 1))
				|| noisy.iter().any(|&(t, lat)| t + lat > at - 1 && t <= at)
				|| clash(&issued, cmp, 0, at - 1)
			{
				at += 1;
			}
			time[cmp] = at - 1;
			Plan {
				alive: alive(&time, at + DELAY + 1),
				time,
				alt,
				len: at + DELAY + 1,
				branch: Some(at),
			}
		}
		Term::Ret | Term::Jump(_) => {
			let at = len.saturating_sub(DELAY + 1);
			Plan {
				alive: alive(&time, at + DELAY + 1),
				time,
				alt,
				len: at + DELAY + 1,
				branch: Some(at),
			}
		}
	})
}

pub fn schedule(vregs: &[VregInfo], items: &[Item], goal: Goal) -> Result<Kernel, String> {
	let n = vregs.len();
	let live = flow(vregs, items)?;
	let mut plans: Vec<Option<(Plan, u32)>> = Vec::new();
	// the cycles (first, last) in which each register holds a value
	let mut held = vec![Vec::<(u32, u32)>::new(); n];
	let mut first_line = vec![0usize; n];
	let mut loops: Vec<(u32, u32, usize)> = Vec::new();
	let mut open: Vec<(u32, usize)> = Vec::new();
	let mut t0 = 0u32;
	for (k, item) in items.iter().enumerate() {
		plans.push(match item {
			Item::Label(_) => None,
			Item::LoopBegin(line) => {
				open.push((t0, *line));
				None
			}
			Item::LoopEnd => {
				let (s, line) = open.pop().expect("loop");
				loops.push((s, t0 - 1, line));
				None
			}
			Item::Block(b) => {
				let p = plan(b, vregs, &live.live_out[k], goal)?;
				let end = t0 + p.len - 1;
				// (register, which of its values in this block): 0 is the one it enters with
				let mut value: HashMap<Vr, u32> = HashMap::new();
				let mut spans: HashMap<(Vr, u32), (u32, u32)> = HashMap::new();
				let mut touch = |key: (Vr, u32), from: u32, to: u32| {
					let s = spans.entry(key).or_insert((from, to));
					*s = (s.0.min(from), s.1.max(to));
				};
				for (op, &t) in b.ops.iter().zip(&p.time) {
					for a in &op.args {
						if let Arg::Use(r, _) | Arg::Def(r, _) = a {
							if first_line[*r] == 0 {
								first_line[*r] = op.line;
							}
						}
						if let Arg::Use(r, _) = a {
							touch((*r, *value.get(r).unwrap_or(&0)), t0 + t, t0 + t);
						}
					}
					if let Some(r) = def(op) {
						let v = value.entry(r).or_insert(0);
						*v += starts(op).is_some() as u32;
						// (a wide load's low half is there a cycle earlier)
						let land = t0 + t + op.lat;
						touch((r, *v), land - two_beats(&op.alts[0]) as u32, land);
					}
				}
				for &r in &live.live_in[k] {
					touch((r, 0), t0, t0);
				}
				for &r in &live.live_out[k] {
					touch((r, *value.get(&r).unwrap_or(&0)), end, end);
				}
				for ((r, _), s) in spans {
					held[r].push(s);
				}
				let at = t0;
				t0 += p.len;
				Some((p, at))
			}
		});
	}
	// Machine registers: value by value in the order they begin, the lowest
	// that is free in every cycle the value is held. For values held in one
	// stretch that needs the fewest registers possible; with gaps it may not,
	// so before giving up every value is held from its first cycle to its last.
	let mut order: Vec<Vr> = (0..n)
		.filter(|&r| vregs[r].pin.is_none() && !held[r].is_empty())
		.collect();
	for h in &mut held {
		// (one stretch where two touch)
		h.sort();
		let mut merged: Vec<(u32, u32)> = Vec::new();
		for &(s, e) in h.iter() {
			match merged.last_mut() {
				Some(m) if s <= m.1 + 1 => m.1 = m.1.max(e),
				_ => merged.push((s, e)),
			}
		}
		*h = merged;
	}
	order.sort_by_key(|&r| (held[r][0].0, held[r].last().unwrap().1, r));
	let assign = |held: &[Vec<(u32, u32)>]| -> Result<Vec<u32>, String> {
		let mut phys: Vec<u32> = vregs.iter().map(|v| v.pin.unwrap_or(u32::MAX)).collect();
		// per machine register: first cycle -> (last cycle, the value's register)
		let mut taken: [Vec<BTreeMap<u32, (u32, Vr)>>; 2] = [
			vec![BTreeMap::new(); IREGS as usize],
			vec![BTreeMap::new(); VREGS as usize],
		];
		for &r in &order {
			let class = vregs[r].vector as usize;
			let owner = |m: &BTreeMap<u32, (u32, Vr)>| {
				held[r].iter().find_map(|&(s, e)| {
					m.range(..=e)
						.next_back()
						.filter(|(_, o)| o.0 >= s)
						.map(|(_, o)| o.1)
				})
			};
			let Some(p) = taken[class].iter().position(|m| owner(m).is_none()) else {
				let mut live: Vec<&str> = taken[class]
					.iter()
					.filter_map(owner)
					.map(|o| vregs[o].name.as_str())
					.collect();
				live.sort();
				return Err(format!(
					"line {}: no {} register left for {} ({} hold: {})",
					first_line[r],
					if class == 1 { "vector" } else { "int" },
					vregs[r].name,
					taken[class].len(),
					live.join(", ")
				));
			};
			taken[class][p].extend(held[r].iter().map(|&(s, e)| (s, (e, r))));
			phys[r] = p as u32;
		}
		Ok(phys)
	};
	let whole = || {
		held.iter()
			.map(|h| {
				h.first()
					.map(|f| (f.0, h.last().unwrap().1))
					.into_iter()
					.collect()
			})
			.collect::<Vec<_>>()
	};
	let phys = assign(&held).or_else(|e| assign(&whole()).map_err(|_| e))?;
	// the bundles
	let mut lines = Vec::new();
	let label = |k: usize| format!(".L{k}");
	for (item, plan) in items.iter().zip(&plans) {
		let (b, (p, _)) = match (item, plan) {
			(Item::Label(k), _) => {
				lines.push(Line::Label(label(*k)));
				continue;
			}
			(Item::Block(b), Some(p)) => (b, p),
			_ => continue,
		};
		let mut cycles: Vec<Vec<Op>> = vec![Vec::new(); p.len as usize];
		for (j, op) in b.ops.iter().enumerate() {
			let args = op
				.args
				.iter()
				.map(|a| match a {
					Arg::Def(r, elem) | Arg::Use(r, elem) => Operand::Reg {
						vector: vregs[*r].vector,
						n: phys[*r],
						elem: *elem,
					},
					Arg::Int(v) => Operand::Int(*v),
				})
				.collect();
			cycles[p.time[j] as usize].push(Op {
				mnem: op.alts[p.alt[j]].clone(),
				args,
			});
		}
		match (&b.term, p.branch) {
			(Term::Branch { cc, label: to, .. }, Some(at)) => {
				// PC1C makes the whole bundle conditional. Beside other operations
				// the branch takes PCCX, whose mask names the units that are
				// conditional too: every unit not in the bundle
				// (bit 0 IAU, 1 SAU, 2 CMU, 3 VAU, 4 LSU0, 5 LSU1)
				let here = &mut cycles[at as usize];
				if here.is_empty() {
					here.push(Op::new("PEU.PC1C", vec![Operand::Ident(cc.to_string())]));
				} else {
					let mask = ["IAU", "SAU", "CMU", "VAU", "LSU0", "LSU1"]
						.iter()
						.enumerate()
						.filter(|(_, u)| !here.iter().any(|o| o.unit() == **u))
						.map(|(bit, _)| 1i64 << bit)
						.sum();
					here.push(Op::new(
						"PEU.PCCX",
						vec![Operand::Ident(cc.to_string()), Operand::Int(mask)],
					));
				}
				cycles[at as usize].push(Op::new("BRU.BRA", vec![Operand::Ident(label(*to))]));
			}
			(Term::Jump(to), Some(at)) => {
				cycles[at as usize].push(Op::new("BRU.BRA", vec![Operand::Ident(label(*to))]))
			}
			(Term::Ret, Some(at)) => cycles[at as usize].push(Op::new(
				"BRU.JMP",
				vec![Operand::Reg {
					vector: false,
					n: 30,
					elem: None,
				}],
			)),
			_ => {}
		}
		let mut idle = 0;
		for ops in cycles {
			if ops.is_empty() {
				idle += 1;
				continue;
			}
			if idle > 0 {
				lines.push(Line::Nop(idle));
				idle = 0;
			}
			lines.push(Line::Bundle(ops));
		}
		if idle > 0 {
			lines.push(Line::Nop(idle));
		}
	}
	loops.sort_by_key(|l| l.0);
	let report = loops
		.iter()
		.map(|&(s, e, line)| format!("loop at line {line}: {} cycles per round", e - s + 1))
		.collect();
	Ok(Kernel {
		program: Program { lines },
		report,
	})
}
