//! Values instead of variables: what the scheduler and the register
//! assignment work on.
//!
//! **Renaming.** [`rename`] gives every value a variable takes inside a
//! straight-line block a virtual register of its own; only the one a later
//! block reads keeps the variable's. This is SSA form
//! without phi nodes, which a block does not need. The scheduler then sees
//! only the dependencies of the data: a variable assigned twice no longer
//! makes the second value wait for the readers of the first.
//!
//! **Liveness.** [`flow`] computes which registers hold a value at each
//! block's entry and exit (backward data flow over the blocks and their
//! branches). The register assignment keeps a machine register only where
//! its value is alive, and a register alive at the kernel's entry is one read
//! before it is set.

use super::{AOp, Arg, Item, Term, Vr, VregInfo};
use std::collections::{HashMap, HashSet};

pub fn def(op: &AOp) -> Option<Vr> {
	op.args.iter().find_map(|a| {
		if let Arg::Def(r, _) = a {
			Some(*r)
		} else {
			None
		}
	})
}

/// The register whose new value `op` begins: a whole write, or the first of
/// the partial writes that make one up.
pub fn starts(op: &AOp) -> Option<Vr> {
	def(op).filter(|_| op.part.is_none() || op.begin)
}

/// A virtual register for every value of a block but the one each variable
/// leaves it with (`live`: [`flow`] of these items).
pub fn rename(vregs: &mut Vec<VregInfo>, items: &mut [Item], live: &Flow) {
	for (k, item) in items.iter_mut().enumerate() {
		let Item::Block(b) = item else { continue };
		let mut last: HashMap<Vr, usize> = HashMap::new();
		for (j, op) in b.ops.iter().enumerate() {
			last.extend(starts(op).map(|r| (r, j)));
		}
		// the register that holds each variable's current value
		let mut cur: HashMap<Vr, Vr> = HashMap::new();
		for (j, op) in b.ops.iter_mut().enumerate() {
			let begun = starts(op);
			for a in &mut op.args {
				if let Arg::Use(r, _) = a {
					*r = *cur.get(r).unwrap_or(r);
				}
			}
			if let Some(m) = &mut op.mem {
				m.base = *cur.get(&m.base).unwrap_or(&m.base);
			}
			if let Some(r) = begun {
				if (last[&r] == j && live.live_out[k].contains(&r)) || vregs[r].pin.is_some() {
					cur.remove(&r);
				} else {
					vregs.push(vregs[r].clone());
					cur.insert(r, vregs.len() - 1);
				}
			}
			for a in &mut op.args {
				if let Arg::Def(r, _) = a {
					*r = *cur.get(r).unwrap_or(r);
				}
			}
		}
	}
}

/// The registers alive at the entry and at the exit of each item (empty for
/// what is not a block).
pub struct Flow {
	pub live_in: Vec<HashSet<Vr>>,
	pub live_out: Vec<HashSet<Vr>>,
}

pub fn flow(vregs: &[VregInfo], items: &[Item]) -> Result<Flow, String> {
	let n = items.len();
	let blocks: Vec<usize> = (0..n)
		.filter(|&k| matches!(items[k], Item::Block(_)))
		.collect();
	let after = |k: usize| blocks.iter().copied().find(|&b| b > k);
	let target = |label: usize| {
		items
			.iter()
			.position(|i| matches!(i, Item::Label(l) if *l == label))
			.and_then(after)
	};
	let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
	// read before any write in the block (and the line of that read) / begun in it
	let mut exposed: Vec<HashMap<Vr, usize>> = vec![HashMap::new(); n];
	let mut begun: Vec<HashSet<Vr>> = vec![HashSet::new(); n];
	for &k in &blocks {
		let Item::Block(b) = &items[k] else {
			unreachable!()
		};
		succ[k] = match b.term {
			Term::None => after(k).into_iter().collect(),
			Term::Branch { label, .. } => target(label).into_iter().chain(after(k)).collect(),
			Term::Jump(label) => target(label).into_iter().collect(),
			Term::Ret => Vec::new(),
		};
		for op in &b.ops {
			// a partial write keeps the rest of the register: it reads it
			let kept = def(op).filter(|_| starts(op).is_none());
			for r in op
				.args
				.iter()
				.filter_map(|a| {
					if let Arg::Use(r, _) = a {
						Some(*r)
					} else {
						None
					}
				})
				.chain(kept)
			{
				if !begun[k].contains(&r) {
					exposed[k].entry(r).or_insert(op.line);
				}
			}
			begun[k].extend(starts(op));
		}
	}
	let mut live_in: Vec<HashSet<Vr>> = exposed
		.iter()
		.map(|e| e.keys().copied().collect())
		.collect();
	let mut live_out: Vec<HashSet<Vr>> = vec![HashSet::new(); n];
	let mut changed = true;
	while changed {
		changed = false;
		for &k in blocks.iter().rev() {
			let new: Vec<Vr> = succ[k]
				.iter()
				.flat_map(|&s| &live_in[s])
				.copied()
				.filter(|r| !live_out[k].contains(r))
				.collect();
			for r in new {
				live_out[k].insert(r);
				if !begun[k].contains(&r) {
					changed |= live_in[k].insert(r);
				}
			}
		}
	}
	// alive at the entry: some path reads it before any write
	let Some(&entry) = blocks.first() else {
		return Ok(Flow { live_in, live_out });
	};
	let unset = live_in[entry]
		.iter()
		.copied()
		.filter(|&r| vregs[r].pin.is_none())
		.min();
	if let Some(r) = unset {
		// the first such read: through the blocks that do not write it
		let (mut seen, mut todo, mut line) = (HashSet::from([entry]), vec![entry], usize::MAX);
		while let Some(k) = todo.pop() {
			line = line.min(*exposed[k].get(&r).unwrap_or(&usize::MAX));
			if !begun[k].contains(&r) {
				todo.extend(succ[k].iter().copied().filter(|&s| seen.insert(s)));
			}
		}
		return Err(format!(
			"line {line}: {} is read before it is set",
			vregs[r].name
		));
	}
	Ok(Flow { live_in, live_out })
}
