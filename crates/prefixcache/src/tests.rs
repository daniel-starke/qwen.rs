use super::*;

/// A hybrid model's state after `len` tokens: an opaque recurrent part of
/// `r` bytes (its content depends on the whole prefix, here on `len` and
/// `salt`) and a row part, 2 streams of 4-byte rows whose content depends
/// on the row's position and the conversation's `salt` only.
fn state(len: usize, salt: u8, r: usize) -> Snap {
	let mut rows = Vec::with_capacity(2 * len * 4);
	for s in 0..2 {
		for i in 0..len {
			rows.extend_from_slice(&[(i % 251) as u8, (i / 251) as u8, s, salt]);
		}
	}
	Snap {
		pos: len,
		parts: vec![
			quant::snap::part(vec![(len % 256) as u8 ^ salt; r]),
			quant::snap::part(rows),
		],
		layout: vec![
			Layout::Opaque,
			Layout::Rows {
				streams: 2,
				rows: len,
			},
		],
	}
}

fn toks(n: usize, salt: u32) -> Vec<u32> {
	(0..n as u32)
		.map(|i| i.wrapping_mul(7) ^ (salt << 20))
		.collect()
}

fn cfg(budget: usize) -> Config {
	Config {
		budget,
		beta: 0.3,
		half_life: 64.0,
		shared_weight: 2.0,
		interval: 4096,
		rewindable: false,
		prefill_tok_s: 100.0,
	}
}

#[test]
fn longest_proper_prefix_hits_and_hits_do_not_consume() {
	let c = Cache::new(cfg(1 << 30));
	let t = toks(3000, 1);
	c.insert(t[..1000].to_vec(), state(1000, 1, 64).into(), Class::Turn);
	c.insert(t[..2000].to_vec(), state(2000, 1, 64).into(), Class::Turn);
	c.insert(toks(500, 9), state(500, 9, 64).into(), Class::Turn);
	for _ in 0..2 {
		let h = c.lookup(&t).unwrap();
		assert_eq!((h.len, h.from_disk), (2000, false));
		assert_eq!(h.snap, state(2000, 1, 64));
	}
	// an identical prompt leaves nothing to prefill: not a hit on itself
	assert_eq!(c.lookup(&t[..2000]).unwrap().len, 1000);
	assert!(c.lookup(&t[..1000]).is_none());
	assert_eq!(c.len(), 3);
	assert!(c.contains(&t[..1000]) && !c.contains(&t[..999]));
	// the same tokens replace rather than duplicate
	c.insert(t[..1000].to_vec(), state(1000, 1, 64).into(), Class::Turn);
	assert_eq!(c.len(), 3);
}

#[test]
fn checkpoints_of_a_conversation_share_their_rows() {
	let c = Cache::new(cfg(1 << 30));
	let t = toks(8 * PAGE, 1);
	for k in 1..=8 {
		let n = k * PAGE;
		c.insert(t[..n].to_vec(), state(n, 1, 1000).into(), Class::Turn);
	}
	// 8 recurrent parts + the rows once (8 pages of 2 x 256 x 4 bytes)
	assert_eq!(c.bytes(), 8 * 1000 + 8 * PAGE * 8);
	// a branch shares nothing past where its rows differ
	c.insert(
		toks(4 * PAGE, 2),
		state(4 * PAGE, 2, 1000).into(),
		Class::Turn,
	);
	assert_eq!(c.bytes(), 9 * 1000 + 12 * PAGE * 8);
	// every state still comes back byte for byte
	for k in 1..8 {
		let h = c.lookup(&t[..k * PAGE + 1]).unwrap();
		assert_eq!(h.snap, state(k * PAGE, 1, 1000));
	}
}

#[test]
fn eviction_thins_a_chain_and_keeps_its_tip() {
	// recurrent parts dominate: each state costs ~1 MiB on its own
	let r = 1 << 20;
	let c = Cache::new(cfg(6 * r + (1 << 18)));
	let t = toks(12_000, 1);
	let cuts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000];
	for &n in &cuts {
		c.insert(t[..n].to_vec(), state(n, 1, r).into(), Class::Interval);
	}
	c.insert(
		t[..11_000].to_vec(),
		state(11_000, 1, r).into(),
		Class::Turn,
	);
	assert!(c.len() <= 6);
	// the tip stays
	assert!(c.contains(&t[..11_000]));
	assert_eq!(c.lookup(&t).unwrap().len, 11_000);
	// the gaps left are spread out (every second state of equal gaps): no
	// two neighbors dropped, so no replay is over two gaps anywhere
	let kept: Vec<usize> = cuts
		.iter()
		.copied()
		.filter(|&n| c.contains(&t[..n]))
		.collect();
	let mut last = 0;
	for &n in kept.iter().chain([11_000].iter()) {
		assert!(n - last <= 2000, "{kept:?}");
		last = n;
	}
}

#[test]
fn an_old_conversation_loses_to_a_recent_one() {
	let r = 1 << 20;
	let c = Cache::new(cfg(2 * r + (1 << 18)));
	c.insert(toks(4000, 1), state(4000, 1, r).into(), Class::Turn);
	for _ in 0..200 {
		c.lookup(&toks(10, 7));
	}
	c.insert(toks(4000, 2), state(4000, 2, r).into(), Class::Turn);
	c.insert(toks(4000, 3), state(4000, 3, r).into(), Class::Turn);
	assert!(!c.contains(&toks(4000, 1)));
	assert!(c.contains(&toks(4000, 2)) && c.contains(&toks(4000, 3)));
}

#[test]
fn shared_prefixes_outweigh_turns_of_equal_age() {
	let r = 1 << 20;
	let c = Cache::new(cfg(r + (1 << 19)));
	c.insert(toks(4000, 1), state(4000, 1, r).into(), Class::Shared);
	c.insert(toks(4000, 2), state(4000, 2, r).into(), Class::Turn);
	assert!(c.contains(&toks(4000, 1)) && !c.contains(&toks(4000, 2)));
}

#[test]
fn rewindable_models_hit_inside_a_longer_state() {
	let mut k = cfg(1 << 30);
	k.rewindable = true;
	let c = Cache::new(k);
	let t = toks(3000, 1);
	let rows_only = |n: usize| {
		let mut s = state(n, 1, 0);
		s.parts.remove(0);
		s.layout.remove(0);
		s
	};
	c.insert(t.clone(), rows_only(3000).into(), Class::Turn);
	// another continuation of the first 1500 tokens
	let mut q = t[..1500].to_vec();
	q.extend(toks(100, 5));
	let h = c.lookup(&q).unwrap();
	assert_eq!(h.len, 1500);
	assert_eq!(h.snap, rows_only(1500));
	// a prompt equal to the state keeps one token to prefill
	assert_eq!(c.lookup(&t).unwrap().len, 2999);
}

#[test]
fn interval_cuts_are_multiples_inside_the_range() {
	let c = Cache::new(cfg(1));
	assert_eq!(c.interval_cuts(0, 4096), Vec::<usize>::new());
	assert_eq!(c.interval_cuts(100, 12_289), vec![4096, 8192, 12_288]);
	assert_eq!(c.interval_cuts(4096, 8193), vec![8192]);
	assert!(Cache::off().interval_cuts(0, 1 << 20).is_empty());
}

fn tmpdir(name: &str) -> std::path::PathBuf {
	let d = std::env::temp_dir().join(format!("qwen-rs-pc-{name}-{}", std::process::id()));
	let _ = std::fs::remove_dir_all(&d);
	d
}

fn disk_cfg(d: &std::path::Path, spill: bool) -> DiskCfg {
	DiskCfg {
		dir: d.to_path_buf(),
		tag: "m".into(),
		budget: 1 << 30,
		spill,
		wear_s_per_gb: 30.0,
		day_bytes: 1 << 40,
	}
}

#[test]
fn spilled_states_come_back_from_disk_and_survive_a_restart() {
	let d = tmpdir("spill");
	let r = 1 << 20;
	let t = toks(9000, 1);
	{
		let c = Cache::new(cfg(r + (1 << 19)))
			.with_disk(disk_cfg(&d, true))
			.unwrap();
		c.insert(t[..4000].to_vec(), state(4000, 1, r).into(), Class::Turn);
		// over budget: the first state is evicted to disk
		c.insert(toks(4000, 2), state(4000, 2, r).into(), Class::Turn);
		c.flush();
		assert_eq!(c.len(), 1);
		let h = c.lookup(&t).unwrap();
		assert_eq!((h.len, h.from_disk), (4000, true));
		assert_eq!(
			&quant::snap::concat(&h.snap.parts[0])[..],
			state(4000, 1, r).parts[0][0].bytes()
		);
		assert_eq!(
			&quant::snap::concat(&h.snap.parts[1])[..],
			&quant::snap::concat(&state(4000, 1, r).parts[1])[..]
		);
		// now in RAM again, and not written twice
		assert!(c.lookup(&t).is_some());
		// a longer checkpoint of the reloaded conversation shares its pages
		let before = c.bytes();
		c.insert(t[..4096].to_vec(), state(4096, 1, 16).into(), Class::Turn);
		assert!(
			c.bytes() - before < 2 * PAGE * 8 + 16 + (1 << 20),
			"{}",
			c.bytes() - before
		);
		c.flush();
	}
	// a new process finds it (and the other state, spilled by the reload)
	let c = Cache::new(cfg(1 << 30))
		.with_disk(disk_cfg(&d, true))
		.unwrap();
	assert!(c.contains(&t[..4000]));
	assert_eq!(c.lookup(&t).unwrap().len, 4000);
	let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn disk_writes_only_new_chunks_and_only_when_worth_their_wear() {
	let d = tmpdir("wear");
	let disk = disk::Disk::open(disk_cfg(&d, true)).unwrap();
	let t = toks(8 * PAGE, 1);
	let st = |n: usize| Arc::new(page::store(state(n, 1, 4096).into(), None).unwrap());
	let a = st(4 * PAGE);
	disk.offer(key::Key::new(t[..4 * PAGE].to_vec()), a, Class::Turn, 1e6);
	disk.flush();
	let (_, first) = disk.stats();
	// the longer state's first four pages are on disk already
	let b = st(8 * PAGE);
	disk.offer(key::Key::new(t.clone()), b, Class::Turn, 1e6);
	disk.flush();
	let (n, total) = disk.stats();
	assert_eq!(n, 2);
	let second = total - first;
	assert!(
		second < first + 4 * PAGE as u64 * 8 + 4096,
		"{first} {second}"
	);
	assert!(second > 4 * PAGE as u64 * 8, "{second}");
	// not worth it: a few seconds saved for a gigabyte-scale write at 30 s/GB
	let big = Arc::new(page::store(state(10, 3, 64 << 20).into(), None).unwrap());
	disk.offer(key::Key::new(toks(10, 3)), big, Class::Turn, 1.0);
	disk.flush();
	assert_eq!(disk.stats().0, 2);
	// the reload reads it back whole
	let (len, id, _) = disk.best(&toks(8 * PAGE + 1, 1)[..]).unwrap();
	assert_eq!(len, 8 * PAGE);
	let (_, s, class) = disk.load(id).unwrap();
	assert_eq!(class, Class::Turn);
	assert_eq!(page::hand_out(&s).unwrap(), state(8 * PAGE, 1, 4096));
	let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_daily_write_budget_holds_across_restarts() {
	let d = tmpdir("day");
	let mut k = disk_cfg(&d, true);
	k.day_bytes = 3 << 20;
	let st = |salt: u8| Arc::new(page::store(state(10, salt, 1 << 20).into(), None).unwrap());
	{
		let disk = disk::Disk::open(k.clone()).unwrap();
		disk.offer(key::Key::new(toks(10, 1)), st(1), Class::Turn, 1e6);
		disk.flush();
		disk.offer(key::Key::new(toks(10, 2)), st(2), Class::Turn, 1e6);
		disk.flush();
		assert_eq!(disk.stats().0, 2);
	}
	let disk = disk::Disk::open(k).unwrap();
	disk.offer(key::Key::new(toks(10, 3)), st(3), Class::Turn, 1e6);
	disk.flush();
	assert_eq!(disk.stats().0, 2);
	let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn disk_eviction_takes_packs_nothing_references_first() {
	let d = tmpdir("dead");
	let mut k = disk_cfg(&d, true);
	k.budget = u64::MAX;
	let disk = disk::Disk::open(k.clone()).unwrap();
	let t = toks(8 * PAGE, 1);
	let st = |n: usize| Arc::new(page::store(state(n, 1, 1000).into(), None).unwrap());
	disk.offer(
		key::Key::new(t[..4 * PAGE].to_vec()),
		st(4 * PAGE),
		Class::Turn,
		1e6,
	);
	disk.flush();
	disk.offer(key::Key::new(t.clone()), st(8 * PAGE), Class::Turn, 1e6);
	disk.flush();
	let (_, two) = disk.stats();
	drop(disk);
	// room for one more small pack only if one goes: the long one, which
	// nothing references (evicting the short one would free nothing)
	k.budget = two;
	let disk = disk::Disk::open(k.clone()).unwrap();
	disk.offer(key::Key::new(toks(5, 9)), st(5), Class::Turn, 1e6);
	disk.flush();
	assert!(disk.has(&t[..4 * PAGE]) && !disk.has(&t) && disk.has(&toks(5, 9)));
	// a budget under what the newest pack needs with its base: the others
	// go, the referenced base last and as a dead file (the newest reads it)
	k.budget = 20_000;
	drop(disk);
	let disk = disk::Disk::open(k).unwrap();
	disk.offer(key::Key::new(t.clone()), st(8 * PAGE), Class::Turn, 1e6);
	disk.flush();
	assert_eq!(disk.stats().0, 1);
	let files: Vec<String> = std::fs::read_dir(&d)
		.unwrap()
		.flatten()
		.map(|e| e.file_name().to_string_lossy().into_owned())
		.filter(|f| f.ends_with(".qpc") || f.ends_with(".qpd"))
		.collect();
	assert_eq!(files.len(), 2, "{files:?}");
	assert_eq!(files.iter().filter(|f| f.ends_with(".qpd")).count(), 1);
	let (_, id, _) = disk.best(&toks(8 * PAGE + 1, 1)).unwrap();
	let (_, s, _) = disk.load(id).unwrap();
	assert_eq!(page::hand_out(&s).unwrap(), state(8 * PAGE, 1, 1000));
	let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_lookup_waits_for_a_reserved_longer_state() {
	let c = Arc::new(Cache::new(cfg(1 << 30)));
	let t = toks(3000, 1);
	c.insert(
		t[..1000].to_vec(),
		state(1000, 1, 64).into(),
		Class::Interval,
	);
	let c2 = c.clone();
	let t2 = t.clone();
	let (tx, rx) = std::sync::mpsc::channel();
	let h = std::thread::spawn(move || {
		let r = c2.reserve(&t2[..2000]).unwrap();
		tx.send(()).unwrap();
		std::thread::sleep(std::time::Duration::from_millis(200));
		c2.insert(t2[..2000].to_vec(), state(2000, 1, 64).into(), Class::Turn);
		drop(r);
	});
	rx.recv().unwrap();
	assert_eq!(c.lookup(&t).unwrap().len, 2000);
	h.join().unwrap();
	// a reservation for another prompt does not hold a lookup up
	let _r = c.reserve(&toks(50, 7)).unwrap();
	let t0 = std::time::Instant::now();
	assert_eq!(c.lookup(&t).unwrap().len, 2000);
	assert!(t0.elapsed().as_millis() < 100);
}
