//! Process CPU time, for the `[eff]` line: tok/s alone cannot tell whether a
//! change kept its speed by burning more cores, which is what efficiency work
//! is judged on.

/// User + kernel CPU seconds this process has used so far, over all threads.
#[cfg(windows)]
pub fn process_cpu_secs() -> Option<f64> {
	#[link(name = "kernel32")]
	extern "system" {
		fn GetCurrentProcess() -> isize;
		fn GetProcessTimes(
			h: isize,
			create: *mut u64,
			exit: *mut u64,
			kernel: *mut u64,
			user: *mut u64,
		) -> i32;
	}
	let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
	// FILETIME is two u32 halves, little-endian: the same bytes as a u64.
	let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
	(ok != 0).then(|| (k + u) as f64 * 100e-9)
}

#[cfg(not(windows))]
pub fn process_cpu_secs() -> Option<f64> {
	std::fs::read_to_string("/proc/self/stat")
		.ok()
		.and_then(|s| {
			// Fields after the `(comm)`: utime and stime are 14 and 15 overall.
			let rest = &s[s.rfind(')')? + 2..];
			let f: Vec<&str> = rest.split_whitespace().collect();
			let ticks: f64 = f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?;
			Some(ticks / 100.0)
		})
}

/// What a phase used: process CPU seconds and joules per source
/// ([`crate::energy`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
	pub cpu: Option<f64>,
	pub joules: crate::energy::Joules,
}

/// A wall-clock + CPU-time + energy stopwatch.
pub struct Meter {
	wall: std::time::Instant,
	cpu: Option<f64>,
	joules: crate::energy::Joules,
}

impl Meter {
	pub fn start() -> Self {
		// Energy first: its first read opens the counters (tens of ms).
		let joules = crate::energy::now();
		Meter {
			wall: std::time::Instant::now(),
			cpu: process_cpu_secs(),
			joules,
		}
	}

	/// (wall seconds, usage) since `start`.
	pub fn read(&self) -> (f64, Usage) {
		let w = self.wall.elapsed().as_secs_f64();
		let cpu = self.cpu.zip(process_cpu_secs()).map(|(a, b)| b - a);
		let joules = crate::energy::now().since(&self.joules);
		(w, Usage { cpu, joules })
	}
}

/// `"<cpu> CPU-s, <cores> cores busy, <ms> CPU-ms/token; <J> J (pkg <W> W
/// = cores + igpu + rest, gpu <W> W), <tok/J> tok/J"` for one phase. RAPL
/// and NVML count the whole machine, not this process: other load counts too.
pub fn describe(wall: f64, u: Usage, tokens: usize) -> String {
	let mut s = match u.cpu {
		Some(c) if wall > 0.0 && tokens > 0 => format!(
			"{c:.2} CPU-s, {:.1} cores busy, {:.1} CPU-ms/token",
			c / wall,
			c * 1000.0 / tokens as f64
		),
		_ => "n/a".into(),
	};
	let j = u.joules;
	if let Some(t) = j.total().filter(|_| wall > 0.0 && tokens > 0) {
		let w = |x: Option<f64>| x.map_or("-".into(), |x| format!("{:.1}", x / wall));
		s += &format!(
			"; {t:.1} J, {:.2} tok/J (pkg {} W = cores {} + igpu {} + rest, gpu {} W)",
			tokens as f64 / t.max(1e-9),
			w(j.pkg),
			w(j.cores),
			w(j.igpu),
			w(j.gpu)
		);
	}
	s
}
