//! Console progress bar for a prefill (`cli --progress`, or `QWEN_PROGRESS=1`).
//!
//! Off by default. A prefill opens a bar with [`Prefill::start`] and reports
//! into it with [`chunk`] (the span of positions now running) and [`part`]
//! (how far into that span it is, e.g. layers done); both are no-ops when no
//! bar is open, so the reporting sites cost nothing otherwise. The state is
//! per thread: a prefill's reports come from the thread that runs it.
//!
//! A bar started while one is open on the same thread is **nested**: it
//! continues the outer bar where it stands instead of drawing its own. The
//! server opens one bar per request and splits the prefill at its cache
//! checkpoints into several calls, which then fill that one bar.
//!
//! The line is redrawn at most once per second (the first time one second
//! after the start), on stderr: in place with `\r` on a terminal, one line per
//! update otherwise. A prefill that finishes within the first second draws
//! nothing.
//!
//! A decode shows a [`Decode`] line instead: tokens so far, tok/s, elapsed
//! time and, on a drafting engine, tokens per step (decode has no fixed end,
//! so no fill and no ETA). Where the answer streams to the same console
//! (`--generate`, `--interactive`) it goes to the console window's title
//! ([`Show::Title`], restored afterwards), elsewhere (`--serve`) to stderr
//! like the prefill bar.
//!
//! A job of named stages (image generation: loads, text encoder, denoising
//! steps, VAE) shows a [`Stages`] line on stderr: the stage, its steps done,
//! the elapsed time and the stage's ETA, redrawn every second on a terminal.
//!
//! Times of a minute or more read `4 min 05 s` (`1 h 02 min 05 s`):
//! [`fmt_secs`].

use std::cell::RefCell;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);

const INTERVAL: Duration = Duration::from_secs(1);
const WIDTH: usize = 30;

/// Turn the bar on for the rest of the process (`cli --progress`).
pub fn set_enabled(on: bool) {
	ENABLED.store(on, Ordering::Relaxed);
}

/// `--progress` was given, or `QWEN_PROGRESS` is set to anything but `0`.
pub fn enabled() -> bool {
	static ENV: OnceLock<bool> = OnceLock::new();
	ENABLED.load(Ordering::Relaxed)
		|| *ENV.get_or_init(|| std::env::var("QWEN_PROGRESS").is_ok_and(|v| v != "0"))
}

/// `secs` as `12.3 s` (`prec` decimals) below a minute, else in whole
/// seconds as `4 min 05 s` or `1 h 02 min 05 s`.
pub fn fmt_secs(secs: f64, prec: usize) -> String {
	let secs = secs.max(0.0);
	let p = 10f64.powi(prec as i32);
	if (secs * p).round() / p < 60.0 {
		return format!("{secs:.prec$} s");
	}
	let t = secs.round() as u64;
	let (h, m, s) = (t / 3600, t / 60 % 60, t % 60);
	if h > 0 {
		format!("{h} h {m:02} min {s:02} s")
	} else {
		format!("{m} min {s:02} s")
	}
}

/// A bar line is on the terminal without its newline (drawn in place).
static PENDING: AtomicBool = AtomicBool::new(false);

/// Characters of the line last drawn in place: the next draw (or the erase)
/// covers that many, so a line that got shorter (`ETA 1 min 00 s` ->
/// `ETA 30 s`) leaves no tail of the longer one behind (no ANSI erase, for
/// older consoles).
static DRAWN: AtomicUsize = AtomicUsize::new(0);

/// Narrowest a bar's `[###---]` run is shrunk to by [`fit`].
const MIN_RUN: usize = 10;

/// `line` in at most `max` characters: the first `[###---]` run is shrunk
/// first (down to [`MIN_RUN`], its fill kept in proportion), then the end
/// is cut. A line drawn in place must not wrap: `\r` returns only to the
/// start of the last row, so every redraw would leave a row behind.
fn fit(line: &str, max: usize) -> String {
	let chars: Vec<char> = line.chars().collect();
	let mut over = chars.len().saturating_sub(max);
	if over == 0 {
		return line.to_string();
	}
	let mut out: Vec<char> = chars.clone();
	let run = chars.iter().position(|&c| c == '[').and_then(|o| {
		let n = chars[o + 1..]
			.iter()
			.take_while(|&&c| c == '#' || c == '-')
			.count();
		(n > MIN_RUN && chars.get(o + 1 + n) == Some(&']')).then_some((o + 1, n))
	});
	if let Some((at, n)) = run {
		let m = n - over.min(n - MIN_RUN);
		let filled = chars[at..at + n].iter().filter(|&&c| c == '#').count();
		let f = (filled * m).div_ceil(n).min(m);
		let f = if filled < n { f.min(m - 1) } else { f };
		let bar = std::iter::repeat_n('#', f).chain(std::iter::repeat_n('-', m - f));
		out.splice(at..at + n, bar);
		over -= n - m;
	}
	out.truncate(out.len() - over);
	out.into_iter().collect()
}

/// Columns of the console stderr is drawn on, if known.
#[cfg(windows)]
fn columns() -> Option<usize> {
	#[repr(C)]
	struct Info {
		size: [i16; 2],
		cursor: [i16; 2],
		attr: u16,
		window: [i16; 4],
		max: [i16; 2],
	}
	#[link(name = "kernel32")]
	extern "system" {
		fn GetStdHandle(n: u32) -> *mut core::ffi::c_void;
		fn GetConsoleScreenBufferInfo(h: *mut core::ffi::c_void, info: *mut Info) -> i32;
	}
	const STD_ERROR_HANDLE: u32 = -12i32 as u32;
	let mut info = Info {
		size: [0; 2],
		cursor: [0; 2],
		attr: 0,
		window: [0; 4],
		max: [0; 2],
	};
	// SAFETY: `info` is a CONSOLE_SCREEN_BUFFER_INFO; a bad handle fails the call
	let ok = unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) };
	let w = info.window[2] as i32 - info.window[0] as i32 + 1;
	(ok != 0 && w > 0).then_some(w as usize)
}

#[cfg(target_os = "linux")]
fn columns() -> Option<usize> {
	#[repr(C)]
	struct Winsize {
		row: u16,
		col: u16,
		x: u16,
		y: u16,
	}
	extern "C" {
		fn ioctl(fd: i32, req: core::ffi::c_ulong, ...) -> i32;
	}
	const TIOCGWINSZ: core::ffi::c_ulong = 0x5413;
	let mut ws = Winsize {
		row: 0,
		col: 0,
		x: 0,
		y: 0,
	};
	// SAFETY: TIOCGWINSZ fills a `struct winsize`
	let ok = unsafe { ioctl(2, TIOCGWINSZ, &mut ws as *mut Winsize) };
	(ok == 0 && ws.col > 0).then_some(ws.col as usize)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn columns() -> Option<usize> {
	std::env::var("COLUMNS")
		.ok()?
		.parse()
		.ok()
		.filter(|&c| c > 0)
}

/// Characters a line drawn in place may take: one less than the console's
/// columns (a full row leaves the cursor in a pending wrap on some
/// consoles), 80 columns when unknown. Read per draw: the window may be
/// resized.
fn room() -> usize {
	columns().unwrap_or(80).saturating_sub(1).max(1)
}

/// Write a progress line to stderr: in place on a terminal, else one line.
fn emit(line: &str, tty: bool, last: bool) {
	let mut e = std::io::stderr().lock();
	let _ = if tty {
		PENDING.store(!last, Ordering::Relaxed);
		let room = room();
		let line = fit(line, room);
		let n = line.chars().count();
		let w = DRAWN
			.swap(if last { 0 } else { n }, Ordering::Relaxed)
			.min(room);
		write!(e, "\r{line:<w$}{}", if last { "\n" } else { "" })
	} else {
		writeln!(e, "{line}")
	};
	let _ = e.flush();
}

fn clear_in(e: &mut impl Write) {
	if PENDING.swap(false, Ordering::Relaxed) {
		let w = DRAWN.swap(0, Ordering::Relaxed).min(room());
		let _ = write!(e, "\r{:w$}\r", "");
	}
}

/// Erase a bar line left in place on the terminal (the next bar draw puts
/// it back), so that what is printed next starts on a clean line.
pub fn clear() {
	let mut e = std::io::stderr().lock();
	clear_in(&mut e);
	let _ = e.flush();
}

/// A log line on stderr that does not run into a bar drawn in place: the
/// bar is erased first and redrawn by its next update. Use it for lines
/// printed while a bar may be open, and for a summary that replaces a bar's
/// final line ([`Prefill::summarized`], [`Decode::summarized`], [`Stages`]).
pub fn line(s: &str) {
	let mut e = std::io::stderr().lock();
	clear_in(&mut e);
	let _ = writeln!(e, "{s}");
	let _ = e.flush();
}

struct Bar {
	total: usize,
	/// Positions finished before the running nested prefill began.
	off: usize,
	/// The running span, relative to `off`, and the fraction of it done.
	base: usize,
	len: usize,
	frac: f64,
	start: Instant,
	last: Option<Instant>,
	tty: bool,
	/// The caller prints a summary instead of the final line.
	summarized: bool,
}

impl Bar {
	fn done(&self) -> f64 {
		((self.off + self.base) as f64 + self.len as f64 * self.frac).min(self.total as f64)
	}

	fn draw(&mut self, last: bool) {
		let now = Instant::now();
		if !last {
			let due = self.last.map_or(self.start + INTERVAL, |t| t + INTERVAL);
			if now < due {
				return;
			}
		} else if self.last.is_none() {
			return;
		}
		self.last = Some(now);
		let done = self.done();
		let total = self.total.max(1) as f64;
		let secs = now.duration_since(self.start).as_secs_f64();
		let rate = if secs > 0.0 { done / secs } else { 0.0 };
		let fill = ((done / total) * WIDTH as f64) as usize;
		let tail = if last {
			fmt_secs(secs, 1)
		} else if rate > 0.0 {
			format!("ETA {}", fmt_secs((self.total as f64 - done) / rate, 0))
		} else {
			String::new()
		};
		let line = format!(
			"prefill [{}{}] {:>w$}/{} tokens {:>3.0}% {:>7.1} tok/s  {tail}",
			"#".repeat(fill.min(WIDTH)),
			"-".repeat(WIDTH - fill.min(WIDTH)),
			done as usize,
			self.total,
			done / total * 100.0,
			rate,
			w = self.total.to_string().len(),
		);
		emit(&line, self.tty, last);
	}
}

thread_local! {
	static BAR: RefCell<Option<Bar>> = const { RefCell::new(None) };
}

/// An open bar (or a nested prefill inside one); the line is finished when
/// the owning handle is dropped.
#[must_use]
pub struct Prefill {
	own: bool,
	active: bool,
}

impl Prefill {
	/// Open a bar over `total` positions, or continue the one open on this
	/// thread. Inactive when the bar is disabled or `total` is 0.
	pub fn start(total: usize) -> Prefill {
		Prefill::start_at(total, Instant::now())
	}

	/// [`Prefill::start`] with the clock started at `t0` (a new bar only): a
	/// caller whose own timing began before the bar's total was known, so the
	/// bar's rate is the one it reports.
	pub fn start_at(total: usize, t0: Instant) -> Prefill {
		if total == 0 || !enabled() {
			return Prefill {
				own: false,
				active: false,
			};
		}
		BAR.with(|b| {
			let mut b = b.borrow_mut();
			match b.as_mut() {
				Some(bar) => {
					bar.off = bar.done() as usize;
					(bar.base, bar.len, bar.frac) = (0, 0, 0.0);
					Prefill {
						own: false,
						active: true,
					}
				}
				None => {
					*b = Some(Bar {
						total,
						off: 0,
						base: 0,
						len: 0,
						frac: 0.0,
						start: t0,
						last: None,
						tty: std::io::stderr().is_terminal(),
						summarized: false,
					});
					Prefill {
						own: true,
						active: true,
					}
				}
			}
		})
	}

	/// The prefill is complete: the final line shows every position done.
	pub fn done(&self) {
		if self.own {
			with(|bar| bar.off = bar.total);
		}
	}

	/// The caller prints its own summary line (through [`line`]) after the
	/// prefill: the bar draws no final line and erases itself instead.
	/// Only an owning bar's choice counts.
	pub fn summarized(&self) {
		if self.own {
			with(|bar| bar.summarized = true);
		}
	}
}

impl Drop for Prefill {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		BAR.with(|b| {
			let mut b = b.borrow_mut();
			if self.own {
				if let Some(mut bar) = b.take() {
					if bar.summarized {
						clear();
					} else {
						bar.draw(true);
					}
				}
			} else if let Some(bar) = b.as_mut() {
				bar.off = bar.done() as usize;
				(bar.base, bar.len, bar.frac) = (0, 0, 0.0);
			}
		});
	}
}

fn with(f: impl FnOnce(&mut Bar)) {
	BAR.with(|b| {
		if let Some(bar) = b.borrow_mut().as_mut() {
			f(bar);
		}
	});
}

/// Positions `[base, base + len)` of the current prefill are running now
/// (`base` counted from that prefill's first position).
pub fn chunk(base: usize, len: usize) {
	with(|bar| {
		(bar.base, bar.len, bar.frac) = (base, len, 0.0);
		bar.draw(false);
	});
}

/// `num / den` of the running span is done.
pub fn part(num: usize, den: usize) {
	with(|bar| {
		bar.frac = num as f64 / den.max(1) as f64;
		bar.draw(false);
	});
}

/// Where a [`Decode`] line goes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Show {
	/// stderr, in place on a terminal (the console carries nothing else).
	Stderr,
	/// The console window's title: the answer is streaming to the console.
	Title,
}

/// Progress of one decode; the last line is drawn when it is dropped (a
/// decode that finishes within the first second draws nothing).
#[must_use]
pub struct Decode {
	show: Show,
	tokens: usize,
	steps: usize,
	start: Instant,
	last: Option<Instant>,
	tty: bool,
	/// The title before the first draw, restored on drop ([`Show::Title`]).
	saved: Option<title::Saved>,
	active: bool,
	summarized: bool,
}

impl Decode {
	/// Inactive (every call a no-op) unless the bar is enabled.
	pub fn start(show: Show) -> Decode {
		Decode {
			show,
			tokens: 0,
			steps: 0,
			start: Instant::now(),
			last: None,
			tty: std::io::stderr().is_terminal(),
			saved: None,
			active: enabled(),
			summarized: false,
		}
	}

	/// The caller prints its own summary line (through [`line`]) after the
	/// decode: no final line, the stderr line is erased instead.
	pub fn summarized(&mut self) {
		self.summarized = true;
	}

	/// `n` more tokens produced.
	pub fn tokens(&mut self, n: usize) {
		if self.active {
			self.tokens += n;
			self.draw(false);
		}
	}

	/// One more model step (a verify step of several positions counts once):
	/// only an engine that drafts reports these, and the line then shows
	/// tokens per step.
	pub fn step(&mut self) {
		self.steps += 1;
	}

	fn line(&self, secs: f64) -> String {
		let rate = if secs > 0.0 {
			self.tokens as f64 / secs
		} else {
			0.0
		};
		let per_step = if self.steps > 0 {
			format!("  {:.2} tok/step", self.tokens as f64 / self.steps as f64)
		} else {
			String::new()
		};
		format!(
			"decode {} tokens  {rate:.1} tok/s  {}{per_step}",
			self.tokens,
			fmt_secs(secs, 1)
		)
	}

	fn draw(&mut self, last: bool) {
		let now = Instant::now();
		if !last {
			let due = self.last.map_or(self.start + INTERVAL, |t| t + INTERVAL);
			if now < due {
				return;
			}
		} else if self.last.is_none() {
			return;
		}
		self.last = Some(now);
		let line = self.line(now.duration_since(self.start).as_secs_f64());
		match self.show {
			Show::Title => {
				if self.saved.is_none() {
					self.saved = Some(title::save());
				}
				if !last {
					title::set(&line);
				}
			}
			Show::Stderr => emit(&line, self.tty, last),
		}
	}
}

impl Drop for Decode {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		if !self.summarized {
			self.draw(true);
		} else if self.show == Show::Stderr {
			clear();
		}
		if let Some(s) = self.saved.take() {
			title::restore(s);
		}
	}
}

struct StageState {
	label: String,
	start: Instant,
	stage: String,
	done: usize,
	total: usize,
	/// When the stage began, and when its last step finished.
	stage_t: Instant,
	step_t: Instant,
	/// Seconds per step of the stage so far, once a step has finished.
	per_step: Option<f64>,
	last: Option<Instant>,
	tty: bool,
	stop: bool,
}

impl StageState {
	fn line(&self, now: Instant) -> String {
		let el = fmt_secs(now.duration_since(self.start).as_secs_f64(), 1);
		let mut l = format!("{}: {}", self.label, self.stage);
		if let Some(fill) = (self.done * WIDTH).checked_div(self.total) {
			let fill = fill.min(WIDTH);
			l += &format!(
				" [{}{}] {}/{}",
				"#".repeat(fill),
				"-".repeat(WIDTH - fill),
				self.done,
				self.total
			);
		}
		l += &format!("  {el}");
		if let (Some(ps), true) = (self.per_step, self.total > self.done) {
			let rem = ps * (self.total - self.done) as f64
				- now.duration_since(self.step_t).as_secs_f64();
			l += &format!("  ETA {}", fmt_secs(rem, 0));
		}
		l
	}

	fn draw(&mut self) {
		let now = Instant::now();
		self.last = Some(now);
		emit(&self.line(now), self.tty, false);
	}
}

/// Progress of a job of named stages (image generation) on stderr: the
/// stage, its steps done, the elapsed time and the stage's ETA (from its
/// finished steps). On a terminal the line is redrawn in place every second
/// by a thread of its own; elsewhere one line per stage and at most one a
/// second within one. No final line: the caller prints a summary (through
/// [`line`]) once it is dropped, which erases the bar on a terminal.
#[must_use]
pub struct Stages {
	inner: Option<(
		Arc<(Mutex<StageState>, Condvar)>,
		Option<std::thread::JoinHandle<()>>,
	)>,
}

impl Stages {
	/// Inactive (every call a no-op) unless the bar is enabled.
	pub fn start(label: &str) -> Stages {
		if !enabled() {
			return Stages { inner: None };
		}
		let now = Instant::now();
		let tty = std::io::stderr().is_terminal();
		let st = Arc::new((
			Mutex::new(StageState {
				label: label.to_string(),
				start: now,
				stage: String::new(),
				done: 0,
				total: 0,
				stage_t: now,
				step_t: now,
				per_step: None,
				last: None,
				tty,
				stop: false,
			}),
			Condvar::new(),
		));
		let tick = tty.then(|| {
			let st = st.clone();
			std::thread::spawn(move || {
				let (m, cv) = &*st;
				let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
				loop {
					g = cv
						.wait_timeout(g, INTERVAL)
						.unwrap_or_else(|e| e.into_inner())
						.0;
					if g.stop {
						return;
					}
					if !g.stage.is_empty() && g.last.is_none_or(|t| t.elapsed() >= INTERVAL) {
						g.draw();
					}
				}
			})
		});
		Stages {
			inner: Some((st, tick)),
		}
	}

	/// Stage `stage` has `done` of `total` steps finished (`total` 0: a
	/// stage without steps).
	pub fn update(&self, stage: &str, done: usize, total: usize) {
		let Some((st, _)) = &self.inner else {
			return;
		};
		let mut g = st.0.lock().unwrap_or_else(|e| e.into_inner());
		let now = Instant::now();
		let new = g.stage != stage;
		if new {
			g.stage = stage.to_string();
			(g.stage_t, g.step_t, g.per_step, g.done) = (now, now, None, 0);
		}
		g.total = total;
		if done != g.done {
			g.done = done;
			g.step_t = now;
			if done > 0 {
				g.per_step = Some(now.duration_since(g.stage_t).as_secs_f64() / done as f64);
			}
		}
		if new || g.last.is_none_or(|t| now.duration_since(t) >= INTERVAL) {
			g.draw();
		}
	}
}

impl Drop for Stages {
	fn drop(&mut self) {
		let Some((st, tick)) = self.inner.take() else {
			return;
		};
		st.0.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
		st.1.notify_all();
		if let Some(t) = tick {
			let _ = t.join();
		}
		clear();
	}
}

/// The console window's title.
#[cfg(windows)]
mod title {
	#[link(name = "kernel32")]
	extern "system" {
		fn GetConsoleTitleW(buf: *mut u16, size: u32) -> u32;
		fn SetConsoleTitleW(title: *const u16) -> i32;
	}

	pub struct Saved(Vec<u16>);

	pub fn save() -> Saved {
		let mut buf = vec![0u16; 1024];
		// SAFETY: the buffer holds `size` u16s
		let n = unsafe { GetConsoleTitleW(buf.as_mut_ptr(), buf.len() as u32) } as usize;
		buf.truncate(n.min(buf.len() - 1));
		buf.push(0);
		Saved(buf)
	}

	pub fn set(s: &str) {
		let w: Vec<u16> = s.encode_utf16().chain(std::iter::once(0)).collect();
		// SAFETY: NUL-terminated UTF-16
		unsafe { SetConsoleTitleW(w.as_ptr()) };
	}

	pub fn restore(s: Saved) {
		// SAFETY: NUL-terminated UTF-16 (`save`)
		unsafe { SetConsoleTitleW(s.0.as_ptr()) };
	}
}

/// The terminal's title by xterm escapes on stderr (push / set / pop), when
/// stderr is a terminal.
#[cfg(not(windows))]
mod title {
	use std::io::{IsTerminal, Write};

	pub struct Saved;

	fn esc(s: &str) {
		let mut e = std::io::stderr().lock();
		if e.is_terminal() {
			let _ = write!(e, "{s}");
			let _ = e.flush();
		}
	}

	pub fn save() -> Saved {
		esc("\x1b[22;0t");
		Saved
	}

	pub fn set(s: &str) {
		esc(&format!("\x1b]0;{s}\x07"));
	}

	pub fn restore(_: Saved) {
		esc("\x1b[23;0t");
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn done() -> f64 {
		BAR.with(|b| b.borrow().as_ref().map_or(-1.0, |b| b.done()))
	}

	#[test]
	fn lines_fit_the_console() {
		let l = "prefill [###########################---] 55381/60564 tokens  91%   172.0 tok/s  ETA 1 min 00 s";
		assert_eq!(fit(l, 200), l);
		let f = fit(l, 79);
		assert_eq!(f.chars().count(), 79);
		assert!(f.starts_with("prefill [##############-] 55381/"), "{f}");
		assert!(f.ends_with("ETA 1 min 00 s"), "{f}");
		// the run stops at MIN_RUN, then the end is cut
		let f = fit(l, 40);
		assert_eq!(f, "prefill [#########-] 55381/60564 tokens ");
		// a full run stays full, an unfinished one keeps a '-'
		assert_eq!(fit("[####] x", 6), "[####]");
		assert_eq!(
			fit(&format!("[{}] abc", "#".repeat(30)), 20),
			format!("[{}] abc", "#".repeat(14))
		);
		assert_eq!(
			fit(&format!("[{}-] abc", "#".repeat(29)), 20),
			format!("[{}-] abc", "#".repeat(13))
		);
	}

	#[test]
	fn times_split_into_minutes() {
		assert_eq!(fmt_secs(5.25, 1), "5.2 s");
		assert_eq!(fmt_secs(59.4, 0), "59 s");
		assert_eq!(fmt_secs(59.6, 0), "1 min 00 s");
		assert_eq!(fmt_secs(59.96, 1), "1 min 00 s");
		assert_eq!(fmt_secs(125.0, 1), "2 min 05 s");
		assert_eq!(fmt_secs(3725.0, 0), "1 h 02 min 05 s");
		assert_eq!(fmt_secs(-3.0, 0), "0 s");
	}

	#[test]
	fn nested_prefills_fill_one_bar() {
		set_enabled(true);
		let outer = Prefill::start(100);
		{
			let _a = Prefill::start(40);
			chunk(0, 32);
			part(1, 2);
			assert_eq!(done(), 16.0);
			chunk(32, 8);
			part(1, 1);
		}
		assert_eq!(done(), 40.0);
		{
			let _b = Prefill::start(60);
			chunk(0, 60);
			part(1, 4);
			assert_eq!(done(), 55.0);
		}
		outer.done();
		assert_eq!(done(), 100.0);
		drop(outer);
		assert_eq!(done(), -1.0);
		// with no bar open, reports are no-ops
		part(1, 1);
		assert_eq!(done(), -1.0);
	}
}
