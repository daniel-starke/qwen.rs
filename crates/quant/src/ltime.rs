//! Load-time spans (`QWEN_LOAD_TIMING=1`).
//!
//! A model load runs pieces on several threads at once, so a line per piece
//! with its start and end on one clock (seconds since [`start`], called
//! first thing in `main`) shows what overlaps and what the critical path is;
//! a per-piece duration alone does not. [`span`] returns a guard that prints
//! `[load]  START -> END  (DUR s)  WHAT` when dropped. Off, a span costs one
//! `OnceLock` read.

use std::sync::OnceLock;
use std::time::Instant;

static T0: OnceLock<Instant> = OnceLock::new();
static ON: OnceLock<bool> = OnceLock::new();

/// Fix the clock's zero (the process's start, as near as `main` gets).
pub fn start() {
	T0.get_or_init(Instant::now);
}

pub fn on() -> bool {
	*ON.get_or_init(|| std::env::var("QWEN_LOAD_TIMING").is_ok_and(|v| v != "0"))
}

/// Seconds since [`start`].
pub fn now() -> f64 {
	T0.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub struct Span {
	what: Option<String>,
	t: f64,
}

impl Span {
	/// End the span now (else at drop).
	pub fn end(mut self) {
		self.print();
	}

	fn print(&mut self) {
		if let Some(w) = self.what.take() {
			let e = now();
			eprintln!(
				"[load] {:6.2} -> {:6.2}  ({:5.2} s)  {w}",
				self.t,
				e,
				e - self.t
			);
		}
	}
}

impl Drop for Span {
	fn drop(&mut self) {
		self.print();
	}
}

/// A span named `what` from now until the guard drops.
pub fn span(what: impl FnOnce() -> String) -> Span {
	Span {
		what: on().then(what),
		t: now(),
	}
}
