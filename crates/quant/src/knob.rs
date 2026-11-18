//! Per-thread overrides of `QWEN_*` variables (a request's profile).
//!
//! The decode knobs of the speculative paths (draft counts, cut-offs, the
//! context-draft match lengths, Flash-Next's MTP cost model) are read with
//! [`var`] where they are used, every step: a value set for the current
//! thread by a [`Scope`] wins over the process environment. The server
//! opens one per request with the values its profile gives that request's
//! mode and context depth (`server::profile`), and `--tune-profile` opens
//! one per trial. A request's decode loop runs on the thread that handles
//! it, so two requests never see each other's values.
//!
//! A variable the user set in the shell before [`freeze_user_env`] wins
//! over a scope: it is a deliberate override of a profile file. Variables
//! the process sets itself afterwards (a profile's load-time section) do
//! not.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::OnceLock;

thread_local! {
	static OVERLAY: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
}

static USER_ENV: OnceLock<HashSet<String>> = OnceLock::new();

/// Record which `QWEN_*` variables the environment holds now: from here on
/// they beat any [`Scope`]. Call once at startup, before setting any (the
/// server; `--tune-profile` does not call it, so its trials win).
pub fn freeze_user_env() {
	USER_ENV.get_or_init(|| {
		std::env::vars()
			.map(|(k, _)| k)
			.filter(|k| k.starts_with("QWEN_"))
			.collect()
	});
}

/// Whether the user set `name` before [`freeze_user_env`].
pub fn user_set(name: &str) -> bool {
	USER_ENV.get().is_some_and(|u| u.contains(name))
}

/// The value of `name`: the innermost [`Scope`]'s on this thread, else the
/// environment's; unless the user set it (see [`freeze_user_env`]).
pub fn var(name: &str) -> Option<String> {
	let scoped = || {
		OVERLAY.with(|o| {
			o.borrow()
				.iter()
				.rev()
				.find(|(k, _)| k == name)
				.map(|(_, v)| v.clone())
		})
	};
	let env = || std::env::var(name).ok();
	if user_set(name) {
		env().or_else(scoped)
	} else {
		scoped().or_else(env)
	}
}

/// [`var`] parsed; `None` when unset or unparsable.
pub fn parse<T: std::str::FromStr>(name: &str) -> Option<T> {
	var(name).and_then(|v| v.trim().parse().ok())
}

/// Values of `QWEN_*` variables for this thread until dropped.
pub struct Scope {
	n: usize,
}

impl Scope {
	pub fn new<K: AsRef<str>, V: AsRef<str>>(vals: &[(K, V)]) -> Scope {
		OVERLAY.with(|o| {
			let mut o = o.borrow_mut();
			o.extend(
				vals.iter()
					.map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string())),
			);
		});
		Scope { n: vals.len() }
	}
}

impl Drop for Scope {
	fn drop(&mut self) {
		OVERLAY.with(|o| {
			let mut o = o.borrow_mut();
			let keep = o.len() - self.n;
			o.truncate(keep);
		});
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn scopes_nest_and_unwind() {
		let k = "QWEN_KNOB_TEST_UNSET";
		assert_eq!(var(k), None);
		{
			let _a = Scope::new(&[(k, "1")]);
			assert_eq!(parse::<u32>(k), Some(1));
			{
				let _b = Scope::new(&[(k, "2")]);
				assert_eq!(var(k).as_deref(), Some("2"));
			}
			assert_eq!(var(k).as_deref(), Some("1"));
			// other threads do not see it
			std::thread::spawn(move || assert_eq!(var(k), None))
				.join()
				.unwrap();
		}
		assert_eq!(var(k), None);
	}
}
