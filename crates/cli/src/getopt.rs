//! # Command-line argument parser.
//!
//! - **Author:** Daniel Starke
//! - **Copyright:** Copyright 2026 Daniel Starke
//! - **Date:** 2026-02-24
//! - **Version:** 2026-03-12
//!
//! Standalone, general-purpose module: the full public API is kept even where
//! this binary happens not to call every entry point.

#![allow(dead_code)]

use super::tst::*;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
	None,
	Optional,
	Required,
}

/// Error types that can occur during command-line parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
	/// Command requires an option but none was provided.
	MissingArgument(String),
	/// Unknown command was provided.
	InvalidCommand(String),
}

#[derive(Debug, Clone)]
pub struct Config {
	pub short: char,
	pub long: String,
	pub arg: Arg,
}

type Args<'a> = Box<dyn Iterator<Item = String> + 'a>;

pub struct Parser<'a> {
	opts: Vec<Config>,
	args: Args<'a>,
	syms: SymbolMap<usize>,
	cur: Option<String>,
	short: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum Command<'a> {
	Opt(&'a Config, Option<String>),
	Rest(Vec<String>),
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Error::MissingArgument(cmd) => {
				write!(f, "Missing required argument for command {}.", cmd)
			}
			Error::InvalidCommand(cmd) => write!(f, "Invalid command {}.", cmd),
		}
	}
}

impl std::error::Error for Error {}

impl Config {
	pub fn new(short: char, long: &str, arg: Arg) -> Self {
		Config {
			short,
			long: long.to_string(),
			arg,
		}
	}

	pub fn new_short(short: char, arg: Arg) -> Self {
		Config {
			short,
			long: "".to_owned(),
			arg,
		}
	}

	pub fn new_long(long: &str, arg: Arg) -> Self {
		Config {
			short: '\0',
			long: long.to_string(),
			arg,
		}
	}
}

impl<'a> Parser<'a> {
	pub fn new(opts: Vec<Config>, args: impl Iterator<Item = impl Into<String> + 'a> + 'a) -> Self {
		let mut res = Parser {
			opts,
			args: Box::new(args.map(Into::into)),
			syms: Default::default(),
			cur: None,
			short: None,
		};
		for (i, o) in res.opts.iter().enumerate() {
			if o.short != '\0' {
				let opt = format!("-{}", o.short);
				res.syms.add(&opt, i);
			}
			if !o.long.is_empty() {
				let opt = format!("--{}", o.long);
				res.syms.add(&opt, i);
			}
		}
		// add "--" separator for trailing positional arguments
		let last = res.opts.len();
		res.opts.push(Config::new('-', "--", Arg::None));
		res.syms.add("--", last);
		res
	}

	/// Parses a single command.
	pub fn next_opt(&mut self) -> Result<Command<'_>, Error> {
		// We come from `str` or ASCII, hence, all `std::str::from_utf8_unchecked()` calls are guaranteed to succeed.
		if let Some(i) = self.short {
			// check subsequently combined short command
			self.short = Some(i + 1);
			let bytes: [u8; 2];
			let byte = unsafe {
				self.cur.as_ref().unwrap_unchecked(/* cur is guaranteed to be Some here */).as_bytes().get(i)
			};
			match byte {
				Some(&b) => {
					bytes = [b'-', b];
					#[cfg(test)]
					unsafe {
						println!(" => {}", str::from_utf8_unchecked(bytes.as_slice()));
					}
					if b == b'-' {
						unsafe {
							return Err(Error::InvalidCommand(
								std::str::from_utf8_unchecked(bytes.as_slice()).to_string(),
							));
						}
					}
				}
				None => {
					self.short = None;
					#[cfg(test)]
					println!(" => none");
					return self.next_opt();
				}
			}
			// match configured command
			let cmd;
			match self.syms.find(bytes.as_slice()) {
				Some((matched_len, idx)) => {
					if bytes.len() == matched_len {
						cmd = &self.opts[idx];
					} else {
						#[cfg(test)]
						println!(" => size mismatch");
						unsafe {
							return Err(Error::InvalidCommand(
								std::str::from_utf8_unchecked(bytes.as_slice()).to_string(),
							));
						}
					}
				}
				None => {
					#[cfg(test)]
					println!(" => invalid");
					unsafe {
						return Err(Error::InvalidCommand(
							std::str::from_utf8_unchecked(bytes.as_slice()).to_string(),
						));
					}
				}
			}
			// check command parameter
			#[cfg(test)]
			println!("matched: {} ({:?})", cmd.short, cmd.arg);
			return Self::get_arg(cmd, bytes.as_slice(), &mut self.args);
		} else {
			// get next argument
			match self.args.next() {
				Some(x) => self.cur = Some(x),
				None => return Ok(Command::Rest(Default::default())),
			};
		}
		let opt = unsafe {
			self.cur.as_ref().unwrap_unchecked(/* cur is guaranteed to be Some here */).clone()
		};
		let bytes = opt.as_bytes();
		// start of positional arguments?
		if !bytes.starts_with(b"-") {
			let mut rest = vec![opt];
			rest.extend(self.args.by_ref());
			return Ok(Command::Rest(rest));
		}
		// check combined short flags
		if bytes.get(1) != Some(&b'-') {
			if bytes.len() == 1 {
				unsafe {
					return Err(Error::InvalidCommand(
						std::str::from_utf8_unchecked(bytes).to_string(),
					));
				}
			}
			#[cfg(test)]
			println!("short: {:?}", self.cur);
			self.short = Some(1);
			return self.next_opt();
		}
		// match configured command
		let cmd;
		match self.syms.find(bytes) {
			Some((matched_len, idx)) => {
				if bytes.len() == matched_len {
					cmd = &self.opts[idx];
				} else {
					unsafe {
						return Err(Error::InvalidCommand(
							std::str::from_utf8_unchecked(bytes).to_string(),
						));
					}
				}
			}
			None => unsafe {
				return Err(Error::InvalidCommand(
					std::str::from_utf8_unchecked(bytes).to_string(),
				));
			},
		}
		#[cfg(test)]
		println!("matched: {} ({:?})", cmd.long, cmd.arg);
		// start of positional arguments introduced by "--"?
		if cmd.short == '-' {
			return Ok(Command::Rest(self.args.by_ref().collect()));
		}
		// check command parameter
		Self::get_arg(cmd, bytes, &mut self.args)
	}

	/// Parses a single argument. Useful for multi-argument commands.
	pub fn next_arg(&mut self) -> Option<String> {
		self.args.next()
	}

	/// Returns the argument for the given command.
	fn get_arg<'b>(
		cmd: &'b Config,
		raw_cmd: &[u8],
		args: &mut Args<'a>,
	) -> Result<Command<'b>, Error> {
		match cmd.arg {
			Arg::None => Ok(Command::Opt(cmd, None)),
			Arg::Optional => Ok(Command::Opt(cmd, args.next())),
			Arg::Required => match args.next() {
				Some(x) => Ok(Command::Opt(cmd, Some(x))),
				None => unsafe {
					Err(Error::MissingArgument(
						std::str::from_utf8_unchecked(raw_cmd).to_string(),
					))
				},
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// empty / no-arg cases
	#[test]
	fn no_args_returns_rest_empty() {
		let args: &[&str] = &[];
		let mut p = Parser::new(vec![], args.iter().copied());
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	#[test]
	fn next_after_exhaustion_returns_rest() {
		let args: &[&str] = &[];
		let mut p = Parser::new(vec![], args.iter().copied());
		matches!(p.next_opt().unwrap(), Command::Rest(_));
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	// Arg::None
	#[test]
	fn short_flag() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["-v"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn long_flag() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["--verbose"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!("verbose", cfg.long),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn multiple_flags() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('q', "quiet", Arg::None),
		];
		let mut p = Parser::new(opts, vec!["-v", "-q"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('q', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	#[test]
	fn combined_flags() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('q', "quiet", Arg::None),
		];
		let mut p = Parser::new(opts, vec!["-vq"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('q', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	// Arg::Optional
	#[test]
	fn optional_arg_present() {
		let opts = vec![Config::new('o', "output", Arg::Optional)];
		let mut p = Parser::new(opts, vec!["-o", "file.txt"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('o', cfg.short);
				assert_eq!("file.txt", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn optional_arg_absent() {
		let opts = vec![Config::new('o', "output", Arg::Optional)];
		let mut p = Parser::new(opts, vec!["-o"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('o', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn optional_arg_long_present() {
		let opts = vec![Config::new('o', "output", Arg::Optional)];
		let mut p = Parser::new(opts, vec!["--output", "result.bin"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!("output", cfg.long);
				assert_eq!("result.bin", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn optional_arg_long_absent() {
		let opts = vec![Config::new('o', "output", Arg::Optional)];
		let mut p = Parser::new(opts, vec!["--output"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!("output", cfg.long),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn optional_arg_combined_present() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('o', "output", Arg::Optional),
		];
		let mut p = Parser::new(opts, vec!["-vo", "file.txt"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('o', cfg.short);
				assert_eq!("file.txt", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	#[test]
	fn optional_arg_combined_absent() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('o', "output", Arg::Optional),
		];
		let mut p = Parser::new(opts, vec!["-vo"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('o', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest"),
		}
	}

	// Arg::Required
	#[test]
	fn required_arg_present() {
		let opts = vec![Config::new('f', "file", Arg::Required)];
		let mut p = Parser::new(opts, vec!["-f", "data.csv"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('f', cfg.short);
				assert_eq!("data.csv", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn required_arg_missing() {
		let opts = vec![Config::new('f', "file", Arg::Required)];
		let mut p = Parser::new(opts, vec!["-f"].into_iter());
		assert_eq!(
			Error::MissingArgument("-f".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	#[test]
	fn required_arg_long_present() {
		let opts = vec![Config::new('f', "file", Arg::Required)];
		let mut p = Parser::new(opts, vec!["--file", "notes.txt"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!("file", cfg.long);
				assert_eq!("notes.txt", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn required_arg_long_missing() {
		let opts = vec![Config::new('f', "file", Arg::Required)];
		let mut p = Parser::new(opts, vec!["--file"].into_iter());
		assert_eq!(
			Error::MissingArgument("--file".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	#[test]
	fn required_arg_combined_present() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('f', "file", Arg::Required),
		];
		let mut p = Parser::new(opts, vec!["-vf", "data.csv"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('f', cfg.short);
				assert_eq!("data.csv", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn required_arg_combined_missing() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('f', "file", Arg::Required),
		];
		let mut p = Parser::new(opts, vec!["-vf"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		assert_eq!(
			Error::MissingArgument("-f".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	#[test]
	fn required_arg_combined_present_cmd() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('f', "file", Arg::Required),
		];
		let mut p = Parser::new(opts, vec!["-vf", "--command"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('f', cfg.short);
				assert_eq!("--command", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	#[test]
	fn required_arg_combined_multiple_present() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('f', "file", Arg::Required),
		];
		let mut p = Parser::new(opts, vec!["-vff", "file0.txt", "file1.txt"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('f', cfg.short);
				assert_eq!("file0.txt", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('f', cfg.short);
				assert_eq!("file1.txt", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
	}

	// "--" separator handling
	#[test]
	fn double_dash_with_arg() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["--", "positional", "..."].into_iter());
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert_eq!(vec!["positional", "..."], v),
			_ => panic!("expected Command::Rest for '--' separator"),
		}
	}

	#[test]
	fn single_dash_with_arg() {
		let mut p = Parser::new(vec![], vec!["-", "stdin", "more"].into_iter());
		assert_eq!(
			Error::InvalidCommand("-".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	#[test]
	fn double_dash_without_arg() {
		let mut p = Parser::new(vec![], vec!["--"].into_iter());
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!("expected Command::Rest for '--' with no args"),
		}
	}

	#[test]
	fn single_dash_in_combined() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["-v-", "stdin", "more"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		assert_eq!(
			Error::InvalidCommand("--".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	// multiple arguments
	#[test]
	fn required_two_args() {
		let opts = vec![Config::new('c', "convert", Arg::Required)];
		let mut p = Parser::new(opts, vec!["-c", "from", "to"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(val)) => {
				assert_eq!('c', cfg.short);
				assert_eq!("from", val);
			}
			_ => panic!("expected Command::Opt with value"),
		}
		match p.next_arg() {
			Some(val) => assert_eq!("to", val),
			_ => panic!("expected second value"),
		}
	}

	// positional arguments
	#[test]
	fn positional_arg() {
		let mut p = Parser::new(vec![], vec!["stdin", "more"].into_iter());
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert_eq!(vec!["stdin", "more"], v),
			_ => panic!("expected Command::Rest"),
		}
	}

	#[test]
	fn positional_arg_after_flag() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["-v", "stdin", "more"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!("verbose", cfg.long),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert_eq!(vec!["stdin", "more"], v),
			_ => panic!("expected Command::Rest"),
		}
	}

	// error cases
	#[test]
	fn invalid_command_short() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["-x"].into_iter());
		assert_eq!(
			Error::InvalidCommand("-x".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	#[test]
	fn invalid_command_long() {
		let opts = vec![Config::new('v', "verbose", Arg::None)];
		let mut p = Parser::new(opts, vec!["--unknown"].into_iter());
		assert_eq!(
			Error::InvalidCommand("--unknown".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	// short-only / long-only configurations
	#[test]
	fn short_only() {
		let opts = vec![Config::new('s', "", Arg::None)];
		let mut p = Parser::new(opts, vec!["-s"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('s', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn long_only() {
		let opts = vec![Config::new_long("sync", Arg::None)];
		let mut p = Parser::new(opts, vec!["--sync"].into_iter());
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!("sync", cfg.long),
			_ => panic!("expected Command::Opt with None"),
		}
	}

	#[test]
	fn short_only_with_invalid_long() {
		let opts = vec![Config::new_short('s', Arg::None)];
		let mut p = Parser::new(opts, vec!["--sync"].into_iter());
		assert_eq!(
			Error::InvalidCommand("--sync".to_string()),
			p.next_opt().unwrap_err()
		);
	}

	// mixed sequence
	#[test]
	fn mixed_flags_and_params() {
		let opts = vec![
			Config::new('v', "verbose", Arg::None),
			Config::new('n', "count", Arg::Optional),
			Config::new('o', "output", Arg::Required),
		];
		let mut p = Parser::new(
			opts,
			vec!["-v", "-n", "5", "--output", "out.txt"].into_iter(),
		);
		match p.next_opt().unwrap() {
			Command::Opt(cfg, None) => assert_eq!('v', cfg.short),
			_ => panic!("expected Command::Opt with None"),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(v)) => {
				assert_eq!('n', cfg.short);
				assert_eq!("5", v);
			}
			_ => panic!(),
		}
		match p.next_opt().unwrap() {
			Command::Opt(cfg, Some(v)) => {
				assert_eq!("output", cfg.long);
				assert_eq!("out.txt", v);
			}
			_ => panic!(),
		}
		match p.next_opt().unwrap() {
			Command::Rest(v) => assert!(v.is_empty()),
			_ => panic!(),
		}
	}

	#[test]
	fn config_new_fields() {
		let c = Config::new('x', "example", Arg::Required);
		assert_eq!('x', c.short);
		assert_eq!("example", c.long);
		assert_eq!(Arg::Required, c.arg);
		let c = Config::new_short('y', Arg::None);
		assert_eq!('y', c.short);
		assert_eq!("", c.long);
		assert_eq!(Arg::None, c.arg);
		let c = Config::new_long("final", Arg::Optional);
		assert_eq!('\0', c.short);
		assert_eq!("final", c.long);
		assert_eq!(Arg::Optional, c.arg);
	}
}
