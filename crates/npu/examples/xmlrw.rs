//! Apply `ov::stateless` rewrites to an engine IR XML (weights stay in the
//! export's `.bin`), for layer microbenchmarks:
//!
//! ```text
//! xmlrw IN.xml OUT.xml BIN unroll[:S] [...]
//! ```
//!
//! `unroll[:S]`: every delta-net Loop as S explicit recurrent steps
//! (`stateless::unroll_delta`; S = 1 for the decode graph).

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	if a.len() < 4 {
		eprintln!("usage: xmlrw IN.xml OUT.xml BIN unroll[:S] ...");
		std::process::exit(2);
	}
	let bin = std::path::Path::new(&a[2]);
	let mut xml = std::fs::read_to_string(&a[0]).expect("read IN.xml");
	for op in &a[3..] {
		let (name, arg) = op.split_once(':').unwrap_or((op, ""));
		match name {
			"unroll" => {
				let s = arg.parse().unwrap_or(1);
				let (x, n) = ov::stateless::unroll_delta(&xml, bin, s).expect("unroll_delta");
				eprintln!("unroll: {n} Loops -> {s} explicit steps each");
				xml = x;
			}
			o => panic!("unknown rewrite {o}"),
		}
	}
	std::fs::write(&a[1], xml).expect("write OUT.xml");
}
