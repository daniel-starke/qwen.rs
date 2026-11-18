//! Write the transformer's graphs (`npubuild::qimage`; `QWEN_IMAGE_FOLD=0`
//! the unfolded form) to `OUT/<kind>.xml` (+ `.bin`), for timing a graph
//! change with `vpuxc.ps1` + `npubench blob` before building a set.
//!
//!   qigraph OUT [ROWS] [KIND...]      (ROWS: rows per job, default 256)
use npubuild::qimage::{fold, graph_form, S};

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let out = std::path::PathBuf::from(a.first().expect("qigraph OUT [ROWS] [KIND...]"));
	let s: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(S);
	let kinds = &a[a.len().min(2)..];
	std::fs::create_dir_all(&out).unwrap();
	for (n, post, pre) in [
		("dit_pre", false, true),
		("dit_mid", true, true),
		("dit_post", true, false),
	] {
		if !kinds.is_empty() && !kinds.iter().any(|k| k == n) {
			continue;
		}
		let f = out.join(format!("{n}.xml"));
		graph_form(post, pre, s, fold()).save(&f, n).unwrap();
		println!("{}", f.display());
	}
}
