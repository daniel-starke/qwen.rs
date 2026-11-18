//! Variants of the block indexer's `idx_score` graph, for timing with
//! `vpuxc.ps1` + `npubench blob`:
//!
//!   idxx VARIANT OUT.xml
//!
//! - `base`: the shipped graph (`flash::graphs::idx_score`).
//! - `h16`: the head sum in f16 before the f32 Convert (a quarter of the
//!   converted elements).
//! - `o16`: `h16` with an f16 output (no Convert).
use npubuild::flash::graphs::{idx_score, IDX_T};
use npubuild::g::{Graph, Ty};

const IH: usize = 4;
const ID: usize = 128;
const S: usize = 256;

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let (v, out) = (a[0].as_str(), std::path::Path::new(&a[1]));
	let g = if v == "base" {
		idx_score(S)
	} else {
		let mut g = Graph::new();
		let q = g.parameter("iq", Ty::F16, &[1, IH * S, ID]);
		let k = g.parameter("kidx", Ty::F16, &[1, IDX_T, ID]);
		let sc = g.matmul(&q, &k, false, true);
		let z = g.s16(0.0);
		let sc = g.maximum(&sc, &z);
		let sc = g.reshape(&sc, &[1, S as i64, IH as i64, IDX_T as i64]);
		let sc = g.reduce_sum(&sc, 2);
		let sc = g.reshape(&sc, &[1, S as i64, IDX_T as i64]);
		let sc = if v == "h16" {
			g.convert(&sc, Ty::F32)
		} else {
			sc
		};
		g.result(&sc, "sc");
		g
	};
	g.save(out, "idx_score").unwrap();
}
