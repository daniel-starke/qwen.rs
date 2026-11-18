//! The delta layers' prefill pre graph in its two forms, compiled as the
//! `compile` step does (Flash-Next options, `kern/hc`), for timing with
//! `npubench blob`:
//!
//!   dpcx GDIR [S]
//!
//! writes and compiles `GDIR/dpm.blob` (`delta_pre_m`: conv, Swish and the
//! l2 norms on the NPU) and `GDIR/dpc.blob` (`delta_pre_c`: the
//! projections only). `GDIR/kern/hc` must hold the build's kernels (copy
//! `<npuout>/flash/graphs/kern/hc`); the compiler from `NPUC_BIN`.
use npubuild::flash::graphs::{delta_pre_c, delta_pre_m};

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let gdir = std::path::Path::new(&a[0]);
	let s: usize = a.get(1).map_or(256, |v| v.parse().unwrap());
	let t = npubuild::sets::Tools::find(None, 2)?;
	for (stem, g) in [("dpm", delta_pre_m(s)), ("dpc", delta_pre_c(s))] {
		g.save(&gdir.join(format!("{stem}.xml")), stem)?;
		let args: Vec<String> = g.inputs().into_iter().map(|i| i.0).collect();
		std::fs::write(gdir.join(format!("{stem}.args")), args.join("\n") + "\n")
			.map_err(|e| e.to_string())?;
		npubuild::sets::compile_graph_kern(&t, gdir, stem, Some("kern/hc"), false)?;
		println!("{stem}: compiled");
	}
	Ok(())
}
