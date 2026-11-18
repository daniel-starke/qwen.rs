//! Weight swizzling of the Flash-Next graphs (`flash::swizzled`):
//!
//!   swzx GDIR [S]
//!
//! writes `GDIR/{dpc,pdt,pdb}{,_s}.blob` (`delta_pre_c`, `post_delta_t`,
//! `post_delta_b` over `S` positions, compiled as the `compile` step does;
//! `_s`: weight inputs read swizzled, its `.swz` names them) for timing
//! with `npubench blob`. `GDIR/kern/hc` must hold the build's kernels (copy
//! `<npuout>/flash/graphs/kern/hc`).
//!
//!   swzx --set GRAPHS SRC DST
//!
//! compiles every graph of `GRAPHS/SRC` (a build's `graphs` directory and
//! its `flash`) that `flash::swizzled` names again with swizzled weight
//! inputs into `GRAPHS/DST`, and links the others' files there: a set for
//! an A/B against `SRC` (an engine description with `graphs = DST`). The
//! compiler from `NPUC_BIN`.
use npubuild::flash::graphs::{delta_pre_c, post_delta_b, post_delta_t};

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	if a[0] == "--set" {
		return set(std::path::Path::new(&a[1]), &a[2], &a[3]);
	}
	let gdir = std::path::Path::new(&a[0]);
	let s: usize = a.get(1).map_or(256, |v| v.parse().unwrap());
	let t = npubuild::sets::Tools::find(None, 2)?;
	for (stem, g) in [
		("dpc", delta_pre_c(s)),
		("pdt", post_delta_t(s)),
		("pdb", post_delta_b(s)),
	] {
		for swz in [false, true] {
			let stem = if swz {
				format!("{stem}_s")
			} else {
				stem.to_string()
			};
			g.save(&gdir.join(format!("{stem}.xml")), &stem)?;
			let args: Vec<String> = g.inputs().into_iter().map(|i| i.0).collect();
			std::fs::write(gdir.join(format!("{stem}.args")), args.join("\n") + "\n")
				.map_err(|e| e.to_string())?;
			npubuild::sets::compile_graph_kern(&t, gdir, &stem, Some("kern/hc"), swz)?;
			let sw = std::fs::read_to_string(gdir.join(format!("{stem}.swz"))).unwrap_or_default();
			println!(
				"{stem}: compiled, swizzled: {}",
				sw.split_whitespace().collect::<Vec<_>>().join(" ")
			);
		}
	}
	Ok(())
}

fn set(g: &std::path::Path, src: &str, dst: &str) -> Result<(), String> {
	let (sd, dd) = (g.join(src), g.join(dst));
	std::fs::create_dir_all(&dd).map_err(|e| e.to_string())?;
	let t = npubuild::sets::Tools::find(None, 6)?;
	let e = |e: std::io::Error| e.to_string();
	let mut jobs = Vec::new();
	for f in std::fs::read_dir(&sd).map_err(e)? {
		let p = f.map_err(e)?.path();
		let name = p.file_name().unwrap().to_string_lossy().into_owned();
		let Some(stem) = name.strip_suffix(".blob") else {
			continue;
		};
		let ours = npubuild::flash::swizzled(stem) && sd.join(format!("{stem}.xml")).exists();
		let exts: &[&str] = if ours {
			&["xml", "bin", "args"]
		} else {
			&["blob", "args", "swz"]
		};
		for x in exts {
			let (a, b) = (
				sd.join(format!("{stem}.{x}")),
				dd.join(format!("{stem}.{x}")),
			);
			if a.exists() && !b.exists() {
				std::fs::hard_link(&a, &b)
					.or_else(|_| std::fs::copy(&a, &b).map(|_| ()))
					.map_err(e)?;
			}
		}
		if ours && !dd.join(format!("{stem}.blob")).exists() {
			jobs.push(stem.to_string());
		}
	}
	let kern = |s: &str| {
		if s.starts_with("attn_far") {
			"kern/fattn"
		} else {
			"kern/hc"
		}
	};
	let q = std::sync::Mutex::new(jobs.iter());
	let errs = std::sync::Mutex::new(Vec::new());
	std::thread::scope(|sc| {
		for _ in 0..t.jobs {
			sc.spawn(|| loop {
				let Some(s) = q.lock().unwrap().next() else {
					break;
				};
				let stem = format!("{dst}/{s}");
				match npubuild::sets::compile_graph_kern(&t, g, &stem, Some(kern(s)), true) {
					Ok(()) => {
						let sw = std::fs::read_to_string(g.join(format!("{stem}.swz")))
							.unwrap_or_default();
						println!(
							"{s}: swizzled {}",
							sw.split_whitespace().collect::<Vec<_>>().join(" ")
						);
					}
					Err(x) => errs.lock().unwrap().push(x),
				}
			});
		}
	});
	let errs = errs.into_inner().unwrap();
	if let Some(x) = errs.first() {
		return Err(format!("{} failed; the first: {x}", errs.len()));
	}
	Ok(())
}
