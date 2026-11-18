//! NPU microbenchmarks through Level Zero (no OpenVINO).
//!
//! ```text
//! npubench info
//! npubench fc [--k 5120] [--n 17408] [--s 1] [--layers 1] [--shared] [--iters 20]
//!             [--profile] [--flags '--config K="V"'] [--save FILE] [--xml FILE]
//! npubench blob FILE [--iters 20] [--profile]
//! ```
//!
//! `fc` builds `layers` pairs of int4-sym g128 MatMuls (`k -> n -> k`, the
//! FFN up/down shapes without the activation) over `s` positions, compiles
//! them with the installed driver and times the graph: weight GB/s (the
//! decode roofline) and TFLOP/s (prefill). `--shared` makes every pair use
//! the same two weight constants (small IR, fast compile; still streamed from
//! DDR since 90 MB never fit in CMX).

use npu::ir::{f16_bits, f16_to_f32, Ir, Ty};
use npu::ze::{self, Graph, Npu};
use std::time::Instant;

fn arg<T: std::str::FromStr>(a: &[String], name: &str, def: T) -> T {
	a.iter()
		.position(|x| x == name)
		.and_then(|i| a.get(i + 1))
		.and_then(|v| v.parse().ok())
		.unwrap_or(def)
}

fn flag(a: &[String], name: &str) -> bool {
	a.iter().any(|x| x == name)
}

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let r = match a.first().map(String::as_str) {
		Some("info") => info(),
		Some("fc") => fc(&a),
		Some("blob") => blob(&a),
		Some("par") => par(&a),
		Some("ir") => ir_file(&a),
		Some("chain") => chain(&a),
		Some("token") => token(&a),
		Some("gen") => gen(&a),
		Some("spec") => spec(&a),
		Some("prefill") => prefill(&a),
		Some("pf") => pf(&a),
		Some("wtest") => wtest(&a),
		Some("rebind") => rebind(&a),
		Some("bmm") => bmm(&a),
		Some("moe") => moe(&a),
		_ => Err("usage: npubench info | fc [...] | blob FILE [...]".into()),
	};
	if let Err(e) = r {
		eprintln!("npubench: {e}");
		std::process::exit(1);
	}
}

fn open() -> Result<(Npu, npu::lock::NpuLock), String> {
	let lock = npu::lock::NpuLock::acquire()?;
	let n = Npu::open()?;
	Ok((n, lock))
}

fn info() -> Result<(), String> {
	let (n, _l) = open()?;
	let i = &n.info;
	println!(
		"device      {} (id {:#06x}), {} tiles, {} MHz",
		i.device_name,
		i.device_id,
		i.tiles,
		i.clock_mhz / 1_000_000
	);
	println!("driver      {:#x}", i.driver_version);
	println!(
		"graph ext   {}.{}",
		i.graph_ext_version >> 16,
		i.graph_ext_version & 0xffff
	);
	println!(
		"compiler    {}.{}, max opset {}",
		i.compiler.0, i.compiler.1, i.max_opset
	);
	println!(
		"elf         {}.{}.{}, runtime {}.{}.{}",
		i.elf.major, i.elf.minor, i.elf.patch, i.runtime.major, i.runtime.minor, i.runtime.patch
	);
	for (k, what) in [(1, "compiler"), (2, "driver")] {
		match n.supported_options(k) {
			Ok(s) => println!("{what} options: {s}"),
			Err(e) => println!("{what} options: {e}"),
		}
	}
	Ok(())
}

/// Bind a fresh host buffer to every argument; inputs get small f16 noise.
fn bind_all<'a>(n: &'a Npu, g: &mut Graph<'a>) -> Result<Vec<ze::HostBuf<'a>>, String> {
	let mut bufs = Vec::new();
	for i in 0..g.args.len() {
		let arg = g.args[i].clone();
		let mut b = n.alloc(arg.bytes)?;
		// `NPU_INPUTS=DIR`: an input `NAME` is read from `DIR/NAME.in` (raw,
		// the arg's precision) when that file exists.
		let file = std::env::var("NPU_INPUTS")
			.ok()
			.map(|d| std::path::Path::new(&d).join(format!("{}.in", arg.name)));
		if !arg.is_output && file.as_ref().is_some_and(|f| f.exists()) {
			let d = std::fs::read(file.as_ref().unwrap()).map_err(|e| e.to_string())?;
			if d.len() != arg.bytes {
				return Err(format!(
					"{}: {} bytes, the arg has {}",
					file.unwrap().display(),
					d.len(),
					arg.bytes
				));
			}
			b.as_mut_slice()[..d.len()].copy_from_slice(&d);
		} else if !arg.is_output {
			let r = npu::noise(arg.bytes, 7 + i as u64);
			let s = b.as_mut_slice();
			if arg.precision == 0x02 {
				for (j, c) in s.chunks_exact_mut(2).enumerate() {
					let v = (r[j % r.len()] as f32 - 127.5) / 512.0;
					c.copy_from_slice(&f16_bits(v).to_le_bytes());
				}
			} else {
				s.copy_from_slice(&r);
			}
		}
		g.bind(i, &b)?;
		println!(
			"  arg {i}: {} {} {:?} {} ({} bytes)",
			if arg.is_output { "out" } else { "in " },
			arg.name,
			arg.dims,
			ze::precision_name(arg.precision),
			arg.bytes
		);
		bufs.push(b);
	}
	Ok(bufs)
}

struct Stats {
	min: f64,
	med: f64,
}

fn time(g: &mut Graph<'_>, iters: usize) -> Result<Stats, String> {
	g.prepare(false)?;
	let first = g.run()?;
	let mut t: Vec<f64> = (0..iters).map(|_| g.run()).collect::<Result<_, _>>()?;
	t.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let s = Stats {
		min: t[0],
		med: t[t.len() / 2],
	};
	println!(
		"  first run {:.3} ms; {} runs: min {:.3} ms, median {:.3} ms",
		first * 1e3,
		iters,
		s.min * 1e3,
		s.med * 1e3
	);
	Ok(s)
}

fn profile(g: &mut Graph<'_>) -> Result<(), String> {
	g.prepare(true)?;
	g.run()?;
	let tasks = g.tasks()?;
	let (mut t0, mut t1) = (u64::MAX, 0u64);
	let mut by: std::collections::BTreeMap<&str, (u64, usize)> = Default::default();
	for t in &tasks {
		t0 = t0.min(t.start_ns);
		t1 = t1.max(t.start_ns + t.duration_ns);
		let e = by.entry(t.engine()).or_default();
		e.0 += t.duration_ns;
		e.1 += 1;
	}
	println!(
		"  profile: {} tasks, span {:.3} ms",
		tasks.len(),
		(t1.saturating_sub(t0)) as f64 * 1e-6
	);
	for (k, (ns, c)) in &by {
		println!(
			"    {k:4} {c:5} tasks, busy {:.3} ms (sum of durations)",
			*ns as f64 * 1e-6
		);
	}
	if std::env::var("NPU_TASKS").is_ok() {
		for t in &tasks {
			println!(
				"T {} {} {} {} {}",
				t.engine(),
				t.start_ns - t0,
				t.duration_ns,
				t.name(),
				t.layer_type()
			);
		}
	}
	let mut top: Vec<_> = tasks.iter().collect();
	top.sort_by_key(|t| std::cmp::Reverse(t.duration_ns));
	for t in top.iter().take(12) {
		println!(
			"    {:4} {:8.3} ms  start {:8.3}  {} [{}]",
			t.engine(),
			t.duration_ns as f64 * 1e-6,
			(t.start_ns - t0) as f64 * 1e-6,
			t.name(),
			t.layer_type()
		);
	}
	Ok(())
}

fn fc(a: &[String]) -> Result<(), String> {
	let k: usize = arg(a, "--k", 5120);
	let n_: usize = arg(a, "--n", 17408);
	let s: usize = arg(a, "--s", 1);
	let layers: usize = arg(a, "--layers", 1);
	let iters: usize = arg(a, "--iters", 20);
	let shared = flag(a, "--shared");
	let flags: String = arg(a, "--flags", String::new());
	let wtype: String = arg(a, "--wtype", "i4".to_string());
	// Weights (and scales) as graph inputs instead of constants.
	let winput = flag(a, "--winput");
	let post = flag(a, "--post");
	// `--group 0`: one scale per output row (per-channel), for each matrix.
	let g: usize = arg(a, "--group", 128);
	let (gu, gd) = if g == 0 { (k, n_) } else { (g, g) };
	let mut ir = Ir::new();
	let x0 = if flag(a, "--3d") {
		ir.parameter("x", Ty::F16, &[1, s, k])
	} else {
		ir.parameter("x", Ty::F16, &[s, k])
	};
	let mut x = x0;
	let scale_up = vec![1.0 / (5.0 * (k as f32).sqrt()); n_ * k / gu];
	let scale_dn = vec![1.0 / (5.0 * (n_ as f32).sqrt()); k * n_ / gd];
	let mut w: Option<(npu::ir::P, npu::ir::P)> = None;
	for l in 0..layers {
		let (wu, wd) = match (&w, shared) {
			(Some((u, d)), true) => (u.clone(), d.clone()),
			_ => {
				let mk = |ir: &mut Ir, n: usize, kk: usize, seed: u64, sc: &[f32]| {
					let g = if g == 0 { kk } else { g };
					match wtype.as_str() {
						"i8" => ir.weight_i8(n, kk, &npu::noise(n * kk, seed), &sc[..n]),
						"f16" => {
							let b: Vec<u8> = npu::noise(n * kk, seed)
								.iter()
								.flat_map(|&r| {
									f16_bits((r as f32 - 127.5) * sc[0] / 16.0).to_le_bytes()
								})
								.collect();
							ir.weight_f16(n, kk, &b)
						}
						// `--post`: i4 [n, k] input, per-row scale applied after the MatMul
						// (the `--allin` form); here only a fixed 2^-3 before it.
						_ if winput && post => {
							let id = ir.layer_count();
							let w = ir.parameter(&format!("w{id}"), Ty::I4, &[n, kk]);
							let w = ir.convert(&w, Ty::F16);
							let c = ir.const_f16(&[1, 1], &[0.125]);
							ir.multiply(&w, &c)
						}
						_ if winput => {
							let id = ir.layer_count();
							let w = ir.parameter(&format!("w{id}"), Ty::I4, &[n, kk / g, g]);
							let w = ir.convert(&w, Ty::F16);
							let s = ir.parameter(&format!("s{id}"), Ty::F16, &[n, kk / g, 1]);
							let w = ir.multiply(&w, &s);
							ir.reshape(&w, &[n, kk])
						}
						_ => ir.weight_i4(n, kk, g, &npu::noise(n * kk / 2, seed), sc),
					}
				};
				let u = mk(&mut ir, n_, k, 100 + 2 * l as u64, &scale_up);
				let d = mk(&mut ir, k, n_, 101 + 2 * l as u64, &scale_dn);
				w = Some((u.clone(), d.clone()));
				(u, d)
			}
		};
		let f32p = flag(a, "--f32");
		let (wu, wd) = if f32p {
			(ir.convert(&wu, Ty::F32), ir.convert(&wd, Ty::F32))
		} else {
			(wu, wd)
		};
		if f32p && l == 0 {
			x = ir.convert(&x, Ty::F32);
		}
		let fq = flag(a, "--fq");
		let xi = if fq {
			ir.fake_quantize(&x, -4.0, 4.0, 256)
		} else {
			x.clone()
		};
		if flag(a, "--gbatch") {
			let c = 1.0 / 64.0;
			let h = ir.linear_i4_gbatch(
				&xi,
				n_,
				k,
				g,
				&npu::noise(n_ * k / 2, 100 + 2 * l as u64),
				&scale_up,
				c,
			);
			x = ir.linear_i4_gbatch(
				&h,
				k,
				n_,
				g,
				&npu::noise(k * n_ / 2, 101 + 2 * l as u64),
				&scale_dn,
				c,
			);
			continue;
		}
		if flag(a, "--dq") {
			// Dynamic per-token int8 activations (`Ir::dyn_quant`).
			let (xq, sx) = ir.dyn_quant(&x);
			let h = ir.matmul(&xq, &wu, false, true);
			let h = ir.multiply(&h, &sx);
			let (hq, sh) = ir.dyn_quant(&h);
			let y = ir.matmul(&hq, &wd, false, true);
			x = ir.multiply(&y, &sh);
			continue;
		}
		let h = ir.matmul(&xi, &wu, false, true);
		let h = if post {
			let id = ir.layer_count();
			let sp = ir.parameter(&format!("su{id}"), Ty::F16, &[1, n_]);
			ir.multiply(&h, &sp)
		} else {
			h
		};
		let hi = if fq {
			ir.fake_quantize(&h, -4.0, 4.0, 256)
		} else {
			h
		};
		x = ir.matmul(&hi, &wd, false, true);
		if post {
			let id = ir.layer_count();
			let sp = ir.parameter(&format!("sd{id}"), Ty::F16, &[1, k]);
			x = ir.multiply(&x, &sp);
		}
	}
	ir.result(&x, "y");
	let xml = ir.xml("fc_bench");
	if let Some(p) = a
		.iter()
		.position(|x| x == "--xml")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(p, &xml).map_err(|e| e.to_string())?;
		std::fs::write(format!("{p}.bin"), &ir.bin).map_err(|e| e.to_string())?;
	}
	let wbytes = 2
		* layers
		* match wtype.as_str() {
			"i8" => n_ * k + n_ * 2,
			"f16" => n_ * k * 2,
			_ => n_ * k / 2 + n_ * k / gu,
		};
	let flops = 2.0 * 2.0 * (layers * s * n_ * k) as f64;
	println!(
		"fc: {wtype} s={s} k={k} n={n_} layers={layers}{} -> {:.1} MB weights read per run, {:.2} GFLOP; IR bin {:.1} MB",
		if shared { " (shared weights)" } else { "" },
		wbytes as f64 / 1e6,
		flops / 1e9,
		ir.bin.len() as f64 / 1e6
	);
	let (npu, _l) = open()?;
	let t = Instant::now();
	let slot = npu::lock::CompileSlot::acquire()?;
	let mut gflags = 0;
	if flag(a, "--profile") {
		gflags |= ze::GRAPH_FLAG_ENABLE_PROFILING;
	}
	let mut graph = npu.compile(xml.as_bytes(), &ir.bin, &flags, gflags)?;
	drop(slot);
	println!("  compiled in {:.1} s", t.elapsed().as_secs_f64());
	if let Some(p) = a
		.iter()
		.position(|x| x == "--save")
		.and_then(|i| a.get(i + 1))
	{
		let b = graph.native()?;
		std::fs::write(p, &b).map_err(|e| e.to_string())?;
		println!("  blob {} bytes -> {p}", b.len());
	}
	let bufs = bind_all(&npu, &mut graph)?;
	let reps: usize = arg(a, "--reps", 0);
	if reps > 0 {
		// The same native blob loaded `reps` times (separate weights copies,
		// as per-layer blobs would be), all bound to the same buffers, one
		// command list.
		let blob = graph.native()?;
		let mut gs = Vec::new();
		for _ in 0..reps {
			let mut g = npu.load(&blob, 0)?;
			for (i, b) in bufs.iter().enumerate() {
				g.bind(i, b)?;
			}
			gs.push(g);
		}
		let mut refs: Vec<&mut Graph<'_>> = gs.iter_mut().collect();
		let mut batch = npu.batch(&mut refs)?;
		let first = batch.run()?;
		let mut t: Vec<f64> = (0..iters).map(|_| batch.run()).collect::<Result<_, _>>()?;
		t.sort_by(|a, b| a.partial_cmp(b).unwrap());
		println!(
			"  batch of {reps}: first {:.3} ms, min {:.3} ms ({:.3} ms per graph), median {:.3} ms => {:.1} GB/s",
			first * 1e3,
			t[0] * 1e3,
			t[0] * 1e3 / reps as f64,
			t[t.len() / 2] * 1e3,
			(wbytes * reps) as f64 / t[0] / 1e9
		);
	}
	let st = time(&mut graph, iters)?;
	println!(
		"  => weights {:.1} GB/s, {:.3} TFLOP/s (at min)",
		wbytes as f64 / st.min / 1e9,
		flops / st.min / 1e12
	);
	let _ = st.med;
	// Sanity: outputs finite?
	let out = bufs.last().unwrap().as_slice();
	let vals: Vec<f32> = out
		.chunks_exact(2)
		.take(8)
		.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect();
	println!("  y[..8] = {vals:?}");
	if flag(a, "--profile") {
		profile(&mut graph)?;
	}
	Ok(())
}

/// Batched activation x activation MatMul: `--shape B,M,K,N` (`[1, B, M, K] x
/// [1, B, K, N]`, both graph inputs), `--chain C` dependent products in
/// sequence (each output feeds the next as the left operand; needs K = N),
/// f16. Times the graph and reports GFLOP/s: the cost of the delta-net's
/// small per-head matmuls.
/// Qwen3.8-Flash-Next's routed experts as one job: `--g` experts of `--c`
/// tokens each (`x{e} [c, 2560]` f16), every expert's int4 gate, up
/// `[640, 2560]` and down `[2560, 640]` inputs of their own (bound per job
/// from wherever the expert lives), static int8 activations, per-row scales
/// after the MatMuls, SwiGLU between. Prints the time per job and the weight
/// bytes read per second.
fn moe(a: &[String]) -> Result<(), String> {
	let g: usize = arg(a, "--g", 16);
	let c: usize = arg(a, "--c", 32);
	let iters: usize = arg(a, "--iters", 20);
	let nofq = flag(a, "--nofq");
	let (w, f) = (2560usize, 640usize);
	let mut ir = Ir::new();
	let fc = |ir: &mut Ir, x: &npu::ir::P, name: &str, n: usize, k: usize| {
		let wt = ir.parameter(name, Ty::I4, &[n, k]);
		let wt = ir.convert(&wt, Ty::F16);
		let sc = ir.parameter(&format!("s_{name}"), Ty::F16, &[1, n]);
		let y = ir.matmul(x, &wt, false, true);
		ir.multiply(&y, &sc)
	};
	for e in 0..g {
		let x = ir.parameter(&format!("x{e}"), Ty::F16, &[c, w]);
		let xq = if nofq {
			x.clone()
		} else {
			ir.fake_quantize(&x, -4.0, 4.0, 256)
		};
		let gt = fc(&mut ir, &xq, &format!("g{e}"), f, w);
		let up = fc(&mut ir, &xq, &format!("u{e}"), f, w);
		let sg = ir.unary("Sigmoid", &gt);
		let sg = ir.multiply(&gt, &sg);
		let hh = ir.multiply(&sg, &up);
		let hq = if nofq {
			hh.clone()
		} else {
			ir.fake_quantize(&hh, -4.0, 4.0, 256)
		};
		let y = fc(&mut ir, &hq, &format!("d{e}"), w, f);
		ir.result(&y, &format!("y{e}"));
	}
	let xml = ir.xml("moe");
	if let Some(p) = a
		.iter()
		.position(|x| x == "--xml")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(p, &xml).map_err(|e| e.to_string())?;
		let bin = p.strip_suffix(".xml").unwrap_or(p).to_string() + ".bin";
		std::fs::write(bin, &ir.bin).map_err(|e| e.to_string())?;
		return Ok(());
	}
	let (npu, _l) = open()?;
	let t = std::time::Instant::now();
	let mut gr = npu.compile(xml.as_bytes(), &ir.bin, "", 0)?;
	println!("  compiled in {:.1} s", t.elapsed().as_secs_f64());
	let _b = bind_all(&npu, &mut gr)?;
	let st = time(&mut gr, iters)?;
	let wb = (g * 3 * f * w / 2) as f64;
	println!(
		"  moe {g} experts x {c} tokens: {:.3} ms per job, {:.3} ms per expert, {:.1} GB/s of weights, {:.2} TOPS",
		st.min * 1e3,
		st.min * 1e3 / g as f64,
		wb / st.min / 1e9,
		2.0 * (g * c * 3 * f * w) as f64 / st.min / 1e12
	);
	Ok(())
}

fn bmm(a: &[String]) -> Result<(), String> {
	let sh: String = arg(a, "--shape", "384,32,32,32".to_string());
	let d: Vec<usize> = sh.split(',').map(|v| v.parse().unwrap()).collect();
	let (b, m, k, n) = (d[0], d[1], d[2], d[3]);
	let chain: usize = arg(a, "--chain", 1);
	let iters: usize = arg(a, "--iters", 20);
	let mut ir = Ir::new();
	let x = ir.parameter("x", Ty::F16, &[1, b, m, k]);
	let w = ir.parameter("w", Ty::F16, &[1, b, k, n]);
	let mut y = ir.matmul(&x, &w, false, false);
	for _ in 1..chain {
		y = ir.matmul(&y, &w, false, false);
	}
	ir.result(&y, "y");
	let xml = ir.xml("bmm");
	if let Some(p) = a
		.iter()
		.position(|x| x == "--xml")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(p, &xml).map_err(|e| e.to_string())?;
		std::fs::write(
			p.strip_suffix(".xml").unwrap_or(p).to_string() + ".bin",
			&ir.bin,
		)
		.map_err(|e| e.to_string())?;
		if flag(a, "--xml-only") {
			return Ok(());
		}
	}
	let (npu, _l) = open()?;
	let gflags = if flag(a, "--profile") {
		ze::GRAPH_FLAG_ENABLE_PROFILING
	} else {
		0
	};
	let mut g = npu.compile(xml.as_bytes(), &ir.bin, "", gflags)?;
	let _b = bind_all(&npu, &mut g)?;
	let st = time(&mut g, iters)?;
	let fl = 2.0 * (b * m * k * n * chain) as f64;
	println!("  bmm {b}x[{m}x{k}]x[{k}x{n}] chain {chain}: {:.3} ms, {:.1} GFLOP/s, {:.1} us per product", st.min * 1e3, fl / st.min / 1e9, st.min * 1e6 / chain as f64);
	if flag(a, "--profile") {
		profile(&mut g)?;
	}
	Ok(())
}

/// Compile an IR file (`X.xml` + `X.bin`) with the driver and time it.
fn ir_file(a: &[String]) -> Result<(), String> {
	let path = a.get(1).ok_or("ir FILE.xml")?;
	let iters: usize = arg(a, "--iters", 20);
	let flags: String = arg(a, "--flags", String::new());
	let xml = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
	let binp = path.strip_suffix(".xml").unwrap_or(path).to_string() + ".bin";
	let bin = std::fs::read(&binp).map_err(|e| format!("{binp}: {e}"))?;
	let (npu, _l) = open()?;
	let gflags = if flag(a, "--profile") {
		ze::GRAPH_FLAG_ENABLE_PROFILING
	} else {
		0
	};
	let t = Instant::now();
	let slot = npu::lock::CompileSlot::acquire()?;
	let mut g = npu.compile(&xml, &bin, &flags, gflags)?;
	drop(slot);
	println!(
		"  compiled {} MB of weights in {:.1} s",
		bin.len() / 1_000_000,
		t.elapsed().as_secs_f64()
	);
	if let Some(p) = a
		.iter()
		.position(|x| x == "--save")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(p, g.native()?).map_err(|e| e.to_string())?;
	}
	let _bufs = bind_all(&npu, &mut g)?;
	let st = time(&mut g, iters)?;
	println!(
		"  => {:.1} GB/s of IR weights",
		bin.len() as f64 / st.min / 1e9
	);
	if flag(a, "--profile") {
		profile(&mut g)?;
	}
	Ok(())
}

/// Load several per-layer blobs and run them as a decode chain in ONE command
/// list: each blob's `y`/`x` residual is a shared host buffer linking it to
/// the next, every other arg gets its own buffer. Proves per-layer blobs
/// compose with zero inter-layer copies (the NPU-only decode architecture).
///
///   npubench chain A.blob B.blob ... [--iters N] [--reps R]
///
/// `--reps R` repeats the given list R times (so 2 blobs x 32 ~ a 64-layer
/// token's cost). Input `x` of a blob is bound to the previous blob's `y`.
fn chain(a: &[String]) -> Result<(), String> {
	let iters: usize = arg(a, "--iters", 20);
	let reps: usize = arg(a, "--reps", 1);
	let blobs: Vec<&String> = a[1..].iter().filter(|s| s.ends_with(".blob")).collect();
	if blobs.is_empty() {
		return Err("chain needs at least one .blob".into());
	}
	let (npu, _l) = open()?;
	let raw: Vec<Vec<u8>> = blobs
		.iter()
		.map(|p| std::fs::read(p).map_err(|e| format!("{p}: {e}")))
		.collect::<Result<_, _>>()?;
	let mut graphs = Vec::new();
	for _ in 0..reps {
		for b in &raw {
			graphs.push(npu.load(b, 0)?);
		}
	}
	// One residual buffer shared by the whole chain (every blob's x and y).
	let w = graphs[0]
		.args
		.iter()
		.find(|a| a.name == "x")
		.map(|a| a.bytes)
		.ok_or("no x arg")?;
	let residual = npu.alloc(w)?;
	let mut owned = Vec::new();
	for g in graphs.iter_mut() {
		for i in 0..g.args.len() {
			let arg = g.args[i].clone();
			if arg.name == "x" || arg.name == "y" {
				g.bind(i, &residual)?;
			} else {
				let b = npu.alloc(arg.bytes)?;
				g.bind(i, &b)?;
				owned.push(b);
			}
		}
	}
	let mut refs: Vec<&mut Graph<'_>> = graphs.iter_mut().collect();
	let mut batch = npu.batch(&mut refs)?;
	let first = batch.run()?;
	let mut t: Vec<f64> = (0..iters).map(|_| batch.run()).collect::<Result<_, _>>()?;
	t.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let n = graphs.len();
	println!(
		"  chain of {n} blobs: first {:.2} ms, min {:.2} ms ({:.2} ms per blob), median {:.2} ms",
		first * 1e3,
		t[0] * 1e3,
		t[0] * 1e3 / n as f64,
		t[t.len() / 2] * 1e3
	);
	let _ = owned;
	Ok(())
}

/// A full decode token from per-layer blobs: load layer0..layerN + head from
/// DIR, thread the residual through one shared buffer, zero all conv/ssm/KV
/// state (a first token from a zero state), run every blob + the head in one
/// command list, and report time + argmax of `logits`.
///
///   npubench token DIR [--layers 64] [--emb FILE] [--iters 10] [--dump FILE]
///
/// `--emb FILE`: raw f16 [5120] inputs_embeds for layer 0 (else small noise).
/// attention_mask marks only position 0 valid; position_ids = 0.
fn token(a: &[String]) -> Result<(), String> {
	let dir = a.get(1).ok_or("token DIR")?;
	let nlayers: usize = arg(a, "--layers", 64);
	let iters: usize = arg(a, "--iters", 10);
	let (npu, _l) = open()?;
	let names: Vec<String> = (0..nlayers)
		.map(|i| format!("layer{i}"))
		.chain(Some("head".to_string()))
		.collect();
	let raw: Vec<Vec<u8>> = names
		.iter()
		.map(|n| std::fs::read(format!("{dir}/{n}.blob")).map_err(|e| format!("{n}: {e}")))
		.collect::<Result<_, _>>()?;
	let mut graphs: Vec<Graph<'_>> = raw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	// Shared residual (every layer's x and y; the head's x).
	let w = graphs[0]
		.args
		.iter()
		.find(|a| a.name == "x")
		.map(|a| a.bytes)
		.ok_or("no x")?;
	// A distinct residual buffer per layer boundary (emb -> b[0] -> b[1] ...
	// -> head). One shared buffer would alias each layer's x and y (the NPU
	// graph reads x and writes y, and aliasing corrupts it -> zeros).
	let resid: Vec<ze::HostBuf<'_>> = (0..=nlayers)
		.map(|_| npu.alloc(w))
		.collect::<Result<_, _>>()?;
	let mut emb = npu.alloc(w)?;
	if let Some(f) = a
		.iter()
		.position(|x| x == "--emb")
		.and_then(|i| a.get(i + 1))
	{
		let d = std::fs::read(f).map_err(|e| e.to_string())?;
		emb.as_mut_slice()[..d.len().min(w)].copy_from_slice(&d[..d.len().min(w)]);
	} else {
		for (i, c) in emb.as_mut_slice().chunks_exact_mut(2).enumerate() {
			c.copy_from_slice(&f16_bits(((i % 97) as f32 - 48.0) / 256.0).to_le_bytes());
		}
	}
	let mut owned = Vec::new();
	let mut logits_idx = None;
	for (gi, g) in graphs.iter_mut().enumerate() {
		let is_head = gi == nlayers;
		for i in 0..g.args.len() {
			let arg = g.args[i].clone();
			// The head graph (last) has one output: the logits, whatever its
			// op name.
			if is_head && arg.is_output {
				let b = npu.alloc(arg.bytes)?;
				logits_idx = Some((owned.len(), arg.dims.clone()));
				g.bind(i, &b)?;
				owned.push(b);
				continue;
			}
			// The residual input kept the name "x"; the residual *output* did
			// not survive save_model (it reports the aten::add op name), so
			// bind by position/type: the one non-state output is the residual.
			if arg.is_output {
				// Output names don't survive save_model, so the residual
				// output is identified by its [1,1,5120] size (= w bytes);
				// every other output (present conv/ssm/KV state) gets its own
				// throwaway buffer (states don't matter for one token).
				if arg.bytes == w {
					g.bind(i, &resid[gi])?;
				} else {
					let b = npu.alloc(arg.bytes)?;
					g.bind(i, &b)?;
					owned.push(b);
				}
				continue;
			}
			match arg.name.as_str() {
				// layer gi reads resid[gi-1] (emb for gi 0); head reads
				// resid[nlayers-1].
				"x" => g.bind(i, if gi == 0 { &emb } else { &resid[gi - 1] })?,
				nm => {
					let mut b = npu.alloc(arg.bytes)?;
					// attention_mask: only position 0 (the new slot at index CAP) valid.
					if nm == "attention_mask" {
						for e in b.as_mut_slice().chunks_exact_mut(8) {
							e.copy_from_slice(&0i64.to_le_bytes());
						}
						let s = b.as_mut_slice();
						let n = s.len() / 8;
						s[(n - 1) * 8..].copy_from_slice(&1i64.to_le_bytes());
					}
					g.bind(i, &b)?;
					owned.push(b);
				}
			}
		}
	}
	let mut refs: Vec<&mut Graph<'_>> = graphs.iter_mut().collect();
	let mut batch = npu.batch(&mut refs)?;
	let first = batch.run()?;
	let mut t: Vec<f64> = (0..iters).map(|_| batch.run()).collect::<Result<_, _>>()?;
	t.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let (li, dims) = logits_idx.ok_or("head produced no logits output")?;
	let out = owned[li].as_slice();
	// logits are f32 (lm_head output kept f32).
	let vocab = dims.iter().product::<u32>() as usize;
	let fl: Vec<f32> = out[..vocab * 4]
		.chunks_exact(4)
		.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
		.collect();
	let (arg, mx) =
		fl.iter().enumerate().fold(
			(0usize, f32::MIN),
			|(ai, am), (i, &v)| if v > am { (i, v) } else { (ai, am) },
		);
	println!(
		"  token: {} blobs + head, first {:.1} ms, min {:.1} ms ({:.2} ms/layer), median {:.1} ms",
		nlayers,
		first * 1e3,
		t[0] * 1e3,
		t[0] * 1e3 / (nlayers + 1) as f64,
		t[t.len() / 2] * 1e3
	);
	println!("  argmax {arg} = {mx:.4}, vocab {vocab}");
	if let Some(f) = a
		.iter()
		.position(|x| x == "--dump")
		.and_then(|i| a.get(i + 1))
	{
		let bytes: Vec<u8> = fl.iter().flat_map(|v| v.to_le_bytes()).collect();
		std::fs::write(f, &bytes).map_err(|e| e.to_string())?;
	}
	Ok(())
}

/// `gen DIR --emb TABLE.f16 --prompt IDS.txt [--new N] [--layers 64]
/// [--dump-logits FILE]`: real greedy generation through the per-layer
/// decode blobs (`cut_all.py` + compile, DIR/layer{L}.blob + head.blob;
/// DIR/outs.txt names each layer's outputs in order). ONE mutable command
/// list of all 65 appends, recorded once:
/// - delta-net conv / ssm states ping-pong between two buffers per state
///   (token t reads set t % 2 and writes the other; the list's state
///   arguments are re-pointed each token, nothing copied);
/// - attention KV is a sliding window of CAP rows: the graph reads past
///   [1, 4, CAP, 256] and writes present [1, 4, CAP + 1, 256]; the host copies
///   present rows 1..=CAP back to past (per head), the mask marks the last t
///   past slots and the new one valid, position_ids = t.
///
/// Embeddings come from TABLE (`[vocab, 5120]` f16, the export's embedding
/// model evaluated for every id); the next token is the argmax of the logits.
fn gen(a: &[String]) -> Result<(), String> {
	let dir = a.get(1).ok_or("gen DIR")?;
	let nlayers: usize = arg(a, "--layers", 64);
	let new_tokens: usize = arg(a, "--new", 16);
	let embf: String = arg(a, "--emb", String::new());
	let promptf: String = arg(a, "--prompt", String::new());
	let prompt: Vec<usize> = std::fs::read_to_string(&promptf)
		.map_err(|e| format!("{promptf}: {e}"))?
		.split_whitespace()
		.map(|v| v.parse().map_err(|_| format!("bad id {v}")))
		.collect::<Result<_, _>>()?;
	if prompt.is_empty() {
		return Err("empty prompt".into());
	}
	let outs: Vec<Vec<String>> = {
		let t = std::fs::read_to_string(format!("{dir}/outs.txt"))
			.map_err(|e| format!("outs.txt: {e}"))?;
		t.lines()
			.map(|l| {
				l.split_whitespace()
					.skip(1)
					.map(|s| s.to_string())
					.collect()
			})
			.collect()
	};
	let (npu, _l) = open()?;
	let t0 = Instant::now();
	let names: Vec<String> = (0..nlayers)
		.map(|i| format!("layer{i}"))
		.chain(Some("head".to_string()))
		.collect();
	let raw: Vec<Vec<u8>> = names
		.iter()
		.map(|n| std::fs::read(format!("{dir}/{n}.blob")).map_err(|e| format!("{n}: {e}")))
		.collect::<Result<_, _>>()?;
	let mut graphs: Vec<Graph<'_>> = raw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	let w = graphs[0]
		.args
		.iter()
		.find(|a| a.name == "x")
		.map(|a| a.bytes)
		.ok_or("no x")?;
	let table = std::fs::File::open(&embf).map_err(|e| format!("{embf}: {e}"))?;
	let read_emb = |id: usize, dst: &mut [u8]| -> Result<(), String> {
		use npu::ReadAt;
		table
			.seek_read(&mut dst[..w], (id * w) as u64)
			.map_err(|e| e.to_string())?;
		Ok(())
	};

	// buffer arena
	let mut bufs: Vec<ze::HostBuf<'_>> = Vec::new();
	fn new_buf<'n>(
		npu: &'n Npu,
		bufs: &mut Vec<ze::HostBuf<'n>>,
		bytes: usize,
	) -> Result<usize, String> {
		let mut b = npu.alloc(bytes)?;
		b.as_mut_slice().fill(0);
		bufs.push(b);
		Ok(bufs.len() - 1)
	}
	let emb = new_buf(&npu, &mut bufs, w)?;
	let resid: Vec<usize> = (0..nlayers)
		.map(|_| new_buf(&npu, &mut bufs, w))
		.collect::<Result<_, _>>()?;
	let mut mask_b: Option<usize> = None;
	let mut bias_b: Option<usize> = None;
	let mut pos_b: Option<usize> = None;
	let mut logits_b: Option<(usize, usize)> = None;
	// per append: arg -> buffer for parity 0 and 1
	let mut sets: [Vec<Vec<usize>>; 2] = [Vec::new(), Vec::new()];
	// (append, arg) pairs whose buffer differs between parities
	let mut flip: Vec<(usize, usize)> = Vec::new();
	// attention KV windows: (present, past, is value) buffers
	let mut kv: Vec<(usize, usize, bool)> = Vec::new();
	let mut cap = 0usize;
	for gi in 0..=nlayers {
		let args = graphs[gi].args.clone();
		let is_head = gi == nlayers;
		let out_names: Vec<String> = if is_head {
			vec!["logits".into()]
		} else {
			outs.get(gi).cloned().ok_or("outs.txt too short")?
		};
		let mut s0 = Vec::new();
		let mut s1 = Vec::new();
		// the past input buffer(s) of each state, by state name
		let mut past: Vec<(String, usize, usize)> = Vec::new();
		let mut k_out = 0;
		// inputs first (they precede the outputs in the blob)
		for (i, arg) in args.iter().enumerate() {
			if arg.is_output {
				let nm = out_names.get(k_out).cloned().unwrap_or_default();
				k_out += 1;
				let (b0, b1) = if is_head {
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					logits_b = Some((b, arg.bytes / 4));
					(b, b)
				} else if nm == "y" {
					(resid[gi], resid[gi])
				} else if let Some(st) = nm.strip_prefix("cache_params.present.") {
					let (_, p0, p1) = past
						.iter()
						.find(|(n, _, _)| n == st)
						.cloned()
						.ok_or(format!("layer {gi}: no past for {nm}"))?;
					if st.starts_with("key") || st.starts_with("value") {
						let q = new_buf(&npu, &mut bufs, arg.bytes)?;
						kv.push((q, p0, st.starts_with("value")));
						(q, q)
					} else {
						// ping-pong: parity p reads set p, writes the other
						flip.push((gi, i));
						(p1, p0)
					}
				} else {
					return Err(format!("layer {gi}: unknown output {nm}"));
				};
				s0.push(b0);
				s1.push(b1);
				continue;
			}
			let (b0, b1) = match arg.name.as_str() {
				"x" => {
					let b = if gi == 0 { emb } else { resid[gi - 1] };
					(b, b)
				}
				"attention_mask" => {
					let b = match mask_b {
						Some(b) => b,
						None => {
							let b = new_buf(&npu, &mut bufs, arg.bytes)?;
							cap = arg.bytes / 8 - 1;
							mask_b = Some(b);
							b
						}
					};
					(b, b)
				}
				"attn_bias" => {
					let b = match bias_b {
						Some(b) => b,
						None => {
							let b = new_buf(&npu, &mut bufs, arg.bytes)?;
							cap = arg.bytes / 2;
							bias_b = Some(b);
							b
						}
					};
					(b, b)
				}
				"position_ids" => {
					let b = match pos_b {
						Some(b) => b,
						None => {
							let b = new_buf(&npu, &mut bufs, arg.bytes)?;
							pos_b = Some(b);
							b
						}
					};
					(b, b)
				}
				nm if nm.starts_with("cache_params.past.") => {
					let st = nm.trim_start_matches("cache_params.past.").to_string();
					if st.starts_with("key") || st.starts_with("value") {
						let b = new_buf(&npu, &mut bufs, arg.bytes)?;
						past.push((st, b, b));
						(b, b)
					} else {
						let p0 = new_buf(&npu, &mut bufs, arg.bytes)?;
						let p1 = new_buf(&npu, &mut bufs, arg.bytes)?;
						past.push((st, p0, p1));
						flip.push((gi, i));
						(p0, p1)
					}
				}
				nm => return Err(format!("layer {gi}: unknown input {nm}")),
			};
			s0.push(b0);
			s1.push(b1);
		}
		sets[0].push(s0);
		sets[1].push(s1);
	}
	let (logits_i, vocab) = logits_b.ok_or("no logits")?;
	let lean = bias_b.is_some();
	let mask_i = mask_b.or(bias_b).ok_or("no attention layer")?;
	if lean {
		// every ring slot starts empty
		let neg = f16_bits(-30000.0).to_le_bytes();
		for c in bufs[mask_i].as_mut_slice().chunks_exact_mut(2) {
			c.copy_from_slice(&neg);
		}
	}
	let pos_i = pos_b.ok_or("no position_ids")?;
	println!("  {nlayers} layers + head loaded, {} KV windows (cap {cap}), {} state args flip, in {:.1} s", kv.len(), flip.len(), t0.elapsed().as_secs_f64());

	// The buffers are written by the host between runs through raw pointers
	// (the batch holds shared references for the list's lifetime).
	let ptrs: Vec<(*mut u8, usize)> = bufs
		.iter_mut()
		.map(|b| {
			let s = b.as_mut_slice();
			(s.as_mut_ptr(), s.len())
		})
		.collect();
	fn host_slice<'x>(p: (*mut u8, usize)) -> &'x mut [u8] {
		// SAFETY: a live NPU host allocation of p.1 bytes, only touched by the
		// host while no list runs.
		unsafe { std::slice::from_raw_parts_mut(p.0, p.1) }
	}
	let slice = |i: usize| host_slice(ptrs[i]);
	let plan0: Vec<(usize, Vec<&ze::HostBuf<'_>>)> = sets[0]
		.iter()
		.enumerate()
		.map(|(g, s)| (g, s.iter().map(|&i| &bufs[i]).collect()))
		.collect();
	let t1 = Instant::now();
	let mut batch = npu.batch_plan_mutable(&mut graphs, &plan0)?;
	println!(
		"  recorded 1 mutable list of {} appends in {:.2} s",
		plan0.len(),
		t1.elapsed().as_secs_f64()
	);

	let dump = a
		.iter()
		.position(|x| x == "--dump-logits")
		.and_then(|i| a.get(i + 1))
		.cloned();
	let mut dumpf = match &dump {
		Some(f) => Some(std::fs::File::create(f).map_err(|e| e.to_string())?),
		None => None,
	};
	let mut parity = 0usize;
	let mut out_ids: Vec<usize> = Vec::new();
	let mut tok = prompt[0];
	let total = prompt.len() + new_tokens;
	let mut step_t: Vec<f64> = Vec::new();
	let mut gen_wall = 0.0f64;
	for t in 0..total {
		let ts = Instant::now();
		if t < prompt.len() {
			tok = prompt[t];
		}
		read_emb(tok, slice(emb))?;
		// mask: past slots [cap - t, cap) valid (the last t tokens), new slot cap
		if !lean {
			let m = slice(mask_i);
			m.fill(0);
			for s in cap.saturating_sub(t)..=cap {
				m[s * 8..s * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
			}
		}
		{
			let p = slice(pos_i);
			for c in p.chunks_exact_mut(8) {
				c.copy_from_slice(&(t as i64).to_le_bytes());
			}
		}
		if t > 0 {
			let src = &sets[parity];
			let upd: Vec<(usize, usize, &ze::HostBuf<'_>)> = flip
				.iter()
				.map(|&(g, i)| (g, i, &bufs[src[g][i]]))
				.collect();
			batch.update(&upd)?;
		}
		let run = batch.run()?;
		step_t.push(run);
		// greedy
		let lg = slice(logits_i);
		let mut best = (0usize, f32::MIN);
		for (i, c) in lg[..vocab * 4].chunks_exact(4).enumerate() {
			let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
			if v > best.1 {
				best = (i, v);
			}
		}
		if let Some(f) = dumpf.as_mut() {
			use std::io::Write;
			f.write_all(&lg[..vocab * 4]).map_err(|e| e.to_string())?;
		}
		if lean {
			// ring: key row -> slot, value row -> column slot of Vt, open the slot
			let slot = t % cap;
			for &(q, p, is_v) in &kv {
				let qs = host_slice(ptrs[q]);
				let ps = host_slice(ptrs[p]);
				for h in 0..qs.len() / 512 {
					let src = &qs[h * 512..(h + 1) * 512];
					if is_v {
						for d in 0..256 {
							let o = ((h * 256 + d) * cap + slot) * 2;
							ps[o..o + 2].copy_from_slice(&src[d * 2..d * 2 + 2]);
						}
					} else {
						let o = (h * cap + slot) * 512;
						ps[o..o + 512].copy_from_slice(src);
					}
				}
			}
			slice(mask_i)[slot * 2..slot * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		} else {
			// slide every KV window by one row: past[h][0..cap] = present[h][1..=cap]
			let row = 256 * 2;
			std::thread::scope(|sc| {
				for &(q, p, _) in &kv {
					let (qp, ql) = (ptrs[q].0 as usize, ptrs[q].1);
					let (pp, pl) = (ptrs[p].0 as usize, ptrs[p].1);
					sc.spawn(move || {
						let heads = pl / (cap * row);
						// SAFETY: disjoint live buffers, no list running.
						let qs = unsafe { std::slice::from_raw_parts(qp as *const u8, ql) };
						let ps = unsafe { std::slice::from_raw_parts_mut(pp as *mut u8, pl) };
						for h in 0..heads {
							let s = (h * (cap + 1) + 1) * row;
							ps[h * cap * row..(h + 1) * cap * row]
								.copy_from_slice(&qs[s..s + cap * row]);
						}
					});
				}
			});
		}
		parity ^= 1;
		if t + 1 >= prompt.len() {
			tok = best.0;
			out_ids.push(tok);
			if t + 1 > prompt.len() {
				gen_wall += ts.elapsed().as_secs_f64();
			}
		}
	}
	let ng = out_ids.len().saturating_sub(1);
	let gen_runs: f64 = step_t[prompt.len()..].iter().sum();
	println!(
		"  prompt {} tokens in {:.2} s; generated {} tokens: NPU {:.1} ms/token, wall {:.1} ms/token -> {:.2} tok/s",
		prompt.len(),
		step_t[..prompt.len()].iter().sum::<f64>(),
		ng,
		gen_runs * 1e3 / ng.max(1) as f64,
		gen_wall * 1e3 / ng.max(1) as f64,
		ng as f64 / gen_wall.max(1e-9)
	);
	println!(
		"  ids: {}",
		out_ids
			.iter()
			.map(|v| v.to_string())
			.collect::<Vec<_>>()
			.join(" ")
	);
	Ok(())
}

/// Prompt-lookup drafts: the `want` tokens that followed the most recent
/// earlier occurrence of the longest suffix (`nmax` down to 1 tokens) of
/// `hist`; empty if there is none.
fn lookup_drafts(hist: &[usize], want: usize, nmax: usize) -> Vec<usize> {
	for n in (1..=nmax.min(hist.len().saturating_sub(1))).rev() {
		let pat = &hist[hist.len() - n..];
		// most recent match that ends before the current suffix
		for s in (0..hist.len() - n).rev() {
			if &hist[s..s + n] == pat {
				let f = s + n;
				let d: Vec<usize> = hist[f..(f + want).min(hist.len())].to_vec();
				if !d.is_empty() {
					return d;
				}
			}
		}
	}
	Vec::new()
}

/// `spec DIR --emb TABLE.f16 --prompt IDS.txt [--new N] [--dump-logits FILE]
/// [--ngram 3]`: greedy speculative decoding through K-position verify blobs
/// (`verify_cut.py` on `cut_all.py --s K`, compiled; DIR/outs.txt). Each step
/// runs [last token, K - 1 drafts] (prompt-lookup drafts from the context,
/// padded with the last token when short), accepts the drafts that match the
/// argmax chain, and rolls the states back to the accepted length:
/// - delta-net ssm: the graph outputs the state after every position
///   (`state{i}`) into one of two banks; the next step's past input is
///   re-pointed to bank[accepted - 1] (no copy);
/// - delta-net conv: the graph outputs the whole window (`conv_full`, past 4 +
///   K new columns); the host copies the 4 columns ending at the last
///   accepted position into the past buffer;
/// - attention KV: present has CAP + K rows per head; past = rows
///   [accepted, accepted + CAP).
///
/// The prompt runs K tokens per step (all accepted). Output must equal plain
/// greedy decoding (`gen`) up to numeric near-ties.
fn spec(a: &[String]) -> Result<(), String> {
	let dir = a.get(1).ok_or("spec DIR")?;
	let nlayers: usize = arg(a, "--layers", 64);
	let new_tokens: usize = arg(a, "--new", 16);
	let ngram: usize = arg(a, "--ngram", 3);
	// `--mtp DIR`: MTP drafts (DIR: head.blob with the `hn` output,
	// mtp_s{K}.blob, mtp_s1.blob, mtphead.blob; `cut_mtp.py`, `heads.py`)
	// instead of prompt-lookup drafts.
	let mtpd: Option<String> = a
		.iter()
		.position(|x| x == "--mtp")
		.and_then(|i| a.get(i + 1))
		.cloned();
	let embf: String = arg(a, "--emb", String::new());
	let promptf: String = arg(a, "--prompt", String::new());
	let prompt: Vec<usize> = std::fs::read_to_string(&promptf)
		.map_err(|e| format!("{promptf}: {e}"))?
		.split_whitespace()
		.map(|v| v.parse().map_err(|_| format!("bad id {v}")))
		.collect::<Result<_, _>>()?;
	if prompt.is_empty() {
		return Err("empty prompt".into());
	}
	let outs: Vec<Vec<String>> = {
		let t = std::fs::read_to_string(format!("{dir}/outs.txt"))
			.map_err(|e| format!("outs.txt: {e}"))?;
		t.lines()
			.map(|l| {
				l.split_whitespace()
					.skip(1)
					.map(|s| s.to_string())
					.collect()
			})
			.collect()
	};
	let (npu, _l) = open()?;
	let t0 = Instant::now();
	let names: Vec<String> = (0..nlayers)
		.map(|i| format!("layer{i}"))
		.chain(Some("head".to_string()))
		.collect();
	let raw: Vec<Vec<u8>> = names
		.iter()
		.map(|n| {
			let f = match (&mtpd, n.as_str()) {
				(Some(m), "head") => format!("{m}/head.blob"),
				_ => format!("{dir}/{n}.blob"),
			};
			std::fs::read(&f).map_err(|e| format!("{f}: {e}"))
		})
		.collect::<Result<_, _>>()?;
	let mut graphs: Vec<Graph<'_>> = raw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	let xa = graphs[0]
		.args
		.iter()
		.find(|a| a.name == "x")
		.ok_or("no x")?
		.clone();
	let kk = xa.dims[1] as usize;
	let mraw: Vec<Vec<u8>> = match &mtpd {
		Some(m) => [
			format!("mtp_s{kk}"),
			"mtp_s1".to_string(),
			"mtphead".to_string(),
		]
		.iter()
		.map(|n| std::fs::read(format!("{m}/{n}.blob")).map_err(|e| format!("{m}/{n}.blob: {e}")))
		.collect::<Result<_, _>>()?,
		None => Vec::new(),
	};
	let mut mgraphs: Vec<Graph<'_>> = mraw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	let k = xa.dims[1] as usize;
	let wrow = xa.bytes / k; // one position's residual (5120 f16)
	let table = std::fs::File::open(&embf).map_err(|e| format!("{embf}: {e}"))?;
	let read_emb = |id: usize, dst: &mut [u8]| -> Result<(), String> {
		use npu::ReadAt;
		table
			.seek_read(&mut dst[..wrow], (id * wrow) as u64)
			.map_err(|e| e.to_string())?;
		Ok(())
	};
	fn new_buf<'n>(
		npu: &'n Npu,
		bufs: &mut Vec<ze::HostBuf<'n>>,
		bytes: usize,
	) -> Result<usize, String> {
		let mut b = npu.alloc(bytes)?;
		b.as_mut_slice().fill(0);
		bufs.push(b);
		Ok(bufs.len() - 1)
	}
	let mut bufs: Vec<ze::HostBuf<'_>> = Vec::new();
	let emb = new_buf(&npu, &mut bufs, xa.bytes)?;
	let resid: Vec<usize> = (0..nlayers)
		.map(|_| new_buf(&npu, &mut bufs, xa.bytes))
		.collect::<Result<_, _>>()?;
	let (mut mask_b, mut pos_b, mut logits_b): (
		Option<usize>,
		Option<usize>,
		Option<(usize, usize)>,
	) = (None, None, None);
	let mut hn_b: Option<usize> = None;
	let mut plan: Vec<Vec<usize>> = Vec::new();
	// delta layers: (append, ssm past arg, [bank][i] (out arg, buffer)), zero buffer
	struct Delta {
		g: usize,
		past_arg: usize,
		zero: usize,
		out_args: Vec<usize>,
		banks: [Vec<usize>; 2],
		conv_past: usize,
		conv_full: usize,
		// deferred-rollback WY graphs (`wyrec.py`): arg indices and the
		// two buffer sets of (sbase, kcur / kp, dnew / dp)
		wy: Option<Wy>,
	}
	#[derive(Default, Clone)]
	struct Wy {
		sb_out: usize,
		k_in: usize,
		k_out: usize,
		d_in: usize,
		d_out: usize,
		sb: [usize; 2],
		kb: [usize; 2],
		db: [usize; 2],
		w: usize,
		alpha: usize,
		gcum: usize,
	}
	let mut deltas: Vec<Delta> = Vec::new();
	let mut kv: Vec<(usize, usize, bool)> = Vec::new(); // (present, past, is value)
	let mut kv_layer: Vec<usize> = Vec::new();
	let mut cap = 0usize;
	let mut conv_cols = 0usize;
	for gi in 0..=nlayers {
		let args = graphs[gi].args.clone();
		let is_head = gi == nlayers;
		let out_names: Vec<String> = if is_head {
			vec!["logits".into(), "hn".into()]
		} else {
			outs.get(gi).cloned().ok_or("outs.txt too short")?
		};
		let mut set = vec![usize::MAX; args.len()];
		let mut ko = 0;
		let mut d = Delta {
			g: gi,
			past_arg: usize::MAX,
			zero: 0,
			out_args: Vec::new(),
			banks: [Vec::new(), Vec::new()],
			conv_past: 0,
			conv_full: 0,
			wy: None,
		};
		let mut pasts: Vec<(String, usize)> = Vec::new();
		for (i, arg) in args.iter().enumerate() {
			if arg.is_output {
				let nm = out_names.get(ko).cloned().unwrap_or_default();
				ko += 1;
				set[i] = if is_head && nm == "hn" {
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					hn_b = Some(b);
					b
				} else if is_head {
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					logits_b = Some((b, arg.bytes / 4 / k));
					b
				} else if nm == "y" {
					resid[gi]
				} else if nm == "conv_full" {
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					d.conv_full = b;
					conv_cols = arg.dims[2] as usize;
					b
				} else if matches!(nm.as_str(), "sbase" | "kcur" | "dnew" | "gcum") {
					let wy = d.wy.get_or_insert_with(Wy::default);
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					match nm.as_str() {
						"sbase" => {
							wy.sb_out = i;
							wy.sb[1] = b;
						}
						"kcur" => {
							wy.k_out = i;
							wy.kb[1] = b;
						}
						"dnew" => {
							wy.d_out = i;
							wy.db[1] = b;
						}
						_ => wy.gcum = b,
					}
					b
				} else if let Some(si) = nm.strip_prefix("state") {
					let si: usize = si.parse().map_err(|_| format!("bad {nm}"))?;
					let b0 = new_buf(&npu, &mut bufs, arg.bytes)?;
					let b1 = new_buf(&npu, &mut bufs, arg.bytes)?;
					if d.banks[0].len() != si {
						return Err(format!("layer {gi}: states out of order"));
					}
					d.banks[0].push(b0);
					d.banks[1].push(b1);
					d.out_args.push(i);
					b0
				} else if let Some(st) = nm.strip_prefix("cache_params.present.") {
					let p = pasts
						.iter()
						.find(|(n, _)| n == st)
						.map(|x| x.1)
						.ok_or(format!("layer {gi}: no past for {nm}"))?;
					let q = new_buf(&npu, &mut bufs, arg.bytes)?;
					kv.push((q, p, st.starts_with("value")));
					kv_layer.push(gi);
					q
				} else {
					return Err(format!("layer {gi}: unknown output {nm}"));
				};
				continue;
			}
			set[i] = match arg.name.as_str() {
				"x" => {
					if gi == 0 {
						emb
					} else {
						resid[gi - 1]
					}
				}
				"attention_mask" => *mask_b.get_or_insert_with(|| {
					cap = arg.bytes / 8 - k;
					usize::MAX
				}),
				// lean attention (`attn_lean.py --vt`): the window is a ring
				"attn_bias" => *mask_b.get_or_insert_with(|| {
					cap = arg.bytes / 2;
					usize::MAX
				}),
				"position_ids" => *pos_b.get_or_insert(usize::MAX),
				"qwen_rs.kp" | "qwen_rs.dp" | "qwen_rs.w" | "qwen_rs.alpha" => {
					let wy = d.wy.get_or_insert_with(Wy::default);
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					match arg.name.as_str() {
						"qwen_rs.kp" => {
							wy.k_in = i;
							wy.kb[0] = b;
						}
						"qwen_rs.dp" => {
							wy.d_in = i;
							wy.db[0] = b;
						}
						"qwen_rs.w" => wy.w = b,
						_ => {
							// alpha = 1: the first step starts from sbase as is
							let one = f16_bits(1.0).to_le_bytes();
							for c in bufs[b].as_mut_slice().chunks_exact_mut(2) {
								c.copy_from_slice(&one);
							}
							wy.alpha = b;
						}
					}
					b
				}
				nm if nm.starts_with("cache_params.past.") => {
					let st = nm.trim_start_matches("cache_params.past.").to_string();
					let b = new_buf(&npu, &mut bufs, arg.bytes)?;
					if st.starts_with("ssm") {
						d.past_arg = i;
						d.zero = b;
					} else if st.starts_with("conv") {
						d.conv_past = b;
					}
					pasts.push((st, b));
					b
				}
				nm => return Err(format!("layer {gi}: unknown input {nm}")),
			};
		}
		if let Some(wy) = d.wy.as_mut() {
			wy.sb[0] = d.zero;
		}
		plan.push(set);
		if d.past_arg != usize::MAX {
			deltas.push(d);
		}
	}
	// the shared mask / position buffers (allocated once, patched into the plan)
	let lean = graphs
		.iter()
		.flat_map(|g| g.args.iter())
		.any(|a| a.name == "attn_bias");
	let mask_i = {
		let bytes = graphs
			.iter()
			.flat_map(|g| g.args.iter())
			.find(|a| a.name == "attention_mask" || a.name == "attn_bias")
			.map(|a| a.bytes)
			.ok_or("no mask")?;
		let b = new_buf(&npu, &mut bufs, bytes)?;
		if lean {
			let neg = f16_bits(-30000.0).to_le_bytes();
			for c in bufs[b].as_mut_slice().chunks_exact_mut(2) {
				c.copy_from_slice(&neg);
			}
		}
		b
	};
	let pos_i = {
		let bytes = graphs
			.iter()
			.flat_map(|g| g.args.iter())
			.find(|a| a.name == "position_ids")
			.map(|a| a.bytes)
			.ok_or("no position_ids")?;
		new_buf(&npu, &mut bufs, bytes)?
	};
	for (gi, set) in plan.iter_mut().enumerate() {
		for (i, a_) in graphs[gi].args.iter().enumerate() {
			if a_.name == "attention_mask" || a_.name == "attn_bias" {
				set[i] = mask_i;
			} else if a_.name == "position_ids" {
				set[i] = pos_i;
			}
		}
	}
	let (logits_i, vocab) = logits_b.ok_or("no logits")?;
	// MTP graphs: buffers by "tag.arg" (m4 = the K-position catch-up, m1 =
	// one chained draft, mh = the draft lm_head); outputs in result order.
	let mut mb: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
	let mut mset: Vec<Vec<usize>> = Vec::new();
	for (gi, g) in mgraphs.iter().enumerate() {
		let tag = ["m4", "m1", "mh"][gi];
		let onames: &[&str] = if gi == 2 {
			&["logits"]
		} else {
			&["h", "present_key", "present_value"]
		};
		let mut ko = 0;
		let mut set = Vec::new();
		for arg in &g.args {
			let key = if arg.is_output {
				ko += 1;
				format!("{tag}.{}", onames[ko - 1])
			} else {
				format!("{tag}.{}", arg.name)
			};
			let b = new_buf(&npu, &mut bufs, arg.bytes)?;
			mb.insert(key, b);
			set.push(b);
		}
		mset.push(set);
	}
	let mtp_on = !mgraphs.is_empty();
	// Row `r` of an MTP `h` output as f16 bytes (the compiled blobs keep that
	// output f32 even though the IR converts it to f16).
	fn h_row(src: &[u8], r: usize, w: usize, f32out: bool) -> Vec<u8> {
		if f32out {
			src[r * w * 2..(r + 1) * w * 2]
				.chunks_exact(4)
				.flat_map(|c| f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
				.collect()
		} else {
			src[r * w..(r + 1) * w].to_vec()
		}
	}
	let mvocab = if mtp_on {
		bufs[mb["mh.logits"]].len() / 4
	} else {
		0
	};
	let hn_f32 = mtp_on && hn_b.is_some_and(|b| bufs[b].len() == 2 * xa.bytes);
	let m4_f32 = mtp_on && bufs[mb["m4.h"]].len() == 2 * xa.bytes;
	let m1_f32 = mtp_on && bufs[mb["m1.h"]].len() == 2 * (xa.bytes / xa.dims[1] as usize);
	let hn_i = hn_b.unwrap_or(usize::MAX);
	if mtp_on && hn_b.is_none() {
		return Err("--mtp needs the head with the `hn` output".into());
	}
	println!(
		"  K = {k}: {nlayers} layers + head loaded, {} delta layers, {} KV windows (cap {cap}), conv window {conv_cols} cols, in {:.1} s",
		deltas.len(),
		kv.len(),
		t0.elapsed().as_secs_f64()
	);
	let ptrs: Vec<(*mut u8, usize)> = bufs
		.iter_mut()
		.map(|b| {
			let s = b.as_mut_slice();
			(s.as_mut_ptr(), s.len())
		})
		.collect();
	fn host_slice<'x>(p: (*mut u8, usize)) -> &'x mut [u8] {
		// SAFETY: a live NPU host allocation of p.1 bytes, only touched by the
		// host while no list runs.
		unsafe { std::slice::from_raw_parts_mut(p.0, p.1) }
	}
	let slice = |i: usize| host_slice(ptrs[i]);
	let plan_refs: Vec<(usize, Vec<&ze::HostBuf<'_>>)> = plan
		.iter()
		.enumerate()
		.map(|(g, s)| (g, s.iter().map(|&i| &bufs[i]).collect()))
		.collect();
	let t1 = Instant::now();
	let mut batch = npu.batch_plan_mutable(&mut graphs, &plan_refs)?;
	println!(
		"  recorded 1 mutable list of {} appends in {:.2} s",
		plan_refs.len(),
		t1.elapsed().as_secs_f64()
	);
	let mut mlists: Vec<ze::Batch<'_>> = Vec::new();
	let mut m1_past_args = (usize::MAX, usize::MAX);
	if mtp_on {
		for gi in 0..3 {
			let refs: Vec<&ze::HostBuf<'_>> = mset[gi].iter().map(|&i| &bufs[i]).collect();
			mlists.push(
				npu.batch_plan_mutable(std::slice::from_mut(&mut mgraphs[gi]), &[(0, refs)])?,
			);
		}
		let pk = mgraphs[1]
			.args
			.iter()
			.position(|x| x.name == "past_key")
			.ok_or("m1: no past_key")?;
		let pv = mgraphs[1]
			.args
			.iter()
			.position(|x| x.name == "past_value")
			.ok_or("m1: no past_value")?;
		m1_past_args = (pk, pv);
		println!(
			"  MTP drafter: K-position catch-up, chained single drafts, {mvocab}-row draft head"
		);
	}
	let dump = a
		.iter()
		.position(|x| x == "--dump-logits")
		.and_then(|i| a.get(i + 1))
		.cloned();
	let mut dumpf = match &dump {
		Some(f) => Some(std::fs::File::create(f).map_err(|e| e.to_string())?),
		None => None,
	};

	let argmax = |lg: &[u8], pos: usize| -> usize {
		let row = &lg[pos * vocab * 4..(pos + 1) * vocab * 4];
		let mut best = (0usize, f32::MIN);
		for (i, c) in row.chunks_exact(4).enumerate() {
			let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
			if v > best.1 {
				best = (i, v);
			}
		}
		best.0
	};
	let mut hist: Vec<usize> = Vec::new(); // committed tokens fed so far
	let mut next = prompt[0]; // the committed token not yet fed
	let mut pi = 1usize; // prompt tokens consumed into `next` / fed
	let mut t = 0usize; // positions in the states
					 // `--init DIR --start N`: start from a prefill of prompt[..N - 1]
					 // (`prefill --dump-states DIR`, one 256-position chunk, right-padded):
					 // delta ssm -> the start state, conv rows -> the 4-column window, attention
					 // key / value rows -> the ring (lean graphs only); MTP has no prompt context.
	if let Some(d) = a
		.iter()
		.position(|x| x == "--init")
		.and_then(|i| a.get(i + 1))
		.cloned()
	{
		let n: usize = arg(a, "--start", 0);
		if n < 2 || n > prompt.len() || !lean {
			return Err(
				"--init needs --start N (2 <= N <= prompt length) and lean attention graphs".into(),
			);
		}
		let rd = |l: usize, k: usize| -> Result<Vec<u8>, String> {
			let f = format!("{d}/L{l}.o{k}.out");
			std::fs::read(&f).map_err(|e| format!("{f}: {e}"))
		};
		for dl in &deltas {
			let conv = rd(dl.g, 0)?; // [1, 3, C] token-major
			let ssm = rd(dl.g, 1)?;
			let sb = dl.wy.as_ref().map(|w| w.sb[0]).unwrap_or(dl.zero);
			// the prefill graphs keep the export's state convention (x vmul:
			// `--init-vscale L0,REST`, default 16,256 = QWEN_OV_DSCALE's), the
			// decode cuts an unscaled state
			let vs: Vec<f32> = arg::<String>(a, "--init-vscale", "16,256".into())
				.split(',')
				.filter_map(|v| v.parse().ok())
				.collect();
			let inv = 1.0 / if dl.g == 0 { vs[0] } else { vs[vs.len() - 1] };
			let dst = slice(sb);
			for (o, c) in dst[..ssm.len()]
				.chunks_exact_mut(2)
				.zip(ssm.chunks_exact(2))
			{
				let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]])) * inv;
				o.copy_from_slice(&f16_bits(v).to_le_bytes());
			}
			let cp = slice(dl.conv_past); // [1, C, 4], column 3 the newest
			let ch = conv.len() / 2 / 3;
			for c in 0..ch {
				cp[(c * 4) * 2..(c * 4) * 2 + 2].fill(0);
				for r in 0..3 {
					let o = ((c * 4) + 1 + r) * 2;
					cp[o..o + 2].copy_from_slice(&conv[(r * ch + c) * 2..(r * ch + c) * 2 + 2]);
				}
			}
		}
		let row = 512;
		for (&(_, p, is_v), &l) in kv.iter().zip(kv_layer.iter()) {
			let src = rd(l, if is_v { 1 } else { 0 })?; // [1, S, 4, 256] token-major
			let ps = slice(p);
			let heads = 4;
			for pos in 0..n - 1 {
				for h in 0..heads {
					let r = &src[(pos * heads + h) * row..(pos * heads + h + 1) * row];
					if is_v {
						for d_ in 0..256 {
							let o = ((h * 256 + d_) * cap + pos) * 2;
							ps[o..o + 2].copy_from_slice(&r[d_ * 2..d_ * 2 + 2]);
						}
					} else {
						let o = (h * cap + pos) * row;
						ps[o..o + row].copy_from_slice(r);
					}
				}
			}
		}
		let m = slice(mask_i);
		for pos in 0..n - 1 {
			m[pos * 2..pos * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
		}
		hist.extend_from_slice(&prompt[..n - 1]);
		next = prompt[n - 1];
		pi = n;
		t = n - 1;
		println!("  started from {d}: {} prompt positions prefilled", n - 1);
	}
	let mut bank = 0usize;
	// state input of each delta layer: (bank, index) or zero
	let mut generated: Vec<usize> = Vec::new();
	let (mut steps, mut gen_steps, mut accepted_drafts, mut proposed) =
		(0usize, 0usize, 0usize, 0usize);
	let mut gen_time = 0.0f64;
	let mut prompt_time = 0.0f64;
	let mut npu_time = 0.0f64;
	let mut draft_time = 0.0f64;
	let mut mt_last = t; // last position the MTP KV holds (positions start at 1)
	let mut mt_count = 0usize;
	let mut mtp_drafts: Vec<usize> = Vec::new();
	while generated.len() < new_tokens {
		let ts = Instant::now();
		let in_prompt = pi < prompt.len();
		// inputs: next + K-1 followers (prompt tokens, or drafts)
		let mut inp = vec![next];
		#[allow(clippy::needless_late_init)]
		let nreal_prompt;
		if in_prompt {
			while inp.len() < k && pi < prompt.len() {
				inp.push(prompt[pi]);
				pi += 1;
			}
			nreal_prompt = inp.len();
		} else {
			let d = if mtp_on {
				mtp_drafts.clone()
			} else {
				let mut h = hist.clone();
				h.push(next);
				lookup_drafts(&h, k - 1, ngram)
			};
			proposed += d.len();
			inp.extend(d);
			nreal_prompt = 0;
		}
		let ndraft_real = inp.len();
		while inp.len() < k {
			inp.push(next);
		}
		for (j, &id) in inp.iter().enumerate() {
			read_emb(id, &mut slice(emb)[j * wrow..(j + 1) * wrow])?;
		}
		{
			if !lean {
				let m = slice(mask_i);
				m.fill(0);
				for s in cap.saturating_sub(t)..cap + k {
					m[s * 8..s * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
				}
			}
			let p = slice(pos_i);
			let rows = p.len() / 8 / k; // [4, 1, K]
			for r in 0..rows {
				for j in 0..k {
					let o = (r * k + j) * 8;
					p[o..o + 8].copy_from_slice(&((t + j) as i64).to_le_bytes());
				}
			}
		}
		let tr = Instant::now();
		batch.run()?;
		npu_time += tr.elapsed().as_secs_f64();
		steps += 1;
		let lg = slice(logits_i);
		if let Some(f) = dumpf.as_mut() {
			use std::io::Write;
			f.write_all(&lg[..k * vocab * 4])
				.map_err(|e| e.to_string())?;
		}
		// acceptance
		let acc;
		let mut emitted = Vec::new();
		if in_prompt {
			acc = nreal_prompt;
			if pi >= prompt.len() {
				// the prompt is done: the last real position predicts the first token
				emitted.push(argmax(lg, acc - 1));
			}
		} else {
			let mut a_ = 1;
			let mut cur = argmax(lg, 0);
			while a_ < ndraft_real && inp[a_] == cur {
				emitted.push(cur);
				a_ += 1;
				cur = argmax(lg, a_ - 1);
			}
			emitted.push(cur);
			if std::env::var("NPU_SPEC_DEBUG").is_ok() {
				println!(
					"    step t={t}: fed {:?} -> argmax {:?}",
					inp,
					(0..k).map(|j| argmax(lg, j)).collect::<Vec<_>>()
				);
			}
			accepted_drafts += a_ - 1;
			acc = a_;
			gen_steps += 1;
		}
		// commit the fed tokens 0..acc
		hist.extend_from_slice(&inp[..acc]);
		t += acc;
		// roll back / advance the states to position acc - 1
		let wy_mode = deltas.first().is_some_and(|d| d.wy.is_some());
		if wy_mode {
			// deferred: the next step forms S_(acc-1) from this step's sbase and
			// factors; the host only sets alpha / w and swaps the buffer sets
			let j = acc - 1;
			for d in &deltas {
				let wy = d.wy.as_ref().unwrap();
				let gb = slice(wy.gcum);
				let heads = slice(wy.alpha).len() / 2;
				let f32g = gb.len() == heads * k * 4;
				let gv = |h: usize, m: usize| -> f32 {
					if f32g {
						let o = (h * k + m) * 4;
						f32::from_le_bytes([gb[o], gb[o + 1], gb[o + 2], gb[o + 3]])
					} else {
						let o = (h * k + m) * 2;
						f16_to_f32(u16::from_le_bytes([gb[o], gb[o + 1]]))
					}
				};
				let al = slice(wy.alpha);
				let ww = slice(wy.w);
				for h in 0..heads {
					let gj = gv(h, j);
					al[h * 2..h * 2 + 2].copy_from_slice(&f16_bits(gj.exp()).to_le_bytes());
					for m in 0..k {
						let v = if m <= j { (gj - gv(h, m)).exp() } else { 0.0 };
						ww[(h * k + m) * 2..(h * k + m) * 2 + 2]
							.copy_from_slice(&f16_bits(v).to_le_bytes());
					}
				}
			}
			// parity: inputs read set `p`, outputs write set `p ^ 1`
			let p = bank ^ 1;
			let upd: Vec<(usize, usize, &ze::HostBuf<'_>)> = deltas
				.iter()
				.flat_map(|d| {
					let wy = d.wy.as_ref().unwrap();
					[
						(d.g, d.past_arg, &bufs[wy.sb[p]]),
						(d.g, wy.k_in, &bufs[wy.kb[p]]),
						(d.g, wy.d_in, &bufs[wy.db[p]]),
						(d.g, wy.sb_out, &bufs[wy.sb[p ^ 1]]),
						(d.g, wy.k_out, &bufs[wy.kb[p ^ 1]]),
						(d.g, wy.d_out, &bufs[wy.db[p ^ 1]]),
					]
				})
				.collect();
			batch.update(&upd)?;
			bank ^= 1;
		} else {
			let upd: Vec<(usize, usize, &ze::HostBuf<'_>)> = deltas
				.iter()
				.flat_map(|d| {
					let src = &bufs[d.banks[bank][acc - 1]];
					let outs_next: Vec<(usize, usize, &ze::HostBuf<'_>)> = d
						.out_args
						.iter()
						.zip(d.banks[bank ^ 1].iter())
						.map(|(&ai, &b)| (d.g, ai, &bufs[b]))
						.collect();
					std::iter::once((d.g, d.past_arg, src)).chain(outs_next)
				})
				.collect();
			batch.update(&upd)?;
			bank ^= 1;
		}
		// conv windows and KV windows (host copies, in parallel)
		{
			let row = 256 * 2;
			let cc = conv_cols;
			std::thread::scope(|sc| {
				for d in &deltas {
					let (fp, fl) = (ptrs[d.conv_full].0 as usize, ptrs[d.conv_full].1);
					let (pp, pl) = (ptrs[d.conv_past].0 as usize, ptrs[d.conv_past].1);
					sc.spawn(move || {
						// SAFETY: disjoint live buffers, no list running.
						let f = unsafe { std::slice::from_raw_parts(fp as *const u16, fl / 2) };
						let p = unsafe { std::slice::from_raw_parts_mut(pp as *mut u16, pl / 2) };
						let w = pl / 2 / (fl / 2 / cc); // past window columns (4)
						let ch = fl / 2 / cc;
						for c in 0..ch {
							p[c * w..(c + 1) * w]
								.copy_from_slice(&f[c * cc + acc..c * cc + acc + w]);
						}
					});
				}
				for &(q, p, is_v) in &kv {
					let (qp, ql) = (ptrs[q].0 as usize, ptrs[q].1);
					let (pp, pl) = (ptrs[p].0 as usize, ptrs[p].1);
					if lean {
						// ring: the accepted rows 0..acc go to slots (t - acc + j) % cap
						sc.spawn(move || {
							// SAFETY: disjoint live buffers, no list running.
							let qs = unsafe { std::slice::from_raw_parts(qp as *const u8, ql) };
							let ps = unsafe { std::slice::from_raw_parts_mut(pp as *mut u8, pl) };
							let heads = ql / (k * row);
							for j in 0..acc {
								let slot = (t - acc + j) % cap;
								for h in 0..heads {
									let src = &qs[(h * k + j) * row..(h * k + j + 1) * row];
									if is_v {
										for d_ in 0..256 {
											let o = ((h * 256 + d_) * cap + slot) * 2;
											ps[o..o + 2].copy_from_slice(&src[d_ * 2..d_ * 2 + 2]);
										}
									} else {
										let o = (h * cap + slot) * row;
										ps[o..o + row].copy_from_slice(src);
									}
								}
							}
						});
						continue;
					}
					sc.spawn(move || {
						let heads = pl / (cap * row);
						// SAFETY: disjoint live buffers, no list running.
						let qs = unsafe { std::slice::from_raw_parts(qp as *const u8, ql) };
						let ps = unsafe { std::slice::from_raw_parts_mut(pp as *mut u8, pl) };
						for h in 0..heads {
							let s = (h * (cap + k) + acc) * row;
							ps[h * cap * row..(h + 1) * cap * row]
								.copy_from_slice(&qs[s..s + cap * row]);
						}
					});
				}
			});
		}
		if lean {
			let m = slice(mask_i);
			for j in 0..acc {
				let slot = (t - acc + j) % cap;
				m[slot * 2..slot * 2 + 2].copy_from_slice(&0u16.to_le_bytes());
			}
		}
		if let Some(&last) = emitted.last() {
			next = last;
		} else {
			next = prompt[pi];
			pi += 1;
		}
		if mtp_on {
			let tm = Instant::now();
			let t_old = t - acc;
			let c = t - mt_last; // MTP positions mt_last + 1 ..= t (t: the token `next`)
			let row = 256 * 2;
			{
				let hn = slice(hn_i);
				let he = slice(mb["m4.hidden"]);
				for j in 0..k {
					let jj = j.min(c - 1);
					let p = mt_last + 1 + jj;
					let tokp = if p < t { hist[p] } else { next };
					read_emb(tokp, &mut slice(mb["m4.emb"])[j * wrow..(j + 1) * wrow])?;
					let hr = p - 1 - t_old;
					he[j * wrow..(j + 1) * wrow].copy_from_slice(&h_row(hn, hr, wrow, hn_f32));
				}
				let m = slice(mb["m4.attention_mask"]);
				m.fill(0);
				for s_ in cap.saturating_sub(mt_count)..cap + k {
					m[s_ * 8..s_ * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
				}
				let pp = slice(mb["m4.position_ids"]);
				for j in 0..k {
					pp[j * 8..j * 8 + 8].copy_from_slice(&((mt_last + 1 + j) as i64).to_le_bytes());
				}
			}
			mlists[0].run()?;
			// the committed MTP window: present rows [c, c + cap)
			for (q, p_) in [
				("m4.present_key", "m4.past_key"),
				("m4.present_value", "m4.past_value"),
			] {
				let qs = slice(mb[q]);
				let ps = slice(mb[p_]);
				let heads = ps.len() / (cap * row);
				for h in 0..heads {
					let s_ = (h * (cap + k) + c) * row;
					ps[h * cap * row..(h + 1) * cap * row].copy_from_slice(&qs[s_..s_ + cap * row]);
				}
			}
			mt_count = (mt_count + c).min(cap);
			mt_last = t;
			mtp_drafts.clear();
			if pi >= prompt.len() {
				let draft_argmax = |lg: &[u8]| -> usize {
					let mut best = (0usize, f32::MIN);
					for (i, c4) in lg[..mvocab * 4].chunks_exact(4).enumerate() {
						let v = f32::from_le_bytes([c4[0], c4[1], c4[2], c4[3]]);
						if v > best.1 {
							best = (i, v);
						}
					}
					best.0
				};
				// d0 from the catch-up's last real row
				let h4 = h_row(slice(mb["m4.h"]), c - 1, wrow, m4_f32);
				slice(mb["mh.x"])[..wrow].copy_from_slice(&h4);
				mlists[2].run()?;
				mtp_drafts.push(draft_argmax(slice(mb["mh.logits"])));
				slice(mb["m1.hidden"])[..wrow].copy_from_slice(&h4);
				for i in 1..k - 1 {
					read_emb(mtp_drafts[i - 1], slice(mb["m1.emb"]))?;
					if i > 1 {
						let h1 = h_row(slice(mb["m1.h"]), 0, wrow, m1_f32);
						slice(mb["m1.hidden"])[..wrow].copy_from_slice(&h1);
					}
					let (pk, pv) = if i == 1 {
						("m4.past_key", "m4.past_value")
					} else {
						("m1.past_key", "m1.past_value")
					};
					mlists[1].update(&[
						(0, m1_past_args.0, &bufs[mb[pk]]),
						(0, m1_past_args.1, &bufs[mb[pv]]),
					])?;
					let valid = (mt_count + i - 1).min(cap);
					{
						let m = slice(mb["m1.attention_mask"]);
						m.fill(0);
						for s_ in cap - valid..=cap {
							m[s_ * 8..s_ * 8 + 8].copy_from_slice(&1i64.to_le_bytes());
						}
						slice(mb["m1.position_ids"])[..8]
							.copy_from_slice(&((t + i) as i64).to_le_bytes());
					}
					mlists[1].run()?;
					let h1 = h_row(slice(mb["m1.h"]), 0, wrow, m1_f32);
					slice(mb["mh.x"])[..wrow].copy_from_slice(&h1);
					mlists[2].run()?;
					mtp_drafts.push(draft_argmax(slice(mb["mh.logits"])));
					if i + 1 < k - 1 {
						// the next chained draft sees this one's KV row
						for (q, p_) in [
							("m1.present_key", "m1.past_key"),
							("m1.present_value", "m1.past_value"),
						] {
							let qs = slice(mb[q]);
							let ps = slice(mb[p_]);
							let heads = ps.len() / (cap * row);
							for h in 0..heads {
								let s_ = (h * (cap + 1) + 1) * row;
								ps[h * cap * row..(h + 1) * cap * row]
									.copy_from_slice(&qs[s_..s_ + cap * row]);
							}
						}
					}
				}
			}
			draft_time += tm.elapsed().as_secs_f64();
		}
		let dt = ts.elapsed().as_secs_f64();
		if in_prompt {
			prompt_time += dt;
		} else {
			gen_time += dt;
		}
		if !in_prompt || pi >= prompt.len() {
			generated.extend(emitted.iter().copied());
		}
	}
	generated.truncate(new_tokens);
	if mtp_on {
		println!(
			"  MTP: {:.1} ms per step (catch-up + {} chained drafts)",
			draft_time * 1e3 / steps.max(1) as f64,
			k - 2
		);
	}
	println!(
		"  prompt {} tokens in {prompt_time:.2} s; {} tokens in {gen_steps} verify steps ({:.2} tokens/step, drafts accepted {accepted_drafts}/{proposed}): {:.0} ms/step -> {:.2} tok/s (NPU {:.0} ms/step)",
		prompt.len(),
		generated.len(),
		generated.len() as f64 / gen_steps.max(1) as f64,
		gen_time * 1e3 / gen_steps.max(1) as f64,
		generated.len() as f64 / gen_time.max(1e-9),
		npu_time * 1e3 / steps.max(1) as f64
	);
	println!(
		"  ids: {}",
		generated
			.iter()
			.map(|v| v.to_string())
			.collect::<Vec<_>>()
			.join(" ")
	);
	Ok(())
}

/// `prefill DIR --inputs INDIR [--layers 64] [--iters 5] [--dump FILE]`: one
/// lean prefill chunk through every layer `DIR/lean{L}` (`build_delta.py` /
/// `build_attn.py`, compiled with `compile_lean.sh`) in ONE command list.
/// A layer with `DIR/lean{L}.use` runs the blob named there (`dedupe.py`:
/// byte-identical `--allin` graphs), so each distinct blob is loaded once and
/// appended per layer with that layer's buffers (the NPU holds only ~27
/// distinct delta-net blobs). The residual runs through a distinct buffer per
/// layer boundary (`INDIR/x.in` -> b0 -> b1 ...); an input with a file
/// `DIR/lean{L}.NAME.in` (weights, per-layer constants) gets its own buffer
/// filled from it; other inputs are read from `INDIR/NAME.in` (mask / cos /
/// sin / valid / keep, one buffer shared by all layers) or zeroed (the past
/// states: a first chunk). Outputs other than the residual get their own
/// buffers. `--dump FILE`: the final residual (raw f16), for the head on the
/// host (`lean_chain.py --resid FILE`).
fn prefill(a: &[String]) -> Result<(), String> {
	let dir = a.get(1).ok_or("prefill DIR")?;
	let ind = arg::<String>(a, "--inputs", String::new());
	let nlayers: usize = arg(a, "--layers", 64);
	let iters: usize = arg(a, "--iters", 5);
	let (npu, _l) = open()?;
	let t0 = Instant::now();
	// `--share N`: at most N layers per loaded instance of a shared blob
	// (another instance of the same blob after that).
	let share: usize = arg(a, "--share", usize::MAX);
	let mut names: Vec<String> = Vec::new();
	let mut uses: Vec<usize> = Vec::new();
	let mut layer_blob = Vec::new();
	// `--only-delta`: the first `--layers` delta-net layers (1, 2, 4, ...)
	// chained, not the model, a uniform graph sequence for `--mutable`.
	let lid: Vec<usize> = if flag(a, "--only-delta") {
		(1..64).filter(|l| l % 4 != 3).take(nlayers).collect()
	} else {
		(0..nlayers).collect()
	};
	for &l in &lid {
		let u = std::path::Path::new(dir).join(format!("lean{l}.use"));
		let nm = if u.exists() {
			std::fs::read_to_string(&u)
				.map_err(|e| e.to_string())?
				.trim()
				.to_string()
		} else {
			format!("lean{l}")
		};
		let gi = match names
			.iter()
			.enumerate()
			.position(|(i, n)| *n == nm && uses[i] < share)
		{
			Some(i) => i,
			None => {
				names.push(nm);
				uses.push(0);
				names.len() - 1
			}
		};
		uses[gi] += 1;
		layer_blob.push(gi);
	}
	let raw: Vec<Vec<u8>> = names
		.iter()
		.map(|n| std::fs::read(format!("{dir}/{n}.blob")).map_err(|e| format!("{n}: {e}")))
		.collect::<Result<_, _>>()?;
	// `raw` stays alive: the driver reads the blob bytes at initialize time.
	let mut graphs: Vec<Graph<'_>> = raw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	println!(
		"  {nlayers} layers, {} distinct blobs, loaded in {:.1} s",
		graphs.len(),
		t0.elapsed().as_secs_f64()
	);
	let w = graphs[layer_blob[0]]
		.args
		.iter()
		.find(|a| a.name == "x")
		.map(|a| a.bytes)
		.ok_or("no x")?;
	let read_file = |f: &std::path::Path, bytes: usize| -> Result<Vec<u8>, String> {
		let d = std::fs::read(f).map_err(|e| e.to_string())?;
		if d.len() != bytes {
			return Err(format!("{}: {} bytes, want {bytes}", f.display(), d.len()));
		}
		Ok(d)
	};
	// buffer arena; the plan refers to it by index
	let mut bufs: Vec<ze::HostBuf<'_>> = Vec::new();
	fn new_buf<'n>(
		npu: &'n Npu,
		bufs: &mut Vec<ze::HostBuf<'n>>,
		bytes: usize,
		data: Option<Vec<u8>>,
	) -> Result<usize, String> {
		let mut b = npu.alloc(bytes)?;
		match data {
			Some(d) => b.as_mut_slice()[..bytes].copy_from_slice(&d),
			None => b.as_mut_slice().fill(0),
		}
		bufs.push(b);
		Ok(bufs.len() - 1)
	}
	let x0 = {
		let f = std::path::Path::new(&ind).join("x.in");
		let d = if !ind.is_empty() && f.exists() {
			Some(read_file(&f, w)?)
		} else {
			None
		};
		new_buf(&npu, &mut bufs, w, d)?
	};
	let resid: Vec<usize> = (0..nlayers)
		.map(|_| new_buf(&npu, &mut bufs, w, None))
		.collect::<Result<_, _>>()?;
	let mut shared: Vec<(String, usize, usize)> = Vec::new();
	let mut plan_idx: Vec<(usize, Vec<usize>)> = Vec::new();
	let mut weight_bytes = 0usize;
	for l in 0..nlayers {
		let gi = layer_blob[l];
		let args = graphs[gi].args.clone();
		let mut set = Vec::new();
		for arg in &args {
			let bi = if arg.is_output {
				if arg.bytes == w {
					resid[l]
				} else {
					new_buf(&npu, &mut bufs, arg.bytes, None)?
				}
			} else if arg.name == "x" {
				if l == 0 {
					x0
				} else {
					resid[l - 1]
				}
			} else {
				let own = std::path::Path::new(dir).join(format!("lean{}.{}.in", lid[l], arg.name));
				let sh = std::path::Path::new(&ind).join(format!("{}.in", arg.name));
				if own.exists() {
					weight_bytes += arg.bytes;
					new_buf(
						&npu,
						&mut bufs,
						arg.bytes,
						Some(read_file(&own, arg.bytes)?),
					)?
				} else if !ind.is_empty() && sh.exists() {
					match shared
						.iter()
						.find(|(n, s, _)| *n == arg.name && *s == arg.bytes)
					{
						Some(&(_, _, bi)) => bi,
						None => {
							let bi = new_buf(
								&npu,
								&mut bufs,
								arg.bytes,
								Some(read_file(&sh, arg.bytes)?),
							)?;
							shared.push((arg.name.clone(), arg.bytes, bi));
							bi
						}
					}
				} else {
					new_buf(&npu, &mut bufs, arg.bytes, None)?
				}
			};
			set.push(bi);
		}
		plan_idx.push((gi, set));
	}
	println!(
		"  {:.1} GB of per-layer inputs, {} buffers, ready in {:.1} s",
		weight_bytes as f64 / 1e9,
		bufs.len(),
		t0.elapsed().as_secs_f64()
	);
	// `--hold DIR`: also load (initialize) every decode blob of DIR
	// (layer{0..63}.blob + head.blob) and keep them resident while prefill
	// runs; whether the decode set's constants and the prefill set's bound
	// weights fit on the device together (no weight-set switch per phase).
	let hold_raw: Vec<Vec<u8>> = match a
		.iter()
		.position(|x| x == "--hold")
		.and_then(|i| a.get(i + 1))
	{
		Some(d) => (0..64)
			.map(|l| format!("{d}/layer{l}.blob"))
			.chain(std::iter::once(format!("{d}/head.blob")))
			.map(|f| std::fs::read(&f).map_err(|e| format!("{f}: {e}")))
			.collect::<Result<_, _>>()?,
		None => Vec::new(),
	};
	let th = Instant::now();
	let mut _held: Vec<Graph<'_>> = hold_raw
		.iter()
		.map(|b| npu.load(b, 0))
		.collect::<Result<_, _>>()?;
	for g in _held.iter_mut() {
		g.initialize()?;
	}
	if !hold_raw.is_empty() {
		println!(
			"  holding {} decode blobs ({:.1} GB) loaded in {:.1} s",
			hold_raw.len(),
			hold_raw.iter().map(|b| b.len()).sum::<usize>() as f64 / 1e9,
			th.elapsed().as_secs_f64()
		);
	}
	let plan: Vec<(usize, Vec<&ze::HostBuf<'_>>)> = plan_idx
		.iter()
		.map(|(g, s)| (*g, s.iter().map(|&i| &bufs[i]).collect()))
		.collect();
	// `--split K`: the layers as K command lists, each recorded, run and
	// destroyed in turn (the driver's per-append descriptor copies of all
	// lists share one heap, which holds only ~25 delta-net appends). The
	// time is the sum of the lists' runs; recording is reported apart.
	let split: usize = arg(a, "--split", 1);
	let per = nlayers.div_ceil(split);
	let mut first = 0.0;
	let mut t = vec![0.0f64; iters];
	let mut rec = 0.0;
	if flag(a, "--per-layer") {
		// `--per-layer`: one mutable command list per distinct blob with a
		// single append, recorded once; each layer re-points its blob's list
		// to that layer's buffers and submits it (the whole model in 64
		// submissions, nothing re-recorded per chunk, and only as many appends
		// in the driver's descriptor heap as there are distinct blobs).
		// `--ring R`: R lists per blob used more than once, so layer l + 1 is
		// re-pointed and queued while layer l runs (in-order queue); a list
		// is waited for only when its turn comes round again.
		let ring: usize = arg(a, "--ring", 1);
		let t1 = Instant::now();
		let mut nuse = vec![0usize; graphs.len()];
		for (gi, _) in plan.iter() {
			nuse[*gi] += 1;
		}
		// lists[gi][r], with the layer each is currently pointed at
		let mut lists: Vec<Vec<ze::Batch<'_>>> = (0..graphs.len()).map(|_| Vec::new()).collect();
		let mut cur: Vec<Vec<usize>> = (0..graphs.len()).map(|_| Vec::new()).collect();
		for (l, (gi, set)) in plan.iter().enumerate() {
			if lists[*gi].len() < ring.min(nuse[*gi]) {
				lists[*gi].push(npu.batch_plan_mutable(
					std::slice::from_mut(&mut graphs[*gi]),
					&[(0, set.clone())],
				)?);
				cur[*gi].push(l);
			}
		}
		rec = t1.elapsed().as_secs_f64();
		let mut upd = 0.0f64;
		let mut busy: Vec<Vec<bool>> = lists.iter().map(|v| vec![false; v.len()]).collect();
		let mut pass = |lists: &mut Vec<Vec<ze::Batch<'_>>>| -> Result<f64, String> {
			let t = Instant::now();
			let mut k = vec![0usize; lists.len()];
			for (l, (gi, set)) in plan.iter().enumerate() {
				let r = k[*gi] % lists[*gi].len();
				k[*gi] += 1;
				let b = &mut lists[*gi][r];
				if busy[*gi][r] {
					b.wait()?;
					busy[*gi][r] = false;
				}
				if cur[*gi][r] != l {
					let tu = Instant::now();
					let args: Vec<(usize, usize, &ze::HostBuf<'_>)> =
						set.iter().enumerate().map(|(i, b)| (0, i, *b)).collect();
					b.update(&args)?;
					upd += tu.elapsed().as_secs_f64();
					cur[*gi][r] = l;
				}
				b.submit()?;
				busy[*gi][r] = true;
			}
			for (gi, v) in lists.iter_mut().enumerate() {
				for (r, b) in v.iter_mut().enumerate() {
					if busy[gi][r] {
						b.wait()?;
						busy[gi][r] = false;
					}
				}
			}
			Ok(t.elapsed().as_secs_f64())
		};
		first = pass(&mut lists)?;
		for v in t.iter_mut() {
			*v = pass(&mut lists)?;
		}
		println!(
			"  record {rec:.2} s ({} mutable lists); updates {:.1} ms per pass",
			lists.iter().map(|v| v.len()).sum::<usize>(),
			upd * 1e3 / (iters + 1) as f64
		);
	} else if flag(a, "--mutable") {
		// `--mutable`: ONE mutable command list holding the first part's
		// appends, re-pointed (zeCommandListUpdateMutableCommandsExp) to each
		// part in turn: recorded once, no per-chunk re-recording. Every part
		// must repeat part 0's graph sequence. The time includes the updates.
		let parts: Vec<&[(usize, Vec<&ze::HostBuf<'_>>)]> = plan.chunks(per).collect();
		for p in &parts {
			if p.len() != per || p.iter().zip(parts[0].iter()).any(|(x, y)| x.0 != y.0) {
				return Err("--mutable: every part must repeat part 0's graph sequence (use --layers divisible by --split)".into());
			}
		}
		let t1 = Instant::now();
		let mut batch = npu.batch_plan_mutable(&mut graphs, parts[0])?;
		rec = t1.elapsed().as_secs_f64();
		let mut cur = 0usize;
		let mut upd = 0.0f64;
		let mut nupd = 0usize;
		let mut pass = |batch: &mut ze::Batch<'_>| -> Result<f64, String> {
			let mut tot = 0.0;
			for (pi, p) in parts.iter().enumerate() {
				if pi != cur {
					let tu = Instant::now();
					// NPU_MUT_NOOP: re-point to part 0's own buffers; NPU_MUT_ARGS=i,j: only those
					let src = if std::env::var("NPU_MUT_NOOP").is_ok() {
						parts[0]
					} else {
						*p
					};
					let only: Option<Vec<usize>> = std::env::var("NPU_MUT_ARGS")
						.ok()
						.map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
					let args: Vec<(usize, usize, &ze::HostBuf<'_>)> = src
						.iter()
						.enumerate()
						.flat_map(|(k, (_, set))| {
							set.iter().enumerate().map(move |(i, b)| (k, i, *b))
						})
						.filter(|(_, i, _)| only.as_ref().is_none_or(|o| o.contains(i)))
						.collect();
					batch.update(&args)?;
					let du = tu.elapsed().as_secs_f64();
					upd += du;
					nupd += 1;
					tot += du;
					cur = pi;
				}
				tot += batch.run()?;
			}
			Ok(tot)
		};
		first = pass(&mut batch)?;
		for v in t.iter_mut() {
			*v = pass(&mut batch)?;
		}
		println!(
			"  record {rec:.2} s (1 mutable list of {per}); {nupd} updates, {:.2} ms each",
			upd * 1e3 / nupd.max(1) as f64
		);
	} else {
		for part in plan.chunks(per) {
			let t1 = Instant::now();
			let mut batch = npu.batch_plan(&mut graphs, part)?;
			rec += t1.elapsed().as_secs_f64();
			first += batch.run()?;
			for v in t.iter_mut() {
				*v += batch.run()?;
			}
		}
		println!("  init + record {rec:.2} s ({split} lists)");
	}
	t.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let s = w / (5120 * 2);
	println!(
		"  prefill: {nlayers} layers, {s} positions: first {:.1} ms, min {:.1} ms ({:.2} ms/layer), median {:.1} ms -> {:.1} tok/s",
		first * 1e3,
		t[0] * 1e3,
		t[0] * 1e3 / nlayers as f64,
		t[t.len() / 2] * 1e3,
		s as f64 / t[t.len() / 2]
	);
	// `--dump-layers DIR`: every layer's output residual as DIR/res{L}.f16
	if let Some(d) = a
		.iter()
		.position(|x| x == "--dump-layers")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
		for (l, &bi) in resid.iter().enumerate() {
			std::fs::write(
				std::path::Path::new(d).join(format!("res{l}.f16")),
				&bufs[bi].as_slice()[..w],
			)
			.map_err(|e| e.to_string())?;
		}
	}
	if let Some(f) = a
		.iter()
		.position(|x| x == "--dump")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(f, &bufs[resid[nlayers - 1]].as_slice()[..w]).map_err(|e| e.to_string())?;
	}
	// `--dump-states DIR`: every layer's outputs other than the residual, in
	// output order, as DIR/L{l}.o{k}.out (delta: conv_out, ssm_out;
	// attention: knew, vnew), plus the last residual as DIR/res.f16; the
	// start state of a decode (`spec --init DIR`).
	if let Some(d) = a
		.iter()
		.position(|x| x == "--dump-states")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
		for (l, (gi, set)) in plan_idx.iter().enumerate() {
			let mut k = 0;
			for (arg, &bi) in graphs[*gi].args.iter().zip(set.iter()) {
				if arg.is_output && bi != resid[l] {
					std::fs::write(
						std::path::Path::new(d).join(format!("L{l}.o{k}.out")),
						&bufs[bi].as_slice()[..arg.bytes],
					)
					.map_err(|e| e.to_string())?;
					k += 1;
				}
			}
		}
		std::fs::write(
			std::path::Path::new(d).join("res.f16"),
			&bufs[resid[nlayers - 1]].as_slice()[..w],
		)
		.map_err(|e| e.to_string())?;
	}
	Ok(())
}

/// `wtest BLOB N`: N instances of one blob in one command list, every input
/// but `x` (the weights, when they are graph inputs) its own host buffer;
/// whether input weights escape the ~14 GB limit on resident graph constants.
fn wtest(a: &[String]) -> Result<(), String> {
	let path = a.get(1).ok_or("wtest BLOB N")?;
	let n: usize = a
		.get(2)
		.and_then(|v| v.parse().ok())
		.ok_or("wtest BLOB N")?;
	let b = std::fs::read(path).map_err(|e| e.to_string())?;
	// `--fill V`: the byte the buffers are filled with (0 may stay unbacked).
	let fillv: u8 = arg(a, "--fill", 0);
	let (npu, _l) = open()?;
	let mut graphs: Vec<Graph<'_>> = (0..n).map(|_| npu.load(&b, 0)).collect::<Result<_, _>>()?;
	let xa = graphs[0]
		.args
		.iter()
		.find(|a| a.name == "x")
		.map(|a| a.bytes)
		.unwrap_or(1);
	let x = npu.alloc(xa)?;
	let mut owned = Vec::new();
	let mut total = 0usize;
	for g in graphs.iter_mut() {
		for i in 0..g.args.len() {
			let arg = g.args[i].clone();
			if arg.name == "x" && !arg.is_output {
				g.bind(i, &x)?;
				continue;
			}
			let mut buf = npu.alloc(arg.bytes)?;
			buf.as_mut_slice().fill(fillv);
			if !arg.is_output {
				total += arg.bytes;
			}
			g.bind(i, &buf)?;
			owned.push(buf);
		}
	}
	println!(
		"  {n} instances, {:.1} GB of non-x inputs",
		total as f64 / 1e9
	);
	let t0 = Instant::now();
	let mut refs: Vec<&mut Graph<'_>> = graphs.iter_mut().collect();
	let mut batch = npu.batch(&mut refs)?;
	let ti = t0.elapsed().as_secs_f64();
	batch.run()?;
	println!(
		"  init {:.2} s, first run done {:.2} s",
		ti,
		t0.elapsed().as_secs_f64()
	);
	let mut t: Vec<f64> = (0..3).map(|_| batch.run()).collect::<Result<_, _>>()?;
	t.sort_by(|a, b| a.partial_cmp(b).unwrap());
	println!(
		"  min {:.1} ms ({:.2} ms/instance)",
		t[0] * 1e3,
		t[0] * 1e3 / n as f64
	);
	Ok(())
}

/// `rebind BLOB [N]`: one instance of BLOB appended N times (default 2) in one
/// command list, each with its own noise inputs and output buffers, against
/// N separate runs of the same sets; prints whether every append computed
/// its own set (the arguments are captured at append time).
fn rebind(a: &[String]) -> Result<(), String> {
	let path = a.get(1).ok_or("rebind BLOB [N]")?;
	let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(2);
	let b = std::fs::read(path).map_err(|e| e.to_string())?;
	let (npu, _l) = open()?;
	let mut g = npu.load(&b, 0)?;
	let args = g.args.clone();
	let mut sets: Vec<Vec<ze::HostBuf<'_>>> = Vec::new();
	for k in 0..n {
		let mut v = Vec::new();
		for (i, arg) in args.iter().enumerate() {
			let mut buf = npu.alloc(arg.bytes)?;
			if arg.is_output {
				buf.as_mut_slice().fill(0);
			} else if arg.precision == 0x02 {
				let r = npu::noise(arg.bytes, 100 * k as u64 + i as u64 + 1);
				for (j, c) in buf.as_mut_slice().chunks_exact_mut(2).enumerate() {
					c.copy_from_slice(
						&f16_bits((r[j % r.len()] as f32 - 127.5) / 512.0).to_le_bytes(),
					);
				}
			} else {
				buf.as_mut_slice().fill(0);
			}
			v.push(buf);
		}
		sets.push(v);
	}
	let refs: Vec<Vec<&ze::HostBuf<'_>>> = sets.iter().map(|v| v.iter().collect()).collect();
	// `--lists K`: the N appends split over K command lists that all exist
	// at once (whether the per-append footprint is per list or global).
	let lists: usize = arg(a, "--lists", 1);
	if lists > 1 {
		let per = n.div_ceil(lists);
		let mut bs = Vec::new();
		for c in refs.chunks(per) {
			bs.push(npu.batch_sets(&mut g, c)?);
			println!("  list {} recorded ({} appends)", bs.len(), c.len());
		}
		let t = Instant::now();
		for b in bs.iter_mut() {
			b.run()?;
		}
		println!(
			"  {lists} lists run: {:.1} ms",
			t.elapsed().as_secs_f64() * 1e3
		);
		return Ok(());
	}
	let mut batch = npu.batch_sets(&mut g, &refs)?;
	let t = batch.run()?;
	let t2 = batch.run()?;
	println!(
		"  {n} appends of one instance: {:.1} ms / {:.1} ms",
		t * 1e3,
		t2 * 1e3
	);
	let outs: Vec<usize> = args
		.iter()
		.enumerate()
		.filter(|(_, a)| a.is_output)
		.map(|(i, _)| i)
		.collect();
	let together: Vec<Vec<Vec<u8>>> = sets
		.iter()
		.map(|v| {
			outs.iter()
				.map(|&i| v[i].as_slice()[..args[i].bytes].to_vec())
				.collect()
		})
		.collect();
	drop(batch);
	for k in 0..n {
		for &i in &outs {
			sets[k][i].as_mut_slice().fill(0);
		}
		let one = vec![sets[k].iter().collect::<Vec<_>>()];
		let mut bk = npu.batch_sets(&mut g, &one)?;
		bk.run()?;
		let same = outs
			.iter()
			.enumerate()
			.all(|(o, &i)| sets[k][i].as_slice()[..args[i].bytes] == together[k][o][..]);
		let zero = together[k].iter().all(|v| v.iter().all(|&x| x == 0));
		println!(
			"  set {k}: batched == separate: {same}{}",
			if zero {
				" (batched output all zero!)"
			} else {
				""
			}
		);
	}
	Ok(())
}

fn blob(a: &[String]) -> Result<(), String> {
	let path = a.get(1).ok_or("blob FILE")?;
	let iters: usize = arg(a, "--iters", 20);
	let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
	let (npu, _l) = open()?;
	let gflags = if flag(a, "--profile") {
		ze::GRAPH_FLAG_ENABLE_PROFILING
	} else {
		0
	};
	let t = Instant::now();
	let mut g = npu.load(&b, gflags)?;
	println!(
		"  loaded {} bytes in {:.2} s",
		b.len(),
		t.elapsed().as_secs_f64()
	);
	let bufs = bind_all(&npu, &mut g)?;
	time(&mut g, iters)?;
	// `--check N`: N more runs on the same inputs, each output compared
	// byte for byte with the first (a graph must repeat exactly)
	let check: usize = arg(a, "--check", 0);
	if check > 0 {
		let outs: Vec<usize> = (0..g.args.len()).filter(|&i| g.args[i].is_output).collect();
		let first: Vec<Vec<u8>> = outs.iter().map(|&i| bufs[i].as_slice().to_vec()).collect();
		let mut bad = 0;
		for r in 0..check {
			g.run()?;
			let d: Vec<usize> = outs
				.iter()
				.zip(&first)
				.map(|(&i, f)| {
					bufs[i]
						.as_slice()
						.iter()
						.zip(f)
						.filter(|(x, y)| x != y)
						.count()
				})
				.collect();
			if d.iter().any(|&x| x > 0) {
				bad += 1;
				println!("  run {r}: differing bytes per output {d:?}");
			}
		}
		println!("  --check: {bad} of {check} runs differ from the first");
	}
	print_outputs(&g, &bufs, a)?;
	if flag(a, "--profile") {
		profile(&mut g)?;
	}
	Ok(())
}

/// First values of every f16 output; `--dump FILE` writes all outputs raw.
fn print_outputs(g: &Graph<'_>, bufs: &[ze::HostBuf<'_>], a: &[String]) -> Result<(), String> {
	let mut all = Vec::new();
	for (arg, b) in g.args.iter().zip(bufs) {
		if !arg.is_output {
			continue;
		}
		let s = &b.as_slice()[..arg.bytes];
		all.extend_from_slice(s);
		if arg.precision == 0x02 {
			let v: Vec<f32> = s
				.chunks_exact(2)
				.take(6)
				.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
				.collect();
			println!("  {} [..6] = {v:?}", arg.name);
		}
	}
	if let Some(p) = a
		.iter()
		.position(|x| x == "--dump")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::write(p, &all).map_err(|e| e.to_string())?;
	}
	// `--dump-dir DIR`: every output as DIR/NAME.out (raw).
	if let Some(d) = a
		.iter()
		.position(|x| x == "--dump-dir")
		.and_then(|i| a.get(i + 1))
	{
		for (arg, b) in g.args.iter().zip(bufs) {
			if arg.is_output {
				let nm: String = arg
					.name
					.chars()
					.map(|c| {
						if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
							c
						} else {
							'_'
						}
					})
					.collect();
				std::fs::write(
					std::path::Path::new(d).join(format!("{nm}.out")),
					&b.as_slice()[..arg.bytes],
				)
				.map_err(|e| e.to_string())?;
			}
		}
	}
	Ok(())
}

/// `par BLOB [--n 2] [--iters 20]`: N instances of one blob, each on its own
/// Level Zero context / queue (one `Npu` each), timed back to back and then
/// submitted together; whether independent work overlaps on the NPU (e.g.
/// two one-tile blobs, `NPU_RS_TILES=1` at compile time).
fn par(a: &[String]) -> Result<(), String> {
	let path = a.get(1).ok_or("par BLOB")?;
	let n: usize = arg(a, "--n", 2);
	let iters: usize = arg(a, "--iters", 20);
	let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
	let (first, _l) = open()?;
	let mut npus = vec![first];
	for _ in 1..n {
		npus.push(Npu::open()?);
	}
	let mut graphs: Vec<Graph<'_>> = npus
		.iter()
		.map(|u| u.load(&b, 0))
		.collect::<Result<_, _>>()?;
	let mut bufs = Vec::new();
	for (u, g) in npus.iter().zip(graphs.iter_mut()) {
		bufs.push(bind_all(u, g)?);
	}
	let mut batches: Vec<ze::Batch<'_>> = Vec::new();
	for (u, g) in npus.iter().zip(graphs.iter_mut()) {
		batches.push(u.batch(&mut [g])?);
	}
	let med = |mut v: Vec<f64>| {
		v.sort_by(|x, y| x.partial_cmp(y).unwrap());
		v[v.len() / 2]
	};
	for bt in batches.iter_mut() {
		bt.run()?;
	}
	let mut seq = Vec::new();
	let mut con = Vec::new();
	for _ in 0..iters {
		let t = Instant::now();
		for bt in batches.iter_mut() {
			bt.run()?;
		}
		seq.push(t.elapsed().as_secs_f64());
		let t = Instant::now();
		for bt in batches.iter_mut() {
			bt.submit()?;
		}
		for bt in batches.iter_mut() {
			bt.wait()?;
		}
		con.push(t.elapsed().as_secs_f64());
	}
	println!(
		"  {n} instances: back to back {:.3} ms, concurrent {:.3} ms (median)",
		med(seq) * 1e3,
		med(con) * 1e3
	);
	Ok(())
}

/// `pf DIR --emb TABLE.f16 --prompt IDS.txt [--n N] [--small DIR64]
/// [--dump-states ST]`: prefill the first N prompt tokens (default all)
/// through the lean graphs of DIR (`build_all.sh`: lean{L}.blob / .use /
/// per-layer .in files) in chunks of the graphs' S positions, the last one
/// right-padded. `--small DIR64`: a second set built for 64 positions
/// (`--s 64 --c 64`, the same weights): a tail of up to 128 tokens runs as
/// 64-position chunks (a 64-position delta layer is ~19 ms vs ~52 at 256).
/// The sets share the bound weights, the delta states and the K / V windows.
/// Between chunks the delta conv / ssm outputs become the next chunk's inputs
/// (ping-pong buffers) and the attention layers' new K / V rows are appended
/// on the host to their past windows (token-major [1, CAP, 4, 256],
/// positions as rows: prompts up to CAP + S tokens). Per chunk the host
/// writes x, the mask (past rows < pos open, causal inside), cos / sin (theta
/// 1e7, 64 rotary dims), valid and keep. `--dump-states ST` writes the final
/// states like `prefill --dump-states` (attention: the window rows 0..N) for
/// `spec --init ST --start N + 1`.
fn pf(a: &[String]) -> Result<(), String> {
	let dir = a.get(1).ok_or("pf DIR")?.clone();
	let embf: String = arg(a, "--emb", String::new());
	let promptf: String = arg(a, "--prompt", String::new());
	let prompt: Vec<usize> = std::fs::read_to_string(&promptf)
		.map_err(|e| format!("{promptf}: {e}"))?
		.split_whitespace()
		.map(|v| v.parse().map_err(|_| format!("bad id {v}")))
		.collect::<Result<_, _>>()?;
	let n: usize = arg(a, "--n", prompt.len());
	let ring: usize = arg(a, "--ring", 3);
	let small: String = arg(a, "--small", String::new());
	// `--long DIR4096`: the main set built with a 4096-row attention window,
	// for chunks that start past the main set's window
	let long: String = arg(a, "--long", String::new());
	let nlayers = 64usize;
	const W: usize = 5120;
	let (npu, _l) = open()?;
	let t0 = Instant::now();
	let dirs: Vec<String> = std::iter::once(dir.clone())
		.chain((!small.is_empty()).then(|| small.clone()))
		.chain((!long.is_empty()).then(|| long.clone()))
		.collect();
	let small_i = (!small.is_empty()).then_some(1usize);
	let long_i = (!long.is_empty()).then_some(dirs.len() - 1);
	// per set: distinct blobs and the layer -> blob map
	let mut set_names: Vec<Vec<String>> = Vec::new();
	let mut set_lb: Vec<Vec<usize>> = Vec::new();
	for d in &dirs {
		let mut names: Vec<String> = Vec::new();
		let mut lb = Vec::new();
		for l in 0..nlayers {
			let u = std::path::Path::new(d).join(format!("lean{l}.use"));
			let nm = if u.exists() {
				std::fs::read_to_string(&u)
					.map_err(|e| e.to_string())?
					.trim()
					.to_string()
			} else {
				format!("lean{l}")
			};
			let gi = match names.iter().position(|x| *x == nm) {
				Some(i) => i,
				None => {
					names.push(nm);
					names.len() - 1
				}
			};
			lb.push(gi);
		}
		set_names.push(names);
		set_lb.push(lb);
	}
	let raws: Vec<Vec<Vec<u8>>> = dirs
		.iter()
		.zip(set_names.iter())
		.map(|(d, names)| {
			names
				.iter()
				.map(|nm| {
					std::fs::read(format!("{d}/{nm}.blob")).map_err(|e| format!("{d}/{nm}: {e}"))
				})
				.collect::<Result<Vec<_>, _>>()
		})
		.collect::<Result<_, _>>()?;
	let mut graphs: Vec<Vec<Graph<'_>>> = raws
		.iter()
		.map(|r| {
			r.iter()
				.map(|b| npu.load(b, 0))
				.collect::<Result<Vec<_>, _>>()
		})
		.collect::<Result<_, _>>()?;
	let svals: Vec<usize> = graphs
		.iter()
		.map(|g| {
			g[0].args
				.iter()
				.find(|x| x.name == "x")
				.map(|x| x.bytes / (W * 2))
				.unwrap_or(0)
		})
		.collect();
	let smax = *svals.iter().max().unwrap();
	// windows are allocated for the largest cap; a smaller-window graph reads
	// their first rows (token-major, positions as rows)
	let caps: Vec<usize> = graphs
		.iter()
		.map(|g| {
			g.iter()
				.flat_map(|x| x.args.iter())
				.find(|x| x.name == "kpast")
				.map(|x| x.dims[1] as usize)
				.unwrap_or(0)
		})
		.collect();
	let maxcap = *caps.iter().max().unwrap();
	let table = std::fs::File::open(&embf).map_err(|e| format!("{embf}: {e}"))?;
	let mut bufs: Vec<ze::HostBuf<'_>> = Vec::new();
	fn nb<'n>(
		npu: &'n Npu,
		bufs: &mut Vec<ze::HostBuf<'n>>,
		bytes: usize,
		data: Option<Vec<u8>>,
	) -> Result<usize, String> {
		let mut b = npu.alloc(bytes)?;
		match data {
			Some(d) => b.as_mut_slice()[..bytes].copy_from_slice(&d),
			None => b.as_mut_slice().fill(0),
		}
		bufs.push(b);
		Ok(bufs.len() - 1)
	}
	// shared across sets (sized for the largest S): residuals, weights by
	// (layer, name), delta state ping-pong, K / V windows and new rows
	let x0 = nb(&npu, &mut bufs, smax * W * 2, None)?;
	let resid: Vec<usize> = (0..nlayers)
		.map(|_| nb(&npu, &mut bufs, smax * W * 2, None))
		.collect::<Result<_, _>>()?;
	let mut weights: std::collections::HashMap<(usize, String), usize> = Default::default();
	let mut st_in: std::collections::HashMap<(usize, String), usize> = Default::default();
	struct Dl {
		args: [usize; 4],
		cb: [usize; 2],
		sb: [usize; 2],
	}
	struct Set {
		sets: Vec<Vec<usize>>,
		dls: Vec<Option<Dl>>,
		atts: Vec<Option<[usize; 4]>>,
		shared: std::collections::HashMap<String, usize>,
		s: usize,
	}
	let mut csets: Vec<Set> = Vec::new();
	for (si, gs) in graphs.iter().enumerate() {
		let mut st = Set {
			sets: Vec::new(),
			dls: Vec::new(),
			atts: Vec::new(),
			shared: Default::default(),
			s: svals[si],
		};
		for l in 0..nlayers {
			let args = gs[set_lb[si][l]].args.clone();
			let mut set = Vec::new();
			let mut outs_k = 0;
			let mut dl = Dl {
				args: [usize::MAX; 4],
				cb: [0; 2],
				sb: [0; 2],
			};
			let mut at = [usize::MAX; 4];
			for (i, arg) in args.iter().enumerate() {
				let bi = if arg.is_output {
					if arg.bytes == svals[si] * W * 2 {
						resid[l]
					} else {
						// outputs after y: delta conv_out, ssm_out; attention knew, vnew
						let key = (l, format!("out{outs_k}"));
						let b = match st_in.get(&key) {
							Some(&b) => b,
							None => {
								let sz = if l % 4 == 3 {
									arg.bytes / svals[si] * smax
								} else {
									arg.bytes
								};
								let b = nb(&npu, &mut bufs, sz, None)?;
								st_in.insert(key, b);
								b
							}
						};
						if l % 4 == 3 {
							at[2 + outs_k] = b;
						} else if outs_k == 0 {
							dl.args[2] = i;
							dl.cb[1] = b;
						} else {
							dl.args[3] = i;
							dl.sb[1] = b;
						}
						outs_k += 1;
						b
					}
				} else {
					let nm = arg.name.clone();
					match nm.as_str() {
						"x" => {
							if l == 0 {
								x0
							} else {
								resid[l - 1]
							}
						}
						"conv" | "ssm" | "kpast" | "vpast" => {
							let key = (l, nm.clone());
							let b = match st_in.get(&key) {
								Some(&b) => b,
								None => {
									let sz = if nm == "kpast" || nm == "vpast" {
										arg.bytes / arg.dims[1] as usize * maxcap
									} else {
										arg.bytes
									};
									let b = nb(&npu, &mut bufs, sz, None)?;
									st_in.insert(key, b);
									b
								}
							};
							match nm.as_str() {
								"conv" => {
									dl.args[0] = i;
									dl.cb[0] = b;
								}
								"ssm" => {
									dl.args[1] = i;
									dl.sb[0] = b;
								}
								"kpast" => {
									at[0] = b;
								}
								_ => at[1] = b,
							}
							b
						}
						_ => {
							let own =
								std::path::Path::new(&dirs[si]).join(format!("lean{l}.{nm}.in"));
							if own.exists() {
								match weights.get(&(l, nm.clone())) {
									Some(&b) => b,
									None => {
										let d = std::fs::read(&own).map_err(|e| e.to_string())?;
										let b = nb(&npu, &mut bufs, arg.bytes, Some(d))?;
										weights.insert((l, nm.clone()), b);
										b
									}
								}
							} else {
								match st.shared.get(&nm) {
									Some(&b) => b,
									None => {
										let b = nb(&npu, &mut bufs, arg.bytes, None)?;
										st.shared.insert(nm.clone(), b);
										b
									}
								}
							}
						}
					}
				};
				set.push(bi);
			}
			st.sets.push(set);
			if l % 4 == 3 {
				st.atts.push(Some(at));
				st.dls.push(None);
			} else {
				st.atts.push(None);
				st.dls.push(Some(dl));
			}
		}
		csets.push(st);
	}
	println!("  windows {caps:?}");
	println!(
		"  sets {:?} positions, {} distinct blobs, buffers ready in {:.1} s",
		svals,
		graphs.iter().map(|g| g.len()).sum::<usize>(),
		t0.elapsed().as_secs_f64()
	);
	// chunk plan: full chunks of the main set (or the long one once the
	// chunk starts past the main window); a tail of <= 2 small chunks goes to
	// the small set while it starts inside the small set's window
	let mut plan: Vec<(usize, usize, usize)> = Vec::new(); // (set, p0, real)
	let mut p0 = 0;
	while p0 < n {
		let r = n - p0;
		let si = match (small_i, long_i) {
			(Some(si), _) if r <= 2 * csets[si].s && p0 <= caps[si] => si,
			_ if p0 <= caps[0] => 0,
			(_, Some(li)) if p0 <= caps[li] => li,
			_ => return Err(format!("chunk at {p0}: past every window ({caps:?})")),
		};
		let real = r.min(csets[si].s);
		plan.push((si, p0, real));
		p0 += real;
	}
	let cap = maxcap;
	let ptrs: Vec<(*mut u8, usize)> = bufs
		.iter_mut()
		.map(|b| {
			let s = b.as_mut_slice();
			(s.as_mut_ptr(), s.len())
		})
		.collect();
	fn hs<'x>(p: (*mut u8, usize)) -> &'x mut [u8] {
		// SAFETY: a live NPU host allocation, touched by the host only while no list runs.
		unsafe { std::slice::from_raw_parts_mut(p.0, p.1) }
	}
	let sl = |i: usize| hs(ptrs[i]);
	// per set and blob: mutable lists (a ring of up to `ring`), re-pointed per layer
	let mut lists: Vec<Vec<Vec<ze::Batch<'_>>>> = Vec::new();
	for (si, gs) in graphs.iter_mut().enumerate() {
		let mut nuse = vec![0usize; gs.len()];
		for &gi in &set_lb[si] {
			nuse[gi] += 1;
		}
		let mut ls: Vec<Vec<ze::Batch<'_>>> = (0..gs.len()).map(|_| Vec::new()).collect();
		// one list per blob at a time: each gets its own slice of `gs`
		let mut gref: Vec<Option<&mut Graph<'_>>> = gs.iter_mut().map(Some).collect();
		for (l, &gi) in set_lb[si].iter().enumerate() {
			if ls[gi].len() < ring.min(nuse[gi]) {
				let refs: Vec<&ze::HostBuf<'_>> =
					csets[si].sets[l].iter().map(|&i| &bufs[i]).collect();
				let g = gref[gi].as_deref_mut().unwrap();
				ls[gi].push(npu.batch_plan_mutable(std::slice::from_mut(g), &[(0, refs)])?);
			}
		}
		lists.push(ls);
	}
	let mut busy: Vec<Vec<Vec<bool>>> = lists
		.iter()
		.map(|ls| ls.iter().map(|v| vec![false; v.len()]).collect())
		.collect();
	let mut cur: Vec<Vec<Vec<(usize, usize)>>> = lists
		.iter()
		.map(|ls| {
			ls.iter()
				.map(|v| vec![(usize::MAX, usize::MAX); v.len()])
				.collect()
		})
		.collect();
	let inv: Vec<f64> = (0..32)
		.map(|i| 1e7f64.powf(-((2 * i) as f64) / 64.0))
		.collect();
	let f16b = |v: f32| f16_bits(v).to_le_bytes();
	let mut par = 0usize;
	let tp = Instant::now();
	for &(si, p0, real) in &plan {
		let st = &csets[si];
		let s = st.s;
		{
			let x = sl(x0);
			x[..s * W * 2].fill(0);
			use npu::ReadAt;
			for j in 0..real {
				table
					.seek_read(
						&mut x[j * W * 2..(j + 1) * W * 2],
						(prompt[p0 + j] * W * 2) as u64,
					)
					.map_err(|e| e.to_string())?;
			}
			if let Some(&m) = st.shared.get("mask") {
				let mk = sl(m);
				let cap = caps[si];
				let row = cap + s;
				let (zero, neg) = (f16b(0.0), f16b(-30000.0));
				for i in 0..s {
					for k in 0..row {
						let open = if k < cap { k < p0 } else { k - cap <= i };
						let o = (i * row + k) * 2;
						mk[o..o + 2].copy_from_slice(if open { &zero } else { &neg });
					}
				}
			}
			for (nm, f) in [("cos", 0usize), ("sin", 1)] {
				if let Some(&b) = st.shared.get(nm) {
					let t = sl(b);
					for i in 0..s {
						for d in 0..64 {
							let ang = (p0 + i) as f64 * inv[d % 32];
							let v = if f == 0 { ang.cos() } else { ang.sin() };
							t[(i * 64 + d) * 2..(i * 64 + d) * 2 + 2]
								.copy_from_slice(&f16b(v as f32));
						}
					}
				}
			}
			if let Some(&b) = st.shared.get("valid") {
				let t = sl(b);
				for i in 0..s {
					t[i * 2..i * 2 + 2].copy_from_slice(&f16b(if i < real { 1.0 } else { 0.0 }));
				}
			}
			if let Some(&b) = st.shared.get("keep") {
				let t = sl(b);
				for j in 0..3 {
					t[j * 4..j * 4 + 4].copy_from_slice(&((real + j) as i32).to_le_bytes());
				}
			}
		}
		// the 64 layers; delta states read set `par`, write `par ^ 1`
		let ls = &mut lists[si];
		let mut k = vec![0usize; ls.len()];
		for l in 0..nlayers {
			let gi = set_lb[si][l];
			let r = k[gi] % ls[gi].len();
			k[gi] += 1;
			if busy[si][gi][r] {
				ls[gi][r].wait()?;
				busy[si][gi][r] = false;
			}
			if cur[si][gi][r] != (l, par) {
				let mut set = st.sets[l].clone();
				if let Some(d) = &st.dls[l] {
					set[d.args[0]] = d.cb[par];
					set[d.args[1]] = d.sb[par];
					set[d.args[2]] = d.cb[par ^ 1];
					set[d.args[3]] = d.sb[par ^ 1];
				}
				let upd: Vec<(usize, usize, &ze::HostBuf<'_>)> = set
					.iter()
					.enumerate()
					.map(|(i, &b)| (0, i, &bufs[b]))
					.collect();
				ls[gi][r].update(&upd)?;
				cur[si][gi][r] = (l, par);
			}
			ls[gi][r].submit()?;
			busy[si][gi][r] = true;
		}
		for (gi, v) in ls.iter_mut().enumerate() {
			for (r, b) in v.iter_mut().enumerate() {
				if busy[si][gi][r] {
					b.wait()?;
					busy[si][gi][r] = false;
				}
			}
		}
		par ^= 1;
		// append the new K / V rows to each attention window
		let rowb = 4 * 256 * 2;
		for at in st.atts.iter().flatten() {
			for (src, dst) in [(at[2], at[0]), (at[3], at[1])] {
				let m = real.min(cap.saturating_sub(p0));
				let (s_, d_) = (sl(src), sl(dst));
				d_[p0 * rowb..(p0 + m) * rowb].copy_from_slice(&s_[..m * rowb]);
			}
		}
	}
	let dt = tp.elapsed().as_secs_f64();
	let shape: Vec<usize> = plan.iter().map(|p| csets[p.0].s).collect();
	println!(
		"  prefill {n} tokens in chunks {shape:?}: {dt:.2} s -> {:.1} tok/s",
		n as f64 / dt
	);
	if let Some(d) = a
		.iter()
		.position(|x| x == "--dump-states")
		.and_then(|i| a.get(i + 1))
	{
		std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
		let rowb = 4 * 256 * 2;
		let st = &csets[0];
		for l in 0..nlayers {
			let w = |k: usize, b: &[u8]| {
				std::fs::write(std::path::Path::new(d).join(format!("L{l}.o{k}.out")), b)
					.map_err(|e| e.to_string())
			};
			if let Some(dl) = &st.dls[l] {
				w(0, sl(dl.cb[par]))?;
				w(1, sl(dl.sb[par]))?;
			} else if let Some(at) = &st.atts[l] {
				w(0, &sl(at[0])[..n.min(cap) * rowb])?;
				w(1, &sl(at[1])[..n.min(cap) * rowb])?;
			}
		}
	}
	Ok(())
}
