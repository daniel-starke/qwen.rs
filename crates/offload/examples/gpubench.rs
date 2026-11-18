//! GPU fused dequant-matmul microbenchmark.
//!
//! GPU matmul speed is a *throughput* question, but without this the only way to measure GPU matmul throughput was a full 67 GiB model load
//! plus a 16-token generation: minutes per data point, with the CPU path, the
//! aux kernels and the driver all mixed into the number. This isolates the fused
//! `q4k`/`q6k` shaders on synthetic weights of the real model's shapes and
//! reports achieved GB/s and GFLOP/s per weight, so a shader edit can be judged
//! in seconds.
//!
//! Weight bytes are synthesized from a deterministic PRNG. Any byte pattern is a
//! valid Q4_K/Q6_K block and the kernels are branch-free w.r.t. the data, so the
//! timing is representative; the `--check` pass additionally compares the GPU
//! result against the CPU `gemv_q4k`/`gemv_q6k` on the *same* bytes, which makes
//! this a correctness gate for shader edits as well.
//!
//! Run (needs a Vulkan device):
//!   cargo run --release -p offload --features vulkan --example gpubench
//!   cargo run --release -p offload --features vulkan --example gpubench -- --iters 50 --seq 1

use offload::{BatchJobSeq, OffloadManager, WeightId};
use quant::{GgmlType, LinearWeight};
use std::time::Instant;

/// xoshiro-ish deterministic byte source (no rand dependency, matching the
/// project's no-external-crates policy).
struct Rng(u64);
impl Rng {
	fn next_u32(&mut self) -> u32 {
		// splitmix64
		self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
		let mut z = self.0;
		z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
		((z ^ (z >> 31)) >> 32) as u32
	}
	fn fill(&mut self, buf: &mut [u8]) {
		for c in buf.chunks_mut(4) {
			let v = self.next_u32().to_le_bytes();
			c.copy_from_slice(&v[..c.len()]);
		}
	}
}

const QK_K: usize = 256;
const Q4K_BLOCK: usize = 144;
const Q6K_BLOCK: usize = 210;

/// Build synthetic Q4_K blocks. The two f16 header words (`d`, `dmin`) are given
/// small sane magnitudes so the accumulated dot products stay in a normal float
/// range; everything else is random, which is exactly what the kernel sees.
fn synth_q4k(rng: &mut Rng, out_dim: usize, in_dim: usize) -> Vec<u8> {
	let nb = out_dim * in_dim / QK_K;
	let mut raw = vec![0u8; nb * Q4K_BLOCK];
	for b in 0..nb {
		let o = b * Q4K_BLOCK;
		rng.fill(&mut raw[o + 4..o + Q4K_BLOCK]);
		let d = quant::f32_to_fp16(0.01 + (rng.next_u32() % 100) as f32 * 1e-4);
		let dmin = quant::f32_to_fp16(0.005 + (rng.next_u32() % 100) as f32 * 5e-5);
		raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
		raw[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
	}
	raw
}

/// Build synthetic Q6_K blocks (`d` is the trailing f16 at bytes 208..210).
fn synth_q6k(rng: &mut Rng, out_dim: usize, in_dim: usize) -> Vec<u8> {
	let nb = out_dim * in_dim / QK_K;
	let mut raw = vec![0u8; nb * Q6K_BLOCK];
	for b in 0..nb {
		let o = b * Q6K_BLOCK;
		rng.fill(&mut raw[o..o + 208]);
		let d = quant::f32_to_fp16(0.01 + (rng.next_u32() % 100) as f32 * 1e-4);
		raw[o + 208..o + 210].copy_from_slice(&d.to_le_bytes());
	}
	raw
}

struct Shape {
	name: &'static str,
	id: WeightId,
	ty: GgmlType,
	out_dim: usize,
	in_dim: usize,
}

fn main() {
	let mut iters = 20usize;
	let mut seq = 1usize;
	let mut check = false;
	let mut sweep = false;
	let mut gate = false;
	let mut gemm_only = false;
	let mut probe = false;
	let mut reps = 4usize;
	let mut spv: Option<String> = None;
	let mut gemv_ab = false;
	let mut q6k = false;
	let mut upload = false;
	let mut stream = false;
	let mut crossover = false;
	let mut gemv_probe = false;
	let mut x_f32 = false;
	let mut rows: Option<u32> = None;
	let mut soak: Option<f64> = None;
	let mut delta = false;
	let mut delta_npu: Option<String> = None;
	let mut lanes: Option<usize> = None;
	let args: Vec<String> = std::env::args().skip(1).collect();
	let mut i = 0;
	while i < args.len() {
		match args[i].as_str() {
			"--iters" => {
				i += 1;
				iters = args[i].parse().expect("--iters N");
			}
			"--seq" => {
				i += 1;
				seq = args[i].parse().expect("--seq N");
			}
			"--check" => check = true,
			"--sweep" => sweep = true,
			"--gate" => gate = true,
			// Just the FFN GEMM at one seq: the tight loop for kernel edits,
			// without the ALU probes or the other three shapes.
			"--gemm" => gemm_only = true,
			// Kernel-only timing plus the wrapper's share, and `--spv` to swap
			// the shader without a Rust rebuild: the edit/measure loop.
			"--probe" => probe = true,
			"--reps" => {
				i += 1;
				reps = args[i].parse().expect("--reps N");
			}
			"--spv" => {
				i += 1;
				spv = Some(args[i].clone());
			}
			// Also time the batched-GEMV kernel this replaced, same buffers.
			"--gemv" => gemv_ab = true,
			// Probe the Q6_K `FfnDown` shape instead of the Q4_K `FfnUp` one.
			"--q6k" => q6k = true,
			// Host->VRAM bandwidth: the gate for layer streaming.
			"--upload" => upload = true,
			// The *streaming* prefill path end to end: what one FFN block costs
			// when its weights are not resident, split into upload and GEMM.
			"--stream" => stream = true,
			// Sweep `seq` and print, for one FFN block, the measured CPU time
			// next to the GPU's resident and streamed times. This is the gate
			// for routing prefill: the CPU GEMM was rewritten twice after the
			// first GPU numbers were taken, so the only honest way to say
			// "the GPU is faster here" is to time both, now, on the same shapes.
			"--crossover" => crossover = true,
			// Decode-shape kernel-only timing: the shipped GEMV against a
			// candidate `--spv`, which may read raw f32 activations (`--xf32`)
			// and cover `--rows N` output rows per workgroup.
			"--gemv-probe" => gemv_probe = true,
			"--xf32" => x_f32 = true,
			"--rows" => {
				i += 1;
				rows = Some(args[i].parse().expect("--rows N"));
			}
			// Sustained decode-shape GEMV for SECS seconds, GB/s per window:
			// a DRAM load to run next to another device (iGPU next to the NPU).
			// The prefill delta-net scan of one layer vs the CPU (`--lanes N`
			// = invocations per state column the `--spv` module was built for).
			"--delta" => delta = true,
			// `--delta-npu DIR`: the recurrence between the NPU's `--cut pre` and
			// `--cut post` graphs, on pre's outputs DIR/{qn,kn,v,gb}.out;
			// writes DIR/o.in (and DIR/ssm.f32, the state after).
			"--delta-npu" => {
				i += 1;
				delta_npu = Some(args[i].clone());
			}
			"--lanes" => {
				i += 1;
				lanes = Some(args[i].parse().expect("--lanes N"));
			}
			"--soak" => {
				i += 1;
				soak = Some(args[i].parse().expect("--soak SECS"));
			}
			other => {
				eprintln!("unknown arg {other}");
				std::process::exit(2);
			}
		}
		i += 1;
	}

	// Qwen3.8-27B shapes: n_embd=5120, n_ff=17408,
	// head_dim=256, n_head=24, n_head_kv=4.
	let n_embd = 5120usize;
	let n_ff = 17408usize;
	let shapes = [
		Shape {
			name: "FfnUp   [17408x5120] Q4_K",
			id: WeightId::FfnUp,
			ty: GgmlType::Q4_K,
			out_dim: n_ff,
			in_dim: n_embd,
		},
		Shape {
			name: "FfnDown [5120x17408] Q6_K",
			id: WeightId::FfnDown,
			ty: GgmlType::Q6_K,
			out_dim: n_embd,
			in_dim: n_ff,
		},
		Shape {
			name: "Wq      [6144x5120]  Q4_K",
			id: WeightId::Wq,
			ty: GgmlType::Q4_K,
			out_dim: 24 * 256,
			in_dim: n_embd,
		},
		Shape {
			name: "Wo      [5120x6144]  Q4_K",
			id: WeightId::Wo,
			ty: GgmlType::Q4_K,
			out_dim: n_embd,
			in_dim: 24 * 256,
		},
	];

	let mut mgr = OffloadManager::new(1, 1);
	if !mgr.available {
		eprintln!("no Vulkan device available; nothing to benchmark");
		std::process::exit(1);
	}
	println!("gpubench: iters={iters} seq={seq} check={check}");

	if upload {
		run_upload(&mut mgr, n_ff, n_embd, iters.clamp(2, 5));
		return;
	}

	if crossover {
		run_crossover(&mut mgr, n_ff, n_embd, iters.clamp(2, 5));
		return;
	}

	if stream {
		let seq = if seq == 1 { 2048 } else { seq };
		run_stream(&mut mgr, n_ff, n_embd, seq, iters.clamp(2, 5));
		return;
	}

	if let Some(dir) = delta_npu {
		run_delta_npu(
			&mut mgr,
			&dir,
			reps.max(10),
			spv.as_deref(),
			lanes.unwrap_or(8),
		);
		return;
	}

	if delta {
		run_delta(
			&mut mgr,
			if seq == 1 { 256 } else { seq },
			reps.max(10),
			spv.as_deref(),
			lanes.unwrap_or(8),
		);
		return;
	}

	if let Some(secs) = soak {
		run_soak(&mut mgr, &shapes[q6k as usize], secs, reps.max(20));
		return;
	}

	if gemv_probe {
		run_gemv_probe(
			&mut mgr,
			&shapes,
			seq,
			reps.max(20),
			spv.as_deref(),
			x_f32,
			rows,
			q6k,
		);
		return;
	}

	if probe {
		let seq = if seq == 1 { 2048 } else { seq };
		run_probe(
			&mut mgr,
			n_ff,
			n_embd,
			seq,
			reps,
			spv.as_deref(),
			gemv_ab,
			q6k,
		);
		return;
	}

	if gate {
		run_gate(&mut mgr, n_ff, n_embd);
		return;
	}

	if gemm_only {
		run_gemm_only(&mut mgr, n_ff, n_embd, seq, iters);
		return;
	}

	if sweep {
		// out_dim sweep at fixed in_dim, to separate "work grew" from
		// "something degrades with grid size".
		let mut rng = Rng(0xABCD_EF01_2345_6789);
		for &od in &[1024usize, 2048, 4096, 5120, 6144, 8192, 12288, 17408] {
			let raw = synth_q4k(&mut rng, od, n_embd);
			mgr.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K, od, n_embd);
			let cpu_w = LinearWeight::Q4K {
				raw: raw.clone(),
				out_dim: od,
				in_dim: n_embd,
			};
			let x: Vec<f32> = (0..n_embd)
				.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
				.collect();
			let _ = mgr.matmul(0, WeightId::FfnUp, &cpu_w, &x, od, n_embd);
			let t0 = Instant::now();
			for _ in 0..iters {
				let _ = mgr.matmul(0, WeightId::FfnUp, &cpu_w, &x, od, n_embd);
			}
			let el = t0.elapsed().as_secs_f64() / iters as f64;
			println!(
				"  out_dim {:>6}  {:8.3} ms  {:7.1} GB/s  ({} workgroups)",
				od,
				el * 1e3,
				raw.len() as f64 / el / 1e9,
				od.div_ceil(64)
			);
		}
		return;
	}

	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	for s in &shapes {
		let raw = match s.ty {
			GgmlType::Q6_K => synth_q6k(&mut rng, s.out_dim, s.in_dim),
			_ => synth_q4k(&mut rng, s.out_dim, s.in_dim),
		};
		mgr.upload(0, s.id, &raw, s.ty, s.out_dim, s.in_dim);

		let cpu_w = match s.ty {
			GgmlType::Q6_K => LinearWeight::Q6K {
				raw: raw.clone(),
				out_dim: s.out_dim,
				in_dim: s.in_dim,
			},
			_ => LinearWeight::Q4K {
				raw: raw.clone(),
				out_dim: s.out_dim,
				in_dim: s.in_dim,
			},
		};
		let x: Vec<f32> = (0..seq * s.in_dim)
			.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
			.collect();

		// Warm-up (first dispatch pays pipeline/descriptor warm-up).
		let _ = run(&mgr, s, &cpu_w, &x, seq);

		let t0 = Instant::now();
		let mut sink = 0.0f32;
		for _ in 0..iters {
			let y = run(&mgr, s, &cpu_w, &x, seq);
			sink += y[0];
		}
		let el = t0.elapsed().as_secs_f64() / iters as f64;

		let wbytes = raw.len() as f64;
		let flops = 2.0 * (s.out_dim * s.in_dim * seq) as f64;
		println!(
			"  {:<28} {:8.3} ms   {:7.1} GB/s   {:7.1} GFLOP/s  (sink {:.3e})",
			s.name,
			el * 1e3,
			wbytes / el / 1e9,
			flops / el / 1e9,
			sink,
		);

		if check {
			let gpu = run(&mgr, s, &cpu_w, &x, seq);
			let cpu = if seq == 1 {
				quant::apply_linear(&cpu_w, &x, s.out_dim, s.in_dim)
			} else {
				quant::apply_linear_batch(&cpu_w, &x, seq, s.out_dim, s.in_dim)
			};
			let (mut maxabs, mut denom) = (0.0f32, 0.0f32);
			for j in 0..cpu.len() {
				maxabs = maxabs.max((gpu[j] - cpu[j]).abs());
				denom = denom.max(cpu[j].abs());
			}
			println!("    check: maxabsdiff {maxabs:.6}  (max |cpu| {denom:.3})");
		}
	}
}

/// One shape, one `seq`, correctness plus rate - the edit/measure loop for
/// `gemm_q4k.glsl`. `--gate` gives the full picture; this runs in a few seconds.
fn run_gemm_only(mgr: &mut OffloadManager, n_ff: usize, n_embd: usize, seq: usize, iters: usize) {
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	let raw = synth_q4k(&mut rng, n_ff, n_embd);
	mgr.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K, n_ff, n_embd);
	let cpu_w = LinearWeight::Q4K {
		raw,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let shape = Shape {
		name: "FfnUp",
		id: WeightId::FfnUp,
		ty: GgmlType::Q4_K,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let x: Vec<f32> = (0..seq * n_embd)
		.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
		.collect();
	let gpu = run(mgr, &shape, &cpu_w, &x, seq);
	let t0 = Instant::now();
	let mut sink = 0.0f32;
	for _ in 0..iters {
		sink += run(mgr, &shape, &cpu_w, &x, seq)[0];
	}
	let el = t0.elapsed().as_secs_f64() / iters as f64;
	let flops = 2.0 * (n_ff * n_embd * seq) as f64;

	// Correctness on a slice: the full CPU reference at seq=2048 costs more wall
	// time than the measurement itself, and every workgroup runs the same code,
	// so a kernel error shows up in the first tiles as readily as anywhere.
	let cs = seq.min(64);
	let cpu = quant::apply_linear_batch(&cpu_w, &x[..cs * n_embd], cs, n_ff, n_embd);
	let (mut maxabs, mut denom) = (0.0f32, 0.0f32);
	for j in 0..cpu.len() {
		maxabs = maxabs.max((gpu[j] - cpu[j]).abs());
		denom = denom.max(cpu[j].abs());
	}
	println!(
		"  FfnUp [{n_ff}x{n_embd}] seq {seq}  {:8.2} ms  {:7.1} GFLOP/s  {:5.2}x CPU  maxabsdiff {maxabs:.6} (max |cpu| {denom:.1}, sink {sink:.3e})",
		el * 1e3,
		flops / el / 1e9,
		flops / el / 1e9 / CPU_PREFILL_GFLOPS,
	);
}

/// The CPU's prefill GEMM rate, for the `x CPU` columns below.
///
/// **A constant here is a trap**: it goes stale whenever the CPU prefill GEMM
/// gets faster, and every `x CPU` column then overstates the GPU (a stale 966
/// against 1836 GFLOP/s on Q4_K gave ~1.5x). The value below is the *blended*
/// FFN-block rate measured by `--crossover` (1095 GFLOP in 765 ms at seq 2048),
/// which is right for a Q4_K+Q6_K mix and still wrong for either kernel alone.
///
/// Treat these columns as a rough orientation only. **`--crossover` times both
/// sides in the same process** and is the number to use for any routing or
/// ship/don't-ship decision; `--probe` prints its own measured CPU line too.
const CPU_PREFILL_GFLOPS: f64 = 1431.0;

/// GO/NO-GO gate for GPU-offloading the prefill GEMMs.
///
/// Two independent measurements, because they answer different questions:
///
///  1. **What the existing kernel does at `seq = 2048`.** `q4k.glsl` maps the
///     sequence position to `gl_WorkGroupID.y`, so every (row, position) pair
///     re-reads the weight row from VRAM. That makes it a batched GEMV whose
///     weight traffic grows linearly with `seq`, and the seq sweep below shows
///     the per-token cost staying flat instead of falling, which is the
///     signature of no reuse at all.
///
///  2. **What the card's ALUs could do** if a real tiled GEMM were written
///     (`shaders/alu_probe.glsl`). This is the number the plan actually rests
///     on: no kernel on this GPU can beat it, so if it is not comfortably above
///     `CPU_PREFILL_GFLOPS` the whole offload idea is dead regardless of how
///     good the kernel is.
fn run_gate(mgr: &mut OffloadManager, n_ff: usize, n_embd: usize) {
	println!();
	println!("=== 1. ALU ceiling (no global memory traffic in the inner loop) ===");
	println!("    An upper bound on any kernel this card could run.");
	// 320 workgroups x 256 invocations = 81920 threads: ~8x the 1060's 1280
	// cores, so every SM is saturated and latency is fully hidden.
	let (groups, inner, iters) = (320u32, 4096u32, 7usize);

	let mut peak = 0.0f64;
	let mut fma_chain = 0.0f64;
	if let Some((s, ops, sink)) = mgr.alu_probe(0, 1, inner, groups, iters) {
		fma_chain = ops / s / 1e9;
		peak = peak.max(fma_chain);
		println!(
			"  fp32 FFMA (16 chains) {:8.3} ms   {:8.1} GFLOP/s   (sink {sink:.3e})",
			s * 1e3,
			fma_chain
		);
	}
	if let Some((s, ops, sink)) = mgr.alu_probe(1, 1, inner, groups, iters) {
		println!(
			"  int32 MAD             {:8.3} ms   {:8.1} G MAD/s    (sink {sink:.3e})",
			s * 1e3,
			ops / s / 1e9
		);
	}

	// Two passes: the "% of peak" column needs the peak, and the 16-chain FFMA
	// loop is *not* it; it tops out ~15 % below the widest tiled loop (all 16 of
	// its FMAs read the same two source registers, which the operand collector
	// cannot feed at full rate). The best rate any mode reaches is the honest
	// reference, so collect first and print after.
	let tiles = [1u32, 2, 4, 8, 16];
	let mut rows = Vec::new();
	for &tile in &tiles {
		if let Some((s, ops, sink)) = mgr.alu_probe(2, tile, inner, groups, iters) {
			let g = ops / s / 1e9;
			peak = peak.max(g);
			rows.push((tile, s, g, sink));
		}
	}
	println!();
	println!("  tiled Q4_K GEMM inner loop (8 nibbles unpacked once, reused across TILE tokens):");
	for (tile, s, g, sink) in rows {
		println!(
			"    TILE {:>2}             {:8.3} ms   {:8.1} GFLOP/s   {:5.2}x CPU   ({:5.1} % of peak, sink {sink:.3e})",
			tile,
			s * 1e3,
			g,
			g / CPU_PREFILL_GFLOPS,
			100.0 * g / peak,
		);
	}
	let _ = fma_chain;
	println!();
	println!("  measured ALU peak: {peak:.0} GFLOP/s");

	// Linearity self-check. If dispatch/submission overhead were a material part
	// of these timings the reported rate would climb with `inner`; a ratio near
	// 1.00 says the number is the loop and not the harness. This is the cheap
	// guard against the failure mode that produced a flattering 300 GB/s during
	// the coalescing work (a grid that did not match the kernel's contract).
	if let (Some((s1, o1, _)), Some((s2, o2, _))) = (
		mgr.alu_probe(2, 8, inner, groups, iters),
		mgr.alu_probe(2, 8, inner * 4, groups, iters),
	) {
		let (g1, g2) = (o1 / s1 / 1e9, o2 / s2 / 1e9);
		println!(
			"  linearity check (TILE 8, inner x4): {g1:.0} -> {g2:.0} GFLOP/s  ratio {:.3}",
			g2 / g1
		);
	}

	println!();
	println!("=== 2. The existing q4k.glsl on the FFN GEMM shape [{n_ff}x{n_embd}] ===");
	println!("    Weight bytes re-read per token is the bug; per-token cost should");
	println!("    fall with seq for a real GEMM and stay flat for a batched GEMV.");
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	let raw = synth_q4k(&mut rng, n_ff, n_embd);
	mgr.upload(0, WeightId::FfnUp, &raw, GgmlType::Q4_K, n_ff, n_embd);
	let cpu_w = LinearWeight::Q4K {
		raw,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let shape = Shape {
		name: "FfnUp",
		id: WeightId::FfnUp,
		ty: GgmlType::Q4_K,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	for &seq in &[1usize, 16, 64, 256, 2048] {
		// Longer sequences cost seconds per dispatch; three timed repeats is
		// enough once the variance at seq=1 has been seen.
		let reps = if seq >= 256 { 3 } else { 10 };
		let x: Vec<f32> = (0..seq * n_embd)
			.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
			.collect();
		let _ = run(mgr, &shape, &cpu_w, &x, seq);
		let t0 = Instant::now();
		let mut sink = 0.0f32;
		for _ in 0..reps {
			sink += run(mgr, &shape, &cpu_w, &x, seq)[0];
		}
		let el = t0.elapsed().as_secs_f64() / reps as f64;
		let flops = 2.0 * (n_ff * n_embd * seq) as f64;
		println!(
			"  seq {:>5}   {:9.2} ms   {:8.2} ms/token   {:7.1} GFLOP/s   {:5.3}x CPU   (sink {sink:.3e})",
			seq,
			el * 1e3,
			el * 1e3 / seq as f64,
			flops / el / 1e9,
			flops / el / 1e9 / CPU_PREFILL_GFLOPS,
		);
	}

	println!();
	println!("=== verdict inputs ===");
	println!("  CPU prefill GEMM rate (measured): {CPU_PREFILL_GFLOPS:.0} GFLOP/s");
	println!(
		"  Gate: a tiled GEMM must clear ~{:.0} GFLOP/s (1.5x CPU) to be worth",
		1.5 * CPU_PREFILL_GFLOPS
	);
	println!("  streaming 235 MB of weights per layer over PCIe. Compare against the");
	println!("  best TILE row above: that is the ceiling, and a real kernel will land");
	println!("  below it once VRAM traffic, tail effects and dispatch overhead are paid.");
}

/// The kernel-iteration loop: measure `gemm_q4k.glsl` with the wrapper's costs
/// removed, and say what the wrapper's costs actually are.
///
/// `matmul_batch_seq` at seq=2048 x out_dim=17408 wraps the dispatch in two
/// 143 MB allocations, a 143 MB device->host copy and a 143 MB memcpy, plus a
/// host-side q8_K quantization of 10 M activations. Timing through it cannot
/// tell a slow shader from a slow wrapper. This prints both, so a `.glsl` edit
/// can be judged against the kernel number alone:
///
///   glslc -fshader-stage=compute -O crates/offload/src/shaders/gemm_q4k.glsl \
///       -o /tmp/v.spv
///   cargo run --release -p offload --features vulkan --example gpubench -- \
///       --probe --spv /tmp/v.spv
///
/// (no Rust rebuild: `--spv` loads the module at runtime).
fn run_probe(
	mgr: &mut OffloadManager,
	n_ff: usize,
	n_embd: usize,
	seq: usize,
	reps: usize,
	spv: Option<&str>,
	gemv: bool,
	q6k: bool,
) {
	// `FfnDown` is the Q6_K shape and `FfnUp` the Q4_K one; both are the real
	// model's, and between them they are most of prefill.
	let (name, id, ty, out_dim, in_dim) = if q6k {
		("FfnDown", WeightId::FfnDown, GgmlType::Q6_K, n_embd, n_ff)
	} else {
		("FfnUp", WeightId::FfnUp, GgmlType::Q4_K, n_ff, n_embd)
	};
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	let raw = if q6k {
		synth_q6k(&mut rng, out_dim, in_dim)
	} else {
		synth_q4k(&mut rng, out_dim, in_dim)
	};
	mgr.upload(0, id, &raw, ty, out_dim, in_dim);
	let cpu_w = if q6k {
		LinearWeight::Q6K {
			raw,
			out_dim,
			in_dim,
		}
	} else {
		LinearWeight::Q4K {
			raw,
			out_dim,
			in_dim,
		}
	};
	let x: Vec<f32> = (0..seq * in_dim)
		.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
		.collect();
	let flops = 2.0 * (out_dim * in_dim * seq) as f64;

	println!();
	println!("{name} [{out_dim}x{in_dim}] {ty:?}  seq {seq}  reps {reps}");
	if let Some(p) = spv {
		println!("  shader override: {p}");
	}

	// The kernel this replaced, on the same buffers: `q4k.glsl` / `q6k.glsl` map
	// the sequence position to `gl_WorkGroupID.y`, so they re-read the weight row
	// for every (row, position) pair. This is what "batched GEMV, not GEMM" costs.
	if gemv {
		if let Some((s, _)) = mgr.gemm_probe(0, id, &x, seq, out_dim, in_dim, 1, None, true) {
			println!(
				"  GEMV kernel    {:9.2} ms   {:7.1} GFLOP/s   {:5.2}x CPU   (kernel only)",
				s * 1e3,
				flops / s / 1e9,
				flops / s / 1e9 / CPU_PREFILL_GFLOPS,
			);
		}
	}

	let mut kernel_secs = 0.0;
	if let Some((s, head)) = mgr.gemm_probe(0, id, &x, seq, out_dim, in_dim, reps, spv, false) {
		kernel_secs = s;
		// Correctness on the first tiles. Every workgroup runs the same code, so
		// a kernel error shows up here as readily as anywhere, and the full CPU
		// reference at seq=2048 costs more wall time than the measurement.
		let cs = seq.min(64);
		let cpu = quant::apply_linear_batch(&cpu_w, &x[..cs * in_dim], cs, out_dim, in_dim);
		let (mut maxabs, mut denom) = (0.0f32, 0.0f32);
		for j in 0..cpu.len() {
			maxabs = maxabs.max((head[j] - cpu[j]).abs());
			denom = denom.max(cpu[j].abs());
		}
		println!(
			"  kernel only     {:8.2} ms   {:7.1} GFLOP/s   {:5.2}x const   maxabsdiff {maxabs:.6} (max |cpu| {denom:.1})",
			s * 1e3,
			flops / s / 1e9,
			flops / s / 1e9 / CPU_PREFILL_GFLOPS,
		);
	}

	// The CPU, on this exact shape and seq, timed here rather than taken from
	// `CPU_PREFILL_GFLOPS`. A ratio against a constant cannot notice that the
	// CPU kernel changed; this one can.
	let mut cpu_secs = f64::MAX;
	for _ in 0..reps.clamp(1, 2) {
		let t0 = Instant::now();
		let r = quant::apply_linear_batch(&cpu_w, &x, seq, out_dim, in_dim);
		cpu_secs = cpu_secs.min(t0.elapsed().as_secs_f64());
		std::hint::black_box(r[0]);
	}
	println!(
		"  CPU (measured)  {:8.2} ms   {:7.1} GFLOP/s   {:5.2}x CPU",
		cpu_secs * 1e3,
		flops / cpu_secs / 1e9,
		1.0,
	);

	// Same shape through the real entry point, so the gap between the two lines
	// is exactly what the wrapper costs.
	let shape = Shape {
		name,
		id,
		ty,
		out_dim,
		in_dim,
	};
	let _ = run(mgr, &shape, &cpu_w, &x, seq);
	let t0 = Instant::now();
	let mut sink = 0.0f32;
	for _ in 0..reps {
		sink += run(mgr, &shape, &cpu_w, &x, seq)[0];
	}
	let wall = t0.elapsed().as_secs_f64() / reps as f64;
	println!(
		"  matmul_batch_seq{:8.2} ms   {:7.1} GFLOP/s   {:5.2}x CPU   (sink {sink:.3e})",
		wall * 1e3,
		flops / wall / 1e9,
		cpu_secs / wall,
	);
	if kernel_secs > 0.0 {
		println!(
			"  wrapper overhead{:8.2} ms   ({:4.1} % of the call)",
			(wall - kernel_secs) * 1e3,
			100.0 * (wall - kernel_secs) / wall,
		);
	}

	// The host-side q8_K quantization of the activation, timed on its own. It is
	// part of the wrapper overhead above and is the one piece a better shader
	// cannot remove, only hoisting it out of the per-weight call can.
	let t0 = Instant::now();
	for _ in 0..reps {
		std::hint::black_box(quant::quantize_q8k(&x[..in_dim], in_dim));
	}
	let q1 = t0.elapsed().as_secs_f64() / reps as f64;
	println!(
		"  ... of which q8_K pack (est) {:6.2} ms",
		q1 * seq as f64 * 1e3
	);
}

/// Host->VRAM upload bandwidth, split into its two halves.
///
/// The layer-streaming plan rests entirely on this number: a layer is ~235 MB
/// of weights, and streaming is only worth it if that lands in VRAM in a small
/// fraction of the layer's compute time. Both halves are measured because they
/// have different fixes: the `memcpy` into the write-combined staging buffer is
/// a host cost (fix: a wider or threaded copy, or a persistently mapped ring),
/// and the `vkCmdCopyBuffer` is the PCIe DMA (fix: nothing, it is the bus).
///
/// `alloc` is reported separately because a naive streaming ring would pay it
/// per layer per chunk; a slot pool pays it once.
/// One FFN block (`FfnUp` + `FfnGate` grouped, then `FfnDown`) through the real
/// `matmul_batch_seq` with **no weight resident**, i.e. exactly what a streamed
/// layer costs during prefill.
///
/// `--upload` measures `mgr.upload`, which is the *residency* path: it allocates
/// a buffer per call and, for Q6_K, repacks through a zeroed 73 MB `Vec`. The
/// streaming path does neither, so its bytes/s are different, and it is the
/// streaming path that prefill actually pays. This also splits that cost into
/// the host repack into the staging map and the submit + DMA after it, which is
/// what says whether double-buffering onto the transfer queue can hide it: only
/// the second half is the copy engine's, the first is a CPU `memcpy` that no
/// amount of queue overlap removes.
fn run_stream(mgr: &mut OffloadManager, n_ff: usize, n_embd: usize, seq: usize, iters: usize) {
	// Streaming can be switched off, and measuring the streaming path with streaming disabled
	// would silently time the CPU fallback instead.
	if matches!(
		std::env::var("QWEN_GPU_STREAM").as_deref().map(str::trim),
		Ok("0") | Ok("no") | Ok("off") | Ok("false")
	) {
		eprintln!("--stream measures the streaming path; QWEN_GPU_STREAM is off");
		std::process::exit(2);
	}
	let mut rng = Rng(0x51EA_0000_1234_5678);
	let up = synth_q4k(&mut rng, n_ff, n_embd);
	let gate = synth_q4k(&mut rng, n_ff, n_embd);
	let down = synth_q6k(&mut rng, n_embd, n_ff);
	let w_up = LinearWeight::Q4K {
		raw: up,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let w_gate = LinearWeight::Q4K {
		raw: gate,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let w_down = LinearWeight::Q6K {
		raw: down,
		out_dim: n_embd,
		in_dim: n_ff,
	};
	let x: Vec<f32> = (0..seq * n_embd)
		.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
		.collect();
	let h: Vec<f32> = (0..seq * n_ff)
		.map(|j| ((j % 89) as f32 - 44.0) * 0.02)
		.collect();

	println!();
	println!("=== streamed FFN block, seq={seq} (non-resident weights) ===");
	let mut best_total = f64::MAX;
	let mut best: (f64, f64, f64) = (0.0, 0.0, 0.0);
	let mut mb = 0.0f64;
	// Every iteration uses a **fresh layer index**, so no weight is ever a
	// pass-1 hit in the slot pool. That is what prefill does (a chunk reads
	// each weight once) and reusing `il = 0` here quietly let `FfnGate` stay
	// resident between iterations and under-reported the upload by a third.
	for it in 0..iters {
		let il = it + 1;
		let _ = mgr.stream_take_stats();
		let t0 = Instant::now();
		let _ = mgr.matmul_batch_seq(&[
			BatchJobSeq {
				il,
				id: WeightId::FfnUp,
				w: &w_up,
				x: &x,
				seq,
				out_dim: n_ff,
				in_dim: n_embd,
			},
			BatchJobSeq {
				il,
				id: WeightId::FfnGate,
				w: &w_gate,
				x: &x,
				seq,
				out_dim: n_ff,
				in_dim: n_embd,
			},
		]);
		let _ = mgr.matmul_batch_seq(&[BatchJobSeq {
			il,
			id: WeightId::FfnDown,
			w: &w_down,
			x: &h,
			seq,
			out_dim: n_embd,
			in_dim: n_ff,
		}]);
		let total = t0.elapsed().as_secs_f64();
		let (bytes, pack, copy, _allocs) = mgr.stream_take_stats();
		if total < best_total {
			best_total = total;
			best = (pack, copy, total - pack - copy);
			mb = bytes as f64 / 1e6;
		}
	}
	let (pack, copy, rest) = best;
	// `repack` is the transfer thread's own elapsed time, which since the worker
	// landed is **not** serial cost: it runs beside the caller's host work and
	// only what spills past `stream_sync` shows up in the block total. Read the
	// block total for the answer; read this pair to see *why* it moved.
	// `QWEN_GPU_STREAM_WORKER=0` puts both back on the calling thread.
	println!(
		"  upload {:7.2} ms = repack {:7.2} ({:6.1} MB, {:5.2} GB/s host, off-thread) + DMA wait {:7.2}",
		(pack + copy) * 1e3, pack * 1e3, mb, mb * 1e-3 / pack, copy * 1e3,
	);
	println!(
		"  everything else (pack x, alloc, GEMM, readback, free) {:7.2} ms",
		rest * 1e3
	);
	println!(
		"  block total {:7.2} ms   upload share {:5.1} %",
		best_total * 1e3,
		100.0 * (pack + copy) / best_total,
	);
}

/// **Where GPU prefill is actually faster than the CPU**, swept over `seq`.
///
/// Everything else in this file compares the GPU against `CPU_PREFILL_GFLOPS`,
/// a *constant*, which goes stale as the CPU GEMM improves. Routing prefill to the GPU
/// "only where it improves speed" needs both sides timed in the same process on
/// the same shapes, which is what this does.
///
/// One **FFN block** per data point (`FfnUp` + `FfnGate` grouped, then
/// `FfnDown`), because that is the unit the model dispatches and it mixes the
/// Q4_K and Q6_K kernels in the ratio prefill actually uses.
///
/// Run with the routing gates disabled, or the GPU arms silently measure the
/// CPU fallback:
///
///     QWEN_GPU_PREFILL_MIN_SEQ=0 QWEN_GPU_STREAM_MIN_SEQ=0 gpubench --crossover
fn run_crossover(mgr: &mut OffloadManager, n_ff: usize, n_embd: usize, iters: usize) {
	let mut rng = Rng(0xC205_5000_1234_5678);
	let up = synth_q4k(&mut rng, n_ff, n_embd);
	let gate = synth_q4k(&mut rng, n_ff, n_embd);
	let down = synth_q6k(&mut rng, n_embd, n_ff);
	// Resident copies live at layer 0; the streamed arm uses fresh layer indices
	// so no weight is ever a slot-pool hit (the same trap `--stream` documents).
	mgr.upload(0, WeightId::FfnUp, &up, GgmlType::Q4_K, n_ff, n_embd);
	mgr.upload(0, WeightId::FfnGate, &gate, GgmlType::Q4_K, n_ff, n_embd);
	mgr.upload(0, WeightId::FfnDown, &down, GgmlType::Q6_K, n_embd, n_ff);
	let w_up = LinearWeight::Q4K {
		raw: up,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let w_gate = LinearWeight::Q4K {
		raw: gate,
		out_dim: n_ff,
		in_dim: n_embd,
	};
	let w_down = LinearWeight::Q6K {
		raw: down,
		out_dim: n_embd,
		in_dim: n_ff,
	};

	let seqs: Vec<usize> = match std::env::var("CROSSOVER_SEQS") {
		Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
		Err(_) => vec![16, 32, 64, 128, 256, 512, 1024, 2048],
	};
	println!();
	println!("=== prefill crossover: one FFN block, CPU vs GPU (best of {iters}) ===");
	println!("  seq |    CPU ms |  GPU res ms  x CPU |  GPU stream ms  x CPU |  block GFLOP");
	for &seq in &seqs {
		let x: Vec<f32> = (0..seq * n_embd)
			.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
			.collect();
		let h: Vec<f32> = (0..seq * n_ff)
			.map(|j| ((j % 89) as f32 - 44.0) * 0.02)
			.collect();
		let flops = 3.0 * 2.0 * (n_ff * n_embd * seq) as f64;

		// CPU: exactly what `Model::lin_seq_group` runs when prefill stays on
		// the CPU; one `apply_linear_batch` per weight, no grouping.
		let mut cpu = f64::MAX;
		for _ in 0..iters {
			let t0 = Instant::now();
			let a = quant::apply_linear_batch(&w_up, &x, seq, n_ff, n_embd);
			let b = quant::apply_linear_batch(&w_gate, &x, seq, n_ff, n_embd);
			let c = quant::apply_linear_batch(&w_down, &h, seq, n_embd, n_ff);
			cpu = cpu.min(t0.elapsed().as_secs_f64());
			std::hint::black_box((a[0], b[0], c[0]));
		}

		let block = |mgr: &OffloadManager, il: usize| -> f64 {
			let t0 = Instant::now();
			let _ = mgr.matmul_batch_seq(&[
				BatchJobSeq {
					il,
					id: WeightId::FfnUp,
					w: &w_up,
					x: &x,
					seq,
					out_dim: n_ff,
					in_dim: n_embd,
				},
				BatchJobSeq {
					il,
					id: WeightId::FfnGate,
					w: &w_gate,
					x: &x,
					seq,
					out_dim: n_ff,
					in_dim: n_embd,
				},
			]);
			let _ = mgr.matmul_batch_seq(&[BatchJobSeq {
				il,
				id: WeightId::FfnDown,
				w: &w_down,
				x: &h,
				seq,
				out_dim: n_embd,
				in_dim: n_ff,
			}]);
			t0.elapsed().as_secs_f64()
		};
		let _ = block(mgr, 0); // warm the per-call buffer pool
		let mut res = f64::MAX;
		for _ in 0..iters {
			res = res.min(block(mgr, 0));
		}
		let mut stream = f64::MAX;
		for it in 0..iters {
			stream = stream.min(block(mgr, 1 + it + seq));
		}
		println!(
			"{seq:5} | {:9.2} | {:11.2} {:6.2}x | {:14.2} {:6.2}x | {:12.1}",
			cpu * 1e3,
			res * 1e3,
			cpu / res,
			stream * 1e3,
			cpu / stream,
			flops / 1e9,
		);
	}
	println!();
	println!("  x CPU > 1.00 means the GPU is faster at that chunk length.");

	// FFN is ~70 % of a layer's prefill compute, but the routing gate applies to
	// every grouped matmul, so the *narrow* shapes have to clear it too: `Wk`
	// and `Wv` are 1024 rows against `FfnUp`'s 17408, which is far less work per
	// unit of wrapper overhead. If the gate were chosen from the FFN table alone
	// it could hand the GPU a group it loses on.
	let n_q = 2 * 256 * 24;
	let n_kv = 256 * 4;
	let n_o = 256 * 24;
	let wq_raw = synth_q4k(&mut rng, n_q, n_embd);
	let wk_raw = synth_q4k(&mut rng, n_kv, n_embd);
	let wv_raw = synth_q4k(&mut rng, n_kv, n_embd);
	let wo_raw = synth_q4k(&mut rng, n_embd, n_o);
	mgr.upload(0, WeightId::Wq, &wq_raw, GgmlType::Q4_K, n_q, n_embd);
	mgr.upload(0, WeightId::Wk, &wk_raw, GgmlType::Q4_K, n_kv, n_embd);
	mgr.upload(0, WeightId::Wv, &wv_raw, GgmlType::Q4_K, n_kv, n_embd);
	mgr.upload(0, WeightId::Wo, &wo_raw, GgmlType::Q4_K, n_embd, n_o);
	let w_q = LinearWeight::Q4K {
		raw: wq_raw,
		out_dim: n_q,
		in_dim: n_embd,
	};
	let w_k = LinearWeight::Q4K {
		raw: wk_raw,
		out_dim: n_kv,
		in_dim: n_embd,
	};
	let w_v = LinearWeight::Q4K {
		raw: wv_raw,
		out_dim: n_kv,
		in_dim: n_embd,
	};
	let w_o = LinearWeight::Q4K {
		raw: wo_raw,
		out_dim: n_embd,
		in_dim: n_o,
	};

	println!();
	println!("=== prefill crossover: one attention block (Wq+Wk+Wv, Wo) ===");
	println!("  seq |    CPU ms |  GPU res ms  x CPU |  GPU stream ms  x CPU |  block GFLOP");
	for &seq in &seqs {
		let x: Vec<f32> = (0..seq * n_embd)
			.map(|j| ((j % 97) as f32 - 48.0) * 0.02)
			.collect();
		let a: Vec<f32> = (0..seq * n_o)
			.map(|j| ((j % 89) as f32 - 44.0) * 0.02)
			.collect();
		let flops = 2.0 * ((n_q + 2 * n_kv) * n_embd + n_embd * n_o) as f64 * seq as f64;
		let mut cpu = f64::MAX;
		for _ in 0..iters {
			let t0 = Instant::now();
			let r0 = quant::apply_linear_batch(&w_q, &x, seq, n_q, n_embd);
			let r1 = quant::apply_linear_batch(&w_k, &x, seq, n_kv, n_embd);
			let r2 = quant::apply_linear_batch(&w_v, &x, seq, n_kv, n_embd);
			let r3 = quant::apply_linear_batch(&w_o, &a, seq, n_embd, n_o);
			cpu = cpu.min(t0.elapsed().as_secs_f64());
			std::hint::black_box((r0[0], r1[0], r2[0], r3[0]));
		}
		let block = |mgr: &OffloadManager, il: usize| -> f64 {
			let t0 = Instant::now();
			let _ = mgr.matmul_batch_seq(&[
				BatchJobSeq {
					il,
					id: WeightId::Wq,
					w: &w_q,
					x: &x,
					seq,
					out_dim: n_q,
					in_dim: n_embd,
				},
				BatchJobSeq {
					il,
					id: WeightId::Wk,
					w: &w_k,
					x: &x,
					seq,
					out_dim: n_kv,
					in_dim: n_embd,
				},
				BatchJobSeq {
					il,
					id: WeightId::Wv,
					w: &w_v,
					x: &x,
					seq,
					out_dim: n_kv,
					in_dim: n_embd,
				},
			]);
			let _ = mgr.matmul_batch_seq(&[BatchJobSeq {
				il,
				id: WeightId::Wo,
				w: &w_o,
				x: &a,
				seq,
				out_dim: n_embd,
				in_dim: n_o,
			}]);
			t0.elapsed().as_secs_f64()
		};
		let _ = block(mgr, 0);
		let mut res = f64::MAX;
		for _ in 0..iters {
			res = res.min(block(mgr, 0));
		}
		let mut stream = f64::MAX;
		for it in 0..iters {
			stream = stream.min(block(mgr, 4096 + it + seq));
		}
		println!(
			"{seq:5} | {:9.2} | {:11.2} {:6.2}x | {:14.2} {:6.2}x | {:12.1}",
			cpu * 1e3,
			res * 1e3,
			cpu / res,
			stream * 1e3,
			cpu / stream,
			flops / 1e9,
		);
	}
}

/// Kernel-only GEMV timing on the real decode shapes: the shipped kernel (q8_K
/// activations, embedded pipeline) next to an optional candidate `.spv`, both
/// checked against the CPU `apply_linear` on the same bytes. The candidate runs
/// on the Q4_K shapes, or on the Q6_K one with `--q6k`.
#[allow(clippy::too_many_arguments)]
/// `--delta`: the prefill delta-net scan of one layer (`delta_prefill.glsl`,
/// Qwen3.8's 16 key / 48 value heads of 128) over `seq` positions, against
/// `quant::delta_net_cols` run position by position on the CPU.
/// `--delta-npu DIR`: see the flag. The state starts from DIR/ssm_in.f32
/// (true scale, [48][128][128] f32) when present, else zero.
fn run_delta_npu(
	mgr: &mut OffloadManager,
	dir: &str,
	reps: usize,
	spv: Option<&str>,
	lanes: usize,
) {
	let p = |n: &str| std::path::Path::new(dir).join(n);
	let rd = |n: &str| std::fs::read(p(n)).unwrap_or_else(|e| panic!("{}: {e}", p(n).display()));
	let (qn, kn, v, gb) = (rd("qn.out"), rd("kn.out"), rd("v.out"), rd("gb.out"));
	let n = qn.len() / (2048 * 2);
	let st0: Vec<f32> = match std::fs::read(p("ssm_in.f32")) {
		Ok(b) => b
			.chunks_exact(4)
			.map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
			.collect(),
		Err(_) => vec![0.0; 48 * 128 * 128],
	};
	let Some((secs, o, st)) =
		mgr.delta_npu_probe(n, &st0, &qn, &kn, &v, &gb, 16.0, lanes, reps, spv)
	else {
		eprintln!("delta-npu: no GPU");
		return;
	};
	std::fs::write(p("o.in"), &o).unwrap();
	std::fs::write(
		p("ssm.f32"),
		st.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
	)
	.unwrap();
	println!(
		"delta-npu: n {n} lanes {lanes}  {:.3} ms  -> {}",
		secs * 1e3,
		p("o.in").display()
	);
}

fn run_delta(mgr: &mut OffloadManager, seq: usize, reps: usize, spv: Option<&str>, lanes: usize) {
	const S: usize = 128;
	let (hk, hv, n) = (16usize, 48usize, seq);
	let mut rng = Rng(0x0DE1_7A5E_ED00_0001);
	let mut f = |lo: f32, hi: f32| lo + (hi - lo) * (rng.next_u32() as f32 / u32::MAX as f32);
	// k / q L2-normalized per (head, position), as the model feeds them.
	let mut unit = |len: usize| {
		let mut v: Vec<f32> = (0..len).map(|_| f(-1.0, 1.0)).collect();
		for c in v.chunks_mut(S) {
			let nrm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
			c.iter_mut().for_each(|x| *x /= nrm);
		}
		v
	};
	let k = unit(hk * n * S);
	let q = unit(hk * n * S);
	let mut rng = Rng(0x0DE1_7A5E_ED00_0002);
	let mut f = |lo: f32, hi: f32| lo + (hi - lo) * (rng.next_u32() as f32 / u32::MAX as f32);
	let v: Vec<f32> = (0..hv * n * S).map(|_| f(-1.0, 1.0)).collect();
	let mut bg: Vec<f32> = (0..n * hv).map(|_| f(0.05, 0.95)).collect();
	bg.extend((0..n * hv).map(|_| f(0.85, 1.0)));
	let st0: Vec<f32> = (0..hv * S * S).map(|_| f(-0.1, 0.1)).collect();

	let Some((secs, o, st)) = mgr.delta_probe(n, hk, hv, &st0, &k, &q, &v, &bg, lanes, reps, spv)
	else {
		eprintln!("delta: no GPU");
		return;
	};
	// CPU reference.
	let mut st_c = st0.clone();
	let mut o_c = vec![0.0f32; hv * n * S];
	let qscale = 1.0 / (S as f32).sqrt();
	for h in 0..hv {
		let kh = h % hk;
		for t in 0..n {
			let kb = (kh * n + t) * S;
			let vb = (h * n + t) * S;
			quant::delta_net_cols(
				&mut st_c[h * S * S..(h + 1) * S * S],
				S,
				0,
				&k[kb..kb + S],
				&q[kb..kb + S],
				&v[vb..vb + S],
				bg[t * hv + h],
				bg[n * hv + t * hv + h],
				qscale,
				&mut o_c[vb..vb + S],
			);
		}
	}
	let md = |a: &[f32], b: &[f32]| {
		let m = a
			.iter()
			.zip(b)
			.map(|(x, y)| (x - y).abs())
			.fold(0.0f32, f32::max);
		let r = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
		(m, r)
	};
	let (mo, ro) = md(&o, &o_c);
	let (ms, rs) = md(&st, &st_c);
	let flop = (hv * n * S * S * 6) as f64;
	println!(
		"delta: seq {n} lanes {lanes}  {:8.3} ms  {:6.1} GFLOP/s  ({:.2} ms per 256 positions)  max|o-cpu| {mo:.2e} (max|o| {ro:.2})  max|st-cpu| {ms:.2e} (max|st| {rs:.2})",
		secs * 1e3,
		flop / secs / 1e9,
		secs * 1e3 * 256.0 / n as f64
	);
}

fn run_soak(mgr: &mut OffloadManager, s: &Shape, secs: f64, reps: usize) {
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	let raw = match s.ty {
		GgmlType::Q6_K => synth_q6k(&mut rng, s.out_dim, s.in_dim),
		_ => synth_q4k(&mut rng, s.out_dim, s.in_dim),
	};
	mgr.upload(0, s.id, &raw, s.ty, s.out_dim, s.in_dim);
	let bytes = raw.len() as f64;
	let x: Vec<f32> = (0..s.in_dim)
		.map(|j| ((j * 7919 % 997) as f32 - 498.0) * 0.004)
		.collect();
	println!(
		"soak: {} ({:.1} MB) x{reps} per call, {secs} s",
		s.name,
		bytes / 1e6
	);
	let t0 = std::time::Instant::now();
	let (mut win_t, mut win_b, mut all_b) = (t0, 0.0f64, 0.0f64);
	while t0.elapsed().as_secs_f64() < secs {
		if mgr
			.gemv_probe(0, s.id, &x, 1, s.out_dim, s.in_dim, reps, None, None, None)
			.is_none()
		{
			eprintln!("soak: no GPU weight");
			return;
		}
		win_b += bytes * (reps + 1) as f64;
		let w = win_t.elapsed().as_secs_f64();
		if w >= 2.0 {
			println!(
				"  t {:6.1} s  {:6.1} GB/s",
				t0.elapsed().as_secs_f64(),
				win_b / w / 1e9
			);
			all_b += win_b;
			win_b = 0.0;
			win_t = std::time::Instant::now();
		}
	}
	all_b += win_b;
	let t = t0.elapsed().as_secs_f64();
	println!("soak: mean {:.1} GB/s over {t:.1} s", all_b / t / 1e9);
}

fn run_gemv_probe(
	mgr: &mut OffloadManager,
	shapes: &[Shape],
	seq: usize,
	reps: usize,
	spv: Option<&str>,
	x_f32: bool,
	rows: Option<u32>,
	q6k: bool,
) {
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	println!(
		"gemv-probe: seq {seq} reps {reps} candidate {:?} xf32={x_f32} rows={rows:?}",
		spv
	);
	for s in shapes {
		let raw = match s.ty {
			GgmlType::Q6_K => synth_q6k(&mut rng, s.out_dim, s.in_dim),
			_ => synth_q4k(&mut rng, s.out_dim, s.in_dim),
		};
		mgr.upload(0, s.id, &raw, s.ty, s.out_dim, s.in_dim);
		let bytes = raw.len() as f64;
		let cpu_w = match s.ty {
			GgmlType::Q6_K => LinearWeight::Q6K {
				raw,
				out_dim: s.out_dim,
				in_dim: s.in_dim,
			},
			_ => LinearWeight::Q4K {
				raw,
				out_dim: s.out_dim,
				in_dim: s.in_dim,
			},
		};
		let x: Vec<f32> = (0..seq * s.in_dim)
			.map(|j| ((j * 7919 % 997) as f32 - 498.0) * 0.004)
			.collect();
		// Exact reference: dequantized weights, f64 dot. The shipped kernel
		// quantizes the activation to q8_K and so carries that noise; a raw-f32
		// candidate should land far closer to this than the shipped one does.
		let wf = {
			let raw = match &cpu_w {
				LinearWeight::Q4K { raw, .. } | LinearWeight::Q6K { raw, .. } => raw,
				_ => unreachable!(),
			};
			quant::dequantize(s.ty, raw, s.out_dim * s.in_dim)
		};
		let mut cpu = vec![0.0f32; seq * s.out_dim];
		for p in 0..seq {
			let xp = &x[p * s.in_dim..(p + 1) * s.in_dim];
			for o in 0..s.out_dim {
				let wr = &wf[o * s.in_dim..(o + 1) * s.in_dim];
				cpu[p * s.out_dim + o] = wr
					.iter()
					.zip(xp)
					.map(|(a, b)| *a as f64 * *b as f64)
					.sum::<f64>() as f32;
			}
		}
		let diff = |y: &[f32]| {
			let n = y.len().min(cpu.len());
			let (mut m, mut d) = (0.0f32, 0.0f32);
			for j in 0..n {
				m = m.max((y[j] - cpu[j]).abs());
				d = d.max(cpu[j].abs());
			}
			(m, d)
		};
		let is_target = (s.ty == GgmlType::Q6_K) == q6k;
		let line = |tag: &str, r: Option<(f64, Vec<f32>)>| {
			if let Some((t, head)) = r {
				let (m, d) = diff(&head);
				println!(
					"  {:<28} {tag:<9} {:8.1} us  {:6.1} GB/s  maxabsdiff {m:.5} (max |cpu| {d:.1})",
					s.name, t * 1e6, bytes * seq as f64 / t / 1e9
				);
			}
		};
		line(
			"shipped",
			mgr.gemv_probe(
				0, s.id, &x, seq, s.out_dim, s.in_dim, reps, None, None, None,
			),
		);
		if is_target && spv.is_some() {
			line(
				"candidate",
				mgr.gemv_probe(
					0,
					s.id,
					&x,
					seq,
					s.out_dim,
					s.in_dim,
					reps,
					spv,
					Some(x_f32),
					rows,
				),
			);
		}
	}
}

fn run_upload(mgr: &mut OffloadManager, n_ff: usize, n_embd: usize, iters: usize) {
	let mut rng = Rng(0x1234_5678_9ABC_DEF0);
	let cases: [(&str, WeightId, GgmlType, usize, usize); 3] = [
		(
			"FfnUp   Q4_K",
			WeightId::FfnUp,
			GgmlType::Q4_K,
			n_ff,
			n_embd,
		),
		(
			"FfnDown Q6_K",
			WeightId::FfnDown,
			GgmlType::Q6_K,
			n_embd,
			n_ff,
		),
		(
			"Wq      Q4_K",
			WeightId::Wq,
			GgmlType::Q4_K,
			24 * 256,
			n_embd,
		),
	];
	println!();
	println!("=== host -> VRAM upload (the layer-streaming gate) ===");
	let mut layer_bytes = 0usize;
	let mut layer_secs = 0.0f64;
	for (name, id, ty, out_dim, in_dim) in cases {
		let raw = match ty {
			GgmlType::Q6_K => synth_q6k(&mut rng, out_dim, in_dim),
			_ => synth_q4k(&mut rng, out_dim, in_dim),
		};
		let mut best = f64::MAX;
		for _ in 0..iters {
			let t0 = Instant::now();
			mgr.upload(0, id, &raw, ty, out_dim, in_dim);
			best = best.min(t0.elapsed().as_secs_f64());
		}
		println!(
			"  {name}  {:7.1} MB   {:7.2} ms   {:5.2} GB/s   (alloc + memcpy + DMA)",
			raw.len() as f64 / 1e6,
			best * 1e3,
			raw.len() as f64 / best / 1e9,
		);
		// An FFN block is up + gate + down; up and gate are the same shape.
		let mult = if name.starts_with("FfnUp") {
			2
		} else if name.starts_with("FfnDown") {
			1
		} else {
			0
		};
		layer_bytes += raw.len() * mult;
		layer_secs += best * mult as f64;
	}
	println!(
		"  FFN block (up+gate+down) {:7.1} MB   {:7.2} ms   {:5.2} GB/s",
		layer_bytes as f64 / 1e6,
		layer_secs * 1e3,
		layer_bytes as f64 / layer_secs / 1e9,
	);
	println!();
	println!("  Break-even: streaming pays off when the upload is small against the");
	println!("  GEMM time it enables. A layer's FFN GEMMs at seq=2048 are ~330 ms on");
	println!(
		"  this card ({:.0} GFLOP/s x 3 x 2*17408*5120*2048 FLOP);",
		CPU_PREFILL_GFLOPS * 2.2
	);
	println!("  compare that against the FFN-block figure above.");
}

fn run(mgr: &OffloadManager, s: &Shape, w: &LinearWeight, x: &[f32], seq: usize) -> Vec<f32> {
	if seq == 1 {
		mgr.matmul(0, s.id, w, x, s.out_dim, s.in_dim)
	} else {
		let job = offload::BatchJobSeq {
			il: 0,
			id: s.id,
			w,
			x,
			seq,
			out_dim: s.out_dim,
			in_dim: s.in_dim,
		};
		mgr.matmul_batch_seq(std::slice::from_ref(&job))
			.pop()
			.unwrap()
	}
}
