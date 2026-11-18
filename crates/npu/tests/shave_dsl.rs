//! The kernel compiler (`npu::shave::dsl`, `SHAVE-DSL.md`) and the SHAVE
//! emulator its output is checked in (`npu::shave::mach`).
//!
//! The emulator's timing model (latencies, delay slots, predication, the
//! rules measured on the device): every kernel-language program must run
//! correctly in it and break none of the rules. On the NPU itself:
//! `scripts/sktest.py`.

use npu::shave::mach::{d2h, h2d, Machine, Stats};
use npu::shave::prog::Program;

pub fn source(name: &str) -> String {
	let path = format!("{}/shave/{name}", env!("CARGO_MANIFEST_DIR"));
	std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// A deterministic spread of attention scores.
pub fn scores(count: usize) -> Vec<f64> {
	let mut s = 0x2545f4914f6cdd1du64;
	(0..count)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			h2d(d2h((s >> 11) as f64 / (1u64 << 53) as f64 * 24.0 - 16.0))
		})
		.collect()
}

/// Run a softmax_lse kernel over `rows` rows of `n`; returns the output and the statistics.
pub fn softmax(prog: &Program, n: usize, rows: usize, x: &[f64]) -> (Vec<f64>, Stats) {
	let (xa, ya, pa, da, sa) = (
		0x2e00_0000u32,
		0x2e10_0000u32,
		0x8003_0000u32,
		0x8003_1000u32,
		0x2e20_0000u32,
	);
	let mut m = Machine::new();
	m.map(xa, x.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect());
	m.map(ya, vec![0xee; 2 * n * rows]);
	let mut p = vec![0u8; 0x48];
	p[0..4].copy_from_slice(&xa.to_le_bytes());
	p[8..12].copy_from_slice(&2u32.to_le_bytes());
	p[12..16].copy_from_slice(&da.to_le_bytes());
	p[0x24..0x28].copy_from_slice(&ya.to_le_bytes());
	m.map(pa, p);
	m.map(
		da,
		[(n as u32).to_le_bytes(), (rows as u32).to_le_bytes()].concat(),
	);
	m.map(sa, vec![0; 0x1c00]);
	m.i[18] = pa;
	m.i[19] = sa + 0x1c00;
	m.run(prog, None, 10_000_000).unwrap();
	let y = m.read(ya, 2 * n * rows).unwrap();
	(
		(0..n * rows)
			.map(|k| h2d(u16::from_le_bytes([y[2 * k], y[2 * k + 1]])))
			.collect(),
		m.stats,
	)
}

/// The kernel's contract: p over the first n - 2 slots, then max and sum.
pub fn check_softmax(y: &[f64], x: &[f64], n: usize, norm: bool) {
	check_softmax_keys(y, x, n, n - 2, norm)
}

/// The same with `keys` key slots (`-DKL=4`: n - 4; the slots between are
/// 0).
pub fn check_softmax_keys(y: &[f64], x: &[f64], n: usize, keys: usize, norm: bool) {
	for (row, (y, x)) in y.chunks(n).zip(x.chunks(n)).enumerate() {
		for k in keys..n - 2 {
			assert_eq!(y[k], 0.0, "row {row} slot {k}");
		}
		let max = x[..keys].iter().cloned().fold(f64::MIN, f64::max);
		let sum: f64 = x[..keys].iter().map(|e| (e - max).exp()).sum();
		assert_eq!(y[n - 2], max, "row {row}: max");
		assert!(
			(y[n - 1] - sum).abs() <= 4e-3 * sum,
			"row {row}: sum {} for {sum}",
			y[n - 1]
		);
		for k in 0..keys {
			let want = (x[k] - max).exp() / if norm { sum } else { 1.0 };
			assert!(
				(y[k] - want).abs() <= 6e-3 * want + 7e-5,
				"row {row} slot {k}: {} for {want}",
				y[k]
			);
		}
	}
}

/// `-DKL=4` (the Flash-Next far attention blocks: 1020 keys, two empty
/// slots, max and sum).
#[test]
fn softmax_with_four_key_lanes_at_the_end() {
	for nonorm in [0, 1] {
		let k = kernel(
			"softmax_lse.sk",
			&[("N", 1024), ("NONORM", nonorm), ("KL", 4)],
		);
		let x = scores(3 * 1024);
		let (y, stats) = softmax(&k.program, 1024, 3, &x);
		check_softmax_keys(&y, &x, 1024, 1020, nonorm == 0);
		assert_eq!(
			(stats.undefined, stats.flags, stats.conflicts),
			(0, 0, 0),
			"{:?}",
			stats.notes
		);
	}
}

fn kernel(file: &str, defs: &[(&str, i64)]) -> npu::shave::dsl::Kernel {
	let defs: Vec<(String, i64)> = defs.iter().map(|d| (d.0.to_string(), d.1)).collect();
	npu::shave::dsl::compile(&source(file), &defs).unwrap_or_else(|e| panic!("{file}: {e}"))
}

#[test]
fn softmax_from_the_kernel_language() {
	for (n, nonorm, u) in [
		(1024, 0, 8),
		(1024, 1, 8),
		(1088, 0, 8),
		(1280, 1, 8),
		(1024, 0, 4),
		(4096, 0, 8),
	] {
		let k = kernel("softmax_lse.sk", &[("N", n), ("NONORM", nonorm), ("U", u)]);
		let n = n as usize;
		let x = scores(3 * n);
		let (y, stats) = softmax(&k.program, n, 3, &x);
		check_softmax(&y, &x, n, nonorm == 0);
		assert_eq!(
			(
				stats.landings,
				stats.sau_vau,
				stats.undefined,
				stats.flags,
				stats.conflicts
			),
			([0; 8], 0, 0, 0, 0),
			"{:?}",
			stats.notes
		);
		println!(
			"N {n} NONORM {nonorm} U {u}: {} cycles per row; {}",
			stats.cycles / 3,
			k.report.join("; ")
		);
		// and it assembles and links
		assert!(
			k.elf()
				.unwrap_or_else(|e| panic!("{e}\n{}", k.text()))
				.len() > 500
		);
	}
}

/// Compile `src` and run it: `input` at the address in the first MemRefData
/// record, 256 bytes of output behind the second. The output and the statistics.
fn run(src: &str, input: &[u8]) -> (Vec<u8>, Stats) {
	let k = npu::shave::dsl::compile(src, &[]).unwrap_or_else(|e| panic!("{e}"));
	let (xa, ya, pa, sa) = (
		0x2e00_0000u32,
		0x2e10_0000u32,
		0x8003_0000u32,
		0x2e20_0000u32,
	);
	let mut m = Machine::new();
	m.map(xa, input.to_vec());
	m.map(ya, vec![0xee; 256]);
	let mut p = vec![0u8; 0x48];
	p[0..4].copy_from_slice(&xa.to_le_bytes());
	p[0x24..0x28].copy_from_slice(&ya.to_le_bytes());
	m.map(pa, p);
	m.map(sa, vec![0; 0x1c00]);
	m.i[18] = pa;
	m.i[19] = sa + 0x1c00;
	m.run(&k.program, None, 1_000_000).unwrap_or_else(|e| {
		panic!(
			"{e}
{}",
			k.text()
		)
	});
	let s = &m.stats;
	assert_eq!(
		(s.landings, s.sau_vau, s.undefined, s.flags, s.conflicts),
		([0; 8], 0, 0, 0, 0),
		"{:?}
{}",
		s.notes,
		k.text()
	);
	assert_eq!(m.i[19], sa + 0x1c00, "the stack pointer is not restored");
	assert!(
		k.elf()
			.unwrap_or_else(|e| panic!(
				"{e}
{}",
				k.text()
			))
			.len() > 300
	);
	(m.read(ya, 256).unwrap().to_vec(), m.stats)
}

fn words(bytes: &[u8]) -> Vec<u32> {
	bytes
		.chunks(4)
		.map(|c| u32::from_le_bytes(c.try_into().unwrap()))
		.collect()
}

#[test]
fn integers_loops_and_conditions() {
	let src = "
        #define K 3
        void controlled_act(const u32 *params) {
            const u32 *x = (const u32 *)params[0];
            u32 *y = (u32 *)params[9];
            int a = 100, big = 0x12345678, neg = -8;
            y[0] = a - 1;
            y[1] = a + 300;
            y[2] = a * (a + K);
            y[3] = big;
            y[4] = neg;
            /* a counted loop over the input, and one that runs to a bound */
            u32 sum = 0;
            const u32 *p = x;
            for (u32 i = 0; i < 4; i++) { sum += *p; p++; }
            y[5] = sum;
            int n = 0;
            do { n += 2; } while (n < 9);
            y[6] = n;
            /* taken and not taken; signed and unsigned */
            u32 r = 1, ua = 100;
            if (sum == 10) r += 10;
            if (sum != 10) r += 100; else r += 2000;
            if (neg < a) r += 1000;
            if ((u32)neg < ua) r += 10000;
            y[7] = r;
            for (int k = 0; k < K; k++) {
                if (k != 1) y[8 + k] = a << k;
                else y[9] = a ^ a;
            }
            /* loops whose count is known only at run time, and one that does not run */
            u32 t = 0, lim = y[5] & 7;
            for (u32 i = 0; i < lim; i++) t += 5;
            for (u32 i = lim; i < 3; i++) t += 100;
            for (u32 i = lim + 4; i < 3; i++) t += 7;
            u32 w = 7;
            while (w > 2) { w -= 2; t += 1000; }
            y[11] = t;
        }";
	let input: Vec<u8> = [1u32, 2, 3, 4]
		.iter()
		.flat_map(|v| v.to_le_bytes())
		.collect();
	let (y, _) = run(src, &input);
	assert_eq!(
		words(&y[..48]),
		[
			99,
			400,
			10300,
			0x12345678,
			(-8i32) as u32,
			10,
			10,
			3011,
			100,
			0,
			400,
			3110
		]
	);
}

#[test]
fn vectors_memory_and_the_stack() {
	let src = "
        void controlled_act(const u32 *params) {
            const half8 *restrict x = (const half8 *)params[0];
            half8 *restrict y = (half8 *)params[9];
            half8 buf[2];
            /* the maximum of 6 vectors, two per round: CMU.MAX rewrites the flags the loop's branch reads */
            register half8 m[2];
            m[0] = x[0]; m[1] = x[1];
            const half8 *p = x + 2;
            for (u32 r = 0; r < 2; r++) {
                for (int u = 0; u < 2; u++) m[u] = max(m[u], p[u]);
                p += 2;
            }
            half8 t = max(m[0], m[1]);
            y[0] = t;
            /* a 128-bit store and load next to 64-bit halves, through the stack */
            buf[0] = t;
            buf[1] = t + t;
            barrier();
            y[1] = buf[1];
            half8 *b = buf;
            y[2] = *b * 0.5;
            ((half *)y)[24] = t[3];
            ((float *)y)[13] = sum(f16_to_f32(t, 1));
        }";
	let x: Vec<f64> = (0..48).map(|k| ((k * 37) % 23) as f64 - 7.5).collect();
	let input: Vec<u8> = x.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect();
	let (y, _) = run(src, &input);
	let h = |k: usize| h2d(u16::from_le_bytes([y[2 * k], y[2 * k + 1]]));
	let max: Vec<f64> = (0..8)
		.map(|l| (0..6).map(|v| x[8 * v + l]).fold(f64::MIN, f64::max))
		.collect();
	for l in 0..8 {
		assert_eq!(
			(h(l), h(8 + l), h(16 + l)),
			(max[l], 2.0 * max[l], 0.5 * max[l]),
			"lane {l}"
		);
	}
	assert_eq!(h(24), max[3]);
	assert_eq!(
		f32::from_le_bytes(y[52..56].try_into().unwrap()),
		max[4..].iter().sum::<f64>() as f32
	);
}

#[test]
fn values_not_variables() {
	// A variable assigned again is a new value: the copies of an unrolled body
	// that share `t` are scheduled as freely as those with a `t` each.
	let src = |shared: i64| {
		let k = npu::shave::dsl::compile(
			"
        void controlled_act(const u32 *params) {
            const half8 *restrict x = (const half8 *)params[0];
            half8 *restrict y = (half8 *)params[9];
            u32 n = 2;
            const half8 k = 2.0;
        #if SHARED
            half8 t;
        #endif
            do {
                for (int u = 0; u < 8; u++) {
        #if !SHARED
                    half8 t;
        #endif
                    t = x[u] * k;
                    t = t + k;
                    y[u] = t * t;
                }
                x += 8;
                y += 8;
                n--;
            } while (n);
        }",
			&[("SHARED".into(), shared)],
		);
		k.unwrap_or_else(|e| panic!("{e}")).report
	};
	assert_eq!(src(1), src(0));
	// A variable holds no register between its last read and its next value:
	// `t` around a loop that needs all 24 vector registers.
	let src = "
        void controlled_act(const u32 *params) {
            const half8 *restrict x = (const half8 *)params[0];
            half8 *restrict y = (half8 *)params[9];
            half8 t = x[0];
            y[12] = t + t;
            u32 n = 1;
            do {
                register half8 m[24];
                for (int u = 0; u < 24; u++) m[u] = x[u];
                barrier();
                for (int u = 0; u < 12; u++) y[u] = m[u] + m[u + 12];
                n--;
            } while (n);
            t = x[1];
            y[13] = t + t;
            /* set on every path, by either branch */
            u32 a;
            if (n) a = 1; else a = 2;
            ((u32 *)y)[56] = a;
        }";
	let x: Vec<f64> = (0..192).map(|k| ((k * 29) % 31) as f64 - 9.0).collect();
	let input: Vec<u8> = x.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect();
	let (y, _) = run(src, &input);
	let h = |k: usize| h2d(u16::from_le_bytes([y[2 * k], y[2 * k + 1]]));
	for k in 0..96 {
		assert_eq!(h(k), x[k] + x[k + 96], "element {k}");
	}
	for l in 0..8 {
		assert_eq!(
			(h(96 + l), h(104 + l)),
			(2.0 * x[l], 2.0 * x[8 + l]),
			"lane {l}"
		);
	}
	assert_eq!(words(&y[224..228]), [2]);
}

#[test]
fn blocks_and_pointer_arrays() {
	// Two 128-bit stores in a block's last cycle keep their LSUs for one more:
	// the next block's loads must not start in it. The stores go through an
	// array of restrict pointers, each element memory of its own, so nothing
	// orders them.
	let src = "
        void controlled_act(const u32 *params) {
            const half8 *restrict x = (const half8 *)params[0];
            half8 *restrict y = (half8 *)params[9];
            register half8 *restrict p[2];
            p[0] = y;
            p[1] = y + 2;
            half8 a = x[0], b = x[1];
            barrier();
            p[0][0] = a;
            p[1][0] = b;
            barrier();
            half8 c = x[2], d = x[3];
            y[1] = c + d;
            /* a loop of one round is no loop */
            for (u32 r = 0; r < 1; r++) y[3] = c;
        }";
	let x: Vec<f64> = (0..32).map(|k| k as f64 - 11.0).collect();
	let input: Vec<u8> = x.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect();
	let (y, _) = run(src, &input);
	let h = |k: usize| h2d(u16::from_le_bytes([y[2 * k], y[2 * k + 1]]));
	for l in 0..8 {
		assert_eq!(
			(h(l), h(8 + l), h(16 + l), h(24 + l)),
			(x[l], x[16 + l] + x[24 + l], x[8 + l], x[16 + l]),
			"lane {l}"
		);
	}
	assert!(npu::shave::dsl::compile(src, &[])
		.unwrap()
		.report
		.is_empty());
}

/// Run a kernel on tensors given as MemRefData records (inputs, then
/// outputs): (bytes, dims innermost first), contiguous f16. The tensors after
/// the run and the statistics; None when the emulator lacks an instruction.
fn invoke(prog: &Program, tensors: &[(Vec<u8>, Vec<u32>)]) -> Option<(Vec<Vec<u8>>, Stats)> {
	let (pa, sa) = (0x8003_0000u32, 0x2e80_0000u32);
	let mut m = Machine::new();
	let mut p = Vec::new();
	for (k, (bytes, dims)) in tensors.iter().enumerate() {
		let (data, meta) = (
			0x2e00_0000 + 0x10_0000 * k as u32,
			0x8004_0000 + 0x100 * k as u32,
		);
		m.map(data, bytes.clone());
		// dims (u32), then the strides (i64, in bits)
		let mut md: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
		md.resize(0x40, 0);
		let mut stride = 16u64;
		for d in dims {
			md.extend(stride.to_le_bytes());
			stride *= *d as u64;
		}
		m.map(meta, md);
		let mut rec = [0u32; 9];
		(rec[0], rec[2], rec[3], rec[4]) = (data, dims.len() as u32, meta, meta + 0x40);
		p.extend(rec.iter().flat_map(|w| w.to_le_bytes()));
	}
	m.map(pa, p);
	m.map(sa, vec![0; 0x1c00]);
	m.i[18] = pa;
	m.i[19] = sa + 0x1c00;
	if let Err(e) = m.run(prog, None, 50_000_000) {
		assert!(e.contains("not modeled") || e.contains("unknown"), "{e}");
		return None;
	}
	assert_eq!(m.i[19], sa + 0x1c00, "the stack pointer is not restored");
	let out = tensors
		.iter()
		.enumerate()
		.map(|(k, t)| {
			m.read(0x2e00_0000 + 0x10_0000 * k as u32, t.0.len())
				.unwrap()
				.to_vec()
		})
		.collect();
	Some((out, m.stats))
}

fn halves(v: &[f64]) -> Vec<u8> {
	v.iter().flat_map(|&e| d2h(e).to_le_bytes()).collect()
}

fn unhalves(b: &[u8]) -> Vec<f64> {
	b.chunks(2)
		.map(|c| h2d(u16::from_le_bytes([c[0], c[1]])))
		.collect()
}

/// A deterministic spread over -1 .. 1.
fn noise(count: usize, seed: u64) -> Vec<f64> {
	let mut s = seed;
	(0..count)
		.map(|_| {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			(s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
		})
		.collect()
}

#[test]
fn gated_norm_from_the_kernel_language() {
	// [S, H, 2, D]: rows (h, 0) = o, (h, 1) = z; B rows (h, 0) = gw, eps in row (0, 1)
	let (s, h, d, eps) = (3usize, 8usize, 128usize, 1e-3f32);
	let a: Vec<f64> = noise(s * h * 2 * d, 0x9e3779b97f4a7c15)
		.iter()
		.map(|x| h2d(d2h(x * 6.0)))
		.collect();
	let mut b: Vec<f64> = noise(h * 2 * d, 0x2545f4914f6cdd1d)
		.iter()
		.map(|x| h2d(d2h(1.25 + 0.75 * x)))
		.collect();
	b[d] = 0.0;
	let mut bb = halves(&b);
	bb[2 * d..2 * d + 4].copy_from_slice(&eps.to_le_bytes());
	let dims = |n: usize| vec![d as u32, 2 * h as u32, n as u32];
	let tensors = [
		(halves(&a), dims(s)),
		(bb, dims(1)),
		(vec![0xee; 2 * a.len()], dims(s)),
	];
	let k = kernel("gnorm.sk", &[]);
	let (out, stats) = invoke(&k.program, &tensors).unwrap();
	assert_eq!(
		(
			stats.landings,
			stats.sau_vau,
			stats.undefined,
			stats.flags,
			stats.conflicts
		),
		([0; 8], 0, 0, 0, 0),
		"{:?}",
		stats.notes
	);
	let y = unhalves(&out[2]);
	for q in 0..s {
		for hh in 0..h {
			let o = &a[(q * h + hh) * 2 * d..][..d];
			let z = &a[(q * h + hh) * 2 * d + d..][..d];
			let rs = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / d as f64 + eps as f64).sqrt();
			for j in 0..d {
				let want = o[j] * rs * b[hh * 2 * d + j] * z[j] / (1.0 + (-z[j]).exp());
				let got = y[(q * h + hh) * 2 * d + j];
				assert!(
					(got - want).abs() <= 4e-3 * want.abs() + 1e-4,
					"position {q} head {hh} element {j}: {got} for {want}"
				);
			}
		}
	}
	println!(
		"gnorm: {} cycles per head; {}",
		stats.cycles / (s * h) as u64,
		k.report.join("; ")
	);
	assert!(k.elf().unwrap().len() > 500);
}

#[test]
fn swish_l2_norm_from_the_kernel_language() {
	// [n, 32, 128] rows: s = x * sigmoid(x), y = s / sqrt(sum(s^2) + 1e-6);
	// rows at scales from 1e-3 to 30, and one of zeros
	let (n, r, d) = (3usize, 32usize, 128usize);
	let mut a: Vec<f64> = noise(n * r * d, 0x9e3779b97f4a7c15);
	for (i, row) in a.chunks_mut(d).enumerate() {
		let sc = [1e-3, 0.05, 1.0, 4.0, 30.0, 0.0][i % 6];
		for x in row {
			*x = h2d(d2h(*x * sc));
		}
	}
	let dims = vec![d as u32, r as u32, n as u32, 1];
	let tensors = [(halves(&a), dims.clone()), (vec![0xee; 2 * a.len()], dims)];
	let k = kernel("l2n.sk", &[]);
	let (out, stats) = invoke(&k.program, &tensors).unwrap();
	assert_eq!(
		(
			stats.landings,
			stats.sau_vau,
			stats.undefined,
			stats.flags,
			stats.conflicts
		),
		([0; 8], 0, 0, 0, 0),
		"{:?}",
		stats.notes
	);
	let y = unhalves(&out[1]);
	for (i, row) in a.chunks(d).enumerate() {
		// the kernel's s is f16, times 16; subnormal inputs read as zero
		let s: Vec<f64> = row
			.iter()
			.map(|&x| if x.abs() < 6.1035e-5 { 0.0 } else { x })
			.map(|x| h2d(d2h(16.0 * x / (1.0 + (-x).exp()))) / 16.0)
			.collect();
		let rs = 1.0 / (s.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
		for j in 0..d {
			let (want, got) = (s[j] * rs, y[i * d + j]);
			assert!(
				(got - want).abs() <= 2e-3 * want.abs() + 2e-4,
				"row {i} element {j}: {got} for {want}"
			);
		}
	}
	println!(
		"l2n: {} cycles per row; {}",
		stats.cycles / (n * r) as u64,
		k.report.join("; ")
	);
	assert!(k.elf().unwrap().len() > 500);
}

/// `rowq.sk`: each row scaled to a max |y| of 127 / 128 (a floor of 1e-3
/// on the row's max), rows of mixed magnitude, one all zero; plain and
/// with `PAD` lanes that must come back holding the max.
#[test]
fn per_row_int8_scale_from_the_kernel_language() {
	let (l, n) = (640usize, 7usize);
	for pad in [0usize, 32] {
		let row = l + pad;
		let mut a: Vec<f64> = noise(n * row, 0x2545f4914f6cdd1d);
		for (i, r) in a.chunks_mut(row).enumerate() {
			let sc = [1e-4, 0.05, 1.0, 9.0, 400.0, 0.0, 3e3][i % 7];
			for x in r.iter_mut() {
				*x = h2d(d2h(*x * sc));
			}
			r[l..].iter_mut().for_each(|x| *x = 0.0);
		}
		// one outlier channel
		a[3 * row + 77] = -h2d(d2h(900.0));
		let dims = vec![row as u32, n as u32, 1];
		let tensors = [(halves(&a), dims.clone()), (vec![0xee; 2 * a.len()], dims)];
		let defs: &[(&str, i64)] = if pad > 0 { &[("PAD", 32)] } else { &[] };
		let k = kernel("rowq.sk", defs);
		let (out, stats) = invoke(&k.program, &tensors).unwrap();
		assert_eq!(
			(
				stats.landings,
				stats.sau_vau,
				stats.undefined,
				stats.flags,
				stats.conflicts
			),
			([0; 8], 0, 0, 0, 0),
			"{:?}",
			stats.notes
		);
		let y = unhalves(&out[1]);
		for (i, r) in a.chunks(row).enumerate() {
			// subnormal inputs read as zero
			let m = r[..l]
				.iter()
				.map(|x| if x.abs() < 6.1035e-5 { 0.0 } else { x.abs() })
				.fold(0.0, f64::max)
				.max(h2d(d2h(1e-3)));
			let rc = 127.0 / 128.0 / m;
			for j in 0..l {
				let x = if r[j].abs() < 6.1035e-5 { 0.0 } else { r[j] };
				let (want, got) = (x * rc, y[i * row + j]);
				assert!(
					(got - want).abs() <= 1.5e-3 * want.abs() + 1e-6,
					"pad {pad} row {i} element {j}: {got} for {want}"
				);
				assert!(got.abs() <= 127.0 / 128.0 * 1.001);
			}
			for j in l..row {
				let got = y[i * row + j];
				assert!(
					(got - m).abs() <= 1e-3 * m,
					"pad {pad} row {i} pad lane {j}: {got} for max {m}"
				);
			}
		}
		println!(
			"rowq pad {pad}: {} cycles per row of {l}; {}",
			stats.cycles / n as u64,
			k.report.join("; ")
		);
		assert!(k.elf().unwrap().len() > 300);
	}
}

#[test]
fn softmax_with_8_bit_probabilities() {
	let (n, rows) = (1024usize, 3usize);
	let x = scores(rows * n);
	let dims = vec![n as u32, rows as u32];
	let tensors = [
		(halves(&x), dims.clone()),
		(halves(&[0.0]), vec![1]),
		(vec![0xee; rows * n], dims),
	];
	let k = kernel("softmax_q8.sk", &[("N", n as i64)]);
	let (out, stats) = invoke(&k.program, &tensors).unwrap();
	assert_eq!(
		(
			stats.landings,
			stats.sau_vau,
			stats.undefined,
			stats.flags,
			stats.conflicts
		),
		([0; 8], 0, 0, 0, 0),
		"{:?}",
		stats.notes
	);
	for (row, (c, x)) in out[2].chunks(n).zip(x.chunks(n)).enumerate() {
		let max = x[..n - 2].iter().cloned().fold(f64::MIN, f64::max);
		assert_eq!(
			h2d(u16::from_le_bytes([c[n - 2], c[n - 1]])),
			max,
			"row {row}: max"
		);
		// every code within one of its value (dithered: half a code more than
		// rounding), the row's sum close: the offsets cancel over 64 slots
		let want: Vec<f64> = x[..n - 2].iter().map(|e| 255.0 * (e - max).exp()).collect();
		for j in 0..n - 2 {
			assert!(
				(c[j] as f64 - want[j]).abs() <= 1.0 + 6e-3 * want[j],
				"row {row} slot {j}: {} for {}",
				c[j],
				want[j]
			);
		}
		let (got, w): (f64, f64) = (
			c[..n - 2].iter().map(|&v| v as f64).sum(),
			want.iter().sum(),
		);
		assert!(
			(got - w).abs() <= 12.0 + 6e-3 * w,
			"row {row}: sum {got} for {w}"
		);
	}
	println!(
		"softmax_q8: {} cycles per row; {}",
		stats.cycles / rows as u64,
		k.report.join("; ")
	);
	assert!(k.elf().unwrap().len() > 500);
}

#[test]
fn block_inverse_from_the_kernel_language() {
	// 6 matrices: a pass of 4 and two single ones
	let (g, c, bs) = (6usize, 64usize, 16usize);
	let a: Vec<f64> = noise(g * c * c, 0x6a09e667f3bcc909)
		.iter()
		.map(|x| h2d(d2h(0.6 * x)))
		.collect();
	let dims = vec![(c * c) as u32, g as u32];
	let tensors = [(halves(&a), dims.clone()), (vec![0xee; 2 * a.len()], dims)];
	let k = kernel("trinv.sk", &[("BS", bs as i64)]);
	let (out, stats) = invoke(&k.program, &tensors).unwrap();
	assert_eq!(
		(
			stats.landings,
			stats.sau_vau,
			stats.undefined,
			stats.flags,
			stats.conflicts
		),
		([0; 8], 0, 0, 0, 0),
		"{:?}",
		stats.notes
	);
	let t = unhalves(&out[1]);
	// forward substitution in the kernel's order, every step rounded to f16
	// (a multiply takes a subnormal factor as zero)
	let r = |x: f64| h2d(d2h(x));
	let fl = |x: f64| if x.abs() < 6.103515625e-5 { 0.0 } else { x };
	for m in 0..g {
		let (am, tm) = (&a[m * c * c..][..c * c], &t[m * c * c..][..c * c]);
		let mut want = vec![0.0; c * c];
		for i in 0..c {
			let j0 = i / bs * bs;
			for k in j0..i {
				for j in j0..j0 + bs {
					want[i * c + j] =
						r(want[i * c + j] - r(fl(am[i * c + k]) * fl(want[k * c + j])));
				}
			}
			want[i * c + i] = 1.0;
		}
		assert_eq!(tm, &want[..], "matrix {m}");
	}
	println!(
		"trinv: {} cycles per matrix; {}",
		stats.cycles / g as u64,
		k.report.join("; ")
	);
	assert!(k.elf().unwrap().len() > 500);
}

#[test]
fn floating_constants_are_folded() {
	let src = "
        #define D 128
        const float INVD = 1.0 / D;
        const half THIRD = 1.0 / 3;
        _Static_assert(INVD * D == 1.0 && INVD < 1 && !(INVD > 0.5), \"folded\");
        void controlled_act(const u32 *params) {
            float *y = (float *)params[9];
            y[0] = INVD;
            y[1] = -(2.0 * D) + 0.5;
            y[2] = 1 / D;
            y[3] = THIRD;
            float4 v = INVD * 4;
            y[4] = v[3];
            half8 h = THIRD;
            float4 hf = f16_to_f32(h, 0);
            y[5] = hf[0];
            #if 3 / 2 == 1
            y[6] = 3 / 2.0;
            #endif
            float4 w = v * (1.0 / 4);
            y[7] = w[0];
        }";
	let (out, _) = run(src, &[]);
	let y: Vec<f32> = words(&out).iter().map(|w| f32::from_bits(*w)).collect();
	assert_eq!(
		y[..5],
		[1.0 / 128.0, -255.5, 0.0, (1.0f64 / 3.0) as f32, 1.0 / 32.0]
	);
	assert_eq!(y[5] as f64, h2d(d2h(1.0 / 3.0)));
	assert_eq!(y[6..8], [1.5, 1.0 / 128.0]);
	// a constant expression costs what its value costs
	let ops = |e: &str| {
		npu::shave::dsl::compile(&format!("void controlled_act(u32 *params) {{ float *y = (float *)params[9]; float4 v = {e}; y[0] = v[0]; }}"), &[]).unwrap().text()
	};
	assert_eq!(ops("0.5 * 0.25 + 1.0 / 8"), ops("0.25"));
	let err = |text: &str| {
		npu::shave::dsl::compile(text, &[])
			.err()
			.unwrap_or_else(|| panic!("compiled: {text}"))
	};
	assert!(err("const float X = 1.0 / 0;
void controlled_act(u32 *params) { }")
	.contains("line 1: / works on constants only"));
	assert!(err("const float X = 1.5;
void controlled_act(u32 *params) { u32 a = X; }")
	.contains("line 2: a floating constant where u32 is needed"));
	assert!(err("const float X = 1.5;
void controlled_act(u32 *params) { X = 2.0; }")
	.contains("line 2: X cannot be assigned"));
	assert!(
		err("void controlled_act(u32 *params) { u32 a[2]; a[1.0 + 1] = 0; }")
			.contains("must be a compile-time constant")
	);
	assert!(err("#if 1.0 / 2
#endif
void controlled_act(u32 *params) { }")
	.contains("a float where an integer constant is needed"));
}

#[test]
fn errors_name_the_line() {
	let err = |body: &str| {
		npu::shave::dsl::compile(
			&format!(
				"void controlled_act(u32 *params) {{
{body}
}}"
			),
			&[],
		)
		.err()
		.unwrap_or_else(|| panic!("compiled: {body}"))
	};
	assert!(err("u32 a;
u32 b = a + 1;")
	.contains("line 3: a is read before it is set"));
	// set on one path only
	assert!(err("u32 a;
if (params[0]) a = 1;
u32 b = a + 1;")
	.contains("line 4: a is read before it is set"));
	assert!(err("half8 v = frob(params);").contains("line 2: unknown function frob"));
	assert!(
		err("half8 v = exp(params);").contains("line 2: exp: the arguments do not fit exp(half8)")
	);
	assert!(err("half8 v = 1.5;
float4 w = v;")
	.contains("line 3: half8 where float4 is needed"));
	assert!(err("half8 *p = params;").contains("u32 * where half8 * is needed (cast it)"));
	assert!(err("u32 a = params[0];
u32 b = a / 3;")
	.contains("line 3: / works on constants only"));
	assert!(err("u32 a = params[N];").contains("line 2"));
	assert!(err("u32 i = 2;
u32 a = params[i];")
	.contains("line 3: an index"));
	assert!(err("half8 *restrict x = (half8 *)params[0];
half8 *restrict y = (half8 *)params[9];
half8 *p = x;
p = y;")
	.contains("line 5: a pointer based on a restrict pointer"));
	assert!(err("#pragma unroll
for (u32 i = 0; i < params[0]; i++) { }")
	.contains("the condition of an unrolled loop must be a compile-time constant"));
	assert!(npu::shave::dsl::compile(
		"#if N > 2
#error N is too large
#endif
void k(u32 *p) { }",
		&[("N".into(), 3)]
	)
	.err()
	.unwrap()
	.contains("line 2: #error N is too large"));
	// more values alive than the processor has registers
	let wide = npu::shave::dsl::compile(
		&source("softmax_lse.sk"),
		&[("N".into(), 4096), ("U".into(), 16)],
	)
	.err()
	.unwrap();
	assert!(wide.contains("no vector register left"), "{wide}");
	assert!(
		npu::shave::dsl::compile(&source("softmax_lse.sk"), &[("N".into(), 1000)])
			.err()
			.unwrap()
			.contains("static assertion failed: N / 8 / U")
	);
	assert!(
		npu::shave::dsl::compile(&source("softmax_lse.sk"), &[("M".into(), 1)])
			.err()
			.unwrap()
			.contains("-DM: the kernel does not use M")
	);
}

#[test]
fn every_operation_assembles() {
	use npu::shave::dsl::lower::{INTRINSICS, OPERATORS};
	// one use per intrinsic and operator: the instruction form must be in isa.tbl
	let var = |c: char| match c {
		'H' | 'V' => "vh",
		'F' => "vf",
		'I' => "vi",
		'h' => "sh",
		'f' => "sf",
		'i' => "si",
		'p' => "ph",
		_ => "1",
	};
	let ty = |c: char| match c {
		'H' | 'V' => "half8",
		'F' => "float4",
		'I' => "int4",
		'h' => "half",
		'f' => "float",
		_ => "u32",
	};
	let mut body = String::from("half8 vh = 1.5; float4 vf = 2.5; int4 vi = 3; half sh = vh[0]; float sf = vf[0]; u32 si = 3; half *ph = (half *)params[9];
");
	for (k, (name, args, ret, ..)) in INTRINSICS.iter().enumerate() {
		let a: Vec<&str> = args.chars().map(var).collect();
		body += &format!(
			"{} r{k} = {name}({});
",
			ty(*ret),
			a.join(", ")
		);
	}
	for (k, (op, class, _)) in OPERATORS.iter().enumerate() {
		let (v, rhs) = (
			var(*class),
			if matches!(*op, "<<" | ">>") {
				"2"
			} else {
				var(*class)
			},
		);
		body += &format!(
			"{} o{k} = {v} {op} {rhs};
",
			ty(*class)
		);
	}
	let k = npu::shave::dsl::compile(
		&format!(
			"void controlled_act(u32 *params) {{
{body}}}"
		),
		&[],
	)
	.unwrap_or_else(|e| {
		panic!(
			"{e}
{body}"
		)
	});
	k.elf().unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn hyper_connection_kernels() {
	let (n, w, hs) = (3usize, 2560usize, 4usize);
	let k = kernel("hcmix.sk", &[]);
	let clean = |s: &Stats| {
		assert_eq!(
			(s.landings, s.sau_vau, s.undefined, s.flags, s.conflicts),
			([0; 8], 0, 0, 0, 0),
			"{:?}",
			s.notes
		)
	};
	// the rows as [20, 4] or merged into [80] (the compiler does either)
	for merged in [false, true] {
		let rows = |head: &[u32], p: usize| -> Vec<u32> {
			let mut d = head.to_vec();
			if merged {
				d.push(80);
			} else {
				d.extend([20, hs as u32]);
			}
			d.push(p as u32);
			d
		};
		// mode N: A = x [128, rows, n], B = w [128, rows, 1]
		let x: Vec<f64> = noise(n * hs * w, 0x9e3779b97f4a7c15)
			.iter()
			.enumerate()
			// streams of different scales
			.map(|(i, v)| h2d(d2h(v * [0.5, 3.0, 40.0, 200.0][(i / w) % hs])))
			.collect();
		let g: Vec<f64> = noise(hs * w, 0x2545f4914f6cdd1d)
			.iter()
			.map(|v| h2d(d2h(1.0 + 0.5 * v)))
			.collect();
		let tensors = [
			(halves(&x), rows(&[128], n)),
			(halves(&g), rows(&[128], 1)),
			(vec![0xee; 2 * x.len()], rows(&[128], n)),
		];
		let (out, stats) = invoke(&k.program, &tensors).unwrap();
		clean(&stats);
		let y = unhalves(&out[2]);
		let cyc = stats.cycles / n as u64;
		for q in 0..n {
			for s in 0..hs {
				let r = &x[(q * hs + s) * w..][..w];
				let rs = 1.0 / (r.iter().map(|v| v * v).sum::<f64>() / w as f64 + 1e-6).sqrt();
				for j in 0..w {
					let want = r[j] * rs * g[s * w + j];
					let got = y[(q * hs + s) * w + j];
					assert!(
						(got - want).abs() <= 3e-3 * want.abs() + 1e-4,
						"N: position {q} stream {s} element {j}: {got} for {want}"
					);
				}
			}
		}
		// mode M: A [128, 2, rows, n] (pair 0 = u, 1 = xn), B [128, 1, rows, 1] = su
		let a: Vec<f64> = noise(n * hs * 2 * w, 0x1234567)
			.iter()
			.map(|v| h2d(d2h(v * 4.0)))
			.collect();
		let su: Vec<f64> = noise(hs * w, 0x7654321)
			.iter()
			.map(|v| h2d(d2h(1.0 + 0.5 * v)))
			.collect();
		let tensors = [
			(halves(&a), rows(&[128, 2], n)),
			(halves(&su), rows(&[128, 1], 1)),
			(vec![0xee; 2 * a.len()], rows(&[128, 2], n)),
		];
		let (out, stats) = invoke(&k.program, &tensors).unwrap();
		clean(&stats);
		let y = unhalves(&out[2]);
		// element (q, s, block b, pair p, e)
		let at = |q: usize, s: usize, j: usize, p: usize| {
			(((q * hs + s) * 20 + j / 128) * 2 + p) * 128 + j % 128
		};
		for q in 0..n {
			for j in 0..w {
				let want: f64 = (0..hs)
					.map(|s| a[at(q, s, j, 1)] / (1.0 + (-a[at(q, s, j, 0)] * su[s * w + j]).exp()))
					.sum::<f64>() * 0.25;
				let got = y[at(q, 0, j, 0)];
				assert!(
					(got - want).abs() <= 4e-3 * want.abs() + 2e-3,
					"M: position {q} element {j}: {got} for {want}"
				);
			}
		}
		println!(
			"hcmix (rows merged {merged}): {} cycles per position (N + M)",
			cyc + stats.cycles / n as u64
		);
		// mode G: A [128, 2, 48 | 96, n] (pair 0 = o, 1 = z), B = gw [128]
		let (gh, d) = (48usize, 128usize);
		let gdims = |p: usize| -> Vec<u32> {
			if merged {
				vec![d as u32, 2 * gh as u32, p as u32]
			} else {
				vec![d as u32, 2, gh as u32, p as u32]
			}
		};
		let a: Vec<f64> = noise(n * gh * 2 * d, 0x51afd7ed558ccd)
			.iter()
			.enumerate()
			// o at the output scale 64 (heads of different sizes), z around 0
			.map(|(i, v)| {
				h2d(d2h(if (i / d) % 2 == 0 {
					v * [8.0, 64.0, 360.0][(i / (2 * d)) % 3]
				} else {
					v * 3.0
				}))
			})
			.collect();
		let gw: Vec<f64> = noise(d, 0xc4ceb9fe1a85ec53)
			.iter()
			.map(|v| h2d(d2h(1.0 + 0.5 * v)))
			.collect();
		let tensors = [
			(halves(&a), gdims(n)),
			(halves(&gw), vec![d as u32, 1, 1]),
			(vec![0xee; 2 * a.len()], gdims(n)),
		];
		let (out, stats) = invoke(&k.program, &tensors).unwrap();
		clean(&stats);
		let y = unhalves(&out[2]);
		for q in 0..n {
			for hh in 0..gh {
				let o = &a[(q * gh + hh) * 2 * d..][..d];
				let z = &a[(q * gh + hh) * 2 * d + d..][..d];
				let rs = 1.0
					/ (o.iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-6 * 64.0 * 64.0).sqrt();
				for j in 0..d {
					let want = o[j] * rs * gw[j] / (1.0 + (-z[j]).exp());
					let got = y[(q * gh + hh) * 2 * d + j];
					assert!(
						(got - want).abs() <= 4e-3 * want.abs() + 1e-4,
						"G: position {q} head {hh} element {j}: {got} for {want}"
					);
				}
			}
		}
		println!(
			"hcmix (rows merged {merged}): G {} cycles per position",
			stats.cycles / n as u64
		);
		// mode G with B per head (`graphs::GN_FOLD`): B [128, 2, 48 | 96, 1]
		// = (gw times a prescale, z's row scale)
		let wz: Vec<f64> = noise(gh * 2 * d, 0x3c6ef372fe94f82b)
			.iter()
			.enumerate()
			.map(|(i, v)| {
				h2d(d2h(if (i / d) % 2 == 0 {
					v * 40.0
				} else {
					1.0 + 0.5 * v
				}))
			})
			.collect();
		let tensors = [
			(halves(&a), gdims(n)),
			(halves(&wz), gdims(1)),
			(vec![0xee; 2 * a.len()], gdims(n)),
		];
		let (out, stats) = invoke(&k.program, &tensors).unwrap();
		clean(&stats);
		let y = unhalves(&out[2]);
		for q in 0..n {
			for hh in 0..gh {
				let o = &a[(q * gh + hh) * 2 * d..][..d];
				let z = &a[(q * gh + hh) * 2 * d + d..][..d];
				let w = &wz[hh * 2 * d..][..d];
				let zs = &wz[hh * 2 * d + d..][..d];
				let rs = 1.0
					/ (o.iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-6 * 64.0 * 64.0).sqrt();
				for j in 0..d {
					let want = o[j] * rs * w[j] / (1.0 + (-z[j] * zs[j]).exp());
					let got = y[(q * gh + hh) * 2 * d + j];
					assert!(
						(got - want).abs() <= 5e-3 * want.abs() + 1e-3,
						"G per head: position {q} head {hh} element {j}: {got} for {want}"
					);
				}
			}
		}
		println!(
			"hcmix (rows merged {merged}): G per head {} cycles per position",
			stats.cycles / n as u64
		);
		// the same with (128, pair, head) merged by the compiler, whole or a
		// tile of 12 heads: A [256 nh, n], B [256 nh, 1]
		for nh in [gh, 12] {
			let a: Vec<f64> = (0..n)
				.flat_map(|q| a[q * gh * 2 * d..][..nh * 2 * d].to_vec())
				.collect();
			let wz = &wz[..nh * 2 * d];
			let tensors = [
				(halves(&a), vec![(nh * 2 * d) as u32, n as u32]),
				(halves(wz), vec![(nh * 2 * d) as u32, 1]),
				(vec![0xee; 2 * a.len()], vec![(nh * 2 * d) as u32, n as u32]),
			];
			let (out, stats) = invoke(&k.program, &tensors).unwrap();
			clean(&stats);
			let y = unhalves(&out[2]);
			for q in 0..n {
				for hh in 0..nh {
					let o = &a[(q * nh + hh) * 2 * d..][..d];
					let z = &a[(q * nh + hh) * 2 * d + d..][..d];
					let w = &wz[hh * 2 * d..][..d];
					let zs = &wz[hh * 2 * d + d..][..d];
					let rs = 1.0
						/ (o.iter().map(|x| x * x).sum::<f64>() / d as f64 + 1e-6 * 64.0 * 64.0)
							.sqrt();
					for j in 0..d {
						let want = o[j] * rs * w[j] / (1.0 + (-z[j] * zs[j]).exp());
						let got = y[(q * nh + hh) * 2 * d + j];
						assert!(
							(got - want).abs() <= 5e-3 * want.abs() + 1e-3,
							"G merged {nh}: position {q} head {hh} element {j}: {got} for {want}"
						);
					}
				}
			}
		}
	}
}
