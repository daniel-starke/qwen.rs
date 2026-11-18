//! Experimental far-block attention graphs (one block of `cap` key slots, `s`
//! positions) around u8 probabilities and P V as the DPU's integer FC:
//! measured and rejected (no gain: 3.42-3.46 ms per block against the far
//! graph's 3.38-3.44, which fuses its MatMul -> Softmax -> MatMul). Kept to
//! repeat the measurements.
//!
//!   farx VARIANT S CAP OUT.xml [RV] [T]
//!
//! - `qa` / `vb`: the two graphs of a block. `qa`: scores -> the
//!   `softmax_q8` kernel (rename the IR's Maximum to Greater: its carrier);
//!   `vb`: the codes as a u8 input, V and a row of ones as int8 weights
//!   `vq` [1, 4, 257, CAP] (unit `RV`). `qas` (`RV` = slices) / `qac`: `qa`
//!   in row slices / on a Convert.
//! - `pv1` / `pv2` / `pv1b`: one graph, P through one / two static
//!   FakeQuantizes (`pv2`: min(P, T) and max(P, T)); `pin` / `pinb`: the
//!   integer FC alone; `pvk`: the SHAVE Quantize as the carrier.
//! - `nosm`: the far graph without its Softmax (a timing stand-in).
use npubuild::g::{Graph, Ty, V};

const HKV: usize = 4;
const G: usize = 6;
const DA: usize = 256;

fn main() {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let (s, cap): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
	let rv: f64 = a.get(4).map_or(32.0, |v| v.parse().unwrap());
	let t: f64 = a.get(5).map_or(1.0 / 32.0, |v| v.parse().unwrap());
	if a[0] == "vb" {
		// the second graph of a far block: P as the first graph's bytes
		// (p = code / 255), V and a row of ones as int8 weights [257, cap]
		// per KV head (unit `rv`) -> o [1, 4, 6 s, 257]
		let mut g = Graph::new();
		let pq = g.parameter("pq", Ty::U8, &[1, HKV, G * s, cap]);
		let c = g.convert(&pq, Ty::F16);
		let k = g.sf(Ty::F16, 1.0 / 255.0);
		let p = g.mul(&c, &k);
		let w = g.parameter("vq", Ty::I8, &[1, HKV, DA + 1, cap]);
		let c = g.convert(&w, Ty::F16);
		let k = g.c16(&[1, 1, 1, 1], &[rv]);
		let w = g.mul(&c, &k);
		let o = g.matmul(&p, &w, false, true);
		g.result(&o, "o");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0].starts_with("pin") {
		// the integer FC alone: P as a u8 input (p = code / 255); `pinb`: V
		// as one batched weight input
		let mut g = Graph::new();
		let pq = g.parameter("pq", Ty::U8, &[1, HKV, G * s, cap]);
		let c = g.convert(&pq, Ty::F16);
		let sc = g.sf(Ty::F16, 1.0 / 255.0);
		let p = g.mul(&c, &sc);
		let o = if a[0] == "pinb" {
			let w = g.parameter("vq", Ty::I8, &[1, HKV, DA, cap]);
			let c = g.convert(&w, Ty::F16);
			let sc = g.c16(&[1, 1, 1, 1], &[rv / 127.0]);
			let w = g.mul(&c, &sc);
			g.matmul(&p, &w, false, true)
		} else {
			let hs = g.split(&p, 1, HKV);
			let os: Vec<V> = (0..HKV)
				.map(|h| {
					let w = g.parameter(&format!("vq{h}"), Ty::I8, &[DA, cap]);
					let c = g.convert(&w, Ty::F16);
					let sc = g.c16(&[1, 1], &[rv / 127.0]);
					let w = g.mul(&c, &sc);
					g.matmul(&hs[h], &w, false, true)
				})
				.collect();
			g.concat(&os, 1)
		};
		g.result(&o, "o");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	let mut g = Graph::new();
	let qg = g.parameter("qg", Ty::F16, &[1, HKV, G * s, DA]);
	let kb = g.parameter("kblk", Ty::F16, &[1, cap, HKV, DA]);
	let k = g.transpose(&kb, &[0, 2, 1, 3]);
	let sc = g.matmul(&qg, &k, false, true);
	let p = g.softmax(&sc, 3);
	if a[0] == "qas" {
		// `qa` with the queries in `rv` slices (rows), a MatMul and a kernel
		// call each: the DPU works on a slice while the SHAVEs have the one before
		let n = rv as usize;
		let qs = g.split(&qg, 2, n);
		let z = g.sf(Ty::F16, 0.0);
		let ps: Vec<V> = qs
			.iter()
			.map(|q| {
				let sc = g.matmul(q, &k, false, true);
				g.maximum(&sc, &z)
			})
			.collect();
		// (no Concat: on booleans it becomes a SHAVE op of its own)
		for (i, p) in ps.iter().enumerate() {
			g.result(p, &format!("p{i}"));
		}
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0] == "qac" {
		// `qa` on a Convert (f16 -> u8) as the carrier
		let p = g.convert(&sc, Ty::U8);
		g.result(&p, "p");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0] == "qa" {
		// the first graph of a far block: scores -> the softmax_q8 kernel on
		// its carrier (the IR's Maximum is renamed to Greater afterwards)
		let z = g.sf(Ty::F16, 0.0);
		let p = g.maximum(&sc, &z);
		g.result(&p, "p");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0] == "nosm" {
		// timing stand-in with the far graph's interface and no Softmax
		let vb = g.parameter("vblk", Ty::F16, &[1, cap, HKV, DA]);
		let v = g.transpose(&vb, &[0, 2, 1, 3]);
		let o = g.matmul(&sc, &v, false, false);
		g.result(&o, "o");
		let n = sc.shape[3];
		let l = g.variadic_split(&sc, 3, &[n - 2, 2]).remove(1);
		g.result(&l, "lse");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0] == "pv1b" {
		// one graph: the Softmax kernel's f16 P through a static FakeQuantize
		// (the compiler's own Quantize), then the integer FC on batched weights
		let p = g.softmax(&sc, 3);
		let pq = g.fake_quantize(&p, 0.0, 1.0, 256);
		let w = g.parameter("vq", Ty::I8, &[1, HKV, DA + 1, cap]);
		let c = g.convert(&w, Ty::F16);
		let k = g.c16(&[1, 1, 1, 1], &[rv]);
		let w = g.mul(&c, &k);
		let o = g.matmul(&pq, &w, false, true);
		g.result(&o, "o");
		let n = p.shape[3];
		let l = g.variadic_split(&p, 3, &[n - 2, 2]).remove(1);
		g.result(&l, "lse");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	if a[0] == "pvk" {
		// Convert f16 -> u8 as the carrier of a Softmax kernel that stores
		// p = exp(score - max) as u8 fractions; V with 3 more rows that
		// bring the kernel's stash (max, sum) out through the FC
		// a second reader of the scores keeps the MatMul's output f16 and
		// the Convert on the SHAVEs
		let n = sc.shape[3];
		let tail = g.variadic_split(&sc, 3, &[n - 2, 2]).remove(1);
		g.result(&tail, "tail");
		// (the IR's Maximum is renamed to Greater afterwards: f16 -> boolean)
		let p = g.fake_quantize(&sc, 0.0, 1.0, 256);
		let w = g.parameter("vq", Ty::I8, &[1, HKV, DA + 3, cap]);
		let c = g.convert(&w, Ty::F16);
		let k = g.c16(&[1, 1, 1, 1], &[rv / 127.0]);
		let w = g.mul(&c, &k);
		let o = g.matmul(&p, &w, false, true);
		g.result(&o, "o");
		g.save(std::path::Path::new(&a[3]), "farx").unwrap();
		return;
	}
	let levels: Vec<V> = match a[0].as_str() {
		"pv1" => vec![g.fake_quantize(&p, 0.0, 1.0, 256)],
		"pv2" => vec![
			g.fake_quantize(&p, 0.0, t, 256),
			g.fake_quantize(&p, t, 1.0, 256),
		],
		v => panic!("variant {v}"),
	};
	let ws: Vec<V> = (0..HKV)
		.map(|h| {
			let w = g.parameter(&format!("vq{h}"), Ty::I8, &[DA, cap]);
			let c = g.convert(&w, Ty::F16);
			let sc = g.c16(&[1, 1], &[rv / 127.0]);
			g.mul(&c, &sc)
		})
		.collect();
	let mut o: Option<V> = None;
	for l in &levels {
		let hs = g.split(l, 1, HKV);
		let os: Vec<V> = hs
			.iter()
			.zip(&ws)
			.map(|(ph, w)| g.matmul(ph, w, false, true))
			.collect();
		let ol = g.concat(&os, 1);
		o = Some(match o {
			Some(x) => g.add(&x, &ol),
			None => ol,
		});
	}
	g.result(&o.unwrap(), "o");
	let n = p.shape[3];
	let l = g.variadic_split(&p, 3, &[n - 2, 2]).remove(1);
	g.result(&l, "lse");
	g.save(std::path::Path::new(&a[3]), "farx").unwrap();
}
