//! Flash-Next's delta pre graphs on the NPU, `delta_pre_m` against
//! `delta_pre_c` + the host conv (`conv_npu.glsl`'s math) on the same
//! inputs:
//!
//!   dpccheck GRAPHS_DIR WEIGHTS_DIR LAYER [N]
//!
//! Prints per output (`qn`, `kn`, `v`, `gb`, `z`, the next conv window)
//! the max abs difference and the reference's max abs.
use npu::ir::{f16_bits, f16_to_f32};
use npu::ze::Npu;
use std::collections::HashMap;

const NQKV: usize = 10240;
const S: usize = 256;

fn f16s(b: &[u8]) -> Vec<f32> {
	b.chunks_exact(2)
		.map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
		.collect()
}

fn h(v: &[f32]) -> Vec<u8> {
	v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect()
}

fn run(
	npu: &Npu,
	blob: &std::path::Path,
	w: &std::path::Path,
	l: usize,
	ins: &HashMap<&str, Vec<u8>>,
) -> Result<Vec<Vec<u8>>, String> {
	let b = std::fs::read(blob).map_err(|e| format!("{}: {e}", blob.display()))?;
	let mut g = npu.load(&b, 0)?;
	g.initialize()?;
	let args_f = blob.with_extension("args");
	let names: Vec<String> = std::fs::read_to_string(&args_f)
		.map_err(|e| format!("{}: {e}", args_f.display()))?
		.lines()
		.map(str::to_string)
		.collect();
	let mut bufs = Vec::new();
	let mut ii = 0;
	for i in 0..g.args.len() {
		let a = g.args[i].clone();
		let mut hb = npu.alloc(a.bytes)?;
		if !a.is_output {
			let name = &names[ii];
			ii += 1;
			let d = match ins.get(name.as_str()) {
				Some(d) => d.clone(),
				None => {
					let p = w.join(format!("fl{l}.{name}.in"));
					std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?
				}
			};
			if d.len() != a.bytes {
				return Err(format!("{name}: {} bytes, wants {}", d.len(), a.bytes));
			}
			hb.as_mut_slice().copy_from_slice(&d);
		}
		g.bind(i, &hb)?;
		bufs.push((a.is_output, hb));
	}
	g.run()?;
	Ok(bufs
		.into_iter()
		.filter(|b| b.0)
		.map(|b| b.1.as_slice().to_vec())
		.collect())
}

fn cmp(name: &str, a: &[f32], r: &[f32]) {
	let d = a
		.iter()
		.zip(r)
		.map(|(x, y)| (x - y).abs())
		.fold(0f32, f32::max);
	let m = r.iter().map(|x| x.abs()).fold(0f32, f32::max);
	let mean = a.iter().zip(r).map(|(x, y)| (x - y).abs()).sum::<f32>() / r.len() as f32;
	println!("{name:9} max diff {d:.4e}  mean diff {mean:.4e}  ref max {m:.4e}");
}

fn main() -> Result<(), String> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let (gd, wd) = (std::path::Path::new(&a[0]), std::path::Path::new(&a[1]));
	let l: usize = a[2].parse().unwrap();
	let n: usize = a.get(3).map_or(S, |v| v.parse().unwrap());
	let noise = |len: usize, seed: u64, sc: f32| -> Vec<f32> {
		npu::noise(len, seed)
			.iter()
			.map(|&b| (b as f32 - 127.5) / 127.5 * sc)
			.collect()
	};
	let mut ins: HashMap<&str, Vec<u8>> = HashMap::new();
	ins.insert("mixed", h(&noise(S * 2560, 1, 2.0)));
	ins.insert("hinj", h(&noise(S * 4, 2, 1.0)));
	let cpast = noise(3 * NQKV, 3, 0.5);
	ins.insert("cpast", h(&cpast));
	ins.insert(
		"valid",
		h(&(0..S)
			.map(|t| if t < n { 1.0 } else { 0.0 })
			.collect::<Vec<_>>()),
	);
	ins.insert(
		"keep",
		(0..3)
			.flat_map(|j| ((n + j) as i32).to_le_bytes())
			.collect(),
	);
	let _lock = npu::lock::NpuLock::acquire()?;
	let npu = Npu::open()?;
	let m = run(&npu, &gd.join("delta_pre_m.blob"), wd, l, &ins)?;
	let c = run(&npu, &gd.join("delta_pre_c.blob"), wd, l, &ins)?;
	// conv on the host from c's qkv
	let rd = |nm: &str| f16s(&std::fs::read(wd.join(format!("fl{l}.{nm}.in"))).unwrap());
	let (cw, qs) = (rd("conv"), rd("qkv.s"));
	let qkv = f16s(&c[0]);
	// the conv's inputs: [cpast; qkv * qs], rounded to f16 as the engine's window
	let x = |r: usize, ch: usize| -> f32 {
		if r < 3 {
			cpast[r * NQKV + ch]
		} else {
			qkv[(r - 3) * NQKV + ch] * qs[ch]
		}
	};
	let (mut qn, mut kn, mut v) = (
		vec![0f32; n * 2048],
		vec![0f32; n * 2048],
		vec![0f32; n * 6144],
	);
	let mut y = vec![0f32; NQKV];
	for t in 0..n {
		for (ch, yc) in y.iter_mut().enumerate() {
			let s: f32 = (0..4).map(|k| cw[4 * ch + k] * x(t + k, ch)).sum();
			*yc = s / (1.0 + (-s).exp());
		}
		for hr in 0..32 {
			let r = &mut y[hr * 128..][..128];
			let ss: f32 = r.iter().map(|a| a * a).sum();
			let inv = 1.0 / (ss + 1e-6).sqrt();
			r.iter_mut().for_each(|a| *a *= inv);
		}
		qn[t * 2048..][..2048].copy_from_slice(&y[..2048]);
		kn[t * 2048..][..2048].copy_from_slice(&y[2048..4096]);
		v[t * 6144..][..6144].copy_from_slice(&y[4096..]);
	}
	let win: Vec<f32> = (0..3)
		.flat_map(|r| (0..NQKV).map(move |ch| (r, ch)))
		.map(|(r, ch)| x(n + r, ch))
		.collect();
	let mo: Vec<Vec<f32>> = m.iter().map(|b| f16s(b)).collect();
	cmp("qn", &qn, &mo[0][..n * 2048]);
	cmp("kn", &kn, &mo[1][..n * 2048]);
	cmp("v", &v, &mo[2][..n * 6144]);
	cmp("gb", &f16s(&c[1])[..n * 96], &mo[3][..n * 96]);
	cmp("z", &f16s(&c[2])[..n * 6144], &mo[4][..n * 6144]);
	cmp("conv_out", &win, &mo[5]);
	cmp("inj", &f16s(&c[3])[..n * 4], &mo[6][..n * 4]);
	Ok(())
}
