//! A minimal OpenVINO IR (v11) writer: enough to build microbenchmark graphs
//! (weight-compressed MatMuls, elementwise ops) in memory and hand them to the
//! driver's compiler. Shapes are static; broadcasting is numpy-style.

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
	F32,
	F16,
	I64,
	I32,
	I8,
	U8,
	I4,
	U4,
}

impl Ty {
	fn et(self) -> &'static str {
		match self {
			Ty::F32 => "f32",
			Ty::F16 => "f16",
			Ty::I64 => "i64",
			Ty::I32 => "i32",
			Ty::I8 => "i8",
			Ty::U8 => "u8",
			Ty::I4 => "i4",
			Ty::U4 => "u4",
		}
	}
	fn precision(self) -> &'static str {
		match self {
			Ty::F32 => "FP32",
			Ty::F16 => "FP16",
			Ty::I64 => "I64",
			Ty::I32 => "I32",
			Ty::I8 => "I8",
			Ty::U8 => "U8",
			Ty::I4 => "I4",
			Ty::U4 => "U4",
		}
	}
	pub fn bits(self) -> usize {
		match self {
			Ty::F32 | Ty::I32 => 32,
			Ty::F16 => 16,
			Ty::I64 => 64,
			Ty::I8 | Ty::U8 => 8,
			Ty::I4 | Ty::U4 => 4,
		}
	}
}

/// An output port of a layer.
#[derive(Clone, Debug)]
pub struct P {
	layer: usize,
	port: usize,
	pub ty: Ty,
	pub shape: Vec<usize>,
}

struct Layer {
	name: String,
	kind: &'static str,
	data: String,
	inputs: Vec<P>,
	outputs: Vec<(Ty, Vec<usize>, Option<String>)>,
}

#[derive(Default)]
pub struct Ir {
	layers: Vec<Layer>,
	pub bin: Vec<u8>,
}

fn dims(shape: &[usize]) -> String {
	shape.iter().map(|d| format!("<dim>{d}</dim>")).collect()
}

fn csv(shape: &[usize]) -> String {
	shape
		.iter()
		.map(|d| d.to_string())
		.collect::<Vec<_>>()
		.join(",")
}

fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
	let n = a.len().max(b.len());
	(0..n)
		.map(|i| {
			let x = if i + a.len() >= n {
				a[i + a.len() - n]
			} else {
				1
			};
			let y = if i + b.len() >= n {
				b[i + b.len() - n]
			} else {
				1
			};
			assert!(
				x == y || x == 1 || y == 1,
				"cannot broadcast {a:?} with {b:?}"
			);
			x.max(y)
		})
		.collect()
}

impl Ir {
	pub fn new() -> Ir {
		Ir::default()
	}

	pub fn layer_count(&self) -> usize {
		self.layers.len()
	}

	fn push(
		&mut self,
		kind: &'static str,
		data: String,
		inputs: Vec<P>,
		outputs: Vec<(Ty, Vec<usize>, Option<String>)>,
	) -> usize {
		let id = self.layers.len();
		self.layers.push(Layer {
			name: format!("{}_{id}", kind.to_lowercase()),
			kind,
			data,
			inputs,
			outputs,
		});
		id
	}

	fn out(&self, layer: usize, k: usize) -> P {
		let l = &self.layers[layer];
		P {
			layer,
			port: l.inputs.len() + k,
			ty: l.outputs[k].0,
			shape: l.outputs[k].1.clone(),
		}
	}

	fn op1(
		&mut self,
		kind: &'static str,
		data: String,
		inputs: Vec<P>,
		ty: Ty,
		shape: Vec<usize>,
	) -> P {
		let id = self.push(kind, data, inputs, vec![(ty, shape, None)]);
		self.out(id, 0)
	}

	pub fn parameter(&mut self, name: &str, ty: Ty, shape: &[usize]) -> P {
		let id = self.push(
			"Parameter",
			format!(r#"shape="{}" element_type="{}""#, csv(shape), ty.et()),
			vec![],
			vec![(ty, shape.to_vec(), Some(name.to_string()))],
		);
		self.layers[id].name = name.to_string();
		self.out(id, 0)
	}

	/// A constant whose raw bytes are `data` (packed little-endian, 4-bit
	/// types two per byte, low nibble first).
	pub fn constant(&mut self, ty: Ty, shape: &[usize], data: &[u8]) -> P {
		let n: usize = shape.iter().product();
		assert_eq!(data.len(), (n * ty.bits()).div_ceil(8), "constant size");
		while !self.bin.len().is_multiple_of(64) {
			self.bin.push(0);
		}
		let off = self.bin.len();
		self.bin.extend_from_slice(data);
		self.op1(
			"Const",
			format!(
				r#"element_type="{}" shape="{}" offset="{off}" size="{}""#,
				ty.et(),
				csv(shape),
				data.len()
			),
			vec![],
			ty,
			shape.to_vec(),
		)
	}

	pub fn const_i64(&mut self, v: &[i64]) -> P {
		let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
		self.constant(Ty::I64, &[v.len()], &b)
	}

	pub fn const_f16(&mut self, shape: &[usize], v: &[f32]) -> P {
		let b: Vec<u8> = v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect();
		self.constant(Ty::F16, shape, &b)
	}

	pub fn convert(&mut self, x: &P, ty: Ty) -> P {
		let s = x.shape.clone();
		self.op1(
			"Convert",
			format!(r#"destination_type="{}""#, ty.et()),
			vec![x.clone()],
			ty,
			s,
		)
	}

	fn binary(&mut self, kind: &'static str, a: &P, b: &P) -> P {
		let s = broadcast(&a.shape, &b.shape);
		self.op1(
			kind,
			r#"auto_broadcast="numpy""#.into(),
			vec![a.clone(), b.clone()],
			a.ty,
			s,
		)
	}

	pub fn add(&mut self, a: &P, b: &P) -> P {
		self.binary("Add", a, b)
	}

	pub fn multiply(&mut self, a: &P, b: &P) -> P {
		self.binary("Multiply", a, b)
	}

	pub fn subtract(&mut self, a: &P, b: &P) -> P {
		self.binary("Subtract", a, b)
	}

	pub fn reshape(&mut self, x: &P, shape: &[usize]) -> P {
		assert_eq!(
			x.shape.iter().product::<usize>(),
			shape.iter().product::<usize>(),
			"reshape size"
		);
		let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
		let c = self.const_i64(&s);
		self.op1(
			"Reshape",
			r#"special_zero="false""#.into(),
			vec![x.clone(), c],
			x.ty,
			shape.to_vec(),
		)
	}

	pub fn matmul(&mut self, a: &P, b: &P, ta: bool, tb: bool) -> P {
		let (ra, ca) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
		let (rb, cb) = (b.shape[b.shape.len() - 2], b.shape[b.shape.len() - 1]);
		let (m, k1) = if ta { (ca, ra) } else { (ra, ca) };
		let (k2, n) = if tb { (cb, rb) } else { (rb, cb) };
		assert_eq!(k1, k2, "matmul inner dims {:?} x {:?}", a.shape, b.shape);
		let mut s = broadcast(&a.shape[..a.shape.len() - 2], &b.shape[..b.shape.len() - 2]);
		s.push(m);
		s.push(n);
		self.op1(
			"MatMul",
			format!(r#"transpose_a="{ta}" transpose_b="{tb}""#),
			vec![a.clone(), b.clone()],
			a.ty,
			s,
		)
	}

	/// Any unary op without attributes (`Sigmoid`, `Exp`, `Abs`, ...).
	pub fn unary(&mut self, kind: &'static str, x: &P) -> P {
		let s = x.shape.clone();
		self.op1(kind, String::new(), vec![x.clone()], x.ty, s)
	}

	/// `ReduceSum` over `axes` (no keep_dims).
	pub fn reduce_sum(&mut self, x: &P, axes: &[usize]) -> P {
		let a: Vec<i64> = axes.iter().map(|&v| v as i64).collect();
		let c = self.const_i64(&a);
		let shape: Vec<usize> = x
			.shape
			.iter()
			.enumerate()
			.filter(|(i, _)| !axes.contains(i))
			.map(|(_, &d)| d)
			.collect();
		self.op1(
			"ReduceSum",
			r#"keep_dims="false""#.into(),
			vec![x.clone(), c],
			x.ty,
			shape,
		)
	}

	/// Any `Reduce*` op over `axes`, optionally keeping the reduced dims.
	pub fn reduce(&mut self, kind: &'static str, x: &P, axes: &[usize], keep: bool) -> P {
		let a: Vec<i64> = axes.iter().map(|&v| v as i64).collect();
		let c = self.const_i64(&a);
		let shape: Vec<usize> = x
			.shape
			.iter()
			.enumerate()
			.filter_map(|(i, &d)| {
				if !axes.contains(&i) {
					Some(d)
				} else if keep {
					Some(1)
				} else {
					None
				}
			})
			.collect();
		self.op1(
			kind,
			format!(r#"keep_dims="{keep}""#),
			vec![x.clone(), c],
			x.ty,
			shape,
		)
	}

	pub fn divide(&mut self, a: &P, b: &P) -> P {
		self.binary("Divide", a, b)
	}

	/// Dynamic per-row (per-token) symmetric int8 quantization of `x [.., k]`:
	/// `(FakeQuantize(x / s, +-127, 255 levels), s)` with `s = max|x| / 127`
	/// per row, so a MatMul on the first output and a multiply by `s` after
	/// it is an int8 x int8 product on the DPU.
	pub fn dyn_quant(&mut self, x: &P) -> (P, P) {
		let r = x.shape.len() - 1;
		let a = self.unary("Abs", x);
		let m = self.reduce("ReduceMax", &a, &[r], true);
		let eps = self.const_f16(&[1], &[1e-4]);
		let m = self.binary("Maximum", &m, &eps);
		let c = self.const_f16(&[1], &[1.0 / 127.0]);
		let s = self.multiply(&m, &c);
		let xs = self.divide(x, &s);
		(self.fake_quantize(&xs, -127.0, 127.0, 255), s)
	}

	/// Group-batched int4 linear: `x [1, k] -> [1, n]` as G = k / g batched
	/// MatMuls over unscaled int4 weights (one common constant scale `c`, so
	/// every sub-conv shares one weight table), then the per-(row, group)
	/// scales on the `[G, 1, n]` partials and a sum over G.
	pub fn linear_i4_gbatch(
		&mut self,
		x: &P,
		n: usize,
		k: usize,
		g: usize,
		q_gnk: &[u8],
		scale_gn: &[f32],
		c: f32,
	) -> P {
		let gs = k / g;
		let xr = self.reshape(x, &[gs, 1, g]);
		let cc = self.const_f16(&[1, 1, 1], &[c]);
		let xr = self.multiply(&xr, &cc);
		let w = self.constant(Ty::I4, &[gs, n, g], q_gnk);
		let w = self.convert(&w, Ty::F16);
		let p = self.matmul(&xr, &w, false, true);
		let sc: Vec<f32> = scale_gn.iter().map(|v| v / c).collect();
		let s = self.const_f16(&[gs, 1, n], &sc);
		let p = self.multiply(&p, &s);
		let y = self.reduce_sum(&p, &[0]);
		self.reshape(&y, &[1, n])
	}

	/// `FakeQuantize` with scalar ranges (`levels` steps over `[lo, hi]`).
	pub fn fake_quantize(&mut self, x: &P, lo: f32, hi: f32, levels: u32) -> P {
		let c = |ir: &mut Ir, v: f32| ir.const_f16(&[1, 1], &[v]);
		let (a, b, c2, d) = (c(self, lo), c(self, hi), c(self, lo), c(self, hi));
		let s = x.shape.clone();
		self.op1(
			"FakeQuantize",
			format!(r#"levels="{levels}" auto_broadcast="numpy""#),
			vec![x.clone(), a, b, c2, d],
			x.ty,
			s,
		)
	}

	/// Symmetric int4, group-quantized weight `[n, k]` (groups of `g` along
	/// `k`) as NNCF exports it: i4 `[n, k/g, g]` -> f16, x scale `[n, k/g, 1]`,
	/// reshape to `[n, k]`. `q` holds `n * k / 2` packed bytes.
	pub fn weight_i4(&mut self, n: usize, k: usize, g: usize, q: &[u8], scale: &[f32]) -> P {
		let w = self.constant(Ty::I4, &[n, k / g, g], q);
		let w = self.convert(&w, Ty::F16);
		let s = self.const_f16(&[n, k / g, 1], scale);
		let w = self.multiply(&w, &s);
		self.reshape(&w, &[n, k])
	}

	/// Int8 weight `[n, k]` with one f16 scale per output row.
	pub fn weight_i8(&mut self, n: usize, k: usize, q: &[u8], scale: &[f32]) -> P {
		let w = self.constant(Ty::I8, &[n, k], q);
		let w = self.convert(&w, Ty::F16);
		let s = self.const_f16(&[n, 1], scale);
		self.multiply(&w, &s)
	}

	/// Plain f16 weight `[n, k]` (raw f16 bits in `q`).
	pub fn weight_f16(&mut self, n: usize, k: usize, q: &[u8]) -> P {
		self.constant(Ty::F16, &[n, k], q)
	}

	pub fn result(&mut self, x: &P, name: &str) {
		let id = self.push("Result", String::new(), vec![x.clone()], vec![]);
		self.layers[id].name = name.to_string();
		// Name the tensor feeding the result, so the output argument has it.
		let l = &mut self.layers[x.layer];
		let k = x.port - l.inputs.len();
		l.outputs[k].2 = Some(name.to_string());
	}

	pub fn xml(&self, name: &str) -> String {
		let mut s = String::new();
		let _ = writeln!(s, r#"<?xml version="1.0"?>"#);
		let _ = writeln!(s, r#"<net name="{name}" version="11">"#);
		s.push_str("\t<layers>\n");
		for (id, l) in self.layers.iter().enumerate() {
			let _ = write!(
				s,
				r#"		<layer id="{id}" name="{}" type="{}" version="opset1">"#,
				l.name, l.kind
			);
			if !l.data.is_empty() {
				let _ = write!(s, "<data {}/>", l.data);
			}
			if !l.inputs.is_empty() {
				s.push_str("<input>");
				for (i, p) in l.inputs.iter().enumerate() {
					let _ = write!(
						s,
						r#"<port id="{i}" precision="{}">{}</port>"#,
						p.ty.precision(),
						dims(&p.shape)
					);
				}
				s.push_str("</input>");
			}
			if !l.outputs.is_empty() {
				s.push_str("<output>");
				for (k, (ty, shape, nm)) in l.outputs.iter().enumerate() {
					let names = nm
						.as_ref()
						.map(|n| format!(r#" names="{n}""#))
						.unwrap_or_default();
					let _ = write!(
						s,
						r#"<port id="{}" precision="{}"{names}>{}</port>"#,
						l.inputs.len() + k,
						ty.precision(),
						dims(shape)
					);
				}
				s.push_str("</output>");
			}
			s.push_str("</layer>\n");
		}
		s.push_str("\t</layers>\n\t<edges>\n");
		for (id, l) in self.layers.iter().enumerate() {
			for (i, p) in l.inputs.iter().enumerate() {
				let _ = writeln!(
					s,
					r#"		<edge from-layer="{}" from-port="{}" to-layer="{id}" to-port="{i}"/>"#,
					p.layer, p.port
				);
			}
		}
		s.push_str("\t</edges>\n</net>\n");
		s
	}
}

/// f32 -> IEEE half (round to nearest even).
pub fn f16_bits(x: f32) -> u16 {
	let b = x.to_bits();
	let sign = ((b >> 16) & 0x8000) as u16;
	let e = ((b >> 23) & 0xff) as i32;
	let m = b & 0x7f_ffff;
	if e == 0xff {
		return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
	}
	let e2 = e - 127 + 15;
	if e2 >= 0x1f {
		return sign | 0x7c00;
	}
	if e2 <= 0 {
		if e2 < -10 {
			return sign;
		}
		let m = m | 0x80_0000;
		let shift = (14 - e2) as u32;
		let half = 1u32 << (shift - 1);
		let r = m & ((1 << shift) - 1);
		let mut v = m >> shift;
		if r > half || (r == half && v & 1 == 1) {
			v += 1;
		}
		return sign | v as u16;
	}
	let mut v = ((e2 as u32) << 10) | (m >> 13);
	let r = m & 0x1fff;
	if r > 0x1000 || (r == 0x1000 && v & 1 == 1) {
		v += 1;
	}
	sign | v as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
	let sign = ((h & 0x8000) as u32) << 16;
	let e = ((h >> 10) & 0x1f) as u32;
	let m = (h & 0x3ff) as u32;
	let bits = if e == 0 {
		if m == 0 {
			sign
		} else {
			let mut e = 127 - 15 + 1;
			let mut m = m;
			while m & 0x400 == 0 {
				m <<= 1;
				e -= 1;
			}
			sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
		}
	} else if e == 0x1f {
		sign | 0x7f80_0000 | (m << 13)
	} else {
		sign | ((e + 127 - 15) << 23) | (m << 13)
	};
	f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn f16_roundtrip() {
		for &x in &[0.0f32, 1.0, -2.5, 65504.0, 6.1e-5, 3.0e-7, 1e9] {
			let y = f16_to_f32(f16_bits(x));
			let rel = if x == 0.0 {
				y.abs()
			} else {
				((y - x) / x).abs()
			};
			assert!(
				rel < 1e-3 || x == 1e9 && y.is_infinite() || x == 3.0e-7 && (y - x).abs() < 6e-8,
				"{x} -> {y}"
			);
		}
	}

	#[test]
	fn matmul_shapes_and_xml() {
		let mut ir = Ir::new();
		let x = ir.parameter("x", Ty::F16, &[1, 256]);
		let q = vec![0u8; 64 * 256 / 2];
		let w = ir.weight_i4(64, 256, 128, &q, &[0.01; 128]);
		let y = ir.matmul(&x, &w, false, true);
		assert_eq!(y.shape, vec![1, 64]);
		ir.result(&y, "y");
		let s = ir.xml("t");
		assert!(s.contains(r#"element_type="i4" shape="64,2,128""#));
		assert!(s.contains(r#"names="y""#));
	}
}

/// `dst[i] = f16(src[i]) * mul` for little-endian f16 bytes: F16C where the
/// CPU has it (8 at a time), else [`f16_to_f32`]. Exact either way.
pub fn f16s_to_f32(src: &[u8], dst: &mut [f32], mul: f32) {
	let n = dst.len();
	assert!(src.len() >= n * 2);
	let mut i = 0;
	#[cfg(target_arch = "x86_64")]
	if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
		// SAFETY: features checked; bounds asserted above
		unsafe { i = f16s_to_f32_f16c(src, dst, mul) };
	}
	for j in i..n {
		dst[j] = f16_to_f32(u16::from_le_bytes([src[j * 2], src[j * 2 + 1]])) * mul;
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16s_to_f32_f16c(src: &[u8], dst: &mut [f32], mul: f32) -> usize {
	use std::arch::x86_64::*;
	let (n, m) = (dst.len() / 8 * 8, _mm256_set1_ps(mul));
	for i in (0..n).step_by(8) {
		let h = _mm_loadu_si128(src.as_ptr().add(i * 2) as *const __m128i);
		_mm256_storeu_ps(
			dst.as_mut_ptr().add(i),
			_mm256_mul_ps(_mm256_cvtph_ps(h), m),
		);
	}
	n
}

/// `dst[i] = f16(src[i] * mul)` as little-endian bytes, rounded to nearest
/// even: F16C where the CPU has it, else [`f16_bits`] (the same bits for
/// every non-NaN value).
pub fn f32s_to_f16(src: &[f32], dst: &mut [u8], mul: f32) {
	let n = src.len();
	assert!(dst.len() >= n * 2);
	let mut i = 0;
	#[cfg(target_arch = "x86_64")]
	if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
		// SAFETY: features checked; bounds asserted above
		unsafe { i = f32s_to_f16_f16c(src, dst, mul) };
	}
	for j in i..n {
		dst[j * 2..j * 2 + 2].copy_from_slice(&f16_bits(src[j] * mul).to_le_bytes());
	}
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32s_to_f16_f16c(src: &[f32], dst: &mut [u8], mul: f32) -> usize {
	use std::arch::x86_64::*;
	let (n, m) = (src.len() / 8 * 8, _mm256_set1_ps(mul));
	for i in (0..n).step_by(8) {
		let v = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), m);
		_mm_storeu_si128(
			dst.as_mut_ptr().add(i * 2) as *mut __m128i,
			_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT),
		);
	}
	n
}

#[cfg(test)]
mod cvt_tests {
	use super::*;

	/// The F16C paths give the software conversions' bits, also for
	/// subnormals, overflow and ties.
	#[test]
	fn vector_conversions_match_the_scalar_ones() {
		let mut xs: Vec<f32> = crate::noise(4 * 4096, 7)
			.chunks_exact(4)
			.map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
			.filter(|x| !x.is_nan())
			.collect();
		xs.extend([
			0.0,
			-0.0,
			65504.0,
			65520.0,
			1e-8,
			6e-8,
			2.98e-8,
			1.0 + 1.0 / 2048.0,
			-3.0e-5,
		]);
		xs.truncate(xs.len() / 8 * 8);
		for mul in [1.0f32, 256.0, 1.0 / 16.0] {
			let mut h = vec![0u8; xs.len() * 2];
			f32s_to_f16(&xs, &mut h, mul);
			for (i, &x) in xs.iter().enumerate() {
				assert_eq!(
					u16::from_le_bytes([h[i * 2], h[i * 2 + 1]]),
					f16_bits(x * mul),
					"{x} x {mul}"
				);
			}
			let mut back = vec![0f32; xs.len()];
			f16s_to_f32(&h, &mut back, mul);
			for (i, &y) in back.iter().enumerate() {
				let r = f16_to_f32(u16::from_le_bytes([h[i * 2], h[i * 2 + 1]])) * mul;
				assert!(
					y.to_bits() == r.to_bits() || (y.is_nan() && r.is_nan()),
					"{i}: {y} vs {r}"
				);
			}
		}
	}
}
