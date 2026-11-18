//! An OpenVINO IR (v11) graph builder with static shape inference: the ops
//! the NPU graph sets are made of, written as `ov.save_model` writes them
//! (opset versions, `<data>` attributes, port precisions), so the driver's
//! and our build of the NPU compiler read them as they read the IRs the
//! Python builders produced.
//!
//! A [`V`] is an output of an op. Parameters and results keep the order they
//! were declared in (the compiled graph's argument order).

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
}

impl Ty {
	pub fn et(self) -> &'static str {
		match self {
			Ty::F32 => "f32",
			Ty::F16 => "f16",
			Ty::I64 => "i64",
			Ty::I32 => "i32",
			Ty::I8 => "i8",
			Ty::U8 => "u8",
			Ty::I4 => "i4",
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
		}
	}
	/// Bytes of `n` elements (4-bit types pack two per byte).
	pub fn bytes(self, n: usize) -> usize {
		match self {
			Ty::F32 | Ty::I32 => 4 * n,
			Ty::F16 => 2 * n,
			Ty::I64 => 8 * n,
			Ty::I8 | Ty::U8 => n,
			Ty::I4 => n.div_ceil(2),
		}
	}
}

/// An output of an op.
#[derive(Clone, Debug)]
pub struct V {
	op: usize,
	out: usize,
	pub ty: Ty,
	pub shape: Vec<usize>,
}

impl V {
	pub fn rank(&self) -> usize {
		self.shape.len()
	}
	pub fn size(&self) -> usize {
		self.shape.iter().product()
	}
}

struct Op {
	kind: &'static str,
	ver: &'static str,
	name: String,
	data: String,
	inputs: Vec<V>,
	outputs: Vec<(Ty, Vec<usize>, Option<String>)>,
}

#[derive(Default)]
pub struct Graph {
	ops: Vec<Op>,
	params: Vec<usize>,
	results: Vec<usize>,
	pub bin: Vec<u8>,
	/// A prefix for the names of the ops made from now on (profiling tags).
	pub region: String,
}

/// f64 -> IEEE half, round to nearest even (as `np.float16(x)`).
pub fn f16_bits(x: f64) -> u16 {
	let b = x.to_bits();
	let sign = ((b >> 48) & 0x8000) as u16;
	let e = ((b >> 52) & 0x7ff) as i32;
	let m = b & 0x000f_ffff_ffff_ffff;
	if e == 0x7ff {
		return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
	}
	if e == 0 && m == 0 {
		return sign;
	}
	// value = 1.m * 2^(e - 1023) (subnormal doubles are far below half's range)
	let exp = e - 1023;
	if exp > 15 {
		return sign | 0x7c00;
	}
	let full = m | (1u64 << 52);
	// half: normal for exp >= -14 with 10 mantissa bits, else subnormal
	let shift = if exp >= -14 { 42 } else { 42 + (-14 - exp) };
	if shift > 63 {
		return sign;
	}
	let mut q = full >> shift;
	let rem = full & ((1u64 << shift) - 1);
	let half = 1u64 << (shift - 1);
	if rem > half || (rem == half && q & 1 == 1) {
		q += 1;
	}
	if exp >= -14 {
		// q in [2^10, 2^11]: a carry to 2^11 moves to the next exponent
		let mut he = (exp + 15) as u64;
		if q == 1 << 11 {
			q >>= 1;
			he += 1;
		}
		if he >= 31 {
			return sign | 0x7c00;
		}
		sign | ((he as u16) << 10) | ((q & 0x3ff) as u16)
	} else {
		// subnormal (q may round up to 2^10 = the smallest normal: same bits)
		sign | q as u16
	}
}

pub fn f16_to_f32(h: u16) -> f32 {
	quant::fp16_to_f32(h)
}

fn csv(shape: &[usize]) -> String {
	shape
		.iter()
		.map(|d| d.to_string())
		.collect::<Vec<_>>()
		.join(",")
}

pub fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
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

fn axis(a: i64, rank: usize) -> usize {
	let r = if a < 0 { a + rank as i64 } else { a };
	assert!(
		r >= 0 && (r as usize) < rank.max(1),
		"axis {a} of rank {rank}"
	);
	r as usize
}

impl Graph {
	pub fn new() -> Graph {
		Graph::default()
	}

	pub fn op_count(&self) -> usize {
		self.ops.len()
	}

	fn push(
		&mut self,
		kind: &'static str,
		ver: &'static str,
		data: String,
		inputs: Vec<V>,
		outputs: Vec<(Ty, Vec<usize>)>,
	) -> usize {
		let id = self.ops.len();
		let name = format!("{}{kind}_{id}", self.region);
		self.ops.push(Op {
			kind,
			ver,
			name,
			data,
			inputs,
			outputs: outputs.into_iter().map(|(t, s)| (t, s, None)).collect(),
		});
		id
	}

	fn out(&self, op: usize, k: usize) -> V {
		let o = &self.ops[op].outputs[k];
		V {
			op,
			out: k,
			ty: o.0,
			shape: o.1.clone(),
		}
	}

	fn op1(
		&mut self,
		kind: &'static str,
		ver: &'static str,
		data: String,
		inputs: Vec<V>,
		ty: Ty,
		shape: Vec<usize>,
	) -> V {
		let id = self.push(kind, ver, data, inputs, vec![(ty, shape)]);
		self.out(id, 0)
	}

	pub fn parameter(&mut self, name: &str, ty: Ty, shape: &[usize]) -> V {
		let id = self.push(
			"Parameter",
			"opset1",
			format!(r#"shape="{}" element_type="{}""#, csv(shape), ty.et()),
			vec![],
			vec![(ty, shape.to_vec())],
		);
		self.ops[id].name = name.to_string();
		self.ops[id].outputs[0].2 = Some(name.to_string());
		self.params.push(id);
		self.out(id, 0)
	}

	/// Parameters declared so far, as (name, type, shape), in order.
	pub fn inputs(&self) -> Vec<(String, Ty, Vec<usize>)> {
		self.params
			.iter()
			.map(|&i| {
				(
					self.ops[i].name.clone(),
					self.ops[i].outputs[0].0,
					self.ops[i].outputs[0].1.clone(),
				)
			})
			.collect()
	}

	/// The names of the parameters a result depends on.
	pub fn used_inputs(&self) -> Vec<String> {
		let live = self.live();
		self.params
			.iter()
			.filter(|&&i| live[i])
			.map(|&i| self.ops[i].name.clone())
			.collect()
	}

	fn live(&self) -> Vec<bool> {
		let mut live = vec![false; self.ops.len()];
		let mut stack: Vec<usize> = self.results.clone();
		while let Some(i) = stack.pop() {
			if !std::mem::replace(&mut live[i], true) {
				stack.extend(self.ops[i].inputs.iter().map(|v| v.op));
			}
		}
		live
	}

	/// Result names, in order.
	pub fn outputs(&self) -> Vec<String> {
		self.results
			.iter()
			.map(|&i| self.ops[i].name.clone())
			.collect()
	}

	/// Move parameter `name` to position `at` of the argument order.
	pub fn move_param(&mut self, name: &str, at: usize) {
		let i = self
			.params
			.iter()
			.position(|&p| self.ops[p].name == name)
			.expect("move_param: no such parameter");
		let p = self.params.remove(i);
		self.params.insert(at, p);
	}

	pub fn constant(&mut self, ty: Ty, shape: &[usize], data: &[u8]) -> V {
		let n: usize = shape.iter().product();
		assert_eq!(data.len(), ty.bytes(n), "constant {ty:?} {shape:?}");
		let off = self.bin.len();
		self.bin.extend_from_slice(data);
		// (a Constant's shape is written with ", ", a Parameter's with ",":
		// as `ov.save_model` does, and as the engine looks the head's norm up)
		let sh = shape
			.iter()
			.map(|d| d.to_string())
			.collect::<Vec<_>>()
			.join(", ");
		self.op1(
			"Const",
			"opset1",
			format!(
				r#"element_type="{}" shape="{sh}" offset="{off}" size="{}""#,
				ty.et(),
				data.len()
			),
			vec![],
			ty,
			shape.to_vec(),
		)
	}

	pub fn c16(&mut self, shape: &[usize], v: &[f64]) -> V {
		let b: Vec<u8> = v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect();
		self.constant(Ty::F16, shape, &b)
	}

	/// A scalar f16 constant.
	pub fn s16(&mut self, v: f64) -> V {
		self.c16(&[], &[v])
	}

	pub fn c32(&mut self, shape: &[usize], v: &[f32]) -> V {
		let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
		self.constant(Ty::F32, shape, &b)
	}

	pub fn s32(&mut self, v: f32) -> V {
		self.c32(&[], &[v])
	}

	/// A float constant of type `ty` (f16 or f32).
	pub fn cf(&mut self, ty: Ty, shape: &[usize], v: &[f64]) -> V {
		match ty {
			Ty::F16 => self.c16(shape, v),
			Ty::F32 => {
				let w: Vec<f32> = v.iter().map(|&x| x as f32).collect();
				self.c32(shape, &w)
			}
			_ => panic!("cf: {ty:?}"),
		}
	}

	pub fn sf(&mut self, ty: Ty, v: f64) -> V {
		self.cf(ty, &[], &[v])
	}

	/// A 1-D i64 constant.
	pub fn ci64(&mut self, v: &[i64]) -> V {
		let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
		self.constant(Ty::I64, &[v.len()], &b)
	}

	/// A scalar i64 constant.
	pub fn si64(&mut self, v: i64) -> V {
		self.constant(Ty::I64, &[], &v.to_le_bytes())
	}

	pub fn ci32(&mut self, v: &[i32]) -> V {
		let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
		self.constant(Ty::I32, &[v.len()], &b)
	}

	pub fn convert(&mut self, x: &V, ty: Ty) -> V {
		self.op1(
			"Convert",
			"opset1",
			format!(r#"destination_type="{}""#, ty.et()),
			vec![x.clone()],
			ty,
			x.shape.clone(),
		)
	}

	/// `x` as `ty`: a Convert unless it has that type already.
	pub fn to(&mut self, x: &V, ty: Ty) -> V {
		if x.ty == ty {
			x.clone()
		} else {
			self.convert(x, ty)
		}
	}

	fn binary(&mut self, kind: &'static str, a: &V, b: &V) -> V {
		assert_eq!(
			a.ty, b.ty,
			"{kind}: {:?} {:?} with {:?} {:?}",
			a.ty, a.shape, b.ty, b.shape
		);
		let shape = broadcast(&a.shape, &b.shape);
		let data = if kind == "Divide" {
			r#"auto_broadcast="numpy" m_pythondiv="true""#
		} else {
			r#"auto_broadcast="numpy""#
		};
		self.op1(
			kind,
			"opset1",
			data.to_string(),
			vec![a.clone(), b.clone()],
			a.ty,
			shape,
		)
	}

	pub fn add(&mut self, a: &V, b: &V) -> V {
		self.binary("Add", a, b)
	}
	pub fn sub(&mut self, a: &V, b: &V) -> V {
		self.binary("Subtract", a, b)
	}
	pub fn mul(&mut self, a: &V, b: &V) -> V {
		self.binary("Multiply", a, b)
	}
	pub fn div(&mut self, a: &V, b: &V) -> V {
		self.binary("Divide", a, b)
	}
	pub fn maximum(&mut self, a: &V, b: &V) -> V {
		self.binary("Maximum", a, b)
	}
	pub fn minimum(&mut self, a: &V, b: &V) -> V {
		self.binary("Minimum", a, b)
	}
	pub fn power(&mut self, a: &V, b: &V) -> V {
		self.binary("Power", a, b)
	}

	pub fn matmul(&mut self, a: &V, b: &V, ta: bool, tb: bool) -> V {
		assert_eq!(
			a.ty, b.ty,
			"MatMul: {:?} {:?} with {:?} {:?}",
			a.ty, a.shape, b.ty, b.shape
		);
		assert!(
			a.rank() >= 2 && b.rank() >= 2,
			"MatMul: {:?} x {:?}",
			a.shape,
			b.shape
		);
		let (mut sa, mut sb) = (a.shape.clone(), b.shape.clone());
		let (ra, rb) = (sa.len(), sb.len());
		if ta {
			sa.swap(ra - 1, ra - 2);
		}
		if tb {
			sb.swap(rb - 1, rb - 2);
		}
		assert_eq!(
			sa[ra - 1],
			sb[rb - 2],
			"MatMul: {:?} x {:?} (ta {ta}, tb {tb})",
			a.shape,
			b.shape
		);
		let mut shape = broadcast(&sa[..ra - 2], &sb[..rb - 2]);
		shape.push(sa[ra - 2]);
		shape.push(sb[rb - 1]);
		self.op1(
			"MatMul",
			"opset1",
			format!(r#"transpose_a="{ta}" transpose_b="{tb}""#),
			vec![a.clone(), b.clone()],
			a.ty,
			shape,
		)
	}

	/// Reshape to `shape` (one entry may be -1), `special_zero = false`.
	pub fn reshape(&mut self, x: &V, shape: &[i64]) -> V {
		let known: usize = shape
			.iter()
			.filter(|&&d| d >= 0)
			.map(|&d| d as usize)
			.product();
		let out: Vec<usize> = shape
			.iter()
			.map(|&d| if d < 0 { x.size() / known } else { d as usize })
			.collect();
		assert_eq!(
			out.iter().product::<usize>(),
			x.size(),
			"Reshape {:?} -> {shape:?}",
			x.shape
		);
		let c = self.ci64(shape);
		self.op1(
			"Reshape",
			"opset1",
			r#"special_zero="false""#.to_string(),
			vec![x.clone(), c],
			x.ty,
			out,
		)
	}

	pub fn transpose(&mut self, x: &V, perm: &[i64]) -> V {
		assert_eq!(perm.len(), x.rank());
		let out: Vec<usize> = perm.iter().map(|&p| x.shape[p as usize]).collect();
		let c = self.ci64(perm);
		self.op1(
			"Transpose",
			"opset1",
			String::new(),
			vec![x.clone(), c],
			x.ty,
			out,
		)
	}

	pub fn concat(&mut self, xs: &[V], ax: i64) -> V {
		let r = axis(ax, xs[0].rank());
		let mut out = xs[0].shape.clone();
		out[r] = 0;
		for x in xs {
			assert_eq!(x.ty, xs[0].ty, "Concat types");
			assert_eq!(x.rank(), xs[0].rank(), "Concat ranks");
			for (i, (&d, &e)) in x.shape.iter().zip(&xs[0].shape).enumerate() {
				assert!(
					i == r || d == e,
					"Concat {:?} with {:?} on axis {ax}",
					x.shape,
					xs[0].shape
				);
			}
			out[r] += x.shape[r];
		}
		self.op1(
			"Concat",
			"opset1",
			format!(r#"axis="{ax}""#),
			xs.to_vec(),
			xs[0].ty,
			out,
		)
	}

	pub fn variadic_split(&mut self, x: &V, ax: i64, sizes: &[usize]) -> Vec<V> {
		let r = axis(ax, x.rank());
		assert_eq!(
			sizes.iter().sum::<usize>(),
			x.shape[r],
			"VariadicSplit {:?} axis {ax} into {sizes:?}",
			x.shape
		);
		let a = self.si64(ax);
		let s: Vec<i64> = sizes.iter().map(|&v| v as i64).collect();
		let s = self.ci64(&s);
		let outs = sizes
			.iter()
			.map(|&n| {
				let mut sh = x.shape.clone();
				sh[r] = n;
				(x.ty, sh)
			})
			.collect();
		let id = self.push(
			"VariadicSplit",
			"opset1",
			String::new(),
			vec![x.clone(), a, s],
			outs,
		);
		(0..sizes.len()).map(|k| self.out(id, k)).collect()
	}

	pub fn split(&mut self, x: &V, ax: i64, n: usize) -> Vec<V> {
		let r = axis(ax, x.rank());
		assert_eq!(x.shape[r] % n, 0);
		let a = self.si64(ax);
		let mut sh = x.shape.clone();
		sh[r] /= n;
		let id = self.push(
			"Split",
			"opset1",
			format!(r#"num_splits="{n}""#),
			vec![x.clone(), a],
			vec![(x.ty, sh); n],
		);
		(0..n).map(|k| self.out(id, k)).collect()
	}

	/// `x[start:stop:step]` along `ax`.
	pub fn slice(&mut self, x: &V, start: usize, stop: usize, step: usize, ax: usize) -> V {
		assert!(
			start < stop && stop <= x.shape[ax],
			"Slice {start}..{stop} of {:?} axis {ax}",
			x.shape
		);
		let mut sh = x.shape.clone();
		sh[ax] = (stop - start).div_ceil(step);
		let ins = vec![
			x.clone(),
			self.ci64(&[start as i64]),
			self.ci64(&[stop as i64]),
			self.ci64(&[step as i64]),
			self.ci64(&[ax as i64]),
		];
		self.op1("Slice", "opset8", String::new(), ins, x.ty, sh)
	}

	pub fn squeeze(&mut self, x: &V, ax: usize) -> V {
		assert_eq!(x.shape[ax], 1);
		let mut sh = x.shape.clone();
		sh.remove(ax);
		let c = self.ci64(&[ax as i64]);
		self.op1(
			"Squeeze",
			"opset1",
			String::new(),
			vec![x.clone(), c],
			x.ty,
			sh,
		)
	}

	/// Insert an axis at `ax` (negative: from the end of the result).
	pub fn unsqueeze(&mut self, x: &V, ax: i64) -> V {
		let r = axis(ax, x.rank() + 1);
		let mut sh = x.shape.clone();
		sh.insert(r, 1);
		let c = self.si64(ax);
		self.op1(
			"Unsqueeze",
			"opset1",
			String::new(),
			vec![x.clone(), c],
			x.ty,
			sh,
		)
	}

	pub fn broadcast_to(&mut self, x: &V, shape: &[usize]) -> V {
		let out = broadcast(&x.shape, shape);
		assert_eq!(out, shape);
		let s: Vec<i64> = shape.iter().map(|&v| v as i64).collect();
		let c = self.ci64(&s);
		self.op1(
			"Broadcast",
			"opset3",
			r#"mode="numpy""#.to_string(),
			vec![x.clone(), c],
			x.ty,
			out,
		)
	}

	/// FakeQuantize with the same input and output range `[lo, hi]`.
	pub fn fake_quantize(&mut self, x: &V, lo: f64, hi: f64, levels: u32) -> V {
		let ins = vec![
			x.clone(),
			self.sf(x.ty, lo),
			self.sf(x.ty, hi),
			self.sf(x.ty, lo),
			self.sf(x.ty, hi),
		];
		self.op1(
			"FakeQuantize",
			"opset1",
			format!(r#"levels="{levels}" auto_broadcast="numpy""#),
			ins,
			x.ty,
			x.shape.clone(),
		)
	}

	/// FakeQuantize with the given bounds as its input and output range.
	pub fn fq(&mut self, x: &V, lo: &V, hi: &V, levels: u32) -> V {
		let ins = vec![x.clone(), lo.clone(), hi.clone(), lo.clone(), hi.clone()];
		self.op1(
			"FakeQuantize",
			"opset1",
			format!(r#"levels="{levels}" auto_broadcast="numpy""#),
			ins,
			x.ty,
			x.shape.clone(),
		)
	}

	fn reduce(&mut self, kind: &'static str, x: &V, ax: i64) -> V {
		let r = axis(ax, x.rank());
		let mut sh = x.shape.clone();
		sh[r] = 1;
		let c = self.ci64(&[ax]);
		self.op1(
			kind,
			"opset1",
			r#"keep_dims="true""#.to_string(),
			vec![x.clone(), c],
			x.ty,
			sh,
		)
	}

	/// Reductions over one axis, `keep_dims = true`.
	pub fn reduce_mean(&mut self, x: &V, ax: i64) -> V {
		self.reduce("ReduceMean", x, ax)
	}
	pub fn reduce_max(&mut self, x: &V, ax: i64) -> V {
		self.reduce("ReduceMax", x, ax)
	}
	pub fn reduce_sum(&mut self, x: &V, ax: i64) -> V {
		self.reduce("ReduceSum", x, ax)
	}

	fn unary(&mut self, kind: &'static str, ver: &'static str, x: &V) -> V {
		self.op1(
			kind,
			ver,
			String::new(),
			vec![x.clone()],
			x.ty,
			x.shape.clone(),
		)
	}

	pub fn sqrt(&mut self, x: &V) -> V {
		self.unary("Sqrt", "opset1", x)
	}
	pub fn exp(&mut self, x: &V) -> V {
		self.unary("Exp", "opset1", x)
	}
	pub fn abs(&mut self, x: &V) -> V {
		self.unary("Abs", "opset1", x)
	}
	pub fn negative(&mut self, x: &V) -> V {
		self.unary("Negative", "opset1", x)
	}
	pub fn sigmoid(&mut self, x: &V) -> V {
		self.unary("Sigmoid", "opset1", x)
	}
	pub fn mish(&mut self, x: &V) -> V {
		self.unary("Mish", "opset4", x)
	}
	pub fn cos(&mut self, x: &V) -> V {
		self.unary("Cos", "opset1", x)
	}
	pub fn sin(&mut self, x: &V) -> V {
		self.unary("Sin", "opset1", x)
	}
	pub fn softplus(&mut self, x: &V) -> V {
		self.unary("SoftPlus", "opset4", x)
	}

	/// `x * sigmoid(beta x)` with an explicit `beta = 1` input (as the lean
	/// builders write it).
	pub fn swish1(&mut self, x: &V) -> V {
		let b = self.sf(x.ty, 1.0);
		self.op1(
			"Swish",
			"opset4",
			String::new(),
			vec![x.clone(), b],
			x.ty,
			x.shape.clone(),
		)
	}

	/// `x * sigmoid(x)` (one input).
	pub fn swish(&mut self, x: &V) -> V {
		self.unary("Swish", "opset4", x)
	}

	/// GELU in its tanh form (`gelu_pytorch_tanh`).
	pub fn gelu_tanh(&mut self, x: &V) -> V {
		self.op1(
			"Gelu",
			"opset7",
			r#"approximation_mode="TANH""#.to_string(),
			vec![x.clone()],
			x.ty,
			x.shape.clone(),
		)
	}

	/// GELU with the exact `erf`.
	pub fn gelu_erf(&mut self, x: &V) -> V {
		self.op1(
			"Gelu",
			"opset7",
			r#"approximation_mode="ERF""#.to_string(),
			vec![x.clone()],
			x.ty,
			x.shape.clone(),
		)
	}

	/// `(x - mean) / sqrt(var + eps)` over the last axis (a LayerNorm
	/// without its weight and bias).
	pub fn mvn(&mut self, x: &V, eps: f64) -> V {
		let ax = self.ci64(&[-1]);
		self.op1(
			"MVN",
			"opset6",
			format!(r#"eps="{eps:e}" normalize_variance="true" eps_mode="INSIDE_SQRT""#),
			vec![x.clone(), ax],
			x.ty,
			x.shape.clone(),
		)
	}

	pub fn softmax(&mut self, x: &V, ax: usize) -> V {
		self.op1(
			"SoftMax",
			"opset8",
			format!(r#"axis="{ax}""#),
			vec![x.clone()],
			x.ty,
			x.shape.clone(),
		)
	}

	pub fn log_softmax(&mut self, x: &V, ax: usize) -> V {
		self.op1(
			"LogSoftmax",
			"opset5",
			format!(r#"axis="{ax}""#),
			vec![x.clone()],
			x.ty,
			x.shape.clone(),
		)
	}

	/// 1-D depthwise convolution: `x` [1, C, T], `w` [C, 1, 1, K], stride 1,
	/// no padding.
	pub fn group_conv1d(&mut self, x: &V, w: &V) -> V {
		assert_eq!(x.rank(), 3);
		assert_eq!(w.rank(), 4);
		assert_eq!((w.shape[0], w.shape[1], w.shape[2]), (x.shape[1], 1, 1));
		let sh = vec![x.shape[0], x.shape[1], x.shape[2] + 1 - w.shape[3]];
		self.op1(
			"GroupConvolution",
			"opset1",
			r#"strides="1" pads_begin="0" pads_end="0" dilations="1" auto_pad="explicit""#
				.to_string(),
			vec![x.clone(), w.clone()],
			x.ty,
			sh,
		)
	}

	/// 2-D convolution: `x` [N, C, H, W], `w` [O, C, KH, KW], stride 1, no
	/// padding -> [N, O, H - KH + 1, W - KW + 1].
	pub fn conv2d(&mut self, x: &V, w: &V) -> V {
		assert_eq!(x.rank(), 4);
		assert_eq!(w.rank(), 4);
		assert_eq!(w.shape[1], x.shape[1]);
		let sh = vec![
			x.shape[0],
			w.shape[0],
			x.shape[2] + 1 - w.shape[2],
			x.shape[3] + 1 - w.shape[3],
		];
		self.op1(
			"Convolution",
			"opset1",
			r#"strides="1,1" pads_begin="0,0" pads_end="0,0" dilations="1,1" auto_pad="explicit""#
				.to_string(),
			vec![x.clone(), w.clone()],
			x.ty,
			sh,
		)
	}

	/// `x` gathered along `ax` by the index tensor `idx`.
	pub fn gather(&mut self, x: &V, idx: &V, ax: usize) -> V {
		let mut sh = x.shape[..ax].to_vec();
		sh.extend_from_slice(&idx.shape);
		sh.extend_from_slice(&x.shape[ax + 1..]);
		let a = self.si64(ax as i64);
		self.op1(
			"Gather",
			"opset8",
			r#"batch_dims="0""#.to_string(),
			vec![x.clone(), idx.clone(), a],
			x.ty,
			sh,
		)
	}

	pub fn result(&mut self, x: &V, name: &str) {
		let id = self.push("Result", "opset1", String::new(), vec![x.clone()], vec![]);
		self.ops[id].name = name.to_string();
		let o = &mut self.ops[x.op].outputs[x.out].2;
		assert!(
			o.is_none() || o.as_deref() == Some(name),
			"result {name}: the tensor is named {o:?} already"
		);
		*o = Some(name.to_string());
		self.results.push(id);
	}

	/// The IR: parameters first (declared order), results last; ops no
	/// result depends on are left out (unused parameters stay arguments).
	pub fn xml(&self, name: &str) -> String {
		let live = self.live();
		let mut order: Vec<usize> = self.params.clone();
		order.extend(
			(0..self.ops.len())
				.filter(|&i| live[i] && !matches!(self.ops[i].kind, "Parameter" | "Result")),
		);
		order.extend(&self.results);
		let mut id_of = vec![0usize; self.ops.len()];
		for (n, &i) in order.iter().enumerate() {
			id_of[i] = n;
		}
		let port = |s: &mut String, id: usize, ty: Ty, shape: &[usize], names: Option<&str>| {
			let nm = names
				.map(|n| format!(r#" names="{n}""#))
				.unwrap_or_default();
			if shape.is_empty() {
				let _ = writeln!(
					s,
					"\t\t\t\t<port id=\"{id}\" precision=\"{}\"{nm} />",
					ty.precision()
				);
			} else {
				let _ = writeln!(
					s,
					"\t\t\t\t<port id=\"{id}\" precision=\"{}\"{nm}>",
					ty.precision()
				);
				for d in shape {
					let _ = writeln!(s, "\t\t\t\t\t<dim>{d}</dim>");
				}
				s.push_str("\t\t\t\t</port>\n");
			}
		};
		let mut s = String::with_capacity(self.ops.len() * 400);
		s.push_str("<?xml version=\"1.0\"?>\n");
		let _ = writeln!(s, "<net name=\"{name}\" version=\"11\">\n\t<layers>");
		for &i in &order {
			let o = &self.ops[i];
			let _ = writeln!(
				s,
				"\t\t<layer id=\"{}\" name=\"{}\" type=\"{}\" version=\"{}\">",
				id_of[i], o.name, o.kind, o.ver
			);
			if !o.data.is_empty() {
				let _ = writeln!(s, "\t\t\t<data {} />", o.data);
			}
			if !o.inputs.is_empty() {
				s.push_str("\t\t\t<input>\n");
				for (k, p) in o.inputs.iter().enumerate() {
					port(&mut s, k, p.ty, &p.shape, None);
				}
				s.push_str("\t\t\t</input>\n");
			}
			if !o.outputs.is_empty() {
				s.push_str("\t\t\t<output>\n");
				for (k, (ty, shape, nm)) in o.outputs.iter().enumerate() {
					port(&mut s, o.inputs.len() + k, *ty, shape, nm.as_deref());
				}
				s.push_str("\t\t\t</output>\n");
			}
			s.push_str("\t\t</layer>\n");
		}
		s.push_str("\t</layers>\n\t<edges>\n");
		for &i in &order {
			let o = &self.ops[i];
			for (k, p) in o.inputs.iter().enumerate() {
				let from = &self.ops[p.op];
				let _ = writeln!(s, "\t\t<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{}\" to-port=\"{k}\" />", id_of[p.op], from.inputs.len() + p.out, id_of[i]);
			}
		}
		s.push_str("\t</edges>\n</net>\n");
		s
	}

	/// Write `PATH.xml` and `PATH.bin` (`path` ends in `.xml`).
	pub fn save(&self, path: &std::path::Path, name: &str) -> Result<(), String> {
		let e = |x: std::io::Error| format!("{}: {x}", path.display());
		std::fs::write(path, self.xml(name)).map_err(e)?;
		std::fs::write(path.with_extension("bin"), &self.bin).map_err(e)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn half_rounds_like_numpy() {
		// values checked against np.float16
		for (x, h) in [
			(0.0, 0x0000u16),
			(1.0, 0x3c00),
			(-2.0, 0xc000),
			(65504.0, 0x7bff),
			(65520.0, 0x7c00),
			(0.0001, 0x068e),
			(6.2e-05, 0x0410),
			(5.960464477539063e-08, 0x0001),
			(2.9802322387695312e-08, 0x0000),
			(2.98023223876954e-08, 0x0001),
			(0.1, 0x2e66),
			(0.3333333333333333, 0x3555),
			(0.00390625, 0x1c00),
			(38.484848484848484, 0x50d0),
			(3e-05, 0x01f7),
			(-7.3e-06, 0x807a),
			(1234.5678, 0x64d3),
			(0.9999999, 0x3c00),
		] {
			assert_eq!(f16_bits(x), h, "{x}");
		}
		assert_eq!(f16_bits(f64::INFINITY), 0x7c00);
	}

	#[test]
	fn shapes_infer() {
		let mut g = Graph::new();
		let x = g.parameter("x", Ty::F16, &[1, 8, 16]);
		let w = g.c16(&[4, 16], &[0.5; 64]);
		let y = g.matmul(&x, &w, false, true);
		assert_eq!(y.shape, [1, 8, 4]);
		let p = g.variadic_split(&y, 2, &[1, 3]);
		assert_eq!(p[1].shape, [1, 8, 3]);
		let r = g.reshape(&p[1], &[1, -1, 6]);
		assert_eq!(r.shape, [1, 4, 6]);
		let t = g.transpose(&r, &[0, 2, 1]);
		assert_eq!(t.shape, [1, 6, 4]);
		let s = g.slice(&t, 1, 3, 1, 1);
		assert_eq!(s.shape, [1, 2, 4]);
		g.result(&s, "y");
		let xml = g.xml("t");
		assert!(xml.contains(r#"<layer id="0" name="x" type="Parameter""#));
		assert!(xml.trim_end().ends_with("</net>"));
	}
}
