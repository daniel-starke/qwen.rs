//! Owning wrappers over the raw C handles in [`crate::ffi`].

use std::ffi::{c_char, c_void, CString};
use std::path::Path;
use std::sync::Arc;

use crate::ffi::{self, Api};

pub struct Runtime {
	pub api: Arc<Api>,
	core: *mut ffi::Core,
}

// The C API documents `ov_core_t` as thread-safe; requests are not shared.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Drop for Runtime {
	fn drop(&mut self) {
		// SAFETY: created by `ov_core_create`, freed once.
		unsafe { (self.api.core_free)(self.core) }
	}
}

fn cs(s: &str) -> CString {
	CString::new(s).expect("interior NUL")
}

impl Runtime {
	pub fn new(lib_dir: Option<&Path>) -> Result<Runtime, String> {
		let api = Arc::new(Api::load(lib_dir)?);
		let mut core = std::ptr::null_mut();
		// SAFETY: valid out-parameter.
		api.check(unsafe { (api.core_create)(&mut core) }, "ov_core_create")?;
		Ok(Runtime { api, core })
	}

	pub fn devices(&self) -> Vec<String> {
		let mut d = ffi::Devices {
			devices: std::ptr::null_mut(),
			size: 0,
		};
		// SAFETY: valid out-parameter, freed below.
		unsafe {
			if (self.api.core_get_available_devices)(self.core, &mut d) != 0 {
				return Vec::new();
			}
			let v = (0..d.size).map(|i| ffi::cstr(*d.devices.add(i))).collect();
			(self.api.available_devices_free)(&mut d);
			v
		}
	}

	pub fn property(&self, device: &str, key: &str) -> Option<String> {
		let (d, k) = (cs(device), cs(key));
		let mut out: *mut c_char = std::ptr::null_mut();
		// SAFETY: valid strings and out-parameter; freed with `ov_free`.
		unsafe {
			if (self.api.core_get_property)(self.core, d.as_ptr(), k.as_ptr(), &mut out) != 0 {
				return None;
			}
			let s = ffi::cstr(out);
			(self.api.free)(out);
			Some(s)
		}
	}

	pub fn read_model(&self, xml: &Path) -> Result<OvModel, String> {
		self.read_model_bin(xml, None)
	}

	/// Read `xml` with its weights from `bin` (default: beside the `.xml`).
	pub fn read_model_bin(&self, xml: &Path, bin: Option<&Path>) -> Result<OvModel, String> {
		let p = cs(&xml.to_string_lossy());
		let b = bin.map(|b| cs(&b.to_string_lossy()));
		let mut m = std::ptr::null_mut();
		// SAFETY: valid strings and out-parameter; a null bin path means
		// "the .bin beside the .xml".
		let st = unsafe {
			(self.api.core_read_model)(
				self.core,
				p.as_ptr(),
				b.as_ref().map_or(std::ptr::null(), |b| b.as_ptr()),
				&mut m,
			)
		};
		self.api.check(st, &format!("read {}", xml.display()))?;
		Ok(OvModel {
			api: self.api.clone(),
			ptr: m,
		})
	}

	/// Compile for `device` with string properties (`("CACHE_DIR", "…")`).
	pub fn compile(
		&self,
		model: &OvModel,
		device: &str,
		props: &[(&str, &str)],
	) -> Result<Compiled, String> {
		let dev = cs(device);
		let owned: Vec<CString> = props.iter().flat_map(|(k, v)| [cs(k), cs(v)]).collect();
		let mut a: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
		const MAX: usize = 16;
		if a.len() > 2 * MAX {
			return Err(format!(
				"compile: at most {MAX} properties are supported, got {}",
				a.len() / 2
			));
		}
		let f = self.api.core_compile_model;
		let mut out = std::ptr::null_mut();
		let (c, m, d, o) = (
			self.core as *const _,
			model.ptr as *const _,
			dev.as_ptr(),
			&mut out as *mut _,
		);
		// The property list is C varargs, so the arity is fixed at the call
		// site: pad to `MAX` pairs by repeating the last one (setting a
		// property twice to the same value is a no-op).
		let st = if a.is_empty() {
			// SAFETY: no varargs.
			unsafe { f(c, m, d, 0, o) }
		} else {
			while a.len() < 2 * MAX {
				let (k, v) = (a[a.len() - 2], a[a.len() - 1]);
				a.extend([k, v]);
			}
			// SAFETY: every pointer is valid for the call; `a` holds `2 * MAX`
			// NUL-terminated strings alternating key / value.
			unsafe {
				f(
					c,
					m,
					d,
					2 * MAX,
					o,
					a[0],
					a[1],
					a[2],
					a[3],
					a[4],
					a[5],
					a[6],
					a[7],
					a[8],
					a[9],
					a[10],
					a[11],
					a[12],
					a[13],
					a[14],
					a[15],
					a[16],
					a[17],
					a[18],
					a[19],
					a[20],
					a[21],
					a[22],
					a[23],
					a[24],
					a[25],
					a[26],
					a[27],
					a[28],
					a[29],
					a[30],
					a[31],
				)
			}
		};
		self.api.check(st, &format!("compile for {device}"))?;
		Ok(Compiled {
			api: self.api.clone(),
			ptr: out,
		})
	}
}

pub struct OvModel {
	api: Arc<Api>,
	ptr: *mut ffi::Model,
}

impl OvModel {
	/// Reshape the named inputs to static shapes.
	pub fn reshape(&self, shapes: &[(&str, Vec<i64>)]) -> Result<(), String> {
		let names: Vec<CString> = shapes.iter().map(|(n, _)| cs(n)).collect();
		let ptrs: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
		let mut ps: Vec<ffi::PartialShape> = Vec::with_capacity(shapes.len());
		let mut err = Ok(());
		for (_, d) in shapes {
			let dims: Vec<ffi::Dim> = d.iter().map(|&v| ffi::Dim { min: v, max: v }).collect();
			let mut p = ffi::PartialShape {
				rank: ffi::Dim { min: 0, max: 0 },
				dims: std::ptr::null_mut(),
			};
			// SAFETY: `dims` outlives the call, which copies it.
			let st = unsafe {
				(self.api.partial_shape_create)(dims.len() as i64, dims.as_ptr(), &mut p)
			};
			if let Err(e) = self.api.check(st, "ov_partial_shape_create") {
				err = Err(e);
				break;
			}
			ps.push(p);
		}
		if err.is_ok() {
			// SAFETY: `ptrs` and `ps` hold `shapes.len()` valid entries.
			let st =
				unsafe { (self.api.model_reshape)(self.ptr, ptrs.as_ptr(), ps.as_ptr(), ps.len()) };
			err = self.api.check(st, "reshape");
		}
		for p in &mut ps {
			// SAFETY: created above, freed once.
			unsafe { (self.api.partial_shape_free)(p) };
		}
		err
	}

	/// A copy of the model whose named inputs and outputs are `f16` tensors
	/// (converted inside the model).
	pub fn with_f16_io(&self, inputs: &[&str], outputs: &[&str]) -> Result<OvModel, String> {
		let a = &self.api;
		let mut ppp = std::ptr::null_mut();
		// SAFETY: every handle below is created, used and freed in order;
		// `build` returns a new model the caller owns.
		unsafe {
			a.check((a.ppp_create)(self.ptr, &mut ppp), "ppp create")?;
			let res = (|| {
				for n in inputs {
					let c = cs(n);
					let mut info = std::ptr::null_mut();
					a.check(
						(a.ppp_input_by_name)(ppp, c.as_ptr(), &mut info),
						&format!("ppp input {n}"),
					)?;
					let mut ti = std::ptr::null_mut();
					let st = (a.ppp_input_tensor_info)(info, &mut ti);
					if st != 0 {
						(a.ppp_input_info_free)(info);
						return a.check(st, "ppp input tensor info").map(|_| unreachable!());
					}
					let st = (a.ppp_input_set_element_type)(ti, ffi::ET_F16);
					(a.ppp_input_tensor_info_free)(ti);
					(a.ppp_input_info_free)(info);
					a.check(st, &format!("ppp f16 input {n}"))?;
				}
				for n in outputs {
					let c = cs(n);
					let mut info = std::ptr::null_mut();
					a.check(
						(a.ppp_output_by_name)(ppp, c.as_ptr(), &mut info),
						&format!("ppp output {n}"),
					)?;
					let mut ti = std::ptr::null_mut();
					let st = (a.ppp_output_tensor_info)(info, &mut ti);
					if st != 0 {
						(a.ppp_output_info_free)(info);
						return a
							.check(st, "ppp output tensor info")
							.map(|_| unreachable!());
					}
					let st = (a.ppp_output_set_element_type)(ti, ffi::ET_F16);
					(a.ppp_output_tensor_info_free)(ti);
					(a.ppp_output_info_free)(info);
					a.check(st, &format!("ppp f16 output {n}"))?;
				}
				let mut m = std::ptr::null_mut();
				a.check((a.ppp_build)(ppp, &mut m), "ppp build")?;
				Ok(OvModel {
					api: a.clone(),
					ptr: m,
				})
			})();
			(a.ppp_free)(ppp);
			res
		}
	}
}

impl Drop for OvModel {
	fn drop(&mut self) {
		// SAFETY: owned handle, freed once.
		unsafe { (self.api.model_free)(self.ptr) }
	}
}

pub struct Compiled {
	api: Arc<Api>,
	ptr: *mut ffi::Compiled,
}

unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Drop for Compiled {
	fn drop(&mut self) {
		// SAFETY: owned handle, freed once.
		unsafe { (self.api.compiled_model_free)(self.ptr) }
	}
}

impl Compiled {
	/// A fresh request. Its state variables (KV cache, SSM and conv state)
	/// start at their initial values: the C API has no `reset_state`, so a
	/// new conversation is a new request.
	pub fn request(&self) -> Result<Request, String> {
		let mut r = std::ptr::null_mut();
		// SAFETY: valid handle and out-parameter.
		let st = unsafe { (self.api.compiled_model_create_infer_request)(self.ptr, &mut r) };
		self.api.check(st, "create infer request")?;
		Ok(Request {
			api: self.api.clone(),
			ptr: r,
			inputs: Vec::new(),
		})
	}
}

/// An input buffer and the tensor that borrows it. The request keeps a
/// reference to the tensor, so both must outlive the next `infer`.
struct Input {
	name: CString,
	_buf: Vec<u64>,
	tensor: *mut ffi::Tensor,
}

pub struct Request {
	api: Arc<Api>,
	ptr: *mut ffi::Request,
	inputs: Vec<Input>,
}

unsafe impl Send for Request {}

impl Drop for Request {
	fn drop(&mut self) {
		// SAFETY: owned handles, freed once; the request goes first so no
		// tensor is freed while it still refers to it.
		unsafe {
			(self.api.infer_request_free)(self.ptr);
			for i in &self.inputs {
				(self.api.tensor_free)(i.tensor);
			}
		}
	}
}

/// Element types an input may be built from.
pub trait Elem: Copy {
	const ET: u32;
}
impl Elem for f32 {
	const ET: u32 = ffi::ET_F32;
}
impl Elem for i32 {
	const ET: u32 = ffi::ET_I32;
}
impl Elem for i64 {
	const ET: u32 = ffi::ET_I64;
}
/// Raw `f16` bits.
impl Elem for u16 {
	const ET: u32 = ffi::ET_F16;
}

/// A host buffer wrapped as an OpenVINO tensor and bound to requests by
/// reference, so state moves between steps without a copy (a step's output
/// tensor is the next step's input).
pub struct HostTensor {
	api: Arc<Api>,
	buf: Vec<u64>,
	t: *mut ffi::Tensor,
	pub dims: Vec<i64>,
	elem_bytes: usize,
}

unsafe impl Send for HostTensor {}

impl Drop for HostTensor {
	fn drop(&mut self) {
		// SAFETY: owned handle. A request that still binds it holds its own
		// reference to the tensor object, but not to `buf`: owners drop
		// their requests first.
		unsafe { (self.api.tensor_free)(self.t) }
	}
}

impl HostTensor {
	pub fn zeros<T: Elem>(rt: &Runtime, dims: &[i64]) -> Result<HostTensor, String> {
		let n: i64 = dims.iter().product();
		let eb = std::mem::size_of::<T>();
		let mut buf = vec![0u64; (n as usize * eb).div_ceil(8).max(1)];
		let api = rt.api.clone();
		let mut shape = ffi::Shape {
			rank: 0,
			dims: std::ptr::null_mut(),
		};
		let mut t = std::ptr::null_mut();
		// SAFETY: as in `Request::set`; `buf` is owned by the result and its
		// heap block never moves.
		unsafe {
			api.check(
				(api.shape_create)(dims.len() as i64, dims.as_ptr(), &mut shape),
				"ov_shape_create",
			)?;
			let st = (api.tensor_create_from_host_ptr)(
				T::ET,
				ffi::Shape {
					rank: shape.rank,
					dims: shape.dims,
				},
				buf.as_mut_ptr() as *mut c_void,
				&mut t,
			);
			(api.shape_free)(&mut shape);
			api.check(st, "host tensor")?;
		}
		Ok(HostTensor {
			api,
			buf,
			t,
			dims: dims.to_vec(),
			elem_bytes: eb,
		})
	}

	#[allow(clippy::should_implement_trait)] // generic over the element type
	pub fn as_mut<T: Elem>(&mut self) -> &mut [T] {
		assert_eq!(std::mem::size_of::<T>(), self.elem_bytes);
		let n = self.dims.iter().product::<i64>() as usize;
		// SAFETY: 8-aligned buffer of `n` elements of `T`'s size.
		unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut T, n) }
	}

	#[allow(clippy::should_implement_trait)] // generic over the element type
	pub fn as_ref<T: Elem>(&self) -> &[T] {
		assert_eq!(std::mem::size_of::<T>(), self.elem_bytes);
		let n = self.dims.iter().product::<i64>() as usize;
		// SAFETY: as above.
		unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const T, n) }
	}
}

impl Request {
	/// Set input `name` to a copy of `data` with shape `dims`.
	pub fn set<T: Elem>(&mut self, name: &str, dims: &[i64], data: &[T]) -> Result<(), String> {
		let n: i64 = dims.iter().product();
		assert_eq!(
			n as usize,
			data.len(),
			"input {name}: shape {dims:?} vs {} elements",
			data.len()
		);
		// SAFETY: plain old data viewed as bytes.
		let raw = unsafe {
			std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
		};
		self.set_raw(name, T::ET, dims, raw)
	}

	/// Set input `name` (element type `et`, `ffi::ET_*`) to a copy of the
	/// packed bytes `data` (int4: two per byte, low nibble first).
	pub fn set_raw(
		&mut self,
		name: &str,
		et: u32,
		dims: &[i64],
		data: &[u8],
	) -> Result<(), String> {
		let bytes = data.len();
		let mut buf = vec![0u64; bytes.div_ceil(8).max(1)];
		// SAFETY: `buf` holds at least `bytes` bytes and is 8-aligned.
		unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr() as *mut u8, bytes) };
		let mut shape = ffi::Shape {
			rank: 0,
			dims: std::ptr::null_mut(),
		};
		let mut t = std::ptr::null_mut();
		let name_c = cs(name);
		// SAFETY: `dims` outlives the call; `shape` is freed after the tensor
		// copied it; `buf` lives in `self.inputs` until replaced or dropped.
		unsafe {
			self.api.check(
				(self.api.shape_create)(dims.len() as i64, dims.as_ptr(), &mut shape),
				"ov_shape_create",
			)?;
			let st = (self.api.tensor_create_from_host_ptr)(
				et,
				ffi::Shape {
					rank: shape.rank,
					dims: shape.dims,
				},
				buf.as_mut_ptr() as *mut c_void,
				&mut t,
			);
			(self.api.shape_free)(&mut shape);
			self.api.check(st, &format!("tensor for {name}"))?;
			let st = (self.api.infer_request_set_tensor)(self.ptr, name_c.as_ptr(), t);
			if st != 0 {
				(self.api.tensor_free)(t);
				return self.api.check(st, &format!("set input {name}"));
			}
		}
		if let Some(pos) = self.inputs.iter().position(|i| i.name == name_c) {
			let old = self.inputs.swap_remove(pos);
			// SAFETY: the request now refers to the new tensor instead.
			unsafe { (self.api.tensor_free)(old.tensor) };
		}
		self.inputs.push(Input {
			name: name_c,
			_buf: buf,
			tensor: t,
		});
		Ok(())
	}

	/// Bind `t` as input or output `name`. The caller keeps `t` alive while
	/// it is bound.
	pub fn bind(&mut self, name: &str, t: &HostTensor) -> Result<(), String> {
		let c = cs(name);
		// SAFETY: valid request, name and tensor.
		let st = unsafe { (self.api.infer_request_set_tensor)(self.ptr, c.as_ptr(), t.t) };
		self.api.check(st, &format!("bind {name}"))
	}

	pub fn infer(&mut self) -> Result<(), String> {
		// SAFETY: valid request with every input set.
		let st = unsafe { (self.api.infer_request_infer)(self.ptr) };
		self.api.check(st, "infer")
	}

	/// Output `name` as (shape, f32 data). The slice is valid until the next
	/// `infer` or `set`.
	pub fn output_f32(&self, name: &str) -> Result<(Vec<i64>, &[f32]), String> {
		let (d, b) = self.output_raw(name, 4)?;
		// SAFETY: the tensor's data is f32, `b` holds whole elements.
		Ok((d, unsafe {
			std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4)
		}))
	}

	/// Output `name` as (shape, raw bytes) for elements of `elem` bytes.
	/// The slice is valid until the next `infer` or `set`.
	pub fn output_raw(&self, name: &str, elem: usize) -> Result<(Vec<i64>, &[u8]), String> {
		let n = cs(name);
		let mut t = std::ptr::null_mut();
		// SAFETY: valid request, name and out-parameters; the tensor handle
		// is a reference into the request, freed here, its data is not.
		unsafe {
			self.api.check(
				(self.api.infer_request_get_tensor)(self.ptr, n.as_ptr(), &mut t),
				&format!("output {name}"),
			)?;
			let mut shape = ffi::Shape {
				rank: 0,
				dims: std::ptr::null_mut(),
			};
			let st = (self.api.tensor_get_shape)(t, &mut shape);
			if st != 0 {
				(self.api.tensor_free)(t);
				return self.api.check(st, "output shape").map(|_| unreachable!());
			}
			let dims = std::slice::from_raw_parts(shape.dims, shape.rank as usize).to_vec();
			(self.api.shape_free)(&mut shape);
			let mut p: *mut c_void = std::ptr::null_mut();
			let st = (self.api.tensor_data)(t, &mut p);
			(self.api.tensor_free)(t);
			self.api.check(st, "output data")?;
			let len = dims.iter().product::<i64>() as usize;
			Ok((dims, std::slice::from_raw_parts(p as *const u8, len * elem)))
		}
	}
}
