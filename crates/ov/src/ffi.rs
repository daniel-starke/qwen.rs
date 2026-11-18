//! The subset of OpenVINO's C API (`openvino/c/*.h`) this crate calls,
//! resolved at run time from `openvino_c.dll` / `libopenvino_c.so`.
//!
//! Loading instead of linking keeps OpenVINO out of the build: no import
//! library, no headers, and a binary that starts (and runs every other path)
//! on a host that has never seen OpenVINO.

use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};

/// Opaque handles (`ov_core_t`, …).
#[repr(C)]
pub struct Core([u8; 0]);
#[repr(C)]
pub struct Model([u8; 0]);
#[repr(C)]
pub struct Compiled([u8; 0]);
#[repr(C)]
pub struct Request([u8; 0]);
#[repr(C)]
pub struct Tensor([u8; 0]);
#[repr(C)]
pub struct Ppp([u8; 0]);
#[repr(C)]
pub struct PppInfo([u8; 0]);
#[repr(C)]
pub struct PppTensorInfo([u8; 0]);

/// `ov_dimension_t` (`min == max` for a static dimension).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dim {
	pub min: i64,
	pub max: i64,
}

/// `ov_partial_shape_t`.
#[repr(C)]
pub struct PartialShape {
	pub rank: Dim,
	pub dims: *mut Dim,
}

/// `ov_shape_t`.
#[repr(C)]
pub struct Shape {
	pub rank: i64,
	pub dims: *mut i64,
}

/// `ov_available_devices_t`.
#[repr(C)]
pub struct Devices {
	pub devices: *mut *mut c_char,
	pub size: usize,
}

/// `ov_version_t`.
#[repr(C)]
pub struct Version {
	pub build_number: *const c_char,
	pub description: *const c_char,
}

/// `ov_element_type_e` values used here.
pub const ET_F16: u32 = 3;
pub const ET_F32: u32 = 4;
pub const ET_I4: u32 = 6;
pub const ET_I8: u32 = 7;
pub const ET_I32: u32 = 9;
pub const ET_I64: u32 = 10;

type St = i32;

/// `ov_core_compile_model` is variadic: `property_args_size` counts the
/// `const char*` key/value arguments that follow.
pub type CompileFn = unsafe extern "C" fn(
	*const Core,
	*const Model,
	*const c_char,
	usize,
	*mut *mut Compiled,
	...
) -> St;

pub struct Api {
	_lib: Library,
	pub dir: PathBuf,
	pub get_openvino_version: unsafe extern "C" fn(*mut Version) -> St,
	pub version_free: unsafe extern "C" fn(*mut Version),
	pub get_error_info: unsafe extern "C" fn(St) -> *const c_char,
	pub get_last_err_msg: unsafe extern "C" fn() -> *const c_char,
	pub free: unsafe extern "C" fn(*const c_char),
	pub core_create: unsafe extern "C" fn(*mut *mut Core) -> St,
	pub core_free: unsafe extern "C" fn(*mut Core),
	pub core_read_model:
		unsafe extern "C" fn(*const Core, *const c_char, *const c_char, *mut *mut Model) -> St,
	pub core_compile_model: CompileFn,
	pub core_get_available_devices: unsafe extern "C" fn(*const Core, *mut Devices) -> St,
	pub available_devices_free: unsafe extern "C" fn(*mut Devices),
	pub core_get_property:
		unsafe extern "C" fn(*const Core, *const c_char, *const c_char, *mut *mut c_char) -> St,
	pub model_free: unsafe extern "C" fn(*mut Model),
	pub compiled_model_create_infer_request:
		unsafe extern "C" fn(*const Compiled, *mut *mut Request) -> St,
	pub compiled_model_free: unsafe extern "C" fn(*mut Compiled),
	pub infer_request_set_tensor:
		unsafe extern "C" fn(*mut Request, *const c_char, *const Tensor) -> St,
	pub infer_request_get_tensor:
		unsafe extern "C" fn(*const Request, *const c_char, *mut *mut Tensor) -> St,
	pub infer_request_infer: unsafe extern "C" fn(*mut Request) -> St,
	pub infer_request_free: unsafe extern "C" fn(*mut Request),
	pub tensor_create_from_host_ptr:
		unsafe extern "C" fn(u32, Shape, *mut c_void, *mut *mut Tensor) -> St,
	pub tensor_get_shape: unsafe extern "C" fn(*const Tensor, *mut Shape) -> St,
	pub tensor_data: unsafe extern "C" fn(*const Tensor, *mut *mut c_void) -> St,
	pub tensor_free: unsafe extern "C" fn(*mut Tensor),
	pub shape_create: unsafe extern "C" fn(i64, *const i64, *mut Shape) -> St,
	pub shape_free: unsafe extern "C" fn(*mut Shape) -> St,
	pub partial_shape_create: unsafe extern "C" fn(i64, *const Dim, *mut PartialShape) -> St,
	pub partial_shape_free: unsafe extern "C" fn(*mut PartialShape),
	pub model_reshape:
		unsafe extern "C" fn(*const Model, *const *const c_char, *const PartialShape, usize) -> St,
	pub ppp_create: unsafe extern "C" fn(*const Model, *mut *mut Ppp) -> St,
	pub ppp_free: unsafe extern "C" fn(*mut Ppp),
	pub ppp_build: unsafe extern "C" fn(*const Ppp, *mut *mut Model) -> St,
	pub ppp_input_by_name: unsafe extern "C" fn(*const Ppp, *const c_char, *mut *mut PppInfo) -> St,
	pub ppp_output_by_name:
		unsafe extern "C" fn(*const Ppp, *const c_char, *mut *mut PppInfo) -> St,
	pub ppp_input_info_free: unsafe extern "C" fn(*mut PppInfo),
	pub ppp_output_info_free: unsafe extern "C" fn(*mut PppInfo),
	pub ppp_input_tensor_info: unsafe extern "C" fn(*const PppInfo, *mut *mut PppTensorInfo) -> St,
	pub ppp_output_tensor_info: unsafe extern "C" fn(*const PppInfo, *mut *mut PppTensorInfo) -> St,
	pub ppp_input_tensor_info_free: unsafe extern "C" fn(*mut PppTensorInfo),
	pub ppp_output_tensor_info_free: unsafe extern "C" fn(*mut PppTensorInfo),
	pub ppp_input_set_element_type: unsafe extern "C" fn(*mut PppTensorInfo, u32) -> St,
	pub ppp_output_set_element_type: unsafe extern "C" fn(*mut PppTensorInfo, u32) -> St,
}

// Every entry is a plain function pointer into a library that is never
// unloaded while an `Api` exists.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
	/// Load `openvino_c` from `dir` (the runtime's `bin` directory, or a
	/// pip wheel's `openvino/libs`). `None` tries, in order, `$OPENVINO_LIB`,
	/// `$INTEL_OPENVINO_DIR/runtime/bin/intel64/Release`, and the default
	/// install location.
	pub fn load(dir: Option<&Path>) -> Result<Api, String> {
		let cands: Vec<PathBuf> = match dir {
			Some(d) => vec![d.to_path_buf()],
			None => default_dirs(),
		};
		let mut errs = Vec::new();
		for d in &cands {
			let f = d.join(LIB_NAME);
			if !f.is_file() {
				errs.push(format!("{}: not found", f.display()));
				continue;
			}
			// An installed runtime keeps TBB in `runtime/3rdparty/tbb/bin`, not
			// beside `openvino.dll` (a pip wheel does); the loader finds it
			// through PATH.
			let tbb = d.join("../../../3rdparty/tbb/bin");
			if tbb.is_dir() {
				let mut p = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
					.collect::<Vec<_>>();
				p.insert(0, tbb);
				if let Ok(j) = std::env::join_paths(p) {
					std::env::set_var("PATH", j);
				}
			}
			match Library::open(&f) {
				Ok(lib) => return Api::resolve(lib, d.clone()),
				Err(e) => errs.push(format!("{}: {e}", f.display())),
			}
		}
		Err(format!(
			"could not load the OpenVINO runtime ({LIB_NAME}); pass --ov-lib DIR or set OPENVINO_LIB:\n  {}",
			errs.join("\n  ")
		))
	}

	fn resolve(lib: Library, dir: PathBuf) -> Result<Api, String> {
		macro_rules! sym {
			($n:literal) => {{
				let p = lib.sym(concat!($n, "\0"));
				if p.is_null() {
					return Err(format!("{LIB_NAME} has no symbol {}", $n));
				}
				// SAFETY: the C declaration of `$n` matches the field type.
				// (The target type is the field's: a macro cannot name it.)
				#[allow(clippy::missing_transmute_annotations)]
				let f = unsafe { std::mem::transmute::<*mut c_void, _>(p) };
				f
			}};
		}
		Ok(Api {
			get_openvino_version: sym!("ov_get_openvino_version"),
			version_free: sym!("ov_version_free"),
			get_error_info: sym!("ov_get_error_info"),
			get_last_err_msg: sym!("ov_get_last_err_msg"),
			free: sym!("ov_free"),
			core_create: sym!("ov_core_create"),
			core_free: sym!("ov_core_free"),
			core_read_model: sym!("ov_core_read_model"),
			core_compile_model: sym!("ov_core_compile_model"),
			core_get_available_devices: sym!("ov_core_get_available_devices"),
			available_devices_free: sym!("ov_available_devices_free"),
			core_get_property: sym!("ov_core_get_property"),
			model_free: sym!("ov_model_free"),
			compiled_model_create_infer_request: sym!("ov_compiled_model_create_infer_request"),
			compiled_model_free: sym!("ov_compiled_model_free"),
			infer_request_set_tensor: sym!("ov_infer_request_set_tensor"),
			infer_request_get_tensor: sym!("ov_infer_request_get_tensor"),
			infer_request_infer: sym!("ov_infer_request_infer"),
			infer_request_free: sym!("ov_infer_request_free"),
			tensor_create_from_host_ptr: sym!("ov_tensor_create_from_host_ptr"),
			tensor_get_shape: sym!("ov_tensor_get_shape"),
			tensor_data: sym!("ov_tensor_data"),
			tensor_free: sym!("ov_tensor_free"),
			shape_create: sym!("ov_shape_create"),
			shape_free: sym!("ov_shape_free"),
			partial_shape_create: sym!("ov_partial_shape_create"),
			partial_shape_free: sym!("ov_partial_shape_free"),
			model_reshape: sym!("ov_model_reshape"),
			ppp_create: sym!("ov_preprocess_prepostprocessor_create"),
			ppp_free: sym!("ov_preprocess_prepostprocessor_free"),
			ppp_build: sym!("ov_preprocess_prepostprocessor_build"),
			ppp_input_by_name: sym!("ov_preprocess_prepostprocessor_get_input_info_by_name"),
			ppp_output_by_name: sym!("ov_preprocess_prepostprocessor_get_output_info_by_name"),
			ppp_input_info_free: sym!("ov_preprocess_input_info_free"),
			ppp_output_info_free: sym!("ov_preprocess_output_info_free"),
			ppp_input_tensor_info: sym!("ov_preprocess_input_info_get_tensor_info"),
			ppp_output_tensor_info: sym!("ov_preprocess_output_info_get_tensor_info"),
			ppp_input_tensor_info_free: sym!("ov_preprocess_input_tensor_info_free"),
			ppp_output_tensor_info_free: sym!("ov_preprocess_output_tensor_info_free"),
			ppp_input_set_element_type: sym!("ov_preprocess_input_tensor_info_set_element_type"),
			ppp_output_set_element_type: sym!("ov_preprocess_output_set_element_type"),
			_lib: lib,
			dir,
		})
	}

	/// Turn a status into `Ok` or an error carrying OpenVINO's own message
	/// (the exception text, which names the failing op or property).
	pub fn check(&self, st: St, what: &str) -> Result<(), String> {
		if st == 0 {
			return Ok(());
		}
		// SAFETY: both return static / thread-local C strings or null.
		let info = unsafe { cstr((self.get_error_info)(st)) };
		let msg = unsafe { cstr((self.get_last_err_msg)()) };
		Err(format!("{what}: {info} ({st}): {}", msg.trim()))
	}

	pub fn version(&self) -> String {
		let mut v = Version {
			build_number: std::ptr::null(),
			description: std::ptr::null(),
		};
		// SAFETY: `v` is a valid out-parameter; freed below.
		unsafe {
			if (self.get_openvino_version)(&mut v) != 0 {
				return "unknown".into();
			}
			let s = cstr(v.build_number);
			(self.version_free)(&mut v);
			s
		}
	}
}

/// # Safety
/// `p` is null or a NUL-terminated string.
pub unsafe fn cstr(p: *const c_char) -> String {
	if p.is_null() {
		String::new()
	} else {
		CStr::from_ptr(p).to_string_lossy().into_owned()
	}
}

fn default_dirs() -> Vec<PathBuf> {
	let mut v = Vec::new();
	if let Some(d) = std::env::var_os("OPENVINO_LIB") {
		v.push(PathBuf::from(d));
	}
	if let Some(d) = std::env::var_os("INTEL_OPENVINO_DIR") {
		v.push(PathBuf::from(d).join(RUNTIME_SUBDIR));
	}
	for d in DEFAULT_INSTALLS {
		v.push(PathBuf::from(d).join(RUNTIME_SUBDIR));
	}
	v
}

#[cfg(windows)]
const LIB_NAME: &str = "openvino_c.dll";
#[cfg(not(windows))]
const LIB_NAME: &str = "libopenvino_c.so";
#[cfg(windows)]
const RUNTIME_SUBDIR: &str = "runtime/bin/intel64/Release";
#[cfg(not(windows))]
const RUNTIME_SUBDIR: &str = "runtime/lib/intel64";
#[cfg(windows)]
const DEFAULT_INSTALLS: &[&str] = &["C:/Program Files (x86)/Intel/openvino_2026"];
#[cfg(not(windows))]
const DEFAULT_INSTALLS: &[&str] = &["/opt/intel/openvino_2026"];

struct Library(*mut c_void);

#[cfg(windows)]
mod sys {
	use std::ffi::c_void;
	#[link(name = "kernel32")]
	extern "system" {
		pub fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
		pub fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
		pub fn GetLastError() -> u32;
	}
	/// Resolve the DLL's own imports (`openvino.dll`, `tbb12.dll`) from its
	/// directory rather than the process's.
	pub const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
}

#[cfg(not(windows))]
mod sys {
	use std::ffi::{c_char, c_int, c_void};
	extern "C" {
		pub fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
		pub fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
		pub fn dlerror() -> *const c_char;
	}
	pub const RTLD_NOW: c_int = 2;
}

impl Library {
	#[cfg(windows)]
	fn open(path: &Path) -> Result<Library, String> {
		use std::os::windows::ffi::OsStrExt;
		// `LOAD_WITH_ALTERED_SEARCH_PATH` needs an absolute path with
		// backslashes to take effect.
		let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
		let s = abs.to_string_lossy().replace('/', "\\");
		let s = s.strip_prefix(r"\\?\").unwrap_or(&s).to_string();
		let w: Vec<u16> = std::ffi::OsStr::new(&s)
			.encode_wide()
			.chain(Some(0))
			.collect();
		// SAFETY: `w` is NUL-terminated.
		let h = unsafe {
			sys::LoadLibraryExW(
				w.as_ptr(),
				std::ptr::null_mut(),
				sys::LOAD_WITH_ALTERED_SEARCH_PATH,
			)
		};
		if h.is_null() {
			return Err(format!("LoadLibraryExW failed, error {}", unsafe {
				sys::GetLastError()
			}));
		}
		Ok(Library(h))
	}

	#[cfg(not(windows))]
	fn open(path: &Path) -> Result<Library, String> {
		let c =
			std::ffi::CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
		// SAFETY: `c` is NUL-terminated.
		let h = unsafe { sys::dlopen(c.as_ptr(), sys::RTLD_NOW) };
		if h.is_null() {
			return Err(unsafe { cstr(sys::dlerror()) });
		}
		Ok(Library(h))
	}

	/// `name` must end in NUL.
	fn sym(&self, name: &str) -> *mut c_void {
		debug_assert!(name.ends_with('\0'));
		#[cfg(windows)]
		// SAFETY: valid module handle, NUL-terminated name.
		unsafe {
			sys::GetProcAddress(self.0, name.as_ptr())
		}
		#[cfg(not(windows))]
		// SAFETY: valid handle, NUL-terminated name.
		unsafe {
			sys::dlsym(self.0, name.as_ptr() as *const c_char)
		}
	}
}
