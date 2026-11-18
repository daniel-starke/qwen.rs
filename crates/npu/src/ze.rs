//! Level Zero + the Intel NPU graph extension, loaded at run time.
//!
//! Layouts follow the MIT-licensed `level-zero` v1.x `ze_api.h` and
//! `intel/level-zero-npu-extensions` (`ze_graph_ext.h`,
//! `ze_graph_profiling_ext.h`). The graph DDI table is read by slot index
//! (the table only ever grows), so a newer driver works unchanged.

use std::ffi::{c_char, c_void};
use std::ptr::{null, null_mut};

type H = *mut c_void;
type Res = u32;

pub const SUCCESS: Res = 0;

/// `QWEN_NPU_SUBMIT_LOG=FILE`: one line per queue submission, `FILETIME
/// tag file:line` (UTC 100 ns ticks, [`trace_tag`]'s value, the caller of
/// `submit` / `run`), to match ETW DxgKrnl NPU packets to the graphs that
/// made them. Buffered: flushed every 256 lines and by [`trace_flush`].
mod trace {
	use std::io::Write;
	use std::sync::atomic::{AtomicU32, Ordering};
	use std::sync::Mutex;

	pub static TAG: AtomicU32 = AtomicU32::new(0);
	static W: Mutex<Option<(std::io::BufWriter<std::fs::File>, u32)>> = Mutex::new(None);
	static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

	fn on() -> bool {
		*ON.get_or_init(|| {
			let Some(f) = std::env::var_os("QWEN_NPU_SUBMIT_LOG") else {
				return false;
			};
			match std::fs::File::create(&f) {
				Ok(f) => {
					*W.lock().unwrap() = Some((std::io::BufWriter::new(f), 0));
					true
				}
				Err(e) => {
					eprintln!("QWEN_NPU_SUBMIT_LOG: {e}");
					false
				}
			}
		})
	}

	pub fn rec(at: &std::panic::Location<'_>) {
		if !on() {
			return;
		}
		let ft = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|d| d.as_nanos() as u64 / 100 + 116_444_736_000_000_000)
			.unwrap_or(0);
		let mut g = W.lock().unwrap();
		if let Some((w, n)) = g.as_mut() {
			let _ = writeln!(
				w,
				"{ft} {} {}:{}",
				TAG.load(Ordering::Relaxed),
				at.file(),
				at.line()
			);
			*n += 1;
			if *n % 256 == 0 {
				let _ = w.flush();
			}
		}
	}

	pub fn flush() {
		if let Some((w, _)) = W.lock().unwrap().as_mut() {
			let _ = w.flush();
		}
	}
}

/// The tag [`trace`] lines carry from now on (e.g. the layer).
pub fn trace_tag(t: u32) {
	trace::TAG.store(t, std::sync::atomic::Ordering::Relaxed);
}

/// Flush the `QWEN_NPU_SUBMIT_LOG` lines written so far.
pub fn trace_flush() {
	trace::flush();
}

// ze_structure_type_t (core)
const ST_DRIVER_PROPERTIES: u32 = 0x1;
const ST_DEVICE_PROPERTIES: u32 = 0x3;
const ST_QUEUE_GROUP_PROPERTIES: u32 = 0x6;
const ST_CONTEXT_DESC: u32 = 0xd;
const ST_COMMAND_QUEUE_DESC: u32 = 0xe;
const ST_COMMAND_LIST_DESC: u32 = 0xf;
const ST_FENCE_DESC: u32 = 0x12;
const ST_HOST_MEM_ALLOC_DESC: u32 = 0x16;
const ST_INIT_DRIVER_TYPE_DESC: u32 = 0x0002_0021;
// ZE_experimental_mutable_command_list
const ST_MUTABLE_COMMAND_LIST_EXP_DESC: u32 = 0x0002_0016;
const ST_MUTABLE_COMMAND_ID_EXP_DESC: u32 = 0x0002_0017;
const ST_MUTABLE_COMMANDS_EXP_DESC: u32 = 0x0002_0018;
const ST_MUTABLE_GRAPH_ARGUMENT_EXP_DESC: u32 = 0x0002_0020;
const MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS: u32 = 1 << 7;
const INIT_DRIVER_TYPE_FLAG_NPU: u32 = 0x2;
const DEVICE_TYPE_VPU: u32 = 5;
const QUEUE_GROUP_COMPUTE: u32 = 0x1;

// graph extension
const ST_DEVICE_GRAPH_PROPERTIES_2: u32 = 0xf;
const ST_GRAPH_DESC_2: u32 = 0xe;
const ST_GRAPH_PROPERTIES_3: u32 = 0x11;
const ST_GRAPH_ARGUMENT_PROPERTIES_3: u32 = 0xd;
pub const FORMAT_NATIVE: u32 = 0x1;
pub const FORMAT_NGRAPH_LITE: u32 = 0x2;
pub const GRAPH_FLAG_DISABLE_CACHING: u32 = 1;
pub const GRAPH_FLAG_ENABLE_PROFILING: u32 = 2;
const STAGE_COMMAND_LIST_INITIALIZE: u32 = 0x1;
const STAGE_INITIALIZE: u32 = 0x2;

// ze_graph_dditable_ext_t slots
const DDI_DESTROY: usize = 1;
const DDI_SET_ARGUMENT: usize = 4;
const DDI_APPEND_INITIALIZE: usize = 5;
const DDI_APPEND_EXECUTE: usize = 6;
const DDI_GET_ARGUMENT_PROPERTIES3: usize = 11;
const DDI_BUILD_LOG: usize = 15;
const DDI_CREATE2: usize = 16;
const DDI_DEVICE_PROPERTIES2: usize = 19;
const DDI_GET_NATIVE2: usize = 20;
const DDI_INITIALIZE: usize = 22;
const DDI_SUPPORTED_OPTIONS: usize = 23;
const DDI_GET_PROPERTIES3: usize = 26;

// ze_graph_profiling_dditable_ext_t slots
const PROF_POOL_CREATE: usize = 0;
const PROF_POOL_DESTROY: usize = 1;
const PROF_QUERY_CREATE: usize = 2;
const PROF_QUERY_DESTROY: usize = 3;
const PROF_QUERY_GET_DATA: usize = 4;
pub const PROFILING_LAYER_LEVEL: u32 = 0x1;
pub const PROFILING_TASK_LEVEL: u32 = 0x2;

#[repr(C)]
struct InitDriverTypeDesc {
	stype: u32,
	pnext: *const c_void,
	flags: u32,
}

#[repr(C)]
struct DriverProperties {
	stype: u32,
	pnext: *mut c_void,
	uuid: [u8; 16],
	driver_version: u32,
}

#[repr(C)]
struct ExtensionProperties {
	name: [u8; 256],
	version: u32,
}

#[repr(C)]
pub struct DeviceProperties {
	stype: u32,
	pnext: *mut c_void,
	pub ty: u32,
	pub vendor_id: u32,
	pub device_id: u32,
	pub flags: u32,
	pub subdevice_id: u32,
	pub core_clock_rate: u32,
	pub max_mem_alloc_size: u64,
	pub max_hardware_contexts: u32,
	pub max_command_queue_priority: u32,
	pub num_threads_per_eu: u32,
	pub physical_eu_simd_width: u32,
	pub num_eus_per_subslice: u32,
	pub num_subslices_per_slice: u32,
	pub num_slices: u32,
	pub timer_resolution: u64,
	pub timestamp_valid_bits: u32,
	pub kernel_timestamp_valid_bits: u32,
	pub uuid: [u8; 16],
	pub name: [u8; 256],
}
const _: () = assert!(std::mem::size_of::<DeviceProperties>() == 368);

#[repr(C)]
struct MutableFlagsDesc {
	stype: u32,
	pnext: *const c_void,
	flags: u32,
}

#[repr(C)]
struct MutableGraphArgDesc {
	stype: u32,
	pnext: *const c_void,
	command_id: u64,
	arg_index: u32,
	arg_value: *const c_void,
}

#[repr(C)]
struct ContextDesc {
	stype: u32,
	pnext: *const c_void,
	flags: u32,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct Version3 {
	pub major: u32,
	pub minor: u32,
	pub patch: u32,
}

#[repr(C)]
struct DeviceGraphProperties2 {
	stype: u32,
	pnext: *mut c_void,
	graph_extension_version: u32,
	compiler_major: u16,
	compiler_minor: u16,
	graph_formats_supported: u32,
	max_ov_opset: u32,
	elf: Version3,
	runtime: Version3,
}

#[repr(C)]
struct GraphDesc2 {
	stype: u32,
	pnext: *mut c_void,
	format: u32,
	input_size: usize,
	input: *const u8,
	build_flags: *const c_char,
	flags: u32,
}
const _: () = assert!(std::mem::size_of::<GraphDesc2>() == 56);

#[repr(C)]
struct GraphProperties3 {
	stype: u32,
	pnext: *mut c_void,
	num_graph_args: u32,
	init_stage_required: u32,
	flags: u32,
}

#[repr(C)]
struct ArgumentProperties3 {
	stype: u32,
	pnext: *mut c_void,
	name: [u8; 256],
	ty: u32,
	dims: [u32; 5],
	network_precision: u32,
	network_layout: u32,
	device_precision: u32,
	device_layout: u32,
	quant_reverse_scale: f32,
	quant_zero_point: u8,
	dims_count: u32,
	debug_friendly_name: [u8; 256],
	associated_tensor_names: [[u8; 256]; 32],
	associated_tensor_names_count: u32,
}

#[repr(C)]
struct QueueGroupProperties {
	stype: u32,
	pnext: *mut c_void,
	flags: u32,
	max_memory_fill_pattern_size: usize,
	num_queues: u32,
}

#[repr(C)]
struct CommandQueueDesc {
	stype: u32,
	pnext: *const c_void,
	ordinal: u32,
	index: u32,
	flags: u32,
	mode: u32,
	priority: u32,
}

#[repr(C)]
struct QueueDescNpuExt1 {
	stype: u32,
	pnext: *const c_void,
	turbo: bool,
}

#[repr(C)]
struct QueueDescNpuExt2 {
	stype: u32,
	pnext: *const c_void,
	options: u32,
}

#[repr(C)]
struct CommandListDesc {
	stype: u32,
	pnext: *const c_void,
	ordinal: u32,
	flags: u32,
}

#[repr(C)]
struct FenceDesc {
	stype: u32,
	pnext: *const c_void,
	flags: u32,
}

#[repr(C)]
struct HostMemAllocDesc {
	stype: u32,
	pnext: *const c_void,
	flags: u32,
}

/// `ze_profiling_task_info` (344 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TaskInfo {
	pub name: [u8; 256],
	pub layer_type: [u8; 50],
	pub exec_type: u32,
	pub start_ns: u64,
	pub duration_ns: u64,
	pub active_cycles: u32,
	pub stall_cycles: u32,
	pub task_id: u32,
	pub parent_layer_id: u32,
}
const _: () = assert!(std::mem::size_of::<TaskInfo>() == 344);

/// `ze_profiling_layer_info` (368 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerInfo {
	pub name: [u8; 256],
	pub layer_type: [u8; 50],
	pub status: u32,
	pub start_ns: u64,
	pub duration_ns: u64,
	pub layer_id: u32,
	pub fused_layer_id: u64,
	pub dpu_ns: u64,
	pub sw_ns: u64,
	pub dma_ns: u64,
}
const _: () = assert!(std::mem::size_of::<LayerInfo>() == 368);

pub fn cstr(b: &[u8]) -> String {
	let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
	String::from_utf8_lossy(&b[..n]).into_owned()
}

impl TaskInfo {
	pub fn name(&self) -> String {
		cstr(&self.name)
	}
	pub fn layer_type(&self) -> String {
		cstr(&self.layer_type)
	}
	/// `DPU`, `SW` (ACT-SHAVE), `DMA` or `-`.
	pub fn engine(&self) -> &'static str {
		match self.exec_type {
			1 => "DPU",
			2 => "SW",
			3 => "DMA",
			_ => "-",
		}
	}
}

impl LayerInfo {
	pub fn name(&self) -> String {
		cstr(&self.name)
	}
	pub fn layer_type(&self) -> String {
		cstr(&self.layer_type)
	}
}

#[cfg(windows)]
mod sys {
	use std::ffi::c_void;
	#[link(name = "kernel32")]
	extern "system" {
		pub fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
		pub fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
		pub fn GetLastError() -> u32;
	}
	pub const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
}

macro_rules! api {
	($($field:ident = $sym:literal : fn($($a:ty),*);)*) => {
		#[allow(non_snake_case)]
		struct Api {
			$($field: unsafe extern "C" fn($($a),*) -> Res,)*
		}
		impl Api {
			#[cfg(windows)]
			fn load() -> Result<Api, String> {
				let w: Vec<u16> = "ze_loader.dll".encode_utf16().chain(Some(0)).collect();
				// SAFETY: NUL-terminated name; the module stays loaded for the process.
				let lib = unsafe { sys::LoadLibraryExW(w.as_ptr(), null_mut(), sys::LOAD_LIBRARY_SEARCH_SYSTEM32) };
				if lib.is_null() {
					return Err(format!("ze_loader.dll not found in System32 (error {})", unsafe { sys::GetLastError() }));
				}
				Ok(Api {
					$($field: {
						// SAFETY: NUL-terminated symbol name on a valid module.
						let p = unsafe { sys::GetProcAddress(lib, concat!($sym, "\0").as_ptr()) };
						if p.is_null() {
							return Err(format!("ze_loader.dll has no {}", $sym));
						}
						// SAFETY: the export has this C signature (ze_api.h).
						unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($a),*) -> Res>(p) }
					},)*
				})
			}
			#[cfg(not(windows))]
			fn load() -> Result<Api, String> {
				Err("the NPU harness is Windows only".into())
			}
		}
	};
}

api! {
	init_drivers = "zeInitDrivers": fn(*mut u32, *mut H, *mut InitDriverTypeDesc);
	driver_get_properties = "zeDriverGetProperties": fn(H, *mut DriverProperties);
	driver_get_extensions = "zeDriverGetExtensionProperties": fn(H, *mut u32, *mut ExtensionProperties);
	driver_get_extension_address = "zeDriverGetExtensionFunctionAddress": fn(H, *const c_char, *mut *mut c_void);
	device_get = "zeDeviceGet": fn(H, *mut u32, *mut H);
	device_get_properties = "zeDeviceGetProperties": fn(H, *mut DeviceProperties);
	device_get_queue_groups = "zeDeviceGetCommandQueueGroupProperties": fn(H, *mut u32, *mut QueueGroupProperties);
	context_create = "zeContextCreate": fn(H, *const ContextDesc, *mut H);
	mem_alloc_host = "zeMemAllocHost": fn(H, *const HostMemAllocDesc, usize, usize, *mut *mut c_void);
	mem_free = "zeMemFree": fn(H, *mut c_void);
	queue_create = "zeCommandQueueCreate": fn(H, H, *const CommandQueueDesc, *mut H);
	queue_execute = "zeCommandQueueExecuteCommandLists": fn(H, u32, *mut H, H);
	queue_synchronize = "zeCommandQueueSynchronize": fn(H, u64);
	list_create = "zeCommandListCreate": fn(H, H, *const CommandListDesc, *mut H);
	list_close = "zeCommandListClose": fn(H);
	list_destroy = "zeCommandListDestroy": fn(H);
	fence_create = "zeFenceCreate": fn(H, *const FenceDesc, *mut H);
	fence_sync = "zeFenceHostSynchronize": fn(H, u64);
	fence_reset = "zeFenceReset": fn(H);
	fence_destroy = "zeFenceDestroy": fn(H);
}

fn check(r: Res, what: &str) -> Result<(), String> {
	if r == SUCCESS {
		Ok(())
	} else {
		Err(format!("{what} failed: ze_result {r:#x}"))
	}
}

/// Slot `i` of a DDI table as a function of type `F`.
///
/// # Safety
/// `table` points at a DDI table with at least `i + 1` entries, and slot `i`
/// has signature `F`.
unsafe fn slot<F: Copy>(table: *const *mut c_void, i: usize, what: &str) -> Result<F, String> {
	let p = *table.add(i);
	if p.is_null() {
		return Err(format!("graph extension has no {what}"));
	}
	Ok(std::mem::transmute_copy::<*mut c_void, F>(&p))
}

type FCreate2 = unsafe extern "C" fn(H, H, *const GraphDesc2, *mut H) -> Res;
type FDestroy = unsafe extern "C" fn(H) -> Res;
type FBuildLog = unsafe extern "C" fn(H, *mut u32, *mut c_char) -> Res;
type FGetNative2 = unsafe extern "C" fn(H, *mut usize, *mut *const u8) -> Res;
type FGetProps3 = unsafe extern "C" fn(H, *mut GraphProperties3) -> Res;
type FGetArgProps3 = unsafe extern "C" fn(H, u32, *mut ArgumentProperties3) -> Res;
type FSetArg = unsafe extern "C" fn(H, u32, *const c_void) -> Res;
type FInit = unsafe extern "C" fn(H) -> Res;
type FAppendInit = unsafe extern "C" fn(H, H, H, u32, *mut H) -> Res;
type FAppendExec = unsafe extern "C" fn(H, H, H, H, u32, *mut H) -> Res;
type FDevProps2 = unsafe extern "C" fn(H, *mut DeviceGraphProperties2) -> Res;
type FPoolCreate = unsafe extern "C" fn(H, u32, *mut H) -> Res;
type FQueryCreate = unsafe extern "C" fn(H, u32, *mut H) -> Res;
type FQueryGetData = unsafe extern "C" fn(H, u32, *mut u32, *mut u8) -> Res;
type FNextCommandId = unsafe extern "C" fn(H, *const MutableFlagsDesc, *mut u64) -> Res;
type FUpdateMutable = unsafe extern "C" fn(H, *const MutableFlagsDesc) -> Res;

/// The mutable-command-list entry points (`zeCommandListGetNextCommandIdExp`,
/// `zeCommandListUpdateMutableCommandsExp`), when ze_loader.dll has them.
#[cfg(windows)]
fn mutable_api() -> Option<(FNextCommandId, FUpdateMutable)> {
	let w: Vec<u16> = "ze_loader.dll".encode_utf16().chain(Some(0)).collect();
	// SAFETY: the module is already loaded (Api::load); GetProcAddress on
	// NUL-terminated names; the exports have these C signatures (ze_api.h).
	unsafe {
		let lib = sys::LoadLibraryExW(w.as_ptr(), null_mut(), sys::LOAD_LIBRARY_SEARCH_SYSTEM32);
		if lib.is_null() {
			return None;
		}
		let a = sys::GetProcAddress(lib, c"zeCommandListGetNextCommandIdExp".as_ptr().cast());
		let b = sys::GetProcAddress(
			lib,
			c"zeCommandListUpdateMutableCommandsExp".as_ptr().cast(),
		);
		if a.is_null() || b.is_null() {
			return None;
		}
		Some((
			std::mem::transmute::<*mut c_void, FNextCommandId>(a),
			std::mem::transmute::<*mut c_void, FUpdateMutable>(b),
		))
	}
}
#[cfg(not(windows))]
fn mutable_api() -> Option<(FNextCommandId, FUpdateMutable)> {
	None
}

/// What the driver reports about itself and its graph compiler.
#[derive(Debug, Clone, Default)]
pub struct Info {
	pub driver_version: u32,
	pub device_name: String,
	pub device_id: u32,
	pub graph_ext_version: u32,
	pub compiler: (u16, u16),
	pub max_opset: u32,
	pub elf: Version3,
	pub runtime: Version3,
	pub tiles: u32,
	pub clock_mhz: u32,
}

/// An opened NPU: driver, device, context, one compute queue.
pub struct Npu {
	api: Api,
	#[allow(dead_code)]
	driver: H,
	device: H,
	context: H,
	queue: H,
	ordinal: u32,
	graph: *const *mut c_void,
	prof: *const *mut c_void,
	pub info: Info,
}

unsafe impl Send for Npu {}

impl Npu {
	pub fn open() -> Result<Npu, String> {
		let api = Api::load()?;
		// SAFETY: every call below is a documented Level Zero entry point
		// given correctly typed, initialized arguments.
		unsafe {
			let mut init = InitDriverTypeDesc {
				stype: ST_INIT_DRIVER_TYPE_DESC,
				pnext: null(),
				flags: INIT_DRIVER_TYPE_FLAG_NPU,
			};
			let mut n = 0u32;
			check(
				(api.init_drivers)(&mut n, null_mut(), &mut init),
				"zeInitDrivers",
			)?;
			let mut drivers = vec![null_mut(); n as usize];
			check(
				(api.init_drivers)(&mut n, drivers.as_mut_ptr(), &mut init),
				"zeInitDrivers",
			)?;
			let mut found = None;
			'outer: for &d in &drivers {
				let mut m = 0u32;
				if (api.device_get)(d, &mut m, null_mut()) != SUCCESS {
					continue;
				}
				let mut devs = vec![null_mut(); m as usize];
				if (api.device_get)(d, &mut m, devs.as_mut_ptr()) != SUCCESS {
					continue;
				}
				for &dev in &devs {
					let mut p: DeviceProperties = std::mem::zeroed();
					p.stype = ST_DEVICE_PROPERTIES;
					if (api.device_get_properties)(dev, &mut p) == SUCCESS
						&& p.ty == DEVICE_TYPE_VPU
					{
						found = Some((d, dev, p));
						break 'outer;
					}
				}
			}
			let (driver, device, props) = found.ok_or("no Level Zero NPU device")?;
			let mut dp = DriverProperties {
				stype: ST_DRIVER_PROPERTIES,
				pnext: null_mut(),
				uuid: [0; 16],
				driver_version: 0,
			};
			check(
				(api.driver_get_properties)(driver, &mut dp),
				"zeDriverGetProperties",
			)?;

			// Extension versions.
			let mut ne = 0u32;
			check(
				(api.driver_get_extensions)(driver, &mut ne, null_mut()),
				"zeDriverGetExtensionProperties",
			)?;
			let mut exts: Vec<ExtensionProperties> = (0..ne).map(|_| std::mem::zeroed()).collect();
			check(
				(api.driver_get_extensions)(driver, &mut ne, exts.as_mut_ptr()),
				"zeDriverGetExtensionProperties",
			)?;
			let ver = |name: &str| {
				exts.iter()
					.find(|e| cstr(&e.name) == name)
					.map(|e| e.version)
			};
			let graph_ver = ver("ZE_extension_graph").ok_or("driver has no ZE_extension_graph")?;

			if std::env::var("NPU_DEBUG").is_ok() {
				for e in &exts {
					eprintln!(
						"  ze ext {} {}.{}",
						cstr(&e.name),
						e.version >> 16,
						e.version & 0xffff
					);
				}
			}
			// Extension tables: `ZE_extension_graph_<major>_<minor>` through
			// zeDriverGetExtensionFunctionAddress (what the OpenVINO NPU
			// plugin does); older drivers only name it without the version.
			let table = |name: &str, version: u32| -> Result<*const *mut c_void, String> {
				let mut last = 0;
				for n in [
					format!("{name}_{}_{}", version >> 16, version & 0xffff),
					name.to_string(),
				] {
					let c = std::ffi::CString::new(n).unwrap();
					let mut t: *mut c_void = null_mut();
					last = (api.driver_get_extension_address)(driver, c.as_ptr(), &mut t);
					if last == SUCCESS && !t.is_null() {
						return Ok(t as *const *mut c_void);
					}
				}
				Err(format!(
					"zeDriverGetExtensionFunctionAddress({name}) failed: ze_result {last:#x}"
				))
			};
			let graph = table("ZE_extension_graph", graph_ver)?;
			let prof = match ver("ZE_extension_profiling_data") {
				Some(v) => table("ZE_extension_profiling_data", v).unwrap_or(null()),
				None => null(),
			};

			let devprops2: FDevProps2 = slot(
				graph,
				DDI_DEVICE_PROPERTIES2,
				"pfnDeviceGetGraphProperties2",
			)?;
			let mut gp: DeviceGraphProperties2 = std::mem::zeroed();
			gp.stype = ST_DEVICE_GRAPH_PROPERTIES_2;
			check(devprops2(device, &mut gp), "pfnDeviceGetGraphProperties2")?;
			if gp.graph_formats_supported & FORMAT_NGRAPH_LITE == 0 {
				return Err("NPU compiler does not accept NGRAPH_LITE (OpenVINO IR)".into());
			}

			let ctx_desc = ContextDesc {
				stype: ST_CONTEXT_DESC,
				pnext: null(),
				flags: 0,
			};
			let mut context = null_mut();
			check(
				(api.context_create)(driver, &ctx_desc, &mut context),
				"zeContextCreate",
			)?;

			let mut ng = 0u32;
			check(
				(api.device_get_queue_groups)(device, &mut ng, null_mut()),
				"zeDeviceGetCommandQueueGroupProperties",
			)?;
			let mut groups: Vec<QueueGroupProperties> = (0..ng)
				.map(|_| QueueGroupProperties {
					stype: ST_QUEUE_GROUP_PROPERTIES,
					pnext: null_mut(),
					flags: 0,
					max_memory_fill_pattern_size: 0,
					num_queues: 0,
				})
				.collect();
			check(
				(api.device_get_queue_groups)(device, &mut ng, groups.as_mut_ptr()),
				"zeDeviceGetCommandQueueGroupProperties",
			)?;
			let ordinal = groups
				.iter()
				.position(|g| g.num_queues > 0 && g.flags & QUEUE_GROUP_COMPUTE != 0)
				.ok_or("NPU exposes no compute queue")? as u32;
			// `NPU_TURBO=1`: the NPU extension's turbo option (what OpenVINO's
			// `NPU_TURBO` sets); `NPU_QPRIO=0|1|2`: normal / low / high queue
			// priority. Version 2 of the descriptor first, then version 1.
			let turbo = std::env::var("NPU_TURBO")
				.map(|v| v == "1")
				.unwrap_or(false);
			let priority: u32 = std::env::var("NPU_QPRIO")
				.ok()
				.and_then(|v| v.parse().ok())
				.unwrap_or(0);
			let ext2 = QueueDescNpuExt2 {
				stype: 2,
				pnext: null(),
				options: 1,
			};
			let ext1 = QueueDescNpuExt1 {
				stype: 1,
				pnext: null(),
				turbo: true,
			};
			let mut queue = null_mut();
			let mut made = false;
			if turbo {
				for ext in [
					&ext2 as *const _ as *const c_void,
					&ext1 as *const _ as *const c_void,
				] {
					let qd = CommandQueueDesc {
						stype: ST_COMMAND_QUEUE_DESC,
						pnext: ext,
						ordinal,
						index: 0,
						flags: 0,
						mode: 0,
						priority,
					};
					if (api.queue_create)(context, device, &qd, &mut queue) == 0 {
						eprintln!(
							"npu: turbo queue ({})",
							if ext == &ext2 as *const _ as *const c_void {
								"ext 2"
							} else {
								"ext 1"
							}
						);
						made = true;
						break;
					}
				}
				if !made {
					eprintln!("npu: turbo queue refused, plain queue");
				}
			}
			if !made {
				let qd = CommandQueueDesc {
					stype: ST_COMMAND_QUEUE_DESC,
					pnext: null(),
					ordinal,
					index: 0,
					flags: 0,
					mode: 0,
					priority,
				};
				check(
					(api.queue_create)(context, device, &qd, &mut queue),
					"zeCommandQueueCreate",
				)?;
			}

			let info = Info {
				driver_version: dp.driver_version,
				device_name: cstr(&props.name),
				device_id: props.device_id,
				graph_ext_version: graph_ver,
				compiler: (gp.compiler_major, gp.compiler_minor),
				max_opset: gp.max_ov_opset,
				elf: gp.elf,
				runtime: gp.runtime,
				tiles: props.num_slices,
				clock_mhz: props.core_clock_rate,
			};
			Ok(Npu {
				api,
				driver,
				device,
				context,
				queue,
				ordinal,
				graph,
				prof,
				info,
			})
		}
	}

	/// Compile an OpenVINO IR (XML + weights) with the driver's compiler.
	/// `flags` is the build-flag string (`--config K="V" ...`); `gflags`
	/// the `ZE_GRAPH_FLAG_*` bits.
	pub fn compile(
		&self,
		xml: &[u8],
		bin: &[u8],
		flags: &str,
		gflags: u32,
	) -> Result<Graph<'_>, String> {
		let mut buf = Vec::with_capacity(24 + xml.len() + bin.len());
		buf.extend_from_slice(&self.info.compiler.0.to_le_bytes());
		buf.extend_from_slice(&self.info.compiler.1.to_le_bytes());
		buf.extend_from_slice(&2u32.to_le_bytes());
		buf.extend_from_slice(&(xml.len() as u64).to_le_bytes());
		buf.extend_from_slice(xml);
		buf.extend_from_slice(&(bin.len() as u64).to_le_bytes());
		buf.extend_from_slice(bin);
		self.create(FORMAT_NGRAPH_LITE, &buf, flags, gflags)
	}

	/// Load a native graph blob (an earlier `Graph::native`).
	pub fn load(&self, blob: &[u8], gflags: u32) -> Result<Graph<'_>, String> {
		self.create(FORMAT_NATIVE, blob, "", gflags)
	}

	fn create(
		&self,
		format: u32,
		input: &[u8],
		flags: &str,
		gflags: u32,
	) -> Result<Graph<'_>, String> {
		let cflags = std::ffi::CString::new(flags).map_err(|e| e.to_string())?;
		// SAFETY: valid DDI table; `desc` points at live buffers for the call.
		unsafe {
			let create2: FCreate2 = slot(self.graph, DDI_CREATE2, "pfnCreate2")?;
			let desc = GraphDesc2 {
				stype: ST_GRAPH_DESC_2,
				pnext: null_mut(),
				format,
				input_size: input.len(),
				input: input.as_ptr(),
				build_flags: cflags.as_ptr(),
				flags: gflags,
			};
			let mut h = null_mut();
			let r = create2(self.context, self.device, &desc, &mut h);
			if r != SUCCESS || h.is_null() {
				let log = self.build_log(h);
				return Err(format!("pfnCreate2 failed: ze_result {r:#x}\n{log}"));
			}
			let mut g = Graph {
				npu: self,
				h,
				args: Vec::new(),
				init_stages: 0,
				initialized: false,
				profiled: gflags & GRAPH_FLAG_ENABLE_PROFILING != 0,
				list: null_mut(),
				fence: null_mut(),
				pool: null_mut(),
				query: null_mut(),
			};
			g.query_args()?;
			Ok(g)
		}
	}

	/// The option string the compiler (`kind` 1) or driver (2) accepts.
	pub fn supported_options(&self, kind: u32) -> Result<String, String> {
		type F = unsafe extern "C" fn(H, u32, *mut usize, *mut c_char) -> Res;
		// SAFETY: valid DDI table; size query then fill.
		unsafe {
			let f: F = slot(
				self.graph,
				DDI_SUPPORTED_OPTIONS,
				"pfnCompilerGetSupportedOptions",
			)?;
			let mut n = 0usize;
			check(
				f(self.device, kind, &mut n, null_mut()),
				"pfnCompilerGetSupportedOptions",
			)?;
			let mut b = vec![0u8; n + 1];
			check(
				f(self.device, kind, &mut n, b.as_mut_ptr() as *mut c_char),
				"pfnCompilerGetSupportedOptions",
			)?;
			Ok(cstr(&b))
		}
	}

	/// The compiler's log for `h` (or the last failed compile, for null).
	#[allow(clippy::not_unsafe_ptr_arg_deref)] // `h` is an opaque driver handle, passed through
	pub fn build_log(&self, h: H) -> String {
		// SAFETY: valid DDI table; size query then fill.
		unsafe {
			let Ok(f) = slot::<FBuildLog>(self.graph, DDI_BUILD_LOG, "pfnBuildLogGetString") else {
				return String::new();
			};
			let mut n = 0u32;
			if f(h, &mut n, null_mut()) != SUCCESS || n == 0 {
				return String::new();
			}
			let mut b = vec![0u8; n as usize];
			if f(h, &mut n, b.as_mut_ptr() as *mut c_char) != SUCCESS {
				return String::new();
			}
			cstr(&b)
		}
	}

	/// Host memory the NPU reads and writes directly.
	pub fn alloc(&self, size: usize) -> Result<HostBuf<'_>, String> {
		self.alloc_with(size, 4096, 0)
	}

	/// [`Npu::alloc`] at `align` with `ze_host_mem_alloc_flags`.
	pub fn alloc_with(&self, size: usize, align: usize, flags: u32) -> Result<HostBuf<'_>, String> {
		let d = HostMemAllocDesc {
			stype: ST_HOST_MEM_ALLOC_DESC,
			pnext: null(),
			flags,
		};
		let mut p = null_mut();
		// SAFETY: valid context; a power-of-two alignment.
		check(
			unsafe { (self.api.mem_alloc_host)(self.context, &d, size.max(1), align, &mut p) },
			"zeMemAllocHost",
		)?;
		Ok(HostBuf {
			npu: self,
			p: p as *mut u8,
			len: size,
		})
	}

	fn submit_once(&self, record: impl FnOnce(H) -> Result<(), String>) -> Result<(), String> {
		// SAFETY: plain Level Zero object lifecycle on handles owned here.
		unsafe {
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: null(),
				ordinal: self.ordinal,
				flags: 0,
			};
			let mut list = null_mut();
			check(
				(self.api.list_create)(self.context, self.device, &ld, &mut list),
				"zeCommandListCreate",
			)?;
			let r = record(list)
				.and_then(|_| check((self.api.list_close)(list), "zeCommandListClose"))
				.and_then(|_| {
					check(
						(self.api.queue_execute)(self.queue, 1, &mut list, null_mut()),
						"zeCommandQueueExecuteCommandLists",
					)
				})
				.and_then(|_| {
					check(
						(self.api.queue_synchronize)(self.queue, u64::MAX),
						"zeCommandQueueSynchronize",
					)
				});
			(self.api.list_destroy)(list);
			r
		}
	}
}

/// Several graph executions recorded into one command list, submitted as one.
pub struct Batch<'a> {
	npu: &'a Npu,
	list: H,
	fence: H,
	/// Mutable lists (`batch_plan_mutable`): the command id of each append
	/// and the update entry point.
	ids: Vec<u64>,
	update: Option<FUpdateMutable>,
}

impl Npu {
	/// Record `graphs` (each already initialized and bound) in order.
	pub fn batch<'a>(&'a self, graphs: &mut [&mut Graph<'a>]) -> Result<Batch<'a>, String> {
		for g in graphs.iter_mut() {
			g.initialize()?;
		}
		// SAFETY: Level Zero object lifecycle on handles owned by the batch.
		unsafe {
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: null(),
				ordinal: self.ordinal,
				flags: 0,
			};
			let mut list = null_mut();
			check(
				(self.api.list_create)(self.context, self.device, &ld, &mut list),
				"zeCommandListCreate",
			)?;
			let f: FAppendExec = slot(self.graph, DDI_APPEND_EXECUTE, "pfnAppendGraphExecute")?;
			for g in graphs.iter() {
				let r = f(list, g.h, null_mut(), null_mut(), 0, null_mut());
				if r != SUCCESS {
					(self.api.list_destroy)(list);
					return Err(format!("pfnAppendGraphExecute failed: ze_result {r:#x}"));
				}
			}
			check((self.api.list_close)(list), "zeCommandListClose")?;
			let fd = FenceDesc {
				stype: ST_FENCE_DESC,
				pnext: null(),
				flags: 0,
			};
			let mut fence = null_mut();
			check(
				(self.api.fence_create)(self.queue, &fd, &mut fence),
				"zeFenceCreate",
			)?;
			Ok(Batch {
				npu: self,
				list,
				fence,
				ids: Vec::new(),
				update: None,
			})
		}
	}
}

impl Npu {
	/// One graph appended once per argument set (`sets[k][i]` = the buffer of
	/// argument `i` in execution `k`), all in one command list: whether the
	/// driver captures the arguments at append time (one blob for many
	/// layers) or reads them at execute time (every append sees the last set).
	pub fn batch_sets<'a>(
		&'a self,
		g: &mut Graph<'a>,
		sets: &[Vec<&HostBuf<'_>>],
	) -> Result<Batch<'a>, String> {
		let plan: Vec<(usize, Vec<&HostBuf<'_>>)> = sets.iter().map(|s| (0, s.clone())).collect();
		self.batch_plan(std::slice::from_mut(g), &plan)
	}

	/// A command list of appends `(graph, buffers)`: graph `plan[k].0` with
	/// argument `i` bound to `plan[k].1[i]`, in order. The driver captures the
	/// arguments at append time (`npubench rebind`), so one graph instance can
	/// run many layers.
	pub fn batch_plan<'a>(
		&'a self,
		graphs: &mut [Graph<'a>],
		plan: &[(usize, Vec<&HostBuf<'_>>)],
	) -> Result<Batch<'a>, String> {
		for g in graphs.iter_mut() {
			g.initialize()?;
		}
		// SAFETY: as `batch`.
		unsafe {
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: null(),
				ordinal: self.ordinal,
				flags: 0,
			};
			let mut list = null_mut();
			check(
				(self.api.list_create)(self.context, self.device, &ld, &mut list),
				"zeCommandListCreate",
			)?;
			let f: FAppendExec = slot(self.graph, DDI_APPEND_EXECUTE, "pfnAppendGraphExecute")?;
			for (gi, set) in plan {
				let g = &mut graphs[*gi];
				for (i, b) in set.iter().enumerate() {
					g.bind(i, b)?;
				}
				let r = f(list, g.h, null_mut(), null_mut(), 0, null_mut());
				if r != SUCCESS {
					(self.api.list_destroy)(list);
					return Err(format!("pfnAppendGraphExecute failed: ze_result {r:#x}"));
				}
			}
			check((self.api.list_close)(list), "zeCommandListClose")?;
			let fd = FenceDesc {
				stype: ST_FENCE_DESC,
				pnext: null(),
				flags: 0,
			};
			let mut fence = null_mut();
			check(
				(self.api.fence_create)(self.queue, &fd, &mut fence),
				"zeFenceCreate",
			)?;
			Ok(Batch {
				npu: self,
				list,
				fence,
				ids: Vec::new(),
				update: None,
			})
		}
	}

	/// One graph appended `n` times into one command list, `bind(g, k)`
	/// binding its arguments before append `k` (the driver captures them at
	/// append time, as in `batch_plan`).
	pub fn batch_each<'a>(
		&'a self,
		g: &mut Graph<'a>,
		n: usize,
		bind: &mut dyn FnMut(&mut Graph<'a>, usize) -> Result<(), String>,
	) -> Result<Batch<'a>, String> {
		g.initialize()?;
		// SAFETY: as `batch`.
		unsafe {
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: null(),
				ordinal: self.ordinal,
				flags: 0,
			};
			let mut list = null_mut();
			check(
				(self.api.list_create)(self.context, self.device, &ld, &mut list),
				"zeCommandListCreate",
			)?;
			let f: FAppendExec = slot(self.graph, DDI_APPEND_EXECUTE, "pfnAppendGraphExecute")?;
			for k in 0..n {
				if let Err(e) = bind(g, k) {
					(self.api.list_destroy)(list);
					return Err(e);
				}
				let r = f(list, g.h, null_mut(), null_mut(), 0, null_mut());
				if r != SUCCESS {
					(self.api.list_destroy)(list);
					return Err(format!("pfnAppendGraphExecute failed: ze_result {r:#x}"));
				}
			}
			check((self.api.list_close)(list), "zeCommandListClose")?;
			let fd = FenceDesc {
				stype: ST_FENCE_DESC,
				pnext: null(),
				flags: 0,
			};
			let mut fence = null_mut();
			check(
				(self.api.fence_create)(self.queue, &fd, &mut fence),
				"zeFenceCreate",
			)?;
			Ok(Batch {
				npu: self,
				list,
				fence,
				ids: Vec::new(),
				update: None,
			})
		}
	}

	/// `batch_plan` on a *mutable* command list (ZE_experimental_mutable_command_list):
	/// every append's arguments can later be re-pointed with `Batch::update`
	/// without recording again. One recorded list can then run more appends
	/// than the driver's descriptor heap holds at once (the heap is charged
	/// per append, not per run): run it, update, run again.
	pub fn batch_plan_mutable<'a>(
		&'a self,
		graphs: &mut [Graph<'a>],
		plan: &[(usize, Vec<&HostBuf<'_>>)],
	) -> Result<Batch<'a>, String> {
		let (next_id, update) =
			mutable_api().ok_or("ze_loader.dll has no mutable command list entry points")?;
		for g in graphs.iter_mut() {
			g.initialize()?;
		}
		// SAFETY: as `batch`; the descriptors outlive the calls that read them.
		unsafe {
			let md = MutableFlagsDesc {
				stype: ST_MUTABLE_COMMAND_LIST_EXP_DESC,
				pnext: null(),
				flags: 0,
			};
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: &md as *const _ as *const c_void,
				ordinal: self.ordinal,
				flags: 0,
			};
			let mut list = null_mut();
			check(
				(self.api.list_create)(self.context, self.device, &ld, &mut list),
				"zeCommandListCreate (mutable)",
			)?;
			let f: FAppendExec = slot(self.graph, DDI_APPEND_EXECUTE, "pfnAppendGraphExecute")?;
			let idd = MutableFlagsDesc {
				stype: ST_MUTABLE_COMMAND_ID_EXP_DESC,
				pnext: null(),
				flags: MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS,
			};
			let mut ids = Vec::with_capacity(plan.len());
			for (gi, set) in plan {
				let g = &mut graphs[*gi];
				for (i, b) in set.iter().enumerate() {
					g.bind(i, b)?;
				}
				let mut id = 0u64;
				let r = next_id(list, &idd, &mut id);
				if r != SUCCESS {
					(self.api.list_destroy)(list);
					return Err(format!(
						"zeCommandListGetNextCommandIdExp failed: ze_result {r:#x}"
					));
				}
				ids.push(id);
				let r = f(list, g.h, null_mut(), null_mut(), 0, null_mut());
				if r != SUCCESS {
					(self.api.list_destroy)(list);
					return Err(format!("pfnAppendGraphExecute failed: ze_result {r:#x}"));
				}
			}
			check((self.api.list_close)(list), "zeCommandListClose")?;
			let fd = FenceDesc {
				stype: ST_FENCE_DESC,
				pnext: null(),
				flags: 0,
			};
			let mut fence = null_mut();
			check(
				(self.api.fence_create)(self.queue, &fd, &mut fence),
				"zeFenceCreate",
			)?;
			Ok(Batch {
				npu: self,
				list,
				fence,
				ids,
				update: Some(update),
			})
		}
	}
}

impl Batch<'_> {
	/// Submit without waiting (pair with `wait`). The queue runs submissions
	/// in order.
	#[track_caller]
	pub fn submit(&mut self) -> Result<(), String> {
		trace::rec(std::panic::Location::caller());
		let api = &self.npu.api;
		// SAFETY: closed list, fence on the same queue, not in flight.
		check(
			unsafe { (api.queue_execute)(self.npu.queue, 1, &mut self.list, self.fence) },
			"zeCommandQueueExecuteCommandLists",
		)
	}

	/// Wait for the last `submit` and reset the fence.
	pub fn wait(&mut self) -> Result<(), String> {
		let api = &self.npu.api;
		// SAFETY: fence of a submitted list.
		unsafe {
			check(
				(api.fence_sync)(self.fence, 60_000_000_000),
				"zeFenceHostSynchronize",
			)?;
			check((api.fence_reset)(self.fence), "zeFenceReset")
		}
	}

	/// Re-point arguments of a mutable batch (`batch_plan_mutable`):
	/// `(append k, argument i, buffer)`, applied in one driver call. The
	/// buffers must outlive every later run.
	pub fn update(&mut self, args: &[(usize, usize, &HostBuf<'_>)]) -> Result<(), String> {
		let a: Vec<_> = args.iter().map(|&(k, i, b)| (k, i, b, 0)).collect();
		self.update_at(&a)
	}

	/// `update` with each buffer at a byte offset `(append k, argument i,
	/// buffer, offset)`: the argument then reads / writes from `offset` on
	/// (as `Graph::bind_at`). The caller keeps the argument's bytes inside
	/// the buffer.
	pub fn update_at(
		&mut self,
		args: &[(usize, usize, &HostBuf<'_>, usize)],
	) -> Result<(), String> {
		let update = self.update.ok_or("not a mutable batch")?;
		if args.is_empty() {
			return Ok(());
		}
		// NPU_MUT_STYPE: the graph-argument struct type (0x7 = the id drivers
		// older than Level Zero 1.11 use).
		let stype = std::env::var("NPU_MUT_STYPE")
			.ok()
			.and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
			.unwrap_or(ST_MUTABLE_GRAPH_ARGUMENT_EXP_DESC);
		if std::env::var("NPU_MUT_SINGLE").is_ok() {
			for &(k, i, b, o) in args {
				if o > b.len() {
					return Err(format!("update: offset {o} past a buffer of {}", b.len()));
				}
				let g = MutableGraphArgDesc {
					stype,
					pnext: null(),
					command_id: self.ids[k],
					arg_index: i as u32,
					// SAFETY: inside the allocation (checked above)
					arg_value: unsafe { (b.ptr() as *mut u8).add(o) } as *const c_void,
				};
				let d = MutableFlagsDesc {
					stype: ST_MUTABLE_COMMANDS_EXP_DESC,
					pnext: &g as *const _ as *const c_void,
					flags: 0,
				};
				// SAFETY: as below.
				check(
					unsafe { update(self.list, &d) },
					"zeCommandListUpdateMutableCommandsExp",
				)?;
			}
			// SAFETY: the list is closed again after its updates (required).
			return check(
				unsafe { (self.npu.api.list_close)(self.list) },
				"zeCommandListClose",
			);
		}
		if let Some(&(_, _, b, o)) = args.iter().find(|a| a.3 > a.2.len()) {
			return Err(format!("update: offset {o} past a buffer of {}", b.len()));
		}
		let mut descs: Vec<MutableGraphArgDesc> = args
			.iter()
			.map(|&(k, i, b, o)| MutableGraphArgDesc {
				stype,
				pnext: null(),
				command_id: self.ids[k],
				arg_index: i as u32,
				// SAFETY: inside the allocation (checked above)
				arg_value: unsafe { (b.ptr() as *mut u8).add(o) } as *const c_void,
			})
			.collect();
		// chain them through pNext
		for j in (0..descs.len() - 1).rev() {
			let next = &descs[j + 1] as *const MutableGraphArgDesc as *const c_void;
			descs[j].pnext = next;
		}
		let d = MutableFlagsDesc {
			stype: ST_MUTABLE_COMMANDS_EXP_DESC,
			pnext: &descs[0] as *const _ as *const c_void,
			flags: 0,
		};
		// SAFETY: closed mutable list, not executing; descriptors live across
		// the call. The list must be closed again after its updates.
		check(
			unsafe { update(self.list, &d) },
			"zeCommandListUpdateMutableCommandsExp",
		)?;
		check(
			unsafe { (self.npu.api.list_close)(self.list) },
			"zeCommandListClose",
		)
	}

	/// Appends recorded (mutable batches).
	pub fn len(&self) -> usize {
		self.ids.len()
	}

	/// No appends recorded.
	pub fn is_empty(&self) -> bool {
		self.ids.is_empty()
	}
	/// Submit and wait; seconds of wall time.
	#[track_caller]
	pub fn run(&mut self) -> Result<f64, String> {
		trace::rec(std::panic::Location::caller());
		let api = &self.npu.api;
		let t = std::time::Instant::now();
		// SAFETY: closed list, fence on the same queue.
		unsafe {
			check(
				(api.queue_execute)(self.npu.queue, 1, &mut self.list, self.fence),
				"zeCommandQueueExecuteCommandLists",
			)?;
			check(
				(api.fence_sync)(self.fence, 60_000_000_000),
				"zeFenceHostSynchronize",
			)?;
			let dt = t.elapsed().as_secs_f64();
			check((api.fence_reset)(self.fence), "zeFenceReset")?;
			Ok(dt)
		}
	}
}

impl Drop for Batch<'_> {
	fn drop(&mut self) {
		// SAFETY: owned handles, destroyed once.
		unsafe {
			(self.npu.api.fence_destroy)(self.fence);
			(self.npu.api.list_destroy)(self.list);
		}
	}
}

/// Key of the weight swizzling our compiler builds graphs for (compiler
/// patch 0009, `NPU_RS_SWZKEY`; `NPU.md` §4).
pub const SWZ_KEY: usize = 2;

/// Swizzle a weight buffer in place for the NPU's CMX, as a graph compiled
/// with `NPU_RS_SWZIN` reads the inputs its `.swz` file names: the 16-byte
/// blocks of every 512-byte row `r` rotate by `(r mod 2^key) * 2^(5 -
/// key)` blocks. Large buffers are split over a few threads.
pub fn swizzle(b: &mut [u8], key: usize) -> Result<(), String> {
	let rows = 1usize << key;
	if !b.len().is_multiple_of(512 * rows) {
		return Err(format!(
			"a buffer of {} bytes cannot be swizzled with key {key}",
			b.len()
		));
	}
	let one = |c: &mut [u8], r0: usize| {
		for (ri, row) in c.chunks_exact_mut(512).enumerate() {
			let sh = ((r0 + ri) % rows) << (5 - key);
			row.rotate_right(sh * 16);
		}
	};
	// pieces of whole periods, so each piece starts at row 0 mod `rows`
	let piece = (512 * rows) << 10;
	if b.len() <= piece {
		one(b, 0);
		return Ok(());
	}
	let nt = std::thread::available_parallelism()
		.map_or(4, |n| n.get())
		.min(8);
	let per = b.len().div_ceil(nt).div_ceil(piece) * piece;
	std::thread::scope(|s| {
		for c in b.chunks_mut(per) {
			s.spawn(move || one(c, 0));
		}
	});
	Ok(())
}

pub struct HostBuf<'a> {
	npu: &'a Npu,
	p: *mut u8,
	len: usize,
}

impl HostBuf<'_> {
	pub fn as_mut_slice(&mut self) -> &mut [u8] {
		// SAFETY: `p` is a live allocation of `len` bytes owned by self.
		unsafe { std::slice::from_raw_parts_mut(self.p, self.len) }
	}
	pub fn as_slice(&self) -> &[u8] {
		// SAFETY: as above.
		unsafe { std::slice::from_raw_parts(self.p, self.len) }
	}
	pub fn ptr(&self) -> *mut c_void {
		self.p as *mut c_void
	}
	pub fn len(&self) -> usize {
		self.len
	}
	pub fn is_empty(&self) -> bool {
		self.len == 0
	}
}

impl Drop for HostBuf<'_> {
	fn drop(&mut self) {
		// SAFETY: allocated by zeMemAllocHost on this context.
		unsafe { (self.npu.api.mem_free)(self.npu.context, self.p as *mut c_void) };
	}
}

/// One graph argument (input or output) as the compiled graph sees it.
#[derive(Debug, Clone)]
pub struct Arg {
	pub name: String,
	pub is_output: bool,
	pub dims: Vec<u32>,
	/// `ZE_GRAPH_ARGUMENT_PRECISION_*` of the device side.
	pub precision: u32,
	pub bytes: usize,
}

pub fn precision_bits(p: u32) -> usize {
	match p {
		0x0F..=0x11 => 64,
		0x01 | 0x0A | 0x05 => 32,
		0x02 | 0x09 | 0x03 | 0x06 => 16,
		0x04 | 0x07 | 0x0E | 0x13 | 0x14 | 0x15 => 8,
		0x0B | 0x0C | 0x12 | 0x18 => 4,
		0x16 | 0x17 => 2,
		0x08 => 1,
		_ => 0,
	}
}

pub fn precision_name(p: u32) -> &'static str {
	match p {
		0x01 => "f32",
		0x02 => "f16",
		0x09 => "bf16",
		0x04 => "u8",
		0x07 => "i8",
		0x0B => "u4",
		0x0C => "i4",
		0x05 => "i32",
		0x11 => "i64",
		_ => "?",
	}
}

pub struct Graph<'a> {
	npu: &'a Npu,
	h: H,
	pub args: Vec<Arg>,
	init_stages: u32,
	initialized: bool,
	profiled: bool,
	list: H,
	fence: H,
	pool: H,
	query: H,
}

impl<'a> Graph<'a> {
	fn query_args(&mut self) -> Result<(), String> {
		// SAFETY: valid graph handle and DDI table.
		unsafe {
			let props3: FGetProps3 =
				slot(self.npu.graph, DDI_GET_PROPERTIES3, "pfnGetProperties3")?;
			let mut p = GraphProperties3 {
				stype: ST_GRAPH_PROPERTIES_3,
				pnext: null_mut(),
				num_graph_args: 0,
				init_stage_required: 0,
				flags: 0,
			};
			check(props3(self.h, &mut p), "pfnGetProperties3")?;
			self.init_stages = p.init_stage_required;
			let argp: FGetArgProps3 = slot(
				self.npu.graph,
				DDI_GET_ARGUMENT_PROPERTIES3,
				"pfnGetArgumentProperties3",
			)?;
			for i in 0..p.num_graph_args {
				let mut a: Box<ArgumentProperties3> = Box::new(std::mem::zeroed());
				a.stype = ST_GRAPH_ARGUMENT_PROPERTIES_3;
				check(argp(self.h, i, &mut *a), "pfnGetArgumentProperties3")?;
				let nd = (a.dims_count as usize).min(5);
				let dims: Vec<u32> = a.dims[..nd].to_vec();
				let elems: usize = dims.iter().map(|&d| d as usize).product();
				self.args.push(Arg {
					name: cstr(&a.name),
					is_output: a.ty == 1,
					dims,
					precision: a.device_precision,
					bytes: (elems * precision_bits(a.device_precision)).div_ceil(8),
				});
			}
		}
		Ok(())
	}

	/// The compiled native blob (ELF).
	pub fn native(&self) -> Result<Vec<u8>, String> {
		// SAFETY: the view stays valid while the graph lives; copied out.
		unsafe {
			let f: FGetNative2 = slot(self.npu.graph, DDI_GET_NATIVE2, "pfnGetNativeBinary2")?;
			let mut n = 0usize;
			let mut p: *const u8 = null();
			check(f(self.h, &mut n, &mut p), "pfnGetNativeBinary2")?;
			Ok(std::slice::from_raw_parts(p, n).to_vec())
		}
	}

	/// Bind argument `i` to `buf` (must stay alive while the graph runs).
	pub fn bind(&mut self, i: usize, buf: &HostBuf<'_>) -> Result<(), String> {
		if buf.len() < self.args[i].bytes {
			return Err(format!(
				"argument {} ({}) needs {} bytes, buffer has {}",
				i,
				self.args[i].name,
				self.args[i].bytes,
				buf.len()
			));
		}
		// SAFETY: valid graph; the pointer is a live NPU-visible allocation.
		unsafe {
			let f: FSetArg = slot(self.npu.graph, DDI_SET_ARGUMENT, "pfnSetArgumentValue")?;
			check(f(self.h, i as u32, buf.ptr()), "pfnSetArgumentValue")
		}
	}

	/// Bind argument `i` to `buf` from byte `off` on (a part of a larger
	/// allocation, e.g. one expert of a layer's expert pool).
	pub fn bind_at(&mut self, i: usize, buf: &HostBuf<'_>, off: usize) -> Result<(), String> {
		if off + self.args[i].bytes > buf.len() {
			return Err(format!(
				"argument {} ({}) needs {} bytes at {off}, buffer has {}",
				i,
				self.args[i].name,
				self.args[i].bytes,
				buf.len()
			));
		}
		// SAFETY: valid graph; the pointer lies inside a live NPU-visible
		// allocation with the argument's bytes after it.
		unsafe {
			let f: FSetArg = slot(self.npu.graph, DDI_SET_ARGUMENT, "pfnSetArgumentValue")?;
			check(
				f(
					self.h,
					i as u32,
					(buf.ptr() as *mut u8).add(off) as *mut c_void,
				),
				"pfnSetArgumentValue",
			)
		}
	}

	/// Run the initialization stages once (weights copied in, etc.).
	pub fn initialize(&mut self) -> Result<(), String> {
		if self.initialized {
			return Ok(());
		}
		// SAFETY: valid graph and DDI table.
		unsafe {
			if self.init_stages & STAGE_INITIALIZE != 0 {
				let f: FInit = slot(self.npu.graph, DDI_INITIALIZE, "pfnGraphInitialize")?;
				check(f(self.h), "pfnGraphInitialize")?;
			}
			if self.init_stages & STAGE_COMMAND_LIST_INITIALIZE != 0 {
				let f: FAppendInit = slot(
					self.npu.graph,
					DDI_APPEND_INITIALIZE,
					"pfnAppendGraphInitialize",
				)?;
				let h = self.h;
				self.npu.submit_once(|list| {
					check(
						f(list, h, null_mut(), 0, null_mut()),
						"pfnAppendGraphInitialize",
					)
				})?;
			}
		}
		self.initialized = true;
		Ok(())
	}

	/// Record the execute command list once (optionally with a profiling
	/// query), so `run` only submits and waits.
	pub fn prepare(&mut self, profile: bool) -> Result<(), String> {
		// A graph compiled for profiling must be executed with a query.
		let profile = profile || self.profiled;
		self.initialize()?;
		let api = &self.npu.api;
		// SAFETY: Level Zero object lifecycle on handles owned by self.
		unsafe {
			if profile && self.query.is_null() {
				if self.npu.prof.is_null() {
					return Err("driver has no ZE_extension_profiling_data".into());
				}
				let pc: FPoolCreate =
					slot(self.npu.prof, PROF_POOL_CREATE, "pfnProfilingPoolCreate")?;
				check(pc(self.h, 1, &mut self.pool), "pfnProfilingPoolCreate")?;
				let qc: FQueryCreate =
					slot(self.npu.prof, PROF_QUERY_CREATE, "pfnProfilingQueryCreate")?;
				check(qc(self.pool, 0, &mut self.query), "pfnProfilingQueryCreate")?;
			}
			if !self.list.is_null() {
				(api.list_destroy)(self.list);
				self.list = null_mut();
			}
			let ld = CommandListDesc {
				stype: ST_COMMAND_LIST_DESC,
				pnext: null(),
				ordinal: self.npu.ordinal,
				flags: 0,
			};
			check(
				(api.list_create)(self.npu.context, self.npu.device, &ld, &mut self.list),
				"zeCommandListCreate",
			)?;
			let f: FAppendExec = slot(self.npu.graph, DDI_APPEND_EXECUTE, "pfnAppendGraphExecute")?;
			let q = if profile { self.query } else { null_mut() };
			check(
				f(self.list, self.h, q, null_mut(), 0, null_mut()),
				"pfnAppendGraphExecute",
			)?;
			check((api.list_close)(self.list), "zeCommandListClose")?;
			if self.fence.is_null() {
				let fd = FenceDesc {
					stype: ST_FENCE_DESC,
					pnext: null(),
					flags: 0,
				};
				check(
					(api.fence_create)(self.npu.queue, &fd, &mut self.fence),
					"zeFenceCreate",
				)?;
			}
		}
		Ok(())
	}

	/// Submit the recorded list and wait; returns the wall time in seconds.
	#[track_caller]
	pub fn run(&mut self) -> Result<f64, String> {
		let at = std::panic::Location::caller();
		if self.list.is_null() {
			self.prepare(false)?;
		}
		trace::rec(at);
		let api = &self.npu.api;
		let t = std::time::Instant::now();
		// SAFETY: recorded, closed list; fence on the same queue.
		unsafe {
			check(
				(api.queue_execute)(self.npu.queue, 1, &mut self.list, self.fence),
				"zeCommandQueueExecuteCommandLists",
			)?;
			check(
				(api.fence_sync)(self.fence, 60_000_000_000),
				"zeFenceHostSynchronize",
			)?;
			let dt = t.elapsed().as_secs_f64();
			check((api.fence_reset)(self.fence), "zeFenceReset")?;
			Ok(dt)
		}
	}

	/// Drop the recorded execute list (`run` records it again): the driver
	/// charges its descriptor heap per recorded append, and a process with
	/// many graphs runs out of it (`0x70000002` from the next append).
	pub fn release(&mut self) {
		if !self.list.is_null() {
			// SAFETY: an owned list, not executing (every run is waited for).
			unsafe { (self.npu.api.list_destroy)(self.list) };
			self.list = null_mut();
		}
	}

	fn prof_data<T: Copy>(&self, kind: u32) -> Result<Vec<T>, String> {
		if self.query.is_null() {
			return Err("graph was not prepared with profiling".into());
		}
		// SAFETY: valid query; size query then fill, records are POD.
		unsafe {
			let f: FQueryGetData = slot(
				self.npu.prof,
				PROF_QUERY_GET_DATA,
				"pfnProfilingQueryGetData",
			)?;
			let mut n = 0u32;
			check(
				f(self.query, kind, &mut n, null_mut()),
				"pfnProfilingQueryGetData(size)",
			)?;
			let k = n as usize / std::mem::size_of::<T>();
			let mut v: Vec<T> = Vec::with_capacity(k);
			check(
				f(self.query, kind, &mut n, v.as_mut_ptr() as *mut u8),
				"pfnProfilingQueryGetData",
			)?;
			v.set_len(k);
			Ok(v)
		}
	}

	/// Per-task timings of the last profiled run.
	pub fn tasks(&self) -> Result<Vec<TaskInfo>, String> {
		self.prof_data(PROFILING_TASK_LEVEL)
	}

	/// Per-layer timings of the last profiled run.
	pub fn layers(&self) -> Result<Vec<LayerInfo>, String> {
		self.prof_data(PROFILING_LAYER_LEVEL)
	}
}

impl Drop for Graph<'_> {
	fn drop(&mut self) {
		let api = &self.npu.api;
		// SAFETY: handles owned by self, destroyed once, children first.
		unsafe {
			if !self.fence.is_null() {
				(api.fence_destroy)(self.fence);
			}
			if !self.list.is_null() {
				(api.list_destroy)(self.list);
			}
			if !self.query.is_null() {
				if let Ok(f) = slot::<FDestroy>(self.npu.prof, PROF_QUERY_DESTROY, "") {
					f(self.query);
				}
			}
			if !self.pool.is_null() {
				if let Ok(f) = slot::<FDestroy>(self.npu.prof, PROF_POOL_DESTROY, "") {
					f(self.pool);
				}
			}
			if let Ok(f) = slot::<FDestroy>(self.npu.graph, DDI_DESTROY, "") {
				f(self.h);
			}
		}
	}
}
