//! Default on (`--no-keep-awake` / `QWEN_KEEP_AWAKE=0` turn it off): hold
//! the display and the system on for the life of the process. With the display in standby the iGPU (the NPU
//! engine's scan device) stalls for 0.4-0.5 s at random points of a prefill
//! (81-88 instead of 127 tok/s); Windows has no
//! per-device power control for applications, but an execution-state request
//! keeps the display, and with it the iGPU, out of that state.

/// Request the display and the system stay on until the process exits
/// (`ES_CONTINUOUS` on the main thread, which lives as long as the process).
/// Returns false where the request is not available or was refused.
#[cfg(windows)]
pub fn keep_awake() -> bool {
	#[link(name = "kernel32")]
	extern "system" {
		fn SetThreadExecutionState(flags: u32) -> u32;
	}
	const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
	const ES_DISPLAY_REQUIRED: u32 = 0x0000_0002;
	const ES_CONTINUOUS: u32 = 0x8000_0000;
	// SAFETY: plain flags, no pointers
	unsafe {
		SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) != 0
	}
}

#[cfg(not(windows))]
pub fn keep_awake() -> bool {
	false
}

/// False when `QWEN_KEEP_AWAKE` is `0` / `off` / `no` / `false`.
pub fn from_env() -> bool {
	!std::env::var("QWEN_KEEP_AWAKE").is_ok_and(|v| {
		matches!(
			v.trim().to_ascii_lowercase().as_str(),
			"0" | "off" | "no" | "false"
		)
	})
}
