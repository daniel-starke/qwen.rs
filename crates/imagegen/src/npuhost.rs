//! The process's NPU and the f16 host-buffer helpers the image models'
//! NPU backends share.

use npu::ze::Npu;

/// The process's NPU (opened once; the device is shared by every model of
/// the process).
pub fn npu() -> Result<&'static Npu, String> {
	// (the address: `Npu` holds raw handles; the driver's are thread safe)
	static N: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
	let a = N
		.get_or_init(|| Npu::open().map(|n| Box::leak(Box::new(n)) as *const Npu as usize))
		.clone()?;
	// SAFETY: leaked above, never freed
	Ok(unsafe { &*(a as *const Npu) })
}

/// f32 -> f16 bytes.
pub fn f16_row(dst: &mut [u8], src: &[f32]) {
	for (d, &v) in dst.chunks_exact_mut(2).zip(src) {
		d.copy_from_slice(&quant::f32_to_fp16(v).to_le_bytes());
	}
}

/// An NPU host buffer's bytes as f16 bits.
pub fn as_u16(b: &[u8]) -> &[u16] {
	// SAFETY: host buffers are 4 KiB aligned; any bit pattern is a u16.
	unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
}

/// f16 bits -> f32, on the pool.
pub fn widen(src: &[u16], dst: &mut [f32]) {
	let p = crate::ops::Ptr(dst.as_mut_ptr());
	let n = src.len();
	const C: usize = 1 << 16;
	quant::par_for(n.div_ceil(C), move |i| {
		let p = p;
		let (a, b) = (i * C, ((i + 1) * C).min(n));
		// SAFETY: task i owns [a, b)
		let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(a), b - a) };
		for (o, &h) in d.iter_mut().zip(&src[a..b]) {
			*o = quant::fp16_to_f32(h);
		}
	});
}

/// f32 -> f16 bytes, on the pool.
pub fn narrow(src: &[f32], dst: &mut [u8]) {
	let n = src.len();
	let p = dst.as_mut_ptr() as usize;
	const C: usize = 1 << 16;
	quant::par_for(n.div_ceil(C), move |i| {
		let (a, b) = (i * C, ((i + 1) * C).min(n));
		// SAFETY: task i owns bytes [2a, 2b)
		let d = unsafe { std::slice::from_raw_parts_mut((p as *mut u8).add(2 * a), 2 * (b - a)) };
		f16_row(d, &src[a..b]);
	});
}
