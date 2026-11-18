//! The SHAVE toolchain end to end on the kernels of `crates/npu/shave`:
//! kernel language -> assembly -> object -> linked ELF. The object must read
//! back as written, and the files must stay what they are: each one's size
//! and FNV-1a hash is recorded from our own build. A change of
//! the compiler, the assembler or the linker that moves a byte shows up
//! here; whether the new bytes are right is the device's to say
//! (`scripts/sktest.py`).

use npu::shave::{asm, dsl, elf::Object, link};

/// (kernel, defines, object bytes, object hash, ELF bytes, ELF hash)
type Build = (
	&'static str,
	&'static [(&'static str, i64)],
	usize,
	u64,
	usize,
	u64,
);

const KERNELS: &[Build] = &[
	(
		"softmax_lse.sk",
		&[("N", 1024)],
		1734,
		0xf091_3ba4_19a5_e981,
		1868,
		0x7237_f1f8_1fb6_0f43,
	),
	(
		"softmax_lse.sk",
		&[("N", 1024), ("NONORM", 1)],
		1522,
		0xdcf4_c974_527e_0799,
		1676,
		0x4efc_3b8f_f232_3760,
	),
	(
		"softmax_lse.sk",
		&[("N", 1024), ("KL", 4)],
		1734,
		0xda32_49cc_42c3_9d01,
		1868,
		0xf285_7029_0803_d443,
	),
	(
		"gnorm.sk",
		&[],
		2614,
		0xea52_2441_d931_14b5,
		2700,
		0xffef_46ff_6bbc_33b8,
	),
	(
		"trinv.sk",
		&[("BS", 16)],
		2081,
		0x0368_4e8f_6624_3a9c,
		2060,
		0xd31f_ea65_69e0_fcdc,
	),
	(
		"softmax_q8.sk",
		&[("N", 1024)],
		2141,
		0xb4f0_5802_e55d_f95b,
		2300,
		0x0dfc_6bbb_e415_0625,
	),
	(
		"hcmix.sk",
		&[],
		7532,
		0x8c0a_228c_8b15_58c7,
		7180,
		0xf805_69d6_31f8_26fd,
	),
];

fn fnv(bytes: &[u8]) -> u64 {
	bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
		(h ^ b as u64).wrapping_mul(0x100000001b3)
	})
}

fn source(name: &str) -> String {
	let path = format!("{}/shave/{name}", env!("CARGO_MANIFEST_DIR"));
	std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn build(name: &str, defs: &[(&str, i64)]) -> (Vec<u8>, Vec<u8>) {
	let defs: Vec<(String, i64)> = defs.iter().map(|&(k, v)| (k.to_string(), v)).collect();
	let k = dsl::compile(&source(name), &defs).unwrap_or_else(|e| panic!("{name}: {e}"));
	let obj = asm::assemble(&k.text()).unwrap_or_else(|e| panic!("{name}: {e}"));
	let file = obj.write();
	// through the file, as `shave as` + `shave ld` go
	let read = Object::read(&file).unwrap();
	assert_eq!(read, obj, "{name}: object does not read back");
	let elf = link::link(&[read], "controlled_act").unwrap();
	(file, elf)
}

#[test]
fn kernels_are_stable() {
	let mut got = Vec::new();
	for &(name, defs, ..) in KERNELS_TO_BUILD {
		let (obj, elf) = build(name, defs);
		got.push((name, defs, obj.len(), fnv(&obj), elf.len(), fnv(&elf)));
	}
	assert_eq!(got, KERNELS, "{got:#x?}");
}

/// The kernels and defines `kernels_are_stable` builds (the shipped ones).
const KERNELS_TO_BUILD: &[(&str, &[(&str, i64)])] = &[
	("softmax_lse.sk", &[("N", 1024)]),
	("softmax_lse.sk", &[("N", 1024), ("NONORM", 1)]),
	("softmax_lse.sk", &[("N", 1024), ("KL", 4)]),
	("gnorm.sk", &[]),
	("trinv.sk", &[("BS", 16)]),
	("softmax_q8.sk", &[("N", 1024)]),
	("hcmix.sk", &[]),
];

#[test]
fn kernel_shape() {
	let (_, elf) = build("gnorm.sk", &[]);
	// entry = start of .text, where the compiler expects the kernel
	assert_eq!(elf[24..28], link::TEXT_ADDR.to_le_bytes());
	assert!(elf
		.windows(link::COMMENT.len())
		.any(|w| w == link::COMMENT.as_bytes()));
}
