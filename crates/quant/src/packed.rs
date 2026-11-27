//! Embedded files stored deflated: a crate's `build.rs` writes
//! `OUT_DIR/<name>.z` (`miniz_oxide`, level 10) and [`packed!`] /
//! [`packed_str!`] embed that and inflate it on first use, once per call
//! site. Large tables, texts and SPIR-V shrink the binary 3-5x this way.
//! `miniz_oxide` is linked anyway (PNG through `image`).

/// Inflate a zlib-less deflate stream written by a `build.rs`.
pub fn inflate(z: &[u8]) -> Vec<u8> {
	miniz_oxide::inflate::decompress_to_vec(z).expect("embedded deflate stream")
}

/// `&'static [u8]` of `OUT_DIR/<name>.z` inflated, the calling crate's
/// `OUT_DIR`; a drop-in for `include_bytes!`.
#[macro_export]
macro_rules! packed {
	($name:literal) => {{
		static DATA: ::std::sync::OnceLock<::std::vec::Vec<u8>> = ::std::sync::OnceLock::new();
		DATA.get_or_init(|| {
			$crate::packed::inflate(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".z")))
		})
		.as_slice()
	}};
}

/// `&'static str` of `OUT_DIR/<name>.z` inflated; a drop-in for
/// `include_str!`.
#[macro_export]
macro_rules! packed_str {
	($name:literal) => {{
		static TEXT: ::std::sync::OnceLock<::std::string::String> = ::std::sync::OnceLock::new();
		TEXT.get_or_init(|| {
			::std::string::String::from_utf8($crate::packed::inflate(include_bytes!(concat!(
				env!("OUT_DIR"),
				"/",
				$name,
				".z"
			))))
			.expect("embedded text is UTF-8")
		})
		.as_str()
	}};
}
