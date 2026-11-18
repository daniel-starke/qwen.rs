//! Post recorded requests (`cli ... --serve --record-requests DIR`) to a
//! running server again, one after the other, without the original gaps:
//!
//! `cargo run --release -p server --example replay -- DIR [HOST:PORT] [--only N,M,..] [--rounds R] [--out OUT]`
//!
//! `--out OUT`: each response as received (headers and body) to
//! `OUT/NAME.resp` (the last round's), to compare two servers' text.
//!
//! Per request: the time to the response's first content
//! (an SSE `content_block_delta` / OpenAI `delta` event; for a whole response
//! the end), the total, and the response size. The server's own
//! `QWEN_NPU_TIMING=1` request timeline splits the time further.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let mut dir = None;
	let mut addr = "127.0.0.1:8080".to_string();
	let mut only: Option<Vec<usize>> = None;
	let mut rounds = 1usize;
	let mut out: Option<String> = None;
	let mut i = 0;
	while i < args.len() {
		match args[i].as_str() {
			"--only" => {
				i += 1;
				only = Some(args[i].split(',').filter_map(|v| v.parse().ok()).collect());
			}
			"--rounds" => {
				i += 1;
				rounds = args[i].parse().expect("--rounds N");
			}
			"--out" => {
				i += 1;
				out = Some(args[i].clone());
			}
			a if dir.is_none() => dir = Some(a.to_string()),
			a => addr = a.to_string(),
		}
		i += 1;
	}
	let Some(dir) = dir else {
		eprintln!("usage: replay DIR [HOST:PORT] [--only N,M,..] [--rounds R] [--out OUT]");
		std::process::exit(2);
	};
	let mut files: Vec<_> = std::fs::read_dir(&dir)
		.expect("read DIR")
		.filter_map(|e| e.ok().map(|e| e.path()))
		.filter(|p| p.extension().is_some_and(|x| x == "req"))
		.collect();
	files.sort();
	let (mut sum_first, mut sum_total) = (0f64, 0f64);
	for round in 0..rounds {
		for (n, p) in files.iter().enumerate() {
			if only.as_ref().is_some_and(|o| !o.contains(&n)) {
				continue;
			}
			let raw = std::fs::read(p).expect("read request");
			let (first, total, got) = post(&addr, &raw);
			let bytes = got.len();
			if let Some(o) = &out {
				let name = p.file_stem().unwrap().to_string_lossy();
				std::fs::create_dir_all(o).expect("create OUT");
				std::fs::write(std::path::Path::new(o).join(format!("{name}.resp")), &got)
					.expect("write a response");
			}
			sum_first += first;
			sum_total += total;
			println!(
				"round {round} #{n} {}: first content {first:.0} ms, total {total:.0} ms, {bytes} bytes",
				p.file_name().unwrap().to_string_lossy()
			);
		}
	}
	println!("sum: first content {sum_first:.0} ms, total {sum_total:.0} ms");
}

/// Post one recorded request; (ms to first content, ms in all, the
/// response).
fn post(addr: &str, raw: &[u8]) -> (f64, f64, Vec<u8>) {
	let split = raw
		.windows(2)
		.position(|w| w == b"\n\n")
		.expect("recorded request without a blank line");
	let head = std::str::from_utf8(&raw[..split]).expect("header text");
	let body = &raw[split + 2..];
	let mut lines = head.lines();
	let first = lines.next().unwrap_or("POST /v1/messages");
	let mut req = format!("{first} HTTP/1.1\r\n");
	for l in lines {
		let k = l.split(':').next().unwrap_or("");
		if matches!(k, "content-length" | "connection" | "host") {
			continue;
		}
		req.push_str(l);
		req.push_str("\r\n");
	}
	req.push_str(&format!(
		"host: {addr}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
		body.len()
	));
	let t = Instant::now();
	let mut s = TcpStream::connect(addr).expect("connect to the server");
	let _ = s.set_nodelay(true);
	s.write_all(req.as_bytes()).expect("send");
	s.write_all(body).expect("send");
	let mut got = Vec::new();
	let mut first_content = None;
	let mut buf = vec![0u8; 1 << 16];
	loop {
		let k = match s.read(&mut buf) {
			Ok(0) | Err(_) => break,
			Ok(k) => k,
		};
		got.extend_from_slice(&buf[..k]);
		if first_content.is_none() {
			let tail = &got[got.len().saturating_sub(k + 32)..];
			if find(tail, b"content_block_delta") || find(tail, b"\"delta\"") {
				first_content = Some(t.elapsed().as_secs_f64() * 1e3);
			}
		}
	}
	let total = t.elapsed().as_secs_f64() * 1e3;
	(first_content.unwrap_or(total), total, got)
}

fn find(h: &[u8], n: &[u8]) -> bool {
	h.windows(n.len()).any(|w| w == n)
}
