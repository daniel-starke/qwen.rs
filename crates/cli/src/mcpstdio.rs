//! `cli --mcp-stdio [URL]`: an MCP stdio transport in front of the server's
//! Streamable HTTP endpoint (`URL/mcp`), for clients that only start
//! commands. Each line on stdin is one JSON-RPC message, POSTed as is; a
//! JSON answer is written to stdout as one line, an SSE answer as one line
//! per event as the events arrive (progress notifications, the server's
//! `roots/list` request, then the result). The client's answers to the
//! server's requests are POSTed like any other message. No model in this
//! process.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

struct Target {
	addr: String,
	host: String,
	path: String,
}

fn target(url: &str) -> Result<Target, String> {
	let rest = url
		.strip_prefix("http://")
		.ok_or_else(|| format!("{url}: only http:// URLs"))?;
	let (hostport, base) = rest.split_once('/').unwrap_or((rest, ""));
	let base = base.trim_end_matches('/');
	let path = if base.ends_with("mcp") {
		format!("/{base}")
	} else if base.is_empty() {
		"/mcp".to_string()
	} else {
		format!("/{base}/mcp")
	};
	let addr = if hostport.contains(':') {
		hostport.to_string()
	} else {
		format!("{hostport}:80")
	};
	Ok(Target {
		addr,
		host: hostport.to_string(),
		path,
	})
}

/// POST one message; every JSON-RPC message of the answer goes to `out`.
/// Returns the `Mcp-Session-Id` the answer carried.
fn post(
	t: &Target,
	session: Option<&str>,
	body: &str,
	out: &dyn Fn(&str),
) -> Result<Option<String>, String> {
	let mut s = TcpStream::connect(&t.addr).map_err(|e| format!("{}: {e}", t.addr))?;
	let sid = session.map_or(String::new(), |s| format!("Mcp-Session-Id: {s}\r\n"));
	let req = format!(
		"POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nMCP-Protocol-Version: 2025-06-18\r\n{sid}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
		t.path,
		t.host,
		body.len()
	);
	s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
	let mut r = BufReader::new(s);
	let mut status = String::new();
	r.read_line(&mut status).map_err(|e| e.to_string())?;
	let code: u16 = status
		.split_whitespace()
		.nth(1)
		.and_then(|c| c.parse().ok())
		.unwrap_or(0);
	let (mut len, mut chunked, mut sse, mut new_sid) = (None, false, false, None);
	loop {
		let mut l = String::new();
		if r.read_line(&mut l).map_err(|e| e.to_string())? == 0 {
			break;
		}
		let l = l.trim_end();
		if l.is_empty() {
			break;
		}
		if let Some((k, v)) = l.split_once(':') {
			let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
			match k.as_str() {
				"content-length" => len = v.parse::<usize>().ok(),
				"transfer-encoding" => chunked = v.eq_ignore_ascii_case("chunked"),
				"content-type" => sse = v.starts_with("text/event-stream"),
				"mcp-session-id" => new_sid = Some(v.to_string()),
				_ => {}
			}
		}
	}
	if code == 202 {
		return Ok(new_sid);
	}
	// the body as a byte stream (chunked or not)
	let mut data = Vec::new();
	let feed = |bytes: &[u8], data: &mut Vec<u8>| {
		data.extend_from_slice(bytes);
		if !sse {
			return;
		}
		// emit every complete event
		while let Some(p) = data.windows(2).position(|w| w == b"\n\n") {
			let ev: Vec<u8> = data.drain(..p + 2).collect();
			let ev = String::from_utf8_lossy(&ev);
			let payload: Vec<&str> = ev
				.lines()
				.filter_map(|l| l.strip_prefix("data:"))
				.map(str::trim_start)
				.collect();
			if !payload.is_empty() {
				out(&payload.join(""));
			}
		}
	};
	if chunked {
		loop {
			let mut l = String::new();
			if r.read_line(&mut l).map_err(|e| e.to_string())? == 0 {
				break;
			}
			let n = usize::from_str_radix(l.trim(), 16).map_err(|_| format!("bad chunk {l:?}"))?;
			if n == 0 {
				break;
			}
			let mut b = vec![0u8; n + 2];
			r.read_exact(&mut b).map_err(|e| e.to_string())?;
			feed(&b[..n], &mut data);
		}
	} else {
		let mut b = Vec::new();
		match len {
			Some(n) => {
				b.resize(n, 0);
				r.read_exact(&mut b).map_err(|e| e.to_string())?;
			}
			None => {
				r.read_to_end(&mut b).map_err(|e| e.to_string())?;
			}
		}
		feed(&b, &mut data);
	}
	if !sse {
		let s = String::from_utf8_lossy(&data);
		let s = s.trim();
		if s.starts_with('{') {
			out(s);
		} else if code >= 400 {
			return Err(format!("HTTP {code}: {s}"));
		}
	}
	Ok(new_sid)
}

/// Run the bridge until stdin closes; the exit code.
pub fn main(url: &str) -> i32 {
	let t = match target(url) {
		Ok(t) => Arc::new(t),
		Err(e) => {
			eprintln!("error: {e}");
			return 2;
		}
	};
	let stdout = Arc::new(Mutex::new(std::io::stdout()));
	let session: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
	let emit = {
		let stdout = stdout.clone();
		move |s: &str| {
			let mut o = stdout.lock().unwrap_or_else(|e| e.into_inner());
			let _ = writeln!(o, "{}", s.replace('\n', " "));
			let _ = o.flush();
		}
	};
	let emit = Arc::new(emit);
	let mut workers = Vec::new();
	for line in std::io::stdin().lock().lines() {
		let Ok(line) = line else { break };
		let line = line.trim().to_string();
		if line.is_empty() {
			continue;
		}
		let init = line.contains("\"initialize\"");
		let (t, session, emit) = (t.clone(), session.clone(), emit.clone());
		let job = move || {
			let sid = session.lock().unwrap_or_else(|e| e.into_inner()).clone();
			match post(&t, sid.as_deref(), &line, &*emit) {
				Ok(Some(s)) => *session.lock().unwrap_or_else(|e| e.into_inner()) = Some(s),
				Ok(None) => {}
				Err(e) => eprintln!("mcp-stdio: {e}"),
			}
		};
		if init {
			// the session id must exist before anything else is sent
			job();
		} else {
			workers.push(std::thread::spawn(job));
		}
	}
	for w in workers {
		let _ = w.join();
	}
	0
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn urls() {
		let t = target("http://127.0.0.1:8080").unwrap();
		assert_eq!(
			(t.addr.as_str(), t.path.as_str()),
			("127.0.0.1:8080", "/mcp")
		);
		assert_eq!(target("http://h:1/mcp").unwrap().path, "/mcp");
		assert_eq!(target("http://h/x/").unwrap().path, "/x/mcp");
		assert!(target("https://h").is_err());
	}
}
