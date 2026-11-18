//! State carried across `forward_step` calls so generation is O(n) in the
//! number of tokens produced, instead of re-running the whole growing prefix
//! on every step.

/// One attention layer's K and V cache, `[pos][n_kv * head_dim]` each, as
/// IEEE half bits: decode at long context streams the whole cache once per
/// token per layer, so half the bytes is half that time (and half the memory
/// a prefix-cache entry or snapshot holds). llama.cpp's default cache type is
/// f16 too. Rows are written with [`quant::kv_extend`].
pub type KvCache = (Vec<u16>, Vec<u16>);

/// State carried across `forward_step` calls so generation is O(n) in the
/// number of tokens produced, instead of re-running the whole growing prefix
/// on every step.
///
/// - `kv`     : per-layer attention key/value cache (full-attention layers).
/// - `state`  : per-layer gated-delta-net recurrent state `[S*S*Hv]`.
/// - `convbuf`: per-layer depthwise-conv rolling buffer `[conv_kernel-1][conv_dim]`.
/// - `pos`    : absolute position of the next token to be fed.
/// - `mpos`   : its **MRoPE** coordinate triple `(t, h, w)`, which is *not*
///   the same thing. For text every axis equals `pos`, but an image occupies
///   one `pos` per merged patch while its `h`/`w` axes range over the patch
///   grid, so after an image the two counters diverge permanently. `pos`
///   indexes the KV cache; `mpos` only ever feeds RoPE.
/// - `mrope`  : set once a non-degenerate `mpos` has been used, i.e. an image
///   is in the context.
pub struct RunState {
	pub kv: Vec<Option<KvCache>>,
	pub state: Vec<Vec<f32>>,
	pub convbuf: Vec<Vec<Vec<f32>>>,
	pub pos: usize,
	pub mpos: [i64; 3],
	pub mrope: bool,
	/// The MTP draft head's state, when the model was loaded with MTP enabled
	/// (`Model::enable_mtp`). `None` costs nothing anywhere.
	pub mtp: Option<MtpRun>,
	/// The DFlash2 drafter's context (`Model::enable_dflash`). Best effort:
	/// losing it only lowers acceptance, so snapshots on disk leave it out.
	pub dflash: Option<crate::DflashRun>,
}

/// One position waiting for its row in the MTP head's KV cache: the MTP block
/// at position `p` reads the embedding of token `x_p` and the main model's
/// final (output-normed) hidden state at `p - 1`.
#[derive(Clone)]
pub(crate) struct MtpEntry {
	pub(crate) emb: Vec<f32>,
	pub(crate) h: Vec<f32>,
	pub(crate) mpos: [i64; 3],
}

/// Per-run state of the MTP (next-token-prediction) head, `blk.{n_layer_main}`.
///
/// Invariant, whenever the main model is not mid-verify:
/// `kv rows + pending.len() == RunState::pos`. Rows are computed lazily: the
/// positions a verify pass accepts are queued in `pending` and folded into the
/// next draft's MTP batch, so the head's weights are read once per step.
pub struct MtpRun {
	pub(crate) kv: Option<KvCache>,
	/// The main model's output-normed hidden state at `pos - 1` (zeros at 0).
	pub(crate) last_h: Vec<f32>,
	pub(crate) pending: Vec<MtpEntry>,
	/// Rollback scratch for a verify pass, reused across steps: SSM state and
	/// conv window of every layer after each of the batch's first tokens.
	pub(crate) snap: SpecSnap,
}

/// See [`MtpRun::snap`]. `conv[t][il]` holds layer `il`'s conv window after
/// verify token `t`. The recurrent state is not snapshotted per token (that
/// was a full ~150 MB copy per verified token): `base[il]` is the state
/// *before* the batch and `inp[il]` the batch's per-token scan inputs, so a
/// rollback replays just the accepted tokens (`Model::ssm_replay`).
/// Full-attention layers are left empty (their KV cache rolls back by
/// truncation).
#[derive(Default)]
pub(crate) struct SpecSnap {
	pub(crate) conv: Vec<Vec<Vec<Vec<f32>>>>,
	pub(crate) base: Vec<Vec<f32>>,
	pub(crate) inp: Vec<Option<SsmReplay>>,
	/// `QWEN_SPEC_CHECK=1` only: `[il][t]` the state after verify token `t`,
	/// copied the old (expensive) way, to check the replay against.
	pub(crate) check: Vec<Vec<Vec<f32>>>,
}

/// `QWEN_SPEC_CHECK=1`: verify that every rollback replay reproduces the
/// state the live scan had at that token, bit for bit.
pub(crate) fn spec_check() -> bool {
	static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*C.get_or_init(|| std::env::var_os("QWEN_SPEC_CHECK").is_some())
}

/// One SSM layer's scan inputs for a verify batch of `n` tokens, in the
/// head-major layouts of `prefill_chunk`'s pass 1.
#[derive(Default)]
pub(crate) struct SsmReplay {
	pub(crate) n: usize,
	pub(crate) q: Vec<f32>,
	pub(crate) k: Vec<f32>,
	pub(crate) v: Vec<f32>,
	pub(crate) beta: Vec<f32>,
	pub(crate) gate: Vec<f32>,
}

impl SpecSnap {
	pub(crate) fn floats(&self) -> usize {
		self.conv
			.iter()
			.flatten()
			.flatten()
			.map(|r| r.len())
			.sum::<usize>()
			+ self.base.iter().map(|s| s.len()).sum::<usize>()
			+ self
				.inp
				.iter()
				.flatten()
				.map(|r| r.q.len() + r.k.len() + r.v.len())
				.sum::<usize>()
	}
}

impl RunState {
	/// Consume the current MRoPE position for one *text* token and advance:
	/// the next token sits one past the largest coordinate used so far, on
	/// every axis. For a text-only run this is just `pos + 1` three times.
	pub fn advance_text_mpos(&mut self) -> [i64; 3] {
		let cur = self.mpos;
		let next = cur[0].max(cur[1]).max(cur[2]) + 1;
		self.mpos = [next; 3];
		cur
	}
}

/// The part of a [`RunState`] that cannot be recovered by truncation.
///
/// The KV cache is append-only, so rolling it back to an earlier position is
/// just shortening each layer's `Vec`. The gated-delta-net state and the conv
/// buffer are overwritten in place every token, so going back means having kept
/// a copy, but that copy is small (~150 MB here, against ~128 KB of KV *per
/// token*), which is what makes a prefix cache affordable: the multi-GB KV is
/// never duplicated.
pub struct Checkpoint {
	state: Vec<Vec<f32>>,
	convbuf: Vec<Vec<Vec<f32>>>,
	kv_len: Vec<usize>,
	pos: usize,
	mpos: [i64; 3],
	mrope: bool,
	/// MTP head: KV length, `last_h`, `pending` (all small but the KV).
	mtp: Option<(usize, Vec<f32>, Vec<MtpEntry>)>,
	/// The drafter's K/V row width, to cut its context back on rewind.
	dflash_kvd: Option<usize>,
}

impl Checkpoint {
	/// Position the checkpoint was taken at.
	pub fn pos(&self) -> usize {
		self.pos
	}
}

impl RunState {
	/// Record the current position so [`RunState::rewind`] can return to it.
	pub fn checkpoint(&self) -> Checkpoint {
		Checkpoint {
			state: self.state.clone(),
			convbuf: self.convbuf.clone(),
			kv_len: self
				.kv
				.iter()
				.map(|c| c.as_ref().map_or(0, |(k, _)| k.len()))
				.collect(),
			pos: self.pos,
			mpos: self.mpos,
			mrope: self.mrope,
			mtp: self.mtp.as_ref().map(|m| {
				(
					m.kv.as_ref().map_or(0, |(k, _)| k.len()),
					m.last_h.clone(),
					m.pending.clone(),
				)
			}),
			dflash_kvd: self.dflash.as_ref().map(|d| d.kv_dim),
		}
	}

	/// Return to `cp`, which must have been taken from this same state at or
	/// before its current position (every token since then is forgotten).
	/// Returns `false`, leaving the state untouched, if it was not.
	pub fn rewind(&mut self, cp: Checkpoint) -> bool {
		let fits = cp.pos <= self.pos
			&& cp.kv_len.len() == self.kv.len()
			&& self.kv.iter().zip(&cp.kv_len).all(|(c, &n)| match c {
				Some((k, v)) => n <= k.len() && n <= v.len(),
				None => n == 0,
			}) && match (&self.mtp, &cp.mtp) {
			(None, None) => true,
			(Some(m), Some((n, _, _))) => *n <= m.kv.as_ref().map_or(0, |(k, _)| k.len()),
			_ => false,
		};
		if !fits {
			return false;
		}
		if let (Some(d), Some(kvd)) = (&mut self.dflash, cp.dflash_kvd) {
			d.truncate(cp.pos, kvd);
		}
		if let (Some(m), Some((n, h, p))) = (&mut self.mtp, cp.mtp) {
			if let Some((k, v)) = &mut m.kv {
				k.truncate(n);
				v.truncate(n);
			}
			m.last_h = h;
			m.pending = p;
		}
		for (c, &n) in self.kv.iter_mut().zip(&cp.kv_len) {
			if let Some((k, v)) = c {
				k.truncate(n);
				v.truncate(n);
			}
		}
		self.state = cp.state;
		self.convbuf = cp.convbuf;
		self.pos = cp.pos;
		self.mpos = cp.mpos;
		self.mrope = cp.mrope;
		true
	}

	/// Bytes this state holds, counting the KV cache by its **capacity**:
	/// `new_run_state` reserves the whole `--ctx` up front, and on Windows that
	/// reservation is committed memory whether or not it has been written.
	pub fn bytes(&self) -> usize {
		let f = std::mem::size_of::<f32>();
		let h = std::mem::size_of::<u16>();
		let kv: usize = self
			.kv
			.iter()
			.flatten()
			.map(|(k, v)| k.capacity() + v.capacity())
			.sum();
		let st: usize = self.state.iter().map(|s| s.len()).sum();
		let cb: usize = self.convbuf.iter().flatten().map(|r| r.len()).sum();
		let mkv = self
			.mtp
			.as_ref()
			.and_then(|m| m.kv.as_ref())
			.map_or(0, |(k, v)| k.capacity() + v.capacity());
		let mtp: usize = self.mtp.as_ref().map_or(0, |m| {
			let sn = m.snap.floats();
			let pe: usize = m.pending.iter().map(|e| e.emb.len() + e.h.len()).sum();
			sn + pe + m.last_h.len()
		});
		(kv + mkv) * h + (st + cb + mtp) * f + self.dflash.as_ref().map_or(0, |d| d.bytes())
	}
}

/// Compact copies and on-disk persistence, for the server's **shared** prefix
/// entries: a tool preamble or system prompt that many conversations start
/// with. A live state reserves KV for the whole `--ctx`; a snapshot keeps only
/// the rows it has, and [`crate::Model::fork_run_state`] re-reserves when a
/// request picks it up.
impl RunState {
	/// A copy whose KV caches hold exactly their rows (capacity == length).
	/// The MTP rollback scratch is not copied: it is rebuilt on demand.
	pub fn clone_compact(&self) -> RunState {
		RunState {
			kv: self
				.kv
				.iter()
				.map(|c| c.as_ref().map(|(k, v)| (k.to_vec(), v.to_vec())))
				.collect(),
			state: self.state.clone(),
			convbuf: self.convbuf.clone(),
			pos: self.pos,
			mpos: self.mpos,
			mrope: self.mrope,
			mtp: self.mtp.as_ref().map(|m| MtpRun {
				kv: m.kv.as_ref().map(|(k, v)| (k.to_vec(), v.to_vec())),
				last_h: m.last_h.clone(),
				pending: m.pending.clone(),
				snap: SpecSnap::default(),
			}),
			dflash: self.dflash.clone(),
		}
	}

	/// Serialize to `w`, preceded by `tag` (an opaque string the reader uses
	/// to refuse a file written for a different model) and the token ids the
	/// state has consumed. Raw host-endian f32 (the KV cache: half bits).
	pub fn write_to(
		&self,
		w: &mut impl std::io::Write,
		tag: &str,
		tokens: &[u32],
	) -> std::io::Result<()> {
		w.write_all(SNAP_MAGIC)?;
		put_u64(w, tag.len() as u64)?;
		w.write_all(tag.as_bytes())?;
		put_u64(w, tokens.len() as u64)?;
		w.write_all(as_bytes(tokens))?;
		put_u64(w, self.pos as u64)?;
		for m in self.mpos {
			put_u64(w, m as u64)?;
		}
		put_u64(w, self.mrope as u64)?;
		put_u64(w, self.kv.len() as u64)?;
		for c in &self.kv {
			put_opt_kv(w, c.as_ref())?;
		}
		put_u64(w, self.state.len() as u64)?;
		for s in &self.state {
			put_f32s(w, s)?;
		}
		put_u64(w, self.convbuf.len() as u64)?;
		for l in &self.convbuf {
			put_u64(w, l.len() as u64)?;
			for r in l {
				put_f32s(w, r)?;
			}
		}
		match &self.mtp {
			None => put_u64(w, 0)?,
			Some(m) => {
				put_u64(w, 1)?;
				put_opt_kv(w, m.kv.as_ref())?;
				put_f32s(w, &m.last_h)?;
				put_u64(w, m.pending.len() as u64)?;
				for e in &m.pending {
					put_f32s(w, &e.emb)?;
					put_f32s(w, &e.h)?;
					for x in e.mpos {
						put_u64(w, x as u64)?;
					}
				}
			}
		}
		Ok(())
	}

	/// Read a snapshot's header: its tag and token ids. Leaves `r` at the
	/// state, for [`RunState::read_body`], so a caller can reject a file for
	/// the wrong model without reading gigabytes.
	pub fn read_header(r: &mut impl std::io::Read) -> std::io::Result<(String, Vec<u32>)> {
		let mut magic = [0u8; 8];
		r.read_exact(&mut magic)?;
		if &magic != SNAP_MAGIC {
			return Err(if magic[..4] == SNAP_MAGIC[..4] {
				bad("format version (older build?)")
			} else {
				bad("magic")
			});
		}
		let n = get_len(r, 1 << 16)?;
		let mut tag = vec![0u8; n];
		r.read_exact(&mut tag)?;
		let tag = String::from_utf8(tag).map_err(|_| bad("tag"))?;
		let n = get_len(r, 1 << 28)?;
		let mut tokens = vec![0u32; n];
		r.read_exact(as_bytes_mut(&mut tokens))?;
		Ok((tag, tokens))
	}

	/// The state following [`RunState::read_header`].
	pub fn read_body(r: &mut impl std::io::Read) -> std::io::Result<RunState> {
		let pos = get_u64(r)? as usize;
		let mut mpos = [0i64; 3];
		for m in &mut mpos {
			*m = get_u64(r)? as i64;
		}
		let mrope = get_u64(r)? != 0;
		let n = get_len(r, 1 << 16)?;
		let mut kv = Vec::with_capacity(n);
		for _ in 0..n {
			kv.push(get_opt_kv(r)?);
		}
		let n = get_len(r, 1 << 16)?;
		let mut state = Vec::with_capacity(n);
		for _ in 0..n {
			state.push(get_f32s(r)?);
		}
		let n = get_len(r, 1 << 16)?;
		let mut convbuf = Vec::with_capacity(n);
		for _ in 0..n {
			let m = get_len(r, 1 << 16)?;
			let mut l = Vec::with_capacity(m);
			for _ in 0..m {
				l.push(get_f32s(r)?);
			}
			convbuf.push(l);
		}
		let mtp = match get_u64(r)? {
			0 => None,
			_ => {
				let kv = get_opt_kv(r)?;
				let last_h = get_f32s(r)?;
				let n = get_len(r, 1 << 20)?;
				let mut pending = Vec::with_capacity(n);
				for _ in 0..n {
					let emb = get_f32s(r)?;
					let h = get_f32s(r)?;
					let mut mpos = [0i64; 3];
					for m in &mut mpos {
						*m = get_u64(r)? as i64;
					}
					pending.push(MtpEntry { emb, h, mpos });
				}
				Some(MtpRun {
					kv,
					last_h,
					pending,
					snap: SpecSnap::default(),
				})
			}
		};
		Ok(RunState {
			kv,
			state,
			convbuf,
			pos,
			mpos,
			mrope,
			mtp,
			dflash: None,
		})
	}
}

/// The prefix cache's form of a state (`crates/prefixcache`): the KV rows
/// as row streams it pages and shares between checkpoints, everything else
/// as one blob. Unlike [`RunState::clone_compact`] nothing is copied here:
/// the cache copies only the pages no other checkpoint has.
impl crate::Model {
	/// Bytes per KV row (one position of one layer's K or V).
	fn kv_row(&self) -> usize {
		self.hp.head_dim * self.hp.n_head_kv
	}

	/// `rs` as (blob, row streams): each attention layer's K then V, then
	/// the MTP head's K and V, each `(half bits, rows)`. The DFlash
	/// drafter's context is left out, as on disk: losing it only lowers
	/// acceptance until it has seen new tokens.
	pub fn state_parts<'a>(&self, rs: &'a RunState) -> (Vec<u8>, Vec<(&'a [u16], usize)>) {
		let w = self.kv_row().max(1);
		let mut b = Vec::new();
		let mut rows = Vec::new();
		let mut kv = |c: Option<&'a KvCache>, b: &mut Vec<u8>| {
			let _ = put_u64(b, c.is_some() as u64);
			if let Some((k, v)) = c {
				rows.push((&k[..], k.len() / w));
				rows.push((&v[..], v.len() / w));
			}
		};
		let _ = put_u64(&mut b, rs.pos as u64);
		for m in rs.mpos {
			let _ = put_u64(&mut b, m as u64);
		}
		let _ = put_u64(&mut b, rs.mrope as u64);
		let _ = put_u64(&mut b, rs.kv.len() as u64);
		for c in &rs.kv {
			kv(c.as_ref(), &mut b);
		}
		let _ = put_u64(&mut b, rs.state.len() as u64);
		for x in &rs.state {
			let _ = put_f32s(&mut b, x);
		}
		let _ = put_u64(&mut b, rs.convbuf.len() as u64);
		for l in &rs.convbuf {
			let _ = put_u64(&mut b, l.len() as u64);
			for r in l {
				let _ = put_f32s(&mut b, r);
			}
		}
		match &rs.mtp {
			None => {
				let _ = put_u64(&mut b, 0);
			}
			Some(m) => {
				let _ = put_u64(&mut b, 1);
				kv(m.kv.as_ref(), &mut b);
				let _ = put_f32s(&mut b, &m.last_h);
				let _ = put_u64(&mut b, m.pending.len() as u64);
				for e in &m.pending {
					let _ = put_f32s(&mut b, &e.emb);
					let _ = put_f32s(&mut b, &e.h);
					for x in e.mpos {
						let _ = put_u64(&mut b, x as u64);
					}
				}
			}
		}
		(b, rows)
	}

	/// A live state from [`crate::Model::state_parts`]' blob, its KV
	/// reserved for the whole context as [`crate::Model::new_run_state`]
	/// would: `fill(i, out)` writes row stream `i`'s bytes into `out`
	/// (exactly its size: `rows` rows of a KV row). An error when the state
	/// does not fit this model (another layout, MTP on one side only, no
	/// room left in `--ctx`).
	pub fn state_from_parts(
		&self,
		blob: &[u8],
		rows: &[usize],
		fill: &mut dyn FnMut(usize, &mut [u8]) -> Result<(), String>,
	) -> Result<RunState, String> {
		let e = |x: std::io::Error| format!("prefix cache state: {x}");
		let mut r = blob;
		let mut rs = self.new_run_state();
		let w = self.kv_row();
		let mut next = 0usize;
		let mut kv = |r: &mut &[u8], c: Option<&mut KvCache>| -> Result<(), String> {
			let present = get_u64(r).map_err(e)? != 0;
			match (present, c) {
				(false, None) => Ok(()),
				(true, Some((k, v))) => {
					for x in [k, v] {
						let n = *rows
							.get(next)
							.ok_or("prefix cache state: too few row streams")?
							* w;
						if n > x.capacity() {
							return Err("prefix cache state: longer than --ctx".into());
						}
						x.resize(n, 0);
						fill(next, as_bytes_mut(x))?;
						next += 1;
					}
					Ok(())
				}
				_ => Err("prefix cache state: another layer layout".into()),
			}
		};
		rs.pos = get_u64(&mut r).map_err(e)? as usize;
		for m in &mut rs.mpos {
			*m = get_u64(&mut r).map_err(e)? as i64;
		}
		rs.mrope = get_u64(&mut r).map_err(e)? != 0;
		if get_len(&mut r, 1 << 16).map_err(e)? != rs.kv.len() {
			return Err("prefix cache state: another layer count".into());
		}
		for c in rs.kv.iter_mut() {
			kv(&mut r, c.as_mut())?;
		}
		let same = |a: &[f32], n: usize| a.len() == n;
		if get_len(&mut r, 1 << 16).map_err(e)? != rs.state.len() {
			return Err("prefix cache state: another layer count".into());
		}
		for x in rs.state.iter_mut() {
			let v = get_f32s(&mut r).map_err(e)?;
			if !same(&v, x.len()) {
				return Err("prefix cache state: another state size".into());
			}
			*x = v;
		}
		if get_len(&mut r, 1 << 16).map_err(e)? != rs.convbuf.len() {
			return Err("prefix cache state: another layer count".into());
		}
		for l in rs.convbuf.iter_mut() {
			if get_len(&mut r, 1 << 16).map_err(e)? != l.len() {
				return Err("prefix cache state: another conv window".into());
			}
			for x in l.iter_mut() {
				let v = get_f32s(&mut r).map_err(e)?;
				if !same(&v, x.len()) {
					return Err("prefix cache state: another conv width".into());
				}
				*x = v;
			}
		}
		let has_mtp = get_u64(&mut r).map_err(e)? != 0;
		match (has_mtp, rs.mtp.as_mut()) {
			(false, None) => {}
			(true, Some(m)) => {
				kv(&mut r, m.kv.as_mut())?;
				m.last_h = get_f32s(&mut r).map_err(e)?;
				let n = get_len(&mut r, 1 << 20).map_err(e)?;
				m.pending.clear();
				for _ in 0..n {
					let emb = get_f32s(&mut r).map_err(e)?;
					let h = get_f32s(&mut r).map_err(e)?;
					let mut mpos = [0i64; 3];
					for x in &mut mpos {
						*x = get_u64(&mut r).map_err(e)? as i64;
					}
					m.pending.push(MtpEntry { emb, h, mpos });
				}
			}
			_ => return Err("prefix cache state: MTP on one side only".into()),
		}
		if rs.pos >= self.ctx {
			return Err("prefix cache state: no room left in --ctx".into());
		}
		Ok(rs)
	}
}

/// Version 2: f16 KV cache (version 1 stored it as f32).
const SNAP_MAGIC: &[u8; 8] = b"QWRS\x00\x00\x00\x02";

fn bad(what: &str) -> std::io::Error {
	std::io::Error::new(
		std::io::ErrorKind::InvalidData,
		format!("run-state snapshot: bad {what}"),
	)
}

// f32 / u16 / u32 only: plain data with no padding. The files are host-endian, which
// is little-endian on every target this crate builds for.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
	unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
	unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

fn put_u64(w: &mut impl std::io::Write, x: u64) -> std::io::Result<()> {
	w.write_all(&x.to_le_bytes())
}

fn get_u64(r: &mut impl std::io::Read) -> std::io::Result<u64> {
	let mut b = [0u8; 8];
	r.read_exact(&mut b)?;
	Ok(u64::from_le_bytes(b))
}

/// A length prefix, bounded so a corrupt file fails cleanly instead of
/// attempting a huge allocation.
fn get_len(r: &mut impl std::io::Read, max: u64) -> std::io::Result<usize> {
	let n = get_u64(r)?;
	if n > max {
		return Err(bad("length"));
	}
	Ok(n as usize)
}

fn put_f32s(w: &mut impl std::io::Write, v: &[f32]) -> std::io::Result<()> {
	put_u64(w, v.len() as u64)?;
	w.write_all(as_bytes(v))
}

fn get_f32s(r: &mut impl std::io::Read) -> std::io::Result<Vec<f32>> {
	let n = get_len(r, 1 << 34)?;
	let mut v = vec![0.0f32; n];
	r.read_exact(as_bytes_mut(&mut v))?;
	Ok(v)
}

fn put_u16s(w: &mut impl std::io::Write, v: &[u16]) -> std::io::Result<()> {
	put_u64(w, v.len() as u64)?;
	w.write_all(as_bytes(v))
}

fn get_u16s(r: &mut impl std::io::Read) -> std::io::Result<Vec<u16>> {
	let n = get_len(r, 1 << 35)?;
	let mut v = vec![0u16; n];
	r.read_exact(as_bytes_mut(&mut v))?;
	Ok(v)
}

fn put_opt_kv(w: &mut impl std::io::Write, c: Option<&KvCache>) -> std::io::Result<()> {
	match c {
		None => put_u64(w, 0),
		Some((k, v)) => {
			put_u64(w, 1)?;
			put_u16s(w, k)?;
			put_u16s(w, v)
		}
	}
}

fn get_opt_kv(r: &mut impl std::io::Read) -> std::io::Result<Option<KvCache>> {
	Ok(match get_u64(r)? {
		0 => None,
		_ => Some((get_u16s(r)?, get_u16s(r)?)),
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	fn toy() -> RunState {
		RunState {
			kv: vec![Some((vec![], vec![])), None],
			state: vec![vec![0.0; 4]; 2],
			convbuf: vec![vec![vec![0.0; 3]; 2]; 2],
			pos: 0,
			mpos: [0; 3],
			mrope: false,
			mtp: None,
			dflash: None,
		}
	}

	// Stand-in for a forward step: append one KV row and mutate the recurrent
	// state in a way that depends on the token.
	fn step(rs: &mut RunState, t: f32) {
		let (k, v) = rs.kv[0].as_mut().unwrap();
		k.extend_from_slice(&[t as u16, t as u16]);
		v.extend_from_slice(&[t as u16 + 100, t as u16 + 100]);
		for s in rs.state.iter_mut().flatten() {
			*s = *s * 0.5 + t;
		}
		rs.convbuf[1][0][0] += t;
		rs.advance_text_mpos();
		rs.pos += 1;
	}

	#[test]
	fn rewind_then_replay_matches_a_straight_run() {
		let mut a = toy();
		for t in [1.0, 2.0, 3.0] {
			step(&mut a, t);
		}
		let cp = a.checkpoint();
		for t in [4.0, 5.0] {
			step(&mut a, t);
		}
		assert!(a.rewind(cp));
		step(&mut a, 9.0);

		let mut b = toy();
		for t in [1.0, 2.0, 3.0, 9.0] {
			step(&mut b, t);
		}
		assert_eq!(a.kv, b.kv);
		assert_eq!(a.state, b.state);
		assert_eq!(a.convbuf, b.convbuf);
		assert_eq!((a.pos, a.mpos), (b.pos, b.mpos));
	}

	#[test]
	fn rewind_restores_the_mtp_head_state() {
		let mut a = toy();
		a.mtp = Some(MtpRun {
			kv: Some((vec![], vec![])),
			last_h: vec![0.0; 2],
			pending: Vec::new(),
			snap: SpecSnap::default(),
		});
		let feed = |rs: &mut RunState, t: f32| {
			let m = rs.mtp.as_mut().unwrap();
			let (k, v) = m.kv.as_mut().unwrap();
			k.push(t as u16);
			v.push(t as u16 + 100);
			let prev = std::mem::replace(&mut m.last_h, vec![t; 2]);
			m.pending.push(MtpEntry {
				emb: vec![t],
				h: prev,
				mpos: [0; 3],
			});
			step(rs, t);
		};
		feed(&mut a, 1.0);
		let cp = a.checkpoint();
		feed(&mut a, 2.0);
		feed(&mut a, 3.0);
		assert!(a.rewind(cp));
		let m = a.mtp.as_ref().unwrap();
		assert_eq!(m.kv.as_ref().unwrap().0, vec![1]);
		assert_eq!(m.last_h, vec![1.0; 2]);
		assert_eq!(m.pending.len(), 1);
		assert_eq!(a.pos, 1);
	}

	#[test]
	fn snapshot_round_trips_through_bytes() {
		let mut a = toy();
		a.mtp = Some(MtpRun {
			kv: Some((vec![15, 25], vec![0xbc00, 0x3400])),
			last_h: vec![3.0; 2],
			pending: vec![MtpEntry {
				emb: vec![7.0],
				h: vec![8.0, 9.0],
				mpos: [4, 5, 6],
			}],
			snap: SpecSnap::default(),
		});
		for t in [1.0, 2.0, 3.0] {
			step(&mut a, t);
		}
		let c = a.clone_compact();
		assert_eq!(c.kv, a.kv);
		let mut buf = Vec::new();
		c.write_to(&mut buf, "model-x", &[10, 11, 12]).unwrap();
		let mut r = &buf[..];
		let (tag, toks) = RunState::read_header(&mut r).unwrap();
		assert_eq!((tag.as_str(), toks), ("model-x", vec![10, 11, 12]));
		let b = RunState::read_body(&mut r).unwrap();
		assert!(r.is_empty());
		assert_eq!(b.kv, a.kv);
		assert_eq!(b.state, a.state);
		assert_eq!(b.convbuf, a.convbuf);
		assert_eq!((b.pos, b.mpos, b.mrope), (a.pos, a.mpos, a.mrope));
		let (bm, am) = (b.mtp.unwrap(), a.mtp.unwrap());
		assert_eq!((bm.kv, bm.last_h), (am.kv, am.last_h));
		assert_eq!(bm.pending[0].mpos, [4, 5, 6]);
		// Truncated or foreign bytes are an error, not a panic.
		assert!(RunState::read_header(&mut &buf[..5]).is_err());
		assert!(RunState::read_body(&mut &buf[..buf.len() / 2]).is_err());
	}

	#[test]
	fn rewind_refuses_a_checkpoint_from_the_future() {
		let mut a = toy();
		step(&mut a, 1.0);
		let cp = a.checkpoint();
		let mut b = toy();
		assert!(!b.rewind(cp));
		assert_eq!(b.pos, 0);
	}
}
