//! The MTP head's files (`--npu-steps mtp`, `OUT/mtp`): the checkpoint's
//! `mtp.*` layer for the engine's host drafter (`npu::flash::mtp`). Every
//! matrix int8 per row (`NAME.i8` codes, `NAME.s` f32 row scales: `w ~= c
//! s`), every vector f32 (`NAME.f32`, RMSNorm weights as `1 + w`). No
//! calibration: the drafter's activations stay f32, and int8 per row needs
//! none; a draft only decides how many positions a verify step takes.

use std::path::Path;

use super::refm::{norm_w, Hc, DA, FE, HCW, HKV, HQ, NE};
use crate::st::Checkpoint;
use npu::flash::{HC_RANK, W};

const P: &str = "mtp.";
const L: &str = "mtp.layers.0.";

/// `[n][k]` as int8 codes per row and f32 row scales.
fn q8(w: &[f32], k: usize) -> (Vec<u8>, Vec<u8>) {
	let mut c = Vec::with_capacity(w.len());
	let mut s = Vec::with_capacity(w.len() / k * 4);
	for r in w.chunks_exact(k) {
		let m = r.iter().fold(0f32, |a, v| a.max(v.abs()));
		let sc = if m > 0.0 { m / 127.0 } else { 1.0 };
		c.extend(
			r.iter()
				.map(|v| (v / sc).round().clamp(-127.0, 127.0) as i8 as u8),
		);
		s.extend_from_slice(&sc.to_le_bytes());
	}
	(c, s)
}

fn put(d: &Path, n: &str, b: &[u8]) -> Result<(), String> {
	let p = d.join(n);
	std::fs::write(&p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn mat(d: &Path, n: &str, w: &[f32], k: usize) -> Result<(), String> {
	let (c, s) = q8(w, k);
	put(d, &format!("{n}.i8"), &c)?;
	put(d, &format!("{n}.s"), &s)
}

fn vecf(d: &Path, n: &str, v: &[f32]) -> Result<(), String> {
	let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
	put(d, &format!("{n}.f32"), &b)
}

fn hc(d: &Path, n: &str, h: &Hc) -> Result<(), String> {
	vecf(d, &format!("{n}.norm"), &h.norm)?;
	mat(d, &format!("{n}.down"), &h.down, HCW)?;
	mat(d, &format!("{n}.up"), &h.up, HC_RANK)?;
	if let Some(j) = &h.inject {
		mat(d, &format!("{n}.inj"), j, HCW)?;
	}
	Ok(())
}

/// Write `OUT/mtp` unless its last file exists.
pub fn step_mtp(out: &Path, ck: &Checkpoint, say: &mut dyn FnMut(&str)) -> Result<(), String> {
	let d = out.join("mtp");
	if d.join("done").exists() {
		return Ok(());
	}
	if !ck.has(&format!("{P}fc_embedding.weight")) {
		return Err("the checkpoint has no mtp.* layer".into());
	}
	crate::mkdir(&d)?;
	say("MTP head (int8 per row)");
	vecf(
		&d,
		"pne",
		&norm_w(ck, &format!("{P}pre_fc_norm_embedding.weight"), W)?,
	)?;
	vecf(
		&d,
		"pnh",
		&norm_w(ck, &format!("{P}pre_fc_norm_hidden.weight"), HCW)?,
	)?;
	mat(
		&d,
		"fce",
		&ck.mat(&format!("{P}fc_embedding.weight"), W, W)?,
		W,
	)?;
	mat(
		&d,
		"fch",
		&ck.mat(&format!("{P}fc_hidden.weight"), W, W)?,
		W,
	)?;
	hc(
		&d,
		"hcf",
		&Hc::load(ck, &format!("{P}hyper_connection_mixer."), false)?,
	)?;
	hc(
		&d,
		"hca",
		&Hc::load(ck, &format!("{L}attn_hyper_connection."), true)?,
	)?;
	hc(
		&d,
		"hcm",
		&Hc::load(ck, &format!("{L}mlp_hyper_connection."), true)?,
	)?;
	let a = format!("{L}self_attn.");
	mat(
		&d,
		"q",
		&ck.mat(&format!("{a}q_proj.weight"), 2 * HQ * DA, W)?,
		W,
	)?;
	mat(
		&d,
		"k",
		&ck.mat(&format!("{a}k_proj.weight"), HKV * DA, W)?,
		W,
	)?;
	mat(
		&d,
		"v",
		&ck.mat(&format!("{a}v_proj.weight"), HKV * DA, W)?,
		W,
	)?;
	mat(
		&d,
		"o",
		&ck.mat(&format!("{a}o_proj.weight"), W, HQ * DA)?,
		HQ * DA,
	)?;
	vecf(&d, "qn", &norm_w(ck, &format!("{a}q_norm.weight"), DA)?)?;
	vecf(&d, "kn", &norm_w(ck, &format!("{a}k_norm.weight"), DA)?)?;
	let m = format!("{L}mlp.");
	mat(&d, "router", &ck.mat(&format!("{m}gate.weight"), NE, W)?, W)?;
	mat(
		&d,
		"sg",
		&ck.mat(&format!("{m}shared_expert.gate_proj.weight"), FE, W)?,
		W,
	)?;
	mat(
		&d,
		"su",
		&ck.mat(&format!("{m}shared_expert.up_proj.weight"), FE, W)?,
		W,
	)?;
	mat(
		&d,
		"sd",
		&ck.mat(&format!("{m}shared_expert.down_proj.weight"), W, FE)?,
		FE,
	)?;
	vecf(
		&d,
		"sgate",
		&ck.vec(&format!("{m}shared_expert_gate.weight"), W)?,
	)?;
	// the routed experts, concatenated: `egu` `[NE][2 FE][W]`, `ed` `[NE][W][FE]`
	let (mut gc, mut gs, mut dc, mut ds) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
	for e in 0..NE {
		if e % 64 == 0 {
			say(&format!("MTP experts {e}..{}", e + 64));
		}
		let gu = ck.rows(&format!("{m}experts.gate_up_proj"), e, e + 1)?;
		let dn = ck.rows(&format!("{m}experts.down_proj"), e, e + 1)?;
		if gu.len() != 2 * FE * W || dn.len() != W * FE {
			return Err(format!("MTP expert {e}: unexpected shapes"));
		}
		let (c, s) = q8(&gu, W);
		gc.extend(c);
		gs.extend(s);
		let (c, s) = q8(&dn, FE);
		dc.extend(c);
		ds.extend(s);
	}
	put(&d, "egu.i8", &gc)?;
	put(&d, "egu.s", &gs)?;
	put(&d, "ed.i8", &dc)?;
	put(&d, "ed.s", &ds)?;
	put(&d, "done", b"")
}
