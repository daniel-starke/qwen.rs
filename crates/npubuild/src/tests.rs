//! The graphs' interfaces: what the engine binds by name and by order.

use crate::dec::{self, AttnC, DeltaC, MlpC, MtpC};
use crate::g::Ty;
use crate::lean::{self, AttnConsts, Cut, Part};
use crate::refm::{D, DA, FF, H, HK, HKV, HQ, W};

fn names(g: &crate::g::Graph) -> Vec<String> {
	g.inputs().into_iter().map(|i| i.0).collect()
}

#[test]
fn delta_prefill_graph_interface() {
	let g = lean::delta(256, Cut::Whole, 256.0);
	// the op count of the graph the Python builder made with the shipped options
	assert_eq!(g.op_count(), 2044);
	let n = names(&g);
	assert_eq!(n[..6], ["x", "conv", "ssm", "valid", "keep", "ln1"]);
	assert_eq!(n[6..14], ["w1", "s1", "w3", "s3", "w5", "s5", "w7", "s7"]);
	assert_eq!(
		n[14..],
		[
			"cw", "dtb", "acoef", "gnb", "w13", "s13", "ln2", "w16", "s16", "w18", "s18", "w20",
			"s20", "w22", "s22", "w24", "s24"
		]
	);
	assert_eq!(g.outputs(), ["y", "conv_out", "ssm_out"]);
	let ins = g.inputs();
	assert_eq!(ins[6], ("w1".to_string(), Ty::I4, vec![5120, W]));
	assert_eq!(ins[12], ("w7".to_string(), Ty::I8, vec![2 * H, W]));
	// a 64-position chunk is one sub-chunk
	assert_eq!(
		lean::delta(64, Cut::Whole, 16.0).outputs(),
		["y", "conv_out", "ssm_out"]
	);
}

#[test]
fn cut_delta_graphs_keep_the_layers_argument_names() {
	let pre = lean::delta(256, Cut::Pre, 256.0);
	assert_eq!(pre.outputs(), ["qn", "kn", "v", "gb", "z", "conv_out"]);
	assert!(!pre.used_inputs().contains(&"ssm".to_string()));
	let post = lean::delta(256, Cut::Post, 256.0);
	assert_eq!(names(&post)[..4], ["x", "o", "z", "ln1"]);
	assert_eq!(post.outputs(), ["y"]);
	// the first half's weights are arguments it does not read
	let used = post.used_inputs();
	assert!(
		!used.contains(&"w1".to_string())
			&& used.contains(&"w13".to_string())
			&& used.contains(&"w24".to_string())
	);
}

fn attn_consts() -> AttnConsts {
	AttnConsts {
		ln1: vec![1.0; W],
		ln2: vec![1.0; W],
		qnw: vec![1.0; DA],
		knw: vec![1.0; DA],
		s_proj: vec![0.01; 2 * HQ * DA + 2 * HKV * DA],
		s_out: vec![0.01; W],
		s_gu: vec![0.01; 2 * FF],
		s_down: vec![0.01; W],
		r_proj: 4.0,
		r_out: 1.0,
		r_gu: 2.0,
		r_down: 1.0,
	}
}

#[test]
fn attention_prefill_graph_interfaces() {
	let c = attn_consts();
	let all = lean::attn(64, 1024, Part::All, &c);
	assert_eq!(
		names(&all),
		["x", "kpast", "vpast", "mask", "cos", "sin", "w0", "w1", "w2", "w3", "w4", "w5", "w6"]
	);
	assert_eq!(all.outputs(), ["y", "knew", "vnew"]);
	assert_eq!(
		names(&lean::attn(64, 1024, Part::Pre, &c)),
		["x", "cos", "sin", "w0"]
	);
	let post = lean::attn(4, 1024, Part::Post, &c);
	assert_eq!(
		names(&post),
		["x", "o", "gate", "w1", "w2", "w3", "w4", "w5", "w6"]
	);
	let near = lean::attn_core(64, 1024, Part::Near);
	assert_eq!(
		names(&near),
		["qg", "kpast", "vpast", "knew", "vnew", "mask"]
	);
	assert_eq!(near.outputs(), ["o", "lse"]);
	assert_eq!(
		names(&lean::attn_core(4, 1024, Part::Near0)),
		["qg", "kpast", "vpast", "mask"]
	);
	assert_eq!(names(&lean::attn_core(4, 1024, Part::Far(8))).len(), 17);
}

fn mlp_c() -> MlpC {
	MlpC {
		ln2: vec![1.0; W],
		s_gu: vec![0.01; 2 * FF],
		s_down: vec![0.01; W],
	}
}

fn attn_c() -> AttnC {
	AttnC {
		ln1: vec![1.0; W],
		qnw: vec![1.0; DA],
		knw: vec![1.0; DA],
		s_proj: vec![0.01; 2 * HQ * DA + 2 * HKV * DA],
		s_out: vec![0.01; W],
		mlp: mlp_c(),
	}
}

#[test]
fn verify_graph_interfaces() {
	let c = DeltaC {
		ln1: vec![1.0; W],
		s_proj: vec![0.01; 2 * HK * D + 2 * H * D],
		s_out: vec![0.01; W],
		ab: vec![1; 2 * H * W],
		s_ab: vec![0.01; 2 * H],
		conv: vec![0.1; (2 * HK * D + H * D) * 4],
		dtb: vec![0.0; H],
		acoef: vec![-1.0; H],
		gnw: vec![1.0; D],
		mlp: mlp_c(),
	};
	let g = dec::delta_verify(4, 7, &c, false);
	assert_eq!(
		names(&g),
		[
			"x",
			"cache_params.past.conv.7",
			"cache_params.past.ssm.7",
			"qwen_rs.kp",
			"qwen_rs.dp",
			"qwen_rs.w",
			"qwen_rs.alpha",
			"w1",
			"w3",
			"w5",
			"w13"
		]
	);
	assert_eq!(g.outputs().join(" "), dec::DELTA_OUTS);
	assert_eq!(g.inputs()[7], ("w1".to_string(), Ty::I4, vec![5120, 1, W]));
	let g = dec::delta_verify(4, 7, &c, true);
	assert_eq!(names(&g)[11..], ["w16", "w18", "w20", "w22", "w24"]);
	let a = dec::attn_verify(4, 4096, 2, &attn_c(), true);
	assert_eq!(
		names(&a),
		[
			"x",
			"cache_params.past.key.2",
			"cache_params.past.value.2",
			"position_ids",
			"attn_bias",
			"w0",
			"w1",
			"w2",
			"w3",
			"w4",
			"w5",
			"w6"
		]
	);
	assert_eq!(a.outputs().join(" "), dec::attn_outs(2));
	assert_eq!(a.inputs()[2].2, [1, HKV, DA, 4096]);
	assert_eq!(
		names(&dec::attn_verify(4, 4096, 2, &attn_c(), false))[5..],
		["w0", "w1"]
	);
}

#[test]
fn head_and_mtp_interfaces() {
	let h = dec::head(4, &vec![1.0; W], &vec![0.01; 1000]);
	assert_eq!(names(&h), ["x", dec::LM_HEAD]);
	assert_eq!(h.outputs(), ["logits", "hn"]);
	// the engine finds the final norm's weight as the first f32 [1, 1, 5120] constant
	assert!(h
		.xml("head")
		.contains(r#"element_type="f32" shape="1, 1, 5120" offset=""#));
	assert_eq!(dec::mtp_head(&vec![0.01; 64]).outputs(), ["logits"]);
	let c = MtpC {
		ne: vec![1.0; W],
		nh: vec![1.0; W],
		norm: vec![1.0; W],
		s_fc: vec![0.01; W],
		layer: attn_c(),
	};
	let m = dec::mtp(4, 4096, &c);
	assert_eq!(
		names(&m)[..7],
		[
			"hidden",
			"emb",
			dec::MTP_FC,
			"past_key",
			"past_value",
			"position_ids",
			"attention_mask"
		]
	);
	assert_eq!(
		names(&m)[7..],
		[
			"w.mtp.proj",
			"w.mtp.out",
			"w.mtp.gu0",
			"w.mtp.gu1",
			"w.mtp.gu2",
			"w.mtp.gu3",
			"w.mtp.down"
		]
	);
	assert_eq!(m.outputs(), ["h", "present_key", "present_value"]);
	let ins = m.inputs();
	assert_eq!(
		ins[6],
		("attention_mask".to_string(), Ty::I64, vec![1, 4100])
	);
}

/// The tower graphs' arguments: what `npu::vis` binds by name (inputs) and
/// by order (outputs), on a small tower.
#[test]
fn vision_tower_graph_interfaces() {
	use vision::{VisionBlock, VisionConfig, VisionModel};
	let (e, ff, heads, patch, merge, proj, s) = (16usize, 24, 2, 2, 2, 12, 8);
	let cfg = VisionConfig {
		image_size: 0,
		patch_size: patch,
		n_embd: e,
		n_ff: ff,
		n_layer: 2,
		n_head: heads,
		proj_dim: proj,
		merge_size: merge,
		eps: 1e-6,
		mean: [0.5; 3],
		std: [0.5; 3],
		n_pos: 4,
	};
	let v = |n: usize| vec![0.01f32; n];
	let block = || VisionBlock {
		ln1_w: v(e),
		ln1_b: v(e),
		ln2_w: v(e),
		ln2_b: v(e),
		qkv_w: v(3 * e * e),
		qkv_b: v(3 * e),
		out_w: v(e * e),
		out_b: v(e),
		up_w: v(ff * e),
		up_b: v(ff),
		down_w: v(e * ff),
		down_b: v(e),
	};
	let merged = e * merge * merge;
	let m = VisionModel {
		blocks: vec![block(), block()],
		patch_w: v(e * 3 * patch * patch),
		patch_b: v(e),
		pos_w: v(4 * e),
		post_ln_w: v(e),
		post_ln_b: v(e),
		mm0_w: v(merged * merged),
		mm0_b: v(merged),
		mm2_w: v(proj * merged),
		mm2_b: v(proj),
		deepstack: Vec::new(),
		gelu_erf: false,
		cfg,
	};
	let first = crate::vis::graph(&m, 0, s);
	assert_eq!(
		first.inputs(),
		[
			("patch".to_string(), Ty::F16, vec![1, s, 3 * patch * patch]),
			("pos".to_string(), Ty::F16, vec![1, s, e])
		]
	);
	assert_eq!(first.outputs(), ["xo", "qkv"]);
	let mid = crate::vis::graph(&m, 1, s);
	assert_eq!(
		mid.inputs(),
		[
			("x".to_string(), Ty::F16, vec![1, s, e]),
			("a".to_string(), Ty::F16, vec![1, s, e])
		]
	);
	assert_eq!(mid.outputs(), ["xo", "qkv"]);
	// the last graph ends in the merger: a token per 2 x 2 patches
	let last = crate::vis::graph(&m, 2, s);
	assert_eq!(names(&last), ["x", "a"]);
	assert_eq!(last.outputs(), ["emb"]);
	let xml = last.xml("vis2");
	assert!(
		xml.contains(r#"approximation_mode="TANH""#) && xml.contains("normalize_variance"),
		"GELU (tanh) and the layer norms"
	);
	assert!(crate::vis::cfg_text(&m.cfg, s)
		.contains("embd = 16\nheads = 2\nlayers = 2\nproj = 12\nnpos = 4\ns = 8\n"));
}

/// Flash-Next: the blobs name their outputs after ops, so the engine binds
/// them by order ([`npu::flash::engine::OUTS`]; expert jobs: `y`, or `y{e}`
/// per expert at row `e c`). The builders' result order must match.
#[test]
fn flash_graph_outputs_match_the_engines_order() {
	use npu::flash::engine::OUTS;
	for (kind, g) in crate::flash::graph_kinds() {
		let base = kind.strip_suffix("_d1").unwrap_or(kind);
		// `delta_pre_m` takes `delta_pre`'s lists and outputs
		let base = if base == "delta_pre_m" {
			"delta_pre"
		} else {
			base
		};
		let outs = g.outputs();
		if base.starts_with("experts") {
			// a job's experts: its `x{e}` inputs (`experts` / `experts_c64`
			// have one `x{e}` per expert too)
			let ng = g
				.inputs()
				.iter()
				.filter(|i| {
					i.0.strip_prefix('x')
						.is_some_and(|n| n.parse::<usize>().is_ok())
				})
				.count();
			let per: Vec<String> = (0..ng).map(|e| format!("y{e}")).collect();
			assert!(outs == ["y"] || outs == per, "{kind}: outputs {outs:?}");
		} else {
			let want = OUTS
				.iter()
				.find(|o| o.0 == base)
				.unwrap_or_else(|| panic!("{kind}: not in OUTS"))
				.1;
			assert_eq!(outs, want, "{kind}");
		}
	}
}
