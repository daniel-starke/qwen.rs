"""Build a lean prefill graph for one delta-net (linear attention) layer from
a cut layer IR (`cut_layer.py`, Loop or chunked form; only its constants are
used), for the NPU low-level prefill track.

    python build_delta.py CUT.xml OUT.xml [--s 256] [--c 32] [--act dyn|none]
                          [--conv eltwise|group]

Interface (all f16):
    x      [1, S, 5120]       residual in
    conv   [1, 3, 10240]      past conv window (last 3 positions' qkv, token-major)
    ssm    [1, 48, 128, 128]  past recurrent state S[h, k, v]
    valid  [1, S, 1]          1 for real positions, 0 for right padding
  -> y [1, S, 5120], conv_out [1, 3, 10240] (window at the last real
     position: `keep` [3] i32 = n - 3 + (3 .. 5) rows of concat(conv, qkv)),
     ssm_out [1, 48, 128, 128].

Design (vs the export's graph): activations f16, token-major everywhere;
in_proj_qkv / z / a / b fused into one int8 MatMul (one activation
quantization), gate / up into one; conv1d (kernel 4, depthwise) as four
shifted multiply-adds; q / k l2-norm on the 16 qk heads before the 3x head
broadcast; the delta rule in the chunked WY form with everything but the
state chain batched over sub-chunks; the norms pre-scaled so x^2 cannot
overflow f16."""
import sys
import numpy as np
import wq as wqmod
import bf16src
from quantrot import block_factors, rot_factors, rotate_any, rotate_graph_any
import openvino as ov
import openvino.opset13 as _op
import os as _os

# `NPU_TAGS=1`: every op's friendly name gets its layer region as a prefix
# (`proj.`, `conv.`, `rec.`, ...), so profiled tasks map back to the graph.
_REGION = ["pre"]
_TAGGED = []


class _OpTag:
    def __getattr__(self, name):
        f = getattr(_op, name)
        if not callable(f) or name in ("parameter", "result"):
            return f

        def g(*a_, **k_):
            r = f(*a_, **k_)
            _TAGGED.append((_REGION[0], r))
            return r
        return g


op = _OpTag()


def region(n):
    _REGION[0] = n

a = sys.argv[1:]


def opt(name, d, conv=str):
    if name in a:
        i = a.index(name)
        v = a[i + 1]
        del a[i:i + 2]
        return conv(v)
    return d


S = opt("--s", 256, int)
C = opt("--c", 128, int)
ACT = opt("--act", "mixed")
SQ_ALPHA = opt("--alpha", 0.65, float)
SQ_STATS = opt("--stats", None)
# `--rotb B`: with `--act rot`, block-diagonal Hadamard rotations of block B
# (one MatMul) instead of the Kronecker rotation; `--rotb -B`: the same B
# signs in every block, folded into the matrix (no sign multiply).
ROTB = opt("--rotb", 0, int)
# `--wai`: the int8 weights as graph inputs `w0, w1, ...` (raw i8 in
# OUT.wN.in) instead of constants: they then live in host buffers the
# runner binds, outside the NPU's ~14 GB cap on graph constants.
WAI = "--wai" in a
a = [v for v in a if v != "--wai"]
# `--allin`: every per-layer constant an input (implies --wai; the int8 row
# scales as inputs applied after the MatMul, the norm / conv / gate vectors
# as f16 inputs), so one compiled blob serves every layer of the same shape
# (the runner appends it once per layer with that layer's buffers; the NPU
# fits only ~27 distinct delta blobs at once).
# `--ablate norec,...`: timing probes that drop a part of the layer.
ABLATE = [v for v in opt("--ablate", "").split(",") if v]
# `--signfold`: the block rotation's random input signs are folded into the
# op before each rotated MatMul input (ln1 / ln2 norm weights, the gated
# norm weight per channel, the up rows' post scales) instead of a SHAVE
# multiply over the activation. Exact.
SIGNFOLD = "--signfold" in a
a = [v for v in a if v != "--signfold"]
ALLIN = "--allin" in a
a = [v for v in a if v != "--allin"]
WAI = WAI or ALLIN
# `--act rot` applies the RMSNorm weight in the graph and rotates /
# quantizes the weighted input; `--fold` folds the weight into the MatMul
# instead (worse: 256-token chain KL 0.0034 vs 0.0017, max 0.23 vs 0.03).
NOFOLD = "--fold" not in a
a = [v for v in a if v not in ("--fold", "--nofold")]
# `--abf16`: in_proj_a / in_proj_b as their own MatMul with f16 activations
# (int8 weights) instead of fused into the int8-activation projection.
ABF16 = "--abf16" in a
a = [v for v in a if v != "--abf16"]
# `--act mixed`: the two MatMuls that read an RMSNorm output (qkvz, gate/up)
# get a static per-tensor int8 range `--rstat R` on the *unweighted*
# normalized input (the norm weight folded into the int8 weights), so no
# per-token rescale follows them; out_proj / down_proj stay dynamic.
RSTAT = opt("--rstat", 16.0, float)
# `--amax minmax`: per-token max |x| as max(ReduceMax(x), -ReduceMin(x))
# (no Abs pass over the big tensor); `abs`: ReduceMax(Abs(x)).
AMAX = opt("--amax", "abs")
# `--ostack 0`: the delta outputs per sub-chunk (no stacked states).
OSTACK = opt("--ostack", 0, int)
# `--debug A,T,X`: extra outputs of the (last half's) recurrence intermediates.
DEBUG = [v for v in opt("--debug", "").split(",") if v]
DBG = {}
CONV = opt("--conv", "group")
# `--halves 2`: the chunk as two halves with the state chained, so the
# compiler can overlap one half's recurrence with the other's matmuls.
HALVES_N = opt("--halves", 1, int)
# `--sq`: how the norms square: `power` (Power(x, 2)), `mul` (x * x; the
# compiler fuses it with the reduce into one untiled `reduce_mean_square`
# SHAVE task, 10.7 ms on [256, 5120]), `mm` (sum of squares as a MatMul
# with a ones vector, on the DPU).
SQ = opt("--sq", "mm")
GSUM = opt("--gsum", "mm")
RGROUPS = opt("--rgroups", 1, int)
# `--gusplit P`: the gate / up MatMul as P MatMuls over row blocks (cached
# int4 codes only); P = 2 is gate and up apart.
GUSPLIT = opt("--gusplit", 1, int)
# `--sgated FILE.json:THR`: static input ranges also for the gated MatMuls
# (out_proj / o_proj: "out", down_proj: "down"; `rcal.py --all`), each only
# where that input's per-token max over its median is <= THR in this layer
SGATED = opt("--sgated", None)
ROUT = RDOWN = None
if SGATED:
    import json as _json
    import re as _re
    _f, _thr = SGATED.rsplit(":", 1)
    _e = _json.load(open(_f))[_re.search(r"layer(\d+)\.xml$", a[0].replace("\\", "/")).group(1)]
    ROUT = _e["out"] if _e["out"] / _e["out_p50"] <= float(_thr) else None
    RDOWN = _e["down"] if _e["down"] / _e["down_p50"] <= float(_thr) else None
# `--projsplit 10240,6144`: the qkvz rows of the projection as MatMuls over
# these row blocks (a boundary at 10240 = qkv | z; cached int4 codes only)
PROJSPLIT = [int(v) for v in opt("--projsplit", "").split(",") if v]
# `--reclean`: fewer elementwise ops in the WY recurrence (see `rec`)
RECLEAN = "--reclean" in a
a = [v for v in a if v != "--reclean"]
# `--addonly`: no Subtract in the recurrence (each lowers to a DPU negate +
# add, with their own DMAs): the doubling masks and the rhs's decay factor
# carry the sign instead, T + T(-L)T, I + (-L), U + (-W)S
ADDONLY = "--addonly" in a
a = [v for v in a if v != "--addonly"]
# `--gsum1`: the four gate-sum MatMuls of `rec` (cum, diff, rsum, gtot: all
# g x a constant) as one MatMul over the concatenated constants, then split
GSUM1 = "--gsum1" in a
a = [v for v in a if v != "--gsum1"]
# `--lt`: the doubling's T L as T x (L^T)^T, with L^T built from k kb^T and
# the transposed decay (one more gate-sum output), so that MatMul's right
# operand needs no transpose (each was a DPU maxpool + DMAs)
LT = "--lt" in a
a = [v for v in a if v != "--lt"]
# `--tcol`: the rhs row scales (beta e^gc on k, beta on v) applied as column
# scales of the [C, C] T instead of the [C, D] rows, X as two MatMuls
# (T diag(s) k, T diag(beta) v): half the multiplied elements, no rhs concat
# or split. (beta on A as e beta_i instead of kb as well: KL 0.0335 -> 0.0356)
TCOL = "--tcol" in a
a = [v for v in a if v != "--tcol"]
# `--shinv`: T = (I + A)^-1 by our SHAVE kernel (forward substitution,
# `crates/npu/shave/trinv.c`) instead of the doubling's MatMuls. The kernel
# rides on a LogSoftmax over the flattened [C*C] axis (never split by the
# compiler, unused elsewhere in a delta layer); compile with
# NPU_RS_KERNEL_DIR holding it as log_softmax_3720xx(_lsu0_wo).elf.
# `--shinv BS`: the kernel built with -DBS=BS inverts only the BS x BS
# diagonal blocks (the doubling's T before level BS); the doubling goes on
# from b = BS on the DPU. BS = C: the whole inverse on the SHAVE.
SHINV = opt("--shinv", 0, int)
# `--gnk`: the gated RMSNorm (o / rms(o) * gw * swish(z), per head) as one
# SHAVE kernel of ours (`crates/npu/shave/gnorm.c`) riding on a Minimum (an
# eltwise op with multi-cluster / multi-SHAVE support, unused elsewhere in a
# delta layer): A = [1, S, H, 2, D] with rows (h, 0) = o, (h, 1) = z, B =
# input `gnb` [1, 1, H, 2, D] with rows (h, 0) = gw and eps (f32) in row
# (0, 1). The compiler tiles it over S, so every row stays whole. Compile
# with NPU_RS_KERNEL_DIR holding it as eltwise_min_3720xx(_lsu0_wo).elf.
GNK = "--gnk" in a
# `--cut pre|post`: the layer as two graphs around the recurrence, which then
# runs elsewhere (the iGPU, `offload` `delta_prefill.glsl`). pre: projections,
# conv, l2-norm, gates -> qn, kn [1, S, 2048] (l2-normalized, unscaled), v
# [1, S, 6144], gb [1, S, 96] (log decay g, then beta; both 0 at padding),
# z [1, S, 6144], conv_out. post: inputs x, o [1, S, 6144] (the recurrence's
# output, token-major) and z -> y.
CUT = opt("--cut", "")
a = [v for v in a if v != "--gnk"]
SG = -1.0 if ADDONLY else 1.0
# `--rmsk`: the norms (ln1, ln2, the gated norm, the q / k l2-norms) in the
# form the compiler's FuseRMSNorm turns into one IE.RMS SHAVE op (f32
# accumulation inside, so no pre-scale): Power -> ReduceMean -> Add eps ->
# Sqrt -> Divide(1, .) -> Multiply. [256, 5120]: 0.89 ms vs 2.38.
# `--rmsk ln,gn,l2` selects norms (default all: ln1 / ln2, the gated norm,
# the q / k l2-norms).
# `--nosep`: with i4 storage, fold each static range's 127 / r into the
# per-channel input in front of it (ln1 / ln2, the gated-norm weight, the up
# rows' scales) instead of a scalar multiply (`sep_scale`) before the FQ.
# For proj / gate-up (one quantized input, several int4 MatMuls) this needs
# the compiler patch 0004 (else the input comes out u8:
# `unrealized_conversion_cast`); `--nosep gated` folds only out / down.
NOSEP = NOSEP_ALL = False
if "--nosep" in a:
    i_ = a.index("--nosep")
    NOSEP = True
    NOSEP_ALL = not (i_ + 1 < len(a) and a[i_ + 1] == "gated")
    del a[i_:i_ + (1 if NOSEP_ALL else 2)]
RMSK = set()
if "--rmsk" in a:
    i_ = a.index("--rmsk")
    if i_ + 1 < len(a) and not a[i_ + 1].startswith("--") and set(a[i_ + 1].split(",")) <= {"ln", "gn", "l2"}:
        RMSK = set(a[i_ + 1].split(","))
        del a[i_ + 1]
    else:
        RMSK = {"ln", "gn", "l2"}
    del a[i_]
assert RECLEAN or not ADDONLY, "--addonly needs --reclean"
# (parsed before the positional arguments; see the comments where used)
QS = opt("--qs", 1.0, float)
VS = opt("--vs", 1.0, float)
PRE = opt("--pre", 1.0 / 16, float)
OPRE = opt("--opre", 16.0, float)
# `--srange FILE.json | R`: with `--act rot`, the two norm-fed MatMuls (the
# projection, gate / up) quantize their rotated input with a static range r
# instead of per token (no max / divide before them, no per-token rescale
# after). FILE.json (`rcal.py`): {"L": {"proj": r, "gu": r}} for layer L
# (from the cut's file name `layerL.xml`); a number: that r everywhere. The
# range folds into the norm weight (x 127 / r) and the int8 row scales
# (x r / 127), so with --allin every layer keeps one graph.
SRANGE = opt("--srange", None)
inp, outp = a
RPROJ = RGU = None
if SRANGE is not None:
    import json
    import re
    try:
        RPROJ = RGU = float(SRANGE)
    except ValueError:
        _lay = re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1)
        _r = json.load(open(SRANGE))[_lay]
        RPROJ, RGU = float(_r["proj"]), float(_r["gu"])
assert S % C == 0
W, H, HK, D = 5120, 48, 16, 128
core = ov.Core()
m = core.read_model(inp)
ops = {o.get_friendly_name(): o for o in m.get_ordered_ops()}
F16, F32 = ov.Type.f16, ov.Type.f32


def const_only(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return True
    seen.add(id(node))
    t = node.get_type_name()
    if t == "Constant":
        return True
    if t == "Parameter":
        return False
    return all(const_only(v.get_node(), seen) for v in node.input_values())


def evaluate(output):
    sub = ov.Model([op.result(output)], [], "w")
    return core.compile_model(sub, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})({})[0].astype(np.float32)


def find(suffix):
    hits = [o for n, o in ops.items() if n.endswith(suffix)]
    if len(hits) != 1:
        raise SystemExit(f"{suffix}: {len(hits)} matches")
    return hits[0]


def const_in(suffix, k=None):
    o = find(suffix)
    idx = [i for i in range(o.get_input_size()) if const_only(o.input_value(i).get_node())]
    i = idx[0] if k is None else k
    return evaluate(o.input_value(i))


def _weight_export(suffix):
    o = find(suffix)
    w = evaluate(o.input_value(1))
    if not o.get_attributes()["transpose_b"]:
        w = w.T
    return w.reshape(-1, w.shape[-1])


def _export_shape(suffix):
    """[n, k] of the export's MatMul weight, without evaluating it."""
    o = find(suffix)
    sh = list(o.input_value(1).get_partial_shape().get_shape())
    if not o.get_attributes()["transpose_b"]:
        sh = sh[::-1]
    return (int(np.prod(sh[:-1])), int(sh[-1]))


def weight(suffix):
    if not bf16src.SRC:
        return _weight_export(suffix)
    import re
    lay = int(re.search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1))
    wb = bf16src.layer_weight(lay, suffix).reshape(_export_shape(suffix))
    if __import__("os").environ.get("NPU_SRC_CHECK") == "1":
        # dequantizes the export's int4 matrix on the CPU: slow, a check only
        w = _weight_export(suffix)
        print(f"bf16 {suffix.split('/')[0]}: export vs bf16 rel {np.linalg.norm(w - wb) / np.linalg.norm(wb):.4f}", flush=True)
    return wb


# --- weights -----------------------------------------------------------------
ln1 = const_in("input_layernorm/aten::mul/Multiply_1").reshape(-1)
eps1 = float(const_in("input_layernorm/aten::add/Add").reshape(-1)[0])
Wqkv = weight("linear_attn.in_proj_qkv/ov_ext::linear/MatMul")
Wz = weight("linear_attn.in_proj_z/ov_ext::linear/MatMul")
Wa = weight("linear_attn.in_proj_a/ov_ext::linear/MatMul")
Wb = weight("linear_attn.in_proj_b/ov_ext::linear/MatMul")
convw = const_in("linear_attn/aten::_convolution/GroupConvolution").reshape(-1, 4)  # [10240, 4]
if any(n.endswith("linear_attn/qwen_rs.convunscale") for n in ops):
    convw = convw * 2.0 ** -5
l2eps = float(const_in("linear_attn/aten::add/Add").reshape(-1)[0])
qsref = float(const_in("linear_attn/qwen_rs.qscale").reshape(-1)[0]) if any(n.endswith("qwen_rs.qscale") for n in ops) else 1.0
if any(n.endswith("linear_attn/aten::mul/Multiply_3") for n in ops):
    qmul = float(const_in("linear_attn/aten::mul/Multiply_3").reshape(-1)[0])  # 1 / sqrt(D)
else:
    # the chunked-form cuts fold 1 / sqrt(D) into the export's q scale
    qmul = 1.0 / np.sqrt(128.0)
    qsref /= qmul
vmul = float(const_in("linear_attn/qwen_rs.vscale").reshape(-1)[0]) if any(n.endswith("qwen_rs.vscale") for n in ops) else 1.0
omul = float(const_in("linear_attn/qwen_rs.oscale").reshape(-1)[0]) if any(n.endswith("qwen_rs.oscale") for n in ops) else 1.0
dtb = const_in("linear_attn/aten::add/Add_2").reshape(-1)  # [48]
acoef = const_in("linear_attn/aten::mul/Multiply_6").reshape(-1)  # [48] = -exp(A_log)
gnw = const_in("linear_attn.norm/aten::mul/Multiply_1").reshape(-1)  # [128]
gneps = float(const_in("linear_attn.norm/aten::add/Add").reshape(-1)[0])
# The export's delta-net scales (`QWEN_OV_DSCALE`, for NPUW's subnormal
# flush) put q x qsref, v / state x vmul into the recurrence; at sub-chunk
# 128 the sums overflow f16 with them. Here the recurrence runs at `--qs` /
# `--vs` (default 1): the state I/O keeps the export's convention (x vmul)
# and is converted at the boundary; the gated norm's eps is rescaled.
gneps /= (omul * qsref * vmul) ** 2
# Residual pre-scale of the RMSNorms (x^2 must not overflow f16, the mean
# square must stay above the f16 floor): `--pre` (default 1/16; the
# embedding-sized layer-0 input wants 1).
# The gated norm's input (the delta output at true scale, often 1e-2-small)
# is scaled up by `--opre` so its mean square clears the f16 floor.
Wo = weight("linear_attn.out_proj/ov_ext::linear/MatMul")
ln2 = const_in("post_attention_layernorm/aten::mul/Multiply_1").reshape(-1)
eps2 = float(const_in("post_attention_layernorm/aten::add/Add").reshape(-1)[0])
Wg = weight("mlp.gate_proj/ov_ext::linear/MatMul")
Wu = weight("mlp.up_proj/ov_ext::linear/MatMul")
Wd = weight("mlp.down_proj/ov_ext::linear/MatMul")
Wproj = np.concatenate([Wqkv, Wz, Wa, Wb], 0)
Wproj_main = Wproj[:-2 * Wa.shape[0]]
Wab_rows = Wproj[-2 * Wa.shape[0]:]
Wgu = np.concatenate([Wg, Wu], 0)
if ACT == "mixed" or (ACT == "rot" and not NOFOLD):
    # the norm weight folded into W (`--act mixed`, `--fold`)
    Wproj_n = Wproj * ln1.reshape(1, -1)
    Wqkvz_n, Wab_n = Wproj_n[:-2 * Wa.shape[0]], Wproj_n[-2 * Wa.shape[0]:]
    Wgu_n = Wgu * ln2.reshape(1, -1)
# `--act sq`: the norm-fed MatMuls (projection, gate / up) per-token int8 on
# SmoothQuant-scaled inputs (`--stats CAL.npz` from `actq.py --stats`,
# `--alpha`), the gated-product inputs (out / o / down projection, whose
# outlier channels move per token) f16 activations with int8 weights.
# Measured on the CPU chain (256 tokens): mean KL 0.0026 vs 0.045 for
# per-token int8 everywhere.
GATED = "none" if ACT == "sq" else None
if ACT == "sq":
    from smooth import smooth_c
    _st = dict(np.load(SQ_STATS))
    c1 = smooth_c(_st, find("linear_attn.in_proj_qkv/ov_ext::linear/MatMul").get_friendly_name(), Wproj, SQ_ALPHA)
    c2 = smooth_c(_st, find("mlp.gate_proj/ov_ext::linear/MatMul").get_friendly_name(), Wgu, SQ_ALPHA)
    Wproj_s, ln1_s = Wproj * c1.reshape(1, -1), ln1 / c1
    Wgu_s, ln2_s = Wgu * c2.reshape(1, -1), ln2 / c2
print(f"weights read; export scales q x{qsref} v x{vmul} o x{omul}; internal q x{QS} v x{VS}; eps {eps1} {l2eps} {gneps}", flush=True)


# --- graph helpers -----------------------------------------------------------
def c16(v):
    return op.constant(np.asarray(v, np.float16))


def ci64(v):
    return op.constant(np.asarray(v, np.int64))


WPARAMS = []
_RET_TUPLE = True
# `NPU_WQ_STORE=i4`: int4 weights as i4 inputs; the delta-net gate rows
# (in_proj_a / b, int8) then run as their own MatMul on the same input
STORE_I4 = __import__("os").environ.get("NPU_WQ_STORE") == "i4"


def sep_scale(x, name, r):
    """x * 127 / r with 127 / r a per-layer runtime scalar (input `name`):
    the static range as a SHAVE multiply between the rotation MatMul and the
    FakeQuantize. Folded into the rotation's output the quantization comes
    out u8 and the i4-input MatMul cannot take it (see `fq_static`)."""
    return op.multiply(x, pconst(name, np.full([1, 1, 1], 127.0 / r)))


def fq_static(x):
    """Static int8 quantization of an input pre-scaled to +-127. With i4
    weights the range is [-128, 127] in 256 levels (exactly i8): the
    symmetric [-127, 127] / 255 form lowers to u8 with zero point 127, which
    the compiler cannot cast to the i8 an i4-input MatMul needs
    (`unrealized_conversion_cast` in ConvertLayers2VPU)."""
    if STORE_I4:
        return op.fake_quantize(x, c16(-128.0), c16(127.0), c16(-128.0), c16(127.0), 256)
    lo, hi = c16(-127.0), c16(127.0)
    return op.fake_quantize(x, lo, hi, lo, hi, 255)


def i8_weight(w, fold=1.0, H=None, name=None, rowsign=None):
    """int8 per-row weights; `fold` multiplies the row scales (a static
    activation range's r / 127, applied with the dequantization). `w` may be
    a callable (the rotated weight, formed only if no cached codes fit)."""
    c = wqmod.cached(name, w.shape) if WAI and STORE_I4 and ALLIN and wqmod.MODE != "f" else None
    if c is not None:
        # cached GPTQ codes: bound as stored, the weight never formed
        packed, sc = c
        sc = sc * fold
        if rowsign is not None:
            sc = sc * np.asarray(rowsign).reshape(-1, 1)
        rows = [w.shape[0] // GUSPLIT] * GUSPLIT if name == "gu" and GUSPLIT > 1 else             PROJSPLIT if name == "proj" and PROJSPLIT and w.shape[0] == sum(PROJSPLIT) else None
        if rows:
            # row blocks as their own inputs: a list of (weight, post) pairs
            res, r0 = [], 0
            for n in rows:
                prm = op.parameter([n, w.shape[1]], ov.Type.i4, name=f"w{len(WPARAMS)}")
                WPARAMS.append((prm, np.ascontiguousarray(packed[r0:r0 + n])))
                res.append((op.multiply(op.convert(prm, "f16"), c16(2.0 ** -8)),
                            pconst(f"s{len(WPARAMS) - 1}", sc[r0:r0 + n].reshape(1, 1, -1) * 2.0 ** 8)))
                r0 += n
            return res, None
        prm = op.parameter(list(w.shape), ov.Type.i4, name=f"w{len(WPARAMS)}")
        WPARAMS.append((prm, packed))
        return (op.multiply(op.convert(prm, "f16"), c16(2.0 ** -8)),
                pconst(f"s{len(WPARAMS) - 1}", sc.reshape(1, 1, -1) * 2.0 ** 8))
    if callable(w):
        w = w()
    if wqmod.MODE == "f":
        # float weights (accuracy reference)
        wexp = op.constant(np.asarray(w * fold, np.float16))
        return (wexp, None) if isinstance(_RET_TUPLE, bool) and _RET_TUPLE else wexp
    q, sc = wqmod.quant(w, H, name)
    if sc.shape[1] > 1:
        # int4 groups (accuracy simulation only): q x per-group scale, no
        # row-scale factor after the MatMul
        n, k = q.shape
        gs = sc.shape[1]
        prm = op.parameter([n, gs, k // gs], ov.Type.i8, name=f"w{len(WPARAMS)}")
        WPARAMS.append((prm, q.reshape(n, gs, k // gs)))
        wexp = op.reshape(op.multiply(op.convert(prm, "f16"), op.constant((sc * fold).reshape(n, gs, 1).astype(np.float16))),
                          op.constant(np.array([n, k], np.int64)), False)
        return (wexp, None)
    sc = sc * fold
    if rowsign is not None:
        sc = sc * np.asarray(rowsign).reshape(sc.shape[0], *([1] * (sc.ndim - 1)))
    if WAI:
        # an i8 graph input (as fast as a constant), written to OUT.wN.in;
        # the scale a constant before the MatMul, or (--allin) an input
        # applied after it (as an input before it the dequant runs on SHAVE:
        # 4x slower; after: +10 % on that MatMul)
        if STORE_I4 and q.min() >= -8 and q.max() <= 7:
            # int4 storage (`NPU_WQ_STORE=i4`): packed nibbles in OUT.wN.in,
            # the same bytes the decode graphs' constants hold
            prm = op.parameter(list(q.shape), ov.Type.i4, name=f"w{len(WPARAMS)}")
            WPARAMS.append((prm, wqmod.pack4(q)))
        else:
            prm = op.parameter(list(q.shape), ov.Type.i8, name=f"w{len(WPARAMS)}")
            WPARAMS.append((prm, q))
        if ALLIN:
            # A fixed constant 2^-8 (the same in every layer, so the graphs
            # stay identical) keeps the MatMul's output in f16 range: in raw
            # int8 x int8 units it reaches ~1e7 and came out inf on the NPU.
            return (op.multiply(op.convert(prm, "f16"), c16(2.0 ** -8)),
                    pconst(f"s{len(WPARAMS) - 1}", sc.reshape(1, 1, -1) * 2.0 ** 8))
        return op.multiply(op.convert(prm, "f16"), c16(sc)), None
    return op.multiply(op.convert(op.constant(q), "f16"), c16(sc)), None


# `NPU_NOPOST=1`: drop the post-MatMul row scales (timing only, wrong output)
NOPOST = __import__("os").environ.get("NPU_NOPOST") == "1"


class LazyW:
    """A weight formed on first call (`rotate_any`), with its shape known."""

    def __init__(self, shape, fn):
        self.shape, self.fn = tuple(shape), fn

    def __call__(self):
        return self.fn()


def pconst(name, v):
    """A per-layer f16 constant: an input `name` (OUT.name.in) with --allin."""
    v = np.asarray(v, np.float16)
    if not ALLIN:
        return op.constant(v)
    prm = op.parameter(list(v.shape), F16, name=name)
    WPARAMS.append((prm, v))
    return prm


_wcache = {}
_LAYER = int(__import__("re").search(r"layer(\d+)\.xml$", inp.replace("\\", "/")).group(1)) if wqmod.HDIR else None


def _wname(w):
    return {id(Wproj): "proj", id(Wproj_main): "proj", id(Wab_rows): "ab", id(Wo): "out", id(Wgu): "gu", id(Wd): "down"}.get(id(w))


def _hess(w, f):
    """GPTQ Hessian (NPU_WQ=gptq:DIR) of the MatMul input `w` reads."""
    nm = _wname(w)
    return None if nm in (None, "ab") or nm in wqmod.KEEP8 else wqmod.hessian(_LAYER, nm, f, rotate_any)


def linear(x, w, static=None, mode=None, srange=None, rsname=None, presigned=False, rowsign=None):
    """x [.., k] f16 @ w[n, k]^T, int8 weights, per-token int8 activations
    (or a static per-tensor range `static`). `rot`: per token after a
    Kronecker rotation of x, folded into the weight (quantrot.py).
    `srange` r: `rot` with a static range; x arrives pre-scaled by 127 / r
    (folded into the norm weight), so it is quantized as is, and r / 127
    goes into the row scales."""
    mode = mode or ACT
    if mode == "rot":
        f = block_factors(w.shape[1], abs(ROTB), periodic=ROTB < 0) if ROTB else rot_factors(w.shape[1])
        x = rotate_graph_any(op, x, nosign(f) if presigned else f)
        key = (id(w), "rot")
        if key not in _wcache:
            _wcache[key] = i8_weight(LazyW(w.shape, lambda: rotate_any(w, f)), 1.0 if srange is None else srange / 127.0, H=_hess(w, f), name=_wname(w), rowsign=rowsign)
        wq, post = _wcache[key]
        if srange is not None:
            lo, hi = c16(-127.0), c16(127.0)
            if STORE_I4 and rsname:
                x = x if NOSEP and (NOSEP_ALL or rsname in ("rs3", "rs4")) else sep_scale(x, rsname, srange)
            xq = fq_static(x)
            if isinstance(wq, list):
                return [op.multiply(op.matmul(xq, w_, False, True), p_) for w_, p_ in wq]
            y = op.matmul(xq, wq, False, True)
            return y if post is None or NOPOST else op.multiply(y, post)
        mode = "dyn"
    else:
        if id(w) not in _wcache:
            _wcache[id(w)] = i8_weight(w)
        wq, post = _wcache[id(w)]
    y = linear_q(x, wq, static, mode)
    return y if post is None or NOPOST else op.multiply(y, post)


def nosign(f):
    """The graph half of a block rotation whose signs were folded upstream."""
    return (f[0], np.ones_like(f[1]), f[2])


def rsign(k):
    """The random input signs of the block rotation of a K = k input."""
    return block_factors(k, abs(ROTB), periodic=ROTB < 0)[1] if SIGNFOLD else np.ones(k)


def linear_q(x, wq, static, mode):
    if mode == "none":
        return op.matmul(x, wq, False, True)
    if static is not None and mode == "mixed":
        lo, hi = c16(-static), c16(static)
        return op.matmul(op.fake_quantize(x, lo, hi, lo, hi, 255), wq, False, True)
    if mode == "sq":
        mode = "dyn"
    if AMAX == "minmax":
        mx = op.maximum(op.reduce_max(x, ci64([-1]), True), op.negative(op.reduce_min(x, ci64([-1]), True)))
    else:
        mx = op.reduce_max(op.absolute(x), ci64([-1]), True)
    mx = op.maximum(mx, c16(1e-4))
    lo, hi = c16(-127.0), c16(127.0)
    xq = op.fake_quantize(op.multiply(x, op.divide(c16(127.0), mx)), lo, hi, lo, hi, 255)
    return op.multiply(op.matmul(xq, wq, False, True), op.multiply(mx, c16(1.0 / 127.0)))


def rmsnorm(x, w, eps, pre=None, fused=True):
    """x * rsqrt(mean(x^2) + eps) * w, with x pre-scaled by `pre` (exact:
    the norm is scale invariant up to eps, which is scaled with it)."""
    if "ln" in RMSK and fused:
        y = rms_fused(x, eps)
        return y if w is None else op.multiply(y, w if isinstance(w, (ov.Node, ov.Output)) else c16(w))
    pre = PRE if pre is None else pre
    xs = op.multiply(x, c16(pre)) if pre != 1.0 else x
    ms = sumsq(xs, mean=True)
    # per-row reciprocal, then a multiply: a Divide of the big tensor runs
    # on SHAVE (~4x slower). `w = None`: unweighted (folded elsewhere).
    y = op.multiply(xs, rsqrt(ms, eps * pre * pre))
    if w is None:
        return y
    return op.multiply(y, w if isinstance(w, (ov.Node, ov.Output)) else c16(w))


def rms_fused(x, eps):
    """x / sqrt(mean(x^2) + eps) over the last axis, as FuseRMSNorm matches it."""
    ms = op.reduce_mean(op.power(x, c16(2.0)), ci64([-1]), True)
    return op.multiply(x, op.divide(c16(1.0), op.sqrt(op.add(ms, c16(eps)))))


def rsqrt(t, eps):
    """1 / sqrt(t + eps) of a small per-row tensor, f16-safe: an eps below
    f16's normal range becomes a floor (max) instead of an add; an eps that
    underflows to 0 made all-zero rows inf, and the NPU's Power(x, -0.5)
    gave non-finite rows. The callers pre-scale so real rows sit far above
    the floor."""
    t = op.add(t, c16(eps)) if eps >= 6.2e-5 else op.maximum(t, c16(6.2e-5))
    return op.divide(c16(1.0), op.sqrt(t))


def sumsq(t, mean=False):
    n = t.get_output_partial_shape(0)[-1].get_length()
    if SQ == "mm":
        ones = c16(np.full((n, 1), 1.0 / n if mean else 1.0, np.float16))
        return op.matmul(op.power(t, c16(2.0)), ones, False, False)
    sq = op.power(t, c16(2.0)) if SQ == "power" else op.multiply(t, t)
    return (op.reduce_mean if mean else op.reduce_sum)(sq, ci64([-1]), True)


def split(x, axis, sizes):
    return op.variadic_split(x, ci64(axis), ci64(sizes)).outputs()


# --- graph -------------------------------------------------------------------
x = op.parameter([1, S, W], F16, name="x")
cpast = op.parameter([1, 3, 10240], F16, name="conv")
spast = op.parameter([1, H, D, D], F16, name="ssm")
valid = op.parameter([1, S, 1], F16, name="valid")
keep = op.parameter([3], ov.Type.i32, name="keep")
params = [x, cpast, spast, valid, keep]
if CUT == "post":
    o_in = op.parameter([1, S, 6144], F16, name="o")
    z_in = op.parameter([1, S, 6144], F16, name="z")
    params = [x, o_in, z_in]

def half(x, cpast, spast, valid, S):
    """One layer over S positions (a half of the chunk with `--halves 2`):
    returns (y, qkv [1, S, 10240] pre-conv, ssm state after)."""
    P = S // C
    region("proj")
    if ACT == "sq":
        proj = linear(rmsnorm(x, ln1_s, eps1), Wproj_s)
    elif ACT == "mixed" or (ACT == "rot" and not NOFOLD):
        xn = rmsnorm(x, None, eps1)
        if ABF16:
            # the gate projections (a / b, 96 rows, feeding exp / sigmoid)
            # apart, f16 activations
            proj = op.concat([linear(xn, Wqkvz_n, RSTAT), linear(xn, Wab_n, mode="none")], 2)
        else:
            proj = linear(xn, Wproj_n, RSTAT)
    else:
        # with i4 storage the range scale 127 / r is a runtime scalar after the
        # rotation (`sep_scale`), not folded into the norm weight
        xn1 = rmsnorm(x, pconst("ln1", ln1 * rsign(W) * (1.0 if RPROJ is None or STORE_I4 and not NOSEP_ALL else 127.0 / RPROJ)), eps1)
        if STORE_I4 and RPROJ is not None and ACT == "rot" and ROTB:
            # one rotation + one static int8 quantization for both MatMuls
            # (two FakeQuantize ops on the same rotated input fail to compile:
            # `unrealized_conversion_cast` with an i4-input weight)
            f = block_factors(W, abs(ROTB), periodic=ROTB < 0)
            xr_ = rotate_graph_any(op, xn1, nosign(f) if SIGNFOLD else f)
            xq = fq_static(xr_ if NOSEP_ALL else sep_scale(xr_, "rs1", RPROJ))
            ys = []
            for Wm in (Wproj_main, Wab_rows):
                key = (id(Wm), "rot")
                if key not in _wcache:
                    _wcache[key] = i8_weight(LazyW(Wm.shape, lambda Wm=Wm: rotate_any(Wm, f)), RPROJ / 127.0, H=_hess(Wm, f), name=_wname(Wm))
                wqq, post = _wcache[key]
                if isinstance(wqq, list):
                    ys.extend(op.multiply(op.matmul(xq, w_, False, True), p_) for w_, p_ in wqq)
                    continue
                y_ = op.matmul(xq, wqq, False, True)
                ys.append(y_ if post is None else op.multiply(y_, post))
            if len(ys) > 2:
                # --projsplit: qkv and z straight from their row blocks
                i_, acc = 0, 0
                while acc < 10240:
                    acc += PROJSPLIT[i_]
                    i_ += 1
                assert acc == 10240, "--projsplit needs a boundary at 10240"
                PSPLIT = (ys[0] if i_ == 1 else op.concat(ys[:i_], 2),
                          ys[i_] if len(ys) - 1 - i_ == 1 else op.concat(ys[i_:-1], 2), ys[-1])
            proj = op.concat(ys, 2) if len(ys) <= 2 else None
        elif STORE_I4:
            proj = op.concat([linear(xn1, Wproj_main, srange=RPROJ), linear(xn1, Wab_rows, srange=RPROJ)], 2)
        else:
            proj = linear(xn1, Wproj, srange=RPROJ)  # [1, S, 16480]
    region("conv")
    if proj is None:
        qkv, z, ab_ = PSPLIT
        pa, pb = split(ab_, 2, [H, H])
    else:
        qkv, z, pa, pb = split(proj, 2, [10240, 6144, H, H])

    # conv1d, kernel 4, causal, depthwise: y[t] = sum_j w[:, j] * ext[t + j]
    ext = op.concat([cpast, qkv], 1)  # [1, S + 3, 10240]
    if CONV == "group":
        # a depthwise GroupConvolution runs on the DPU also with the filter
        # as an input (--allin): 1.4 ms vs 5.3 for the eltwise form
        xt = op.transpose(ext, ci64([0, 2, 1]))
        cw = pconst("cw", convw.reshape(10240, 1, 1, 4))
        cv = op.group_convolution(xt, cw, [1], [0], [0], [1])
        conv = op.transpose(cv, ci64([0, 2, 1]))
    else:
        conv = None
        for j in range(4):
            sl = op.slice(ext, ci64([j]), ci64([j + S]), ci64([1]), ci64([1]))
            t = op.multiply(sl, pconst(f"conv{j}", convw[:, j].reshape(1, 1, -1)) if ALLIN else c16(convw[:, j].reshape(1, 1, -1)))
            conv = t if conv is None else op.add(conv, t)
    conv = op.swish(conv, c16(1.0))
    q, k, v = split(conv, 2, [2048, 2048, 6144])


    def l2n(t):
        # Per-row max-abs pre-scale: the largest element becomes 1, so the
        # sum of squares is >= 1 and no row's squares sink into f16
        # subnormals, which the NPU flushes (a fixed 1/4 lost the rows of
        # small k: 11 % off). Exact up to eps (1e-6 of a sum >= 1).
        t = op.reshape(t, ci64([1, S, HK, D]), False)
        if "l2" in RMSK:
            # t / sqrt(sum t^2 + eps) = rms(t) / sqrt(D); the caller applies
            # 1 / sqrt(D) after a reshape (a Multiply right after the norm is
            # taken for its gamma, and a non-1-D one stops the fusion)
            return op.reshape(rms_fused(t, l2eps / D), ci64([1, S, HK * D]), False)
        m = op.maximum(op.reduce_max(op.absolute(t), ci64([-1]), True), c16(1e-4))
        ts = op.multiply(t, op.divide(c16(1.0), m))
        ss = sumsq(ts)
        r = rsqrt(ss, 0.0)
        DBG.update(l2ss=ss, l2r=r, l2ts=ts)
        return op.multiply(ts, r)


    def heads48(t):
        """[1, S, 16, D] -> [1, 48, S, D] (v head h reads qk head h // 3)."""
        t = op.reshape(t, ci64([1, S, HK, 1, D]), False)
        t = op.broadcast(t, ci64([1, S, HK, 3, D]))
        t = op.reshape(t, ci64([1, S, H, D]), False)
        return op.transpose(t, ci64([0, 2, 1, 3]))


    region("heads")
    qh = heads48(op.multiply(l2n(q), c16(qmul * QS / (np.sqrt(D) if "l2" in RMSK else 1.0))))
    kh = heads48(op.multiply(l2n(k), c16(1.0 / np.sqrt(D))) if "l2" in RMSK else l2n(k))
    vh = op.transpose(op.multiply(op.reshape(v, ci64([1, S, H, D]), False), c16(VS)), ci64([0, 2, 1, 3]))
    # gates, [1, 48, S]
    vt = op.transpose(valid, ci64([0, 2, 1]))  # [1, 1, S]
    g = op.multiply(op.softplus(op.add(pa, pconst("dtb", dtb))), pconst("acoef", acoef))
    g = op.multiply(op.transpose(g, ci64([0, 2, 1])), vt)
    beta = op.multiply(op.transpose(op.sigmoid(pb), ci64([0, 2, 1])), vt)
    if CUT == "pre":
        tm = lambda t: op.reshape(t, ci64([1, S, HK * D]), False)
        g_tm = op.multiply(op.multiply(op.softplus(op.add(pa, pconst("dtb", dtb))), pconst("acoef", acoef)), valid)
        b_tm = op.multiply(op.sigmoid(pb), valid)
        return dict(qn=tm(l2n(q)), kn=tm(l2n(k)), v=v, gb=op.concat([g_tm, b_tm], 2), z=z), qkv, None

    def rec(qh, kh, vh, g, beta, spast_, Hn):
        """The chunked WY recurrence over Hn heads: (o [1, Hn, S, D], state)."""
        # chunked WY delta rule, batched over the P sub-chunks
        x5 = [1, Hn, P, C, D]
        q5, k5, v5 = (op.reshape(t, ci64(x5), False) for t in (qh, kh, vh))
        DBG.update(conv=conv, k=k, kn=l2n(k), kh=kh, k5=k5)
        g5 = op.reshape(g, ci64([1, Hn, P, C]), False)
        b5 = op.reshape(beta, ci64([1, Hn, P, C]), False)
        tri_s = np.tril(np.ones((C, C), np.float16), -1)
        tri_i = np.tril(np.ones((C, C), np.float16), 0)
        # Gate sums as masked MatMuls (fp32 accumulation, one f16 rounding)
        # instead of an f16 CumSum and differences of it: the cumsum reaches
        # |500| within a sub-chunk, where f16's step is 0.25-0.5, so
        # `gc_i - gc_j` and `gsum - gc` canceled to +-0.5 in the exponent
        # (kd 5 % off on the NPU). `--gsum cumsum` is the old form.
        if GSUM == "mm":
            i_, j_ = np.meshgrid(np.arange(C), np.arange(C), indexing="ij")
            g4 = op.unsqueeze(g5, ci64(3))  # [.., 1, C]
            sr = np.arange(C).reshape(C, 1, 1)
            mdiff = ((j_[None] < sr) & (sr <= i_[None])).astype(np.float16).reshape(C, C * C)
            mcum, mrs, mtot = (i_ <= j_).astype(np.float16), (i_ > j_).astype(np.float16), np.ones((C, 1), np.float16)
            mdiffT = mdiff.reshape(C, C, C).transpose(0, 2, 1).reshape(C, C * C)
            if GSUM1 and LT:
                allm = op.matmul(g4, c16(np.concatenate([mdiff, mdiffT, mcum, mrs, mtot], 1)), False, False)
                d_, dT_, c_, r_, t_ = split(allm, 4, [C * C, C * C, C, C, 1])
                eT = op.exp(op.reshape(dT_, ci64([1, Hn, P, C, C]), False))
            elif GSUM1:
                allm = op.matmul(g4, c16(np.concatenate([mdiff, mcum, mrs, mtot], 1)), False, False)
                d_, c_, r_, t_ = split(allm, 4, [C * C, C, C, 1])
            else:
                d_, c_, r_, t_ = (op.matmul(g4, c16(m_), False, False) for m_ in (mdiff, mcum, mrs, mtot))
            cum = op.squeeze(c_, ci64([3]))
            diff = op.reshape(d_, ci64([1, Hn, P, C, C]), False)
            e = op.exp(diff)
            rsum = op.unsqueeze(op.squeeze(r_, ci64([3])), ci64(-1))
            gtot = t_  # [.., 1, 1]
        else:
            cum = op.cumsum(g5, ci64(3))
            e = None
        gc = op.unsqueeze(cum, ci64(-1))  # [.., C, 1]
        gr = op.unsqueeze(cum, ci64(3))  # [.., 1, C]
        if e is None:
            e = op.exp(op.multiply(op.subtract(gc, gr), c16(tri_i)))
        bc = op.unsqueeze(b5, ci64(-1))
        eg = op.exp(gc)
        if RECLEAN:
            # beta applied to k once (A's rows and the rhs share it); no
            # strictly-lower mask on A: every read of A (the doubling's L)
            # is masked strictly lower already
            assert not (TCOL and (LT or ADDONLY)), "--tcol excludes --lt / --addonly"
            kb = op.multiply(k5, bc)
            kk = op.matmul(kb, k5, False, True)
            A = op.multiply(kk, e)
            if LT:
                assert GSUM == "mm" and GSUM1, "--lt needs --gsum1"
                AT = op.multiply(op.matmul(k5, kb, False, True), eT)
            rhs = None if TCOL else op.concat([op.multiply(kb, op.multiply(eg, c16(SG)) if ADDONLY else eg), op.multiply(v5, bc)], 4)
        else:
            kk = op.matmul(k5, k5, False, True)
            A = op.multiply(op.multiply(op.multiply(kk, e), bc), c16(tri_s))
            rhs = op.concat([op.multiply(op.multiply(k5, bc), eg), op.multiply(v5, bc)], 4)
        T = None
        b = 1
        if SHINV:
            assert RECLEAN and not ADDONLY, "--shinv needs --reclean"
            T = op.reshape(op.log_softmax(op.reshape(A, ci64([1, Hn * P, 1, C * C]), False), 3), ci64([1, Hn, P, C, C]), False)
            b = SHINV
        while b < (2 if "nodbl" in ABLATE else C):
            i, j = np.meshgrid(np.arange(C), np.arange(C), indexing="ij")
            msk = ((i // (2 * b) == j // (2 * b)) & (i // b > j // b)).astype(np.float16)
            L = op.multiply(A, c16(msk * SG))
            if T is None:
                T = op.add(c16(np.eye(C, dtype=np.float16)), L) if ADDONLY else op.subtract(c16(np.eye(C, dtype=np.float16)), L)
            else:
                if LT:
                    LTb = op.multiply(AT, c16(np.ascontiguousarray(msk.T) * SG))
                    TLT = op.matmul(op.matmul(T, LTb, False, True), T, False, False)
                else:
                    TLT = op.matmul(op.matmul(T, L, False, False), T, False, False)
                T = op.add(T, TLT) if ADDONLY else op.subtract(T, TLT)
            b *= 2
        if TCOL:
            # X = T diag(beta e^gc) k | T diag(beta) v, the scales as T's columns
            row = lambda t: op.reshape(t, ci64([1, Hn, P, 1, C]), False)
            Wm = op.matmul(op.multiply(T, row(op.multiply(eg, bc))), k5, False, False)
            Um = op.matmul(op.multiply(T, row(bc)), v5, False, False)
            X = Wm
        else:
            X = op.matmul(T, rhs, False, False) if T is not None else rhs
            Wm, Um = split(X, 4, [D, D])
        DBG.update(A=A, T=T, X=X, kk=kk, e=e, cum=cum)
        qk = op.matmul(q5, k5, False, True)
        qkm = op.multiply(op.multiply(qk, e), c16(tri_i))
        qe = op.multiply(q5, eg)
        if GSUM == "mm":
            kd = op.multiply(k5, op.exp(rsum))
            egc = op.exp(gtot)
        else:
            gsum = op.unsqueeze(op.reduce_sum(g5, ci64([3]), True), ci64(-1))  # [.., 1, 1]
            kd = op.multiply(k5, op.exp(op.subtract(gsum, gc)))
            egc = op.exp(gsum)


        def part(t, p):
            sl = op.slice(t, ci64([p]), ci64([p + 1]), ci64([1]), ci64([2]))
            return op.squeeze(sl, ci64([2]))


        st = spast_
        ss, ds = [], []
        for p in range(P):
            ss.append(st)
            WS = op.matmul(part(Wm, p), st, False, False)
            dm = op.add(part(Um, p), WS) if ADDONLY else op.subtract(part(Um, p), WS)
            st = op.add(op.multiply(st, part(egc, p)), op.matmul(part(kd, p), dm, True, False))
            ds.append(dm)
            DBG[f"dm{p}"] = dm
            DBG[f"st{p}"] = st
        DBG.update(kd=kd, egc=egc, Wm=Wm, Um=Um, qe=qe)
        if OSTACK:
            sstk = op.concat([op.unsqueeze(t, ci64(2)) for t in ss], 2) if P > 1 else op.unsqueeze(ss[0], ci64(2))
            dstk = op.concat([op.unsqueeze(t, ci64(2)) for t in ds], 2) if P > 1 else op.unsqueeze(ds[0], ci64(2))
            o = op.add(op.matmul(qe, sstk, False, False), op.matmul(qkm, dstk, False, False))  # [1, H, P, C, D]
            o = op.reshape(o, ci64([1, Hn, S, D]), False)
        else:
            # per sub-chunk: no stacked copies of the states
            os_ = [op.add(op.matmul(part(qe, p), ss[p], False, False), op.matmul(part(qkm, p), ds[p], False, False)) for p in range(P)]
            o = op.concat(os_, 2) if P > 1 else os_[0]  # [1, H, S, D]
        return o, st

    region("rec")
    # `--rgroups G`: the recurrence as G independent head groups, so the
    # scheduler can overlap one group's SHAVE ops with another's DPU MatMuls
    if CUT == "post":
        o, st, z = op.reshape(o_in, ci64([1, S, H, D]), False), spast, z_in
    elif RGROUPS > 1:
        hg = H // RGROUPS
        sl = lambda t, i: op.slice(t, ci64([i * hg]), ci64([(i + 1) * hg]), ci64([1]), ci64([1]))
        res = []
        for i in range(RGROUPS):
            region(f"rec{i}")
            res.append(rec(sl(qh, i), sl(kh, i), sl(vh, i), sl(g, i), sl(beta, i), sl(spast, i), hg))
        region("rec")
        o = op.concat([r_[0] for r_ in res], 1)
        st = op.concat([r_[1] for r_ in res], 1)
    else:
        o, st = rec(qh, kh, vh, g, beta, spast, H)
    if CUT != "post":
        o = op.transpose(o, ci64([0, 2, 1, 3]))  # [1, S, H, D]
    if "norec" in ABLATE:
        # timing probe: everything but the recurrence (its inputs still
        # computed, the state passed through)
        o = op.transpose(op.add(op.add(op.add(vh, qh), kh), op.unsqueeze(op.add(g, beta), ci64(-1))), ci64([0, 2, 1, 3]))
        st = spast

    region("onorm")
    # gated RMSNorm per head, out_proj, residual
    gw = gnw * rsign(H * D).reshape(H, D) if SIGNFOLD else gnw
    if NOSEP and ROUT:
        gw = gw * (127.0 / ROUT)
    if GNK:
        gb = np.zeros((1, 1, H, 2, D), np.float16)
        gb[0, 0, :, 0, :] = np.broadcast_to(np.asarray(gw, np.float32), (H, D))
        gb.reshape(-1)[D:D + 2] = np.frombuffer(np.float32(gneps * (QS * VS) ** 2).tobytes(), np.float16)
        zz = op.reshape(z, ci64([1, S, H, D]), False)
        A_ = op.concat([op.unsqueeze(o, ci64(3)), op.unsqueeze(zz, ci64(3))], 3)
        y_ = op.minimum(A_, pconst("gnb", gb))
        attn = op.reshape(op.slice(y_, ci64([0]), ci64([1]), ci64([1]), ci64([3])), ci64([1, S, H * D]), False)
    elif "gn" in RMSK:
        # fused norm (gamma 1), a reshape, then the [48, 128] weight and the gate
        on = op.reshape(rms_fused(o, gneps * (QS * VS) ** 2), ci64([1, S, H * D]), False)
        gwf = np.broadcast_to(np.asarray(gw, np.float32), (H, D)).reshape(1, 1, H * D)
        attn = op.multiply(op.multiply(on, pconst("gnw", gwf)), op.swish(z, c16(1.0)))
    else:
        on = rmsnorm(o, pconst("gnw", gw), gneps * (QS * VS) ** 2, pre=OPRE, fused=False)
        zz = op.swish(op.reshape(z, ci64([1, S, H, D]), False), c16(1.0))
        attn = op.reshape(op.multiply(on, zz), ci64([1, S, H * D]), False)
    DBG.update(onorm_o=o, onorm_z=z, attn=attn)
    x1 = op.add(x, linear(attn, Wo, mode=GATED, presigned=SIGNFOLD, srange=ROUT, rsname="rs3" if ROUT else None))
    region("gu")
    if ACT == "sq":
        gu = linear(rmsnorm(x1, ln2_s, eps2), Wgu_s)
    elif ACT == "mixed" or (ACT == "rot" and not NOFOLD):
        gu = linear(rmsnorm(x1, None, eps2), Wgu_n, RSTAT)
    else:
        gu = linear(rmsnorm(x1, pconst("ln2", ln2 * rsign(W) * (1.0 if RGU is None or STORE_I4 and not NOSEP_ALL else 127.0 / RGU)), eps2), Wgu, srange=RGU, rsname="rs2",
                    presigned=SIGNFOLD, rowsign=np.concatenate([np.ones(Wg.shape[0]), rsign(Wd.shape[1]) *
                                                                (127.0 / RDOWN if NOSEP and RDOWN else 1.0)])
                    if SIGNFOLD or NOSEP and RDOWN else None)
    region("down")
    if isinstance(gu, list):
        h_ = len(gu) // 2
        gg = gu[0] if h_ == 1 else op.concat(gu[:h_], 2)
        uu = gu[h_] if h_ == 1 else op.concat(gu[h_:], 2)
    else:
        gg, uu = split(gu, 2, [Wg.shape[0], Wg.shape[0]])
    y = op.add(x1, linear(op.multiply(op.swish(gg, c16(1.0)), uu), Wd, mode=GATED, presigned=SIGNFOLD,
                          srange=RDOWN, rsname="rs4" if RDOWN else None))


    return y, qkv, st


HALVES = HALVES_N
s_in = op.multiply(spast, c16(VS / vmul)) if VS != vmul else spast
if HALVES == 1:
    y, qkv, st = half(x, cpast, s_in, valid, S)
    ext = op.concat([cpast, qkv], 1)
else:
    hs = S // HALVES
    ys, qkvs = [], []
    cp, st = cpast, s_in
    for i in range(HALVES):
        xh = op.slice(x, ci64([i * hs]), ci64([(i + 1) * hs]), ci64([1]), ci64([1]))
        vh_ = op.slice(valid, ci64([i * hs]), ci64([(i + 1) * hs]), ci64([1]), ci64([1]))
        yh, qkvh, st = half(xh, cp, st, vh_, hs)
        ys.append(yh)
        qkvs.append(qkvh)
        cp = op.slice(qkvh, ci64([hs - 3]), ci64([hs]), ci64([1]), ci64([1]))
    y = op.concat(ys, 1)
    ext = op.concat([cpast] + qkvs, 1)
cnext = op.gather(ext, keep, ci64(1))  # [1, 3, 10240]
if VS != vmul and st is not None:
    st = op.multiply(st, c16(vmul / VS))

outs = []
dbg = [(DBG[k], "dbg_" + k) for k in DEBUG]
named = [(y, "y"), (cnext, "conv_out"), (st, "ssm_out")]
if CUT == "pre":
    named = [(y[k_], k_) for k_ in ("qn", "kn", "v", "gb", "z")] + [(cnext, "conv_out")]
elif CUT == "post":
    named = [(y, "y")]
for t, nm in named + dbg:
    r = op.result(t)
    r.output(0).get_tensor().set_names({nm})
    outs.append(r)
model = ov.Model(outs, params + [prm for prm, _ in WPARAMS], "delta")
for prm, q in WPARAMS:
    q.tofile(outp[:-4] + "." + prm.get_friendly_name() + ".in")
if _os.environ.get("NPU_TAGS") == "1":
    for _rg, _n in _TAGGED:
        _n = _n.get_node() if hasattr(_n, "get_node") else _n
        if _n.get_type_name() not in ("Parameter", "Result", "Constant"):
            _n.set_friendly_name(f"{_rg}.{_n.get_friendly_name()}")
model.validate_nodes_and_infer_types()
ov.save_model(model, outp, compress_to_fp16=False)
print(f"S={S} c={C} act={ACT} conv={CONV}: {len(model.get_ops())} ops -> {outp}")
