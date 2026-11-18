"""ACT-SHAVE instruction tests on the NPU itself: assembly kernels run through
a `Minimum` carrier (compiler patch 0008), one invocation per SHAVE.

    python shavedev.py who                 # stack / code / params addresses and ids of the four SHAVEs
    python shavedev.py lat [SET ...]       # latency and result of each instruction (sets: v1 v2 shfl mac i1 div)
    python shavedev.py time [NAME:GATE ..] # cost of a bundle, in SHAVE cycles (gates: solo same diff all)
    python shavedev.py misc                # branch delay slots, predication, 128-bit load / store
    python shavedev.py swz                 # register zeroing, lane copy, shuffle, operand swizzle modifiers

Run it in an empty work directory: it writes the carrier graph (`m.xml`),
the inputs, and per test `NAME.s`, `k_NAME/` (the kernel under the carrier's
name), `NAME.blob`, `NAME.out`. Needs our `shave` tool (`SHAVE`, default the
release example: `cargo build --release -p npu --example shave`), our compiler
build (`vpuxc.ps1`) and `npubench` (`NPUBENCH`, default the release example).

The graph is `y = Minimum(a, b)` on `[1, 1, 64, 256]` f16: the compiler makes
four invocations (two per tile), each with 16 rows = 8192 bytes of `a`, `b`
and `y` in its tile's CMX. A kernel gets `a` in i16, `b` in i15, `y` in i17
(preamble) and must leave i18-i31 alone, i19 being the stack pointer and i30
the return address.

`lat` issues the instruction once and copies its destination register to a
new register in each of the following cycles; the latency is the first
cycle whose copy holds the final value, and that value is compared with
numpy. `time` runs a 100-bundle block 1000 times and reads the SW task's
duration from the profile: `solo` gates the loop to one SHAVE, `same` to the
two SHAVEs of one tile, `diff` to one SHAVE on each tile."""
import os
import statistics
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHAVE = os.environ.get("SHAVE", os.path.join(HERE, "..", "..", "..", "target", "release", "examples", "shave.exe"))
BENCH = os.environ.get("NPUBENCH", os.path.join(HERE, "..", "..", "..", "target", "release", "examples", "npubench.exe"))
CYCLE = 1.836   # ns per bundle (`time`: iau2 - iau)
PRE = ["LSU0.LDO.32 i17 i18 0x48 || LSU1.LDO.32 i16 i18 0", "LSU0.LDO.32 i15 i18 0x24", "NOP 8"]
POST = ["NOP 8", "BRU.JMP i30", "NOP 6"]
PORT = '<port id="%d" precision="FP16"%s><dim>1</dim><dim>1</dim><dim>64</dim><dim>256</dim></port>'
XML = ('<?xml version="1.0"?><net name="m" version="11"><layers>'
       + "".join('<layer id="%d" name="%s" type="Parameter" version="opset1"><data shape="1,1,64,256" element_type="f16"/><output>%s</output></layer>'
                 % (i, n, PORT % (0, ' names="%s"' % n)) for i, n in enumerate("ab"))
       + '<layer id="2" name="min" type="Minimum" version="opset1"><data auto_broadcast="numpy"/><input>' + PORT % (0, "") + PORT % (1, "")
       + "</input><output>" + PORT % (2, ' names="y"') + "</output></layer>"
       + '<layer id="3" name="res" type="Result" version="opset1"><input>' + PORT % (0, "") + "</input></layer></layers><edges>"
       + '<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>'
       + '<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/></edges></net>')

# operands: 16 bytes of each type at these offsets of every invocation's `a` / `b`
F16, F32, I32, I16, I8, POS16, POS32, U8, FR16 = 0, 16, 32, 48, 64, 96, 112, 128, 144
A = {F16: np.array([0.5, 1.5, -2.0, 3.25, 0.1, -0.7, 4.0, 2.5], np.float16), F32: np.array([1.5, -2.25, 3.0, 0.125], np.float32),
     I32: np.array([100000, -7, 123456789, 5], np.int32), I16: np.array([1000, -7, 30000, 5, -32768, 255, 77, -1], np.int16),
     I8: np.array([100, -7, 127, 5, -128, 25, 77, -1, 1, 2, 3, 4, 5, 6, 7, 8], np.int8), POS16: np.array([0.25, 1, 2, 4, 9, 0.5, 16, 100], np.float16),
     POS32: np.array([0.25, 2.0, 9.0, 100.0], np.float32), U8: np.array([200, 7, 255, 5, 128, 25, 77, 1, 1, 2, 3, 4, 5, 6, 7, 250], np.uint8)}
B = {F16: np.array([2.0, -1.0, 0.5, 4.0, 3.0, 0.3, -1.5, 8.0], np.float16), F32: np.array([2.0, 0.5, -4.0, 8.0], np.float32),
     I32: np.array([3, 9, -1000, 7], np.int32), I16: np.array([3, 9, 10000, 7, -1, 2, -77, 1], np.int16),
     I8: np.array([3, 9, 100, 7, -1, 2, -77, 1, 8, 7, 6, 5, 4, 3, 2, 1], np.int8), POS16: np.array([3, 2, 0.5, 1, 4, 8, 0.25, 10], np.float16),
     POS32: np.array([3.0, 0.5, 4.0, 10.0], np.float32), U8: np.array([100, 9, 100, 7, 255, 2, 177, 1, 8, 7, 6, 5, 4, 3, 2, 1], np.uint8)}


def setup():
    os.makedirs("in", exist_ok=True)
    open("m.xml", "w", newline="\n").write(XML)
    open("m.bin", "wb").close()
    for nm, d in (("a", A), ("b", B)):
        c = bytearray(8192)
        for off, v in d.items():
            c[off:off + 16] = v.tobytes()
        open(f"in/{nm}.in", "wb").write(bytes(c) * 4)


def run(name, body, prof=False):
    """Assemble `body` (bundles) into a kernel, compile the carrier graph with it and run it:
    (the four invocations' 8192 output bytes, the SW tasks' durations in ns when profiled)"""
    lines = PRE + body + POST
    s = (".nowarn\n.code .text.entry\n.align 16\n.type controlled_act,@function\ncontrolled_act:\n"
         + "".join(("" if l.endswith(":") else "    ") + l + "\n" for l in lines) + ".nowarnend\n.end\n")
    open(name + ".s", "w", newline="\n").write(s)
    p = subprocess.run([SHAVE, "build", name + ".s", name + ".elf"], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit("assembly failed: " + (p.stdout + p.stderr)[-800:])
    elf = open(name + ".elf", "rb").read()
    os.makedirs("k_" + name, exist_ok=True)
    for f in ("eltwise_min_3720xx.elf", "eltwise_min_3720xx_lsu0_wo.elf"):
        open(f"k_{name}/{f}", "wb").write(elf)
    env = dict(os.environ, NPU_RS_KERNEL_DIR=os.path.abspath("k_" + name).replace("\\", "/"), NPU_TASKS="1", NPU_INPUTS="in")
    p = subprocess.run(["powershell", "-NoProfile", "-File", os.path.join(HERE, "vpuxc.ps1"), "m.xml", name + ".blob"] + (["-Profile"] if prof else []),
                       env=env, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit("compile failed: " + (p.stdout + p.stderr)[-800:])
    p = subprocess.run([BENCH, "blob", name + ".blob", "--iters", "5", "--dump", name + ".out"] + (["--profile"] if prof else []),
                       env=env, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise SystemExit("run failed: " + (p.stdout + p.stderr)[-800:])
    o = open(name + ".out", "rb").read()
    return [o[i * 8192:(i + 1) * 8192] for i in range(4)], [int(l.split()[3]) for l in p.stdout.splitlines() if l.startswith("T SW")]


# ---------------------------------------------------------------- who
def who():
    body = ["CMU.CPTI i1 P_SVID", "CMU.CPTI i2 P_CFG", "CMU.CPTI i3 P_GPI", "NOP 4",
            "LSU0.STO.32 i19 i17 0 || LSU1.STO.32 i30 i17 4", "LSU0.STO.32 i18 i17 8 || LSU1.STO.32 i1 i17 12",
            "LSU0.STO.32 i2 i17 16 || LSU1.STO.32 i3 i17 20", "LSU0.STO.32 i16 i17 24 || LSU1.STO.32 i15 i17 28", "LSU0.STO.32 i17 i17 32"]
    o, _ = run("who", body)
    print("     stack     return    params    P_SVID    P_CFG     P_GPI     a         b         y")
    for i in range(4):
        print("  " + " ".join("%08x " % x for x in struct.unpack("<9I", o[i][:36])))


# ---------------------------------------------------------------- lat
NV, NI = 20, 12   # samples of a vector (v4..v23) / scalar (i0..i11) result


def ldv(r, base, off):
    return [f"LSU0.LDO.64 {r}.0 {base} {off} || LSU1.LDO.64 {r}.1 {base} {off + 8}"]


def block(s, vec, skip):
    """Operands v1 / i14 from `a`, v2 / i12 from `b`; result in v3 (vector) or i13. A CMU instruction is
    sampled by another unit: two CMU results cannot be written in one cycle."""
    op, off = s["op"], s["off"]
    cmu = op.startswith("CMU.")
    L = ldv("v1", "i16", off) + ldv("v2", "i15", off) + [f"LSU0.LDO.32 i14 i16 {off} || LSU1.LDO.32 i12 i15 {off}", "NOP 8",
                                                         "CMU.CP.128 v3 v1" if vec else "CMU.CP.32 i13 i14"]
    L += list(s.get("pre", ())) + ["NOP 30", op] + ([f"NOP {skip}"] if skip else [])
    if vec:
        L += [(f"VAU.AND v{4 + k} v3 v3" if cmu else f"CMU.CP.128 v{4 + k} v3") for k in range(NV)] + ["NOP 30"]
        L += [f"LSU0.STO.64 v{4 + k}.0 i17 0 || LSU1.STO.64 v{4 + k}.1 i17 8 || IAU.ADD i17 i17 16" for k in range(NV)]
    else:
        L += [(f"IAU.OR i{k} i13 i13" if cmu else f"CMU.CP.32 i{k} i13") for k in range(NI)] + ["NOP 30"]
        L += [f"LSU0.STO.32 i{k} i17 0 || IAU.ADD i17 i17 4" for k in range(NI)]
    return L


def lat(name, specs, vec, skip=0):
    body = []
    for s in specs:
        body += block(s, vec, skip)
    o, _ = run("lat_" + name, body)
    n, w = (NV, 16) if vec else (NI, 4)
    for k, s in enumerate(specs):
        smp = [[o[i][(k * n + j) * w:(k * n + j + 1) * w] for j in range(n)] for i in range(4)]
        fin = smp[0][-1]
        first = next(j for j in range(n) if smp[0][j] == fin)
        got = np.frombuffer(fin, s["dt"])
        note = ""
        if "ref" in s:
            exp = np.asarray(s["ref"]()).astype(s["dt"])
            m = min(len(exp), len(got))
            if np.issubdtype(np.dtype(s["dt"]), np.floating):
                e, g = exp[:m].astype(np.float64), got[:m].astype(np.float64)
                err = np.max(np.abs(g - e) / np.maximum(1e-3, np.abs(e)))
                note = ("OK" if err < 2e-2 else "MISMATCH") + " (rel err %.0e)" % err
            else:
                note = "OK" if (got[:m] == exp[:m]).all() else "MISMATCH"
        if not note.startswith("OK"):
            note += " got " + str(got.tolist())
        if smp[0][-1] != smp[0][-2]:
            note += " (not settled in the window)"
        if any(smp[i] != smp[0] for i in range(1, 4)):
            note += " (SHAVEs differ)"
        print("%-36s latency %-3s %s" % (op_name(s), (skip + first + 1) if first or skip else "1", note), flush=True)


def op_name(s):
    return s["op"] + (" *" if s.get("pre") else "")


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


a16, b16, a32, b32, ai, bi, ah, bh, ab, bb, p16, au, bu = A[F16], B[F16], A[F32], B[F32], A[I32], B[I32], A[I16], B[I16], A[I8], B[I8], A[POS16], A[U8], B[U8]
f64 = np.float64
V1 = [
    dict(op="VAU.ADD.f16 v3 v1 v2", off=F16, dt=np.float16, ref=lambda: a16 + b16), dict(op="VAU.SUB.f16 v3 v1 v2", off=F16, dt=np.float16, ref=lambda: a16 - b16),
    dict(op="VAU.MUL.f16 v3 v1 v2", off=F16, dt=np.float16, ref=lambda: a16 * b16),
    dict(op="VAU.ADD.f32 v3 v1 v2", off=F32, dt=np.float32, ref=lambda: a32 + b32), dict(op="VAU.SUB.f32 v3 v1 v2", off=F32, dt=np.float32, ref=lambda: a32 - b32),
    dict(op="VAU.MUL.f32 v3 v1 v2", off=F32, dt=np.float32, ref=lambda: a32 * b32),
    dict(op="VAU.ADD.i32 v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai + bi), dict(op="VAU.SUB.i32 v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai - bi),
    dict(op="VAU.MUL.i32 v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai * bi),
    dict(op="VAU.ADD.i16 v3 v1 v2", off=I16, dt=np.int16, ref=lambda: ah + bh), dict(op="VAU.MUL.i16 v3 v1 v2", off=I16, dt=np.int16, ref=lambda: ah * bh),
    dict(op="VAU.ADD.i8 v3 v1 v2", off=I8, dt=np.int8, ref=lambda: ab + bb), dict(op="VAU.MUL.i8 v3 v1 v2", off=I8, dt=np.int8, ref=lambda: ab * bb),
    dict(op="VAU.ABS.f16 v3 v1", off=F16, dt=np.float16, ref=lambda: np.abs(a16)), dict(op="VAU.ABS.i32 v3 v1", off=I32, dt=np.int32, ref=lambda: np.abs(ai)),
    dict(op="VAU.AND v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai & bi), dict(op="VAU.XOR v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai ^ bi),
    dict(op="VAU.NOT v3 v1", off=I32, dt=np.int32, ref=lambda: ~ai),
    dict(op="VAU.IADDS.i16 v3 v1 v2", off=I16, dt=np.int16, ref=lambda: np.clip(ah.astype(np.int32) + bh, -32768, 32767)),
    dict(op="VAU.IMULS.i16 v3 v1 v2", off=I16, dt=np.int16, ref=lambda: np.clip(ah.astype(np.int32) * bh, -32768, 32767)),
    dict(op="VAU.ADIFF.f32 v3 v1 v2", off=F32, dt=np.float32, ref=lambda: np.abs(a32 - b32)),
    dict(op="VAU.AVG.x8 v3 v1 v2", off=U8, dt=np.uint8, ref=lambda: (au.astype(np.int32) + bu + 1) >> 1),
    dict(op="VAU.SHL.x32 v3 v1 v2", off=I32, dt=np.int32, ref=lambda: ai << (bi & 31)),
    dict(op="VAU.DP4 v3 v1 v2", off=I8, dt=np.int32, ref=lambda: (ab.astype(np.int32) * bb).reshape(4, 4).sum(1)),
]
V2 = [
    dict(op="VAU.SIGM v3 v1", off=F16, dt=np.float16, ref=lambda: sig(a16)), dict(op="VAU.TANH v3 v1", off=F16, dt=np.float16, ref=lambda: np.tanh(a16.astype(f64))),
    dict(op="VAU.EXP v3 v1", off=F16, dt=np.float16, ref=lambda: np.exp(a16.astype(f64))), dict(op="VAU.LOG v3 v1", off=POS16, dt=np.float16, ref=lambda: np.log(p16.astype(f64))),
    dict(op="VAU.SQT v3 v1", off=POS16, dt=np.float16, ref=lambda: np.sqrt(p16.astype(f64))),
    dict(op="CMU.MIN.f16 v3 v1 v2", off=F16, dt=np.float16, ref=lambda: np.minimum(a16, b16)), dict(op="CMU.MAX.f16 v3 v1 v2", off=F16, dt=np.float16, ref=lambda: np.maximum(a16, b16)),
    dict(op="CMU.MIN.f32 v3 v1 v2", off=F32, dt=np.float32, ref=lambda: np.minimum(a32, b32)), dict(op="CMU.MAX.i32 v3 v1 v2", off=I32, dt=np.int32, ref=lambda: np.maximum(ai, bi)),
    dict(op="CMU.MAX.u8 v3 v1 v2", off=U8, dt=np.uint8, ref=lambda: np.maximum(au, bu)),
    dict(op="CMU.CP.128 v3 v2", off=F16, dt=np.float16, ref=lambda: b16),
    dict(op="CMU.CP.128.f16.f32 v3 v1.0", off=F16, dt=np.float32, ref=lambda: a16[:4].astype(np.float32)),
    dict(op="CMU.CP.128.i32.f32 v3 v1", off=I32, dt=np.float32, ref=lambda: ai.astype(np.float32)),
    dict(op="CMU.CP.128.f32.i32s v3 v1", off=F32, dt=np.int32, ref=lambda: np.trunc(a32)),
    dict(op="CMU.CP.128.i16.i32 v3 v1.0", off=I16, dt=np.int32, ref=lambda: ah[:4].astype(np.int32)),
    dict(op="CMU.CP.128.i8.i16 v3 v1.0", off=I8, dt=np.int16, ref=lambda: ab[:8].astype(np.int16)),
    dict(op="CMU.CP.128.32.r v3 i14", off=I32, dt=np.int32, ref=lambda: np.full(4, ai[0])),
    dict(op="CMU.ALIGNVEC v3 v1 v2 4", off=I8, dt=np.int8, ref=lambda: np.concatenate([ab[4:], bb[:4]])),
    dict(op="LSU0.LD.128 v3 i15", off=F16, dt=np.float16, ref=lambda: b16),
    dict(op="LSU0.LD.64 v3.0 i15", off=F16, dt=np.float16, ref=lambda: np.concatenate([b16[:4], a16[4:]])),
    dict(op="LSU0.LD.128.f16.f32 v3 i15", off=F16, dt=np.float32, ref=lambda: b16[:4].astype(np.float32)),
    dict(op="LSU0.LD.128.i8.i16 v3 i15", off=F16, dt=np.int16, ref=lambda: np.frombuffer(b16.tobytes(), np.int8)[:8].astype(np.int16)),
    dict(op="LSU0.LD.128.u8.f16 v3 i15", off=F16, dt=np.float16, ref=lambda: np.frombuffer(b16.tobytes(), np.uint8)[:8].astype(np.float16)),
    dict(op="LSU0.LD.128.i4.i8 v3 i15", off=F16, dt=np.int8,
         ref=lambda: ((np.repeat(np.frombuffer(b16.tobytes(), np.uint8)[:8], 2) >> np.tile([0, 4], 8) & 15) ^ 8).astype(np.int8) - 8),
]
SHFL = [   # a set holds at most 25 vector tests (8192 bytes of output)
    # window bits, element bits, K: K-way de-interleave of every window (K = half the elements: interleave its halves)
    dict(op="CMU.SHFL v3 v1 128 8 4", off=I8, dt=np.int8, ref=lambda: ab.reshape(4, 4).T.ravel()),
    dict(op="CMU.SHFL v3 v1 128 8 8", off=I8, dt=np.int8, ref=lambda: ab.reshape(2, 8).T.ravel()),
    dict(op="CMU.SHFL v3 v1 64 16 2", off=I16, dt=np.int16, ref=lambda: ah.reshape(2, 2, 2).transpose(0, 2, 1).ravel()),
    dict(op="CMU.SHFL v3 v1 16 4 2", off=I8, dt=np.uint8,
         ref=lambda: [(lambda n: n[0] | n[2] << 4 | n[1] << 8 | n[3] << 12)([w >> 4 * k & 15 for k in range(4)]) >> 8 * j & 255
                      for w in np.frombuffer(ab.tobytes(), np.uint16).tolist() for j in range(2)]),
]
# `*`: the accumulator is prepared first (`pre`); the last two issue three accumulations, every other cycle and ...
Z = ["VAU.XOR v0 v0 v0", "NOP 4", "VAU.ACCPZ.f32 v0 V_ACC0", "NOP 8"]
MAC = [
    dict(op="VAU.MACPW.f32 v3 v1 v2 V_ACC0", off=F32, pre=Z, dt=np.float32, ref=lambda: a32 * b32),
    dict(op="VAU.MACPW.f32 v3 v1 v2 V_ACC0", off=F32, pre=Z + ["VAU.MACPZ.f32 v2 v2 V_ACC0", "NOP 8"], dt=np.float32, ref=lambda: b32 * b32 + a32 * b32),
    dict(op="VAU.ACCPW.f32 v3 v1 V_ACC0", off=F32, pre=Z + ["VAU.MACPZ.f32 v1 v2 V_ACC0", "NOP 8"], dt=np.float32, ref=lambda: a32 * b32 + a32),
    dict(op="VAU.MACNW.f32 v3 v1 v2 V_ACC0", off=F32, pre=Z + ["VAU.ACCPZ.f32 v1 V_ACC0", "NOP 8"], dt=np.float32, ref=lambda: a32 - a32 * b32),
    dict(op="VAU.MACPW.f16 v3 v1 v2 V_ACC0", off=F16, pre=Z + ["VAU.MACPZ.f16 v1 v1 V_ACC0", "NOP 8"], dt=np.float16, ref=lambda: a16 * a16 + a16 * b16),
    dict(op="VAU.MACPW.i32 v3 v1 v2 V_ACC0", off=I32, pre=Z + ["VAU.MACPZ.i32 v2 v2 V_ACC0", "NOP 8"], dt=np.int32, ref=lambda: bi * bi + ai * bi),
    dict(op="VAU.MACPW.f32 v3 v1 v2 V_ACC1", off=F32, pre=["VAU.MACPZ.f32 v2 v2 V_ACC1", "NOP 8"], dt=np.float32, ref=lambda: b32 * b32 + a32 * b32),
    dict(op="VAU.MACPW.f32 v3 v1 v2 V_ACC0", off=F32, pre=Z + ["VAU.MACPZ.f32 v2 v2 V_ACC0", "NOP", "VAU.MACP.f32 v1 v1 V_ACC0", "NOP", "VAU.MACP.f32 v1 v2 V_ACC0", "NOP"],
         dt=np.float32, ref=lambda: b32 * b32 + a32 * a32 + 2 * a32 * b32),
    # ... in consecutive cycles: the second (a * a) is lost, an accumulator takes one accumulation per two cycles
    dict(op="VAU.MACPW.f32 v3 v1 v2 V_ACC0", off=F32, pre=Z + ["VAU.MACPZ.f32 v2 v2 V_ACC0", "VAU.MACP.f32 v1 v1 V_ACC0", "VAU.MACP.f32 v1 v2 V_ACC0"],
         dt=np.float32, ref=lambda: b32 * b32 + 2 * a32 * b32),
]
I1 = [
    dict(op="IAU.ADD i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] + bi[0]]), dict(op="IAU.SUB i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] - bi[0]]),
    dict(op="IAU.MUL i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] * bi[0]]), dict(op="IAU.SHL i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] << bi[0]]),
    dict(op="IAU.AND i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] & bi[0]]), dict(op="IAU.LMULH i13 i14 i14", off=I32, dt=np.int32, ref=lambda: [(int(ai[0]) * int(ai[0])) >> 32]),
    dict(op="IAU.BSR i13 i14", off=I32, dt=np.int32, ref=lambda: [16]), dict(op="IAU.ONES i13 i14", off=I32, dt=np.int32, ref=lambda: [bin(100000).count("1")]),
    dict(op="SAU.ADD.i32 i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] + bi[0]]), dict(op="SAU.MUL.i32 i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] * bi[0]]),
    dict(op="SAU.ADD.f32 i13 i14 i12", off=F32, dt=np.float32, ref=lambda: [a32[0] + b32[0]]), dict(op="SAU.MUL.f32 i13 i14 i12", off=F32, dt=np.float32, ref=lambda: [a32[0] * b32[0]]),
    dict(op="SAU.ADD.f16 i13 i14 i12", off=F16, dt=np.float16, ref=lambda: [a16[0] + b16[0]]), dict(op="SAU.MUL.f16 i13 i14 i12", off=F16, dt=np.float16, ref=lambda: [a16[0] * b16[0]]),
    dict(op="SAU.DIV.f16 i13 i14 i12", off=F16, dt=np.float16, ref=lambda: [a16[0] / b16[0]]),
    dict(op="SAU.SQT i13 v1.0", off=POS16, dt=np.float16, ref=lambda: [np.sqrt(p16[0])]), dict(op="SAU.RQT i13 v1.0", off=POS16, dt=np.float16, ref=lambda: [1 / np.sqrt(p16[0])]),
    dict(op="SAU.RCP i13 v1.0", off=POS16, dt=np.float16, ref=lambda: [1 / p16[0]]), dict(op="SAU.SIN i13 v1.0", off=F16, dt=np.float16, ref=lambda: [np.sin(a16[0])]),
    dict(op="SAU.COS i13 v1.0", off=F16, dt=np.float16, ref=lambda: [np.cos(a16[0])]), dict(op="SAU.ATN i13 v1.0", off=F16, dt=np.float16, ref=lambda: [np.arctan(a16[0])]),
    dict(op="SAU.EXP i13 v1.0", off=F16, dt=np.float16, ref=lambda: [np.exp(a16[0])]), dict(op="SAU.EXP2 i13 v1.0", off=F16, dt=np.float16, ref=lambda: [2.0 ** a16[0]]),
    dict(op="SAU.LOG i13 v1.0", off=POS16, dt=np.float16, ref=lambda: [np.log(p16[0])]), dict(op="SAU.LOG2 i13 v1.0", off=POS16, dt=np.float16, ref=lambda: [np.log2(p16[0])]),
    dict(op="SAU.SIGM i13 v1.0", off=F16, dt=np.float16, ref=lambda: [sig(a16[0])]), dict(op="SAU.TANH i13 v1.0", off=F16, dt=np.float16, ref=lambda: [np.tanh(a16[0])]),
    dict(op="SAU.SUMX.f32 i13 v1", off=F32, dt=np.float32, ref=lambda: [a32.sum()]), dict(op="SAU.SUMX.i32 i13 v1", off=I32, dt=np.int32, ref=lambda: [ai.sum()]),
    dict(op="SAU.DP4 i13 i14 i12", off=I8, dt=np.int32, ref=lambda: [(ab[:4].astype(np.int32) * bb[:4]).sum()]),
    dict(op="LSU0.LDO.32 i13 i15 32", off=I32, dt=np.int32, ref=lambda: [bi[0]]), dict(op="LSU0.LDIL i13 0x1234", off=I32, dt=np.int32, ref=lambda: [0x1234]),
    dict(op="CMU.CP.32 i13 i12", off=I32, dt=np.int32, ref=lambda: [bi[0]]), dict(op="CMU.CP.32 i13 v2.1", off=I32, dt=np.int32, ref=lambda: [bi[1]]),
    dict(op="CMU.CP.32.i32.f32 i13 i14", off=I32, dt=np.float32, ref=lambda: [np.float32(ai[0])]), dict(op="CMU.CP.32.f32.i32s i13 i14", off=F32, dt=np.int32, ref=lambda: [np.trunc(a32[0])]),
    dict(op="CMU.CP.32.f16.f32 i13 v1.0", off=F16, dt=np.float32, ref=lambda: [np.float32(a16[0])]),
]
DIV = [   # sampled from 10 cycles after the instruction
    dict(op="SAU.DIV.f32 i13 i14 i12", off=F32, dt=np.float32, ref=lambda: [a32[0] / b32[0]]), dict(op="SAU.DIV.i32 i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] // bi[0]]),
    dict(op="SAU.MOD.i32 i13 i14 i12", off=I32, dt=np.int32, ref=lambda: [ai[0] % bi[0]]),
]
# u8 fractions (x / 255) and f16 <-> i16: the conversions of an 8-bit softmax
fr = np.array([0.0, 0.25, 0.5, 0.501, 0.998, 1.0, 0.002, 1.5], np.float16)
A[FR16] = B[FR16] = fr
Q8 = [
    dict(op="CMU.CP.64.f16.u8fs v3.0 v1", off=FR16, dt=np.uint8, ref=lambda: np.clip(np.rint(fr.astype(f64) * 255), 0, 255)),
    dict(op="CMU.CP.128.u8f.f16 v3 v1.0", off=U8, dt=np.float16, ref=lambda: au[:8].astype(f64) / 255),
    dict(op="CMU.CP.128.f16.i16s v3 v1", off=F16, dt=np.int16, ref=lambda: np.rint(a16.astype(f64))),
    dict(op="CMU.CP.128.i16.f16 v3 v1", off=I16, dt=np.float16, ref=lambda: ah.astype(f64)),
    dict(op="CMU.CP.64.f16.u8s v3.0 v1", off=F16, dt=np.uint8, ref=lambda: np.clip(np.rint(a16.astype(f64)), 0, 255)),
]
SETS = {"v1": (V1, True, 0), "v2": (V2, True, 0), "shfl": (SHFL, True, 0), "mac": (MAC, True, 0), "i1": (I1, False, 0), "div": (DIV, False, 10), "q8": (Q8, True, 0)}


# ---------------------------------------------------------------- time
def gate(mask, val):
    """skip the loop unless (y pointer & mask) == val: the tile is bit 21 of the address, the SHAVE of a tile bit 13"""
    return ["LSU0.LDIL i10 0x%04x" % (mask & 0xffff), "LSU0.LDIH i10 0x%04x" % (mask >> 16), "LSU0.LDIL i11 0x%04x" % (val & 0xffff), "LSU0.LDIH i11 0x%04x" % (val >> 16),
            "NOP 2", "IAU.AND i12 i17 i10", "NOP 2", "CMU.CMII.u32 i12 i11", "PEU.PC1C NEQ || BRU.BRA tdone", "NOP 6"]


GATES = {"all": [], "solo": gate(0x00202000, 0), "same": gate(0x00200000, 0), "diff": gate(0x00002000, 0)}
LOOPS = {
    "empty": [],
    "iau": ["IAU.ADD i1 i1 1"] * 100,
    "iau2": ["IAU.ADD i1 i1 1"] * 200,
    "nop": ["NOP"] * 100,
    "vmul": ["VAU.MUL.f16 v3 v1 v2"] * 100,
    "sigm": ["VAU.SIGM v3 v1"] * 100,
    "div": ["SAU.DIV.f32 i8 i5 i3"] * 100,
    "wide": ["LSU0.LDIL i6 7 || LSU1.LDIL i7 9 || CMU.CP.32 i4 i5 || IAU.ADD i1 i1 1 || VAU.ADD.i32 v1 v1 v2"] * 100,
    "ld32": ["LSU0.LDO.32 i8 i16 0"] * 100,
    "ld32seq": ["LSU0.LDO.32 i8 i16 %d" % (4 * k) for k in range(100)],
    "ld32x2": ["LSU0.LDO.32 i8 i16 %d || LSU1.LDO.32 i9 i15 %d" % (4 * k, 4 * k) for k in range(100)],
    "ld128": ["LSU0.LD.128 v4 i16"] * 100,
    "ld64x2": ["LSU0.LDO.64 v4.0 i16 %d || LSU1.LDO.64 v4.1 i16 %d" % (16 * k, 16 * k + 8) for k in range(100)],
    "st32": ["LSU0.STO.32 i5 i17 64"] * 100,
    "st128": ["LSU0.ST.128 v2 i17"] * 100,
    "st64x2": ["LSU0.STO.64 v2.0 i17 %d || LSU1.STO.64 v2.1 i17 %d" % (64 + 16 * k, 72 + 16 * k) for k in range(100)],
    "ldst": ["LSU0.LDO.64 v4.0 i16 %d || LSU1.STO.64 v2.0 i17 %d" % (8 * k, 64 + 8 * k) for k in range(100)],
    "ldstack": ["LSU0.LDO.32 i8 i19 -8"] * 100,
    "ldparams": ["LSU0.LDO.32 i8 i18 0"] * 100,
    "ldcode": ["LSU0.LDO.32 i8 i30 1"] * 100,
}


def time_loops(args):
    base = None
    for a in ["empty:solo"] + (args or [k + ":solo" for k in LOOPS if k != "empty"]):
        k, g = a.split(":")
        body = GATES[g] + ["LSU0.LDIL i0 1000", "LSU0.LDIL i1 0 || LSU1.LDIL i2 0", "LSU0.LDIL i3 3 || LSU1.LDIL i5 5", "CMU.CP.128.32.r v2 i3", "CMU.CP.128.32.r v1 i2",
                           "NOP 4", "tloop:"] + LOOPS[k] + ["IAU.INCS i0 -1", "CMU.CMZ.i32 i0", "PEU.PC1C NEQ || BRU.BRA tloop", "NOP 6", "tdone:"]
        _, sw = run(f"time_{k}_{g}", body, prof=True)
        us = [x / 1e3 for x in sw if x > 15000]   # the gated-out invocations return in ~3 us
        if k == "empty":
            base = statistics.median(us)
            continue
        n = len(LOOPS[k])
        print("%-9s %-4s %s us -> %s cycles per bundle" % (k, g, " ".join("%.1f" % x for x in us), " ".join("%.2f" % ((x - base) / n / CYCLE) for x in us)), flush=True)


# ---------------------------------------------------------------- misc
def misc():
    body = ["LSU0.LDIL i1 0", "NOP 2", "BRU.BRA d1"] + ["IAU.ADD i1 i1 1"] * 12 + ["d1:", "NOP 4", "LSU0.STO.32 i1 i17 0",
            "LSU0.LDIL i1 0 || LSU1.LDIL i2 d2", "LSU0.LDIH i2 d2", "NOP 2", "BRU.JMP i2"] + ["IAU.ADD i1 i1 1"] * 12 + ["d2:", "NOP 4", "LSU0.STO.32 i1 i17 4",
            "LSU0.LDIL i3 0 || LSU1.LDIL i4 5", "LSU0.LDIL i5 100 || LSU1.LDIL i6 100", "NOP 2", "CMU.CMZ.i32 i3", "PEU.PC1C EQ || IAU.ADD i5 i5 1", "PEU.PC1C NEQ || IAU.ADD i6 i6 1",
            "CMU.CMZ.i32 i4", "PEU.PC1C EQ || IAU.ADD i5 i5 16", "PEU.PC1C NEQ || IAU.ADD i6 i6 16", "NOP 4", "LSU0.STO.32 i5 i17 8 || LSU1.STO.32 i6 i17 12",
            "LSU0.LDIL i5 100", "CMU.CMZ.i32 i4", "NOP 2", "CMU.CMZ.i32 i3 || PEU.PC1C EQ || IAU.ADD i5 i5 1", "PEU.PC1C EQ || IAU.ADD i5 i5 16", "NOP 4", "LSU0.STO.32 i5 i17 16",
            "LSU0.LD.128 v1 i16", "NOP 10", "IAU.ADD i7 i17 64", "NOP 2", "LSU0.ST.128 v1 i7", "IAU.ADD i7 i17 80 || LSU1.LD.128 v2 i15", "NOP 10", "LSU1.ST.128 v2 i7"]
    o, _ = run("misc", body)
    w = struct.unpack("<5I", o[0][:20])
    print(f"delay slots executed after a taken BRU.BRA: {w[0]}, after BRU.JMP: {w[1]}")
    print(f"PEU.PC1C: EQ counter {w[2]}, NEQ counter {w[3]} (101, 116 = each ran exactly when its condition held)")
    print(f"a predicated-off bundle that also holds the compare: counter {w[4]} (100 = the compare was squashed too, 116 = it ran)")
    print("LD.128 / ST.128 round trip:", o[0][64:80] == A[F16].tobytes() and o[0][80:96] == B[F16].tobytes())


# ---------------------------------------------------------------- swz
def swz():
    """`a` = bytes 00..0f, `b` = bytes 10..1f; each line shows v3 (or i13) after the bundle, v3 = v1 = a before it"""
    for nm, lo in (("a", 0), ("b", 16)):
        c = bytearray(8192)
        c[:16] = bytes(range(lo, lo + 16))
        open(f"in/{nm}.in", "wb").write(bytes(c) * 4)
    tests = [("CMU.CPVRC v3.0 v1", True), ("CMU.CPVRC v3.2 v1", True), ("CMU.SHFL v3 v1 128 8 2", True), ("CMU.SHFL v3 v1 32 8 4", True)]
    for sel in ("[3210] [UUUU]", "[0123] [UUUU]", "[1032] [UUUU]", "[3210] [VVVV]", "[3210] [ZZZZ]", "[3210] [1111]", "[3210] [UVZ1]"):
        tests.append((f"LSU0.SWZMV4.word {sel} || VAU.OR v3 v1 v2", True))
    tests += [("LSU0.SWZMV4.byte [0123] [UUUU] || VAU.OR v3 v1 v2", True), ("LSU0.SWZMV8.L [0123] [UUUU] || VAU.OR v3 v1 v2", True),
              ("LSU0.SWZMV8.H [0123] [UUUU] || VAU.OR v3 v1 v2", True), ("LSU0.SWZMC4.word [3210] [UVZ1] || CMU.MAX.u8 v3 v1 v2", True),
              ("LSU0.SWZMS4.byte [0123] [VVVV] || SAU.OR i13 i14 i12", False)]
    body = []
    for line, _ in tests:
        body += ldv("v1", "i16", 0) + ldv("v2", "i15", 0) + ["LSU0.LDO.32 i14 i16 0 || LSU1.LDO.32 i12 i15 0", "NOP 8", "CMU.CP.128 v3 v1", "CMU.CP.32 i13 i14",
                                                             "NOP 8", line, "NOP 12", "LSU0.STO.64 v3.0 i17 0 || LSU1.STO.64 v3.1 i17 8",
                                                             "LSU0.STO.32 i13 i17 16 || IAU.ADD i17 i17 32"]
    # CPZI / CPZV last: they clear the registers after the named one as well
    for n in range(4):
        body += [f"LSU0.LDIL i{k} {0x100 + k}" for k in range(1, 8)] + [f"CMU.CP.128.32.r v{k} i{k}" for k in range(1, 8)] + ["NOP 4", f"CMU.CPZV v2 {n}", f"CMU.CPZI i2 {n}", "NOP 4"]
        body += [f"LSU0.STO.32 i{k} i17 {4 * (k - 1)}" for k in range(1, 8)] + [f"LSU0.STO.64 v{k}.0 i17 {24 + 8 * k}" for k in range(1, 8)] + ["IAU.ADD i17 i17 96"]
    o, _ = run("swz", body)
    for k, (line, vec) in enumerate(tests):
        r = o[0][k * 32:k * 32 + 20]
        print("%-52s %s" % (line, r[:16].hex(" ") if vec else r[16:20].hex(" ")))
    base = len(tests) * 32
    for n in range(4):
        r = o[0][base + n * 96:base + (n + 1) * 96]
        print(f"CMU.CPZI i2 {n}: i1..i7 = " + " ".join("%x" % x for x in struct.unpack("<7I", r[:28]))
              + f"   CMU.CPZV v2 {n}: v1..v7 = " + " ".join("%x" % x for x in struct.unpack("<14I", r[32:88])[::2]))


def main():
    a = sys.argv[1:]
    setup()
    if a[:1] == ["who"]:
        who()
    elif a[:1] == ["lat"]:
        for k in a[1:] or list(SETS):
            lat(k, *SETS[k])
    elif a[:1] == ["time"]:
        time_loops(a[1:])
    elif a[:1] == ["misc"]:
        misc()
    elif a[:1] == ["swz"]:
        swz()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
