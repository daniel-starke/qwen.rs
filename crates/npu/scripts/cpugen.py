"""Exact-CPU reference of `npubench gen`: the same per-layer decode graphs
(DIR/layer{L}.xml + head.xml) chained on the CPU in f32 over a prompt plus
greedy tokens, with the same state handling (conv / ssm carried, KV as a
sliding window of CAP rows, mask = the last t past slots + the new one,
position_ids = t). Compares with the NPU run's logits.

    python cpugen.py DIR PROMPT_IDS.txt NEW [--npu NPU.f32] [--emb TABLE.f16]"""
import os
import sys
import numpy as np
import openvino as ov

a = sys.argv[1:]
def opt(name, d, conv=str):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return conv(v)
    return d
npu = opt("--npu", None)
dumpx = opt("--dump-x", None)  # every step's head input (f32 [5120] rows)
xs = []
embf = opt("--emb", os.environ.get("NPU_EMB")) or sys.exit("--emb FILE or NPU_EMB: emb_table.f16")
d, pf, new = a[0], a[1], int(a[2])
prompt = [int(v) for v in open(pf).read().split()]
W, V = 5120, 248320
core = ov.Core()
EXACT = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "INFERENCE_PRECISION_HINT": "f32"}
table = np.memmap(embf, np.float16, "r", shape=(V, W))
layers = []
for L in range(64):
    cm = core.compile_model(core.read_model(f"{d}\layer{L}.xml"), "CPU", EXACT)
    layers.append(cm)
head = core.compile_model(core.read_model(f"{d}\head.xml"), "CPU", EXACT)
state = {}
for cm in layers:
    for p in cm.inputs:
        nm = p.get_any_name()
        if nm.startswith("cache_params.past."):
            state[nm] = np.zeros(list(p.get_shape()), np.float16)
nref = np.fromfile(npu, np.float32).reshape(-1, V) if npu else None

def lsm(z):
    z = z - z.max(); return z - np.log(np.exp(z).sum())

ids, tok = [], prompt[0]
for t in range(len(prompt) + new):
    if t < len(prompt):
        tok = prompt[t]
    x = table[tok].astype(np.float32).reshape(1, 1, W)
    for cm in layers:
        feed = {"x": x}
        for p in cm.inputs:
            nm = p.get_any_name(); shp = list(p.get_shape())
            if nm == "attention_mask":
                cap = shp[-1] - 1
                m = np.zeros(shp, np.int64); m[..., max(0, cap - t):] = 1; feed[nm] = m
            elif nm == "position_ids":
                feed[nm] = np.full(shp, t, np.int64)
            elif nm.startswith("cache_params.past."):
                feed[nm] = state[nm]
        r = cm(feed)
        for o in cm.outputs:
            on = o.get_any_name()
            if on == "y":
                x = r[o].astype(np.float32)
            elif on.startswith("cache_params.present."):
                pn = on.replace("present", "past")
                v = r[o].astype(np.float16)
                if "key" in on or "value" in on:
                    v = v[:, :, 1:]  # slide the window
                state[pn] = v
    xs.append(x.reshape(-1).copy())
    lg = list(head({"x": x}).values())[0].astype(np.float64).ravel()
    line = f"step {t:2d} tok {tok:6d} cpu argmax {int(lg.argmax()):6d}"
    if nref is not None and t < len(nref):
        n = nref[t].astype(np.float64)
        pc, pn_ = lsm(lg), lsm(n)
        kl = float((np.exp(pc) * (pc - pn_)).sum())
        line += f"  npu argmax {int(n.argmax()):6d}  KL {kl:.5f}"
    print(line, flush=True)
    if t + 1 >= len(prompt):
        tok = int(lg.argmax()) if nref is None else int(nref[t].argmax())  # follow the NPU's tokens
        ids.append(tok)
print("cpu-followed ids:", ids)
if dumpx:
    np.stack(xs).astype(np.float32).tofile(dumpx)
