"""Exact-CPU reference for the per-layer decode decomposition: chain the cut
graphs (layer0..layerN + head) on CPU from a zero state for one token, so the
NPU `npubench token` run can be checked against it.

    python cpuref.py DIR TOKEN [--layers 64] [--export DIR]

Writes DIR/emb.f16 (inputs_embeds of TOKEN, [5120] f16) and DIR/logits_cpu.f32
(the chained logits), and prints the argmax. Uses the export's embeddings
model for inputs_embeds and compiles every layer with f32 activations
(DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_PRECISION_HINT f32)."""
import os, sys, numpy as np
import openvino as ov

a = sys.argv[1:]
def opt(name, d, conv=str):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return conv(v)
    return d
export = opt("--export", os.environ.get("OV_EXPORT")) or sys.exit("--export DIR or OV_EXPORT: the OpenVINO export")
nlayers = opt("--layers", 64, int)
d, tok = a[0], int(a[1])
W, CAP = 5120, 4096
core = ov.Core()
EXACT = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "INFERENCE_PRECISION_HINT": "f32"}

emb_m = core.read_model(export + r"\openvino_text_embeddings_model.xml")
emb = core.compile_model(emb_m, "CPU")
e = emb.create_infer_request().infer({0: np.array([[tok]], np.int64)})
x = list(e.values())[0].reshape(1, 1, W).astype(np.float32)
x.astype(np.float16).tofile(f"{d}\\emb.f16")
print(f"token {tok}: inputs_embeds[:4] = {x.ravel()[:4]}", flush=True)

def run_layer(xmlp, x):
    m = core.read_model(xmlp)
    cm = core.compile_model(m, "CPU", EXACT)
    req = cm.create_infer_request()
    names = {i.get_any_name(): i for i in cm.inputs}
    feed = {"x": x}
    for nm, p in names.items():
        if nm == "x":
            continue
        shp = list(p.get_shape())
        if nm == "attention_mask":
            mk = np.zeros(shp, np.int64); mk[..., -1] = 1; feed[nm] = mk
        elif nm == "position_ids":
            feed[nm] = np.zeros(shp, np.int64)
        elif nm == "beam_idx":
            feed[nm] = np.zeros(shp, np.int32)
        elif nm.startswith("cache_params.past."):
            feed[nm] = np.zeros(shp, np.float16)
        else:
            feed[nm] = np.zeros(shp, np.float32)
    r = req.infer(feed)
    return {o.get_any_name(): r[o] for o in cm.outputs}

for L in range(nlayers):
    out = run_layer(f"{d}\\layer{L}.xml", x)
    x = out["y"].astype(np.float32)
    print(f"layer{L}: y[:3]={x.ravel()[:3]}", flush=True)
out = run_layer(f"{d}\\head.xml", x)
logits = list(out.values())[0].astype(np.float32).ravel()
logits.tofile(f"{d}\\logits_cpu.f32")
print(f"CPU argmax {int(logits.argmax())} = {logits.max():.4f}, vocab {logits.size}", flush=True)
