"""Original BF16 weights (Hugging Face safetensors) for the lean builders:
`NPU_SRC=DIR` makes `weight(suffix)` read
model.language_model.layers.{L}.{module}.weight from DIR instead of
dequantizing the export's int4 g128 constant (requantization from the
original, `wq.py`)."""
import json
import os
import numpy as np

SRC = os.environ.get("NPU_SRC")


def tensor(name):
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    fn = os.path.join(SRC, idx[name])
    with open(fn, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        e = hdr[name]
        a, b = e["data_offsets"]
        f.seek(8 + n + a)
        raw = np.frombuffer(f.read(b - a), np.uint16)
    assert e["dtype"] == "BF16", e["dtype"]
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])


def layer_weight(layer, suffix):
    """`suffix` as the builders name a MatMul (`linear_attn.in_proj_qkv/...`)."""
    mod = suffix.split("/")[0]
    return tensor(f"model.language_model.layers.{layer}.{mod}.weight")
