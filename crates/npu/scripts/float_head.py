"""Float decode head for BF16 reference logits: the head graph HEAD_DIR/head.xml
(final RMSNorm + quantized lm_head) with the lm_head replaced by the BF16
checkpoint's `lm_head.weight` (as f16, widened to f32 for the MatMul).

    python float_head.py HEAD_DIR BF16_DIR OUT_DIR

Writes OUT_DIR/head.xml (+ .bin, ~2.5 GB). Use as `lean_chain.py --head OUT_DIR`
with the BF16 per-layer graphs to make a `truth.f32` for a new eval text."""
import json
import os
import struct
import sys

import numpy as np
import openvino as ov
import openvino.opset13 as op

head_dir, src, out = sys.argv[1:4]


def tensor(name):
    """One tensor of the safetensors checkpoint, bf16 -> f32."""
    shard = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"][name]
    with open(os.path.join(src, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = np.frombuffer(f.read(b - a), np.uint16)
    assert meta["dtype"] == "BF16", meta["dtype"]
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


w = tensor("lm_head.weight")
m = ov.Core().read_model(os.path.join(head_dir, "head.xml"))
mm = [n for n in m.get_ordered_ops() if n.get_type_name() == "MatMul"]
assert len(mm) == 1, "expected one MatMul (the lm_head)"
mm = mm[0]
assert tuple(mm.input(1).get_partial_shape().to_shape()) == w.shape, (mm.input(1).get_partial_shape(), w.shape)
mm.input(1).replace_source_output(op.convert(op.constant(w.astype(np.float16)), ov.Type.f32).output(0))
os.makedirs(out, exist_ok=True)
ov.save_model(m, os.path.join(out, "head.xml"), compress_to_fp16=False)
print(f"lm_head {w.shape} (bf16 -> f16) -> {out}/head.xml")
