"""Qwen3.8-Flash-Next: the PLE n-gram table's 128 checkpoint shards as one
row-major BF16 file (rows of 160), for the engine to read per token.

    python flash_ngram.py BF16_DIR OUT.bf16 [LAYER]

LAYER: the PLE layer index (0-based, default 1). Rows past the last shard
are not written (the table is padded to a multiple of 128 rows).
"""

import json
import os
import sys

d, out = sys.argv[1], sys.argv[2]
layer = int(sys.argv[3]) if len(sys.argv) > 3 else 1
pre = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_"
wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
n = sum(1 for k in wm if k.startswith(pre))
tmp = out + ".part"
rows = 0
with open(tmp, "wb") as o:
    for i in range(n):
        name = f"{pre}{i}.weight"
        fn = os.path.join(d, wm[name])
        with open(fn, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            e = json.loads(f.read(hl))[name]
            assert e["dtype"] == "BF16" and e["shape"][1] == 160, e
            a, b = e["data_offsets"]
            f.seek(8 + hl + a)
            left = b - a
            while left:
                c = f.read(min(left, 64 << 20))
                o.write(c)
                left -= len(c)
            rows += e["shape"][0]
        print(f"shard {i}: {rows} rows", flush=True)
os.replace(tmp, out)
print(f"{rows} rows x 160 -> {out}")
