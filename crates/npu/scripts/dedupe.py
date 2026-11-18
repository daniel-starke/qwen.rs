"""Group the lean layer graphs of DIR (lean{L}.xml + .bin) that are byte-
identical (`build_delta.py --allin`: every per-layer constant an input) and
write `lean{L}.use` = the representative's name for each duplicate, so only
the representatives are compiled (`compile_lean.sh`) and `npubench prefill`
loads one instance per distinct blob.

    python dedupe.py DIR"""
import hashlib
import os
import sys

d = sys.argv[1]
groups = {}
L = 0
while os.path.exists(os.path.join(d, f"lean{L}.xml")):
    h = hashlib.sha256()
    for ext in ("xml", "bin"):
        h.update(open(os.path.join(d, f"lean{L}.{ext}"), "rb").read())
    groups.setdefault(h.hexdigest(), []).append(L)
    L += 1
for ls in groups.values():
    for l in ls[1:]:
        open(os.path.join(d, f"lean{l}.use"), "w").write(f"lean{ls[0]}")
    for l in ls[:1]:
        p = os.path.join(d, f"lean{l}.use")
        if os.path.exists(p):
            os.remove(p)
print(f"{L} layers, {len(groups)} distinct graphs: " + ", ".join(f"lean{ls[0]} x{len(ls)}" for ls in groups.values()))
