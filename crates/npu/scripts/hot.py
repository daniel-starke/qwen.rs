"""Heaviest ops of an `NPU_TASKS=1 npubench ... --profile` dump, joined with
their type and input shapes from the IR (`XML`), plus the time per op class.

    python hot.py DUMP XML [N]

Columns: busy ms (sum of task durations), window ms (first start to last
end), engines, op name, type, input shapes."""
import collections
import re
import sys
import openvino as ov

f, xml = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = []
for line in open(f, encoding="utf-8", errors="replace"):
    if not line.startswith("T "):
        continue
    _, eng, st, du, rest = line.rstrip("\n").split(" ", 4)
    op = rest.rsplit(" [", 1)[0].split("?", 1)[0]
    rows.append((eng, int(st), int(du), op))
t0 = min(r[1] for r in rows)
span = max(r[1] + r[2] for r in rows) - t0
info = {}
for o in ov.Core().read_model(xml).get_ordered_ops():
    info[o.get_friendly_name()] = (o.get_type_name(), [str(o.input_value(i).get_partial_shape()).replace(" ", "") for i in range(o.get_input_size())])
agg = {}
for eng, st, du, op in rows:
    a = agg.setdefault(op, [0, st, st + du, set()])
    a[0] += du
    a[1] = min(a[1], st)
    a[2] = max(a[2], st + du)
    a[3].add(eng)
print(f"{len(rows)} tasks, span {span / 1e6:.2f} ms")
cls = collections.defaultdict(float)
for op, (du, a, b, e) in agg.items():
    ty = info.get(op, ("?", []))[0]
    cls[ty] += (b - a) / 1e6
print("window ms by type:", ", ".join(f"{k} {v:.1f}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])[:10]))
for op, (du, a, b, e) in sorted(agg.items(), key=lambda kv: -(kv[1][2] - kv[1][1]))[:top]:
    ty, sh = info.get(op, ("?", []))
    print(f"  {du / 1e6:7.2f} {(b - a) / 1e6:7.2f}  {'/'.join(sorted(e)):10s} {op[:40]:40s} {ty:14s} {' '.join(sh)[:70]}")
