"""ringsim.py FILE.elf.mlir [--model all|fin] [--half-dpu 16] [--half-sw 16] [--half-dma 128]

Simulates the NPU37XX runtime's descriptor rings over a lowered ELF IR
(vpuxc.ps1 -Keep): one DPU invariant list, one ActShave list, one DMA list per
port, each a ring loaded in halves. Half k + 2 of a list is loaded only when
every task of half k is done (--model fin) or has its clean_after barrier
consumed (--model all). Reports a deadlock and the tasks / barriers involved.
"""
import re, sys
from collections import defaultdict

a = sys.argv[1:]
def opt(n, d, c=str):
    if n in a:
        i = a.index(n); v = a[i + 1]; del a[i:i + 2]; return c(v)
    return d
MODEL = opt("--model", "all")
INORDER = "--inorder" in a
a = [v for v in a if v != "--inorder"]
HALF = {"var": opt("--half-var", 128, int), "dpu": opt("--half-dpu", 16, int), "sw": opt("--half-sw", 16, int),
        "dma0": opt("--half-dma", 128, int), "dma1": opt("--half-dma", 128, int)}
txt = open(a[0]).read().split("\n")

bars = {}  # id -> [producers, consumers, pid, next]
tasks = []  # dict(list, idx, fifo, waits, updates, clean, start)
idxre = re.compile(r"Index<0:0:(\d+)>")
nvar = defaultdict(int)
for ln in txt:
    m = re.search(r'"VPUMI37XX.DPUVariant"\((%\d+)\)', ln)
    if m:
        nvar[m.group(1)] += 1
for ln in txt:
    if "VPUMI37XX.ConfigureBarrier" in ln:
        m = re.search(r"consumer_count = (\d+) : ui8, producer_count = (\d+) : ui8}\s*<(\d+), (-?\d+)>[^-]*-> !VPURegMapped.Index<0:0:(\d+)>", ln)
        if m is None:
            print("unparsed barrier:", ln.strip()[:200]); sys.exit(1)
        c, p, pid, nxt, i = map(int, m.groups())
        bars[i] = [p, c, pid, nxt]
        continue
    kind = None
    if "VPUMI37XX.DPUInvariant" in ln:
        kind = "dpu"
    elif "VPUMI37XX.ActKernelInvocation" in ln:
        kind = "sw"
    elif "= VPUMI37XX.NNDMA" in ln:
        kind = "dma"
    if kind is None:
        continue
    w = re.search(r"waits\(([^)]*)\)", ln); u = re.search(r"updates\(([^)]*)\)", ln)
    waits = [int(x) for x in idxre.findall(w.group(1))] if w else []
    upd = [int(x) for x in idxre.findall(u.group(1))] if u else []
    if kind == "dpu":
        clean = int(re.search(r"clean_after = (\d+)", ln).group(1)); start = int(re.search(r"start_after = (\d+)", ln).group(1))
        res = re.search(r"-> <(\d+):(\d+):(\d+)>", ln)
        cl = re.search(r"input\(%\d+ : memref<[^>]*\[@CMX_NN, (\d)\]", ln) or re.search(r"\[@CMX_NN, (\d)\]", ln)
        lst, fifo = "dpu", ("dpu" if INORDER else f"dpu{cl.group(1)}")
    elif kind == "sw":
        clean = int(re.search(r"clean_after\((\d+)\)", ln).group(1)); start = int(re.search(r"start_after\((\d+)\)", ln).group(1))
        res = re.search(r"-> !VPURegMapped.Index<(\d+):(\d+):(\d+)>$", ln.strip())
        lst, fifo = "sw", ("sw" if INORDER else "sw" + re.search(r"tile\((\d+)\)", ln).group(1))
    else:
        clean = int(re.search(r"clean_after\((\d+)\)", ln).group(1)); start = int(re.search(r"start_after\((\d+)\)", ln).group(1))
        res = re.search(r"-> !VPURegMapped.Index<(\d+):(\d+):(\d+)>$", ln.strip())
        lst = fifo = "dma" + res.group(2)
    hits = nvar[re.match(r"\s*(%\d+)", ln).group(1)] if kind == "dpu" else 1
    tasks.append(dict(hits=hits, list=lst, idx=int(res.group(3)), fifo=fifo, waits=waits, upd=upd, clean=clean, start=start, line=ln.strip()[:160]))

lists = defaultdict(list)
for t in tasks:
    lists[t["list"]].append(t)
for l in lists.values():
    l.sort(key=lambda t: t["idx"])
vstart = {}
v = 0
for t in lists["dpu"]:
    vstart[id(t)] = v; v += max(t["hits"], 1)
def var_ok(t):
    # all variants of t must be in loaded halves of the variant ring
    if t["list"] != "dpu":
        return True
    last = vstart[id(t)] + max(t["hits"], 1) - 1
    return last // HALF["var"] < vloaded[0]
vloaded = [2]
fifos = defaultdict(list)
for l in lists.values():
    for t in l:
        fifos[t["fifo"]].append(t)
print({k: len(v) for k, v in lists.items()}, {k: len(v) for k, v in fifos.items()}, "barriers", len(bars))

prod = {i: b[0] for i, b in bars.items()}; cons = {i: b[1] for i, b in bars.items()}
consumed = set(); released = set(i for i in bars if prod[i] == 0)
# physical barrier programming: virtual v is programmed once its predecessor with the same pid is consumed
prev = {}
for i in sorted(bars):
    if bars[i][3] >= 0:
        prev[bars[i][3]] = i
programmed = set(i for i in bars if i not in prev)
loaded = {k: 2 for k in lists}  # halves loaded
pos = {k: 0 for k in fifos}
done = set(); started = set()
def half(t):
    return t["idx"] // HALF[t["list"]]
progress = True
steps = 0
while progress:
    progress = False
    for f, q in fifos.items():
        if pos[f] >= len(q):
            continue
        t = q[pos[f]]
        if half(t) >= loaded[t["list"]]:
            continue
        if not var_ok(t):
            continue
        if any(b not in released for b in t["waits"]):
            continue
        if any(b not in programmed for b in range(t["start"])) if t["start"] < 0 else False:
            continue
        if any(b not in programmed for b in t["upd"] + t["waits"]):
            continue
        # start + finish
        for b in t["waits"]:
            cons[b] -= t["hits"]
            if cons[b] == 0:
                consumed.add(b)
                if bars[b][3] >= 0:
                    programmed.add(bars[b][3])
        for b in t["upd"]:
            prod[b] -= t["hits"]
            if prod[b] == 0:
                released.add(b)
        done.add(id(t)); pos[f] += 1; progress = True; steps += 1
    # variant ring reload: half h + 2 once every task owning a variant of half h is cleaned
    if "dpu" in lists:
        while True:
            h = vloaded[0] - 2
            own = [t for t in lists["dpu"] if vstart[id(t)] // HALF["var"] <= h and (vstart[id(t)] + max(t["hits"], 1) - 1) // HALF["var"] >= h]
            if not own or vloaded[0] * HALF["var"] >= v:
                break
            if not all(id(t) in done and t["clean"] in consumed for t in own):
                break
            vloaded[0] += 1; progress = True
    # ring reload
    for k, l in lists.items():
        while True:
            h = loaded[k] - 2
            seg = l[h * HALF[k]:(h + 1) * HALF[k]]
            if not seg or loaded[k] * HALF[k] >= len(l):
                break
            if MODEL == "fin":
                ok = all(id(t) in done for t in seg)
            else:
                ok = all(id(t) in done and t["clean"] in consumed for t in seg)
            if not ok:
                break
            loaded[k] += 1; progress = True
if all(pos[f] >= len(q) for f, q in fifos.items()):
    print("OK: no deadlock,", steps, "tasks")
else:
    print("DEADLOCK after", steps, "tasks")
    for f, q in fifos.items():
        if pos[f] < len(q):
            t = q[pos[f]]
            why = []
            if half(t) >= loaded[t["list"]]:
                h = loaded[t["list"]] - 2
                seg = lists[t["list"]][h * HALF[t["list"]]:(h + 1) * HALF[t["list"]]]
                bad = [(s["idx"], s["clean"], id(s) in done) for s in seg if not (id(s) in done and (MODEL == "fin" or s["clean"] in consumed))]
                why.append(f"half {half(t)} not loaded; half {h} blocked by (idx, clean_after, done) {bad[:6]}")
            nw = [b for b in t["waits"] if b not in released]
            if nw:
                why.append(f"waits on unreleased {nw} (producers left {[prod[b] for b in nw]})")
            print(f"  {f}: next idx {t['idx']} ({pos[f]}/{len(q)}):", "; ".join(why))
