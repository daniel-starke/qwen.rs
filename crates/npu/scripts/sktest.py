"""A kernel of `../shave` on the NPU itself, against numpy and against another build.

    python sktest.py KERNEL.elf [REF.elf] [--kernel softmax|gnorm|trinv|rowq] [--iters 20]
        softmax (default): [--n 1024] [--nonorm] [--rows 64] [--heads 4]
        gnorm:             [--pos 64] [--heads 48] [--d 128]
        trinv:             [--mats 32] [--c 64] [--bs 16]
        rowq:              [--rows 256] [--l 12288] [--pad 0]

Run it in an empty work directory. The graph is the kernel's carrier op
alone, compiled with our compiler build (`vpuxc.ps1`, compiler patch 0008)
with KERNEL.elf as that op's kernel, and run with `npubench` (`NPUBENCH`,
default the release example), which holds the NPU lock. Printed: the
largest errors against numpy (float64 on the f16 input) and the SW tasks'
time from the profile. With REF.elf (another build of the kernel) the same
for it, and how far the two outputs are apart.

    softmax_lse.sk: y = Softmax(x, axis 3) on [1, heads, rows, n]
    gnorm.sk:       y = Minimum(a, b), a [1, pos, heads, 2, d], b [1, 1, heads, 2, d]
    trinv.sk:       y = LogSoftmax(x, axis 3) on [1, mats, 1, c * c]
    rowq.sk:        y = Softmax(x, axis 2) on [1, rows, l + pad]

    shave build ../shave/softmax_lse.sk new.elf -DN=1024
    shave build ../shave/rowq.sk new.elf -DPAD=32   (then --kernel rowq --pad 32)
    python sktest.py new.elf old.elf"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("NPUBENCH", os.path.join(HERE, "..", "..", "..", "target", "release", "examples", "npubench.exe"))


def opt(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v if isinstance(default, str) else int(v)
    return default


KERNEL, ITERS = opt("--kernel", "softmax"), opt("--iters", 20)
N, ROWS, HEADS = opt("--n", 1024), opt("--rows", 256 if KERNEL == "rowq" else 64), opt("--heads", 4 if KERNEL == "softmax" else 48)
POS, D = opt("--pos", 64), opt("--d", 128)
MATS, C, BS = opt("--mats", 32), opt("--c", 64), opt("--bs", 16)
L, PAD = opt("--l", 12288), opt("--pad", 0)
NONORM = "--nonorm" in sys.argv
if NONORM:
    sys.argv.remove("--nonorm")
EPS = np.float32(1e-3)


def port(i, shape, name=None):
    return ('<port id="%d" precision="FP16"%s>' % (i, ' names="%s"' % name if name else "")) + "".join(f"<dim>{d}</dim>" for d in shape) + "</port>"


def graph(inputs, op, attrs, out):
    """One op over the inputs (name -> shape)."""
    n = len(inputs)
    layers = "".join(f'<layer id="{i}" name="{nm}" type="Parameter" version="opset1"><data shape="{",".join(map(str, sh))}" element_type="f16"/><output>'
                     + port(0, sh, nm) + "</output></layer>" for i, (nm, sh) in enumerate(inputs.items()))
    layers += (f'<layer id="{n}" name="op" type="{op[0]}" version="{op[1]}"><data {attrs}/><input>' + "".join(port(i, sh) for i, sh in enumerate(inputs.values()))
               + "</input><output>" + port(n, out, "y") + f'</output></layer><layer id="{n + 1}" name="res" type="Result" version="opset1"><input>' + port(0, out)
               + "</input></layer>")
    edges = "".join(f'<edge from-layer="{i}" from-port="0" to-layer="{n}" to-port="{i}"/>' for i in range(n))
    edges += f'<edge from-layer="{n}" from-port="{n}" to-layer="{n + 1}" to-port="0"/>'
    return f'<?xml version="1.0"?><net name="m" version="11"><layers>{layers}</layers><edges>{edges}</edges></net>'


def softmax():
    shape = (1, HEADS, ROWS, N)
    rng = np.random.default_rng(1)
    # attention scores: a wide spread, some rows with one dominant key
    x = (rng.standard_normal((HEADS * ROWS, N)) * 4 - 6).astype(np.float16)
    x[::7, rng.integers(0, N - 2, 1)] += np.float16(12)
    xd = x.astype(np.float64)
    m = xd[:, :N - 2].max(1, keepdims=True)
    e = np.exp(xd[:, :N - 2] - m)
    s = e.sum(1, keepdims=True)
    p = e if NONORM else e / s

    def report(y):
        y = y.reshape(HEADS * ROWS, N)
        return (f"p max abs error {np.abs(y[:, :N - 2] - p).max():.2e} (largest p {p.max():.3f}), max wrong in {int((y[:, N - 2] != m[:, 0]).sum())} of {len(y)} rows, "
                f"sum max rel error {np.abs(y[:, N - 1] / s[:, 0] - 1).max():.2e}")
    return graph({"x": shape}, ("SoftMax", "opset8"), 'axis="3"', shape), {"x": x}, "softmax", report, lambda y: y


def gnorm():
    rng = np.random.default_rng(2)
    a = np.empty((POS, HEADS, 2, D), np.float16)
    a[:, :, 0] = rng.standard_normal((POS, HEADS, D)) * 4
    a[:, :, 1] = rng.standard_normal((POS, HEADS, D)) * 2
    b = np.zeros((HEADS, 2, D), np.float16)
    b[:, 0] = rng.uniform(0.5, 2.0, (HEADS, D))
    b.reshape(-1)[D:D + 2] = np.frombuffer(EPS.tobytes(), np.float16)
    o, z = a[:, :, 0].astype(np.float64), a[:, :, 1].astype(np.float64)
    want = o / np.sqrt((o * o).mean(2, keepdims=True) + float(EPS)) * b[:, 0].astype(np.float64) * z / (1 + np.exp(-z))

    def rows(y):
        # only rows (h, 0) are written
        return y.reshape(POS, HEADS, 2, D)[:, :, 0]

    def report(y):
        err = np.abs(rows(y) - want)
        return f"max abs error {err.max():.2e}, max rel error {(err / np.maximum(np.abs(want), 1e-2)).max():.2e} (largest |y| {np.abs(want).max():.1f})"
    return (graph({"a": (1, POS, HEADS, 2, D), "b": (1, 1, HEADS, 2, D)}, ("Minimum", "opset1"), 'auto_broadcast="numpy"', (1, POS, HEADS, 2, D)),
            {"a": a, "b": b}, "eltwise_min", report, rows)


def trinv():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((MATS, C, C)) * 0.2).astype(np.float16)
    i, j = np.meshgrid(np.arange(C), np.arange(C), indexing="ij")
    low = ((i // BS == j // BS) & (i > j)).astype(np.float64)
    want = np.linalg.inv(np.eye(C) + x.astype(np.float64) * low)

    def report(y):
        return f"max abs error {np.abs(y.reshape(MATS, C, C) - want).max():.2e} (largest |T| {np.abs(want).max():.2f})"
    shape = (1, MATS, 1, C * C)
    return graph({"x": shape}, ("LogSoftmax", "opset5"), 'axis="3"', shape), {"x": x}, "log_softmax", report, lambda y: y


def rowq():
    rows = ROWS
    rng = np.random.default_rng(4)
    w = L + PAD
    x = np.zeros((rows, w), np.float16)
    sc = np.array([1e-4, 0.05, 1.0, 9.0, 400.0, 0.0, 3e3])[np.arange(rows) % 7]
    x[:, :L] = rng.standard_normal((rows, L)) * sc[:, None]
    x[3, 77] = -900
    xd = x[:, :L].astype(np.float64)
    xd[np.abs(xd) < 6.1035e-5] = 0
    m = np.maximum(np.abs(xd).max(1), float(np.float16(1e-3)))
    want = xd * (127 / 128) / m[:, None]

    def report(y):
        y = y.reshape(rows, w)
        e = np.abs(y[:, :L] - want).max()
        s = f"y max abs error {e:.2e} (of at most 0.99)"
        if PAD:
            s += f", pad lanes max rel error {np.abs(y[:, L:] / m[:, None] - 1).max():.2e}"
        return s
    shape = (1, rows, w)
    return graph({"x": shape}, ("SoftMax", "opset8"), 'axis="2"', shape), {"x": x}, "softmax", report, lambda y: y


def run(name, elf, carrier):
    d = "k_" + name
    os.makedirs(d, exist_ok=True)
    data = open(elf, "rb").read()
    for f in (carrier + "_3720xx.elf", carrier + "_3720xx_lsu0_wo.elf"):
        open(os.path.join(d, f), "wb").write(data)
    env = dict(os.environ, NPU_RS_KERNEL_DIR=os.path.abspath(d).replace("\\", "/"), NPU_TASKS="1", NPU_INPUTS="in")
    p = subprocess.run(["powershell", "-NoProfile", "-File", os.path.join(HERE, "vpuxc.ps1"), "m.xml", name + ".blob", "-Profile"], env=env, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit("compile failed: " + (p.stdout + p.stderr)[-1500:])
    p = subprocess.run([BENCH, "blob", name + ".blob", "--iters", str(ITERS), "--dump", name + ".out", "--profile"], env=env, capture_output=True, text=True, timeout=300)
    if p.returncode:
        raise SystemExit("run failed: " + (p.stdout + p.stderr)[-1500:])
    open(name + ".log", "w", newline="\n").write(p.stdout)
    sw = [int(l.split()[3]) for l in p.stdout.splitlines() if l.startswith("T SW")]
    return np.frombuffer(open(name + ".out", "rb").read(), np.float16).astype(np.float64), sw


def main():
    if len(sys.argv) < 2 or KERNEL not in ("softmax", "gnorm", "trinv", "rowq"):
        raise SystemExit(__doc__)
    xml, inputs, carrier, report, part = {"softmax": softmax, "gnorm": gnorm, "trinv": trinv, "rowq": rowq}[KERNEL]()
    os.makedirs("in", exist_ok=True)
    open("m.xml", "w", newline="\n").write(xml)
    open("m.bin", "wb").close()
    for name, x in inputs.items():
        x.tofile(f"in/{name}.in")
    outs = []
    for k, elf in enumerate(sys.argv[1:3]):
        y, sw = run("new" if k == 0 else "ref", elf, carrier)
        print(f"{elf}: {report(y)}; {len(sw)} SW task(s), {sum(sw) / 1e6:.3f} ms in all, longest {max(sw) / 1e3:.0f} us")
        outs.append(part(y))
    if len(outs) == 2:
        d = np.abs(outs[0] - outs[1])
        print(f"the two builds: {int((d != 0).sum())} of {d.size} outputs differ, by at most {d.max():.2e}")


if __name__ == "__main__":
    main()
