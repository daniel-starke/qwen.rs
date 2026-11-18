"""A graph set whose int4 / int8 weight inputs are read pre-swizzled
(compiler patch 0009, `NPU_RS_SWZIN`): the DPU then reads them at the
speed of compile-time constants (+22 % on the FC MatMuls).

    python swzset.py OUT SRC WDIR [--decode]

OUT gets every file of SRC as a hard link, except the blobs, which are
compiled again with `NPU_RS_SWZIN=1 NPU_RS_SWZKEY=2` (prefill sets:
`lean{L}.xml` -> `lean{L}.blob`; `--decode`: `layer{L}.xml` ->
`layer{L}.blob`), and the weight files of every input the compiler swizzled,
which are replaced by swizzled copies. A swizzled copy is written once to
WDIR (`<name>.k2`, keyed by the plain file's inode) and hard-linked from
there, so every set that shares a weight shares its swizzled copy too.
Each compile's swizzled inputs (by name) go to OUT/<blob>.swz.

Compile options: prefill and decode delta layers with the shipped prefill
options (grouped MatMul, patches 0006 / 0007, `NPU_RS_KERNEL_DIR`), decode
attention layers with swizzling only (the grouped-MatMul options make the
s = 4 attention 50 % slower). A blob is compiled once per distinct graph
file (inode) and hard-linked from WDIR/blobs.

`swzcheck.py` checks that every weight buffer the engine shares between
graphs (prefill sets and decode sets) is read in one form by all of them."""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swz import swizzle  # noqa: E402

KEY = 2
HERE = os.path.dirname(os.path.abspath(__file__))
VPUXC = os.path.join(HERE, "vpuxc.ps1")
PREFILL_ENV = {
    "NPU_RS_DPU1LIST": "1",
    "NPU_RS_GMM_MAXG": "96",
    "NPU_RS_VIEWFOLD": "1",
    "NPU_RS_DISTVIEW": "1",
    "NPUC_OPTS": "enable-grouped-matmul=true",
    "NPU_RS_KERNEL_DIR": os.environ.get("NPU_RS_KERNEL_DIR", ""),
}


def params(xml):
    """Input names of an IR graph, in order (= the compiler's network inputs)."""
    import openvino as ov

    return [p.get_friendly_name() for p in ov.Core().read_model(xml).get_parameters()]


def compile_blob(xml, blob, swzlog, attention_decode):
    env = dict(os.environ)
    for k in list(PREFILL_ENV) + ["NPU_RS_SWZIN", "NPU_RS_SWZKEY", "NPU_RS_SWZIN_LOG"]:
        env.pop(k, None)
    if not attention_decode:
        env.update(PREFILL_ENV)
    env.update(NPU_RS_SWZIN="1", NPU_RS_SWZKEY=str(KEY), NPU_RS_SWZIN_LOG=swzlog)
    log = blob + ".log"
    with open(log, "w") as f:
        r = subprocess.run(["powershell", "-File", VPUXC, xml, blob], env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not os.path.exists(blob):
        raise SystemExit(f"compile of {xml} failed, see {log}")


def main():
    a = sys.argv[1:]
    decode = "--decode" in a
    a = [v for v in a if v != "--decode"]
    out, src, wdir = a
    if not PREFILL_ENV["NPU_RS_KERNEL_DIR"]:
        sys.exit("NPU_RS_KERNEL_DIR: the directory with the SHAVE kernels (gnorm, trinv)")
    os.makedirs(out, exist_ok=True)
    # SWZ_BLOBS: the blob cache's directory name (another one per compiler /
    # kernel version; the weight files stay shared)
    blobs = os.environ.get("SWZ_BLOBS", "blobs")
    os.makedirs(os.path.join(wdir, blobs), exist_ok=True)
    stem = "layer" if decode else "lean"
    for f in os.listdir(src):
        if f.endswith(".blob") and f.startswith(stem) or f.endswith(".swz"):
            continue
        d = os.path.join(out, f)
        if not os.path.exists(d):
            os.link(os.path.join(src, f), d)
    # compile every graph (once per distinct xml inode)
    jobs = []
    for L in range(64):
        if os.path.exists(os.path.join(out, f"{stem}{L}.use")):
            continue
        xml = os.path.join(out, f"{stem}{L}.xml")
        ino = os.stat(xml).st_ino
        tag = "d" if decode else "p"
        cb = os.path.join(wdir, blobs, f"{tag}{ino}.blob")
        jobs.append((L, xml, cb, decode and L % 4 == 3))
    todo = [j for j in jobs if not os.path.exists(j[2])]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(int(os.environ.get("JOBS", "6"))) as ex:
        list(ex.map(lambda j: compile_blob(j[1], j[2][:-5] + ".tmp.blob", j[2] + ".idx", j[3]) or os.replace(j[2][:-5] + ".tmp.blob", j[2]), todo))
    for L, xml, cb, _ in jobs:
        names = params(xml)
        idx = [int(v) for v in open(cb + ".idx").read().split()]
        swz = [names[i] for i in idx]
        b = os.path.join(out, f"{stem}{L}.blob")
        if os.path.exists(b):
            os.remove(b)
        os.link(cb, b)
        with open(os.path.join(out, f"{stem}{L}.swz"), "w") as f:
            f.write("\n".join(swz) + "\n")
    # swizzled weight files, for every layer that uses a blob
    nsw = 0
    for L in range(64):
        u = os.path.join(out, f"{stem}{L}.use")
        g = open(u).read().strip() if os.path.exists(u) else f"{stem}{L}"
        for name in open(os.path.join(out, f"{g}.swz")).read().split():
            # decode graphs bind the prefill set's files (`bindw.py`): nothing to write
            p = os.path.join(out, f"{stem}{L}.{name}.in")
            if not os.path.exists(p):
                continue
            ino = os.stat(p).st_ino
            k = os.path.join(wdir, f"{ino}.k{KEY}")
            if not os.path.exists(k):
                tmp = k + ".tmp"
                with open(p, "rb") as f:
                    data = f.read()
                with open(tmp, "wb") as f:
                    f.write(swizzle(data, KEY))
                os.replace(tmp, k)
            os.remove(p)
            os.link(k, p)
            nsw += 1
    print(f"{out}: {len(jobs)} graphs ({len(todo)} compiled), {nsw} weight files swizzled")


if __name__ == "__main__":
    main()
