#!/usr/bin/env python3
"""Summarize a VTune result of `cli` with our own functions named.

VTune names system DLLs from their PDBs, but a mingw-built `cli.exe` has none,
so its functions show as `func@0x140527...`. This script runs `vtune -report`
on the result and resolves those addresses against the executable's symbol
table (binutils `nm -n -C`: the function whose start is the closest at or
below the address; needs a binary that is not stripped, e.g. the
`target/release` one). Rust folds identical functions together (ICF), so an
address can carry an unrelated name of the same code; read such a name as
"the same machine code as".

Capture (system-wide, from an elevated shell; never launch `cli` under VTune,
its Level Zero interception loses the NPU): start the run, then

    vtune -collect xpu-offload -analyze-system -d 40 -knob profile-npu=true \
        -knob npu-profiling-mode=time-based -knob npu-metrics-set=NOC \
        -knob npu-sampling-interval=1 -knob trace-npu-programming-api=false \
        -knob collect-programming-api=false -knob collect-memory-bandwidth=true \
        -knob enable-characterization-insights=true -data-limit=0 -r RESULT

Then:

    python scripts/vtune_summary.py RESULT --exe path/to/the/cli.exe [--top N]
        [--vtune path/to/vtune.exe] [--nm path/to/nm.exe] [--no-summary]

Prints the summary report (elapsed time, GPU time, the hotspots while the GPU
was idle, DRAM bandwidth, NPU traffic) and the top functions by CPU time,
every `func@0x...` of the exe's module replaced by `name [0x...+offset]`
(other modules' addresses, e.g. the kernel's, stay as they are). The exe must be the
very binary that ran (a rebuild moves every address).
"""
import argparse
import bisect
import glob
import os
import re
import shutil
import subprocess
import sys


def find_vtune(arg):
    if arg:
        return arg
    p = shutil.which('vtune')
    if p:
        return p
    roots = [os.environ.get('VTUNE_PROFILER_DIR', '')]
    pf = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    roots += sorted(glob.glob(os.path.join(pf, 'Intel', 'oneAPI', 'vtune', '*')), reverse=True)
    for r in roots:
        c = os.path.join(r, 'bin64', 'vtune.exe')
        if r and os.path.exists(c):
            return c
    sys.exit('vtune not found: pass --vtune')


def find_nm(arg):
    if arg:
        return arg
    for n in ('nm', 'x86_64-w64-mingw32-nm'):
        p = shutil.which(n)
        if p:
            return p
    sys.exit('nm not found: pass --nm (binutils, e.g. mingw-w64 bin)')


def symbols(nm, exe):
    out = subprocess.run([nm, '-n', '-C', '--defined-only', exe],
                         capture_output=True, text=True, errors='replace').stdout
    syms = []
    for line in out.splitlines():
        m = re.match(r'([0-9a-fA-F]+) ([tTwW]) (.*)', line)
        if m:
            syms.append((int(m[1], 16), m[3].strip()))
    if not syms:
        sys.exit(f'{exe}: no symbols (stripped?)')
    syms.sort()
    return [a for a, _ in syms], [n for _, n in syms]


def resolver(addrs, names, width):
    def name(m):
        a = int(m[1], 16)
        i = bisect.bisect_right(addrs, a) - 1
        if i < 0:
            return m[0]
        n = names[i]
        if len(n) > width:
            n = n[:width - 3] + '...'
        off = a - addrs[i]
        return f'{n} [0x{addrs[i]:x}+{off:#x}]' if off else f'{n} [0x{a:x}]'
    return name


def report(vtune, result, args):
    r = subprocess.run([vtune] + args + ['-r', result], capture_output=True)
    for enc in ('utf-8', 'utf-16'):
        try:
            return r.stdout.decode(enc)
        except UnicodeDecodeError:
            pass
    return r.stdout.decode('latin-1')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('result')
    ap.add_argument('--exe', required=True)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--width', type=int, default=110)
    ap.add_argument('--vtune')
    ap.add_argument('--nm')
    ap.add_argument('--no-summary', action='store_true')
    a = ap.parse_args()
    vtune, nm = find_vtune(a.vtune), find_nm(a.nm)
    addrs, names = symbols(nm, a.exe)
    fix = resolver(addrs, names, a.width)
    pat = re.compile(r'func@0x([0-9a-fA-F]+)')
    if not a.no_summary:
        s = report(vtune, a.result, ['-report', 'summary', '-report-knob', 'show-issues=false'])
        exe = os.path.basename(a.exe).lower()
        for line in s.splitlines():
            print(pat.sub(fix, line) if exe in line.lower() else line)
    csv = report(vtune, a.result, ['-report', 'hotspots', '-group-by', 'function',
                                   '-format', 'csv', '-csv-delimiter', 'tab'])
    rows = [l.split('\t') for l in csv.splitlines() if '\t' in l]
    if not rows:
        sys.exit('no hotspots report:\n' + csv[:2000])
    head, body = rows[0], rows[1:]
    ci = next((i for i, h in enumerate(head) if h.strip() == 'CPU Time'), 1)
    mi = next((i for i, h in enumerate(head) if h.strip() == 'Module'), None)
    tot = sum(float(r[ci] or 0) for r in body if len(r) > ci)
    exe = os.path.basename(a.exe).lower()
    print(f'Top functions by CPU time (total {tot:.1f} CPU-s)')
    for r in body[:a.top]:
        t = float(r[ci] or 0)
        m = r[mi] if mi is not None and len(r) > mi else ''
        f = pat.sub(fix, r[0]) if m.lower() == exe or not m else r[0]
        print(f'{t:9.2f} s {100 * t / max(tot, 1e-9):5.1f} %  {f}  ({m})')


if __name__ == '__main__':
    main()
