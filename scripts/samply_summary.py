#!/usr/bin/env python3
"""Summarize a samply profile of `cli` by CPU time, with our own frames symbolized.

samply symbolizes system DLLs (PDB) but not a mingw-built `cli.exe`, whose
debug info is DWARF. This script resolves those frames with binutils
`addr2line` (inline chains included, each frame tagged with its source file
and line), then prints, per time window, the CPU time by thread kind, by leaf
frame and by inclusive function. Optionally it writes collapsed stacks (one
`frame;frame;... weight` line per stack) for flame-graph tools.

Build a binary that carries symbols, in its own target directory so the
shipped `target/deploy` is not touched:

    CARGO_PROFILE_DEPLOY_DEBUG=line-tables-only CARGO_PROFILE_DEPLOY_STRIP=false \
    CARGO_TARGET_DIR=<prof-target> cargo build --profile deploy -p cli --features vulkan

Record (samply asks for elevation once on Windows; ETW needs it):

    samply record -s --unstable-presymbolicate -o run.json.gz -- <prof-target>/deploy/cli.exe MODEL --serve ...

`--unstable-presymbolicate` writes `run.json.syms.json` beside the profile;
without it system frames show as `lib.dll` only.

Then:

    python scripts/samply_summary.py run.json.gz [--exe <prof-target>/deploy/cli.exe]
        [--window NAME=START:END]... [--marks FILE] [--top N] [--folded OUT.txt]

Windows: `--window decode=12.5:40` is seconds since the profile's start;
`--marks FILE` takes lines `NAME START END` in Unix epoch seconds (what a
client script can print around its requests). Windows sharing a NAME are
summed (overlaps counted once), and the wall time and cores busy are printed
for each NAME. A sample inside windows of several NAMEs counts in each.
Without any window the whole profile is one window.

CPU time is each sample's `threadCPUDelta` (converted to microseconds from the
profile's declared unit), so idle samples weigh nothing. Every own frame is
looked up one byte before its address: a caller's is a return address, and a
sample's is the instruction after the one executing (checked on the
disassembly: a `pause` spin samples on the instruction after the `pause`). A
leaf sampled on a function's first instruction keeps its own address. A hot
leaf whose instruction reads memory (e.g. `vpxor (mem), ...`) usually means
that load stalls. Thread kinds and inclusive times are per function and source
file; leaves and folded stacks keep the line.

The addr2line results are cached beside the profile, keyed by the binary's
size and modification time; a rebuilt binary invalidates the cache.
"""
import argparse
import bisect
import collections
import gzip
import json
import os
import re
import struct
import subprocess
import sys

CACHE_HEAD = '# samply_summary addr2line v3 exe '
LOC = re.compile(r'(.*):(\d+|\?)(?: \(discriminator \d+\))?')
CLOSURE = re.compile(r'\{closure#\d+\}( @.*)?')
# threadCPUDelta units -> microseconds
UNITS = {'µs': 1.0, 'μs': 1.0, 'us': 1.0, 'ns': 1e-3, 'ms': 1e3}


def load_json(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def pe_image_base(path):
    """ImageBase from a PE file's optional header (PE32+ or PE32)."""
    with open(path, 'rb') as f:
        mz = f.read(0x40)
        if len(mz) < 0x40 or mz[:2] != b'MZ':
            raise ValueError(f'{path}: not a PE file')
        off = struct.unpack_from('<I', mz, 0x3C)[0]
        f.seek(off)
        h = f.read(24 + 32)  # signature + COFF header, optional header through ImageBase
    if len(h) < 56 or h[:4] != b'PE\0\0':
        raise ValueError(f'{path}: not a PE file')
    magic = struct.unpack_from('<H', h, 24)[0]
    if magic == 0x20B:
        return struct.unpack_from('<Q', h, 24 + 24)[0]
    if magic == 0x10B:
        return struct.unpack_from('<I', h, 24 + 28)[0]
    raise ValueError(f'{path}: unknown optional header magic {magic:#x}')


def exe_ident(exe):
    st = os.stat(exe)
    return f'{st.st_size} {st.st_mtime_ns}'


def symbolize(exe, addr2line, addrs, cache_path):
    """lookup rva -> [(function, 'file:line'), ...] innermost first, cached on disk.

    `addrs` are the addresses to look up as given (callers already moved back
    into their call instruction)."""
    ident = exe_ident(exe) if exe and os.path.exists(exe) else None
    names = {}
    unverified = False
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            head = f.readline().rstrip('\n')
            cached = head[len(CACHE_HEAD):] if head.startswith(CACHE_HEAD) else None
            if cached is not None and (ident is None or cached == ident):
                unverified = ident is None
                for line in f:
                    a, _, n = line.rstrip('\n').partition('\t')
                    names[int(a)] = [tuple(x.split('\x02', 1)) for x in n.split('\x01')] if n else []
    todo = sorted(a for a in addrs if a not in names)
    if todo:
        if ident is None:
            sys.exit(f'binary {exe!r} not found: pass --exe')
        base = pe_image_base(exe)
        back = {base + a: a for a in todo}
        inp = ''.join(f'{base + a:#x}\n' for a in todo)
        try:
            out = subprocess.run([addr2line, '-a', '-i', '-f', '-C', '-e', exe], input=inp,
                                 capture_output=True, text=True, encoding='utf-8',
                                 errors='replace', check=True).stdout
        except FileNotFoundError:
            sys.exit(f'{addr2line} not found: pass --addr2line (binutils / mingw-w64)')
        cur = None
        fn = None
        for line in out.split('\n'):
            line = line.rstrip('\r')
            if re.fullmatch(r'0x[0-9a-fA-F]+', line):
                cur = back.get(int(line, 16))
                if cur is not None:
                    names[cur] = []
                fn = None
                continue
            if cur is None or not line:
                continue
            # addr2line -f -i prints function / file:line pairs, innermost first.
            if fn is None:
                fn = line
            else:
                m = LOC.fullmatch(line)
                where = re.split(r'[\\/]', m[1])[-1] + ':' + m[2] if m else line
                names[cur].append((fn, where))
                fn = None
        if cache_path:
            with open(cache_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(CACHE_HEAD + ident + '\n')
                for a, n in names.items():
                    f.write(f'{a}\t' + '\x01'.join(f'{x}\x02{w}' for x, w in n) + '\n')
    elif unverified:
        print(f'warning: {exe!r} not found; using the addr2line cache unverified', file=sys.stderr)
    return names


def short(n):
    n = re.sub(r'::h[0-9a-f]{16}', '', n)
    n = re.sub(r'\{closure_env#\d+\}', '', n)
    for _ in range(3):
        n = re.sub(r'<[^<>]*>', '<>', n)
    return n


# Frames that say nothing about where the time goes: runtime glue, intrinsics.
NOISE = re.compile(r'^(std::|core::|alloc::|<std::|<alloc::|<core::|__|call_once|catch_unwind|'
                   r'do_call|spawn_unchecked|\{closure#\d+\}<>|_mm|copy_nonoverlapping|atomic_|'
                   r'fold<|sum<|map_fold|\?\?|<>::new::thread_start|\?)')


def sample_times(s):
    if 'time' in s:
        return s['time']
    out, acc = [], 0.0
    for d in s['timeDeltas']:
        acc += d
        out.append(acc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('profile')
    ap.add_argument('--exe', help='the profiled binary with DWARF (default: its path in the profile)')
    ap.add_argument('--addr2line', default='addr2line')
    ap.add_argument('--process', default='cli',
                    help='process name to keep, with or without .exe (default cli)')
    ap.add_argument('--window', action='append', default=[], metavar='NAME=START:END')
    ap.add_argument('--marks', help='file of "NAME START END" lines, Unix epoch seconds')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--folded', help='write collapsed stacks (CPU us) to this file')
    a = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    p = load_json(a.profile)
    stem = re.sub(r'\.json(\.gz)?$', '', a.profile)
    syms_path = stem + '.json.syms.json'
    syms = load_json(syms_path) if os.path.exists(syms_path) else {'string_table': [], 'data': []}
    meta = p['meta']
    t0 = meta['startTime']  # ms since the epoch
    libs = p['libs']
    shared_strings = (p.get('shared') or {}).get('stringArray')

    def strings(t):
        s = t.get('stringArray')
        if s is None:
            s = shared_strings
        if s is None:
            sys.exit('profile has no stringArray (per thread or shared)')
        return s

    want = a.process.lower()

    def name_matches(n):
        n = (n or '').lower()
        return n == want or n == want + '.exe'

    windows = []  # (name, start ms, end ms) relative to the profile's start
    for w in a.window:
        m = re.fullmatch(r'([^=]+)=([\d.eE+]+):([\d.eE+]+)', w)
        if not m:
            sys.exit(f'bad --window {w!r}: NAME=START:END in seconds')
        windows.append((m[1], float(m[2]) * 1e3, float(m[3]) * 1e3))
    if a.marks:
        with open(a.marks, encoding='utf-8-sig') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    windows.append((parts[0], float(parts[1]) * 1e3 - t0, float(parts[2]) * 1e3 - t0))
    if not windows:
        windows.append(('all', float('-inf'), float('inf')))
    order = list(dict.fromkeys(n for n, _, _ in windows))

    # the profile's span, so a window reaching past it counts only the recorded part
    end = meta.get('profilingEndTime')
    if not end:
        end = 0.0
        for t in p['threads']:
            ts = sample_times(t['samples'])
            end = max(end, ts[-1] if ts else 0.0, t.get('unregisterTime') or 0.0)
    span = (meta.get('profilingStartTime', 0.0), end)
    per_name = collections.defaultdict(list)
    for n, s, e in windows:
        s, e = max(s, span[0]), min(e, span[1])
        if e > s:
            per_name[n].append((s, e))
    wall = collections.Counter()
    for n, iv in per_name.items():  # union of each name's intervals
        iv.sort()
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                wall[n] += (ce - cs) / 1e3
                cs, ce = s, e
            else:
                ce = max(ce, e)
        wall[n] += (ce - cs) / 1e3

    def windows_of(tm):
        return {n for n, s, e in windows if s <= tm < e}

    st = syms['string_table']
    libsym = {}  # debug name -> (starts, ends or None, names)
    for d in syms['data']:
        tab = sorted((x['rva'], x.get('size'), st[x['symbol']]) for x in d['symbol_table'])
        libsym[d['debug_name']] = ([r for r, _, _ in tab],
                                   [r + z if z else None for r, z, _ in tab],
                                   [n for _, _, n in tab])

    threads = [t for t in p['threads'] if name_matches(t.get('processName'))]
    if not threads:
        sys.exit(f'no threads of a process named {a.process}')
    cand = [i for i, l in enumerate(libs) if name_matches(l.get('name'))]
    paths = {libs[i].get('path') for i in cand}
    if a.exe:
        exe = a.exe
        sym_libs = set(cand)
        if len(paths) > 1:
            print(f'warning: {len(paths)} different {a.process} binaries in the profile; '
                  f'all symbolized with {exe}', file=sys.stderr)
    else:
        exe = libs[cand[0]].get('path') if cand else None
        sym_libs = {i for i in cand if libs[i].get('path') == exe}
        if len(paths) > 1:
            print(f'warning: {len(paths)} different {a.process} binaries in the profile; '
                  f'only {exe} symbolized (pass --exe to choose)', file=sys.stderr)

    def own_lib(t, fi):
        r = t['funcTable']['resource'][t['frameTable']['func'][fi]]
        if r is None or r < 0:
            return False
        return t['resourceTable']['lib'][r] in sym_libs

    addrs = set()
    for t in threads:
        ft, stk = t['frameTable'], t['stackTable']
        leaf = {stk['frame'][x] for x in t['samples']['stack'] if x is not None}
        caller = {stk['frame'][y] for y in stk['prefix'] if y is not None}
        for fi in leaf | caller:
            if own_lib(t, fi):
                addrs.add(max(ft['address'][fi] - 1, 0))
        for fi in leaf:
            if own_lib(t, fi):
                addrs.add(ft['address'][fi])
    names = symbolize(exe, a.addr2line, addrs, stem + '.addr2line.cache') if addrs else {}

    def lookup(addr, leaf):
        """Symbolized frames for a frame's address. Callers: one byte back, inside
        the call. Leaves: the sample's address is the instruction after the one
        executing, so one byte back too, unless that byte is in another function
        (the sample sits on a function's first instruction)."""
        got = names.get(max(addr - 1, 0))
        if leaf:
            at = names.get(addr)
            if at and (not got or got[-1][0] != at[-1][0]):
                return at
        return got

    def frames(t, i, leaf):
        """Frames for frame index i, innermost first, as (label, function, own)."""
        ft, fn, rt = t['frameTable'], t['funcTable'], t['resourceTable']
        f = ft['func'][i]
        r = fn['resource'][f]
        addr = ft['address'][i]
        li = rt['lib'][r] if r is not None and r >= 0 else None
        if li is None:
            n = strings(t)[fn['name'][f]]
            return ((n, n, False),)
        lib = libs[li]
        if li in sym_libs:
            got = lookup(addr, leaf)
            if got:
                # label with the line; function key with the file only (bare
                # names such as {closure#0} or worker are not unique)
                return tuple((f'{s} @{w}', f'{s} @{w.rsplit(":", 1)[0]}', True)
                             for s, w in ((short(x), w) for x, w in got))
            n = f'{lib["name"]}+{addr:#x}'
            return ((n, n, False),)
        tab = libsym.get(lib.get('debugName'))
        if tab:
            k = bisect.bisect_right(tab[0], addr) - 1
            if k >= 0 and (tab[1][k] is None or addr < tab[1][k]):
                n = lib['name'] + '!' + tab[2][k]
                return ((n, n, False),)
        return ((lib['name'], lib['name'], False),)

    unit = (meta.get('sampleUnits') or {}).get('threadCPUDelta', 'µs')
    scale = UNITS.get(unit)
    tot = collections.Counter()
    self_t = collections.defaultdict(collections.Counter)
    incl_t = collections.defaultdict(collections.Counter)
    kind_t = collections.defaultdict(collections.Counter)
    folded = collections.Counter()
    nthreads = collections.Counter()
    for t in threads:
        s = t['samples']
        stk = t['stackTable']
        fcache = {}
        cchain = {}  # stack index -> its frames as callers, leaf first
        info = {}  # sample stack index -> (leaf label, kind, inclusive functions, folded key)

        def frames_of(fi, leaf):
            k = (fi, leaf)
            if k not in fcache:
                fcache[k] = frames(t, fi, leaf)
            return fcache[k]

        def caller_chain(y):
            path = []
            while y is not None and y not in cchain:
                path.append(y)
                y = stk['prefix'][y]
            tail = cchain[y] if y is not None else ()
            for z in reversed(path):
                tail = frames_of(stk['frame'][z], False) + tail
                cchain[z] = tail
            return tail

        def stack_info(x):
            chain = frames_of(stk['frame'][x], True) + caller_chain(stk['prefix'][x])
            # thread kind: the outermost own function that is not glue or a bare closure
            kind = next((fn for _, fn, own in reversed(chain)
                         if own and not NOISE.match(fn) and not CLOSURE.fullmatch(fn)), chain[-1][0])
            incl = frozenset(fn for _, fn, own in chain if own and not NOISE.match(fn))
            fold = ';'.join(lb.replace(';', ',') for lb, _, _ in reversed(chain)) if a.folded else None
            return chain[0][0], kind, incl, fold

        cpu_d = s.get('threadCPUDelta')
        if cpu_d is not None and scale is None:
            sys.exit(f'unsupported threadCPUDelta unit {unit!r}')
        weights = s.get('weight')
        times = sample_times(s)
        seen = set()
        for k in range(s['length']):
            if cpu_d is not None:
                cpu = (cpu_d[k] or 0) * scale
            else:  # no CPU deltas recorded: count samples x interval
                cpu = meta.get('interval', 1.0) * 1e3 * (weights[k] if weights else 1)
            x = s['stack'][k]
            if cpu <= 0 or x is None:
                continue
            ws = windows_of(times[k])
            if not ws:
                continue
            seen |= ws
            if x not in info:
                info[x] = stack_info(x)
            leaf, kind, incl, fold = info[x]
            for w in ws:
                tot[w] += cpu
                self_t[w][leaf] += cpu
                for n in incl:
                    incl_t[w][n] += cpu
                kind_t[w][kind] += cpu
            if fold is not None:
                folded[fold] += cpu
        for w in seen:
            nthreads[w] += 1

    for w in order:
        T = tot[w]
        if not T:
            continue
        head = f'\n===== {w}: CPU {T / 1e6:.1f} s'
        if wall[w]:
            head += f' over {wall[w]:.1f} s wall = {T / 1e6 / wall[w]:.1f} cores busy'
        print(head + f', {nthreads[w]} threads')
        print('  -- by thread kind:')
        for k, v in kind_t[w].most_common(12):
            print(f'   {v / T * 100:5.1f}%  {k}')
        print('  -- self (leaf frame):')
        for k, v in self_t[w].most_common(a.top):
            print(f'   {v / T * 100:5.1f}%  {k[:200]}')
        print('  -- inclusive (own functions):')
        for shown, (k, v) in enumerate(incl_t[w].most_common()):
            if shown >= 2 * a.top or v / T < 0.005:
                break
            print(f'   {v / T * 100:5.1f}%  {k[:200]}')
    if a.folded:
        with open(a.folded, 'w', encoding='utf-8', newline='\n') as f:
            for k, v in folded.most_common():
                f.write(f'{k} {int(v)}\n')


if __name__ == '__main__':
    main()
