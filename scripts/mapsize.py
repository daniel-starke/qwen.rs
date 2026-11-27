#!/usr/bin/env python3
"""Break a GNU ld linker map of `cli` down by section, crate and symbol.

`crates/cli/build.rs` makes the linker write `cli.map` beside the binary
(`target/<profile>/cli.map`) on every link. This script reads it and prints

- the output sections' sizes (what the image is made of),
- per section, the bytes by crate: by the object a piece came from
  (`deps/<crate>-<hash>...rcgu.o`, `lib<crate>-<hash>.rlib(...)`, a mingw
  library), and, because a fat-LTO build (`--profile deploy`) puts every
  crate into one object, also by the crate the symbol names (its first path
  segment, `<T as Trait>` resolved to `T`'s crate),
- the largest symbols (a symbol's size = the distance to the next symbol or
  the end of its input section; anonymous constants as `anon.*`).

Symbols in the map are already demangled by ld, but the map lists only
global symbols: a local function's bytes count toward the global symbol
before it. `--exe` reads all symbols, locals included, from the binary that
was linked with this map (`nm -n -C`), which gives exact symbol sizes. A fat-LTO
build merges all crates' constants into unnamed pools, so its `.rdata`
symbol sizes run up to the next named symbol: read `.rdata` from a
`--release` map, where each constant keeps its `anon.*` symbol and object.
`--exe` needs an unstripped binary (`--release` is; for the shipped profile build
`CARGO_PROFILE_DEPLOY_STRIP=false cargo build --profile deploy ...`, in a
separate `CARGO_TARGET_DIR` so `target/deploy` stays the stripped one).

    python scripts/mapsize.py target/release/cli.map [--exe target/release/cli.exe]
        [--nm NM] [--top N] [--section .text]... [--crate NAME] [--grep REGEX]

`--crate` limits the symbol list to one crate (as attributed by name),
`--grep` to symbols matching a regex.
"""

import argparse
import bisect
import collections
import re
import subprocess
import sys

SECTIONS = (".text", ".rdata", ".data", ".pdata", ".xdata", ".bss", ".tls",
            ".idata", ".reloc", ".rsrc", ".edata", ".CRT")

OUT_RE = re.compile(r"^(\.[\w$.]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)")
OUT_NAME_RE = re.compile(r"^(\.[\w$.]+)\s*$")
IN_RE = re.compile(r"^ (\.[\w$.]+|COMMON)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(.+)$")
IN_NAME_RE = re.compile(r"^ (\.[\w$.]+|COMMON)\s*$")
IN_TAIL_RE = re.compile(r"^\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(.+)$")
SYM_RE = re.compile(r"^\s{16,}0x([0-9a-f]+)\s+(\S.*)$")
FILL_RE = re.compile(r"^ \*fill\*\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)")


def obj_crate(path):
    """The crate (or library) an input object belongs to."""
    p = path.replace("\\", "/")
    m = re.search(r"/lib([\w]+?)-[0-9a-f]{16}\.rlib\(", p)
    if m:
        return m.group(1)
    m = re.search(r"/deps/([\w]+?)-[0-9a-f]{16}\.", p)
    if m:
        return m.group(1)
    m = re.search(r"/lib([\w+-]+?)\.a\(", p)
    if m:
        return "lib" + m.group(1)
    base = p.rsplit("/", 1)[-1]
    return base or "?"


def strip_generics(s):
    """Remove the outermost <...> groups' contents, keeping their position."""
    out, depth = [], 0
    for c in s:
        if c == "<":
            depth += 1
            if depth == 1:
                out.append("<")
            continue
        if c == ">" and depth:
            depth -= 1
            if depth == 0:
                out.append(">")
            continue
        if depth == 0:
            out.append(c)
    return "".join(out)


def sym_crate(name):
    """The crate a demangled symbol path names."""
    if name.startswith("anon.") or name.startswith(".L"):
        return "(anon)"
    s = name.lstrip("&*").strip()
    # `<T as Trait>::f` / `<T>::f`: attribute to T.
    while s.startswith("<"):
        depth, i = 0, 0
        for i, c in enumerate(s):
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    break
        inner = s[1:i]
        head = strip_generics(inner).split(" as ")[0]
        s = head.lstrip("&*").replace("mut ", "", 1).replace("dyn ", "", 1).lstrip("[(").strip()
        if s.startswith("<"):
            continue
        break
    if "::" in s:
        return s.split("::", 1)[0]
    return "(C)"  # unmangled: C runtime, libgcc, import thunks


def parse(path):
    """Yield (out_section, in_section, obj, addr, size, [(addr, name)])."""
    out = None
    cur = None
    pending_in = None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = iter(f)
        for line in lines:
            line = line.rstrip("\n")
            m = OUT_RE.match(line)
            if m:
                out = m.group(1)
                continue
            m = OUT_NAME_RE.match(line)
            if m:
                out = m.group(1)
                continue
            if pending_in is not None:
                m = IN_TAIL_RE.match(line)
                pending, pending_in = pending_in, None
                if m:
                    if cur:
                        yield cur
                    cur = (out, pending, m.group(3), int(m.group(1), 16), int(m.group(2), 16), [])
                    continue
            m = IN_RE.match(line)
            if m:
                if cur:
                    yield cur
                cur = (out, m.group(1), m.group(4), int(m.group(2), 16), int(m.group(3), 16), [])
                continue
            m = IN_NAME_RE.match(line)
            if m:
                pending_in = m.group(1)
                continue
            m = SYM_RE.match(line)
            if m and cur and not m.group(2).startswith("PROVIDE") and "=" not in m.group(2):
                cur[5].append((int(m.group(1), 16), m.group(2).strip()))
                continue
            if FILL_RE.match(line) or line.startswith(" *("):
                continue
    if cur:
        yield cur


def nm_symbols(exe, nm):
    """Sorted (addr, name) of every defined code/data symbol in `exe`."""
    res = subprocess.run([nm, "-n", "-C", "--defined-only", exe],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3 or parts[1].lower() not in "tdrb":
            continue
        name = parts[2]
        if name.startswith(".") or name.startswith("__") and name.endswith("__"):
            continue
        out.append((int(parts[0], 16), name))
    out.sort()
    return out


def human(n):
    for unit in ("B", "KiB", "MiB"):
        if abs(n) < 1024 or unit == "MiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map")
    ap.add_argument("--exe", help="the binary linked with this map: exact symbol sizes from nm")
    ap.add_argument("--nm", default="nm", help="binutils nm (default: nm on PATH)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--section", action="append", help="sections for the crate/symbol tables (default .text .rdata)")
    ap.add_argument("--crate", help="symbol list: only this crate (by symbol name)")
    ap.add_argument("--grep", help="symbol list: only symbols matching this regex")
    a = ap.parse_args()
    secs = a.section or [".text", ".rdata"]
    grep = re.compile(a.grep) if a.grep else None
    exe_syms = nm_symbols(a.exe, a.nm) if a.exe else None
    exe_addrs = [x[0] for x in exe_syms] if exe_syms else None

    out_size = collections.Counter()
    by_obj = collections.defaultdict(collections.Counter)
    by_sym_crate = collections.defaultdict(collections.Counter)
    syms = collections.Counter()   # (section, name) -> bytes
    sym_obj = {}

    for out, _in, obj, addr, size, ss in parse(a.map):
        if size == 0 or out is None or addr == 0:
            continue
        out_size[out] += size
        crate = obj_crate(obj)
        by_obj[out][crate] += size
        if out not in secs:
            continue
        if exe_syms is not None:
            ss = exe_syms[bisect.bisect_left(exe_addrs, addr):bisect.bisect_left(exe_addrs, addr + size)]
        ss = sorted(s for s in ss if addr <= s[0] < addr + size)
        if not ss or ss[0][0] > addr:
            ss.insert(0, (addr, f"[{crate} {_in}]"))
        # Aliases (identical-code folding) share an address: count once.
        uniq = []
        for s in ss:
            if uniq and uniq[-1][0] == s[0]:
                continue
            uniq.append(s)
        for i, (sa, name) in enumerate(uniq):
            end = uniq[i + 1][0] if i + 1 < len(uniq) else addr + size
            n = end - sa
            sc = sym_crate(name) if not name.startswith("[") else crate
            if sc == "(C)" or sc == "(anon)":
                sc = f"{sc} {crate}"
            by_sym_crate[out][sc] += n
            syms[(out, name)] += n
            sym_obj[(out, name)] = (sc, crate)

    debug = {s: n for s, n in out_size.items() if s.startswith((".debug", ".zdebug"))}
    total = sum(n for s, n in out_size.items() if s not in debug)
    print(f"{a.map}: {human(total)} in loaded sections (file size adds headers, symbols, debug info)\n")
    print("Output sections")
    for s, n in out_size.most_common():
        if s not in debug:
            print(f"  {s:10} {human(n):>11}  {100 * n / total:5.1f} %")
    if debug:
        print(f"  debug info {human(sum(debug.values())):>11}  (not loaded; removed by strip = true)")

    for s in secs:
        tot = out_size[s] or 1
        print(f"\n{s}: by crate of the object")
        for c, n in by_obj[s].most_common(a.top):
            print(f"  {human(n):>11}  {100 * n / tot:5.1f} %  {c}")
        print(f"\n{s}: by crate of the symbol name")
        for c, n in by_sym_crate[s].most_common(a.top):
            print(f"  {human(n):>11}  {100 * n / tot:5.1f} %  {c}")

    print(f"\nLargest symbols ({', '.join(secs)})")
    shown = 0
    for (s, name), n in syms.most_common():
        sc, crate = sym_obj[(s, name)]
        if a.crate and not sc.split(" ")[-1] == a.crate and sc != a.crate:
            continue
        if grep and not grep.search(name):
            continue
        print(f"  {human(n):>11}  {s:6}  {crate:14}  {name[:160]}")
        shown += 1
        if shown >= a.top:
            break


if __name__ == "__main__":
    sys.exit(main())
