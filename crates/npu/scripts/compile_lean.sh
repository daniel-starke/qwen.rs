#!/bin/sh
# Compile every lean layer graph of DIR that has no lean{L}.use (dedupe.py),
# JOBS (default 4) at a time, with vpuxc.ps1 -> DIR/lean{L}.blob (skipped when
# the blob is newer than the graph).
D=$1
V="$(cygpath -w "$(cd "$(dirname "$0")" && pwd)/vpuxc.ps1")"
export V D
ls "$D" | sed -n 's/^lean\([0-9]*\)\.xml$/\1/p' | while read L; do [ -f "$D/lean$L.use" ] || echo $L; done |
  xargs -P ${JOBS:-4} -I{} sh -c 'cd "$D" && [ lean{}.blob -nt lean{}.xml ] || powershell -File "$V" lean{}.xml lean{}.blob > c{}.log 2>&1 || echo "lean{} failed (see $D/c{}.log)"'
