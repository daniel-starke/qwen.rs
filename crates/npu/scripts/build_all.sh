#!/bin/sh
# Build the lean prefill graph of every layer from the per-layer cuts
# (`cut_all.py`, CUTDIR/layer{L}.xml): build_attn.py for L % 4 == 3,
# build_delta.py otherwise, 8 at a time.
#
#   sh build_all.sh CUTDIR OUTDIR [builder options, e.g. --act rot --rotb 256]
#
# Writes OUTDIR/lean{L}.xml (+ .bin) and OUTDIR/b{L}.log. PY selects the
# Python (default `python`: one with OpenVINO); ATTN_OPTS / DELTA_OPTS add options for
# the attention / delta-net layers only (e.g. "--cap 256", "--allin").
CUT=$1; OUT=$2; shift 2
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export CUT OUT PY HERE ATTN_OPTS DELTA_OPTS
export OPTS="$*"
seq 0 63 | xargs -P ${JOBS:-8} -I{} sh -c '
L={}
if [ $((L % 4)) -eq 3 ]; then B=build_attn.py; X="$ATTN_OPTS"; else B=build_delta.py; X="$DELTA_OPTS"; fi
[ $L -eq 0 ] && X="$X --pre 1"
"$PY" "$HERE/$B" "$CUT/layer$L.xml" "$OUT/lean$L.xml" $OPTS $X > "$OUT/b$L.log" 2>&1 || echo "layer $L failed (see $OUT/b$L.log)"
'
