#!/bin/sh
# A prefill graph set that differs from an existing one only in its
# delta-net layers (L % 4 != 3): the 16 attention layers' graphs, blobs and
# weights are hard-linked from SRC, the 48 delta layers are built with
# DELTA_OPTS (layer 0 adds `--pre 1`), deduplicated and compiled (with the
# compile environment of the caller: NPU_RS_*, NPUC_OPTS).
#
#   DELTA_OPTS="..." sh mkdelta.sh OUT SRC
#
# NPU_SRC / NPU_WQ / NPU_WQ_STORE as for build_all.sh.
OUT=$1; SRC=$2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
CUT=${CUT:?"set CUT: the per-layer cuts (cut_all.py)"}
mkdir -p "$OUT" || exit 1
for f in "$SRC"/*; do
  b=$(basename "$f")
  # the attention layers' files: lean{L}.*, b{L}.log, c{L}.log with L % 4 == 3
  L=$(echo "$b" | sed -n 's/^lean\([0-9]*\)\..*/\1/p; s/^[bc]\([0-9]*\)\.log$/\1/p')
  [ -n "$L" ] && [ $((L % 4)) -eq 3 ] || continue
  [ -e "$OUT/$b" ] || ln "$f" "$OUT/$b" || exit 1
done
export PY HERE CUT OUT DELTA_OPTS
seq 0 63 | awk '$1 % 4 != 3' | xargs -P ${JOBS:-8} -I{} sh -c '
X="$DELTA_OPTS"; [ {} -eq 0 ] && X="$X --pre 1"
"$PY" "$HERE/build_delta.py" "$CUT/layer{}.xml" "$OUT/lean{}.xml" $X > "$OUT/b{}.log" 2>&1 || echo "layer {} failed (see $OUT/b{}.log)"
'
# weight files equal to SRC's become hard links (one copy on disk)
for f in "$OUT"/lean*.in; do
  b=$(basename "$f")
  [ -f "$SRC/$b" ] && ! [ "$f" -ef "$SRC/$b" ] && cmp -s "$f" "$SRC/$b" && ln -f "$SRC/$b" "$f"
done
"$PY" "$HERE/dedupe.py" "$OUT" > "$OUT/dedupe.log" || exit 1
JOBS=${CJOBS:-6} sh "$HERE/compile_lean.sh" "$OUT"
echo "set $OUT: $(ls "$OUT"/lean*.blob | wc -l) blobs; $(head -1 "$OUT/dedupe.log")"
