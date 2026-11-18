#!/bin/sh
# A prefill graph set that differs from an existing one only in its
# attention layers (L % 4 == 3): the delta-net layers' graphs, blobs and
# weights are hard-linked from SRC, the 16 attention layers are built with
# the given positions / window and compiled. Attention weight files equal to
# WREF's (another set built with the same attention options) are replaced by
# hard links, so N sets cost one copy of the weights.
#
#   sh mkset.sh OUT SRC S CAP [WREF]
#
# ACC is the directory of rcal.py's rcal.json / rcal_all.json. The attention
# options are the shipped ones (ATTN_OPTS overrides) plus
# `--s S --cap CAP`; NPU_SRC / NPU_WQ / NPU_WQ_STORE as for build_all.sh.
OUT=$1; SRC=$2; S=$3; CAP=$4; WREF=$5
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
CUT=${CUT:?"set CUT: the per-layer cuts (cut_all.py)"}
ACC=${ACC:?"set ACC: the calibration directory (rcal.json, rcal_all.json)"}
ATTN_OPTS=${ATTN_OPTS:---act rot --rotb 256 --wai --rmsk ln,hn --srange $ACC/rcal.json --sgated $ACC/rcal_all.json:1000 --gusplit 4}
mkdir -p "$OUT" || exit 1
for f in "$SRC"/*; do
  b=$(basename "$f")
  L=$(echo "$b" | sed -n 's/^lean\([0-9]*\)\..*/\1/p')
  [ -n "$L" ] && [ $((L % 4)) -eq 3 ] && continue
  case $b in b[0-9]*.log|c[0-9]*.log) continue;; esac
  [ -e "$OUT/$b" ] || ln "$f" "$OUT/$b" || exit 1
done
export PY HERE CUT OUT ATTN_OPTS S CAP
seq 3 4 63 | xargs -P ${JOBS:-8} -I{} sh -c '
"$PY" "$HERE/build_attn.py" "$CUT/layer{}.xml" "$OUT/lean{}.xml" $ATTN_OPTS --s $S --cap $CAP > "$OUT/b{}.log" 2>&1 || echo "layer {} failed (see $OUT/b{}.log)"
'
JOBS=${CJOBS:-6} sh "$HERE/compile_lean.sh" "$OUT"
if [ -n "$WREF" ]; then
  for f in "$OUT"/lean*.in; do
    b=$(basename "$f")
    [ -f "$WREF/$b" ] && cmp -s "$f" "$WREF/$b" && ln -f "$WREF/$b" "$f"
  done
fi
echo "set $OUT: $(ls "$OUT"/lean*.blob | wc -l) blobs"
