#!/bin/sh
# A prefill set for positions past every attention window ("far" set): SRC's
# files hard-linked (a cut set: delta layers, weights), its 16 attention
# layers as two graphs each: lean{L}a (`build_attn.py --part pre`: x, cos,
# sin -> qg, gate, knew, vnew) and lean{L}b (`--part post`: x, o, gate -> y),
# plus two weight-free graphs every attention layer shares: near.blob
# (`--part near`: the chunk's keys and one partly filled block) and far.blob
# (`--part far`: one full block of past keys). The engine keeps K / V in
# host blocks of CAP - 2 keys, runs far.blob once per full block and merges
# the blocks by their log sum exp (shave/softmax_lse.sk as the Softmax
# kernel of near / far, built for each graph's key axis by our kernel
# compiler: `shave`, the release example, or SHAVE).
#
#   sh mkfar.sh OUT SRC S [CAP]       (S = SRC's positions; CAP default 1024)
#
# DEC=1 (with S = 1): the decode form (`fardec = OUT` in the .npu file),
# only the attention graphs, near.blob as `--part near0` (the step's key is
# in the block already); the delta layers are the decode set's, the weights
# the prefill sets' buffers; plus farm.blob (`--part far --nblk 8`: 8 blocks
# per far job, 0.29 ms per block against 0.35).
OUT=$1; SRC=$2; S=$3; CAP=${4:-1024}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
SHAVE=${SHAVE:-$HERE/../../../target/release/examples/shave}
CUT=${CUT:?"set CUT: the per-layer cuts (cut_all.py)"}
ACC=${ACC:?"set ACC: the calibration directory (rcal.json, rcal_all.json)"}
ATTN_OPTS=${ATTN_OPTS:---act rot --rotb 256 --wai --rmsk ln,hn --srange $ACC/rcal.json --sgated $ACC/rcal_all.json:1000 --gusplit 4}
B=$(dirname "$OUT")/farbuild-$S
mkdir -p "$OUT" "$B" || exit 1
export NPU_SRC=${NPU_SRC:?"set NPU_SRC: the BF16 checkpoint"} NPU_WQ=${NPU_WQ:?"set NPU_WQ, e.g. gptq:HESS_DIR"} NPU_WQ_STORE=${NPU_WQ_STORE:-i4}
export NPU_RS_DPU1LIST=1 NPU_RS_GMM_MAXG=96 NPU_RS_VIEWFOLD=1 NPU_RS_DISTVIEW=1 NPUC_OPTS=enable-grouped-matmul=true
V="$(cygpath -w "$HERE/vpuxc.ps1")"
export PY HERE CUT B ATTN_OPTS S CAP V
# near / far: the Softmax kernel built for each graph's key axis
for g in near far; do
  f=$g$CAP
  [ -f "$B/$f.blob" ] && continue
  n=$CAP; [ $g = near ] && [ -z "$DEC" ] && n=$((CAP + S))
  part=$g; [ $g = near ] && [ -n "$DEC" ] && part=near0
  mkdir -p "$B/k$f"
  # far: P left undivided (-DNONORM, the kernel's third pass; `far.nonorm`
  # tells the engine to divide in its merge): 5.0 -> 3.7 ms per block
  kd=; [ $g = far ] && kd=-DNONORM=1
  "$SHAVE" build "$HERE/../shave/softmax_lse.sk" "$B/k$f/softmax_3720xx.elf" -DN=$n $kd > "$B/$f.k.log" 2>&1 || { echo "kernel $f failed (see $B/$f.k.log)"; exit 1; }
  cp "$B/k$f/softmax_3720xx.elf" "$B/k$f/softmax_3720xx_lsu0_wo.elf"
  ( "$PY" "$HERE/build_attn.py" "$CUT/layer3.xml" "$B/$f.xml" --part $part --s $S --cap $CAP > "$B/$f.b.log" 2>&1 &&
    cd "$B" && NPU_RS_KERNEL_DIR="$(cygpath -m "$B/k$f")" powershell -File "$V" $f.xml $f.blob > $f.c.log 2>&1 || echo "$f failed (see $B/$f.*.log)" ) &
done
[ -n "$DEC" ] && [ ! -f "$B/farm$CAP.blob" ] && ( "$PY" "$HERE/build_attn.py" "$CUT/layer3.xml" "$B/farm$CAP.xml" --part far --nblk 8 --s $S --cap $CAP > "$B/farm$CAP.b.log" 2>&1 &&
  cd "$B" && until [ -f kfar$CAP/softmax_3720xx_lsu0_wo.elf ]; do sleep 1; done &&
  NPU_RS_KERNEL_DIR="$(cygpath -m "$B/kfar$CAP")" powershell -File "$V" farm$CAP.xml farm$CAP.blob > farm$CAP.c.log 2>&1 || echo "farm$CAP failed (see $B/farm$CAP.*.log)" ) &
seq 3 4 63 | xargs -P ${JOBS:-6} -I{} sh -c '
for c in a b; do
  n=lean{}$c; [ -f "$B/$n.blob" ] && continue
  o="--part pre"; [ $c = b ] && o="--part post --w0 1"
  "$PY" "$HERE/build_attn.py" "$CUT/layer{}.xml" "$B/$n.xml" $ATTN_OPTS --s $S $o > "$B/$n.b.log" 2>&1 || { echo "build $n failed"; continue; }
  rm -f "$B"/$n.w*.in
  (cd "$B" && NPU_RS_SWZIN=1 NPU_RS_SWZKEY=2 NPU_RS_SWZIN_LOG=$n.swz powershell -File "$V" $n.xml $n.blob > $n.c.log 2>&1) || echo "compile $n failed (see $B/$n.c.log)"
done'
wait
[ -z "$DEC" ] && for f in "$SRC"/*; do
  b=$(basename "$f")
  L=$(echo "$b" | sed -n 's/^lean\([0-9]*\)\.\(blob\|xml\|bin\|use\)$/\1/p')
  [ -n "$L" ] && [ $((L % 4)) -eq 3 ] && continue
  [ -e "$OUT/$b" ] || ln "$f" "$OUT/$b" || exit 1
done
for L in $(seq 3 4 63); do
  for c in a b; do ln -f "$B/lean$L$c.blob" "$OUT/lean$L$c.blob" || exit 1; done
  # the cut graphs must read each weight in the form the whole layer's did
  # (the compiler logs input indices: pre x, cos, sin, w0; post x, o, gate, w1..)
  c=$({ tr -d '\r' < "$B/lean${L}a.swz" | awk 'NF {print "w" $1 - 3}'; tr -d '\r' < "$B/lean${L}b.swz" | awk 'NF {print "w" $1 - 2}'; } | sort | tr '\n' ' ')
  r=$(tr -d '\r' < "$SRC/lean$L.swz" | awk 'NF' | sort | tr '\n' ' ')
  [ "$c" = "$r" ] || echo "layer $L: swizzled inputs [$c] differ from $SRC's [$r]"
done
ln -f "$B/near$CAP.blob" "$OUT/near.blob" && ln -f "$B/far$CAP.blob" "$OUT/far.blob" || exit 1
[ -n "$DEC" ] && { ln -f "$B/farm$CAP.blob" "$OUT/farm.blob" || exit 1; }
: > "$OUT/far.nonorm"
echo "far set $OUT (S=$S, blocks of $((CAP - 2)) keys) from $SRC"
