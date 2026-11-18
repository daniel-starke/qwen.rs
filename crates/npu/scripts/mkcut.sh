#!/bin/sh
# A prefill set whose delta layers are cut around the recurrence
# (`build_delta.py --cut pre|post` -> lean{L}a / lean{L}b; the engine runs
# the recurrence on its scan device, `scan = intel` in the .npu file):
# every file of SRC hard-linked, plus the pre / post graphs of layer 0 and
# of the graph layers 1..62 share, compiled with the shipped prefill
# environment and weight-input swizzling (the weights are SRC's swizzled
# files, which these graphs read in the same form; `.swz` logs kept).
#
#   sh mkcut.sh OUT SRC S          (S = 256 or 64, SRC's positions)
#
# Graphs built in OUT/../cutbuild-S and shared by every set of that S.
# Their two SHAVE kernels (shave/gnorm.sk as `eltwise_min`, shave/trinv.sk
# -DBS=16 as `log_softmax`) are built into cutbuild-S/kern by our kernel
# compiler (`shave`, the release example, or SHAVE) unless NPU_RS_KERNEL_DIR
# names a directory that has them.
OUT=$1; SRC=$2; S=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
SHAVE=${SHAVE:-$HERE/../../../target/release/examples/shave}
CUT=${CUT:?"set CUT: the per-layer cuts (cut_all.py)"}
ACC=${ACC:?"set ACC: the calibration directory (rcal.json, rcal_all.json)"}
F="--act rot --rotb 256 --allin --srange $ACC/rcal.json --rgroups 6 --signfold --gusplit 4
 --projsplit 5120,5120,6144 --reclean --rmsk ln --sgated $ACC/rcal_all.json:1000 --gsum1 --c 64
 --nosep --shinv 16 --qs 16 --gnk --s $S"
B=$(dirname "$OUT")/cutbuild-$S
mkdir -p "$OUT" "$B" || exit 1
export NPU_SRC=${NPU_SRC:?"set NPU_SRC: the BF16 checkpoint"} NPU_WQ=${NPU_WQ:?"set NPU_WQ, e.g. gptq:HESS_DIR"} NPU_WQ_STORE=${NPU_WQ_STORE:-i4}
export NPU_RS_DPU1LIST=1 NPU_RS_GMM_MAXG=96 NPU_RS_VIEWFOLD=1 NPU_RS_DISTVIEW=1 NPUC_OPTS=enable-grouped-matmul=true
if [ -z "$NPU_RS_KERNEL_DIR" ]; then
  mkdir -p "$B/kern" || exit 1
  for k in gnorm:eltwise_min trinv:log_softmax; do
    "$SHAVE" build "$HERE/../shave/${k%%:*}.sk" "$B/kern/${k##*:}_3720xx.elf" > "$B/kern/${k%%:*}.log" 2>&1 || { echo "kernel ${k%%:*} failed (see $B/kern/${k%%:*}.log)"; exit 1; }
    cp "$B/kern/${k##*:}_3720xx.elf" "$B/kern/${k##*:}_3720xx_lsu0_wo.elf"
  done
  NPU_RS_KERNEL_DIR="$(cygpath -m "$B/kern")"
fi
export NPU_RS_KERNEL_DIR NPU_RS_SWZIN=1 NPU_RS_SWZKEY=2
V="$(cygpath -w "$HERE/vpuxc.ps1")"
for L in 0 1; do
  X=""; [ $L -eq 0 ] && X="--pre 1"
  for c in a b; do
    cut=pre; [ $c = b ] && cut=post
    n=lean$L$c
    [ -f "$B/$n.blob" ] && continue
    "$PY" "$HERE/build_delta.py" "$CUT/layer$L.xml" "$B/$n.xml" $F $X --cut $cut > "$B/$n.b.log" 2>&1 || { echo "build $n failed"; tail "$B/$n.b.log"; exit 1; } &
  done
done
wait
for L in 0 1; do
  for c in a b; do
    n=lean$L$c
    [ -f "$B/$n.blob" ] || (cd "$B" && NPU_RS_SWZIN_LOG=$n.swz powershell -File "$V" $n.xml $n.blob > $n.c.log 2>&1 || echo "compile $n failed (see $B/$n.c.log)") &
  done
done
wait
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ -e "$OUT/$b" ] || ln "$f" "$OUT/$b" || exit 1
done
for n in lean0a lean0b lean1a lean1b; do
  ln -f "$B/$n.blob" "$OUT/$n.blob" || exit 1
done
for L in $(seq 2 62); do
  [ $((L % 4)) -eq 3 ] && continue
  echo lean1a > "$OUT/lean${L}a.use"
  echo lean1b > "$OUT/lean${L}b.use"
done
echo "cut set $OUT (S=$S) from $SRC"
