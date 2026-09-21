#!/usr/bin/env bash
# Parallel FlashRec RecIF ablations across idle GPUs.
#
# One-at-a-time + full combo, RecIF video quality+speed:
#   baseline / cascade / tbb / unary / sparse / full
# Default grid: beam {50,128,512} × conc {8,32} (skip conc=1; unique-beam collapse).
#
#   MODEL_PATH=/path/to/OneRec-1.7B \
#   DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
#   bash benchmark/recif/run_ablation.sh
#
# Override: GPUS="0 1 2 6 7" SAMPLE_SIZE=200 BEAMS="50 128 512" CONCS="8 32"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "$ROOT/benchmark/recif/_local_env.sh"

PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/datadisk0/mcu/chongyangwang/sglang-beam-search/data/OneRec-1.7B}"
DATA_DIR="${DATA_DIR:-/mnt/datadisk0/mcu/chongyangwang/sglang-beam-search/data/OpenOneRec-RecIF/benchmark_data}"
SID_VOCAB_FILE="${SID_VOCAB_FILE:-$ROOT/data/catalogs/sid2pid_beamrec_l4.json}"
_flashrec_apply_local_defaults
SID_VOCAB_FILE="$(_flashrec_resolve_path "$SID_VOCAB_FILE")"
MODEL_PATH="$(_flashrec_resolve_model "$MODEL_PATH")"
DATA_DIR="$(_flashrec_resolve_path "$DATA_DIR")"

if [[ "${SMOKE:-0}" == "1" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-4}"
  WARMUP="${WARMUP:-1}"
  BEAMS="${BEAMS:-50}"
  CONCS="${CONCS:-8}"
else
  SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
  WARMUP="${WARMUP:-8}"
  BEAMS="${BEAMS:-50 128 512}"
  CONCS="${CONCS:-8 32}"
fi

PORT_BASE="${PORT_BASE:-18940}"
MEM_USED_MAX_MIB="${MEM_USED_MAX_MIB:-512}"

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v cap="$MEM_USED_MAX_MIB" '{
        gsub(/ /, "", $1); gsub(/ /, "", $2);
        if ($2 + 0 < cap + 0) print $1
      }'
}

if [[ -n "${GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  GPU_ARR=($GPUS)
else
  mapfile -t GPU_ARR < <(idle_gpus)
fi
if ((${#GPU_ARR[@]} == 0)); then
  echo "no idle GPUs (memory.used < ${MEM_USED_MAX_MIB} MiB)." >&2
  exit 1
fi

# Isolate each new feature against a common off-base. TBB / unary / sparse
# default ON in this branch; cascade defaults OFF. TBB is built on the fused
# sparse scorer (trie_budget_score feeds the batched budget select), so the
# tbb variant keeps sparse on; read its delta against the sparse row.
OFF="--disable-trie-budget --disable-unary-finish --disable-sparse-lm-head"
VARIANTS=(
  "baseline|$OFF"
  "cascade|$OFF --enable-cascade-attention"
  "tbb|--disable-unary-finish"
  "full|--enable-cascade-attention"
  "unary|--disable-trie-budget --disable-sparse-lm-head"
  "sparse|--disable-trie-budget --disable-unary-finish"
)

TS="$(date +%Y%m%d_%H%M%S)"
if [[ "${SMOKE:-0}" == "1" ]]; then
  PARENT="${OUTDIR:-$ROOT/results/ablation_smoke_$TS}"
else
  PARENT="${OUTDIR:-$ROOT/results/ablation_$TS}"
fi
mkdir -p "$PARENT"
echo "$PARENT" > "$ROOT/results/LATEST_ABLATION_DIR.txt"

{
  echo "parent=$PARENT"
  echo "model=$MODEL_PATH"
  echo "data=$DATA_DIR"
  echo "catalog=$SID_VOCAB_FILE"
  echo "gpus=${GPU_ARR[*]}"
  echo "sample_size=$SAMPLE_SIZE warmup=$WARMUP"
  echo "beams=$BEAMS concs=$CONCS"
  echo "python=$PYTHON"
  date --iso-8601=seconds
} | tee "$PARENT/run_meta.txt"

run_variant() {
  local gpu="$1" name="$2" extra="$3"
  local port=$((PORT_BASE + gpu))
  local out="$PARENT/$name"
  mkdir -p "$out"
  echo "[ablation] gpu=$gpu port=$port variant=$name extra='$extra'"
  echo "gpu=$gpu port=$port extra=$extra" > "$out/variant_meta.txt"
  MODEL_PATH="$MODEL_PATH" \
  DATA_DIR="$DATA_DIR" \
  SID_VOCAB_FILE="$SID_VOCAB_FILE" \
  PYTHON="$PYTHON" \
  FLASHREC_GPU="$gpu" \
  FLASHREC_PORT="$port" \
  EXTRA_FLASHREC_ARGS="$extra" \
  SAMPLE_SIZE="$SAMPLE_SIZE" \
  WARMUP="$WARMUP" \
  BEAMS="$BEAMS" \
  CONCS="$CONCS" \
  SMOKE=0 \
  OUTDIR="$out" \
    bash "$ROOT/benchmark/recif/bench_compare.sh" \
    >"$out/orchestrator.log" 2>&1
  local rc=$?
  echo "[ablation] variant=$name rc=$rc" | tee -a "$PARENT/run_meta.txt"
  return "$rc"
}

# 1:1 GPU mapping for the first N-1 variants; leftover variants run
# sequentially on the last GPU so every idle card stays busy.
n_gpu=${#GPU_ARR[@]}
n_var=${#VARIANTS[@]}
primary=()
overflow=()
if ((n_var <= n_gpu)); then
  primary=("${VARIANTS[@]}")
else
  primary=("${VARIANTS[@]:0:n_gpu-1}")
  overflow=("${VARIANTS[@]:n_gpu-1}")
fi

pids=()
names=()
idx=0
for spec in "${primary[@]}"; do
  name="${spec%%|*}"
  extra="${spec#*|}"
  gpu="${GPU_ARR[$idx]}"
  run_variant "$gpu" "$name" "$extra" &
  pids+=($!)
  names+=("$name")
  idx=$((idx + 1))
done

if ((${#overflow[@]} > 0)); then
  gpu="${GPU_ARR[$((n_gpu - 1))]}"
  (
    rc=0
    for spec in "${overflow[@]}"; do
      name="${spec%%|*}"
      extra="${spec#*|}"
      run_variant "$gpu" "$name" "$extra" || rc=1
    done
    exit "$rc"
  ) &
  pids+=($!)
  names+=("overflow:${gpu}")
fi

echo "launched ${#pids[@]} workers pids=${pids[*]} names=${names[*]}" | tee -a "$PARENT/run_meta.txt"
echo "${pids[*]}" > "$PARENT/workers.pids"

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[ablation] worker ${names[$i]} pid=${pids[$i]} failed" | tee -a "$PARENT/run_meta.txt"
    fail=1
  fi
done

"$PYTHON" "$ROOT/benchmark/recif/summarize_ablation.py" "$PARENT" \
  | tee "$PARENT/ABLATION_REPORT.print.txt" || true

echo "DONE $PARENT fail=$fail"
exit "$fail"
