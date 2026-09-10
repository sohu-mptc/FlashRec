#!/usr/bin/env bash
# FlashRec RecIF beam × concurrency matrix (quality + QPS). FlashRec only.
#
# Default grid: n in {50,128,512} × concurrency in {1,8,16,32}
# RecIF video. Needs one GPU (FLASHREC_GPU, default 0).
#
# FlashRec: MODEL_PATH + SID_VOCAB_FILE; layout inferred from tokenizer.
# Optional SID=START:END/SIZE,... override.
#
#   MODEL_PATH=... DATA_DIR=... SAMPLE_SIZE=200 bash benchmark/recif/run_matrix.sh
#   MODEL_PATH=... DATA_DIR=... SMOKE=1 bash benchmark/recif/run_matrix.sh
#   MODELS="OneRec-1.7B OneRec-0.6B" bash benchmark/recif/run_matrix.sh
#
# Writes MATRIX_REPORT.md and matrix_summary.json under results/ (gitignored).
# Compatibility shim: bash scripts/run_sglang_flashrec_matrix.sh
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
# Optional: RHEL/CentOS gcc-toolset (no-op if the file is absent).
if [[ -f /opt/rh/gcc-toolset-11/enable ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-11/enable
  set -u
fi

# shellcheck disable=SC1091
source "$ROOT/benchmark/recif/_local_env.sh"
_flashrec_maybe_loop_models "$ROOT/benchmark/recif/run_matrix.sh"

PYTHON="${PYTHON:-python3}"
# No defaults: both point at operator-supplied data. MODEL_PATH is any
# Qwen3-architecture GenRec checkpoint; DATA_DIR is OpenOneRec-RecIF
# benchmark_data. Prefer benchmark/local/ (see benchmark/local/README.md).
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:-}"
_flashrec_apply_local_defaults
SID_VOCAB_FILE="${SID_VOCAB_FILE:-$ROOT/data/catalogs/sid2pid_beamrec_l4.json}"
SID_VOCAB_FILE="$(_flashrec_resolve_path "$SID_VOCAB_FILE")"
FLASHREC_GPU="${FLASHREC_GPU:-0}"
FLASHREC_PORT="${FLASHREC_PORT:-18931}"
WAIT_HEALTH_SEC="${WAIT_HEALTH_SEC:-600}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"

for _var in MODEL_PATH DATA_DIR; do
  if [[ -z "${!_var}" ]]; then
    echo "$_var is not set." >&2
    echo "Put files under benchmark/local/ (see benchmark/local/README.md), or:" >&2
    echo "usage: MODEL_PATH=/path/to/checkpoint \\" >&2
    echo "       DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \\" >&2
    echo "       bash $0" >&2
    echo "       MODELS=\"OneRec-1.7B OneRec-0.6B\" bash $0" >&2
    exit 2
  fi
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-4}"
  WARMUP="${WARMUP:-1}"
  BEAMS="${BEAMS:-50 128}"
  CONCS="${CONCS:-1 8}"
else
  SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
  WARMUP="${WARMUP:-8}"
  BEAMS="${BEAMS:-50 128 512}"
  CONCS="${CONCS:-1 8 16 32}"
fi

MODEL_TAG="$(_flashrec_model_tag "$MODEL_PATH")"
TS="$(date +%Y%m%d_%H%M%S)"
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTDIR="${OUTDIR:-$ROOT/results/onerec_beam_conc_smoke_${MODEL_TAG}_$TS}"
else
  OUTDIR="${OUTDIR:-$ROOT/results/onerec_beam_conc_matrix_${MODEL_TAG}_$TS}"
fi
mkdir -p "$OUTDIR"
echo "$OUTDIR" > "$OUTDIR/../LATEST_MATRIX_DIR.txt" 2>/dev/null || true
echo "$OUTDIR" > "$ROOT/results/LATEST_MATRIX_DIR.txt"

SID="${SID:-}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "missing model: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$SID_VOCAB_FILE" ]]; then
  echo "missing SID catalog: $SID_VOCAB_FILE" >&2
  echo "build it with: DATA_DIR=$DATA_DIR bash $ROOT/scripts/build_catalog.sh" >&2
  exit 1
fi

kill_port() {
  local port="$1"
  local pids
  pids=$(ss -ltnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  if [[ -z "$pids" ]]; then
    return 0
  fi
  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    local left
    left=$(ss -ltnp 2>/dev/null | grep ":${port} " || true)
    [[ -z "$left" ]] && break
    sleep 1
  done
  pids=$(ss -ltnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
}

wait_health() {
  local url="$1"
  local log="$2"
  local pid="$3"
  local i
  for i in $(seq 1 "$WAIT_HEALTH_SEC"); do
    if curl -sf -m 2 "${url}/health" >/dev/null 2>&1; then
      echo "  healthy ${url} after ${i}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  process $pid died — last 40 lines of $log:"
      tail -40 "$log" || true
      return 1
    fi
    sleep 1
  done
  echo "  timeout waiting for ${url} — last 40 lines of $log:"
  tail -40 "$log" || true
  return 1
}

start_flashrec() {
  local beam="$1"
  local log="$OUTDIR/server_flashrec_n${beam}.log"
  kill_port "$FLASHREC_PORT"
  local graph_bs slots aging max_bs_req
  case "$beam" in
    50) graph_bs=800; slots=800; aging=300; max_bs_req=16 ;;
    128) graph_bs=2048; slots=2048; aging=200; max_bs_req=16 ;;
    *) graph_bs=4096; slots=4096; aging=150; max_bs_req=8 ;;
  esac

  echo "[flashrec] starting n=${beam} gpu=${FLASHREC_GPU} port=${FLASHREC_PORT}"
  local extra_sid=()
  [[ -n "${SID}" ]] && extra_sid+=(--sid "$SID")
  # shellcheck disable=SC2206
  local extra_fr=()
  if [[ -n "${EXTRA_FLASHREC_ARGS:-}" ]]; then
    extra_fr=(${EXTRA_FLASHREC_ARGS})
    echo "[flashrec] EXTRA_FLASHREC_ARGS=${EXTRA_FLASHREC_ARGS}"
  fi
  CUDA_VISIBLE_DEVICES="$FLASHREC_GPU" \
  SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  FLASHREC_TORCH_PROFILER_DIR="$OUTDIR/profiles_flashrec" \
  "$PYTHON" -m flashrec \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$FLASHREC_PORT" \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend flashinfer \
    --mem-fraction-static "${FLASHREC_MEM_FRACTION:-0.80}" \
    --cuda-graph-max-bs "$graph_bs" \
    --batch-slots "$slots" \
    --batch-wait-ms 4 \
    --batch-wait-max-ms 10 \
    --target-batch-requests 8 \
    --max-batch-requests "$max_bs_req" \
    --max-running-requests 32 \
    --schedule-policy lpm \
    --lpm-aging-ms "$aging" \
    --beam-width "$beam" \
    --max-tokens 5 \
    "${extra_sid[@]}" \
    "${extra_fr[@]}" \
    --sid-vocab-file "$SID_VOCAB_FILE" \
    --host-worker-threads 4 \
    --serve \
    >"$log" 2>&1 &
  FLASHREC_PID=$!
  echo "$FLASHREC_PID" > "$OUTDIR/flashrec.pid"
  wait_health "http://127.0.0.1:${FLASHREC_PORT}" "$log" "$FLASHREC_PID"
}

run_cell() {
  local beam="$1"
  local conc="$2"
  local cell_dir="$OUTDIR/flashrec/n${beam}_c${conc}"
  mkdir -p "$cell_dir"
  if [[ -s "$cell_dir/summary.json" ]]; then
    echo "[eval] skip existing flashrec n=$beam conc=$conc"
    return 0
  fi
  echo "[eval] flashrec n=$beam conc=$conc -> $cell_dir"
  "$PYTHON" -m flashrec.benchmark.recif \
    --engine flashrec \
    --server-url "http://127.0.0.1:${FLASHREC_PORT}" \
    --samples-json "$OUTDIR/samples.json" \
    --catalog "$SID_VOCAB_FILE" \
    --out-dir "$cell_dir" \
    --task video \
    --n "$beam" \
    --concurrency "$conc" \
    --sample-size "$SAMPLE_SIZE" \
    --sample-mode random \
    --seed 42 \
    --max-tokens 5 \
    --warmup "$WARMUP" \
    --request-timeout "${REQUEST_TIMEOUT:-300}" \
    >"$cell_dir/eval.log" 2>&1
  local rc=$?
  echo "  rc=$rc"
  return $rc
}

summarize() {
  "$PYTHON" "$ROOT/benchmark/recif/summarize_compare.py" "$OUTDIR"
}

cleanup() {
  if [[ "$SKIP_LAUNCH" != "1" ]]; then
    kill_port "$FLASHREC_PORT"
  fi
}
trap cleanup EXIT

{
  echo "model=$MODEL_PATH"
  echo "data=$DATA_DIR"
  echo "catalog=$SID_VOCAB_FILE"
  echo "sample_size=$SAMPLE_SIZE warmup=$WARMUP"
  echo "beams=$BEAMS concs=$CONCS"
  echo "flashrec_gpu=$FLASHREC_GPU"
  echo "skip_launch=$SKIP_LAUNCH"
  echo "extra_flashrec_args=${EXTRA_FLASHREC_ARGS:-}"
  echo "outdir=$OUTDIR"
  date --iso-8601=seconds
} | tee "$OUTDIR/run_meta.txt"

echo "[setup] dumping RecIF samples"
"$PYTHON" - "$DATA_DIR" "$OUTDIR/samples.json" "$SAMPLE_SIZE" "$WARMUP" <<'PY'
import json, random, re, sys
from pathlib import Path
import pandas as pd

data_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
sample_size = int(sys.argv[3])
warmup = int(sys.argv[4])
need = sample_size + max(warmup, 0)
sid_re = re.compile(r"<s_[abc]_\d+>")
block_re = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")
frame = pd.read_parquet(data_dir / "video" / "video_test.parquet", columns=["metadata", "messages"])
order = list(range(len(frame)))
random.Random(42).shuffle(order)
samples = []
for row_idx in order:
    row = frame.iloc[row_idx]
    metadata, messages = row["metadata"], row["messages"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(metadata, dict) or not isinstance(messages, list):
        continue
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        converted.append({"role": message.get("role"), "content": content})
    blocks = block_re.findall(str(metadata.get("answer", "")))
    gt = list(dict.fromkeys("".join(sid_re.findall(b)) for b in blocks if sid_re.findall(b)))
    if converted and gt:
        samples.append({"id": str(int(row_idx)), "messages": converted, "ground_truth_sids": gt})
    if len(samples) >= need:
        break
out_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(samples)} samples -> {out_path}")
PY

for beam in $BEAMS; do
  echo "========== beam=$beam =========="
  flashrec_ok=0
  if [[ "$SKIP_LAUNCH" == "1" ]]; then
    curl -sf -m 2 "http://127.0.0.1:${FLASHREC_PORT}/health" >/dev/null && flashrec_ok=1
  else
    if start_flashrec "$beam"; then flashrec_ok=1; fi
  fi

  if [[ "$flashrec_ok" != "1" ]]; then
    echo "[flashrec] not healthy for beam=$beam, skipping cells" >&2
    continue
  fi

  for conc in $CONCS; do
    if [[ "$SKIP_LAUNCH" != "1" ]] \
      && ! curl -sf -m 2 "http://127.0.0.1:${FLASHREC_PORT}/health" >/dev/null; then
      echo "[flashrec] health lost before conc=$conc, restarting"
      if start_flashrec "$beam"; then flashrec_ok=1; else flashrec_ok=0; fi
    fi
    if [[ "$flashrec_ok" == "1" ]]; then
      run_cell "$beam" "$conc" || true
    fi
    summarize >/dev/null || true
  done

  if [[ "$SKIP_LAUNCH" != "1" ]]; then
    kill_port "$FLASHREC_PORT"
    sleep 3
  fi
done

summarize | tee "$OUTDIR/MATRIX_REPORT.print.txt"
echo "DONE $OUTDIR"
