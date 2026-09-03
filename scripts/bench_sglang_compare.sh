#!/usr/bin/env bash
# Speed benchmark: FlashRec vs SGLang beam search (PR #31626).
#
# Default grid: beam {50,128,512} × concurrency {1,8,16,32} on RecIF video.
# FlashRec: SID trie + FP8. SGLang: open-vocab beam via POST /generate
# (sampling_params.beam_width). Do not send OpenAI `n` to SGLang — that is
# parallel sampling, not beam search.
#
#   MODEL_PATH=/path/to/OneRec-1.7B \
#   DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
#   bash scripts/bench_sglang_compare.sh
#
#   SMOKE=1 ...                         # 4 samples, beam 50, conc 1 8
#   SGLANG_LAUNCH=docker|native|skip    # default docker
#   DOCKER='sudo docker'                # if the docker CLI needs wrapping
#   SKIP_LAUNCH=1                       # eval against already-running servers
#
# Writes results/onerec_beam_conc_bench_<stamp>/{MATRIX_REPORT.md,SPEED_COMPARE.md}.
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
# Optional: RHEL/CentOS gcc-toolset (no-op if the file is absent).
if [[ -f /opt/rh/gcc-toolset-11/enable ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-11/enable
  set -u
fi

PYTHON="${PYTHON:-python3}"
SGLANG_PY="${SGLANG_PY:-$PYTHON}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:-}"
SID_VOCAB_FILE="${SID_VOCAB_FILE:-$ROOT/data/catalogs/sid2pid_beamrec_l4.json}"
FLASHREC_GPU="${FLASHREC_GPU:-0}"
SGL_GPU="${SGL_GPU:-1}"
FLASHREC_PORT="${FLASHREC_PORT:-18931}"
SGL_PORT="${SGL_PORT:-18910}"
WAIT_HEALTH_SEC="${WAIT_HEALTH_SEC:-600}"
SGLANG_LAUNCH="${SGLANG_LAUNCH:-docker}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
DOCKER="${DOCKER:-docker}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1}"
SGLANG_CONTAINER="${SGLANG_CONTAINER:-flashrec-sglang-bench}"

for _var in MODEL_PATH DATA_DIR; do
  if [[ -z "${!_var}" ]]; then
    echo "$_var is not set." >&2
    echo "usage: MODEL_PATH=/path/to/checkpoint \\" >&2
    echo "       DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \\" >&2
    echo "       bash $0" >&2
    exit 2
  fi
done

if [[ "${SMOKE:-0}" == "1" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-4}"
  WARMUP="${WARMUP:-1}"
  BEAMS="${BEAMS:-50}"
  CONCS="${CONCS:-1 8}"
else
  SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
  WARMUP="${WARMUP:-8}"
  BEAMS="${BEAMS:-50 128 512}"
  CONCS="${CONCS:-1 8 16 32}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTDIR="${OUTDIR:-$ROOT/results/onerec_beam_conc_bench_smoke_$TS}"
else
  OUTDIR="${OUTDIR:-$ROOT/results/onerec_beam_conc_bench_$TS}"
fi
mkdir -p "$OUTDIR" "$ROOT/results"
echo "$OUTDIR" > "$ROOT/results/LATEST_MATRIX_DIR.txt"

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

wait_url() {
  local url="$1"
  local log="$2"
  local pid="$3"
  local i
  for i in $(seq 1 "$WAIT_HEALTH_SEC"); do
    if curl -sf -m 2 "${url}/health" >/dev/null 2>&1 \
      || curl -sf -m 2 "${url}/get_model_info" >/dev/null 2>&1 \
      || curl -sf -m 2 "${url}/v1/models" >/dev/null 2>&1; then
      echo "  healthy ${url} after ${i}s"
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "  process $pid died — last 40 lines of $log:"
      tail -40 "$log" || true
      return 1
    fi
    if [[ "$SGLANG_LAUNCH" == "docker" && -n "${SGLANG_CONTAINER:-}" ]]; then
      local st
      st=$($DOCKER inspect -f '{{.State.Running}}' "$SGLANG_CONTAINER" 2>/dev/null || echo false)
      if [[ "$st" != "true" && "$url" == *":${SGL_PORT}"* ]]; then
        echo "  docker $SGLANG_CONTAINER not running — last 40 lines of $log:"
        tail -40 "$log" || true
        return 1
      fi
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
  [[ -n "${SID:-}" ]] && extra_sid+=(--sid "$SID")
  CUDA_VISIBLE_DEVICES="$FLASHREC_GPU" \
  SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
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
    --sid-vocab-file "$SID_VOCAB_FILE" \
    --host-worker-threads 4 \
    --serve \
    >"$log" 2>&1 &
  FLASHREC_PID=$!
  echo "$FLASHREC_PID" > "$OUTDIR/flashrec.pid"
  wait_url "http://127.0.0.1:${FLASHREC_PORT}" "$log" "$FLASHREC_PID"
}

stop_sglang() {
  if [[ -f "$OUTDIR/sglang_log.pid" ]]; then
    kill "$(cat "$OUTDIR/sglang_log.pid")" 2>/dev/null || true
    rm -f "$OUTDIR/sglang_log.pid"
  fi
  if [[ "$SGLANG_LAUNCH" == "docker" ]]; then
    $DOCKER rm -f "$SGLANG_CONTAINER" >/dev/null 2>&1 || true
  else
    kill_port "$SGL_PORT"
  fi
}

start_sglang_docker() {
  local beam="$1"
  local log="$2"
  local graph_bs="$beam"
  local max_running=$(( (beam + 1) * 32 ))
  if (( max_running < 256 )); then max_running=256; fi
  echo "[sglang] docker n=${beam} gpu=${SGL_GPU} port=${SGL_PORT} max_running=${max_running} graph=${graph_bs}"
  $DOCKER rm -f "$SGLANG_CONTAINER" >/dev/null 2>&1 || true
  $DOCKER run -d --name "$SGLANG_CONTAINER" \
    --gpus "device=${SGL_GPU}" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p "${SGL_PORT}:30000" \
    -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    "$SGLANG_IMAGE" \
    python3 -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --host 0.0.0.0 --port 30000 \
      --dtype bfloat16 \
      --attention-backend flashinfer \
      --mem-fraction-static "${SGL_MEM_FRACTION:-0.70}" \
      --context-length "${SGL_CONTEXT_LENGTH:-4096}" \
      --random-seed 0 \
      --max-running-requests "$max_running" \
      --chunked-prefill-size "${SGL_CHUNKED_PREFILL:-8192}" \
      --disable-radix-cache \
      --disable-chunked-prefix-cache \
      --disable-overlap-schedule \
      --cuda-graph-max-bs-decode "$graph_bs" \
      --cuda-graph-backend-decode full \
      >>"$log" 2>&1
  $DOCKER logs -f "$SGLANG_CONTAINER" >>"$log" 2>&1 &
  echo $! > "$OUTDIR/sglang_log.pid"
}

start_sglang_native() {
  local beam="$1"
  local log="$2"
  local graph_bs="$beam"
  local max_running=$(( (beam + 1) * 32 ))
  if (( max_running < 256 )); then max_running=256; fi
  echo "[sglang] native n=${beam} gpu=${SGL_GPU} port=${SGL_PORT} max_running=${max_running} graph=${graph_bs}"
  kill_port "$SGL_PORT"
  CUDA_VISIBLE_DEVICES="$SGL_GPU" \
  SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$SGL_PORT" \
    --dtype bfloat16 \
    --mem-fraction-static "${SGL_MEM_FRACTION:-0.70}" \
    --cuda-graph-max-bs-decode "$graph_bs" \
    --max-running-requests "$max_running" \
    --attention-backend flashinfer \
    --disable-overlap-schedule \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --context-length "${SGL_CONTEXT_LENGTH:-4096}" \
    --chunked-prefill-size "${SGL_CHUNKED_PREFILL:-8192}" \
    >"$log" 2>&1 &
  echo $! > "$OUTDIR/sglang.pid"
}

start_sglang() {
  local beam="$1"
  local log="$OUTDIR/server_sglang_n${beam}.log"
  stop_sglang
  case "$SGLANG_LAUNCH" in
    docker) start_sglang_docker "$beam" "$log" ;;
    native)
      start_sglang_native "$beam" "$log"
      wait_url "http://127.0.0.1:${SGL_PORT}" "$log" "$(cat "$OUTDIR/sglang.pid")"
      return
      ;;
    skip) echo "[sglang] SKIP_LAUNCH — using http://127.0.0.1:${SGL_PORT}"; return 0 ;;
    *) echo "unknown SGLANG_LAUNCH=$SGLANG_LAUNCH (docker|native|skip)" >&2; return 1 ;;
  esac
  wait_url "http://127.0.0.1:${SGL_PORT}" "$log" ""
}

run_cell() {
  local engine="$1"
  local url="$2"
  local beam="$3"
  local conc="$4"
  local cell_dir="$OUTDIR/${engine}/n${beam}_c${conc}"
  mkdir -p "$cell_dir"
  if [[ -s "$cell_dir/summary.json" ]]; then
    echo "[eval] skip existing $engine n=$beam conc=$conc"
    return 0
  fi
  echo "[eval] $engine n=$beam conc=$conc -> $cell_dir"
  local extra=()
  if [[ "$engine" == "sglang" ]]; then
    extra+=(--protocol generate --model-path "$MODEL_PATH")
  fi
  "$PYTHON" "$ROOT/scripts/eval_beam_matrix.py" \
    --engine "$engine" \
    --server-url "$url" \
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
    --request-timeout "${REQUEST_TIMEOUT:-600}" \
    "${extra[@]}" \
    >"$cell_dir/eval.log" 2>&1
  local rc=$?
  if [[ "$rc" -ne 0 || ! -s "$cell_dir/summary.json" ]]; then
    echo "  FAIL rc=$rc"
    return 1
  fi
  echo "  ok"
  return 0
}

summarize() {
  "$PYTHON" - "$OUTDIR" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary in sorted(root.glob("*/n*_c*/summary.json")):
    payload = json.loads(summary.read_text(encoding="utf-8"))
    m = payload.get("metrics") or {}
    rows.append({
        "engine": payload.get("engine"),
        "beam": payload.get("beam_size"),
        "conc": payload.get("concurrency"),
        "samples": payload.get("samples"),
        "qps": payload.get("qps"),
        "wall_s": payload.get("wall_seconds"),
        "p50_s": payload.get("latency_p50_s"),
        "p99_s": payload.get("latency_p99_s"),
        "recall@32": m.get("recall@32"),
        "ndcg@32": m.get("ndcg@32"),
        "mrr@32": m.get("mrr@32"),
        "hit@32": m.get("hit@32"),
        "invalid_rate": payload.get("invalid_rate"),
        "path": str(summary.parent.relative_to(root)),
    })
rows.sort(key=lambda r: (str(r["engine"]), int(r["beam"] or 0), int(r["conc"] or 0)))
(root / "matrix_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

def fmt(v, nd=4):
    if v is None:
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)

lines = [
    "# OneRec × RecIF video — FlashRec vs SGLang (speed bench)",
    "",
    f"- output: `{root}`",
    f"- cells: {len(rows)}",
    "",
    "| engine | beam | conc | samples | QPS | wall(s) | p50(s) | p99(s) | recall@32 | ndcg@32 | invalid_rate |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {engine} | {beam} | {conc} | {samples} | {qps} | {wall} | {p50} | {p99} | {rec} | {ndcg} | {inv} |".format(
            engine=r["engine"],
            beam=r["beam"],
            conc=r["conc"],
            samples=r["samples"],
            qps=fmt(r["qps"], 3),
            wall=fmt(r["wall_s"], 1),
            p50=fmt(r["p50_s"], 3),
            p99=fmt(r["p99_s"], 3),
            rec=fmt(r.get("recall@32"), 5),
            ndcg=fmt(r.get("ndcg@32"), 5),
            inv=fmt(r.get("invalid_rate"), 4),
        )
    )
(root / "MATRIX_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
  "$PYTHON" "$ROOT/scripts/summarize_sglang_compare.py" "$OUTDIR" || true
}

cleanup() {
  if [[ "$SKIP_LAUNCH" != "1" ]]; then
    kill_port "$FLASHREC_PORT"
    stop_sglang
  fi
}
trap cleanup EXIT

{
  echo "model=$MODEL_PATH"
  echo "data=$DATA_DIR"
  echo "catalog=$SID_VOCAB_FILE"
  echo "sample_size=$SAMPLE_SIZE warmup=$WARMUP"
  echo "beams=$BEAMS concs=$CONCS"
  echo "flashrec_gpu=$FLASHREC_GPU sgl_gpu=$SGL_GPU"
  echo "sglang_launch=$SGLANG_LAUNCH skip_launch=$SKIP_LAUNCH"
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
        samples.append({"id": str(row_idx), "messages": converted, "ground_truth_sids": gt})
    if len(samples) >= need:
        break
out_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(samples)} samples -> {out_path}")
PY

sgl_health() {
  curl -sf -m 2 "http://127.0.0.1:${SGL_PORT}/health" >/dev/null \
    || curl -sf -m 2 "http://127.0.0.1:${SGL_PORT}/get_model_info" >/dev/null
}

for beam in $BEAMS; do
  echo "========== beam=$beam =========="
  flashrec_ok=0
  sgl_ok=0
  if [[ "$SKIP_LAUNCH" == "1" ]]; then
    curl -sf -m 2 "http://127.0.0.1:${FLASHREC_PORT}/health" >/dev/null && flashrec_ok=1
    sgl_health && sgl_ok=1
  else
    if start_flashrec "$beam"; then flashrec_ok=1; fi
    if start_sglang "$beam"; then sgl_ok=1; fi
  fi

  for conc in $CONCS; do
    if [[ "$SKIP_LAUNCH" != "1" && "$sgl_ok" == "1" ]] && ! sgl_health; then
      echo "[sglang] health lost before conc=$conc, restarting"
      if start_sglang "$beam"; then sgl_ok=1; else sgl_ok=0; fi
    fi
    if [[ "$SKIP_LAUNCH" != "1" && "$flashrec_ok" == "1" ]] \
      && ! curl -sf -m 2 "http://127.0.0.1:${FLASHREC_PORT}/health" >/dev/null; then
      echo "[flashrec] health lost before conc=$conc, restarting"
      if start_flashrec "$beam"; then flashrec_ok=1; else flashrec_ok=0; fi
    fi
    pids=()
    if [[ "$flashrec_ok" == "1" ]]; then
      run_cell flashrec "http://127.0.0.1:${FLASHREC_PORT}" "$beam" "$conc" &
      pids+=($!)
    fi
    if [[ "$sgl_ok" == "1" ]]; then
      run_cell sglang "http://127.0.0.1:${SGL_PORT}" "$beam" "$conc" &
      pids+=($!)
    fi
    for pid in "${pids[@]:-}"; do
      wait "$pid" || true
    done
    summarize >/dev/null || true
  done

  if [[ "$SKIP_LAUNCH" != "1" ]]; then
    kill_port "$FLASHREC_PORT"
    stop_sglang
    sleep 3
  fi
done

summarize | tee "$OUTDIR/MATRIX_REPORT.print.txt"
echo "DONE $OUTDIR"
