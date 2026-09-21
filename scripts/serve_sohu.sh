#!/usr/bin/env bash
#
# Launch FlashRec for SoHuRec (production short-prompt GenRec).
#
# Defaults match the SoHuRec-1.7B serving recipe used in docs/baselines:
#   SID layout 512×512×512 at token range 151669:153204
#   FP8 weights + fp8_e4m3 KV, beam=50
#
#   bash scripts/serve_sohu.sh
#   MODEL=0.6B BEAM_WIDTH=512 bash scripts/serve_sohu.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/serve_sohu.sh
#   EXTRA_SERVER_ARGS='--enable-cascade-attention' bash scripts/serve_sohu.sh
#
# Override MODEL_PATH / SID / SYSTEM_PROMPT_FILE if your local layout differs.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Prefer the benchmark_local backup symlink tree; fall back to sglang-beam-search.
_SOHU_ROOT="${SOHU_DATA_ROOT:-/mnt/datadisk0/mcu/chongyangwang/_flashrec_benchmark_local_backup/local}"
_SOHU_FALLBACK="/mnt/datadisk0/mcu/chongyangwang/sglang-beam-search/data"

_resolve() {
  local primary="$1" fallback="$2"
  if [[ -e "$primary" ]]; then
    echo "$primary"
  else
    echo "$fallback"
  fi
}

MODEL_SIZE="${MODEL:-${MODEL_SIZE:-1.7B}}"
case "$MODEL_SIZE" in
  0.6B|0.6b|06)
    _MODEL_NAME="SoHuRec-0.6B"
    _MODEL_FALLBACK="$_SOHU_FALLBACK/v1.1.23.2-07-itemic-0.6B"
    ;;
  1.7B|1.7b|17|*)
    _MODEL_NAME="SoHuRec-1.7B"
    _MODEL_FALLBACK="$_SOHU_FALLBACK/v1.1.23.2-07-itemic"
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-$(_resolve "$_SOHU_ROOT/models/$_MODEL_NAME" "$_MODEL_FALLBACK")}"
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$(_resolve \
  "$_SOHU_ROOT/data/genrec_shared_system_prompt.txt" \
  "$_SOHU_FALLBACK/genrec_shared_system_prompt.txt")}"

# SoHu codebook constraint (not the OneRec RecIF trie catalog).
# Leave SID_VOCAB_FILE empty so serve.sh does not attach beamrec_l4.json.
export SID="${SID:-151669:153204/512,512,512}"
export SID_VOCAB_FILE=""

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QUANTIZATION="${QUANTIZATION:-fp8}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
export BEAM_WIDTH="${BEAM_WIDTH:-50}"
export MAX_TOKENS="${MAX_TOKENS:-5}"

# Slot / graph recipe by beam width (same as historical SoHu serving matrix).
if [[ -z "${CUDA_GRAPH_MAX_BS:-}" || -z "${BATCH_SLOTS:-}" || -z "${LPM_AGING_MS:-}" ]]; then
  case "$BEAM_WIDTH" in
    50)
      : "${CUDA_GRAPH_MAX_BS:=800}"
      : "${BATCH_SLOTS:=800}"
      : "${LPM_AGING_MS:=300}"
      : "${MAX_BATCH_REQUESTS:=16}"
      ;;
    100|128)
      : "${CUDA_GRAPH_MAX_BS:=1600}"
      : "${BATCH_SLOTS:=1600}"
      : "${LPM_AGING_MS:=250}"
      : "${MAX_BATCH_REQUESTS:=16}"
      ;;
    200|256)
      : "${CUDA_GRAPH_MAX_BS:=3200}"
      : "${BATCH_SLOTS:=3200}"
      : "${LPM_AGING_MS:=200}"
      : "${MAX_BATCH_REQUESTS:=16}"
      ;;
    500|512)
      : "${CUDA_GRAPH_MAX_BS:=4096}"
      : "${BATCH_SLOTS:=4096}"
      : "${LPM_AGING_MS:=150}"
      : "${MAX_BATCH_REQUESTS:=8}"
      ;;
    *)
      # n≥1000
      : "${CUDA_GRAPH_MAX_BS:=4096}"
      : "${BATCH_SLOTS:=4096}"
      : "${LPM_AGING_MS:=150}"
      : "${MAX_BATCH_REQUESTS:=8}"
      ;;
  esac
fi
export CUDA_GRAPH_MAX_BS BATCH_SLOTS LPM_AGING_MS MAX_BATCH_REQUESTS

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "serve_sohu.sh: missing model at $MODEL_PATH" >&2
  exit 1
fi
if [[ -n "${SYSTEM_PROMPT_FILE:-}" && ! -f "$SYSTEM_PROMPT_FILE" ]]; then
  echo "serve_sohu.sh: SYSTEM_PROMPT_FILE not found: $SYSTEM_PROMPT_FILE (continuing without it)" >&2
  unset SYSTEM_PROMPT_FILE
fi

echo "serve_sohu.sh: model=$MODEL_PATH"
echo "serve_sohu.sh: sid=$SID beam=$BEAM_WIDTH graph=$CUDA_GRAPH_MAX_BS slots=$BATCH_SLOTS"
echo "serve_sohu.sh: host=${HOST}:${PORT} gpu=${CUDA_VISIBLE_DEVICES}"
[[ -n "${SYSTEM_PROMPT_FILE:-}" ]] && echo "serve_sohu.sh: system_prompt=$SYSTEM_PROMPT_FILE"

exec bash "$ROOT/scripts/serve.sh"
