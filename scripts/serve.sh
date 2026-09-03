#!/usr/bin/env bash
#
# FlashRec HTTP server.
#
# Production (layout inferred from tokenizer + catalog):
#   MODEL_PATH=/path/to/model ./serve.sh
#   MODEL_PATH=/path/to/model SID_VOCAB_FILE=data/catalogs/sid2pid_beamrec_l4.json ./serve.sh
#
# Default catalog is data/catalogs/sid2pid_beamrec_l4.json (repo-relative).
# Smoke test (full vocabulary, no trie): SID_VOCAB_FILE= ./serve.sh
#
#   CUDA_VISIBLE_DEVICES=0 PORT=8000 MODEL_PATH=... SID_VOCAB_FILE=... ./serve.sh
#   EXTRA_SERVER_ARGS='--disable-radix' MODEL_PATH=... SID_VOCAB_FILE=... ./serve.sh
#
# Tunable knobs: copy scripts/serve.env.example → scripts/serve.env (gitignored)
# and edit, or pass SERVE_ENV=/path/to.env. Files that use ${VAR:-...} keep
# already-exported environment variables. Bare VAR=value overwrites them.
#
# Effective defaults match cli.py/config.py. --batch-slots follows
# --cuda-graph-max-bs unless BATCH_SLOTS is set. --pipeline-stages 0 = off.
#
# SID=START:END/SIZE,... overrides tokenizer inference. The three SID_* split
# vars are last-resort overrides.
#
# The server has no authentication and binds 127.0.0.1 by default; only set
# HOST=0.0.0.0 on a trusted network or behind a proxy.
#
# Wide-beam (n=512 / n=1000) serving: the defaults admit only one 512-beam
# request at a time (slot budget 800) and starve short prompts under LPM. Use
# the n=512 / n=1000 recipe in serve.env.example, or:
#   CUDA_GRAPH_MAX_BS=4096 BATCH_SLOTS=4096 LPM_AGING_MS=150 \
#     BEAM_WIDTH=512 MODEL_PATH=... SID_VOCAB_FILE=... ./serve.sh
# --beam-width sets the fused-expand capture width; graph sizes auto-extend to
# k*n (config.resolved_cuda_graph_sizes).
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

_load_serve_env() {
    local env_file="$1"
    if [[ ! -f "$env_file" ]]; then
        echo "serve.sh: env file not found: $env_file" >&2
        exit 1
    fi
    echo "serve.sh: loading $env_file"
    set -a
    # Runtime path (SERVE_ENV / serve.env); example file is the committed template.
    # shellcheck disable=SC1090,SC1091
    source "$env_file"
    set +a
}

if [[ -n "${SERVE_ENV:-}" ]]; then
    _load_serve_env "$SERVE_ENV"
elif [[ -f "$ROOT/scripts/serve.env" ]]; then
    _load_serve_env "$ROOT/scripts/serve.env"
fi

export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export FLASHREC_TORCH_PROFILER_DIR="${FLASHREC_TORCH_PROFILER_DIR:-$ROOT/profiles}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "serve.sh: set MODEL_PATH to your model directory" >&2
    echo "  production: MODEL_PATH=/path/to/model $0" >&2
    echo "  (SID_VOCAB_FILE defaults to data/catalogs/sid2pid_beamrec_l4.json)" >&2
    exit 1
fi

CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-800}"
BATCH_SLOTS="${BATCH_SLOTS:-$CUDA_GRAPH_MAX_BS}"

# shellcheck disable=SC2206
EXTRA_ARGS=(${EXTRA_SERVER_ARGS:-})

# Unset → conventional catalog. SID_VOCAB_FILE= (empty) → unconstrained smoke.
if [[ -z "${SID_VOCAB_FILE+x}" ]]; then
    SID_VOCAB_FILE="data/catalogs/sid2pid_beamrec_l4.json"
fi

SID_ARGS=()
if [[ -n "${SID_VOCAB_FILE:-}" ]]; then
    if [[ "$SID_VOCAB_FILE" != /* ]]; then
        SID_VOCAB_FILE="$ROOT/$SID_VOCAB_FILE"
    fi
    if [[ ! -f "$SID_VOCAB_FILE" ]]; then
        echo "serve.sh: SID_VOCAB_FILE not found: $SID_VOCAB_FILE" >&2
        echo "  build it with: DATA_DIR=... bash $ROOT/scripts/build_catalog.sh" >&2
        exit 1
    fi
    SID_ARGS+=(--sid-vocab-file "$SID_VOCAB_FILE")
else
    echo "serve.sh: SID_VOCAB_FILE empty — unconstrained full-vocab decode (no trie)" >&2
    echo "  production: omit SID_VOCAB_FILE to use data/catalogs/sid2pid_beamrec_l4.json" >&2
fi
# Overrides only; omit so tokenizer + catalog infer the layout.
[[ -n "${SID:-}" ]] && SID_ARGS+=(--sid "$SID")
[[ -n "${SID_TOKEN_RANGE:-}" ]] && SID_ARGS+=(--sid-token-range "$SID_TOKEN_RANGE")
[[ -n "${SID_CODEBOOK_SIZES:-}" ]] && SID_ARGS+=(--sid-codebook-sizes "$SID_CODEBOOK_SIZES")
[[ -n "${SID_BOUNDARY_TOKENS:-}" ]] && SID_ARGS+=(--sid-boundary-tokens "$SID_BOUNDARY_TOKENS")
[[ -n "${SYSTEM_PROMPT_FILE:-}" ]] && SID_ARGS+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
[[ -n "${WARMUP_USER_A:-}" ]] && SID_ARGS+=(--warmup-user-a "$WARMUP_USER_A")
[[ -n "${WARMUP_USER_B:-}" ]] && SID_ARGS+=(--warmup-user-b "$WARMUP_USER_B")

OPTIONAL_ARGS=()
[[ -n "${BEAM_WIDTH:-}" ]] && OPTIONAL_ARGS+=(--beam-width "$BEAM_WIDTH")
[[ -n "${MAX_TOKENS:-}" ]] && OPTIONAL_ARGS+=(--max-tokens "$MAX_TOKENS")
[[ -n "${LOG_LEVEL:-}" ]] && OPTIONAL_ARGS+=(--log-level "$LOG_LEVEL")

echo "serve.sh: starting FlashRec on ${HOST:-127.0.0.1}:${PORT:-8000} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}${SID_VOCAB_FILE:+, catalog=$SID_VOCAB_FILE})"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m flashrec \
    --model-path "$MODEL_PATH" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8000}" \
    --quantization "${QUANTIZATION:-fp8}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
    --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --batch-slots "$BATCH_SLOTS" \
    --batch-wait-ms "${BATCH_WAIT_MS:-4}" \
    --batch-wait-max-ms "${BATCH_WAIT_MAX_MS:-10}" \
    --target-batch-requests "${TARGET_BATCH_REQUESTS:-8}" \
    --max-batch-requests "${MAX_BATCH_REQUESTS:-16}" \
    --schedule-policy "${SCHEDULE_POLICY:-lpm}" \
    --lpm-aging-ms "${LPM_AGING_MS:-300}" \
    ${TORCH_COMPILE:+--torch-compile} \
    --pipeline-stages "${PIPELINE_STAGES:-0}" \
    --host-worker-threads "${HOST_WORKER_THREADS:-4}" \
    "${OPTIONAL_ARGS[@]}" \
    "${SID_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --serve
