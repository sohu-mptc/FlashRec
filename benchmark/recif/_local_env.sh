# Shared local-path defaults for RecIF orchestration scripts.
# Sourced by bench_compare.sh / run_matrix.sh (not meant to be run alone).
#
# Layout (gitignored payloads):
#   benchmark/local/models/<name>/
#   benchmark/local/data/benchmark_data/
#   benchmark/local/env   (optional; copy from env.example)

LOCAL_DIR="${LOCAL_DIR:-$ROOT/benchmark/local}"

if [[ -f "$LOCAL_DIR/env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$LOCAL_DIR/env"
  set +a
fi

_flashrec_resolve_path() {
  local p="${1:-}"
  if [[ -z "$p" ]]; then
    echo ""
    return 0
  fi
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "$ROOT/$p"
  fi
}

_flashrec_resolve_model() {
  local m="${1:-}"
  if [[ -z "$m" ]]; then
    echo ""
    return 0
  fi
  if [[ "$m" = /* ]]; then
    echo "$m"
    return 0
  fi
  if [[ -d "$LOCAL_DIR/models/$m" ]]; then
    echo "$LOCAL_DIR/models/$m"
    return 0
  fi
  if [[ -d "$ROOT/$m" ]]; then
    echo "$ROOT/$m"
    return 0
  fi
  echo "$ROOT/$m"
}

_flashrec_model_tag() {
  local path="$1"
  local tag
  tag="$(basename "$path")"
  tag="${tag// /_}"
  echo "$tag"
}

# If MODELS lists 2+ checkpoints, re-exec this script once per model.
_flashrec_maybe_loop_models() {
  local self="$1"
  if [[ -z "${MODELS:-}" || "${_FLASHREC_MODEL_LOOP:-0}" == "1" ]]; then
    return 0
  fi
  local -a models=()
  # shellcheck disable=SC2206
  models=($MODELS)
  if ((${#models[@]} <= 1)); then
    if ((${#models[@]} == 1)); then
      export MODEL_PATH="$(_flashrec_resolve_model "${models[0]}")"
    fi
    return 0
  fi
  local rc=0
  local m resolved
  echo "[models] running ${#models[@]} checkpoints: ${models[*]}"
  for m in "${models[@]}"; do
    resolved="$(_flashrec_resolve_model "$m")"
    echo "========== MODEL=$resolved =========="
    if ! _FLASHREC_MODEL_LOOP=1 MODEL_PATH="$resolved" MODELS="$MODELS" \
      OUTDIR= bash "$self"; then
      rc=1
    fi
  done
  exit "$rc"
}

_flashrec_apply_local_defaults() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(_flashrec_resolve_model "$MODEL_PATH")"
  elif [[ -f "$LOCAL_DIR/models/OneRec-1.7B/config.json" ]]; then
    MODEL_PATH="$LOCAL_DIR/models/OneRec-1.7B"
  fi

  if [[ -n "${DATA_DIR:-}" ]]; then
    DATA_DIR="$(_flashrec_resolve_path "$DATA_DIR")"
  elif [[ -d "$LOCAL_DIR/data/benchmark_data" ]]; then
    DATA_DIR="$LOCAL_DIR/data/benchmark_data"
  fi

  if [[ -n "${SID_VOCAB_FILE:-}" ]]; then
    SID_VOCAB_FILE="$(_flashrec_resolve_path "$SID_VOCAB_FILE")"
  fi
}
