#!/usr/bin/env bash
# Build FlashRec SID catalog(s) from OpenOneRec RecIF packed mappings.
# Serve with --model-path plus --sid-vocab-file; layout is inferred.
#
#   DATA_DIR=/path/to/benchmark_data bash scripts/build_catalog.sh
#   DATA_DIR=/path/to/benchmark_data TASK=both bash scripts/build_catalog.sh
#   flashrec --catalog /path/to/benchmark_data
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
# No default: the RecIF dataset lives wherever the operator downloaded it.
DATA_DIR="${DATA_DIR:-}"
OUT_DIR="${OUT_DIR:-$ROOT/data/catalogs}"
TASK="${TASK:-video}"
LEVELS="${LEVELS:-}"

if [[ -z "$DATA_DIR" ]]; then
  echo "DATA_DIR is not set." >&2
  echo "Point it at an OpenOneRec-RecIF benchmark_data directory:" >&2
  echo "  DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data bash $0" >&2
  echo "Or: flashrec --catalog /path/to/OpenOneRec-RecIF/benchmark_data" >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "missing RecIF data dir: $DATA_DIR" >&2
  echo "set DATA_DIR to OpenOneRec-RecIF/benchmark_data" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cmd=("$PYTHON" -m flashrec --catalog "$DATA_DIR" --catalog-out "$OUT_DIR" --catalog-task "$TASK")
if [[ -n "$LEVELS" ]]; then
  cmd+=(--catalog-levels "$LEVELS")
fi
exec "${cmd[@]}"
