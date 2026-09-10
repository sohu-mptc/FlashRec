#!/usr/bin/env bash
# Compatibility shim — see benchmark/recif/run_matrix.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/benchmark/recif/run_matrix.sh" "$@"
