#!/usr/bin/env bash
# Compatibility shim — see benchmark/recif/bench_compare.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/benchmark/recif/bench_compare.sh" "$@"
