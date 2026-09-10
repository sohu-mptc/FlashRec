# RecIF beam × concurrency eval

FlashRec-only matrix on OpenOneRec RecIF-Bench **video**.
Measures QPS, latency (p50/p90/p99), plus Recall / NDCG / invalid_rate.

## Local models & data

Prefer [`../local/`](../local/) (gitignored payloads). See that README for layout.

```bash
mkdir -p benchmark/local/models benchmark/local/data
ln -s /path/to/OneRec-1.7B benchmark/local/models/OneRec-1.7B
ln -s /path/to/OneRec-0.6B benchmark/local/models/OneRec-0.6B
ln -s /path/to/OpenOneRec-RecIF/benchmark_data benchmark/local/data/benchmark_data
cp benchmark/local/env.example benchmark/local/env
```

## Prerequisites

```bash
# catalog (once) — DATA_DIR defaults to benchmark/local/data/benchmark_data when present
bash scripts/build_catalog.sh
# or: DATA_DIR=/path/to/benchmark_data bash scripts/build_catalog.sh

# optional: pandas for sample dump / client
pip install 'flashrec[eval]'
```

Needs one GPU (`FLASHREC_GPU`, default `0`).

## Speed / quality matrix (recommended)

```bash
# one model (from local/env or MODEL_PATH)
SMOKE=1 bash benchmark/recif/bench_compare.sh
bash benchmark/recif/bench_compare.sh

# two models sequentially
MODELS="OneRec-1.7B OneRec-0.6B" SMOKE=1 bash benchmark/recif/bench_compare.sh
MODELS="OneRec-1.7B OneRec-0.6B" bash benchmark/recif/bench_compare.sh
```

Outputs under `results/onerec_beam_conc_bench_<model>_<stamp>/`
(`MATRIX_REPORT.md`, `matrix_summary.json`, per-cell `summary.json`).

Regenerate the report:

```bash
python benchmark/recif/summarize_compare.py results/onerec_beam_conc_bench_<model>_<stamp>
```

## Full quality + speed matrix

Same FlashRec-only loop; default output prefix is `onerec_beam_conc_matrix_*`
(and smoke beams include `128`):

```bash
bash benchmark/recif/run_matrix.sh
MODELS="OneRec-1.7B OneRec-0.6B" bash benchmark/recif/run_matrix.sh
```

## Single cell (server already running)

```bash
python -m flashrec.benchmark.recif \
  --engine flashrec \
  --server-url http://127.0.0.1:8000 \
  --data-dir benchmark/local/data/benchmark_data \
  --catalog data/catalogs/sid2pid_beamrec_l4.json \
  --out-dir /tmp/cell --task video --n 50 --concurrency 8 --sample-size 200
```

`--engine sglang` still works for manual baseline clients (`POST /generate` +
`--model-path`); the matrix scripts no longer launch or compare SGLang.
