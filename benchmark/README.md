# FlashRec benchmarks

Layout mirrors SGLang: repo-level orchestration under `benchmark/`, importable
clients under `python/flashrec/benchmark/`.

| Area | Repo scripts | Python module |
| --- | --- | --- |
| RecIF quality + throughput matrix | [`recif/`](recif/) | `python -m flashrec.benchmark.recif` |
| Online serving throughput (wide beam) | [`serving/`](serving/) | `python -m flashrec.benchmark.serving` |
| Local models / RecIF data (gitignored) | [`local/`](local/) | — |

Published numbers and fairness notes: [`docs/baselines.md`](../docs/baselines.md).

## Local models & data (not in git)

Put checkpoints and RecIF `benchmark_data` under [`local/`](local/) (only the
README / `.gitignore` / `env.example` are tracked):

```bash
mkdir -p benchmark/local/models benchmark/local/data
ln -s /path/to/OneRec-1.7B benchmark/local/models/OneRec-1.7B
ln -s /path/to/OneRec-0.6B benchmark/local/models/OneRec-0.6B
ln -s /path/to/OpenOneRec-RecIF/benchmark_data benchmark/local/data/benchmark_data
cp benchmark/local/env.example benchmark/local/env
```

Run **one** model (defaults from `local/env` or `local/models/OneRec-1.7B`):

```bash
bash benchmark/recif/bench_compare.sh
```

Run **two** (or more) models back-to-back:

```bash
MODELS="OneRec-1.7B OneRec-0.6B" SMOKE=1 bash benchmark/recif/bench_compare.sh
```

Each model gets its own `results/…_<model>_…` directory.

RecIF matrix scripts run **FlashRec only** (no SGLang launch/compare).
Compatibility shims under `scripts/` still work (`scripts/bench_sglang_compare.sh`,
`scripts/eval_beam_matrix.py`, …) and forward here.
