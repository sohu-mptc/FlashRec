# Local models & datasets (not committed)

Put checkpoints and RecIF data here. Everything except this README /
`.gitignore` / `env.example` is ignored by git.

```
benchmark/local/
  models/
    OneRec-1.7B/          # HF checkpoint (config.json + weights)
    OneRec-0.6B/          # optional second model
  data/
    benchmark_data/       # OpenOneRec-RecIF benchmark_data layout
  env                     # optional; copy from env.example
```

## Setup

```bash
mkdir -p benchmark/local/models benchmark/local/data

# example: symlink existing downloads
ln -s /path/to/OneRec-1.7B benchmark/local/models/OneRec-1.7B
ln -s /path/to/OneRec-0.6B benchmark/local/models/OneRec-0.6B
ln -s /path/to/OpenOneRec-RecIF/benchmark_data benchmark/local/data/benchmark_data

cp benchmark/local/env.example benchmark/local/env
# edit MODEL_PATH / MODELS / DATA_DIR if needed
```

## Run one or two models

```bash
# uses benchmark/local/env if present; else defaults under this directory
bash benchmark/recif/bench_compare.sh

# two checkpoints sequentially (names relative to models/)
MODELS="OneRec-1.7B OneRec-0.6B" bash benchmark/recif/bench_compare.sh

# absolute paths also work
MODELS="/abs/OneRec-1.7B /abs/SoHuRec-1.7B" \
  DATA_DIR=benchmark/local/data/benchmark_data \
  bash benchmark/recif/bench_compare.sh
```

Each model writes its own `results/…_<model_tag>_…` directory.
