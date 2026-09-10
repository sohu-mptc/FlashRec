---
name: recif-eval
description: >-
  Run RecIF beam×concurrency eval for FlashRec (and optional manual
  SGLang/vLLM/TRT-LLM clients). Use when the user asks to 评测, 对照, 压测,
  matrix, RecIF, recall@32, invalid_rate, run_sglang_flashrec_matrix,
  bench_sglang_compare, or eval_beam_matrix.
---

# RecIF 评测与 baseline

公开数字用 **OneRec-1.7B**（快手 OneRec-1.7B × OpenOneRec RecIF-Bench **video**）。
历史对照协议与表格见 `docs/baselines.zh-CN.md`。编排脚本在 `benchmark/recif/`；
客户端在 `python -m flashrec.benchmark.recif`（`scripts/` 下仍有兼容 shim）。

**矩阵脚本只跑 FlashRec**（不再自动拉起 / 对照 SGLang）。需要 baseline 时用单格
客户端手动打 `--engine sglang|vllm|…`，启动命令见 `docs/baselines.zh-CN.md` 文末。

**不要把不同 workload 的 QPS 直接相除。** OneRec-1.7B prompt 约 2.5k token（~500 条历史 SID）；
同一引擎 `n=50` 并发 8 时约 26 QPS，饱和约 28 QPS。生产短 prompt（约 300 token）
`n=50` 饱和约 220 QPS（1.7B）/ 303 QPS（0.6B）；`n=1000` 为 24.6 QPS（1.7B）/
32 QPS（0.6B）。见 `docs/baselines.zh-CN.md`。

**不公平对照警告：** FlashRec 默认 SID trie；SGLang-master / SGLang 0801 / vLLM / HF /
TensorRT-LLM 在本仓库文档中是**开放词表** beam。比吞吐时对齐 `beam_width`、
`max_tokens`、模型、硬件；比质量时必须报 `invalid_rate`。不要引用 FlashRec
conc=1 的 recall@32（unique-beam 塌缩）。

## 矩阵：FlashRec only

需要一张卡（默认 `FLASHREC_GPU=0`）。

```bash
# 1. catalog（一次）
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
  bash scripts/build_catalog.sh

# 2. beam {50,128,512} × 并发 {1,8,16,32}
MODEL_PATH=/path/to/OneRec-1.7B \
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
SMOKE=1 bash benchmark/recif/bench_compare.sh

MODEL_PATH=/path/to/OneRec-1.7B \
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
bash benchmark/recif/bench_compare.sh
```

产物：`results/onerec_beam_conc_bench_<stamp>/`（`MATRIX_REPORT.md`、
`matrix_summary.json`、每格 `summary.json` /
`candidates.csv` / `per_sample_metrics.csv` / `latencies.jsonl`）。
`results/` 已 gitignore。从一次 run 重生成报告：

```bash
python benchmark/recif/summarize_compare.py results/onerec_beam_conc_bench_<stamp>
```

可覆盖：`BEAMS`、`CONCS`、`SAMPLE_SIZE`、`FLASHREC_GPU`、`FLASHREC_PORT`、
`PYTHON`、`SKIP_LAUNCH`、`OUTDIR`、`EXTRA_FLASHREC_ARGS`。FlashRec 按 n 自动设
graph / 槽位（50→800、128→2048、≥512→4096），换宽度会重启服务。

FlashRec 侧只需要 `SID_VOCAB_FILE`（脚本默认 `data/catalogs/sid2pid_beamrec_l4.json`）；
布局从 tokenizer 推断，不必设 `SID`。

## 单格客户端

服务已在跑时：

```bash
python -m flashrec.benchmark.recif \
  --engine flashrec \
  --server-url http://127.0.0.1:8000 \
  --data-dir /path/to/benchmark_data \
  --catalog data/catalogs/sid2pid_beamrec_l4.json \
  --out-dir /tmp/cell --task video --n 50 --concurrency 8 --sample-size 200
```

`--engine sglang` 走 `POST /generate`（还要 `--model-path`）。其它引擎走
`/v1/chat/completions`。看 `summary.json` 的 `qps`、`latency_p50/90/99_s`、
`invalid_rate`、`metrics.recall@32` / `ndcg@32`。

纯吞吐（无 RecIF 质量指标）见 `python -m flashrec.benchmark.serving` /
`benchmark/serving/README.md`。

## 其他引擎（手动）

启动命令在 `docs/baselines.zh-CN.md` 文末，镜像不随仓库提供。要点：

- SGLang-master：请求 `sampling_params.beam_width`（`POST /generate`）、
  `--disable-overlap-schedule`、`--max-running-requests ≥ (n+1) × 并发`
  （expand 会复制 request-pool 槽位）
- vLLM：`use_beam_search=true`；`--max-logprobs ≥ 2n`（n=32 至少 100）
- TensorRT-LLM：`--max_beam_width` 与请求 `best_of` 一致；32 GB 上 n=512 通常不支持

## 精度 / 对齐测试（非检索质量）

```bash
PYTHONPATH=python python -m unittest discover -s tests -v

SGLANG_BEAM_URL=http://127.0.0.1:PORT FLASHREC_URL=http://127.0.0.1:8000 \
  PYTHONPATH=python python -m unittest tests.test_parity -v

FLASHREC_DIFF_MODEL=/path/to/model \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff -v
```
