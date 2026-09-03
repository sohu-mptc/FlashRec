# 参数

[English](configuration.md)

默认值与 `flashrec --help` 及 `BeamRecConfig`（`python/flashrec/config.py`）一致。

## 模型与 SID 词表

服务配置为 `--model-path` 加 `--sid-vocab-file`。引擎从 tokenizer 读取
`<s_a_0>` 风格 codebook token 以及 `<|sid_begin|>` / `<|sid_end|>`，自动推出
token 区间、codebook 大小和 boundary。不设 catalog 时做无约束解码（全词表
`lm_head`，不建 trie），只适合冒烟测试。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | （必填） | 模型目录（权重 + tokenizer）。 |
| `--quantization` | `fp8` | 权重量化。`fp8` = W8A8 per-channel；`nvfp4` 预留。 |
| `--kv-cache-dtype` | `fp8_e4m3` | KV cache 存储精度。 |
| `--sid-vocab-file` | 不设 | 合法 SID 目录（JSON），构建约束 trie，并触发布局推断。 |
| `--sid` | 不设 | 可选覆盖 `START:END/SIZE,...`。仅当 tokenizer 不用 `<s_a_0>` / `<\|sid_begin\|>` 约定时需要。 |
| `--system-prompt` / `--system-prompt-file` | 不设 | 共享 system prompt，拼接到请求 messages 前。 |
| `--warmup-user-a` / `--warmup-user-b` | 通用 probe | 两段不同的 user 文本，它们的最长公共前缀会钉进 radix cache。不设则只钉 chat template + system prompt。两边都设时，还可以钉上你自己流量里共享的 user 头。 |

OpenOneRec RecIF 的 `sid2pid.json` / `sid2iid.json` 用打包整数做 SID key。
转换成 `--sid-vocab-file` 需要的逗号分隔 key JSON：

```bash
bash scripts/build_catalog.sh
# DATA_DIR=/path/to/benchmark_data TASK=video|product|both LEVELS=4
python scripts/convert_recif_catalog.py --data-dir /path/to/benchmark_data
```

默认写出 `data/catalogs/sid2pid_beamrec_l4.json`（key 为 `"a,b,c,1"`）。
4 层 catalog 会推断末层 codebook 大小为 2，从而把 `<|sid_end|>` 计入序列分数。
启动日志会打印推断出的 `--sid ...`。

### SID 高级覆盖

只在推断不对、或 tokenizer 命名不同时使用：

| 参数 | 说明 |
|------|------|
| `--sid-token-range` | 限制 `lm_head` / beam 候选的 token 集合。闭区间 `start:end` 或逗号列表。 |
| `--sid-codebook-sizes` | 各层 codebook 大小，与 SID 深度对应，如 `8192,8192,8192`。 |
| `--sid-boundary-tokens` | 包裹 SID 的 `begin,end` token id。 |

显式 `--sid` 或拆分旗标若与推断结果冲突，启动会报错。

## Beam search 与生成

| 参数 | 默认 | 说明 |
|------|------|------|
| `--beam-width`（别名 `--n`） | `50` | 默认 beam 宽度；请求体 `n` 可覆盖。也决定 fused-expand 的 graph 捕获宽度——单实例服务一种主力 beam 宽度性能最优。 |
| `--max-tokens` | `5` | 默认最大生成 token 数（SID 步数）；请求体可覆盖。 |
| `--length-penalty` | `1.0` | beam 分数的长度惩罚。 |
| `--prompt` / `--messages-json` | 不设 | 离线单次生成（与 `--serve` 互斥）。 |

## 显存与后端

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mem-fraction-static` | `0.8` | 预留给 KV cache / 静态缓冲的显存比例。 |
| `--max-seq-len` | `4096` | 最大序列长度。 |
| `--cuda-graph-max-bs` | `800` | CUDA graph 捕获的最大 batch（beam 行数）。 |
| `--cuda-graph-capture-sizes` | `50,100,…,400,800` | 显式捕获尺寸；自动扩展到 beam 宽度的整数倍。 |
| `--attention-backend` | `flashinfer` | Attention 后端。 |
| `--flashinfer-variant` | `fa2` | FlashInfer kernel 变体。 |
| `--gpu-id` | `0` | CUDA 设备编号。 |

## 合批与调度

| 参数 | 默认 | 说明 |
|------|------|------|
| `--batch-slots` | 跟随 `--cuda-graph-max-bs` | 单波 beam 行槽位预算（≈ 请求数 × beam 宽度）。 |
| `--batch-wait-ms` | `4` | 第一次 expand 前的基础合批等待。 |
| `--batch-wait-max-ms` | `10` | 欠填时最长再等；也是队头请求可被拖延的上限。 |
| `--target-batch-requests` | `8` | 未凑够该请求数时继续等（受最长等待约束）；也是在途拉新请求的阈值。 |
| `--max-batch-requests` | `16` | 单波最多 admit 这么多请求。 |
| `--max-running-requests` | `64` | 最大并发运行请求数。 |
| `--decode-pack-min-requests` | `6` | 触发 decode 合包的最小在途请求数。 |
| `--decode-pack-ratio` | `0.75` | decode 合包的填充比例阈值。 |
| `--schedule-policy` | `lpm` | 出队策略：`lpm`（最长公共前缀优先，利于 radix）或 `fcfs`。 |
| `--lpm-aging-ms` | `300` | 等待超过该时长的请求会被提到 LPM 排序之前；`0` 关闭（纯 LPM 可能饿死短 prompt）。 |
| `--host-worker-threads` | `4` | host 侧 worker 线程数（tokenize / 回包）。 |
| `--pipeline-stages` | `0` | decode 流水线级数；`0` 关，`>=1` 开（压测中常不及默认 eager，启用前先测）。 |

### 宽 beam 调优（如 `n = 512` / `n = 1000`）

默认配置下一波只容得下一个 512-beam 请求（槽位预算 800），且纯 LPM 可能
饿死短 prompt。同一套 4096 配方覆盖 `n = 1000`（一波约 4 个 n=1000 请求）：

```bash
CUDA_GRAPH_MAX_BS=4096 BATCH_SLOTS=4096 LPM_AGING_MS=150 \
  BEAM_WIDTH=512 MODEL_PATH=... SID_VOCAB_FILE=... bash scripts/serve.sh
```

`--beam-width` 决定 fused-expand 的捕获宽度；graph 尺寸自动扩展到
`k × n`（见 `BeamRecConfig.resolved_cuda_graph_sizes`）。

## 功能开关（消融）

优化默认全开，可单独关闭：
`--disable-radix`、`--disable-cuda-graph`、`--disable-prefill-batch`、
`--disable-fused-expand`、`--disable-graph-expand`、`--disable-decode-pack`、
`--disable-fused-rms-fp8`、`--disable-fused-silu-fp8`、
`--disable-fused-qk-rope-kv`、`--disable-warmup`。
`--torch-compile`（仅 prefill；decode 仍走 CUDA graph）默认关。

## HTTP 服务

| 参数 | 默认 | 说明 |
|------|------|------|
| `--serve` | 关 | 启动 FastAPI 服务（否则走离线 generate）。 |
| `--host` | `127.0.0.1` | 监听地址。**无鉴权**——仅在可信网络或代理之后才绑定公网地址。 |
| `--port` | `8000` | 端口。 |
| `--log-level` | `info` | 日志级别。 |

`/v1/chat/completions` 的请求与响应字段见 [API](api.md)。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `FLASHREC_TORCH_PROFILER_DIR` | 不设 | Profiler 默认输出目录（回退到 `SGLANG_TORCH_PROFILER_DIR`，再回退 `/tmp`）。 |
| `FLASHREC_DENSE_FINALIZE` | `1` | beam finalize 的稠密快速路径；`0` 关闭。 |
| `FLASHREC_TRIE_DENSE_MAX_CELLS` | `2^29` | SID trie 从稠密切到 CSR 稀疏存储的最大 cell 数。 |

`scripts/serve.sh` 另外读取 `MODEL_PATH`、`HOST`、`PORT`、
`CUDA_VISIBLE_DEVICES`、`MEM_FRACTION_STATIC`、`QUANTIZATION`、
`KV_CACHE_DTYPE`、`TORCH_COMPILE`、`CUDA_GRAPH_MAX_BS`、
`BATCH_SLOTS`、`BATCH_WAIT_MS`、`BATCH_WAIT_MAX_MS`、`TARGET_BATCH_REQUESTS`、
`MAX_BATCH_REQUESTS`、`SCHEDULE_POLICY`、`LPM_AGING_MS`、`PIPELINE_STAGES`、
`HOST_WORKER_THREADS`、`BEAM_WIDTH`、`MAX_TOKENS`、`LOG_LEVEL`、
`ATTENTION_BACKEND`、`SID_VOCAB_FILE`、`SID`、`SID_TOKEN_RANGE`、
`SID_CODEBOOK_SIZES`、`SID_BOUNDARY_TOKENS`、`SYSTEM_PROMPT_FILE`、
`WARMUP_USER_A`、`WARMUP_USER_B`，以及
`EXTRA_SERVER_ARGS`（追加/覆盖任意 CLI 参数）。`SID_VOCAB_FILE` 默认为
`data/catalogs/sid2pid_beamrec_l4.json`（相对仓库根目录）；布局从 tokenizer
推断。`SID_VOCAB_FILE=`（置空）表示无约束冒烟测试。`SID`
（`START:END/SIZE,...`）和三个拆分 `SID_*` 变量用来覆盖推断结果。

可调参数模板见 `scripts/serve.env.example`。复制为 `scripts/serve.env`
（已在 gitignore 中）后编辑，或用 `SERVE_ENV=/path/to.env bash scripts/serve.sh`
指定。文件里写 `${VAR:-...}` 时，启动命令已导出的环境变量优先；写成裸
`VAR=value` 则会覆盖命令行导出的值。该文件会被 bash `source` 执行，只加载
可信路径的文件。

## 测试

```bash
python -m pytest
```

与一个正在运行的 SGLang beam 服务做在线对齐（可选）：

```bash
SGLANG_BEAM_URL=http://127.0.0.1:PORT FLASHREC_URL=http://127.0.0.1:8000 \
  PYTHONPATH=python python -m unittest tests.test_parity -v
```

与 HuggingFace `transformers` 做精度对比（需要 CUDA + GenRec 模型目录）：

```bash
FLASHREC_DIFF_MODEL=/path/to/OneRec-1.7B \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff -v

FLASHREC_DIFF_MODEL=/path/to/OneRec-1.7B \
FLASHREC_DIFF_BEAMS=1,20,50,128,512 \
FLASHREC_DIFF_QUANT=bf16 \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff.TestBeamSearchDiff -v
```

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `FLASHREC_DIFF_MODEL` | （必填） | checkpoint 目录 |
| `FLASHREC_DIFF_QUANT` | `fp8` | `fp8` 为服务路径；`bf16` 与 HF 更近 |
| `FLASHREC_DIFF_BEAMS` | `4,8` | beam 宽度列表 |
| `FLASHREC_DIFF_CATALOG` | 空 | 设置后走 trie 约束，与 HF 的 codebook-only 口径不再一致 |

测的是 codebook 约束的 log-prob / SID 集合重叠，不是 OneRec recall。
OneRec-1.7B 公开权重为 BF16 训练，FP8 路径是加载时量化，重叠下降不是框架问题；
SoHuRec-1.7B / SoHuRec-0.6B（FP8 训练）在同一路径上重叠更高。
两者均在 RTX 5090 上，见 [评测](baselines.zh-CN.md)。生产建议使用 FP8 训练的
模型。可运行入口：[示例](../examples/README.md)。

## Profiling 与 Trace

接口与 SGLang 对齐，可直接给 `sglang.bench_serving --profile` 用：

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"./profiles","num_steps":20,"activities":["CPU","GPU"],"with_stack":true}'
curl -s -X POST http://127.0.0.1:8000/stop_profile   # 达到 num_steps 时会自动停止
```

Body 字段：`output_dir`、`num_steps`、`start_step`、`activities`
（`CPU`/`GPU`）、`profile_by_stage`、`with_stack`、`record_shapes`、
`profile_prefix`。

Chrome trace 里会有 `flashrec.batch.wait` / `flashrec.prefill` /
`flashrec.decode_fwd` / `flashrec.expand` / `flashrec.finalize`
区间。start/stop 在 GPU worker 线程上生效——不要在 HTTP 线程上直接包
`torch.profiler`。
