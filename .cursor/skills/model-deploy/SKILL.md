---
name: model-deploy
description: >-
  Deploy and serve GenRec checkpoints with FlashRec (install, serve.sh,
  SID trie, wide-beam knobs, health/curl, FP8, profiling). Use when the user
  asks to 部署模型, 启动服务, 上线, serve, launch FlashRec, MODEL_PATH,
  /v1/chat/completions, or expose an HTTP beam-search endpoint.
---

# 部署 FlashRec

默认单进程部署。不要引入张量并行、多卡、Docker 编排或鉴权网关，除非用户明确要求。
完整旋钮见 `docs/configuration.zh-CN.md`。目录与评测分别走 `sid-catalog`、`recif-eval`。

## 前置

- Linux + NVIDIA GPU + CUDA 12 工具链；Python ≥ 3.10
- HuggingFace 格式 checkpoint（`config.json` + tokenizer + safetensors）
- 当前只支持 **Qwen3 dense**（0.6B–14B、OneRec-1.7B、同架构 GenRec）。MoE / 其他架构走 `add-model`
- 32 GB 显存下 FP8 大约到 14B

先确认环境：

```bash
python -m flashrec.check_env
```

## 安装

```bash
pip install -e .
# 或 wheel：
bash scripts/build_wheel.sh && pip install dist/flashrec-*.whl
```

## 启动服务

默认入口是 `scripts/serve.sh`（会设 `PYTHONPATH` 并展开调度环境变量）。
可调参数模板：`scripts/serve.env.example`（复制为 `scripts/serve.env` 或设 `SERVE_ENV`）。

**服务配置**是 `MODEL_PATH` + `SID_VOCAB_FILE`。引擎从 tokenizer 推断 SID 布局
（`<s_a_0>` codebook + `<|sid_begin|>` / `<|sid_end|>`）。不设 catalog 时做全词表
无约束解码（无 trie），只适合冒烟。

```bash
SID_VOCAB_FILE=data/catalogs/sid2pid_beamrec_l4.json \
MODEL_PATH=/path/to/model bash scripts/serve.sh
# 等价：
flashrec --serve --model-path /path/to/model --port 8000 \
  --sid-vocab-file data/catalogs/sid2pid_beamrec_l4.json
```

常用覆盖：`HOST`、`PORT`、`CUDA_VISIBLE_DEVICES`、`QUANTIZATION`、`KV_CACHE_DTYPE`、
`MEM_FRACTION_STATIC`、`CUDA_GRAPH_MAX_BS`、`BATCH_SLOTS`、`BEAM_WIDTH`、
`EXTRA_SERVER_ARGS`。

### OneRec-1.7B（文档中的服务配置）

先构建 catalog（见 `sid-catalog`），再：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/OneRec-1.7B PORT=8000 HOST=127.0.0.1 \
QUANTIZATION=fp8 KV_CACHE_DTYPE=fp8_e4m3 \
CUDA_GRAPH_MAX_BS=800 BATCH_SLOTS=800 \
SID_VOCAB_FILE=data/catalogs/sid2pid_beamrec_l4.json \
BEAM_WIDTH=50 \
  bash scripts/serve.sh
```

tokenizer 不用 `<s_a_0>` / `<|sid_begin|>` 约定时，再设
`SID=START:END/SIZE,...` 覆盖推断。

### 宽 beam（`n = 512` / `n = 1000`）

默认槽位 800，一波只能进一个 512-beam 请求，且纯 LPM 可能饿死短 prompt。
`n = 1000` 用同一套 4096 槽位（约 4 路并发）：

```bash
CUDA_GRAPH_MAX_BS=4096 BATCH_SLOTS=4096 LPM_AGING_MS=150 \
  BEAM_WIDTH=512 MODEL_PATH=/path/to/model \
  SID_VOCAB_FILE=data/catalogs/sid2pid_beamrec_l4.json \
  bash scripts/serve.sh
```

`--beam-width` 决定 fused-expand 的捕获宽度；graph 尺寸会扩成 `k × n`。
**单实例只服务一种主力 beam 宽度。**

## 就绪检查与请求

```bash
curl -sf http://127.0.0.1:8000/health
# {"status":"ok"}

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"..."}],"n":50,"max_tokens":5,"temperature":0}'
```

- `n`：该请求 beam 宽度（覆盖服务端 `--beam-width`）
- `temperature=0`：确定性 top-k；`>0` 为 Gumbel top-k，噪声只用于排序，返回的 `sglext.sequence_score` 不含噪声
- 离线单次：`flashrec --model-path ... --sid-vocab-file ... --prompt "..." --beam-width 50 --max-tokens 5`

Profiling（与 `sglang.bench_serving --profile` 对齐）：`POST /start_profile`、`POST /stop_profile`。
不要在 HTTP 线程上包 `torch.profiler`。

## 精度与显存

| 目标 | 设置 |
|------|------|
| 默认生产 | `--quantization fp8`（W8A8 per-channel）+ `--kv-cache-dtype fp8_e4m3` |
| 纯 BF16 | `--quantization` 传非 `fp8` 的值 |
| 预量化 FP8 checkpoint | 带 `weight_scale` 即可加载 |
| OOM | 降 `--mem-fraction-static`，或改小 `--cuda-graph-max-bs` / `--batch-slots` / `--beam-width` |

FP8 在小模型窄 beam 上不一定更快，主要省权重与 KV 显存。
OneRec-1.7B 等 **BF16 训练** 的 checkpoint 走加载时量化时，相对 HuggingFace 的
beam 重叠会下降，这不是框架问题；生产建议用 **FP8 训练**（或带 `weight_scale`
的预量化）权重。对照见 `docs/baselines.zh-CN.md`。

## 安全

服务**无鉴权**，默认绑 `127.0.0.1`。只在可信网或反向代理后才设 `HOST=0.0.0.0`。

## 故障排查

1. `python -m flashrec.check_env`：CUDA / flashinfer / sgl-kernel / 驱动
2. `/health` 不通：看进程是否还在、端口、CUDA graph 捕获是否卡在启动
3. 非法 SID 或超长输出：确认 `SID_VOCAB_FILE` 对应该 checkpoint；启动日志应有 `Inferred --sid ...`。tokenizer 命名不同时才设 `SID=`
4. 宽 beam 吞吐不随并发上升：槽位不够（`BATCH_SLOTS` < `n` × 并发）
5. 新架构加载失败：当前 `ModelEngine` 写死 `Qwen3ForCausalLM`，见 `add-model`
