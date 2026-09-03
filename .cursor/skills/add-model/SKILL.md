---
name: add-model
description: 给 FlashRec 引擎接入一个新模型架构（新的 HF checkpoint / 非 Qwen3 结构）。涵盖模型定义、权重合并加载、FP8 双路径、融合 kernel 接线、CUDA graph 兼容、精度校验、以及压测+trace 验证闭环。当用户要"增加/支持/接入新模型"时使用。
---

# 给 FlashRec 增加新模型

FlashRec 当前 TP=1，唯一模型是 Qwen3（`python/flashrec/models/qwen3.py`）。接入新模型不是"写个 nn.Module"就完了——现有优化大多在模型层有接线点，漏掉任何一个都会直接掉性能或破坏 CUDA graph。按下面顺序做。

## 0. 先读这些（每次都读，不要凭记忆）

- `python/flashrec/models/qwen3.py` — 参考实现，所有优化点的接线范例
- `python/flashrec/models/weight.py` — 权重合并加载范例
- `python/flashrec/engine/engine.py` — 装配点（config 解析、KV pool 预算、AttentionBackend、模型实例化）
- `python/flashrec/sid_layout.py` — tokenizer 约定与 SID 布局推断
- `docs/architecture.md` — 请求路径与模块地图

## 1. 模型定义（models/<name>.py）

复制 `qwen3.py` 作为骨架，逐项对照新模型的 HF `config.json` 改。**必须保留的优化结构**（每一项都有实测收益，不是风格偏好）：

1. **QKV 融合单 GEMM**（实测吞吐收益明显，三个小 GEMM 换一个大 GEMM）：不要写 q_proj/k_proj/v_proj 三个 Linear，写一个 `qkv_proj = Linear(hidden, (n_q + 2*n_kv) * head_dim)`，forward 里用 `view` 切成 per-head 视图（零拷贝），rope/store kernel 接受显式 stride。同理 MLP 用 `gate_up_proj` 融合。
2. **FP8 双路径 forward**：每个模块的 forward 接受可选 `q_fp8` / `a_scale`（上一层 RMSNorm 融合量化的输出），有则走 `Linear.forward_fp8`，无则走普通路径。DecoderLayer 里 `_use_fused_quant()` 门控，`input_layernorm.quant_fp8(x, residual)` 产出下一个 GEMM 的 FP8 激活。
3. **三个融合 kernel 开关**，构造参数逐层透传，默认 True，engine 从 config 读（`enable_fused_rms_fp8` / `enable_fused_silu_fp8` / `enable_fused_qk_rope_kv`）：
   - `fused_qk_norm_rope_store_fp8`（kernel/qk_rope_kv.py）：QK-RMSNorm + RoPE + FP8 KV 写入单 kernel，仅当 `pool.dtype == float8_e4m3fn` 且 `out_cache_loc` 非空时走；返回 False 时必须有完整 fallback（q_norm/k_norm → `apply_rope_and_store_kv` → 再 fallback 到 `self.rotary`）。**新模型若没有 qk-norm，需要改这个 kernel 或跳过 norm 部分，不要静默传单位权重以外的东西**。
   - `silu_and_mul_per_token_quant_fp8`（kernel/silu_fp8.py）：激活函数不是 SwiGLU 的模型（如 GELU）不能直接用，需新写融合 kernel 或退化为 eager + `per_token_quant_fp8`。
   - `RMSNorm.quant_fp8`（layers/norm.py）：LayerNorm 模型（非 RMSNorm）同理需要新路径。
4. **attention 调用约定**：`self.attn.forward(q.contiguous(), k, v, layer_id, batch, skip_store=stored)` — 融合 kernel 已写过 KV 时必须传 `skip_store=True`，否则双写。q 在融合 rope 后是 qkv buffer 的 strided view，FlashInfer 要 contiguous，这一次小 copy 是刻意保留的（k/v 不需要）。
5. **residual 双流水**：forward 返回 `(hidden, residual)` 二元组，RMSNorm 的 fused add-residual 签名 `norm(x, residual)`。不要改成单张量往返（会多一次 add kernel，且破坏 `quant_fp8` 融合）。
6. **lm_head()**：返回权重张量（不是 Linear），tie_word_embeddings 时回退 `embed_tokens.weight`。`RestrictedLMHead`（logits.py）会对它做 `index_select` 出 special-token 子集——GenRec 场景 LM-head GEMM 从 vocab 全量缩到 ~1.5k 行，不要绕过。

**GQA 参数**：`num_kv < num_qo` 时 KV pool、AttentionBackend、qkv 融合的维度全部由 config 的 `num_key_value_heads`/`head_dim` 驱动，不要 hardcode。MoE / MLA / 滑窗注意力等结构性差异超出现有 AttentionBackend 能力，先在 `attention/flashinfer.py` 层面评估，再动模型层。

## 2. 权重加载（weight.py）

在 `load_hf_config` 里加新架构分支（按 `config.json` 的 `architectures` 字段分发；当前它无条件按 Qwen3 解析，第一个新模型进来时要重构成注册表）。`load_weights` 的关键约定：

- **合并在加载时做**：`merge_qkv_weights` / `merge_gate_up_weights` 沿输出维 cat，weight_scale 同步 cat，**不 requantize**（FP8 checkpoint 的 per-channel scale 直接拼）。新模型的投影名不同（如 `wqkv`、`w1/w3`）就加对应的 pending bucket。
- FP8 checkpoint：`.weight_scale` / `.weight_scale_inv` 先扫一遍建 `scale_map`，权重 dtype 为 float8_e4m3fn 时带 scale 调 `Linear.load(w, weight_scale=...)`；BF16 checkpoint + `--quantization fp8` 时 `Linear.load(w, quantize_fp8=True)` 在线量化（per-channel absmax）。
- 模型目录名带 "fp8" ≠ 推理走 FP8。检查 `config.json` 有无 `quantization_config`；没有的话必须显式 `--quantization fp8` 才会启用 FP8 GEMM。

## 3. 引擎装配（engine/engine.py + config.py）

- `engine.py` 目前无条件 `load_hf_config`（`ModelEngine.__init__` 内）后实例化 `Qwen3ForCausalLM`，两处都要改成按架构分发。KV pool 的 token 预算用 `bytes_per = 2 * n_kv * head_dim * n_layers * dtype_size`，新模型只要 config 字段对就自动正确。
- SID 布局由 `sid_layout.py` 从 tokenizer 的 `<s_a_0>` codebook 与 `<|sid_begin|>` / `<|sid_end|>`，再结合 `--sid-vocab-file` 推断，**不再写死在 `config.py`**。新词表沿用这套 added-token 命名即可；否则扩展 `sid_layout.py`，或让用户传 `--sid`。`RestrictedLMHead` 仍按推断出的 token 区间做 `index_select`。
- **CUDA graph**：`DecodeGraphRunner`（engine/graph.py）捕获 model forward + restricted LM-head + fused expand 整段。新模型 forward 里不能有 capture-unsafe 操作：不能有 `.item()` / `.tolist()` / host 分支依赖张量值 / 动态 shape 分配。凡是 `if tensor 条件` 的门控（如 fused kernel 的 fallback 判断）必须在 capture 前静态确定。捕获失败通常表现为 warmup 阶段 crash 或静默 fallback 到 eager（QPS 直接腰斩，去 trace 里看有没有 `replay`）。

## 4. 精度校验（改一行验一行，不要攒到最后)

1. **单测**：仿照 `tests/test_mlp.py`（kernel vs PyTorch 参考实现，`torch.testing.assert_close`）给新模块写等价性测试；`tests/test_parity.py` 有算法层 parity 模式。跑 `PYTHONPATH=python python -m unittest discover -s tests`。
2. **对照 SGLang / HF**：起 SGLang 跑同一 checkpoint，`SGLANG_BEAM_URL=http://... PYTHONPATH=python python -m unittest tests.test_parity`（temperature=0 比 choices 与 scores）。
3. **temperature=0 复现性**：跨请求结果必须逐位稳定。**采样 N≥120 且跨独立进程**，5–10 次采样下过的结论历史上错过两次。若有漂移，先 hash 每步 forward 的输入（positions/seq_lens/kv_indices）确认输入恒定，再怀疑 kernel——上次漂移的根因是 host 侧缓存键用了 `id(req)`（地址回收串状态），不是数值噪声。注意 instrumentation 本身的 sync 会破坏 CUDA graph capture 掩盖现象。

## 5. 性能验证 + trace 闭环

每一轮改动都走同一个闭环，**不优于基线就回退**（"感觉会更快"不算数）：重启 server（当前工作树代码）→ 固定负载压测 → 采 trace → 等 trace 落盘完成。每轮固定 model-path / `--sid-vocab-file` / beam 宽度 / 并发，换了任何一项旧基线就不可复现。生产路径是 `--model-path` + `--sid-vocab-file`，不要把旧 checkpoint 的 token id 写进脚本。

压测用 `scripts/eval_beam_matrix.py`（单格）或 `scripts/run_sglang_flashrec_matrix.sh`（beam × 并发矩阵），起服务见 `scripts/serve.sh` 与 model-deploy skill。

采 trace（server 端口按 `PORT`，`serve.sh` 默认 8000）：

```bash
export FLASHREC_TORCH_PROFILER_DIR=./profiles   # 落盘目录，或在请求体传 output_dir
curl -sf -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"num_steps":200,"profile_prefix":"mymodel","output_dir":"./profiles/mymodel_r1"}'
# ... 打压测流量 ...
curl -sf -X POST http://127.0.0.1:8000/stop_profile
```

接口细节与区间名见 profile-serving skill。

**trace 落盘要 10–30s**，期间压测数据会被污染（p99 明显抬高）——必须等文件大小稳定，落盘完成前的压测数字全部作废。

trace 用 https://ui.perfetto.dev/ 打开，按怀疑方向看：整体 GPU 空泡率先定位哪段疼；gap 归因到具体 python frame；forward 是 GPU-bound 还是 launch-bound（新模型最常见问题：eager 逐层 launch）；decode 步间节奏 / duty cycle；cudaMalloc/Free/Sync stall 与 graph replay 是否生效；以及某个埋点 span 的内部分解。

新代码里加埋点用 `from flashrec.profiler import trace_range`，`with trace_range("flashrec.<区域名>"):` 包住 scheduler/模型的 host 区域，会同时进 chrome trace 和 NVTX。

**测量三坑**（违反任何一条得出的结论直接作废）：

1. torch profiler 把 host/Python 开销放大 2–3 倍——优化方向要用**无 profiler 的压测数字**定，trace 只用来定位相对热点。
2. radix 冷热差可达单轮噪声的数倍——对比必须同温（都跑 warmup，或都冷启动）。
3. 单轮压测本身有噪声，先跑几轮基线量出噪声带——收益落在噪声内的改动一律回退，保持代码简单。

## 6. 验收清单

- [ ] `PYTHONPATH=python python -m unittest discover -s tests` 全绿
- [ ] SGLang/HF parity：temperature=0 choices + scores 一致
- [ ] t=0 复现性：N≥120 跨进程无漂移
- [ ] trace 确认 decode 走 graph `replay`（不是 eager），forward 为 GPU-bound
- [ ] FP8 生效确认：trace 里 GEMM kernel 是 fp8_scaled_mm 系，不是 bf16 cutlass
- [ ] 记录新模型基线（commit hash、负载参数、QPS）
