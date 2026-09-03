---
name: profile-serving
description: >-
  Capture torch.profiler traces on a running FlashRec server. Use when the
  user asks to profile, trace, Chrome trace, start_profile, stop_profile,
  sglang.bench_serving --profile, or 性能剖析.
---

# 服务端 Profiling

接口与 SGLang 对齐，可直接给 `sglang.bench_serving --profile` 用。start/stop
在 **GPU worker 线程**上生效——不要在 HTTP 线程上包 `torch.profiler`。

## 采集

默认输出目录：`FLASHREC_TORCH_PROFILER_DIR` → `SGLANG_TORCH_PROFILER_DIR` → `/tmp`。
`scripts/serve.sh` 会设成仓库下 `profiles/`。

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"./profiles","num_steps":20,"activities":["CPU","GPU"],"with_stack":true}'

curl -s -X POST http://127.0.0.1:8000/stop_profile
```

到 `num_steps` 会自动停。Body：`output_dir`、`num_steps`、`start_step`、
`activities`（`CPU`/`GPU`）、`profile_by_stage`、`with_stack`、`record_shapes`、
`profile_prefix`。

Chrome trace 区间名：`flashrec.batch.wait` / `prefill` / `decode_fwd` /
`expand` / `finalize`。

## 解读

- 窄 beam + 低并发：`batch.wait` 或单请求开销主导，不要据此调 graph
- 宽 beam：看 `expand` 是否在 CUDA graph 内；eager expand 说明 `bs % n != 0` 或捕获宽度不够
- `finalize` 过重：检查 `FLASHREC_DENSE_FINALIZE`（默认 `1`）
- 对照 SGLang 时两侧都要用 FlashInfer，并关掉 SGLang overlap schedule
