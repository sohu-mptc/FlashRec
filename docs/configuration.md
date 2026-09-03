# Configuration

[简体中文](configuration.zh-CN.md)

Defaults match `flashrec --help` and `BeamRecConfig` (`python/flashrec/config.py`).

## Model & SID vocabulary

Recommended serving configuration is `--model-path` plus `--sid-vocab-file`. The engine reads
`<s_a_0>`-style codebook tokens and `<|sid_begin|>` / `<|sid_end|>` from the
tokenizer and infers the token range, codebook sizes, and boundary tokens.
Leave the catalog unset for an unconstrained smoke test (full-vocabulary
`lm_head`, no trie).

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Model directory (weights + tokenizer). |
| `--quantization` | `fp8` | Weight quantization. `fp8` = W8A8 per-channel; `nvfp4` reserved. |
| `--kv-cache-dtype` | `fp8_e4m3` | KV-cache storage dtype. |
| `--sid-vocab-file` | unset | Valid-SID catalog (JSON); builds the constraint trie and triggers layout inference. |
| `--sid` | unset | Optional override `START:END/SIZE,...` for a tokenizer with a different codebook-token naming than `<s_a_0>` / `<\|sid_begin\|>`. |
| `--system-prompt` / `--system-prompt-file` | unset | Shared system prompt prepended to request messages. |
| `--warmup-user-a` / `--warmup-user-b` | generic probes | Two distinct user texts whose longest common prefix is pinned in the radix cache. Unset pins chat template + system prompt only. Set both to also pin a shared user-head from your own traffic. |

OpenOneRec RecIF stores SIDs as packed integers in `sid2pid.json` /
`sid2iid.json`. Convert them to the comma-key JSON `--sid-vocab-file` expects:

```bash
bash scripts/build_catalog.sh
# DATA_DIR=/path/to/benchmark_data TASK=video|product|both LEVELS=4
python scripts/convert_recif_catalog.py --data-dir /path/to/benchmark_data
```

Default output is `data/catalogs/sid2pid_beamrec_l4.json` (keys `"a,b,c,1"`).
A 4-level catalog infers a last codebook of size 2 so `<|sid_end|>` is included
in the sequence score. Startup logs the inferred `--sid ...`.

### Advanced SID overrides

Needed only when inference is wrong or the tokenizer uses a different naming
scheme:

| Flag | Description |
|------|-------------|
| `--sid-token-range` | Token ids the `lm_head` / beam candidates are restricted to. Closed range `start:end` or comma list. |
| `--sid-codebook-sizes` | Per-level codebook sizes matching SID depth, e.g. `8192,8192,8192`. |
| `--sid-boundary-tokens` | `begin,end` token ids wrapping a SID. |

An explicit `--sid` or split flag that disagrees with the inferred layout is an
error.

## Beam search & generation

| Flag | Default | Description |
|------|---------|-------------|
| `--beam-width` (alias `--n`) | `50` | Default beam width; per-request `n` overrides it. Also sets the fused-expand graph capture width — a single instance performs best serving one dominant beam width. |
| `--max-tokens` | `5` | Default max generated tokens (SID steps); per-request override. |
| `--length-penalty` | `1.0` | Beam-score length penalty. |
| `--prompt` / `--messages-json` | unset | Offline single-shot generation (mutually exclusive with `--serve`). |

## Memory & backends

| Flag | Default | Description |
|------|---------|-------------|
| `--mem-fraction-static` | `0.8` | GPU memory fraction reserved for KV cache and static buffers. |
| `--max-seq-len` | `4096` | Maximum sequence length. |
| `--cuda-graph-max-bs` | `800` | Largest CUDA-graph batch (beam rows). |
| `--cuda-graph-capture-sizes` | `50,100,…,400,800` | Explicit capture sizes; automatically extended to multiples of the beam width. |
| `--attention-backend` | `flashinfer` | Attention backend. |
| `--flashinfer-variant` | `fa2` | FlashInfer kernel variant. |
| `--gpu-id` | `0` | CUDA device index. |

## Batching & scheduling

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-slots` | follows `--cuda-graph-max-bs` | Beam-row slot budget per wave (≈ requests × beam width). |
| `--batch-wait-ms` | `4` | Base batching wait before the first expand. |
| `--batch-wait-max-ms` | `10` | Max extra wait when under-filled; upper bound on head-of-queue delay. |
| `--target-batch-requests` | `8` | Keep waiting (bounded by max wait) until this many requests are queued; also the in-flight refill threshold. |
| `--max-batch-requests` | `16` | Admit up to this many requests into one wave. |
| `--max-running-requests` | `64` | Max concurrently running requests. |
| `--decode-pack-min-requests` | `6` | Min in-flight requests to trigger decode packing. |
| `--decode-pack-ratio` | `0.75` | Decode-pack fill-ratio threshold. |
| `--schedule-policy` | `lpm` | Queue policy: `lpm` (longest prefix match, radix-friendly) or `fcfs`. |
| `--lpm-aging-ms` | `300` | Promote jobs waiting longer than this ahead of LPM ranking; `0` disables (pure LPM can starve short prompts). |
| `--host-worker-threads` | `4` | Host worker threads (tokenize / response). |
| `--pipeline-stages` | `0` | Decode pipelining stages; `0` off, `>=1` on (usually measured slower than the default eager path — benchmark first). |

### Wide-beam tuning (e.g. `n = 512` / `n = 1000`)

The defaults admit only one 512-beam request per wave (slot budget 800) and
pure LPM can starve short prompts. The same 4096 recipe covers `n = 1000`
(~4 concurrent n=1000 requests per wave):

```bash
CUDA_GRAPH_MAX_BS=4096 BATCH_SLOTS=4096 LPM_AGING_MS=150 \
  BEAM_WIDTH=512 MODEL_PATH=... SID_VOCAB_FILE=... bash scripts/serve.sh
```

`--beam-width` sets the fused-expand capture width; graph sizes automatically
extend to `k × n` (`BeamRecConfig.resolved_cuda_graph_sizes`).

## Ablation switches

Optimizations default to on; disable individually with:
`--disable-radix`, `--disable-cuda-graph`, `--disable-prefill-batch`,
`--disable-fused-expand`, `--disable-graph-expand`, `--disable-decode-pack`,
`--disable-fused-rms-fp8`, `--disable-fused-silu-fp8`,
`--disable-fused-qk-rope-kv`, `--disable-warmup`.
`--torch-compile` (prefill only; decode stays on CUDA graphs) defaults to off.

## HTTP server

| Flag | Default | Description |
|------|---------|-------------|
| `--serve` | off | Start the FastAPI server (otherwise offline generate). |
| `--host` | `127.0.0.1` | Bind address. **No authentication** — only bind a public address on a trusted network or behind a proxy. |
| `--port` | `8000` | Port. |
| `--log-level` | `info` | Log level. |

Request and response fields for `/v1/chat/completions` are in
[API](api.md).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASHREC_TORCH_PROFILER_DIR` | unset | Default profiler output directory (falls back to `SGLANG_TORCH_PROFILER_DIR`, then `/tmp`). |
| `FLASHREC_DENSE_FINALIZE` | `1` | Dense fast path for beam finalize; `0` disables. |
| `FLASHREC_TRIE_DENSE_MAX_CELLS` | `2^29` | Max cells before the SID trie switches from dense to CSR-sparse storage. |

`scripts/serve.sh` additionally reads `MODEL_PATH`, `HOST`, `PORT`,
`CUDA_VISIBLE_DEVICES`, `MEM_FRACTION_STATIC`, `QUANTIZATION`,
`KV_CACHE_DTYPE`, `TORCH_COMPILE`, `CUDA_GRAPH_MAX_BS`, `BATCH_SLOTS`,
`BATCH_WAIT_MS`, `BATCH_WAIT_MAX_MS`, `TARGET_BATCH_REQUESTS`,
`MAX_BATCH_REQUESTS`, `SCHEDULE_POLICY`, `LPM_AGING_MS`, `PIPELINE_STAGES`,
`HOST_WORKER_THREADS`, `BEAM_WIDTH`, `MAX_TOKENS`, `LOG_LEVEL`,
`ATTENTION_BACKEND`, `SID_VOCAB_FILE`, `SID`, `SID_TOKEN_RANGE`,
`SID_CODEBOOK_SIZES`, `SID_BOUNDARY_TOKENS`, `SYSTEM_PROMPT_FILE`,
`WARMUP_USER_A`, `WARMUP_USER_B`, and
`EXTRA_SERVER_ARGS` (append/override any CLI flag). `SID_VOCAB_FILE` defaults
to `data/catalogs/sid2pid_beamrec_l4.json` (repo-relative); layout is inferred.
`SID_VOCAB_FILE=` (empty) is unconstrained smoke. `SID` (`START:END/SIZE,...`)
and the three split `SID_*` vars override inference.

The tunable template is `scripts/serve.env.example`. Copy it to
`scripts/serve.env` (gitignored) and edit, or pass
`SERVE_ENV=/path/to.env bash scripts/serve.sh`. With `${VAR:-...}` in the
file, already-exported environment variables win; a bare `VAR=value`
overwrites a CLI export. The file is bash-sourced — load a trusted path only.

## Tests

```bash
python -m pytest
```

Live parity against a running SGLang beam server (optional):

```bash
SGLANG_BEAM_URL=http://127.0.0.1:PORT FLASHREC_URL=http://127.0.0.1:8000 \
  PYTHONPATH=python python -m unittest tests.test_parity -v
```

Accuracy diff against HuggingFace `transformers` (needs CUDA and a GenRec
model directory):

```bash
FLASHREC_DIFF_MODEL=/path/to/OneRec-1.7B \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff -v

FLASHREC_DIFF_MODEL=/path/to/OneRec-1.7B \
FLASHREC_DIFF_BEAMS=1,20,50,128,512 \
FLASHREC_DIFF_QUANT=bf16 \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff.TestBeamSearchDiff -v
```

| Env | Default | Meaning |
| --- | --- | --- |
| `FLASHREC_DIFF_MODEL` | (required) | Checkpoint directory |
| `FLASHREC_DIFF_QUANT` | `fp8` | `fp8` is the serving path; `bf16` is closer to HF |
| `FLASHREC_DIFF_BEAMS` | `4,8` | Beam widths |
| `FLASHREC_DIFF_CATALOG` | unset | If set, runs a trie; no longer the same protocol as HF |

This measures codebook-constrained log-probs / SID-set overlap, not OneRec
recall. OneRec-1.7B's public weights are BF16-trained; the FP8 path quantizes
on load, so the overlap drop is not a framework bug. FP8-trained SoHuRec-1.7B /
SoHuRec-0.6B recover higher overlap on the same path.
Both are on RTX 5090; see [Evaluation](baselines.md). Prefer an FP8-trained
model for production. Runnable entry: [Examples](../examples/README.md).

## Profiling & tracing

The interface mirrors SGLang's, so `sglang.bench_serving --profile` works
directly:

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"./profiles","num_steps":20,"activities":["CPU","GPU"],"with_stack":true}'
curl -s -X POST http://127.0.0.1:8000/stop_profile   # auto-stops at num_steps
```

Body fields: `output_dir`, `num_steps`, `start_step`, `activities`
(`CPU`/`GPU`), `profile_by_stage`, `with_stack`, `record_shapes`,
`profile_prefix`.

Chrome traces contain `flashrec.batch.wait` / `flashrec.prefill` /
`flashrec.decode_fwd` / `flashrec.expand` / `flashrec.finalize`
ranges. Start/stop take effect on the GPU worker thread — do not wrap
`torch.profiler` around the HTTP thread.
