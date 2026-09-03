# Architecture

FlashRec serves an HTTP request in the same process that owns the weights and
the KV pool. The request is admitted into a decode wave and executed there.
Scheduling and tokenization run as threads inside that process, sharing one
address space.

The module layout follows
[mini-sglang](https://github.com/sgl-project/mini-sglang). At runtime the engine
depends on `sgl-kernel`, `flashinfer_python`, and `triton`.

```
python/flashrec/
├── sid_layout.py  infer SID range / codebooks / boundary from tokenizer + catalog
├── engine/      model forward, CUDA-graph capture & replay
├── scheduler/   wave batching, admission, decode loop, pipelining
├── search/      beam expansion, scoring, SID trie (dense / CSR)
├── kernel/      JIT-compiled fused beam-trie CUDA kernel
├── attention/   FlashInfer backend & KV indexing
├── kvcache/     paged KV pool, radix prefix cache
├── layers/, models/   Qwen3-family model, FP8 linear / norm / rotary
└── server/      FastAPI OpenAI-compatible endpoint
```

## Workload

GenRec retrieval generates **semantic IDs**: short, fixed-depth token sequences
that index an item catalog. The shape differs from chat serving.

| | Chat / general LLM | GenRec beam retrieval |
| --- | --- | --- |
| Decode depth | hundreds to thousands of steps | 3–5 steps |
| Width per request | 1 sequence | 50–512+ beams |
| Vocabulary per step | full, unconstrained | only valid-SID continuations |
| Dominant cost | attention over long context | beam bookkeeping and expansion |

A general-purpose engine amortizes per-step overhead over a long decode. Over
five SID steps that overhead is not hidden, so wide-beam throughput on general
engines is roughly flat in concurrency: one wide-beam request occupies the
engine.

## Request path

1. **Admission.** Requests wait until the beam-row slot budget can hold them.
   The queue is longest-prefix-match by default, with aging so a request behind
   better-matching neighbours keeps gaining priority until it is admitted.
2. **Tokenize.** Host worker threads, off the GPU critical path.
3. **Radix prefix match.** Shared prompt prefixes reuse cached KV; the hit shows
   up as `usage.prompt_tokens_details.cached_tokens`.
4. **Batched prefill.** All requests admitted into the wave prefill together.
5. **Trie-constrained expand.** At engine start the SID layout is inferred from
   the tokenizer (`<s_a_0>` codebooks, `<|sid_begin|>` / `<|sid_end|>`) plus
   `--sid-vocab-file`. A fused CUDA kernel then scores continuations against
   the SID trie, so illegal SIDs are never candidates and `invalid_rate` is
   0. `lm_head` is evaluated only over that inferred token range.
6. **Decode.** One CUDA-graph replay per step, covering the full beam width
   including the expansion step. Steps 5–6 repeat until the SID is complete.
7. **Finalize.** Length penalty, ranking, and the OpenAI-shaped response.

## Design

**Beam expansion lives inside the CUDA graph.** Graph capture sizes extend to
multiples of the configured beam width, so a wide beam replays as a single graph.

**The trie is the decoding constraint.** Dense for small catalogs, CSR sparse for
large ones (a public 8192³ catalog would need ~25G dense cells). Constraining
during expansion keeps `invalid_rate = 0` while open-vocabulary baselines leave
roughly 17–27% of beams illegal.

**FP8 reduces footprint.** W8A8 per-channel weights and `fp8_e4m3` KV by default,
with fused RMSNorm→FP8, SiLU→FP8, and QK-RoPE+KV-write kernels. On 1.7B-class
checkpoints at `n ≤ 128` it runs ~0.6–0.9× the throughput of BF16 and returns
weight and KV-cache room; the two precisions converge at `n = 512`, where the
step is bound by bookkeeping.

**Scheduling is throughput-oriented.** Requests are admitted between decode steps
under a slot budget, which lets beam rows from different requests share a step.

## Scope

Training and experimentation remain outside this engine. FlashRec covers the
serving hop: beam search over a semantic-ID catalog, behind an OpenAI-compatible
endpoint.

Model size is bound by device memory — roughly up to 14B in FP8 on 32 GB. The hot
path it optimizes is trie-constrained wide beam over a few SID steps.

## Further reading

- [Configuration](configuration.md) — flags and environment variables
- [API](api.md) — HTTP and Python interfaces
- [Evaluation](baselines.md) — OneRec-1.7B / SoHuRec-1.7B / SoHuRec-0.6B comparisons
- [FAQ](faq.md) — common questions
