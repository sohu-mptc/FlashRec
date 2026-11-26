# Architecture

FlashRec serves generative recommendation with catalog-constrained wide-beam
search. Its HTTP server, model and KV pool share a process. Host threads handle
scheduling and tokenization while search state stays on the GPU.

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
| Decode depth | hundreds to thousands of steps | typically 3–4 SID tokens |
| Width per request | 1 sequence | up to 1,000 beams in the paper |
| Valid continuations | full vocabulary | legal children in the catalog trie |
| Dominant cost | attention over long context | beam bookkeeping and expansion |

A general-purpose engine amortizes per-step overhead over a long decode. Over
five SID steps that overhead is not hidden, so wide-beam throughput on general
engines is roughly flat in concurrency: one wide-beam request occupies the
engine.

## Request path

1. **Beam-row admission.** By default, each request charges its requested beam
   width against the row budget. The queue uses longest-prefix match with aging.
2. **Tokenize.** Host worker threads, off the GPU critical path.
3. **Radix prefix match.** Shared prompt prefixes reuse cached KV; the hit shows
   up as `usage.prompt_tokens_details.cached_tokens`.
4. **Shared prefill.** Batched prefill computes each prompt once and shares its
   prompt KV page references across that request's beams.
5. **GPU-resident expansion.** At startup, FlashRec infers the SID layout from
   the tokenizer and `--sid-vocab-file`. The search masks illegal trie edges and
   selects the surviving parent-token pairs for each request. Eligible shapes
   fuse selection and state advancement in CUDA.
6. **Zero-copy state and KV inheritance.** One parent map advances SID histories
   and trie nodes and remaps the generated-suffix KV references without copying
   KV tensors. The next forward pass writes newly selected tokens to fresh KV
   slots. Eligible shapes replay CUDA graphs over persistent device buffers.
7. **Finalize.** FlashRec applies the length penalty, ranks the complete SIDs,
   releases the request's row charge and decode pages, and returns the response.

## Design

**CUDA graph replay.** Eligible model, SID head and expansion paths replay a
per-level graph. Other shapes can use a base graph with separate expansion or
run eagerly.

**Catalog trie and scoring.** Small catalogs use dense transition tables and
larger catalogs use compressed sparse row (CSR) adjacency. Full-codebook
scoring (FR-full) normalizes over the current SID level's codebook before trie
masking. Legal-set scoring (FR-legal) normalizes over each prefix's legal
children. With deterministic trie decoding, unary finish can append the forced
suffix without model calls once all live beams have unique remaining paths.

**Beam-shared attention (BSA).** The optional FlashInfer path reuses each
request's prompt KV reads across beams and computes attention over each beam's
generated suffix. The per-level graph takes precedence when eligible. Use
`--disable-level-graph` to exercise BSA in batches that would otherwise use
that graph. [Configuration](configuration.md#memory--backends)

**FP8 reduces footprint.** W8A8 per-channel weights and `fp8_e4m3` KV by default,
with fused RMSNorm→FP8, SiLU→FP8, and QK-RoPE+KV-write kernels. On 1.7B-class
checkpoints at `n ≤ 128` it runs ~0.6–0.9× the throughput of BF16 and returns
weight and KV-cache room; the two precisions converge at `n = 512`, where the
step is bound by bookkeeping.

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
