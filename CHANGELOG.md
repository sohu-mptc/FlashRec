# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the [FlashRec preprint](docs/papers/paper.pdf), *FlashRec: Serving
  Generative Recommendation with Catalog-Constrained Wide-Beam Search*, with
  links and citation information in both READMEs.
- Added `--full-codebook-scoring` (FR-full), `--legal-set-scoring` (FR-legal),
  `--enable-beam-shared-attention` (BSA), and
  `--beam-shared-attention-min-rows` as aliases for the existing options.

### Changed

- Aligned the CLI and documentation with the paper's terminology. The Python
  API now uses `FlashRecConfig`, `FlashRecEngine`, and `FlashRecServer`.
  Existing `BeamRec*` imports and server launch commands remain compatible.

## [0.1.3] - 2026-11-26

Added cascade attention, CSR trie operators, deadline-aware SLO admission,
and decode/memory optimizations.

### Added

- Cascade attention with FlashInfer prefix sharing and CUDA-graph
  capture/replay (`attention/flashinfer.py`, `engine/graph.py`,
  `scheduler/scheduler.py`).
- Standard CSR trie CUDA operators (`kernel/csr_trie.py`,
  `kernel/csrc/csr_trie.cuh`) to replace dense transition tables in trie
  mode, significantly reducing resident trie index VRAM while preserving
  output semantics and throughput.
- Trie-budget grouped beam expansion, unary finish, and leaf flush
  (`search/budget.py`, `search/unary.py`, `kernel/trie_budget.py`,
  `kernel/csrc/trie_budget.cuh`).
- Deadline-aware SLO admission gate (`--admission-slo-budget-ms`,
  `--admission-unit`, `--max-queue-len`, and affine service-time estimator
  flags), `token_ids` in chat responses, and scheduler admission/path
  telemetry (`scheduler/loop.py`, `scheduler/batching.py`, `server/api.py`,
  `cli.py`, `scripts/serve.sh`).
- Configurable `qk_norm` override for Qwen2/Qwen3 checkpoints, optional
  next-norm fusion, and explicit SID boundary token support.

### Changed

- Default `enable_sparse_lm_head` to `False` (full-codebook scoring);
  legal-set renormalized scoring remains opt-in via
  `--enable-sparse-lm-head`.
- Trigger full idle GC only after sustained worker inactivity
  (`server/api.py`, `cli.py`, `config.py`).
- Size `ReqToTokenPool` from slot and batch-request budgets and keep
  evicting unlocked radix cache entries until KV allocation succeeds.

### Fixed

- Run unary and leaf finish paths under `torch.inference_mode()` during
  decode ticks to avoid in-place inference-tensor update errors.
- Prevent beam padding from emitting illegal SIDs when legal trie children
  run out, and size cascade CUDA-graph metadata to the live beam width.
- Fix an undefined `device` reference in `Qwen3ForCausalLM._wire_next_norm()`.

## [0.1.2] - 2026-09-12

Fixed within-request SID collapse under CUDA-graph decode.

### Fixed

- Score each SID codebook level separately (or narrow the restricted head
  before top-*k*) so wrong-level tokens no longer starve the candidate pool
  and collapse beams onto duplicate SIDs.
- Keep FlashInfer `seq_lens_cpu` coherent without racing D2H after prompt-len
  changes, and ping-pong pinned H2D staging so in-flight DMA cannot corrupt
  fused-expand SID columns.

### Added

- `enable_per_codebook_lm_head` (default on) and per-level `lm_head` binding
  in `RestrictedLMHead`.

## [0.1.1] - 2026-09-08

Improved decode throughput, added a catalog CLI, and warmed up serving.

### Added

- `flashrec.catalog` module and `flashrec catalog` CLI for building and
  inspecting SID catalogs from RecIF packed mappings.
- Serving warm-up so the first request does not pay graph-capture cost.
- `tests/test_packaging.py` guarding the version/tag consistency check.

### Changed

- Decode path throughput improvements in the CUDA-graph capture and beam
  expansion (`engine/graph.py`, `search/expand.py`, `search/trie.py`).
- Server API (`server/api.py`) and scheduler (`scheduler/scheduler.py`)
  refinements for wave scheduling and decode packing.
- `scripts/convert_recif_catalog.py` slimmed down; catalog logic moved
  into `flashrec.catalog`.
- Banner and architecture assets refreshed.

## [0.1.0] - 2026-09-02

First public release: an inference engine for generative recommendation,
wide beam search over a semantic-ID catalog, executed inside CUDA graphs.

### Added

- Trie-constrained wide-beam decoding (dense and CSR-sparse SID catalogs)
  with a restricted `lm_head` over the SID token range.
- CUDA-graph capture of the decode path, including fused beam-trie expansion.
- End-to-end FP8 (W8A8 per-channel weights, `fp8_e4m3` KV cache) with fused
  RMSNorm→FP8, SiLU→FP8, and QK-RoPE+KV-write kernels.
- Wave scheduling with radix prefix KV cache, LPM + aging admission, and
  decode packing.
- Deterministic (`temperature = 0`) and Gumbel top-*k* (`temperature > 0`)
  beam search.
- OpenAI-compatible `/v1/chat/completions` and a `torch.profiler` HTTP
  interface compatible with `sglang.bench_serving --profile`.
- Qwen3 dense model path (GQA, `head_dim`, qk-norm, tied embeddings read
  from `config.json`), including OneRec-1.7B.
- SID catalog builder from OpenOneRec RecIF packed mappings
  (`scripts/build_catalog.sh`).
- Open-vocabulary beam baseline docs and RecIF compare runners (SGLang,
  vLLM, HuggingFace, TensorRT-LLM) in `docs/baselines.md`.
- Runnable `examples/`, plus architecture, API, configuration, and FAQ docs.
- CPU unit tests, pre-commit hooks, GitHub Actions CI, and a PyPI release
  workflow on `v*` tags.

[Unreleased]: https://github.com/sohu-mptc/FlashRec/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.3
[0.1.2]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.2
[0.1.1]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.1
[0.1.0]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.0
