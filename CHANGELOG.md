# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/sohu-mptc/FlashRec/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.2
[0.1.1]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.1
[0.1.0]: https://github.com/sohu-mptc/FlashRec/releases/tag/v0.1.0
