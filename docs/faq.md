# FAQ

## Scope and fit

**What does it serve?** Beam search over a constrained vocabulary: a prompt in,
ranked semantic IDs out, in one response.

**What does it need installed?** `pip install` pulls the runtime —
`sgl-kernel`, `flashinfer_python`, and `triton` — on Linux with a CUDA 12
toolchain. The layout follows
[mini-sglang](https://github.com/sgl-project/mini-sglang) and the fused kernel
headers are vendored. SGLang itself is needed to reproduce the *baseline* side of
the comparisons.

**Which models work?** Qwen3 dense (0.6B through 14B), OneRec-1.7B,
SoHuRec-1.7B / SoHuRec-0.6B, and Qwen3-based GenRec checkpoints. Architectural parameters are read from
`config.json`; validated at the 1.7B scale. Broader coverage, including MoE, is
on the roadmap.

**How large a model fits?** Model size is bound by device memory — roughly up to
14B in FP8 on 32 GB. Tensor and expert parallelism for larger models are on the
roadmap.

## Getting valid output

**How do I get valid semantic IDs?** Serving uses `--model-path` plus
`--sid-vocab-file` (build the catalog with `scripts/build_catalog.sh`). The
token range, codebook sizes, and boundary tokens are inferred from the
checkpoint tokenizer. Leave the catalog unset only for a connectivity check over
the full vocabulary. Pass `--sid START:END/SIZE,...` (or the three split flags)
only to override that inference.

**Output SIDs are too long, or carry trailing junk.** Check that the inferred
(or explicit `--sid`) layout matches the checkpoint, and that `--max-tokens`
matches the SID depth (depth-3 SIDs plus boundary tokens want `--max-tokens 5`).

**I asked for `n` candidates and got fewer.** The trie ran out of valid
continuations for that prompt, so the response carries every legal beam it found.
Catalog size sets that ceiling.

**Are results reproducible?** Yes at `temperature = 0`, which is deterministic
top-*k*. With `temperature > 0` the engine samples via Gumbel top-*k* without
replacement; the reported `sequence_score` stays noise-free either way.

## Performance

**Throughput is lower than the published numbers.** First check which workload
you measured. OneRec-1.7B prompts are ~2.5k tokens (~500 history SIDs);
SoHuRec-1.7B serving prompts are ~300 tokens. Same engine, `n=50`, concurrency 8
is ~26 QPS on OneRec-1.7B and ~160 QPS on SoHuRec-1.7B; saturation (concurrency
32) is ~28 QPS and ~220 QPS. Do not divide OneRec QPS by SoHuRec QPS, and do not
collapse SGLang-master and SGLang 0801 into one baseline — see
[Evaluation](baselines.md). The most common cause at `n ≥ 512` (including
`n = 1000`) is a graph and slot budget left at the defaults. Wide beam needs
`--cuda-graph-max-bs 4096 --batch-slots 4096`, and it helps to raise
`--lpm-aging-ms` to ~150. See [Configuration](configuration.md).

**FP8 beam overlap vs HuggingFace is low on OneRec.** OneRec-1.7B is
BF16-trained; `--quantization fp8` quantizes on load. That post-training
quantization — not the engine — drives the drop: the same engine in BF16 matches
HuggingFace's best SID at every width. FP8-trained SoHuRec-1.7B / SoHuRec-0.6B
recover **88–96%** / **88–93%** beam-set overlap on the same FP8 decode path.
Prefer an FP8-trained checkpoint when serving FP8; see
[Evaluation](baselines.md).

**FP8 runs slower than BF16 here.** Expected on 1.7B-class checkpoints at
`n ≤ 128`, where FP8 lands at ~0.6–0.9× the throughput of BF16 and returns weight
and KV-cache footprint. The two precisions converge at `n = 512`, where the step
is bound by beam bookkeeping.

**At `n = 50` and concurrency 1 the gap is smaller.** Expected. FlashRec is
still ahead (~1.2–1.3× on OneRec, ~1.8–2.1× on SoHuRec-1.7B versus
SGLang-master and SGLang 0801), but per-request overhead matters more at that width. The gap
widens as beam and concurrency grow; that is the design target.

**Is the prefix cache working?** Read
`usage.prompt_tokens_details.cached_tokens` in the response. A GenRec workload
with a shared system prompt should show a large hit.

**How do I profile a step?** `/start_profile` and `/stop_profile` are compatible
with `sglang.bench_serving --profile`; the output directory comes from
`FLASHREC_TORCH_PROFILER_DIR`. See
[Configuration](configuration.md).

## Failure modes

| Symptom | What to do |
| --- | --- |
| OOM at startup | Lower `--mem-fraction-static` (e.g. `0.70`), or `--max-seq-len` |
| OOM only at wide beam | Lower `--batch-slots` / `--cuda-graph-max-bs`, or the beam width |
| Long startup on first run | The fused beam-trie kernel is JIT-compiled once, then cached |
| Trie build is slow or memory-hungry | Large catalogs switch to CSR-sparse automatically; tune the threshold with `FLASHREC_TRIE_DENSE_MAX_CELLS` |
| Sequence longer than the checkpoint's native range | Sequence length is capped by `--max-seq-len` and stays inside the model's native positional range |
| GPU shows full load with no process | `nvidia-smi --gpu-reset -i <id>`, after releasing any monitoring process holding `/dev/nvidia*` |

## Reproducing the comparisons

Launch commands for SGLang-master, vLLM, TensorRT-LLM, and HF Transformers, plus the
full result tables and the baseline-side pitfalls (vLLM needs
`--max-logprobs ≥ 2n`; SGLang returns ranked beams in `meta_info.beam_results`)
are in [Evaluation](baselines.md).
