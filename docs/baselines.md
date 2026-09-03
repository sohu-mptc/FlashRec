# Evaluation

[简体中文](baselines.zh-CN.md)

Hardware: **NVIDIA RTX 5090 (32 GB)**. Default sample size **5,000**,
`max_tokens=5`, `temperature=0`. FlashRec: FP8 + SID trie. Baselines:
open-vocabulary.

**Do not divide OneRec QPS by SoHuRec QPS.** RecIF prompt p50 ≈ **2,465**
tokens; production SoHuRec prompts p50 ≈ **303** tokens (~8×). Catalog and
graph paths also differ.

| Model / workload | Description |
| --- | --- |
| **OneRec** | [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) RecIF-Bench video, model [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B) |
| **SoHuRec-1.7B / SoHuRec-0.6B** | Sohu internal generative-recommendation serving traffic on the corresponding models |

Engines are reported **separately**. Do not collapse SGLang-master and
SGLang 0801 into one row.

| Name | Implementation | Constraint |
| --- | --- | --- |
| **FlashRec** | this repo | FP8 + SID trie, `invalid_rate = 0` |
| **SGLang-master** | SGLang mainline ([PR #31626](https://github.com/sgl-project/sglang/pull/31626) wide-beam) | open-vocabulary |
| **SGLang 0801** | [`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801) | open-vocabulary (SoHuRec-0.6B still uses a SID `lm_head` restriction) |
| **vLLM / TensorRT-LLM** | OneRec only | open-vocabulary |

Open-vocabulary runs must report `invalid_rate` (~17–27% illegal SIDs; those
beams cannot be served). A cell is filled only if it completed the full
sample; empty / `—` means it did not finish or failed (see failure modes), not
that it was skipped.

## Takeaways

1. **Public RecIF, same long prompts.** FlashRec (trie) is fastest in every
   completed cell. At `n=50` it saturates around **28 QPS**, SGLang 0801 **18**,
   SGLang-master **14**, TensorRT-LLM **13**, vLLM **3.8**. At `n=1000`
   FlashRec saturates at **4.62 QPS**; SGLang-master and SGLang 0801 only hold concurrency 1
   (2.5–3.0).
2. **Quality.** Open-vocabulary engines have essentially the same recall@32
   (~0.034 at `n=50`). The gap is **whether SID constraint is on**, not the
   engine. Open-vocab `invalid ≈ 18–27%`; FlashRec `invalid = 0`. **Do not cite
   FlashRec concurrency-1 recall@32** (unique-beam collapse inflates it).
3. **Production short prompts (SoHuRec ~300 tokens).** FlashRec is about
   **2.2–3.4×** versus SGLang-master and SGLang 0801; the gap grows as beam widens. At `n=50`,
   1.7B FlashRec saturates at **220 QPS**, 0.6B at **303 QPS**. At `n=512`,
   1.7B FlashRec is **~48 QPS** vs 0801 / SGLang-master **~16 QPS**.
4. **Wide beam + high concurrency** is the shared weak spot for SGLang-master,
   SGLang 0801, TensorRT-LLM, and vLLM (OOM, slots, disconnects). After
   per-cell restarts and `max-running=(n+1)×conc`, 0801 completes 50 / 128 / 512
   on production 1.7B.

## SoHuRec-1.7B / SoHuRec-0.6B

FP8, production prompts (~300 tokens), **5,000** samples. SGLang-master is
[PR #31626](https://github.com/sgl-project/sglang/pull/31626); SGLang 0801 is
[`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801).

On 1.7B, 0801 **dropped** `--beam-search-lm-head-special-token-ids` (internal
`topk(k=2n)`; at `n=1000`, `2n=2000` exceeds the ~1,535 SID tokens). 0.6B has
enough SID tokens and kept the restriction. The 1.7B 0801 cells are therefore
open-vocabulary beam, speed-only.

Saturated throughput (highest completed concurrency at that `n`):

| n | FlashRec / SGLang 0801 | FlashRec / SGLang-master |
| ---: | ---: | ---: |
| 50 | **2.26×** | **2.71×** |
| 128 | **2.52×** | **2.46×** |
| 512 | **2.95×** | **2.87×** |
| 1000 | **2.16×** (conc=1 only) | **2.14×** (conc=1 only) |

At `n=1000`, SGLang-master and SGLang 0801 die at conc ≥ 8, so peak throughput cannot be
compared.

<div align="center">
  <img src="figures/perf-serving-throughput-en.svg" alt="SoHuRec-1.7B throughput vs SGLang 0801 and SGLang-master (concurrency 32)" width="620"/>
</div>

### SoHuRec-1.7B QPS

| n | Engine | conc=1 | conc=8 | conc=16 | conc=32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | **FlashRec** | **41.11** | **159.72** | **168.68** | **219.54** |
| 50 | SGLang 0801 | 22.84 | 68.56 | 85.94 | 97.20 |
| 50 | SGLang-master | 19.75 | 56.05 | 74.16 | 81.12 |
| 128 | **FlashRec** | **25.09** | **101.99** | **114.61** | **136.78** |
| 128 | SGLang 0801 | 20.41 | 48.37 | 52.15 | 54.22 |
| 128 | SGLang-master | 18.48 | 44.66 | 53.61 | 55.65 |
| 512 | **FlashRec** | **19.48** | **41.47** | **47.76** | **47.16** |
| 512 | SGLang 0801 | 11.45 | 15.98 | 16.20 | 16.03 |
| 512 | SGLang-master | 11.13 | 16.56 | 16.47 | 16.63 |
| 1000 | **FlashRec** | **15.54** | **24.64** | 24.53 | 24.64 |
| 1000 | SGLang 0801 | 7.19 | — | — | — |
| 1000 | SGLang-master | 7.26 | — | — | — |

### SoHuRec-0.6B QPS

| n | Engine | conc=1 | conc=8 | conc=16 | conc=32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | **FlashRec** | **47.43** | **184.44** | **251.17** | **302.60** |
| 50 | SGLang 0801 | 28.17 | 75.79 | 99.43 | 116.66 |
| 50 | SGLang-master | 23.51 | 60.53 | 87.17 | 104.76 |
| 128 | **FlashRec** | **28.31** | **101.81** | **179.10** | **185.86** |
| 128 | SGLang 0801 | 24.12 | 57.14 | 61.95 | 65.44 |
| 128 | SGLang-master | 21.05 | 51.22 | 59.76 | 72.42 |
| 512 | **FlashRec** | **21.77** | **52.99** | **59.75** | **60.16** |
| 512 | SGLang 0801 | 13.30 | 19.53 | 19.19 | — |
| 512 | SGLang-master | 12.55 | — | — | — |
| 1000 | **FlashRec** | **18.01** | **32.19** | 32.02 | 32.11 |
| 1000 | SGLang 0801 | 8.51 | — | — | — |
| 1000 | SGLang-master | 8.34 | — | — | — |

Peak 0.6B FlashRec is **303 QPS** (`n=50`, conc=32), about **1.4×** the 1.7B
peak. At `n=1000` FlashRec saturates at **32 QPS**, about **1.3×** the 1.7B
figure.

## OneRec

[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) RecIF-Bench video,
model [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B). 5,000
samples (`video_test.parquet`, random, seed=42). FlashRec catalog:
`sid2pid_beamrec_l4.json` (~1.52M SIDs, depth 4, codebook 8,192). Baselines
have no trie.

QPS is filled only for cells that finished all 5,000 samples.

<div align="center">
  <img src="figures/perf-onerec-qps-en.svg" alt="OneRec-1.7B throughput vs SGLang 0801, SGLang-master, TensorRT-LLM, and vLLM (n=50 saturation)" width="620"/>
</div>

### Throughput (QPS)

| n | Engine | conc=1 | conc=8 | conc=16 | conc=32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | **FlashRec** | **15.09** | **26.47** | **27.68** | **28.08** |
| 50 | SGLang 0801 | 12.36 | 17.14 | 17.95 | 18.20 |
| 50 | SGLang-master | 11.43 | 13.84 | 13.88 | 13.85 |
| 50 | TensorRT-LLM | 11.64 | 12.71 | 12.70 | 12.69 |
| 50 | vLLM | 2.47 | 3.85 | 3.89 | 3.78 |
| 128 | **FlashRec** | **12.17** | **19.18** | **19.41** | **19.91** |
| 128 | SGLang 0801 | 9.97 | 12.83 | 13.19 | 13.20 |
| 128 | SGLang-master | 9.09 | 9.80 | 9.79 | 9.79 |
| 128 | vLLM | 0.99 | 1.57 | 1.57 | 1.56 |
| 128 | TensorRT-LLM | 7 samples | 7 samples | 7 samples | 7 samples |
| 512 | **FlashRec** | **7.10** | **8.09** | **8.18** | **8.17** |
| 512 | SGLang 0801 | 5.01 | 5.61 | OOM | OOM |
| 512 | SGLang-master | 4.20 | 4.42 | failed | failed |
| 512 | vLLM | 0.16 | interrupted | — | — |
| 512 | TensorRT-LLM | unsupported on 32 GB | same | same | same |
| 1000 | **FlashRec** | **4.38** | **4.62** | **4.62** | **4.62** |
| 1000 | SGLang 0801 | 3.03 | died | died | died |
| 1000 | SGLang-master | 2.51 | died | — | — |

Saturated throughput vs FlashRec (highest completed concurrency at that `n`):

| n | FlashRec | SGLang 0801 | SGLang-master | TensorRT-LLM | vLLM | FlashRec / 0801 | FlashRec / SGLang-master |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 28.08 | 18.20 | 13.88 | 12.71 | 3.89 | **1.54×** | **2.02×** |
| 128 | 19.91 | 13.20 | 9.80 | — | 1.57 | **1.51×** | **2.03×** |
| 512 | 8.18 | 5.61 | 4.42 | — | 0.16 (conc=1 only) | **1.46×** | **1.85×** |
| 1000 | 4.62 | 3.03 (conc=1 only) | 2.51 (conc=1 only) | — | — | **1.52×** (vs conc=1) | **1.84×** (vs conc=1) |

### Quality (cite conc ≥ 8 only)

FlashRec concurrency-1 `recall@32 ≈ 0.12` is unique-beam collapse — **do not
cite it**. Open-vocabulary illegal SIDs ≈ **17–27%**; FlashRec trie is **0**.

| n | Engine | conc | recall@32 | hit@32 | invalid |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | FlashRec | 8 | 0.034 | 0.180 | 0 |
| 50 | SGLang 0801 | 8 | 0.034 | 0.193 | 0.270 |
| 50 | SGLang-master | 8 | 0.034 | 0.195 | 0.260 |
| 50 | TensorRT-LLM | 8 | 0.034 | 0.195 | 0.259 |
| 50 | vLLM | 8 | 0.034 | 0.194 | 0.260 |
| 128 | FlashRec | 8 | 0.054 | 0.219 | 0 |
| 128 | SGLang 0801 | 8 | 0.038 | 0.216 | 0.236 |
| 128 | SGLang-master | 8 | 0.038 | 0.215 | 0.227 |
| 128 | vLLM | 8 | 0.038 | 0.214 | 0.227 |
| 512 | FlashRec | 8 | 0.051 | 0.250 | 0 |
| 512 | SGLang 0801 | 8 | 0.040 | 0.224 | 0.194 |
| 512 | SGLang-master | 8 | 0.040 | 0.223 | 0.185 |
| 512 | vLLM | 1 | 0.040 | 0.223 | 0.185 |
| 1000 | FlashRec | 8 | **0.074** | 0.218 | 0 |
| 1000 | SGLang 0801 | 1 | 0.040 | 0.223 | 0.179 |
| 1000 | SGLang-master | 1 | 0.040 | 0.224 | 0.170 |

Open-vocabulary engines match on recall@32. FlashRec searches the legal set;
hit@32 rises with `n` from 0.18 to 0.25. Open-vocab hit looks similar, but
about one in four beams is illegal and cannot be served. FlashRec recall@32 at
`n=1000` is **0.074** vs **0.040** open-vocab.

### RecIF failure modes

- **SGLang 0801:** OOM at `n=512` conc=16/32 (`set a smaller --max-running-requests or --beam-width`); dies at `n=1000` conc ≥ 8. After per-cell restarts and `max_running=(n+1)×conc`, `n=50/128` are complete and `n=512` only reaches conc=8.
- **SGLang-master:** `n=50/128` complete; `n=512` conc=1/8 is 4.20 / **4.42**, conc=16/32 failed; `n=1000` conc ≥ 8 dies.
- **TensorRT-LLM:** stops after 7 samples at `n=128`; unsupported at `n=512` on 32 GB.
- **vLLM:** completes but is slow (`n=512` conc=1 = 0.16 QPS); `n=512` conc ≥ 8 failed.

## OneRec-1.7B and SoHuRec-1.7B

Same engine (FlashRec FP8 + SID trie, `n=50`, 5,000 requests each).

| | SoHuRec-1.7B | OneRec-1.7B |
| --- | ---: | ---: |
| prompt tokens p50 (p10–p90) | **303** (205–389) | **2,465** (2,220–2,563) |
| prompt shape | short serving requests | long SID history (~500 SIDs) |
| SID depth | 3 | 4 |
| codebook size | 512 | 8,192 |
| catalog | ~0.43M SIDs | ~1.52M SIDs |
| trie | dense | CSR |

Prompt length differs by about 8×; absolute QPS should not be compared directly.

| conc | OneRec-1.7B QPS | SoHuRec-1.7B QPS |
| ---: | ---: | ---: |
| 1 | 15.09 | 41.11 |
| 8 | 26.47 | 159.72 |
| 16 | 27.68 | 168.68 |
| 32 | 28.08 | 219.54 |

OneRec-1.7B saturates at 26–28 QPS from concurrency ≥ 8; SoHuRec-1.7B is 160–220 QPS.

## NVIDIA SID-GR (reference; hardware is not comparable)

Against [NVIDIA SID-GR inference](https://github.com/NVIDIA/recsys-examples/tree/main/examples/sid-gr-inference).
NVIDIA's published numbers are **H100 / Qwen3-1.7B / BF16**; this run is
**RTX 5090 / FlashRec FP8**. Do not treat the ratio as a speed-up. Same-GPU,
same-precision comparison was not run.

Offline synthetic context (beam=256, decode=3), OneRec-1.7B FlashRec:

| ctx | batch | FlashRec ms | FlashRec QPS | NVIDIA SID-GR ms (H100) | NVIDIA SGLang ms (H100) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 39.6 | 25.2 | 17.6 | 33.5 |
| 1000 | 8 | 211.5 | 37.8 | 93.2 | 199.3 |
| 5000 | 1 | 117.0 | 8.55 | 42.3 | 94.6 |
| 5000 | 8 | 844.1 | 9.48 | 307.9 | 685.4 |

Online HTTP (ctx=5000, beam=256, conc=4, 64 requests): OneRec-1.7B FlashRec
**9.02 req/s** (median 436 ms). NVIDIA SID-GR about **19.7 req/s** (median
~199 ms).

Real prompts, offline (not aligned to NVIDIA ctx/beam; beam=50, decode=5, SID
trie). Faster internal short prompts are expected:

| batch | RecIF OneRec QPS | internal SoHuRec-1.7B QPS | internal SoHuRec-0.6B QPS |
| ---: | ---: | ---: | ---: |
| 1 | 17.3 | 42.5 | 51.6 |
| 8 | 28.5 | 179.4 | 282.3 |

## Numerical match vs HuggingFace

Codebook-constrained beam, no SID trie. HuggingFace is a BF16 reference.
Commands: [Examples](../examples/README.md).

**OneRec-1.7B**

| FlashRec | max \|Δlogprob\| | mean \|Δ\| | top-1 | top-10 | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.196 | 0.042 | 100% | 90% | 0.993 |
| FP8 | 0.571 | 0.088 | top-2 swap | 90% | 0.992 |

| n | BF16 overlap | BF16 top-1 | FP8 overlap | FP8 top-1 |
| ---: | ---: | --- | ---: | ---: |
| 1 | 100% | yes | 0% | no |
| 20 | 90% | yes | 80% | no |
| 50 | 86% | yes | 70% | no |
| 128 | 89% | yes | 72% | no |
| 512 | 92% | yes | 76% | no |

OneRec-1.7B's public checkpoint is **BF16-trained**; the FP8 path quantizes on
load (W8A8). The FP8 drop vs HuggingFace in the table above (`n = 1` top-1 swap,
`n ≥ 50` overlap only 70–76%) is that quantization error, **not an engine bug**
— the same engine's BF16 path matches HuggingFace's best SID at every width.

**SoHuRec-1.7B / SoHuRec-0.6B** (FP8-trained; FP8 decoder GEMM on both sides)

| Model | n=4 | n=8 | n=50 | n=128 | n=512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SoHuRec-1.7B FP8 | 88% | 94% | 92% | 95% | 96% |
| SoHuRec-0.6B FP8 | 88% | 94% | 90% | 91% | 93% |

Prefill top-1: 100% (SoHuRec-1.7B), 94% (SoHuRec-0.6B). On the same FP8 decode path, FP8-trained
checkpoints recover beam-set overlap to **88–96%** / **88–93%**. FP8 serving
accuracy is dominated by whether the model was trained in FP8; prefer an
FP8-trained (or `weight_scale` pre-quantized) checkpoint in production rather
than post-training quantization of BF16 weights.

## Failed cells

- RecIF SGLang 0801 `n=512` conc=16/32; RecIF SGLang 0801 / SGLang-master `n=1000` conc ≥ 8
- RecIF SGLang-master `n=512` conc=16/32
- RecIF vLLM `n=512` conc=8/16/32; TensorRT-LLM `n ≥ 128`
- NVIDIA sid-gr-inference **same-GPU, same-precision** comparison was not run (paper H100 numbers only)

## Reproduction

```bash
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data bash scripts/build_catalog.sh

MODEL_PATH=/path/to/OneRec-1.7B \
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
bash scripts/bench_sglang_compare.sh
```

| Engine | Beam entry | Version |
|---|---|---|
| SGLang-master | `sampling_params.beam_width` ([PR #31626](https://github.com/sgl-project/sglang/pull/31626)) | `lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1` |
| SGLang 0801 | same `/generate` + `beam_width` | [`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801) |
| vLLM | `use_beam_search=true` + `n` | `vllm/vllm-openai:v0.28.0` (`--max-logprobs ≥ 2n`) |
| TensorRT-LLM | `--max_beam_width` == request `best_of` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24` |
| HuggingFace | `num_beams` / `num_return_sequences` | local `transformers` |

Single cell: `python scripts/eval_beam_matrix.py --engine <name> --server-url <url> …`.
SGLang-master / SGLang 0801 ranked beams are in `meta_info.beam_results[]`.
