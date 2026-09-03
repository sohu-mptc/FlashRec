# Examples

Two runnable entry points, both against a real downloadable checkpoint.

| File | What it does | Setup |
| --- | --- | --- |
| [offline_single_request.py](offline_single_request.py) | In-process beam search on one prompt | Runs standalone |
| [http_client.py](http_client.py) | Queries `/v1/chat/completions`, prints ranked beams + scores | Needs a running server |

## 0. Get a checkpoint

Everything below uses **OneRec-1.7B**, the Kuaishou OneRec team's public
GenRec checkpoint (Qwen3-1.7B backbone):

```bash
pip install -U "huggingface_hub[cli]"
hf download OpenOneRec/OneRec-1.7B --local-dir ./OneRec-1.7B
export MODEL_PATH=$PWD/OneRec-1.7B
```

Model card: [`OpenOneRec/OneRec-1.7B`](https://huggingface.co/OpenOneRec/OneRec-1.7B).
Any Qwen3-architecture GenRec checkpoint works the same way.

## 1. Offline, in-process

```bash
pip install -e .
python examples/offline_single_request.py --model-path "$MODEL_PATH" --n 8
```

That decodes over the **full vocabulary**, which is enough to confirm weights
load and CUDA graphs capture. Step 2 adds the catalog that makes the output
valid semantic IDs.

## 2. Constrain decoding to a catalog

Valid-SID output needs the checkpoint's SID vocabulary plus a catalog trie. The
catalog is built from OneRec RecIF-Bench
([OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec), `benchmark_data`,
which ships `sid2pid.json`):

```bash
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data bash scripts/build_catalog.sh
# writes data/catalogs/sid2pid_beamrec_l4.json
```

```bash
python examples/offline_single_request.py \
  --model-path "$MODEL_PATH" --n 32 \
  --sid-vocab-file data/catalogs/sid2pid_beamrec_l4.json
```

The catalog plus the checkpoint tokenizer are enough: codebook sizes, token
range, and `<|sid_begin|>` / `<|sid_end|>` are inferred. Pass `--sid` to override
that inference, e.g. for a tokenizer with a different codebook-token naming — see
[Configuration](../docs/configuration.md).

## 3. Serve and query over HTTP

```bash
flashrec --serve --model-path "$MODEL_PATH" --port 8000 \
  --beam-width 32 --max-tokens 5 \
  --sid-vocab-file data/catalogs/sid2pid_beamrec_l4.json
```

Then, in another shell:

```bash
python examples/http_client.py --n 32 --top 10
```

The server binds `127.0.0.1` and has **no authentication**. Bind a public
address only on a trusted network or behind an authenticating proxy.

## 4. Numerical accuracy vs HuggingFace

This is **not** OneRec retrieval recall. It checks that codebook-constrained beam search
matches `transformers` on the same OneRec-1.7B checkpoint. SID layout inferred
from the tokenizer is **3-level codebook only**:
`151669:176244/8192,8192,8192` (24,576 tokens, no catalog). OneRec serving with
`sid2pid_beamrec_l4.json` infers `151669:176246/8192,8192,8192,2` (24,578; last
codebook is the two boundary tokens). HuggingFace is a BF16 reference;
FlashRec is measured in BF16 and in FP8
(W8A8 + FP8 KV). **Device: NVIDIA RTX 5090.** Full protocol:
[Evaluation](../docs/baselines.md).

```bash
# Prefill log-probs + small-n beam overlap (default n=4,8)
FLASHREC_DIFF_MODEL="$MODEL_PATH" \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff -v

# Beam-set overlap at the serving widths
FLASHREC_DIFF_MODEL="$MODEL_PATH" \
FLASHREC_DIFF_BEAMS=1,20,50,128,512 \
FLASHREC_DIFF_QUANT=bf16 \
  PYTHONPATH=python python -m unittest tests.test_beam_search_diff.TestBeamSearchDiff -v
```

`FLASHREC_DIFF_QUANT=fp8` (default) is the serving path.

**Prefill** (last-token log-probs over 24,576 SID tokens):

| FlashRec | max \|Δlogprob\| | mean \|Δ\| | top-1 | top-10 | rank corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.196 | 0.042 | 100% | 90% | 0.993 |
| FP8 | 0.571 | 0.088 | top-2 swap | 90% | 0.992 |

BF16 hidden-state cosine vs HF is 0.99995; the 0.20 log-prob gap is FlashInfer vs
HF attention, not a weight-load bug. HF's top-1 / top-2 are 0.125 nats apart, so
FP8 can swap them.

**Beam overlap** (`|set(HF) ∩ set(FlashRec)| / n`; both codebook-constrained):

| n | BF16 overlap | BF16 top-1 | FP8 overlap | FP8 top-1 |
| ---: | ---: | --- | ---: | --- |
| 1 | 100% | yes | 0% | no |
| 20 | 90% | yes | 80% | no |
| 50 | 86% | yes | 70% | no |
| 128 | 89% | yes | 72% | no |
| 512 | 92% | yes | 76% | no |

BF16 matches HF's best SID at every width:
`<|sid_begin|><s_a_0><s_b_104><s_c_5764><|sid_end|>`.
FP8 n=1 follows the near-tie (`s_a_5719…`); from n=20 the HF-best sequence is
inside the FlashRec beam.

The FP8 drop on OneRec is post-training quantization of a **BF16-trained**
checkpoint, not an engine bug. Prefer FP8-trained weights (SoHuRec-1.7B /
SoHuRec-0.6B table below).

SoHuRec-1.7B / SoHuRec-0.6B (FP8-trained, no channel scale;
decoder GEMM is FP8 on both sides; **device: NVIDIA RTX 5090**; eval prompts
from the corresponding serving traffic, not a public corpus).
Full protocol: [Evaluation](../docs/baselines.md).

| Checkpoint | n=4 | n=8 | n=50 | n=128 | n=512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SoHuRec-1.7B FP8 | 88% | 94% | 92% | 95% | 96% |
| SoHuRec-0.6B FP8 | 88% | 94% | 90% | 91% | 93% |

`n = 512` is the first 3 prompts (HF OOM on a longer remaining prompt). Prefill: SoHuRec-1.7B
top-1 100%, rank corr 0.9996; SoHuRec-0.6B top-1 94% with a noisy tail (rank corr 0.641)
but beam-set overlap still around 90%.

## Notes

- `--n` is beam width and candidate count at once; `--max-tokens` is SID depth.
  Depth-3 SIDs plus boundary tokens want `--max-tokens 5`.
- `temperature=0` (default) is deterministic and repeatable across runs.
- Wide beam (`n ≥ 512`, including `n = 1000`) needs a larger graph and slot budget:
  `--cuda-graph-max-bs 4096 --batch-slots 4096`. See
  [Configuration](../docs/configuration.md).
- Both scripts ship a placeholder prompt so they run as-is; real prompts come
  from your own recommendation stack. OneRec quality (Recall / NDCG /
  invalid_rate) is [scripts/eval_beam_matrix.py](../scripts/eval_beam_matrix.py).
  Numerical match vs HuggingFace is the section above, not those retrieval metrics.
