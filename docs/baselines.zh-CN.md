# 评测

[English](baselines.md)

硬件 **NVIDIA RTX 5090（32 GB）**。样本默认 **5,000**，
`max_tokens=5`，`temperature=0`。FlashRec 为 FP8 + SID trie；对照引擎为开放词表。

**不要把 OneRec QPS 与 SoHuRec QPS 直接相除。** RecIF prompt p50 ≈ **2,465**
token，生产 SoHuRec 画像 p50 ≈ **303** token（约 8×）；catalog / graph 路径也不同。


| 模型 / 语料                         | 说明                                                                                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **OneRec**                      | [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) RecIF-Bench video，模型 [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B) |
| **SoHuRec-1.7B / SoHuRec-0.6B** | 对应模型上的搜狐内部生成式推荐服务请求                                                                                                                           |


引擎分列，**不要把 SGLang-master 与 SGLang 0801 写成一行**：


| 简称             | 实现                                                                                                                     | 约束                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **FlashRec**   | 本仓库                                                                                                                    | FP8 + SID trie，`invalid_rate = 0`       |
| **SGLang-master** | SGLang 主线（[PR #31626](https://github.com/sgl-project/sglang/pull/31626) wide-beam） | 开放词表 |
| **SGLang 0801** | [`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801) | 开放词表（SoHuRec-0.6B 仍带 SID `lm_head` 限制） |
| **vLLM / TensorRT-LLM** | 仅 OneRec                                                                                                         | 开放词表                                    |


开放词表必须同时报 `invalid_rate`（约 17–27% 非法 SID，不能上架）。只填**跑满**的格子；空或
`—` 表示未跑满 / 失败（见各节失败模式），不要写成「没开跑」。

## 结论

1. **公开 RecIF、同一套长 prompt**：FlashRec（trie）在能跑满的格子里都是最快。`n=50`
   饱和约 **28 QPS**，SGLang 0801 **18**，SGLang-master **14**，TensorRT-LLM **13**，
   vLLM **3.8**。`n=1000` FlashRec 饱和 **4.62 QPS**，SGLang-master 与 SGLang 0801 只能稳住 conc=1
   （2.5–3.0）。
2. **质量**：开放词表各引擎 recall@32 几乎一样（`n=50` 约 0.034），差主要来自**有没有
   SID 约束**，不是引擎实现。开放词表 `invalid ≈ 18–27%`；FlashRec `invalid = 0`。
   **不要引用 FlashRec conc=1 的 recall@32**（unique-beam 塌缩，看起来虚高）。
3. **生产短 prompt（SoHuRec ~300 token）**：FlashRec 相对 SGLang-master 与 SGLang 0801 约
   **2.2–3.4×**，随 beam 变宽差距拉大。`n=50` 时 1.7B FlashRec 饱和
   **220 QPS**，0.6B **303 QPS**；`n=512` 时 1.7B FlashRec **~48 QPS** vs
   0801 / SGLang-master **~16 QPS**。
4. **宽 beam + 高并发**是 SGLang-master、SGLang 0801、TensorRT-LLM、vLLM 的共同短板（OOM、
   槽位、断连）。按格重启 + `max-running=(n+1)×conc` 之后，0801 在生产 1.7B 上能跑满
   50 / 128 / 512。

## SoHuRec-1.7B / SoHuRec-0.6B

FP8，生产短 prompt（约 300 token），**5,000** 条。SGLang-master 为
[PR #31626](https://github.com/sgl-project/sglang/pull/31626)；SGLang 0801 为
[`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801)。

1.7B 上 0801 **去掉了** `--beam-search-lm-head-special-token-ids`（内部
`topk(k=2n)`，`n=1000` 时 `2n=2000` 超过约 1,535 个 SID token 会炸）；0.6B SID
足够，带着限制跑。因此 1.7B 的 0801 是开词表 beam，只比速度。

相对饱和吞吐（该 `n` 下各引擎最高已跑满 conc）：


| n    | FlashRec / SGLang 0801 | FlashRec / SGLang-master |
| ---: | ---------------------: | ----------------: |
| 50   |                 **2.26×** |            **2.71×** |
| 128  |                 **2.52×** |            **2.46×** |
| 512  |                 **2.95×** |            **2.87×** |
| 1000 |        **2.16×**（仅 conc=1） |   **2.14×**（仅 conc=1） |


`n=1000` 时 conc ≥ 8 SGLang-master 与 SGLang 0801 都失败，峰值不可比。

![SoHuRec-1.7B 吞吐对比 SGLang 0801 / SGLang-master（并发 32）](figures/perf-serving-throughput.svg)

### SoHuRec-1.7B QPS


| n    | 引擎               | conc=1     | conc=8      | conc=16     | conc=32     |
| ---: | ---------------- | ----------: | -----------: | -----------: | -----------: |
| 50   | **FlashRec**     |     **41.11** |     **159.72** |     **168.68** |     **219.54** |
| 50   | SGLang 0801      |       22.84 |        68.56 |        85.94 |        97.20 |
| 50   | SGLang-master           |       19.75 |        56.05 |        74.16 |        81.12 |
| 128  | **FlashRec**     |     **25.09** |     **101.99** |     **114.61** |     **136.78** |
| 128  | SGLang 0801      |       20.41 |        48.37 |        52.15 |        54.22 |
| 128  | SGLang-master           |       18.48 |        44.66 |        53.61 |        55.65 |
| 512  | **FlashRec**     |     **19.48** |      **41.47** |      **47.76** |      **47.16** |
| 512  | SGLang 0801      |       11.45 |        15.98 |        16.20 |        16.03 |
| 512  | SGLang-master           |       11.13 |        16.56 |        16.47 |        16.63 |
| 1000 | **FlashRec**     |     **15.54** |      **24.64** |        24.53 |        24.64 |
| 1000 | SGLang 0801      |        7.19 |            — |            — |            — |
| 1000 | SGLang-master           |        7.26 |            — |            — |            — |


### SoHuRec-0.6B QPS


| n    | 引擎               | conc=1     | conc=8      | conc=16     | conc=32     |
| ---: | ---------------- | ----------: | -----------: | -----------: | -----------: |
| 50   | **FlashRec**     |     **47.43** |     **184.44** |     **251.17** |     **302.60** |
| 50   | SGLang 0801      |       28.17 |        75.79 |        99.43 |       116.66 |
| 50   | SGLang-master           |       23.51 |        60.53 |        87.17 |       104.76 |
| 128  | **FlashRec**     |     **28.31** |     **101.81** |     **179.10** |     **185.86** |
| 128  | SGLang 0801      |       24.12 |        57.14 |        61.95 |        65.44 |
| 128  | SGLang-master           |       21.05 |        51.22 |        59.76 |        72.42 |
| 512  | **FlashRec**     |     **21.77** |      **52.99** |      **59.75** |      **60.16** |
| 512  | SGLang 0801      |       13.30 |        19.53 |        19.19 |            — |
| 512  | SGLang-master           |       12.55 |            — |            — |            — |
| 1000 | **FlashRec**     |     **18.01** |      **32.19** |        32.02 |        32.11 |
| 1000 | SGLang 0801      |        8.51 |            — |            — |            — |
| 1000 | SGLang-master           |        8.34 |            — |            — |            — |


0.6B FlashRec 峰值 **303 QPS**（`n=50` conc=32），约 1.7B 的 **1.4×**。`n=1000`
FlashRec 饱和 **32 QPS**，约 1.7B 的 **1.3×**。

## OneRec

[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) RecIF-Bench video，
模型 [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B)。5,000 条
（`video_test.parquet` 随机，seed=42）。FlashRec catalog 为
`sid2pid_beamrec_l4.json`（约 152 万 SID，深度 4，码本 8,192）。对照无 trie。

QPS 只填跑满 5,000 的格子。

![OneRec-1.7B 吞吐对比 SGLang 0801 / SGLang-master / TensorRT-LLM / vLLM（n=50 饱和）](figures/perf-onerec-qps.svg)

### 速度 QPS


| n    | 引擎               | conc=1     | conc=8     | conc=16    | conc=32    |
| ---: | ---------------- | ----------: | ----------: | ----------: | ----------: |
| 50   | **FlashRec**     |     **15.09** |    **26.47** |    **27.68** |    **28.08** |
| 50   | SGLang 0801      |       12.36 |       17.14 |       17.95 |       18.20 |
| 50   | SGLang-master           |       11.43 |       13.84 |       13.88 |       13.85 |
| 50   | TensorRT-LLM     |       11.64 |       12.71 |       12.70 |       12.69 |
| 50   | vLLM             |        2.47 |        3.85 |        3.89 |        3.78 |
| 128  | **FlashRec**     |     **12.17** |    **19.18** |    **19.41** |    **19.91** |
| 128  | SGLang 0801      |        9.97 |       12.83 |       13.19 |       13.20 |
| 128  | SGLang-master           |        9.09 |        9.80 |        9.79 |        9.79 |
| 128  | vLLM             |        0.99 |        1.57 |        1.57 |        1.56 |
| 128  | TensorRT-LLM     | 仅 7 条      | 仅 7 条      | 仅 7 条      | 仅 7 条      |
| 512  | **FlashRec**     |      **7.10** |     **8.09** |     **8.18** |     **8.17** |
| 512  | SGLang 0801      |        5.01 |        5.61 |        OOM |        OOM |
| 512  | SGLang-master           |        4.20 |        4.42 |        失败 |        失败 |
| 512  | vLLM             |        0.16 |       中断   |          — |          — |
| 512  | TensorRT-LLM     | 32 GB 不支持  | 同左         | 同左         | 同左         |
| 1000 | **FlashRec**     |      **4.38** |     **4.62** |     **4.62** |     **4.62** |
| 1000 | SGLang 0801      |        3.03 |        失败 |        失败 |        失败 |
| 1000 | SGLang-master           |        2.51 |        失败 |          — |          — |


相对 FlashRec 饱和吞吐（该 `n` 下各引擎最高已跑满 conc）：


|    n | FlashRec | SGLang 0801     | SGLang-master           | TensorRT-LLM | vLLM             | FlashRec / 0801          | FlashRec / SGLang-master        |
| ---: | -------: | --------------: | ---------------: | -----------: | ---------------: | -----------------------: | -----------------------: |
|   50 |    28.08 |           18.20 |            13.88 |        12.71 |             3.89 |                   **1.54×** |                   **2.02×** |
|  128 |    19.91 |           13.20 |             9.80 |            — |             1.57 |                   **1.51×** |                   **2.03×** |
|  512 |     8.18 |            5.61 |             4.42 |            — |    0.16（仅 conc=1） |                   **1.46×** |                   **1.85×** |
| 1000 |     4.62 | 3.03（仅 conc=1） |    2.51（仅 conc=1） |            — |               — | **1.52×**（相对 conc=1） | **1.84×**（相对 conc=1） |


### 质量（只引用 conc ≥ 8）

FlashRec conc=1 的 `recall@32 ≈ 0.12` 是 unique-beam 塌缩，**不要引用**。开放词表
illegal SID ≈ **17–27%**；FlashRec trie 为 **0**。


| n    | 引擎           | conc | recall@32 | hit@32 | invalid |
| ---: | ------------ | ---: | --------: | -----: | ------: |
|   50 | FlashRec     |    8 |     0.034 |  0.180 |       0 |
|   50 | SGLang 0801  |    8 |     0.034 |  0.193 |   0.270 |
|   50 | SGLang-master       |    8 |     0.034 |  0.195 |   0.260 |
|   50 | TensorRT-LLM |    8 |     0.034 |  0.195 |   0.259 |
|   50 | vLLM         |    8 |     0.034 |  0.194 |   0.260 |
|  128 | FlashRec     |    8 |     0.054 |  0.219 |       0 |
|  128 | SGLang 0801  |    8 |     0.038 |  0.216 |   0.236 |
|  128 | SGLang-master       |    8 |     0.038 |  0.215 |   0.227 |
|  128 | vLLM         |    8 |     0.038 |  0.214 |   0.227 |
|  512 | FlashRec     |    8 |     0.051 |  0.250 |       0 |
|  512 | SGLang 0801  |    8 |     0.040 |  0.224 |   0.194 |
|  512 | SGLang-master       |    8 |     0.040 |  0.223 |   0.185 |
|  512 | vLLM         |    1 |     0.040 |  0.223 |   0.185 |
| 1000 | FlashRec     |    8 | **0.074** |  0.218 |       0 |
| 1000 | SGLang 0801  |    1 |     0.040 |  0.223 |   0.179 |
| 1000 | SGLang-master       |    1 |     0.040 |  0.224 |   0.170 |


开放词表各引擎 recall@32 几乎一样。FlashRec 在合法集合里打，hit@32 随 `n` 从 0.18
升到 0.25；开放词表 hit 看起来不低，但约 1/4 beam 非法、不能上架。FlashRec
`n=1000` 的 recall@32 = 0.074 明显高于开放词表的 0.040。

### RecIF 失败模式

- **SGLang 0801**：`n=512` conc=16/32 OOM（`set a smaller --max-running-requests or --beam-width`）；
  `n=1000` conc ≥ 8 失败。按格重启 + `max_running=(n+1)×conc` 后，`n=50/128` 全满，
  `n=512` 只能到 conc=8。
- **SGLang-master**：`n=50/128` 全满；`n=512` conc=1/8 为 4.20 / **4.42**，conc=16/32
  失败；`n=1000` conc ≥ 8 失败。
- **TensorRT-LLM**：`n=128` 各 7 条即停；`n=512` 在 32 GB 上声明不支持。
- **vLLM**：能跑但极慢（`n=512` conc=1 = 0.16 QPS）；`n=512` conc ≥ 8 失败。

## OneRec-1.7B 与 SoHuRec-1.7B

同一引擎（FlashRec FP8 + SID trie，`n=50`，各 5,000 条）。


|                            | SoHuRec-1.7B     | OneRec-1.7B            |
| -------------------------- | ---------------- | ---------------------- |
| prompt tokens p50（p10–p90） | **303**（205–389） | **2,465**（2,220–2,563） |
| prompt 形态                  | 短请求              | 长 SID 历史（约 500 条）      |
| SID 深度                     | 3                | 4                      |
| 码本大小                       | 512              | 8,192                  |
| catalog                    | 约 43 万 SID       | 约 152 万 SID            |
| trie                       | 稠密               | CSR                    |


prompt 长度差约 8 倍，绝对 QPS 不宜直接比较。


| conc | OneRec-1.7B QPS | SoHuRec-1.7B QPS |
| ---: | --------------: | ---------------: |
|    1 |           15.09 |            41.11 |
|    8 |           26.47 |           159.72 |
|   16 |           27.68 |           168.68 |
|   32 |           28.08 |           219.54 |


OneRec-1.7B 在并发 ≥ 8 后饱和于 26–28 QPS；SoHuRec-1.7B 为 160–220 QPS。

## NVIDIA SID-GR（参考，硬件不可比）

对照 [NVIDIA SID-GR inference](https://github.com/NVIDIA/recsys-examples/tree/main/examples/sid-gr-inference)。
NVIDIA 公开数字是 **H100 / Qwen3-1.7B / BF16**；本组是 **RTX 5090 / FlashRec FP8**，
不能直接当加速比。同卡同精度对照未做。

离线合成 ctx（beam=256，decode=3），OneRec-1.7B FlashRec：


|  ctx | batch | FlashRec ms | FlashRec QPS | NVIDIA SID-GR ms（H100） | NVIDIA SGLang ms（H100） |
| ---: | ----: | ----------: | -----------: | ----------------------: | ----------------------: |
| 1000 |     1 |        39.6 |         25.2 |                    17.6 |                    33.5 |
| 1000 |     8 |       211.5 |         37.8 |                    93.2 |                   199.3 |
| 5000 |     1 |       117.0 |         8.55 |                    42.3 |                    94.6 |
| 5000 |     8 |       844.1 |         9.48 |                   307.9 |                   685.4 |


在线 HTTP（ctx=5000，beam=256，conc=4，64 请求）：OneRec-1.7B FlashRec
**9.02 req/s**（median 436 ms）。NVIDIA SID-GR 约 **19.7 req/s**（median ~199 ms）。

真实 prompt 离线（不对齐 NVIDIA 的 ctx / beam；beam=50、decode=5、SID trie）。内部短
prompt 更快是预期：


| batch | RecIF OneRec QPS | 内部 SoHuRec-1.7B QPS | 内部 SoHuRec-0.6B QPS |
| ----: | ---------------: | --------------------: | --------------------: |
|     1 |             17.3 |                  42.5 |                  51.6 |
|     8 |             28.5 |                 179.4 |                 282.3 |


## 与 HuggingFace 的数值对齐

codebook 约束 beam，无 SID trie。HuggingFace 为 BF16 参考。命令见
[示例](../examples/README.md)。

**OneRec-1.7B**


| FlashRec | max \|Δlogprob\| | mean \|Δ\| | top-1    | top-10 | Spearman |
| -------- | --------------: | ---------: | -------- | -----: | -------: |
| BF16     |           0.196 |      0.042 | 100%     |    90% |    0.993 |
| FP8      |           0.571 |      0.088 | top-2 对调 |    90% |    0.992 |



| n   | BF16 重叠 | BF16 top-1 | FP8 重叠 | FP8 top-1 |
| --- | ------- | ---------- | ------ | --------- |
| 1   | 100%    | 是          | 0%     | 否         |
| 20  | 90%     | 是          | 80%    | 否         |
| 50  | 86%     | 是          | 70%    | 否         |
| 128 | 89%     | 是          | 72%    | 否         |
| 512 | 92%     | 是          | 76%    | 否         |


OneRec-1.7B 公开 checkpoint 为 **BF16 训练**，FP8 路径是加载时在线量化（W8A8）。
上表中 FP8 相对 HuggingFace 的下降（`n = 1` top-1 对调，`n ≥ 50` 重叠仅 70–76%）
来自该量化误差，**不是框架实现问题**：同一引擎的 BF16 路径在所有宽度上与
HuggingFace 最优 SID 一致。

**SoHuRec-1.7B / SoHuRec-0.6B**（FP8 训练，两侧 decoder GEMM 均为 FP8）


| 模型               | n=4 | n=8 | n=50 | n=128 | n=512 |
| ---------------- | --- | --- | ---- | ----- | ----- |
| SoHuRec-1.7B FP8 | 88% | 94% | 92%  | 95%   | 96%   |
| SoHuRec-0.6B FP8 | 88% | 94% | 90%  | 91%   | 93%   |


prefill top-1：SoHuRec-1.7B 100%，SoHuRec-0.6B 94%。同一套 FP8 解码路径上，经 FP8 训练的
checkpoint 将 beam 集合重叠恢复到 **88–96%** / **88–93%**。FP8 服务精度主要由
模型是否经过 FP8 训练决定；生产路径建议使用 FP8 训练（或带 `weight_scale` 的
预量化）checkpoint，避免仅对 BF16 权重做 post-training 量化。

## 失败格子

- RecIF SGLang 0801 `n=512` conc=16/32；RecIF SGLang 0801 / SGLang-master `n=1000` conc ≥ 8
- RecIF SGLang-master `n=512` conc=16/32
- RecIF vLLM `n=512` conc=8/16/32；TensorRT-LLM `n ≥ 128`
- NVIDIA sid-gr-inference **同卡同精度**对照未做（目前只有论文 H100 数字）

## 复现

```bash
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data bash scripts/build_catalog.sh

MODEL_PATH=/path/to/OneRec-1.7B \
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
bash scripts/bench_sglang_compare.sh
```


| 引擎           | Beam 入口                                                                                        | 版本                                                                                                                               |
| ------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| SGLang-master       | 请求 `sampling_params.beam_width`（[PR #31626](https://github.com/sgl-project/sglang/pull/31626)） | `lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1`                                                                             |
| SGLang 0801  | 同样 `/generate` + `beam_width`                                                                  | [`cswuyg/sglang` `feature/beam_search_update_0801`](https://github.com/cswuyg/sglang/tree/feature/beam_search_update_0801)        |
| vLLM         | `use_beam_search=true` + `n`                                                                   | `vllm/vllm-openai:v0.28.0`（`--max-logprobs ≥ 2n`）                                                                                |
| TensorRT-LLM | `--max_beam_width` 与请求 `best_of` 一致                                                            | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`                                                                                  |
| HuggingFace  | `num_beams` / `num_return_sequences`                                                           | 本机 `transformers`                                                                                                                |


单 cell：`python scripts/eval_beam_matrix.py --engine <name> --server-url <url> …`。
SGLang-master / SGLang 0801 排名结果在 `meta_info.beam_results[]`。
