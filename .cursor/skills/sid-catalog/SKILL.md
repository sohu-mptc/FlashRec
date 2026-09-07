---
name: sid-catalog
description: >-
  Build SID trie catalogs for FlashRec from OpenOneRec RecIF packed
  mappings. Use when the user mentions sid-vocab-file, sid2pid, 物品目录,
  catalog, trie, flashrec --catalog, convert_recif_catalog, build_catalog.sh,
  or illegal SID rate.
---

# 构建 SID catalog

`--sid-vocab-file` 需要逗号 key 的 JSON（如 `"a,b,c,1"`）。RecIF 原始
`sid2pid.json` / `sid2iid.json` 用打包整数，必须先转换。不转换直接喂引擎会建错 trie。

## 默认（按源文件自动分层）

```bash
flashrec --catalog /path/to/OpenOneRec-RecIF/benchmark_data
```

RecIF 打包整数（`a*8192^2+b*8192+c`）会写成 4 层 `data/catalogs/sid2pid_beamrec_l4.json`（末段 `1` = `<|sid_end|>`）。逗号 key 按源文件段数原样保留。`data/` 已 gitignore，不要提交生成物。

| 旗标 / 变量 | 默认 | 含义 |
|------|------|------|
| `--catalog` / `DATA_DIR` | （必填） | RecIF `benchmark_data` 目录或 `sid2pid.json` |
| `--catalog-task` / `TASK` | `video` | `video` / `product` / `both` |
| `--catalog-levels` / `LEVELS` | （自动） | 覆盖推断层数。`3` = `"a,b,c"`；`4` = 在 3 层 RecIF 后追加 `{sid_begin, sid_end}` |
| `--catalog-out` / `OUT_DIR` | `data/catalogs` | 输出目录或 JSON 文件 |

```bash
flashrec --catalog sid2pid.json --catalog-out out.json
flashrec --catalog sid2pid.json --catalog-out out.json --catalog-levels 3
DATA_DIR=/path/to/benchmark_data bash scripts/build_catalog.sh
```

## 与服务参数对齐

服务只需 `--model-path` + `--sid-vocab-file`。引擎从 tokenizer 读取
`<s_a_*>` / `<|sid_begin|>` / `<|sid_end|>`，再结合 catalog 层数推断 token
区间、codebook 大小和 boundary。启动日志会打印
`Inferred --sid RANGE/SIZES (boundary B) from tokenizer + catalog`。

4 层 catalog（key `"a,b,c,1"`）把结束符做成末层 codebook，生成序列为
`[a, b, c, sid_end]`，结束符 logprob 计入 `sequence_score`。3 层 catalog
（`"a,b,c"`）的 range 只含 codebook，boundary 包在后面。

tokenizer 不用这套命名时才显式传 `--sid`。换模型后会重新推断，不要把旧
checkpoint 的 token id 抄过去。

## 检查

- 输出 JSON 根对象的 key 是逗号分隔整数，不是打包十进制
- RecIF 打包源默认 4 段且末段为 `1`；逗号源保持原段数（不要把真实第 4 码改成 `1`）
- 启动服务后 `invalid_rate` 应为 0；若 ~0.28，说明仍在跑开放词表（没挂 vocab 文件）
