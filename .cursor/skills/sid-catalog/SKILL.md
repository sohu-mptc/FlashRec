---
name: sid-catalog
description: >-
  Build SID trie catalogs for FlashRec from OpenOneRec RecIF packed
  mappings. Use when the user mentions sid-vocab-file, sid2pid, 物品目录,
  catalog, trie, convert_recif_catalog, build_catalog.sh, or illegal SID rate.
---

# 构建 SID catalog

`--sid-vocab-file` 需要逗号 key 的 JSON（如 `"a,b,c,1"`）。RecIF 原始
`sid2pid.json` / `sid2iid.json` 用打包整数，必须先转换。不转换直接喂引擎会建错 trie。

## 默认（RecIF video，4 层）

```bash
DATA_DIR=/path/to/OpenOneRec-RecIF/benchmark_data \
  bash scripts/build_catalog.sh
```

写出 `data/catalogs/sid2pid_beamrec_l4.json`。`data/` 已 gitignore，不要提交生成物。

| 变量 | 默认 | 含义 |
|------|------|------|
| `DATA_DIR` | （必填） | RecIF `benchmark_data` 目录 |
| `TASK` | `video` | `video` / `product` / `both` |
| `LEVELS` | `4` | `3` = `"a,b,c"`；`4` = `"a,b,c,1"`（第 4 层是 `{sid_begin, sid_end}`，code `1` = `<\|sid_end\|>`） |
| `OUT_DIR` | `data/catalogs` | 输出目录 |

等价 Python：

```bash
python scripts/convert_recif_catalog.py --data-dir /path/to/benchmark_data
python scripts/convert_recif_catalog.py sid2pid.json out.json --levels 3
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
- `LEVELS=4` 时 key 为 4 段且末段为 `1`
- 启动服务后 `invalid_rate` 应为 0；若 ~0.28，说明仍在跑开放词表（没挂 vocab 文件）
