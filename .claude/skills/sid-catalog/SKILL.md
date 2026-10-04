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

## 已知 codebook（务必对号入座）

完整表：NAS `flashrec_data/data/catalogs/CODEBOOK.md` + `codebook.json`；
Cursor rule `.cursor/rules/sid-codebook.mdc`。

| 模型 | 逻辑大小 | 推荐 `--sid` / 边界 | Catalog |
|------|---------:|---------------------|---------|
| OneRec-1.7B | **8192³** | 从 tokenizer+catalog 推断；begin/end=176245/176246 | `sid2pid_beamrec_l4.json` 或 RecIF `sid2pid.json` |
| SoHuRec-1.7B | **512³** | `151669:153204/512,512,512`（begin/end=153205/153206） | `sid2pid_sohurec_l3.json` |
| SoHuRec-0.6B | **512³** | `151669:153204/512,512,512@176245,176246`（**必须** `@`） | 同上 |

RecIF 打包基数是 **8192**。SoHu GT 数据在
`/mnt/nas_new_1/flashrec_data/data/sohu_gt/`（说明见同级 `sohu_gt.CODEBOOK.md`）。

## 解码命名（不要为了消 catalog_miss 去改 OneRec catalog）

Catalog key 始终是各层的 dense code。OneRec L4 是 `"a,b,c,1"`，末段 `1` 是 `<|sid_end|>`。token id = 该层 `<s_*_0>` + code。

SoHu-0.6B 的服务窗口是 `151669 + level*512 + code`，tokenizer 仍把这 1536 个 id 叫 `<s_a_*>`。`TokenizerAdapter` **只在** `codebook_sizes` 为 `512,512,512` 且 `<s_b_0>` 距离 `<s_a_0>` 超过 512 时，把这段重写成 `<s_a/b/c_code>`。SoHu-1.7B / v2 的 `<s_b_0>` 就在 `s_a_0+512`，这个函数直接返回。

OneRec 是 8192 宽的真三层，禁止这层重写。重写后 `<s_a_565>` 会印成 `<s_b_53>`，`<s_a_1523>` 会印成 `<s_c_499>`。trie 里的 SID 仍合法，评测按印出来的字符串查 catalog 就记 `catalog_miss`（约 0.09，duplicate=0）。还原命名后这些 beam 都在 `sid2pid_beamrec_l4.json` 里。

`canonicalize_sohu_sid_tokens` 只处理三个 token 全是 `<s_a_*>` 的 SoHu beam，不会把已经印成 `<s_b_*>` 的 OneRec 改回去。

## 检查

- 输出 JSON 根对象的 key 是逗号分隔整数，不是打包十进制
- RecIF 打包源默认 4 段且末段为 `1`；逗号源保持原段数（不要把真实第 4 码改成 `1`）
- 开放词表（没挂 vocab 文件）时 invalid 接近 1，或约 0.28
- OneRec invalid 约 0.09 且首码是 code 小于 512 的 `<s_b_*>` / `<s_c_*>`：解码重写用错了，不要重建 catalog
