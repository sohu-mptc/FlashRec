# 为 FlashRec 做贡献

[English](CONTRIBUTING.md)

感谢你抽出时间参与贡献。本文介绍贡献流程，以及 CI 强制执行的代码规范。

提交贡献即表示你同意该贡献以本项目的 [Apache-2.0](LICENSE) 许可证授权。

产品文档见 [文档](docs/README.zh-CN.md)。

## 开发环境

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

之后每次 commit 会自动跑 hook，与 CI（`lint.yml`）一致：isort、black、一组
收窄的 ruff 规则（`F401`、`F821`、`UP037`）、codespell、针对融合 CUDA kernel
的 clang-format，以及标准的空白字符 / YAML / TOML / 私钥检查。

`python/flashrec/kernel/include/` 是上游 SGLang JIT 头文件的 vendored
副本——请与上游保持逐字节一致，这样与上游重新同步时 diff 保持干净。

```bash
pre-commit run --all-files     # 随时全量检查
```

## 测试

CPU 单测跑在合成 fixture 上：

```bash
python -m pytest
```

可选的集成校验（与 SGLang 在线对齐、HuggingFace 精度 diff、profiling）见
[参数](docs/configuration.zh-CN.md)。

改动 beam 语义、SID trie、融合 kernel 或调度器时，请补充或扩展对应测试。仅能
在 GPU 上运行的测试要在 CUDA 不可用时 `skipTest`，让 CPU CI 任务保持绿色。

## Pull request

1. 从 `main` 拉分支后提 PR；`no-commit-to-branch` hook 会拦住直接提交到
   `main`。
2. 变更尽量聚焦，一个 PR 解决一件事。
3. 文档跟代码一起改：
   - 新增或改名的 CLI 参数 → `docs/configuration.md`、
     `docs/configuration.zh-CN.md`；若是常用参数，同步两个 README
   - HTTP 请求或响应字段 → `docs/api.md`
   - 用户可见行为 → [Changelog](CHANGELOG.md) 的 `Unreleased`
4. 填好 PR 模板（摘要 + 测试计划）。
5. 等 CI（`lint.yml` + `ci.yml`）通过后再请求 review。

## 代码风格

- Python 用 black（行宽 88），isort 使用 `profile=black`。
- 不要保留未使用的 import，不要引用未定义的名字；ruff `F401`/`F821` 会拦下
  构建。`__init__.py` 的再导出（re-export）除外。
- `python/flashrec/kernel/csrc/` 下的 CUDA / C++ 用该目录的
  `.clang-format`。
- 沿用现有模块切分（`engine/`、`scheduler/`、`search/`、`kernel/` 等）。
  运行时依赖 `sgl-kernel`、`flashinfer_python` 与 `triton`。

## 报告缺陷

功能性缺陷与功能请求：请通过 GitHub Issues 提交（使用模板）。

安全漏洞请按 [安全披露](SECURITY.md) 的私下披露流程报告。

参与本项目请遵守 [行为准则](CODE_OF_CONDUCT.md)。

## 接入新模型系列

引擎当前实现的是 Qwen3 dense（`python/flashrec/models/qwen3.py`）。接入
一个新系列需要完成同样的接线：融合 QKV / gate-up GEMM、FP8 双路径、三个融合
kernel、residual 二元组 forward、CUDA graph 捕获安全性（捕获的 decode 路径上
不要 `.item()` / 依赖张量值的 host 分支），以及 SID 布局推断（`sid_layout.py`：
`<s_a_0>` codebook 与 `<|sid_begin|>` / `<|sid_end|>`，由 `--sid-vocab-file`
驱动）。
