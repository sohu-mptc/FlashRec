# Contributing to FlashRec

[简体中文](CONTRIBUTING.zh-CN.md)

Thanks for taking the time. This is the contribution process and the
code-style contract that CI enforces.

By submitting a contribution you agree that it is licensed under the project's
[Apache-2.0](LICENSE) license.

Product docs: [Documentation](docs/README.md).

## Development setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

`pre-commit` then runs on every commit and matches CI (`lint.yml`): isort,
black, a narrow ruff rule set (`F401`, `F821`, `UP037`), codespell,
clang-format for the fused CUDA kernel, plus the standard whitespace / YAML /
TOML / private-key checks.

`python/flashrec/kernel/include/` is a vendored copy of upstream SGLang JIT
headers — leave it byte-identical so a re-sync stays a clean diff.

```bash
pre-commit run --all-files     # check the whole tree on demand
```

## Tests

CPU unit tests run against synthetic fixtures:

```bash
python -m pytest
```

Optional integration checks (live SGLang parity, HuggingFace accuracy diff,
profiling) are documented in
[Configuration](docs/configuration.md).

Please add or extend a test when you change beam semantics, the SID trie, a
fused kernel, or the scheduler. GPU-only tests must `skipTest` when CUDA is
unavailable so the CPU CI job stays green.

## Pull requests

1. Branch off `main`. Direct commits to `main` are blocked by the
   `no-commit-to-branch` hook.
2. Keep the change focused. One concern per PR.
3. Update docs with the code:
   - new or renamed CLI flags → `docs/configuration.md`,
     `docs/configuration.zh-CN.md`, and both READMEs if the flag is commonly
     tuned
   - HTTP request or response fields → `docs/api.md`
   - user-visible behaviour → [Changelog](CHANGELOG.md) under `Unreleased`
4. Fill in the PR template (summary + test plan).
5. Wait for CI (`lint.yml` + `ci.yml`) before asking for review.

## Code style

- Python formatting is black (line length 88) with isort `profile=black`.
- Keep imports used and names defined; ruff `F401`/`F821` gate the build.
  `__init__.py` re-exports are exempt.
- CUDA / C++ in `python/flashrec/kernel/csrc/` is formatted with
  clang-format using the file-local `.clang-format`.
- Prefer existing module layout (`engine/`, `scheduler/`, `search/`,
  `kernel/`, …). The runtime stands on `sgl-kernel`, `flashinfer_python`, and
  `triton`.

## Reporting bugs

Functional bugs and feature requests: GitHub Issues (use the templates).

Security vulnerabilities: follow the private disclosure process in
[Security](SECURITY.md).

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Adding a model family

The engine currently implements Qwen3 dense
(`python/flashrec/models/qwen3.py`). A new family needs the same wiring:
fused QKV / gate-up GEMMs, the FP8 dual path, the three fused kernels,
residual-pair forwards, CUDA-graph capture safety (no `.item()` / host
branches on tensor values on the captured decode path), and SID layout
inference (`sid_layout.py`: `<s_a_0>` codebooks plus `<|sid_begin|>` /
`<|sid_end|>`, driven by `--sid-vocab-file`).
