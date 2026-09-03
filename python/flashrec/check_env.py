"""Collect environment info for bug reports.

    python3 -m flashrec.check_env

Everything is probed defensively: this has to produce useful output on the
broken environments it exists to diagnose, so a missing or half-installed
dependency is reported as such instead of raising.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

# Distribution names, not import names: flashinfer_python installs as
# "flashinfer", apache-tvm-ffi as "tvm_ffi".
_DISTS = (
    "flashrec",
    "torch",
    "sgl-kernel",
    "flashinfer_python",
    "triton",
    "transformers",
    "safetensors",
    "numpy",
    "fastapi",
    "pydantic",
    "uvicorn",
    "apache-tvm-ffi",
    "ninja",
)

# Env vars that change numerics or kernel selection, so they belong in a report.
_ENV_VARS = (
    "FLASHREC_DENSE_FINALIZE",
    "FLASHREC_DIFF_MODEL",
    "FLASHREC_TORCH_PROFILER_DIR",
    "FLASHREC_TRIE_DENSE_MAX_CELLS",
    "FLASHREC_URL",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


def _dist_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"
    except Exception as e:  # pragma: no cover - corrupt metadata
        return f"error: {type(e).__name__}"


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _git_commit() -> Optional[str]:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isdir(os.path.join(repo, ".git")):
        return None
    sha = _run(["git", "-C", repo, "rev-parse", "--short", "HEAD"])
    if sha is None:
        return None
    dirty = _run(["git", "-C", repo, "status", "--porcelain"])
    return f"{sha}{'-dirty' if dirty else ''}"


def _torch_info() -> List[Tuple[str, str]]:
    try:
        import torch
    except Exception as e:
        return [("torch import", f"failed: {type(e).__name__}: {e}")]

    rows = [
        ("torch build CUDA", str(torch.version.cuda)),
        (
            "torch build cuDNN",
            str(getattr(torch.backends.cudnn, "version", lambda: None)()),
        ),
        ("CUDA available", str(torch.cuda.is_available())),
    ]
    if not torch.cuda.is_available():
        return rows

    try:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            rows.append((f"GPU {i}", f"{name} (sm_{major}{minor}, {total:.1f} GiB)"))
    except Exception as e:  # pragma: no cover - driver-dependent
        rows.append(("GPU probe", f"failed: {type(e).__name__}: {e}"))
    return rows


def collect() -> List[Tuple[str, str]]:
    """Return (label, value) pairs in report order."""
    rows: List[Tuple[str, str]] = [
        ("Python", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
    ]

    commit = _git_commit()
    if commit:
        rows.append(("FlashRec commit", commit))

    rows.append(("", ""))
    rows.extend((name, _dist_version(name)) for name in _DISTS)

    rows.append(("", ""))
    rows.extend(_torch_info())

    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver:
        rows.append(("NVIDIA driver", driver.splitlines()[0].strip()))
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        line = [ln for ln in nvcc.splitlines() if "release" in ln]
        rows.append(("nvcc", line[0].strip() if line else "present"))

    env = [(k, os.environ[k]) for k in _ENV_VARS if k in os.environ]
    if env:
        rows.append(("", ""))
        rows.extend(env)
    return rows


def format_report(rows: List[Tuple[str, str]]) -> str:
    width = max((len(k) for k, _ in rows if k), default=0)
    lines = []
    for key, value in rows:
        lines.append("" if not key else f"{key.ljust(width)} : {value}")
    return "\n".join(lines)


def main() -> int:
    print(format_report(collect()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
