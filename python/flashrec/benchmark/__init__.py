"""FlashRec benchmark clients (RecIF eval and online serving throughput).

Run from the CLI::

    python -m flashrec.benchmark.recif --help
    python -m flashrec.benchmark.serving --help

Orchestration scripts live under the repo ``benchmark/`` directory.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["recif", "serving"]


def __getattr__(name: str) -> Any:
    if name in ("recif", "serving"):
        return importlib.import_module(f"flashrec.benchmark.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
