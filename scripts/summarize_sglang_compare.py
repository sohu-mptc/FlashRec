#!/usr/bin/env python3
"""Compatibility shim — see benchmark/recif/summarize_compare.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "benchmark" / "recif" / "summarize_compare.py"

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
