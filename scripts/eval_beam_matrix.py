#!/usr/bin/env python3
"""Compatibility shim — implementation lives in flashrec.benchmark.recif."""

from __future__ import annotations

from flashrec.benchmark.recif import main

if __name__ == "__main__":
    raise SystemExit(main())
