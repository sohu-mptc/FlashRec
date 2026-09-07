#!/usr/bin/env python3
"""Build a FlashRec SID catalog from RecIF packed mappings.

Prefer the engine CLI::

    flashrec --catalog /path/to/benchmark_data
    flashrec --catalog sid2pid.json --catalog-out out.json
"""

from __future__ import annotations

import sys

from flashrec.catalog import script_main

if __name__ == "__main__":
    sys.exit(script_main())
