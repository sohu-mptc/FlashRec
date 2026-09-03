"""Shared pytest configuration.

``flashrec`` is imported directly once the project is installed in editable
mode (``pip install -e .``), so tests no longer need a ``sys.path`` hack. The
one exception is ``scripts/``: it is not a package, and
``tests/test_convert_recif_catalog.py`` imports ``convert_recif_catalog`` from
it, so we expose that directory here instead of in the test file.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
