"""Packaging invariants that only break once a wheel reaches a user.

``import flashrec`` must not pull in torch or numpy: ``__version__`` and
``check_env`` have to work on the half-installed environments they exist to
report on. The version is single-sourced from ``version.py`` via setuptools
``dynamic`` metadata, so the declaration stays asserted rather than duplicated.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import flashrec

ROOT = Path(__file__).resolve().parent.parent
HEAVY = ("torch", "numpy", "transformers")


def _in_subprocess(code):
    """Run code in a fresh interpreter so this suite's own imports don't leak."""
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    if out.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{out.stderr}")
    return out.stdout.strip()


class TestLazyImport:
    def test_bare_import_pulls_no_heavy_deps(self):
        loaded = _in_subprocess(
            "import flashrec, sys;"
            f"print(' '.join(m for m in {HEAVY!r} if m in sys.modules))"
        )
        assert loaded == "", f"import flashrec loaded: {loaded}"

    def test_version_readable_without_heavy_deps(self):
        assert (
            _in_subprocess("import flashrec;print(flashrec.__version__)")
            == flashrec.__version__
        )

    def test_lazy_attrs_resolve(self):
        assert flashrec.BeamRecConfig.__name__ == "BeamRecConfig"
        assert flashrec.BeamRecEngine.__name__ == "BeamRecEngine"

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            flashrec.definitely_not_here

    def test_dir_lists_public_api(self):
        assert sorted(flashrec.__all__) == sorted(dir(flashrec))

    def test_cli_help_lists_catalog_without_torch(self):
        out = _in_subprocess(
            "from flashrec.cli import _build_parser; import sys;"
            "text=_build_parser().format_help();"
            "print('yes' if '--catalog PATH' in text else 'no');"
            f"print('none' if not any(m in sys.modules for m in {HEAVY!r}) else "
            f"' '.join(m for m in {HEAVY!r} if m in sys.modules))"
        )
        lines = out.splitlines()
        assert lines[0] == "yes"
        assert lines[1] == "none", f"flashrec.cli loaded: {lines[1]}"


class TestVersionSingleSource:
    """version.py is the only place the version is written down."""

    def _pyproject(self):
        if sys.version_info < (3, 11):
            pytest.skip("tomllib requires Python 3.11+")
        import tomllib

        with open(ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_version_is_dynamic(self):
        project = self._pyproject()["project"]
        assert "version" in project.get("dynamic", [])
        assert "version" not in project, "static version would drift from version.py"

    def test_dynamic_source_is_version_module(self):
        dynamic = self._pyproject()["tool"]["setuptools"]["dynamic"]
        assert dynamic["version"] == {"attr": "flashrec.version.__version__"}

    def test_build_wheel_stamps_version_module_not_pyproject(self):
        # ``pip install -e .`` never touches this script; the wheel path does.
        # A leftover regex for static ``version =`` in pyproject.toml is what
        # made ``bash scripts/build_wheel.sh`` fail after version went dynamic.
        script = (ROOT / "scripts" / "build_wheel.sh").read_text()
        assert "python/flashrec/version.py" in script
        assert "failed to patch version" not in script
        assert 'r\'^version\\s*=\\s*"[^"]*"\'' not in script


class TestCheckEnv:
    def test_runs_and_reports_version(self):
        out = _in_subprocess(
            "import flashrec.check_env as c;print(c.format_report(c.collect()))"
        )
        assert "flashrec" in out
        assert "Python" in out
