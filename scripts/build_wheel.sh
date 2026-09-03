#!/usr/bin/env bash
# Build a py3-none-any wheel, same flow as SGLang's python/ package.
# Version is BASE+g<shortsha>[.dirty] so the artifact identifies the git commit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION_FILE="$ROOT/python/flashrec/version.py"
PYPROJECT="$ROOT/pyproject.toml"

BASE_VERSION="$(sed -n 's/^__version__ = "\([^"+]*\).*/\1/p' "$VERSION_FILE")"
if [[ -z "${BASE_VERSION}" ]]; then
  BASE_VERSION="$(sed -n 's/^version = "\([^"+]*\).*/\1/p' "$PYPROJECT")"
fi
BASE_VERSION="${BASE_VERSION:-0.1.0}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_HASH="$(git rev-parse --short=7 HEAD)"
  VERSION="${BASE_VERSION}+g${GIT_HASH}"
  if [[ -n "$(git status --porcelain)" ]]; then
    VERSION="${VERSION}.dirty"
  fi
else
  GIT_HASH="unknown"
  VERSION="${BASE_VERSION}"
fi

echo "Building flashrec==${VERSION} (git ${GIT_HASH})"

VERSION_BAK="$(mktemp)"
PYPROJECT_BAK="$(mktemp)"
cp "$VERSION_FILE" "$VERSION_BAK"
cp "$PYPROJECT" "$PYPROJECT_BAK"
restore() {
  cp "$VERSION_BAK" "$VERSION_FILE"
  cp "$PYPROJECT_BAK" "$PYPROJECT"
  rm -f "$VERSION_BAK" "$PYPROJECT_BAK"
}
trap restore EXIT

python3 - "$PYPROJECT" "$VERSION" <<'PY'
import re
import sys

path, version = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
text, n = re.subn(
    r'^version\s*=\s*"[^"]*"',
    f'version = "{version}"',
    text,
    count=1,
    flags=re.M,
)
if n != 1:
    raise SystemExit(f"failed to patch version in {path}")
open(path, "w", encoding="utf-8").write(text)
PY

cat > "$VERSION_FILE" <<EOF
"""Package version. Wheel builds stamp __git_commit__ via scripts/build_wheel.sh."""

__version__ = "${VERSION}"
__git_commit__ = "${GIT_HASH}"
EOF

python -m pip install -q --upgrade build wheel
python -m build --wheel --outdir "${OUTDIR:-$ROOT/dist}"
ls -lh "${OUTDIR:-$ROOT/dist}"/flashrec-*.whl
