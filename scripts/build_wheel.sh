#!/usr/bin/env bash
# Build a py3-none-any wheel, same flow as SGLang's python/ package.
# Version is BASE+g<shortsha>[.dirty] so the artifact identifies the git commit.
#
# pyproject.toml takes version dynamically from flashrec.version.__version__,
# so this script only stamps version.py (and restores it afterwards). Do not
# write a static ``version =`` back into pyproject.toml: that field is gone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION_FILE="$ROOT/python/flashrec/version.py"

BASE_VERSION="$(sed -n 's/^__version__ = "\([^"+]*\).*/\1/p' "$VERSION_FILE")"
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
cp "$VERSION_FILE" "$VERSION_BAK"
restore() {
  cp "$VERSION_BAK" "$VERSION_FILE"
  rm -f "$VERSION_BAK"
}
trap restore EXIT

cat > "$VERSION_FILE" <<EOF
"""Package version. Wheel builds stamp __git_commit__ via scripts/build_wheel.sh."""

__version__ = "${VERSION}"
__git_commit__ = "${GIT_HASH}"
EOF

python -m pip install -q --upgrade build wheel
python -m build --wheel --outdir "${OUTDIR:-$ROOT/dist}"
ls -lh "${OUTDIR:-$ROOT/dist}"/flashrec-*.whl
