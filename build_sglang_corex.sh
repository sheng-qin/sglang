#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
COREX_VERSION="${COREX_VERSION:-latest}"
SGLANG_COREX_SKIP_RUST="${SGLANG_COREX_SKIP_RUST:-1}"

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION="$(date --utc +%Y%m%d%H%M%S)"
fi

export LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"
export SGLANG_USE_IXFORMER="${SGLANG_USE_IXFORMER:-1}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0+corex.${COREX_VERSION}}"

cd "${SCRIPT_DIR}/python"

ln -sf ../LICENSE LICENSE
ln -sf ../README.md README.md

cleanup() {
  rm -f LICENSE README.md
  if [[ -f pyproject.toml.corex.bak ]]; then
    mv pyproject.toml.corex.bak pyproject.toml
  fi
}
trap cleanup EXIT

if [[ "${SGLANG_COREX_SKIP_RUST}" == "1" ]]; then
  cp pyproject.toml pyproject.toml.corex.bak
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
lines = path.read_text().splitlines()
out = []
skip_rust_section = False
for line in lines:
    stripped = line.strip()
    if stripped == '[[tool.setuptools-rust.ext-modules]]':
        skip_rust_section = True
        continue
    if skip_rust_section and stripped.startswith("["):
        skip_rust_section = False
    if skip_rust_section:
        continue
    if '"setuptools-rust' in line:
        line = line.replace('"setuptools-rust>=1.10", ', "")
        line = line.replace(', "setuptools-rust>=1.10"', "")
    out.append(line)
path.write_text("\n".join(out) + "\n")
PY
fi

"${PYTHON_BIN}" - <<'PY'
import os
import importlib.util
import sys

required = ["build", "setuptools", "setuptools_scm", "wheel"]
if os.environ.get("SGLANG_COREX_SKIP_RUST") != "1":
    required.append("setuptools_rust")
missing = [
    name
    for name in required
    if importlib.util.find_spec(name) is None
]
if missing:
    print(
        "ERROR: Missing build tools: "
        + ", ".join(missing)
        + ". Install them from the approved CoreX/private environment first.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

"${PYTHON_BIN}" -m build --wheel --no-isolation --outdir build_pip
