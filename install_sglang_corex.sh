#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
TARGET_DIR="${TARGET_DIR:-}"
PKG_DIR="${SCRIPT_DIR}/python/build_pip"
ILU_REQUIREMENTS="${ILU_REQUIREMENTS:-${SCRIPT_DIR}/requirements/ilu.txt}"
INSTALL_PROJECT_DEPS="${INSTALL_PROJECT_DEPS:-1}"

install_wheel_reqs() {
  local pip_target_args=()
  if [[ -n "${TARGET_DIR}" ]]; then
    pip_target_args=(-t "${TARGET_DIR}/lib/python3/dist-packages" --upgrade)
  fi

  if [[ ! -f "${ILU_REQUIREMENTS}" ]]; then
    echo "INFO: ${ILU_REQUIREMENTS} not found; skip Iluvatar/CoreX bootstrap wheels."
    return
  fi

  while IFS= read -r req || [[ -n "${req}" ]]; do
    req="${req%%#*}"
    req="${req#"${req%%[![:space:]]*}"}"
    req="${req%"${req##*[![:space:]]}"}"
    [[ -z "${req}" ]] && continue
    echo "INFO: Installing CoreX bootstrap wheel: ${req}"
    "${PYTHON_BIN}" -m pip install \
      --only-binary=:all: \
      --no-deps \
      "${pip_target_args[@]}" \
      "${req}"
  done < "${ILU_REQUIREMENTS}"
}

install_project_deps() {
  [[ "${INSTALL_PROJECT_DEPS}" == "1" ]] || return

  local deps_file
  deps_file="$(mktemp)"
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/python/pyproject.toml" > "${deps_file}" <<'PY'
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)

for dep in data["project"].get("dependencies", []):
    print(dep)
PY

  local pip_target_args=()
  if [[ -n "${TARGET_DIR}" ]]; then
    pip_target_args=(-t "${TARGET_DIR}/lib/python3/dist-packages" --upgrade)
  fi

  while IFS= read -r req || [[ -n "${req}" ]]; do
    req="${req%%#*}"
    req="${req#"${req%%[![:space:]]*}"}"
    req="${req%"${req##*[![:space:]]}"}"
    [[ -z "${req}" ]] && continue
    echo "INFO: Installing SGLang pyproject dependency: ${req}"
    "${PYTHON_BIN}" -m pip install \
      --only-binary=:all: \
      --no-deps \
      "${pip_target_args[@]}" \
      "${req}"
  done < "${deps_file}"
  rm -f "${deps_file}"
}

install_wheel_reqs
install_project_deps

if [[ ! -d "${PKG_DIR}" ]]; then
  echo "ERROR: Package directory ${PKG_DIR} does not exist. Run build_sglang_corex.sh first."
  exit 1
fi

latest_pkg="$(ls -t "${PKG_DIR}"/sglang-*.whl 2>/dev/null | head -n 1)"
if [[ -z "${latest_pkg}" ]]; then
  echo "ERROR: Cannot find an sglang wheel in ${PKG_DIR}."
  exit 1
fi

if [[ -n "${TARGET_DIR}" ]]; then
  install_dir="${TARGET_DIR}/lib/python3/dist-packages"
  "${PYTHON_BIN}" -m pip install --upgrade --no-deps -t "${install_dir}" "${latest_pkg}"
  echo "sglang installed in ${install_dir}; add it to PYTHONPATH before launching."
else
  "${PYTHON_BIN}" -m pip uninstall sglang -y || true
  "${PYTHON_BIN}" -m pip install --no-deps "${latest_pkg}"
fi
