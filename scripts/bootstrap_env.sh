#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${SYMON_REQUIREMENTS_FILE:-requirements.txt}"
INSTALL_VLLM="${SYMON_INSTALL_VLLM:-0}"
VLLM_REQ_FILE="${SYMON_VLLM_REQUIREMENTS_FILE:-requirements.vllm.txt}"

echo "===== SyMoNeuRaL bootstrap ====="
echo "Repo: $ROOT"

echo
echo "===== verify repo root ====="
test -f README.md
test -f "$REQ_FILE"
test -d app
test -d scripts

echo
echo "===== python ====="
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PY_BIN not found"
  echo "Run with:"
  echo "  PYTHON_BIN=/path/to/python scripts/bootstrap_env.sh"
  exit 1
fi

"$PY_BIN" --version

echo
echo "===== create/reuse .venv ====="
if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
  echo "created .venv"
else
  echo "reusing .venv"
fi

source .venv/bin/activate

echo
echo "===== upgrade pip/setuptools/wheel ====="
python -m pip install --upgrade pip "setuptools<82" wheel

echo
echo "===== install $REQ_FILE ====="
python -m pip install -r "$REQ_FILE"

if [ "$INSTALL_VLLM" = "1" ]; then
  echo
  echo "===== install optional vLLM profile ($VLLM_REQ_FILE) ====="
  if [ ! -f "$VLLM_REQ_FILE" ]; then
    echo "ERROR: optional vLLM requirements file not found: $VLLM_REQ_FILE"
    exit 1
  fi
  python -m pip install -r "$VLLM_REQ_FILE"
fi

echo
echo "===== pip check ====="
python -m pip check

echo
echo "===== verify app imports ====="
python - <<'PY'
import importlib

mods = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "anyio",
    "app.main",
    "app.symoneural",
]

failed = False

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"PASS import {mod}")
    except Exception as exc:
        failed = True
        print(f"FAIL import {mod}: {exc!r}")

if failed:
    raise SystemExit(1)
PY

echo
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "===== production audit ====="
  python production_audit.py
else
  echo "===== production audit (skipped: vllm not installed) ====="
  echo "Install optional profile with:"
  echo "  SYMON_INSTALL_VLLM=1 scripts/bootstrap_env.sh"
fi

echo
echo "===== bootstrap complete ====="
echo "Next:"
echo "  source .venv/bin/activate"
echo "  SYMON_INSTALL_VLLM=1 scripts/bootstrap_env.sh   # optional"
echo "  python -m app.symon_cli --help"
echo "  scripts/check_models_magic.sh"
echo "  python3 production_audit.py --live"
echo "  scripts/start.sh"
echo "  scripts/test.sh"
