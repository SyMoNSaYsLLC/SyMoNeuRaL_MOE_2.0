#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_BIN="${PYTHON_BIN:-python3}"

echo "===== SyMoNeuRaL bootstrap ====="
echo "Repo: $ROOT"

echo
echo "===== verify repo root ====="
test -f README.md
test -f requirements.txt
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
python -m pip install --upgrade pip setuptools wheel

echo
echo "===== install requirements.txt ====="
python -m pip install -r requirements.txt

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
echo "===== production audit ====="
python production_audit.py

echo
echo "===== bootstrap complete ====="
echo "Next:"
echo "  source .venv/bin/activate"
echo "  scripts/check_models_magic.sh"
echo "  python3 production_audit.py --live"
echo "  scripts/start.sh"
echo "  scripts/test.sh"
