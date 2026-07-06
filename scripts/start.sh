#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
elif [ -f config/symoneural.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./config/symoneural.env
  set +a
fi

: "${SYMON_BIND_HOST:=0.0.0.0}"
: "${SYMON_BIND_PORT:=8000}"

mkdir -p data/runtime data/output data/uploads data/hf-cache
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "ERROR: missing .venv python. Run docs/INSTALL_CUDA_VLLM.md first."
  exit 1
fi

# Never write .pyc bytecode cache. This isn't for speed -- python startup cost
# is trivial next to the multi-second model load anyway -- it's so a fixed
# source file is *guaranteed* to actually run, with zero chance of Python
# silently executing stale compiled bytecode instead (this bit us for real
# while debugging earlier: a source-level fix looked applied but wasn't
# actually running, and stale __pycache__ was the reason).
export PYTHONDONTWRITEBYTECODE=1

# Auto-prune the HuggingFace download cache, but only when disk space is
# actually tight -- not on every launch. This cache holds tokenizer/config
# downloads and re-populates itself on next use, so clearing it is always
# safe, just wasteful to do more often than necessary. Override the
# threshold with SYMON_MIN_FREE_DISK_GB in .env if 5GB isn't right for you.
: "${SYMON_MIN_FREE_DISK_GB:=5}"
if command -v df >/dev/null 2>&1; then
  AVAIL_GB="$(df --output=avail -B G "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
  if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "$SYMON_MIN_FREE_DISK_GB" ]; then
    HF_SIZE="$(du -sh data/hf-cache 2>/dev/null | cut -f1)"
    echo "Low disk space (${AVAIL_GB}GB free, min is ${SYMON_MIN_FREE_DISK_GB}GB) -- clearing data/hf-cache (${HF_SIZE:-unknown size}) to free room."
    rm -rf data/hf-cache
    mkdir -p data/hf-cache
  fi
fi

PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  echo "SyMoNeuRaL already running pid=$PID"
  exit 0
fi

nohup "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
  --host "$SYMON_BIND_HOST" \
  --port "$SYMON_BIND_PORT" \
  > data/runtime/symon-manager.log 2>&1 &

echo "$!" > data/runtime/symon-manager.pid
echo "SyMoNeuRaL started pid=$(cat data/runtime/symon-manager.pid) on ${SYMON_BIND_HOST}:${SYMON_BIND_PORT}"
