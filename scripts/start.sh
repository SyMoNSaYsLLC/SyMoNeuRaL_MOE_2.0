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
