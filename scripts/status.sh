#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${SYMON_BIND_PORT:-8000}"

echo "===== process ====="
PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  echo "running pid=$PID"
else
  echo "not running"
fi

echo
echo "===== port ====="
ss -ltnp 2>/dev/null | grep ":$PORT" || echo "$PORT not listening"

echo
echo "===== gpu ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi unavailable"

echo
echo "===== static audit ====="
if [ -x .venv/bin/python ]; then
  .venv/bin/python production_audit.py --static-only
else
  python3 production_audit.py --static-only
fi
