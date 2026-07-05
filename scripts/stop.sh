#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped pid=$PID"
else
  echo "not running"
fi
rm -f data/runtime/symon-manager.pid
