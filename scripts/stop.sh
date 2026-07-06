#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PID="$(cat data/runtime/symon-manager.pid 2>/dev/null || true)"

if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  # Kill descendants first (uvicorn's own worker/reload children if any),
  # then the parent itself. SIGTERM first, then SIGKILL if it doesn't die
  # promptly -- uvicorn/vLLM shutdown can be slow but shouldn't hang forever.
  CHILDREN="$(pgrep -P "$PID" 2>/dev/null || true)"
  if [ -n "$CHILDREN" ]; then
    kill $CHILDREN 2>/dev/null || true
  fi
  kill "$PID" 2>/dev/null || true

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$PID" 2>/dev/null; then
    echo "pid=$PID did not exit after SIGTERM, sending SIGKILL"
    kill -9 "$PID" 2>/dev/null || true
    [ -n "$CHILDREN" ] && kill -9 $CHILDREN 2>/dev/null || true
  fi

  echo "stopped pid=$PID"
else
  echo "not running"
fi

rm -f data/runtime/symon-manager.pid

# vLLM's V1 engine runs its actual model in a SEPARATE child OS process
# (visible as "VLLM::EngineCore" in ps/nvidia-smi). Killing only the parent
# uvicorn PID above does not guarantee these get reaped -- if the parent
# dies abruptly (crash, earlier bug, manual kill) they can be orphaned and
# keep holding GPU memory indefinitely, surviving every future start/stop
# cycle until killed explicitly. Sweep for and clear any stragglers here so
# `stop.sh` actually leaves the GPU clean, regardless of parent linkage.
STRAY_PIDS="$(pgrep -f 'VLLM::EngineCore' 2>/dev/null || true)"
if [ -n "$STRAY_PIDS" ]; then
  echo "found stray VLLM::EngineCore process(es), cleaning up: $STRAY_PIDS"
  kill -9 $STRAY_PIDS 2>/dev/null || true
fi

STRAY_TRACKERS="$(pgrep -f 'multiprocessing.resource_tracker' 2>/dev/null || true)"
if [ -n "$STRAY_TRACKERS" ]; then
  kill -9 $STRAY_TRACKERS 2>/dev/null || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  REMAINING="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$REMAINING" ]; then
    echo "WARNING: GPU still reports active compute processes after stop:"
    echo "$REMAINING"
    echo "Check manually with: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
  fi
fi
