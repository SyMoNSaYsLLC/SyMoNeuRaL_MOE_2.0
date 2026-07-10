#!/usr/bin/env bash
# Clears the two caches that tend to silently accumulate on this project:
#   1. data/hf-cache -- HuggingFace's own download cache (tokenizer/config
#      lookups, and any repo-id-based downloads). Safe to clear: anything
#      needed again just re-downloads on next use. Does NOT touch your
#      actual model weights under models/llm/ or models/sd/ -- those are
#      separate, plain local directories, untouched by this script.
#   2. __pycache__ directories -- stale compiled bytecode can occasionally
#      cause Python to run old code even after you've fixed a source file
#      (we hit exactly this while debugging earlier). Clearing forces a
#      fresh compile on next run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== SyMoNeuRaL cache inspector ====="
echo "Repo: $ROOT"
echo

if [ -d data/hf-cache ]; then
  HF_SIZE="$(du -sh data/hf-cache 2>/dev/null | cut -f1)"
  echo "HuggingFace cache (data/hf-cache): ${HF_SIZE:-unknown size}"
else
  echo "HuggingFace cache (data/hf-cache): not present"
fi

PYCACHE_COUNT="$(find app tools -name '__pycache__' -type d 2>/dev/null | wc -l | tr -d ' ')"
echo "Stale __pycache__ directories under app/ and tools/: $PYCACHE_COUNT"
echo

read -r -p "Clear __pycache__ directories now? [y/N] " clear_pycache
if [[ "$clear_pycache" =~ ^[Yy]$ ]]; then
  find app tools -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
  echo "Cleared."
else
  echo "Skipped."
fi
echo

read -r -p "Clear the HuggingFace download cache (data/hf-cache)? Anything needed again will just re-download. [y/N] " clear_hf
if [[ "$clear_hf" =~ ^[Yy]$ ]]; then
  rm -rf data/hf-cache
  mkdir -p data/hf-cache
  echo "Cleared."
else
  echo "Skipped."
fi

echo
echo "===== done ====="
echo "If the server is currently running, restart it to pick up any code changes:"
echo "  scripts/stop.sh && scripts/start.sh"
