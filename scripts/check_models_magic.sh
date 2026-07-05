#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "===== SyMoNeuRaL model magic check ====="
echo "Repo: $ROOT"

GEMMA_MODEL="${SYMON_GEMMA_MODEL:-models/llm/gemma-4-12b-it-Q4_K_M.gguf}"
QWEN_MODEL="${SYMON_QWEN_MODEL:-models/llm/qwen2.5-coder-1.5b-instruct-q8_0.gguf}"
STABLE_MODEL="${SYMON_STABLE_MODEL:-models/sd/v1-5-pruned-emaonly.safetensors}"

python3 - "$GEMMA_MODEL" "$QWEN_MODEL" "$STABLE_MODEL" <<'PY'
from pathlib import Path
import json
import struct
import sys

checks = {
    "gemma": Path(sys.argv[1]),
    "qwen": Path(sys.argv[2]),
    "stable": Path(sys.argv[3]),
}

failed = False

def fail(msg: str):
    global failed
    failed = True
    print(f"FAIL {msg}")

def ok(msg: str):
    print(f"PASS {msg}")

for name, path in checks.items():
    print()
    print(f"===== {name} =====")
    print(f"path: {path}")

    if not path.exists():
        fail(f"{name}: missing file")
        continue

    if path.is_symlink():
        fail(f"{name}: is symlink, expected real file")
        continue

    if not path.is_file():
        fail(f"{name}: not a regular file")
        continue

    size = path.stat().st_size
    print(f"size_bytes: {size}")
    print(f"size_gb: {size / 1024**3:.2f}")

    if size < 1024 * 1024:
        fail(f"{name}: file is suspiciously small")
        continue

    with path.open("rb") as f:
        head = f.read(16)

    if path.suffix.lower() == ".gguf":
        magic = head[:4]
        print(f"magic: {magic!r}")

        if magic != b"GGUF":
            fail(f"{name}: expected GGUF magic, got {magic!r}")
            continue

        version = struct.unpack("<I", head[4:8])[0]
        print(f"gguf_version: {version}")

        if version <= 0 or version > 10:
            fail(f"{name}: suspicious GGUF version {version}")
            continue

        ok(f"{name}: valid GGUF header")

    elif path.suffix.lower() == ".safetensors":
        if len(head) < 8:
            fail(f"{name}: too small for safetensors header")
            continue

        header_len = struct.unpack("<Q", head[:8])[0]
        print(f"safetensors_header_len: {header_len}")

        if header_len <= 0 or header_len > 100_000_000:
            fail(f"{name}: invalid safetensors header length {header_len}")
            continue

        with path.open("rb") as f:
            f.seek(8)
            raw_header = f.read(header_len)

        try:
            header = json.loads(raw_header)
        except Exception as exc:
            fail(f"{name}: safetensors header is not valid JSON: {exc!r}")
            continue

        tensor_names = [k for k in header.keys() if k != "__metadata__"]
        print(f"tensor_count: {len(tensor_names)}")

        if not tensor_names:
            fail(f"{name}: no tensors found in safetensors header")
            continue

        ok(f"{name}: valid safetensors header")

    else:
        fail(f"{name}: unsupported extension {path.suffix}")

print()
if failed:
    print("FAIL model magic check")
    raise SystemExit(1)

print("PASS model magic check")
PY
