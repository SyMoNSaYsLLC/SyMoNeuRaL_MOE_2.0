#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${SYMON_BIND_PORT:-8000}"
python3 - <<PY
import json, urllib.request
base = 'http://127.0.0.1:${PORT}'
print('GET /health')
print(urllib.request.urlopen(base + '/health', timeout=10).read().decode()[:1000])
body = json.dumps({
    'model': 'symoneural_moe',
    'messages': [{'role': 'user', 'content': 'hello from production test'}],
    'max_tokens': 64,
}).encode()
req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
print('POST /v1/chat/completions')
print(urllib.request.urlopen(req, timeout=120).read().decode()[:2000])
PY
