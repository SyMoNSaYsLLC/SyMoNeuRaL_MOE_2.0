#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import vllm
    print('vllm', getattr(vllm, '__version__', 'present'))
except Exception as exc:
    print('vllm import failed', exc)
    raise
PY
