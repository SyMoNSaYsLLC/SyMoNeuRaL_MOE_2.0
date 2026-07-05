#!/usr/bin/env python3
from __future__ import annotations

from huggingface_hub import snapshot_download

# Edit model IDs after accepting model licenses where required.
MODELS = {
    "models/llm/qwen": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "models/vlm/qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    # "models/llm/gemma": "google/gemma-2-2b-it",
}

for local_dir, repo_id in MODELS.items():
    print(f"downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
