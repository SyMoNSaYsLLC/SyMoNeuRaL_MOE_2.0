from __future__ import annotations

from app.runtime.worker_types import WorkerResult
from app.workers import gemma, safetensor


def run(original_prompt: str, manager_task: str) -> WorkerResult:
    result = safetensor.image(manager_task or original_prompt)
    if not result.ok:
        return result
    # Do not have Gemma rewrite image markdown; preserve generated file path/url.
    result.metadata.update({"handoff": "gemma->safetensor", "manager": "gemma"})
    return result
