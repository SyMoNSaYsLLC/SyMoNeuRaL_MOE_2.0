from __future__ import annotations

from app.runtime.worker_types import WorkerResult
from app.workers import gemma, qwen


def run(original_prompt: str, manager_task: str, max_tokens: int) -> WorkerResult:
    draft = qwen.code(manager_task, max_tokens=max_tokens)
    if not draft.ok:
        return draft
    checked = gemma.final_check(original_prompt, draft.content, max_tokens=min(512, max_tokens))
    checked.metadata.update({"specialist": "qwen", "handoff": "gemma->qwen->gemma"})
    return checked
