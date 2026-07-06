from __future__ import annotations

from app.runtime.config import SETTINGS
from app.runtime.worker_types import WorkerResult
from app.workers import gemma, vision


def run(original_prompt: str, image_paths: list[str], max_tokens: int) -> WorkerResult:
    draft = vision.answer(original_prompt, image_paths=image_paths, max_tokens=max_tokens)
    if not draft.ok:
        return draft

    if not SETTINGS.gemma_enabled:
        draft.metadata.update({"specialist": "vision", "handoff": "vision_only", "final_check": "skipped_gemma_disabled", "images": len(image_paths)})
        return draft

    checked = gemma.final_check(original_prompt, draft.content, max_tokens=min(384, max_tokens))
    checked.metadata.update({"specialist": "vision", "handoff": "gemma->vision->gemma", "images": len(image_paths)})
    return checked
