from __future__ import annotations

from app.runtime import flow_control, vision_engine
from app.runtime.config import MODELS, SETTINGS
from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "vision"


def ready_detail() -> dict:
    return vision_engine.ready_detail(MODELS.vision)


def ready() -> bool:
    return bool(ready_detail().get("ok"))


def answer(prompt: str, image_paths: list[str], max_tokens: int = 512) -> WorkerResult:
    if not image_paths:
        return WorkerResult(worker=NAME, content="No image was provided for the vision worker.", ok=False, error="missing_image")
    vram_guard.assert_safe_for_job(NAME)
    with flow_control.controller.job("vision"):
        content = vision_engine.generate(
            name=NAME,
            model_ref=MODELS.vision,
            prompt=prompt,
            image_paths=image_paths,
            max_new_tokens=max_tokens,
            temperature=SETTINGS.temperature_vision,
        )
    return WorkerResult(worker=NAME, content=content, metadata={"backend": SETTINGS.vision_backend, "images": len(image_paths)})
