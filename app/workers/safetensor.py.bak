from __future__ import annotations

from app.runtime import flow_control, image_engine, text_engine
from app.runtime.config import MODELS, SETTINGS
from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "safetensor"


def ready_detail() -> dict:
    return image_engine.ready_detail(MODELS.stable)


def ready() -> bool:
    return bool(ready_detail().get("ok"))


def image(prompt: str) -> WorkerResult:
    vram_guard.assert_safe_for_job(NAME)
    if SETTINGS.unload_text_before_image:
        text_engine.unload()
    with flow_control.controller.job("image"):
        path = image_engine.generate_image(model_path=MODELS.stable, prompt=prompt)
    rel = path.relative_to(image_engine.OUTPUT.parent) if hasattr(image_engine, "OUTPUT") else path.name
    content = f"Generated image.\n\n![generated](/outputs/{path.name})\n\nFile: `{path}`"
    return WorkerResult(worker=NAME, content=content, metadata={"path": str(path), "url": f"/outputs/{path.name}"})
