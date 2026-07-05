from __future__ import annotations

from typing import Any

from app.runtime.config import SETTINGS


def ready_detail(model_ref: str) -> dict[str, Any]:
    if SETTINGS.vision_backend == "vllm":
        from app.runtime.backends import vllm_engine
        return vllm_engine.ready_detail(model_ref, multimodal=True)
    return {"ok": False, "reason": f"unsupported_vision_backend:{SETTINGS.vision_backend}"}


def ready(model_ref: str) -> bool:
    return bool(ready_detail(model_ref).get("ok"))


def generate(*, name: str, model_ref: str, prompt: str, image_paths: list[str], max_new_tokens: int, temperature: float) -> str:
    if SETTINGS.vision_backend != "vllm":
        raise RuntimeError(f"unsupported_vision_backend:{SETTINGS.vision_backend}")
    from app.runtime.backends import vllm_engine
    return vllm_engine.generate_vision(
        name=name,
        model_ref=model_ref,
        prompt=prompt,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
