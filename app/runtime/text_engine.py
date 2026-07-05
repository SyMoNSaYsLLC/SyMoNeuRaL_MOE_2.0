from __future__ import annotations

from typing import Any

from app.runtime.config import SETTINGS


def backend_name() -> str:
    return SETTINGS.text_backend


def dependency_status() -> dict[str, Any]:
    if SETTINGS.text_backend == "vllm":
        from app.runtime.backends import vllm_engine
        return vllm_engine.dependency_status()
    return {"backend": SETTINGS.text_backend, "error": "unsupported_text_backend"}


def ready_detail(model_ref: str) -> dict[str, Any]:
    if SETTINGS.text_backend == "vllm":
        from app.runtime.backends import vllm_engine
        return vllm_engine.ready_detail(model_ref)
    return {"ok": False, "reason": f"unsupported_text_backend:{SETTINGS.text_backend}"}


def ready(model_ref: str) -> bool:
    return bool(ready_detail(model_ref).get("ok"))


def unload(name: str | None = None) -> None:
    if SETTINGS.text_backend == "vllm":
        from app.runtime.backends import vllm_engine
        vllm_engine.unload(name)


def generate(
    *,
    name: str,
    model_ref: str,
    system: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    if SETTINGS.text_backend != "vllm":
        raise RuntimeError(f"unsupported_text_backend:{SETTINGS.text_backend}")
    from app.runtime.backends import vllm_engine
    return vllm_engine.generate(
        name=name,
        model_ref=model_ref,
        system=system,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
