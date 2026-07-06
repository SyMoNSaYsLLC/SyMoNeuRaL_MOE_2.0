from __future__ import annotations

from app.runtime import flow_control, kv_cache, text_engine
from app.runtime.config import MODELS, SETTINGS
from app.runtime.worker_types import WorkerDecision, WorkerResult
from app.workers import vram_guard

NAME = "gemma"
SYSTEM = (
    "You are SyMoNeuRaL Gemma, the manager/writer worker. "
    "Route only when useful, answer directly when the request is general, and verify specialist answers before final output."
)


def ready_detail() -> dict:
    return text_engine.ready_detail(MODELS.gemma)


def ready() -> bool:
    return bool(ready_detail().get("ok"))


def decide(prompt: str, has_images: bool = False) -> WorkerDecision:
    p = prompt.lower()
    if has_images:
        return WorkerDecision(worker="vision", task=prompt, kind="vision", reason="image_input")
    if any(x in p for x in ("draw", "image", "picture", "photo", "render", "generate a logo", "stable", "safetensor")):
        return WorkerDecision(worker="safetensor", task=prompt, kind="image", reason="image_generation_request")
    if any(x in p for x in ("code", "script", "python", "bash", "rust", "cpp", "c++", "debug", "repo", "bitbake", ".bb", "kernel", "cuda")):
        return WorkerDecision(worker="qwen", task=prompt, kind="code", reason="code_or_build_request")
    return WorkerDecision(worker="gemma", task=prompt, kind="chat", reason="general_chat")


def write(prompt: str, max_tokens: int = 768) -> WorkerResult:
    vram_guard.assert_safe_for_job(NAME)
    with flow_control.controller.job("text"):
        kv_cache.observe(prompt, NAME)
        content = text_engine.generate(
            name=NAME,
            model_ref=MODELS.gemma,
            system=SYSTEM,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=SETTINGS.temperature_chat,
        )
    return WorkerResult(worker=NAME, content=content, metadata={"backend": text_engine.backend_name()})


def final_check(original: str, draft: str, max_tokens: int = 384) -> WorkerResult:
    prompt = (
        "Review the specialist answer for correctness, safety, and directness. "
        "Return the final answer only.\n\n"
        f"Original request:\n{original}\n\nSpecialist draft:\n{draft}"
    )
    try:
        return write(prompt, max_tokens=max_tokens)
    except Exception:
        # Fail useful, not silent. If the specialist has a valid result but Gemma reload/check fails, return it with metadata.
        return WorkerResult(worker=NAME, content=draft, metadata={"final_check": "skipped_due_to_gemma_unavailable"})
