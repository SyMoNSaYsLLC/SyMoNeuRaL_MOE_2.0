from __future__ import annotations

from app.runtime import flow_control, kv_cache, text_engine
from app.runtime.config import MODELS, SETTINGS
from app.runtime.worker_types import WorkerResult
from app.workers import vram_guard

NAME = "qwen"
SYSTEM = (
    "You are SyMoNeuRaL Qwen Coder. Produce exact, runnable code and system commands. "
    "Call out destructive commands and production risks clearly."
)
CHAT_SYSTEM = (
    "You are SyMoNeuRaL, a helpful assistant. Answer directly and conversationally. "
    "If a request needs code, scripts, or debugging, switch into precise, runnable-code mode."
)


def ready_detail() -> dict:
    return text_engine.ready_detail(MODELS.qwen)


def ready() -> bool:
    return bool(ready_detail().get("ok"))


def code(task: str, max_tokens: int = 1024) -> WorkerResult:
    vram_guard.assert_safe_for_job(NAME)
    with flow_control.controller.job("text"):
        kv_cache.observe(task, NAME)
        content = text_engine.generate(
            name=NAME,
            model_ref=MODELS.qwen,
            system=SYSTEM,
            prompt=task,
            max_new_tokens=max_tokens,
            temperature=SETTINGS.temperature_code,
        )
    return WorkerResult(worker=NAME, content=content, metadata={"backend": text_engine.backend_name()})


def chat(task: str, max_tokens: int = 768) -> WorkerResult:
    vram_guard.assert_safe_for_job(NAME)
    with flow_control.controller.job("text"):
        kv_cache.observe(task, NAME)
        content = text_engine.generate(
            name=NAME,
            model_ref=MODELS.qwen,
            system=CHAT_SYSTEM,
            prompt=task,
            max_new_tokens=max_tokens,
            temperature=SETTINGS.temperature_chat,
        )
    return WorkerResult(worker=NAME, content=content, metadata={"backend": text_engine.backend_name()})
