from __future__ import annotations

from app.comms import gemmacodercomms, gemmaimagingcomms, gemmavisioncomms
from app.runtime.config import SETTINGS
from app.runtime.worker_types import RequestContext
from app.workers import gemma, qwen, safetensor, vision


def answer(
    prompt: str,
    max_tokens: int = 768,
    images: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str, dict]:
    image_tuple = tuple(images or ())
    ctx = RequestContext(prompt=prompt, max_tokens=max_tokens, images=image_tuple)
    decision = gemma.decide(ctx.prompt, has_images=bool(ctx.images))

    if decision.worker == "qwen":
        result = gemmacodercomms.run(ctx.prompt, decision.task, ctx.max_tokens)
    elif decision.worker == "safetensor":
        result = gemmaimagingcomms.run(ctx.prompt, decision.task)
    elif decision.worker == "vision":
        result = gemmavisioncomms.run(ctx.prompt, list(ctx.images), ctx.max_tokens)
    else:
        result = gemma.write(decision.task, ctx.max_tokens)

    metadata = result.metadata.copy()
    metadata.update(
        {
            "decision": decision.__dict__,
            "ok": result.ok,
            "error": result.error,
        }
    )
    return result.worker, result.content, metadata


def worker_readiness() -> dict[str, bool]:
    workers = {
        "gemma": gemma.ready(),
        "qwen": qwen.ready(),
        "safetensor": safetensor.ready(),
        "vision": vision.ready(),
    }

    if not SETTINGS.require_all_workers:
        return {
            "gemma": workers["gemma"],
            "qwen": workers["qwen"],
            "safetensor": workers["safetensor"],
            "vision": True,
        }

    return workers


def worker_status() -> dict:
    return {
        "gemma": gemma.ready_detail(),
        "qwen": qwen.ready_detail(),
        "safetensor": safetensor.ready_detail(),
        "vision": vision.ready_detail(),
    }
