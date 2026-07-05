from __future__ import annotations

from pathlib import Path
from typing import Any

from app.runtime.config import MODELS, SETTINGS, resolve_local_path

_MODEL: Any | None = None
_PROCESSOR: Any | None = None


def dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {"backend": "transformers_vit"}
    try:
        import torch
        status["torch"] = getattr(torch, "__version__", "present")
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        status["torch_error"] = str(exc)
    try:
        import transformers
        status["transformers"] = getattr(transformers, "__version__", "present")
    except Exception as exc:
        status["transformers_error"] = str(exc)
    return status


def ready_detail(model_ref: str | None = None) -> dict[str, Any]:
    model_ref = model_ref or MODELS.vit_encoder
    path = resolve_local_path(model_ref)
    deps = dependency_status()
    deps_ok = "torch_error" not in deps and "transformers_error" not in deps
    path_ok = path.exists() and path.is_dir() and (path / "config.json").exists()
    if SETTINGS.allow_remote_models and "/" in model_ref and not path.exists():
        path_ok = True
    return {"ok": deps_ok and path_ok, "path": str(path), "model_ref": model_ref, "deps": deps, "path_ok": path_ok}


def encode_image(image_path: str, model_ref: str | None = None) -> list[float]:
    """Optional lightweight ViT/CLIP embedding path.

    This is not the primary VLM chat path. The flagship multimodal path is vLLM VLM.
    This helper exists for future routing, image similarity, and prefix/semantic routing.
    """
    global _MODEL, _PROCESSOR
    model_ref = model_ref or MODELS.vit_encoder
    detail = ready_detail(model_ref)
    if not detail["ok"]:
        raise RuntimeError(f"vit_encoder_not_ready:{detail}")
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    if _MODEL is None or _PROCESSOR is None:
        resolved = str(resolve_local_path(model_ref)) if resolve_local_path(model_ref).exists() else model_ref
        _PROCESSOR = AutoProcessor.from_pretrained(resolved, local_files_only=not SETTINGS.allow_remote_models, trust_remote_code=True)
        _MODEL = AutoModel.from_pretrained(resolved, local_files_only=not SETTINGS.allow_remote_models, trust_remote_code=True)
        if SETTINGS.device == "cuda" and torch.cuda.is_available():
            _MODEL = _MODEL.to("cuda")
    image = Image.open(Path(image_path).expanduser()).convert("RGB")
    inputs = _PROCESSOR(images=image, return_tensors="pt")
    if SETTINGS.device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state.mean(dim=1)
    return pooled.detach().float().cpu().numpy()[0].tolist()
