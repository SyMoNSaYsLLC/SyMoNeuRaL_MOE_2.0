from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.runtime.config import OUTPUT, SETTINGS, resolve_local_path


@dataclass
class LoadedImageModel:
    path: Path
    pipe: Any


# A1111-familiar sampler names mapped to their diffusers scheduler classes.
# Kept deliberately small -- these are the handful that make a genuinely
# noticeable difference and that most people already have an opinion about.
_SCHEDULER_MAP = {
    "euler_a": "EulerAncestralDiscreteScheduler",
    "euler": "EulerDiscreteScheduler",
    "dpm++_2m": "DPMSolverMultistepScheduler",
    "dpm++_2m_karras": "DPMSolverMultistepScheduler",  # + use_karras_sigmas below
    "ddim": "DDIMScheduler",
    "unipc": "UniPCMultistepScheduler",
}


def _apply_scheduler(pipe: Any, name: str) -> None:
    name = (name or "default").strip().lower()
    if name in ("", "default"):
        return
    class_name = _SCHEDULER_MAP.get(name)
    if not class_name:
        print(f"[image_engine] Unknown SYMON_IMAGE_SCHEDULER '{name}', leaving checkpoint default in place. "
              f"Recognized: {', '.join(_SCHEDULER_MAP)}")
        return
    try:
        import diffusers
        scheduler_cls = getattr(diffusers, class_name)
        config = dict(pipe.scheduler.config)
        if name == "dpm++_2m_karras":
            config["use_karras_sigmas"] = True
        pipe.scheduler = scheduler_cls.from_config(config)
    except Exception as exc:
        print(f"[image_engine] Failed to apply scheduler '{name}': {exc!r}. Leaving checkpoint default in place.")


_LOCK = threading.RLock()
_CACHE: LoadedImageModel | None = None


def dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {"backend": SETTINGS.image_backend}
    try:
        import torch
        status["torch"] = getattr(torch, "__version__", "present")
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        status["torch_error"] = str(exc)
    try:
        import diffusers
        status["diffusers"] = getattr(diffusers, "__version__", "present")
    except Exception as exc:
        status["diffusers_error"] = str(exc)
    try:
        import PIL
        status["pillow"] = getattr(PIL, "__version__", "present")
    except Exception as exc:
        status["pillow_error"] = str(exc)
    return status


def deps_ready() -> tuple[bool, str]:
    d = dependency_status()
    for key in ("torch_error", "diffusers_error", "pillow_error"):
        if key in d:
            return False, f"{key}:{d[key]}"
    if SETTINGS.device == "cuda" and not d.get("cuda_available"):
        return False, "cuda_unavailable"
    return True, "ok"


def model_path_ready(model_path: Path | str) -> tuple[bool, str]:
    model_path = resolve_local_path(str(model_path))
    if not model_path.exists():
        return False, f"model_path_missing:{model_path}"
    if model_path.is_file():
        if model_path.suffix.lower() != ".safetensors":
            return False, f"image_checkpoint_not_safetensors:{model_path}"
        return True, "ok"
    if not (model_path / "model_index.json").exists():
        return False, f"missing_diffusers_model_index:{model_path}"
    return True, "ok"


def ready_detail(model_path: Path | str) -> dict[str, Any]:
    model_path = resolve_local_path(str(model_path))
    path_ok, path_reason = model_path_ready(model_path)
    deps_ok, deps_reason = deps_ready()
    return {
        "ok": bool(path_ok and deps_ok),
        "backend": SETTINGS.image_backend,
        "path": str(model_path),
        "path_ok": path_ok,
        "path_reason": path_reason,
        "deps_ok": deps_ok,
        "deps_reason": deps_reason,
        "dependencies": dependency_status(),
        "device": SETTINGS.device,
        "settings": {
            "width": SETTINGS.image_width,
            "height": SETTINGS.image_height,
            "steps": SETTINGS.image_steps,
            "guidance": SETTINGS.image_guidance,
        },
    }


def ready(model_path: Path | str) -> bool:
    return bool(ready_detail(model_path).get("ok"))


def load(model_path: Path | str) -> LoadedImageModel:
    if SETTINGS.image_backend != "diffusers":
        raise RuntimeError(f"unsupported_image_backend:{SETTINGS.image_backend}")
    model_path = resolve_local_path(str(model_path))
    detail = ready_detail(model_path)
    if not detail["ok"]:
        raise RuntimeError(f"image_engine_not_ready:{detail}")
    global _CACHE
    with _LOCK:
        if _CACHE is not None and _CACHE.path == model_path:
            return _CACHE
        unload()
        import torch
        dtype = torch.float16 if SETTINGS.device == "cuda" and torch.cuda.is_available() else torch.float32
        if model_path.is_file():
            # Single-file .safetensors checkpoints are a legacy SD1.x/SDXL-merge
            # format; StableDiffusionPipeline's from_single_file loader handles
            # both. Newer model families (SDXL-as-directory, and anything
            # trained/exported later) ship as a proper diffusers directory and
            # go through the generic auto-detecting loader below instead.
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_single_file(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            # DiffusionPipeline auto-selects the correct pipeline subclass by
            # reading model_index.json's _class_name -- this is what lets this
            # server swap between SD1.x, SDXL, or a future custom-trained model
            # dropped into the same directory shape, with no code change here.
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
            for unsafe_attr in ("safety_checker", "requires_safety_checker"):
                if hasattr(pipe, unsafe_attr):
                    try:
                        setattr(pipe, unsafe_attr, None if unsafe_attr == "safety_checker" else False)
                    except Exception:
                        pass
        if SETTINGS.device == "cuda" and torch.cuda.is_available():
            pipe = pipe.to("cuda")
            for feature in ("enable_attention_slicing", "enable_vae_tiling", "enable_model_cpu_offload"):
                try:
                    getattr(pipe, feature)()
                except Exception:
                    pass
        else:
            pipe = pipe.to("cpu")

        _apply_scheduler(pipe, SETTINGS.image_scheduler)
        _CACHE = LoadedImageModel(path=model_path, pipe=pipe)
        return _CACHE


def unload() -> None:
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            try:
                del _CACHE.pipe
            except Exception:
                pass
            _CACHE = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def generate_image(*, model_path: Path | str, prompt: str, negative_prompt: str | None = None, width: int | None = None, height: int | None = None, steps: int | None = None, guidance_scale: float | None = None, seed: int | None = None) -> tuple[Path, int]:
    loaded = load(model_path)

    effective_seed = seed if seed is not None else SETTINGS.image_seed
    candidate_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt if negative_prompt is not None else SETTINGS.image_negative_prompt,
        "width": width or SETTINGS.image_width,
        "height": height or SETTINGS.image_height,
        "num_inference_steps": steps or SETTINGS.image_steps,
        "guidance_scale": guidance_scale or SETTINGS.image_guidance,
    }

    if effective_seed is not None and effective_seed >= 0:
        import torch
        device = "cuda" if (SETTINGS.device == "cuda" and torch.cuda.is_available()) else "cpu"
        candidate_kwargs["generator"] = torch.Generator(device=device).manual_seed(int(effective_seed))
    else:
        # -1 / unset: pick a real random seed ourselves (instead of leaving it
        # to the pipeline's own RNG) so we can still report which seed was
        # actually used -- needed for "regenerate this exact image later".
        import random
        import torch
        effective_seed = random.randint(0, 2**32 - 1)
        device = "cuda" if (SETTINGS.device == "cuda" and torch.cuda.is_available()) else "cpu"
        candidate_kwargs["generator"] = torch.Generator(device=device).manual_seed(effective_seed)

    import inspect
    try:
        accepted = set(inspect.signature(loaded.pipe.__call__).parameters)
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        kwargs = candidate_kwargs

    result = loaded.pipe(**kwargs)
    image = result.images[0]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    next_id = len(list(OUTPUT.glob("symon_image_*.png"))) + 1
    path = OUTPUT / f"symon_image_{next_id:06d}.png"
    image.save(path)
    return path, effective_seed
