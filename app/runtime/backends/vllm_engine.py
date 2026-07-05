from __future__ import annotations

import gc
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.runtime.config import HF_CACHE, SETTINGS, looks_like_hf_repo_id, resolve_local_path
from app.runtime import kv_cache


@dataclass
class LoadedModel:
    name: str
    model_ref: str
    llm: Any
    is_multimodal: bool = False


_LOCK = threading.RLock()
_ACTIVE: LoadedModel | None = None


def dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {"backend": "vllm"}
    try:
        import torch
        status["torch"] = getattr(torch, "__version__", "present")
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            status["cuda_device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            status["cuda_compute_capability"] = f"{cap[0]}.{cap[1]}"
    except Exception as exc:
        status["torch_error"] = str(exc)
    try:
        import vllm
        status["vllm"] = getattr(vllm, "__version__", "present")
    except Exception as exc:
        status["vllm_error"] = str(exc)
    return status


def deps_ready() -> tuple[bool, str]:
    d = dependency_status()
    for key in ("torch_error", "vllm_error"):
        if key in d:
            return False, f"{key}:{d[key]}"
    if SETTINGS.device == "cuda" and not d.get("cuda_available"):
        return False, "cuda_unavailable"
    return True, "ok"


def model_ref_ready(model_ref: str, *, multimodal: bool = False) -> tuple[bool, str, str]:
    ref = str(model_ref).strip()
    if not ref:
        return False, "empty_model_ref", ref
    local = resolve_local_path(ref)
    if local.exists():
        if not local.is_dir():
            return False, f"model_must_be_hf_directory_for_vllm:{local}", str(local)
        if not (local / "config.json").exists():
            return False, f"missing_config_json:{local}", str(local)
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        if not any((local / item).exists() for item in tokenizer_files):
            return False, f"missing_tokenizer_files:{local}", str(local)
        weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        has_weights = any(any(local.glob(pattern)) for pattern in weight_patterns)
        has_index = any((local / name).exists() for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"))
        if not has_weights and not has_index:
            return False, f"missing_model_weights:{local}", str(local)
        return True, "ok", str(local)
    if SETTINGS.allow_remote_models and looks_like_hf_repo_id(ref):
        return True, "remote_hf_model_allowed", ref
    return False, f"model_path_missing:{local}", str(local)


def ready_detail(model_ref: str, *, multimodal: bool = False) -> dict[str, Any]:
    ref_ok, ref_reason, resolved = model_ref_ready(model_ref, multimodal=multimodal)
    deps_ok, deps_reason = deps_ready()
    return {
        "ok": bool(ref_ok and deps_ok),
        "backend": "vllm",
        "model_ref": str(model_ref),
        "resolved_model": resolved,
        "model_ok": ref_ok,
        "model_reason": ref_reason,
        "deps_ok": deps_ok,
        "deps_reason": deps_reason,
        "dependencies": dependency_status(),
        "device": SETTINGS.device,
        "allow_remote_models": SETTINGS.allow_remote_models,
        "multimodal": multimodal,
        "vllm": {
            "dtype": SETTINGS.vllm_dtype,
            "gpu_memory_utilization": SETTINGS.vllm_gpu_memory_utilization,
            "max_model_len": SETTINGS.vllm_max_model_len,
            "cpu_offload_gb": SETTINGS.vllm_cpu_offload_gb,
            "swap_space_gb": SETTINGS.vllm_swap_space_gb,
            "quantization": SETTINGS.vllm_quantization or None,
            "enforce_eager": SETTINGS.vllm_enforce_eager,
            "enable_prefix_caching": SETTINGS.vllm_enable_prefix_caching,
            "max_num_seqs": SETTINGS.vllm_max_num_seqs,
            "max_num_batched_tokens": SETTINGS.vllm_max_num_batched_tokens,
        },
    }


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def unload(name: str | None = None) -> None:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is None:
            _cleanup_cuda()
            return
        if name is not None and _ACTIVE.name != name:
            return
        try:
            try:
                from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
            del _ACTIVE.llm
        finally:
            _ACTIVE = None
            _cleanup_cuda()


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def _llm_kwargs(resolved_model: str, *, multimodal: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "trust_remote_code": SETTINGS.vllm_trust_remote_code,
        "tensor_parallel_size": 1,
        "dtype": SETTINGS.vllm_dtype,
        "gpu_memory_utilization": SETTINGS.vllm_gpu_memory_utilization,
        "max_model_len": SETTINGS.vllm_max_model_len,
        "cpu_offload_gb": SETTINGS.vllm_cpu_offload_gb,
        "swap_space": SETTINGS.vllm_swap_space_gb,
        "enforce_eager": SETTINGS.vllm_enforce_eager,
        "seed": SETTINGS.vllm_seed,
        "download_dir": str(HF_CACHE),
        "disable_log_stats": SETTINGS.vllm_disable_log_stats,
        "enable_prefix_caching": SETTINGS.vllm_enable_prefix_caching,
        "max_num_seqs": SETTINGS.vllm_max_num_seqs,
        "max_num_batched_tokens": SETTINGS.vllm_max_num_batched_tokens,
    }
    if SETTINGS.vllm_quantization:
        kwargs["quantization"] = SETTINGS.vllm_quantization
    if multimodal:
        kwargs["limit_mm_per_prompt"] = {"image": SETTINGS.vllm_limit_mm_images}
    return kwargs


def load(name: str, model_ref: str, *, multimodal: bool = False) -> LoadedModel:
    global _ACTIVE
    detail = ready_detail(model_ref, multimodal=multimodal)
    if not detail["ok"]:
        raise RuntimeError(f"{name}_vllm_not_ready:{detail}")
    resolved_model = detail["resolved_model"]
    with _LOCK:
        if _ACTIVE is not None and _ACTIVE.name == name and _ACTIVE.model_ref == resolved_model and _ACTIVE.is_multimodal == multimodal:
            return _ACTIVE
        if _ACTIVE is not None and SETTINGS.single_active_model:
            unload()
        try:
            from app.runtime import image_engine
            image_engine.unload()
        except Exception:
            pass
        from vllm import LLM
        kwargs = _filter_kwargs(LLM, _llm_kwargs(resolved_model, multimodal=multimodal))
        llm = LLM(**kwargs)
        _ACTIVE = LoadedModel(name=name, model_ref=resolved_model, llm=llm, is_multimodal=multimodal)
        return _ACTIVE


def _sampling(max_new_tokens: int, temperature: float) -> Any:
    from vllm import SamplingParams
    return SamplingParams(
        max_tokens=max(1, int(max_new_tokens)),
        temperature=max(0.0, float(temperature)),
        top_p=SETTINGS.top_p,
        repetition_penalty=SETTINGS.repetition_penalty,
    )


def generate(*, name: str, model_ref: str, system: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    with _LOCK:
        kv_cache.observe(prompt, worker=name)
        loaded = load(name, model_ref, multimodal=False)
        sampling = _sampling(max_new_tokens, temperature)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if hasattr(loaded.llm, "chat"):
            outputs = loaded.llm.chat(messages, sampling_params=sampling, use_tqdm=False)
        else:
            joined = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
            outputs = loaded.llm.generate(joined, sampling_params=sampling, use_tqdm=False)
        if not outputs or not getattr(outputs[0], "outputs", None):
            raise RuntimeError(f"{name}_vllm_empty_generation")
        return outputs[0].outputs[0].text.strip()


def generate_vision(*, name: str, model_ref: str, prompt: str, image_paths: list[str], max_new_tokens: int, temperature: float) -> str:
    with _LOCK:
        loaded = load(name, model_ref, multimodal=True)
        from PIL import Image
        images = [Image.open(Path(p).expanduser()).convert("RGB") for p in image_paths]
        sampling = _sampling(max_new_tokens, temperature)
        # vLLM's multimodal offline API accepts a prompt plus multi_modal_data.
        # For chat-template-heavy VLMs, keep prompt plain and let the model processor handle images.
        multi_modal_data: dict[str, Any]
        if len(images) == 1:
            multi_modal_data = {"image": images[0]}
        else:
            multi_modal_data = {"image": images}
        inputs = {"prompt": prompt, "multi_modal_data": multi_modal_data}
        outputs = loaded.llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
        if not outputs or not getattr(outputs[0], "outputs", None):
            raise RuntimeError(f"{name}_vllm_vlm_empty_generation")
        return outputs[0].outputs[0].text.strip()
