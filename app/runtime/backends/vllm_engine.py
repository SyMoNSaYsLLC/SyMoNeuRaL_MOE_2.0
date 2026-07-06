from __future__ import annotations

import gc
import inspect
import os
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from app.runtime import kv_cache
from app.runtime.config import HF_CACHE, SETTINGS, looks_like_hf_repo_id, resolve_local_path


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

    try:
        status["vllm_gguf_plugin"] = metadata.version("vllm-gguf-plugin")
    except Exception as exc:
        status["vllm_gguf_plugin_error"] = str(exc)

    return status


def deps_ready() -> tuple[bool, str]:
    d = dependency_status()

    for key in ("torch_error", "vllm_error"):
        if key in d:
            return False, f"{key}:{d[key]}"

    if SETTINGS.device == "cuda" and not d.get("cuda_available"):
        return False, "cuda_unavailable"

    return True, "ok"


def gguf_plugin_ready() -> tuple[bool, str]:
    try:
        version = metadata.version("vllm-gguf-plugin")
        return True, f"vllm-gguf-plugin:{version}"
    except Exception as exc:
        return False, f"missing_vllm_gguf_plugin:{exc}"


def _is_gguf_path(path: str | Path) -> bool:
    return Path(str(path)).suffix.lower() == ".gguf"


def _read_magic(path: Path, n: int = 4) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _normalize_support_ref(value: str) -> str:
    ref = value.strip()
    if not ref:
        return ref

    if looks_like_hf_repo_id(ref):
        return ref

    path = Path(ref).expanduser()
    if not path.is_absolute():
        path = resolve_local_path(ref)

    return str(path)


def _env_model_value(name: str, suffix: str) -> str | None:
    safe_name = name.upper().replace("-", "_")

    candidates = [
        f"SYMON_{safe_name}_{suffix}",
        f"SYMON_{safe_name}_{suffix}_PATH",
        f"SYMON_VLLM_{suffix}",
        f"SYMON_VLLM_{suffix}_PATH",
    ]

    if suffix == "HF_CONFIG":
        candidates.extend(
            [
                f"SYMON_{safe_name}_BASE_MODEL",
                "SYMON_VLLM_BASE_MODEL",
            ]
        )

    for env_name in candidates:
        value = os.environ.get(env_name)
        if value and value.strip():
            return _normalize_support_ref(value)

    return None


def _env_model_float(name: str, suffix: str, default: float) -> float:
    """Per-model override with global fallback, e.g. SYMON_GEMMA_TOP_P then SYMON_TOP_P."""
    safe_name = name.upper().replace("-", "_")
    for env_name in (f"SYMON_{safe_name}_{suffix}", f"SYMON_{suffix}"):
        value = os.environ.get(env_name)
        if value and value.strip():
            try:
                return float(value)
            except ValueError:
                continue
    return default


def _env_model_int(name: str, suffix: str, default: int) -> int:
    safe_name = name.upper().replace("-", "_")
    for env_name in (f"SYMON_{safe_name}_{suffix}", f"SYMON_{suffix}"):
        value = os.environ.get(env_name)
        if value and value.strip():
            try:
                return int(float(value))
            except ValueError:
                continue
    return default


def _env_model_str(name: str, suffix: str, default: str) -> str:
    safe_name = name.upper().replace("-", "_")
    for env_name in (f"SYMON_{safe_name}_{suffix}", f"SYMON_{suffix}"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return default


def _hf_token() -> str | None:
    for env_name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return None


def _worker_name_for_resolved_model(resolved_model: str) -> str:
    filename = Path(resolved_model).name.lower()

    if "qwen" in filename:
        return "qwen"

    if "gemma" in filename:
        return "gemma"

    return "model"


def inferred_base_model(name: str, resolved_model: str) -> str | None:
    filename = Path(resolved_model).name.lower()

    if "qwen2.5-coder-1.5b" in filename:
        return "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    if "qwen2.5-coder" in filename:
        return "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    if "gemma-4-12b" in filename:
        return "google/gemma-4-12B-it"

    if "gemma-3-12b" in filename:
        return "google/gemma-3-12b-it"

    if "gemma-2-27b" in filename:
        return "google/gemma-2-27b-it"

    if "gemma-2-9b" in filename:
        return "google/gemma-2-9b-it"

    if "gemma" in filename and "12b" in filename:
        return "google/gemma-3-12b-it"

    if "gemma" in filename:
        return "google/gemma-2-9b-it"

    return None


def tokenizer_for(name: str, resolved_model: str) -> str | None:
    return _env_model_value(name, "TOKENIZER") or inferred_base_model(name, resolved_model)


def hf_config_for(name: str, resolved_model: str) -> str | None:
    return _env_model_value(name, "HF_CONFIG") or inferred_base_model(name, resolved_model)


def _dtype_for_gguf(dtype: str) -> str:
    normalized = str(dtype).strip().lower()

    if normalized in {"float32", "fp32", "f32", "torch.float32"}:
        return "float32"

    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return "bfloat16"

    if normalized in {"half", "fp16", "f16", "float16", "torch.float16"}:
        return "float16"

    # "auto" (or unset) -- let vLLM pick based on the checkpoint's own dtype and
    # the GPU's capability, instead of silently forcing float16. This is what
    # was causing "Casting torch.bfloat16 to torch.float16" on every load for
    # bf16 GGUF checkpoints (e.g. Gemma) on GPUs that support bf16 natively.
    if normalized in {"", "auto"}:
        return "auto"

    return dtype


def model_ref_ready(model_ref: str, *, multimodal: bool = False) -> tuple[bool, str, str]:
    ref = str(model_ref).strip()

    if not ref:
        return False, "empty_model_ref", ref

    local = resolve_local_path(ref)

    if local.exists():
        if local.is_file():
            if local.suffix.lower() != ".gguf":
                return False, f"unsupported_vllm_model_file:{local}", str(local)

            if local.stat().st_size < 1024 * 1024:
                return False, f"gguf_file_too_small:{local}", str(local)

            magic = _read_magic(local)
            if magic != b"GGUF":
                return False, f"invalid_gguf_magic:{magic!r}:{local}", str(local)

            if multimodal:
                return False, f"gguf_vlm_not_supported_by_this_worker:{local}", str(local)

            return True, "ok_gguf_file", str(local)

        if not local.is_dir():
            return False, f"model_path_not_file_or_directory:{local}", str(local)

        if not (local / "config.json").exists():
            return False, f"missing_config_json:{local}", str(local)

        tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        if not any((local / item).exists() for item in tokenizer_files):
            return False, f"missing_tokenizer_files:{local}", str(local)

        weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        has_weights = any(any(local.glob(pattern)) for pattern in weight_patterns)
        has_index = any(
            (local / name).exists()
            for name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
        )

        if not has_weights and not has_index:
            return False, f"missing_model_weights:{local}", str(local)

        return True, "ok_hf_directory", str(local)

    if SETTINGS.allow_remote_models and looks_like_hf_repo_id(ref):
        return True, "remote_hf_model_allowed", ref

    return False, f"model_path_missing:{local}", str(local)


def ready_detail(model_ref: str, *, multimodal: bool = False) -> dict[str, Any]:
    ref_ok, ref_reason, resolved = model_ref_ready(model_ref, multimodal=multimodal)
    deps_ok, deps_reason = deps_ready()

    is_gguf = _is_gguf_path(resolved)
    plugin_ok = True
    plugin_reason = "not_required"
    name_hint = _worker_name_for_resolved_model(resolved)

    if is_gguf:
        plugin_ok, plugin_reason = gguf_plugin_ready()

    tokenizer = tokenizer_for(name_hint, resolved) if is_gguf else None
    hf_config = hf_config_for(name_hint, resolved) if is_gguf else None
    dtype = _dtype_for_gguf(SETTINGS.vllm_dtype) if is_gguf else SETTINGS.vllm_dtype

    return {
        "ok": bool(ref_ok and deps_ok and plugin_ok),
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
        "gguf": {
            "enabled": is_gguf,
            "plugin_ok": plugin_ok,
            "plugin_reason": plugin_reason,
            "tokenizer_hint": tokenizer,
            "hf_config_hint": hf_config,
            "dtype_effective": dtype,
            "tokenizer_env": [
                "SYMON_GEMMA_TOKENIZER",
                "SYMON_QWEN_TOKENIZER",
                "SYMON_VLLM_TOKENIZER",
            ],
            "hf_config_env": [
                "SYMON_GEMMA_HF_CONFIG",
                "SYMON_QWEN_HF_CONFIG",
                "SYMON_VLLM_HF_CONFIG",
                "SYMON_GEMMA_BASE_MODEL",
                "SYMON_QWEN_BASE_MODEL",
                "SYMON_VLLM_BASE_MODEL",
            ],
        },
        "vllm": {
            "dtype": dtype,
            "gpu_memory_utilization": SETTINGS.vllm_gpu_memory_utilization,
            "max_model_len": SETTINGS.vllm_max_model_len,
            "cpu_offload_gb": SETTINGS.vllm_cpu_offload_gb,
            "swap_space_gb": SETTINGS.vllm_swap_space_gb,
            "quantization": None if is_gguf else (SETTINGS.vllm_quantization or None),
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


def _force_kill_stray_engine_processes() -> None:
    # Last resort when waiting didn't work: del + vLLM's high-level teardown
    # calls do not reliably terminate the actual EngineCore child OS process
    # in every vLLM version/configuration. Rather than proceed into a
    # guaranteed-OOM engine boot, actively find and kill any stray
    # "VLLM::EngineCore" processes by name (mirrors scripts/stop.sh's sweep,
    # but self-healing at runtime instead of only at shutdown). Since this
    # app only ever intends one active model at a time (SETTINGS.
    # single_active_model), it is safe to clear all of them here -- unload()
    # is only called when we're intentionally tearing down the old engine
    # to make room for a new one.
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-f", "VLLM::EngineCore"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p for p in result.stdout.split() if p.strip()]
    except Exception as exc:
        print(f"[vllm_engine] _force_kill_stray_engine_processes: pgrep failed: {exc!r}")
        return

    if not pids:
        return

    print(f"[vllm_engine] Found stray EngineCore process(es) still holding GPU memory: {pids}. Killing.")
    try:
        subprocess.run(["kill", "-9", *pids], timeout=5)
    except Exception as exc:
        print(f"[vllm_engine] _force_kill_stray_engine_processes: kill failed: {exc!r}")


def _wait_for_vram_release(timeout_s: float = 30.0, poll_interval_s: float = 0.5) -> None:
    # vLLM's V1 engine runs the actual model in a separate child OS process
    # (visible as "EngineCore pid=..." in logs). torch.cuda.empty_cache() in
    # THIS (parent) process cannot free memory held by a different process's
    # CUDA context -- that's an OS-level boundary. Deleting our Python
    # reference to the LLM wrapper doesn't guarantee that child process has
    # actually exited and released its VRAM yet. Rather than trust vLLM's
    # internal teardown timing (which varies by version and isn't always
    # synchronous), poll real GPU state until memory is actually free, or
    # give up after timeout_s and proceed anyway (logged, not silent).
    import time

    from app.runtime import gpu as gpu_status_module

    deadline = time.monotonic() + timeout_s
    last_free_gb = None

    while time.monotonic() < deadline:
        try:
            status = gpu_status_module.status()
            devices = status.get("devices") or []
            if not devices:
                return
            last_free_gb = float(devices[0].get("free_gb", 0.0))
            # A conservative bar: enough free VRAM for a typical next engine's
            # gpu_memory_utilization share of total, with some margin. We
            # don't know the next model's exact requirement here, so just
            # require most of the card to look free again.
            total_gb = float(devices[0].get("total_gb", 0.0))
            if total_gb > 0 and last_free_gb >= total_gb * 0.6:
                return
        except Exception:
            return
        time.sleep(poll_interval_s)

    print(
        f"[vllm_engine] _wait_for_vram_release: timed out after {timeout_s}s, "
        f"last observed free_gb={last_free_gb}. The old engine process likely "
        f"did not release its memory on its own -- killing it directly."
    )

    _force_kill_stray_engine_processes()

    # After force-killing, give the OS/driver a brief moment to actually
    # reclaim the memory, and check once more (short budget this time --
    # if it's still stuck after a real kill -9, waiting longer won't help).
    import time as _time

    for _ in range(10):
        try:
            status = gpu_status_module.status()
            devices = status.get("devices") or []
            if not devices:
                return
            free_gb = float(devices[0].get("free_gb", 0.0))
            total_gb = float(devices[0].get("total_gb", 0.0))
            if total_gb > 0 and free_gb >= total_gb * 0.5:
                print(f"[vllm_engine] VRAM recovered after force-kill: free_gb={free_gb}")
                return
        except Exception:
            return
        _time.sleep(1)

    print(
        "[vllm_engine] VRAM still not recovered after force-killing stray "
        "EngineCore process(es). The next engine boot may still fail -- "
        "this may require a manual check (nvidia-smi --query-compute-apps=...)."
    )


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
            _wait_for_vram_release()


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs

        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def _llm_kwargs(name: str, resolved_model: str, *, multimodal: bool) -> dict[str, Any]:
    # Per-model overrides fall back to the shared SYMON_VLLM_* value, e.g.
    # SYMON_GEMMA_MAX_MODEL_LEN (llama.cpp: --ctx-size) or
    # SYMON_QWEN_GPU_MEMORY_UTILIZATION (llama.cpp: how much VRAM a model is
    # allowed to claim -- vLLM's analog of --n-gpu-layers on a single-GPU,
    # fully-offloaded setup is the fraction of VRAM it's allowed to reserve).
    max_model_len = _env_model_int(name, "MAX_MODEL_LEN", SETTINGS.vllm_max_model_len)
    gpu_memory_utilization = _env_model_float(
        name, "GPU_MEMORY_UTILIZATION", SETTINGS.vllm_gpu_memory_utilization
    )
    kv_cache_dtype = _env_model_str(name, "KV_CACHE_DTYPE", SETTINGS.vllm_kv_cache_dtype)

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "trust_remote_code": SETTINGS.vllm_trust_remote_code,
        "tensor_parallel_size": 1,
        "dtype": SETTINGS.vllm_dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
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

    if kv_cache_dtype and kv_cache_dtype.lower() != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    if SETTINGS.vllm_quantization:
        kwargs["quantization"] = SETTINGS.vllm_quantization

    if _is_gguf_path(resolved_model):
        kwargs["load_format"] = "gguf"
        kwargs["dtype"] = _dtype_for_gguf(str(kwargs.get("dtype", "")))

        # GGUF load_format is the quantized loader. Do not also pass a
        # separate vLLM quantization method.
        kwargs.pop("quantization", None)

        tokenizer = tokenizer_for(name, resolved_model)
        if tokenizer:
            kwargs["tokenizer"] = tokenizer

        hf_config = hf_config_for(name, resolved_model)
        if hf_config:
            kwargs["hf_config_path"] = hf_config

        hf_token = _hf_token()
        if hf_token:
            kwargs["hf_token"] = hf_token

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
        if (
            _ACTIVE is not None
            and _ACTIVE.name == name
            and _ACTIVE.model_ref == resolved_model
            and _ACTIVE.is_multimodal == multimodal
        ):
            return _ACTIVE

        if _ACTIVE is not None and SETTINGS.single_active_model:
            unload()

        try:
            from app.runtime import image_engine

            image_engine.unload()
        except Exception:
            pass

        # llama.cpp's `-fa on` forces flash-attention. vLLM auto-selects the best
        # backend already, but this lets you pin one (e.g. FLASH_ATTN, FLASHINFER,
        # XFORMERS) via SYMON_<NAME>_ATTENTION_BACKEND / SYMON_VLLM_ATTENTION_BACKEND.
        attention_backend = _env_model_str(name, "ATTENTION_BACKEND", SETTINGS.vllm_attention_backend)
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

        from vllm import LLM

        kwargs = _filter_kwargs(LLM, _llm_kwargs(name, resolved_model, multimodal=multimodal))
        llm = LLM(**kwargs)

        _ACTIVE = LoadedModel(
            name=name,
            model_ref=resolved_model,
            llm=llm,
            is_multimodal=multimodal,
        )
        return _ACTIVE


def _sampling(name: str, max_new_tokens: int, temperature: float) -> Any:
    from vllm import SamplingParams

    # Per-model overrides (e.g. SYMON_GEMMA_TOP_P / SYMON_QWEN_TOP_K) fall back to
    # the shared SYMON_TOP_P / SYMON_TOP_K / SYMON_REPETITION_PENALTY values.
    top_p = _env_model_float(name, "TOP_P", SETTINGS.top_p)
    top_k = _env_model_int(name, "TOP_K", SETTINGS.top_k)
    repetition_penalty = _env_model_float(name, "REPETITION_PENALTY", SETTINGS.repetition_penalty)
    min_p = _env_model_float(name, "MIN_P", SETTINGS.min_p)
    presence_penalty = _env_model_float(name, "PRESENCE_PENALTY", SETTINGS.presence_penalty)
    frequency_penalty = _env_model_float(name, "FREQUENCY_PENALTY", SETTINGS.frequency_penalty)

    kwargs: dict[str, Any] = {
        "max_tokens": max(1, int(max_new_tokens)),
        "temperature": max(0.0, float(temperature)),
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    # llama.cpp's --top-k 0 means "disabled"; vLLM uses -1 for that. 0 or fewer
    # from the .env is treated as "off" here so both conventions behave the same.
    if top_k and top_k > 0:
        kwargs["top_k"] = top_k

    # min_p 0 (the SamplingParams default) means "disabled" already, so no
    # special-casing needed the way top_k needs -- just pass it through.
    if min_p and min_p > 0:
        kwargs["min_p"] = min_p

    return SamplingParams(**kwargs)


def generate(
    *,
    name: str,
    model_ref: str,
    system: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    with _LOCK:
        kv_cache.observe(prompt, worker=name)

        loaded = load(name, model_ref, multimodal=False)
        sampling = _sampling(name, max_new_tokens, temperature)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        if hasattr(loaded.llm, "chat"):
            outputs = loaded.llm.chat(messages, sampling_params=sampling, use_tqdm=False)
        else:
            joined = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
            outputs = loaded.llm.generate(joined, sampling_params=sampling, use_tqdm=False)

        if not outputs or not getattr(outputs[0], "outputs", None):
            raise RuntimeError(f"{name}_vllm_empty_generation")

        return outputs[0].outputs[0].text.strip()


def generate_vision(
    *,
    name: str,
    model_ref: str,
    prompt: str,
    image_paths: list[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    with _LOCK:
        loaded = load(name, model_ref, multimodal=True)

        from PIL import Image

        images = [Image.open(Path(p).expanduser()).convert("RGB") for p in image_paths]
        sampling = _sampling(name, max_new_tokens, temperature)

        if len(images) == 1:
            multi_modal_data: dict[str, Any] = {"image": images[0]}
        else:
            multi_modal_data = {"image": images}

        inputs = {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
        }

        outputs = loaded.llm.generate(inputs, sampling_params=sampling, use_tqdm=False)

        if not outputs or not getattr(outputs[0], "outputs", None):
            raise RuntimeError(f"{name}_vllm_vlm_empty_generation")

        return outputs[0].outputs[0].text.strip()
