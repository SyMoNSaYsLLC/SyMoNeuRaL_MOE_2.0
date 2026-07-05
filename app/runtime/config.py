from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "data" / "runtime"
OUTPUT = ROOT / "data" / "output"
HF_CACHE = ROOT / "data" / "hf-cache"

for path in (RUNTIME, OUTPUT, HF_CACHE):
    path.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def resolve_local_path(model_ref: str) -> Path:
    path = Path(str(model_ref)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def looks_like_hf_repo_id(value: str) -> bool:
    value = value.strip()
    if value.startswith(("/", "./", "../", "~")):
        return False
    return "/" in value and not any(ch in value for ch in " \\:")


@dataclass(frozen=True)
class ModelRefs:
    gemma: str = _env_str("SYMON_GEMMA_MODEL", "models/llm/gemma")
    qwen: str = _env_str("SYMON_QWEN_MODEL", "models/llm/qwen")
    vision: str = _env_str("SYMON_VISION_MODEL", "models/vlm/qwen2_5_vl")
    stable: str = _env_str("SYMON_STABLE_MODEL", "models/sd/stable-diffusion")
    vit_encoder: str = _env_str("SYMON_VIT_ENCODER_MODEL", "models/vlm/vit-encoder")


@dataclass(frozen=True)
class RuntimeSettings:
    service_name: str = _env_str("SYMON_SERVICE_NAME", "symoneural_moe")
    bind_host: str = _env_str("SYMON_BIND_HOST", "0.0.0.0")
    bind_port: int = _env_int("SYMON_BIND_PORT", 8000)
    device: str = _env_str("SYMON_DEVICE", "cuda").lower()
    text_backend: str = _env_str("SYMON_TEXT_BACKEND", "vllm").lower()
    vision_backend: str = _env_str("SYMON_VISION_BACKEND", "vllm").lower()
    image_backend: str = _env_str("SYMON_IMAGE_BACKEND", "diffusers").lower()
    allow_remote_models: bool = _env_bool("SYMON_ALLOW_REMOTE_MODELS", False)
    require_all_workers: bool = _env_bool("SYMON_REQUIRE_ALL_WORKERS", True)

    # vLLM CUDA text/VLM runtime.
    vllm_dtype: str = _env_str("SYMON_VLLM_DTYPE", "auto")
    vllm_gpu_memory_utilization: float = _env_float("SYMON_VLLM_GPU_MEMORY_UTILIZATION", 0.82)
    vllm_max_model_len: int = _env_int("SYMON_VLLM_MAX_MODEL_LEN", 8192)
    vllm_cpu_offload_gb: float = _env_float("SYMON_VLLM_CPU_OFFLOAD_GB", 0.0)
    vllm_swap_space_gb: float = _env_float("SYMON_VLLM_SWAP_SPACE_GB", 4.0)
    vllm_quantization: str = _env_str("SYMON_VLLM_QUANTIZATION", "")
    vllm_enforce_eager: bool = _env_bool("SYMON_VLLM_ENFORCE_EAGER", False)
    vllm_trust_remote_code: bool = _env_bool("SYMON_VLLM_TRUST_REMOTE_CODE", True)
    vllm_disable_log_stats: bool = _env_bool("SYMON_VLLM_DISABLE_LOG_STATS", True)
    vllm_seed: int = _env_int("SYMON_VLLM_SEED", 0)
    vllm_enable_prefix_caching: bool = _env_bool("SYMON_VLLM_ENABLE_PREFIX_CACHING", True)
    vllm_max_num_seqs: int = _env_int("SYMON_VLLM_MAX_NUM_SEQS", 16)
    vllm_max_num_batched_tokens: int = _env_int("SYMON_VLLM_MAX_NUM_BATCHED_TOKENS", 8192)
    vllm_limit_mm_images: int = _env_int("SYMON_VLLM_LIMIT_MM_IMAGES", 4)

    # Sampling defaults.
    max_new_tokens_default: int = _env_int("SYMON_MAX_NEW_TOKENS", 768)
    temperature_chat: float = _env_float("SYMON_TEMPERATURE_CHAT", 0.7)
    temperature_code: float = _env_float("SYMON_TEMPERATURE_CODE", 0.2)
    temperature_vision: float = _env_float("SYMON_TEMPERATURE_VISION", 0.2)
    top_p: float = _env_float("SYMON_TOP_P", 0.95)
    repetition_penalty: float = _env_float("SYMON_REPETITION_PENALTY", 1.05)

    # Diffusers image runtime.
    image_width: int = _env_int("SYMON_IMAGE_WIDTH", 512)
    image_height: int = _env_int("SYMON_IMAGE_HEIGHT", 512)
    image_steps: int = _env_int("SYMON_IMAGE_STEPS", 24)
    image_guidance: float = _env_float("SYMON_IMAGE_GUIDANCE", 7.0)
    image_negative_prompt: str = _env_str("SYMON_IMAGE_NEGATIVE_PROMPT", "low quality, blurry, distorted, watermark")

    # GPU/VRAM guardrails. Defaults fit one 16 GB class card; increase per docs for datacenter GPUs.
    vram_target_gb: float = _env_float("SYMON_VRAM_TARGET_GB", 14.0)
    vram_hard_limit_gb: float = _env_float("SYMON_VRAM_HARD_LIMIT_GB", 15.2)
    unload_text_before_image: bool = _env_bool("SYMON_UNLOAD_TEXT_BEFORE_IMAGE", True)
    single_active_model: bool = _env_bool("SYMON_SINGLE_ACTIVE_MODEL", True)

    # SLO-aware admission and flow control.
    max_queue: int = _env_int("SYMON_MAX_QUEUE", 32)
    max_concurrent_text: int = _env_int("SYMON_MAX_CONCURRENT_TEXT", 1)
    max_concurrent_image: int = _env_int("SYMON_MAX_CONCURRENT_IMAGE", 1)
    max_concurrent_vision: int = _env_int("SYMON_MAX_CONCURRENT_VISION", 1)
    slo_ttft_ms: int = _env_int("SYMON_SLO_TTFT_MS", 2500)
    slo_itl_ms: int = _env_int("SYMON_SLO_ITL_MS", 150)
    slo_image_ms: int = _env_int("SYMON_SLO_IMAGE_MS", 45000)

    # Prefix-aware routing.
    prefix_cache_chars: int = _env_int("SYMON_PREFIX_CACHE_CHARS", 4096)
    prefix_route_affinity_seconds: int = _env_int("SYMON_PREFIX_ROUTE_AFFINITY_SECONDS", 600)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


MODELS = ModelRefs()
SETTINGS = RuntimeSettings()
