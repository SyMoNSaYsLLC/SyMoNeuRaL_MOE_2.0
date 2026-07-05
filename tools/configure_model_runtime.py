#!/usr/bin/env python3
"""
SyMoNeuRaL runtime configurator (llama.cpp-flag equivalents for the vLLM/diffusers stack)
==========================================================================================

This project does NOT run llama.cpp / llama-server. Gemma and Qwen are served through
vLLM (with the GGUF loader), and the v1-5 checkpoint is served through diffusers. So
flags like:

    --ctx-size 16384 --n-gpu-layers 99 --no-mmap -fa on
    --cache-type-k q8_0 --cache-type-v q8_0
    --temp 1.0 --top-p 0.95 --top-k 64

don't exist as CLI flags here -- but nearly all of them have a direct vLLM equivalent,
set through environment variables in .env. This script edits those variables for you.

    llama.cpp flag              ->  vLLM setting                  ->  env var
    ---------------------------------------------------------------------------------
    --ctx-size N                ->  max_model_len                 ->  SYMON_<MODEL>_MAX_MODEL_LEN
    --n-gpu-layers 99           ->  gpu_memory_utilization         ->  SYMON_<MODEL>_GPU_MEMORY_UTILIZATION
                                     (vLLM always runs fully on GPU; this is "how big
                                      a slice of VRAM this model is allowed to reserve")
    --cache-type-k/v q8_0       ->  kv_cache_dtype                 ->  SYMON_<MODEL>_KV_CACHE_DTYPE
                                     (use "fp8" -- vLLM has no q8_0 KV cache, fp8 is
                                      the closest quantized KV cache it supports)
    -fa on                      ->  attention backend auto-select  ->  SYMON_<MODEL>_ATTENTION_BACKEND
                                     (vLLM already prefers FlashAttention/FlashInfer;
                                      set this only to pin a specific backend)
    --temp / --top-p / --top-k  ->  SamplingParams                 ->  SYMON_<MODEL>_TEMPERATURE*
                                                                        SYMON_<MODEL>_TOP_P
                                                                        SYMON_<MODEL>_TOP_K
    --no-mmap                   ->  no equivalent (not applicable to vLLM's loader)

Every <MODEL>-prefixed var falls back to a shared SYMON_VLLM_*/SYMON_TOP_P/etc. default
if you don't set it, so you only need to touch what you want to change per-model.

Run it:
    python3 tools/configure_model_runtime.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

# ---------------------------------------------------------------------------
# Key groups. "temperature_key" differs per model because the app already has
# dedicated chat/code temperature knobs (SYMON_TEMPERATURE_CHAT / _CODE); the
# rest use the generic <MODEL>_<SUFFIX> pattern read by app/runtime.
# ---------------------------------------------------------------------------

MODEL_KEYS = {
    "gemma": {
        "label": "Gemma (manager / writer, GGUF via vLLM)",
        "temperature_key": "SYMON_TEMPERATURE_CHAT",
        "ctx_key": "SYMON_GEMMA_MAX_MODEL_LEN",
        "vram_key": "SYMON_GEMMA_GPU_MEMORY_UTILIZATION",
        "kv_key": "SYMON_GEMMA_KV_CACHE_DTYPE",
        "attn_key": "SYMON_GEMMA_ATTENTION_BACKEND",
        "top_p_key": "SYMON_GEMMA_TOP_P",
        "top_k_key": "SYMON_GEMMA_TOP_K",
        "rep_key": "SYMON_GEMMA_REPETITION_PENALTY",
        "defaults": {
            "SYMON_TEMPERATURE_CHAT": "1.0",
            "SYMON_GEMMA_MAX_MODEL_LEN": "16384",
            "SYMON_GEMMA_GPU_MEMORY_UTILIZATION": "0.55",
            "SYMON_GEMMA_KV_CACHE_DTYPE": "fp8",
            "SYMON_GEMMA_ATTENTION_BACKEND": "",
            "SYMON_GEMMA_TOP_P": "0.95",
            "SYMON_GEMMA_TOP_K": "64",
            "SYMON_GEMMA_REPETITION_PENALTY": "1.05",
        },
    },
    "qwen": {
        "label": "Qwen (code specialist, GGUF via vLLM)",
        "temperature_key": "SYMON_TEMPERATURE_CODE",
        "ctx_key": "SYMON_QWEN_MAX_MODEL_LEN",
        "vram_key": "SYMON_QWEN_GPU_MEMORY_UTILIZATION",
        "kv_key": "SYMON_QWEN_KV_CACHE_DTYPE",
        "attn_key": "SYMON_QWEN_ATTENTION_BACKEND",
        "top_p_key": "SYMON_QWEN_TOP_P",
        "top_k_key": "SYMON_QWEN_TOP_K",
        "rep_key": "SYMON_QWEN_REPETITION_PENALTY",
        "defaults": {
            "SYMON_TEMPERATURE_CODE": "0.2",
            "SYMON_QWEN_MAX_MODEL_LEN": "8192",
            "SYMON_QWEN_GPU_MEMORY_UTILIZATION": "0.35",
            "SYMON_QWEN_KV_CACHE_DTYPE": "fp8",
            "SYMON_QWEN_ATTENTION_BACKEND": "",
            "SYMON_QWEN_TOP_P": "0.90",
            "SYMON_QWEN_TOP_K": "40",
            "SYMON_QWEN_REPETITION_PENALTY": "1.05",
        },
    },
}

# Stable Diffusion v1-5 (.safetensors, served via diffusers -- no ctx/kv-cache concept,
# but width/height/steps/guidance are the equivalent "how much VRAM and time per image"
# knobs, and diffusers already auto-enables attention slicing / VAE tiling for you).
SD_KEYS = {
    "width": "SYMON_IMAGE_WIDTH",
    "height": "SYMON_IMAGE_HEIGHT",
    "steps": "SYMON_IMAGE_STEPS",
    "guidance": "SYMON_IMAGE_GUIDANCE",
    "negative_prompt": "SYMON_IMAGE_NEGATIVE_PROMPT",
}
SD_DEFAULTS = {
    "SYMON_IMAGE_WIDTH": "512",
    "SYMON_IMAGE_HEIGHT": "512",
    "SYMON_IMAGE_STEPS": "24",
    "SYMON_IMAGE_GUIDANCE": "7.0",
    "SYMON_IMAGE_NEGATIVE_PROMPT": "low quality, blurry, distorted, watermark",
}

GLOBAL_KEYS = {
    "SYMON_VLLM_MAX_MODEL_LEN": "8192",
    "SYMON_VLLM_GPU_MEMORY_UTILIZATION": "0.82",
    "SYMON_VLLM_KV_CACHE_DTYPE": "auto",
    "SYMON_VLLM_ATTENTION_BACKEND": "",
    "SYMON_TOP_P": "0.95",
    "SYMON_TOP_K": "64",
    "SYMON_REPETITION_PENALTY": "1.05",
}

KV_CACHE_CHOICES = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
ATTN_CHOICES = ["", "FLASH_ATTN", "FLASHINFER", "XFORMERS"]


# ---------------------------------------------------------------------------
# .env read/write helpers (same approach as configure_vram_policy.py)
# ---------------------------------------------------------------------------

def read_env() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(errors="ignore").splitlines()


def parse_env(lines: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def update_env_lines(lines: list[str], values: dict[str, str], header: str) -> list[str]:
    key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
    remaining = dict(values)
    output: list[str] = []

    for line in lines:
        match = key_re.match(line.strip())
        if not match:
            output.append(line)
            continue
        key = match.group(1)
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)

    if remaining:
        if output and output[-1].strip():
            output.append("")
        output.append(f"# {header}")
        for key, value in remaining.items():
            output.append(f"{key}={value}")

    return output


def write_env(lines: list[str], values: dict[str, str], header: str) -> None:
    updated = update_env_lines(lines, values, header)
    ENV_PATH.write_text("\n".join(updated).rstrip() + "\n")
    print(f"Wrote {ENV_PATH}")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    print()
    print(prompt)
    for i, item in enumerate(choices, start=1):
        shown = item if item else "(blank = let vLLM auto-select)"
        marker = "  <- default" if item == default else ""
        print(f"  {i}) {shown}{marker}")
    raw = input(f"Select number or type a value [{default or 'blank'}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1]
    return raw


def require_float(key: str, value: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        f = float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid number for {key}: {value}") from exc
    if lo is not None and f < lo:
        raise SystemExit(f"{key} must be >= {lo}")
    if hi is not None and f > hi:
        raise SystemExit(f"{key} must be <= {hi}")
    return f


def require_int(key: str, value: str, lo: int | None = None) -> int:
    try:
        i = int(float(value))
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {key}: {value}") from exc
    if lo is not None and i < lo:
        raise SystemExit(f"{key} must be >= {lo}")
    return i


# ---------------------------------------------------------------------------
# Per-model configuration flow
# ---------------------------------------------------------------------------

def configure_model(model_key: str, current: dict[str, str]) -> dict[str, str]:
    spec = MODEL_KEYS[model_key]
    print()
    print(spec["label"])
    print("=" * len(spec["label"]))

    values = {k: current.get(k, v) for k, v in spec["defaults"].items()}

    print("Press Enter to keep the shown value. These map to llama.cpp flags as noted.")

    values[spec["ctx_key"]] = ask(
        f"Context length / --ctx-size  ({spec['ctx_key']})", values[spec["ctx_key"]]
    )
    values[spec["vram_key"]] = ask(
        f"VRAM fraction this model may reserve / --n-gpu-layers 99 equivalent, 0.05-0.95  ({spec['vram_key']})",
        values[spec["vram_key"]],
    )
    values[spec["kv_key"]] = ask_choice(
        f"KV cache dtype / --cache-type-k --cache-type-v  ({spec['kv_key']})",
        KV_CACHE_CHOICES,
        values[spec["kv_key"]],
    )
    values[spec["attn_key"]] = ask_choice(
        f"Attention backend pin / -fa on  ({spec['attn_key']})",
        ATTN_CHOICES,
        values[spec["attn_key"]],
    )
    values[spec["temperature_key"]] = ask(
        f"Temperature / --temp  ({spec['temperature_key']})", values[spec["temperature_key"]]
    )
    values[spec["top_p_key"]] = ask(
        f"top-p / --top-p  ({spec['top_p_key']})", values[spec["top_p_key"]]
    )
    values[spec["top_k_key"]] = ask(
        f"top-k / --top-k (0 = disabled)  ({spec['top_k_key']})", values[spec["top_k_key"]]
    )
    values[spec["rep_key"]] = ask(
        f"Repetition penalty (no direct llama.cpp flag; similar intent to --repeat-penalty)  ({spec['rep_key']})",
        values[spec["rep_key"]],
    )

    # Validate
    ctx = require_int(spec["ctx_key"], values[spec["ctx_key"]], lo=256)
    vram = require_float(spec["vram_key"], values[spec["vram_key"]], lo=0.05, hi=0.98)
    require_float(spec["temperature_key"], values[spec["temperature_key"]], lo=0.0, hi=2.0)
    require_float(spec["top_p_key"], values[spec["top_p_key"]], lo=0.0, hi=1.0)
    require_int(spec["top_k_key"], values[spec["top_k_key"]], lo=0)
    require_float(spec["rep_key"], values[spec["rep_key"]], lo=0.5, hi=2.0)
    kv = values[spec["kv_key"]]
    if kv not in KV_CACHE_CHOICES:
        raise SystemExit(f"{spec['kv_key']} must be one of {KV_CACHE_CHOICES}")

    print()
    print(f"  ctx={ctx}  vram_share={vram:.2f}  kv_cache={kv or 'auto'}  "
          f"attn={values[spec['attn_key']] or 'auto'}  temp={values[spec['temperature_key']]}  "
          f"top_p={values[spec['top_p_key']]}  top_k={values[spec['top_k_key']]}")

    return values


def configure_stable_diffusion(current: dict[str, str]) -> dict[str, str]:
    print()
    print("Stable Diffusion v1-5 (.safetensors, served via diffusers)")
    print("============================================================")
    print("No ctx-size / n-gpu-layers / KV cache here -- diffusers already enables")
    print("attention slicing + VAE tiling automatically. These are the equivalent")
    print("VRAM/quality/time knobs for a single-image pipeline.")

    values = {k: current.get(k, v) for k, v in SD_DEFAULTS.items()}

    values[SD_KEYS["width"]] = ask("Width", values[SD_KEYS["width"]])
    values[SD_KEYS["height"]] = ask("Height", values[SD_KEYS["height"]])
    values[SD_KEYS["steps"]] = ask("Inference steps", values[SD_KEYS["steps"]])
    values[SD_KEYS["guidance"]] = ask("Guidance scale (CFG)", values[SD_KEYS["guidance"]])
    values[SD_KEYS["negative_prompt"]] = ask(
        "Default negative prompt", values[SD_KEYS["negative_prompt"]]
    )

    require_int(SD_KEYS["width"], values[SD_KEYS["width"]], lo=64)
    require_int(SD_KEYS["height"], values[SD_KEYS["height"]], lo=64)
    require_int(SD_KEYS["steps"], values[SD_KEYS["steps"]], lo=1)
    require_float(SD_KEYS["guidance"], values[SD_KEYS["guidance"]], lo=0.0, hi=30.0)

    return values


def configure_global(current: dict[str, str]) -> dict[str, str]:
    print()
    print("Global vLLM defaults (used when a model has no per-model override)")
    print("======================================================================")

    values = {k: current.get(k, v) for k, v in GLOBAL_KEYS.items()}

    values["SYMON_VLLM_MAX_MODEL_LEN"] = ask(
        "Default context length / --ctx-size", values["SYMON_VLLM_MAX_MODEL_LEN"]
    )
    values["SYMON_VLLM_GPU_MEMORY_UTILIZATION"] = ask(
        "Default VRAM fraction / --n-gpu-layers 99 equivalent",
        values["SYMON_VLLM_GPU_MEMORY_UTILIZATION"],
    )
    values["SYMON_VLLM_KV_CACHE_DTYPE"] = ask_choice(
        "Default KV cache dtype / --cache-type-k/v", KV_CACHE_CHOICES,
        values["SYMON_VLLM_KV_CACHE_DTYPE"],
    )
    values["SYMON_VLLM_ATTENTION_BACKEND"] = ask_choice(
        "Default attention backend / -fa on", ATTN_CHOICES,
        values["SYMON_VLLM_ATTENTION_BACKEND"],
    )
    values["SYMON_TOP_P"] = ask("Default top-p", values["SYMON_TOP_P"])
    values["SYMON_TOP_K"] = ask("Default top-k", values["SYMON_TOP_K"])
    values["SYMON_REPETITION_PENALTY"] = ask(
        "Default repetition penalty", values["SYMON_REPETITION_PENALTY"]
    )

    require_int("SYMON_VLLM_MAX_MODEL_LEN", values["SYMON_VLLM_MAX_MODEL_LEN"], lo=256)
    require_float(
        "SYMON_VLLM_GPU_MEMORY_UTILIZATION", values["SYMON_VLLM_GPU_MEMORY_UTILIZATION"],
        lo=0.05, hi=0.98,
    )
    require_float("SYMON_TOP_P", values["SYMON_TOP_P"], lo=0.0, hi=1.0)
    require_int("SYMON_TOP_K", values["SYMON_TOP_K"], lo=0)
    require_float("SYMON_REPETITION_PENALTY", values["SYMON_REPETITION_PENALTY"], lo=0.5, hi=2.0)

    return values


def show_effective(current: dict[str, str]) -> None:
    print()
    print("Current .env values relevant to model runtime")
    print("================================================")
    all_keys: dict[str, Any] = {}
    for spec in MODEL_KEYS.values():
        for suffix in ("temperature_key", "ctx_key", "vram_key", "kv_key", "attn_key", "top_p_key", "top_k_key", "rep_key"):
            all_keys[spec[suffix]] = None
    all_keys.update({k: None for k in SD_KEYS.values()})
    all_keys.update({k: None for k in GLOBAL_KEYS})

    for key in sorted(all_keys):
        print(f"  {key} = {current.get(key, '(unset, uses code default)')}")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main() -> int:
    lines = read_env()
    current = parse_env(lines)
    pending: dict[str, str] = {}

    print(__doc__.strip().splitlines()[0])
    print()
    print("SyMoNeuRaL model runtime configurator")
    print("======================================")

    while True:
        print()
        print("1) Configure Gemma  (--ctx-size/--n-gpu-layers/-fa/--cache-type/--temp equivalents)")
        print("2) Configure Qwen   (same, coder specialist)")
        print("3) Configure Stable Diffusion v1-5 (.safetensors)")
        print("4) Configure global vLLM defaults (fallback for 1 & 2)")
        print("5) Show current effective .env values")
        print("6) Save changes and exit")
        print("7) Exit without saving")

        choice = input("Select [1-7]: ").strip()

        if choice == "1":
            pending.update(configure_model("gemma", {**current, **pending}))
        elif choice == "2":
            pending.update(configure_model("qwen", {**current, **pending}))
        elif choice == "3":
            pending.update(configure_stable_diffusion({**current, **pending}))
        elif choice == "4":
            pending.update(configure_global({**current, **pending}))
        elif choice == "5":
            show_effective({**current, **pending})
        elif choice == "6":
            if not pending:
                print("Nothing changed.")
                return 0
            write_env(lines, pending, "SyMoNeuRaL model runtime settings")
            print()
            print("Restart the service for these to take effect:")
            print("  scripts/stop.sh && scripts/start.sh")
            return 0
        elif choice == "7":
            print("No changes written.")
            return 1
        else:
            print("Unknown option.")


if __name__ == "__main__":
    raise SystemExit(main())
