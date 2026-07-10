# SyMoNeuRaL Flagship MoE Server

One zipped production project merged from the three submitted SyMoNeuRaL sources. This build keeps your single public endpoint and internal name-based workers, but replaces the missing text inference layer with a flagship **vLLM CUDA** backend.

## What this is

```text
client / web UI / OpenAI-compatible caller
  -> FastAPI :8000
  -> SyMoNeuRaL router
  -> Gemma manager worker
     -> Qwen coder worker when code/build/debug is requested
     -> Vision worker when image input is present
     -> safetensor image worker when image generation is requested
  -> vLLM CUDA for text/VLM execution
  -> diffusers CUDA for image generation
```

No exposed LLM worker ports are required. vLLM runs **in process** through Python, not as a separate localhost server.

## Production gates

Static packaging check:

```bash
python3 production_audit.py --static-only
```

Production check on the target NVIDIA box:

```bash
python3 production_audit.py
```

Live generation check:

```bash
python3 production_audit.py --live
```

A production pass means the app imports, CUDA is visible, vLLM is installed, model paths are valid, image backend is valid, worker readiness is true, and live generation works.

## Quick install on NVIDIA CUDA Linux

```bash
cd SyMoNeuRaL_Flagship_MoE
cp config/symoneural.env.example .env
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
python3 production_audit.py --static-only
```

Put local Hugging Face model folders here by default:

```text
models/llm/gemma/
models/llm/qwen/
models/vlm/qwen2_5_vl/
models/sd/stable-diffusion/
```

Then run:

```bash
python3 production_audit.py
scripts/start.sh
scripts/test.sh
```

## Important model note

vLLM consumes Hugging Face/safetensors model directories directly. GGUF conversion is documented for legacy/export workflows, but GGUF is not the runtime format for this vLLM CUDA backend.

## Core docs

- `docs/ARCHITECTURE.md` — complete worker/linker/runtime layout
- `docs/INSTALL_CUDA_VLLM.md` — NVIDIA CUDA/vLLM install
- `docs/NVIDIA_HARDWARE.md` — RTX/Ada/Hopper/Blackwell tuning references
- `docs/MODEL_TUNING.md` — 16 GB, 24 GB, 48 GB, 80 GB model profiles
- `docs/MODEL_CONVERSION.md` — HF layout, safetensors, HF to GGUF legacy export
- `docs/SLO_FLOW_CONTROL.md` — admission control, queue limits, TTFT/ITL knobs
- `docs/KV_CACHE_PREFIX_ROUTING.md` — vLLM prefix caching and SyMoNeuRaL prefix affinity
- `docs/MULTIMODAL_VISION.md` — vLLM VLM path and optional ViT encoder path
- `docs/PRODUCTION_RUNBOOK.md` — operations commands
- `docs/SOURCES.md` — source zip merge notes and official references

## Environment summary

```bash
SYMON_TEXT_BACKEND=vllm
SYMON_VISION_BACKEND=vllm
SYMON_IMAGE_BACKEND=diffusers
SYMON_DEVICE=cuda
SYMON_VLLM_ENABLE_PREFIX_CACHING=1
SYMON_SINGLE_ACTIVE_MODEL=1
SYMON_REQUIRE_ALL_WORKERS=1
```

For one 16 GB GPU, keep `SYMON_SINGLE_ACTIVE_MODEL=1` and unload text before images. For larger cards, see `docs/MODEL_TUNING.md`.
