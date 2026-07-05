<!-- SYMONEURAL_BOOTSTRAP_START -->
# First-time setup

Run the repo bootstrap first. It creates the Python environment, installs requirements, verifies imports, and runs the first production audit.

From the repo root:

```bash
cd ~/SyMoNeuRaL_MoE
scripts/bootstrap_env.sh
```

What `scripts/bootstrap_env.sh` does:

```text
1. verifies repo root
2. verifies Python is available
3. creates or reuses .venv
4. activates .venv
5. upgrades pip, setuptools, and wheel
6. installs requirements.txt
7. runs pip check
8. verifies FastAPI/SyMoNeuRaL app imports
9. runs production_audit.py
```

After bootstrap succeeds, activate the environment:

```bash
source .venv/bin/activate
```

Run the model file check:

```bash
scripts/check_models_magic.sh
```

Run the full production audit:

```bash
python3 production_audit.py
```

Run the live generation audit after CUDA, vLLM, GGUF plugin, and model paths are configured:

```bash
python3 production_audit.py --live
```

Start and test the server:

```bash
scripts/start.sh
scripts/test.sh
```

For CUDA/vLLM repair or reinstall:

```bash
scripts/install_cuda_vllm.sh
```

The CUDA PyTorch wheel index used by this project is:

```text
https://download.pytorch.org/whl/cu129
```

<!-- SYMONEURAL_BOOTSTRAP_END -->

---

# SyMoNeuRaL Flagship MoE Server

SyMoNeuRaL is a single-endpoint MoE server with internal name-based workers and a CUDA-backed inference path.

This build keeps one public FastAPI endpoint and routes internally through SyMoNeuRaL workers/linkers. It uses CUDA-backed text/VLM execution paths and diffusers CUDA for image generation.

## Runtime shape

```text
client / web UI / OpenAI-compatible caller
  -> FastAPI :8000
  -> SyMoNeuRaL router
  -> Gemma manager/writer worker
     -> Qwen coder worker when code/build/debug is requested
     -> Vision worker when image input is present
     -> safetensor image worker when image generation is requested
  -> CUDA-backed model runtime
```

Workers are routed internally by Python module name. The app exposes one public FastAPI endpoint.

## Current local model layout

The current repo model files are expected here:

```text
models/
  llm/
    gemma-4-12b-it-Q4_K_M.gguf
    qwen2.5-coder-1.5b-instruct-q8_0.gguf
  sd/
    v1-5-pruned-emaonly.safetensors
```

These are local model files stored directly in the repo model tree.

The model magic check verifies:

```text
Gemma -> GGUF header
Qwen  -> GGUF header
Stable Diffusion -> safetensors JSON header
```

Run:

```bash
scripts/check_models_magic.sh
```

Expected current result:

```text
PASS gemma: valid GGUF header
PASS qwen: valid GGUF header
PASS stable: valid safetensors header
PASS model magic check
```

## Environment summary

Default `.env` for the current local model files:

```bash
SYMON_BIND_HOST=0.0.0.0
SYMON_BIND_PORT=8000

SYMON_TEXT_BACKEND=vllm
SYMON_VISION_BACKEND=vllm
SYMON_IMAGE_BACKEND=diffusers
SYMON_DEVICE=cuda

SYMON_GEMMA_MODEL=models/llm/gemma-4-12b-it-Q4_K_M.gguf
SYMON_QWEN_MODEL=models/llm/qwen2.5-coder-1.5b-instruct-q8_0.gguf
SYMON_STABLE_MODEL=models/sd/v1-5-pruned-emaonly.safetensors

SYMON_VLLM_ENABLE_PREFIX_CACHING=1
SYMON_SINGLE_ACTIVE_MODEL=1
SYMON_REQUIRE_ALL_WORKERS=0
```

`SYMON_REQUIRE_ALL_WORKERS=0` is used while the VLM model is staged later. Set it to `1` after a VLM model is added and verified.

For one 16 GB GPU, keep:

```bash
SYMON_SINGLE_ACTIVE_MODEL=1
```

That forces one heavy model path active at a time and protects VRAM while switching between text, vision, and image generation.

## Model preparation tools

Active model-prep tools:

```text
tools/hf_download_interactive.py
tools/hf_to_gguf_interactive.py
tools/quantize_gguf_interactive.py
scripts/check_models_magic.sh
```

### Download a Hugging Face model interactively

Use:

```bash
tools/hf_download_interactive.py
```

The script reminds the user to authenticate with Hugging Face first:

```bash
hf auth login
hf auth whoami
```

It installs or updates the Hugging Face CLI using:

```bash
uv tool install hf
```

Then it asks for a model URL or repo id, for example:

```text
google/gemma-4-E2B-it-qat-mobile-transformers
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

### Convert Hugging Face model folder to GGUF

Use:

```bash
tools/hf_to_gguf_interactive.py
```

The script asks for:

```text
Hugging Face model folder
output GGUF path
output type: auto, f32, f16, bf16, q8_0, tq1_0, tq2_0
optional model metadata name
```

This uses a local `llama.cpp` converter script when available.

### Quantize GGUF interactively

Use:

```bash
tools/quantize_gguf_interactive.py
```

The script asks for:

```text
input GGUF
output GGUF
target quantization type
optional tensor type overrides
optional imatrix
optional thread count
```

Typical quantization choices include:

```text
Q8_0
Q6_K
Q5_K_M
Q5_K_S
Q4_K_M
Q4_K_S
Q3_K_L
Q3_K_M
Q3_K_S
Q2_K
IQ4_NL
IQ4_XS
IQ3_M
IQ2_M
```

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

A production pass means:

```text
1. app imports successfully
2. CUDA is visible
3. runtime dependencies are installed
4. configured model paths are valid
5. image backend is valid
6. worker readiness is true
7. live generation works
```

## Core layout

```text
SyMoNeuRaL_MoE/
  app/
    main.py
    symoneural.py
    workers/
      gemma.py
      qwen.py
      vision.py
      safetensor.py
      vram_guard.py
    comms/
      gemmacodercomms.py
      gemmaimagingcomms.py
      gemmavisioncomms.py
    runtime/
      config.py
      gpu.py
      flow_control.py
      kv_cache.py
      text_engine.py
      vision_engine.py
      image_engine.py
      vit_encoder.py
      worker_types.py
      backends/
        vllm_engine.py
  config/
    symoneural.env.example
  docs/
    ARCHITECTURE.md
    INSTALL_CUDA_VLLM.md
    NVIDIA_HARDWARE.md
    MODEL_TUNING.md
    MODEL_CONVERSION.md
    SLO_FLOW_CONTROL.md
    KV_CACHE_PREFIX_ROUTING.md
    MULTIMODAL_VISION.md
    PRODUCTION_RUNBOOK.md
    SOURCES.md
  scripts/
    bootstrap_env.sh
    install_cuda_vllm.sh
    check_models_magic.sh
    start.sh
    stop.sh
    status.sh
    test.sh
  tools/
    hf_download_interactive.py
    hf_to_gguf_interactive.py
    quantize_gguf_interactive.py
  production_audit.py
  requirements.txt
  README.md
```

## Core docs

- `docs/ARCHITECTURE.md` — worker/linker/runtime layout
- `docs/INSTALL_CUDA_VLLM.md` — NVIDIA CUDA/vLLM install
- `docs/NVIDIA_HARDWARE.md` — RTX/Ada/Hopper/Blackwell tuning references
- `docs/MODEL_TUNING.md` — 16 GB, 24 GB, 48 GB, 80 GB model profiles
- `docs/MODEL_CONVERSION.md` — HF layout, safetensors, HF to GGUF workflows
- `docs/SLO_FLOW_CONTROL.md` — admission control, queue limits, TTFT/ITL knobs
- `docs/KV_CACHE_PREFIX_ROUTING.md` — prefix caching and SyMoNeuRaL prefix affinity
- `docs/MULTIMODAL_VISION.md` — VLM path and optional ViT encoder path
- `docs/PRODUCTION_RUNBOOK.md` — operations commands
- `docs/SOURCES.md` — source zip merge notes and official references

## Server commands

Start:

```bash
scripts/start.sh
```

Stop:

```bash
scripts/stop.sh
```

Status:

```bash
scripts/status.sh
```

Test:

```bash
scripts/test.sh
```

## OpenAI-compatible endpoint

The server exposes an OpenAI-compatible chat completions route:

```text
POST /v1/chat/completions
```

Basic test after the server is running:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "symoneural",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "max_tokens": 64
  }'
```

## Health

Health endpoint:

```text
GET /health
```

Health reports the FastAPI manager state and worker readiness. Production audit and live audit are the readiness gates.

## Expected serving shape

```text
one public FastAPI endpoint
internal Python worker routing
CUDA-backed model runtime
diffusers CUDA image backend
```

Static audit verifies packaging, syntax, and wiring. Live readiness verifies CUDA, runtime dependencies, valid model paths, and successful generation.
