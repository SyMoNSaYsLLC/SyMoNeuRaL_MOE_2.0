# Architecture

## Public surface

Only FastAPI is public:

```text
0.0.0.0:8000
```

Main API routes:

```text
GET  /
GET  /health
GET  /ready
GET  /metrics
GET  /v1/models
POST /v1/chat/completions
POST /completion
```

## Internal name-based worker flow

```text
app/main.py
  -> app/symoneural.py
     -> app/workers/gemma.py
     -> app/comms/gemmacodercomms.py
        -> app/workers/qwen.py
     -> app/comms/gemmaimagingcomms.py
        -> app/workers/safetensor.py
     -> app/comms/gemmavisioncomms.py
        -> app/workers/vision.py
```

## Runtime layer

```text
app/runtime/backends/vllm_engine.py   flagship text + VLM backend
app/runtime/text_engine.py            backend selector for Gemma/Qwen
app/runtime/vision_engine.py          backend selector for image input / VLM
app/runtime/image_engine.py           diffusers Stable image backend
app/runtime/gpu.py                    pynvml/nvidia-smi VRAM checks
app/runtime/flow_control.py           SLO-aware queue/admission control
app/runtime/kv_cache.py               prefix-aware routing affinity
app/runtime/vit_encoder.py            optional ViT/CLIP-style embedding helper
```

## Request routing

Gemma decides the worker:

```text
normal chat      -> gemma
code/build/debug -> qwen via gemmacodercomms
image generation -> safetensor via gemmaimagingcomms
image input      -> vision via gemmavisioncomms
```

## GPU ownership

For one 16 GB card, default mode is one heavy active model at a time:

```text
SYMON_SINGLE_ACTIVE_MODEL=1
SYMON_UNLOAD_TEXT_BEFORE_IMAGE=1
```

That avoids the previous class of failures where the web server is online but workers cannot fit in VRAM.
