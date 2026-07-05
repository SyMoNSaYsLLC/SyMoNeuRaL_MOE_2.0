# Model Tuning Guide

## First production target on 16 GB

Use small-but-real models first:

```bash
SYMON_GEMMA_MODEL=models/llm/gemma-2-2b-it
SYMON_QWEN_MODEL=models/llm/qwen2.5-coder-1.5b-instruct
SYMON_VISION_MODEL=models/vlm/qwen2.5-vl-3b-instruct
```

Keep one active model:

```bash
SYMON_SINGLE_ACTIVE_MODEL=1
```

## 7B text on 16 GB

Possible, but tight:

```bash
SYMON_VLLM_GPU_MEMORY_UTILIZATION=0.88
SYMON_VLLM_MAX_MODEL_LEN=4096
SYMON_VLLM_CPU_OFFLOAD_GB=4
SYMON_VLLM_MAX_NUM_SEQS=4
```

For code generation, reduce context before increasing output tokens.

## Prefix cache

Enable:

```bash
SYMON_VLLM_ENABLE_PREFIX_CACHING=1
SYMON_PREFIX_CACHE_CHARS=4096
```

This helps repeated system prompts, repo audits, and repeated build context.

## Image generation

On 16 GB:

```bash
SYMON_UNLOAD_TEXT_BEFORE_IMAGE=1
SYMON_IMAGE_WIDTH=512
SYMON_IMAGE_HEIGHT=512
SYMON_IMAGE_STEPS=20
```

Raise steps/resolution only after `nvidia-smi` shows safe headroom.

## Multimodal / VLM

Start with a small VLM. For larger VLMs, cap images:

```bash
SYMON_VLLM_LIMIT_MM_IMAGES=1
SYMON_VLLM_MAX_MODEL_LEN=4096
```

## Production symptoms

| Symptom | Likely cause | Fix |
|---|---|---|
| `cuda_unavailable` | PyTorch cannot see GPU | driver/PyTorch wheel mismatch |
| `model_path_missing` | env path wrong | set `SYMON_*_MODEL` correctly |
| OOM during image | text model still loaded | keep `SYMON_UNLOAD_TEXT_BEFORE_IMAGE=1` |
| slow first token | prefill/context too large | reduce context or use prefix caching |
| queue rejected | SLO flow control engaged | reduce concurrent clients or increase limits on larger GPU |
