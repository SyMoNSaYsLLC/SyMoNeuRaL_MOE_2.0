# NVIDIA Hardware Notes

## Tiers

| Tier | Example GPUs | Recommended use |
|---|---|---|
| 16 GB RTX | RTX 4070 Ti SUPER, RTX 5070 Ti class | one active model at a time, 2B-7B models, image on demand |
| 24 GB RTX | RTX 4090 / 5090 class | 7B text with larger context, small VLM, image on demand |
| 48 GB workstation | RTX 6000 Ada / RTX PRO class | multiple active workers, larger VLM, higher queue |
| 80 GB datacenter | A100/H100/H200/B200 class | tensor parallel, higher concurrency, disaggregated prefill/decode |

## 16 GB defaults

```bash
SYMON_VLLM_GPU_MEMORY_UTILIZATION=0.82
SYMON_VLLM_MAX_MODEL_LEN=8192
SYMON_VRAM_TARGET_GB=14.0
SYMON_VRAM_HARD_LIMIT_GB=15.2
SYMON_SINGLE_ACTIVE_MODEL=1
SYMON_MAX_CONCURRENT_TEXT=1
SYMON_MAX_CONCURRENT_IMAGE=1
SYMON_MAX_CONCURRENT_VISION=1
```

## Larger GPU direction

For 24 GB+:

```bash
SYMON_VLLM_GPU_MEMORY_UTILIZATION=0.88
SYMON_VLLM_MAX_MODEL_LEN=16384
SYMON_MAX_CONCURRENT_TEXT=2
```

For 48 GB+:

```bash
SYMON_SINGLE_ACTIVE_MODEL=0
SYMON_VLLM_MAX_NUM_SEQS=32
SYMON_VLLM_MAX_NUM_BATCHED_TOKENS=16384
```

For 80 GB datacenter:

- consider tensor parallel
- consider vLLM serving replicas behind a router
- consider prefill/decode separation
- consider TensorRT-LLM benchmarking as an alternate backend

## Low-level NVIDIA stack references

- CUDA runtime and driver provide GPU execution and memory control.
- cuBLAS/cuBLASLt provide GEMM/matmul backbone.
- CUTLASS is the source-level CUDA C++ template path for custom kernels.
- vLLM is the chosen runtime engine here because it already implements CUDA model execution, paged KV cache behavior, prefix caching, scheduling, and supported VLM paths.
