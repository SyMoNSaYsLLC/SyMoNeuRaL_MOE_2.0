# Production Review Notes

This merged folder is the canonical project output for this pass.

## Selected base

The vLLM CUDA backend source was selected as the base because it was the closest to the required production architecture.

## What was removed/avoided

- no separate localhost LLM worker servers
- no old 18080/18081/18082 worker process contract
- no stale pycache/runtime files
- no duplicate legacy text runtime files

## What was added

- in-process vLLM CUDA text backend
- in-process vLLM VLM/multimodal backend
- diffusers image backend
- optional ViT encoder helper
- SLO-aware admission control
- prefix-aware routing state
- hard production audit
- full docs and runbooks

## Verification performed in this sandbox

Passed:

```text
python3 production_audit.py --static-only
python3 production_audit.py --static-only --imports
```

Full production audit correctly fails in this sandbox because this environment has CPU-only torch, no NVIDIA GPU, no vLLM package, and no model weights.
