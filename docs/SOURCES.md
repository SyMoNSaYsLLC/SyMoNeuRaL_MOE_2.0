# Sources and Merge Notes

Merged uploaded project zips:

- `SyMoNeRaL_Clean-vllm-cuda-backend.zip`
- `SyMoNeRaL_Clean-main.zip`
- `SyMoNeuRaL_Prod-main.zip`

Base selected: the vLLM CUDA backend zip, because it had the most recent worker/runtime/audit direction.

Carried forward:

- web app assets from Clean/Prod sources
- single FastAPI public service
- name-based Gemma/Qwen/safetensor workers
- production hard-fail audit direction

Added in this merged folder:

- vLLM text backend cleanup and config filtering
- vLLM multimodal VLM path
- optional ViT encoder helper
- SLO flow-control controller
- prefix-aware routing affinity layer
- GPU/VRAM status via pynvml/nvidia-smi
- full docs/runbooks/model tuning/conversion notes

Official reference links to keep current:

- vLLM GPU install: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- vLLM automatic prefix caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- vLLM multimodal inputs: https://docs.vllm.ai/en/latest/features/multimodal_inputs/
- TensorRT-LLM docs: https://nvidia.github.io/TensorRT-LLM/
- CUTLASS docs: https://docs.nvidia.com/cutlass/
- NVIDIA Blackwell compatibility guide: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
