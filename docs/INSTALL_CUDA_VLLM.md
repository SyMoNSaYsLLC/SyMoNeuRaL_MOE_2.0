# Install CUDA vLLM Backend

## Required base

- Linux x86_64
- NVIDIA driver visible through `nvidia-smi`
- Python 3.10+ recommended
- CUDA-compatible PyTorch/vLLM wheels

## Install

```bash
cd SyMoNeuRaL_Flagship_MoE
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Or run:

```bash
scripts/install_cuda_vllm.sh
```

## Verify CUDA

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
```

## Verify vLLM

```bash
python3 - <<'PY'
import vllm
print(vllm.__version__)
PY
```

## Model paths

Production is local-only by default:

```bash
SYMON_ALLOW_REMOTE_MODELS=0
```

Use local folders:

```text
models/llm/gemma/config.json
models/llm/gemma/tokenizer.json or tokenizer.model
models/llm/gemma/*.safetensors

models/llm/qwen/config.json
models/llm/qwen/tokenizer.json or tokenizer.model
models/llm/qwen/*.safetensors

models/vlm/qwen2_5_vl/config.json
models/vlm/qwen2_5_vl/tokenizer.json or tokenizer.model
models/vlm/qwen2_5_vl/*.safetensors

models/sd/stable-diffusion/model_index.json
```

Remote bootstrap only:

```bash
export SYMON_ALLOW_REMOTE_MODELS=1
export SYMON_GEMMA_MODEL=google/gemma-2-2b-it
export SYMON_QWEN_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
export SYMON_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

Then switch back to local-only after models are cached/downloaded.
