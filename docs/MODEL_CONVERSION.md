# Model Layout and Conversion

## Runtime format

This build uses **vLLM CUDA**, so Gemma/Qwen/VLM models should stay in Hugging Face directory format with safetensors weights.

Required text/VLM structure:

```text
model-folder/
  config.json
  tokenizer.json or tokenizer.model or tokenizer_config.json
  model.safetensors or model-00001-of-000xx.safetensors
  model.safetensors.index.json when sharded
```

## Download from Hugging Face

```bash
. .venv/bin/activate
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', local_dir='models/llm/qwen')
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='models/vlm/qwen2_5_vl')
PY
```

Gemma models may require Hugging Face license acceptance and auth token.

## HF to GGUF legacy export

GGUF is **not** the runtime format for this vLLM backend. Use GGUF only if you need to export for a separate legacy/runtime target.

Typical external flow:

```bash
export LLAMA_CPP_DIR=$HOME/src/llama.cpp
export HF_MODEL=$HOME/SyMoNeuRaL_Flagship_MoE/models/llm/qwen
export OUT=$HOME/SyMoNeuRaL_Flagship_MoE/models/export/qwen.gguf
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_MODEL" --outfile "$OUT" --outtype f16
```

Quantized legacy export example:

```bash
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$OUT" "${OUT%.gguf}.Q4_K_M.gguf" Q4_K_M
```

Keep this separate from production vLLM deployment. Production vLLM should point at the HF/safetensors directory directly.
