#!/usr/bin/env bash
set -euo pipefail
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to an external llama.cpp checkout}"
: "${HF_MODEL:?set HF_MODEL to a local HF model directory}"
: "${OUT:?set OUT to output gguf path}"
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_MODEL" --outfile "$OUT" --outtype "${OUTTYPE:-f16}"
if [ -n "${QUANT:-}" ]; then
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$OUT" "${OUT%.gguf}.${QUANT}.gguf" "$QUANT"
fi
