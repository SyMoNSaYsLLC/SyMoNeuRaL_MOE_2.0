# Multimodal Vision

## Flagship path

The primary multimodal path is a vLLM-supported Vision Language Model:

```text
app/workers/vision.py
  -> app/runtime/vision_engine.py
  -> app/runtime/backends/vllm_engine.py
  -> vLLM VLM CUDA backend
```

Suggested first VLM:

```bash
SYMON_VISION_MODEL=models/vlm/qwen2_5_vl
```

Bootstrap option:

```bash
SYMON_ALLOW_REMOTE_MODELS=1
SYMON_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

## OpenAI-style image input

The `/v1/chat/completions` route accepts content lists containing text and `image_url` entries. Local paths and `data:image/...` URLs are accepted. Remote image URLs are rejected by default so the production server does not fetch arbitrary external content.

Example content shape:

```json
{
  "model": "symoneural_moe",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "file:///home/google/image.png"}}
    ]
  }]
}
```

## Optional ViT encoder

`app/runtime/vit_encoder.py` is included for future image embeddings, semantic routing, or image similarity. It is not the main chat/VLM answer path.

```bash
SYMON_VIT_ENCODER_MODEL=models/vlm/vit-encoder
```

For production answers about images, use the vLLM VLM path.
