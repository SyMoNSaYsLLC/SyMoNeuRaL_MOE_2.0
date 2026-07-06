from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import symoneural
from app.runtime import flow_control, gpu, kv_cache
from app.runtime.config import MODELS, OUTPUT, RUNTIME, SETTINGS

# High-frequency polling endpoints (GPU panel, health chip) otherwise flood
# data/runtime/symon-manager.log within seconds, burying real request errors.
# Filtering them out of the access log keeps the log usable for debugging;
# they're still served normally, just not logged on every call.
_QUIET_PATHS = ("/gpu/stats", "/health")


class _QuietPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(f'"GET {path} ' in message for path in _QUIET_PATHS)


logging.getLogger("uvicorn.access").addFilter(_QuietPollingFilter())

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "apps" / "web"
STATIC_ROOT = WEB_ROOT / "static"
UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SyMoNeuRaL Flagship MoE", version="1.0.0")
if STATIC_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")

# Paths that stay open even when SYMON_API_KEY is set. Browsers can't attach
# an Authorization header to plain <img>/<link> requests, so the assets the
# page itself needs to render (index page, static JS/CSS, generated images,
# favicons) must stay reachable without a token, or the UI breaks. Everything
# else -- the actual API surface -- requires the key.
_AUTH_EXEMPT_PREFIXES = ("/static", "/outputs", "/uploads")
_AUTH_EXEMPT_EXACT = {"/", "/favicon.ico", "/favicon.svg", "/favicon-dark.ico", "/favicon-dark.svg",
                      "/apple-touch-icon.png", "/apple-touch-icon-180x180.png"}


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if not SETTINGS.api_key:
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT_EXACT or any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES):
        return await call_next(request)

    header = request.headers.get("authorization", "")
    expected = f"Bearer {SETTINGS.api_key}"
    if header != expected:
        return PlainTextResponse("unauthorized", status_code=401)

    return await call_next(request)


@app.on_event("startup")
async def _warm_up_qwen() -> None:
    # Load Qwen in the background the moment the server boots, instead of
    # waiting for the first real visitor's message to trigger the cold load.
    # Runs as a background task so it doesn't delay "Uvicorn running on...".
    #
    # Deliberately calls text_engine.generate() directly instead of going
    # through qwen.chat() -- chat() acquires flow_control's admission slot
    # (SYMON_MAX_CONCURRENT_TEXT, default 1), and holding that slot for the
    # full ~20-30s cold load would flatly REJECT any real request that
    # arrives during warm-up with a 503, which is worse than the old
    # behavior (a real early request just took a while and then succeeded).
    # Bypassing it here means: if a real request races in during warm-up, it
    # correctly *waits* on vllm_engine's own load lock (shared, one-time
    # cost) instead of being rejected outright.
    import asyncio

    async def _run() -> None:
        try:
            from app.runtime import text_engine
            from app.runtime.config import MODELS, SETTINGS
            from app.workers.qwen import CHAT_SYSTEM
            await run_in_threadpool(
                text_engine.generate,
                name="qwen",
                model_ref=MODELS.qwen,
                system=CHAT_SYSTEM,
                prompt="Hi",
                max_new_tokens=4,
                temperature=SETTINGS.temperature_chat,
            )
            print("[startup] Qwen warm-up complete.")
        except Exception as exc:
            print(f"[startup] Qwen warm-up failed (non-fatal): {exc!r}")

    asyncio.create_task(_run())


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = "symoneural_moe"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    stream: bool | None = False
    temperature: float | None = None
    metadata: dict[str, Any] | None = Field(default=None)


def _save_data_url(data_url: str) -> str:
    header, payload = data_url.split(",", 1)
    ext = "png"
    if "image/jpeg" in header:
        ext = "jpg"
    elif "image/webp" in header:
        ext = "webp"
    path = UPLOADS / f"input_{uuid.uuid4().hex}.{ext}"
    path.write_bytes(base64.b64decode(payload))
    return str(path)


def _resolve_image_url(url: str) -> str:
    url = url.strip()
    if url.startswith("data:image/"):
        return _save_data_url(url)
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        return str(Path(parsed.path or url).expanduser())
    # Do not fetch remote images inside production server. Download explicitly before submit.
    raise HTTPException(400, f"remote_image_url_not_allowed:{parsed.scheme}")


def extract_user_payload(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    prompt_parts: list[str] = []
    image_paths: list[str] = []
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            prompt_parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                typ = item.get("type")
                if typ in {"text", "input_text"}:
                    prompt_parts.append(str(item.get("text", "")).strip())
                elif typ in {"image_url", "input_image"}:
                    raw = item.get("image_url") or item.get("image") or item.get("url")
                    if isinstance(raw, dict):
                        raw = raw.get("url")
                    if raw:
                        image_paths.append(_resolve_image_url(str(raw)))
        elif isinstance(content, dict):
            text = content.get("text") or content.get("prompt")
            if text:
                prompt_parts.append(str(text).strip())
        break
    prompt = "\n".join(x for x in prompt_parts if x).strip()
    return prompt, image_paths


@app.get("/")
async def root():
    index = WEB_ROOT / "index.html"
    if not index.exists():
        raise HTTPException(500, "web_index_missing")
    return FileResponse(index)


_ICON_FILES = {
    "/favicon.ico": ("favicon.ico", "image/x-icon"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/favicon-dark.ico": ("favicon-dark.ico", "image/x-icon"),
    "/favicon-dark.svg": ("favicon-dark.svg", "image/svg+xml"),
    "/apple-touch-icon.png": ("apple-touch-icon-180x180.png", "image/png"),
    "/apple-touch-icon-180x180.png": ("apple-touch-icon-180x180.png", "image/png"),
}


def _icon_route(filename: str, media_type: str):
    async def _serve():
        path = WEB_ROOT / filename
        if not path.exists():
            raise HTTPException(404, f"icon_missing:{filename}")
        return FileResponse(path, media_type=media_type)
    return _serve


for _route_path, (_filename, _media_type) in _ICON_FILES.items():
    app.add_api_route(_route_path, _icon_route(_filename, _media_type), methods=["GET"], include_in_schema=False)


@app.get("/health")
async def health():
    workers = symoneural.worker_readiness()
    worker_status = symoneural.worker_status()
    ok = all(workers.values())
    return {
        "ok": ok,
        "status": "online" if ok else "degraded",
        "service": SETTINGS.service_name,
        "gpu": gpu.status(),
        "workers": workers,
        "worker_status": worker_status,
        "flow": flow_control.controller.snapshot(),
        "kv_cache": kv_cache.snapshot(),
        "runtime": {
            "text_backend": SETTINGS.text_backend,
            "vision_backend": SETTINGS.vision_backend,
            "image_backend": SETTINGS.image_backend,
            "device": SETTINGS.device,
            "single_active_model": SETTINGS.single_active_model,
            "models": {
                "gemma": MODELS.gemma,
                "qwen": MODELS.qwen,
                "vision": MODELS.vision,
                "stable": MODELS.stable,
            },
        },
    }


@app.get("/gpu/stats")
async def gpu_stats():
    return gpu.status()


@app.get("/ready")
async def ready():
    h = await health()
    if not h["ok"]:
        raise HTTPException(503, h)
    return h


@app.get("/metrics")
async def metrics():
    g = gpu.status()
    f = flow_control.controller.snapshot()
    k = kv_cache.snapshot()
    lines = [
        f"symon_gpu_vram_used_gb {g.get('used_gb', 0)}",
        f"symon_gpu_vram_total_gb {g.get('total_gb', 0)}",
        f"symon_flow_queue_depth {f['queue_depth']}",
        f"symon_flow_rejected_total {f['rejected']}",
        f"symon_kv_prefix_entries {k['entries']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "symoneural_moe", "object": "model", "backend": SETTINGS.text_backend},
            {"id": "gemma", "object": "model", "path": MODELS.gemma},
            {"id": "qwen", "object": "model", "path": MODELS.qwen},
            {"id": "vision", "object": "model", "path": MODELS.vision},
            {"id": "safetensor", "object": "model", "path": MODELS.stable},
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    if req.stream:
        raise HTTPException(400, "streaming_not_enabled_for_inprocess_backend")
    prompt, images = extract_user_payload(req.messages)
    if not prompt and not images:
        raise HTTPException(400, "missing_user_prompt")
    if not prompt and images:
        prompt = "Describe this image and answer any obvious question about it."
    started = time.monotonic()
    try:
        worker, content, metadata = await run_in_threadpool(
            symoneural.answer,
            prompt,
            req.max_tokens or SETTINGS.max_new_tokens_default,
            images,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "id": f"symon-chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": worker,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "symoneural": {
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "worker": worker,
            "images": len(images),
            "client": request.client.host if request.client else None,
            **metadata,
        },
    }


@app.post("/completion")
async def legacy_completion(req: dict[str, Any]):
    prompt = str(req.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(400, "missing_prompt")
    worker, content, metadata = await run_in_threadpool(
        symoneural.answer,
        prompt,
        int(req.get("max_tokens") or SETTINGS.max_new_tokens_default),
        [],
    )
    return {"content": content, "model": worker, "symoneural": metadata}


@app.get("/props")
async def props():
    return {"service": SETTINGS.service_name, "backend": SETTINGS.text_backend, "device": SETTINGS.device}


@app.get("/slots")
async def slots():
    return flow_control.controller.snapshot()


@app.get("/tools")
async def tools():
    return {"tools": ["gemma", "qwen", "vision", "safetensor"], "public_endpoint": ":8000"}


@app.post("/v1/streams/lookup")
async def streams_lookup(req: dict[str, Any]):
    return {"streaming": False, "reason": "single-process in-process backend; enable streaming after vLLM async engine integration"}
