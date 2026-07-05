from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
REQUIRED = [
    "app/main.py",
    "app/symoneural.py",
    "app/runtime/config.py",
    "app/runtime/gpu.py",
    "app/runtime/flow_control.py",
    "app/runtime/kv_cache.py",
    "app/runtime/text_engine.py",
    "app/runtime/vision_engine.py",
    "app/runtime/image_engine.py",
    "app/runtime/backends/vllm_engine.py",
    "app/runtime/vit_encoder.py",
    "app/runtime/worker_types.py",
    "app/workers/gemma.py",
    "app/workers/qwen.py",
    "app/workers/safetensor.py",
    "app/workers/vision.py",
    "app/workers/vram_guard.py",
    "app/comms/gemmacodercomms.py",
    "app/comms/gemmaimagingcomms.py",
    "app/comms/gemmavisioncomms.py",
    "scripts/start.sh",
    "scripts/stop.sh",
    "scripts/status.sh",
    "scripts/test.sh",
    "scripts/install_cuda_vllm.sh",
    "README.md",
    "requirements.txt",
    "docs/ARCHITECTURE.md",
    "docs/INSTALL_CUDA_VLLM.md",
    "docs/NVIDIA_HARDWARE.md",
    "docs/MODEL_TUNING.md",
    "docs/MODEL_CONVERSION.md",
    "docs/SLO_FLOW_CONTROL.md",
    "docs/KV_CACHE_PREFIX_ROUTING.md",
    "docs/MULTIMODAL_VISION.md",
    "docs/PRODUCTION_RUNBOOK.md",
]
PY_MODULES = [
    "app.main",
    "app.symoneural",
    "app.runtime.config",
    "app.runtime.gpu",
    "app.runtime.flow_control",
    "app.runtime.kv_cache",
    "app.runtime.text_engine",
    "app.runtime.vision_engine",
    "app.runtime.image_engine",
    "app.runtime.backends.vllm_engine",
    "app.workers.gemma",
    "app.workers.qwen",
    "app.workers.safetensor",
    "app.workers.vision",
    "app.comms.gemmacodercomms",
    "app.comms.gemmaimagingcomms",
    "app.comms.gemmavisioncomms",
]
RUNTIME_FORBIDDEN = [
    "llama-server",
    "symon-llm-server",
    "stable-diffusion.cpp",
    "http://127.0.0.1:18080",
    "http://127.0.0.1:18081",
    "http://127.0.0.1:18082",
]
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def text(rel: str) -> str:
    return (ROOT / rel).read_text(errors="ignore")


def static_checks() -> None:
    for rel in REQUIRED:
        check(f"exists {rel}", (ROOT / rel).exists())

    for py in (ROOT / "app").rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            ast.parse(py.read_text(), filename=str(py))
            check(f"syntax {py.relative_to(ROOT)}", True)
        except SyntaxError as exc:
            check(f"syntax {py.relative_to(ROOT)}", False, f"{exc.lineno}:{exc.offset} {exc.msg}")

    hits = []
    for base in (ROOT / "app", ROOT / "scripts"):
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                body = path.read_text(errors="ignore")
                for token in RUNTIME_FORBIDDEN:
                    if token in body:
                        hits.append(f"{path.relative_to(ROOT)} contains {token}")
    check("no stale local worker servers", not hits, "; ".join(hits[:20]))

    main = text("app/main.py")
    sym = text("app/symoneural.py")
    vllm = text("app/runtime/backends/vllm_engine.py")
    flow = text("app/runtime/flow_control.py")
    kv = text("app/runtime/kv_cache.py")
    req = text("requirements.txt")
    readme = text("README.md")

    check("main exposes OpenAI chat completions", "/v1/chat/completions" in main and "ChatRequest" in main)
    check("main extracts multimodal OpenAI content", "image_url" in main and "extract_user_payload" in main)
    check("symoneural routes all four workers", all(x in sym for x in ("gemmacodercomms", "gemmaimagingcomms", "gemmavisioncomms")))
    check("vLLM is in-process not HTTP server", "from vllm import LLM" in vllm and "llm.chat" in vllm and "llm.generate" in vllm)
    check("vLLM prefix caching configured", "enable_prefix_caching" in vllm and "SYMON_VLLM_ENABLE_PREFIX_CACHING" in text("app/runtime/config.py"))
    check("vLLM multimodal path exists", "generate_vision" in vllm and "multi_modal_data" in vllm)
    check("flow control enforces concurrency", "max_concurrent_text" in flow and "admission_rejected" in flow)
    check("prefix routing tracks hot prefixes", "prefix_hash" in kv and "preferred_worker" in kv)
    check("requirements includes vllm/diffusers/pynvml", all(x in req for x in ("vllm", "diffusers", "pynvml")))
    check("README documents flagship backend", "vLLM CUDA" in readme and "Vision" in readme and "SLO" in readme)
    check("docs include HF to GGUF guide", "GGUF" in text("docs/MODEL_CONVERSION.md"))


def import_checks() -> None:
    sys.path.insert(0, str(ROOT))
    for mod in PY_MODULES:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:
            check(f"import {mod}", False, repr(exc))


def production_checks(live: bool) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        from app import symoneural
        from app.runtime.config import MODELS, SETTINGS, resolve_local_path
        from app.runtime.backends import vllm_engine
        from app.runtime import image_engine
    except Exception as exc:
        check("load app for production checks", False, repr(exc))
        return

    client = TestClient(app)
    health = client.get("/health")
    check("GET /health returns 200", health.status_code == 200, health.text[:500])
    models = client.get("/v1/models")
    check("GET /v1/models returns 200", models.status_code == 200, models.text[:500])

    deps = vllm_engine.dependency_status()
    check("torch import present", "torch_error" not in deps, str(deps.get("torch_error", "")))
    check("vllm import present", "vllm_error" not in deps, str(deps.get("vllm_error", "")))
    if SETTINGS.device == "cuda":
        check("CUDA visible to torch", bool(deps.get("cuda_available")), json.dumps(deps))

    for name, ref in ("gemma", MODELS.gemma), ("qwen", MODELS.qwen), ("vision", MODELS.vision):
        detail = vllm_engine.ready_detail(ref, multimodal=(name == "vision"))
        check(f"{name} vLLM model ready", bool(detail.get("ok")), json.dumps(detail)[:1200])
    img = image_engine.ready_detail(MODELS.stable)
    check("image model ready", bool(img.get("ok")), json.dumps(img)[:1200])

    workers = symoneural.worker_readiness()
    check("worker readiness map all true", all(workers.values()), json.dumps(symoneural.worker_status())[:1500])

    if live:
        body = {"model": "symoneural_moe", "messages": [{"role": "user", "content": "Reply with exactly: symoneural_live_ok"}], "max_tokens": 16}
        response = client.post("/v1/chat/completions", json=body, timeout=180)
        check("live text generation returns 200", response.status_code == 200, response.text[:1200])
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            check("live text generation non-empty", bool(content.strip()), content[:200])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true", help="syntax/config/docs wiring only; does not require GPU/models/deps")
    parser.add_argument("--imports", action="store_true", help="also import app modules")
    parser.add_argument("--live", action="store_true", help="run live generation; requires GPU, deps, models")
    args = parser.parse_args()

    print("SyMoNeuRaL flagship production audit")
    print("=" * 44)
    static_checks()
    if args.imports or not args.static_only:
        import_checks()
    if not args.static_only:
        production_checks(live=args.live)

    failed = 0
    for name, ok, detail in checks:
        print(("PASS" if ok else "FAIL"), name)
        if not ok:
            failed += 1
            if detail:
                print("      ", detail)
    print("=" * 44)
    print(f"Total: {len(checks)} checks | Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
