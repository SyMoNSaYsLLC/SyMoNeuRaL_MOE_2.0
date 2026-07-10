from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

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
    "scripts/bootstrap_env.sh",
    "scripts/start.sh",
    "scripts/stop.sh",
    "scripts/status.sh",
    "scripts/test.sh",
    "scripts/install_cuda_vllm.sh",
    "scripts/check_models_magic.sh",
    "tools/hf_download_interactive.py",
    "tools/hf_to_gguf_interactive.py",
    "tools/quantize_gguf_interactive.py",
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

DEPRECATED = [
    "tools/convert_hf_to_gguf_legacy.sh",
    "tools/download_models_example.py",
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
    "app.workers.vram_guard",
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
    path = ROOT / rel
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file() -> bool:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return False

    key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for raw in env_path.read_text(errors="ignore").splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key_re.match(key):
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return True


def env_first(names: list[str], fallback: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return fallback


def model_ref_from_env_or_attr(env_names: list[str], models_obj: Any, attr: str) -> str:
    fallback = str(getattr(models_obj, attr))
    return env_first(env_names, fallback)


def model_refs(models_obj: Any) -> dict[str, str]:
    return {
        "gemma": model_ref_from_env_or_attr(["SYMON_GEMMA_MODEL"], models_obj, "gemma"),
        "qwen": model_ref_from_env_or_attr(["SYMON_QWEN_MODEL"], models_obj, "qwen"),
        "vision": model_ref_from_env_or_attr(
            ["SYMON_VISION_MODEL", "SYMON_VLM_MODEL"],
            models_obj,
            "vision",
        ),
        "stable": model_ref_from_env_or_attr(
            ["SYMON_STABLE_MODEL", "SYMON_IMAGE_MODEL"],
            models_obj,
            "stable",
        ),
    }


def apply_model_refs_to_config(config_module: Any, models_obj: Any, refs: dict[str, str]) -> Any:
    updates = {
        "gemma": refs["gemma"],
        "qwen": refs["qwen"],
        "vision": refs["vision"],
        "stable": refs["stable"],
    }

    if is_dataclass(models_obj):
        try:
            new_models = replace(models_obj, **updates)
            setattr(config_module, "MODELS", new_models)
            return new_models
        except Exception:
            pass

    if hasattr(models_obj, "model_copy"):
        try:
            new_models = models_obj.model_copy(update=updates)
            setattr(config_module, "MODELS", new_models)
            return new_models
        except Exception:
            pass

    for key, value in updates.items():
        try:
            setattr(models_obj, key, value)
        except Exception:
            pass

    try:
        setattr(config_module, "MODELS", models_obj)
    except Exception:
        pass

    return models_obj


def detail_ok(detail: Any) -> bool:
    if isinstance(detail, dict):
        return bool(detail.get("ok"))
    return bool(detail)


def call_ready_detail(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        detail = fn(*args, **kwargs)
        if isinstance(detail, dict):
            return detail
        return {"ok": bool(detail), "detail": detail}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _contains_all(body: str, tokens: tuple[str, ...]) -> bool:
    return all(token in body for token in tokens)


def static_checks() -> None:
    for rel in REQUIRED:
        check(f"exists {rel}", (ROOT / rel).exists())

    for rel in DEPRECATED:
        check(f"deprecated removed {rel}", not (ROOT / rel).exists())

    for base in (ROOT / "app", ROOT / "tools"):
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                ast.parse(py.read_text(), filename=str(py))
                check(f"syntax {py.relative_to(ROOT)}", True)
            except SyntaxError as exc:
                check(
                    f"syntax {py.relative_to(ROOT)}",
                    False,
                    f"{exc.lineno}:{exc.offset} {exc.msg}",
                )

    hits = []
    for base in (ROOT / "app", ROOT / "scripts"):
        if not base.exists():
            continue
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
    req_vllm = text("requirements.vllm.txt")
    req_all = f"{req}\n{req_vllm}"
    readme = text("README.md")
    config = text("app/runtime/config.py")
    vram_guard = text("app/workers/vram_guard.py")

    check(
        "main exposes OpenAI chat completions",
        "/v1/chat/completions" in main and "ChatRequest" in main,
    )
    check(
        "main extracts multimodal OpenAI content",
        "image_url" in main and "extract_user_payload" in main,
    )
    check(
        "symoneural routes text, code, vision, and image-runtime requests",
        all(x in sym for x in ("gemmacodercomms", "gemmaimagingcomms", "gemmavisioncomms"))
        and "vram_profile" in sym
        and "profile_for_decision" in sym,
    )
    check(
        "vLLM is in-process not HTTP server",
        "from vllm import LLM" in vllm and "llm.chat" in vllm and "llm.generate" in vllm,
    )
    check(
        "vLLM prefix caching configured",
        "enable_prefix_caching" in vllm and "SYMON_VLLM_ENABLE_PREFIX_CACHING" in config,
    )
    check(
        "vLLM multimodal path exists",
        "generate_vision" in vllm and "multi_modal_data" in vllm,
    )
    check(
        "flow control enforces concurrency",
        "max_concurrent_text" in flow and "admission_rejected" in flow,
    )
    check(
        "prefix routing tracks hot prefixes",
        "prefix_hash" in kv and "preferred_worker" in kv,
    )
    check(
        "requirements/profile includes vllm/diffusers/nvidia-ml-py/httpx",
        all(x in req_all for x in ("vllm", "diffusers", "nvidia-ml-py", "httpx"))
        and "pynvml" not in req_all,
    )
    check(
        "config defines MoE VRAM split policy",
        _contains_all(
            config,
            (
                "vram_text_gemma_share",
                "vram_text_qwen_share",
                "vram_image_gemma_share",
                "vram_image_runtime_share",
                "vram_param_gemma_share",
                "vram_param_qwen_share",
                "vram_param_runtime_share",
                "parameter_governor",
            ),
        ),
    )
    check(
        "vram guard owns role budget profiles",
        _contains_all(
            vram_guard,
            (
                "VramProfile",
                "profile_for_decision",
                "text_profile",
                "image_profile",
                "parameter_profile",
                "policy_status",
            ),
        ),
    )
    check(
        "README documents CUDA backend",
        ("CUDA-backed" in readme or "vLLM CUDA" in readme)
        and "Vision" in readme
        and "SLO" in readme,
    )
    check(
        "docs include HF to GGUF guide",
        "GGUF" in text("docs/MODEL_CONVERSION.md"),
    )

    deprecated_hits = []
    for base in ("README.md", "docs", "tools", "scripts", "app"):
        path = ROOT / base
        if path.is_file():
            body = path.read_text(errors="ignore")
            for token in ("convert_hf_to_gguf_legacy", "download_models_example"):
                if token in body:
                    deprecated_hits.append(f"{base} contains {token}")
        elif path.is_dir():
            for sub in path.rglob("*"):
                if sub.is_file() and "__pycache__" not in sub.parts:
                    body = sub.read_text(errors="ignore")
                    for token in ("convert_hf_to_gguf_legacy", "download_models_example"):
                        if token in body:
                            deprecated_hits.append(f"{sub.relative_to(ROOT)} contains {token}")

    check("no deprecated tool references", not deprecated_hits, "; ".join(deprecated_hits[:20]))


def import_checks() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for mod in PY_MODULES:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:
            check(f"import {mod}", False, repr(exc))


def _shares_close(values: list[float], expected: list[float], tolerance: float = 0.001) -> bool:
    if len(values) != len(expected):
        return False

    return all(abs(a - b) <= tolerance for a, b in zip(values, expected, strict=True))


def production_checks(live: bool) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from starlette.testclient import TestClient

        from app.main import app
        from app import symoneural
        import app.runtime.config as runtime_config
        from app.runtime.config import MODELS, SETTINGS, resolve_local_path
        from app.runtime.backends import vllm_engine
        from app.runtime import image_engine
        from app.workers import vram_guard
    except Exception as exc:
        check("load app for production checks", False, repr(exc))
        return

    refs = model_refs(MODELS)
    active_models = apply_model_refs_to_config(runtime_config, MODELS, refs)
    refs = model_refs(active_models)

    require_all_workers = bool_from_env(
        "SYMON_REQUIRE_ALL_WORKERS",
        bool(getattr(SETTINGS, "require_all_workers", True)),
    )

    check(
        "audit resolved model refs",
        True,
        json.dumps(
            {
                "gemma": refs["gemma"],
                "qwen": refs["qwen"],
                "vision": refs["vision"],
                "stable": refs["stable"],
                "require_all_workers": require_all_workers,
            }
        ),
    )

    client = TestClient(app)
    if SETTINGS.api_key:
        client.headers.update({"Authorization": f"Bearer {SETTINGS.api_key}"})

    health = client.get("/health")
    check("GET /health returns 200", health.status_code == 200, health.text[:500])

    models = client.get("/v1/models")
    check("GET /v1/models returns 200", models.status_code == 200, models.text[:500])

    deps = vllm_engine.dependency_status()
    check("torch import present", "torch_error" not in deps, str(deps.get("torch_error", "")))
    check("vllm import present", "vllm_error" not in deps, str(deps.get("vllm_error", "")))

    device = os.environ.get("SYMON_DEVICE", str(getattr(SETTINGS, "device", "cpu")))
    if device == "cuda":
        check("CUDA visible to torch", bool(deps.get("cuda_available")), json.dumps(deps))

    gemma_detail = call_ready_detail(vllm_engine.ready_detail, refs["gemma"], multimodal=False)
    if SETTINGS.gemma_enabled:
        check("gemma manager model path ready", detail_ok(gemma_detail), json.dumps(gemma_detail)[:1200])
    else:
        check(
            "gemma manager model optional (disabled)",
            True,
            json.dumps({"required": False, "reason": "SYMON_GEMMA_ENABLED=0", **gemma_detail})[:1200],
        )

    qwen_detail = call_ready_detail(vllm_engine.ready_detail, refs["qwen"], multimodal=False)
    check("qwen specialist model ready", detail_ok(qwen_detail), json.dumps(qwen_detail)[:1200])

    vision_detail = call_ready_detail(vllm_engine.ready_detail, refs["vision"], multimodal=True)

    try:
        vision_path = resolve_local_path(refs["vision"])
        vision_exists = vision_path.exists()
    except Exception:
        vision_path = Path(str(refs["vision"]))
        vision_exists = False

    if require_all_workers or vision_exists:
        check("vision runtime model ready", detail_ok(vision_detail), json.dumps(vision_detail)[:1200])
    else:
        check(
            "vision runtime model optional",
            True,
            json.dumps(
                {
                    "required": False,
                    "reason": "SYMON_REQUIRE_ALL_WORKERS=0 and vision model path is not present",
                    "model_ref": refs["vision"],
                    "resolved_model": str(vision_path),
                }
            )[:1200],
        )

    image_detail = call_ready_detail(image_engine.ready_detail, refs["stable"])
    check("image runtime model ready", detail_ok(image_detail), json.dumps(image_detail)[:1200])

    required_models_ok = (
        (detail_ok(gemma_detail) if SETTINGS.gemma_enabled else True)
        and detail_ok(qwen_detail)
        and detail_ok(image_detail)
        and (detail_ok(vision_detail) if require_all_workers or vision_exists else True)
    )

    check(
        "required model readiness true",
        required_models_ok,
        json.dumps(
            {
                "require_all_workers": require_all_workers,
                "gemma_enabled": SETTINGS.gemma_enabled,
                "gemma": gemma_detail,
                "qwen": qwen_detail,
                "vision": vision_detail,
                "image": image_detail,
            }
        )[:1500],
    )

    policy = vram_guard.policy_status()
    text_profile = vram_guard.text_profile()
    image_profile = vram_guard.image_profile()
    param_profile = vram_guard.parameter_profile()

    check(
        "VRAM target is 13-15 GB",
        13.0 <= SETTINGS.vram_target_gb <= 15.0,
        json.dumps(policy)[:1500],
    )
    check(
        "VRAM hard limit <= 15.2 GB",
        SETTINGS.vram_hard_limit_gb <= 15.2,
        json.dumps(policy)[:1500],
    )
    text_expected = [0.20, 0.80] if SETTINGS.gemma_enabled else [1.0]
    check(
        f"text split is {'20 Gemma / 80 Qwen' if SETTINGS.gemma_enabled else '100 Qwen (Gemma disabled)'}",
        _shares_close([role.share for role in text_profile.roles], text_expected),
        json.dumps(text_profile.as_dict())[:1500],
    )
    image_expected = [0.20, 0.80] if SETTINGS.gemma_enabled else [1.0]
    check(
        f"image split is {'20 Gemma / 80 image runtime' if SETTINGS.gemma_enabled else '100 image runtime (Gemma disabled)'}",
        _shares_close([role.share for role in image_profile.roles], image_expected),
        json.dumps(image_profile.as_dict())[:1500],
    )
    param_expected = [0.25, 0.05, 0.70] if SETTINGS.gemma_enabled else [0.05, 0.95]
    check(
        f"parameter split is {'25 Gemma / 5 Qwen / 70 runtime' if SETTINGS.gemma_enabled else '5 Qwen / 95 runtime (Gemma disabled)'}",
        _shares_close([role.share for role in param_profile.roles], param_expected),
        json.dumps(param_profile.as_dict())[:1500],
    )
    check(
        "parameter governor is qwen",
        vram_guard.parameter_governor() == "qwen",
        json.dumps(policy)[:1500],
    )

    for profile_name, profile in (
        ("manager", vram_guard.manager_profile()),
        ("text", text_profile),
        ("image", image_profile),
        ("vision", vram_guard.vision_profile()),
        ("parameter", param_profile),
    ):
        ok, reason = vram_guard.profile_valid(profile)
        check(f"VRAM profile valid {profile_name}", ok, reason)

    try:
        worker_status = symoneural.worker_status()
        check("worker status available", True, json.dumps(worker_status)[:1500])
    except Exception as exc:
        check("worker status available", False, repr(exc))

    if live:
        body = {
            "model": "symoneural_moe",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Code audit route test. Reply with exactly this Python string literal: "
                        "symoneural_live_ok"
                    ),
                }
            ],
            "max_tokens": 32,
        }

        response = client.post("/v1/chat/completions", json=body)
        check("live specialist text generation returns 200", response.status_code == 200, response.text[:1200])

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            check("live specialist text generation non-empty", bool(content.strip()), content[:200])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="syntax/config/docs wiring only; does not require GPU/models/deps",
    )
    parser.add_argument("--imports", action="store_true", help="also import app modules")
    parser.add_argument("--live", action="store_true", help="run live generation; requires GPU, deps, models")
    args = parser.parse_args()

    env_loaded = load_env_file()

    print("SyMoNeuRaL flagship production audit")
    print("=" * 44)

    check("load .env", True, "loaded .env" if env_loaded else ".env not present")

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
