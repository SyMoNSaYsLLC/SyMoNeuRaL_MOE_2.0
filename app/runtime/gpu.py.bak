from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from app.runtime.config import SETTINGS


@dataclass(frozen=True)
class GpuStatus:
    ok: bool
    used_gb: float
    total_gb: float
    free_gb: float
    target_gb: float
    hard_limit_gb: float
    critical: bool
    devices: list[dict[str, Any]]
    reason: str = "ok"


def _from_pynvml() -> list[dict[str, Any]]:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    devices = []
    for idx in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        devices.append({
            "index": idx,
            "name": str(name),
            "used_gb": round(mem.used / 1024**3, 3),
            "total_gb": round(mem.total / 1024**3, 3),
            "free_gb": round(mem.free / 1024**3, 3),
        })
    return devices


def _from_nvidia_smi() -> list[dict[str, Any]]:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    devices = []
    for line in out.strip().splitlines():
        idx, name, used, total, free = [x.strip() for x in line.split(",", 4)]
        devices.append({
            "index": int(idx),
            "name": name,
            "used_gb": round(float(used) / 1024, 3),
            "total_gb": round(float(total) / 1024, 3),
            "free_gb": round(float(free) / 1024, 3),
        })
    return devices


def devices() -> list[dict[str, Any]]:
    try:
        return _from_pynvml()
    except Exception:
        try:
            return _from_nvidia_smi()
        except Exception:
            return []


def status() -> dict[str, Any]:
    devs = devices()
    if not devs:
        return GpuStatus(
            ok=SETTINGS.device != "cuda",
            used_gb=0.0,
            total_gb=0.0,
            free_gb=0.0,
            target_gb=SETTINGS.vram_target_gb,
            hard_limit_gb=SETTINGS.vram_hard_limit_gb,
            critical=SETTINGS.device == "cuda",
            devices=[],
            reason="no_nvidia_gpu_detected",
        ).__dict__

    primary = devs[0]
    used = float(primary["used_gb"])
    total = float(primary["total_gb"])
    free = float(primary["free_gb"])
    critical = used >= SETTINGS.vram_hard_limit_gb
    ok = used < SETTINGS.vram_target_gb and not critical
    return GpuStatus(
        ok=ok,
        used_gb=round(used, 3),
        total_gb=round(total, 3),
        free_gb=round(free, 3),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        critical=critical,
        devices=devs,
        reason="ok" if ok else "vram_pressure",
    ).__dict__


def assert_safe_for_job(job: str) -> None:
    s = status()
    if s["critical"]:
        raise RuntimeError(f"gpu_vram_hard_limit:{job}:{s}")
