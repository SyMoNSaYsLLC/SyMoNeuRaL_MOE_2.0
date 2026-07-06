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

        util_pct = None
        try:
            util_pct = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            pass

        temp_c = None
        try:
            temp_c = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass

        power_w = None
        try:
            power_w = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
        except Exception:
            pass

        power_limit_w = None
        try:
            power_limit_w = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1)
        except Exception:
            pass

        devices.append({
            "index": idx,
            "name": str(name),
            "used_gb": round(mem.used / 1024**3, 3),
            "total_gb": round(mem.total / 1024**3, 3),
            "free_gb": round(mem.free / 1024**3, 3),
            "utilization_pct": util_pct,
            "temperature_c": temp_c,
            "power_w": power_w,
            "power_limit_w": power_limit_w,
        })
    return devices


def _from_nvidia_smi() -> list[dict[str, Any]]:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    devices = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",", 8)]
        idx, name, used, total, free, util, temp, power, power_limit = parts

        def _num(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        devices.append({
            "index": int(idx),
            "name": name,
            "used_gb": round(float(used) / 1024, 3),
            "total_gb": round(float(total) / 1024, 3),
            "free_gb": round(float(free) / 1024, 3),
            "utilization_pct": int(_num(util)) if _num(util) is not None else None,
            "temperature_c": int(_num(temp)) if _num(temp) is not None else None,
            "power_w": _num(power),
            "power_limit_w": _num(power_limit),
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
