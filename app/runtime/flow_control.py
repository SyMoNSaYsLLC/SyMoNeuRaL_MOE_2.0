from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from app.runtime.config import SETTINGS
from app.runtime import gpu


@dataclass
class FlowStats:
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    in_flight_text: int = 0
    in_flight_image: int = 0
    in_flight_vision: int = 0
    queue_depth: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class AdmissionController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.stats = FlowStats()

    def _limit_for(self, kind: str) -> int:
        if kind == "image":
            return SETTINGS.max_concurrent_image
        if kind == "vision":
            return SETTINGS.max_concurrent_vision
        return SETTINGS.max_concurrent_text

    def _in_flight_for(self, kind: str) -> int:
        if kind == "image":
            return self.stats.in_flight_image
        if kind == "vision":
            return self.stats.in_flight_vision
        return self.stats.in_flight_text

    def _inc(self, kind: str) -> None:
        if kind == "image":
            self.stats.in_flight_image += 1
        elif kind == "vision":
            self.stats.in_flight_vision += 1
        else:
            self.stats.in_flight_text += 1

    def _dec(self, kind: str) -> None:
        if kind == "image":
            self.stats.in_flight_image = max(0, self.stats.in_flight_image - 1)
        elif kind == "vision":
            self.stats.in_flight_vision = max(0, self.stats.in_flight_vision - 1)
        else:
            self.stats.in_flight_text = max(0, self.stats.in_flight_text - 1)

    def admit(self, kind: str) -> tuple[bool, str]:
        with self._lock:
            g = gpu.status()
            if g.get("critical"):
                self.stats.rejected += 1
                return False, f"gpu_vram_hard_limit:{g}"
            if self.stats.queue_depth >= SETTINGS.max_queue:
                self.stats.rejected += 1
                return False, "queue_full"
            if self._in_flight_for(kind) >= self._limit_for(kind):
                self.stats.rejected += 1
                return False, f"{kind}_concurrency_limit"
            self.stats.accepted += 1
            self.stats.queue_depth += 1
            self._inc(kind)
            return True, "ok"

    def finish(self, kind: str, started: float) -> None:
        with self._lock:
            self._dec(kind)
            self.stats.queue_depth = max(0, self.stats.queue_depth - 1)
            self.stats.completed += 1
            self.stats.latencies_ms.append((time.monotonic() - started) * 1000)
            if len(self.stats.latencies_ms) > 512:
                self.stats.latencies_ms = self.stats.latencies_ms[-512:]

    @contextmanager
    def job(self, kind: str) -> Iterator[None]:
        ok, reason = self.admit(kind)
        if not ok:
            raise RuntimeError(f"admission_rejected:{reason}")
        started = time.monotonic()
        try:
            yield
        finally:
            self.finish(kind, started)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.stats.latencies_ms)
            p95 = None
            if latencies:
                latencies_sorted = sorted(latencies)
                p95 = latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))]
            return {
                "accepted": self.stats.accepted,
                "rejected": self.stats.rejected,
                "completed": self.stats.completed,
                "in_flight": {
                    "text": self.stats.in_flight_text,
                    "image": self.stats.in_flight_image,
                    "vision": self.stats.in_flight_vision,
                },
                "queue_depth": self.stats.queue_depth,
                "latency_p95_ms": round(p95, 2) if p95 is not None else None,
                "slo": {
                    "ttft_ms": SETTINGS.slo_ttft_ms,
                    "itl_ms": SETTINGS.slo_itl_ms,
                    "image_ms": SETTINGS.slo_image_ms,
                },
            }


controller = AdmissionController()
