from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WorkerName = Literal["gemma", "qwen", "safetensor", "vision"]
TaskKind = Literal["chat", "code", "image", "vision"]


@dataclass(frozen=True)
class WorkerDecision:
    worker: WorkerName
    task: str
    kind: TaskKind
    reason: str


@dataclass
class WorkerResult:
    worker: str
    content: str
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestContext:
    prompt: str
    max_tokens: int = 768
    request_id: str | None = None
    images: tuple[str, ...] = ()
    client: str | None = None


@dataclass(frozen=True)
class Readiness:
    ok: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)
