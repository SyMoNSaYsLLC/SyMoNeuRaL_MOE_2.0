from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from app.runtime.config import SETTINGS


@dataclass
class PrefixHint:
    prefix_hash: str
    chars: int
    seen_count: int
    last_seen: float
    preferred_worker: str | None = None


_PREFIXES: dict[str, PrefixHint] = {}


def prefix_key(prompt: str) -> str:
    prefix = prompt[: SETTINGS.prefix_cache_chars]
    return hashlib.sha256(prefix.encode("utf-8", "ignore")).hexdigest()[:24]


def observe(prompt: str, worker: str | None = None) -> PrefixHint:
    now = time.time()
    key = prefix_key(prompt)
    hint = _PREFIXES.get(key)
    if hint is None:
        hint = PrefixHint(prefix_hash=key, chars=min(len(prompt), SETTINGS.prefix_cache_chars), seen_count=0, last_seen=now)
    hint.seen_count += 1
    hint.last_seen = now
    if worker:
        hint.preferred_worker = worker
    _PREFIXES[key] = hint
    cleanup(now)
    return hint


def cleanup(now: float | None = None) -> None:
    now = now or time.time()
    expiry = SETTINGS.prefix_route_affinity_seconds
    stale = [key for key, hint in _PREFIXES.items() if now - hint.last_seen > expiry]
    for key in stale:
        _PREFIXES.pop(key, None)


def snapshot() -> dict:
    cleanup()
    return {
        "enabled": SETTINGS.vllm_enable_prefix_caching,
        "prefix_cache_chars": SETTINGS.prefix_cache_chars,
        "affinity_seconds": SETTINGS.prefix_route_affinity_seconds,
        "entries": len(_PREFIXES),
        "hot_prefixes": [
            {
                "prefix_hash": h.prefix_hash,
                "seen_count": h.seen_count,
                "preferred_worker": h.preferred_worker,
                "chars": h.chars,
            }
            for h in sorted(_PREFIXES.values(), key=lambda item: item.seen_count, reverse=True)[:10]
        ],
        "note": "Actual KV block storage is owned by vLLM. This table is SyMoNeuRaL's prefix-aware routing/affinity layer.",
    }
