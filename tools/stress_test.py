#!/usr/bin/env python3
"""
SyMoNeuRaL stress test
=======================

production_audit.py --live proves the wiring works with ONE request. This
tool answers a different question: does it hold up under realistic,
REPEATED use? Specifically, given today's VRAM-release fixes, the thing
worth actually verifying is whether VRAM comes back down to a stable
baseline after each request, or creeps upward over time (a leak).

It sends a rotating mix of plain chat, code, and (occasionally, since
they're slow/expensive) image requests, times each one, and polls
/gpu/stats after every single request. At the end it prints a full table
plus a verdict on whether VRAM trended upward across the run.

Usage:
    python3 tools/stress_test.py
    python3 tools/stress_test.py --iterations 40 --image-every 8
    python3 tools/stress_test.py --host http://127.0.0.1:8000 --api-key XXXX
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHAT_PROMPTS = [
    "Hey, how are you?",
    "What's a good name for a coffee shop?",
    "Explain what a firewall does in one paragraph.",
    "Give me three tips for staying focused while working.",
]

CODE_PROMPTS = [
    "write a python function to check if a number is prime",
    "write a bash one-liner to find the 5 largest files in a directory",
    "write a python function to flatten a nested list",
    "write a sql query to find duplicate emails in a users table",
]

IMAGE_PROMPTS = [
    "a small cabin in a snowy forest at dusk",
    "a red vintage bicycle leaning against a brick wall",
]


def _read_env_value(key: str) -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return ""


def _request(url: str, api_key: str, method: str = "GET", body: dict | None = None, timeout: float = 300.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            payload = json.loads(exc.read().decode())
        except Exception:
            payload = {"detail": "non-json error body"}
    except Exception as exc:
        return {"ok": False, "status": None, "error": repr(exc), "latency_ms": (time.monotonic() - t0) * 1000}
    return {"ok": 200 <= status < 300, "status": status, "payload": payload, "latency_ms": (time.monotonic() - t0) * 1000}


def _gpu_snapshot(host: str, api_key: str) -> dict:
    result = _request(f"{host}/gpu/stats", api_key)
    if not result.get("ok"):
        return {"used_gb": None, "free_gb": None}
    devices = (result.get("payload") or {}).get("devices") or []
    if not devices:
        return {"used_gb": None, "free_gb": None}
    return {"used_gb": devices[0].get("used_gb"), "free_gb": devices[0].get("free_gb")}


def _wait_for_ready(host: str, api_key: str, timeout_s: float = 120.0) -> bool:
    # scripts/start.sh backgrounds the server and returns immediately -- it
    # does NOT wait for uvicorn to bind the port or for the model to load.
    # Firing requests right after start.sh returns is a real race, not a
    # server bug (every failure will be a near-instant ConnectionRefusedError
    # if you hit this).
    #
    # Waiting for *any* response from /health isn't enough either: the app
    # answers /health immediately, well before the background warm-up
    # actually finishes loading Qwen. Taking the VRAM "baseline" during that
    # window makes a normal warm-up look like a leak (VRAM legitimately goes
    # up as the model finishes loading). Wait for the qwen worker to actually
    # report ready before proceeding.
    print(f"Waiting for {host} to come up and Qwen to finish loading (up to {timeout_s:.0f}s)...")
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        result = _request(f"{host}/health", api_key, timeout=5.0)
        if result.get("status") is not None:
            workers = (result.get("payload") or {}).get("workers") or {}
            if workers.get("qwen") is True:
                print(f"Qwen ready after {attempt} attempt(s).")
                return True
        time.sleep(2)
    print(f"Qwen never reported ready within {timeout_s:.0f}s -- giving up. "
          f"Check data/runtime/symon-manager.log.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Base URL of the running server")
    parser.add_argument("--api-key", default=None, help="Overrides SYMON_API_KEY from .env if given")
    parser.add_argument("--iterations", type=int, default=20, help="Total requests to send")
    parser.add_argument("--image-every", type=int, default=6, help="Send one image request every N iterations (0 = never)")
    parser.add_argument("--max-tokens", type=int, default=128, help="max_tokens for chat/code requests")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for the server to come up first")
    parser.add_argument("--ready-timeout", type=float, default=120.0, help="Max seconds to wait for the server before giving up")
    args = parser.parse_args()

    api_key = args.api_key if args.api_key is not None else _read_env_value("SYMON_API_KEY")

    print("===== SyMoNeuRaL stress test =====")
    print(f"host: {args.host}")
    print(f"iterations: {args.iterations}")
    print(f"image request every: {args.image_every or 'never'}")
    print(f"auth: {'using API key from .env/--api-key' if api_key else 'no key set'}")
    print()

    if not args.no_wait:
        if not _wait_for_ready(args.host, api_key, timeout_s=args.ready_timeout):
            return 1
        print()

    baseline = _gpu_snapshot(args.host, api_key)
    print(f"baseline VRAM before run: used={baseline['used_gb']} GB, free={baseline['free_gb']} GB")
    print()

    rows: list[dict] = []

    for i in range(1, args.iterations + 1):
        if args.image_every and i % args.image_every == 0:
            kind = "image"
            prompt = "generate an image of " + IMAGE_PROMPTS[(i // args.image_every - 1) % len(IMAGE_PROMPTS)]
            max_tokens = 64
        elif i % 2 == 0:
            kind = "code"
            prompt = CODE_PROMPTS[i % len(CODE_PROMPTS)]
            max_tokens = args.max_tokens
        else:
            kind = "chat"
            prompt = CHAT_PROMPTS[i % len(CHAT_PROMPTS)]
            max_tokens = args.max_tokens

        result = _request(
            f"{args.host}/v1/chat/completions",
            api_key,
            method="POST",
            body={
                "model": "symoneural_moe",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )

        gpu = _gpu_snapshot(args.host, api_key)

        row = {
            "n": i,
            "kind": kind,
            "ok": result.get("ok"),
            "status": result.get("status"),
            "latency_ms": round(result.get("latency_ms", 0), 1),
            "used_gb": gpu["used_gb"],
        }
        rows.append(row)

        status_marker = "OK" if row["ok"] else f"FAIL({row['status']})"
        print(f"[{i:3d}/{args.iterations}] {kind:6s} {status_marker:10s} "
              f"latency={row['latency_ms']:>9.1f}ms  vram_used={row['used_gb']} GB")

        if not row["ok"]:
            err = result.get("error") or result.get("payload")
            print(f"          -> {err}")

    print()
    print("===== summary =====")

    failures = [r for r in rows if not r["ok"]]
    print(f"Total requests: {len(rows)}  Failures: {len(failures)}")

    vram_values = [r["used_gb"] for r in rows if r["used_gb"] is not None]
    if len(vram_values) >= 4:
        first_quarter = vram_values[: max(1, len(vram_values) // 4)]
        last_quarter = vram_values[-max(1, len(vram_values) // 4):]
        avg_first = statistics.mean(first_quarter)
        avg_last = statistics.mean(last_quarter)
        drift = avg_last - avg_first
        print(f"VRAM used, first quarter avg: {avg_first:.2f} GB")
        print(f"VRAM used, last quarter avg:  {avg_last:.2f} GB")
        print(f"Drift: {drift:+.2f} GB")
        if drift > 1.0:
            print("VERDICT: VRAM trended upward significantly across the run -- "
                  "looks like a leak, something isn't releasing memory between requests.")
        elif drift > 0.3:
            print("VERDICT: Mild upward drift. Possibly noise, possibly a slow leak -- "
                  "worth a longer run (--iterations 60+) to be sure.")
        else:
            print("VERDICT: VRAM stayed stable across the run. No leak detected.")
    else:
        print("Not enough VRAM samples to assess a trend (need at least 4 successful readings).")

    latencies_by_kind: dict[str, list[float]] = {}
    for r in rows:
        if r["ok"]:
            latencies_by_kind.setdefault(r["kind"], []).append(r["latency_ms"])
    print()
    for kind, values in latencies_by_kind.items():
        print(f"{kind:6s} latency -- min={min(values):.0f}ms  median={statistics.median(values):.0f}ms  max={max(values):.0f}ms  n={len(values)}")

    if failures:
        print()
        print(f"{len(failures)} request(s) failed -- see details above. This is a real problem, not noise.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
