#!/usr/bin/env python3
"""
SyMoNeuRaL VRAM calibration tool
=================================

Measures, for ONE worker at a time (qwen or image), the real timing of:

    low (baseline, nothing loaded)
      -> nominal (model finished loading, steady state)
        -> peak (maximum VRAM seen during actual generation)
          -> settle (what happens after the request completes)

This exists because our own "SDXL auto-unloads right after generating" fix
means /gpu/stats taken AFTER a request always shows the released state --
you can never see the real peak that way. This tool polls continuously
THROUGH the request instead, in a background thread, so it actually catches
the peak.

"Only one model active" is already guaranteed by SYMON_SINGLE_ACTIVE_MODEL --
this tool doesn't need to enforce that itself, it just triggers one request
to one worker and watches.

Usage:
    python3 tools/calibrate_vram.py --worker qwen
    python3 tools/calibrate_vram.py --worker image
    python3 tools/calibrate_vram.py --worker qwen --tolerance 0.2 --settle-window 20
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "payload": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode())
        except Exception:
            payload = {"detail": "non-json error body"}
        return {"status": exc.code, "payload": payload}
    except Exception as exc:
        return {"status": None, "error": repr(exc)}


def _gpu_used_gb(host: str, api_key: str) -> float | None:
    result = _request(f"{host}/gpu/stats", api_key, timeout=5.0)
    devices = (result.get("payload") or {}).get("devices") or []
    if not devices:
        return None
    return devices[0].get("used_gb")


def _wait_for_ready(host: str, api_key: str, timeout_s: float = 120.0) -> bool:
    # Same race as stress_test.py: scripts/start.sh backgrounds the server
    # and returns immediately, well before uvicorn binds the port or Qwen's
    # startup warm-up finishes. Firing the real test request before that
    # happens gets an instant connection failure (or, worse, silently
    # measures the warm-up instead of the thing you actually asked for).
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
    parser.add_argument("--worker", choices=["qwen", "image"], required=True, help="Which worker to calibrate")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tolerance", type=float, default=0.3, help="GB change considered 'stable' for plateau detection")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between VRAM samples")
    parser.add_argument("--settle-window", type=float, default=15.0, help="Seconds to keep polling after the request completes")
    parser.add_argument("--prompt", default=None, help="Override the default test prompt")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for the server/Qwen to be ready first")
    parser.add_argument("--ready-timeout", type=float, default=120.0, help="Max seconds to wait for readiness")
    args = parser.parse_args()

    api_key = args.api_key if args.api_key is not None else _read_env_value("SYMON_API_KEY")

    if args.worker == "qwen":
        prompt = args.prompt or "Hey, how are you?"
        body = {"model": "symoneural_moe", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64}
    else:
        prompt = args.prompt or "generate an image of a lighthouse at sunset"
        body = {"model": "symoneural_moe", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64}

    print(f"===== VRAM calibration: {args.worker} =====")
    print(f"host: {args.host}")
    print(f"prompt: {prompt}")
    print()

    if not args.no_wait:
        if not _wait_for_ready(args.host, api_key, timeout_s=args.ready_timeout):
            return 1
        print()

    samples: list[tuple[float, float]] = []  # (elapsed_s, used_gb)
    stop_event = threading.Event()
    request_result: dict = {}

    t_start = time.monotonic()

    def _poller():
        while not stop_event.is_set():
            used = _gpu_used_gb(args.host, api_key)
            if used is not None:
                samples.append((time.monotonic() - t_start, used))
            time.sleep(args.poll_interval)

    def _sender():
        request_result["response"] = _request(f"{args.host}/v1/chat/completions", api_key, method="POST", body=body)
        request_result["done_at"] = time.monotonic() - t_start

    poller_thread = threading.Thread(target=_poller, daemon=True)
    poller_thread.start()

    baseline = _gpu_used_gb(args.host, api_key)
    print(f"baseline (low) VRAM: {baseline} GB")
    print("sending request and polling continuously...")

    sender_thread = threading.Thread(target=_sender)
    sender_thread.start()
    sender_thread.join()

    request_done_at = request_result.get("done_at", 0.0)
    print(f"request completed at t={request_done_at:.1f}s, status={request_result['response'].get('status')}")

    print(f"watching for {args.settle_window:.0f}s more to see what happens after completion...")
    time.sleep(args.settle_window)
    stop_event.set()
    poller_thread.join(timeout=2)

    print()
    print("===== timeline =====")
    for elapsed, used in samples:
        marker = " <- request completed" if abs(elapsed - request_done_at) < args.poll_interval else ""
        print(f"t={elapsed:6.1f}s  used={used:6.2f} GB{marker}")

    if not samples:
        print("No VRAM samples captured -- is /gpu/stats reachable?")
        return 1

    print()
    print("===== analysis =====")

    during_request = [(t, u) for t, u in samples if t <= request_done_at]
    after_request = [(t, u) for t, u in samples if t > request_done_at]

    peak_t, peak_u = max(during_request, key=lambda x: x[1]) if during_request else (None, None)
    print(f"baseline (low):   {baseline:.2f} GB" if baseline is not None else "baseline: unknown")
    print(f"peak (during request): {peak_u:.2f} GB at t={peak_t:.1f}s" if peak_u is not None else "peak: no data")

    # nominal/loaded plateau: first point where VRAM has risen meaningfully
    # above baseline AND the next couple samples don't change much (tolerance)
    nominal_t = nominal_u = None
    if baseline is not None:
        for i in range(len(during_request) - 2):
            t0, u0 = during_request[i]
            t1, u1 = during_request[i + 1]
            t2, u2 = during_request[i + 2]
            risen = u0 - baseline > args.tolerance
            stable = abs(u1 - u0) <= args.tolerance and abs(u2 - u1) <= args.tolerance
            if risen and stable:
                nominal_t, nominal_u = t0, u0
                break
    if nominal_t is not None:
        print(f"nominal (loaded, steady): {nominal_u:.2f} GB at t={nominal_t:.1f}s "
              f"(load took ~{nominal_t:.1f}s)")
    else:
        print("nominal (loaded, steady): not clearly detected -- see raw timeline above")

    if after_request:
        settle_u = after_request[-1][1]
        settle_t = after_request[-1][0]
        print(f"settle (end of {args.settle_window:.0f}s post-request window): {settle_u:.2f} GB at t={settle_t:.1f}s")
        if baseline is not None:
            ever_rose_meaningfully = peak_u is not None and (peak_u - baseline) > args.tolerance
            if not ever_rose_meaningfully:
                print("  -> VRAM never moved during this test -- the model was already loaded "
                      "before the test started (this measured a warm request, not a fresh load).")
            elif settle_u <= baseline + args.tolerance:
                print("  -> rose during the request, then genuinely released back to baseline after completion.")
            else:
                print("  -> still resident above baseline (staying loaded, as expected for single_active_model caching)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
