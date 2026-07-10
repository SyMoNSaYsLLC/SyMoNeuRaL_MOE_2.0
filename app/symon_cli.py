from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from typing import Sequence


def _default_model() -> str:
    # Keep the model source explicit and environment-driven for portability.
    return os.environ.get("SYMON_VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _must_find(binary: str) -> str:
    path = shutil.which(binary)
    if path:
        return path
    raise SystemExit(
        f"Missing '{binary}' in PATH. Activate .venv and install requirements.vllm.txt first."
    )


def _run_exec(argv: Sequence[str]) -> int:
    cmd = [str(x) for x in argv]
    # Use subprocess instead of os.exec* to return exit status cleanly.
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def cmd_serve(args: argparse.Namespace, passthrough: list[str]) -> int:
    vllm_bin = _must_find("vllm")

    cmd = [
        vllm_bin,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])

    cmd.extend(passthrough)

    print("Launching:", " ".join(shlex.quote(c) for c in cmd))
    return _run_exec(cmd)


def cmd_bench(_: argparse.Namespace, passthrough: list[str]) -> int:
    vllm_bin = _must_find("vllm")
    cmd = [vllm_bin, "bench", *passthrough]
    if len(cmd) == 2:
        print("Usage: python -m app.symon_cli bench <latency|throughput|serve> [args...]")
        return 2
    print("Launching:", " ".join(shlex.quote(c) for c in cmd))
    return _run_exec(cmd)


def cmd_api(args: argparse.Namespace, passthrough: list[str]) -> int:
    py = sys.executable
    cmd = [
        py,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    cmd.extend(passthrough)
    print("Launching:", " ".join(shlex.quote(c) for c in cmd))
    return _run_exec(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symon",
        description="SyMoNeuRaL CLI with vLLM-compatible API/CLI delegation.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser(
        "serve",
        help="Start vLLM OpenAI-compatible server (delegates to: vllm serve).",
    )
    p_serve.add_argument("--model", default=_default_model())
    p_serve.add_argument("--host", default=os.environ.get("SYMON_BIND_HOST", "0.0.0.0"))
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("SYMON_BIND_PORT", "8000")))
    p_serve.add_argument("--dtype", default=os.environ.get("SYMON_VLLM_DTYPE", "auto"))
    p_serve.add_argument(
        "--max-model-len",
        type=int,
        default=(int(os.environ["SYMON_VLLM_MAX_MODEL_LEN"]) if "SYMON_VLLM_MAX_MODEL_LEN" in os.environ else None),
    )
    p_serve.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=(
            float(os.environ["SYMON_VLLM_GPU_MEMORY_UTILIZATION"])
            if "SYMON_VLLM_GPU_MEMORY_UTILIZATION" in os.environ
            else None
        ),
    )

    p_api = sub.add_parser(
        "api",
        help="Start SyMoNeuRaL FastAPI app (app.main:app).",
    )
    p_api.add_argument("--host", default=os.environ.get("SYMON_BIND_HOST", "0.0.0.0"))
    p_api.add_argument("--port", type=int, default=int(os.environ.get("SYMON_BIND_PORT", "8000")))

    sub.add_parser(
        "bench",
        help="Run vLLM benchmark tooling (delegates to: vllm bench ...).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns, passthrough = parser.parse_known_args(argv)

    if ns.command == "serve":
        return cmd_serve(ns, passthrough)
    if ns.command == "api":
        return cmd_api(ns, passthrough)
    if ns.command == "bench":
        return cmd_bench(ns, passthrough)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
