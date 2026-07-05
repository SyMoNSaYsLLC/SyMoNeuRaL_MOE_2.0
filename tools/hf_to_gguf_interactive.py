#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTTYPES = [
    "auto",
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "tq1_0",
    "tq2_0",
]


def ask(prompt: str, default: str | None = None) -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    print()
    print(prompt)
    for i, item in enumerate(choices, start=1):
        marker = " default" if default and item == default else ""
        print(f"  {i}) {item}{marker}")

    while True:
        raw = ask("Select number or value", default or "")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]

        raw_l = raw.lower()
        for item in choices:
            if raw_l == item.lower():
                return item

        print(f"Invalid choice: {raw}")


def find_converter() -> Path | None:
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    candidates: list[Path] = []

    if env_dir:
        candidates.append(Path(env_dir) / "convert_hf_to_gguf.py")

    candidates.extend(
        [
            REPO_ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py",
            REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "SyMoNeuRaL_MCP" / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "SyMoNeuRaL_CLEAN" / "llama.cpp" / "convert_hf_to_gguf.py",
        ]
    )

    for path in candidates:
        if path.is_file():
            return path

    return None


def read_magic(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read(4)


def main() -> int:
    print("===== SyMoNeuRaL HF -> GGUF converter =====")

    converter = find_converter()

    if converter:
        print(f"Found converter: {converter}")
        use_found = ask("Use this converter? yes/no", "yes").lower()
        if use_found not in {"y", "yes"}:
            converter = None

    if converter is None:
        converter_raw = ask(
            "Path to llama.cpp convert_hf_to_gguf.py",
            str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
        )
        converter = Path(converter_raw).expanduser().resolve()

    if not converter.is_file():
        print(f"ERROR: converter not found: {converter}")
        print("Set LLAMA_CPP_DIR or enter the path to convert_hf_to_gguf.py.")
        return 1

    model_dir = Path(ask("Path to Hugging Face model folder")).expanduser().resolve()

    if not model_dir.is_dir():
        print(f"ERROR: model folder not found: {model_dir}")
        return 1

    if not (model_dir / "config.json").is_file():
        print(f"ERROR: missing config.json in {model_dir}")
        print("This script expects a Hugging Face model directory.")
        return 1

    outtype = ask_choice(
        "Select GGUF output tensor type",
        DEFAULT_OUTTYPES,
        default="bf16",
    )

    default_name = f"{model_dir.name}-{outtype}.gguf"
    default_out = REPO_ROOT / "models" / "llm" / default_name

    outfile = Path(ask("Output GGUF path", str(default_out))).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists():
        overwrite = ask(f"Output exists: {outfile}. Overwrite? yes/no", "no").lower()
        if overwrite not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    model_name = ask("Optional model name metadata, blank to skip", "")

    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]

    if model_name:
        cmd.extend(["--model-name", model_name])

    print()
    print("===== command =====")
    print(" ".join(shlex.quote(x) for x in cmd))

    confirm = ask("Run conversion? yes/no", "yes").lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return 1

    subprocess.run(cmd, check=True)

    if not outfile.is_file():
        print(f"ERROR: conversion finished but output missing: {outfile}")
        return 1

    magic = read_magic(outfile)

    if magic != b"GGUF":
        print(f"ERROR: output is not GGUF. Magic={magic!r}")
        return 1

    print()
    print("PASS HF -> GGUF conversion")
    print(f"Output: {outfile}")
    print(f"Size GB: {outfile.stat().st_size / 1024**3:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
