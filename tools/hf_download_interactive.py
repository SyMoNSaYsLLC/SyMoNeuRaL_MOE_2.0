#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ask(prompt: str, default: str | None = None) -> str:
    if default is not None and default != "":
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(" ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(cmd, text=True, check=check)


def find_hf() -> str | None:
    found = shutil.which("hf")
    if found:
        return found

    candidates = [
        Path.home() / ".local" / "bin" / "hf",
        Path.home() / ".cargo" / "bin" / "hf",
    ]

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    return None


def normalize_model_id(value: str) -> str:
    model_id = value.strip()

    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "huggingface.co/",
    ):
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix) :]

    model_id = model_id.strip().strip("/")
    for marker in ("/tree/", "/blob/", "/resolve/"):
        if marker in model_id:
            model_id = model_id.split(marker, 1)[0]

    return model_id.strip().strip("/")


def main() -> int:
    print("===== SyMoNeuRaL Hugging Face model downloader =====")
    print()
    print("Before running gated/private downloads, make sure Hugging Face auth is linked first:")
    print()
    print("  hf auth login")
    print("  hf auth whoami")
    print()
    print("If using Hugging Face SSH/git access, make sure SSH is linked to the Hugging Face account first.")
    print("This script runs: uv tool install hf  +  hf download")
    print()

    uv = shutil.which("uv")
    if not uv:
        print("ERROR: uv is not installed or not in PATH.")
        return 1

    print("===== install/update hf CLI through uv =====")
    install = subprocess.run([uv, "tool", "install", "hf"], text=True, check=False)
    if install.returncode != 0:
        print("uv tool install hf did not complete cleanly; trying uv tool upgrade hf")
        subprocess.run([uv, "tool", "upgrade", "hf"], text=True, check=False)

    hf = find_hf()
    if not hf:
        print("ERROR: hf command is still not available.")
        print("Reload the shell or add uv's tool bin directory to PATH.")
        print("Common path:")
        print("  export PATH=\"$HOME/.local/bin:$PATH\"")
        return 1

    print()
    print("===== hf CLI status =====")
    subprocess.run([hf, "--version"], text=True, check=False)
    subprocess.run([hf, "auth", "whoami"], text=True, check=False)

    print()
    model_input = ask(
        "Enter Hugging Face model URL or repo id",
        "google/gemma-4-E2B-it-qat-mobile-transformers",
    )

    model_id = normalize_model_id(model_input)
    if "/" not in model_id:
        print(f"ERROR: model must look like user/model or org/model. Got: {model_id}")
        return 1

    default_name = model_id.rsplit("/", 1)[-1]
    default_dir = REPO_ROOT / "models" / "hf" / default_name

    print()
    print(f"Resolved model id: {model_id}")

    local_dir = Path(ask("Local output directory", str(default_dir))).expanduser().resolve()
    revision = ask("Optional revision/branch/commit, blank for default", "")
    repo_type = ask("Download repo type", "model")

    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        hf,
        "download",
        model_id,
        "--repo-type",
        repo_type,
        "--local-dir",
        str(local_dir),
    ]

    if revision:
        cmd.extend(["--revision", revision])

    print()
    print("===== download command =====")
    print(" ".join(shlex.quote(x) for x in cmd))
    print()

    confirm = ask("Run download? yes/no", "yes").lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return 1

    subprocess.run(cmd, text=True, check=True)

    if not local_dir.is_dir():
        print(f"ERROR: local directory missing after download: {local_dir}")
        return 1

    print()
    print("===== downloaded files =====")
    shown = 0
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            print(path.relative_to(REPO_ROOT))
            shown += 1
            if shown >= 50:
                break

    print()
    print("PASS Hugging Face download complete")
    print(f"Model id: {model_id}")
    print(f"Local dir: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
