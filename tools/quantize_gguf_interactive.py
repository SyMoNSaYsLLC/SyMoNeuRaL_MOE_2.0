#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import struct
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FALLBACK_QUANT_TYPES = [
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
    "IQ4_NL",
    "IQ4_XS",
    "IQ3_M",
    "IQ3_S",
    "IQ3_XXS",
    "IQ2_M",
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "IQ1_M",
    "IQ1_S",
    "TQ1_0",
    "TQ2_0",
]

TENSOR_TYPES = [
    "none",
    "f32",
    "f16",
    "bf16",
    "q8_0",
]


def ask(prompt: str, default: str | None = None) -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def ask_choice(prompt: str, choices: list[str], default: str | None = None, allow_custom: bool = True) -> str:
    print()
    print(prompt)
    for i, item in enumerate(choices, start=1):
        marker = " default" if default and item == default else ""
        print(f"  {i}) {item}{marker}")

    if allow_custom:
        print("  custom) type manually")

    while True:
        raw = ask("Select number or value", default or "")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]

        if allow_custom and raw.lower() == "custom":
            return ask("Enter custom quant type").upper()

        for item in choices:
            if raw.upper() == item.upper():
                return item

        print(f"Invalid choice: {raw}")


def find_quantize_bin() -> Path | None:
    env_bin = os.environ.get("LLAMA_QUANTIZE_BIN")
    env_dir = os.environ.get("LLAMA_CPP_DIR")

    candidates: list[Path] = []

    if env_bin:
        candidates.append(Path(env_bin))

    if env_dir:
        base = Path(env_dir)
        candidates.extend(
            [
                base / "build" / "bin" / "llama-quantize",
                base / "build" / "bin" / "quantize",
                base / "bin" / "llama-quantize",
                base / "bin" / "quantize",
            ]
        )

    candidates.extend(
        [
            REPO_ROOT / "bin" / "llama-quantize",
            REPO_ROOT / "tools" / "llama.cpp" / "build" / "bin" / "llama-quantize",
            REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path.home() / "SyMoNeuRaL_MCP" / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path.home() / "SyMoNeuRaL_CLEAN" / "llama.cpp" / "build" / "bin" / "llama-quantize",
        ]
    )

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path

    return None


def get_supported_quant_types(binary: Path) -> list[str]:
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except Exception:
        return FALLBACK_QUANT_TYPES

    text = proc.stdout or ""
    found = set()

    for token in re.findall(r"\b(?:Q|IQ|TQ)[A-Z0-9_]+\b", text):
        if token not in {"QUANTIZE"}:
            found.add(token)

    ordered = [q for q in FALLBACK_QUANT_TYPES if q in found]
    extra = sorted(found - set(ordered))

    return ordered + extra if found else FALLBACK_QUANT_TYPES


def check_gguf(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing GGUF file: {path}")

    with path.open("rb") as f:
        head = f.read(8)

    if head[:4] != b"GGUF":
        raise SystemExit(f"ERROR: not a GGUF file: {path}")

    version = struct.unpack("<I", head[4:8])[0]
    if version <= 0 or version > 10:
        raise SystemExit(f"ERROR: suspicious GGUF version {version}: {path}")


def main() -> int:
    print("===== SyMoNeuRaL GGUF quantizer =====")

    quantize_bin = find_quantize_bin()

    if quantize_bin:
        print(f"Found quantizer: {quantize_bin}")
        use_found = ask("Use this quantizer? yes/no", "yes").lower()
        if use_found not in {"y", "yes"}:
            quantize_bin = None

    if quantize_bin is None:
        raw = ask(
            "Path to llama-quantize binary",
            str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"),
        )
        quantize_bin = Path(raw).expanduser().resolve()

    if not quantize_bin.is_file():
        print(f"ERROR: quantizer not found: {quantize_bin}")
        print("Set LLAMA_CPP_DIR or LLAMA_QUANTIZE_BIN.")
        return 1

    if not os.access(quantize_bin, os.X_OK):
        print(f"ERROR: quantizer is not executable: {quantize_bin}")
        return 1

    infile = Path(ask("Input GGUF file")).expanduser().resolve()
    check_gguf(infile)

    supported = get_supported_quant_types(quantize_bin)

    qtype = ask_choice(
        "Select output quantization type",
        supported,
        default="Q4_K_M",
        allow_custom=True,
    ).upper()

    default_out = infile.with_name(f"{infile.stem}-{qtype}.gguf")
    outfile = Path(ask("Output GGUF path", str(default_out))).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists():
        overwrite = ask(f"Output exists: {outfile}. Overwrite? yes/no", "no").lower()
        if overwrite not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    allow_requantize = ask("Allow requantizing an already-quantized model? yes/no", "no").lower()
    pure = ask("Use --pure? yes/no", "no").lower()
    imatrix = ask("Optional imatrix path, blank to skip", "")

    output_tensor_type = ask_choice(
        "Optional --output-tensor-type",
        TENSOR_TYPES,
        default="none",
        allow_custom=True,
    )

    token_embedding_type = ask_choice(
        "Optional --token-embedding-type",
        TENSOR_TYPES,
        default="none",
        allow_custom=True,
    )

    threads = ask("Optional thread count, blank to skip", "")

    cmd = [str(quantize_bin)]

    if allow_requantize in {"y", "yes"}:
        cmd.append("--allow-requantize")

    if pure in {"y", "yes"}:
        cmd.append("--pure")

    if imatrix:
        imatrix_path = Path(imatrix).expanduser().resolve()
        if not imatrix_path.is_file():
            print(f"ERROR: imatrix file not found: {imatrix_path}")
            return 1
        cmd.extend(["--imatrix", str(imatrix_path)])

    if output_tensor_type != "none":
        cmd.extend(["--output-tensor-type", output_tensor_type])

    if token_embedding_type != "none":
        cmd.extend(["--token-embedding-type", token_embedding_type])

    cmd.extend([str(infile), str(outfile), qtype])

    if threads:
        if not threads.isdigit():
            print(f"ERROR: thread count must be numeric: {threads}")
            return 1
        cmd.append(threads)

    print()
    print("===== command =====")
    print(" ".join(shlex.quote(x) for x in cmd))

    confirm = ask("Run quantization? yes/no", "yes").lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return 1

    subprocess.run(cmd, check=True)

    check_gguf(outfile)

    print()
    print("PASS GGUF quantization")
    print(f"Output: {outfile}")
    print(f"Size GB: {outfile.stat().st_size / 1024**3:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
