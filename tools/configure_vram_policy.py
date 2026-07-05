#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

POLICY_KEYS = {
    "SYMON_VRAM_TARGET_GB": "14.5",
    "SYMON_VRAM_HARD_LIMIT_GB": "15.2",
    "SYMON_VRAM_TEXT_GEMMA_SHARE": "0.20",
    "SYMON_VRAM_TEXT_QWEN_SHARE": "0.80",
    "SYMON_VRAM_IMAGE_GEMMA_SHARE": "0.20",
    "SYMON_VRAM_IMAGE_RUNTIME_SHARE": "0.80",
    "SYMON_VRAM_PARAM_GEMMA_SHARE": "0.25",
    "SYMON_VRAM_PARAM_QWEN_SHARE": "0.05",
    "SYMON_VRAM_PARAM_RUNTIME_SHARE": "0.70",
    "SYMON_PARAMETER_GOVERNOR": "qwen",
}

PRESETS = {
    "1": {
        "name": "RTX 5070 Ti 16 GB balanced",
        "values": POLICY_KEYS,
    },
    "2": {
        "name": "Text-heavy Qwen specialist",
        "values": {
            **POLICY_KEYS,
            "SYMON_VRAM_TEXT_GEMMA_SHARE": "0.15",
            "SYMON_VRAM_TEXT_QWEN_SHARE": "0.85",
            "SYMON_VRAM_PARAM_GEMMA_SHARE": "0.20",
            "SYMON_VRAM_PARAM_QWEN_SHARE": "0.10",
            "SYMON_VRAM_PARAM_RUNTIME_SHARE": "0.70",
        },
    },
    "3": {
        "name": "Image-heavy runtime",
        "values": {
            **POLICY_KEYS,
            "SYMON_VRAM_IMAGE_GEMMA_SHARE": "0.15",
            "SYMON_VRAM_IMAGE_RUNTIME_SHARE": "0.85",
        },
    },
    "4": {
        "name": "Manual entry",
        "values": {},
    },
}


def read_env() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(errors="ignore").splitlines()


def parse_env(lines: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def prompt_value(key: str, default: str) -> str:
    raw = input(f"{key} [{default}]: ").strip()
    return raw if raw else default


def require_float(key: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {key}: {value}") from exc


def validate(values: dict[str, str]) -> None:
    target = require_float("SYMON_VRAM_TARGET_GB", values["SYMON_VRAM_TARGET_GB"])
    hard = require_float("SYMON_VRAM_HARD_LIMIT_GB", values["SYMON_VRAM_HARD_LIMIT_GB"])

    if target <= 0:
        raise SystemExit("SYMON_VRAM_TARGET_GB must be positive")

    if hard <= 0:
        raise SystemExit("SYMON_VRAM_HARD_LIMIT_GB must be positive")

    if target > hard:
        raise SystemExit("SYMON_VRAM_TARGET_GB cannot exceed SYMON_VRAM_HARD_LIMIT_GB")

    if hard > 15.2:
        raise SystemExit("SYMON_VRAM_HARD_LIMIT_GB should not exceed 15.2 on the 16 GB profile")

    groups = {
        "text": [
            "SYMON_VRAM_TEXT_GEMMA_SHARE",
            "SYMON_VRAM_TEXT_QWEN_SHARE",
        ],
        "image": [
            "SYMON_VRAM_IMAGE_GEMMA_SHARE",
            "SYMON_VRAM_IMAGE_RUNTIME_SHARE",
        ],
        "parameter": [
            "SYMON_VRAM_PARAM_GEMMA_SHARE",
            "SYMON_VRAM_PARAM_QWEN_SHARE",
            "SYMON_VRAM_PARAM_RUNTIME_SHARE",
        ],
    }

    for group_name, keys in groups.items():
        total = sum(require_float(key, values[key]) for key in keys)
        if abs(total - 1.0) > 0.001:
            raise SystemExit(f"{group_name} shares must sum to 1.0, got {total:.4f}")

    governor = values["SYMON_PARAMETER_GOVERNOR"].strip().lower()
    if governor not in {"qwen", "gemma", "dedicated"}:
        raise SystemExit("SYMON_PARAMETER_GOVERNOR must be qwen, gemma, or dedicated")


def update_env_lines(lines: list[str], values: dict[str, str]) -> list[str]:
    key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
    remaining = dict(values)
    output: list[str] = []

    for line in lines:
        match = key_re.match(line.strip())
        if not match:
            output.append(line)
            continue

        key = match.group(1)
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)

    if remaining:
        if output and output[-1].strip():
            output.append("")
        output.append("# SyMoNeuRaL MoE VRAM policy")
        for key, value in remaining.items():
            output.append(f"{key}={value}")

    return output


def print_summary(values: dict[str, str]) -> None:
    target = float(values["SYMON_VRAM_TARGET_GB"])
    hard = float(values["SYMON_VRAM_HARD_LIMIT_GB"])

    text_gemma = float(values["SYMON_VRAM_TEXT_GEMMA_SHARE"])
    text_qwen = float(values["SYMON_VRAM_TEXT_QWEN_SHARE"])

    image_gemma = float(values["SYMON_VRAM_IMAGE_GEMMA_SHARE"])
    image_runtime = float(values["SYMON_VRAM_IMAGE_RUNTIME_SHARE"])

    param_gemma = float(values["SYMON_VRAM_PARAM_GEMMA_SHARE"])
    param_qwen = float(values["SYMON_VRAM_PARAM_QWEN_SHARE"])
    param_runtime = float(values["SYMON_VRAM_PARAM_RUNTIME_SHARE"])

    print()
    print("Selected VRAM policy")
    print("====================")
    print(f"Target budget: {target:.2f} GB")
    print(f"Hard ceiling:  {hard:.2f} GB")
    print(f"Text route:    {text_gemma:.0%} Gemma manager / {text_qwen:.0%} Qwen specialist")
    print(f"Image route:   {image_gemma:.0%} Gemma manager / {image_runtime:.0%} image runtime")
    print(
        "Param route:   "
        f"{param_gemma:.0%} Gemma / "
        f"{param_qwen:.0%} Qwen governor / "
        f"{param_runtime:.0%} active runtime"
    )
    print(f"Governor:      {values['SYMON_PARAMETER_GOVERNOR']}")


def choose_values(current: dict[str, str]) -> dict[str, str]:
    print("SyMoNeuRaL VRAM policy configurator")
    print("===================================")
    print("1) RTX 5070 Ti 16 GB balanced")
    print("2) Text-heavy Qwen specialist")
    print("3) Image-heavy runtime")
    print("4) Manual entry")
    print()

    choice = input("Select preset [1]: ").strip() or "1"
    if choice not in PRESETS:
        raise SystemExit(f"Unknown preset: {choice}")

    preset = PRESETS[choice]
    print(f"Selected: {preset['name']}")

    values = {key: current.get(key, default) for key, default in POLICY_KEYS.items()}
    values.update(preset["values"])

    if choice == "4":
        print()
        print("Press Enter to keep the shown value.")
        for key in POLICY_KEYS:
            values[key] = prompt_value(key, values[key])

    validate(values)
    return values


def main() -> int:
    lines = read_env()
    current = parse_env(lines)
    values = choose_values(current)
    print_summary(values)

    print()
    confirm = input(f"Write policy to {ENV_PATH}? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("No changes written.")
        return 1

    updated = update_env_lines(lines, values)
    ENV_PATH.write_text("\n".join(updated).rstrip() + "\n")
    print(f"Wrote {ENV_PATH}")

    print()
    print("Next:")
    print("  python3 production_audit.py")
    print("  python3 production_audit.py --live")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
