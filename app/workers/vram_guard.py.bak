from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.runtime.config import SETTINGS
from app.runtime.gpu import assert_safe_for_job as gpu_assert_safe_for_job
from app.runtime.gpu import devices, status


@dataclass(frozen=True)
class VramRoleBudget:
    role: str
    share: float
    target_gb: float
    hard_limit_gb: float


@dataclass(frozen=True)
class VramProfile:
    name: str
    worker: str
    governor: str
    target_gb: float
    hard_limit_gb: float
    roles: tuple[VramRoleBudget, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "worker": self.worker,
            "governor": self.governor,
            "target_gb": self.target_gb,
            "hard_limit_gb": self.hard_limit_gb,
            "roles": [role.__dict__.copy() for role in self.roles],
        }


def _budget(role: str, share: float) -> VramRoleBudget:
    return VramRoleBudget(
        role=role,
        share=share,
        target_gb=round(SETTINGS.vram_target_gb * share, 3),
        hard_limit_gb=round(SETTINGS.vram_hard_limit_gb * share, 3),
    )


def parameter_governor() -> str:
    governor = SETTINGS.parameter_governor.strip().lower()
    if governor not in {"gemma", "qwen", "dedicated"}:
        return "qwen"
    return governor


def manager_profile() -> VramProfile:
    return VramProfile(
        name="manager",
        worker="gemma",
        governor=parameter_governor(),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        roles=(
            _budget("gemma_manager_quality", 1.0),
        ),
    )


def text_profile() -> VramProfile:
    return VramProfile(
        name="text",
        worker="qwen",
        governor=parameter_governor(),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        roles=(
            _budget("gemma_manager_quality", SETTINGS.vram_text_gemma_share),
            _budget("qwen_text_runtime", SETTINGS.vram_text_qwen_share),
        ),
    )


def image_profile() -> VramProfile:
    return VramProfile(
        name="image",
        worker="safetensor",
        governor=parameter_governor(),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        roles=(
            _budget("gemma_manager_quality", SETTINGS.vram_image_gemma_share),
            _budget("image_runtime", SETTINGS.vram_image_runtime_share),
        ),
    )


def vision_profile() -> VramProfile:
    return VramProfile(
        name="vision",
        worker="vision",
        governor=parameter_governor(),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        roles=(
            _budget("gemma_manager_quality", SETTINGS.vram_text_gemma_share),
            _budget("vision_runtime", SETTINGS.vram_text_qwen_share),
        ),
    )


def parameter_profile() -> VramProfile:
    return VramProfile(
        name="parameter_governance",
        worker=parameter_governor(),
        governor=parameter_governor(),
        target_gb=SETTINGS.vram_target_gb,
        hard_limit_gb=SETTINGS.vram_hard_limit_gb,
        roles=(
            _budget("gemma_manager_quality", SETTINGS.vram_param_gemma_share),
            _budget("qwen_parameter_governor", SETTINGS.vram_param_qwen_share),
            _budget("active_runtime", SETTINGS.vram_param_runtime_share),
        ),
    )


def profile_for_decision(worker: str) -> VramProfile:
    worker = worker.strip().lower()

    if worker == "qwen":
        return text_profile()

    if worker == "safetensor":
        return image_profile()

    if worker == "vision":
        return vision_profile()

    if worker == "parameter" or worker == "governor":
        return parameter_profile()

    return manager_profile()


def _sum_shares(profile: VramProfile) -> float:
    return round(sum(role.share for role in profile.roles), 6)


def profile_valid(profile: VramProfile) -> tuple[bool, str]:
    if profile.target_gb <= 0:
        return False, "vram_target_gb_must_be_positive"

    if profile.hard_limit_gb <= 0:
        return False, "vram_hard_limit_gb_must_be_positive"

    if profile.target_gb > profile.hard_limit_gb:
        return False, "vram_target_exceeds_hard_limit"

    if profile.hard_limit_gb > 15.2:
        return False, "vram_hard_limit_exceeds_16gb_guardrail"

    for role in profile.roles:
        if role.share < 0:
            return False, f"negative_share:{role.role}"
        if role.target_gb < 0:
            return False, f"negative_target_gb:{role.role}"
        if role.hard_limit_gb < 0:
            return False, f"negative_hard_limit_gb:{role.role}"

    total = _sum_shares(profile)
    if abs(total - 1.0) > 0.001:
        return False, f"profile_shares_do_not_sum_to_one:{profile.name}:{total}"

    return True, "ok"


def assert_profile_safe(profile: VramProfile) -> None:
    ok, reason = profile_valid(profile)
    if not ok:
        raise RuntimeError(f"vram_profile_invalid:{reason}:{profile.as_dict()}")

    gpu_assert_safe_for_job(profile.worker)


def assert_safe_for_job(worker: str) -> None:
    profile = profile_for_decision(worker)
    assert_profile_safe(profile)


def policy_status() -> dict[str, Any]:
    profiles = {
        "manager": manager_profile().as_dict(),
        "text": text_profile().as_dict(),
        "image": image_profile().as_dict(),
        "vision": vision_profile().as_dict(),
        "parameter_governance": parameter_profile().as_dict(),
    }

    validations = {
        name: profile_valid(profile_for_decision(profile["worker"]))
        for name, profile in profiles.items()
    }

    validations["parameter_governance"] = profile_valid(parameter_profile())

    return {
        "target_gb": SETTINGS.vram_target_gb,
        "hard_limit_gb": SETTINGS.vram_hard_limit_gb,
        "single_active_model": SETTINGS.single_active_model,
        "unload_text_before_image": SETTINGS.unload_text_before_image,
        "parameter_governor": parameter_governor(),
        "profiles": profiles,
        "validations": {
            name: {"ok": ok, "reason": reason}
            for name, (ok, reason) in validations.items()
        },
    }


__all__ = [
    "VramRoleBudget",
    "VramProfile",
    "status",
    "devices",
    "assert_safe_for_job",
    "assert_profile_safe",
    "profile_for_decision",
    "profile_valid",
    "policy_status",
    "parameter_governor",
    "manager_profile",
    "text_profile",
    "image_profile",
    "vision_profile",
    "parameter_profile",
]
