from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .domain import IntakePayload

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "config" / "policy.v1.json"


@lru_cache(maxsize=4)
def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict:
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def missing_and_implausible(payload: IntakePayload, policy: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing, implausible = [], []
    for field in policy["required_fast_track_fields"]:
        value = getattr(payload, field)
        if value is None:
            missing.append(field)
        elif not (policy["plausible_ranges"][field][0] <= float(value) <= policy["plausible_ranges"][field][1]):
            implausible.append(field)
    return tuple(missing), tuple(implausible)


def red_flags(payload: IntakePayload, policy: dict) -> tuple[str, ...]:
    allowed = set(policy["nurse_selectable_red_flags"])
    return tuple(sorted(set(payload.observable_cues).intersection(allowed)))


def vital_red_flags(payload: IntakePayload, policy: dict) -> tuple[str, ...]:
    limits = policy["rules_only_red_flag_vitals"]
    flags = []
    if payload.spo2 is not None and payload.spo2 <= limits["spo2_at_or_below"]:
        flags.append("low_spo2")
    if payload.systolic_bp is not None and payload.systolic_bp <= limits["systolic_bp_at_or_below"]:
        flags.append("low_systolic_bp")
    if payload.heart_rate is not None and payload.heart_rate >= limits["heart_rate_at_or_above"]:
        flags.append("high_heart_rate")
    if payload.respiratory_rate is not None and payload.respiratory_rate >= limits["respiratory_rate_at_or_above"]:
        flags.append("high_respiratory_rate")
    return tuple(flags)


def out_of_distribution(payload: IntakePayload, policy: dict) -> bool:
    for field, bounds in policy["out_of_distribution_ranges"].items():
        value = getattr(payload, field)
        if value is not None and not bounds[0] <= float(value) <= bounds[1]:
            return True
    return False
