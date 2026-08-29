from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IntakePayload:
    patient_token: str
    age: float | None
    systolic_bp: float | None
    diastolic_bp: float | None
    heart_rate: float | None
    respiratory_rate: float | None
    temperature_c: float | None
    spo2: float | None
    pain_score: float | None
    sex: str = "Not recorded"  # Context only - never passed to the ML model.
    complaint: str = ""  # Context only - never parsed for red flags or ML features.
    history_available: bool = False
    observable_cues: tuple[str, ...] = ()  # Nurse-selected, versioned policy cues.
    entered_at: str = field(default_factory=utc_now)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriageAssessment:
    status: Literal["assessment", "clinician_review_required", "rules_only"]
    recommendation: str
    suggested_esi: int | None
    confidence: float | None
    urgent_risk: float | None
    low_acuity_probability: float | None
    safety_gate: str
    fast_track_eligible: bool
    why_not_fast_track: tuple[str, ...]
    missing_fields: tuple[str, ...]
    implausible_fields: tuple[str, ...]
    red_flags: tuple[str, ...]
    top_factors: tuple[str, ...]
    model_version: str
    rule_version: str
    out_of_distribution: bool = False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionPayload:
    patient_token: str
    nurse_disposition: str
    accepted: bool
    override_reason: str | None = None
    override_note: str | None = None
    decided_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class VitalsPayload:
    patient_token: str
    systolic_bp: float | None
    diastolic_bp: float | None
    heart_rate: float | None
    respiratory_rate: float | None
    temperature_c: float | None
    spo2: float | None
    pain_score: float | None
    recorded_at: str = field(default_factory=utc_now)
