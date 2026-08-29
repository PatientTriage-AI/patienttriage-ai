from __future__ import annotations

from .domain import IntakePayload, TriageAssessment
from .model import LocalModel, ModelUnavailable
from .policy import load_policy, missing_and_implausible, out_of_distribution, red_flags, vital_red_flags


def _review(
    recommendation: str, gate: str, policy: dict, missing: tuple[str, ...] = (),
    implausible: tuple[str, ...] = (), flags: tuple[str, ...] = (), factors: tuple[str, ...] = (),
    model_version: str = "unavailable", outlier: bool = False,
) -> TriageAssessment:
    reasons = list(missing) + list(implausible) + list(flags)
    reasons.append(policy["fast_track"]["disabled_reason"] if not policy["fast_track"]["enabled"] else "Clinician confirmation required.")
    return TriageAssessment(
        status="clinician_review_required", recommendation=recommendation, suggested_esi=None,
        confidence=None, urgent_risk=None, low_acuity_probability=None, safety_gate=gate,
        fast_track_eligible=False, why_not_fast_track=tuple(reasons), missing_fields=missing,
        implausible_fields=implausible, red_flags=flags, top_factors=factors,
        model_version=model_version, rule_version=policy["version"], out_of_distribution=outlier,
    )


def evaluate_intake(payload: IntakePayload, model_failure: bool = False, policy: dict | None = None) -> TriageAssessment:
    """Assess intake safely. This function never infers red flags from complaint text."""
    policy = policy or load_policy()
    missing, implausible = missing_and_implausible(payload, policy)
    selected_flags = red_flags(payload, policy)
    vital_flags = vital_red_flags(payload, policy)
    all_flags = tuple(sorted(set(selected_flags + vital_flags)))

    if payload.age is None or payload.age < policy["adult_minimum_age"]:
        return _review("Rules-only escalation: adult ML model unavailable for this age cohort.", "pediatric_rules_only", policy, missing, implausible, all_flags)
    if all_flags:
        return _review("Immediate clinician review required: red-flag safety gate triggered.", "red_flag", policy, missing, implausible, all_flags)
    if missing:
        return _review("Clinician review required: complete current vitals and pain score.", "missing_data", policy, missing, implausible)
    if implausible:
        return _review("Clinician review required: verify implausible intake value(s).", "implausible_data", policy, missing, implausible)
    if model_failure:
        return TriageAssessment(
            status="rules_only", recommendation="Rules-only fallback: nurse assessment required; confidence unavailable.",
            suggested_esi=None, confidence=None, urgent_risk=None, low_acuity_probability=None,
            safety_gate="model_failure", fast_track_eligible=False,
            why_not_fast_track=("Model-failure toggle is active.",), missing_fields=(), implausible_fields=(),
            red_flags=(), top_factors=(), model_version="unavailable", rule_version=policy["version"],
        )
    outlier = out_of_distribution(payload, policy)
    if outlier:
        return _review("Clinician review required: intake is outside the model's validated range.", "out_of_distribution", policy, outlier=True)
    try:
        prediction = LocalModel().predict(payload)
    except ModelUnavailable as exc:
        return _review(f"Clinician review required: {exc}", "model_unavailable", policy)
    if prediction.confidence < policy["low_confidence_threshold"]:
        return _review("Clinician review required: calibrated confidence is below the safety threshold.", "low_confidence", policy, factors=prediction.top_factors, model_version=prediction.model_version)

    fast_track = (
        policy["fast_track"]["enabled"]
        and prediction.low_acuity_probability >= policy["fast_track"]["low_acuity_threshold"]
        and prediction.urgent_risk == 0
    )
    reasons = () if fast_track else (policy["fast_track"]["disabled_reason"],)
    return TriageAssessment(
        status="assessment", recommendation="Nurse confirmation required before routing.",
        suggested_esi=prediction.suggested_esi, confidence=prediction.confidence,
        urgent_risk=prediction.urgent_risk, low_acuity_probability=prediction.low_acuity_probability,
        safety_gate="passed", fast_track_eligible=fast_track, why_not_fast_track=reasons,
        missing_fields=(), implausible_fields=(), red_flags=(), top_factors=prediction.top_factors,
        model_version=prediction.model_version, rule_version=policy["version"], out_of_distribution=False,
    )
