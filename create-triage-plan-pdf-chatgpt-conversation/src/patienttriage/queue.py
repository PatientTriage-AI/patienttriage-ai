from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .domain import TriageAssessment


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class QueuePatient:
    patient_token: str
    arrived_at: str
    final_esi: int | None
    assessment: TriageAssessment
    reassessment_alerts: list[str] = field(default_factory=list)

    def elapsed_minutes(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        return max(0, int((now - _parse(self.arrived_at)).total_seconds() // 60))


def sorted_queue(patients: list[QueuePatient]) -> list[QueuePatient]:
    # None is intentionally placed last; an unconfirmed recommendation cannot displace confirmed urgency.
    return sorted(patients, key=lambda person: (person.final_esi is None, person.final_esi or 99, person.arrived_at))


def wait_limit_alert(patient: QueuePatient, policy: dict, now: datetime | None = None) -> str | None:
    if patient.final_esi is None:
        return "Awaiting nurse disposition; reassessment needed."
    limit = policy["reassessment_wait_minutes"].get(str(patient.final_esi))
    if limit is not None and patient.elapsed_minutes(now) >= limit:
        return f"Wait-limit breach: ESI {patient.final_esi} has waited {patient.elapsed_minutes(now)} minutes (limit {limit})."
    return None


def deterioration_alert(previous_esi: int | None, latest: TriageAssessment) -> str | None:
    if latest.red_flags:
        return "New red flag: reassessment required and Fast-Track removed."
    if previous_esi and latest.suggested_esi and latest.suggested_esi < previous_esi:
        return f"Predicted urgency increased from ESI {previous_esi} to ESI {latest.suggested_esi}: reassessment required."
    if latest.fast_track_eligible is False and latest.safety_gate != "passed":
        return "Fast-Track eligibility removed: reassessment required."
    return None
