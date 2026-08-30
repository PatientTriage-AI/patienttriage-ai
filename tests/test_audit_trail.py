"""Tests for AuditTrail — create, retrieve, clear, and hash-chain integrity.

Run with:
    python -m pytest tests/test_audit_trail.py -v
"""

import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from patienttriage.audit import AuditTrail
from patienttriage.domain import (
    TriageAssessment,
    DecisionPayload,
    ReassessmentEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_audit(tmp_path: Path) -> AuditTrail:
    """Return a fresh AuditTrail backed by a temporary SQLite file."""
    return AuditTrail(tmp_path / "test_audit.sqlite3")


def _make_assessment(token: str = "PT-TEST01") -> TriageAssessment:
    return TriageAssessment(
        patient_token=token,
        timestamp=datetime.now(timezone.utc),
        is_pediatric=False,
        safety_gate_passed=True,
        missing_fields=[],
        red_flags=[],
        suggested_esi=3,
        confidence=0.75,
        urgent_risk=0.20,
        probabilities=None,
        shap_factors=None,
        fast_track_eligible=False,
        fast_track_reason="Low urgency risk",
        clinician_review_reason=None,
        model_version="v1.0",
        policy_version="policy.v1",
    )


def _make_decision(token: str = "PT-TEST01") -> DecisionPayload:
    return DecisionPayload(
        patient_token=token,
        nurse_accepted=True,
        override_esi=None,
        override_reason=None,
        override_text="",
    )


def _make_reassessment(token: str = "PT-TEST01") -> ReassessmentEvent:
    return ReassessmentEvent(
        patient_token=token,
        timestamp=datetime.now(timezone.utc),
        trigger="Vital sign deterioration",
        old_esi=3,
        new_esi=2,
        vitals_updated=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — Records can be created
# ---------------------------------------------------------------------------

class TestAuditCreation:
    def test_record_assessment_creates_event(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        events = tmp_audit.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "ASSESSMENT"

    def test_record_decision_creates_event(self, tmp_audit: AuditTrail):
        tmp_audit.record_decision(_make_decision())
        events = tmp_audit.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "DECISION"

    def test_record_reassessment_creates_event(self, tmp_audit: AuditTrail):
        tmp_audit.record_reassessment(_make_reassessment())
        events = tmp_audit.get_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "REASSESSMENT"

    def test_multiple_events_all_stored(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment("PT-A"))
        tmp_audit.record_decision(_make_decision("PT-A"))
        tmp_audit.record_reassessment(_make_reassessment("PT-A"))
        assert len(tmp_audit.get_events()) == 3


# ---------------------------------------------------------------------------
# Test 2 — Records can be retrieved
# ---------------------------------------------------------------------------

class TestAuditRetrieval:
    def test_get_events_returns_list(self, tmp_audit: AuditTrail):
        assert isinstance(tmp_audit.get_events(), list)

    def test_empty_trail_returns_empty_list(self, tmp_audit: AuditTrail):
        assert tmp_audit.get_events() == []

    def test_event_has_expected_keys(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        event = tmp_audit.get_events()[0]
        for key in ("id", "patient_token", "timestamp", "event_type", "payload", "previous_hash", "hash"):
            assert key in event, f"Missing key: {key}"

    def test_patient_token_stored_correctly(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment("PT-XYZ99"))
        assert tmp_audit.get_events()[0]["patient_token"] == "PT-XYZ99"


# ---------------------------------------------------------------------------
# Test 3 — Records can be cleared
# ---------------------------------------------------------------------------

class TestAuditClear:
    def test_clear_removes_all_events(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        tmp_audit.record_decision(_make_decision())
        assert len(tmp_audit.get_events()) == 2  # precondition

        tmp_audit.clear_audit_trail()
        assert tmp_audit.get_events() == []

    def test_clear_on_empty_trail_is_safe(self, tmp_audit: AuditTrail):
        """Clearing an already-empty trail must not raise."""
        tmp_audit.clear_audit_trail()
        assert tmp_audit.get_events() == []

    def test_clear_is_persistent_in_sqlite(self, tmp_audit: AuditTrail):
        """Clearing must write to disk, not just session state."""
        tmp_audit.record_assessment(_make_assessment())
        db_path = tmp_audit.db_path

        tmp_audit.clear_audit_trail()

        # Open the same file directly and count rows.
        with sqlite3.connect(db_path) as conn:
            row_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        assert row_count == 0, "SQLite rows were not deleted — clear is not persistent"

    def test_clear_multiple_times_is_idempotent(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        tmp_audit.clear_audit_trail()
        tmp_audit.clear_audit_trail()  # second clear must not raise
        assert tmp_audit.get_events() == []


# ---------------------------------------------------------------------------
# Test 4 — New events can be created after clearing
# ---------------------------------------------------------------------------

class TestAuditPostClear:
    def test_new_assessment_after_clear(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment("PT-BEFORE"))
        tmp_audit.clear_audit_trail()

        tmp_audit.record_assessment(_make_assessment("PT-AFTER"))
        events = tmp_audit.get_events()
        assert len(events) == 1
        assert events[0]["patient_token"] == "PT-AFTER"

    def test_multiple_new_events_after_clear(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        tmp_audit.clear_audit_trail()

        tmp_audit.record_assessment(_make_assessment("PT-1"))
        tmp_audit.record_decision(_make_decision("PT-1"))
        tmp_audit.record_reassessment(_make_reassessment("PT-1"))
        assert len(tmp_audit.get_events()) == 3


# ---------------------------------------------------------------------------
# Test 5 — Hash-chain integrity
# ---------------------------------------------------------------------------

class TestHashChain:
    def test_first_event_references_genesis(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        events = sorted(tmp_audit.get_events(), key=lambda e: e["id"])
        assert events[0]["previous_hash"] == "GENESIS"

    def test_chain_links_correctly(self, tmp_audit: AuditTrail):
        """Each event's previous_hash must equal the preceding event's hash."""
        tmp_audit.record_assessment(_make_assessment("PT-C1"))
        tmp_audit.record_decision(_make_decision("PT-C1"))
        tmp_audit.record_reassessment(_make_reassessment("PT-C1"))

        events = sorted(tmp_audit.get_events(), key=lambda e: e["id"])
        for i in range(1, len(events)):
            assert events[i]["previous_hash"] == events[i - 1]["hash"], (
                f"Chain broken between event {i - 1} and event {i}"
            )

    def test_chain_resets_to_genesis_after_clear(self, tmp_audit: AuditTrail):
        """After clearing, the next event must start a fresh chain from GENESIS."""
        tmp_audit.record_assessment(_make_assessment("PT-PRE"))
        tmp_audit.clear_audit_trail()

        tmp_audit.record_assessment(_make_assessment("PT-POST"))
        events = tmp_audit.get_events()
        assert len(events) == 1
        assert events[0]["previous_hash"] == "GENESIS", (
            "Post-clear event did not restart chain from GENESIS"
        )

    def test_last_hash_in_memory_reset_after_clear(self, tmp_audit: AuditTrail):
        """self.last_hash must equal 'GENESIS' immediately after clear."""
        tmp_audit.record_assessment(_make_assessment())
        assert tmp_audit.last_hash != "GENESIS"  # precondition: chain advanced

        tmp_audit.clear_audit_trail()
        assert tmp_audit.last_hash == "GENESIS"

    def test_hashes_are_non_empty_strings(self, tmp_audit: AuditTrail):
        tmp_audit.record_assessment(_make_assessment())
        event = tmp_audit.get_events()[0]
        assert isinstance(event["hash"], str) and len(event["hash"]) > 0
        assert isinstance(event["previous_hash"], str) and len(event["previous_hash"]) > 0

    def test_chain_valid_after_clear_and_multiple_new_events(self, tmp_audit: AuditTrail):
        """Full workflow: pre-clear events then clear then new events; chain must be valid."""
        # Before clear
        tmp_audit.record_assessment(_make_assessment("PT-OLD"))
        tmp_audit.clear_audit_trail()

        # After clear
        tmp_audit.record_assessment(_make_assessment("PT-NEW1"))
        tmp_audit.record_decision(_make_decision("PT-NEW1"))
        tmp_audit.record_reassessment(_make_reassessment("PT-NEW1"))

        events = sorted(tmp_audit.get_events(), key=lambda e: e["id"])
        assert events[0]["previous_hash"] == "GENESIS"
        for i in range(1, len(events)):
            assert events[i]["previous_hash"] == events[i - 1]["hash"]
