from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from patienttriage.audit import AuditStore
from patienttriage.domain import DecisionPayload, IntakePayload, TriageAssessment
from patienttriage.policy import load_policy
from patienttriage.queue import QueuePatient, deterioration_alert, wait_limit_alert
from patienttriage.service import evaluate_intake
from patienttriage.model import ModelPrediction, ModelUnavailable


def valid_adult(**changes) -> IntakePayload:
    base = IntakePayload("test-adult", 35, 120, 75, 76, 16, 36.8, 99, 2, history_available=False)
    return replace(base, **changes)


class SafetyTests(unittest.TestCase):
    def test_pediatric_never_fast_track(self):
        outcome = evaluate_intake(valid_adult(age=12))
        self.assertEqual(outcome.safety_gate, "pediatric_rules_only")
        self.assertFalse(outcome.fast_track_eligible)

    def test_red_flag_never_fast_track(self):
        outcome = evaluate_intake(valid_adult(observable_cues=("severe_chest_pain",)))
        self.assertEqual(outcome.safety_gate, "red_flag")
        self.assertFalse(outcome.fast_track_eligible)

    def test_missing_vital_never_fast_track(self):
        outcome = evaluate_intake(valid_adult(spo2=None))
        self.assertEqual(outcome.safety_gate, "missing_data")
        self.assertFalse(outcome.fast_track_eligible)

    def test_failure_never_fast_track(self):
        outcome = evaluate_intake(valid_adult(), model_failure=True)
        self.assertEqual(outcome.status, "rules_only")
        self.assertIsNone(outcome.confidence)
        self.assertFalse(outcome.fast_track_eligible)

    def test_unavailable_model_does_not_claim_assessment(self):
        with patch("patienttriage.service.LocalModel") as model:
            model.return_value.predict.side_effect = ModelUnavailable("No local model installed.")
            outcome = evaluate_intake(valid_adult())
        self.assertEqual(outcome.safety_gate, "model_unavailable")
        self.assertFalse(outcome.fast_track_eligible)

    def test_low_confidence_never_fast_track(self):
        prediction = ModelPrediction({1: .10, 2: .10, 3: .15, 4: .35, 5: .30}, "test", ("age", "pain", "spo2"))
        with patch("patienttriage.service.LocalModel") as model:
            model.return_value.predict.return_value = prediction
            outcome = evaluate_intake(valid_adult())
        self.assertEqual(outcome.safety_gate, "low_confidence")
        self.assertFalse(outcome.fast_track_eligible)

    def test_zero_history_adult_can_be_assessed_from_intake(self):
        prediction = ModelPrediction({1: .01, 2: .01, 3: .03, 4: .15, 5: .80}, "test", ("age", "pain", "spo2"))
        with patch("patienttriage.service.LocalModel") as model:
            model.return_value.predict.return_value = prediction
            outcome = evaluate_intake(valid_adult(history_available=False))
        self.assertEqual(outcome.safety_gate, "passed")
        self.assertEqual(outcome.status, "assessment")
        self.assertFalse(outcome.fast_track_eligible)  # Validation policy remains disabled.

    def test_override_is_complete_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = AuditStore(Path(directory) / "audit.sqlite3")
            with self.assertRaises(ValueError):
                audit.record_decision(DecisionPayload("test-adult", "ESI 3", accepted=False))
            audit.record_decision(DecisionPayload("test-adult", "ESI 3", accepted=False, override_reason="Clinical judgement"))
            self.assertTrue(audit.verify_chain()[0])

    def test_wait_limit_and_deterioration_alerts(self):
        assessment = evaluate_intake(valid_adult(), model_failure=True)
        queued = QueuePatient("test-adult", (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat(), 3, assessment)
        self.assertIn("Wait-limit breach", wait_limit_alert(queued, load_policy()) or "")
        deteriorated = evaluate_intake(valid_adult(spo2=89), model_failure=True)
        self.assertIn("reassessment", deterioration_alert(3, deteriorated) or "")

    def test_surge_does_not_change_policy(self):
        self.assertFalse(load_policy()["fast_track"]["enabled"])


if __name__ == "__main__":
    unittest.main()
