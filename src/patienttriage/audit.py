import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from .domain import TriageAssessment, DecisionPayload, ReassessmentEvent

class AuditTrail:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        self.last_hash = "GENESIS"
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_token TEXT,
                    timestamp TEXT,
                    event_type TEXT,
                    payload JSON,
                    previous_hash TEXT,
                    hash TEXT
                )
            """)
            
    def _record(self, patient_token: str, event_type: str, payload: dict):
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(payload, default=str)
        
        # Hash chaining for append-only tamper-evidence
        record_content = f"{patient_token}|{timestamp}|{event_type}|{payload_str}|{self.last_hash}"
        current_hash = hashlib.sha256(record_content.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_events (patient_token, timestamp, event_type, payload, previous_hash, hash) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_token, timestamp, event_type, payload_str, self.last_hash, current_hash)
            )
        self.last_hash = current_hash
        
    def record_assessment(self, assessment: TriageAssessment):
        payload = {
            "suggested_esi": assessment.suggested_esi,
            "confidence": assessment.confidence,
            "urgent_risk": assessment.urgent_risk,
            "safety_gate_passed": assessment.safety_gate_passed,
            "fast_track_eligible": assessment.fast_track_eligible,
            "clinician_review_reason": assessment.clinician_review_reason,
            "model_version": assessment.model_version,
            "policy_version": assessment.policy_version
        }
        self._record(assessment.patient_token, "ASSESSMENT", payload)
        
    def record_decision(self, decision: DecisionPayload):
        payload = {
            "nurse_accepted": decision.nurse_accepted,
            "override_esi": decision.override_esi,
            "override_reason": decision.override_reason,
            "override_text": decision.override_text
        }
        self._record(decision.patient_token, "DECISION", payload)
        
    def record_reassessment(self, event: ReassessmentEvent):
        payload = {
            "trigger": event.trigger,
            "old_esi": event.old_esi,
            "new_esi": event.new_esi,
            "vitals_updated": event.vitals_updated
        }
        self._record(event.patient_token, "REASSESSMENT", payload)
        
    def get_events(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM audit_events ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]

    def clear_audit_trail(self) -> None:
        """Delete all persisted audit events and reset the hash chain.

        After this call:
        - The ``audit_events`` table is empty (persistent clear — survives refresh).
        - ``self.last_hash`` is reset to ``"GENESIS"`` so future events start a
          new valid hash chain without any errors.

        Raises:
            Exception: Re-raises any SQLite error so callers can surface it to
                       the user without silently swallowing failures.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM audit_events")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name='audit_events'")
            except sqlite3.OperationalError:
                pass  # sqlite_sequence may not exist if table was never populated
        # Reset in-memory chain head so the next recorded event is valid.
        self.last_hash = "GENESIS"

    # Backward-compatible alias kept for any existing callers.
    def clear_all(self) -> None:
        """Alias for :meth:`clear_audit_trail` kept for backward compatibility."""
        self.clear_audit_trail()
