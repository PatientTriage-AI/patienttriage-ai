from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .domain import DecisionPayload, IntakePayload, TriageAssessment, VitalsPayload, utc_now


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class AuditStore:
    """Append-only SQLite event ledger with a verifiable SHA-256 hash chain."""

    def __init__(self, path: str | Path = "runtime/triage_audit.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL, patient_token TEXT NOT NULL, created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL, previous_hash TEXT, event_hash TEXT NOT NULL
                )"""
            )
            connection.execute("""CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;""")
            connection.execute("""CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;""")

    def _append(self, event_type: str, patient_token: str, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = hashlib.sha256(f"{patient_token}|{event_type}|{utc_now()}|{_canonical(payload)}".encode()).hexdigest()[:24]
        created_at = utc_now()
        with self._connect() as connection:
            last = connection.execute("SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1").fetchone()
            previous_hash = last["event_hash"] if last else "GENESIS"
            body = {"event_id": event_id, "event_type": event_type, "patient_token": patient_token, "created_at": created_at, "payload": payload, "previous_hash": previous_hash}
            event_hash = hashlib.sha256(_canonical(body).encode()).hexdigest()
            connection.execute(
                "INSERT INTO audit_events (event_id,event_type,patient_token,created_at,payload_json,previous_hash,event_hash) VALUES (?,?,?,?,?,?,?)",
                (event_id, event_type, patient_token, created_at, _canonical(payload), previous_hash, event_hash),
            )
        return {**body, "event_hash": event_hash}

    def record_assessment(self, intake: IntakePayload, assessment: TriageAssessment) -> dict[str, Any]:
        return self._append("assessment", intake.patient_token, {"input_snapshot": intake.snapshot(), "assessment": assessment.snapshot()})

    def record_decision(self, decision: DecisionPayload, assessment: TriageAssessment | None = None) -> dict[str, Any]:
        if not decision.accepted and not decision.override_reason:
            raise ValueError("A structured override reason is required when the recommendation is overridden.")
        payload = {"decision": asdict(decision), "recommendation": assessment.snapshot() if assessment else None}
        return self._append("nurse_decision", decision.patient_token, payload)

    def record_vitals(self, vitals: VitalsPayload) -> dict[str, Any]:
        return self._append("reassessment_vitals", vitals.patient_token, asdict(vitals))

    def timeline(self, query: str = "", limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_events"
        values: tuple[Any, ...] = ()
        if query:
            sql += " WHERE patient_token LIKE ? OR event_type LIKE ? OR payload_json LIKE ?"
            value = f"%{query}%"
            values = (value, value, value)
        sql += " ORDER BY sequence DESC LIMIT ?"
        values += (limit,)
        with self._connect() as connection:
            return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in connection.execute(sql, values)]

    def verify_chain(self) -> tuple[bool, str]:
        previous_hash = "GENESIS"
        with self._connect() as connection:
            for row in connection.execute("SELECT * FROM audit_events ORDER BY sequence"):
                body = {"event_id": row["event_id"], "event_type": row["event_type"], "patient_token": row["patient_token"], "created_at": row["created_at"], "payload": json.loads(row["payload_json"]), "previous_hash": row["previous_hash"]}
                if row["previous_hash"] != previous_hash or hashlib.sha256(_canonical(body).encode()).hexdigest() != row["event_hash"]:
                    return False, f"Hash-chain verification failed at sequence {row['sequence']}."
                previous_hash = row["event_hash"]
        return True, "Hash chain is valid."
