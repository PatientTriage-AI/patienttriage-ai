from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def nurse_facing_scenarios() -> list[dict]:
    """Remove facilitator-only expected ESI values before they reach the nurse UI."""
    rows = json.loads((ROOT / "fixtures" / "facilitator_scenarios.json").read_text(encoding="utf-8"))
    return [{key: value for key, value in row.items() if key != "expected_esi"} for row in rows]
