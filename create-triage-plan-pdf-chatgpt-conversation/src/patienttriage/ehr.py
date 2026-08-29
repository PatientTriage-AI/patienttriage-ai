from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class EhrAdapter(ABC):
    @abstractmethod
    def write_disposition(self, patient_token: str, disposition: str, metadata: dict[str, Any]) -> None: ...


class LocalMockEhrAdapter(EhrAdapter):
    """Prototype-only local mock. A production adapter maps token/disposition to FHIR Encounter."""
    def __init__(self, path: str | Path = "runtime/mock_ehr.ndjson"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_disposition(self, patient_token: str, disposition: str, metadata: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"patient_token": patient_token, "disposition": disposition, "metadata": metadata}, sort_keys=True) + "\n")
