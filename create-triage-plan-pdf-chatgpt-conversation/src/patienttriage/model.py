"""Local calibrated LightGBM artifact loader; no network or external model calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import IntakePayload

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
FEATURES = ("age", "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "spo2", "pain_score")


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrediction:
    probabilities: dict[int, float]
    model_version: str
    top_factors: tuple[str, ...]

    @property
    def suggested_esi(self) -> int:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.suggested_esi]

    @property
    def urgent_risk(self) -> float:
        return sum(self.probabilities.get(level, 0.0) for level in (1, 2, 3))

    @property
    def low_acuity_probability(self) -> float:
        return self.probabilities.get(4, 0.0) + self.probabilities.get(5, 0.0)


class LocalModel:
    """Loads a joblib bundle made by scripts/train_model.py and returns calibrated probabilities."""

    def __init__(self, artifact_path: Path | None = None):
        self.artifact_path = artifact_path or MODEL_DIR / "triage_calibrated.joblib"

    def predict(self, payload: IntakePayload) -> ModelPrediction:
        if not self.artifact_path.exists():
            raise ModelUnavailable("No locally trained calibrated model artifact is installed.")
        try:
            import joblib  # optional until model training/runtime dependencies are installed
            import pandas as pd
        except ImportError as exc:
            raise ModelUnavailable("Model dependencies are not installed.") from exc
        bundle: dict[str, Any] = joblib.load(self.artifact_path)
        values = {name: getattr(payload, name) for name in FEATURES}
        for name in FEATURES:
            values[f"{name}_missing"] = int(values[name] is None)
        dataframe = pd.DataFrame([values])
        probabilities = bundle["model"].predict_proba(dataframe)[0]
        classes = bundle["model"].classes_
        probability_map = {int(label): float(value) for label, value in zip(classes, probabilities)}
        suggested_esi = max(probability_map, key=probability_map.get)
        metadata = bundle.get("metadata", {})
        try:
            import shap
            raw_model = bundle["base_model"]
            values_by_class = shap.TreeExplainer(raw_model).shap_values(dataframe)
            class_index = list(classes).index(suggested_esi)
            if isinstance(values_by_class, list):
                signed_values = values_by_class[class_index][0]
            else:  # SHAP 0.45+ may return sample x feature x class.
                signed_values = values_by_class[0, :, class_index]
            ranked = sorted(zip(dataframe.columns, signed_values), key=lambda item: abs(item[1]), reverse=True)[:3]
            factors = tuple(f"{name.replace('_', ' ')} ({'increases' if value > 0 else 'decreases'} ESI {suggested_esi})" for name, value in ranked)
        except Exception as exc:
            raise ModelUnavailable("SHAP explanation could not be generated for this local model.") from exc
        return ModelPrediction(
            probabilities=probability_map,
            model_version=str(metadata.get("model_version", "unversioned-local-model")),
            top_factors=factors,
        )


def model_metadata() -> dict[str, Any]:
    path = MODEL_DIR / "model_metadata.json"
    if not path.exists():
        return {"model_version": "unavailable", "status": "No trained artifact installed"}
    return json.loads(path.read_text(encoding="utf-8"))
