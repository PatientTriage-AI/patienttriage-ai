import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class TriageModel:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model = None
        self.base_model = None
        self.metadata = None
        self.features = []
        self.load_error = None
        self._load_model()
        
    def _load_model(self):
        try:
            data = joblib.load(self.models_dir / "triage_calibrated.joblib")
            self.model = data["model"]
            self.base_model = data["base_model"]
            self.metadata = data["metadata"]
            self.features = self.metadata["features"]
        except Exception as e:
            self.load_error = str(e)
            print(f"Failed to load model: {self.load_error}")
            self.model = None
            
    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, inputs: Dict[str, Any]) -> Tuple[int, float, float, Dict[int, float], list]:
        """
        Returns: suggested_esi, confidence, urgent_risk, probabilities, shap_factors
        """
        if not self.is_available():
            raise RuntimeError("Model unavailable")
            
        # Prepare DataFrame
        row = {}
        for f in self.features:
            # The UI/domain calls this vital ``temperature`` while the training
            # dataset uses ``temperature_c``. Keep inference aligned with the
            # feature name used to train the artifact.
            val = inputs.get("temperature") if f == "temperature_c" else inputs.get(f)
            row[f] = val
            row[f"{f}_missing"] = 1 if val is None or pd.isna(val) else 0
            if row[f"{f}_missing"] == 1:
                row[f] = 0.0 # Placeholder for missing, standard imputer will handle it if pipeline
                
        df = pd.DataFrame([row])
        
        # Ensure column order matches training
        feature_cols = self.features + [f"{f}_missing" for f in self.features]
        df = df[feature_cols]
        
        probas = self.model.predict_proba(df)[0]
        classes = self.model.classes_
        
        prob_dict = {int(c): float(p) for c, p in zip(classes, probas)}
        
        suggested_esi = int(classes[np.argmax(probas)])
        confidence = float(np.max(probas))
        urgent_risk = sum(prob_dict.get(c, 0.0) for c in [1, 2, 3])
        
        # Mock SHAP for robustness in prototype if real SHAP fails on CalibratedClassifierCV
        # In a real app we'd use shap.Explainer
        shap_factors = self._get_factors(df)
        
        return suggested_esi, confidence, urgent_risk, prob_dict, shap_factors
        
    def _get_factors(self, df: pd.DataFrame) -> list:
        # Simplistic mock of top factors based on deviation from normal
        # A real implementation uses shap.Explainer(self.base_model)
        factors = []
        normals = {"heart_rate": 80, "systolic_bp": 120, "spo2": 98, "temperature_c": 37.0, "respiratory_rate": 16, "pain_score": 0}
        
        deviations = {}
        for col in self.features:
            val = df.iloc[0][col]
            missing = df.iloc[0][f"{col}_missing"]
            if missing == 0 and col in normals:
                # normalize deviation roughly
                if col == "temperature_c":
                    dev = abs(val - normals[col]) / 1.0
                elif col == "spo2":
                    dev = abs(val - normals[col]) / 2.0
                else:
                    dev = abs(val - normals[col]) / (normals[col] * 0.2 + 1)
                deviations[col] = dev
                
        sorted_devs = sorted(deviations.items(), key=lambda x: x[1], reverse=True)
        for col, dev in sorted_devs[:3]:
            val = df.iloc[0][col]
            norm = normals[col]
            direction = "↑" if val > norm else "↓"
            factors.append({"feature": col, "value": val, "direction": direction})
            
        return factors
