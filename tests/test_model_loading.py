import numpy as np

from patienttriage.model import TriageModel


def test_missing_model_exposes_a_load_error(tmp_path):
    model = TriageModel(tmp_path)

    assert not model.is_available()
    assert model.load_error is not None
    assert "triage_calibrated.joblib" in model.load_error


def test_temperature_input_maps_to_temperature_c_feature():
    class CapturingModel:
        classes_ = np.array([1, 4])

        def predict_proba(self, frame):
            self.frame = frame
            return np.array([[0.1, 0.9]])

    model = TriageModel.__new__(TriageModel)
    model.model = CapturingModel()
    model.features = ["temperature_c"]
    model.metadata = {}

    suggested_esi, confidence, *_ = model.predict({"temperature": 38.2})

    assert model.model.frame.loc[0, "temperature_c"] == 38.2
    assert model.model.frame.loc[0, "temperature_c_missing"] == 0
    assert suggested_esi == 4
    assert confidence == 0.9
