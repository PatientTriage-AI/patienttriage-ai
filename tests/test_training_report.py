import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "train_model", Path(__file__).parents[1] / "scripts" / "train_model.py"
)
train_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model)


def test_score_report_includes_dashboard_metrics():
    report = train_model.score_report(
        pd.Series([1, 2, 3, 4, 5]),
        np.eye(5),
        np.array([1, 2, 3, 4, 5]),
        latency_ms=1.0,
        frame=pd.DataFrame({"age": [30, 35, 40, 45, 50]}),
    )

    assert report["accuracy"] == 1.0
    assert report["esi_5_recall"] == 1.0
    assert "macro_f1" in report
