#!/usr/bin/env python3
"""Train the local, adult-only LightGBM model with site-held-out calibration and reporting."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

FEATURES = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "spo2", "pain_score"]
TARGET = "esi_level"
SOURCE_COLUMN_ALIASES = {"site_id": "site", "temperature": "temperature_c"}


def add_features(frame):
    result = frame[FEATURES].copy()
    for column in FEATURES:
        result[f"{column}_missing"] = result[column].isna().astype(int)
    return result


def score_report(y_true, probabilities, labels, latency_ms, frame):
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    urgent_actual = y_true.isin([1, 2, 3])
    urgent_predicted = np.isin(predicted, [1, 2, 3])
    low_actual = y_true.isin([4, 5])
    low_predicted = np.isin(predicted, [4, 5])
    esi_5_actual = y_true == 5
    esi_5_predicted = predicted == 5
    report = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "esi_5_recall": float(recall_score(esi_5_actual, esi_5_predicted, zero_division=0)),
        "confusion_matrix": {"labels": [1, 2, 3, 4, 5], "values": confusion_matrix(y_true, predicted, labels=[1, 2, 3, 4, 5]).tolist()},
        "urgent_case_recall": float(recall_score(urgent_actual, urgent_predicted, zero_division=0)),
        "low_acuity_precision": float(precision_score(low_actual, low_predicted, zero_division=0)),
        "fast_track_false_negative_count": None,
        "median_prediction_latency_ms": round(float(latency_ms), 3),
        "calibration": {}, "subgroups": {},
    }
    for esi in [1, 2, 3, 4, 5]:
        actual, predicted_probability = (y_true == esi).astype(int), probabilities[:, list(labels).index(esi)]
        observed, estimated = calibration_curve(actual, predicted_probability, n_bins=10, strategy="quantile")
        report["calibration"][f"esi_{esi}"] = {"predicted": estimated.tolist(), "observed": observed.tolist()}
    if "sex" in frame:
        groups = {f"sex:{value}": frame["sex"] == value for value in frame["sex"].dropna().unique()}
    else:
        groups = {}
    groups.update({"age:18-64": frame["age"].between(18, 64), "age:65+": frame["age"] >= 65})
    if "site" in frame:
        groups.update({f"site:{value}": frame["site"].astype(str) == str(value) for value in frame["site"].dropna().unique()})
    for name, mask in groups.items():
        if int(mask.sum()) > 0:
            report["subgroups"][name] = {"n": int(mask.sum()), "urgent_recall": float(recall_score(urgent_actual[mask], urgent_predicted[mask], zero_division=0)), "low_acuity_precision": float(precision_score(low_actual[mask], low_predicted[mask], zero_division=0))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Supplied de-identified CSV. It is never committed.")
    parser.add_argument("--out-dir", default=Path("models"), type=Path)
    args = parser.parse_args()
    try:
        import joblib
        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: python -m pip install -e .") from exc
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError) as exc:
        LGBMClassifier = None
        print(f"LightGBM unavailable ({exc}); using the portable LogisticRegression fallback.")
    # Preserve the source file untouched. These are explicit, documented source-to-model aliases.
    dataset = pd.read_csv(args.csv).rename(columns=SOURCE_COLUMN_ALIASES)
    required = set(FEATURES + [TARGET, "site"])
    absent = sorted(required - set(dataset.columns))
    if absent:
        raise SystemExit(f"CSV is missing required columns after source alias mapping: {absent}")
    if len(dataset) != 87234:
        print(f"WARNING: expected 87,234 rows, received {len(dataset):,}; review data provenance before using results.")
    adult = dataset[dataset["age"] >= 18].copy()
    if adult.empty:
        raise SystemExit("No adult records found. The prototype model must never train or score pediatric records.")
    sites = adult["site"].astype(str)
    train, calibration, test = adult[sites.isin(["1", "2", "3", "4"])], adult[sites == "5"], adult[sites == "6"]
    if min(len(train), len(calibration), len(test)) == 0:
        raise SystemExit("Site split requires data from sites 1-4 (train), 5 (calibrate), and 6 (test).")
    weights = train[TARGET].map({1: 8, 2: 6, 3: 4, 4: 1.5, 5: 1}).fillna(1)
    if LGBMClassifier is not None:
        base = LGBMClassifier(objective="multiclass", num_class=5, n_estimators=250, learning_rate=0.05, random_state=42, n_jobs=-1)
        base.fit(add_features(train), train[TARGET], sample_weight=weights)
        model_name = "LightGBM"
    else:
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ])
        base.fit(add_features(train), train[TARGET])
        model_name = "LogisticRegression"
    # scikit-learn 1.6+ replaces cv="prefit" with FrozenEstimator. The sites are
    # deliberately disjoint: train sites 1-4, calibration site 5, test site 6.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(add_features(calibration), calibration[TARGET])
    started = time.perf_counter()
    probabilities = calibrated.predict_proba(add_features(test))
    latency_ms = (time.perf_counter() - started) * 1000 / max(1, len(test))
    report = score_report(test[TARGET], probabilities, calibrated.classes_, latency_ms, test)
    low_probability = probabilities[:, list(calibrated.classes_).index(4)] + probabilities[:, list(calibrated.classes_).index(5)]
    urgent = test[TARGET].isin([1, 2, 3]).to_numpy()
    candidates = low_probability >= 0.85
    report["fast_track_false_negative_count"] = int((candidates & urgent).sum())
    report["fast_track_validation_status"] = "eligible for governance review" if report["fast_track_false_negative_count"] == 0 else "must remain disabled"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    version = f"{model_name.lower()}-site-split-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    joblib.dump({"model": calibrated, "base_model": base, "metadata": {"model_name": model_name, "model_version": version, "features": FEATURES, "training_sites": [1, 2, 3, 4], "calibration_site": 5, "test_site": 6, "fast_track_validation_status": report["fast_track_validation_status"]}}, args.out_dir / "triage_calibrated.joblib")
    (args.out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "model_metadata.json").write_text(json.dumps({"model_name": model_name, "model_version": version, "status": "Trained locally", "training_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test), "pediatric_records_used": 0, "fast_track_validation_status": report["fast_track_validation_status"]}, indent=2), encoding="utf-8")
    print(f"Wrote local model and evaluation report to {args.out_dir}")


if __name__ == "__main__":
    main()
