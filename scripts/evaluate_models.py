#!/usr/bin/env python3
"""Train and evaluate multiple models, selecting the best based on specified criteria."""
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

def score_report(y_true, probabilities, labels, latency_ms, frame, y_pred=None):
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

    if y_pred is None:
        predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    else:
        predicted = y_pred
        
    urgent_actual = y_true.isin([1, 2, 3])
    urgent_predicted = np.isin(predicted, [1, 2, 3])
    low_actual = y_true.isin([4, 5])
    low_predicted = np.isin(predicted, [4, 5])
    
    # Calculate ESI 5 recall specifically
    esi_5_actual = (y_true == 5)
    esi_5_pred = (predicted == 5)
    esi_5_recall = float(recall_score(esi_5_actual, esi_5_pred, zero_division=0))

    report = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "esi_5_recall": esi_5_recall,
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
            report["subgroups"][name] = {
                "n": int(mask.sum()), 
                "urgent_recall": float(recall_score(urgent_actual[mask], urgent_predicted[mask], zero_division=0)), 
                "low_acuity_precision": float(precision_score(low_actual[mask], low_predicted[mask], zero_division=0))
            }
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Supplied de-identified CSV.")
    parser.add_argument("--out-dir", default=Path("models"), type=Path)
    args = parser.parse_args()
    
    import joblib
    import pandas as pd
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    dataset = pd.read_csv(args.csv).rename(columns=SOURCE_COLUMN_ALIASES)
    adult = dataset[dataset["age"] >= 18].copy()
    sites = adult["site"].astype(str)
    train, calibration, test = adult[sites.isin(["1", "2", "3", "4"])], adult[sites == "5"], adult[sites == "6"]
    
    weights = train[TARGET].map({1: 8, 2: 6, 3: 4, 4: 1.5, 5: 1}).fillna(1)
    
    X_train = add_features(train)
    y_train = train[TARGET]
    X_cal = add_features(calibration)
    y_cal = calibration[TARGET]
    X_test = add_features(test)
    y_test = test[TARGET]
    
    models = {
        "LightGBM": LGBMClassifier(objective="multiclass", num_class=5, n_estimators=250, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=250, learning_rate=0.05, random_seed=42, thread_count=-1, verbose=0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ])
    }
    
    results = {}
    best_model_name = None
    best_score = -1
    best_calibrated = None
    best_base = None
    best_report = None
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluating models...")
    for name, base_model in models.items():
        print(f"Training {name}...")
        if name == "LightGBM":
            base_model.fit(X_train, y_train, sample_weight=weights)
        elif name == "CatBoost":
            base_model.fit(X_train, y_train, sample_weight=weights.to_numpy())
        else:
            base_model.fit(X_train, y_train)
            
        print(f"Calibrating {name}...")
        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")
        calibrated.fit(X_cal, y_cal)
        
        started = time.perf_counter()
        probabilities = calibrated.predict_proba(X_test)
        latency_ms = (time.perf_counter() - started) * 1000 / max(1, len(test))
        
        report = score_report(y_test, probabilities, calibrated.classes_, latency_ms, test)
        
        low_probability = probabilities[:, list(calibrated.classes_).index(4)] + probabilities[:, list(calibrated.classes_).index(5)]
        urgent = y_test.isin([1, 2, 3]).to_numpy()
        candidates = low_probability >= 0.85
        ft_fn = int((candidates & urgent).sum())
        report["fast_track_false_negative_count"] = ft_fn
        report["fast_track_validation_status"] = "eligible for governance review" if ft_fn == 0 else "must remain disabled"
        
        results[name] = report
        
        # Scoring logic for selection
        score = (report["urgent_case_recall"] * 0.4 + 
                 report["low_acuity_precision"] * 0.3 + 
                 report["macro_f1"] * 0.2 - 
                 (ft_fn * 0.01))
                 
        if score > best_score:
            best_score = score
            best_model_name = name
            best_calibrated = calibrated
            best_base = base_model
            best_report = report

    print(f"\nSelected Model: {best_model_name}")
    
    # Save best model
    version = f"{best_model_name.lower()}-site-split-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    joblib.dump({
        "model": best_calibrated, 
        "base_model": best_base, 
        "metadata": {
            "model_name": best_model_name,
            "model_version": version, 
            "features": FEATURES, 
            "training_sites": [1, 2, 3, 4], 
            "calibration_site": 5, 
            "test_site": 6
        }
    }, args.out_dir / "triage_calibrated.joblib")
    
    (args.out_dir / "evaluation_report.json").write_text(json.dumps(best_report, indent=2), encoding="utf-8")
    (args.out_dir / "model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "model_metadata.json").write_text(json.dumps({
        "model_name": best_model_name,
        "model_version": version, 
        "status": "Trained locally", 
        "training_rows": len(train), 
        "calibration_rows": len(calibration), 
        "test_rows": len(test), 
        "pediatric_records_used": 0, 
        "fast_track_validation_status": best_report["fast_track_validation_status"]
    }, indent=2), encoding="utf-8")
    print(f"Wrote local model and evaluation reports to {args.out_dir}")

if __name__ == "__main__":
    main()
