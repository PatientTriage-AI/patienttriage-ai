# PatientTriage.ai - Round 2 Prototype

## Overview
PatientTriage.ai is an offline, safety-first clinical decision-support application built for emergency department triage nurses. It helps identify low-risk adult cases for Fast-Track while prioritizing patient safety.

## Key Features
- **Nurse-in-the-Loop:** Clinical decision support only. Nurse confirmation is strictly required.
- **Batch Intake:** Support for CSV and Excel upload to intake multiple patients at once.
- **Safety Gate:** Rule-based fallback for pediatric cases, red flags, missing vitals, and ML uncertainty.
- **Append-Only Audit Trail:** SQLite-backed immutable logging for all assessments and decisions.
- **Surge Simulation:** Simulates 3x volume pressure without compromising safety thresholds.
- **Live ML Evaluation:** Real-time metrics based on 5 selected ML models evaluated on a site-held-out test set.

## Installation
```bash
# Require Python 3.10+ (tested with 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Running the Application
```bash
PYTHONPATH=src streamlit run app.py
```

## Model Training & Evaluation
To regenerate models and select the safest model for production:
```bash
python scripts/evaluate_models.py --csv Data_preProcessing/fedmml_ed_triage_dataset.csv
```

## Testing
Run the automated test suite:
```bash
pytest tests/
```

## Privacy & Limitations
- **No Pediatric Data:** The model was trained entirely on adult data. Pediatric cases trigger an immediate rules-only fallback.
- **Label Leakage Protection:** Clinical notes and chief complaints are explicitly excluded from ML features.
- **Data Protection:** No raw patient datasets are committed to this repo.
- **ESI 5 Weakness:** Explicitly tracked and documented in the model card.
