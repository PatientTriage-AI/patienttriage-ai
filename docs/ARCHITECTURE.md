# Architecture

PatientTriage.ai is designed with a strict clinical decision-support boundary.

1. **Intake Payload:** Captured via single form or batch CSV/Excel.
2. **Safety Gate (`policy.py`):** Runs configurable rules (e.g. pediatric age < 18, missing vitals).
3. **ML Assessor (`model.py`):** Uses adult-only calibrated models (e.g. LogisticRegression) to compute confidence and suggested ESI.
4. **Service Layer (`service.py`):** Orchestrates policy and model, applies Fast-Track eligibility rules.
5. **Audit Trail (`audit.py`):** SQLite append-only log with hash chaining.
6. **Queue System (`queue.py`):** Urgency-sorted queue monitoring wait times.
7. **Streamlit UI (`app.py`):** Nurse-facing dashboard.
