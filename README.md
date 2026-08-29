# PatientTriage.ai - Round 2 prototype

An offline Streamlit clinical decision-support demo. It assists a triage nurse; it does not replace nurse judgement, make a final disposition, or support production use.

## What is implemented

- A five-screen Streamlit workflow: intake, recommendation, nurse decision, waiting queue, and audit/model card.
- Explicit nurse-selected red flags and plausibility checks. Complaint text is contextual only: it is neither keyword-matched nor used by the ML model.
- Adult-only model boundary, including rules-only escalation for every patient under 18.
- A local calibrated-LightGBM training workflow with train sites 1-4, calibration site 5, and final test site 6.
- A hard-disabled Fast-Track policy. It can be enabled only after zero urgent held-out validation passes plus clinical governance approval.
- Local SQLite append-only audit records protected by a SHA-256 hash chain, a local mock EHR adapter, and configurable reassessment alerts.
- 20 non-identifying fixtures. Facilitator-only expected ESI values are stripped before the nurse-facing app receives them.

## Run locally

```bash
# Python 3.10 or later is required.
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src streamlit run app.py
```

The first run works in safe model-unavailable mode. It will not fabricate an ML score or offer Fast-Track.

## Train with the supplied de-identified CSV

The CSV is deliberately absent from this repository. Place it outside version control and run:

```bash
PYTHONPATH=src python scripts/train_model.py --csv /secure/path/supplied_triage.csv
```

The script requires `esi_level`, a site column (`site` or source `site_id`), and the eight intake columns. The supplied CSV's `temperature` column is explicitly mapped to the model's `temperature_c` feature. It rejects absent site splits and filters out all pediatric records before training. It writes a local model, test-set evaluation report, and metadata under `models/` (ignored by Git).

## Verify acceptance checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The checks cover pediatric/red-flag/missing-vital/model-failure Fast-Track blocks, required override reasons plus ledger verification, reassessment alerts, and unchanged surge safety policy.

## Safety and privacy

This demo is for de-identified static data and local operation only. It makes no external model calls and must not receive real patient data. Before any production use, clinical governance must approve the clinical policy and validation; security, access control, retention/deletion, notices/consent, breach response, and deployment must be designed for the actual hospital context. See [the model card](docs/MODEL_CARD.md).

## Project layout

```text
app.py                         Streamlit user interface
config/policy.v1.json          Versioned, site-configurable prototype policy
src/patienttriage/             Domain, safety gates, model, audit, queue, EHR adapter
scripts/train_model.py         Reproducible local model/evaluation workflow
fixtures/                      20 non-identifying facilitator fixtures
tests/                         Automated safety acceptance checks
docs/                          Model card, architecture, demo script, production mapping
```

## Publish to GitHub

The repository is configured to exclude the virtual environment, audit records, raw data, and trained model binary. Create an empty GitHub repository first, then run the following from this project directory:

```bash
git init
git add .
git commit -m "Initial PatientTriage.ai prototype"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY-NAME.git
git push -u origin main
```

Use a private repository unless the project has been approved for public sharing. Full publishing guidance is in `docs/GITHUB_SETUP.md`.
