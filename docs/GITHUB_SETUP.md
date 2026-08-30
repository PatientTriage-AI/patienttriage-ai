# Publish to GitHub

## What is safe to publish

The repository excludes the virtual environment, local audit database, raw data folders, and trained model binaries. Do not add a real-patient or identifying dataset. The supplied local CSV stays outside the repository.

The application runs in safe model-unavailable mode after cloning. Each teammate who needs local ML predictions must use the approved de-identified source data and train a model locally.

## First push

1. Create an empty repository on GitHub. Do not initialize it with a README, `.gitignore`, or license because this project already has those files.
2. From the project directory, use the commands shown in the README's **Publish to GitHub** section.
3. Use a private repository unless the complete source, fixture data, and project documentation have been cleared for public release.

## Before inviting teammates

- Confirm that no raw CSV, `.venv`, `runtime/`, or `models/triage_calibrated.joblib` appears in `git status`.
- Read `docs/MODEL_CARD.md` and keep the clinical-decision-support banner intact.
- Ask collaborators to run automated checks before raising a pull request.
