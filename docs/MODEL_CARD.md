# Model card - PatientTriage.ai Round 2 prototype

## Intended use

This local prototype gives a triage nurse a decision-support recommendation from current adult intake vitals and pain score. It is not an autonomous triage system, does not set care priority, and requires explicit nurse confirmation for every route.

## Inputs and exclusions

The model may use age, systolic/diastolic blood pressure, heart rate, respiratory rate, temperature, SpO2, pain score, and explicit missingness flags. Complaint, sex, history availability, and observable clinician cues are context-only. It must never use patient/encounter identifiers, site, country, timestamps, labs, notes, complaint text, or sex as inference features.

## Material limitations

- The supplied data contain no pediatric records. Patients under 18 are not scored by the adult model; they receive rules-only escalation.
- Chief complaint maps perfectly to ESI in the source data and notes repeat label-revealing templates. Neither can be ML input or evidence of real-world performance.
- No model artifact is bundled because the source CSV is not available in this workspace. The application intentionally runs in clinician-review mode rather than inventing a score.
- Prototype thresholds are not clinical policy. Site-specific policy and all Fast-Track activation require hospital clinical governance approval.

## Safety behavior

Red flags are only nurse-selected cues from the versioned policy; they are not inferred from complaint strings. Missing, implausible, out-of-distribution, low-confidence, pediatric, red-flag, and model-failure inputs are ineligible for Fast-Track. Low confidence is never an automatic downgrade.

Fast-Track is disabled by default. Activation demands a documented held-out validation threshold, zero urgent cases among validation passes, and clinical governance approval. Surge display does not change a threshold or reorder urgent patients.

## Evaluation requirements

The training script uses sites 1-4 for training, site 5 for calibration, and site 6 for final results. Its report contains a five-class confusion matrix, per-class calibration curves, urgent-case recall, low-acuity precision, Fast-Track false-negative count, latency, and sex / 18-64 / 65+ / site subgroup summaries. No pediatric performance claim may be made.

## Privacy and production posture

The demo is local and uses de-identified/static scenarios only. Production design must address role-based access, encryption in transit and at rest, purpose limitation, retention and deletion, notice/consent handling, breach response, and immutable access auditing under the applicable governance framework, including the Digital Personal Data Protection Act, 2023 where applicable.
