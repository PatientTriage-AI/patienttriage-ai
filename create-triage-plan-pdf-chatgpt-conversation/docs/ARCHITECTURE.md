# Prototype architecture

```mermaid
flowchart LR
    Nurse["Triage nurse"] --> Intake["New-arrival intake\ncurrent vitals, pain, context"]
    Intake --> Gate{"Safety gate\nadult? complete? plausible?\nred flags? OOD? model healthy?"}
    Gate -->|"Blocked"| Review["Rules-only or\nclinician review required"]
    Gate -->|"Passes"| Model["Local calibrated\nLightGBM + SHAP"]
    Model --> Rec["Recommendation\nESI, confidence, urgent risk\nwhy not Fast-Track"]
    Review --> Rec
    Rec --> Decision["Nurse accepts or overrides\nstructured reason required"]
    Decision --> Audit["SQLite append-only audit\nSHA-256 hash chain"]
    Decision --> EHR["Local mock EHR adapter\nfuture FHIR mapping"]
    Decision --> Queue["Urgency-sorted waiting queue"]
    Queue --> Reassess["Wait-limit or deterioration\nreassessment alert"]
    Reassess --> Intake
```

## Data boundary

Only the eight adult intake variables and their missingness flags reach the model. The UI may display context (complaint, sex, history availability, observable cues), but that context is excluded from inference. Complaint text is never safety-keyword matched.

## Future FHIR mapping

`EhrAdapter.write_disposition(...)` is implemented locally for the demo. A production mapping should be designed with the hospital, but likely resources include `Encounter` (triage/disposition), `Observation` (vitals), `AuditEvent` (access/changes), `Provenance` (assessment source/version), and a governed extension or `Communication` for the nurse-confirmed recommendation. No FHIR write occurs in this prototype.
