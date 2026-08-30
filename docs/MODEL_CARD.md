# Model Card

**Model Details**
- Active Model: Logistic Regression
- Task: 5-class ESI prediction
- Inputs: Age, vitals (systolic/diastolic BP, HR, RR, Temp, SpO2), pain score, missingness flags.

**Intended Use**
- Adult decision support in triage. NOT for autonomous diagnosis.

**Limitations**
- **Pediatric Limitation:** Training data contains no pediatric records. Automatically escalates age < 18.
- **Label Leakage Protection:** Clinical notes and chief complaints are intentionally excluded.
- **ESI 5 Weakness:** The model struggles to recall true ESI 5 patients due to class imbalance and overlapping vitals with ESI 4.
