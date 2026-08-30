from fpdf import FPDF
from fpdf.enums import XPos, YPos
import markdown

md_text = """
# PatientTriage.ai: Business and Technical Proposal
### AI-Assisted Emergency Department Triage Decision Support

## 1. Executive Summary
Emergency Departments (EDs) are facing unprecedented overcrowding, leading to critical bottlenecks at triage. **PatientTriage.ai** offers a robust, safety-first, AI-assisted clinical decision-support prototype designed to empower triage nurses. By accurately identifying low-risk adult cases eligible for Fast-Track routing, our system reduces wait times, optimizes resource allocation, and alleviates cognitive overload for healthcare staff. Crucially, PatientTriage.ai is built on a nurse-in-the-loop philosophy, ensuring that clinical judgment remains final and that the system never automatically downgrades a patient's urgency.

## 2. The Problem
- **Triage Bottlenecks:** Manual triage during high-volume shifts leads to prolonged wait times and delayed care for critical patients.
- **Cognitive Overload:** Nurses must rapidly process complex vital signs, patient histories, and clinical cues under immense pressure.
- **Resource Misallocation:** Low-acuity patients often consume main ED beds when they could be safely routed to Fast-Track or Primary Care.
- **Safety Risks during Surges:** High operational stress can lead to inconsistencies in ESI (Emergency Severity Index) assignment.

## 3. The PatientTriage.ai Solution
PatientTriage.ai is an offline, privacy-preserving application that provides intelligent ESI recommendations. 
**Key Features Include:**
- **Batch Processing:** Seamlessly intake multiple patients simultaneously via CSV or Excel uploads.
- **Interactive Waiting Queue:** Real-time urgency-sorted queue that monitors wait limits and flags patients for reassessment upon vital deterioration.
- **Surge Simulation:** A built-in 3x Surge Mode that adapts the UI for high capacity pressure without relaxing clinical safety thresholds.
- **Explainable AI:** Displays calibrated confidence scores and highlights the top 3 vital signs influencing the model's recommendation.

## 4. Clinical Safety and Boundaries
PatientTriage.ai is engineered with strict clinical boundaries:
- **Nurse Confirmation Required:** The model is a co-pilot. Nurses can accept or override recommendations, with all overrides requiring a structured reason.
- **Stringent Safety Gate:** Automatic escalation (rules-only fallback) for pediatric cases (Age < 18), missing/implausible vitals, and nurse-observed Red Flags.
- **Zero Fast-Track False Negatives:** Fast-Track eligibility is only unlocked if the active machine learning model achieves zero false negatives on urgent cases during site-held-out validation.

## 5. Technology and Architecture
- **Machine Learning Pipeline:** Evaluates multiple algorithms (LightGBM, CatBoost, HistGradientBoosting, ExtraTrees, Logistic Regression) and automatically selects the safest model based on urgent-case recall and Fast-Track false negative rates.
- **Offline and Local Deployment:** Built with Python and Streamlit, running entirely locally to guarantee zero external API dependencies or latency.
- **Append-Only Audit Trail:** Utilizes SQLite with cryptographic hash-chaining. Every assessment, nurse decision, override, and reassessment is immutably logged for compliance and retrospective analysis.

## 6. Data Privacy and Compliance
- **DPDP Act 2023 Readiness:** Operates offline on de-identified data. No real patient identifiers are exposed or transmitted.
- **Leakage Prevention:** Chief complaints and clinical notes are intentionally excluded from the ML feature set to prevent label leakage and ensure generalizability purely on objective vitals and demographics.

## 7. Implementation Roadmap
1. **Phase 1 (Month 1): System Customization and Shadow Mode.** Deploy the offline prototype alongside standard triage processes to monitor background accuracy without influencing care.
2. **Phase 2 (Month 2): Controlled Pilot.** Enable nurse-facing UI with safety gates active. Monitor Fast-Track routing efficiency and override rates.
3. **Phase 3 (Month 3+): EHR Integration.** Utilize our existing local EHR mock adapters to map final nurse dispositions to hospital records via standard FHIR APIs.

## 8. Expected ROI
- **Reduced Wait Times:** Safely diverting 15-20% of low-acuity cases to Fast-Track.
- **Enhanced Auditability:** 100% cryptographic traceability of all triage decisions and overrides.
- **Standardized Care:** Consistent ESI application even during 3x surge conditions.

*PatientTriage.ai - Empowering nurses, protecting patients.*
"""

html = markdown.markdown(md_text)

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'PatientTriage.ai Business Proposal', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()), align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

pdf = PDF()
pdf.add_page()
pdf.set_font("Helvetica", size=10)
pdf.write_html(html)
pdf.output("Business_Proposal_PatientTriage.pdf")
print("PDF generated successfully.")
