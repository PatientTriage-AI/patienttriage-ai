import streamlit as st
import pandas as pd
import json
import uuid
import io
from pathlib import Path
from datetime import datetime, timezone

from patienttriage.domain import IntakePayload, DecisionPayload, VitalsPayload, ReassessmentEvent
from patienttriage.policy import SafetyPolicy
from patienttriage.model import TriageModel
from patienttriage.audit import AuditTrail
from patienttriage.service import TriageService
from patienttriage.queue import PatientQueue
from patienttriage.ehr import EhrAdapter

st.set_page_config(page_title="PatientTriage.ai", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZATION ---
@st.cache_resource
def init_services():
    base_dir = Path(__file__).parent
    policy = SafetyPolicy(base_dir / "config" / "policy.v1.json")
    model = TriageModel(base_dir / "models")
    audit = AuditTrail(base_dir / "runtime" / "audit.sqlite3")
    # Ensure runtime dir exists
    (base_dir / "runtime").mkdir(exist_ok=True)
    service = TriageService(policy, model, audit)
    return service, policy

if "queue" not in st.session_state:
    st.session_state.queue = PatientQueue()
if "ehr" not in st.session_state:
    st.session_state.ehr = EhrAdapter()
if "simulate_ml_failure" not in st.session_state:
    st.session_state.simulate_ml_failure = False
if "surge_multiplier" not in st.session_state:
    st.session_state.surge_multiplier = 1

service, policy_config = init_services()

# --- CSS / STYLING ---
st.markdown("""
<style>
    .safety-banner {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="safety-banner">CLINICAL DECISION SUPPORT ONLY | NURSE CONFIRMATION REQUIRED | NOT FOR PRODUCTION USE</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings & Simulation")
    st.session_state.simulate_ml_failure = st.toggle("Simulate ML Failure", value=st.session_state.simulate_ml_failure)
    
    surge = st.radio("Operating Mode", ["1x (Normal)", "3x (Surge)"])
    st.session_state.surge_multiplier = 3 if "3x" in surge else 1
    if st.session_state.surge_multiplier == 3:
        st.warning("⚠️ Surge mode changes operational pressure only. Safety thresholds remain unchanged.")
        
    st.markdown("---")
    st.info(f"Policy: {policy_config.version()}")
    if service.model.is_available():
        st.info(f"Model: {service.model.metadata.get('model_version', 'unknown')}")
    else:
        st.error("ML model unavailable. Run `python scripts/train_model.py --csv Data_preProcessing/fedmml_ed_triage_dataset.csv` to create `models/triage_calibrated.joblib`.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["New Arrival", "Batch Intake", "Waiting Queue", "Audit Trail", "Model Evaluation"])

# --- TAB 1: NEW ARRIVAL (SINGLE) ---
with tab1:
    st.header("Single Patient Intake")
    
    with st.form("single_intake_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            sex = st.selectbox("Sex", ["M", "F", "Other"])
            history = st.checkbox("History Available", value=True)
            complaint = st.text_input("Chief Complaint")
        with col2:
            sys_bp = st.number_input("Systolic BP", min_value=0, value=120)
            dia_bp = st.number_input("Diastolic BP", min_value=0, value=80)
            hr = st.number_input("Heart Rate", min_value=0, value=80)
            rr = st.number_input("Resp Rate", min_value=0, value=16)
        with col3:
            temp = st.number_input("Temperature (°C)", min_value=20.0, max_value=45.0, value=37.0)
            spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=98)
            pain = st.number_input("Pain Score", min_value=0, max_value=10, value=0)
            
        st.markdown("#### Red Flags & Cues")
        red_flags = st.multiselect("Select Red Flags if present", policy_config.config["red_flags"])
        
        submitted = st.form_submit_button("Assess Patient")
        
        if submitted:
            intake = IntakePayload(
                age=age, sex=sex, systolic_bp=sys_bp, diastolic_bp=dia_bp,
                heart_rate=hr, respiratory_rate=rr, temperature=temp,
                spo2=spo2, pain_score=pain, history_available=history,
                complaint=complaint, observable_cues=[], red_flags=red_flags
            )
            assessment = service.evaluate_intake(intake, simulate_ml_failure=st.session_state.simulate_ml_failure)
            
            # Show Assessment Results
            st.markdown("### Assessment Result")
            if not assessment.safety_gate_passed:
                st.error(f"CLINICIAN REVIEW REQUIRED: {assessment.clinician_review_reason}")
                if assessment.is_pediatric:
                    st.warning("PEDIATRIC CASE - Adult ML model unavailable. Rules-only escalation.")
            else:
                st.success("Safety Gate: PASSED")
                
            colA, colB, colC = st.columns(3)
            colA.metric("Suggested ESI", assessment.suggested_esi if assessment.suggested_esi else "N/A")
            colB.metric("Confidence", f"{assessment.confidence:.1%}" if assessment.confidence else "N/A")
            colC.metric("Urgent Risk", f"{assessment.urgent_risk:.1%}" if assessment.urgent_risk else "N/A")
            
            if assessment.fast_track_eligible:
                st.success("Fast-Track: ELIGIBLE")
            else:
                st.warning(f"Fast-Track: NOT ELIGIBLE ({assessment.fast_track_reason})")
                
            if assessment.shap_factors:
                st.markdown("#### Model explanation — not clinical reasoning")
                for f in assessment.shap_factors:
                    st.write(f"- {f['direction']} **{f['feature']}**: {f['value']}")
                    
            st.session_state.queue.add_patient(assessment, intake)
            st.info("Patient added to waiting queue for final disposition.")

# --- TAB 2: BATCH INTAKE ---
with tab2:
    st.header("Batch Patient Intake")
    st.write("Interface mein multiple patients ki information ek saath enter honi chahiye — Excel sheet ya CSV file se input le lo.")
    
    colA, colB = st.columns(2)
    with colA:
        csv_template = "age,sex,systolic_bp,diastolic_bp,heart_rate,respiratory_rate,temperature,spo2,pain_score,history_available,complaint\n45,M,120,80,80,16,37.0,98,0,True,Chest pain"
        st.download_button("Download CSV Template", csv_template, "template.csv", "text/csv")
        
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.write(f"Loaded {len(df)} records.")
            
            if st.button("Process Batch"):
                valid_count = 0
                invalid_count = 0
                
                for _, row in df.iterrows():
                    try:
                        intake = IntakePayload(
                            age=int(row.get('age', 45)),
                            sex=str(row.get('sex', 'M')),
                            systolic_bp=float(row.get('systolic_bp', 120)) if pd.notna(row.get('systolic_bp')) else None,
                            diastolic_bp=float(row.get('diastolic_bp', 80)) if pd.notna(row.get('diastolic_bp')) else None,
                            heart_rate=float(row.get('heart_rate', 80)) if pd.notna(row.get('heart_rate')) else None,
                            respiratory_rate=float(row.get('respiratory_rate', 16)) if pd.notna(row.get('respiratory_rate')) else None,
                            temperature=float(row.get('temperature', 37.0)) if pd.notna(row.get('temperature')) else None,
                            spo2=float(row.get('spo2', 98)) if pd.notna(row.get('spo2')) else None,
                            pain_score=float(row.get('pain_score', 0)) if pd.notna(row.get('pain_score')) else None,
                            history_available=bool(row.get('history_available', True)),
                            complaint=str(row.get('complaint', '')),
                            observable_cues=[],
                            red_flags=[]
                        )
                        assessment = service.evaluate_intake(intake, simulate_ml_failure=st.session_state.simulate_ml_failure)
                        st.session_state.queue.add_patient(assessment, intake)
                        valid_count += 1
                    except Exception as e:
                        invalid_count += 1
                        
                st.success(f"Processed {valid_count} valid patients. {invalid_count} errors.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# --- TAB 3: WAITING QUEUE ---
with tab3:
    st.header("Waiting Queue")
    
    st.session_state.queue.check_reassessments(policy_config.config)
    q = st.session_state.queue.get_queue(st.session_state.surge_multiplier)
    
    if st.session_state.surge_multiplier == 3:
        st.metric("Displayed Queue Volume (Surge)", len(q) * 3)
    else:
        st.metric("Displayed Queue Volume", len(q))
        
    for item in q:
        with st.expander(f"{item['token']} | Urgency: {item['final_esi']} | Wait: {(datetime.now(timezone.utc) - item['added_at']).seconds // 60}m"):
            st.write(f"**Status:** {item['disposition']}")
            if item['reassessment_needed']:
                st.error(f"⚠️ REASSESSMENT REQUIRED: {item['reassessment_reason']}")
                
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Intake Details**")
                st.json(item['intake'].__dict__)
            with col2:
                st.write("**Assessment**")
                st.write(f"Suggested ESI: {item['assessment'].suggested_esi}")
                st.write(f"Confidence: {item['assessment'].confidence}")
                
                new_esi = st.selectbox("Final Disposition", [1, 2, 3, 4, 5, "Fast-Track", "Discharge"], key=f"disp_{item['token']}")
                override_reason = st.text_input("Override Reason (if applicable)", key=f"rsn_{item['token']}")
                
                if st.button("Confirm Decision", key=f"btn_{item['token']}"):
                    decision = DecisionPayload(
                        patient_token=item['token'],
                        nurse_accepted=str(new_esi) == str(item['assessment'].suggested_esi),
                        override_esi=int(new_esi) if str(new_esi).isdigit() else None,
                        override_reason=override_reason if str(new_esi) != str(item['assessment'].suggested_esi) else None,
                        override_text=""
                    )
                    service.record_decision(decision)
                    item['disposition'] = "Processed"
                    item['final_esi'] = decision.override_esi or item['assessment'].suggested_esi
                    st.success("Decision recorded.")
                    st.rerun()

# --- TAB 4: AUDIT TRAIL ---
with tab4:
    st.header("Audit Trail")
    events = service.audit.get_events()
    if events:
        df_events = pd.DataFrame(events)
        st.dataframe(df_events, use_container_width=True)
    else:
        st.write("No events recorded yet.")

# --- TAB 5: MODEL EVALUATION ---
with tab5:
    st.header("Model Evaluation Dashboard")
    try:
        report_path = Path(__file__).parent / "models" / "evaluation_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
                
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Accuracy", f"{report.get('accuracy', 0):.1%}")
            col2.metric("Urgent Case Recall", f"{report.get('urgent_case_recall', 0):.1%}")
            col3.metric("Low Acuity Precision", f"{report.get('low_acuity_precision', 0):.1%}")
            col4.metric("ESI 5 Recall", f"{report.get('esi_5_recall', 0):.1%}")
            
            st.write(f"Fast-Track False Negatives: **{report.get('fast_track_false_negative_count')}**")
            st.write(f"Fast-Track Validation Status: **{report.get('fast_track_validation_status')}**")
            
            st.subheader("Confusion Matrix")
            st.write(pd.DataFrame(report["confusion_matrix"]["values"], 
                                  columns=[f"Pred ESI {i}" for i in report["confusion_matrix"]["labels"]],
                                  index=[f"Actual ESI {i}" for i in report["confusion_matrix"]["labels"]]))
        else:
            st.warning("Evaluation report not found. Run model training script.")
    except Exception as e:
        st.error(f"Failed to load report: {e}")
