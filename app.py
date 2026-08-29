from __future__ import annotations

import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from patienttriage.audit import AuditStore
from patienttriage.domain import DecisionPayload, IntakePayload, VitalsPayload
from patienttriage.ehr import LocalMockEhrAdapter
from patienttriage.fixtures import nurse_facing_scenarios
from patienttriage.model import model_metadata
from patienttriage.policy import load_policy
from patienttriage.queue import QueuePatient, deterioration_alert, sorted_queue, wait_limit_alert
from patienttriage.service import evaluate_intake

st.set_page_config(page_title="PatientTriage.ai | Round 2", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    div[data-testid="stMetric"] { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 14px; }
    div[data-testid="stMetricLabel"] { color: #cbd5e1; }
    .demo-step { padding: .65rem .8rem; border-radius: .5rem; background: #172033; margin: .35rem 0; }
</style>
""", unsafe_allow_html=True)
st.error("CLINICAL DECISION SUPPORT ONLY - nurse confirmation required - not for production use.")
policy = load_policy()
store = AuditStore()
PAGES = ["New arrival", "Recommendation", "Nurse decision", "Waiting queue", "Audit & model card"]

if "queue" not in st.session_state:
    st.session_state.queue = []
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "intake" not in st.session_state:
    st.session_state.intake = None
if "active_screen" not in st.session_state:
    st.session_state.active_screen = "New arrival"
if "decision_tokens" not in st.session_state:
    st.session_state.decision_tokens = set()
if st.session_state.get("next_screen"):
    st.session_state.active_screen = st.session_state.pop("next_screen")

st.sidebar.title("PatientTriage.ai")
screen = st.sidebar.radio("Screen", PAGES, key="active_screen")
model_failure = st.sidebar.toggle("Demonstrate model failure", help="Safe rules-only fallback; no ML score is displayed.")
surge = st.sidebar.toggle("3x surge display", help="Shows capacity pressure only; safety gates and order never change.")
with st.sidebar.expander("Guided demo", expanded=False):
    st.markdown("""
    <div class="demo-step">1. Select a scenario and evaluate intake.</div>
    <div class="demo-step">2. Review the recommendation and safety gate.</div>
    <div class="demo-step">3. Record a nurse-confirmed disposition.</div>
    <div class="demo-step">4. Explore queue, surge, and audit events.</div>
    """, unsafe_allow_html=True)


def go_to(page: str) -> None:
    """Navigate on a fresh rerun, before Streamlit constructs the sidebar radio widget."""
    st.session_state.next_screen = page
    st.rerun()


def optional_number(label: str, default: float | None, minimum: float = 0.0, maximum: float = 300.0) -> float | None:
    missing_key = f"missing-{label}"
    if missing_key not in st.session_state:
        st.session_state[missing_key] = default is None
    marked = st.checkbox(f"{label} not available", key=missing_key)
    if marked:
        return None
    value_key = f"value-{label}"
    if value_key not in st.session_state:
        st.session_state[value_key] = float(default if default is not None else minimum)
    return st.number_input(label, min_value=float(minimum), max_value=float(maximum), key=value_key)


if screen == "New arrival":
    st.header("1. New arrival")
    st.caption("Use a non-identifying demo scenario, or enter current intake observations manually.")
    scenarios = nurse_facing_scenarios()
    choices = {"Manual entry": None, **{row["label"]: row for row in scenarios}}
    quick_choices = [
        "SIMULATED: zero-history intake",
        "SIMULATED: pediatric rules-only escalation",
        "SIMULATED: ambiguous presentation",
    ]
    quick_columns = st.columns(3)
    for column, option in zip(quick_columns, quick_choices):
        if column.button(option.replace("SIMULATED: ", "Try: "), use_container_width=True):
            st.session_state.pending_scenario = option
            st.rerun()
    if st.session_state.get("pending_scenario"):
        st.session_state.scenario_choice = st.session_state.pop("pending_scenario")
    selected_label = st.selectbox("Demo scenario (expected ESI is hidden from this screen)", list(choices), key="scenario_choice")
    selected = choices[selected_label] or {}
    numeric_fields = [
        ("Age", "age"), ("Systolic BP", "systolic_bp"), ("Diastolic BP", "diastolic_bp"),
        ("Heart rate", "heart_rate"), ("Respiratory rate", "respiratory_rate"),
        ("Temperature C", "temperature_c"), ("SpO2", "spo2"), ("Pain score", "pain_score"),
    ]
    # A scenario selection is a deliberate new intake: reset prior manual checkbox/widget values
    # before constructing form widgets, otherwise Streamlit preserves stale widget state.
    if st.session_state.get("loaded_scenario") != selected_label:
        st.session_state.loaded_scenario = selected_label
        for label, field in numeric_fields:
            value = selected.get(field)
            st.session_state[f"missing-{label}"] = value is None
            if value is not None:
                st.session_state[f"value-{label}"] = float(value)
    with st.form("intake_form"):
        token = st.text_input("Pseudonymous patient token", value=selected.get("id", f"demo-{uuid.uuid4().hex[:8]}"))
        col1, col2, col3 = st.columns(3)
        with col1:
            age = optional_number("Age", selected.get("age"), 0, 120)
            systolic = optional_number("Systolic BP", selected.get("systolic_bp"), 0, 300)
            diastolic = optional_number("Diastolic BP", selected.get("diastolic_bp"), 0, 250)
        with col2:
            heart_rate = optional_number("Heart rate", selected.get("heart_rate"), 0, 300)
            respiratory = optional_number("Respiratory rate", selected.get("respiratory_rate"), 0, 100)
            temperature = optional_number("Temperature C", selected.get("temperature_c"), 0, 50)
        with col3:
            spo2 = optional_number("SpO2", selected.get("spo2"), 0, 100)
            pain = optional_number("Pain score", selected.get("pain_score"), 0, 10)
            sex = st.selectbox("Sex (context only)", ["Not recorded", "Female", "Male", "Intersex / self-described"])
        history = st.checkbox("History available", value=False)
        complaint = st.text_area("Chief complaint (context only; never used for ML or red-flag matching)")
        cues = st.multiselect("Nurse-observed red flags", policy["nurse_selectable_red_flags"], default=selected.get("observable_cues", []))
        submitted = st.form_submit_button("Evaluate intake")
    if submitted:
        payload = IntakePayload(token, age, systolic, diastolic, heart_rate, respiratory, temperature, spo2, pain, sex, complaint, history, tuple(cues))
        assessment = evaluate_intake(payload, model_failure=model_failure)
        st.session_state.intake, st.session_state.assessment = payload, assessment
        store.record_assessment(payload, assessment)
        st.success("Assessment recorded locally.")
    if st.session_state.assessment and st.session_state.intake:
        if st.button("View recommendation →", type="primary"):
            go_to("Recommendation")

elif screen == "Recommendation":
    st.header("2. Recommendation")
    assessment = st.session_state.assessment
    if not assessment:
        st.info("Complete a New arrival assessment first.")
    else:
        st.subheader(assessment.recommendation)
        st.caption(f"Safety gate: {assessment.safety_gate} | rules: {assessment.rule_version} | model: {assessment.model_version}")
        cols = st.columns(4)
        cols[0].metric("Suggested ESI", assessment.suggested_esi if assessment.suggested_esi else "-" )
        cols[1].metric("Calibrated confidence", f"{assessment.confidence:.0%}" if assessment.confidence is not None else "Unavailable")
        cols[2].metric("Urgent risk", f"{assessment.urgent_risk:.0%}" if assessment.urgent_risk is not None else "Unavailable")
        cols[3].metric("Fast-Track", "Eligible" if assessment.fast_track_eligible else "Not eligible")
        if assessment.missing_fields: st.warning("Missing: " + ", ".join(assessment.missing_fields))
        if assessment.implausible_fields: st.warning("Verify: " + ", ".join(assessment.implausible_fields))
        if assessment.red_flags: st.error("Red flags: " + ", ".join(assessment.red_flags))
        if assessment.top_factors: st.write("Top model factors: " + "; ".join(assessment.top_factors))
        st.info("Why not Fast-Track: " + "; ".join(assessment.why_not_fast_track))
        if st.button("Record nurse decision →", type="primary"):
            go_to("Nurse decision")

elif screen == "Nurse decision":
    st.header("3. Nurse decision")
    assessment, intake = st.session_state.assessment, st.session_state.intake
    if not assessment or not intake:
        st.info("Complete an assessment first.")
    else:
        dispositions = ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5", "Clinician review required"]
        default_disposition = f"ESI {assessment.suggested_esi}" if assessment.suggested_esi else "Clinician review required"
        with st.form("decision"):
            accepted = st.radio("Decision", ["Accept recommendation", "Override recommendation"]) == "Accept recommendation"
            disposition = st.selectbox("Final nurse disposition", dispositions, index=dispositions.index(default_disposition))
            reason = st.selectbox("Override reason", ["", "New clinical information", "Clinical judgement", "Patient deterioration", "Data correction", "Other"])
            note = st.text_area("Optional note")
            submit = st.form_submit_button("Record nurse decision")
        if submit:
            if not accepted and not reason:
                st.error("Select a structured override reason.")
            elif intake.patient_token in st.session_state.decision_tokens:
                st.warning("This intake already has a recorded nurse decision. Start a new arrival to record another case.")
            else:
                decision = DecisionPayload(intake.patient_token, disposition, accepted, reason or None, note or None)
                store.record_decision(decision, assessment)
                LocalMockEhrAdapter().write_disposition(intake.patient_token, disposition, {"prototype": True, "audit_required": True})
                esi = int(disposition[-1]) if disposition.startswith("ESI") else None
                st.session_state.queue.append(QueuePatient(intake.patient_token, intake.entered_at, esi, assessment))
                st.session_state.decision_tokens.add(intake.patient_token)
                st.success("Immutable audit event recorded and local mock EHR disposition written.")
        if intake.patient_token in st.session_state.decision_tokens:
            if st.button("Open waiting queue →", type="primary"):
                go_to("Waiting queue")

elif screen == "Waiting queue":
    st.header("4. Waiting queue")
    queue = sorted_queue(st.session_state.queue)
    display = queue * 3 if surge else queue
    st.caption("3x surge increases displayed volume and capacity pressure only; it does not relax thresholds or reorder urgency.")
    if not display:
        st.info("No nurse-confirmed queue entries yet.")
    for sequence, person in enumerate(display, start=1):
        alert = wait_limit_alert(person, policy)
        with st.container(border=True):
            label = f"Surge simulation {sequence}: " if surge else ""
            st.write(f"**{label}{person.patient_token}** - {('ESI ' + str(person.final_esi)) if person.final_esi else 'Awaiting disposition'} - {person.elapsed_minutes()} min")
            if alert: st.warning(alert)
    st.divider()
    st.subheader("Record reassessment vitals")
    token = st.selectbox("Queue patient", [p.patient_token for p in queue], index=None)
    if token:
        person = next(p for p in queue if p.patient_token == token)
        if st.button("Apply simulated deterioration vitals"):
            payload = IntakePayload(token, 64, 98, 60, 142, 30, 37.9, 90, 8)
            latest = evaluate_intake(payload, model_failure=model_failure)
            latest_vitals = VitalsPayload(token, 98, 60, 142, 30, 37.9, 90, 8)
            store.record_vitals(latest_vitals)
            alert = deterioration_alert(person.final_esi, latest)
            if alert:
                person.reassessment_alerts.append(alert)
                st.error(alert)

else:
    st.header("5. Audit & model card")
    st.caption("Local SQLite append-only audit trail. Expected ESI fixture labels are not shown in the nurse workflow.")
    query = st.text_input("Search local audit timeline")
    valid, message = store.verify_chain()
    (st.success if valid else st.error)(message)
    for event in store.timeline(query):
        with st.expander(f"#{event['sequence']} | {event['event_type']} | {event['patient_token']} | {event['created_at']}"):
            st.json(event["payload"])
    st.subheader("Model status")
    st.json(model_metadata())
    st.markdown("See `docs/MODEL_CARD.md` for limitations, intended use, evaluation requirements, privacy, and fairness reporting.")
