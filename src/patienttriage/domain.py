from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class IntakePayload:
    age: int
    sex: str
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    heart_rate: Optional[float]
    respiratory_rate: Optional[float]
    temperature: Optional[float]
    spo2: Optional[float]
    pain_score: Optional[float]
    history_available: bool
    complaint: str
    observable_cues: List[str]
    red_flags: List[str]

@dataclass
class TriageAssessment:
    patient_token: str
    timestamp: datetime
    is_pediatric: bool
    safety_gate_passed: bool
    missing_fields: List[str]
    red_flags: List[str]
    suggested_esi: Optional[int]
    confidence: Optional[float]
    urgent_risk: Optional[float]
    probabilities: Optional[Dict[int, float]]
    shap_factors: Optional[List[Dict[str, Any]]]
    fast_track_eligible: bool
    fast_track_reason: str
    clinician_review_reason: Optional[str]
    model_version: str
    policy_version: str
    
@dataclass
class DecisionPayload:
    patient_token: str
    nurse_accepted: bool
    override_esi: Optional[int]
    override_reason: Optional[str]
    override_text: Optional[str]

@dataclass
class VitalsPayload:
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    heart_rate: Optional[float]
    respiratory_rate: Optional[float]
    temperature: Optional[float]
    spo2: Optional[float]
    pain_score: Optional[float]
    red_flags: List[str]
@dataclass
class ReassessmentEvent:
    patient_token: str
    timestamp: datetime
    trigger: str
    old_esi: Optional[int]
    new_esi: Optional[int]
    vitals_updated: bool
