from datetime import datetime, timezone
import uuid
import dataclasses

from .domain import IntakePayload, TriageAssessment, DecisionPayload, VitalsPayload, ReassessmentEvent
from .policy import SafetyPolicy
from .model import TriageModel
from .audit import AuditTrail

class TriageService:
    def __init__(self, policy: SafetyPolicy, model: TriageModel, audit: AuditTrail):
        self.policy = policy
        self.model = model
        self.audit = audit
        
    def evaluate_intake(self, intake: IntakePayload, patient_token: str = None, simulate_ml_failure: bool = False) -> TriageAssessment:
        if patient_token is None:
            patient_token = f"PT-{uuid.uuid4().hex[:6].upper()}"
            
        model_available = self.model.is_available() and not simulate_ml_failure
        
        safety = self.policy.evaluate(intake, model_available)
        
        suggested_esi = None
        confidence = None
        urgent_risk = None
        probabilities = None
        shap_factors = None
        
        fast_track = {"eligible": False, "reason": safety.get("reason", "Unknown")}
        
        if safety["passed"]:
            try:
                suggested_esi, confidence, urgent_risk, probabilities, shap_factors = self.model.predict(dataclasses.asdict(intake))
                val_status = self.model.metadata.get("fast_track_validation_status", "must remain disabled")
                fast_track = self.policy.check_fast_track(safety["passed"], urgent_risk, val_status)
            except Exception as e:
                safety["passed"] = False
                safety["reason"] = f"ML Runtime Error: {e}"
                fast_track = {"eligible": False, "reason": safety["reason"]}
                
        assessment = TriageAssessment(
            patient_token=patient_token,
            timestamp=datetime.now(timezone.utc),
            is_pediatric=safety["is_pediatric"],
            safety_gate_passed=safety["passed"],
            missing_fields=safety["missing_fields"],
            red_flags=safety["red_flags"],
            suggested_esi=suggested_esi,
            confidence=confidence,
            urgent_risk=urgent_risk,
            probabilities=probabilities,
            shap_factors=shap_factors,
            fast_track_eligible=fast_track["eligible"],
            fast_track_reason=fast_track["reason"],
            clinician_review_reason=safety["reason"] if not safety["passed"] else None,
            model_version=self.model.metadata.get("model_version", "unknown") if self.model.is_available() else "none",
            policy_version=self.policy.version()
        )
        
        self.audit.record_assessment(assessment)
        return assessment
        
    def record_decision(self, decision: DecisionPayload):
        self.audit.record_decision(decision)
        return decision
