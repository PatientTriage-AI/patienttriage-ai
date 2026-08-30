import json
from pathlib import Path
from .domain import IntakePayload

class SafetyPolicy:
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def version(self) -> str:
        return self.config.get("version", "unknown")
        
    def evaluate(self, intake: IntakePayload, model_available: bool) -> dict:
        is_pediatric = intake.age < self.config["thresholds"]["pediatric_age_limit"]
        
        missing_fields = []
        for req in self.config["required_vitals"]:
            if getattr(intake, req) is None:
                missing_fields.append(req)
                
        if intake.pain_score is None:
            missing_fields.append("pain_score")
            
        implausible = []
        for field, (min_val, max_val) in self.config["plausible_ranges"].items():
            val = getattr(intake, field, None)
            if val is not None and (val < min_val or val > max_val):
                implausible.append(f"{field}={val}")
                
        red_flags = intake.red_flags
        
        passed = True
        reason = None
        
        if is_pediatric:
            passed = False
            reason = "Pediatric case: Adult ML model unavailable"
        elif not model_available:
            passed = False
            reason = "ML Model Unavailable: Rules-only fallback active"
        elif red_flags:
            passed = False
            reason = "Red flag(s) present"
        elif missing_fields:
            passed = False
            reason = f"Missing required vitals: {', '.join(missing_fields)}"
        elif implausible:
            passed = False
            reason = f"Implausible vitals: {', '.join(implausible)}"
            
        return {
            "is_pediatric": is_pediatric,
            "passed": passed,
            "missing_fields": missing_fields,
            "implausible_fields": implausible,
            "red_flags": red_flags,
            "reason": reason
        }

    def check_fast_track(self, passed_safety: bool, urgent_risk: float, validation_status: str) -> dict:
        if not passed_safety:
            return {"eligible": False, "reason": "Failed safety gate"}
            
        if validation_status != "eligible for governance review":
            return {"eligible": False, "reason": "Validation safety standard failed (false negatives > 0)"}
            
        if urgent_risk > self.config["thresholds"]["fast_track_max_urgent_risk"]:
            return {"eligible": False, "reason": f"Urgent risk ({urgent_risk:.1%}) exceeds threshold"}
            
        return {"eligible": True, "reason": "Passed all safety checks"}
