from datetime import datetime, timezone
from typing import List, Dict, Any

class PatientQueue:
    def __init__(self):
        self._queue = []
        
    def add_patient(self, assessment: Any, intake: Any, final_esi: int = None, disposition: str = "Waiting"):
        item = {
            "token": assessment.patient_token,
            "added_at": datetime.now(timezone.utc),
            "assessment": assessment,
            "intake": intake,
            "final_esi": final_esi or assessment.suggested_esi,
            "disposition": disposition,
            "reassessment_needed": False,
            "reassessment_reason": None
        }
        self._queue.append(item)
        
    def get_queue(self, surge_multiplier: int = 1) -> List[Dict[str, Any]]:
        # Sort by urgency (ESI 1 is highest priority), then by wait time (oldest first)
        def sort_key(x):
            esi = x["final_esi"] if x["final_esi"] is not None else 6 # 6 is unassigned/fallback
            return (esi, x["added_at"])
            
        sorted_q = sorted(self._queue, key=sort_key)
        
        # Surge multiplier simulates volume pressure by displaying more phantom patients or multiplying wait counts
        # We'll just return the real ones, but app.py can display pressure stats.
        return sorted_q
        
    def check_reassessments(self, policy_thresholds: dict):
        now = datetime.now(timezone.utc)
        for item in self._queue:
            if item["disposition"] != "Waiting":
                continue
                
            wait_mins = (now - item["added_at"]).total_seconds() / 60.0
            esi = str(item["final_esi"])
            limit = policy_thresholds.get("reassessment_wait_limit_minutes", {}).get(esi, 120)
            
            if wait_mins > limit and not item["reassessment_needed"]:
                item["reassessment_needed"] = True
                item["reassessment_reason"] = f"Wait limit ({limit}m) exceeded"
                
    def update_vitals(self, token: str, new_vitals: dict) -> dict:
        for item in self._queue:
            if item["token"] == token:
                for k, v in new_vitals.items():
                    if hasattr(item["intake"], k):
                        setattr(item["intake"], k, v)
                return item
        return None
