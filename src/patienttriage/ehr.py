class EhrAdapter:
    def __init__(self):
        self.records = []
        
    def write_disposition(self, patient_token: str, esi: int, notes: str = ""):
        self.records.append({
            "patient_token": patient_token,
            "esi": esi,
            "notes": notes,
            "status": "SENT_TO_EHR"
        })
        return True
