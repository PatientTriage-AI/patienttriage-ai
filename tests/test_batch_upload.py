import pytest
from patienttriage.domain import IntakePayload
from patienttriage.queue import PatientQueue

def test_queue_sorting():
    q = PatientQueue()
    
    # Mock assessments
    class MockAssessment:
        def __init__(self, token, esi):
            self.patient_token = token
            self.suggested_esi = esi
            
    q.add_patient(MockAssessment("PT-1", 4), None)
    q.add_patient(MockAssessment("PT-2", 2), None)
    q.add_patient(MockAssessment("PT-3", 1), None)
    
    sorted_q = q.get_queue()
    
    assert sorted_q[0]["token"] == "PT-3"
    assert sorted_q[1]["token"] == "PT-2"
    assert sorted_q[2]["token"] == "PT-1"
