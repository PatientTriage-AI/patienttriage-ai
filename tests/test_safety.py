import pytest
from pathlib import Path
from patienttriage.policy import SafetyPolicy
from patienttriage.domain import IntakePayload

@pytest.fixture
def policy():
    base_dir = Path(__file__).parent.parent
    return SafetyPolicy(base_dir / "config" / "policy.v1.json")

@pytest.fixture
def base_intake():
    return IntakePayload(
        age=45, sex="M", systolic_bp=120, diastolic_bp=80,
        heart_rate=80, respiratory_rate=16, temperature=37.0,
        spo2=98, pain_score=0, history_available=True,
        complaint="Pain", observable_cues=[], red_flags=[]
    )

def test_pediatric_rejected(policy, base_intake):
    base_intake.age = 12
    result = policy.evaluate(base_intake, True)
    assert not result["passed"]
    assert result["is_pediatric"]
    assert "Pediatric case" in result["reason"]

def test_red_flag_rejected(policy, base_intake):
    base_intake.red_flags = ["Cardiac arrest"]
    result = policy.evaluate(base_intake, True)
    assert not result["passed"]
    assert "Red flag" in result["reason"]

def test_missing_vitals_rejected(policy, base_intake):
    base_intake.heart_rate = None
    result = policy.evaluate(base_intake, True)
    assert not result["passed"]
    assert "Missing required vitals" in result["reason"]

def test_implausible_vitals_rejected(policy, base_intake):
    base_intake.heart_rate = 300
    result = policy.evaluate(base_intake, True)
    assert not result["passed"]
    assert "Implausible vitals" in result["reason"]

def test_ml_failure_rejected(policy, base_intake):
    result = policy.evaluate(base_intake, False)
    assert not result["passed"]
    assert "ML Model Unavailable" in result["reason"]
