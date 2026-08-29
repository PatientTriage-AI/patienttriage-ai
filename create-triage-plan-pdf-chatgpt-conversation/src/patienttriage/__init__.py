"""Safety-first clinical decision-support prototype. Not for production use."""

from .domain import IntakePayload, TriageAssessment
from .service import evaluate_intake

__all__ = ["IntakePayload", "TriageAssessment", "evaluate_intake"]
