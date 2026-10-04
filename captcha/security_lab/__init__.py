"""Security-oriented local CAPTCHA benchmark utilities.

This package is intentionally designed for controlled localhost/staging evaluation.
The solver side never consumes detector scores or ground truth.
"""

from .risk import BotRiskScorer, RiskAssessment
from .trace import AttemptTrace, JsonlTraceWriter

__all__ = ["BotRiskScorer", "RiskAssessment", "AttemptTrace", "JsonlTraceWriter"]
