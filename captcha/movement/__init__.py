"""Deterministic movement planning for CaptchaMind benchmark environments.

This package intentionally contains no OS/browser automation backend. It converts
high-level benchmark actions into validated movement primitives that can be
executed by a test backend or inspected in dry-run mode.
"""

from .adapter import action_to_movement_plan
from .executor import MovementExecutor, RecordingBackend
from .models import MovementPlan, MovementStep, Point
from .planner import MovementPlanner

__all__ = [
    "Point",
    "MovementStep",
    "MovementPlan",
    "MovementPlanner",
    "MovementExecutor",
    "RecordingBackend",
    "action_to_movement_plan",
]
