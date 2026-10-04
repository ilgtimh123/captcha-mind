from typing import Optional

from captcha.data_types import Action

from .models import MovementPlan, Point
from .planner import MovementPlanner


def action_to_movement_plan(
    action: Action,
    planner: Optional[MovementPlanner] = None,
) -> MovementPlan:
    """Convert CaptchaMind high-level actions into benchmark movement plans."""

    planner = planner or MovementPlanner()

    if action.name == "click":
        if "position" not in action.kwargs:
            raise ValueError("click action requires 'position'")
        return planner.click(Point.from_pair(action.kwargs["position"]))

    if action.name == "drag":
        if "from" not in action.kwargs or "to" not in action.kwargs:
            raise ValueError("drag action requires 'from' and 'to'")
        return planner.drag(
            Point.from_pair(action.kwargs["from"]),
            Point.from_pair(action.kwargs["to"]),
        )

    raise ValueError(f"action {action.name!r} has no movement representation")
