from typing import Optional, Tuple

from .models import MovementPlan, MovementStep, Point


class MovementPlanner:
    """Build deterministic straight-line movement plans for benchmark actions.

    The planner is intentionally predictable: it does not emulate human input or
    attempt to avoid bot/automation detection. Its purpose is reproducible UI
    testing and synthetic benchmark execution.
    """

    def __init__(
        self,
        interpolation_steps: int = 8,
        move_duration_ms: int = 320,
        bounds: Optional[Tuple[int, int]] = None,
    ) -> None:
        if interpolation_steps < 1 or interpolation_steps > 512:
            raise ValueError("interpolation_steps must be between 1 and 512")
        if move_duration_ms < 0:
            raise ValueError("move_duration_ms must be >= 0")
        if bounds is not None:
            width, height = bounds
            if width <= 0 or height <= 0:
                raise ValueError("bounds must contain positive width and height")
        self.interpolation_steps = interpolation_steps
        self.move_duration_ms = move_duration_ms
        self.bounds = bounds

    def validate_point(self, point: Point) -> Point:
        if self.bounds is not None:
            width, height = self.bounds
            if not (0 <= point.x < width and 0 <= point.y < height):
                raise ValueError(
                    f"point {(point.x, point.y)} is outside bounds {(width, height)}"
                )
        return point

    def interpolate(self, start: Point, end: Point):
        self.validate_point(start)
        self.validate_point(end)
        points = []
        for index in range(1, self.interpolation_steps + 1):
            ratio = index / self.interpolation_steps
            x = round(start.x + (end.x - start.x) * ratio)
            y = round(start.y + (end.y - start.y) * ratio)
            point = Point(x, y)
            if not points or points[-1] != point:
                points.append(point)
        return points

    def move(self, start: Point, end: Point) -> MovementPlan:
        points = self.interpolate(start, end)
        duration = self.move_duration_ms // max(1, len(points))
        return MovementPlan(
            source_action="move",
            steps=[MovementStep("move", point, duration) for point in points],
        )

    def click(self, position: Point) -> MovementPlan:
        self.validate_point(position)
        return MovementPlan(
            source_action="click",
            steps=[MovementStep("click", position, 0)],
        )

    def drag(self, start: Point, end: Point) -> MovementPlan:
        points = self.interpolate(start, end)
        duration = self.move_duration_ms // max(1, len(points))
        steps = [
            MovementStep("move", start, 0),
            MovementStep("press", start, 0),
        ]
        steps.extend(MovementStep("move", point, duration) for point in points)
        steps.append(MovementStep("release", end, 0))
        return MovementPlan(source_action="drag", steps=steps)
