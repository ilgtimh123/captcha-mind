"""Naturalistic pointer trajectories for the local benchmark only.

This module is designed for the bundled loopback-only browser benchmark. It
adds curvature plus an ease-in/ease-out velocity profile so benchmark runs can
exercise realistic pointer event streams instead of teleporting the cursor.
It is not an anti-detection or live-site bypass component.
"""

import math
import random
from typing import Iterable, List, Optional, Tuple

from captcha.movement.models import MovementPlan, MovementStep, Point


Bounds = Tuple[int, int, int, int]


class LocalHumanMotionPlanner:
    """Generate repeatable, naturalistic movement plans for local UI tests."""

    def __init__(
        self,
        seed: int = 0,
        steps: int = 18,
        move_duration_ms: int = 520,
        bounds: Optional[Bounds] = None,
    ) -> None:
        if steps < 4:
            raise ValueError("steps must be >= 4")
        if move_duration_ms < 0:
            raise ValueError("move_duration_ms must be >= 0")
        self.seed = int(seed)
        self.steps = int(steps)
        self.move_duration_ms = int(move_duration_ms)
        self.bounds = bounds

    def _validate_point(self, point: Point) -> Point:
        if self.bounds is None:
            return point
        left, top, right, bottom = self.bounds
        if not (left <= point.x <= right and top <= point.y <= bottom):
            raise ValueError(
                "point ({}, {}) is outside benchmark bounds {}".format(
                    point.x, point.y, self.bounds
                )
            )
        return point

    def _clamp_generated_point(self, point: Point) -> Point:
        if self.bounds is None:
            return point
        left, top, right, bottom = self.bounds
        return Point(
            min(right, max(left, point.x)),
            min(bottom, max(top, point.y)),
        )

    @staticmethod
    def _bezier(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        t: float,
    ) -> Tuple[float, float]:
        omt = 1.0 - t
        x = (
            omt ** 3 * p0[0]
            + 3.0 * omt ** 2 * t * p1[0]
            + 3.0 * omt * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )
        y = (
            omt ** 3 * p0[1]
            + 3.0 * omt ** 2 * t * p1[1]
            + 3.0 * omt * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )
        return x, y

    def trajectory(self, start: Point, end: Point) -> List[Point]:
        """Return a deterministic curved path including start and end."""
        start = self._validate_point(start)
        end = self._validate_point(end)

        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        distance = math.hypot(dx, dy)
        if distance == 0:
            return [start]

        rng = random.Random(
            (self.seed * 1_000_003)
            ^ (start.x * 9_176)
            ^ (start.y * 6_113)
            ^ (end.x * 3_571)
            ^ (end.y * 7_919)
        )

        # A modest perpendicular bend. The bounded magnitude keeps the path
        # plausible for UI testing without adding adversarial evasion logic.
        nx, ny = -dy / distance, dx / distance
        bend = distance * rng.uniform(0.06, 0.14)
        if rng.random() < 0.5:
            bend *= -1.0

        c1 = (
            start.x + dx * rng.uniform(0.24, 0.36) + nx * bend,
            start.y + dy * rng.uniform(0.24, 0.36) + ny * bend,
        )
        c2 = (
            start.x + dx * rng.uniform(0.64, 0.78) + nx * bend * 0.55,
            start.y + dy * rng.uniform(0.64, 0.78) + ny * bend * 0.55,
        )

        result: List[Point] = []
        for index in range(self.steps + 1):
            u = index / float(self.steps)
            # Smoothstep approximates acceleration followed by deceleration.
            t = u * u * (3.0 - 2.0 * u)
            x, y = self._bezier(
                (float(start.x), float(start.y)),
                c1,
                c2,
                (float(end.x), float(end.y)),
                t,
            )
            point = self._clamp_generated_point(
                Point(int(round(x)), int(round(y)))
            )
            if not result or point != result[-1]:
                result.append(point)

        if result[0] != start:
            result.insert(0, start)
        if result[-1] != end:
            result.append(end)
        return result

    def _moves(self, points: Iterable[Point]) -> List[MovementStep]:
        points = list(points)
        if not points:
            return []
        per_step = max(1, self.move_duration_ms // max(1, len(points) - 1))
        return [
            MovementStep(kind="move", position=point, duration_ms=per_step)
            for point in points
        ]

    def click(self, start: Point, target: Point) -> MovementPlan:
        path = self.trajectory(start, target)
        plan = MovementPlan(source_action="click")
        plan.steps.extend(self._moves(path))
        plan.append(MovementStep(kind="press", position=target, duration_ms=38))
        plan.append(MovementStep(kind="release", position=target, duration_ms=42))
        return plan

    def drag(self, start: Point, target: Point) -> MovementPlan:
        path = self.trajectory(start, target)
        plan = MovementPlan(source_action="drag")
        plan.append(MovementStep(kind="move", position=start, duration_ms=0))
        plan.append(MovementStep(kind="press", position=start, duration_ms=45))
        for step in self._moves(path[1:]):
            plan.append(step)
        plan.append(MovementStep(kind="release", position=target, duration_ms=55))
        return plan
