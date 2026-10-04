from typing import List, Protocol, Tuple

from .models import MovementPlan, MovementStep


class MovementBackend(Protocol):
    def move_to(self, x: int, y: int, duration_ms: int = 0) -> None: ...
    def press(self, x: int, y: int) -> None: ...
    def release(self, x: int, y: int) -> None: ...
    def click(self, x: int, y: int) -> None: ...


class RecordingBackend:
    """In-memory backend for tests and dry-run benchmark execution."""

    def __init__(self) -> None:
        self.events: List[Tuple] = []

    def move_to(self, x: int, y: int, duration_ms: int = 0) -> None:
        self.events.append(("move", x, y, duration_ms))

    def press(self, x: int, y: int) -> None:
        self.events.append(("press", x, y))

    def release(self, x: int, y: int) -> None:
        self.events.append(("release", x, y))

    def click(self, x: int, y: int) -> None:
        self.events.append(("click", x, y))


class MovementExecutor:
    """Execute a MovementPlan through an injected benchmark/test backend.

    No operating-system or browser automation implementation is bundled. Callers
    must explicitly provide a backend, which keeps benchmark planning separate
    from any external UI-control mechanism.
    """

    def __init__(self, backend: MovementBackend) -> None:
        self.backend = backend

    def execute_step(self, step: MovementStep) -> None:
        if step.position is None:
            raise ValueError(f"movement step {step.kind} has no position")
        x, y = step.position.x, step.position.y
        if step.kind == "move":
            self.backend.move_to(x, y, step.duration_ms)
        elif step.kind == "press":
            self.backend.press(x, y)
        elif step.kind == "release":
            self.backend.release(x, y)
        elif step.kind == "click":
            self.backend.click(x, y)
        else:
            raise ValueError(f"unsupported movement step: {step.kind}")

    def execute(self, plan: MovementPlan) -> None:
        for step in plan.steps:
            self.execute_step(step)
