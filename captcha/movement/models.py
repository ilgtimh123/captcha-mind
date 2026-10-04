from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    @classmethod
    def from_pair(cls, value) -> "Point":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("point must be a two-item list/tuple")
        x, y = value
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("point coordinates must be numeric")
        try:
            return cls(int(x), int(y))
        except (TypeError, ValueError) as exc:
            raise ValueError("point coordinates must be numeric") from exc

    def as_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class MovementStep:
    kind: str
    position: Optional[Point] = None
    duration_ms: int = 0

    def __post_init__(self):
        if self.kind not in {"move", "press", "release", "click"}:
            raise ValueError(f"unsupported movement step: {self.kind}")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")
        if self.kind in {"move", "click"} and self.position is None:
            raise ValueError(f"{self.kind} requires a position")


@dataclass
class MovementPlan:
    steps: List[MovementStep] = field(default_factory=list)
    source_action: Optional[str] = None

    def append(self, step: MovementStep) -> None:
        self.steps.append(step)

    def as_dict(self):
        result = []
        for step in self.steps:
            item = {"kind": step.kind, "duration_ms": step.duration_ms}
            if step.position is not None:
                item["position"] = [step.position.x, step.position.y]
            result.append(item)
        return {"source_action": self.source_action, "steps": result}
