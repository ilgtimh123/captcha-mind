"""Defender-side feature extraction for pointer and page telemetry."""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class MotionFeatures:
    event_count: int = 0
    move_count: int = 0
    down_count: int = 0
    up_count: int = 0
    click_count: int = 0
    duration_ms: float = 0.0
    path_length_px: float = 0.0
    displacement_px: float = 0.0
    straightness: float = 0.0
    mean_speed_px_s: float = 0.0
    peak_speed_px_s: float = 0.0
    inter_event_mean_ms: float = 0.0
    inter_event_cv: float = 0.0
    long_gap_count: int = 0
    direction_changes: int = 0
    focus_loss_count: int = 0
    visibility_hidden_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _point(event: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        return float(event["x"]), float(event["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _time(event: Dict[str, Any]) -> Optional[float]:
    for key in ("t", "ts_ms"):
        try:
            return float(event[key])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def extract_motion_features(
    motion_events: Iterable[Dict[str, Any]],
    page_events: Optional[Iterable[Dict[str, Any]]] = None,
) -> MotionFeatures:
    events: List[Dict[str, Any]] = list(motion_events)
    page: List[Dict[str, Any]] = list(page_events or [])
    result = MotionFeatures()
    result.event_count = len(events)
    result.move_count = sum(1 for e in events if e.get("kind") == "move")
    result.down_count = sum(1 for e in events if e.get("kind") in {"down", "mousedown"})
    result.up_count = sum(1 for e in events if e.get("kind") in {"up", "mouseup"})
    result.click_count = sum(1 for e in events if e.get("kind") == "click")
    result.focus_loss_count = sum(1 for e in page if e.get("kind") == "blur")
    result.visibility_hidden_count = sum(
        1 for e in page if e.get("kind") == "visibility" and e.get("state") == "hidden"
    )

    timed = [(e, _time(e)) for e in events]
    timed = [(e, t) for e, t in timed if t is not None]
    if len(timed) >= 2:
        times = [t for _, t in timed]
        result.duration_ms = max(0.0, times[-1] - times[0])
        gaps = [max(0.0, b - a) for a, b in zip(times, times[1:])]
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            result.inter_event_mean_ms = mean_gap
            if mean_gap > 0:
                variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                result.inter_event_cv = math.sqrt(variance) / mean_gap
            result.long_gap_count = sum(1 for g in gaps if g >= 750.0)

    move_events = [e for e in events if e.get("kind") == "move" and _point(e) is not None]
    points = [_point(e) for e in move_events]
    points = [p for p in points if p is not None]
    speeds: List[float] = []
    vectors: List[Tuple[float, float]] = []
    if len(points) >= 2:
        for index in range(1, len(points)):
            x0, y0 = points[index - 1]
            x1, y1 = points[index]
            dx, dy = x1 - x0, y1 - y0
            distance = math.hypot(dx, dy)
            result.path_length_px += distance
            vectors.append((dx, dy))
            t0 = _time(move_events[index - 1])
            t1 = _time(move_events[index])
            if t0 is not None and t1 is not None and t1 > t0:
                speeds.append(distance / ((t1 - t0) / 1000.0))
        result.displacement_px = math.hypot(
            points[-1][0] - points[0][0], points[-1][1] - points[0][1]
        )
        if result.path_length_px > 0:
            result.straightness = result.displacement_px / result.path_length_px

    if speeds:
        result.mean_speed_px_s = sum(speeds) / len(speeds)
        result.peak_speed_px_s = max(speeds)

    for first, second in zip(vectors, vectors[1:]):
        a = math.atan2(first[1], first[0])
        b = math.atan2(second[1], second[0])
        delta = abs((b - a + math.pi) % (2.0 * math.pi) - math.pi)
        if delta > 0.35:
            result.direction_changes += 1

    return result
