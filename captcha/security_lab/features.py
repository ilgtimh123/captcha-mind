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
    stroke_count: int = 0
    mean_stroke_points: float = 0.0
    mean_stroke_straightness: float = 0.0
    near_linear_stroke_fraction: float = 0.0
    mean_stroke_turn_rad: float = 0.0
    mean_stroke_step_cv: float = 0.0
    focus_loss_count: int = 0
    visibility_hidden_count: int = 0
    automation_webdriver: bool = False
    plugin_count: int = -1
    language_count: int = -1
    move_centroid_x: float = 0.0
    move_centroid_y: float = 0.0

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
    environment = next((e for e in reversed(page) if e.get("kind") == "environment"), None)
    if environment is not None:
        result.automation_webdriver = bool(environment.get("webdriver", False))
        try:
            result.plugin_count = int(environment.get("plugin_count", -1))
        except (TypeError, ValueError):
            result.plugin_count = -1
        try:
            result.language_count = int(environment.get("language_count", -1))
        except (TypeError, ValueError):
            result.language_count = -1

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
    if points:
        result.move_centroid_x = sum(p[0] for p in points) / len(points)
        result.move_centroid_y = sum(p[1] for p in points) / len(points)
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

    # Analyze each contiguous mouse-move stroke separately. Whole-attempt
    # straightness can hide mechanical paths when a task contains several
    # clicks in different directions; per-stroke geometry preserves that
    # defensive signal without exposing any challenge ground truth.
    strokes: List[List[Dict[str, Any]]] = []
    current_stroke: List[Dict[str, Any]] = []
    boundary_kinds = {"down", "mousedown", "up", "mouseup", "click"}
    for event in events:
        if event.get("kind") == "move" and _point(event) is not None:
            current_stroke.append(event)
            continue
        if event.get("kind") in boundary_kinds:
            if len(current_stroke) >= 2:
                strokes.append(current_stroke)
            current_stroke = []
    if len(current_stroke) >= 2:
        strokes.append(current_stroke)

    stroke_straightness: List[float] = []
    stroke_points: List[int] = []
    stroke_turns: List[float] = []
    stroke_step_cvs: List[float] = []
    for stroke in strokes:
        pts = [_point(event) for event in stroke]
        pts = [point for point in pts if point is not None]
        if len(pts) < 2:
            continue
        stroke_points.append(len(pts))
        lengths: List[float] = []
        headings: List[float] = []
        for first, second in zip(pts, pts[1:]):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length = math.hypot(dx, dy)
            if length > 0:
                lengths.append(length)
                headings.append(math.atan2(dy, dx))
        path = sum(lengths)
        displacement = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        straightness = displacement / path if path > 0 else 0.0
        stroke_straightness.append(straightness)

        turn_values: List[float] = []
        for a, b in zip(headings, headings[1:]):
            turn_values.append(abs((b - a + math.pi) % (2.0 * math.pi) - math.pi))
        stroke_turns.append(sum(turn_values) / len(turn_values) if turn_values else 0.0)

        if lengths:
            mean_length = sum(lengths) / len(lengths)
            if mean_length > 0:
                variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
                stroke_step_cvs.append(math.sqrt(variance) / mean_length)
            else:
                stroke_step_cvs.append(0.0)

    result.stroke_count = len(stroke_straightness)
    if stroke_straightness:
        result.mean_stroke_straightness = sum(stroke_straightness) / len(stroke_straightness)
        result.near_linear_stroke_fraction = (
            sum(1 for value in stroke_straightness if value >= 0.995)
            / len(stroke_straightness)
        )
    if stroke_points:
        result.mean_stroke_points = sum(stroke_points) / len(stroke_points)
    if stroke_turns:
        result.mean_stroke_turn_rad = sum(stroke_turns) / len(stroke_turns)
    if stroke_step_cvs:
        result.mean_stroke_step_cv = sum(stroke_step_cvs) / len(stroke_step_cvs)

    return result
