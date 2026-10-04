"""Structured attempt logging for the local CAPTCHA security lab."""

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return repr(value)


def stable_digest(value: Any) -> str:
    raw = json.dumps(_safe_json(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class AttemptTrace:
    seed: int
    task_type: str
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at_ms: int = field(default_factory=_utc_ms)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, **fields: Any) -> Dict[str, Any]:
        event = {
            "attempt_id": self.attempt_id,
            "seq": len(self.events),
            "ts_ms": _utc_ms(),
            "kind": str(kind),
            "seed": int(self.seed),
            "task_type": str(self.task_type),
        }
        event.update({str(k): _safe_json(v) for k, v in fields.items()})
        self.events.append(event)
        return event

    def snapshot(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "seed": self.seed,
            "task_type": self.task_type,
            "started_at_ms": self.started_at_ms,
            "events": list(self.events),
        }


class JsonlTraceWriter:
    """Append-only JSONL writer.

    One event is written per line so a partially interrupted run remains easy to
    inspect.  The class never stores credentials, cookies, browser storage, or
    HTTP authorization material; callers should pass only benchmark metadata.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    def write_event(self, event: Dict[str, Any]) -> None:
        line = json.dumps(_safe_json(event), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_trace(self, trace: AttemptTrace, start_index: int = 0) -> int:
        events = trace.events[start_index:]
        for event in events:
            self.write_event(event)
        return start_index + len(events)

    def read_events(self, attempt_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        result = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if attempt_id is None or event.get("attempt_id") == attempt_id:
                    result.append(event)
        return result
