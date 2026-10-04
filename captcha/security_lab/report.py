"""Summarize JSONL traces produced by the CAPTCHA security lab."""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def summarize_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for event in events:
        grouped[str(event.get("attempt_id", "unknown"))].append(event)

    rows = []
    for attempt_id, items in grouped.items():
        items = sorted(items, key=lambda e: int(e.get("seq", 0)))
        first = items[0] if items else {}
        row = {
            "attempt_id": attempt_id,
            "seed": first.get("seed"),
            "task_type": first.get("task_type"),
            "event_count": len(items),
            "reward": None,
            "risk_score": None,
            "risk_band": None,
            "risk_reasons": [],
            "motion_event_count": None,
            "page_event_count": None,
        }
        for event in items:
            if event.get("kind") == "browser_step_complete":
                row["reward"] = event.get("reward")
                row["motion_event_count"] = event.get("motion_event_count")
                row["page_event_count"] = event.get("page_event_count")
            elif event.get("kind") == "defender_assessment":
                assessment = event.get("assessment") or {}
                row["risk_score"] = assessment.get("score")
                row["risk_band"] = assessment.get("band")
                row["risk_reasons"] = assessment.get("reasons") or []
        rows.append(row)

    rows.sort(key=lambda row: (str(row.get("task_type")), int(row.get("seed") or 0), row["attempt_id"]))
    return rows


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    result = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CAPTCHA security-lab JSONL traces")
    parser.add_argument("path", help="JSONL trace file")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args()
    rows = summarize_events(load_jsonl(args.path))
    print(json.dumps(rows, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
