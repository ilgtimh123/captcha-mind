#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from captcha.data_types import Action
from captcha.local_benchmark import LocalWebBenchmarkEnv
from captcha.movement.models import MovementPlan, MovementStep, Point
from captcha.security_lab.risk import BotRiskScorer


class RoboticPlanner:
    """Straight, uniform trajectory used as an intentionally mechanical control."""

    def __init__(self, bounds=(0, 0, 959, 719), steps=8, move_duration_ms=160):
        self.bounds = bounds
        self.steps = steps
        self.move_duration_ms = move_duration_ms

    def _check(self, p):
        l, t, r, b = self.bounds
        if not (l <= p.x <= r and t <= p.y <= b):
            raise ValueError(f"point outside bounds: {p}")

    def _points(self, start, target):
        self._check(start); self._check(target)
        out = []
        for i in range(1, self.steps + 1):
            q = i / self.steps
            p = Point(round(start.x + (target.x-start.x)*q), round(start.y + (target.y-start.y)*q))
            if not out or out[-1] != p:
                out.append(p)
        return out

    def click(self, start, target):
        points = self._points(start, target)
        per = max(1, self.move_duration_ms // max(1, len(points)))
        plan = MovementPlan(source_action="click")
        for p in points:
            plan.append(MovementStep("move", p, per))
        plan.append(MovementStep("press", target, 0))
        plan.append(MovementStep("release", target, 0))
        return plan

    def drag(self, start, target):
        points = self._points(start, target)
        per = max(1, self.move_duration_ms // max(1, len(points)))
        plan = MovementPlan(source_action="drag")
        plan.append(MovementStep("move", start, 0))
        plan.append(MovementStep("press", start, 0))
        for p in points:
            plan.append(MovementStep("move", p, per))
        plan.append(MovementStep("release", target, 0))
        return plan


def center(box):
    return [int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)]


def click_verify(env):
    box = env.page.locator("#verify").bounding_box()
    return env.step(Action(name="click", kwargs={"position": center(box)}))


def solve_visible(env, challenge_type):
    if challenge_type == "slider":
        h = env.page.locator(".handle").bounding_box()
        t = env.page.locator(".target").bounding_box()
        env.step(Action(name="drag", kwargs={"from": center(h), "to": center(t)}))
        return click_verify(env)

    if challenge_type == "grid":
        prompt = env.page.locator("#prompt").inner_text()
        target = prompt.split("containing", 1)[1].strip().rstrip(".")
        tiles = env.page.locator(".tile")
        for i in range(tiles.count()):
            item = tiles.nth(i)
            if item.inner_text().strip() == target:
                env.step(Action(name="click", kwargs={"position": center(item.bounding_box())}))
        return click_verify(env)

    if challenge_type == "order":
        prompt = env.page.locator("#prompt").inner_text()
        target = [x.strip() for x in prompt.split(":", 1)[1].split("→")]
        buttons = env.page.locator(".order-btn")
        for symbol in target:
            for i in range(buttons.count()):
                item = buttons.nth(i)
                if item.inner_text().strip() == symbol:
                    env.step(Action(name="click", kwargs={"position": center(item.bounding_box())}))
                    break
        return click_verify(env)

    raise ValueError(challenge_type)


def summarize(rows):
    def avg(name):
        vals = [float(r[name]) for r in rows if r.get(name) is not None]
        return round(statistics.mean(vals), 3) if vals else None
    bands = {b: sum(1 for r in rows if r["risk_band"] == b) for b in ("low", "medium", "high")}
    motion_bands = {b: sum(1 for r in rows if r["motion_band"] == b) for b in ("low", "medium", "high")}
    return {
        "attempts": len(rows),
        "success_rate": round(sum(r["reward"] for r in rows) / max(1, len(rows)), 4),
        "avg_full_risk": avg("risk_score"),
        "avg_motion_only_risk": avg("motion_score"),
        "avg_duration_ms": avg("duration_ms"),
        "avg_straightness": avg("straightness"),
        "avg_inter_event_cv": avg("inter_event_cv"),
        "risk_bands": bands,
        "motion_bands": motion_bands,
    }


def run_profile(profile, challenge_type, seeds, headless=True):
    full_scorer = BotRiskScorer()
    motion_scorer = BotRiskScorer()
    rows = []
    with LocalWebBenchmarkEnv(
        task_index=0,
        challenge_type=challenge_type,
        headless=headless,
        risk_scorer=full_scorer,
    ) as env:
        for idx, seed in enumerate(seeds, 1):
            env.reset(seed)
            if profile == "robotic":
                env.planner = RoboticPlanner()
            started = time.perf_counter()
            response = solve_visible(env, challenge_type)
            wall_ms = (time.perf_counter() - started) * 1000.0
            full = env.last_risk_assessment
            motion = motion_scorer.assess(env.motion_trace(), [])
            feat = full.features
            row = {
                "profile": profile,
                "type": challenge_type,
                "seed": seed,
                "reward": response.reward,
                "risk_score": full.score,
                "risk_band": full.band,
                "motion_score": motion.score,
                "motion_band": motion.band,
                "duration_ms": feat["duration_ms"],
                "wall_ms": round(wall_ms, 3),
                "straightness": feat["straightness"],
                "inter_event_cv": feat["inter_event_cv"],
                "move_count": feat["move_count"],
                "direction_changes": feat["direction_changes"],
                "reasons": full.reasons,
                "motion_reasons": motion.reasons,
            }
            rows.append(row)
            print(f"[{profile:10s} {challenge_type:6s}] {idx:02d}/{len(seeds)} seed={seed:03d} reward={response.reward:.0f} full={full.score:5.1f}/{full.band:6s} motion={motion.score:5.1f}/{motion.band:6s} straight={feat['straightness']:.3f} cv={feat['inter_event_cv']:.3f}", flush=True)
    return rows


def default_output_path():
    artifact_root = os.environ.get("CAPTCHA_LAB_ARTIFACTS")
    if artifact_root:
        return str(Path(artifact_root) / "defense-batch" / "summary.json")
    return "/tmp/captcha-security-batch/summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default=default_output_path())
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    all_rows = []
    for profile in ("robotic", "human_like"):
        for typ in ("slider", "grid", "order"):
            all_rows.extend(run_profile(profile, typ, seeds))
    grouped = {}
    for profile in ("robotic", "human_like"):
        for typ in ("slider", "grid", "order"):
            key = f"{profile}:{typ}"
            grouped[key] = summarize([r for r in all_rows if r["profile"] == profile and r["type"] == typ])
    output = {"groups": grouped, "rows": all_rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(grouped, indent=2, ensure_ascii=False))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
