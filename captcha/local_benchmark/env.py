"""CaptchaMind-compatible environment backed by a real local Chromium page."""

import os
import shutil
import tempfile
from typing import Optional

from captcha.data_types import Action, EnvResetResponse, EnvResponse
from captcha.movement.executor import MovementExecutor
from captcha.movement.models import Point
from captcha.security_lab.risk import BotRiskScorer
from captcha.security_lab.trace import AttemptTrace, JsonlTraceWriter, stable_digest

from .browser_backend import LocalOnlyPlaywrightBackend
from .human_motion import LocalHumanMotionPlanner
from .server import LocalBenchmarkServer


_SECURITY_TELEMETRY_JS = r"""
(() => {
  if (window.__securityTrace) return;
  window.__securityTrace = [];
  const push = (kind, extra={}) => window.__securityTrace.push({
    kind,
    t: performance.now(),
    ...extra
  });
  push('environment', {
    webdriver: navigator.webdriver === true,
    plugin_count: navigator.plugins ? navigator.plugins.length : -1,
    language_count: navigator.languages ? navigator.languages.length : -1
  });
  window.addEventListener('focus', () => push('focus'));
  window.addEventListener('blur', () => push('blur'));
  document.addEventListener('visibilitychange', () => push('visibility', {state: document.visibilityState}));
  document.addEventListener('pointerenter', e => push('pointerenter', {x:e.clientX, y:e.clientY}));
  document.addEventListener('pointerleave', e => push('pointerleave', {x:e.clientX, y:e.clientY}));
  document.addEventListener('click', e => push('click', {x:e.clientX, y:e.clientY, button:e.button}));
})();
"""


class LocalWebBenchmarkEnv:
    """Run synthetic CAPTCHA-style tasks in a real browser on loopback only.

    The agent receives screenshots and may submit the same high-level click/drag
    Action objects used elsewhere in CaptchaMind. Defender telemetry is written
    through a separate operator channel and is never included in observations.
    """

    def __init__(
        self,
        task_index: int = 0,
        challenge_type: Optional[str] = None,
        headless: bool = True,
        trace_writer: Optional[JsonlTraceWriter] = None,
        risk_scorer: Optional[BotRiskScorer] = None,
    ) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "playwright is required for LocalWebBenchmarkEnv; install with "
                "`pip install playwright` and `playwright install chromium`"
            ) from exc

        self.task_index = int(task_index)
        self.challenge_type = challenge_type
        self.actions = []
        self.last_movement_plan = None
        self.last_risk_assessment = None
        self.attempt_trace = None
        self.attempt_id = None
        self.trace_writer = trace_writer
        self.risk_scorer = risk_scorer or BotRiskScorer()

        tmp_parent = os.environ.get("CAPTCHA_LAB_TMPDIR")
        if tmp_parent:
            os.makedirs(tmp_parent, exist_ok=True)
        self._tmpdir = tempfile.mkdtemp(
            prefix="captcha_local_benchmark_",
            dir=tmp_parent or None,
        )
        self._shot_index = 0

        self.server = LocalBenchmarkServer().start()
        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page(viewport={"width": 960, "height": 720})
        self.backend = LocalOnlyPlaywrightBackend(self.page)
        self.executor = MovementExecutor(self.backend)
        self.planner = LocalHumanMotionPlanner(
            seed=self.task_index,
            bounds=(0, 0, 959, 719),
        )

    def _emit(self, kind: str, **fields) -> None:
        if self.attempt_trace is None:
            return
        event = self.attempt_trace.add(kind, **fields)
        if self.trace_writer is not None:
            self.trace_writer.write_event(event)

    def _screenshot(self) -> str:
        self._shot_index += 1
        path = os.path.join(self._tmpdir, "step_{:03d}.png".format(self._shot_index))
        self.page.screenshot(path=path)
        return path

    def _install_security_telemetry(self) -> None:
        self.page.evaluate(_SECURITY_TELEMETRY_JS)

    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = int(task_index)
        self.actions = []
        self.last_movement_plan = None
        self.last_risk_assessment = None
        self._shot_index = 0
        self.planner = LocalHumanMotionPlanner(
            seed=self.task_index,
            bounds=(0, 0, 959, 719),
        )
        self.page.goto(
            self.server.url(self.task_index, self.challenge_type),
            wait_until="domcontentloaded",
        )
        self._install_security_telemetry()
        state = self.page.evaluate("window.__labGetState()")
        actual_type = str(state.get("type", self.challenge_type or "auto"))
        self.attempt_trace = AttemptTrace(seed=self.task_index, task_type=actual_type)
        self.attempt_id = self.attempt_trace.attempt_id
        self.backend.position = (0, 0)
        shot = self._screenshot()
        self._emit(
            "challenge_start",
            url_host="127.0.0.1",
            viewport={"width": 960, "height": 720},
            screenshot_name=os.path.basename(shot),
        )
        return EnvResetResponse(observation=[shot])

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        self._emit(
            "action_received",
            action={"name": action.name, "kwargs": action.kwargs},
        )

        if action.name == "click":
            if "position" not in action.kwargs:
                raise ValueError("click action requires position")
            start = Point(*self.backend.position)
            target = Point.from_pair(action.kwargs["position"])
            plan = self.planner.click(start, target)
        elif action.name == "drag":
            if "from" not in action.kwargs or "to" not in action.kwargs:
                raise ValueError("drag action requires from/to")
            start = Point.from_pair(action.kwargs["from"])
            target = Point.from_pair(action.kwargs["to"])
            plan = self.planner.drag(start, target)
        else:
            raise ValueError(
                "local browser benchmark currently supports click and drag actions"
            )

        self.last_movement_plan = plan
        plan_dict = plan.as_dict()
        self._emit(
            "movement_planned",
            source_action=plan.source_action,
            step_count=len(plan.steps),
            plan_digest=stable_digest(plan_dict),
            plan=plan_dict,
        )
        self.executor.execute(plan)
        state = self.page.evaluate("window.__labGetState()")
        done = bool(state.get("done", False))
        reward = float(state.get("reward", 0.0))
        motion = self.motion_trace()
        page_events = self.security_trace()
        self._emit(
            "browser_step_complete",
            done=done,
            reward=reward,
            motion_event_count=len(motion),
            page_event_count=len(page_events),
        )

        if done:
            self.last_risk_assessment = self.risk_scorer.assess(motion, page_events)
            self._emit(
                "defender_assessment",
                assessment=self.last_risk_assessment.as_dict(),
            )
            observation = []
        else:
            observation = [self._screenshot()]
        return EnvResponse(observation=observation, reward=reward, done=done)

    def motion_trace(self):
        return self.page.evaluate("window.__motionTrace.slice()")

    def security_trace(self):
        return self.page.evaluate("(window.__securityTrace || []).slice()")

    def benchmark_state(self):
        return self.page.evaluate("window.__labGetState()")

    def security_report(self):
        """Operator-only report; never included in agent observations."""
        return {
            "attempt_id": self.attempt_id,
            "assessment": None if self.last_risk_assessment is None else self.last_risk_assessment.as_dict(),
            "motion_trace": self.motion_trace(),
            "page_trace": self.security_trace(),
            "events": [] if self.attempt_trace is None else list(self.attempt_trace.events),
        }

    def close(self) -> None:
        try:
            if getattr(self, "browser", None) is not None:
                self.browser.close()
        finally:
            if getattr(self, "_playwright", None) is not None:
                self._playwright.stop()
            if getattr(self, "server", None) is not None:
                self.server.stop()
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> "LocalWebBenchmarkEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
