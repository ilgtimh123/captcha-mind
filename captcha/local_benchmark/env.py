"""CaptchaMind-compatible environment backed by a real local Chromium page."""

import os
import shutil
import tempfile
from typing import Optional

from captcha.data_types import Action, EnvResetResponse, EnvResponse
from captcha.movement.executor import MovementExecutor
from captcha.movement.models import Point

from .browser_backend import LocalOnlyPlaywrightBackend
from .human_motion import LocalHumanMotionPlanner
from .server import LocalBenchmarkServer


class LocalWebBenchmarkEnv:
    """Run synthetic CAPTCHA-style tasks in a real browser on loopback only.

    The class intentionally exposes only screenshots to the agent and accepts
    the same high-level click/drag Action objects used by CaptchaMind.
    """

    def __init__(
        self,
        task_index: int = 0,
        challenge_type: Optional[str] = None,
        headless: bool = True,
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
        self._tmpdir = tempfile.mkdtemp(prefix="captcha_local_benchmark_")
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

    def _screenshot(self) -> str:
        self._shot_index += 1
        path = os.path.join(self._tmpdir, "step_{:03d}.png".format(self._shot_index))
        self.page.screenshot(path=path)
        return path

    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = int(task_index)
        self.actions = []
        self.last_movement_plan = None
        self._shot_index = 0
        self.planner = LocalHumanMotionPlanner(
            seed=self.task_index,
            bounds=(0, 0, 959, 719),
        )
        self.page.goto(
            self.server.url(self.task_index, self.challenge_type),
            wait_until="domcontentloaded",
        )
        self.backend.position = (0, 0)
        return EnvResetResponse(observation=[self._screenshot()])

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)

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
        self.executor.execute(plan)
        state = self.page.evaluate("window.__labGetState()")
        done = bool(state.get("done", False))
        reward = float(state.get("reward", 0.0))
        observation = [] if done else [self._screenshot()]
        return EnvResponse(observation=observation, reward=reward, done=done)

    def motion_trace(self):
        return self.page.evaluate("window.__motionTrace.slice()")

    def benchmark_state(self):
        return self.page.evaluate("window.__labGetState()")

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
