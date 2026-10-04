import os
import unittest

from captcha.local_benchmark.browser_backend import assert_loopback_url
from captcha.local_benchmark.human_motion import LocalHumanMotionPlanner
from captcha.local_benchmark.server import LocalBenchmarkServer
from captcha.movement.executor import MovementExecutor
from captcha.movement.models import Point


class LocalMotionPlannerTests(unittest.TestCase):
    def test_trajectory_is_repeatable_and_curved(self):
        planner = LocalHumanMotionPlanner(seed=7, steps=12)
        start, end = Point(20, 100), Point(320, 100)
        first = planner.trajectory(start, end)
        second = planner.trajectory(start, end)
        self.assertEqual(first, second)
        self.assertEqual(first[0], start)
        self.assertEqual(first[-1], end)
        self.assertTrue(any(point.y != 100 for point in first[1:-1]))

    def test_loopback_guard_rejects_external_hosts(self):
        assert_loopback_url("http://127.0.0.1:9999/")
        assert_loopback_url("http://localhost:9999/")
        with self.assertRaises(RuntimeError):
            assert_loopback_url("https://example.com/")


class BrowserSmokeTests(unittest.TestCase):
    def test_real_chromium_receives_mouse_trace(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest("playwright not installed: {}".format(exc))

        from captcha.local_benchmark.browser_backend import LocalOnlyPlaywrightBackend

        with LocalBenchmarkServer() as server:
            playwright = sync_playwright().start()
            try:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 960, "height": 720})
                    page.goto(server.url(seed=3, challenge_type="grid"))
                    backend = LocalOnlyPlaywrightBackend(page)
                    planner = LocalHumanMotionPlanner(
                        seed=3,
                        steps=10,
                        move_duration_ms=40,
                        bounds=(0, 0, 959, 719),
                    )
                    plan = planner.click(Point(5, 5), Point(250, 180))
                    MovementExecutor(backend).execute(plan)
                    trace = page.evaluate("window.__motionTrace.slice()")
                    kinds = [event["kind"] for event in trace]
                    self.assertGreaterEqual(len(trace), 5)
                    self.assertIn("move", kinds)
                    self.assertIn("down", kinds)
                    self.assertIn("up", kinds)
                    state = page.evaluate("window.__labGetState()")
                    self.assertEqual(state["type"], "grid")
                    self.assertFalse(state["done"])
                finally:
                    browser.close()
            finally:
                playwright.stop()


if __name__ == "__main__":
    unittest.main()
