import os
import tempfile
import unittest

from captcha.data_types import Action
from captcha.local_benchmark import LocalWebBenchmarkEnv
from captcha.security_lab.features import extract_motion_features
from captcha.security_lab.risk import BotRiskScorer
from captcha.security_lab.trace import AttemptTrace, JsonlTraceWriter


class SecurityLabUnitTests(unittest.TestCase):
    def test_trace_writer_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            writer = JsonlTraceWriter(path)
            trace = AttemptTrace(seed=4, task_type="grid")
            writer.write_event(trace.add("challenge_start", value=1))
            writer.write_event(trace.add("browser_step_complete", reward=1.0))
            events = writer.read_events(trace.attempt_id)
            self.assertEqual([e["kind"] for e in events], ["challenge_start", "browser_step_complete"])

    def test_feature_extraction(self):
        motion = [
            {"kind": "move", "x": 0, "y": 0, "t": 0.0},
            {"kind": "move", "x": 10, "y": 0, "t": 25.0},
            {"kind": "down", "x": 10, "y": 0, "t": 30.0},
            {"kind": "move", "x": 20, "y": 5, "t": 60.0},
            {"kind": "up", "x": 20, "y": 5, "t": 90.0},
        ]
        page = [{"kind": "focus", "t": 0.0}, {"kind": "blur", "t": 91.0}]
        features = extract_motion_features(motion, page)
        self.assertEqual(features.event_count, 5)
        self.assertEqual(features.down_count, 1)
        self.assertEqual(features.up_count, 1)
        self.assertGreater(features.path_length_px, 20.0)
        self.assertEqual(features.focus_loss_count, 1)

    def test_repeat_signature_increases_operator_risk(self):
        scorer = BotRiskScorer()
        motion = [
            {"kind": "move", "x": i * 10, "y": 10, "t": i * 40.0}
            for i in range(10)
        ] + [
            {"kind": "down", "x": 90, "y": 10, "t": 405.0},
            {"kind": "up", "x": 90, "y": 10, "t": 445.0},
        ]
        jittered = [dict(event) for event in motion]
        for index, event in enumerate(jittered):
            event["t"] = float(event["t"]) + (index % 3) * 11.0
        first = scorer.assess(motion)
        second = scorer.assess(jittered)
        third = scorer.assess(motion)
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(second.signature, third.signature)
        self.assertGreaterEqual(third.score, first.score)
        self.assertTrue(any("repeated" in reason for reason in third.reasons))

    def test_per_stroke_features_distinguish_mechanical_paths(self):
        robotic = []
        t = 0.0
        for y in (20, 120):
            for i in range(8):
                robotic.append({"kind": "move", "x": 20 + i * 20, "y": y, "t": t})
                t += 20.0
            robotic.append({"kind": "down", "x": 160, "y": y, "t": t}); t += 40.0
            robotic.append({"kind": "up", "x": 160, "y": y, "t": t}); t += 20.0

        curved = []
        t = 0.0
        for offset in (0, 100):
            for i in range(18):
                x = 20 + i * 8
                y = 30 + offset + int(50 * ((i / 17.0) - 0.5) ** 2)
                curved.append({"kind": "move", "x": x, "y": y, "t": t})
                t += 28.0 + (i % 4) * 3.0
            curved.append({"kind": "down", "x": 156, "y": 42 + offset, "t": t}); t += 40.0
            curved.append({"kind": "up", "x": 156, "y": 42 + offset, "t": t}); t += 20.0

        robot_features = extract_motion_features(robotic)
        curved_features = extract_motion_features(curved)
        self.assertEqual(robot_features.stroke_count, 2)
        self.assertGreaterEqual(robot_features.near_linear_stroke_fraction, 0.99)
        self.assertLessEqual(robot_features.mean_stroke_points, 8.0)
        self.assertGreater(curved_features.mean_stroke_points, robot_features.mean_stroke_points)
        self.assertLess(curved_features.near_linear_stroke_fraction, robot_features.near_linear_stroke_fraction)

        robot_risk = BotRiskScorer().assess(robotic)
        curved_risk = BotRiskScorer().assess(curved)
        self.assertGreater(robot_risk.score, curved_risk.score)

    def test_webdriver_environment_signal_is_defender_visible(self):
        scorer = BotRiskScorer()
        motion = [
            {"kind": "move", "x": i * 8, "y": 20 + i, "t": i * 45.0}
            for i in range(10)
        ] + [
            {"kind": "down", "x": 72, "y": 29, "t": 420.0},
            {"kind": "up", "x": 72, "y": 29, "t": 470.0},
        ]
        assessment = scorer.assess(
            motion,
            [{"kind": "environment", "webdriver": True, "plugin_count": 5, "language_count": 2}],
        )
        self.assertGreaterEqual(assessment.score, 35.0)
        self.assertTrue(any("WebDriver" in reason for reason in assessment.reasons))


class SecurityLabBrowserTests(unittest.TestCase):
    def test_local_env_produces_defender_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = JsonlTraceWriter(os.path.join(tmp, "attempts.jsonl"))
            with LocalWebBenchmarkEnv(
                task_index=9,
                challenge_type="slider",
                headless=True,
                trace_writer=writer,
            ) as env:
                env.reset(9)
                handle = env.page.locator(".handle").bounding_box()
                target = env.page.locator(".target").bounding_box()
                verify = env.page.locator("#verify").bounding_box()
                self.assertIsNotNone(handle)
                self.assertIsNotNone(target)
                self.assertIsNotNone(verify)

                start = [int(handle["x"] + handle["width"] / 2), int(handle["y"] + handle["height"] / 2)]
                end = [int(target["x"] + target["width"] / 2), int(target["y"] + target["height"] / 2)]
                env.step(Action(name="drag", kwargs={"from": start, "to": end}))
                verify_point = [int(verify["x"] + verify["width"] / 2), int(verify["y"] + verify["height"] / 2)]
                response = env.step(Action(name="click", kwargs={"position": verify_point}))

                self.assertTrue(response.done)
                self.assertEqual(response.reward, 1.0)
                report = env.security_report()
                self.assertIsNotNone(report["assessment"])
                self.assertGreater(len(report["motion_trace"]), 5)
                self.assertGreater(len(report["events"]), 3)
                persisted = writer.read_events(env.attempt_id)
                kinds = [e["kind"] for e in persisted]
                self.assertIn("defender_assessment", kinds)


if __name__ == "__main__":
    unittest.main()
