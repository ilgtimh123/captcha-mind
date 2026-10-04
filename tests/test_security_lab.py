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
        first = scorer.assess(motion)
        scorer.assess(motion)
        third = scorer.assess(motion)
        self.assertGreaterEqual(third.score, first.score)
        self.assertTrue(any("repeated" in reason for reason in third.reasons))


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
