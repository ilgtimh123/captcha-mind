import unittest

from captcha.data_types import Action
from captcha.movement import (
    MovementExecutor,
    MovementPlanner,
    Point,
    RecordingBackend,
    action_to_movement_plan,
)


class MovementPlannerTests(unittest.TestCase):
    def test_drag_plan_has_press_waypoints_and_release(self):
        planner = MovementPlanner(interpolation_steps=4, move_duration_ms=400)
        plan = planner.drag(Point(10, 20), Point(50, 60))

        self.assertEqual(plan.source_action, "drag")
        self.assertEqual(plan.steps[0].kind, "move")
        self.assertEqual(plan.steps[0].position, Point(10, 20))
        self.assertEqual(plan.steps[1].kind, "press")
        self.assertEqual(plan.steps[-1].kind, "release")
        self.assertEqual(plan.steps[-1].position, Point(50, 60))
        self.assertEqual(
            [step.position for step in plan.steps if step.kind == "move"][1:],
            [Point(20, 30), Point(30, 40), Point(40, 50), Point(50, 60)],
        )

    def test_action_adapter_supports_click_and_drag(self):
        planner = MovementPlanner(interpolation_steps=2)

        click = action_to_movement_plan(
            Action(name="click", kwargs={"position": [7, 9]}), planner
        )
        self.assertEqual(click.steps[0].kind, "click")
        self.assertEqual(click.steps[0].position, Point(7, 9))

        drag = action_to_movement_plan(
            Action(name="drag", kwargs={"from": [0, 0], "to": [8, 4]}), planner
        )
        self.assertEqual(drag.steps[-1].position, Point(8, 4))

    def test_bounds_reject_out_of_range_target(self):
        planner = MovementPlanner(bounds=(100, 80))
        with self.assertRaises(ValueError):
            planner.drag(Point(10, 10), Point(100, 20))

    def test_recording_executor_preserves_event_order(self):
        planner = MovementPlanner(interpolation_steps=2, move_duration_ms=100)
        plan = planner.drag(Point(2, 3), Point(10, 7))
        backend = RecordingBackend()

        MovementExecutor(backend).execute(plan)

        self.assertEqual(backend.events[0], ("move", 2, 3, 0))
        self.assertEqual(backend.events[1], ("press", 2, 3))
        self.assertEqual(backend.events[-1], ("release", 10, 7))
        self.assertEqual(
            [event[0] for event in backend.events],
            ["move", "press", "move", "move", "release"],
        )

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(ValueError):
            action_to_movement_plan(Action(name="enter_number", kwargs={"number": 4}))


if __name__ == "__main__":
    unittest.main()
