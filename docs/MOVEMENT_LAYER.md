# Benchmark Movement Layer

This fork adds a deterministic movement-planning layer for CaptchaBench and UI-test environments.

## Scope

The layer converts CaptchaMind's existing high-level `click` and `drag` actions into explicit movement primitives. It is designed for reproducible benchmark execution, inspection, replay, and testing.

It intentionally does **not** bundle an operating-system or browser automation backend and does not attempt to imitate human input or evade automation detection.

## Action flow

```text
CaptchaMind Action
       |
       v
action_to_movement_plan()
       |
       v
MovementPlan
       |
       +-- move
       +-- press
       +-- move ... waypoints
       +-- release
       |
       v
MovementExecutor
       |
       v
Injected test backend
```

## Example

```python
from captcha.data_types import Action
from captcha.movement import (
    MovementExecutor,
    MovementPlanner,
    RecordingBackend,
    action_to_movement_plan,
)

planner = MovementPlanner(interpolation_steps=4, move_duration_ms=240)
action = Action(
    name="drag",
    kwargs={"from": [100, 150], "to": [300, 150]},
)

plan = action_to_movement_plan(action, planner)
backend = RecordingBackend()
MovementExecutor(backend).execute(plan)

print(plan.as_dict())
print(backend.events)
```

A drag plan contains a move to the start point, pointer press, deterministic straight-line waypoints, and pointer release at the requested destination.

## Validation

`MovementPlanner` supports optional `(width, height)` bounds. Points outside the configured benchmark surface are rejected. Interpolation is capped at 512 steps and movement duration must be non-negative.

## Slide puzzle integration

`SlidePuzzleEnv.step()` now expands each high-level `drag` into a movement plan and stores it as `env.last_movement_plan`. Reward calculation remains unchanged, preserving the original benchmark metric.

## Tests

Run:

```bash
python -m unittest -v tests.test_movement
```

The GitHub Actions workflow `.github/workflows/movement-tests.yml` runs these tests on pushes to the feature branch and on movement-layer pull requests.
