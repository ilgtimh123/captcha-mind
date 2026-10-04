# Local Browser Benchmark

A self-contained CAPTCHA-style benchmark for testing CaptchaMind visual reasoning
and pointer control in a real Chromium browser.  The benchmark is synthetic,
uses no third-party CAPTCHA service, and binds to `127.0.0.1` only.

## Design references

The structure follows the research-benchmark pattern used by OpenCaptchaWorld:
a local web UI renders interactive visual puzzles, an agent observes the page,
and actions are executed in a real browser.  Browser input uses Playwright's
low-level mouse API (`move`, `down`, `up`) so drag behavior produces actual
browser pointer/mouse events.

The visual design and generated fixtures in this repository are original and
neutral.  They do not copy Google/reCAPTCHA branding, assets, endpoints, or
challenge data.

## Included challenge families

| Type | Goal | Primary capability |
| --- | --- | --- |
| `grid` | Select all 3x3 tiles containing a target symbol | visual selection + clicking |
| `order` | Click three symbols in the requested order | visual grounding + sequencing |
| `slider` | Drag a handle into a generated target zone | visual grounding + motor control |

Each task is deterministic for a given integer seed.  Omitting `type` chooses a
family from the seed, which makes repeated benchmark runs reproducible.

## Naturalistic pointer movement

`LocalHumanMotionPlanner` converts high-level `click` / `drag` actions into a
movement plan with:

- cubic Bezier curvature;
- smooth acceleration/deceleration via a smoothstep timing profile;
- explicit `move -> press -> ... -> release` sequences;
- deterministic seeded variation for repeatable experiments;
- viewport bounds validation.

The purpose is to measure realistic motor-control behavior in the local fixture,
not to evade anti-bot systems.

## Hard localhost boundary

`LocalOnlyPlaywrightBackend` checks `page.url` before every mouse operation and
permits only:

- `localhost`
- `127.0.0.1`
- `::1`

External hosts raise `RuntimeError` before an input event is dispatched.

## Install optional browser dependency

```bash
pip install playwright pydantic
python -m playwright install chromium
```

## Use the benchmark page manually

```python
from captcha.local_benchmark import LocalBenchmarkServer

with LocalBenchmarkServer() as server:
    print(server.url(seed=12, challenge_type="grid"))
    input("Open the printed localhost URL, then press Enter to stop... ")
```

Available explicit `challenge_type` values are `grid`, `order`, and `slider`.

## Use it as a CaptchaMind environment

`LocalWebBenchmarkEnv` implements the same `reset(task_index)` / `step(action)`
contract expected by `ReactAgent`.

```python
from captcha.local_benchmark import LocalWebBenchmarkEnv
from captcha.agents.react_agent import ReactAgent

agent = ReactAgent(
    model="qwen-7b-sft",
    instruction="Solve the visual verification task shown in the browser.",
)

with LocalWebBenchmarkEnv(
    task_index=7,
    challenge_type="slider",
    headless=False,
) as env:
    result = agent.solve(env=env, task_index=7, max_num_steps=12)
    print("reward:", result.reward)
    print("browser state:", env.benchmark_state())
    print("recorded pointer events:", len(env.motion_trace()))
```

The agent receives screenshots only.  Its `click` or `drag` actions are converted
into naturalistic local movement plans and dispatched through Chromium.

## Browser telemetry

The fixture records mouse events in `window.__motionTrace` with:

```text
kind: move | down | up
x, y: viewport coordinates
t: browser performance timestamp
```

`env.motion_trace()` returns the trace for analysis.  `env.benchmark_state()`
returns the current family, seed, completion state, reward, and number of motion
events.

## CI

`.github/workflows/local-browser-benchmark.yml` installs Playwright Chromium on
a GitHub-hosted runner and verifies that a naturalistic plan generates real
`mousemove`, `mousedown`, and `mouseup` events on the local benchmark page.

Run the same test locally with:

```bash
python -m unittest tests.test_local_benchmark -v
```
