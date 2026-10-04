# CAPTCHA Security Lab

This directory documents the defensive test harness built around the repository's
localhost-only visual verification benchmark.  The goal is to measure how well a
CAPTCHA/anti-automation design resists a capable multimodal agent **without**
connecting the solver to a production website.

## Architecture

```text
CaptchaMind / test agent
        |
        | screenshots + click/drag only
        v
LocalWebBenchmarkEnv -----------------------------+
        |                                         |
        | Chromium mouse events                   | operator channel only
        v                                         v
Synthetic challenge                         security_lab JSONL
(grid / order / slider)                     + defender features
        |                                         |
        +---------------- reward -----------------+
                                                  |
                                                  v
                                         BotRiskScorer/report
```

The solver never receives hidden ground truth, detector features, risk score, or
operator logs.  This separation matters: otherwise the benchmark becomes an
optimization oracle for evading the detector rather than a useful defensive test.

## Logs produced

`JsonlTraceWriter` creates append-only JSONL.  Important event families are:

- `challenge_start`: attempt id, seed, task family, viewport
- `action_received`: high-level model action (`click` / `drag`)
- `movement_planned`: generated movement plan, step count, digest
- `browser_step_complete`: reward/done and browser event counts
- `defender_assessment`: final risk band, score, reasons and derived features

The existing `ReactAgent` already prints raw model responses, parse diagnostics and
token usage.  `run_instrumented_attempt()` captures that stdout/stderr verbatim to a
separate text file while keeping the solver algorithm unchanged.

## Browser telemetry

The local benchmark records two independent streams:

1. motion trace: `mousemove`, `mousedown`, `mouseup`
2. page trace: focus/blur, visibility state, pointer enter/leave, click

Derived defender features currently include:

- event counts and down/up balance
- total interaction duration
- path length and displacement
- path straightness
- mean/peak speed
- inter-event timing mean and coefficient of variation
- long gaps
- direction changes
- focus/visibility changes

`BotRiskScorer` is deliberately an explainable baseline.  Treat it as a telemetry
validation tool, not a production bot classifier.  A production deployment should
train/evaluate detection on labelled human and automation traffic and calibrate
false-positive rates before enforcement.

## Running a defensive attempt

```python
from captcha.agents.react_agent import ReactAgent
from captcha.local_benchmark import LocalWebBenchmarkEnv
from captcha.security_lab.harness import run_instrumented_attempt
from captcha.security_lab.trace import JsonlTraceWriter

writer = JsonlTraceWriter("artifacts/security_attempts.jsonl")
agent = ReactAgent(
    model="qwen-7b-sft",
    instruction="Solve the visual verification task.",
)

with LocalWebBenchmarkEnv(
    task_index=12,
    challenge_type="slider",
    headless=True,
    trace_writer=writer,
) as env:
    result = run_instrumented_attempt(
        agent,
        env,
        task_index=12,
        debug_text_path="artifacts/solver_12.log",
    )
    print(result.reward)
    print(result.security_report["assessment"])
```

Summarize JSONL:

```bash
python -m captcha.security_lab.report artifacts/security_attempts.jsonl --pretty
```

## Recommended defensive experiment matrix

For each challenge family, evaluate many deterministic seeds and record at least:

- solve rate
- median/p95 solve time
- actions per attempt
- model/token latency
- detector risk distribution
- false positives on manually collected human runs
- repeated-signature frequency
- challenge abandonment/failure rate

Do not tune a challenge only against one solver.  Rotate model families and compare
humans against automated agents under the same browser/viewport conditions.

## Staging integration pattern

If this lab is later used to evaluate a real service, keep two planes separate:

1. **production/staging application**: issues challenge id and validates result
2. **security lab runner**: receives a mirrored/synthetic challenge in an isolated
   test environment and produces measurement data

Do not point the bundled automated browser backend at a production domain.  The
backend is intentionally hard-coded to loopback hosts.

For an application deployment, log server-side fields such as challenge id/version,
issue time, submit time, result, validation outcome, request-rate bucket and privacy-
preserving client/network identifiers.  Keep secrets, cookies, authorization headers
and raw credentials out of logs.

## Defensive iteration loop

```text
collect labelled attempts
        -> inspect traces
        -> measure human-vs-agent separation
        -> adjust challenge/risk model
        -> replay fixed seed suite
        -> regression test false positives
        -> promote only after staging validation
```

A CAPTCHA should be one layer, not the only control.  Combine it with server-side
rate limiting, request/session reputation, replay prevention, challenge expiry and
strong validation of any challenge token/result.
