"""Operator harness that captures existing agent debug output without altering solving logic."""

import contextlib
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LabRunResult:
    reward: float
    security_report: Dict[str, Any]
    debug_text_path: Optional[str]


def run_instrumented_attempt(
    agent,
    env,
    task_index: int,
    max_num_steps: int = 30,
    debug_text_path: Optional[str] = None,
) -> LabRunResult:
    """Run one local benchmark attempt and persist the agent's normal debug text.

    CaptchaMind's current ReactAgent already prints raw model responses, parsing
    diagnostics and token usage.  This helper captures those streams verbatim so
    operators can correlate them with the environment JSONL trace.  It does not
    parse defender scores or feed any security signal back into the agent.
    """

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = agent.solve(env, int(task_index), max_num_steps=int(max_num_steps))

    path = None
    if debug_text_path:
        path = os.path.abspath(debug_text_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(buffer.getvalue())

    report = env.security_report() if hasattr(env, "security_report") else {}
    return LabRunResult(
        reward=float(result.reward),
        security_report=report,
        debug_text_path=path,
    )
