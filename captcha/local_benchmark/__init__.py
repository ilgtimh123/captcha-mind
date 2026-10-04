"""Local-only browser benchmark for CaptchaMind.

The package intentionally restricts browser execution to localhost/loopback.
"""

from .human_motion import LocalHumanMotionPlanner
from .server import LocalBenchmarkServer

__all__ = ["LocalHumanMotionPlanner", "LocalBenchmarkServer"]
