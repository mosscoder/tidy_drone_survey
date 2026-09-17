"""Progress heartbeat for a caller's stall watchdog.

`beat()` touches the file named by the environment variable TIDYSURVEY_HEARTBEAT (created if
missing), at most once per second per process, and does nothing when the variable is unset. The
engine calls it once per unit of work in the loops that can hang (the scorer's blocks, the match
chunk's tiles, the warp's blocks); a caller that sets the variable can watch the file's mtime and
decide that a loop has stalled independent of map size. The match chunk runs in a subprocess, which
inherits the variable, so the boundary needs nothing else. Never raises.
"""
from __future__ import annotations
import os
import time

HEARTBEAT_ENV = "TIDYSURVEY_HEARTBEAT"
_MIN_INTERVAL = 1.0
_last = -float("inf")


def beat() -> None:
    global _last
    path = os.environ.get(HEARTBEAT_ENV)
    if not path:
        return
    now = time.monotonic()
    if now - _last < _MIN_INTERVAL:
        return
    _last = now
    try:
        if not os.path.exists(path):
            open(path, "a").close()
        os.utime(path, None)
    except OSError:
        pass
