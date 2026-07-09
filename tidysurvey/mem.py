"""Memory telemetry for hours-long runs.

A multi-hour stitch on 40-gigapixel rasters lives or dies by memory. When the
OS memory-pressure killer (jetsam on macOS) SIGKILLs the process, the log just
stops mid-stage: no traceback, no death-rattle line — SIGKILL is uncatchable,
so the process never gets to confess. The only defence is a breadcrumb trail:
sample memory every few seconds so the *last line before a silent cutoff*
quantifies how close to the ceiling we were, which tells a memory kill apart
from an external one (terminal close, manual kill) after the fact.

Process RSS alone understates the truth on Apple-Silicon unified memory — Metal
(MPS) allocations and loky calibration workers don't land in the parent's RSS —
so we log SYSTEM availability too. That system number is what jetsam actually
watches, and it already accounts for child processes and the GPU.

Nothing here may ever raise into the pipeline: on any error the probe goes
quiet rather than taking a 15-hour run down with it.
"""
from __future__ import annotations

import os
import resource
import threading

try:
    import psutil
    _PROC = psutil.Process()
except Exception:                       # psutil should be present; degrade if not
    psutil = None
    _PROC = None

# ru_maxrss is BYTES on macOS, KILOBYTES on Linux.
_MAXRSS_TO_GB = (1 / 1024**3) if os.uname().sysname == "Darwin" else (1 / 1024**2)

# System free memory below this (GB) is flagged — on a 64 GB box this is genuine
# pressure territory where jetsam starts looking for something to kill.
_LOW_FREE_GB = 6.0


def peak_gb() -> float:
    """High-water mark of this process's resident set, in GB (monotonic)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_TO_GB


def line() -> str:
    """One-line snapshot: process RSS, process peak, and system used%/free —
    the system figure is the one that predicts a jetsam kill."""
    peak = peak_gb()
    if psutil is None:
        return f"peak {peak:.1f}G (psutil absent)"
    rss = _PROC.memory_info().rss / 1024**3
    vm = psutil.virtual_memory()
    free_gb = vm.available / 1024**3
    flag = "  ⚠ LOW" if free_gb < _LOW_FREE_GB else ""
    return (f"rss {rss:.1f}G · peak {peak:.1f}G · "
            f"sys {vm.percent:.0f}% used, {free_gb:.1f}G free{flag}")


class Sampler:
    """Daemon thread logging `line()` every `every` seconds, tagged with the
    stage the pipeline is in. Set the interval with $TIDYSURVEY_MEM_EVERY.
    Never raises into the pipeline; on any error it simply stops sampling."""

    def __init__(self, say, every: float = 30.0):
        self._say = say
        try:
            every = float(os.environ.get("TIDYSURVEY_MEM_EVERY", every))
        except (TypeError, ValueError):
            pass
        self._every = max(5.0, every)
        self._stage = "startup"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mem-sampler",
                                        daemon=True)

    def start(self) -> "Sampler":
        try:
            self._thread.start()
        except Exception:
            pass
        return self

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def sample_now(self, note: str = "") -> None:
        try:
            self._say(f"  · mem [{self._stage}{note}] {line()}")
        except Exception:
            pass

    def _loop(self) -> None:
        # Event.wait returns True the moment stop() fires, so the interval never
        # delays shutdown; it returns False on timeout -> take a sample.
        while not self._stop.wait(self._every):
            self.sample_now()

    def stop(self, final: bool = True) -> None:
        self._stop.set()
        if final:
            self.sample_now("·final")
