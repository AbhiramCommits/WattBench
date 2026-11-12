"""Workload base class and deterministic pacing helper.

The workloads execute real compute (numpy, optionally torch/CUDA).  In
synthetic mode the runner also supplies a ``clock_scale`` derived from the
emulated GPU's throttled clock; the workload then paces its *service rate* to
that scale while still computing every unit.  This is what makes a synthetic
power sweep produce a physically meaningful throughput curve.
"""

from __future__ import annotations

import abc
import threading
import time
from typing import Any


class Workload(abc.ABC):
    """A deterministic, repeatable piece of GPU/CPU work."""

    name: str = "workload"
    domain: str = "ai"
    unit_name: str = "units"
    nominal_rate: float = 1.0

    def __init__(self, clock_scale: float = 1.0, pacing: bool = False, seed: int = 0) -> None:
        self.clock_scale = max(0.02, min(1.0, clock_scale))
        self.pacing = pacing
        self.seed = seed
        self._prepared = False
        self._stop = threading.Event()

    def prepare(self, device: Any | None = None) -> None:  # noqa: B027
        """Allocate buffers. Safe to call once per cell."""

    @abc.abstractmethod
    def run(self, duration_s: float) -> dict[str, Any]:
        """Execute work for ``duration_s`` and return a result record."""

    def teardown(self) -> None:  # noqa: B027
        """Release resources."""

    def _pace_after_unit(self, start: float, units: int) -> None:
        """Sleep so the modelled service rate follows ``clock_scale``."""
        if not self.pacing:
            return
        target_elapsed = units / (self.nominal_rate * self.clock_scale)
        delay = start + target_elapsed - time.perf_counter()
        if delay > 0:
            time.sleep(min(delay, 0.25))


def result_record(units: int, unit_name: str, wall_s: float, **extra: Any) -> dict[str, Any]:
    """Build the common result dict, including throughput."""
    record: dict[str, Any] = {
        "work_units": int(units),
        "unit_name": unit_name,
        "wall_s": round(float(wall_s), 6),
        "throughput_per_s": round(units / wall_s, 6) if wall_s > 0 else 0.0,
    }
    record.update(extra)
    return record
