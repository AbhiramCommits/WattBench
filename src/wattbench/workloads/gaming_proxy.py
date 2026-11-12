"""Fragment-shader-style gaming proxy at an uncapped frame rate."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import Workload, result_record


class GamingProxyWorkload(Workload):
    """Fixed pixel-shader workload driving a frame loop as fast as it will go.

    Each frame evaluates several analytic lighting layers over every pixel of
    a fixed 320x180 buffer.  It is not a game engine; it reproduces the shape
    of a frame loop (one deterministic unit of fragment work per frame) so
    uncapped frame rate becomes a stable throughput metric.
    """

    name = "gaming_proxy"
    domain = "gaming"
    unit_name = "frames"
    nominal_rate = 90.0
    width = 320
    height = 180
    shader_layers = 4

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backend = "numpy-cpu"
        self._torch = None
        self._prepared = False
        self._checksum = 0.0

    def prepare(self, device: Any | None = None) -> None:
        """Allocate the pixel grid on CUDA when available, else numpy."""
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.device("cuda")
                x = torch.linspace(-1.0, 1.0, self.width, device=dev)
                y = torch.linspace(-1.0, 1.0, self.height, device=dev)
                xx, yy = torch.meshgrid(x, y, indexing="xy")
                self._torch = torch
                self._x, self._y = xx, yy
                self.backend = "torch-cuda"
                self._prepared = True
                return
        except Exception:
            pass
        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        self._x, self._y = np.meshgrid(x, y)
        self.backend = "numpy-cpu"
        self._prepared = True

    def _frame(self) -> None:
        if self._torch is not None:
            torch = self._torch
            col = torch.sin(self._x * 12.9898 + self._y * 78.233) * 43758.5453
            for k in range(self.shader_layers):
                col = col + torch.sin(
                    self._x * (k + 1) * 3.1 + self._y * (k + 2) * 2.3 + col * 0.1
                ) * 0.5
            self._checksum += float(torch.abs(col).sum().item())
            return
        col = np.sin(self._x * 12.9898 + self._y * 78.233) * 43758.5453
        for k in range(self.shader_layers):
            col = col + np.sin(
                self._x * (k + 1) * 3.1 + self._y * (k + 2) * 2.3 + col * 0.1
            ) * 0.5
        self._checksum += float(np.abs(col).sum())

    def run(self, duration_s: float) -> dict[str, Any]:
        """Run the uncapped shader frame loop for the requested duration."""
        if not self._prepared:
            self.prepare()
        start = time.perf_counter()
        deadline = start + duration_s
        units = 0
        self._checksum = 0.0
        while time.perf_counter() < deadline:
            self._frame()
            units += 1
            self._pace_after_unit(start, units)
        wall = time.perf_counter() - start
        return result_record(
            units,
            self.unit_name,
            wall,
            compute_backend=self.backend,
            resolution=f"{self.width}x{self.height}",
            shader_layers=self.shader_layers,
        )
