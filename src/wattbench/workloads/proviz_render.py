"""ProViz-style tile render: Mandelbrot at a fixed resolution."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import Workload, result_record


class ProvizRenderWorkload(Workload):
    """Fixed-resolution Mandelbrot tile render, one frame per unit.

    The escape-iteration loop is the render cost; resolution and iteration
    count are fixed so frames are directly comparable across power states.
    """

    name = "proviz_render"
    domain = "proviz"
    unit_name = "frames"
    nominal_rate = 12.0
    width = 640
    height = 360
    max_iter = 120

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backend = "numpy-cpu"
        self._torch = None
        self._prepared = False
        self._checksum = 0.0

    def prepare(self, device: Any | None = None) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.device("cuda")
                x = torch.linspace(-2.2, 0.8, self.width, device=dev)
                y = torch.linspace(-1.2, 1.2, self.height, device=dev)
                xx, yy = torch.meshgrid(x, y, indexing="xy")
                self._torch = torch
                self._c = torch.complex(xx, yy)
                self.backend = "torch-cuda"
                self._prepared = True
                return
        except Exception:
            pass
        x = np.linspace(-2.2, 0.8, self.width, dtype=np.float32)
        y = np.linspace(-1.2, 1.2, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        self._c = (xx + 1j * yy).astype(np.complex64)
        self.backend = "numpy-cpu"
        self._prepared = True

    def _frame(self) -> None:
        if self._torch is not None:
            torch = self._torch
            z = torch.zeros_like(self._c)
            out = torch.zeros(self._c.shape, device=self._c.device, dtype=torch.int32)
            alive = torch.ones(self._c.shape, device=self._c.device, dtype=torch.bool)
            for i in range(self.max_iter):
                z = torch.where(alive, z * z + self._c, z)
                escaped = alive & (torch.abs(z) > 2.0)
                out = torch.where(escaped, torch.full_like(out, i), out)
                alive = alive & ~escaped
                if not bool(alive.any().item()):
                    break
            self._checksum += float(out.sum().item())
            return
        z = np.zeros_like(self._c)
        out = np.zeros(self._c.shape, dtype=np.int32)
        alive = np.ones(self._c.shape, dtype=bool)
        for i in range(self.max_iter):
            zz = z[alive]
            zz = zz * zz + self._c[alive]
            z[alive] = zz
            escaped = np.abs(zz) > 2.0
            idx = np.flatnonzero(alive)
            out.flat[idx[escaped]] = i
            alive.flat[idx[escaped]] = False
            if not alive.any():
                break
        self._checksum += float(out.sum())

    def run(self, duration_s: float) -> dict[str, Any]:
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
            max_iter=self.max_iter,
        )
