"""Batched neural-network inference workload (torch CUDA when available)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import Workload, result_record


class AiInferenceWorkload(Workload):
    """Fixed-shape batched MLP inference: 3 x (matmul + ReLU) + reduction.

    Shapes never change between runs, so one "inference" is a deterministic
    unit of work.  With CUDA torch the work really executes on the GPU; the
    numpy path is the hardware-free fallback.
    """

    name = "ai_inference"
    domain = "ai"
    unit_name = "inferences"
    nominal_rate = 60.0
    batch = 32
    features = 768
    layers = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backend = "numpy-cpu"
        self._torch = None
        self._x: Any = None
        self._w: list[Any] = []
        self._checksum = 0.0

    def prepare(self, device: Any | None = None) -> None:
        """Allocate tensors on CUDA when available, else numpy."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.manual_seed(self.seed)
                dev = torch.device("cuda")
                self._torch = torch
                self._x = torch.randn(
                    (self.batch, self.features), device=dev, dtype=torch.float32
                )
                self._w = [
                    (torch.randn((self.features, self.features), device=dev) * 0.02).to(
                        torch.float32
                    )
                    for _ in range(self.layers)
                ]
                self.backend = "torch-cuda"
                self._prepared = True
                return
        except Exception:
            pass
        rng = np.random.default_rng(self.seed)
        self._x = rng.standard_normal((self.batch, self.features), dtype=np.float32)
        self._w = [
            (rng.standard_normal((self.features, self.features)) * 0.02).astype(np.float32)
            for _ in range(self.layers)
        ]
        self.backend = "numpy-cpu"
        self._prepared = True

    def _infer(self) -> None:
        if self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                y = self._x
                for w in self._w:
                    y = torch.relu(y @ w)
                self._checksum += float(y.sum().item())
            return
        y = self._x
        for w in self._w:
            y = np.maximum(y @ w, 0.0)
        self._checksum += float(y.sum())

    def run(self, duration_s: float) -> dict[str, Any]:
        """Run fixed-shape inferences for the requested duration."""
        if not self._prepared:
            self.prepare()
        start = time.perf_counter()
        deadline = start + duration_s
        units = 0
        self._checksum = 0.0
        while time.perf_counter() < deadline:
            self._infer()
            units += 1
            self._pace_after_unit(start, units)
        wall = time.perf_counter() - start
        return result_record(
            units,
            self.unit_name,
            wall,
            compute_backend=self.backend,
            batch=self.batch,
            features=self.features,
            layers=self.layers,
        )
