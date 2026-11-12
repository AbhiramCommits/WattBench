"""Typed loader for ``config/rig.yaml``.

The same config object is serialised into every run manifest, and its hash is
recorded so a result can always be traced back to the exact sweep definition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunSection:
    name: str = "default"
    seed: int = 20261112
    out_dir: str = "runs"


@dataclass
class SamplingSection:
    gpu_hz: float = 20.0
    sensor_hz: float = 50.0
    align_grid_hz: float = 20.0
    bias_threshold_pct: float = 5.0


@dataclass
class GpuSection:
    mode: str = "auto"
    index: int = 0
    seed: int = 42
    default_power_limit_w: float = 250.0


@dataclass
class SensorSection:
    mode: str = "auto"
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    shunt_ohms: float = 0.005
    units: str = "mv"
    sample_rate_hz: float = 50.0
    seed: int = 7


@dataclass
class ClockPoint:
    min_mhz: int = 0
    max_mhz: int = 0
    label: str = "unlocked"


@dataclass
class WorkloadSection:
    name: str = "ai_inference"
    domain: str = "ai"
    duration_s: float = 4.0


@dataclass
class SweepSection:
    power_limits_w: list[float] = field(default_factory=lambda: [90.0, 140.0, 190.0, 240.0])
    clock_points: list[ClockPoint] = field(
        default_factory=lambda: [ClockPoint(0, 0, "unlocked"), ClockPoint(1700, 1700, "locked-1700")]
    )
    repeats: int = 2
    soak_seconds: float = 2.5
    soak_timeout_s: float = 6.0
    soak_slope_c_per_s: float = 0.05
    idle_baseline_seconds: float = 1.2
    measure_seconds: float = 4.0
    randomized: bool = True
    outlier_z: float = 3.5
    settle_seconds: float = 0.25


@dataclass
class AnalysisSection:
    knee_marginal_fraction: float = 0.25
    regression_z: float = 3.0
    regression_rel_tol: float = 0.03


@dataclass
class RigConfig:
    run: RunSection = field(default_factory=RunSection)
    sampling: SamplingSection = field(default_factory=SamplingSection)
    gpu: GpuSection = field(default_factory=GpuSection)
    sensor: SensorSection = field(default_factory=SensorSection)
    sweep: SweepSection = field(default_factory=SweepSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    workloads: list[WorkloadSection] = field(
        default_factory=lambda: [
            WorkloadSection("ai_inference", "ai", 4.0),
            WorkloadSection("proviz_render", "proviz", 4.0),
            WorkloadSection("gaming_proxy", "gaming", 4.0),
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        if self.sweep.repeats < 1:
            raise ValueError("sweep.repeats must be >= 1")
        if not self.sweep.power_limits_w:
            raise ValueError("sweep.power_limits_w must not be empty")
        if self.sampling.gpu_hz <= 0 or self.sensor.sample_rate_hz <= 0:
            raise ValueError("sampling rates must be positive")
        if self.sensor.shunt_ohms <= 0:
            raise ValueError("sensor.shunt_ohms must be positive")
        if self.sensor.units not in ("mv", "a"):
            raise ValueError("sensor.units must be 'mv' or 'a'")
        if self.sweep.measure_seconds <= 0 or self.sweep.idle_baseline_seconds < 0:
            raise ValueError("measurement durations must be positive")
        if self.gpu.mode not in ("auto", "hardware", "synthetic"):
            raise ValueError("gpu.mode must be auto, hardware, or synthetic")
        if self.sensor.mode not in ("auto", "hardware", "synthetic"):
            raise ValueError("sensor.mode must be auto, hardware, or synthetic")
        if not 0 < self.analysis.knee_marginal_fraction <= 1:
            raise ValueError("analysis.knee_marginal_fraction must be in (0, 1]")
        if not self.workloads:
            raise ValueError("at least one workload is required")


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    allowed = set(cls.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> RigConfig:
    """Load and validate a rig config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    allowed = set(RigConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    cfg = RigConfig(
        run=_build(RunSection, raw.get("run")),
        sampling=_build(SamplingSection, raw.get("sampling")),
        gpu=_build(GpuSection, raw.get("gpu")),
        sensor=_build(SensorSection, raw.get("sensor")),
        sweep=_build(SweepSection, raw.get("sweep")),
        analysis=_build(AnalysisSection, raw.get("analysis")),
    )
    wls = raw.get("workloads")
    if wls is not None:
        cfg.workloads = [_build(WorkloadSection, w) for w in wls]
    clock_points = raw.get("sweep", {}).get("clock_points")
    if clock_points is not None:
        cfg.sweep.clock_points = [_build(ClockPoint, cp) for cp in clock_points]
    cfg.validate()
    return cfg
