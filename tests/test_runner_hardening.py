"""Signal-safe cleanup and sweep run logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from wattbench import runner
from wattbench.config import (
    ClockPoint,
    GpuSection,
    RigConfig,
    RunSection,
    SamplingSection,
    SensorSection,
    SweepSection,
    WorkloadSection,
)
from wattbench.devices.gpu import SyntheticGpu


def _config() -> RigConfig:
    return RigConfig(
        run=RunSection(name="interrupt", seed=1, out_dir="runs"),
        sampling=SamplingSection(gpu_hz=30, sensor_hz=50),
        gpu=GpuSection(mode="synthetic", default_power_limit_w=250),
        sensor=SensorSection(mode="synthetic", sample_rate_hz=50),
        sweep=SweepSection(
            power_limits_w=[90.0],
            clock_points=[ClockPoint(0, 0, "unlocked")],
            repeats=1,
            soak_seconds=0.1,
            soak_timeout_s=0.2,
            idle_baseline_seconds=0.1,
            measure_seconds=0.2,
            settle_seconds=0.0,
            randomized=False,
        ),
        workloads=[WorkloadSection("ai_inference", "ai", 0.2)],
    )


def test_sigint_restores_defaults_and_records_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config()
    gpu = SyntheticGpu(seed=4, power_limit_w=cfg.gpu.default_power_limit_w)
    monkeypatch.setattr(runner, "get_gpu", lambda **kwargs: gpu)

    def interrupted(*args, **kwargs):
        gpu.set_power_limit(77)
        gpu.set_locked_clocks(1200, 1200)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_cell", interrupted)
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        runner.sweep(cfg, out=run_dir)

    telemetry = gpu.read_telemetry()
    assert telemetry["power_limit_w"] == pytest.approx(250.0)
    assert gpu.locked_clocks() is None
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
    assert meta["counts"]["complete"] == 0
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "SIGINT received" in log
    assert "restored default power limit" in log


def test_normal_sweep_writes_run_log(tmp_path: Path) -> None:
    cfg = _config()
    cfg.sweep.soak_timeout_s = 0.3
    cfg.sweep.measure_seconds = 0.3
    cfg.workloads[0].duration_s = 0.3
    out = tmp_path / "run"
    meta = runner.sweep(cfg, out=out)
    assert meta["status"] == "complete"
    log = (out / "run.log").read_text(encoding="utf-8")
    assert "sweep finished" in log
    assert "run log:" in log
    # Keep the config file serialisable for provenance parity.
    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["sweep"]["power_limits_w"] == [90.0]
