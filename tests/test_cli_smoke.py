"""End-to-end tiny sweep -> analyze -> report in a temporary directory."""

from __future__ import annotations

from pathlib import Path

import yaml

from wattbench.cli import main
from wattbench.config import (
    AnalysisSection,
    ClockPoint,
    GpuSection,
    RigConfig,
    RunSection,
    SamplingSection,
    SensorSection,
    SweepSection,
    WorkloadSection,
)


def _tiny_config(path: Path) -> Path:
    cfg = RigConfig(
        run=RunSection(name="smoke", seed=123, out_dir="runs"),
        sampling=SamplingSection(gpu_hz=30, sensor_hz=50, align_grid_hz=20, bias_threshold_pct=25.0),
        gpu=GpuSection(mode="synthetic", seed=9, default_power_limit_w=240),
        sensor=SensorSection(mode="synthetic", sample_rate_hz=50, seed=3),
        sweep=SweepSection(
            power_limits_w=[90.0, 140.0],
            clock_points=[ClockPoint(0, 0, "unlocked")],
            repeats=1,
            soak_seconds=0.3,
            soak_timeout_s=0.5,
            idle_baseline_seconds=0.4,
            measure_seconds=0.8,
            randomized=False,
            settle_seconds=0.05,
        ),
        analysis=AnalysisSection(knee_marginal_fraction=0.9),
        workloads=[WorkloadSection("ai_inference", "ai", 0.8)],
    )
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def test_tiny_sweep_analyze_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = _tiny_config(tmp_path / "rig.yaml")
    run_dir = tmp_path / "runs" / "smoke"

    assert main(["sweep", "--config", str(cfg_path), "--out", str(run_dir)]) == 0
    cells = sorted((run_dir / "cells").iterdir())
    assert len(cells) == 2
    for cell in cells:
        assert (cell / "raw.csv").exists()
        assert (cell / "manifest.json").exists()

    assert main(["analyze", str(run_dir), "--out", str(run_dir / "results")]) == 0
    results = run_dir / "results" / "results.json"
    assert results.exists()
    assert (run_dir / "results" / "cells.csv").exists()

    assert main(["report", str(run_dir)]) == 0
    assert (run_dir / "results" / "report.md").exists()
    assert (run_dir / "results" / "report.html").exists()
    assert (run_dir / "results" / "plots" / "perf_vs_power_limit.png").exists()
    report = (run_dir / "results" / "report.md").read_text(encoding="utf-8")
    assert "## Headline results" in report
    assert "| Workload | Peak throughput |" in report
    assert "Measurement source: synthetic" in report
    # Top-level mirror written from the working directory.
    assert (tmp_path / "results" / "report.md").exists()

    # --no-mirror writes only into the run's own results directory.
    (tmp_path / "results" / "report.md").unlink()
    assert main(["report", str(run_dir), "--no-mirror"]) == 0
    assert not (tmp_path / "results" / "report.md").exists()
    assert (run_dir / "results" / "report.md").exists()

    self_cmp = main(
        ["compare", "--baseline", str(run_dir), "--candidate", str(run_dir)]
    )
    assert self_cmp == 0
