"""Perf-per-joule, knee detection, and regression verdicts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._build import make_results_doc
from wattbench.align import AlignmentResult
from wattbench.analysis import _trapz_energy_j, compute_efficiency, knee, regression_compare


def _alignment(t0_ns: int, n: int) -> AlignmentResult:
    return AlignmentResult(
        offset_ms=0.0,
        drift_ppm=0.0,
        sensor_t0_ns=t0_ns,
        correlation=1.0,
        grid_hz=20.0,
        nvml_samples=n,
        sensor_samples=n,
        mean_abs_diff_w=0.0,
        mean_diff_w=0.0,
        percent_bias=0.0,
        pearson_r=1.0,
        pearson_r_full=1.0,
        loa_lower_w=0.0,
        loa_upper_w=0.0,
        flagged=False,
        bias_threshold_pct=5.0,
    )


def test_trapezoid_energy_analytic() -> None:
    t = np.array([0, 5e9, 10e9], dtype=np.int64)
    assert _trapz_energy_j(t, np.array([100.0, 100.0, 100.0])) == pytest.approx(1000.0)
    assert _trapz_energy_j(t[:2], np.array([0.0, 200.0])) == pytest.approx(500.0)
    assert _trapz_energy_j(np.array([1], dtype=np.int64), np.array([5.0])) == 0.0


def test_compute_efficiency_constant_power() -> None:
    t0 = 1_000_000_000
    times = [t0 + i * 50_000_000 for i in range(21)]  # 1.000 s at 20 Hz
    rows = []
    for ts in times:
        rows.append({"source": "nvml", "timestamp_ns": ts, "power_w": 100.0})
        rows.append({"source": "sensor", "timestamp_ns": ts, "power_w": 100.0})
    df = pd.DataFrame(rows)
    manifest = {
        "regions": {"measure_start": t0, "measure_end": t0 + 1_000_000_000},
        "result": {"work_units": 500, "throughput_per_s": 500.0, "unit_name": "inferences", "wall_s": 1.0},
    }
    eff = compute_efficiency(df, manifest, _alignment(t0, 21), 20.0, 20.0)
    assert eff["energy_j_nvml"] == pytest.approx(100.0)
    assert eff["energy_j_sensor"] == pytest.approx(100.0)
    assert eff["work_per_joule_nvml"] == pytest.approx(5.0)
    assert eff["perf_per_watt_nvml"] == pytest.approx(5.0)
    assert eff["idle_subtracted_energy_j_nvml"] == pytest.approx(80.0)
    assert eff["idle_subtracted_work_per_joule_nvml"] == pytest.approx(6.25)


def _cells(throughputs: list[float], powers: list[float]) -> list[dict]:
    return [
        {
            "workload": "ai_inference",
            "clock_label": "unlocked",
            "power_limit_w": p,
            "aggregates": {
                "throughput_per_s": {"mean": tp},
                "nvml_power_w": {"mean": p},
            },
        }
        for p, tp in zip(powers, throughputs, strict=True)
    ]


def test_knee_on_known_curve() -> None:
    cells = _cells([40.0, 70.0, 90.0, 95.0], [100.0, 150.0, 200.0, 250.0])
    result = knee(cells, marginal_fraction=0.5)
    assert result["power_limit_w"] == 200.0
    assert result["throughput_pct_of_peak"] == pytest.approx(100.0 * 90.0 / 95.0, abs=0.01)
    assert result["power_pct_of_peak"] == pytest.approx(80.0)
    assert result["max_efficiency_power_limit_w"] == 150.0
    assert result["peak_power_limit_w"] == 250.0


def test_knee_requires_three_points() -> None:
    with pytest.raises(ValueError):
        knee(_cells([10.0, 20.0], [100.0, 200.0]))


def _write(tmp_path: Path, doc: dict, name: str) -> Path:
    d = tmp_path / name / "results"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path / name


def test_regression_compare_pass_and_fail(tmp_path: Path) -> None:
    base = _write(tmp_path, make_results_doc(throughput=100.0, stdev=1.0), "baseline")
    same = _write(tmp_path, make_results_doc(throughput=101.0, stdev=1.0), "candidate-pass")
    bad = _write(tmp_path, make_results_doc(throughput=90.0, stdev=1.0), "candidate-fail")
    passed = regression_compare(base, same)
    assert passed["overall"] == "PASS"
    assert passed["cells_compared"] == 1
    failed = regression_compare(base, bad)
    assert failed["overall"] == "FAIL"
    assert failed["failures"][0]["delta"] == pytest.approx(-10.0)
    assert failed["failures"][0]["tolerance"] == pytest.approx(3.0)
