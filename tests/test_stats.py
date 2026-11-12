"""Outlier rejection, CV and confidence-interval arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from wattbench.stats import (
    aggregate_cell,
    metric_stats,
    modified_zscore,
    reject_outliers,
    t_critical,
)


def test_planted_outlier_is_caught_and_recorded() -> None:
    records = [
        {"repeat_idx": 0, "throughput_per_s": 100.0},
        {"repeat_idx": 1, "throughput_per_s": 101.0},
        {"repeat_idx": 2, "throughput_per_s": 102.0},
        {"repeat_idx": 3, "throughput_per_s": 150.0},
    ]
    kept, rejected = reject_outliers(records, "throughput_per_s", z_threshold=3.5)
    assert len(kept) == 3
    assert len(rejected) == 1
    assert rejected[0]["repeat_idx"] == 3
    assert rejected[0]["value"] == pytest.approx(150.0)
    assert rejected[0]["modified_z"] > 3.5
    assert "modified z-score" in rejected[0]["reason"]


def test_small_n_is_not_rejected() -> None:
    records = [{"repeat_idx": 0, "throughput_per_s": 10.0}, {"repeat_idx": 1, "throughput_per_s": 99.0}]
    kept, rejected = reject_outliers(records)
    assert len(kept) == 2 and rejected == []


def test_modified_zscore_matches_definition() -> None:
    z = modified_zscore([10.0, 11.0, 12.0, 20.0])
    expected = 0.6745 * (np.array([10.0, 11.0, 12.0, 20.0]) - 11.5) / 1.0
    np.testing.assert_allclose(z, expected)


def test_cv_and_ci_against_hand_checked_values() -> None:
    stats = metric_stats([10.0, 12.0, 14.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(12.0)
    assert stats["median"] == pytest.approx(12.0)
    assert stats["stdev"] == pytest.approx(2.0)
    assert stats["cv_pct"] == pytest.approx(100.0 * 2.0 / 12.0, abs=1e-4)
    # t(0.975, 2) = 4.303; SEM = 2/sqrt(3); half-width = 4.969
    half = 4.303 * (2.0 / np.sqrt(3.0))
    assert stats["ci95_low"] == pytest.approx(12.0 - half, rel=1e-4)
    assert stats["ci95_high"] == pytest.approx(12.0 + half, rel=1e-4)


def test_t_critical_table_and_fallback() -> None:
    assert t_critical(1) == pytest.approx(12.706)
    assert t_critical(2) == pytest.approx(4.303)
    assert t_critical(31) == pytest.approx(1.96)
    with pytest.raises(ValueError):
        t_critical(5, confidence=0.99)


def test_aggregate_cell_records_rejected_outlier() -> None:
    base = {
        "nvml_power_w": 100.0,
        "sensor_power_w": 99.0,
        "sm_clock_mhz": 1500.0,
        "mem_clock_mhz": 9000.0,
        "temp_c": 65.0,
        "nvml_baseline_w": 20.0,
        "sensor_baseline_w": 20.2,
        "util_pct": 95.0,
    }
    repeats = [dict(base, repeat_idx=i, cell_id=f"c__r{i}", throughput_per_s=tp)
               for i, tp in enumerate([100.0, 101.0, 102.0, 160.0])]
    agg = aggregate_cell(repeats, z_threshold=3.5)
    assert agg["repeats_total"] == 4
    assert agg["repeats_used"] == 3
    assert len(agg["rejected_outliers"]) == 1
    assert agg["aggregates"]["throughput_per_s"]["mean"] == pytest.approx(101.0)
