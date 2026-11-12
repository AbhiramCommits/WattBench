"""Recovery of injected clock offset/drift and agreement metrics."""

from __future__ import annotations

import numpy as np
import pytest

from wattbench.align import (
    correct_sensor_times,
    estimate_offset_drift,
    validate_pair,
)


def _pair(duration_s: float = 45.0, drift_ppm: float = 180.0, offset_s: float = 0.120):
    rng = np.random.default_rng(11)
    fs = 50.0
    t = np.arange(0.0, duration_s, 1.0 / fs)
    power = 20.0 + np.where((t > 8) & (t < 14), 70.0, 0.0)
    power += np.where((t > 18) & (t < 27), 130.0, 0.0)
    power += np.where((t > 31) & (t < 38), 95.0, 0.0)
    power += np.sin(t * 1.3) * 1.5 + rng.normal(0.0, 0.3, len(t))
    sensor_t = t * (1.0 + drift_ppm * 1e-6) + offset_s
    sensor_p = power + rng.normal(0.0, 0.4, len(t))
    return (
        (t * 1e9).astype(np.int64),
        power,
        (sensor_t * 1e9).astype(np.int64),
        sensor_p,
    )


def test_recovers_injected_offset_and_drift() -> None:
    t_ref, p_ref, t_s, p_s = _pair(drift_ppm=180.0, offset_s=0.120)
    fit = estimate_offset_drift(
        t_ref, p_ref, t_s, p_s, grid_hz=20.0, max_offset_ms=500, max_drift_ppm=500
    )
    assert fit["offset_ms"] == pytest.approx(-120.0, abs=8.0)
    assert fit["drift_ppm"] == pytest.approx(180.0, abs=30.0)
    assert fit["correlation"] > 0.9


def test_correction_maps_sensor_clock_back() -> None:
    t_ref, _p_ref, t_s, _p_s = _pair()
    corrected = correct_sensor_times(t_s, -120.0, 180.0, int(t_s[0]))
    assert corrected[0] == pytest.approx(t_s[0] - 120e6, abs=1.0)
    # After removing the injected 120 ms skew the two clocks stay within 2 ms.
    assert abs((corrected[-1] - t_ref[-1]) - (corrected[0] - t_ref[0])) < 2e6


def test_bias_and_loa_on_known_biased_pair() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(0, 20, 0.05)
    nvml = 100.0 + 8.0 * np.sin(0.3 * t) + rng.normal(0, 0.05, len(t))
    sensor = nvml * 1.05 + 0.5 + rng.normal(0, 0.05, len(t))
    t_ns = (t * 1e9).astype(np.int64)
    metrics = validate_pair(
        t_ns,
        nvml,
        t_ns,
        sensor,
        {"offset_ms": 0.0, "drift_ppm": 0.0},
        window_ns=None,
        grid_hz=20.0,
        bias_threshold_pct=4.0,
    )
    assert metrics["percent_bias"] == pytest.approx(5.5, abs=1.0)
    assert metrics["flagged"] is True
    assert metrics["loa_lower_w"] < metrics["mean_diff_w"] < metrics["loa_upper_w"]
    assert metrics["mean_abs_diff_w"] > 4.0
    assert metrics["pearson_r_full"] > 0.99


def test_flat_biased_pair_flags_bias() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(0, 10, 0.05)
    nvml = np.full(len(t), 100.0) + rng.normal(0, 0.1, len(t))
    sensor = nvml + 6.0
    t_ns = (t * 1e9).astype(np.int64)
    metrics = validate_pair(
        t_ns,
        nvml,
        t_ns,
        sensor,
        {"offset_ms": 0.0, "drift_ppm": 0.0},
        window_ns=None,
        bias_threshold_pct=5.0,
    )
    assert metrics["percent_bias"] == pytest.approx(6.0, abs=0.3)
    assert metrics["flagged"] is True
