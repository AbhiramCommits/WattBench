"""Helpers for constructing minimal but schema-valid results documents."""

from __future__ import annotations

from typing import Any


def _metric(mean: float, stdev: float = 1.0, n: int = 2) -> dict[str, Any]:
    return {
        "n": n,
        "mean": mean,
        "median": mean,
        "stdev": stdev,
        "cv_pct": 100.0 * stdev / mean if mean else 0.0,
        "ci95_low": mean - stdev,
        "ci95_high": mean + stdev,
    }


def make_results_doc(
    throughput: float = 100.0,
    stdev: float = 1.0,
    run_id: str = "test-run",
    schema_version: str = "1.0",
) -> dict[str, Any]:
    cell_id = "ai_inference__pl090__clk-unlocked"
    alignment = {
        "offset_ms": 1.0,
        "drift_ppm": 2.0,
        "sensor_t0_ns": 1_000_000_000,
        "correlation": 0.99,
        "grid_hz": 20.0,
        "nvml_samples": 100,
        "sensor_samples": 250,
        "mean_abs_diff_w": 1.0,
        "mean_diff_w": 0.5,
        "percent_bias": 0.5,
        "pearson_r": 0.5,
        "pearson_r_full": 0.99,
        "loa_lower_w": -1.5,
        "loa_upper_w": 2.5,
        "flagged": False,
        "bias_threshold_pct": 5.0,
        "window": "measurement",
    }
    repeat = {
        "repeat_idx": 0,
        "cell_id": cell_id + "__r0",
        "work_units": int(throughput),
        "wall_s": 1.0,
        "throughput_per_s": throughput,
        "nvml_power_w": 90.0,
        "sensor_power_w": 89.5,
        "sm_clock_mhz": 1500.0,
        "mem_clock_mhz": 9000.0,
        "temp_c": 60.0,
        "util_pct": 90.0,
        "nvml_baseline_w": 20.0,
        "sensor_baseline_w": 20.5,
        "raw_csv": "cells/x/raw.csv",
        "alignment": alignment,
    }
    cell = {
        "cell_id": cell_id,
        "workload": "ai_inference",
        "domain": "ai",
        "power_limit_w": 90.0,
        "clock_label": "unlocked",
        "clock_min_mhz": 0,
        "clock_max_mhz": 0,
        "source": "synthetic",
        "repeats_total": 1,
        "repeats_used": 1,
        "rejected_outliers": [],
        "aggregates": {
            "throughput_per_s": _metric(throughput, stdev),
            "nvml_power_w": _metric(90.0, 0.5),
            "sensor_power_w": _metric(89.5, 0.6),
        },
        "per_repeat": [repeat],
    }
    return {
        "schema_version": schema_version,
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "run": {
            "run_id": run_id,
            "dir": f"runs/{run_id}",
            "source": "synthetic",
            "config_hash": "deadbeef",
            "matrix": {"configs": 1, "workloads": ["ai_inference"], "repeats": 1},
            "counts": {"complete": 1},
        },
        "provenance": {"config": {"analysis": {"regression_z": 3.0, "regression_rel_tol": 0.03}}},
        "alignment": {
            "bias_threshold_pct": 5.0,
            "offset_ms_mean": 1.0,
            "offset_ms_stdev": 0.1,
            "drift_ppm_mean": 2.0,
            "drift_ppm_stdev": 0.1,
            "mean_abs_diff_w": 1.0,
            "percent_bias": 0.5,
            "pearson_r": 0.5,
            "pearson_r_full": 0.99,
            "loa_lower_w": -1.5,
            "loa_upper_w": 2.5,
            "cells_flagged": [],
            "repeats_flagged": 0,
            "per_cell": {},
        },
        "cells": [cell],
    }
