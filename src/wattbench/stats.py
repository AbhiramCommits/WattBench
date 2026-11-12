"""Per-cell statistics, robust outlier rejection, and confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np

# Two-sided 95% Student-t critical values (scipy is intentionally not a dep).
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Student-t critical value for a two-sided interval (normal beyond 30 df)."""
    if confidence != 0.95:
        raise ValueError("only 95% intervals are tabulated")
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    lo = max(k for k in _T95 if k < df)
    hi = min(k for k in _T95 if k > df)
    frac = (df - lo) / (hi - lo)
    return _T95[lo] + frac * (_T95[hi] - _T95[lo])


def metric_stats(values: list[float | None] | np.ndarray) -> dict[str, float | int]:
    """Mean, median, sample stdev, CV, and 95% CI on the mean."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "stdev": float("nan"),
            "cv_pct": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(np.mean(arr))
    stdev = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = stdev / np.sqrt(n) if n > 1 else 0.0
    half = t_critical(n - 1) * sem if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 6),
        "median": round(float(np.median(arr)), 6),
        "stdev": round(stdev, 6),
        "cv_pct": round(100.0 * stdev / mean, 4) if mean != 0 else float("nan"),
        "ci95_low": round(mean - half, 6),
        "ci95_high": round(mean + half, 6),
    }


def modified_zscore(values: list[float] | np.ndarray) -> np.ndarray:
    """Median/MAD-based modified z-score (Iglewicz-Hoaglin)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        mad = np.mean(np.abs(arr - med))
    if mad == 0:
        return np.zeros_like(arr)
    return 0.6745 * (arr - med) / mad


def reject_outliers(
    records: list[dict[str, Any]],
    value_key: str = "throughput_per_s",
    z_threshold: float = 3.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flag modified-z outliers; returns (kept, rejected) with reasons.

    Rejection is skipped for fewer than three repeats: a z-score cannot be
    estimated robustly from one or two points.  Rejected repeats are always
    returned so they can be recorded rather than silently discarded.
    """
    if len(records) < 3:
        return list(records), []
    values = [float(r[value_key]) for r in records]
    zs = modified_zscore(values)
    keep: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rec, z in zip(records, zs, strict=True):
        if abs(z) > z_threshold:
            rejected.append(
                {
                    "repeat_idx": int(rec.get("repeat_idx", -1)),
                    "cell_id": rec.get("cell_id"),
                    "value": round(float(rec[value_key]), 6),
                    "modified_z": round(float(z), 3),
                    "reason": f"modified z-score {z:.2f} exceeds {z_threshold}",
                }
            )
        else:
            keep.append(rec)
    if not keep and rejected:
        keep = [min(records, key=lambda r: abs(r[value_key] - np.median(values)))]
        rejected = [r for r in rejected if r is not keep[0]]
    return keep, rejected


_AGG_KEYS = (
    "throughput_per_s",
    "nvml_power_w",
    "sensor_power_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "nvml_baseline_w",
    "sensor_baseline_w",
    "util_pct",
)


def aggregate_cell(
    repeats: list[dict[str, Any]], z_threshold: float = 3.5
) -> dict[str, Any]:
    """Aggregate a cell's repeats with robust outlier rejection."""
    kept, rejected = reject_outliers(repeats, "throughput_per_s", z_threshold)
    aggregates = {
        key: metric_stats([r.get(key) for r in kept])
        for key in _AGG_KEYS
        if any(r.get(key) is not None for r in kept)
    }
    return {
        "repeats_total": len(repeats),
        "repeats_used": len(kept),
        "rejected_outliers": rejected,
        "aggregates": aggregates,
    }


def summarize_cv(cells: list[dict[str, Any]], key: str = "nvml_power_w") -> dict[str, float]:
    """Min/median/max CV across cells, for the report's confidence section."""
    cvs = []
    for cell in cells:
        agg = cell.get("aggregates", {}).get(key, {})
        cv = agg.get("cv_pct")
        if cv is not None and np.isfinite(cv):
            cvs.append(float(cv))
    if not cvs:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan"), "n": 0}
    return {
        "min": round(min(cvs), 4),
        "median": round(float(np.median(cvs)), 4),
        "max": round(max(cvs), 4),
        "n": len(cvs),
    }
