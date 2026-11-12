"""Align NVML and external-sensor power streams and cross-validate them.

The two clocks are independent: the sensor starts with an arbitrary offset and
its crystal drifts.  We estimate a constant time offset and a linear drift by
maximising normalized cross-correlation on a common resampled grid, then report
agreement metrics (MAE, percent bias, Pearson r, Bland-Altman limits) over the
steady-state measurement window.

Sign convention
---------------
The fitted correction is applied as::

    t_aligned = (t_sensor - t0) / (1 + drift_ppm * 1e-6) + t0 + offset_ms*1e6

so ``offset_ms`` is the shift *added* to the sensor clock and ``drift_ppm`` is
the sensor's clock error (positive means the sensor runs fast; the division by
``1 + drift_ppm*1e-6`` corrects it).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Raised when the two streams cannot be aligned."""


@dataclass
class AlignmentResult:
    """Fitted clock relationship and agreement statistics."""

    offset_ms: float
    drift_ppm: float
    sensor_t0_ns: int
    correlation: float
    grid_hz: float
    nvml_samples: int
    sensor_samples: int
    mean_abs_diff_w: float
    mean_diff_w: float
    percent_bias: float
    pearson_r: float
    pearson_r_full: float
    loa_lower_w: float
    loa_upper_w: float
    flagged: bool
    bias_threshold_pct: float
    window: str = "measurement"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def correct_sensor_times(
    timestamps_ns: np.ndarray,
    offset_ms: float,
    drift_ppm: float,
    sensor_t0_ns: int,
) -> np.ndarray:
    """Apply the fitted offset/drift correction to sensor timestamps."""
    t = np.asarray(timestamps_ns, dtype=np.float64)
    corrected = (t - sensor_t0_ns) / (1.0 + drift_ppm * 1e-6) + sensor_t0_ns
    return corrected + offset_ms * 1e6


def _smooth_rows(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Moving average along the last axis, edge-padded (handles 1-D or 2-D)."""
    if k <= 1:
        return x
    arr = np.asarray(x, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    pad = k // 2
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(padded, axis=1)
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), cs], axis=1)
    out = (cs[:, k:] - cs[:, :-k]) / k
    return out[0] if squeeze else out


def _normalized_corr_rows(sensor: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation (inputs are already finite)."""
    mean_s = sensor.mean(axis=1)
    centered_s = sensor - mean_s[:, None]
    var_s = (centered_s**2).sum(axis=1)
    mean_r = ref.mean()
    centered_r = ref - mean_r
    var_r = (centered_r**2).sum()
    num = (centered_s * centered_r[None, :]).sum(axis=1)
    den = np.sqrt(np.maximum(var_s, 1e-18) * max(var_r, 1e-18))
    corr = num / den
    return np.where(var_s > 1e-12, corr, -1.0)


def estimate_offset_drift(
    t_ref_ns: np.ndarray,
    p_ref: np.ndarray,
    t_sensor_ns: np.ndarray,
    p_sensor: np.ndarray,
    grid_hz: float = 20.0,
    max_offset_ms: float = 1000.0,
    max_drift_ppm: float = 600.0,
    estimation_hz: float = 100.0,
) -> dict[str, float]:
    """Estimate (offset_ms, drift_ppm, correlation) by grid search.

    Correlation is computed on the *derivative* of the power traces after a
    short moving average.  Raw power is dominated by long flat plateaus at the
    power cap, where a 100 ms shift is invisible; the transitions between
    soak, idle and load carry the timing information and are what make the
    offset identifiable.  Coarse drift is searched on a 20 ppm grid; all lags
    for a given drift are evaluated in one vectorised operation, then a local
    refinement resolves drift to 2 ppm and offset to 0.25 ms.
    """
    t_ref = np.asarray(t_ref_ns, dtype=np.float64) / 1e9
    t_s = np.asarray(t_sensor_ns, dtype=np.float64) / 1e9
    p_ref = np.asarray(p_ref, dtype=np.float64)
    p_s = np.asarray(p_sensor, dtype=np.float64)
    finite_ref = np.isfinite(p_ref)
    finite_s = np.isfinite(p_s)
    t_ref, p_ref = t_ref[finite_ref], p_ref[finite_ref]
    t_s, p_s = t_s[finite_s], p_s[finite_s]
    if len(t_ref) < 10 or len(t_s) < 10:
        raise AlignmentError(f"too few samples to align: nvml={len(t_ref)}, sensor={len(t_s)}")
    order_ref = np.argsort(t_ref)
    order_s = np.argsort(t_s)
    t_ref, p_ref = t_ref[order_ref], p_ref[order_ref]
    t_s, p_s = t_s[order_s], p_s[order_s]
    t0_s = float(t_s[0])
    est_hz = max(float(estimation_hz), float(grid_hz))

    def evaluate(drift_values: np.ndarray, offset_values: np.ndarray) -> tuple[float, float, float]:
        best = (-2.0, 0.0, 0.0)
        offset_values = np.asarray(offset_values, dtype=np.float64)
        for drift in drift_values:
            t_corr = (t_s - t0_s) / (1.0 + drift * 1e-6) + t0_s
            g0 = max(t_ref[0], t_corr[0])
            g1 = min(t_ref[-1], t_corr[-1])
            if g1 - g0 < 0.5:
                continue
            n_grid = max(int((g1 - g0) * est_hz) + 1, 20)
            grid = np.linspace(g0, g1, n_grid)
            ref = _smooth_rows(np.interp(grid, t_ref, p_ref), k=5)
            dref = np.diff(ref)
            valid = (grid[0] - offset_values / 1000.0 >= t_corr[0]) & (
                grid[-1] - offset_values / 1000.0 <= t_corr[-1]
            )
            if not valid.any():
                continue
            offs = offset_values[valid]
            sample_times = grid[None, :] - offs[:, None] / 1000.0
            sensor = np.interp(sample_times.ravel(), t_corr, p_s)
            sensor = _smooth_rows(sensor.reshape(sample_times.shape), k=5)
            dsensor = np.diff(sensor, axis=1)
            corr = _normalized_corr_rows(dsensor, dref)
            idx = int(np.argmax(corr))
            if corr[idx] > best[0]:
                best = (float(corr[idx]), float(drift), float(offs[idx]))
        return best

    coarse_drifts = np.arange(-max_drift_ppm, max_drift_ppm + 1e-9, 20.0)
    coarse_offsets = np.arange(-max_offset_ms, max_offset_ms + 1e-9, 2.0)
    corr, drift, offset = evaluate(coarse_drifts, coarse_offsets)
    if corr < -1.5:
        raise AlignmentError("no overlapping sample window after drift correction")

    fine_drifts = drift + np.arange(-20.0, 20.0 + 1e-9, 2.0)
    fine_offsets = offset + np.arange(-3.0, 3.0 + 1e-9, 0.25)
    corr2, drift2, offset2 = evaluate(fine_drifts, fine_offsets)
    if corr2 > corr:
        corr, drift, offset = corr2, drift2, offset2
    return {"offset_ms": offset, "drift_ppm": drift, "correlation": corr}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else float("nan")


def validate_pair(
    t_nvml_ns: np.ndarray,
    p_nvml: np.ndarray,
    t_sensor_ns: np.ndarray,
    p_sensor: np.ndarray,
    alignment: dict[str, float],
    window_ns: tuple[int, int] | None = None,
    grid_hz: float = 20.0,
    bias_threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Compute MAE, percent bias, Pearson r and Bland-Altman LoA."""
    aligned = correct_sensor_times(
        t_sensor_ns,
        alignment["offset_ms"],
        alignment["drift_ppm"],
        int(np.asarray(t_sensor_ns)[0]),
    )
    t_ref_all = np.asarray(t_nvml_ns) / 1e9
    p_ref_all = np.asarray(p_nvml)
    t_sig_all = aligned / 1e9
    p_sig_all = np.asarray(p_sensor)
    # Full-trace agreement: includes soak and baseline transitions, so it
    # measures whether the streams track at all, not just steady-state noise.
    f0 = max(t_ref_all.min(), t_sig_all.min())
    f1 = min(t_ref_all.max(), t_sig_all.max())
    if f1 > f0:
        grid_full = np.linspace(f0, f1, max(int((f1 - f0) * grid_hz) + 1, 10))
        pearson_full = _pearson(
            np.interp(grid_full, t_sig_all, p_sig_all),
            np.interp(grid_full, t_ref_all, p_ref_all),
        )
    else:
        pearson_full = float("nan")
    if window_ns is not None:
        w0, w1 = window_ns
        ref_mask = (np.asarray(t_nvml_ns) >= w0) & (np.asarray(t_nvml_ns) <= w1)
        sig_mask = (aligned >= w0) & (aligned <= w1)
    else:
        ref_mask = np.ones(len(t_nvml_ns), dtype=bool)
        sig_mask = np.ones(len(aligned), dtype=bool)
    tr, pr = t_ref_all[ref_mask], p_ref_all[ref_mask]
    ts, ps = t_sig_all[sig_mask], p_sig_all[sig_mask]
    if len(tr) < 5 or len(ts) < 5:
        raise AlignmentError(
            f"insufficient samples in validation window: nvml={len(tr)}, sensor={len(ts)}"
        )
    g0, g1 = max(tr[0], ts[0]), min(tr[-1], ts[-1])
    if g1 <= g0:
        raise AlignmentError("streams do not overlap inside the measurement window")
    grid = np.linspace(g0, g1, max(int((g1 - g0) * grid_hz) + 1, 10))
    ref = np.interp(grid, tr, pr)
    sig = np.interp(grid, ts, ps)
    diff = sig - ref
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    mean_ref = float(np.mean(ref))
    bias_pct = (mean_diff / mean_ref * 100.0) if mean_ref else float("nan")
    return {
        "mean_abs_diff_w": float(np.mean(np.abs(diff))),
        "mean_diff_w": mean_diff,
        "percent_bias": float(bias_pct),
        "pearson_r": _pearson(sig, ref),
        "pearson_r_full": pearson_full,
        "loa_lower_w": mean_diff - 1.96 * std_diff,
        "loa_upper_w": mean_diff + 1.96 * std_diff,
        "flagged": bool(abs(bias_pct) > bias_threshold_pct),
        "bias_threshold_pct": bias_threshold_pct,
        "grid_hz": grid_hz,
        "window": "measurement" if window_ns is not None else "overlap",
    }


def load_cell_frames(cell_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a cell's raw CSV and manifest."""
    cell_dir = Path(cell_dir)
    raw_path = cell_dir / "raw.csv"
    manifest_path = cell_dir / "manifest.json"
    if not raw_path.exists() or not manifest_path.exists():
        raise AlignmentError(f"missing raw.csv or manifest.json in {cell_dir}")
    df = pd.read_csv(raw_path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return df, manifest


def align_frame(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    grid_hz: float = 20.0,
    bias_threshold_pct: float = 5.0,
    max_offset_ms: float = 1000.0,
) -> AlignmentResult:
    """Align one cell's streams and validate them over the measurement window."""
    nvml = df[df["source"] == "nvml"].sort_values("timestamp_ns")
    sensor = df[df["source"] == "sensor"].sort_values("timestamp_ns")
    if len(nvml) < 10 or len(sensor) < 10:
        raise AlignmentError(f"cell has too few samples: nvml={len(nvml)}, sensor={len(sensor)}")
    t_ref = nvml["timestamp_ns"].to_numpy(dtype=np.int64)
    p_ref = nvml["power_w"].to_numpy(dtype=np.float64)
    t_s = sensor["timestamp_ns"].to_numpy(dtype=np.int64)
    p_s = sensor["power_w"].to_numpy(dtype=np.float64)
    fit = estimate_offset_drift(t_ref, p_ref, t_s, p_s, grid_hz=grid_hz, max_offset_ms=max_offset_ms)
    regions = manifest.get("regions", {})
    window = None
    if "measure_start" in regions and "measure_end" in regions:
        window = (int(regions["measure_start"]), int(regions["measure_end"]))
    metrics = validate_pair(
        t_ref,
        p_ref,
        t_s,
        p_s,
        fit,
        window_ns=window,
        grid_hz=grid_hz,
        bias_threshold_pct=bias_threshold_pct,
    )
    return AlignmentResult(
        offset_ms=round(fit["offset_ms"], 4),
        drift_ppm=round(fit["drift_ppm"], 3),
        sensor_t0_ns=int(t_s[0]),
        correlation=round(fit["correlation"], 5),
        nvml_samples=len(nvml),
        sensor_samples=len(sensor),
        **metrics,
    )


def align_cell(
    cell_dir: str | Path,
    grid_hz: float = 20.0,
    bias_threshold_pct: float = 5.0,
    max_offset_ms: float = 1000.0,
) -> AlignmentResult:
    """Align a cell from disk (convenience wrapper)."""
    df, manifest = load_cell_frames(cell_dir)
    return align_frame(
        df,
        manifest,
        grid_hz=grid_hz,
        bias_threshold_pct=bias_threshold_pct,
        max_offset_ms=max_offset_ms,
    )


def measurement_arrays(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    alignment: AlignmentResult,
) -> dict[str, np.ndarray]:
    """Return NVML and drift-corrected sensor arrays inside the measurement window."""
    regions = manifest["regions"]
    w0, w1 = int(regions["measure_start"]), int(regions["measure_end"])
    nvml = df[(df["source"] == "nvml") & (df["timestamp_ns"] >= w0) & (df["timestamp_ns"] <= w1)]
    sensor = df[df["source"] == "sensor"]
    aligned = correct_sensor_times(
        sensor["timestamp_ns"].to_numpy(dtype=np.int64),
        alignment.offset_ms,
        alignment.drift_ppm,
        alignment.sensor_t0_ns,
    )
    mask = (aligned >= w0) & (aligned <= w1)
    return {
        "t_nvml_ns": nvml["timestamp_ns"].to_numpy(dtype=np.int64),
        "p_nvml": nvml["power_w"].to_numpy(dtype=np.float64),
        "t_sensor_ns": aligned[mask],
        "p_sensor": sensor["power_w"].to_numpy(dtype=np.float64)[mask],
    }
