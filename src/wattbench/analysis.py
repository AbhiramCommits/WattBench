"""Run-level analysis: align every cell, aggregate repeats, validate, persist."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .align import (
    AlignmentError,
    AlignmentResult,
    align_frame,
    correct_sensor_times,
    measurement_arrays,
)
from .config import load_config
from .schema import SCHEMA_VERSION, load_results, write_results
from .stats import aggregate_cell

logger = logging.getLogger(__name__)


class AnalysisError(RuntimeError):
    """Raised when a run cannot be analysed."""


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        raise AnalysisError(f"{run_dir} is not a run directory (missing run.json)")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_cells(run_dir: Path) -> list[Path]:
    cells_dir = run_dir / "cells"
    if not cells_dir.exists():
        raise AnalysisError(f"no cells/ directory under {run_dir}")
    dirs = []
    for path in sorted(cells_dir.iterdir()):
        if not path.is_dir():
            continue
        manifest = path / "manifest.json"
        raw = path / "raw.csv"
        if not manifest.exists() or not raw.exists():
            logger.warning("skipping %s: missing manifest or raw data", path.name)
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping %s: corrupt manifest", path.name)
            continue
        if data.get("status") != "complete":
            logger.warning("skipping %s: status=%s", path.name, data.get("status"))
            continue
        dirs.append(path)
    if not dirs:
        raise AnalysisError(f"no complete cells found under {cells_dir}")
    return dirs


def _window_mean_nvml(df: pd.DataFrame, start: int, end: int, column: str) -> float:
    rows = df[(df["source"] == "nvml") & (df["timestamp_ns"] >= start) & (df["timestamp_ns"] <= end)]
    if rows.empty:
        return float("nan")
    return float(np.nanmean(rows[column].to_numpy(dtype=np.float64)))


def _baseline_means(
    df: pd.DataFrame, manifest: dict[str, Any], alignment: AlignmentResult
) -> tuple[float | None, float | None]:
    regions = manifest.get("regions", {})
    if "baseline_start" not in regions:
        return None, None
    b0, b1 = int(regions["baseline_start"]), int(regions["baseline_end"])
    nvml = df[(df["source"] == "nvml") & (df["timestamp_ns"] >= b0) & (df["timestamp_ns"] <= b1)]
    nvml_mean = float(np.nanmean(nvml["power_w"])) if not nvml.empty else None
    sensor = df[df["source"] == "sensor"]
    aligned = correct_sensor_times(
        sensor["timestamp_ns"].to_numpy(dtype=np.int64),
        alignment.offset_ms,
        alignment.drift_ppm,
        alignment.sensor_t0_ns,
    )
    mask = (aligned >= b0) & (aligned <= b1)
    sensor_mean = float(np.nanmean(sensor["power_w"].to_numpy()[mask])) if mask.any() else None
    return nvml_mean, sensor_mean


def _measurement_stats(
    df: pd.DataFrame, manifest: dict[str, Any], alignment: AlignmentResult
) -> dict[str, float]:
    arr = measurement_arrays(df, manifest, alignment)
    if len(arr["p_nvml"]) == 0 or len(arr["p_sensor"]) == 0:
        raise AnalysisError(f"empty measurement window for {manifest['cell']['cell_id']}")
    regions = manifest["regions"]
    w0, w1 = int(regions["measure_start"]), int(regions["measure_end"])
    nvml = df[
        (df["source"] == "nvml")
        & (df["timestamp_ns"] >= w0)
        & (df["timestamp_ns"] <= w1)
    ]
    return {
        "nvml_power_w": float(np.nanmean(arr["p_nvml"])),
        "sensor_power_w": float(np.nanmean(arr["p_sensor"])),
        "sm_clock_mhz": float(np.nanmean(nvml["sm_clock_mhz"].to_numpy(dtype=np.float64))),
        "mem_clock_mhz": float(np.nanmean(nvml["mem_clock_mhz"].to_numpy(dtype=np.float64))),
        "temp_c": float(np.nanmean(nvml["temp_c"].to_numpy(dtype=np.float64))),
        "util_pct": float(np.nanmean(nvml["util_pct"].to_numpy(dtype=np.float64))),
    }


def _trapz_energy_j(t_ns: np.ndarray, p_w: np.ndarray) -> float:
    """Trapezoidal energy over a sampled power trace, in joules."""
    if len(t_ns) < 2:
        return 0.0
    t = (np.asarray(t_ns, dtype=np.float64) - float(t_ns[0])) / 1e9
    return float(np.trapezoid(np.asarray(p_w, dtype=np.float64), t))


def compute_efficiency(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    alignment: AlignmentResult,
    nvml_idle_w: float | None,
    sensor_idle_w: float | None,
) -> dict[str, Any]:
    """Work per joule and performance per watt from both power sources.

    Energy is integrated with the trapezoid rule over the *measured* trace,
    not inferred from a setpoint.  Idle-subtracted values are reported
    alongside the raw ones so both conventions are auditable.
    """
    arr = measurement_arrays(df, manifest, alignment)
    result = manifest.get("result", {})
    work = float(result.get("work_units", 0))
    throughput = float(result.get("throughput_per_s", 0.0))
    duration = float(manifest.get("regions", {}).get("measure_end", 0) - manifest.get("regions", {}).get("measure_start", 0)) / 1e9
    if duration <= 0:
        duration = float(result.get("wall_s", 0.0))
    e_n = _trapz_energy_j(arr["t_nvml_ns"], arr["p_nvml"])
    e_s = _trapz_energy_j(arr["t_sensor_ns"], arr["p_sensor"])
    mean_n = float(np.nanmean(arr["p_nvml"]))
    mean_s = float(np.nanmean(arr["p_sensor"]))
    eff_sub_n = max(e_n - (nvml_idle_w or 0.0) * duration, 0.0) if nvml_idle_w else None
    eff_sub_s = max(e_s - (sensor_idle_w or 0.0) * duration, 0.0) if sensor_idle_w else None
    return {
        "work_units": int(work),
        "unit_name": str(result.get("unit_name", "units")),
        "energy_j_nvml": round(e_n, 6),
        "energy_j_sensor": round(e_s, 6),
        "work_per_joule_nvml": round(work / e_n, 8) if e_n > 0 else 0.0,
        "work_per_joule_sensor": round(work / e_s, 8) if e_s > 0 else 0.0,
        "perf_per_watt_nvml": round(throughput / mean_n, 8) if mean_n > 0 else 0.0,
        "perf_per_watt_sensor": round(throughput / mean_s, 8) if mean_s > 0 else 0.0,
        "idle_subtracted_energy_j_nvml": None if eff_sub_n is None else round(eff_sub_n, 6),
        "idle_subtracted_energy_j_sensor": None if eff_sub_s is None else round(eff_sub_s, 6),
        "idle_subtracted_work_per_joule_nvml": (
            round(work / eff_sub_n, 8) if eff_sub_n and eff_sub_n > 0 else None
        ),
        "idle_subtracted_work_per_joule_sensor": (
            round(work / eff_sub_s, 8) if eff_sub_s and eff_sub_s > 0 else None
        ),
        "idle_power_nvml_w": nvml_idle_w,
        "idle_power_sensor_w": sensor_idle_w,
        "mean_power_nvml_w": round(mean_n, 6),
        "mean_power_sensor_w": round(mean_s, 6),
        "duration_s": round(duration, 6),
    }


def knee(cells: list[dict[str, Any]], marginal_fraction: float = 0.9) -> dict[str, Any]:
    """Fit throughput vs power-limit and locate the efficiency knee.

    A quadratic is fit to the measured (power-limit, throughput) points for a
    single workload/clock family.  The marginal return dT/dP is evaluated at
    each tested point; the knee is the first point where it falls below
    ``marginal_fraction`` of the peak marginal return.  The max perf/W point
    is reported separately because it is usually at a lower power limit.
    """
    points = sorted(cells, key=lambda c: float(c["power_limit_w"]))
    pts = [c for c in points if c["aggregates"].get("throughput_per_s", {}).get("mean") is not None]
    if len(pts) < 3:
        raise ValueError("knee analysis requires at least three power-limit points")
    x = np.array([float(c["power_limit_w"]) for c in pts])
    y = np.array([float(c["aggregates"]["throughput_per_s"]["mean"]) for c in pts])
    p_meas = np.array([float(c["aggregates"]["nvml_power_w"]["mean"]) for c in pts])
    perf_w = y / np.maximum(p_meas, 1e-9)
    coef = np.polyfit(x, y, 2)
    deriv = np.gradient(np.polyval(coef, x), x)
    peak_deriv = float(np.max(deriv))
    if not np.isfinite(peak_deriv) or peak_deriv <= 0:
        deriv = np.gradient(y, x)
        peak_deriv = float(np.max(deriv))
    threshold = marginal_fraction * peak_deriv
    knee_idx = 0
    for i in range(1, len(pts)):
        if deriv[i] < threshold:
            knee_idx = i
            break
    else:
        knee_idx = int(np.argmax(y))
    peak_idx = int(np.argmax(y))
    best_idx = int(np.argmax(perf_w))
    method = (
        f"quadratic fit T(P); first point with marginal dT/dP below "
        f"{marginal_fraction:.2f} x peak marginal"
    )
    peak_throughput = float(y[peak_idx])
    peak_plimit = float(x[peak_idx])
    return {
        "workload": cells[0]["workload"],
        "clock_label": cells[0]["clock_label"],
        "method": method,
        "power_limit_w": float(x[knee_idx]),
        "power_w": float(p_meas[knee_idx]),
        "throughput_per_s": float(y[knee_idx]),
        "perf_per_watt": float(perf_w[knee_idx]),
        "peak_power_limit_w": peak_plimit,
        "peak_power_w": float(p_meas[peak_idx]),
        "peak_throughput_per_s": peak_throughput,
        "peak_perf_per_watt": float(perf_w[peak_idx]),
        "max_efficiency_power_limit_w": float(x[best_idx]),
        "max_efficiency_power_w": float(p_meas[best_idx]),
        "max_efficiency_throughput_pct": round(100.0 * y[best_idx] / peak_throughput, 3),
        "throughput_pct_of_peak": round(100.0 * y[knee_idx] / peak_throughput, 3),
        "power_pct_of_peak": round(100.0 * x[knee_idx] / peak_plimit, 3),
        "marginal_fraction": marginal_fraction,
    }


def _resolve_results(path: str | Path) -> tuple[Path, Any]:
    p = Path(path)
    candidates = [p / "results.json", p / "results" / "results.json", p]
    for cand in candidates:
        if cand.is_file():
            return cand, load_results(cand)
    raise AnalysisError(f"no results.json found under {p} (run 'wattbench analyze' first)")


def regression_compare(
    baseline: str | Path,
    candidate: str | Path,
    z: float = 3.0,
    rel_tol: float = 0.03,
) -> dict[str, Any]:
    """Compare per-cell throughput, with a tolerance from measured variance.

    The tolerance is ``max(z * combined_sem, rel_tol * |baseline mean|)`` where
    the standard error of the difference comes from each sweep's own repeat
    stdev.  The relative floor prevents physically meaningless failures when
    the repeat noise rounds to zero (e.g. a locked-clock synthetic cell).
    """
    baseline_path, base = _resolve_results(baseline)
    candidate_path, cand = _resolve_results(candidate)
    base_cells = {c.cell_id: c for c in base.cells}
    cand_cells = {c.cell_id: c for c in cand.cells}
    rows: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    for cell_id in sorted(set(base_cells) & set(cand_cells)):
        b = base_cells[cell_id]
        c = cand_cells[cell_id]
        bstat = b.aggregates.get("throughput_per_s")
        cstat = c.aggregates.get("throughput_per_s")
        if bstat is None or cstat is None or bstat.mean is None or cstat.mean is None:
            continue
        b_mean, c_mean = float(bstat.mean), float(cstat.mean)
        b_sd = float(bstat.stdev or 0.0)
        c_sd = float(cstat.stdev or 0.0)
        b_n = max(int(bstat.n), 1)
        c_n = max(int(cstat.n), 1)
        combined_sem = float(np.sqrt(b_sd**2 / b_n + c_sd**2 / c_n))
        tol = max(z * combined_sem, rel_tol * abs(b_mean))
        delta = c_mean - b_mean
        delta_pct = 100.0 * delta / b_mean if b_mean else float("nan")
        passed = abs(delta) <= tol
        b_pwr = b.aggregates.get("nvml_power_w")
        c_pwr = c.aggregates.get("nvml_power_w")
        row = {
            "cell_id": cell_id,
            "baseline_throughput": round(b_mean, 6),
            "candidate_throughput": round(c_mean, 6),
            "delta": round(delta, 6),
            "delta_pct": round(delta_pct, 4),
            "tolerance": round(tol, 6),
            "stdev_baseline": round(b_sd, 6),
            "stdev_candidate": round(c_sd, 6),
            "power_delta_w": (
                round(float(c_pwr.mean) - float(b_pwr.mean), 6)
                if b_pwr and c_pwr and b_pwr.mean is not None and c_pwr.mean is not None
                else None
            ),
            "verdict": "PASS" if passed else "FAIL",
        }
        rows.append(row)
        if not passed and (worst is None or abs(delta) > abs(worst["delta"])):
            worst = row
    missing = sorted(set(base_cells) - set(cand_cells))
    overall = "PASS" if rows and all(r["verdict"] == "PASS" for r in rows) and not missing else "FAIL"
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_source": base.run.source,
        "candidate_source": cand.run.source,
        "z": z,
        "rel_tol": rel_tol,
        "cells_compared": len(rows),
        "cells_missing": missing,
        "failures": [r for r in rows if r["verdict"] == "FAIL"],
        "worst": worst,
        "overall": overall,
        "rows": rows,
    }


def _group_key(manifest: dict[str, Any]) -> tuple[str, float, str, int, int]:
    cell = manifest["cell"]
    return (
        cell["workload"],
        float(cell["power_limit_w"]),
        cell["clock_label"],
        int(cell["clock_min_mhz"]),
        int(cell["clock_max_mhz"]),
    )


def _base_cell_id(manifest: dict[str, Any]) -> str:
    cell = manifest["cell"]
    label = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in str(cell["clock_label"])
    )
    return f"{cell['workload']}__pl{float(cell['power_limit_w']):03.0f}__clk-{label}"


def analyze_run(
    run_dir: str | Path,
    out: str | Path | None = None,
    config_path: str | Path | None = None,
    write_mirror: bool = True,
) -> int:
    """Align, aggregate, validate, and write ``results.json`` + ``cells.csv``."""
    run_path = Path(run_dir)
    run_meta = load_run_metadata(run_path)
    provenance = run_meta.get("provenance", {})
    cfg_dict = provenance.get("config") or {}
    sampling = cfg_dict.get("sampling", {})
    sweep_cfg = cfg_dict.get("sweep", {})
    grid_hz = float(sampling.get("align_grid_hz", 20.0))
    bias_threshold = float(sampling.get("bias_threshold_pct", 5.0))
    outlier_z = float(sweep_cfg.get("outlier_z", 3.5))
    analysis_cfg = cfg_dict.get("analysis", {})
    knee_fraction = float(analysis_cfg.get("knee_marginal_fraction", 0.9))
    if config_path:
        cfg = load_config(config_path)
        grid_hz = cfg.sampling.align_grid_hz
        bias_threshold = cfg.sampling.bias_threshold_pct
        outlier_z = cfg.sweep.outlier_z
        knee_fraction = cfg.analysis.knee_marginal_fraction

    cell_dirs = discover_cells(run_path)
    logger.info("analyzing %d repeat cells from %s", len(cell_dirs), run_path)
    groups: dict[tuple[str, float, str, int, int], list[tuple[Path, dict[str, Any]]]] = {}
    for cell_dir in cell_dirs:
        manifest = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
        groups.setdefault(_group_key(manifest), []).append((cell_dir, manifest))

    cells: list[dict[str, Any]] = []
    all_alignments: list[AlignmentResult] = []
    per_cell_align: dict[str, dict[str, Any]] = {}
    flagged_cells: set[str] = set()

    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        entries = sorted(groups[key], key=lambda e: int(e[1]["cell"]["repeat_idx"]))
        repeats: list[dict[str, Any]] = []
        alignments: list[AlignmentResult] = []
        for cell_dir, manifest in entries:
            try:
                df = pd.read_csv(cell_dir / "raw.csv")
                alignment = align_frame(
                    df,
                    manifest,
                    grid_hz=grid_hz,
                    bias_threshold_pct=bias_threshold,
                )
            except (AlignmentError, OSError, ValueError) as exc:
                logger.error("alignment failed for %s: %s", cell_dir.name, exc)
                continue
            mstats = _measurement_stats(df, manifest, alignment)
            nvml_base, sensor_base = _baseline_means(df, manifest, alignment)
            eff = compute_efficiency(df, manifest, alignment, nvml_base, sensor_base)
            result = manifest.get("result", {})
            record = {
                "repeat_idx": int(manifest["cell"]["repeat_idx"]),
                "cell_id": manifest["cell"]["cell_id"],
                "work_units": int(result.get("work_units", 0)),
                "wall_s": float(result.get("wall_s", 0.0)),
                "throughput_per_s": float(result.get("throughput_per_s", 0.0)),
                **mstats,
                "nvml_baseline_w": nvml_base,
                "sensor_baseline_w": sensor_base,
                "raw_csv": str((cell_dir / "raw.csv").relative_to(run_path)),
                "efficiency": eff,
                "alignment": alignment.to_dict(),
            }
            repeats.append(record)
            alignments.append(alignment)
            all_alignments.append(alignment)
            if alignment.flagged:
                flagged_cells.add(_base_cell_id(manifest))

        if not repeats:
            logger.error("no usable repeats for %s", key)
            continue
        agg = aggregate_cell(repeats, z_threshold=outlier_z)
        kept_ids = {r["cell_id"] for r in repeats} - {
            o["cell_id"] for o in agg["rejected_outliers"]
        }
        eff_records = [r["efficiency"] for r in repeats if r["cell_id"] in kept_ids and r.get("efficiency")]
        cell_eff = None
        if eff_records:
            cell_eff = {
                "work_units": round(float(np.mean([e["work_units"] for e in eff_records]))),
                "unit_name": eff_records[0]["unit_name"],
                **{
                    k: (
                        round(float(np.mean([e[k] for e in eff_records if e.get(k) is not None])), 8)
                        if any(e.get(k) is not None for e in eff_records)
                        else None
                    )
                    for k in (
                        "energy_j_nvml",
                        "energy_j_sensor",
                        "work_per_joule_nvml",
                        "work_per_joule_sensor",
                        "perf_per_watt_nvml",
                        "perf_per_watt_sensor",
                        "idle_subtracted_energy_j_nvml",
                        "idle_subtracted_energy_j_sensor",
                        "idle_subtracted_work_per_joule_nvml",
                        "idle_subtracted_work_per_joule_sensor",
                        "idle_power_nvml_w",
                        "idle_power_sensor_w",
                        "mean_power_nvml_w",
                        "mean_power_sensor_w",
                        "duration_s",
                    )
                },
            }
        cell = {
            "cell_id": _base_cell_id(entries[0][1]),
            "workload": entries[0][1]["cell"]["workload"],
            "domain": entries[0][1]["cell"]["domain"],
            "power_limit_w": float(entries[0][1]["cell"]["power_limit_w"]),
            "clock_label": entries[0][1]["cell"]["clock_label"],
            "clock_min_mhz": int(entries[0][1]["cell"]["clock_min_mhz"]),
            "clock_max_mhz": int(entries[0][1]["cell"]["clock_max_mhz"]),
            "source": run_meta.get("source", "unknown"),
            "repeats_total": agg["repeats_total"],
            "repeats_used": agg["repeats_used"],
            "rejected_outliers": agg["rejected_outliers"],
            "aggregates": agg["aggregates"],
            "per_repeat": repeats,
            "efficiency": cell_eff,
            "knee": None,
        }
        cells.append(cell)
        align_dict = {
            "offset_ms": float(np.mean([a.offset_ms for a in alignments])),
            "drift_ppm": float(np.mean([a.drift_ppm for a in alignments])),
            "correlation": float(np.nanmean([a.correlation for a in alignments])),
            "nvml_samples": int(np.mean([a.nvml_samples for a in alignments])),
            "sensor_samples": int(np.mean([a.sensor_samples for a in alignments])),
            "mean_abs_diff_w": float(np.mean([a.mean_abs_diff_w for a in alignments])),
            "mean_diff_w": float(np.mean([a.mean_diff_w for a in alignments])),
            "percent_bias": float(np.mean([a.percent_bias for a in alignments])),
            "pearson_r": float(np.nanmean([a.pearson_r for a in alignments])),
            "pearson_r_full": float(np.nanmean([a.pearson_r_full for a in alignments])),
            "loa_lower_w": float(np.mean([a.loa_lower_w for a in alignments])),
            "loa_upper_w": float(np.mean([a.loa_upper_w for a in alignments])),
            "flagged": any(a.flagged for a in alignments),
            "bias_threshold_pct": bias_threshold,
            "sensor_t0_ns": int(np.mean([a.sensor_t0_ns for a in alignments])),
            "grid_hz": grid_hz,
            "window": "measurement",
        }
        per_cell_align[cell["cell_id"]] = align_dict

    if not cells:
        raise AnalysisError("analysis produced no cells; every alignment failed")

    knee_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        knee_groups.setdefault((cell["workload"], cell["clock_label"]), []).append(cell)
    knees: list[dict[str, Any]] = []
    for group_key in sorted(knee_groups):
        group = knee_groups[group_key]
        if len(group) < 3:
            continue
        try:
            k = knee(group, marginal_fraction=knee_fraction)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("knee fit failed for %s: %s", group_key, exc)
            continue
        knees.append(k)
        for cell in group:
            cell["knee"] = k

    offset_vals = np.array([a.offset_ms for a in all_alignments])
    drift_vals = np.array([a.drift_ppm for a in all_alignments])
    alignment_summary = {
        "bias_threshold_pct": bias_threshold,
        "offset_ms_mean": float(np.mean(offset_vals)),
        "offset_ms_stdev": float(np.std(offset_vals, ddof=1)) if len(offset_vals) > 1 else 0.0,
        "drift_ppm_mean": float(np.mean(drift_vals)),
        "drift_ppm_stdev": float(np.std(drift_vals, ddof=1)) if len(drift_vals) > 1 else 0.0,
        "mean_abs_diff_w": float(np.mean([a.mean_abs_diff_w for a in all_alignments])),
        "percent_bias": float(np.mean([a.percent_bias for a in all_alignments])),
        "pearson_r": float(np.nanmean([a.pearson_r for a in all_alignments])),
        "pearson_r_full": float(np.nanmean([a.pearson_r_full for a in all_alignments])),
        "loa_lower_w": float(np.mean([a.loa_lower_w for a in all_alignments])),
        "loa_upper_w": float(np.mean([a.loa_upper_w for a in all_alignments])),
        "cells_flagged": sorted(flagged_cells),
        "repeats_flagged": int(sum(1 for a in all_alignments if a.flagged)),
        "per_cell": per_cell_align,
    }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "run": {
            "run_id": run_meta.get("run_id", run_path.name),
            "dir": str(run_path),
            "source": run_meta.get("source", "unknown"),
            "config_hash": run_meta.get("config_hash", provenance.get("config_hash", "")),
            "git_commit": provenance.get("git_commit"),
            "created_utc": run_meta.get("created_utc"),
            "finished_utc": run_meta.get("finished_utc"),
            "elapsed_s": run_meta.get("elapsed_s"),
            "matrix": run_meta.get("matrix", {}),
            "counts": run_meta.get("counts", {}),
        },
        "provenance": provenance,
        "alignment": alignment_summary,
        "cells": cells,
        "analysis": {
            "knee_marginal_fraction": knee_fraction,
            "knees": knees,
            "cv_power_nvml": _cv_summary(cells),
        },
    }

    out_dir = Path(out) if out else Path("results")
    validated = write_results(doc, out_dir / "results.json")
    _write_cells_csv(validated, out_dir / "cells.csv")
    if write_mirror:
        mirror = run_path / "results"
        if mirror.resolve() != out_dir.resolve():
            mirror.mkdir(parents=True, exist_ok=True)
            write_results(doc, mirror / "results.json")
            _write_cells_csv(validated, mirror / "cells.csv")
            logger.info("wrote run-scoped mirror under %s", mirror)

    cv = _cv_summary(cells)
    print(f"analyzed: {len(cells)} cells, {sum(c['repeats_total'] for c in cells)} repeats")
    print(f"schema: {SCHEMA_VERSION} (validated)  -> {out_dir / 'results.json'}")
    print(
        f"alignment: offset {alignment_summary['offset_ms_mean']:.1f} ms, "
        f"drift {alignment_summary['drift_ppm_mean']:.1f} ppm, "
        f"MAE {alignment_summary['mean_abs_diff_w']:.2f} W, "
        f"bias {alignment_summary['percent_bias']:+.2f}%, "
        f"r(measurement) {alignment_summary['pearson_r']:.4f}, "
        f"r(full trace) {alignment_summary['pearson_r_full']:.4f}"
    )
    print(
        f"repeats flagged: {alignment_summary['repeats_flagged']} "
        f"({len(flagged_cells)} cells)  outliers rejected: "
        f"{sum(len(c['rejected_outliers']) for c in cells)}"
    )
    print(
        f"CV power (NVML): min {cv['min']:.2f}%  median {cv['median']:.2f}%  "
        f"max {cv['max']:.2f}%"
    )
    for k in knees:
        print(
            f"knee {k['workload']:14s} {k['clock_label']:12s} "
            f"{k['power_limit_w']:.0f} W: {k['throughput_pct_of_peak']:.1f}% throughput "
            f"at {k['power_pct_of_peak']:.1f}% power limit"
        )
    return 0


def _cv_summary(cells: list[dict[str, Any]]) -> dict[str, float]:
    vals = []
    for c in cells:
        cv = c["aggregates"].get("nvml_power_w", {}).get("cv_pct")
        if cv is not None and np.isfinite(cv):
            vals.append(float(cv))
    if not vals:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def _write_cells_csv(validated: Any, path: Path) -> None:
    rows = []
    for cell in validated.cells:
        agg = cell.aggregates
        align = cell.per_repeat[0].alignment if cell.per_repeat else None
        row = {
            "cell_id": cell.cell_id,
            "workload": cell.workload,
            "domain": cell.domain,
            "power_limit_w": cell.power_limit_w,
            "clock_label": cell.clock_label,
            "repeats_total": cell.repeats_total,
            "repeats_used": cell.repeats_used,
            "rejected_outliers": len(cell.rejected_outliers),
            "throughput_per_s": agg.get("throughput_per_s").mean if agg.get("throughput_per_s") else None,
            "throughput_cv_pct": agg.get("throughput_per_s").cv_pct if agg.get("throughput_per_s") else None,
            "nvml_power_w": agg.get("nvml_power_w").mean if agg.get("nvml_power_w") else None,
            "nvml_power_cv_pct": agg.get("nvml_power_w").cv_pct if agg.get("nvml_power_w") else None,
            "sensor_power_w": agg.get("sensor_power_w").mean if agg.get("sensor_power_w") else None,
            "sensor_power_cv_pct": agg.get("sensor_power_w").cv_pct if agg.get("sensor_power_w") else None,
            "sm_clock_mhz": agg.get("sm_clock_mhz").mean if agg.get("sm_clock_mhz") else None,
            "temp_c": agg.get("temp_c").mean if agg.get("temp_c") else None,
            "offset_ms": align.offset_ms if align else None,
            "drift_ppm": align.drift_ppm if align else None,
            "percent_bias": align.percent_bias if align else None,
            "pearson_r": align.pearson_r if align else None,
            "flagged": align.flagged if align else None,
            "work_per_joule_nvml": cell.efficiency.work_per_joule_nvml if cell.efficiency else None,
            "work_per_joule_sensor": cell.efficiency.work_per_joule_sensor if cell.efficiency else None,
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["workload", "clock_label", "power_limit_w"])
    df.to_csv(path, index=False, float_format="%.6f")


def compare_runs(
    baseline: str | Path,
    candidate: str | Path,
    rel_tol: float | None = None,
) -> int:
    """CLI adapter: print the regression table and return 1 on any FAIL."""
    _, cand = _resolve_results(candidate)
    cfg = cand.provenance.get("config", {}).get("analysis", {})
    result = regression_compare(
        baseline,
        candidate,
        z=float(cfg.get("regression_z", 3.0)),
        rel_tol=float(rel_tol if rel_tol is not None else cfg.get("regression_rel_tol", 0.03)),
    )
    print(
        f"baseline:  {result['baseline']} ({result['baseline_source']})\n"
        f"candidate: {result['candidate']} ({result['candidate_source']})\n"
        f"tolerance: max(z={result['z']:.1f} * combined SEM, "
        f"{100 * result['rel_tol']:.1f}% of baseline)\n"
    )
    header = f"{'cell':54s} {'base':>10s} {'cand':>10s} {'delta':>9s} {'tol':>8s}  verdict"
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        print(
            f"{row['cell_id'][:54]:54s} {row['baseline_throughput']:10.3f} "
            f"{row['candidate_throughput']:10.3f} {row['delta']:+9.3f} "
            f"{row['tolerance']:8.3f}  {row['verdict']}"
        )
    if result["cells_missing"]:
        print(f"\nmissing cells: {', '.join(result['cells_missing'])}")
    if result["failures"]:
        print(f"\n{len(result['failures'])} failing cell(s); worst:")
        worst = result["worst"]
        if worst:
            print(
                f"  {worst['cell_id']}: {worst['delta']:+.3f} "
                f"({worst['delta_pct']:+.2f}%) exceeds {worst['tolerance']:.3f}"
            )
    print(f"\noverall: {result['overall']} ({result['cells_compared']} cells compared)")
    return 0 if result["overall"] == "PASS" else 1
