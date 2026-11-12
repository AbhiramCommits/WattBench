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
from .schema import SCHEMA_VERSION, write_results
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
    if config_path:
        cfg = load_config(config_path)
        grid_hz = cfg.sampling.align_grid_hz
        bias_threshold = cfg.sampling.bias_threshold_pct
        outlier_z = cfg.sweep.outlier_z

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
            "efficiency": None,
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
