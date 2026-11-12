"""Matplotlib figures for the WattBench report (no seaborn)."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .align import correct_sensor_times
from .schema import load_results

logger = logging.getLogger(__name__)

WORKLOAD_LABELS = {
    "ai_inference": "AI inference",
    "proviz_render": "ProViz render",
    "gaming_proxy": "Gaming proxy",
}
DOMAIN_UNITS = {"ai": "inferences/J", "proviz": "frames/J", "gaming": "frames/J"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _workload_label(name: str) -> str:
    return WORKLOAD_LABELS.get(name, name)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _unlocked_cells(cells: list[Any]) -> list[Any]:
    return [c for c in cells if c.clock_label == "unlocked"] or list(cells)


def plot_perf_vs_power(doc: Any, out_dir: Path) -> Path:
    """Throughput vs power limit for every workload (unlocked + locked)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cells = doc.cells
    workloads = sorted({c.workload for c in cells})
    for ax, wl in zip(axes, workloads, strict=False):
        for j, clock in enumerate(sorted({c.clock_label for c in cells if c.workload == wl})):
            group = sorted(
                [c for c in cells if c.workload == wl and c.clock_label == clock],
                key=lambda c: c.power_limit_w,
            )
            x = [c.power_limit_w for c in group]
            stat = [c.aggregates["throughput_per_s"] for c in group]
            y = [s.mean or 0.0 for s in stat]
            err = [
                max(0.0, (s.ci95_high or s.mean or 0.0) - (s.mean or 0.0)) for s in stat
            ]
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=COLORS[j % len(COLORS)],
                label=clock,
            )
        ax.set_title(_workload_label(wl))
        ax.set_xlabel("Power limit (W)")
        ax.set_ylabel("Throughput (units/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Throughput vs board power limit", fontsize=13)
    return _save(fig, out_dir / "perf_vs_power_limit.png")


def plot_perf_per_watt(doc: Any, out_dir: Path) -> Path:
    """Perf/W curve per workload with max-efficiency and knee markers."""
    workloads = sorted({c.workload for c in doc.cells})
    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4.2), squeeze=False)
    for ax, wl in zip(axes[0], workloads, strict=False):
        group = sorted(
            [c for c in _unlocked_cells(doc.cells) if c.workload == wl],
            key=lambda c: c.power_limit_w,
        )
        x = [c.power_limit_w for c in group]
        y = [c.efficiency.perf_per_watt_nvml if c.efficiency else 0.0 for c in group]
        ax.plot(x, y, marker="o", color="#1f77b4", linewidth=1.8)
        if group and group[0].knee:
            k = group[0].knee
            ax.axvline(k.power_limit_w, color="#d62728", linestyle="--", alpha=0.7)
            ax.scatter(
                [k.power_limit_w],
                [k.perf_per_watt],
                color="#d62728",
                zorder=5,
                label=f"knee {k.power_limit_w:.0f} W",
            )
            ax.scatter(
                [k.max_efficiency_power_limit_w],
                [max(y) if y else 0],
                marker="*",
                s=120,
                color="#2ca02c",
                zorder=5,
                label=f"max perf/W {k.max_efficiency_power_limit_w:.0f} W",
            )
            ax.legend(fontsize=8)
        ax.set_title(_workload_label(wl))
        ax.set_xlabel("Power limit (W)")
        ax.set_ylabel("Performance per watt (units/s/W)")
        ax.grid(alpha=0.3)
    fig.suptitle("Performance per watt vs power limit (NVML power)", fontsize=13)
    return _save(fig, out_dir / "perf_per_watt.png")


def _measurement_samples(run_dir: Path, doc: Any) -> tuple[np.ndarray, np.ndarray]:
    nv_all: list[float] = []
    se_all: list[float] = []
    for cell in doc.cells:
        for rep in cell.per_repeat[:2]:
            cell_dir = run_dir / "cells" / rep.cell_id
            raw = cell_dir / "raw.csv"
            manifest = cell_dir / "manifest.json"
            if not raw.exists() or not manifest.exists():
                continue
            m = json.loads(manifest.read_text(encoding="utf-8"))
            w0, w1 = int(m["regions"]["measure_start"]), int(m["regions"]["measure_end"])
            df = pd.read_csv(raw)
            nvml = df[(df["source"] == "nvml") & (df["timestamp_ns"] >= w0) & (df["timestamp_ns"] <= w1)]
            sensor = df[df["source"] == "sensor"]
            aligned = correct_sensor_times(
                sensor["timestamp_ns"].to_numpy(dtype=np.int64),
                rep.alignment.offset_ms,
                rep.alignment.drift_ppm,
                rep.alignment.sensor_t0_ns or int(sensor["timestamp_ns"].iloc[0]),
            )
            mask = (aligned >= w0) & (aligned <= w1)
            if len(nvml) == 0 or not mask.any():
                continue
            grid = np.linspace(w0, w1, 60)
            nv_all.extend(np.interp(grid, nvml["timestamp_ns"].to_numpy(), nvml["power_w"].to_numpy()))
            se_all.extend(
                np.interp(grid, aligned[mask], sensor["power_w"].to_numpy()[mask])
            )
    return np.asarray(nv_all), np.asarray(se_all)


def plot_power_scatter(run_dir: Path, doc: Any, out_dir: Path) -> Path:
    """NVML vs external sensor, with identity and fitted agreement lines."""
    ref, sig = _measurement_samples(run_dir, doc)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(ref, sig, s=7, alpha=0.35, color="#1f77b4", label="measurement samples")
    if len(ref) > 2:
        lo = float(min(ref.min(), sig.min()))
        hi = float(max(ref.max(), sig.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#444444", label="identity")
        slope, intercept = np.polyfit(ref, sig, 1)
        xs = np.array([lo, hi])
        ax.plot(xs, slope * xs + intercept, color="#d62728", linewidth=1.8,
                label=f"fit: {slope:.3f}x {intercept:+.2f} W")
        ax.text(
            0.03,
            0.97,
            f"n = {len(ref)}\nMAE = {doc.alignment.mean_abs_diff_w:.2f} W\n"
            f"bias = {doc.alignment.percent_bias:+.2f}%\n"
            f"r (full trace) = {doc.alignment.pearson_r_full or 0:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
    ax.set_xlabel("NVML board power (W)")
    ax.set_ylabel("External INA219 power (W)")
    ax.set_title("Dual-source power agreement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_dir / "nvml_vs_external.png")


def plot_time_series(run_dir: Path, doc: Any, out_dir: Path) -> tuple[Path, str]:
    """One cell's power trace with soak, baseline and measurement shaded."""
    cell = next(
        (
            c
            for c in doc.cells
            if c.workload == "ai_inference" and c.clock_label == "unlocked" and c.power_limit_w == 190
        ),
        doc.cells[0],
    )
    rep = cell.per_repeat[0]
    cell_dir = run_dir / "cells" / rep.cell_id
    df = pd.read_csv(cell_dir / "raw.csv")
    manifest = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
    regions = manifest["regions"]
    t0 = int(regions["soak_start"])
    nvml = df[df["source"] == "nvml"]
    sensor = df[df["source"] == "sensor"]
    aligned = correct_sensor_times(
        sensor["timestamp_ns"].to_numpy(dtype=np.int64),
        rep.alignment.offset_ms,
        rep.alignment.drift_ppm,
        rep.alignment.sensor_t0_ns or int(sensor["timestamp_ns"].iloc[0]),
    )
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(
        (nvml["timestamp_ns"] - t0) / 1e9,
        nvml["power_w"],
        color="#1f77b4",
        linewidth=1.2,
        label="NVML board power",
    )
    ax.plot(
        (aligned - t0) / 1e9,
        sensor["power_w"],
        color="#ff7f0e",
        linewidth=0.9,
        alpha=0.8,
        label="External INA219 (aligned)",
    )
    spans = [
        ("soak", regions["soak_start"], regions["soak_end"], "#f2c14e"),
        ("baseline", regions["baseline_start"], regions["baseline_end"], "#8ecae6"),
        ("measurement", regions["measure_start"], regions["measure_end"], "#90be6d"),
    ]
    for name, a, b, color in spans:
        ax.axvspan((a - t0) / 1e9, (b - t0) / 1e9, color=color, alpha=0.18, label=name)
        ax.text(
            ((a + b) / 2 - t0) / 1e9,
            ax.get_ylim()[1],
            name,
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlabel("Seconds from cell start")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"{_workload_label(cell.workload)} at {cell.power_limit_w:.0f} W — "
                 f"{cell.clock_label}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    name = f"timeseries_{cell.workload}_{cell.power_limit_w:.0f}W.png"
    return _save(fig, out_dir / name), name


def plot_domain_efficiency(doc: Any, out_dir: Path) -> Path:
    """Best per-domain work per joule, both sources, with repeat error bars."""
    domains = sorted({c.domain for c in doc.cells})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(domains))
    width = 0.36
    for k, (source, color) in enumerate([("nvml", "#1f77b4"), ("sensor", "#ff7f0e")]):
        heights = []
        errors = []
        labels = []
        for domain in domains:
            group = [c for c in _unlocked_cells(doc.cells) if c.domain == domain]
            key = f"work_per_joule_{source}"
            best = max(
                group,
                key=lambda c: float((c.efficiency.model_dump().get(key) if c.efficiency else 0) or 0),
            )
            val = float(getattr(best.efficiency, key) or 0) if best.efficiency else 0.0
            reps = [
                float(getattr(r.efficiency, key) or 0)
                for r in best.per_repeat
                if r.efficiency is not None
            ]
            heights.append(val)
            errors.append(float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0)
            labels.append(DOMAIN_UNITS.get(domain, "units/J"))
        ax.bar(x + (k - 0.5) * width, heights, width, yerr=errors, capsize=4,
               color=color, alpha=0.85, label=f"{source} power")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}\n({DOMAIN_UNITS.get(d, 'units/J')})" for d in domains], fontsize=9
    )
    ax.set_ylabel("Peak work per joule (best configuration)")
    ax.set_title("Energy efficiency by domain (error bars = repeat stdev)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir / "domain_efficiency.png")


def plot_all(run_dir: str | Path, doc: Any | None = None, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Generate every report figure; returns a name -> path map."""
    run_path = Path(run_dir)
    if doc is None:
        doc = load_results(_results_path(run_path))
    out = Path(out_dir) if out_dir else run_path / "results" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    created = {
        "perf": plot_perf_vs_power(doc, out),
        "efficiency": plot_perf_per_watt(doc, out),
        "scatter": plot_power_scatter(run_path, doc, out),
        "bars": plot_domain_efficiency(doc, out),
    }
    ts_path, ts_name = plot_time_series(run_path, doc, out)
    created["timeseries"] = ts_path
    created["timeseries_name"] = Path(ts_name).name
    logger.info("wrote %d plots to %s", len(created) - 1, out)
    return created


def _results_path(run_dir: Path) -> Path:
    for cand in (run_dir / "results" / "results.json", run_dir / "results.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no results.json under {run_dir}; run 'wattbench analyze' first")


def image_data_uri(path: Path) -> str:
    """Base64 data URI for a self-contained HTML report."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
