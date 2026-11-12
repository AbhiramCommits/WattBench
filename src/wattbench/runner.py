"""Sweeping test runner: soak, baseline, measure, and persist raw dual-source data.

A cell is the atomic unit: one workload x one power limit x one clock setting x
one repeat.  Every cell writes its own raw CSV and manifest, which makes the
sweep resumable and keeps provenance attached to the measurements themselves.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import RigConfig
from .devices.factory import get_gpu, get_sensor
from .devices.gpu import GpuDevice, SyntheticGpu
from .devices.power_sensor import PowerSensor
from .workloads import available_workloads, get_workload

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class SweepError(RuntimeError):
    """Raised when a sweep cannot start or complete safely."""


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {"wattbench": __version__, "python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "matplotlib", "pynvml", "pyserial", "pydantic", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _top_processes(limit: int = 5) -> list[dict[str, Any]]:
    """Top background processes by CPU, captured at sweep start."""
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "pcpu=,comm="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        rows: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            try:
                cpu = float(parts[0])
            except ValueError:
                continue
            rows.append({"cpu_pct": cpu, "command": parts[1][:120]})
        rows.sort(key=lambda r: r["cpu_pct"], reverse=True)
        return rows[:limit]
    except Exception:
        return []


def collect_provenance(cfg: RigConfig, gpu: GpuDevice, sensor: PowerSensor) -> dict[str, Any]:
    """Capture everything needed to reproduce or audit a measurement."""
    gpu_info = gpu.info()
    sensor_info = sensor.info() if hasattr(sensor, "info") else {}
    source = (
        "hardware"
        if gpu_info.get("source") == "hardware" and sensor_info.get("source") == "hardware"
        else "synthetic"
    )
    return {
        "source": source,
        "sources": {
            "gpu": gpu_info.get("source", "unknown"),
            "sensor": sensor_info.get("source", "unknown"),
        },
        "gpu": gpu_info,
        "sensor": sensor_info,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "versions": _package_versions(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_hash": cfg.config_hash(),
        "config": cfg.to_dict(),
        "sensor_port": cfg.sensor.port,
        "shunt_ohms": cfg.sensor.shunt_ohms,
        "background_processes_top_cpu": _top_processes(),
        "captured_utc": utc_now_iso(),
    }


@dataclass(frozen=True)
class Cell:
    """One point in the sweep matrix."""

    workload: str
    domain: str
    power_limit_w: float
    clock_min_mhz: int
    clock_max_mhz: int
    clock_label: str
    repeat_idx: int

    @property
    def cell_id(self) -> str:
        """Stable identifier for this matrix cell."""
        label = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.clock_label)
        return f"{self.workload}__pl{self.power_limit_w:03.0f}__clk-{label}__r{self.repeat_idx}"

    def as_dict(self) -> dict[str, Any]:
        """Return the cell definition as a plain dictionary."""
        return {
            "cell_id": self.cell_id,
            "workload": self.workload,
            "domain": self.domain,
            "power_limit_w": self.power_limit_w,
            "clock_min_mhz": self.clock_min_mhz,
            "clock_max_mhz": self.clock_max_mhz,
            "clock_label": self.clock_label,
            "repeat_idx": self.repeat_idx,
        }


def build_cells(cfg: RigConfig, seed: int | None = None) -> list[Cell]:
    """Build the full cross product, randomized to spread thermal bias."""
    rng = random.Random(cfg.run.seed if seed is None else seed)
    cells: list[Cell] = []
    for wl in cfg.workloads:
        for pl in cfg.sweep.power_limits_w:
            for cp in cfg.sweep.clock_points:
                for rep in range(cfg.sweep.repeats):
                    cells.append(
                        Cell(
                            workload=wl.name,
                            domain=wl.domain,
                            power_limit_w=float(pl),
                            clock_min_mhz=int(cp.min_mhz),
                            clock_max_mhz=int(cp.max_mhz),
                            clock_label=cp.label,
                            repeat_idx=rep,
                        )
                    )
    if cfg.sweep.randomized:
        rng.shuffle(cells)
    return cells


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


class SampleLoop:
    """Thread that reads NVML telemetry at a fixed rate into a buffer."""

    def __init__(self, gpu: GpuDevice, hz: float) -> None:
        self.gpu = gpu
        self.interval = 1.0 / hz
        self.samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nvml-sampler", daemon=True)

    def start(self) -> None:
        """Start the sampler thread."""
        self._thread.start()

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            sample = self.gpu.read_telemetry()
            sample["wall_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")
            with self._lock:
                self.samples.append(sample)
            next_t += self.interval
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()

    def stop(self) -> None:
        """Stop the sampler thread."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of samples collected so far."""
        with self._lock:
            return list(self.samples)


def sample_loop(gpu: GpuDevice, hz: float, stop: threading.Event) -> list[dict[str, Any]]:
    """Functional form of the sampler (kept for scripting/tests)."""
    samples: list[dict[str, Any]] = []
    interval = 1.0 / hz
    next_t = time.monotonic()
    while not stop.is_set():
        samples.append(gpu.read_telemetry())
        next_t += interval
        delay = next_t - time.monotonic()
        if delay > 0:
            stop.wait(delay)
        else:
            next_t = time.monotonic()
    return samples


def _temp_slope(samples: list[dict[str, Any]], window_s: float = 2.0) -> float | None:
    if len(samples) < 5:
        return None
    t_end = samples[-1]["timestamp_ns"]
    cutoff = t_end - int(window_s * 1e9)
    recent = [s for s in samples if s["timestamp_ns"] >= cutoff and np.isfinite(s["temp_c"])]
    if len(recent) < 5:
        return None
    t = np.array([(s["timestamp_ns"] - recent[0]["timestamp_ns"]) / 1e9 for s in recent])
    temp = np.array([s["temp_c"] for s in recent])
    if t[-1] <= 0:
        return None
    return float(np.polyfit(t, temp, 1)[0])


def thermal_soak(
    gpu: GpuDevice,
    workload: Any,
    cfg: RigConfig,
    sampler: SampleLoop,
    domain: str,
) -> dict[str, Any]:
    """Run the workload until the temperature slope flattens or timeout fires."""
    if hasattr(gpu, "set_activity"):
        gpu.set_activity(domain)
    start = time.perf_counter()
    slope: float | None = None
    converged = False
    while True:
        elapsed = time.perf_counter() - start
        remaining = cfg.sweep.soak_timeout_s - elapsed
        if remaining <= 0:
            break
        workload.run(min(0.5, remaining))
        elapsed = time.perf_counter() - start
        if elapsed >= cfg.sweep.soak_seconds:
            slope = _temp_slope(sampler.snapshot())
            if slope is not None and abs(slope) <= cfg.sweep.soak_slope_c_per_s:
                converged = True
                break
        if elapsed >= cfg.sweep.soak_timeout_s:
            break
    duration = time.perf_counter() - start
    slope = _temp_slope(sampler.snapshot()) if slope is None else slope
    stats = _power_window(sampler.snapshot(), time.monotonic_ns() - int(duration * 1e9))
    logger.info(
        "  soak %.1fs, slope=%s C/s, temp=%.1f C, power=%.1f W",
        duration,
        f"{slope:.3f}" if slope is not None else "n/a",
        stats.get("temp_c", float("nan")),
        stats.get("power_w", float("nan")),
    )
    return {
        "duration_s": round(duration, 4),
        "slope_c_per_s": None if slope is None else round(slope, 5),
        "converged": converged,
        "final_temp_c": stats.get("temp_c"),
    }


def _power_window(samples: list[dict[str, Any]], since_ns: int) -> dict[str, float]:
    recent = [s for s in samples if s["timestamp_ns"] >= since_ns]
    if not recent:
        return {}
    return {
        "power_w": float(np.nanmean([s["power_w"] for s in recent])),
        "temp_c": float(np.nanmean([s["temp_c"] for s in recent])),
        "sm_clock_mhz": float(np.nanmean([s["sm_clock_mhz"] for s in recent])),
        "util_pct": float(np.nanmean([s["util_pct"] for s in recent])),
        "n": float(len(recent)),
    }


def idle_baseline(
    gpu: GpuDevice, sampler: SampleLoop, seconds: float, domain: str = "idle"
) -> dict[str, Any]:
    """Measure idle power on both sources immediately before the load run."""
    if hasattr(gpu, "set_activity"):
        gpu.set_activity("idle")
    start_ns = time.monotonic_ns()
    time.sleep(seconds)
    end_ns = time.monotonic_ns()
    window = _power_window(sampler.snapshot(), start_ns)
    logger.info("  idle baseline %.1fs: %.1f W", seconds, window.get("power_w", float("nan")))
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": round((end_ns - start_ns) / 1e9, 4),
        "nvml_power_w": window.get("power_w"),
        "nvml_samples": int(window.get("n", 0)),
        "temp_c": window.get("temp_c"),
    }


# --------------------------------------------------------------------------- #
# Cell execution
# --------------------------------------------------------------------------- #


def _cell_complete(cell_dir: Path) -> bool:
    manifest = cell_dir / "manifest.json"
    raw = cell_dir / "raw.csv"
    if not manifest.exists() or not raw.exists() or raw.stat().st_size < 100:
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("status") == "complete"


def _apply_config(gpu: GpuDevice, cell: Cell) -> float:
    applied = gpu.set_power_limit(cell.power_limit_w)
    if cell.clock_min_mhz <= 0 and cell.clock_max_mhz <= 0:
        gpu.reset_clocks()
    else:
        gpu.set_locked_clocks(cell.clock_min_mhz, cell.clock_max_mhz)
    return float(applied)


def _region_for(ts_ns: int, regions: dict[str, int]) -> str:
    if ts_ns < regions["soak_end"]:
        return "soak"
    if ts_ns < regions["baseline_end"]:
        return "baseline"
    return "measurement"


def _write_raw(
    path: Path,
    nvml: list[dict[str, Any]],
    sensor_samples: list[tuple[int, float, float, float]],
    regions: dict[str, int],
) -> None:
    rows: list[dict[str, Any]] = []
    for s in nvml:
        rows.append(
            {
                "source": "nvml",
                "timestamp_ns": s["timestamp_ns"],
                "wall_utc": s.get("wall_utc"),
                "region": _region_for(s["timestamp_ns"], regions),
                "power_w": s["power_w"],
                "sm_clock_mhz": s["sm_clock_mhz"],
                "mem_clock_mhz": s["mem_clock_mhz"],
                "temp_c": s["temp_c"],
                "util_pct": s["util_pct"],
                "power_limit_w": s["power_limit_w"],
                "bus_voltage_v": np.nan,
                "current_a": np.nan,
            }
        )
    for ts, bus_v, current_a, power_w in sensor_samples:
        rows.append(
            {
                "source": "sensor",
                "timestamp_ns": ts,
                "wall_utc": None,
                "region": _region_for(ts, regions),
                "power_w": power_w,
                "sm_clock_mhz": np.nan,
                "mem_clock_mhz": np.nan,
                "temp_c": np.nan,
                "util_pct": np.nan,
                "power_limit_w": np.nan,
                "bus_voltage_v": bus_v,
                "current_a": current_a,
            }
        )
    df = pd.DataFrame(rows).sort_values("timestamp_ns")
    df.to_csv(path, index=False, float_format="%.6f")


def run_cell(
    cell: Cell,
    cfg: RigConfig,
    gpu: GpuDevice,
    sensor: PowerSensor,
    run_dir: Path,
    provenance: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Apply a config point, soak, baseline, measure, and persist the cell."""
    cell_dir = run_dir / "cells" / cell.cell_id
    if not force and _cell_complete(cell_dir):
        logger.info("skip %s (complete)", cell.cell_id)
        return {"cell_id": cell.cell_id, "status": "skipped"}

    cell_dir.mkdir(parents=True, exist_ok=True)
    wl_cfg = next((w for w in cfg.workloads if w.name == cell.workload), None)
    if wl_cfg is None:
        raise SweepError(f"workload {cell.workload} not in config")

    applied_pl = _apply_config(gpu, cell)
    scale = gpu.compute_scale(cell.domain) if isinstance(gpu, SyntheticGpu) else 1.0
    workload = get_workload(
        cell.workload,
        clock_scale=scale,
        pacing=isinstance(gpu, SyntheticGpu),
        seed=cfg.run.seed + cell.repeat_idx,
    )
    workload.prepare(gpu)
    logger.info(
        "cell %s  %.0f W  clocks %d-%d MHz  scale=%.2f",
        cell.cell_id,
        applied_pl,
        cell.clock_min_mhz,
        cell.clock_max_mhz,
        scale,
    )

    sampler = SampleLoop(gpu, cfg.sampling.gpu_hz)
    started = utc_now_iso()
    t_cell = time.perf_counter()
    sensor.start()
    sampler.start()
    try:
        time.sleep(cfg.sweep.settle_seconds)
        soak_start = time.monotonic_ns()
        soak = thermal_soak(gpu, workload, cfg, sampler, cell.domain)
        soak_end = time.monotonic_ns()

        baseline = idle_baseline(gpu, sampler, cfg.sweep.idle_baseline_seconds)
        baseline_end = time.monotonic_ns()

        if hasattr(gpu, "set_activity"):
            gpu.set_activity(cell.domain)
        measure_start = time.monotonic_ns()
        measure_start_perf = time.perf_counter()
        result = workload.run(cfg.sweep.measure_seconds)
        measure_wall = time.perf_counter() - measure_start_perf
        measure_end = time.monotonic_ns()
        time.sleep(0.15)
    finally:
        sampler.stop()
        sensor.stop()
        if hasattr(gpu, "set_activity"):
            gpu.set_activity("idle")
        workload.teardown()

    nvml_samples = sampler.snapshot()
    sensor_samples = sensor.samples()
    regions = {
        "soak_start": soak_start,
        "soak_end": soak_end,
        "baseline_start": baseline["start_ns"],
        "baseline_end": baseline_end,
        "measure_start": measure_start,
        "measure_end": measure_end,
    }
    raw_path = cell_dir / "raw.csv"
    _write_raw(raw_path, nvml_samples, sensor_samples, regions)

    # Idle baseline from the external sensor, using the aligned window later.
    sensor_baseline = [
        p for (ts, _v, _i, p) in sensor_samples if baseline["start_ns"] <= ts <= baseline["end_ns"]
    ]
    if sensor_baseline:
        baseline["sensor_power_w"] = float(np.mean(sensor_baseline))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell.as_dict(),
        "status": "complete",
        "source": provenance["source"],
        "started_utc": started,
        "finished_utc": utc_now_iso(),
        "duration_s": round(time.perf_counter() - t_cell, 4),
        "applied_power_limit_w": applied_pl,
        "clock_scale": round(scale, 4),
        "regions": regions,
        "soak": soak,
        "baseline": baseline,
        "result": {**result, "measured_wall_s": round(measure_wall, 6)},
        "requested": {
            "workload": cell.workload,
            "power_limit_w": cell.power_limit_w,
            "clock_min_mhz": cell.clock_min_mhz,
            "clock_max_mhz": cell.clock_max_mhz,
            "repeat_idx": cell.repeat_idx,
            "measure_seconds": cfg.sweep.measure_seconds,
            "soak_seconds": cfg.sweep.soak_seconds,
        },
        "sampling": {
            "gpu_hz": cfg.sampling.gpu_hz,
            "sensor_hz": cfg.sensor.sample_rate_hz,
            "nvml_samples": len(nvml_samples),
            "sensor_samples": len(sensor_samples),
        },
        "config_hash": provenance["config_hash"],
        "provenance": provenance,
        "artifacts": {"raw_csv": "raw.csv", "manifest": "manifest.json"},
    }
    tmp = cell_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(cell_dir / "manifest.json")
    logger.info(
        "  -> %d %s in %.2fs (%.1f/s), %d nvml + %d sensor samples",
        result["work_units"],
        result["unit_name"],
        result["wall_s"],
        result["throughput_per_s"],
        len(nvml_samples),
        len(sensor_samples),
    )
    return {
        "cell_id": cell.cell_id,
        "status": "complete",
        "throughput_per_s": result["throughput_per_s"],
    }


# --------------------------------------------------------------------------- #
# Sweep orchestration
# --------------------------------------------------------------------------- #


def _restore_defaults(gpu: GpuDevice, cfg: RigConfig) -> None:
    try:
        if hasattr(gpu, "set_activity"):
            gpu.set_activity("idle")
        gpu.reset_clocks()
    except Exception as exc:  # pragma: no cover - hardware failure path
        logger.error("failed to reset clocks: %s", exc)
    try:
        if hasattr(gpu, "reset_power_limit"):
            gpu.reset_power_limit()
        else:  # pragma: no cover
            gpu.set_power_limit(cfg.gpu.default_power_limit_w)
        logger.info("restored default power limit and clocks")
    except Exception as exc:  # pragma: no cover - hardware failure path
        logger.error("failed to restore default power limit: %s", exc)


def sweep(
    cfg: RigConfig,
    out: str | Path | None = None,
    seed: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full matrix, skipping valid completed cells (resumable)."""
    if cfg.workloads and any(w.name not in available_workloads() for w in cfg.workloads):
        bad = [w.name for w in cfg.workloads if w.name not in available_workloads()]
        raise SweepError(f"unknown workloads: {bad}; available: {available_workloads()}")

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out) if out else Path(cfg.run.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu = get_gpu(
        mode=cfg.gpu.mode,
        index=cfg.gpu.index,
        seed=cfg.gpu.seed,
        power_limit_w=cfg.gpu.default_power_limit_w,
    )
    sensor = get_sensor(
        mode=cfg.sensor.mode,
        gpu=gpu,
        port=cfg.sensor.port,
        baud=cfg.sensor.baud,
        shunt_ohms=cfg.sensor.shunt_ohms,
        units=cfg.sensor.units,
        sample_rate_hz=cfg.sensor.sample_rate_hz,
        seed=cfg.sensor.seed,
    )
    provenance = collect_provenance(cfg, gpu, sensor)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )

    cells = build_cells(cfg, seed)
    if limit is not None:
        cells = cells[:limit]
    total = len(cells)
    log_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    wattbench_logger = logging.getLogger("wattbench")
    previous_level = wattbench_logger.level
    if wattbench_logger.getEffectiveLevel() > logging.INFO:
        wattbench_logger.setLevel(logging.INFO)
    wattbench_logger.addHandler(log_handler)
    logger.info("run log: %s", run_dir / "run.log")

    run_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "created_utc": utc_now_iso(),
        "status": "running",
        "source": provenance["source"],
        "config_hash": cfg.config_hash(),
        "config": cfg.to_dict(),
        "provenance": provenance,
        "matrix": {
            "power_limits_w": cfg.sweep.power_limits_w,
            "clock_points": [
                {"label": c.label, "min_mhz": c.min_mhz, "max_mhz": c.max_mhz}
                for c in cfg.sweep.clock_points
            ],
            "workloads": [w.name for w in cfg.workloads],
            "repeats": cfg.sweep.repeats,
            "cells_total": total,
            "configs": len(cfg.sweep.power_limits_w) * len(cfg.sweep.clock_points),
        },
        "cells": [c.as_dict() for c in cells],
    }
    (run_dir / "run.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    logger.info(
        "sweep %s: %d cells (%d configs x %d workloads x %d repeats), source=%s",
        run_dir,
        total,
        run_manifest["matrix"]["configs"],
        len(cfg.workloads),
        cfg.sweep.repeats,
        provenance["source"],
    )
    logger.info("GPU: %s", provenance["gpu"].get("name"))
    logger.info("sensor: %s on %s", provenance["sensor"].get("backend"), cfg.sensor.port)

    completed = skipped = failed = 0
    failures: list[dict[str, str]] = []
    t0 = time.perf_counter()
    interrupted = False
    finished_normally = False
    try:
        for idx, cell in enumerate(cells, start=1):
            logger.info("[%d/%d] %s", idx, total, cell.cell_id)
            try:
                cell_status = run_cell(cell, cfg, gpu, sensor, run_dir, provenance, force=force)
                if cell_status["status"] == "skipped":
                    skipped += 1
                else:
                    completed += 1
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failed += 1
                logger.exception("cell %s failed: %s", cell.cell_id, exc)
                failures.append({"cell_id": cell.cell_id, "error": str(exc)})
                cell_dir = run_dir / "cells" / cell.cell_id
                cell_dir.mkdir(parents=True, exist_ok=True)
                (cell_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "cell": cell.as_dict(),
                            "status": "failed",
                            "error": str(exc),
                            "source": provenance["source"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        finished_normally = True
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "SIGINT received: stopping the sweep and restoring default power limit and clocks"
        )
        raise
    finally:
        _restore_defaults(gpu, cfg)
        if interrupted:
            final_status = "interrupted"
        elif not finished_normally:
            final_status = "error"
        elif failed:
            final_status = "complete_with_failures"
        else:
            final_status = "complete"
        run_manifest.update(
            {
                "status": final_status,
                "finished_utc": utc_now_iso(),
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "counts": {"complete": completed, "skipped": skipped, "failed": failed},
                "failures": failures,
            }
        )
        try:
            (run_dir / "run.json").write_text(
                json.dumps(run_manifest, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - disk full
            logger.error("could not write run.json: %s", exc)
        logger.info(
            "sweep finished: %d complete, %d skipped, %d failed in %.1fs (status=%s)",
            completed,
            skipped,
            failed,
            time.perf_counter() - t0,
            final_status,
        )
        wattbench_logger.removeHandler(log_handler)
        log_handler.close()
        wattbench_logger.setLevel(previous_level)
        logger.info("closed run log")

    return run_manifest


def run_sweep(
    cfg: RigConfig,
    out: str | Path | None = None,
    seed: int | None = None,
    verbose: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> int:
    """CLI adapter: run a sweep and return a process exit code."""
    manifest = sweep(cfg, out=out, seed=seed, limit=limit, force=force)
    counts = manifest.get("counts", {})
    if counts.get("failed"):
        logger.error("%d cells failed; inspect run.json", counts["failed"])
        return 1
    if counts.get("complete", 0) + counts.get("skipped", 0) == 0:
        logger.error("no cells were completed")
        return 1
    print(f"run: {Path(out) if out else Path(cfg.run.out_dir) / manifest['run_id']}")
    print(
        f"cells: {counts.get('complete', 0)} complete, {counts.get('skipped', 0)} skipped, "
        f"{counts.get('failed', 0)} failed"
    )
    print(f"source: {manifest.get('source')}  elapsed: {manifest.get('elapsed_s')}s")
    return 0


def prune_raw_csvs(run_dir: str | Path, keep: int = 1, max_bytes: int = 2_000_000) -> int:
    """Drop bulk raw CSVs while keeping representative cells for the repo."""
    root = Path(run_dir) / "cells"
    csvs = sorted(root.glob("*/raw.csv"), key=lambda p: p.stat().st_size, reverse=True)
    removed = 0
    for path in csvs[keep:]:
        if path.stat().st_size > max_bytes:
            path.unlink()
            removed += 1
    return removed


def sha256_file(path: str | Path) -> str:
    """SHA-256 digest of a file, used for artifact manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_nvidia_smi() -> str | None:
    """Return the nvidia-smi path if present on this machine."""
    return shutil.which("nvidia-smi")


__all__ = [
    "Cell",
    "SampleLoop",
    "build_cells",
    "collect_provenance",
    "idle_baseline",
    "run_cell",
    "run_sweep",
    "sample_loop",
    "sha256_file",
    "sweep",
    "thermal_soak",
]
