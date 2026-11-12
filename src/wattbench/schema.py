"""Versioned, machine-readable results schema.

Results are validated through pydantic on every write; ``wattbench analyze``
refuses to produce an invalid document and fails loudly with the field path.
Non-finite floats are normalised to JSON ``null`` so the document is valid
JSON, and optional fields are typed accordingly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class MetricStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    cv_pct: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


class AlignmentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_ms: float
    drift_ppm: float
    sensor_t0_ns: int | None = None
    correlation: float | None = None
    grid_hz: float | None = None
    nvml_samples: int | None = None
    sensor_samples: int | None = None
    mean_abs_diff_w: float
    mean_diff_w: float | None = None
    percent_bias: float
    pearson_r: float | None = None
    pearson_r_full: float | None = None
    loa_lower_w: float
    loa_upper_w: float
    flagged: bool
    bias_threshold_pct: float
    window: str = "measurement"


class AlignmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bias_threshold_pct: float
    offset_ms_mean: float
    offset_ms_stdev: float
    drift_ppm_mean: float
    drift_ppm_stdev: float
    mean_abs_diff_w: float
    percent_bias: float
    pearson_r: float | None = None
    pearson_r_full: float | None = None
    loa_lower_w: float
    loa_upper_w: float
    cells_flagged: list[str] = Field(default_factory=list)
    repeats_flagged: int = 0
    per_cell: dict[str, AlignmentMetrics] = Field(default_factory=dict)


class Efficiency(BaseModel):
    """Performance per joule, with and without idle-power subtraction."""

    model_config = ConfigDict(extra="forbid")

    work_units: int
    unit_name: str
    energy_j_nvml: float
    energy_j_sensor: float
    work_per_joule_nvml: float
    work_per_joule_sensor: float
    perf_per_watt_nvml: float
    perf_per_watt_sensor: float
    idle_subtracted_energy_j_nvml: float | None = None
    idle_subtracted_energy_j_sensor: float | None = None
    idle_subtracted_work_per_joule_nvml: float | None = None
    idle_subtracted_work_per_joule_sensor: float | None = None
    idle_power_nvml_w: float | None = None
    idle_power_sensor_w: float | None = None
    mean_power_nvml_w: float
    mean_power_sensor_w: float
    duration_s: float


class Knee(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload: str
    clock_label: str
    method: str
    power_limit_w: float
    power_w: float
    throughput_per_s: float
    perf_per_watt: float
    peak_power_limit_w: float
    peak_power_w: float
    peak_throughput_per_s: float
    peak_perf_per_watt: float
    max_efficiency_power_limit_w: float
    max_efficiency_power_w: float
    max_efficiency_throughput_pct: float
    throughput_pct_of_peak: float
    power_pct_of_peak: float
    marginal_fraction: float


class RepeatRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_idx: int
    cell_id: str
    work_units: int
    wall_s: float
    throughput_per_s: float
    nvml_power_w: float
    sensor_power_w: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    temp_c: float
    util_pct: float
    nvml_baseline_w: float | None = None
    sensor_baseline_w: float | None = None
    raw_csv: str | None = None
    efficiency: Efficiency | None = None
    alignment: AlignmentMetrics


class RejectedOutlier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_idx: int
    cell_id: str | None = None
    value: float
    modified_z: float
    reason: str


class CellResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    workload: str
    domain: str
    power_limit_w: float
    clock_label: str
    clock_min_mhz: int
    clock_max_mhz: int
    source: str
    repeats_total: int
    repeats_used: int
    rejected_outliers: list[RejectedOutlier] = Field(default_factory=list)
    aggregates: dict[str, MetricStats]
    per_repeat: list[RepeatRecord]
    efficiency: Efficiency | None = None
    knee: Knee | None = None


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dir: str
    source: str
    config_hash: str
    git_commit: str | None = None
    created_utc: str | None = None
    finished_utc: str | None = None
    elapsed_s: float | None = None
    matrix: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)


class ResultsDoc(BaseModel):
    """Top-level results document, validated before it is written."""

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="null")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    generated_utc: str
    run: RunMetadata
    provenance: dict[str, Any]
    alignment: AlignmentSummary
    cells: list[CellResult]
    analysis: dict[str, Any] | None = None

    @field_validator("cells")
    @classmethod
    def _non_empty(cls, value: list[CellResult]) -> list[CellResult]:
        if not value:
            raise ValueError("results must contain at least one cell")
        return value


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats with None and recurse through containers."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def validate_results(doc: dict[str, Any]) -> ResultsDoc:
    """Validate a results dict, raising ``ResultsSchemaError`` on failure."""
    try:
        return ResultsDoc.model_validate(_sanitize(doc))
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ResultsSchemaError(f"results schema violation: {errors}") from exc


class ResultsSchemaError(ValueError):
    """Raised when a results document does not satisfy the schema."""


def write_results(doc: dict[str, Any], path: str | Path) -> ResultsDoc:
    """Validate and atomically write a results document."""
    validated = validate_results(doc)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(out)
    return validated


def load_results(path: str | Path) -> ResultsDoc:
    """Load and re-validate an existing results document."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultsSchemaError(f"{p} is not valid JSON: {exc}") from exc
    return validate_results(raw)
