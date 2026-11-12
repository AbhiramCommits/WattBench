"""Stakeholder report: Markdown plus a self-contained HTML rendering."""

from __future__ import annotations

import html
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from .analysis import AnalysisError, _resolve_results, analyze_run, regression_compare
from .plots import WORKLOAD_LABELS, image_data_uri, plot_all
from .stats import summarize_cv

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _workload_cells(doc: Any, workload: str) -> list[Any]:
    unlocked = [
        c for c in doc.cells if c.workload == workload and c.clock_label == "unlocked"
    ]
    return unlocked or [c for c in doc.cells if c.workload == workload]


def _workload_summary(doc: Any, workload: str) -> dict[str, Any]:
    cells = sorted(_workload_cells(doc, workload), key=lambda c: c.power_limit_w)
    with_tp = [c for c in cells if c.aggregates.get("throughput_per_s")]
    peak = max(with_tp, key=lambda c: c.aggregates["throughput_per_s"].mean or 0.0)
    with_eff = [c for c in cells if c.efficiency is not None]
    best = max(with_eff, key=lambda c: c.efficiency.perf_per_watt_nvml)
    knee = cells[0].knee if cells else None
    unit = with_tp[0].efficiency.unit_name if with_tp and with_tp[0].efficiency else "units"
    peak_tp = float(peak.aggregates["throughput_per_s"].mean or 0.0)
    peak_power = float(peak.aggregates["nvml_power_w"].mean or 0.0)
    best_eff = float(best.efficiency.perf_per_watt_nvml) if best.efficiency else 0.0
    best_tp = float(best.aggregates["throughput_per_s"].mean or 0.0)
    best_power = float(best.aggregates["nvml_power_w"].mean or 0.0)
    return {
        "workload": workload,
        "label": WORKLOAD_LABELS.get(workload, workload),
        "unit": unit,
        "cells": cells,
        "peak": peak,
        "peak_tp": peak_tp,
        "peak_power": peak_power,
        "peak_power_limit": float(peak.power_limit_w),
        "best": best,
        "best_eff": best_eff,
        "best_tp": best_tp,
        "best_power": best_power,
        "best_power_limit": float(best.power_limit_w),
        "knee": knee,
    }


def _headline_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Peak throughput | Peak board power | Best perf/W | At power limit | Knee power limit | Knee throughput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        knee = s["knee"]
        knee_tp = f"{knee.throughput_pct_of_peak:.1f}% of peak" if knee else "n/a"
        knee_pl = f"{knee.power_limit_w:.0f} W" if knee else "n/a"
        lines.append(
            f"| {s['label']} | {s['peak_tp']:.2f} {s['unit']}/s | "
            f"{s['peak_power']:.1f} W | {s['best_eff']:.4f} {s['unit']}/s/W | "
            f"{s['best_power_limit']:.0f} W | {knee_pl} | {knee_tp} |"
        )
    return "\n".join(lines)


def _knee_paragraph(s: dict[str, Any]) -> str | None:
    knee = s["knee"]
    if knee is None:
        return None
    peak_delta_tp = s["peak_tp"] - knee.throughput_per_s
    peak_delta_tp_pct = 100.0 * peak_delta_tp / knee.throughput_per_s if knee.throughput_per_s else 0.0
    peak_delta_p = s["peak_power"] - knee.power_w
    peak_delta_p_pct = 100.0 * peak_delta_p / knee.power_w if knee.power_w else 0.0
    return (
        f"**{s['label']}.** At a {knee.power_limit_w:.0f} W power limit "
        f"({knee.power_pct_of_peak:.1f}% of the {knee.peak_power_limit_w:.0f} W tested peak), "
        f"{s['label']} delivers {knee.throughput_per_s:.2f} {s['unit']}/s — "
        f"{knee.throughput_pct_of_peak:.1f}% of peak throughput — at "
        f"{knee.power_w:.1f} W measured — "
        f"{knee.throughput_pct_of_peak / knee.power_pct_of_peak:.2f}x the throughput per unit of "
        f"power-limit range relative to the peak setting. The highest measured efficiency is "
        f"{s['best_eff']:.4f} {s['unit']}/s/W at {s['best_power_limit']:.0f} W "
        f"({100.0 * s['best_tp'] / s['peak_tp'] if s['peak_tp'] else 0:.1f}% of peak throughput). "
        f"Going from the knee to the maximum-performance point adds "
        f"{peak_delta_tp:.2f} {s['unit']}/s (+{peak_delta_tp_pct:.1f}%) for "
        f"{peak_delta_p:.1f} W more board power (+{peak_delta_p_pct:.1f}%)."
    )


def _confidence_section(doc: Any) -> str:
    a = doc.alignment
    cv_power = summarize_cv([c.model_dump() for c in doc.cells], "nvml_power_w")
    cv_tp = summarize_cv([c.model_dump() for c in doc.cells], "throughput_per_s")
    rejected = sum(len(c.rejected_outliers) for c in doc.cells)
    lines = [
        "## Measurement confidence",
        "",
        f"Both power streams were resampled to a common grid and aligned by maximising "
        f"normalized cross-correlation of their derivatives. Across {len(doc.cells)} cells the "
        f"fitted sensor clock offset is **{a.offset_ms_mean:.1f} ms** "
        f"(stdev {a.offset_ms_stdev:.1f} ms) with a fitted drift of "
        f"**{a.drift_ppm_mean:+.1f} ppm** (stdev {a.drift_ppm_stdev:.1f} ppm). The drift is a "
        f"short-window estimate; the synthetic-injection test in `test_align.py` recovers a "
        f"known drift on a longer trace.",
        "",
        f"During the steady-state measurement window, NVML and the external sensor agree to a "
        f"mean absolute difference of **{a.mean_abs_diff_w:.2f} W**, a signed bias of "
        f"**{a.percent_bias:+.2f}%**, and a full-trace Pearson r of "
        f"**{(a.pearson_r_full or 0):.4f}**. The 95% Bland-Altman limits of agreement are "
        f"**{a.loa_lower_w:+.2f} W to {a.loa_upper_w:+.2f} W** (sensor minus NVML). "
        + (
            "No cell exceeded the configured bias threshold."
            if not a.cells_flagged
            else f"{len(a.cells_flagged)} cell(s) exceeded the "
            f"{a.bias_threshold_pct:.1f}% bias threshold: {', '.join(a.cells_flagged)}."
        ),
        "",
        f"Run-to-run repeatability: coefficient of variation for NVML power is "
        f"{cv_power['min']:.2f}% (min) / {cv_power['median']:.2f}% (median) / "
        f"{cv_power['max']:.2f}% (max); for throughput it is {cv_tp['min']:.2f}% / "
        f"{cv_tp['median']:.2f}% / {cv_tp['max']:.2f}%. "
        f"{rejected} repeat(s) were rejected as modified-z outliers "
        f"(threshold {doc.provenance.get('config', {}).get('sweep', {}).get('outlier_z', 3.5)}); "
        f"every rejection is listed in `results.json`, not silently dropped. "
        f"Each cell was measured with "
        f"{doc.cells[0].repeats_used if doc.cells else 0} usable repeat(s) after rejection.",
        "",
    ]
    return "\n".join(lines)


def _regression_section(baseline: str | None, doc: Any, run_dir: Path) -> str:
    if not baseline:
        return ""
    cfg = doc.provenance.get("config", {}).get("analysis", {})
    result = regression_compare(
        baseline,
        run_dir,
        z=float(cfg.get("regression_z", 3.0)),
        rel_tol=float(cfg.get("regression_rel_tol", 0.03)),
    )
    lines = [
        "## Regression check against the baseline run",
        "",
        f"Overall verdict: **{result['overall']}** across {result['cells_compared']} matched "
        f"cells. Tolerance is `max(z x combined standard error, "
        f"{100 * result['rel_tol']:.1f}% of baseline)`, with z = {result['z']}.",
        "",
        "| Cell | Baseline | Candidate | Delta | Tolerance | Verdict |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['cell_id']} | {row['baseline_throughput']:.3f} | "
            f"{row['candidate_throughput']:.3f} | {row['delta']:+.3f} "
            f"({row['delta_pct']:+.2f}%) | {row['tolerance']:.3f} | {row['verdict']} |"
        )
    if result["cells_missing"]:
        lines += ["", f"Cells missing from one side: {', '.join(result['cells_missing'])}"]
    lines.append("")
    return "\n".join(lines)


def _provenance_section(doc: Any) -> str:
    p = doc.provenance
    gpu = p.get("gpu", {})
    sensor = p.get("sensor", {})
    host = p.get("host", {})
    versions = p.get("versions", {})
    config = p.get("config", {})
    background = p.get("background_processes_top_cpu", [])
    lines = [
        "## Rig configuration appendix",
        "",
        "```yaml",
        yaml.safe_dump(config, sort_keys=False).rstrip(),
        "```",
        "",
        "## Provenance",
        "",
        f"- Measurement source: **{doc.run.source}**",
        f"- GPU: {gpu.get('name', 'unknown')} ({gpu.get('backend', 'unknown')} backend, "
        f"driver {gpu.get('driver_version', 'unknown')})",
        f"- Sensor: {sensor.get('backend', 'unknown')} on {sensor.get('port', 'unknown')}, "
        f"shunt {p.get('shunt_ohms', 'unknown')} Ω",
        f"- Host: {host.get('hostname', 'unknown')}, {host.get('platform', 'unknown')}, "
        f"{host.get('cpu_count', 'unknown')} CPUs",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items())),
        f"- Git commit: {p.get('git_commit', 'unknown')} "
        f"(dirty={p.get('git_dirty', 'unknown')})",
        f"- Config hash: {p.get('config_hash', 'unknown')}",
        f"- Captured: {p.get('captured_utc', 'unknown')}",
    ]
    if background:
        lines += ["", "Top background processes by CPU at sweep start:", "", "```"]
        for proc in background:
            lines.append(f"{proc.get('cpu_pct', 0):6.1f}%  {proc.get('command', '')}")
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_markdown(doc: Any, run_dir: Path, plots: dict[str, Path], baseline: str | None) -> str:
    summaries = [_workload_summary(doc, wl) for wl in sorted({c.workload for c in doc.cells})]
    source_note = (
        "**Measurement source: synthetic.** These numbers were produced by WattBench's "
        "deterministic synthetic GPU and synthetic INA219 backends. The power model integrates "
        "a first-order thermal soak, throttles clocks against the configured board cap, and "
        "adds per-sample noise; the sensor stream carries an independent clock skew, drift, "
        "gain error and noise. The identical pipeline runs unchanged on a real rig — set "
        "`gpu.mode: hardware` and `sensor.mode: hardware`."
        if doc.run.source != "hardware"
        else "**Measurement source: hardware.** NVML telemetry and an external INA219 shunt "
        "sensor were read from the physical rig described in `docs/rig_setup.md`."
    )
    sections: list[str] = [
        f"# WattBench report — {doc.run.run_id}",
        "",
        source_note,
        "",
        f"Run completed {doc.run.finished_utc or doc.run.created_utc or 'n/a'} in "
        f"{_fmt(doc.run.elapsed_s, 1)} s. Matrix: "
        f"{doc.run.matrix.get('configs', '?')} power/clock configurations x "
        f"{len(doc.run.matrix.get('workloads', []))} workloads x "
        f"{doc.run.matrix.get('repeats', '?')} repeats = {len(doc.cells)} aggregated cells "
        f"({sum(c.repeats_total for c in doc.cells)} measured repeats).",
        "",
        "## Headline results",
        "",
        _headline_table(summaries),
        "",
        "## Efficiency knee findings",
        "",
        "The knee is the first point where the fitted marginal throughput per added watt falls "
        f"below {doc.analysis.get('knee_marginal_fraction', 0.9):.2f}x its peak value. "
        "It sits between the highest-efficiency point and the maximum-performance point.",
        "",
    ]
    for s in summaries:
        para = _knee_paragraph(s)
        if para:
            sections += [para, ""]
    sections += [
        "## Charts",
        "",
        f"![Throughput vs power limit](plots/{plots['perf'].name})",
        "",
        f"![Performance per watt with knee](plots/{plots['efficiency'].name})",
        "",
        f"![NVML vs external sensor agreement](plots/{plots['scatter'].name})",
        "",
        f"![Power time series with soak, baseline and measurement](plots/{plots['timeseries'].name})",
        "",
        f"![Efficiency by domain](plots/{plots['bars'].name})",
        "",
        _confidence_section(doc),
        _regression_section(baseline, doc, run_dir),
        _provenance_section(doc),
        "## Method notes",
        "",
        "- Each sweep cell applies the power limit and clock lock, runs a thermal soak until "
        "the temperature slope flattens or the timeout expires, measures an idle baseline, "
        "then runs the workload for the configured measurement window.",
        "- Energy is integrated with the trapezoid rule over the sampled power trace, so it "
        "reflects the actual power shape rather than a setpoint. Work per joule is reported "
        "both with and without idle-power subtraction in `results.json`.",
        "- Cells are executed in randomized order to spread thermal bias, and the sweep is "
        "resumable. Repeats are aggregated with mean, median, stdev, CV and a 95% confidence "
        "interval; modified-z outliers are recorded, never silently removed.",
        "- The external sensor is aligned to NVML by a fitted constant offset and linear drift; "
        "agreement is reported as MAE, percent bias, Pearson r, and Bland-Altman limits.",
        "",
    ]
    return "\n".join(sections)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def markdown_to_html(md: str, base_dir: Path) -> str:
    """Minimal Markdown renderer for the report's controlled subset."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code: list[str] = []
    table: list[str] = []

    def flush_table() -> None:
        nonlocal table
        if not table:
            return
        rows = [[c.strip() for c in row.strip().strip("|").split("|")] for row in table]
        out.append("<table>")
        out.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr></thead>")
        out.append("<tbody>")
        for row in rows[2:]:
            out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
        out.append("</tbody></table>")
        table = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_table()
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if line.startswith("|"):
            table.append(line)
            i += 1
            continue
        flush_table()
        if not line.strip():
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            out.append(f"<h{level}>{_inline(text)}</h{level}>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif _IMAGE_RE.fullmatch(line.strip()):
            m = _IMAGE_RE.fullmatch(line.strip())
            assert m is not None
            alt, src = m.group(1), m.group(2)
            img = base_dir / src
            uri = image_data_uri(img) if img.exists() else ""
            out.append(f'<figure><img src="{uri}" alt="{html.escape(alt)}">'
                       f"<figcaption>{_inline(alt)}</figcaption></figure>")
        else:
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    flush_table()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
    return "\n".join(out)


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       max-width: 1050px; margin: 2rem auto; padding: 0 1.2rem; color: #1d232a; line-height: 1.55; }
h1 { border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #d9dee3; padding-bottom: .25rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }
th, td { border: 1px solid #cfd6dc; padding: .4rem .55rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead { background: #eef3f7; }
tbody tr:nth-child(even) { background: #fafbfc; }
code { background: #f0f3f5; padding: .1rem .25rem; border-radius: 3px; font-size: .88em; }
pre { background: #f6f8fa; padding: .9rem; overflow-x: auto; border-radius: 6px; }
pre code { background: none; padding: 0; }
img { max-width: 100%; border: 1px solid #e1e6ea; border-radius: 4px; margin-top: .4rem; }
figure { margin: 1.4rem 0; }
figcaption { color: #5a6672; font-size: .85rem; text-align: center; }
strong { color: #102a43; }
"""


def build_report(run_dir: str | Path, baseline: str | None = None) -> int:
    """Render ``report.md`` and a self-contained ``report.html``."""
    run_path = Path(run_dir)
    try:
        results_path, doc = _resolve_results(run_path)
    except AnalysisError:
        logger.info("no results found for %s; running analysis first", run_path)
        if analyze_run(run_path, out=run_path / "results", write_mirror=False) != 0:
            raise
        results_path, doc = _resolve_results(run_path)
    run_results_dir = results_path.parent
    plots = plot_all(run_path, doc, run_results_dir / "plots")
    md = build_markdown(doc, run_path, plots, baseline)
    (run_results_dir / "report.md").write_text(md, encoding="utf-8")
    html_doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>WattBench report — {html.escape(doc.run.run_id)}</title>"
        f"<style>{_CSS}</style></head><body>"
        + markdown_to_html(md, run_results_dir)
        + "</body></html>"
    )
    (run_results_dir / "report.html").write_text(html_doc, encoding="utf-8")

    top = Path("results")
    if top.resolve() != run_results_dir.resolve():
        top.mkdir(parents=True, exist_ok=True)
        (top / "report.md").write_text(md, encoding="utf-8")
        (top / "report.html").write_text(html_doc, encoding="utf-8")
        plot_dir = top / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for p in plots.values():
            if isinstance(p, Path) and p.is_file():
                shutil.copy2(p, plot_dir / p.name)
    print(f"report: {run_results_dir / 'report.md'}")
    print(f"html:   {run_results_dir / 'report.html'} (self-contained)")
    n_plots = len([p for p in plots.values() if isinstance(p, Path) and p.is_file()])
    print(f"plots:  {run_results_dir / 'plots'} ({n_plots})")
    return 0
