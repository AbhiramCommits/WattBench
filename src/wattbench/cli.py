"""Command-line entrypoint for WattBench."""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
from typing import TYPE_CHECKING

from . import __version__
from .config import RigConfig, load_config

if TYPE_CHECKING:
    from .devices.factory import get_gpu, get_sensor  # noqa: F401

logger = logging.getLogger("wattbench")


def _default_config() -> str | None:
    from pathlib import Path

    p = Path("config/rig.yaml")
    return str(p) if p.exists() else None


def _load(args: argparse.Namespace) -> RigConfig:
    if getattr(args, "config", None):
        return load_config(args.config)
    return RigConfig()


def cmd_devices(args: argparse.Namespace) -> int:
    """Print the resolved backends and their capabilities."""
    from .devices.factory import describe_backends, get_gpu, get_sensor
    from .devices.power_sensor import serial_port_exists

    cfg = _load(args)
    gpu_mode = args.gpu or cfg.gpu.mode
    sensor_mode = args.sensor or cfg.sensor.mode
    gpu = get_gpu(mode=gpu_mode, index=cfg.gpu.index, seed=cfg.gpu.seed)
    sensor = get_sensor(
        mode=sensor_mode,
        gpu=gpu,
        port=cfg.sensor.port,
        baud=cfg.sensor.baud,
        shunt_ohms=cfg.sensor.shunt_ohms,
        units=cfg.sensor.units,
        sample_rate_hz=cfg.sensor.sample_rate_hz,
        seed=cfg.sensor.seed,
    )
    info = gpu.info()
    lo, hi = gpu.supported_power_limit_range()
    desc = describe_backends(gpu, sensor)
    sensor_info = desc["sensor"] or {}

    print(f"WattBench {__version__} — device discovery")
    print()
    print(f"  GPU backend      : {info.get('backend')} ({info.get('source')})")
    print(f"  GPU              : {info.get('name')}")
    print(f"  Driver           : {info.get('driver_version')}")
    print(f"  Power-limit range: {lo:.0f}-{hi:.0f} W")
    print(f"  Telemetry        : {info.get('backend')} @ {cfg.sampling.gpu_hz:.0f} Hz")
    print()
    print(f"  Sensor backend   : {sensor_info.get('backend', 'none')} ({sensor_info.get('source', 'none')})")
    port = sensor_info.get("port", cfg.sensor.port)
    status = "present" if serial_port_exists(str(port)) else "not attached"
    print(f"  Sensor port      : {port} [{status}]")
    print(f"  Shunt            : {sensor_info.get('shunt_ohms', cfg.sensor.shunt_ohms)} Ω")
    print(f"  Sensor rate      : {cfg.sensor.sample_rate_hz:.0f} Hz")
    print()
    if info.get("source") == "synthetic" or sensor_info.get("source") == "synthetic":
        print("  NOTE: synthetic backends are active. Artifacts will be stamped")
        print('        "source": "synthetic"; they are physically modelled, not hardware.')
    print(f"  Config           : {args.config or 'built-in defaults'}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a sweep from parsed CLI arguments."""
    from .runner import run_sweep

    cfg = _load(args)
    out = args.out
    return run_sweep(cfg, out=out, seed=args.seed, verbose=args.verbose)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a run directory from parsed CLI arguments."""
    from .analysis import analyze_run

    return analyze_run(args.run_dir, out=args.out, config_path=args.config)


def cmd_report(args: argparse.Namespace) -> int:
    """Render the stakeholder report from parsed CLI arguments."""
    from .report import build_report

    return build_report(
        args.run_dir,
        baseline=getattr(args, "baseline", None),
        write_top_level=not getattr(args, "no_mirror", False),
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two runs from parsed CLI arguments."""
    from .analysis import compare_runs

    return compare_runs(args.baseline, args.candidate, rel_tol=args.rel_tol)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse command tree."""
    parser = argparse.ArgumentParser(
        prog="wattbench",
        description="GPU performance-per-watt measurement with cross-validated power sensing",
    )
    parser.add_argument("--version", action="version", version=f"wattbench {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("devices", help="detect and describe GPU + power-sensor backends")
    d.add_argument("--config", default=_default_config(), help="rig config YAML")
    d.add_argument("--gpu", choices=["auto", "hardware", "synthetic"], default=None)
    d.add_argument("--sensor", choices=["auto", "hardware", "synthetic"], default=None)
    d.set_defaults(func=cmd_devices)

    s = sub.add_parser("sweep", help="run a power-limit / clock sweep")
    s.add_argument("--config", default=_default_config(), help="rig config YAML")
    s.add_argument("--out", default=None, help="output run directory (default runs/<timestamp>)")
    s.add_argument("--seed", type=int, default=None, help="cell-order / device seed override")
    s.set_defaults(func=cmd_sweep)

    a = sub.add_parser("analyze", help="align, aggregate, and validate a run")
    a.add_argument("run_dir", help="path to runs/<timestamp>")
    a.add_argument("--out", default=None, help="results output directory")
    a.add_argument("--config", default=None, help="optional config override (bias threshold)")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("report", help="render markdown + self-contained HTML report")
    r.add_argument("run_dir", help="path to runs/<timestamp>")
    r.add_argument("--baseline", default=None, help="optional baseline run for regression section")
    r.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not copy the report into the top-level results/ directory",
    )
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="compare a candidate run against a baseline run")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--rel-tol", type=float, default=None, help="override relative tolerance")
    c.set_defaults(func=cmd_compare)
    return parser


def _handle_sigterm(signum: int, frame: object) -> None:
    """Translate SIGTERM into KeyboardInterrupt so cleanup blocks always run."""
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    with contextlib.suppress(ValueError, OSError):  # non-main thread / unsupported
        signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return 130
    except Exception as exc:
        if args.verbose:
            logger.exception("command failed")
        else:
            logger.error("error: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
