"""Backend selection: real hardware when present, synthetic when not.

Auto mode is the default and is always explicit in the artifacts: each device
reports ``source`` as ``hardware`` or ``synthetic``.  Explicit ``hardware`` mode
never silently degrades -- it raises with an actionable error.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from .gpu import (
    DeviceUnavailableError,
    GpuDevice,
    NvmlGpu,
    SyntheticGpu,
    nvml_available,
)
from .power_sensor import (
    PowerSensor,
    SensorUnavailableError,
    SerialIna219Sensor,
    SyntheticIna219Sensor,
    serial_port_exists,
)

logger = logging.getLogger(__name__)

Mode = Literal["auto", "hardware", "synthetic"]


def get_gpu(
    mode: Mode = "auto",
    index: int = 0,
    seed: int = 42,
    power_limit_w: float = 250.0,
) -> GpuDevice:
    """Return an NVML-backed GPU in hardware mode, else a synthetic emulator."""
    if mode not in ("auto", "hardware", "synthetic"):
        raise ValueError(f"unknown GPU mode {mode!r}; use auto, hardware, or synthetic")
    if mode == "synthetic":
        logger.info("using synthetic GPU backend (seed=%d)", seed)
        return SyntheticGpu(seed=seed, power_limit_w=power_limit_w)
    if mode == "auto":
        if nvml_available():
            try:
                gpu = NvmlGpu(index=index)
                logger.info("detected NVIDIA GPU %s via NVML", gpu.info().get("name"))
                return gpu
            except DeviceUnavailableError as exc:
                logger.warning("NVML present but backend failed: %s", exc)
        else:
            logger.info("no NVIDIA GPU detected; using synthetic GPU backend")
        return SyntheticGpu(seed=seed, power_limit_w=power_limit_w)
    # explicit hardware
    try:
        return NvmlGpu(index=index)
    except DeviceUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover
        raise DeviceUnavailableError(f"hardware GPU requested but unavailable: {exc}") from exc


def get_sensor(
    mode: Mode = "auto",
    gpu: GpuDevice | None = None,
    port: str = "/dev/ttyUSB0",
    baud: int = 115200,
    shunt_ohms: float = 0.005,
    units: str = "mv",
    sample_rate_hz: float = 50.0,
    seed: int = 7,
) -> PowerSensor:
    """Return a serial INA219 sensor when usable, else the synthetic sensor."""
    if mode not in ("auto", "hardware", "synthetic"):
        raise ValueError(f"unknown sensor mode {mode!r}; use auto, hardware, or synthetic")

    if mode in ("auto", "hardware"):
        if not serial_port_exists(port):
            if mode == "hardware":
                raise SensorUnavailableError(
                    f"serial port {port} does not exist. Connect the INA219 MCU, pass "
                    "--port, or use --sensor synthetic."
                )
            logger.info("serial port %s not present; using synthetic power sensor", port)
        else:
            sensor = SerialIna219Sensor(
                port=port,
                baud=baud,
                shunt_ohms=shunt_ohms,
                units=units,
                sample_rate_hz=sample_rate_hz,
            )
            try:
                sensor.open()
            except SensorUnavailableError as exc:
                if mode == "hardware":
                    raise
                logger.warning("%s; falling back to synthetic sensor", exc)
            else:
                return sensor

    if gpu is None:
        raise SensorUnavailableError(
            "the synthetic power sensor needs a synthetic GPU to observe; pass gpu=..."
        )
    logger.info("using synthetic INA219 sensor (skew/drift/noise model)")
    return SyntheticIna219Sensor(gpu=gpu, seed=seed, sample_rate_hz=sample_rate_hz)


def describe_backends(gpu: GpuDevice, sensor: PowerSensor | None = None) -> dict[str, Any]:
    """Collect a serialisable description of the active backends."""
    info = gpu.info()
    lo, hi = gpu.supported_power_limit_range()
    out: dict[str, Any] = {
        "gpu": info,
        "gpu_power_limit_range_w": [lo, hi],
        "sensor": sensor.info() if sensor is not None and hasattr(sensor, "info") else None,
    }
    return out
