"""Device protocol conformance and synthetic physics."""

from __future__ import annotations

import time

import pytest

from wattbench.devices.gpu import (
    DeviceUnavailableError,
    GpuDevice,
    NvmlGpu,
    SyntheticGpu,
    nvml_available,
)
from wattbench.devices.power_sensor import (
    PowerSensor,
    SyntheticIna219Sensor,
)


def test_synthetic_gpu_satisfies_protocol() -> None:
    gpu = SyntheticGpu(seed=1)
    assert isinstance(gpu, GpuDevice)


def test_nvml_backend_declares_protocol() -> None:
    for method in (
        "info",
        "read_telemetry",
        "set_power_limit",
        "set_locked_clocks",
        "reset_clocks",
        "supported_power_limit_range",
    ):
        assert callable(getattr(NvmlGpu, method))


def test_synthetic_sensor_satisfies_protocol() -> None:
    gpu = SyntheticGpu(seed=1)
    sensor = SyntheticIna219Sensor(gpu, seed=2)
    assert isinstance(sensor, PowerSensor)


def test_synthetic_gpu_throttles_under_lower_cap() -> None:
    low = SyntheticGpu(seed=10)
    low.set_activity("proviz")
    low.set_power_limit(90)
    low_read = low.read_telemetry()
    high = SyntheticGpu(seed=10)
    high.set_activity("proviz")
    high.set_power_limit(240)
    high_read = high.read_telemetry()
    assert low_read["sm_clock_mhz"] < high_read["sm_clock_mhz"]
    assert low_read["power_w"] <= 90 * 1.12
    assert low.compute_scale("proviz") < high.compute_scale("proviz")


def test_power_limit_clamped_to_supported_range() -> None:
    gpu = SyntheticGpu(seed=3)
    lo, hi = gpu.supported_power_limit_range()
    assert gpu.set_power_limit(1) == lo
    assert gpu.set_power_limit(10_000) == hi
    assert gpu.read_telemetry()["power_limit_w"] == hi


def test_synthetic_thermal_soak_rises() -> None:
    gpu = SyntheticGpu(seed=5)
    gpu.set_activity("proviz")
    gpu.set_power_limit(240)
    first = gpu.read_telemetry()["temp_c"]
    for _ in range(40):
        gpu.read_telemetry()
        time.sleep(0.005)
    later = gpu.read_telemetry()["temp_c"]
    assert later > first


def test_locked_clocks_respected() -> None:
    gpu = SyntheticGpu(seed=7)
    gpu.set_activity("ai")
    gpu.set_locked_clocks(1700, 1700)
    samples = [gpu.read_telemetry() for _ in range(5)]
    assert all(s["sm_clock_mhz"] <= 1710 for s in samples)
    gpu.reset_clocks()
    assert gpu.compute_scale("ai") > 0


def test_hardware_mode_is_explicit() -> None:
    if nvml_available():
        assert isinstance(NvmlGpu(index=0), GpuDevice)
    else:
        with pytest.raises(DeviceUnavailableError):
            NvmlGpu(index=0)
