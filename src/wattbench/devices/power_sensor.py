"""Power-sense abstraction: serial INA219 hardware and a synthetic counterpart.

The external sensor is the independent cross-check on NVML board power.  The
serial protocol is deliberately tiny: the MCU prints newline-delimited

    t_ms,bus_v,shunt_mv

at a fixed rate, where ``shunt_mv`` is the raw differential voltage across the
shunt resistor.  The host converts it to current with the *actual* shunt value,
so a mis-labelled resistor is visible in the data instead of hidden in MCU
firmware.  ``units: a`` also accepts firmware that already prints amps.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SensorUnavailableError(RuntimeError):
    """Raised when the requested serial sensor cannot be opened."""


@dataclass(frozen=True)
class SensorSample:
    """One calibrated external power measurement."""

    timestamp_ns: int
    bus_voltage_v: float
    current_a: float
    power_w: float

    def as_tuple(self) -> tuple[int, float, float, float]:
        return (self.timestamp_ns, self.bus_voltage_v, self.current_a, self.power_w)


@runtime_checkable
class PowerSensor(Protocol):
    """Common interface for external power sensors."""

    def start(self) -> None:
        """Begin acquisition."""

    def stop(self) -> None:
        """Stop acquisition and release the port."""

    def samples(self) -> list[tuple[int, float, float, float]]:
        """Return (timestamp_ns, bus_voltage_v, current_a, power_w) samples."""


def parse_ina219_line(
    line: str,
    shunt_ohms: float = 0.1,
    units: str = "mv",
) -> tuple[int, float, float, float] | None:
    """Parse one ``t_ms,v,i`` line; return ``None`` for partial/garbled input.

    ``units="mv"`` treats the third field as shunt millivolts and derives
    current via Ohm's law.  ``units="a"`` treats it as already-scaled amps.
    The function never raises on malformed serial data: acquisition code must
    survive a USB reconnect mid-line.
    """
    if not line:
        return None
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        t_ms = float(parts[0])
        bus_v = float(parts[1])
        third = float(parts[2])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (t_ms, bus_v, third)):
        return None
    if shunt_ohms <= 0:
        raise ValueError("shunt_ohms must be positive")
    if units == "mv":
        current_a = (third / 1000.0) / shunt_ohms
    elif units == "a":
        current_a = third
    else:
        raise ValueError("units must be 'mv' or 'a'")
    # Physical plausibility guards: a 12 V PCIe rail and a 25 A class shunt.
    if not (0.0 <= bus_v <= 30.0) or abs(current_a) > 60.0:
        return None
    power_w = bus_v * current_a
    return (round(t_ms * 1_000_000), bus_v, current_a, power_w)


class SerialIna219Sensor:
    """Reads INA219-class samples from a USB-serial MCU.

    The reader thread tolerates partial lines and intermittent read errors;
    malformed records are skipped and counted rather than killing a run.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        shunt_ohms: float = 0.005,
        units: str = "mv",
        timeout_s: float = 0.5,
        sample_rate_hz: float = 100.0,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.shunt_ohms = float(shunt_ohms)
        self.units = units
        self.timeout_s = timeout_s
        self.sample_rate_hz = sample_rate_hz
        self._serial: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[tuple[int, float, float, float]] = []
        self._lock = threading.Lock()
        self._bad_lines = 0
        self._first_host_ns: int | None = None
        self._first_mcu_ms: float | None = None
        self._read_errors = 0

    def info(self) -> dict[str, Any]:
        return {
            "source": "hardware",
            "backend": "serial-ina219",
            "port": self.port,
            "baud": self.baud,
            "shunt_ohms": self.shunt_ohms,
            "units": self.units,
        }

    def open(self) -> None:
        """Open the serial port without starting the reader thread."""
        try:
            import serial
        except ImportError as exc:  # pragma: no cover
            raise SensorUnavailableError("pyserial is not installed") from exc
        try:
            self._serial = serial.Serial(
                self.port, self.baud, timeout=self.timeout_s, exclusive=True
            )
        except Exception as exc:
            hint = (
                " Check the USB cable and that the port exists; on Linux add the user to the "
                "'dialout' group. Use --sensor synthetic to run without hardware."
            )
            raise SensorUnavailableError(f"cannot open serial port {self.port}: {exc}.{hint}") from exc
        logger.info("opened INA219 serial sensor on %s @ %d baud", self.port, self.baud)

    def start(self) -> None:
        if self._serial is None:
            self.open()
        with self._lock:
            self._samples.clear()
        self._first_host_ns = None
        self._first_mcu_ms = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="ina219-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._serial is not None
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:  # pragma: no cover - hardware reconnect path
                self._read_errors += 1
                if self._read_errors % 20 == 1:
                    logger.warning("serial read error (%d): %s", self._read_errors, exc)
                time.sleep(0.1)
                continue
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:  # pragma: no cover
                continue
            parsed = parse_ina219_line(text, self.shunt_ohms, self.units)
            if parsed is None:
                if text.strip() and not text.strip().startswith("#"):
                    self._bad_lines += 1
                continue
            _mcu_ns, bus_v, current_a, power_w = parsed
            # Re-anchor the MCU's millisecond clock to the host clock on the
            # first good line, then follow the MCU clock so real drift is kept.
            if self._first_host_ns is None:
                self._first_host_ns = time.monotonic_ns()
                self._first_mcu_ms = _mcu_ns / 1e6
            assert self._first_mcu_ms is not None
            host_ns = int(
                self._first_host_ns + (_mcu_ns / 1e6 - self._first_mcu_ms) * 1_000_000
            )
            with self._lock:
                self._samples.append((host_ns, bus_v, current_a, power_w))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        logger.info(
            "INA219 stopped: %d samples, %d malformed lines, %d read errors",
            len(self._samples),
            self._bad_lines,
            self._read_errors,
        )

    def samples(self) -> list[tuple[int, float, float, float]]:
        with self._lock:
            return list(self._samples)

    @property
    def bad_lines(self) -> int:
        return self._bad_lines


class SyntheticIna219Sensor:
    """Synthetic external sensor derived from the synthetic GPU.

    Emulates the *measurement chain*, not just the GPU: a DC/DC gain error, a
    small fixed current offset, its own Gaussian noise, an independent sampling
    clock with a fixed start skew, and a few hundred ppm of clock drift.  That
    gives the alignment step a real, non-trivial estimation problem.
    """

    def __init__(
        self,
        gpu: Any,
        seed: int = 7,
        sample_rate_hz: float = 50.0,
        drift_ppm: float = 180.0,
        offset_ms: float = 120.0,
        gain: float = 0.985,
        current_offset_a: float = 0.03,
        bus_voltage_v: float = 12.05,
        noise_w: float = 0.35,
    ) -> None:
        self.gpu = gpu
        self.sample_rate_hz = sample_rate_hz
        self.drift_ppm = drift_ppm
        self.offset_ms = offset_ms
        self.gain = gain
        self.current_offset_a = current_offset_a
        self.bus_voltage_v = bus_voltage_v
        self.noise_w = noise_w
        self._rng = random.Random(seed)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[tuple[int, float, float, float]] = []
        self._lock = threading.Lock()
        self._t0_host_ns = 0

    def info(self) -> dict[str, Any]:
        return {
            "source": "synthetic",
            "backend": "synthetic-ina219",
            "port": "synthetic://ina219",
            "shunt_ohms": 0.005,
            "drift_ppm": self.drift_ppm,
            "offset_ms": self.offset_ms,
            "gain": self.gain,
        }

    def start(self) -> None:
        with self._lock:
            self._samples.clear()
        self._stop.clear()
        self._t0_host_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._sample_loop, name="synthetic-ina219", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        interval = 1.0 / self.sample_rate_hz
        t0 = self._t0_host_ns
        while not self._stop.is_set():
            now = time.monotonic_ns()
            # Sensor clock: starts with a fixed skew and runs at (1 + drift).
            sensor_ns = int(t0 + (now - t0) * (1.0 + self.drift_ppm * 1e-6) + self.offset_ms * 1e6)
            if hasattr(self.gpu, "instant_power_w"):
                true_power = float(self.gpu.instant_power_w())
            else:
                true_power = float(self.gpu.current_power_w())
            bus_v = self.bus_voltage_v + self._rng.gauss(0.0, 0.006)
            current = (true_power / self.bus_voltage_v) * self.gain + self.current_offset_a
            current += self._rng.gauss(0.0, 0.004)
            measured = bus_v * current + self._rng.gauss(0.0, self.noise_w)
            with self._lock:
                self._samples.append((sensor_ns, bus_v, current, measured))
            self._stop.wait(interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("synthetic INA219 stopped: %d samples", len(self._samples))

    def samples(self) -> list[tuple[int, float, float, float]]:
        with self._lock:
            return list(self._samples)


def serial_port_exists(port: str) -> bool:
    """Cheap presence check used by the factory before attempting an open."""
    if not port:
        return False
    return os.path.exists(port)
