"""GPU device abstraction with a real NVML backend and a physical synthetic model.

Every hardware touchpoint in WattBench sits behind a Protocol so the whole
pipeline can run unchanged on a workstation with no NVIDIA GPU.  The synthetic
backend is not a constant: it integrates a thermal model, throttles clocks
against the configured power cap, and produces per-sample noise.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import threading
import time
import warnings
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested hardware backend cannot be opened."""


class GpuControlError(RuntimeError):
    """Raised when a power-limit or clock change is rejected by the driver."""


@runtime_checkable
class GpuDevice(Protocol):
    """Common interface for all GPU telemetry/control backends."""

    def info(self) -> dict[str, Any]:
        """Return identifying information about the device and backend."""

    def read_telemetry(self) -> dict[str, Any]:
        """Return one telemetry sample (timestamp_ns is time.monotonic_ns())."""

    def set_power_limit(self, watts: float) -> float:
        """Apply a board power limit in watts; returns the applied value."""

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        """Lock graphics clocks to a range (0,0 means unlocked)."""

    def reset_clocks(self) -> None:
        """Restore default clock behaviour."""

    def supported_power_limit_range(self) -> tuple[float, float]:
        """Return the (minimum, maximum) enforceable board power limit."""


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #


class NvmlGpu:
    """NVIDIA GPU backend.

    Telemetry comes from NVML.  Power-limit and clock changes are issued with
    ``nvidia-smi`` because NVML does not expose all of them portably; every
    command is logged verbatim so a lab operator can audit or replay it.
    """

    def __init__(self, index: int = 0, smi_path: str | None = None) -> None:
        try:
            import pynvml
        except ImportError as exc:  # pragma: no cover - exercised only without dep
            raise DeviceUnavailableError(
                "pynvml is not installed; install the 'pynvml' package or use --gpu synthetic"
            ) from exc

        self._nvml = pynvml
        self.index = index
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:  # pynvml raises its own exception types
            raise DeviceUnavailableError(
                f"NVML could not open GPU {index}: {exc}. "
                "Install the NVIDIA driver, or use the synthetic backend (--gpu synthetic)."
            ) from exc

        self._smi = smi_path or shutil.which("nvidia-smi") or "nvidia-smi"
        if not shutil.which(self._smi) and smi_path is None:
            logger.warning("nvidia-smi not found on PATH; power-limit changes will fail")
        self._default_power_limit_w = 0.0
        self._name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self._name, bytes):  # pragma: no cover - old pynvml
            self._name = self._name.decode()
        try:
            self._default_power_limit_w = (
                pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle) / 1000.0
            )
        except Exception:  # pragma: no cover - not all cards expose it
            self._default_power_limit_w = 0.0

    # -- info ------------------------------------------------------------- #
    def info(self) -> dict[str, Any]:
        """Return device identity and backend information."""
        nv = self._nvml
        driver = nv.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):  # pragma: no cover
            driver = driver.decode()
        try:
            uuid = nv.nvmlDeviceGetUUID(self.handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
        except Exception:
            uuid = "unknown"
        return {
            "source": "hardware",
            "backend": "nvml",
            "name": self._name,
            "driver_version": driver,
            "index": self.index,
            "uuid": uuid,
            "nvml_version": nv.nvmlSystemGetNVMLVersion(),
        }

    # -- telemetry -------------------------------------------------------- #
    def read_telemetry(self) -> dict[str, Any]:
        """Read one telemetry sample from the device."""
        nv = self._nvml
        h = self.handle
        ts = time.monotonic_ns()
        try:
            power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = float("nan")
        try:
            limit = nv.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception:
            limit = float("nan")
        try:
            sm_clock = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        except Exception:
            sm_clock = 0
        try:
            mem_clock = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)
        except Exception:
            mem_clock = 0
        try:
            temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = float("nan")
        try:
            util = nv.nvmlDeviceGetUtilizationRates(h).gpu
        except Exception:
            util = float("nan")
        return {
            "timestamp_ns": ts,
            "power_w": float(power),
            "sm_clock_mhz": int(sm_clock),
            "mem_clock_mhz": int(mem_clock),
            "temp_c": float(temp),
            "util_pct": float(util),
            "power_limit_w": float(limit),
        }

    # -- control ---------------------------------------------------------- #
    def supported_power_limit_range(self) -> tuple[float, float]:
        """Return the enforceable (min, max) board power limit."""
        try:
            lo, hi = self._nvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
            return (lo / 1000.0, hi / 1000.0)
        except Exception as exc:
            logger.warning("NVML power-limit constraints unavailable: %s", exc)
            return (0.0, 0.0)

    def _run_smi(self, args: list[str]) -> str:
        cmd = [self._smi, "-i", str(self.index), *args]
        logger.info("running: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        except FileNotFoundError as exc:
            raise GpuControlError(
                "nvidia-smi was not found. Install the NVIDIA driver or use the synthetic backend."
            ) from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            hint = ""
            if "permission" in detail.lower() or "not permitted" in detail.lower():
                hint = (
                    " Power-limit and clock changes require root. Re-run with sudo, or grant the "
                    "capability to nvidia-smi (see docs/rig_setup.md)."
                )
            raise GpuControlError(f"nvidia-smi failed ({proc.returncode}): {detail}.{hint}")
        return proc.stdout.strip()

    def set_power_limit(self, watts: float) -> float:
        """Apply a board power limit and return the applied value."""
        lo, hi = self.supported_power_limit_range()
        applied = float(watts)
        if hi > 0:
            applied = min(max(float(watts), lo), hi)
            if applied != float(watts):
                logger.warning(
                    "requested %.0f W outside supported [%.0f, %.0f] W; clamping to %.0f W",
                    watts,
                    lo,
                    hi,
                    applied,
                )
        self._run_smi(["-pl", f"{applied:.0f}"])
        return applied

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        """Lock graphics clocks to the given range."""
        if min_mhz <= 0 and max_mhz <= 0:
            self.reset_clocks()
            return
        if min_mhz <= 0:
            min_mhz = max_mhz
        if max_mhz <= 0:
            max_mhz = min_mhz
        self._run_smi(["-lgc", f"{int(min_mhz)},{int(max_mhz)}"])

    def reset_clocks(self) -> None:
        """Release any clock lock and restore defaults."""
        self._run_smi(["-rgc"])

    def reset_power_limit(self) -> None:
        """Restore the driver's default board limit (best effort)."""
        if self._default_power_limit_w > 0:
            self.set_power_limit(self._default_power_limit_w)
        else:  # pragma: no cover
            self._run_smi(["-pl", "default"])


# --------------------------------------------------------------------------- #
# Synthetic backend
# --------------------------------------------------------------------------- #

# Physical constants of the emulated card.  Voltage rises gently with clock;
# dynamic power is modelled as k * f * V^2 * utilization, with a fixed
# leakage/static term.  The thermal model is first order.
_FAKE_NAME = "Synthetic Phoenix X24 (emulated)"
_PLIM_MIN_W = 60.0
_PLIM_MAX_W = 320.0
_PLIM_DEFAULT_W = 250.0
_MAX_SM_MHZ = 2100
_MAX_MEM_MHZ = 10500
_STATIC_W = 20.0
_K_W = 250.0
_V_MAX = 0.85
_AMBIENT_C = 24.0
_RTH_C_PER_W = 0.30
_HEAT_CAPACITY_J_PER_C = 150.0
_THROTTLE_TEMP_C = 78.0
_HARD_TEMP_C = 88.0

_DOMAIN_UTIL = {"ai": 0.86, "proviz": 0.96, "gaming": 0.75, "idle": 0.03}


class SyntheticGpu:
    """Deterministic, physically plausible GPU emulator.

    Power follows ``P = P_static + k * (f/f_max) * (V(f)/V_max)^2 * util``.
    When that exceeds the configured cap the SM clock is throttled until it
    fits, and temperature follows a first-order soak curve with an additional
    thermal throttle above 78 C.  Per-sample measurement noise is seeded.
    """

    def __init__(self, seed: int = 42, power_limit_w: float = _PLIM_DEFAULT_W) -> None:
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._power_limit_w = min(max(power_limit_w, _PLIM_MIN_W), _PLIM_MAX_W)
        self._default_power_limit_w = _PLIM_DEFAULT_W
        self._locked: tuple[int, int] | None = None
        self._domain = "idle"
        self._util = _DOMAIN_UTIL["idle"]
        self._temp_c = _AMBIENT_C + 2.0
        self._last_t_ns = time.monotonic_ns()
        self._last_power_w = self._idle_power()
        self._last_sm_mhz = 300.0
        self._last_mem_mhz = 405.0

    # -- model ------------------------------------------------------------ #
    @staticmethod
    def _voltage(clock_mhz: float) -> float:
        return 0.68 + 0.16 * (clock_mhz / _MAX_SM_MHZ)

    def _power_at(self, clock_mhz: float, util: float) -> float:
        f = max(clock_mhz, 0.0) / _MAX_SM_MHZ
        v = self._voltage(clock_mhz) / _V_MAX
        return _STATIC_W + _K_W * f * (v**2) * util

    def _idle_power(self) -> float:
        return _STATIC_W + self._rng.uniform(-0.4, 0.4)

    def _thermal_clock_limit(self, temp_c: float) -> float:
        if temp_c <= _THROTTLE_TEMP_C:
            return float(_MAX_SM_MHZ)
        over = temp_c - _THROTTLE_TEMP_C
        frac = max(0.30, 1.0 - 0.025 * over)
        if temp_c >= _HARD_TEMP_C:
            frac = 0.30
        return _MAX_SM_MHZ * frac

    def _solve_clock_for_cap(self, cap_w: float, util: float, f_ceiling: float) -> float:
        """Largest clock whose modelled power does not exceed the cap."""
        if self._power_at(f_ceiling, util) <= cap_w:
            return f_ceiling
        lo, hi = 100.0, f_ceiling
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self._power_at(mid, util) <= cap_w:
                lo = mid
            else:
                hi = mid
        return lo

    def _requested_clock(self) -> float:
        if self._locked is not None and self._locked[1] > 0:
            return float(min(self._locked[1], _MAX_SM_MHZ))
        return float(_MAX_SM_MHZ)

    def _effective_clock(self) -> tuple[float, float]:
        """Return (effective SM clock, unthrottled-but-thermally-limited clock)."""
        util = self._util
        f_req = self._requested_clock()
        f_thermal = min(f_req, self._thermal_clock_limit(self._temp_c))
        f_eff = self._solve_clock_for_cap(self._power_limit_w, util, f_thermal)
        return f_eff, f_thermal

    def _integrate_temperature(self, power_w: float) -> None:
        now = time.monotonic_ns()
        dt = min(max((now - self._last_t_ns) / 1e9, 0.0), 2.0)
        self._last_t_ns = now
        if dt <= 0:
            return
        q = power_w - (self._temp_c - _AMBIENT_C) / _RTH_C_PER_W
        self._temp_c += (q / _HEAT_CAPACITY_J_PER_C) * dt
        self._temp_c = max(_AMBIENT_C, min(self._temp_c, 110.0))

    # -- activity hooks (not part of the GpuDevice protocol) -------------- #
    def set_activity(self, domain: str) -> None:
        """Tell the emulator which workload class is currently running."""
        if domain not in _DOMAIN_UTIL:
            domain = "idle"
        with self._lock:
            self._domain = domain
            self._util = _DOMAIN_UTIL[domain]
        logger.debug("synthetic GPU activity -> %s (util %.0f%%)", domain, self._util * 100)

    def compute_scale(self, domain: str | None = None) -> float:
        """Throughput scale vs. an unconstrained card, for workload pacing.

        This is what makes the synthetic sweeps physically meaningful: a lower
        power cap throttles the SM clock, and the workload's modelled service
        rate falls with it.  The work itself is still really executed.
        """
        with self._lock:
            util = _DOMAIN_UTIL.get(domain or self._domain, self._util)
            f_req = self._requested_clock()
            f_thermal = min(f_req, self._thermal_clock_limit(self._temp_c))
            f_eff = self._solve_clock_for_cap(self._power_limit_w, util, f_thermal)
            scale = f_eff / _MAX_SM_MHZ
        if (domain or self._domain) == "ai":
            # Inference is partly memory bound: throughput degrades more slowly.
            scale = 0.55 + 0.45 * scale
        return min(1.0, max(0.05, scale))

    def current_power_w(self) -> float:
        """Return the most recently measured power."""
        with self._lock:
            return self._last_power_w

    def instant_power_w(self) -> float:
        """Modelled board power *now*, without waiting for the next NVML read.

        The external sensor observes the rail continuously, so coupling it to
        the last 20 Hz telemetry value would add up to a sample of latency at
        every transition.  This is the physical value the shunt would see.
        """
        with self._lock:
            if self._domain == "idle":
                return _STATIC_W
            f_eff, _ = self._effective_clock()
            return self._power_at(f_eff, self._util)

    def current_clock_mhz(self) -> float:
        """Return the most recent SM clock in MHz."""
        with self._lock:
            return self._last_sm_mhz

    # -- protocol --------------------------------------------------------- #
    def info(self) -> dict[str, Any]:
        """Return device identity and backend information."""
        return {
            "source": "synthetic",
            "backend": "synthetic",
            "name": _FAKE_NAME,
            "driver_version": "synthetic-1.0",
            "index": 0,
            "uuid": "GPU-SYNTHETIC-0000",
            "model": {
                "static_power_w": _STATIC_W,
                "dynamic_coefficient": _K_W,
                "max_sm_mhz": _MAX_SM_MHZ,
                "max_mem_mhz": _MAX_MEM_MHZ,
                "thermal_rth_c_per_w": _RTH_C_PER_W,
                "heat_capacity_j_per_c": _HEAT_CAPACITY_J_PER_C,
            },
        }

    def read_telemetry(self) -> dict[str, Any]:
        """Read one telemetry sample from the device."""
        with self._lock:
            f_eff, _ = self._effective_clock()
            util = self._util
            if self._domain == "idle":
                base_power = self._idle_power()
            else:
                base_power = self._power_at(f_eff, util)
            # Board-level noise: roughly 1.2% of reading plus a fixed floor.
            noisy_power = base_power + self._rng.gauss(0.0, 0.012 * base_power + 0.5)
            noisy_power = max(3.0, min(noisy_power, self._power_limit_w * 1.02 + 5.0))
            self._integrate_temperature(base_power)
            self._last_power_w = noisy_power
            self._last_sm_mhz = f_eff * (1.0 + self._rng.gauss(0.0, 0.004))
            self._last_mem_mhz = (
                float(min(self._locked[1], _MAX_MEM_MHZ))
                if self._locked is not None and self._locked[1] > 0
                else float(_MAX_MEM_MHZ)
            )
            temp = self._temp_c + self._rng.gauss(0.0, 0.12)
            return {
                "timestamp_ns": time.monotonic_ns(),
                "power_w": round(noisy_power, 4),
                "sm_clock_mhz": int(max(100.0, self._last_sm_mhz)),
                "mem_clock_mhz": int(self._last_mem_mhz),
                "temp_c": round(temp, 3),
                "util_pct": round(util * 100.0 + self._rng.gauss(0.0, 1.0), 2),
                "power_limit_w": round(self._power_limit_w, 3),
            }

    def set_power_limit(self, watts: float) -> float:
        """Apply a board power limit and return the applied value."""
        applied = min(max(float(watts), _PLIM_MIN_W), _PLIM_MAX_W)
        with self._lock:
            self._power_limit_w = applied
        if applied != float(watts):
            logger.warning(
                "synthetic: requested %.0f W outside [%.0f, %.0f] W; clamped to %.0f W",
                watts,
                _PLIM_MIN_W,
                _PLIM_MAX_W,
                applied,
            )
        logger.info("synthetic GPU power limit set to %.0f W", applied)
        return applied

    def set_locked_clocks(self, min_mhz: int, max_mhz: int) -> None:
        """Lock graphics clocks to the given range."""
        if min_mhz <= 0 and max_mhz <= 0:
            self.reset_clocks()
            return
        lo = min_mhz if min_mhz > 0 else max_mhz
        hi = max_mhz if max_mhz > 0 else min_mhz
        lo = max(100, min(int(lo), _MAX_SM_MHZ))
        hi = max(100, min(int(hi), _MAX_SM_MHZ))
        with self._lock:
            self._locked = (lo, hi)
        logger.info("synthetic GPU clocks locked to %d-%d MHz", lo, hi)

    def reset_clocks(self) -> None:
        """Release any clock lock and restore defaults."""
        with self._lock:
            self._locked = None
        logger.info("synthetic GPU clocks reset to default")

    def locked_clocks(self) -> tuple[int, int] | None:
        """Return the active synthetic clock lock, if any."""
        with self._lock:
            return self._locked

    def reset_power_limit(self) -> None:
        """Restore the driver default board power limit."""
        self.set_power_limit(self._default_power_limit_w)

    def supported_power_limit_range(self) -> tuple[float, float]:
        """Return the enforceable (min, max) board power limit."""
        return (_PLIM_MIN_W, _PLIM_MAX_W)

    def temperature_c(self) -> float:
        """Return the modelled GPU temperature."""
        with self._lock:
            return self._temp_c


def nvml_available() -> bool:
    """Best-effort check for a usable NVIDIA GPU without raising."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False

