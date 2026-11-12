# WattBench rig setup

This document covers the external power-sense chain: the shunt, the INA219
breakout, the Arduino logger, and the host configuration. It is written for a
bench where the GPU is powered from a single 12 V PCIe feed that can be
interrupted for measurement.

> **The synthetic backend exists so the software can be evaluated without any
> of this hardware.** You do not need a rig to run WattBench end to end. When a
> rig is present, set `gpu.mode: hardware` and `sensor.mode: hardware` in
> `config/rig.yaml`; when it is not, leave both on `auto`.

## Safety first

- Power the machine down and unplug the PSU before breaking the 12 V rail.
  Never interrupt a live PCIe power lead with a multimeter or loose wire.
- Use a **four-terminal (Kelvin) shunt**: the heavy current path is soldered or
  bolted; the two sense leads tap the shunt body separately. A breadboard or
  alligator-clip current path will add contact resistance and heat.
- Size the shunt for the rail: `P = I²R`. A 5 mΩ, 5 W shunt is 2 W at 20 A and
  drops 100 mV, which is inside the INA219's ±320 mV range. Do not use the
  common 0.1 Ω breakout resistor on a 20 A rail.
- The INA219 is a **26 V maximum** part. It is suitable for the 12 V rail only,
  not the 48 V / 12VHPWR input. Check the actual bus voltage with the logger
  before connecting the GPU.
- Keep the USB ground and the PSU ground common, but use an isolated USB
  adapter if the host and the bench supply can be at different earth
  potentials. Never create a ground loop through the oscilloscope and USB.
- Fuse the measurement branch and use shrouded connectors. A 12 V rail can
  deliver hundreds of amps into a short.
- Do not exceed the GPU's rated current. WattBench only *observes* the rail; it
  does not protect the hardware.

## Wiring

```
 PSU 12 V ----[ shunt ]----+---- GPU PCIe 12 V
                  |        |
              Vin+ Vin-    |
                  |        |
             INA219 Vbus --+
                  |
             INA219 GND ---- PSU GND / host GND (common)
             INA219 SDA ---- Arduino SDA (A4 on Uno/Nano)
             INA219 SCL ---- Arduino SCL (A5 on Uno/Nano)
             INA219 VCC ---- 3.3 V or 5 V (breakout dependent)
```

- `Vin+` is the PSU side and `Vin-` is the GPU side. The INA219 measures the
  differential voltage between them and the bus voltage on `Vin-`.
- Put the shunt on the **12 V feed**, not in the ground return. Ground-return
  shunts work but move the GPU ground above the host ground, which can upset
  PCIe signalling.
- Keep the sense pair twisted and short (< 10 cm). Route them away from the
  GPU's VRM and fan PWM.
- If the GPU uses multiple PCIe cables, measure the total: either one shunt
  rated for the summed current, or one INA219 per cable with a second serial
  port. WattBench currently consumes one sensor stream; a second sensor is a
  natural extension, not a supported configuration yet.

## Firmware

Flash [`firmware/ina219_logger.ino`](../firmware/ina219_logger.ino) with the
Adafruit INA219 library. The sketch prints:

```
t_ms,bus_v,shunt_mv
```

at 100 Hz. At 115200 baud that is about 2.5 kB/s, well within USB-serial
throughput. The host tolerates partial lines, wrong field counts, and USB
reconnects; malformed records are counted and skipped.

To verify the stream from a shell:

```bash
python -m serial.tools.miniterm /dev/ttyUSB0 115200
# or
stty -F /dev/ttyUSB0 115200 raw -echo && head -n 5 /dev/ttyUSB0
```

Expect plausible values: 11.8–12.3 V bus, and a shunt reading that rises with
GPU load. At idle a 5 mΩ shunt should read a few millivolts (tens of watts).

## Host configuration

```yaml
sensor:
  mode: hardware        # auto falls back to synthetic when the port is absent
  port: /dev/ttyUSB0    # macOS: /dev/cu.usbserial-*
  baud: 115200
  shunt_ohms: 0.005     # measure this with a 4-wire meter; do not trust the label
  units: mv             # firmware prints shunt millivolts
  sample_rate_hz: 50
```

Find the port on Linux with `dmesg | tail` or `ls /dev/ttyUSB*`; on macOS use
`ls /dev/cu.*`. Add the user to `dialout` on Linux:

```bash
sudo usermod -aG dialout "$USER"   # then log out and back in
```

## Power-limit and clock control

NVML reads telemetry without special privileges. Changing the board power
limit and locking clocks goes through `nvidia-smi`, which usually needs root:

```bash
sudo nvidia-smi -i 0 -pl 200          # board power limit, watts
sudo nvidia-smi -i 0 -lgc 1700,1700   # lock graphics clock, MHz
sudo nvidia-smi -i 0 -rgc             # release the clock lock
```

WattBench logs the exact command it runs at INFO level. If a change is denied,
it raises an actionable error rather than silently measuring the wrong state.
On some data-centre cards, persistence mode and a supported vBIOS are required.
See the [NVIDIA power-limit documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
WattBench always restores the default limit and releases clocks in a `finally`
block, including on Ctrl-C.

## Verification checklist

1. `wattbench devices` reports `hardware` for both backends, a real GPU name,
   and the correct power-limit range.
2. With the workload idle, the external sensor reads within a few watts of
   `nvidia-smi --query-gpu=power.draw`.
3. During a synthetic reference load, `percent_bias` from `wattbench analyze`
   is under the configured threshold (default 5%).
4. The fitted clock offset is stable across cells (tens of milliseconds) and
   drift is under a few hundred ppm. A large drift suggests a marginal USB
   bridge or a sensor clock that is not crystal-referenced.
5. A full sweep restores the default power limit; verify with
   `nvidia-smi --query-gpu=power.limit,clocks.gr --format=csv`.
