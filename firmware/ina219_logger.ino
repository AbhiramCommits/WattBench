/*
 * WattBench INA219 logger
 * -----------------------
 * Prints one newline-delimited record per sample:
 *
 *     t_ms,bus_v,shunt_mv
 *
 * where t_ms is Arduino millis() (the host re-anchors it to its own monotonic
 * clock), bus_v is the 12 V PCIe rail voltage, and shunt_mv is the raw
 * differential voltage across the shunt in millivolts.  The host divides by
 * the *actual* shunt resistance, so record the resistor value in rig.yaml.
 *
 * Wiring: INA219 Vin+ / Vin- in series with the 12 V feed to the GPU rail,
 * shunt across Vin+/Vin-.  Vbus to the load side, GND common with the PSU.
 * INA219 is a 26 V-max part: never put it on the 48 V / 12VHPWR side.
 *
 * Library: Adafruit INA219 (Library Manager).  Board: any 3.3/5 V Arduino.
 */

#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

static const unsigned long PERIOD_MS = 10;   // 100 Hz; raise if serial overruns
static unsigned long last_ms = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { /* wait briefly for native USB */ }

  Wire.begin();
  if (!ina219.begin()) {
    Serial.println("# INA219 not found; check SDA/SCL and address 0x40");
    while (true) { delay(1000); }
  }
  // 32 V bus range, /8 gain (320 mV full-scale shunt).  With a 5 mOhm shunt
  // this covers +/-64 A, far beyond the 12 V rail's 25 A design limit.
  ina219.setCalibration_32V_2A();
  Serial.println("# t_ms,bus_v,shunt_mv");
}

void loop() {
  unsigned long now = millis();
  if (now - last_ms < PERIOD_MS) return;
  last_ms = now;

  float bus_v = ina219.getBusVoltage_V();
  float shunt_mv = ina219.getShuntVoltage_mV();

  Serial.print(now);
  Serial.print(',');
  Serial.print(bus_v, 4);
  Serial.print(',');
  Serial.println(shunt_mv, 4);
}
