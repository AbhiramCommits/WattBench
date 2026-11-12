"""INA219 serial line parsing and shunt scaling."""

from __future__ import annotations

import pytest

from wattbench.devices.power_sensor import parse_ina219_line


def test_valid_shunt_millivolt_line() -> None:
    parsed = parse_ina219_line("1000,12.0,50.0", shunt_ohms=0.005, units="mv")
    assert parsed is not None
    ts, bus_v, current_a, power_w = parsed
    assert ts == 1_000_000_000
    assert bus_v == pytest.approx(12.0)
    assert current_a == pytest.approx(10.0)  # 50 mV / 5 mOhm
    assert power_w == pytest.approx(120.0)


def test_shunt_resistance_scaling() -> None:
    low = parse_ina219_line("10,12.0,50", shunt_ohms=0.005)
    high = parse_ina219_line("10,12.0,50", shunt_ohms=0.010)
    assert low is not None and high is not None
    assert low[2] == pytest.approx(10.0)
    assert high[2] == pytest.approx(5.0)


def test_amps_units() -> None:
    parsed = parse_ina219_line("25,12.1,3.5", units="a")
    assert parsed is not None
    assert parsed[2] == pytest.approx(3.5)
    assert parsed[3] == pytest.approx(42.35)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "# comment",
        "1000,12.0",  # partial
        "1000,12.0,50,extra",  # wrong field count
        "abc,12.0,50",  # garbled timestamp
        "1000,not-a-voltage,50",
        "1000,12.0,not-a-current",
        "1000,nan,50",
        "1000,120.0,50",  # bus voltage implausible
        "1000,12.0,9000",  # current implausible
    ],
)
def test_bad_lines_are_rejected(line: str) -> None:
    assert parse_ina219_line(line, shunt_ohms=0.005) is None


def test_invalid_shunt_raises() -> None:
    with pytest.raises(ValueError):
        parse_ina219_line("1,12,10", shunt_ohms=0.0)


def test_tolerates_leading_trailing_whitespace() -> None:
    parsed = parse_ina219_line(" 1000 , 12.0 , 10 \r\n", shunt_ohms=0.005)
    assert parsed is not None
    assert parsed[2] == pytest.approx(2.0)
