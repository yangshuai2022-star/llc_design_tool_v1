"""DC-link capacitor hold-up calculations under constant-power discharge."""

from __future__ import annotations

import math


def hold_end_voltage_v(capacitance_f: float, start_voltage_v: float,
                       input_power_w: float, hold_time_s: float) -> float:
    radicand = start_voltage_v**2 - 2.0 * input_power_w * hold_time_s / capacitance_f
    return math.sqrt(max(radicand, 0.0))


def hold_time_s(capacitance_f: float, start_voltage_v: float,
                end_voltage_v: float, input_power_w: float) -> float:
    return capacitance_f * (start_voltage_v**2 - end_voltage_v**2) / (2.0 * input_power_w)


def required_capacitance_f(start_voltage_v: float, end_voltage_v: float,
                           input_power_w: float, hold_time_s_value: float) -> float:
    denominator = start_voltage_v**2 - end_voltage_v**2
    if denominator <= 0:
        raise ValueError("start voltage must exceed end voltage")
    return 2.0 * input_power_w * hold_time_s_value / denominator
