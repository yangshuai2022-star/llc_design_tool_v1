import math

import pytest

from llc_design.core.bus_capacitor import required_capacitance_f
from llc_design.core.operating_point import solve_operating_point
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology
from llc_design.core.tank import design_tank
from llc_design.core.waveform import output_capacitor_ripple
from llc_design.models.capacitors import (bus_capacitor_result,
                                          output_capacitor_result,
                                          resonant_capacitor_result)


def test_bus_capacitance_for_20ms_hold_up():
    capacitance = required_capacitance_f(400.0, 300.0, 3000.0 / 0.96, 20e-3)
    assert capacitance * 1e6 == pytest.approx(1785.714, rel=1e-5)


def test_output_capacitor_rms_matches_rectified_sine_identity():
    io = 3000.0 / 53.0
    result = output_capacitor_ripple(io, 100e3, 1500e-6, 1.5e-3, 0.5)
    expected = io * math.sqrt(math.pi**2 / 8.0 - 1.0)
    assert result.capacitor_current_rms_a == pytest.approx(expected, rel=2e-4)
    assert result.total_ripple_vpp < 0.2


def _nominal_op(spec):
    return solve_operating_point(spec, design_tank(spec), spec.vbus_nom_v, 1.0)


def test_full_bridge_resonant_cap_has_no_dc_bias():
    spec = LLCDesignSpec()
    result = resonant_capacitor_result(spec, design_tank(spec), _nominal_op(spec))
    assert result.voltage_dc_bias_v == 0.0
    assert result.voltage_peak_v == pytest.approx(
        math.sqrt(2.0) * result.voltage_rms_v)
    assert result.voltage_margin > 1.0


def test_half_bridge_resonant_cap_includes_vbus_over_two_dc_bias():
    spec = LLCDesignSpec(primary_topology=PrimaryTopology.HALF_BRIDGE,
                         primary_turns=15)
    result = resonant_capacitor_result(spec, design_tank(spec), _nominal_op(spec))
    assert result.voltage_dc_bias_v == pytest.approx(200.0)
    assert result.voltage_peak_v == pytest.approx(
        200.0 + math.sqrt(2.0) * result.voltage_rms_v)
    assert result.voltage_peak_v > result.voltage_rms_v


def test_resonant_cap_rating_margin_scales_with_rating():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = _nominal_op(spec)
    base = resonant_capacitor_result(spec, tank, op)
    low_rating = spec.clone(resonant_cap_voltage_rating_v=200.0)
    reduced = resonant_capacitor_result(low_rating, tank, op)
    assert reduced.voltage_margin < base.voltage_margin
    assert reduced.voltage_peak_v == base.voltage_peak_v


def test_output_cap_loss_matches_esr_and_budget_is_finite():
    spec = LLCDesignSpec()
    result = output_capacitor_result(spec, _nominal_op(spec))
    assert result.capacitor_loss_w == pytest.approx(
        result.capacitor_current_rms_a**2 * spec.output_cap_esr_ohm)
    assert math.isfinite(result.required_capacitance_f)
    assert result.total_ripple_vpp <= spec.output_ripple_limit_vpp


def test_bus_cap_meets_requested_hold_time_on_baseline():
    spec = LLCDesignSpec()
    result = bus_capacitor_result(spec)
    assert result.hold_time_to_limit_s >= spec.requested_hold_time_s
    assert result.predicted_end_voltage_v >= spec.vbus_hold_end_v
    assert result.energy_end_j < result.energy_start_j
