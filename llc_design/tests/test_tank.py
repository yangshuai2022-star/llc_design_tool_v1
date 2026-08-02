import math

import pytest

from llc_design.core.operating_point import solve_operating_point
from llc_design.core.spec import LLCDesignSpec
from llc_design.core.tank import design_tank, gain


def test_baseline_tank_synthesis():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    assert tank.lr_h * 1e6 == pytest.approx(23.7811, rel=2e-4)
    assert tank.cr_f * 1e9 == pytest.approx(106.5145, rel=2e-4)
    assert tank.lm_h * 1e6 == pytest.approx(118.9054, rel=2e-4)


def test_gain_is_unity_at_series_resonance():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    assert gain(tank, tank.fr_hz, tank.rac_nom_ohm) == pytest.approx(1.0, abs=1e-12)


def test_nominal_frequency_and_currents():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = solve_operating_point(spec, tank, 400.0, 1.0)
    assert op.switching_frequency_hz / 1e3 == pytest.approx(99.689, rel=3e-4)
    assert op.input_phase_deg > 20.0
    assert op.secondary_current_rms_a == pytest.approx(62.87, rel=0.01)
    assert op.resonant_current_rms_a < 11.0


def test_hold_up_endpoint_remains_on_inductive_branch():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = solve_operating_point(spec, tank, 300.0, 1.0)
    assert spec.minimum_frequency_hz <= op.switching_frequency_hz <= spec.maximum_frequency_hz
    assert op.input_phase_deg >= spec.minimum_inductive_angle_deg
    assert op.required_gain > 1.3
