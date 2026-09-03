import math

import pytest

from llc_design.core.waveform import output_capacitor_ripple


def test_ripple_loss_matches_esr_model():
    current, frequency, capacitance, esr, limit = 56.6, 100e3, 1500e-6, 1.5e-3, 0.5
    result = output_capacitor_ripple(current, frequency, capacitance, esr, limit)
    assert result.capacitor_loss_w == pytest.approx(
        result.capacitor_current_rms_a**2 * esr)
    assert math.isfinite(result.required_capacitance_f)
    assert result.total_ripple_vpp <= limit
    assert result.charge_pp_c > 0.0


def test_esr_dominated_budget_reports_unmeetable_capacitance():
    result = output_capacitor_ripple(56.6, 100e3, 1500e-6, 0.2, 0.05)
    assert math.isinf(result.required_capacitance_f)
    assert result.esr_ripple_vpp > 0.05


def test_higher_capacitance_lowers_capacitive_ripple():
    base = output_capacitor_ripple(56.6, 100e3, 1500e-6, 1.5e-3, 0.5)
    bigger = output_capacitor_ripple(56.6, 100e3, 3000e-6, 1.5e-3, 0.5)
    assert bigger.capacitive_ripple_vpp < base.capacitive_ripple_vpp


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        output_capacitor_ripple(0.0, 100e3, 1500e-6, 1.5e-3, 0.5)
