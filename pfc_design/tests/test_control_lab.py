"""Regression tests for PFC Control Lab nested-loop modelling."""

from dataclasses import replace
import math

import numpy as np

from llc_design.control.digital_loop import TwoP2ZControllerConfig
from pfc_design.control import (
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    sense_frequency_response,
    simulate_pfc_line_cycle,
)


def test_default_control_lab_builds_two_distinct_loops():
    result = build_pfc_control_lab_analysis(PFCControlLabConfig())
    assert result.current_loop.name.startswith("PFC current")
    assert result.voltage_loop.name.startswith("PFC bus-voltage")
    assert result.current_loop.controller.sample_time_s == 20e-6
    assert result.voltage_loop.controller.sample_time_s == 100e-6
    assert result.current_loop.margins.critical_gain_crossover_hz is not None
    assert result.voltage_loop.margins.critical_gain_crossover_hz is not None


def test_amc_is_not_modelled_as_third_feedback_loop():
    result = build_pfc_control_lab_analysis(PFCControlLabConfig())
    assert "amc_vff" in result.voltage_loop.responses
    assert "open_current" in result.current_loop.responses
    assert "open_voltage" in result.voltage_loop.responses
    assert not hasattr(result, "amc_loop")


def test_half_wave_observer_is_not_part_of_pfc_control_lab():
    result = build_pfc_control_lab_analysis(PFCControlLabConfig())
    names = " ".join(
        list(result.current_loop.responses) + list(result.voltage_loop.responses)
    ).lower()
    assert "observer" not in names
    assert "nco" not in names


def test_external_current_filter_reduces_high_frequency_gain():
    cfg = PFCControlLabConfig()
    f = np.asarray([10.0, 10_000.0])
    response = sense_frequency_response(cfg.current_sense, f).total
    assert abs(response[1]) < abs(response[0])


def test_vac_filter_has_finite_50hz_phase():
    cfg = PFCControlLabConfig()
    response = sense_frequency_response(cfg.vac_sense, np.asarray([50.0, 60.0])).total
    assert np.all(np.isfinite(response))
    assert abs(np.angle(response[0])) < math.radians(20.0)


def test_current_gain_schedule_changes_with_line_angle():
    base = PFCControlLabConfig()
    low = build_pfc_control_lab_analysis(replace(
        base, power_stage=replace(base.power_stage, line_angle_deg=10.0)))
    high = build_pfc_control_lab_analysis(replace(
        base, power_stage=replace(base.power_stage, line_angle_deg=90.0)))
    assert low.operating_point.indu_comp <= high.operating_point.indu_comp
    assert 0.7 <= low.operating_point.indu_comp <= 1.0
    assert 0.7 <= high.operating_point.indu_comp <= 1.0


def test_two_p_two_z_sign_convention_is_preserved():
    cfg = PFCControlLabConfig(
        current_controller=TwoP2ZControllerConfig(
            b0=0.1, b1=-0.05, b2=0.01,
            a1=-1.2, a2=0.3,
            sample_time_s=20e-6,
            output_min=-2.0, output_max=0.98,
        )
    )
    result = build_pfc_control_lab_analysis(cfg)
    tf = result.current_loop.controller
    assert np.allclose(tf.numerator, [0.1, -0.05, 0.01])
    assert np.allclose(tf.denominator, [1.0, -1.2, 0.3])
    assert "-a1" not in tf.difference_equation()  # Numeric equation is emitted.


def test_line_cycle_waveforms_and_metrics_are_finite():
    result = simulate_pfc_line_cycle(PFCControlLabConfig(waveform_line_cycles=4))
    assert len(result.time_s) > 1000
    assert result.metrics.input_voltage_rms_v > 100.0
    assert result.metrics.input_current_rms_a > 0.0
    assert math.isfinite(result.metrics.power_factor)
    assert math.isfinite(result.metrics.current_thd_percent)
    assert np.all(np.isfinite(result.signals["vbus"]))


def test_switching_waveform_has_expected_period_and_states():
    cfg = PFCControlLabConfig()
    result = build_pfc_switching_waveforms(cfg, line_angle_deg=60.0, samples=500)
    assert len(result.time_s) == 500
    assert result.time_s[-1] < 1.0 / cfg.power_stage.switching_frequency_hz
    assert set(np.unique(result.signals["hf_high_gate"])) <= {0.0, 1.0}
    assert np.max(result.signals["switch_node_voltage"]) == cfg.power_stage.bus_voltage_v



def test_configured_switching_waveform_spans_multiple_cycles():
    cfg = PFCControlLabConfig(
        switching_cycles=3,
        switching_samples_per_cycle=240,
    )
    result = build_pfc_switching_waveforms(cfg, line_angle_deg=45.0)
    period = 1.0 / cfg.power_stage.switching_frequency_hz
    assert len(result.time_s) == 3 * 240
    assert 2.9 * period < result.time_s[-1] < 3.0 * period
    assert set(np.unique(result.signals["cycle_index"])) == {0.0, 1.0, 2.0}
    assert "high_side_current" in result.signals
    assert "bus_cap_current" in result.signals


def test_line_cycle_exposes_full_control_and_power_waveform_set():
    result = simulate_pfc_line_cycle(PFCControlLabConfig(waveform_line_cycles=4))
    required = {
        "vac_rms_estimate", "i_measured_signed", "vbus_ripple",
        "voltage_error", "effective_duty_min", "minimum_pulse_active",
        "load_current", "boost_output_current", "amc_update_strobe",
        "voltage_update_strobe",
    }
    assert required <= set(result.signals)
    for key in required:
        assert len(result.signals[key]) == len(result.time_s)
        assert np.all(np.isfinite(result.signals[key]))

def test_minimum_pulse_constraint_changes_waveform_duty_floor():
    base = PFCControlLabConfig(waveform_line_cycles=3)
    constrained = replace(
        base,
        power_stage=replace(
            base.power_stage,
            minimum_effective_pulse_s=5e-6,
        ),
    )
    result = simulate_pfc_line_cycle(constrained)
    expected = 5e-6 * constrained.power_stage.switching_frequency_hz
    assert result.metrics.duty_min >= expected - 1e-9
