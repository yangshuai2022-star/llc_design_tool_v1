from dataclasses import replace
import numpy as np

from pfc_design.vienna import (
    ViennaControlLabConfig,
    build_vienna_control_lab_analysis,
    build_vienna_switching_waveforms,
    simulate_vienna_line_cycle,
)


def _fast_cfg():
    return replace(
        ViennaControlLabConfig(),
        frequency_points=220,
        waveform_line_cycles=4,
        waveform_integration_rate_hz=250e3,
        switching_samples_per_cycle=200,
        initial_midpoint_imbalance_v=6.0,
    )


def test_vienna_has_three_independent_open_loop_stability_results():
    a = build_vienna_control_lab_analysis(_fast_cfg())
    assert "open_current" in a.current_loop.responses
    assert "open_voltage" in a.voltage_loop.responses
    assert "open_balance" in a.balance_loop.responses
    assert a.current_loop.margins.phase_margin_deg is not None
    assert a.balance_loop.margins.phase_margin_deg is not None


def test_vienna_line_cycle_contains_abc_split_bus_and_sector_signals():
    line = simulate_vienna_line_cycle(_fast_cfg())
    for key in ("va","vb","vc","ia","ib","ic","vdc_plus","vdc_minus","vdc_delta","sector"):
        assert key in line.signals
    # Three-phase voltages sum to approximately zero.
    total = line.signals["va"] + line.signals["vb"] + line.signals["vc"]
    assert np.max(np.abs(total)) < 1e-8
    assert np.min(line.signals["sector"]) >= 1
    assert np.max(line.signals["sector"]) <= 6


def test_vienna_midpoint_balance_reduces_initial_split_bus_error():
    cfg = _fast_cfg()
    line = simulate_vienna_line_cycle(cfg)
    initial = abs(line.signals["vdc_delta"][0])
    final_window = line.signals["vdc_delta"][-1000:]
    assert abs(np.mean(final_window)) < initial


def test_vienna_switching_workpoint_has_three_level_device_signals():
    cfg = _fast_cfg()
    line = simulate_vienna_line_cycle(cfg)
    sw = build_vienna_switching_waveforms(cfg, line, line_angle_deg=30.0)
    for ph in "abc":
        assert f"gate_{ph}" in sw.signals
        assert f"vconv_{ph}" in sw.signals
        assert f"upper_diode_{ph}" in sw.signals
        assert f"lower_diode_{ph}" in sw.signals
    assert "midpoint_current" in sw.signals


def test_vienna_zero_state_duty_matches_signed_modulation_without_min_pulse():
    cfg = _fast_cfg()
    line = simulate_vienna_line_cycle(cfg)
    sl = slice(-2000, None)
    for ph in "abc":
        expected = 1.0 - np.abs(line.signals[f"mod_{ph}"][sl])
        assert np.max(np.abs(expected - line.signals[f"duty_{ph}"][sl])) < 1e-9


def test_vienna_third_harmonic_injection_can_be_enabled_or_disabled():
    cfg = _fast_cfg()
    enabled = simulate_vienna_line_cycle(cfg)
    assert np.max(np.abs(enabled.signals["third_harmonic_injection"][-2000:])) > 1.0

    disabled_cfg = replace(
        cfg,
        firmware=replace(cfg.firmware, third_harmonic_injection_enabled=False),
    )
    disabled = simulate_vienna_line_cycle(disabled_cfg)
    assert np.max(np.abs(disabled.signals["third_harmonic_injection"])) == 0.0


def test_vienna_switching_is_reconstructed_from_final_settled_cycle_and_zero_duty():
    cfg = _fast_cfg()
    line = simulate_vienna_line_cycle(cfg)
    sw = build_vienna_switching_waveforms(cfg, line, line_angle_deg=42.0)
    assert sw.source_time_s is not None
    assert sw.source_time_s >= line.time_s[-1] - 1.0 / cfg.power_stage.line_frequency_hz - 2e-5
    for ph in "abc":
        # Center switch is ON during Vienna zero state; carrier discretization
        # gives at most one sample of duty error.
        mean_gate = float(np.mean(sw.signals[f"gate_{ph}"]))
        command = float(sw.signals[f"duty_{ph}"][0])
        assert abs(mean_gate - command) <= 1.5 / cfg.switching_samples_per_cycle


def test_vienna_sensor_mismatch_path_changes_measured_phase_channels():
    cfg = replace(
        _fast_cfg(),
        phase_current_gain_scale=(1.00, 1.03, 0.97),
        phase_current_offset_a=(0.0, 0.05, -0.03),
        phase_voltage_gain_scale=(1.00, 1.01, 0.99),
    )
    line = simulate_vienna_line_cycle(cfg)
    # Measurement mismatch is explicit and does not rely on GUI state.
    err_b = line.signals["ib_meas"] - line.signals["ib"]
    err_c = line.signals["ic_meas"] - line.signals["ic"]
    assert np.sqrt(np.mean(err_b[-1000:] ** 2)) > 0.02
    assert np.sqrt(np.mean(err_c[-1000:] ** 2)) > 0.02
