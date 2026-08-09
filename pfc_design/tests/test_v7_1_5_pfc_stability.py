from dataclasses import replace
import numpy as np

from pfc_design.control import (
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    simulate_pfc_line_cycle,
    tune_pfc_current_loop,
)
from pfc_design.vienna import (
    ViennaControlLabConfig,
    build_vienna_control_lab_analysis,
    build_vienna_switching_waveforms,
    simulate_vienna_line_cycle,
    validate_vienna_analysis,
    validate_vienna_line_cycle,
    validate_vienna_switching,
)


def test_current_loop_one_click_tuner_returns_stable_starting_point():
    cfg = PFCControlLabConfig()
    result = tune_pfc_current_loop(cfg)
    assert result.accepted
    assert result.controller.kp > 0.0
    assert result.controller.ti_s > 0.0
    assert result.nominal_phase_margin_deg is not None
    assert result.nominal_phase_margin_deg >= 52.0
    assert result.worst_phase_margin_deg is not None
    assert result.worst_phase_margin_deg >= 50.0
    assert result.worst_gain_margin_db is None or result.worst_gain_margin_db >= 10.0


def test_tuned_current_loop_stays_finite_in_ac_cycle_and_switching_view():
    base = PFCControlLabConfig()
    tune = tune_pfc_current_loop(base)
    cfg = replace(
        base,
        current_controller=tune.controller,
        frequency_points=320,
        waveform_line_cycles=4,
        waveform_integration_rate_hz=250e3,
        switching_samples_per_cycle=240,
    )
    analysis = build_pfc_control_lab_analysis(cfg)
    assert analysis.current_loop.margins.phase_margin_deg is not None
    assert analysis.current_loop.margins.phase_margin_deg > 45.0

    line = simulate_pfc_line_cycle(cfg)
    for key in ("i_inductor", "i_ref", "duty_total", "vbus"):
        assert np.all(np.isfinite(line.signals[key]))
    assert float(np.max(np.abs(line.signals["i_inductor"]))) < 10.0 * cfg.power_stage.current_reference_a + 100.0

    sw = build_pfc_switching_waveforms(cfg, line_cycle=line, line_angle_deg=60.0)
    i = np.asarray(sw.signals["inductor_current"])
    assert np.all(np.isfinite(i))
    assert "inductor_current_average" in sw.signals
    assert "current_deviation" in sw.signals
    # No old per-period reset discontinuity: a PWM-boundary step must be of the
    # same order as normal display integration steps, not a full ripple jump.
    spp = cfg.switching_samples_per_cycle
    boundary_steps = [abs(i[k] - i[k - 1]) for k in range(spp, len(i), spp)]
    typical = float(np.median(np.abs(np.diff(i))))
    assert max(boundary_steps, default=0.0) <= max(5.0 * typical, 0.25)


def _vienna_gui_equivalent_fast_config():
    base = ViennaControlLabConfig()
    # Mirrors the GUI defaults, including its explicit 9 us current delay,
    # while shortening only plotting/test resolution.
    fw = replace(base.firmware, current_computation_delay_s=9e-6, pwm_update_delay_s=0.0)
    return replace(
        base,
        firmware=fw,
        frequency_points=320,
        waveform_line_cycles=4,
        waveform_integration_rate_hz=250e3,
        switching_samples_per_cycle=240,
    )


def test_vienna_gui_calculation_pipeline_is_finite_and_complete():
    cfg = _vienna_gui_equivalent_fast_config()
    analysis = build_vienna_control_lab_analysis(cfg)
    validate_vienna_analysis(analysis)
    line = simulate_vienna_line_cycle(cfg)
    validate_vienna_line_cycle(line)
    sw = build_vienna_switching_waveforms(cfg, line, line_angle_deg=cfg.switching_line_angle_deg)
    validate_vienna_switching(sw)
    assert np.isfinite(line.metrics.overall_power_factor)
    assert sw.source_time_s is not None


def test_vienna_all_gui_consumed_waveform_keys_are_present():
    cfg = _vienna_gui_equivalent_fast_config()
    line = simulate_vienna_line_cycle(cfg)
    sw = build_vienna_switching_waveforms(cfg, line, line_angle_deg=30.0)
    line_keys = {
        "va", "vb", "vc", "va_meas", "vb_meas", "vc_meas",
        "ia", "ib", "ic", "ia_ref", "ib_ref", "ic_ref",
        "mod_a", "mod_b", "mod_c", "duty_a", "duty_b", "duty_c",
        "vdc", "vdc_plus", "vdc_minus", "vdc_delta", "vdc_measured",
        "gcmd", "vloop", "balance_output", "midpoint_current",
        "input_power_total", "load_current", "bus_series_current", "sector",
        "third_harmonic_injection",
    }
    assert line_keys.issubset(line.signals)
    sw_keys = {"midpoint_current", "upper_cap_current", "lower_cap_current"}
    for ph in "abc":
        sw_keys |= {f"gate_{ph}", f"vconv_{ph}", f"current_{ph}", f"upper_diode_{ph}", f"lower_diode_{ph}", f"duty_{ph}"}
    assert sw_keys.issubset(sw.signals)
