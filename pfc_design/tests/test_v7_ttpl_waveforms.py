from dataclasses import replace
import numpy as np

from pfc_design.control import PFCControlLabConfig
from pfc_design.control.waveforms import build_pfc_switching_waveforms, simulate_pfc_line_cycle


def _fast_cfg():
    return replace(
        PFCControlLabConfig(),
        waveform_line_cycles=3,
        waveform_integration_rate_hz=250e3,
        switching_samples_per_cycle=200,
    )


def test_ttpl_line_cycle_distinguishes_rectified_inductor_current_and_signed_grid_current():
    result = simulate_pfc_line_cycle(_fast_cfg())
    i_l = result.signals["i_inductor"]
    i_ac = result.signals["i_input_signed"]
    vac = result.signals["vac"]
    assert np.all(i_l >= -1e-12)
    assert np.any(i_ac < 0.0)
    assert np.any(i_ac > 0.0)
    # Signed grid current follows line polarity away from exact zero crossings.
    # Check the settled final line cycle, not the startup interval.
    samples_per_cycle = int(round(250e3 / 50.0))
    vac = vac[-samples_per_cycle:]
    i_ac = i_ac[-samples_per_cycle:]
    mask = (np.abs(vac) > 20.0) & (np.abs(i_ac) > 1e-6)
    assert np.mean(np.sign(i_ac[mask]) == np.sign(vac[mask])) > 0.99


def test_ttpl_metrics_include_strict_pf_thd_and_harmonics():
    m = simulate_pfc_line_cycle(_fast_cfg()).metrics
    assert 0.0 < m.power_factor <= 1.01
    assert 0.0 < m.distortion_factor <= 1.01
    assert m.fundamental_current_rms_a > 0.0
    assert len(m.harmonic_orders) == len(m.harmonic_current_rms_a) == 25
    assert m.harmonic_orders[0] == 1


def test_ttpl_switching_waveform_is_derived_from_line_cycle_workpoint():
    cfg = _fast_cfg()
    line = simulate_pfc_line_cycle(cfg)
    sw = build_pfc_switching_waveforms(cfg, line_cycle=line, line_angle_deg=60.0)
    assert sw.source_time_s is not None
    assert sw.source_time_s >= line.time_s[-1] - 1.0 / cfg.power_stage.line_frequency_hz - 2e-5
    assert len(sw.time_s) == cfg.switching_cycles * cfg.switching_samples_per_cycle
    assert "pwm_state_code" in sw.signals
    assert "input_current_signed" in sw.signals
