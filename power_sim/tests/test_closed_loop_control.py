import math
from pathlib import Path

import numpy as np

from power_control_tools import ControllerKind, DiscretizationMethod, design_controller, discretize_transfer_function
from power_sim import (
    ControllerLimitConfig,
    FirstOrderLLCPlant,
    LLCFMConfig,
    LLCFMMode,
    LLCFMRuntime,
    PWMCountMode,
    SamplerConfig,
    SamplerRuntime,
    StepProfile,
    ClosedLoopTiming,
    ClosedLoopScenario,
    analyze_linear_closed_loop,
    run_closed_loop,
)


def _pi(fs=40_000.0, kp=0.12, ti=0.0025):
    analog = design_controller(ControllerKind.PI, kp=kp, ti_s=ti)
    return discretize_transfer_function(analog, fs, DiscretizationMethod.TUSTIN)


def test_sampler_quantization_and_one_sample_delay():
    cfg = SamplerConfig(sample_rate_hz=40_000, gain=0.05, adc_bits=12, adc_min=0.0, adc_max=3.3, delay_samples=1)
    s = SamplerRuntime(cfg)
    first = s.sample(53.0)
    second = s.sample(54.0)
    assert first == 0.0
    assert abs(second - 53.0) < 0.03


def test_llc_up_down_tbprd_quantization_150khz():
    cfg = LLCFMConfig(
        mode=LLCFMMode.FREQUENCY_COMMAND,
        nominal_frequency_hz=100_000,
        minimum_frequency_hz=50_000,
        maximum_frequency_hz=250_000,
        tbclk_hz=120_000_000,
        count_mode=PWMCountMode.UP_DOWN,
    )
    step = LLCFMRuntime(cfg).map(150_000.0)
    assert step.tbprd == 400
    assert step.actual_frequency_hz == 150_000.0


def test_linear_fm_polarity_and_limits():
    cfg = LLCFMConfig(
        mode=LLCFMMode.LINEAR_FM,
        nominal_frequency_hz=100_000,
        kfm_hz_per_unit=-50_000,
        minimum_frequency_hz=70_000,
        maximum_frequency_hz=180_000,
        tbclk_hz=120_000_000,
    )
    rt = LLCFMRuntime(cfg)
    assert rt.map(0.1).frequency_command_hz == 95_000
    limited = rt.map(2.0)
    assert limited.frequency_command_hz == 70_000
    assert limited.saturated


def test_linear_closed_loop_checks_complete_closed_loop_not_controller_alone():
    fs = 40_000.0
    controller = _pi(fs, kp=0.05, ti=0.004)
    plant_model = FirstOrderLLCPlant(time_constant_s=0.001, frequency_gain_v_per_hz=-5e-4)
    plant = plant_model.discrete_transfer(fs)
    analysis = analyze_linear_closed_loop(controller, plant, modulator_gain=-20_000.0, delay_samples=1)
    # PI itself has z=1, but the complete feedback loop must be judged by its
    # closed-loop poles rather than rejecting the controller pole in isolation.
    assert analysis.stable
    assert analysis.max_pole_radius < 1.0
    assert analysis.phase_margin_deg is not None


def test_event_closed_loop_reference_step_converges_with_quantized_tbprd():
    fs = 40_000.0
    controller = _pi(fs, kp=0.06, ti=0.003)
    plant = FirstOrderLLCPlant(
        nominal_output_v=53.0,
        nominal_frequency_hz=100_000.0,
        frequency_gain_v_per_hz=-5e-4,
        time_constant_s=8e-4,
    )
    result = run_closed_loop(
        plant,
        controller=controller,
        controller_limits=ControllerLimitConfig(-0.8, 0.8),
        sampler=SamplerConfig(sample_rate_hz=fs, sample_phase_s=2e-6, gain=0.05, adc_bits=12, adc_min=0.0, adc_max=3.3),
        modulator=LLCFMConfig(
            mode=LLCFMMode.LINEAR_FM,
            nominal_frequency_hz=100_000.0,
            kfm_hz_per_unit=-20_000.0,
            minimum_frequency_hz=70_000.0,
            maximum_frequency_hz=180_000.0,
            tbclk_hz=120_000_000.0,
            count_mode=PWMCountMode.UP_DOWN,
            quantize_tbprd=True,
        ),
        timing=ClosedLoopTiming(duration_s=0.03, computation_delay_s=1e-6, pwm_update_delay_s=2e-6),
        scenario=ClosedLoopScenario(reference_v=StepProfile(53.0, 0.005, 54.0)),
    )
    assert len(result.samples) > 1000
    assert abs(result.diagnostics.regulation_error_v) < 0.15
    assert result.diagnostics.output_std_v < 0.08
    assert result.diagnostics.controller_saturation_fraction < 0.2


def test_load_step_is_supported_by_scheduler():
    fs = 40_000.0
    controller = _pi(fs, kp=0.05, ti=0.003)
    plant = FirstOrderLLCPlant(load_droop_v_per_pu=2.0, time_constant_s=0.001)
    result = run_closed_loop(
        plant,
        controller=controller,
        controller_limits=ControllerLimitConfig(-1.0, 1.0),
        sampler=SamplerConfig(sample_rate_hz=fs),
        modulator=LLCFMConfig(mode=LLCFMMode.LINEAR_FM, nominal_frequency_hz=100_000, kfm_hz_per_unit=-20_000),
        timing=ClosedLoopTiming(duration_s=0.025),
        scenario=ClosedLoopScenario(reference_v=StepProfile(53.0), load_fraction=StepProfile(1.0, 0.005, 1.5)),
    )
    assert abs(result.diagnostics.regulation_error_v) < 0.2


def test_existing_llc_small_signal_plant_can_feed_v9_closed_loop_stability_check():
    from llc_design.control.analysis import build_small_signal_analysis
    from llc_design.core.spec import LLCDesignSpec
    from llc_design.models.system import LLCSystemAnalyzer
    from power_sim import llc_small_signal_to_digital_plant
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    small = build_small_signal_analysis(spec, system_analysis=system, sample_time_s=20e-6)
    plant = llc_small_signal_to_digital_plant(small)
    controller = discretize_transfer_function(
        design_controller(ControllerKind.PIF, kp=0.002, ti_s=0.003, lpf_pole_hz=3500),
        plant.sample_rate_hz,
        DiscretizationMethod.TUSTIN,
    )
    analysis = analyze_linear_closed_loop(controller, plant, modulator_gain=-100_000.0, delay_samples=1)
    assert analysis.stable
    assert analysis.phase_margin_deg is not None and analysis.phase_margin_deg > 45.0
    assert analysis.gain_margin_db is None or analysis.gain_margin_db > 6.0
