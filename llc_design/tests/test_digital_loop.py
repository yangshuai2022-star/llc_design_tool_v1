"""Tests for complete digital voltage-loop modelling."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from llc_design.control.analysis import build_small_signal_analysis
from llc_design.control.digital_loop import (
    ADCSamplingConfig,
    AnalogSenseConfig,
    CommandTimingConfig,
    DigitalTransferFunction,
    FMLUTMode,
    FrequencyModulatorLUT,
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
    build_digital_loop_analysis,
    export_controller_c99,
)
from llc_design.core.spec import LLCDesignSpec
from llc_design.models.system import LLCSystemAnalyzer


@lru_cache(maxsize=1)
def _baseline_small_signal():
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    return build_small_signal_analysis(spec, system_analysis=system, sample_time_s=20e-6)


def test_firmware_fm_lut_endpoints_and_local_gain():
    lut = FrequencyModulatorLUT.firmware_default()
    assert lut.frequency_hz(0.0) == pytest.approx(250e3)
    assert lut.frequency_hz(1.0) == pytest.approx(120e6 / (2.0 * 857.0))
    command = lut.command_for_frequency(100e3)
    assert 0.0 < command < 1.0
    assert lut.frequency_hz(command) == pytest.approx(100e3, rel=2e-3)
    assert lut.local_gain_hz_per_pu(command) < 0.0
    left, right = lut.gain_sides_hz_per_pu(0.52)
    assert np.isfinite(left) and np.isfinite(right)


def test_user_frequency_lut_and_parser():
    lut = FrequencyModulatorLUT.from_text(
        "pcmd,frequency_hz\n0,250000\n0.5,100000\n1,70000",
        mode=FMLUTMode.PCMD_TO_FREQUENCY,
    )
    assert lut.frequency_hz(0.5) == pytest.approx(100e3)
    assert lut.local_gain_hz_per_pu(0.25) == pytest.approx(-300e3)


def test_pi_and_pif_match_firmware_linear_forms():
    ts = 20e-6
    pi = PIControllerConfig(kp=0.2, ti_s=1e-3, sample_time_s=ts).transfer_function()
    ki2 = ts / (2e-3)
    assert pi.numerator == pytest.approx([0.2 * (1 + ki2), 0.2 * (-1 + ki2)])
    assert pi.denominator == pytest.approx([1.0, -1.0])
    pif_cfg = PIFControllerConfig(kp=0.2, ti_s=1e-3, lpf_cutoff_hz=3500, sample_time_s=ts)
    pif = pif_cfg.transfer_function()
    assert len(pif.denominator) == 3
    assert 0.0 < pif_cfg.alpha < 1.0


def test_two_p_two_z_sign_convention():
    config = TwoP2ZControllerConfig(
        b0=1.0, b1=-0.5, b2=0.1,
        a1=-1.2, a2=0.3,
        sample_time_s=20e-6,
    )
    tf = config.transfer_function()
    assert tf.denominator == pytest.approx([1.0, -1.2, 0.3])
    text = tf.difference_equation()
    assert "1.2*y[k-1]" in text
    assert "- 0.3*y[k-2]" in text


def test_analog_sense_is_calibrated_to_unity_dc():
    sense = AnalogSenseConfig()
    response = sense.frequency_response_components(np.asarray([1e-6, 1e3, 1e6]))
    assert abs(response["calibrated_analog"][0]) == pytest.approx(1.0, rel=1e-8)
    assert sense.divider_pole_hz == pytest.approx(100.85e3, rel=5e-3)
    assert sense.adc_rc_pole_hz == pytest.approx(361.72e3, rel=5e-3)


def test_adc_recursive_average_dc_gain_and_timing():
    adc = ADCSamplingConfig()
    response = adc.frequency_response(np.asarray([1e-6, 1e3]))
    assert abs(response[0]) == pytest.approx(1.0, rel=1e-8)
    assert len(adc.sample_offsets_s) == 3
    assert adc.eoc_delay_s > adc.sample_offsets_s[-1]
    digital = adc.simplified_digital_filter()
    assert digital.numerator == pytest.approx([0.75])
    assert digital.denominator == pytest.approx([1.0, -0.25])


def test_complete_loop_builds_all_required_layers():
    small = _baseline_small_signal()
    result = build_digital_loop_analysis(
        small,
        controller_config=PIFControllerConfig(
            kp=0.002, ti_s=3e-3, lpf_cutoff_hz=3500, sample_time_s=20e-6),
        command_timing=CommandTimingConfig(computation_delay_s=1e-6),
    )
    required = {
        "power_stage", "fm_power_stage", "controller",
        "sense_analog_raw", "sense_analog_calibrated", "adc_sampling",
        "sense_total", "delay_minimum", "delay_nominal", "delay_maximum",
        "open_loop_minimum", "open_loop_nominal", "open_loop_maximum",
        "closed_loop_nominal", "sensitivity_nominal",
        "closed_loop_output_impedance",
    }
    assert required.issubset(result.responses)
    assert result.fm_operating_point.gain_hz_per_pu < 0.0
    assert np.all(np.isfinite(result.nominal_open_loop))
    assert len(result.discrete_approximation.closed_loop_poles) > 0


def test_generated_controller_c99_compiles(tmp_path: Path):
    controller = PIFControllerConfig(
        kp=0.002, ti_s=3e-3, lpf_cutoff_hz=3500, sample_time_s=20e-6,
    ).transfer_function()
    source = export_controller_c99(controller, tmp_path / "controller.c")
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler available")
    subprocess.run(
        [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-c", str(source)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
