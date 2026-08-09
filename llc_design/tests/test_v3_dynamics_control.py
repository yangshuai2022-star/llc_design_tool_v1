"""Regression tests for the V3 waveform and control-object modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from llc_design.control.analysis import (
    build_small_signal_analysis,
    export_small_signal_analysis,
)
from llc_design.control.linearize import ControlInputKind
from llc_design.core.spec import LLCDesignSpec
from llc_design.dynamics.plant import DynamicPhasorModel
from llc_design.dynamics.switched import (
    SwitchedSimulationConfig,
    simulate_switched_steady_state,
)
from llc_design.dynamics.waveforms import reconstruct_dynamic_phasor_waveforms
from llc_design.models.system import LLCSystemAnalyzer


@lru_cache(maxsize=1)
def _baseline():
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    small = build_small_signal_analysis(spec, system_analysis=system)
    return spec, system, small


def test_edf_regulated_equilibrium_and_primary_transfers():
    spec, _, result = _baseline()
    assert result.steady_state.frequency_trimmed
    assert result.steady_state.output_voltage_v == pytest.approx(spec.vout_v, abs=2e-3)
    assert result.continuous_transfer.dc_gain < 0.0  # above-resonance frequency control
    assert np.isfinite(result.line_to_output_transfer.dc_gain)
    assert result.line_to_output_transfer.dc_gain > 0.0
    assert np.isfinite(result.output_impedance_transfer.dc_gain)
    assert result.output_impedance_transfer.dc_gain > 0.0
    assert result.continuous_plant.stable
    assert result.discrete_plant.stable


def test_control_input_unit_transformations_preserve_physics():
    spec, system, frequency = _baseline()
    period = build_small_signal_analysis(
        spec,
        system_analysis=system,
        control_input_kind=ControlInputKind.PERIOD_S,
    )
    counts = build_small_signal_analysis(
        spec,
        system_analysis=system,
        control_input_kind=ControlInputKind.TIMER_COUNTS,
        timer_clock_hz=120e6,
    )
    fs = frequency.operating_point.switching_frequency_hz
    assert period.continuous_transfer.dc_gain == pytest.approx(
        -fs**2 * frequency.continuous_transfer.dc_gain, rel=2e-5)
    assert counts.continuous_transfer.dc_gain == pytest.approx(
        -(fs**2 / 120e6) * frequency.continuous_transfer.dc_gain, rel=2e-5)


def test_exact_zoh_and_difference_equation_are_consistent():
    _, _, result = _baseline()
    plant = result.discrete_plant
    assert plant.ad.shape == (7, 7)
    assert plant.bd.shape == (7, 1)
    assert plant.cd.shape == (1, 7)
    assert plant.denominator[0] == pytest.approx(1.0)
    assert "y[k]" in plant.difference_equation.text()
    frequencies = np.geomspace(1.0, 1e3, 20)
    response = plant.frequency_response(frequencies)
    assert np.all(np.isfinite(response))


def test_fast_waveform_bundle_contains_engineering_nodes_and_measurements():
    _, _, result = _baseline()
    bundle = reconstruct_dynamic_phasor_waveforms(
        DynamicPhasorModel(result.parameters),
        result.steady_state,
        cycles=2,
        samples_per_cycle=512,
    )
    required = {
        "v_leg_a", "v_leg_b", "v_bridge", "i_resonant",
        "v_resonant_cap", "v_resonant_inductor",
        "v_transformer_primary", "i_magnetizing",
        "v_transformer_secondary", "i_transformer_secondary",
        "v_rectified", "i_rectified", "i_output_cap", "v_output_ripple",
        "vds_q1", "ids_q1", "vds_sr1", "ids_sr1",
        "b_transformer", "h_transformer",
        "b_resonant_inductor", "h_resonant_inductor",
    }
    assert required.issubset(bundle.signals)
    fs = result.operating_point.switching_frequency_hz
    assert bundle.signal("i_output_cap").statistics.frequency_hz == pytest.approx(
        2.0 * fs, rel=2e-3)
    assert abs(bundle.signal("i_output_cap").statistics.average) < 0.1
    assert bundle.signal("b_transformer").statistics.absolute_peak > 0.0
    assert bundle.signal("i_transformer_secondary").statistics.rms > spec_output_current()


def spec_output_current() -> float:
    spec, _, _ = _baseline()
    return spec.pout_w / spec.vout_v


def test_piecewise_switched_periodic_shooting_converges():
    _, _, result = _baseline()
    bundle = simulate_switched_steady_state(
        DynamicPhasorModel(result.parameters),
        result.steady_state,
        SwitchedSimulationConfig(
            samples_per_cycle=192,
            output_cycles=1,
            minimum_settling_cycles=5,
            convergence_tolerance=1e-7,
            shooting_max_evaluations=70,
        ),
    )
    assert bundle.metadata["converged"] == "True"
    assert float(bundle.metadata["periodic_mismatch"]) < 1e-6
    assert abs(bundle.signal("i_output_cap").statistics.average) < 0.2
    assert bundle.signal("v_output").statistics.average == pytest.approx(53.0, abs=0.25)


def test_small_signal_export_and_generated_c99_compile(tmp_path: Path):
    _, _, result = _baseline()
    paths = export_small_signal_analysis(result, tmp_path)
    assert paths["report"].exists()
    assert paths["c99_direct_source"].exists()
    assert paths["c99_state_source"].exists()
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler available")
    for source in (paths["c99_direct_source"], paths["c99_state_source"]):
        subprocess.run(
            [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-c", str(source)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
