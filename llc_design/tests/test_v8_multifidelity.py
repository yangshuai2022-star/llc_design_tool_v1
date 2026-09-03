"""Regression tests for the V8.0/V8.1 multi-fidelity LLC core."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from llc_design.analysis import (
    FidelityLevel,
    GoldenSolverConfig,
    HarmonicBalanceConfig,
    LLCAnalysisRequest,
    LLCGoldenSolver,
    TimeDomainConfig,
    exact_rectifier_projection,
    export_multifidelity_analysis,
    extract_peak_phasors,
    solve_harmonic_balance,
    synthesize_real_waveform,
)
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology


def test_peak_phasor_round_trip_and_exact_rectifier_projection():
    harmonics = (1, 3, 5, 7)
    phasors = np.asarray(
        [1.3 - 4.2j, -0.7 + 0.35j, 0.18 - 0.09j, -0.05 + 0.02j],
        dtype=np.complex128,
    )
    phase = 2.0 * math.pi * np.arange(2048) / 2048
    waveform = synthesize_real_waveform(phasors, harmonics, phase)
    recovered = extract_peak_phasors(waveform, harmonics)
    assert recovered == pytest.approx(phasors, rel=1e-12, abs=1e-12)

    polarity, mean_absolute, roots = exact_rectifier_projection(
        np.asarray([-5.0j]), (1,), zero_crossing_samples=512)
    assert roots == pytest.approx((0.0, math.pi), abs=1e-12)
    assert polarity[0] == pytest.approx(-4.0j / math.pi, abs=1e-12)
    assert mean_absolute == pytest.approx(10.0 / math.pi, rel=1e-12)


def test_regulated_multi_harmonic_balance_is_self_consistent():
    request = LLCAnalysisRequest(
        spec=LLCDesignSpec(),
        load_fraction=1.0,
        samples_per_cycle=512,
        waveform_cycles=1,
    )
    result = solve_harmonic_balance(
        request,
        HarmonicBalanceConfig(
            max_harmonic=7,
            samples_per_cycle=512,
            output_cycles=1,
            frequency_scan_points=9,
            zero_crossing_samples=1024,
        ),
    )
    native = result.native_result
    assert result.convergence.converged
    assert result.convergence.method == "exact_zero_crossing_multi_harmonic_balance"
    assert result.harmonic_orders == (1, 3, 5, 7)
    assert result.metrics.output_voltage_v == pytest.approx(53.0, abs=2e-3)
    assert result.convergence.residual_norm < 1e-8
    assert abs(result.metrics.power_balance_error_percent) < 1e-5
    assert native.diagnostics["exact_rectifier_projection"] is True
    assert native.exact_sign_residual_norm < 1e-8
    assert abs(native.resonant_current_phasors_a[1]) > 1e-3
    assert abs(native.resonant_current_phasors_a[2]) > 1e-3

    # Common V8 waveform schema must satisfy output-node KCL sample by sample.
    bundle = result.waveform
    kcl = (
        bundle.signal("i_rectified").values
        - bundle.signal("i_load_output").values
        - bundle.signal("i_output_cap").values
    )
    assert np.max(np.abs(kcl)) < 1e-10


def test_light_load_hb_fallback_is_explicit_not_silent():
    request = LLCAnalysisRequest(
        spec=LLCDesignSpec(),
        load_fraction=0.2,
        samples_per_cycle=512,
        waveform_cycles=1,
    )
    result = solve_harmonic_balance(
        request,
        HarmonicBalanceConfig(
            max_harmonic=7,
            samples_per_cycle=512,
            output_cycles=1,
            frequency_scan_points=9,
            zero_crossing_samples=1024,
        ),
    )
    assert result.convergence.converged
    assert result.diagnostics["regularized_fallback"] is True
    assert result.convergence.method == "regularized_multi_harmonic_balance"
    assert any("regularized rectifier projection" in warning for warning in result.warnings)
    assert result.metrics.output_voltage_v == pytest.approx(53.0, abs=3e-3)


def test_hb_supports_full_and_half_bridge_topologies():
    config = HarmonicBalanceConfig(
        max_harmonic=5,
        minimum_harmonic=5,
        adaptive_harmonics=False,
        samples_per_cycle=256,
        output_cycles=1,
        frequency_scan_points=9,
        zero_crossing_samples=512,
    )
    specifications = (
        LLCDesignSpec(),
        LLCDesignSpec(
            primary_topology=PrimaryTopology.HALF_BRIDGE,
            primary_turns=15,
        ),
    )
    for spec in specifications:
        result = solve_harmonic_balance(
            LLCAnalysisRequest(
                spec=spec,
                samples_per_cycle=256,
                waveform_cycles=1,
            ),
            config,
        )
        assert result.convergence.converged
        assert result.convergence.method == "exact_zero_crossing_multi_harmonic_balance"
        assert result.metrics.output_voltage_v == pytest.approx(53.0, abs=3e-3)
        assert spec.minimum_frequency_hz < result.metrics.switching_frequency_hz < spec.maximum_frequency_hz


def _golden_analysis():
    request = LLCAnalysisRequest(
        spec=LLCDesignSpec(),
        samples_per_cycle=512,
        waveform_cycles=1,
    )
    config = GoldenSolverConfig(
        harmonic_balance=HarmonicBalanceConfig(
            max_harmonic=7,
            samples_per_cycle=512,
            output_cycles=1,
            frequency_scan_points=9,
            zero_crossing_samples=1024,
        ),
        time_domain=TimeDomainConfig(
            samples_per_cycle=512,
            output_cycles=1,
            minimum_settling_cycles=8,
            maximum_settling_cycles=300,
            convergence_tolerance=1e-7,
            shooting_max_evaluations=100,
            retry_samples_per_cycle=512,
        ),
    )
    return LLCGoldenSolver(config).solve(request)


def test_golden_solver_selects_converged_switched_reference():
    analysis = _golden_analysis()
    assert analysis.reference_level is FidelityLevel.SWITCHED_TIME_DOMAIN
    assert set(analysis.results) == {
        FidelityLevel.FHA,
        FidelityLevel.HARMONIC_BALANCE,
        FidelityLevel.SWITCHED_TIME_DOMAIN,
    }
    assert len(analysis.comparison_rows) == 3
    assert all(result.convergence.converged for result in analysis.results.values())
    reference_row = next(
        row for row in analysis.comparison_rows
        if row.fidelity is analysis.reference_level
    )
    assert reference_row.frequency_error_percent == pytest.approx(0.0)
    assert reference_row.resonant_rms_error_percent == pytest.approx(0.0)
    assert analysis.reference.metrics.output_voltage_v == pytest.approx(53.0, abs=0.06)


def test_multifidelity_export_without_waveform_plots(tmp_path):
    analysis = _golden_analysis()
    paths = export_multifidelity_analysis(
        analysis, tmp_path, export_model_waveforms=False)
    assert paths["comparison_csv"].exists()
    assert paths["summary_json"].exists()
    assert paths["report"].exists()

    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["reference_level"] == "switched_time_domain"
    assert set(payload["models"]) == {
        "fha", "harmonic_balance", "switched_time_domain",
    }
    with paths["comparison_csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert sum(row["reference"] == "True" for row in rows) == 1
