from __future__ import annotations

import json
from pathlib import Path

import pytest

from llc_design.control.analysis import build_small_signal_analysis
from llc_design.control.digital_loop import (
    ADCSamplingConfig,
    AnalogSenseConfig,
    CommandTimingConfig,
    PIFControllerConfig,
    build_digital_loop_analysis,
)
from llc_design.core.q_zvs import build_q_zvs_analysis
from llc_design.core.spec import LLCDesignSpec
from llc_design.dynamics.plant import DynamicPhasorModel
from llc_design.dynamics.switched import (
    SwitchedSimulationConfig,
    simulate_switched_steady_state,
)
from llc_design.dynamics.waveforms import reconstruct_dynamic_phasor_waveforms
from llc_design.magnetics.transformer_designer import (
    FerriteCoreInput,
    synthesize_transformer,
)
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.optimization.sweep import LLCOptimizer, OptimizationConfig
from web_api.adapters.llc import SUPPORTED_OPERATIONS, LLCAdapter


def _small_config() -> dict[str, object]:
    return {
        "vbus_v": 400.0,
        "load_fraction": 1.0,
        "sample_time_s": 20e-6,
        "control_input_kind": "frequency_hz",
        "input_delay_samples": 0,
        "trim_frequency_to_output": False,
    }


def test_llc_adapter_exposes_the_complete_operation_whitelist():
    assert SUPPORTED_OPERATIONS == {
        "system",
        "q_zvs",
        "transformer",
        "waveforms_fast",
        "waveforms_detailed",
        "small_signal",
        "digital_loop",
        "optimize",
        "codegen",
    }


def test_llc_adapter_rejects_unknown_operation_and_config_field():
    adapter = LLCAdapter()
    with pytest.raises(ValueError, match="unknown LLC operation"):
        adapter.run("not-an-operation", {})
    with pytest.raises(ValueError, match="unknown config field"):
        adapter.run("system", {"not_a_spec_field": 1})


def test_system_operation_matches_direct_analyzer(tmp_path: Path):
    spec = LLCDesignSpec()
    direct = LLCSystemAnalyzer().analyze(spec)
    result = LLCAdapter().run("system", {}, tmp_path)
    assert result["feasibility"] is direct.feasible
    assert result["metrics"]["nominal_efficiency"] == pytest.approx(
        direct.nominal.efficiency
    )
    assert result["metrics"]["nominal_frequency_hz"] == pytest.approx(
        direct.nominal.operating_point.switching_frequency_hz
    )
    assert result["tables"]["operating_points"][0]["vbus_v"] == pytest.approx(
        direct.operating_points[0].operating_point.vbus_v
    )


def test_q_zvs_operation_matches_direct_map(tmp_path: Path):
    direct = build_q_zvs_analysis(LLCDesignSpec(), frequency_points=80)
    result = LLCAdapter().run("q_zvs", {"frequency_points": 80}, tmp_path)
    assert result["metrics"]["frequency_points"] == 80
    assert result["series"]["frequencies_hz"] == pytest.approx(
        direct.map.frequencies_hz.tolist()
    )
    assert result["tables"]["workpoints"][0]["zvs_margin"] == pytest.approx(
        direct.workpoints[0].zvs_margin
    )


def test_transformer_operation_matches_direct_synthesis(tmp_path: Path):
    spec = LLCDesignSpec()
    core = FerriteCoreInput()
    direct = synthesize_transformer(spec, core)
    result = LLCAdapter().run("transformer", {}, tmp_path)
    assert result["metrics"]["primary_turns"] == direct.primary_turns
    assert result["metrics"]["secondary_turns"] == direct.secondary_turns
    assert result["metrics"]["worst_b_peak_t"] == pytest.approx(direct.worst_b_peak_t)


def test_waveform_operations_match_direct_waveform_models(tmp_path: Path):
    spec = LLCDesignSpec()
    small = build_small_signal_analysis(spec, **_small_config())
    model = DynamicPhasorModel(small.parameters)
    fast = reconstruct_dynamic_phasor_waveforms(
        model, small.steady_state, cycles=1, samples_per_cycle=64
    )
    fast_result = LLCAdapter().run(
        "waveforms_fast",
        {**_small_config(), "cycles": 1, "samples_per_cycle": 64},
        tmp_path,
    )
    assert fast_result["metrics"]["sample_count"] == len(fast.time_s)
    assert fast_result["series"]["i_resonant"] == pytest.approx(
        fast.signals["i_resonant"].values.tolist()
    )

    detailed = simulate_switched_steady_state(
        model,
        small.steady_state,
        SwitchedSimulationConfig(
            samples_per_cycle=128,
            output_cycles=1,
            minimum_settling_cycles=1,
            maximum_settling_cycles=2,
            shooting_max_evaluations=10,
        ),
    )
    detailed_result = LLCAdapter().run(
        "waveforms_detailed",
        {
            **_small_config(),
            "samples_per_cycle": 128,
            "cycles": 1,
            "minimum_settling_cycles": 1,
            "maximum_settling_cycles": 2,
            "shooting_max_evaluations": 10,
        },
        tmp_path,
    )
    assert detailed_result["metrics"]["sample_count"] == len(detailed.time_s)
    assert detailed_result["series"]["v_output"] == pytest.approx(
        detailed.signals["v_output"].values.tolist()
    )


def test_small_signal_operation_matches_direct_analysis(tmp_path: Path):
    spec = LLCDesignSpec()
    direct = build_small_signal_analysis(spec, **_small_config())
    result = LLCAdapter().run("small_signal", _small_config(), tmp_path)
    assert result["metrics"]["sample_time_s"] == pytest.approx(direct.sample_time_s)
    assert result["metrics"]["dc_gain"] == pytest.approx(
        direct.continuous_transfer.dc_gain
    )
    assert result["metrics"]["discrete_stable"] is direct.discrete_plant.stable


def test_digital_loop_operation_matches_direct_analysis(tmp_path: Path):
    spec = LLCDesignSpec()
    small = build_small_signal_analysis(spec, **_small_config())
    direct = build_digital_loop_analysis(
        small,
        controller_config=PIFControllerConfig(sample_time_s=20e-6),
        analog_sense=AnalogSenseConfig(),
        adc_sampling=ADCSamplingConfig(control_sample_time_s=20e-6),
        command_timing=CommandTimingConfig(),
    )
    result = LLCAdapter().run("digital_loop", _small_config(), tmp_path)
    assert result["metrics"]["controller_kind"] == "pif"
    assert result["metrics"]["pcmd"] == pytest.approx(
        direct.fm_operating_point.command_pu
    )
    assert result["metrics"]["likely_stable"] is direct.likely_stable


def test_optimize_operation_returns_explicit_records_and_matches_direct(tmp_path: Path):
    config = {
        "optimization_config": {
            "ln_values": [5.0],
            "q_values": [0.35],
            "fr_values_hz": [100000.0],
            "primary_turn_values": [30],
            "secondary_turn_values": [4],
            "primary_devices": ["REF_650V_SIC_45M"],
            "sr_parallel_values": [2],
        },
        "maximum_candidates": 1,
    }
    direct = LLCOptimizer().run(
        LLCDesignSpec(),
        OptimizationConfig(
            ln_values=(5.0,),
            q_values=(0.35,),
            fr_values_hz=(100000.0,),
            primary_turn_values=(30,),
            secondary_turn_values=(4,),
            primary_devices=("REF_650V_SIC_45M",),
            sr_parallel_values=(2,),
        ),
        1,
    )
    result = LLCAdapter().run("optimize", config, tmp_path)
    assert result["tables"]["all"]["columns"] == [str(c) for c in direct.table.columns]
    assert (
        result["tables"]["all"]["records"][0]["candidate_key"]
        == direct.table.iloc[0]["candidate_key"]
    )


def test_codegen_operation_writes_only_job_artifacts_and_has_stability_gate(
    tmp_path: Path,
):
    result = LLCAdapter().run("codegen", _small_config(), tmp_path)
    assert result["metrics"]["stability_gate_passed"] is True
    paths = [Path(item["path"]) for item in result["artifacts"]]
    assert paths
    assert all(path.is_file() and tmp_path in path.parents for path in paths)
    assert all(path.suffix in {".c", ".h", ".json", ".txt", ".md"} for path in paths)


def test_adapter_payload_is_json_safe(tmp_path: Path):
    result = LLCAdapter().run("small_signal", _small_config(), tmp_path)
    json.dumps(result, allow_nan=False)
