from dataclasses import replace
from pathlib import Path

import pytest

from pfc_design.control import (
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    simulate_pfc_line_cycle,
)
from pfc_design.control.autotune import tune_pfc_current_loop
from pfc_design.control.config import LoadModel
from pfc_design.magnetics import (
    HIGH_FLUX_254,
    PFCInductorDesignRequest,
    design_pfc_inductor,
    high_flux_254_material,
)
from power_codegen import generate_ttpl_control_code
from web_api.adapters.ttpl import (
    build_ttpl_config,
    run_ttpl_operation,
)


def _config_payload() -> dict:
    return {
        "frequency_points": 120,
        "waveform_line_cycles": 3,
        "switching_cycles": 1,
        "switching_samples_per_cycle": 120,
        "waveform_integration_rate_hz": 200.0e3,
    }


def _direct_config() -> PFCControlLabConfig:
    return replace(
        PFCControlLabConfig(),
        frequency_points=120,
        waveform_line_cycles=3,
        switching_cycles=1,
        switching_samples_per_cycle=120,
        waveform_integration_rate_hz=200.0e3,
    )


def test_config_builder_rejects_unknown_nested_fields_and_keeps_three_sense_chains_independent():
    with pytest.raises(ValueError, match="power_stage.*unknown"):
        build_ttpl_config({"power_stage": {"not_a_real_field": 1.0}})

    config = build_ttpl_config(
        {
            "power_stage": {"load_model": "resistive"},
            "current_controller": {"kind": "pi", "kp": 0.02},
            "current_sense": {"timing": {"digital_filter": {"alpha": 0.8}}},
        }
    )
    assert config.power_stage.load_model is LoadModel.RESISTIVE
    assert config.current_controller.kp == pytest.approx(0.02)
    assert config.current_sense is not config.vac_sense
    assert config.current_sense is not config.vbus_sense
    assert config.current_sense.timing.digital_filter.alpha == pytest.approx(0.8)
    assert config.vac_sense.timing.digital_filter.alpha == pytest.approx(1.0)


def test_control_analysis_matches_direct_loop_and_all_sensing_margins():
    config = _direct_config()
    direct = build_pfc_control_lab_analysis(config)
    result = run_ttpl_operation("control_analysis", config=_config_payload())

    current = result["metrics"]["current_loop"]
    voltage = result["metrics"]["voltage_loop"]
    assert current["phase_margin_deg"] == pytest.approx(direct.current_loop.margins.phase_margin_deg)
    assert current["gain_margin_db"] == pytest.approx(direct.current_loop.margins.gain_margin_db)
    assert voltage["phase_margin_deg"] == pytest.approx(direct.voltage_loop.margins.phase_margin_deg)
    assert voltage["gain_margin_db"] == pytest.approx(direct.voltage_loop.margins.gain_margin_db)
    assert set(result["metrics"]["sensing"]) == {"current", "vac", "vbus"}
    assert result["series"]["frequencies_hz"] == pytest.approx(direct.frequencies_hz.tolist())


def test_autotune_is_explicit_and_does_not_apply_by_default():
    config = _direct_config()
    direct = tune_pfc_current_loop(config)
    result = run_ttpl_operation("autotune_current", config=_config_payload())

    assert result["parameters"]["apply"] is False
    assert result["metrics"]["accepted"] is direct.accepted
    assert result["metrics"]["controller"]["kp"] == pytest.approx(direct.controller.kp)
    assert result["metrics"]["controller"]["ti_s"] == pytest.approx(direct.controller.ti_s)
    assert result["config_snapshot"]["current_controller"]["kp"] == pytest.approx(
        config.current_controller.kp
    )


def test_autotune_apply_is_explicit_and_returns_applied_analysis():
    result = run_ttpl_operation(
        "autotune_current",
        config=_config_payload(),
        options={"apply": True},
    )
    assert result["parameters"]["apply"] is True
    assert result["parameters"]["applied"] is True
    assert result["config_snapshot"]["current_controller"]["kp"] != pytest.approx(0.01)
    assert "analysis" in result["tables"]


def test_line_cycle_and_switching_match_direct_waveform_arrays():
    config = _direct_config()
    direct_line = simulate_pfc_line_cycle(config)
    line = run_ttpl_operation("line_cycle", config=_config_payload())

    assert line["metrics"]["power_factor"] == pytest.approx(direct_line.metrics.power_factor)
    assert line["metrics"]["current_thd_percent"] == pytest.approx(
        direct_line.metrics.current_thd_percent
    )
    assert line["series"]["signals"]["i_input_signed"] == pytest.approx(
        direct_line.signals["i_input_signed"].tolist()
    )

    direct_switch = build_pfc_switching_waveforms(config, line_cycle=direct_line)
    switch = run_ttpl_operation("switching", config=_config_payload())
    assert switch["metrics"]["line_angle_deg"] == pytest.approx(direct_switch.line_angle_deg)
    assert switch["series"]["signals"]["inductor_current"] == pytest.approx(
        direct_switch.signals["inductor_current"].tolist()
    )


def test_inductor_design_matches_direct_result():
    cfg = _direct_config()
    request = PFCInductorDesignRequest(
        topology="ttpl",
        input_rms_v=cfg.power_stage.vin_rms_v,
        bus_voltage_v=cfg.power_stage.bus_voltage_v,
        output_power_w=cfg.power_stage.output_power_w,
        switching_frequency_hz=cfg.power_stage.switching_frequency_hz,
        target_inductance_uh=cfg.power_stage.boost_inductance_h * 1.0e6,
        efficiency=cfg.power_stage.efficiency,
        core=HIGH_FLUX_254,
        material=high_flux_254_material(60),
        n_cores=2,
    )
    direct = design_pfc_inductor(request)
    result = run_ttpl_operation("inductor_design", config=_config_payload())
    assert result["metrics"]["turns"] == direct.turns
    assert result["metrics"]["l_full_load_peak_uh"] == pytest.approx(direct.l_full_load_peak_uh)
    assert result["series"]["inductance_uh"] == pytest.approx(direct.inductance_uh.tolist())


def test_full_export_reports_real_paths(tmp_path: Path):
    result = run_ttpl_operation(
        "full_export",
        config=_config_payload(),
        options={"directory": str(tmp_path / "export")},
    )
    paths = result["parameters"]["artifact_paths"]
    assert paths
    assert all(Path(path).exists() for path in paths.values())
    assert (tmp_path / "export" / "pfc_control_lab_summary.json").exists()


def test_codegen_matches_direct_generator_and_emits_c99(tmp_path: Path):
    config = replace(
        _direct_config(),
        current_controller=replace(
            _direct_config().current_controller,
            kp=0.00854059,
            ti_s=0.00064608,
        ),
    )
    direct = generate_ttpl_control_code(
        build_pfc_control_lab_analysis(config), tmp_path / "direct"
    )
    payload = _config_payload()
    payload["current_controller"] = {"kind": "pi", "kp": 0.00854059, "ti_s": 0.00064608}
    result = run_ttpl_operation(
        "codegen",
        config=payload,
        options={"directory": str(tmp_path / "adapter")},
    )
    assert result["metrics"]["validation"]["passed"] is direct.validation.passed
    paths = result["parameters"]["artifact_paths"]
    assert Path(paths["control_c"]).read_text(encoding="utf-8") == (
        direct.directory / "ttpl_control.c"
    ).read_text(encoding="utf-8")
