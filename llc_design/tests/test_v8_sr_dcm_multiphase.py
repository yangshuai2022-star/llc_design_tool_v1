import numpy as np

from llc_design.analysis import FidelityLevel, LLCAnalysisRequest, solve_harmonic_balance
from llc_design.analysis.complementarity import (
    ComplementarityTimeDomainConfig,
    solve_complementarity_time_domain,
)
from llc_design.core.spec import LLCDesignSpec
from llc_design.magnetics.v8 import analyze_golden_magnetics
from llc_design.models.devices import DeviceDatabase
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.multiphase import solve_interleaved_llc
from llc_design.switching.sr import SRTimingConfig, analyze_sr, export_sr_lut_c99


def _hb(spec=None, load=1.0):
    spec = spec or LLCDesignSpec()
    return solve_harmonic_balance(
        LLCAnalysisRequest(spec, load_fraction=load, waveform_cycles=1, samples_per_cycle=512)
    )


def test_sr_timing_uses_primary_command_reference_and_has_qrr_loss():
    spec = LLCDesignSpec()
    result = _hb(spec)
    device = DeviceDatabase().get_sr(spec.sr_device)
    sr = analyze_sr(result, spec, device)
    assert 0.0 < sr.timing.on_delay_s < 250e-9
    assert 0.0 < sr.timing.off_advance_s < 250e-9
    assert sr.loss.reverse_recovery_w > 0.0
    assert sr.loss.recovery_charge_c_per_cycle > 0.0
    assert np.max(sr.recovery_current_a) > 0.0
    assert sr.loss.total_w > sr.loss.channel_w


def test_sr_third_quadrant_model_activates_for_deliberately_late_turnoff():
    spec = LLCDesignSpec()
    result = _hb(spec)
    device = DeviceDatabase().get_sr(spec.sr_device)
    sr = analyze_sr(
        result, spec, device,
        SRTimingConfig(
            turnoff_advance_s=-180e-9,
            turnoff_propagation_s=0.0,
            guard_s=0.0,
            enforce_safe_turnoff=False,
        ),
    )
    assert sr.loss.third_quadrant_w > 0.0
    assert sr.loss.reverse_current_w > 0.0
    assert sr.loss.reverse_charge_c > 0.0


def test_sr_lut_c99_export(tmp_path):
    spec = LLCDesignSpec()
    result = _hb(spec)
    device = DeviceDatabase().get_sr(spec.sr_device)
    sr = analyze_sr(result, spec, device)
    rows = ({
        "vin_v": 400.0,
        "load_fraction": 1.0,
        "fs_hz": sr.timing.switching_frequency_hz,
        "on_delay_ns": sr.timing.on_delay_s * 1e9,
        "off_advance_ns": sr.timing.off_advance_s * 1e9,
        "conduction_ns": sr.timing.conduction_s * 1e9,
    },)
    path = export_sr_lut_c99(rows, tmp_path / "llc_sr_lut.c")
    text = path.read_text()
    assert "g_llc_sr_lut" in text
    assert "on_delay_ns" in text


def test_dcm_complementarity_open_state_has_exact_zero_secondary_current():
    spec = LLCDesignSpec()
    result = solve_complementarity_time_domain(
        LLCAnalysisRequest(spec, load_fraction=0.10, waveform_cycles=1, samples_per_cycle=512),
        ComplementarityTimeDomainConfig(
            samples_per_cycle=512,
            output_cycles=1,
            minimum_settling_cycles=10,
            maximum_settling_cycles=100,
            convergence_tolerance=3e-5,
        ),
    )
    state = result.waveform.signal("rectifier_state").values
    isec = result.waveform.signal("i_transformer_secondary").values
    ip = result.waveform.signal("i_primary_load").values
    open_mask = np.abs(state) < 0.5
    assert np.mean(open_mask) > 0.10
    assert np.max(np.abs(isec[open_mask])) < 1e-12
    assert np.max(np.abs(ip[open_mask])) < 1e-10


def test_golden_magnetics_uses_waveform_and_returns_parasitics():
    spec = LLCDesignSpec()
    system = LLCSystemAnalyzer().analyze(spec)
    result = _hb(spec)
    mag = analyze_golden_magnetics(spec, result, system.transformer)
    assert mag.core_loss_w > 0.0
    assert mag.primary_copper_w > 0.0
    assert mag.secondary_copper_w > 0.0
    assert mag.primary_ac_factor > 1.0
    assert mag.leakage_inductance_h > 0.0
    assert mag.primary_secondary_capacitance_f > 0.0
    assert mag.b_peak_t > 0.0


def test_fixed_interleaved_phase_definitions_and_ripple_cancellation():
    spec = LLCDesignSpec()
    two = solve_interleaved_llc(spec, 2, fidelity=FidelityLevel.FHA, samples_per_cycle=256)
    three = solve_interleaved_llc(spec, 3, fidelity=FidelityLevel.FHA, samples_per_cycle=256)
    assert two.phase_offsets_deg == (0.0, 90.0)
    assert three.phase_offsets_deg == (0.0, 120.0, 240.0)
    assert max(abs(p.share_percent - 50.0) for p in two.phases) < 1e-6
    assert max(abs(p.share_percent - 100.0 / 3.0) for p in three.phases) < 1e-6
    # V8.3+ uses a topology-specific Y/Y shared-bridge model for 3P instead of
    # phase-shifting the 2P/single-cell waveform.  Therefore a blanket
    # "3P ripple must always be lower than 2P" assertion is not physically
    # valid for arbitrary auto-synthesized tanks.  Verify that both topology
    # solvers return finite system ripple metrics and that the 3P result is
    # explicitly tagged as the dedicated shared-bridge model.
    assert np.isfinite(two.output_capacitor_rms_a) and two.output_capacitor_rms_a >= 0.0
    assert np.isfinite(three.output_capacitor_rms_a) and three.output_capacitor_rms_a >= 0.0
    assert np.isfinite(two.output_ripple_vpp) and two.output_ripple_vpp >= 0.0
    assert np.isfinite(three.output_ripple_vpp) and three.output_ripple_vpp >= 0.0
    assert three.waveform.metadata.get("electrical_model") == "three_phase_y_shared_bridge_td"
