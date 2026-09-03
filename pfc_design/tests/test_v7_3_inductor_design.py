import math

from pfc_design.magnetics import (
    HIGH_FLUX_254,
    high_flux_254_material,
    PFCInductorDesignRequest,
    design_pfc_inductor,
)


def test_high_flux_254_official_geometry_and_mu60_fit():
    assert HIGH_FLUX_254.le_mm == 98.4
    assert HIGH_FLUX_254.ae_mm2 == 110.6
    assert HIGH_FLUX_254.ve_mm3 == 10880.0
    mat = high_flux_254_material(60)
    assert mat.al_nh_per_t2 == 81.0
    assert math.isclose(mat.permeability_percent(0.0), 100.0, rel_tol=0, abs_tol=1e-12)
    assert 79.0 < mat.permeability_percent(100.0) < 82.0
    assert mat.core_loss_density_mw_cm3(0.1, 100e3) > 0.0


def test_ttpl_default_inductor_design_hits_biased_target_with_two_cores():
    result = design_pfc_inductor(PFCInductorDesignRequest(
        topology="ttpl", input_rms_v=230.0, bus_voltage_v=400.0,
        output_power_w=3300.0, switching_frequency_hz=50e3,
        target_inductance_uh=220.0, efficiency=0.97,
        core=HIGH_FLUX_254, material=high_flux_254_material(60),
        n_cores=2, wire_copper_diameter_mm=1.0,
    ))
    assert result.inductance_target_met
    assert result.window_ok
    assert result.l_full_load_peak_uh >= 0.98 * 220.0
    assert result.core_loss_w > 0.0
    assert result.copper_loss_w > 0.0
    assert result.inductance_uh[-1] < result.inductance_uh[0]


def test_vienna_default_phase_inductor_design_hits_biased_target():
    result = design_pfc_inductor(PFCInductorDesignRequest(
        topology="vienna", input_rms_v=400.0, bus_voltage_v=700.0,
        output_power_w=10000.0, switching_frequency_hz=65e3,
        target_inductance_uh=600.0, efficiency=0.97,
        core=HIGH_FLUX_254, material=high_flux_254_material(60),
        n_cores=5, wire_copper_diameter_mm=1.0,
    ))
    assert result.inductance_target_met
    assert result.window_ok
    assert result.l_full_load_peak_uh >= 0.98 * 600.0
    assert result.parallel_wires >= 1
    assert result.b_ac_line_max_t > 0.0
