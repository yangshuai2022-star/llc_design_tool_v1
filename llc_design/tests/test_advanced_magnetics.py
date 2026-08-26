import math

import numpy as np

from llc_design.core.spec import LLCDesignSpec
from llc_design.magnetics.core import CoreDatabase
from llc_design.magnetics.litz import (
    LitzWire, StackLayer, layered_litz_stack_loss,
    transverse_field_loss_per_strand_w_per_m,
)
from llc_design.magnetics.material import MaterialDatabase
from llc_design.models.system import LLCSystemAnalyzer


def test_igse_sinusoid_matches_classical_steinmetz():
    material = MaterialDatabase().get("TDK_N97_REF")
    f = 100e3
    bpk = 0.10
    samples = 4096
    t = np.arange(samples) / (samples * f)
    b = bpk * np.sin(2 * math.pi * f * t)
    classical = material.steinmetz_density_w_m3(f, bpk, 100.0)
    igse = material.igse_density_w_m3(t, b, 100.0)
    assert abs(igse / classical - 1.0) < 0.01


def test_point_one_mm_strands_reduce_external_field_loss_vs_point_two_mm():
    f = 100e3
    h = 3000.0
    small = 400 * transverse_field_loss_per_strand_w_per_m(0.1e-3, f, h)
    large = 100 * transverse_field_loss_per_strand_w_per_m(0.2e-3, f, h)
    assert small < 0.35 * large


def test_sandwich_stack_has_lower_proximity_loss_than_grouped_stack():
    samples = 1024
    theta = 2 * math.pi * np.arange(samples) / samples
    primary = tuple(10 * math.sqrt(2) * np.sin(theta))
    secondary = tuple(-10 * math.sqrt(2) * np.sin(theta))
    wire = LitzWire(200, 0.1e-3, 0.112e-3, 0.55, 1)

    def layer(label, turns, waveform):
        return StackLayer(label, turns, turns * 0.08, wire, waveform)

    sandwich = [
        layer("primary", 5, primary),
        layer("secondary", 10, secondary),
        layer("primary", 5, primary),
    ]
    grouped = [
        layer("primary", 5, primary),
        layer("primary", 5, primary),
        layer("secondary", 10, secondary),
    ]
    a = layered_litz_stack_loss(sandwich, 100e3, 0.025)
    b = layered_litz_stack_loss(grouped, 100e3, 0.025)
    prox_a = sum(x.external_proximity_w for x in a.values())
    prox_b = sum(x.external_proximity_w for x in b.values())
    assert prox_a < prox_b


def test_core_database_contains_pq_ee_ec_families():
    families = set(CoreDatabase().families)
    assert {"PQ", "EE", "EC"}.issubset(families)


def test_baseline_exposes_detailed_magnetic_losses_and_hotspot():
    result = LLCSystemAnalyzer().analyze(LLCDesignSpec())
    nominal = result.nominal
    assert nominal.transformer.primary_proximity_w > 0.0
    assert nominal.transformer.secondary_dc_w > 0.0
    assert nominal.resonant_inductor.gap_fringing_w > 0.0
    assert nominal.transformer.estimated_hotspot_c > result.spec.ambient_temperature_c
    assert nominal.resonant_inductor.estimated_hotspot_c > result.spec.ambient_temperature_c


# ---- Dowell foil cross-check (feature C) --------------------------------------

from llc_design.magnetics.dowell import (
    DowellLayer, dowell_optimum_delta, dowell_stack_loss,
)


def _foil_waveform(samples: int = 512, amp: float = 1.0, duty: float = 0.5):
    """Trapezoidal current waveform, half-wave symmetric."""
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    wave = np.where(t < duty, amp, -amp)
    return tuple(float(x) for x in wave)


def test_dowell_foil_ac_factor_exceeds_dc_at_high_frequency():
    # A 0.3 mm foil at 100 kHz, 4 layers: AC copper loss must well exceed the
    # DC-equivalent loss Rdc*Irms^2 (i.e. the AC resistance factor > 1).
    wave = _foil_waveform(amp=10.0)
    layer = DowellLayer(
        label="primary", turns=10, conductor_length_m=0.07 * 10,
        foil_thickness_m=0.3e-3, foil_width_m=10e-3,
        current_waveform_a=wave, layers_in_group=4,
    )
    result = dowell_stack_loss([layer], 100e3, max_harmonic=15, temperature_c=100.0)["primary"]
    from llc_design.magnetics.litz import copper_resistivity
    rho = copper_resistivity(100.0)
    rdc = rho * layer.conductor_length_m / (layer.foil_thickness_m * layer.foil_width_m)
    rms = float(np.sqrt(np.mean(np.asarray(wave) ** 2)))
    dc_equivalent = rdc * rms ** 2
    assert result.total_w > 2.0 * dc_equivalent   # significant AC penalty for thick foil
    assert result.skin_effect_w > 0.0
    assert result.external_proximity_w > 0.0


def test_dowell_foil_loss_is_minimised_near_optimum_delta():
    # Sweep foil thickness; the AC loss should dip near deltaopt * skin_depth.
    wave = _foil_waveform(amp=10.0)
    freq = 100e3
    delta_opt = dowell_optimum_delta(duty=0.5, layers_p=4)
    from llc_design.magnetics.litz import skin_depth_m
    delta_skin = skin_depth_m(freq, 100.0)
    t_opt = delta_opt * delta_skin
    thicknesses = np.geomspace(0.5 * t_opt, 4.0 * t_opt, 25)
    losses = []
    for t in thicknesses:
        layer = DowellLayer(
            label="primary", turns=10, conductor_length_m=0.7,
            foil_thickness_m=float(t), foil_width_m=10e-3,
            current_waveform_a=wave, layers_in_group=4,
        )
        r = dowell_stack_loss([layer], freq, max_harmonic=15, temperature_c=100.0)["primary"]
        losses.append(r.total_w)
    losses = np.array(losses)
    # Minimum near the optimum: the optimum thickness should be within the
    # lowest-loss half of the sweep, and the global min should not be at an edge.
    min_idx = int(np.argmin(losses))
    assert 2 <= min_idx <= len(losses) - 3
    # AC loss at the optimum should be lower than at 4x the optimum thickness.
    opt_loss = losses[np.argmin(np.abs(thicknesses - t_opt))]
    thick_loss = losses[-1]
    assert opt_loss < thick_loss


def test_dowell_foil_loss_matches_book_kpn_formula():
    # Single harmonic (sinusoid), p=6, delta=0.5: total AC loss must equal
    # Rdc * kpn(delta,1,p) * I_rms^2 (DC component is zero for a pure AC sinusoid
    # with no offset).
    samples = 4096
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    i_rms_target = 10.0
    wave = tuple(float(x) for x in (i_rms_target * math.sqrt(2) * np.sin(2 * np.pi * t)))
    from llc_design.magnetics.dowell import dowell_kpn
    from llc_design.magnetics.litz import copper_resistivity, skin_depth_m
    rho = copper_resistivity(100.0)
    rdc = rho * 0.7 / (0.3e-3 * 10e-3)
    delta = 0.3e-3 / skin_depth_m(100e3, 100.0)
    kpn1 = dowell_kpn(delta, 1, 6)
    # The total AC loss for a pure fundamental sinusoid = Rdc * kpn1 * I_rms^2.
    expected_ac = rdc * kpn1 * i_rms_target ** 2
    layer = DowellLayer(
        label="primary", turns=10, conductor_length_m=0.7,
        foil_thickness_m=0.3e-3, foil_width_m=10e-3,
        current_waveform_a=wave, layers_in_group=6,
    )
    r = dowell_stack_loss([layer], 100e3, max_harmonic=15, temperature_c=100.0)["primary"]
    # skin + proximity (the AC part) should match the single-harmonic kpn formula
    # within the small spectral leakage of a finite-length FFT.
    assert math.isclose(r.skin_effect_w + r.external_proximity_w, expected_ac, rel_tol=2e-2)
