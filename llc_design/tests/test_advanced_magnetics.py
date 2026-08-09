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
