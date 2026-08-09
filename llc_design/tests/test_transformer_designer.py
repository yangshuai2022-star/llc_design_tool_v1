from llc_design.core.spec import LLCDesignSpec
from llc_design.magnetics.transformer_designer import (
    TransformerSynthesisSettings,
    load_transformer_core_presets,
    synthesize_transformer,
)


def test_pq35_datasheet_preset_geometry():
    p = load_transformer_core_presets()["TDK_PQ35_35_B65881A_N87"]
    assert p.ae_mm2 == 171.0
    assert p.amin_mm2 == 161.0
    assert p.le_mm == 79.7
    assert p.ve_mm3 == 13650.0
    assert p.winding_area_mm2 == 158.0
    assert p.mean_turn_length_mm == 76.0
    assert p.al_nh == 4500.0
    assert p.mu_e == 1670.0


def test_transformer_auto_turns_litz_and_loss():
    p = load_transformer_core_presets()["TDK_PQ35_35_B65881A_N87"]
    r = synthesize_transformer(LLCDesignSpec(), p, TransformerSynthesisSettings(strand_count_step=50))
    assert r.primary_turns > r.secondary_turns > 0
    assert r.primary_litz.strand_count % 50 == 0
    assert r.secondary_litz.strand_count % 50 == 0
    assert r.primary_litz.strand_diameter_mm == 0.10
    assert r.secondary_litz.strand_diameter_mm == 0.10
    assert r.worst_b_peak_t > 0.0
    assert r.nominal_loss.core_w > 0.0
    assert r.nominal_loss.primary_copper_w > 0.0
    assert r.nominal_loss.secondary_copper_w > 0.0
    assert len(r.workpoints) == 7


def test_n97_material_preset_is_supported():
    p = load_transformer_core_presets()["TDK_PQ35_35_B65881A_N97"]
    r = synthesize_transformer(LLCDesignSpec(), p)
    assert r.core.material_grade == "N97"
    assert r.nominal_loss.total_w > 0.0
