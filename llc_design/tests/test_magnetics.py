from llc_design.core.spec import LLCDesignSpec
from llc_design.magnetics.litz import round_wire_skin_factor, select_litz_wire
from llc_design.models.system import LLCSystemAnalyzer


def test_point_one_mm_strand_has_small_skin_penalty_at_100khz():
    factor = round_wire_skin_factor(0.1e-3, 100e3, 100.0)
    assert 1.0 <= factor < 1.02


def test_litz_selection_rounds_and_splits_large_secondary_bundle():
    wire = select_litz_wire(63.0, 0.1e-3, 0.112e-3, 0.55, 5.0)
    assert wire.strand_count % 25 == 0
    assert wire.strand_count >= 1600
    assert wire.sub_bundle_count >= 4


def test_baseline_magnetics_are_feasible_and_inductor_is_two_layers_max():
    result = LLCSystemAnalyzer().analyze(LLCDesignSpec())
    assert result.transformer.feasible
    assert result.resonant_inductor.feasible
    assert result.resonant_inductor.layers <= 2
    assert result.transformer.primary_wire.strand_copper_diameter_m == 0.1e-3
