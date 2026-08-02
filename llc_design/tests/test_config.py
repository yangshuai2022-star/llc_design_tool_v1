from llc_design.core.config import load_spec, save_spec
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology


def test_config_round_trip(tmp_path):
    source = LLCDesignSpec(primary_topology=PrimaryTopology.HALF_BRIDGE,
                           primary_turns=16)
    path = save_spec(source, tmp_path / "spec.json")
    loaded = load_spec(path)
    assert loaded.primary_topology == PrimaryTopology.HALF_BRIDGE
    assert loaded.primary_turns == 16
    assert loaded.vout_v == 53.0
