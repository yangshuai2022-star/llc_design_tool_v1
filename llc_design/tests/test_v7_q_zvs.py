import numpy as np

from llc_design.core.q_zvs import build_q_zvs_analysis
from llc_design.core.spec import LLCDesignSpec


def test_q_zvs_map_shape_and_q_increases_with_load():
    result = build_q_zvs_analysis(LLCDesignSpec(), frequency_points=100)
    assert result.map.gain.shape == (6, 100)
    assert result.map.zvs_margin.shape == (6, 100)
    assert np.all(np.diff(result.map.q_effective) > 0.0)
    assert np.all(np.isfinite(result.map.gain))


def test_q_zvs_workpoints_are_bound_to_actual_llc_solution():
    result = build_q_zvs_analysis(
        LLCDesignSpec(), load_fractions=(0.25, 0.5, 1.0),
        vbus_points=(360.0, 400.0, 420.0), frequency_points=90,
    )
    assert len(result.workpoints) >= 8
    for point in result.workpoints:
        assert 0.0 < point.frequency_hz
        assert point.q_effective > 0.0
        assert point.zvs_margin >= 0.0
        assert point.zvs_status in {"CAPACITIVE", "ZVS_FAIL", "ZVS_WARNING", "ZVS_SAFE"}
