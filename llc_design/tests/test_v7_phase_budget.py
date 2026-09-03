import numpy as np

from llc_design.control.phase_budget import phase_budget


def test_phase_budget_interpolates_on_log_frequency():
    f = np.array([10.0, 100.0, 1000.0])
    response = np.array([1+0j, 0.1-0.1j, 0.01-0.01j])
    budget = phase_budget(f, {"x": response}, {"x": "X"}, 100.0, ["x"])
    assert len(budget) == 1
    assert budget[0].label == "X"
    assert abs(budget[0].gain_db - 20*np.log10(abs(response[1]))) < 1e-10
