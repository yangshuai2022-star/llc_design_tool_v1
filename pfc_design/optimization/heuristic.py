"""Global heuristic search over the mixed design space.

The grid sweep enumerates discrete choices densely but coarsely in the
continuous dimensions. differential_evolution searches fsw and ripple
jointly with the discrete core/MOSFET indices, using the same
feasibility-penalized objective as the local refine — so it is a true
global search over the whole candidate set, not a local polish.
"""

import numpy as np
from scipy.optimize import differential_evolution

from ..core.spec import DesignSpec, MosfetSpec
from ..core.operating_point import compute_mathcad_operating_point
from ..magnetics.core_entry import CoreSpec
from ..models.system import SystemAnalyzer
from .feasibility import (
    DEFAULT_RIPPLE_MARGIN_LIMIT, DEFAULT_RIPPLE_RATIO_LIMIT, evaluate_feasibility,
)
from .scipy_opt import (
    DEFAULT_FSW_BOUNDS, DEFAULT_RIPPLE_BOUNDS, make_penalty_objective,
)


def global_search(
    analyzer: SystemAnalyzer,
    base_spec: DesignSpec,
    cores: list[CoreSpec],
    mosfets: list[MosfetSpec],
    n_cores: int = 2,
    fsw_bounds: tuple[float, float] = DEFAULT_FSW_BOUNDS,
    ripple_bounds: tuple[float, float] = DEFAULT_RIPPLE_BOUNDS,
    ripple_margin_limit: float = DEFAULT_RIPPLE_MARGIN_LIMIT,
    ripple_ratio_limit: float = DEFAULT_RIPPLE_RATIO_LIMIT,
    popsize: int = 12,
    maxiter: int = 50,
    seed: int = 42,
    polish: bool = True,
) -> dict:
    """Differential-evolution search over (fsw, ripple, core, MOSFET).

    x = (fsw_kHz, ripple_ratio, core_idx, mosfet_idx); the two indices
    are declared integral so scipy rounds them to valid candidates.

    Returns:
        dict with the best design found, its loss/efficiency/
        feasibility and the number of objective evaluations.
    """
    n_core, n_mos = len(cores), len(mosfets)
    if n_core == 0 or n_mos == 0:
        raise ValueError("global_search needs non-empty core and MOSFET lists")

    bounds = [(fsw_bounds[0], fsw_bounds[1]),
              (ripple_bounds[0], ripple_bounds[1]),
              (0.0, float(n_core - 1)),
              (0.0, float(n_mos - 1))]
    integrality = np.array([False, False, True, True])

    def objective(z: np.ndarray) -> float:
        core = cores[int(round(z[2])) % n_core]
        mosfet = mosfets[int(round(z[3])) % n_mos]
        obj = make_penalty_objective(analyzer, base_spec, core, mosfet, n_cores,
                                     ripple_margin_limit, ripple_ratio_limit)
        return obj([z[0], z[1]])

    res = differential_evolution(
        objective, bounds, integrality=integrality,
        popsize=popsize, maxiter=maxiter, seed=seed, polish=polish,
        mutation=(0.5, 1.0), recombination=0.8,
        tol=1e-8, updating="immediate", workers=1,
    )

    core = cores[int(round(res.x[2])) % n_core]
    mosfet = mosfets[int(round(res.x[3])) % n_mos]

    # Re-evaluate the best point to recover the full design details
    spec = base_spec.clone()
    spec.fsw = float(res.x[0]) * 1000.0
    spec.ripple_ratio = float(res.x[1])
    op = compute_mathcad_operating_point(spec)
    result = analyzer.analyze(spec, op, preferred_core=core,
                              mosfet=mosfet, n_cores=n_cores)
    report = evaluate_feasibility(result, core,
                                  ripple_margin_limit, ripple_ratio_limit)

    return {
        "fsw_kHz": float(res.x[0]),
        "ripple_ratio": float(res.x[1]),
        "core": core.part_number,
        "mosfet": mosfet.part_number,
        "loss_W": float(result["total_loss"]),
        "efficiency_pct": float(result["efficiency"]) * 100.0,
        "feasible": report.feasible,
        "actual_ripple_margin": report.actual_ripple_margin,
        "reasons": report.reasons,
        "evaluations": int(res.nfev),
    }
