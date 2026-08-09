"""Continuous local refinement using scipy.

Continuous design variables: fsw (kHz) and ripple_ratio. Core and
MOSFET are discrete and fixed for a given refinement run (selected by
the grid sweep or passed explicitly).

The optimizer minimizes a *feasibility-penalized* total loss so that
infeasible designs are pushed far above any feasible one; this lets a
derivative-free method (Nelder-Mead, robust to the small discontinuities
caused by integer turn counts) polish a seed design within the bounds.
"""

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..core.spec import DesignSpec, MosfetSpec
from ..core.operating_point import compute_mathcad_operating_point
from ..magnetics.core_database import CoreDatabase
from ..magnetics.core_entry import CoreSpec
from ..models.system import SystemAnalyzer
from .feasibility import (
    DEFAULT_RIPPLE_MARGIN_LIMIT, DEFAULT_RIPPLE_RATIO_LIMIT, evaluate_feasibility,
)

# Penalty per violated constraint — dominates any achievable loss (~300 W),
# so every feasible design ranks below every infeasible one.
PENALTY = 1e5

DEFAULT_FSW_BOUNDS = (45.0, 120.0)     # kHz
DEFAULT_RIPPLE_BOUNDS = (0.10, 0.60)   # ripple ratio


def make_penalty_objective(
    analyzer: SystemAnalyzer,
    base_spec: DesignSpec,
    core: CoreSpec,
    mosfet: MosfetSpec | None,
    n_cores: int = 2,
    ripple_margin_limit: float = DEFAULT_RIPPLE_MARGIN_LIMIT,
    ripple_ratio_limit: float = DEFAULT_RIPPLE_RATIO_LIMIT,
) -> Callable[[np.ndarray], float]:
    """Build f(x) = penalized total loss, x = (fsw_kHz, ripple_ratio).

    The returned objective clones the base spec for every evaluation
    (no in-place mutation) and adds PENALTY per violated constraint.
    """
    def objective(x: np.ndarray) -> float:
        fsw_kHz, ripple = float(x[0]), float(x[1])
        spec = base_spec.clone()
        spec.fsw = fsw_kHz * 1000.0
        spec.ripple_ratio = ripple
        try:
            op = compute_mathcad_operating_point(spec)
            result = analyzer.analyze(spec, op, preferred_core=core,
                                      mosfet=mosfet, n_cores=n_cores)
        except Exception:
            return PENALTY * 10.0
        report = evaluate_feasibility(result, core,
                                      ripple_margin_limit, ripple_ratio_limit)
        loss = float(result["total_loss"])
        if report.feasible:
            return loss
        n_failed = (report.failed_bmax + report.failed_window
                    + report.failed_saturation + report.failed_actual_ripple)
        return PENALTY * n_failed + loss

    return objective


def refine_design(
    analyzer: SystemAnalyzer,
    base_spec: DesignSpec,
    core: CoreSpec,
    mosfet: MosfetSpec | None,
    x0: tuple[float, float],
    n_cores: int = 2,
    fsw_bounds: tuple[float, float] = DEFAULT_FSW_BOUNDS,
    ripple_bounds: tuple[float, float] = DEFAULT_RIPPLE_BOUNDS,
    ripple_margin_limit: float = DEFAULT_RIPPLE_MARGIN_LIMIT,
    ripple_ratio_limit: float = DEFAULT_RIPPLE_RATIO_LIMIT,
    method: str = "Nelder-Mead",
    maxiter: int = 200,
) -> dict:
    """Local search from a seed (fsw_kHz, ripple_ratio) for one core/MOSFET.

    Returns:
        dict with best design, its loss/efficiency/feasibility and
        the number of objective evaluations.
    """
    objective = make_penalty_objective(
        analyzer, base_spec, core, mosfet, n_cores,
        ripple_margin_limit, ripple_ratio_limit,
    )
    bounds = [fsw_bounds, ripple_bounds]
    x0_arr = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    res = minimize(objective, x0_arr, method=method, bounds=bounds,
                   options={"maxiter": maxiter, "xatol": 1e-3,
                            "fatol": 1e-3, "disp": False})

    return _finalize(analyzer, base_spec, core, mosfet, res.x, n_cores,
                     ripple_margin_limit, ripple_ratio_limit,
                     evaluations=int(res.nfev), success=bool(res.success))


def optimize_loss(spec: DesignSpec, core: CoreSpec, db: CoreDatabase,
                  initial: tuple[float, float] = (65.0, 0.30),
                  bounds: tuple[tuple[float, float], tuple[float, float]] =
                  (DEFAULT_FSW_BOUNDS, DEFAULT_RIPPLE_BOUNDS)) -> dict:
    """Refine (fsw_kHz, ripple_ratio) for a fixed core, auto-selected MOSFET.

    Convenience entry point equivalent to refine_design with the
    SystemAnalyzer's default MOSFET selection (mosfet=None).
    """
    analyzer = SystemAnalyzer(db)
    return refine_design(analyzer, spec, core, mosfet=None,
                         x0=initial, fsw_bounds=bounds[0], ripple_bounds=bounds[1])


def _finalize(analyzer: SystemAnalyzer, base_spec: DesignSpec, core: CoreSpec,
              mosfet: MosfetSpec | None, x: np.ndarray, n_cores: int,
              ripple_margin_limit: float, ripple_ratio_limit: float,
              evaluations: int, success: bool) -> dict:
    """Re-evaluate the best point and build the result dict."""
    fsw_kHz, ripple = float(x[0]), float(x[1])
    spec = base_spec.clone()
    spec.fsw = fsw_kHz * 1000.0
    spec.ripple_ratio = ripple
    op = compute_mathcad_operating_point(spec)
    result = analyzer.analyze(spec, op, preferred_core=core,
                              mosfet=mosfet, n_cores=n_cores)
    report = evaluate_feasibility(result, core,
                                  ripple_margin_limit, ripple_ratio_limit)
    return {
        "fsw_kHz": fsw_kHz,
        "ripple_ratio": ripple,
        "core": core.part_number,
        "mosfet": (mosfet.part_number if mosfet is not None
                   else result["mosfet"].part_number),
        "loss_W": float(result["total_loss"]),
        "efficiency_pct": float(result["efficiency"]) * 100.0,
        "feasible": report.feasible,
        "actual_ripple_margin": report.actual_ripple_margin,
        "reasons": report.reasons,
        "evaluations": evaluations,
        "success": success,
    }
