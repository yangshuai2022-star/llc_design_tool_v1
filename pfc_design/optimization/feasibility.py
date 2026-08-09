"""Shared design-constraint feasibility checking.

Single source of truth for what makes a design point feasible or not.
Used by the grid sweep (ParamSweep), the local refine (scipy_opt) and
the global heuristic search (heuristic) so all search stages agree on
the exact same constraint thresholds and reason strings.
"""

from dataclasses import dataclass, field

from ..magnetics.core_entry import CoreSpec

# Constraint thresholds (must stay in sync with sweep defaults)
B_MAX_SAT_FACTOR = 0.7        # B_max < 0.7 * B_sat
WINDOW_FILL_LIMIT = 0.60      # window utilization < 60%
SATURATION_PCT_MIN = 20.0     # L_eff / L_noload > 20%
DEFAULT_RIPPLE_MARGIN_LIMIT = 1.50   # actual/target ripple ratio
DEFAULT_RIPPLE_RATIO_LIMIT = 0.50    # absolute ripple ratio at peak


@dataclass
class FeasibilityReport:
    """Outcome of the constraint checks for one design point."""
    feasible: bool
    failed_bmax: bool = False
    failed_window: bool = False
    failed_saturation: bool = False
    failed_actual_ripple: bool = False
    ripple_margin_exceeded: bool = False
    ripple_ratio_exceeded: bool = False
    actual_ripple_margin: float = 0.0
    actual_ripple_ratio_peak_basis: float = 0.0
    L_eff_ratio: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ripple_reasons: list[str] = field(default_factory=list)


def evaluate_feasibility(
    result: dict,
    core: CoreSpec,
    ripple_margin_limit: float = DEFAULT_RIPPLE_MARGIN_LIMIT,
    ripple_ratio_limit: float = DEFAULT_RIPPLE_RATIO_LIMIT,
) -> FeasibilityReport:
    """Classify one analyze() result against the engineering constraints.

    Args:
        result: dict returned by SystemAnalyzer.analyze()
        core: the core used in the design (for B_sat comparison)
        ripple_margin_limit: max actual/target ripple ratio
        ripple_ratio_limit: max absolute ripple ratio at line peak

    Returns:
        FeasibilityReport with per-constraint flags and formatted reasons.
    """
    design = result["inductor_design"]
    ind_meta = result["losses"]["inductor"].metadata
    im = result.get("inductor_metrics", {})

    reasons: list[str] = []
    ripple_reasons: list[str] = []

    b_max = ind_meta.get("B_max_T", 0.0)
    sat_pct = ind_meta.get("saturation_pct", 100.0)

    failed_bmax = b_max > core.bs_T * B_MAX_SAT_FACTOR
    failed_window = design.kw > WINDOW_FILL_LIMIT
    failed_saturation = sat_pct < SATURATION_PCT_MIN

    actual_ripple_margin = im.get("actual_ripple_margin", 0.0)
    actual_ripple_basis = im.get("actual_ripple_ratio_peak_basis", 0.0)
    l_eff_ratio = im.get("L_eff_ratio", 0.0)

    ripple_margin_exceeded = actual_ripple_margin > ripple_margin_limit
    ripple_ratio_exceeded = actual_ripple_basis > ripple_ratio_limit
    failed_actual_ripple = ripple_margin_exceeded or ripple_ratio_exceeded

    if failed_bmax:
        reasons.append(f"Bmax={b_max:.3f}T>{core.bs_T * B_MAX_SAT_FACTOR:.3f}T")
    if failed_window:
        reasons.append(f"kw={design.kw:.0%}>{WINDOW_FILL_LIMIT:.0%}")
    if failed_saturation:
        reasons.append(f"sat={sat_pct:.0f}%<{SATURATION_PCT_MIN:.0f}%")
    if ripple_margin_exceeded:
        ripple_reasons.append(
            f"ripple_margin={actual_ripple_margin:.2f}>{ripple_margin_limit:.2f}")
    if ripple_ratio_exceeded:
        ripple_reasons.append(
            f"ripple_ratio={actual_ripple_basis:.3f}>{ripple_ratio_limit:.3f}")
    reasons.extend(ripple_reasons)

    feasible = not (failed_bmax or failed_window
                    or failed_saturation or failed_actual_ripple)

    return FeasibilityReport(
        feasible=feasible,
        failed_bmax=failed_bmax,
        failed_window=failed_window,
        failed_saturation=failed_saturation,
        failed_actual_ripple=failed_actual_ripple,
        ripple_margin_exceeded=ripple_margin_exceeded,
        ripple_ratio_exceeded=ripple_ratio_exceeded,
        actual_ripple_margin=actual_ripple_margin,
        actual_ripple_ratio_peak_basis=actual_ripple_basis,
        L_eff_ratio=l_eff_ratio,
        reasons=reasons,
        ripple_reasons=ripple_reasons,
    )
