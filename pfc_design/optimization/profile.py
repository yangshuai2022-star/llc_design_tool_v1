"""P0.7: optimizer result classification and reporting.

Design profiles encode per-application constraint sets.
All functions operate on a sweep DataFrame already populated
with actual-ripple and failure-flag columns.
"""

from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Design profiles
# ---------------------------------------------------------------------------

@dataclass
class DesignProfile:
    name: str
    actual_ripple_ratio_limit: float
    actual_ripple_margin_limit: float
    L_eff_ratio_min: float


DESIGN_PROFILE_PRODUCTION = DesignProfile(
    name="production",
    actual_ripple_ratio_limit=0.45,
    actual_ripple_margin_limit=1.25,
    L_eff_ratio_min=0.80,
)

DESIGN_PROFILE_BALANCED = DesignProfile(
    name="balanced",
    actual_ripple_ratio_limit=0.50,
    actual_ripple_margin_limit=1.50,
    L_eff_ratio_min=0.65,
)

DESIGN_PROFILE_AGGRESSIVE = DesignProfile(
    name="aggressive",
    actual_ripple_ratio_limit=0.60,
    actual_ripple_margin_limit=1.80,
    L_eff_ratio_min=0.50,
)

ALL_PROFILES = [
    DESIGN_PROFILE_PRODUCTION,
    DESIGN_PROFILE_BALANCED,
    DESIGN_PROFILE_AGGRESSIVE,
]


# ---------------------------------------------------------------------------
# Profile feasibility
# ---------------------------------------------------------------------------

def compute_profile_feasibility(
    df: pd.DataFrame, profile: DesignProfile
) -> pd.Series:
    """True where *all* profile constraints are satisfied on top of base feasible."""
    base = df["feasible"].fillna(False)
    ratio_ok = df["actual_ripple_ratio_peak_basis"] <= profile.actual_ripple_ratio_limit
    margin_ok = df["actual_ripple_margin"] <= profile.actual_ripple_margin_limit
    l_eff_ok = df["L_eff_ratio"] >= profile.L_eff_ratio_min
    return base & ratio_ok & margin_ok & l_eff_ok


# ---------------------------------------------------------------------------
# Primary reject reason (single most-important failure per row)
# ---------------------------------------------------------------------------

def _compute_primary_reject_reason(row: pd.Series) -> str:
    if row.get("feasible", False):
        return "feasible"
    if row.get("failed_bmax", False):
        return "bmax"
    if row.get("failed_window", False):
        return "window"
    if row.get("failed_saturation", False):
        return "saturation"
    if row.get("failed_actual_ripple", False):
        return "actual_ripple"
    if row.get("L_eff_ratio", 1.0) < 0.50:
        return "l_eff_ratio"
    constraints = row.get("constraints", "")
    if isinstance(constraints, str) and constraints.startswith("Error:"):
        return "error"
    return "unknown"


# ---------------------------------------------------------------------------
# Recommendation classes
# ---------------------------------------------------------------------------

def _append_label(existing: str, new: str) -> str:
    if not existing:
        return new
    return existing + ";" + new


def _assign_recommendation_classes(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("", index=df.index, dtype=str)

    mask_feasible = df["feasible"].fillna(False)
    if mask_feasible.any():
        best_idx = df.loc[mask_feasible, "P_total_W"].idxmin()
        labels.iloc[best_idx] = _append_label(
            labels.iloc[best_idx], "lowest_loss_feasible"
        )

    profile_col_to_label = [
        ("design_profile_feasible_production", "recommended_robust"),
        ("design_profile_feasible_balanced", "recommended_balanced"),
        ("design_profile_feasible_aggressive", "aggressive_low_loss"),
    ]
    for col, label in profile_col_to_label:
        mask = df[col].fillna(False)
        if mask.any():
            best_idx = df.loc[mask, "P_total_W"].idxmin()
            labels.iloc[best_idx] = _append_label(
                labels.iloc[best_idx], label
            )

    return labels


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify_results(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich sweep DataFrame with profile feasibility, reject reason, and
    recommendation labels. Returns a new DataFrame (original unchanged)."""
    df = df.copy()

    for profile in ALL_PROFILES:
        col = f"design_profile_feasible_{profile.name}"
        df[col] = compute_profile_feasibility(df, profile)

    df["primary_reject_reason"] = df.apply(_compute_primary_reject_reason, axis=1)
    df["recommendation_class"] = _assign_recommendation_classes(df)

    return df


# ---------------------------------------------------------------------------
# Extract recommendation rows
# ---------------------------------------------------------------------------

def get_recommended_designs(df: pd.DataFrame) -> dict:
    """Return the single best row for each recommendation class, or None."""
    results: dict = {}

    feasible = df[df["feasible"].fillna(False)]
    if len(feasible) > 0:
        results["lowest_loss_feasible"] = feasible.loc[
            feasible["P_total_W"].idxmin()
        ]
    else:
        results["lowest_loss_feasible"] = None

    profile_map = {
        "recommended_robust": "design_profile_feasible_production",
        "recommended_balanced": "design_profile_feasible_balanced",
        "aggressive_low_loss": "design_profile_feasible_aggressive",
    }
    for rec_name, col in profile_map.items():
        subset = df[df[col].fillna(False)]
        if len(subset) > 0:
            results[rec_name] = subset.loc[subset["P_total_W"].idxmin()]
        else:
            results[rec_name] = None

    return results


# ---------------------------------------------------------------------------
# Infeasible reason summary
# ---------------------------------------------------------------------------

def infeasible_reason_summary(df: pd.DataFrame) -> dict:
    """Count rows that fail each constraint.
    L_eff_ratio uses the production threshold (0.80) — the most informative
    for seeing how many designs are excluded by inductance droop."""
    return {
        "failed_bmax_count": int(df["failed_bmax"].fillna(False).sum()),
        "failed_window_count": int(df["failed_window"].fillna(False).sum()),
        "failed_saturation_count": int(df["failed_saturation"].fillna(False).sum()),
        "failed_actual_ripple_count": int(
            df["failed_actual_ripple"].fillna(False).sum()
        ),
        "failed_l_eff_ratio_count": int(
            (df["L_eff_ratio"].fillna(1.0) < 0.80).sum()
        ),
    }


# ---------------------------------------------------------------------------
# Diversity queries
# ---------------------------------------------------------------------------

def per_core_best(
    df: pd.DataFrame, feasible_col: str = "feasible"
) -> pd.DataFrame:
    """Lowest-loss feasible row for each distinct core."""
    subset = df[df[feasible_col].fillna(False)]
    if len(subset) == 0:
        return subset.iloc[:0]
    idx = subset.groupby("core")["P_total_W"].idxmin()
    return subset.loc[idx].sort_values("P_total_W")


def per_mosfet_best(
    df: pd.DataFrame, feasible_col: str = "feasible"
) -> pd.DataFrame:
    """Lowest-loss feasible row for each distinct MOSFET."""
    subset = df[df[feasible_col].fillna(False)]
    if len(subset) == 0:
        return subset.iloc[:0]
    idx = subset.groupby("mosfet")["P_total_W"].idxmin()
    return subset.loc[idx].sort_values("P_total_W")
