"""P0.7: optimizer result classification and reporting."""

import numpy as np
import pandas as pd
import pytest

from pfc_design.core.spec import DesignSpec
from pfc_design.optimization.sweep import ParamSweep
from pfc_design.optimization.profile import (
    DesignProfile,
    DESIGN_PROFILE_PRODUCTION,
    DESIGN_PROFILE_BALANCED,
    DESIGN_PROFILE_AGGRESSIVE,
    compute_profile_feasibility,
    classify_results,
    get_recommended_designs,
    infeasible_reason_summary,
    per_core_best,
)


class TestProfileClassification:

    def test_production_profile_filters_high_warning_candidate(self):
        """A design with ripple_margin=1.30 (risk=high_warning) fails
        production (margin limit=1.25) but passes balanced (margin limit=1.50)."""
        df = pd.DataFrame([{
            "feasible": True,
            "actual_ripple_margin": 1.30,
            "actual_ripple_ratio_peak_basis": 0.35,
            "L_eff_ratio": 0.85,
            "P_total_W": 50.0,
            "failed_bmax": False,
            "failed_window": False,
            "failed_saturation": False,
            "failed_actual_ripple": False,
        }])

        prod_feasible = compute_profile_feasibility(df, DESIGN_PROFILE_PRODUCTION)
        assert not prod_feasible.iloc[0], "production should reject margin=1.30 > 1.25"

        bal_feasible = compute_profile_feasibility(df, DESIGN_PROFILE_BALANCED)
        assert bal_feasible.iloc[0], "balanced should accept margin=1.30 <= 1.50"

    def test_balanced_profile_accepts_current_candidate(self):
        """A design with ripple_margin=1.30, L_eff=0.85, ratio=0.40 passes
        balanced (margin<=1.50) and aggressive but not production (margin>1.25)."""
        df = pd.DataFrame([{
            "feasible": True,
            "actual_ripple_margin": 1.30,
            "actual_ripple_ratio_peak_basis": 0.40,
            "L_eff_ratio": 0.85,
            "P_total_W": 45.0,
            "failed_bmax": False,
            "failed_window": False,
            "failed_saturation": False,
            "failed_actual_ripple": False,
        }])

        # production: margin=1.30 > 1.25 → rejected
        assert not compute_profile_feasibility(df, DESIGN_PROFILE_PRODUCTION).iloc[0]
        # balanced: margin=1.30 <= 1.50 → accepted
        assert compute_profile_feasibility(df, DESIGN_PROFILE_BALANCED).iloc[0]
        # aggressive: margin=1.30 <= 1.80 → accepted
        assert compute_profile_feasibility(df, DESIGN_PROFILE_AGGRESSIVE).iloc[0]

    def test_recommended_robust_exists_or_reports_none(self, shared_db):
        """After classification, get_recommended_designs returns either
        a valid row or None for each recommendation class."""
        spec = DesignSpec(
            vin_min=176.0, vout=410.0, pout_total=7100.0, n_phases=2,
            fsw=65_000.0, ripple_ratio=0.3, eta_target=0.96,
        )
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=3)
        mosfets = sweeper.mosfet_db.query(vds_min=400, technology="Si")[:2]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array(range(len(cores))),
            "mosfet_idx": np.array(range(len(mosfets))),
        }

        df = sweeper.sweep(spec, sweep_vars, cores=cores, mosfets=mosfets)
        df = classify_results(df)
        recs = get_recommended_designs(df)

        for key in ["lowest_loss_feasible", "recommended_robust",
                     "recommended_balanced", "aggressive_low_loss"]:
            assert key in recs, f"Missing recommendation key: {key}"
            if recs[key] is not None:
                assert isinstance(recs[key], pd.Series)
                assert "P_total_W" in recs[key].index
                assert "core" in recs[key].index

    def test_infeasible_reason_summary_counts(self):
        """infeasible_reason_summary correctly counts each failure type."""
        df = pd.DataFrame([
            {
                "feasible": False,
                "failed_bmax": True, "failed_window": False,
                "failed_saturation": False, "failed_actual_ripple": False,
                "L_eff_ratio": 0.90, "P_total_W": 100.0,
                "constraints": "Bmax=0.310T>0.280T",
            },
            {
                "feasible": False,
                "failed_bmax": False, "failed_window": True,
                "failed_saturation": False, "failed_actual_ripple": False,
                "L_eff_ratio": 0.85, "P_total_W": 120.0,
                "constraints": "kw=65%>60%",
            },
            {
                "feasible": False,
                "failed_bmax": False, "failed_window": False,
                "failed_saturation": False, "failed_actual_ripple": True,
                "L_eff_ratio": 0.80, "P_total_W": 90.0,
                "constraints": "ripple_constraint: margin=1.60>1.50",
            },
            {
                "feasible": True,
                "failed_bmax": False, "failed_window": False,
                "failed_saturation": False, "failed_actual_ripple": False,
                "L_eff_ratio": 0.45, "P_total_W": 80.0,
                "constraints": "",
            },
        ])

        summary = infeasible_reason_summary(df)

        assert summary["failed_bmax_count"] == 1
        assert summary["failed_window_count"] == 1
        assert summary["failed_saturation_count"] == 0
        assert summary["failed_actual_ripple_count"] == 1
        assert summary["failed_l_eff_ratio_count"] == 1  # row 4: L_eff=0.45 < 0.80

    def test_per_core_best_outputs_multiple_cores_when_available(self, shared_db):
        """per_core_best returns one row per distinct core when multiple
        cores produce feasible designs."""
        spec = DesignSpec(
            vin_min=176.0, vout=410.0, pout_total=7100.0, n_phases=2,
            fsw=65_000.0, ripple_ratio=0.3, eta_target=0.96,
        )
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=4)
        mosfets = sweeper.mosfet_db.query(vds_min=400, technology="Si")[:1]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array(range(len(cores))),
            "mosfet_idx": np.array([0]),
        }

        df = sweeper.sweep(spec, sweep_vars, cores=cores, mosfets=mosfets)
        feasible_count = df["feasible"].sum()

        best_per_core = per_core_best(df)
        if feasible_count >= 2:
            unique_cores = best_per_core["core"].nunique()
            assert unique_cores >= 2, (
                f"Expected >=2 unique cores, got {unique_cores}"
            )
            assert len(best_per_core) == unique_cores
        else:
            assert len(best_per_core) <= feasible_count
