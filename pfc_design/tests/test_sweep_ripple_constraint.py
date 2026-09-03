"""P0.6: actual ripple feasibility constraint in optimizer sweep."""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec, MosfetSpec
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.models.system import SystemAnalyzer
from pfc_design.optimization.sweep import ParamSweep


@pytest.fixture(scope="module")
def sweep_spec():
    return DesignSpec(
        vin_min=176.0, vout=410.0, pout_total=7100.0, n_phases=2,
        fsw=65_000.0, ripple_ratio=0.3, eta_target=0.96,
    )


class TestRippleConstraintInSweep:

    def test_high_ripple_margin_marks_infeasible(self, sweep_spec, shared_db):
        """When actual_ripple_margin > limit, feasible_by_actual_ripple=False."""
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=2)
        mosfets = [sweeper.mosfet_db.query(vds_min=400, technology="Si")[0]]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array([0]),
            "mosfet_idx": np.array([0]),
        }

        # Default limits: margin=1.50, ratio=0.50
        # With ripple_ratio=0.3 and L_eff << L_target, margin > 1.50
        df = sweeper.sweep(
            sweep_spec, sweep_vars,
            cores=cores, mosfets=mosfets,
            actual_ripple_margin_limit=1.50,
            actual_ripple_ratio_limit=0.50,
            allow_aggressive_ripple=False,
        )

        assert len(df) == 1
        row = df.iloc[0]
        # With margin > 1.50, feasible_by_actual_ripple should be False
        # and the overall feasible should be False
        assert not row["feasible"]
        assert "ripple_constraint" in row["constraints"]

    def test_actual_ripple_ratio_limit_marks_infeasible(self, sweep_spec, shared_db):
        """When actual ripple ratio > 0.50, design is infeasible."""
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=2)
        mosfets = [sweeper.mosfet_db.query(vds_min=400, technology="Si")[0]]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array([0]),
            "mosfet_idx": np.array([0]),
        }

        # Tight limit: actual ripple (0.47) > limit (0.40) → infeasible
        df = sweeper.sweep(
            sweep_spec, sweep_vars,
            cores=cores, mosfets=mosfets,
            actual_ripple_margin_limit=2.0,   # margin won't trigger
            actual_ripple_ratio_limit=0.40,   # ratio WILL trigger
            allow_aggressive_ripple=False,
        )

        assert len(df) == 1
        row = df.iloc[0]
        assert not row["feasible"]
        assert "ripple_constraint" in row["constraints"]
        assert "ripple_ratio" in row["constraints"]

    def test_allow_aggressive_ripple_keeps_candidate_feasible_with_warning(
            self, sweep_spec, shared_db):
        """allow_aggressive_ripple=True: ripple is warning, not filter.

        Uses a tight margin_limit to guarantee ripple warning appears.
        Other constraints may still cause infeasibility — we only verify
        that ripple produces 'aggressive_ripple_warning' not 'ripple_constraint'.
        """
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=2)
        mosfets = [sweeper.mosfet_db.query(vds_min=400, technology="Si")[0]]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array([0]),
            "mosfet_idx": np.array([0]),
        }

        df = sweeper.sweep(
            sweep_spec, sweep_vars,
            cores=cores, mosfets=mosfets,
            actual_ripple_margin_limit=1.10,   # very tight
            actual_ripple_ratio_limit=0.50,
            allow_aggressive_ripple=True,        # ← keep feasible
        )
        row = df.iloc[0]
        # Ripple warning must appear, not a hard constraint
        assert "aggressive_ripple_warning" in row["constraints"]
        assert "ripple_constraint" not in row["constraints"]
        # feasible_by_actual_ripple == True since we're allowing aggressive ripple
        assert row["feasible_by_actual_ripple"] == True

    def test_optimizer_outputs_actual_ripple_columns(self, sweep_spec, shared_db):
        """Sweep output must contain all P0.6 required columns."""
        sweeper = ParamSweep(shared_db)
        cores = shared_db.top_for_pfc(ae_min_cm2=0.5, material_class="Sendust", n=2)
        mosfets = [sweeper.mosfet_db.query(vds_min=400, technology="Si")[0]]

        sweep_vars = {
            "fsw": np.array([65_000.0]),
            "ripple_ratio": np.array([0.3]),
            "core_idx": np.array([0]),
            "mosfet_idx": np.array([0]),
        }

        df = sweeper.sweep(sweep_spec, sweep_vars, cores=cores, mosfets=mosfets)
        row = df.iloc[0]

        required_cols = [
            "L_eff_ratio",
            "target_ripple_ratio",
            "actual_ripple_ratio_peak_basis",
            "actual_ripple_margin",
            "actual_ripple_margin_limit",
            "actual_ripple_risk_level",
            "feasible_by_actual_ripple",
        ]
        for col in required_cols:
            assert col in row.index, f"Missing column: {col}"

        # Values should be populated
        assert row["L_eff_ratio"] > 0
        assert row["target_ripple_ratio"] > 0
        assert row["actual_ripple_ratio_peak_basis"] > 0
        assert row["actual_ripple_margin"] > 0
        assert row["actual_ripple_margin_limit"] > 0
