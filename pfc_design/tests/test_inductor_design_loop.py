"""T1: inductor design loop targets effective inductance at peak current.

The design must honor the ripple spec through the DC-bias-drooped
inductance (L_eff at IL_peak), not the no-load inductance, while
respecting the B_max < 0.7*Bsat and window-fill guards. Cores that
cannot physically reach the target must report it honestly.
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.magnetics.core_database import CoreDatabase
from pfc_design.models.inductor import InductorDesigner


@pytest.fixture(scope="module")
def designer(shared_db):
    return InductorDesigner(shared_db)


class TestDesignLoopTargetsEffectiveL:

    def test_undersized_core_reports_target_unmet(self, shared_spec,
                                                  shared_db, designer):
        """0077083A7 cannot reach L_eff(Ipk) = L_target under Bmax guard.

        The loop must end at the best feasible N (Bmax boundary) and
        report L_eff_target_met=False with the limiting factor.
        """
        core = shared_db.get_by_part_number("0077083A7")
        spec = shared_spec.clone()
        spec.ripple_ratio = 0.30
        op = compute_mathcad_operating_point(spec)
        design = designer.design(spec, op, preferred_core=core, n_cores=2)

        assert not design.design_metadata["L_eff_target_met"]
        assert design.design_metadata["l_eff_limited_by"] in (
            "droop_peak", "iteration_cap", "window")
        assert design.L_eff_at_ipeak_uh < design.L_target_uh
        assert design.L_eff_at_ipeak_uh > 0

    def test_adequate_core_hits_effective_target(self, shared_spec,
                                                 shared_db, designer):
        """0077004A7 (Kool Mu 60u) supports the effective-L target.

        The design must land L_eff(Ipk) >= 98% of L_target — the whole
        point of the fix: ripple spec honored at the real operating point.
        """
        core = shared_db.get_by_part_number("0077004A7")
        spec = shared_spec.clone()
        spec.ripple_ratio = 0.30
        op = compute_mathcad_operating_point(spec)
        design = designer.design(spec, op, preferred_core=core, n_cores=2)

        assert design.design_metadata["L_eff_target_met"], (
            f"ratio_to_target="
            f"{design.design_metadata['L_eff_ratio_to_target']:.3f}")
        assert design.L_eff_at_ipeak_uh >= design.L_target_uh * 0.98
        # Droop compensation: no-load L must exceed the target
        assert design.L_noload_uh > design.L_target_uh
        # Bmax guard respected
        b_peak = design.L_eff_at_ipeak_uh * 1e-6 * (
            design.design_metadata["IL_peak_with_ripple"]) / (
            design.n_turns * design.ae_total_cm2 * 1e-4)
        assert b_peak <= core.bs_T * 0.7 + 1e-9

    def test_all_top_cores_honor_bmax_guard(self, shared_spec, shared_db,
                                            designer):
        """Designs that meet their target must be Bmax-safe.

        Cores that physically cannot satisfy B_max within the scan are
        allowed to fall back (they get flagged infeasible downstream),
        but every design that claims L_eff_target_met must respect the
        B_max < 0.7*Bsat guard.
        """
        cores = shared_db.top_for_pfc(
            ae_min_cm2=0.5, material_class="Sendust", n=10)
        spec = shared_spec.clone()
        spec.ripple_ratio = 0.30
        op = compute_mathcad_operating_point(spec)
        for core in cores:
            design = designer.design(spec, op,
                                     preferred_core=core, n_cores=2)
            if not design.design_metadata["L_eff_target_met"]:
                continue  # undersized core — flagged infeasible downstream
            b_peak = design.L_eff_at_ipeak_uh * 1e-6 * (
                design.design_metadata["IL_peak_with_ripple"]) / (
                design.n_turns * design.ae_total_cm2 * 1e-4)
            assert b_peak <= core.bs_T * 0.7 + 1e-9, (
                f"{core.part_number}: Bmax={b_peak:.3f}T > "
                f"{core.bs_T*0.7:.3f}T")

    def test_bigger_ripple_ratio_needs_smaller_inductance(self, shared_spec,
                                                          shared_db, designer):
        """Ripple 0.45 vs 0.25: target L scales down; design must follow."""
        core = shared_db.get_by_part_number("0077004A7")
        results = {}
        for ripple in (0.25, 0.45):
            spec = shared_spec.clone()
            spec.ripple_ratio = ripple
            op = compute_mathcad_operating_point(spec)
            design = designer.design(spec, op, preferred_core=core, n_cores=2)
            results[ripple] = design

        assert results[0.45].L_target_uh < results[0.25].L_target_uh
        assert results[0.45].n_turns <= results[0.25].n_turns
