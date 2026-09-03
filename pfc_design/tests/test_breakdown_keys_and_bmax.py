"""Regression tests for the breakdown-key naming and inductor B-guard fixes.

1. Loss breakdown keys must say "_total" for n-phase totals (the values were
   already the phase-summed totals; "_per_phase" labels misled readers into
   multiplying by n again).
2. The inductor design loop must evaluate B_max with the peak current that is
   self-consistent with the realized (drooped) L_eff — the old loop used the
   target ripple and silently exceeded its own 0.7*Bsat red line.
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.models.system import SystemAnalyzer
from pfc_design.models.totem_pole import TotemPoleAnalyzer


def _bmax_from_metadata(design) -> float:
    """B_max at the metadata peak current (independent re-derivation)."""
    i_peak = design.design_metadata["IL_peak_with_ripple"]
    return (design.L_eff_at_ipeak_uh * 1e-6 * i_peak
            / (design.n_turns * design.ae_total_cm2 * 1e-4))


class TestBreakdownKeys:

    @pytest.mark.parametrize("analyzer", [SystemAnalyzer(), TotemPoleAnalyzer()])
    def test_no_per_phase_keys_in_breakdown(self, analyzer):
        r = analyzer.analyze(DesignSpec())
        assert r["breakdown"]
        assert not any("_per_phase" in k for k in r["breakdown"]), (
            f"misleading keys: {[k for k in r['breakdown'] if '_per_phase' in k]}")

    def test_breakdown_values_sum_to_consistent_total(self, shared_spec):
        """Each *_total value must be n * the per-phase sub-loss, and the
        per-phase group must aggregate to total_loss minus shared losses."""
        r = SystemAnalyzer().analyze(shared_spec)
        n = shared_spec.n_phases
        per_phase_keys = [k for k in r["breakdown"]
                          if k.endswith("_total")
                          and not k.startswith(("Bridge_", "Capacitor_"))]
        group = sum(r["breakdown"][k] for k in per_phase_keys)
        shared = r["breakdown"].get("Bridge_forward_Vf_total", 0) \
            + r["breakdown"].get("Bridge_forward_Rd_total", 0) \
            + r["breakdown"].get("Capacitor_ESR_total", 0)
        assert group == pytest.approx(n * r["per_phase_loss"], rel=1e-9)
        assert group + shared == pytest.approx(r["total_loss"], rel=1e-9)
        assert r["per_phase_loss"] > 0


class TestInductorDesignLoopConsistency:

    def test_bmax_guard_uses_self_consistent_peak_current(self, shared_spec,
                                                          shared_db):
        """A design that claims L_eff_target_met must keep B_max <= 0.7*Bsat
        with the true (fixed-point) peak current — the old loop's target-ripple
        basis could exceed its own red line."""
        spec = shared_spec.clone()
        spec.ripple_ratio = 0.30
        analyzer = SystemAnalyzer(shared_db)
        r = analyzer.analyze(spec)
        design = r["inductor_design"]
        dm = design.design_metadata
        assert dm["DeltaI_pp_actual"] > 0
        # If the target is claimed met, the guard must hold; if not, the
        # design must say so honestly (downstream rejects it).
        assert design.L_eff_at_ipeak_uh >= design.L_target_uh * 0.98 \
            or not dm["L_eff_target_met"]
        b = _bmax_from_metadata(design)
        assert b <= design.core.bs_T * 0.7 + 1e-9, (
            f"{design.core.part_number}: Bmax={b:.3f} T > "
            f"{design.core.bs_T*0.7:.3f} T (0.7*Bsat red line)")

    def test_peak_current_consistent_with_actual_ripple(self, shared_spec,
                                                        shared_db):
        """IL_peak_with_ripple must equal Iin_pk + Vin_pk*D_pk/(2*L_eff*fsw).

        Tolerance is loose (0.5%) because for undersized cores the healthy
        branch has no exact fixed point; the old code was off by ~58% here.
        """
        spec = shared_spec.clone()
        spec.ripple_ratio = 0.30
        analyzer = SystemAnalyzer(shared_db)
        r = analyzer.analyze(spec)
        design = r["inductor_design"]
        dm = design.design_metadata
        vin_pk = np.sqrt(2) * r["op"].vin_rms
        d_pk = 1.0 - vin_pk / spec.vout
        want_ripple = (vin_pk * d_pk
                       / (design.L_eff_at_ipeak_uh * 1e-6 * spec.fsw))
        assert dm["DeltaI_pp_actual"] == pytest.approx(want_ripple, rel=1e-6)
        assert dm["IL_peak_with_ripple"] == pytest.approx(
            dm["Iin_pk_phase"] + want_ripple / 2.0, rel=5e-3)
