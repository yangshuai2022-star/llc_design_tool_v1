"""Totem-pole bridgeless PFC model vs bridge-boost baseline.

Verifies:
  1. algebraic identities (SR+active conduction = Rds * full RMS^2)
  2. deadtime loss scales linearly with t_dead
  3. slow-leg conduction = I_rms^2 * Rds
  4. totem beats boost (bridge + boost diode removed)
  5. identical magnetics in both topologies
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec, MosfetDatabase
from pfc_design.core.line_cycle import average_over_half_cycle
from pfc_design.models.system import SystemAnalyzer
from pfc_design.models.totem_pole import TotemPoleAnalyzer


@pytest.fixture(scope="module")
def totem_spec():
    return DesignSpec(diode_part="C4D10120H")


class TestTotemModelIdentities:

    def test_sr_plus_active_conduction_equals_full_rms2(
            self, totem_spec, shared_db):
        """Active conducts during D, SR during 1-D with the same Rds.

        Their sum must equal Rds * mean(i_rms_local_sq) — the exact
        benefit of synchronous rectification (no diode drop at all).
        """
        r = TotemPoleAnalyzer(shared_db).analyze(totem_spec)
        mos = r["mosfet"]
        trace = r["trace"]
        rds = mos.rds_on_25c * (1 + mos.rds_alpha * (80 - 25))
        expected = rds * average_over_half_cycle(trace.i_rms_local_sq,
                                                 trace.theta)
        p_cond_active = r["losses"]["active"].sub_losses["conduction"]
        p_cond_sr = r["losses"]["sr"].sub_losses["conduction"]
        assert p_cond_active + p_cond_sr == pytest.approx(expected, rel=1e-9)

    def test_deadtime_loss_scales_with_t_dead(self, totem_spec, shared_db):
        t = TotemPoleAnalyzer(shared_db)
        r1 = t.analyze(totem_spec, deadtime_s=50e-9)
        r2 = t.analyze(totem_spec, deadtime_s=100e-9)
        p1 = r1["losses"]["sr"].sub_losses["deadtime_body"]
        p2 = r2["losses"]["sr"].sub_losses["deadtime_body"]
        assert p2 == pytest.approx(2 * p1, rel=1e-6)

    def test_slow_leg_conduction_is_iin_rms2_rdstj(self, totem_spec,
                                                   shared_db):
        r = TotemPoleAnalyzer(shared_db).analyze(totem_spec)
        slow = r["slow_mosfet"]
        rds = slow.rds_on_25c * (1 + slow.rds_alpha * (80 - 25))
        expected = r["op"].iin_rms ** 2 * rds
        p_cond = r["losses"]["slow_leg"].sub_losses["conduction"]
        assert p_cond == pytest.approx(expected, rel=1e-9)


class TestTotemVsBoost:

    def test_totem_wins_over_bridge_boost(self, totem_spec, shared_db):
        boost = SystemAnalyzer(shared_db).analyze(totem_spec)
        totem = TotemPoleAnalyzer(shared_db).analyze(totem_spec)
        assert totem["efficiency"] > boost["efficiency"]
        assert totem["total_loss"] < boost["total_loss"]
        # The bridge and the boost diode are gone
        assert not any(k.startswith("Bridge_") for k in totem["breakdown"])
        assert not any(k.startswith("Diode_") for k in totem["breakdown"])
        assert any(k.startswith("Bridge_") for k in boost["breakdown"])
        assert any(k.startswith("Diode_") for k in boost["breakdown"])

    def test_identical_magnetics_in_both(self, totem_spec, shared_db):
        boost = SystemAnalyzer(shared_db).analyze(totem_spec)
        totem = TotemPoleAnalyzer(shared_db).analyze(totem_spec)
        b_design = boost["inductor_design"]
        t_design = totem["inductor_design"]
        assert b_design.core.part_number == t_design.core.part_number
        assert b_design.n_turns == t_design.n_turns
        assert b_design.L_eff_at_ipeak_uh == pytest.approx(
            t_design.L_eff_at_ipeak_uh)

    def test_inductor_metrics_identical(self, totem_spec, shared_db):
        boost = SystemAnalyzer(shared_db).analyze(totem_spec)
        totem = TotemPoleAnalyzer(shared_db).analyze(totem_spec)
        for key in ("actual_ripple_margin", "actual_ripple_risk_level",
                    "L_eff_ratio"):
            assert boost["inductor_metrics"][key] == pytest.approx(
                totem["inductor_metrics"][key])


class TestTotemThermal:

    def test_thermal_on_raises_tj_and_loss(self, totem_spec, shared_db):
        spec = totem_spec.clone()
        spec.thermal_model = True
        t = TotemPoleAnalyzer(shared_db)
        r_off = t.analyze(totem_spec)
        r_on = t.analyze(spec)
        th = r_on["thermal"]
        assert th["enabled"]
        assert th["t_j_hf"] > spec.t_ambient
        assert r_on["total_loss"] >= r_off["total_loss"] - 1e-9
