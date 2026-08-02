"""Verify MOSFET line-cycle-integrated conduction and switching losses."""

import numpy as np
import pytest

from pfc_design.core.line_cycle import build_line_cycle_trace, average_over_half_cycle
from pfc_design.models.mosfet import MosfetLoss
from pfc_design.core.spec import MosfetSpec


@pytest.fixture
def test_mosfet():
    """100 mOhm SiC MOSFET for testing."""
    return MosfetSpec(
        manufacturer="Test", part_number="Test-100m",
        technology="SiC", vds_max=650.0, id_25c=47.0, id_100c=30.0,
        rds_on_25c=0.100, rds_on_150c=0.160, rds_alpha=0.004,
        qg_nc=100.0, coss_er_pF=200.0, tr_ns=5.0, tf_ns=4.5, vgs=14.0
    )


class TestMosfetConductionLoss:

    def test_conduction_degenerates_no_ripple(self, shared_spec, shared_op, test_mosfet):
        """When delta_i=0, Pcond = Rds * mean(D * i_avg^2)."""
        L_large = 1e9  # huge L → negligible ripple
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=L_large)

        rds = 0.100
        m = MosfetLoss()
        p_trace = m._conduction_loss_line_cycle(trace, rds)

        # Manual: Rds * mean(D * i_avg^2)
        expected = rds * average_over_half_cycle(
            trace.duty * trace.i_avg_phase ** 2, trace.theta
        )
        assert p_trace == pytest.approx(expected, rel=1e-6)

    def test_conduction_includes_ripple(self, shared_spec, shared_op, test_mosfet):
        """With nonzero ripple, line-cycle loss > without-ripple version."""
        rds = 0.100
        m = MosfetLoss()

        # Normal L → nonzero ripple
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        p_with_ripple = m._conduction_loss_line_cycle(trace, rds)

        # Very large L → negligible ripple
        trace_no_rip = build_line_cycle_trace(shared_op, shared_spec, L_uh=1e9)
        p_no_ripple = m._conduction_loss_line_cycle(trace_no_rip, rds)

        assert p_with_ripple > p_no_ripple * 1.001, \
            f"With ripple: {p_with_ripple:.4f}, no ripple: {p_no_ripple:.4f}"


class TestMosfetSwitchingLoss:

    def test_switching_uses_valley_for_turnon(self, shared_spec, shared_op):
        """Eon must use I_valley, not I_avg."""
        tr = 5e-9
        tf = 4.5e-9
        m = MosfetLoss()

        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        p_sw = m._switching_loss_line_cycle(shared_spec, trace, tr, tf)

        # Manual check: each angle uses I_valley for Eon, I_peak for Eoff
        e_on = 0.5 * shared_spec.vout * trace.i_valley * tr
        e_off = 0.5 * shared_spec.vout * trace.i_peak * tf
        e_mean = average_over_half_cycle(e_on + e_off, trace.theta)
        expected = shared_spec.fsw * e_mean

        assert p_sw == pytest.approx(expected, rel=1e-10)

    def test_switching_valley_nonnegative_in_action(self, shared_spec, shared_op):
        """Even near zero-crossing, switching loss is finite (valley clamped)."""
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        m = MosfetLoss()
        p = m._switching_loss_line_cycle(shared_spec, trace, 5e-9, 5e-9)
        assert p > 0
        assert np.isfinite(p)

    def test_coss_unchanged_by_trace(self, shared_spec, shared_op, test_mosfet):
        """Coss loss should be the same with or without trace."""
        m = MosfetLoss()
        p_coss_direct = m._coss_loss(shared_spec, test_mosfet.coss_er)
        expected = 0.5 * test_mosfet.coss_er * shared_spec.vout ** 2 * shared_spec.fsw
        assert p_coss_direct == pytest.approx(expected, rel=1e-10)

    def test_compute_with_trace_uses_line_cycle(self, shared_spec, shared_op, test_mosfet):
        """When trace is provided, compute() uses line-cycle methods."""
        m = MosfetLoss()
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        result = m.compute(shared_spec, shared_op, mosfet=test_mosfet, trace=trace)
        assert result.metadata["loss_model"] == "line_cycle"
        assert "risk_flag" in result.metadata
        assert "coss_accounting_note" in result.metadata
