"""Verify LineCycleTrace midpoint sampling and formula correctness."""

import numpy as np
import pytest

from pfc_design.core.line_cycle import build_line_cycle_trace


class TestLineCycleTrace:

    def test_midpoint_sampling_no_endpoints(self, shared_trace):
        """Theta[0] > 0 and theta[-1] < pi."""
        trace = shared_trace
        assert trace.theta[0] > 0, "First theta must be > 0 (midpoint)"
        assert trace.theta[-1] < np.pi, "Last theta must be < pi (midpoint)"

    def test_default_512_points(self, shared_op, shared_spec):
        """Default N=512 yields 512-point arrays."""
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        assert len(trace.theta) == 512
        assert len(trace.vin_abs) == 512
        assert len(trace.delta_i_pp) == 512

    def test_duty_at_line_peak(self, shared_op, shared_spec):
        """Duty at theta=pi/2 ≅ 1 - Vm/Vout."""
        trace = build_line_cycle_trace(shared_op, shared_spec, L_uh=200.0)
        idx_peak = np.argmin(np.abs(trace.theta - np.pi / 2))
        expected = 1.0 - shared_op.vm / shared_spec.vout
        assert trace.duty[idx_peak] == pytest.approx(expected, rel=0.01)

    def test_i_valley_nonnegative(self, shared_trace):
        """i_valley must be >= 0 everywhere (clamped)."""
        assert np.all(shared_trace.i_valley >= 0)

    def test_i_rms_local_sq_formula(self, shared_trace):
        """i_rms_local_sq == i_avg^2 + delta_i_pp^2 / 12."""
        t = shared_trace
        expected = t.i_avg_phase ** 2 + t.delta_i_pp ** 2 / 12.0
        np.testing.assert_allclose(t.i_rms_local_sq, expected, rtol=1e-12)

    def test_i_peak_formula(self, shared_trace):
        """i_peak = i_avg + delta_i_pp/2."""
        t = shared_trace
        expected = t.i_avg_phase + t.delta_i_pp / 2.0
        np.testing.assert_allclose(t.i_peak, expected, rtol=1e-12)

    def test_dcm_flag_near_zero_crossing(self, shared_trace):
        """near_zero_or_dcm should be True near theta→0 and theta→pi."""
        t = shared_trace
        # First few points should have high duty
        assert np.any(t.near_zero_or_dcm[:10])
        # Last few points should have high duty (near pi)
        assert np.any(t.near_zero_or_dcm[-10:])
        # Middle (near line peak) should be comfortable
        mid = len(t.theta) // 2
        assert not t.near_zero_or_dcm[mid]

    def test_zero_L_raises(self, shared_op, shared_spec):
        """L_uh=0 must raise ValueError."""
        with pytest.raises(ValueError):
            build_line_cycle_trace(shared_op, shared_spec, L_uh=0)

    def test_negative_L_raises(self, shared_op, shared_spec):
        """L_uh < 0 must raise ValueError."""
        with pytest.raises(ValueError):
            build_line_cycle_trace(shared_op, shared_spec, L_uh=-100)

    def test_theta_uniform_spacing(self, shared_trace):
        """Uniform dtheta across the grid."""
        dtheta_arr = np.diff(shared_trace.theta)
        np.testing.assert_allclose(dtheta_arr, shared_trace.dtheta, rtol=1e-10)
