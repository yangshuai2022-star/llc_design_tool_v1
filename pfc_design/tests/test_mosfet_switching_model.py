"""MOSFET switching loss: datasheet Eon/Eoff curve model.

Verifies:
  1. linear V/I scaling of reference energies
  2. fallback to the tr/tf ramp when no reference data exists
  3. model marker in loss metadata
"""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec, MosfetSpec
from pfc_design.core.operating_point import OperatingPoint
from pfc_design.core.line_cycle import build_line_cycle_trace, average_over_half_cycle
from pfc_design.models.mosfet import MosfetLoss


@pytest.fixture(scope="module")
def curve_spec():
    return DesignSpec(vout=410.0, fsw=65_000.0, pout_total=5500.0,
                      n_phases=2, eta_target=0.96)


@pytest.fixture(scope="module")
def base_mosfet():
    return MosfetSpec(
        manufacturer="Test", part_number="SW-TEST",
        technology="SiC", vds_max=650.0, id_25c=30.0, id_100c=20.0,
        rds_on_25c=0.040, rds_on_150c=0.060, rds_alpha=0.004,
        qg_nc=60.0, coss_er_pF=120.0, tr_ns=10.0, tf_ns=6.0, vgs=15.0,
    )


class TestEonEoffCurveModel:

    def test_linear_scaling_at_reference_point(self, curve_spec, base_mosfet):
        """E(400V, 20A) must equal Eon_ref exactly (1:1 at the ref point)."""
        m = MosfetSpec(**{**base_mosfet.__dict__,
                          "eon_ref_uj": 100.0, "eoff_ref_uj": 60.0,
                          "e_ref_v": 400.0, "e_ref_i": 20.0})
        # Constant current trace: I_valley = I_peak = 20 A at every theta
        op = OperatingPoint(curve_spec, vin_rms=176.0)
        trace = build_line_cycle_trace(op, curve_spec, L_uh=180.0)
        # Force flat currents via a custom trace substitute
        flat = trace.__class__(
            theta=trace.theta, theta_deg=trace.theta_deg, dtheta=trace.dtheta,
            vm=trace.vm, sin_theta=trace.sin_theta, vin_abs=trace.vin_abs,
            duty=trace.duty,
            i_avg_phase=np.full_like(trace.i_avg_phase, 20.0),
            delta_i_pp=trace.delta_i_pp * 0.0,
            i_peak=np.full_like(trace.i_peak, 20.0),
            i_valley=np.full_like(trace.i_valley, 20.0),
            i_rms_local_sq=np.full_like(trace.i_rms_local_sq, 20.0**2),
            iin_rms_phase=trace.iin_rms_phase, iin_pk_phase=trace.iin_pk_phase,
            L_uh=trace.L_uh, near_zero_or_dcm=trace.near_zero_or_dcm,
        )
        result = MosfetLoss().compute(curve_spec, op, mosfet=m, trace=flat)
        # E_mean = (100uJ + 60uJ) at V=410 (2.5% above ref) -> scaled by 410/400
        expected = 65_000.0 * (100.0 + 60.0) * 1e-6 * (410.0 / 400.0)
        assert result.sub_losses["switching"] == pytest.approx(expected,
                                                               rel=1e-6)
        assert result.metadata["switching_model"] == "datasheet_Eon_Eoff"

    def test_current_scaling_linear(self, curve_spec, base_mosfet):
        """Double the current -> double Eon (linear I scaling)."""
        m = MosfetSpec(**{**base_mosfet.__dict__,
                          "eon_ref_uj": 100.0, "eoff_ref_uj": 60.0,
                          "e_ref_v": 400.0, "e_ref_i": 20.0})
        op = OperatingPoint(curve_spec, vin_rms=176.0)
        trace = build_line_cycle_trace(op, curve_spec, L_uh=180.0)

        def p_sw(scale):
            t = trace.__class__(
                theta=trace.theta, theta_deg=trace.theta_deg,
                dtheta=trace.dtheta, vm=trace.vm, sin_theta=trace.sin_theta,
                vin_abs=trace.vin_abs, duty=trace.duty,
                i_avg_phase=trace.i_avg_phase,
                delta_i_pp=trace.delta_i_pp,
                i_peak=trace.i_peak * scale, i_valley=trace.i_valley * scale,
                i_rms_local_sq=trace.i_rms_local_sq * scale**2,
                iin_rms_phase=trace.iin_rms_phase,
                iin_pk_phase=trace.iin_pk_phase,
                L_uh=trace.L_uh, near_zero_or_dcm=trace.near_zero_or_dcm,
            )
            return MosfetLoss().compute(curve_spec, op, mosfet=m,
                                        trace=t).sub_losses["switching"]

        assert p_sw(2.0) == pytest.approx(2 * p_sw(1.0), rel=1e-9)

    def test_fallback_to_tr_tf_without_reference(self, curve_spec,
                                                 base_mosfet):
        """eon_ref/eoff_ref == 0 -> legacy linear-ramp estimate."""
        m = MosfetSpec(**base_mosfet.__dict__)   # no eon/eoff reference
        op = OperatingPoint(curve_spec, vin_rms=176.0)
        trace = build_line_cycle_trace(op, curve_spec, L_uh=180.0)
        result = MosfetLoss().compute(curve_spec, op, mosfet=m, trace=trace)
        expected = 0.5 * curve_spec.vout * (
            trace.i_valley * m.tr + trace.i_peak * m.tf)
        assert result.sub_losses["switching"] == pytest.approx(
            curve_spec.fsw * average_over_half_cycle(expected, trace.theta))
        assert result.metadata["switching_model"] == "linear_tr_tf"
