"""Verify shared components use total (not per-phase) power/current basis."""

import numpy as np
import pytest

from pfc_design.core.spec import DesignSpec
from pfc_design.core.operating_point import compute_mathcad_operating_point
from pfc_design.models.bridge_rectifier import BridgeRectifierLoss
from pfc_design.models.capacitor import CapacitorBank


class TestBridgeRectifier:

    def test_bridge_uses_total_input_current(self, shared_spec, shared_op):
        """Bridge must use total line current = n_phases * per-phase I_rms."""
        br = BridgeRectifierLoss()
        result = br.compute(shared_spec, shared_op)

        # Manual check: total input current
        pin_total = shared_spec.pout_total / shared_spec.eta_target
        iin_rms_total = pin_total / shared_op.vin_rms
        iin_pk_total = np.sqrt(2) * iin_rms_total
        iin_avg_total = 2 / np.pi * iin_pk_total

        expected_vf = 2 * shared_spec.bridge_vf * iin_avg_total
        assert result.sub_losses["forward_Vf"] == pytest.approx(expected_vf, rel=1e-6)
        assert result.metadata["power_basis"] == "total_input_current"

    def test_bridge_result_is_total_not_per_phase(self, shared_spec, shared_op):
        """Bridge loss is total-system W, not per-phase."""
        br = BridgeRectifierLoss()
        result = br.compute(shared_spec, shared_op)
        # Bridge is shared — result is total watts
        assert result.power_loss_W > 0
        assert "total" in result.component.lower()


class TestCapacitorBank:

    def test_LF_ripple_uses_pout_total(self, shared_spec, shared_op):
        """Low-frequency ripple must use pout_total, not pout_per_phase."""
        cap = CapacitorBank()
        result = cap.compute(shared_spec, shared_op)

        # Verify DeltaV_lf uses pout_total
        c = shared_spec.c_out_total
        f_line = shared_spec.f_line
        vout = shared_spec.vout
        expected_lf = shared_spec.pout_total / (2 * np.pi * f_line * c * vout)

        assert result.metadata["delta_V_LF_Vpp"] == pytest.approx(expected_lf, rel=1e-6)

    def test_LF_ripple_independent_of_n_phases(self, shared_spec, shared_op):
        """Same total power → same LF ripple regardless of n_phases."""
        spec_1ph = DesignSpec(**{**shared_spec.__dict__, "n_phases": 1})
        spec_2ph = DesignSpec(**{**shared_spec.__dict__, "n_phases": 2, "pout_total": 7100})

        cap = CapacitorBank()
        r1 = cap.compute(spec_1ph, shared_op)
        r2 = cap.compute(spec_2ph, shared_op)
        # Same pout_total → same LF ripple
        assert r1.metadata["delta_V_LF_Vpp"] == pytest.approx(
            r2.metadata["delta_V_LF_Vpp"], rel=1e-6)

    def test_Ic_lf_rms_independent_of_C(self, shared_spec, shared_op):
        """Ic_lf_rms does not depend on capacitor value (only power and Vout)."""
        cap = CapacitorBank()
        expected = shared_spec.pout_total / (np.sqrt(2) * shared_spec.vout)
        result = cap.compute(shared_spec, shared_op)
        assert result.metadata["Ic_lf_rms_A"] == pytest.approx(expected, rel=1e-6)

    def test_capacitor_power_basis_is_total(self, shared_spec, shared_op):
        """Metadata must declare power_basis = pout_total."""
        cap = CapacitorBank()
        result = cap.compute(shared_spec, shared_op)
        assert result.metadata["power_basis"] == "pout_total"

    def test_switching_ripple_warning_present(self, shared_spec, shared_op):
        """Simplified switching ripple model must have warning."""
        cap = CapacitorBank()
        result = cap.compute(shared_spec, shared_op)
        assert "switching_ripple_warning" in result.metadata
