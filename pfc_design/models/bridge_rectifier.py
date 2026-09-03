"""Bridge rectifier loss model.

Always 2 diodes conducting in series.
"""

import numpy as np

from ..core.spec import DesignSpec
from ..core.operating_point import OperatingPoint
from .base import LossResult


class BridgeRectifierLoss:
    """Bridge rectifier loss (total, shared across phases).

    The input bridge carries the total line current, not per-phase current.
    """

    def compute(self, spec: DesignSpec, op: OperatingPoint) -> LossResult:
        """Compute bridge rectifier loss.

        Uses the minimum input voltage operating point (worst case).
        Total input current (n_phases * per_phase RMS).

        Pbridge = 2 * Vf * Iin_abs_avg_total + 2 * Rd * Iin_rms_total^2
        """
        # Total input current — bridge is a shared component
        pin_total = spec.pout_total / spec.eta_target
        iin_rms_total = pin_total / op.vin_rms
        iin_pk_total = np.sqrt(2) * iin_rms_total
        iin_abs_avg_total = 2 / np.pi * iin_pk_total

        # Rd term (defaults to 0 if not on spec)
        bridge_rd = getattr(spec, 'bridge_rd', 0.0)

        p_loss = 2 * spec.bridge_vf * iin_abs_avg_total + 2 * bridge_rd * iin_rms_total ** 2

        return LossResult(
            component="Bridge Rectifier (total)",
            power_loss_W=float(p_loss),
            sub_losses={
                "forward_Vf": float(2 * spec.bridge_vf * iin_abs_avg_total),
                "forward_Rd": float(2 * bridge_rd * iin_rms_total ** 2),
            },
            metadata={
                "Iin_rms_total": iin_rms_total,
                "Iin_pk_total": iin_pk_total,
                "Iin_abs_avg_total": iin_abs_avg_total,
                "Vf_per_diode": spec.bridge_vf,
                "Rd_per_diode": bridge_rd,
                "power_basis": "total_input_current",
            }
        )
