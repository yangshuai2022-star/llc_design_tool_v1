"""Output capacitor model for PFC.

Computes:
- Low-frequency voltage ripple (2x line frequency)
- Switching-frequency voltage ripple
- RMS ripple current
- ESR power loss
- Theoretical capacitor lifetime
"""

import numpy as np

from ..core.spec import DesignSpec
from ..core.operating_point import OperatingPoint
from .base import LossResult


class CapacitorBank:
    """Output capacitor bank design and loss model.

    The capacitor is a shared component — all ripple calculations
    use total output power (not per-phase).
    """

    def compute(self, spec: DesignSpec, op: OperatingPoint) -> LossResult:
        """Compute capacitor bank losses and lifetime.

        Returns:
            LossResult with ESR loss, ripple data, and lifetime estimate
        """
        c_total = spec.c_out_total
        esr_total = spec.cap_esr / spec.cap_n_parallel
        f_line = spec.f_line
        vout = spec.vout

        # Low-frequency ripple current RMS (independent of C)
        # For PFC output: Ic_lf_rms = Pout_total / (sqrt(2) * Vout)
        ic_lf_rms = spec.pout_total / (np.sqrt(2) * vout)

        # Low-frequency voltage ripple pk-pk (depends on C)
        # DeltaV_lf_pp = Pout_total / (2 * pi * f_line * C_total * Vout)
        delta_v_lf = spec.pout_total / (2 * np.pi * f_line * c_total * vout)

        # High-frequency switching ripple — simplified for v1
        # Two-phase interleaving provides partial cancellation
        delta_i_sw = spec.ripple_ratio * op.iin_peak
        delta_v_sw = delta_i_sw / (8 * spec.fsw * c_total)

        # Total RMS capacitor current (approximate)
        # For v1: use LF component as dominant
        ic_rms = ic_lf_rms

        # ESR power loss (LF-dominant for v1)
        p_esr = ic_rms ** 2 * esr_total

        # Capacitor lifetime estimation
        lifetime_hours = self._lifetime(spec, p_esr)

        return LossResult(
            component="Output Capacitor (total)",
            power_loss_W=float(p_esr),
            sub_losses={"ESR": float(p_esr)},
            metadata={
                "C_total_uF": c_total * 1e6,
                "ESR_total_Ohm": esr_total,
                "Ic_lf_rms_A": ic_lf_rms,
                "delta_V_LF_Vpp": delta_v_lf,
                "delta_V_SW_Vpp": delta_v_sw,
                "Ic_rms_A": ic_rms,
                "lifetime_hours": lifetime_hours,
                "lifetime_years": lifetime_hours / (365 * 24),
                "power_basis": "pout_total",
                "switching_ripple_model": "simplified",
                "switching_ripple_warning": (
                    "Two-phase interleaving HF ripple cancellation not yet modeled"
                ),
            }
        )

    def _lifetime(self, spec: DesignSpec, p_esr: float) -> float:
        """Estimate capacitor lifetime.

        Arrhenius model:
        L = L_rated * 2^((T_rated - T_core) / 10) * (I_rated / I_actual)^n

        where T_core = T_ambient + delta_T (heating from ripple current)
        """
        # Per-capacitor ripple current
        ic_per_cap = np.sqrt(p_esr / (spec.cap_esr / spec.cap_n_parallel))
        ic_each = ic_per_cap / spec.cap_n_parallel if spec.cap_n_parallel > 0 else ic_per_cap

        # Hot-spot temperature rise (approximate)
        # Rth ≈ 20 K/W for typical aluminum electrolytic
        rth_cap = 20.0  # K/W per capacitor (typical)
        t_rise = ic_each ** 2 * spec.cap_esr * rth_cap
        t_core = spec.cap_ambient + t_rise

        if t_core >= spec.cap_rated_temp:
            return 0.0

        # Lifetime: temperature factor + ripple current factor
        l_temp = spec.cap_rated_life * 2 ** ((spec.cap_rated_temp - t_core) / 10)
        if ic_each <= spec.cap_rated_ripple:
            l_total = l_temp
        else:
            # Ripple current derating (empirical exponent n ≈ 3-5)
            l_total = l_temp * (spec.cap_rated_ripple / max(ic_each, 0.001)) ** spec.cap_life_exponent

        return float(l_total)
