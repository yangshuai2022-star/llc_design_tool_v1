"""MOSFET loss model for PFC boost switch.

Supports both fixed parameters (DesignSpec) and
database-driven MosfetSpec for optimization sweeps.
"""

import numpy as np

from ..core.spec import DesignSpec, MosfetSpec
from ..core.operating_point import OperatingPoint
from .base import LossResult


class MosfetLoss:
    """MOSFET total loss computation."""

    def compute(self, spec: DesignSpec, op: OperatingPoint,
                mosfet: MosfetSpec | None = None,
                t_j: float = 80.0,
                trace=None) -> LossResult:
        """Compute all MOSFET losses per switch.

        Args:
            spec: design specification (or fallback if mosfet is None)
            op: operating point
            mosfet: MosfetSpec from database; uses spec defaults if None
            t_j: junction temperature in °C
            trace: optional LineCycleTrace for line-cycle-integrated loss
        """
        # Use MosfetSpec if provided, otherwise legacy spec fields
        if mosfet is not None:
            vds_max = mosfet.vds_max
            rds_25 = mosfet.rds_on_25c
            rds_alpha = mosfet.rds_alpha
            qg = mosfet.qg
            coss_er = mosfet.coss_er
            tr = mosfet.tr
            tf = mosfet.tf
            vgs = mosfet.vgs
            part = mosfet.part_number
            tech = mosfet.technology
        else:
            vds_max = spec.mosfet_vds_max
            rds_25 = spec.mosfet_rds_on
            rds_alpha = spec.mosfet_rds_alpha
            qg = spec.mosfet_qg
            coss_er = spec.mosfet_coss_er
            tr = spec.mosfet_tr
            tf = spec.mosfet_tf
            vgs = spec.mosfet_vgs
            part = "Mathcad-ref"
            tech = "SiC"

        # Temperature-adjusted Rds_on
        rds_tj = rds_25 * (1 + rds_alpha * (t_j - 25))

        # Datasheet Eon/Eoff model (linear V/I scaling) if reference data
        # exists; otherwise fall back to the tr/tf linear-ramp estimate.
        has_e_curves = (mosfet is not None and mosfet.eon_ref_uj > 0
                        and mosfet.eoff_ref_uj > 0)

        if trace is not None:
            p_cond = self._conduction_loss_line_cycle(trace, rds_tj)
            e_curves = None
            if has_e_curves:
                e_curves = (mosfet.eon_ref_uj, mosfet.eoff_ref_uj,
                            mosfet.e_ref_v, mosfet.e_ref_i)
            p_sw = self._switching_loss_line_cycle(spec, trace, tr, tf,
                                                   e_curves=e_curves)
            loss_model = "line_cycle"
            sw_model = ("datasheet_Eon_Eoff" if e_curves else "linear_tr_tf")
            extra_meta = {
                "switching_model": sw_model,
                "coss_accounting_note": (
                    "Simplified — if datasheet Eoff includes Coss energy, "
                    "future versions must avoid double-counting"
                ),
                "risk_flag": "Coss/Eoff_overlap_risk",
            }
        else:
            p_cond = self._conduction_loss_legacy(op, rds_tj)
            p_sw = self._switching_loss_legacy(spec, op, tr, tf)
            loss_model = "legacy_single_point"
            extra_meta = {}

        p_coss = self._coss_loss(spec, coss_er)
        p_gate = self._gate_loss(spec, qg, vgs)

        total = p_cond + p_sw + p_coss + p_gate

        return LossResult(
            component="MOSFET (per phase)",
            power_loss_W=total,
            sub_losses={
                "conduction": p_cond,
                "switching": p_sw,
                "Coss": p_coss,
                "gate_drive": p_gate,
            },
            metadata={
                "part_number": part,
                "technology": tech,
                "vds_max": vds_max,
                "Rds_on_Tj": rds_tj,
                "T_j": t_j,
                "loss_model": loss_model,
                **extra_meta,
            }
        )

    # ── Legacy methods ──────────────────────────────────────────────

    def _conduction_loss_legacy(self, op: OperatingPoint, rds: float) -> float:
        i_ds_rms = op.rms(op.iin_t, op.duty_t)
        return float(i_ds_rms ** 2 * rds)

    def _switching_loss_legacy(self, spec: DesignSpec, op: OperatingPoint,
                               tr: float, tf: float) -> float:
        i_avg_on = 2 / np.pi * op.iin_peak
        e_on = 0.5 * spec.vout * i_avg_on * tr
        e_off = 0.5 * spec.vout * i_avg_on * tf
        return float(spec.fsw * (e_on + e_off))

    # ── Line-cycle conduction loss (with ripple RMS) ────────────────

    def _conduction_loss_line_cycle(self, trace, rds: float) -> float:
        """Pcond = Rds * mean( D(theta) * [i_avg^2 + delta_i^2/12] ).

        Degenerates to Rds * mean(D * i_avg^2) when delta_i = 0.
        """
        from ..core.line_cycle import average_over_half_cycle
        integrand = trace.duty * trace.i_rms_local_sq
        return float(rds * average_over_half_cycle(integrand, trace.theta))

    # ── Line-cycle switching loss (valley/peak currents) ──────────

    def _switching_loss_line_cycle(self, spec: DesignSpec, trace,
                                   tr: float, tf: float,
                                   e_curves: tuple | None = None) -> float:
        """Psw = fsw * mean( Eon(theta) + Eoff(theta) ).

        Boost CCM:  turn-on  current ≈ I_valley (valley of ripple)
                    turn-off current ≈ I_peak   (peak of ripple)

        With datasheet reference energies (e_curves =
        (Eon_ref_uJ, Eoff_ref_uJ, V_ref, I_ref)):
            Eon(V,I)  = Eon_ref  * (V/V_ref) * (I/I_ref)
            Eoff(V,I) = Eoff_ref * (V/V_ref) * (I/I_ref)
        linear scaling in both voltage and current — the common
        engineering approximation of the datasheet curves.

        Without reference data (fallback, legacy):
            Eon  = 0.5 * Vout * I_valley * tr
            Eoff = 0.5 * Vout * I_peak   * tf
        """
        from ..core.line_cycle import average_over_half_cycle
        vout = spec.vout
        if e_curves is not None:
            eon_ref_uj, eoff_ref_uj, v_ref, i_ref = e_curves
            k_on = eon_ref_uj * 1e-6 / (v_ref * i_ref)
            k_off = eoff_ref_uj * 1e-6 / (v_ref * i_ref)
            e_on = k_on * vout * trace.i_valley
            e_off = k_off * vout * trace.i_peak
        else:
            e_on = 0.5 * vout * trace.i_valley * tr
            e_off = 0.5 * vout * trace.i_peak * tf
        e_mean = average_over_half_cycle(e_on + e_off, trace.theta)
        return float(spec.fsw * e_mean)

    # ── Coss / gate drive (unchanged by line-cycle) ─────────────────

    def _coss_loss(self, spec: DesignSpec, coss_er: float) -> float:
        e_oss = 0.5 * coss_er * spec.vout ** 2
        return float(e_oss * spec.fsw)

    def _gate_loss(self, spec: DesignSpec, qg: float, vgs: float) -> float:
        return float(qg * vgs * spec.fsw)
