"""Diode loss model for PFC boost diode.

- Forward voltage loss (Vf(Tj) + Rd(Tj) model, datasheet-based)
- Reverse recovery loss (significant for Si, negligible for SiC Schottky)

Either a DiodeSpec from the database (spec.diode_part) or the legacy
DesignSpec fields (diode_vf / diode_rd) can drive the model.
"""

import numpy as np

from ..core.spec import DesignSpec, DiodeSpec
from ..core.operating_point import OperatingPoint
from .base import LossResult

# Si diode reverse recovery: Qrr increases with temperature
# (~+0.7 %/°C typical). SiC Schottky has no stored charge.
QRR_TEMP_COEFF = 0.007


class DiodeLoss:
    """Diode loss computation."""

    def compute(self, spec: DesignSpec, op: OperatingPoint,
                trace=None, diode: DiodeSpec | None = None,
                t_j: float = 80.0) -> LossResult:
        """Compute diode losses per phase.

        Args:
            spec: design specification (legacy Vf/Rd fallback)
            op: operating point
            trace: optional LineCycleTrace for line-cycle integration
            diode: DiodeSpec from database; uses spec.diode_vf if None
            t_j: junction temperature in °C (Vf temperature model)

        Returns:
            LossResult with forward and reverse-recovery sub-losses
        """
        if diode is not None:
            vf = diode.vf_at(t_j)
            rd = diode.rd_at(t_j)
            qrr = diode.qrr
            is_sic = diode.is_sic
            part = diode.part_number
            tech = diode.technology
        else:
            vf = spec.diode_vf
            rd = spec.diode_rd
            qrr = 0.0 if spec.diode_type.upper() == "SIC" else 2e-6
            is_sic = spec.diode_type.upper() == "SIC"
            part = "Mathcad-ref"
            tech = spec.diode_type

        if trace is not None:
            p_fwd = self._forward_loss_line_cycle(trace, vf, rd)
            loss_model = "line_cycle"
        else:
            p_fwd = self._forward_loss_legacy(op, vf, rd)
            loss_model = "legacy_single_point"

        p_rr = self._reverse_recovery(spec, qrr, is_sic, t_j)

        metadata = {
            "part_number": part,
            "technology": tech,
            "Vf_Tj": vf,
            "T_j": t_j,
            "type": spec.diode_type,
            "loss_model": loss_model,
        }
        if is_sic:
            metadata["SiC_note"] = "simplified model — Cj/Qc/Ec not accounted"

        return LossResult(
            component="Diode (per phase)",
            power_loss_W=p_fwd + p_rr,
            sub_losses={"forward": p_fwd, "reverse_recovery": p_rr},
            metadata=metadata,
        )

    # ── Legacy ──────────────────────────────────────────────────────

    def _forward_loss_legacy(self, op: OperatingPoint, vf: float,
                             rd: float) -> float:
        diode_duty = 1.0 - op.duty_t
        i_d_rms = op.rms(op.iin_t, diode_duty)
        i_d_avg = op.avg(op.iin_t, diode_duty)
        return float(vf * i_d_avg + rd * i_d_rms ** 2)

    # ── Line-cycle forward loss (includes ripple RMS) ───────────────

    def _forward_loss_line_cycle(self, trace, vf: float, rd: float) -> float:
        """Pdiode_forward = Vf * avg((1-D)*i_avg) + Rd * avg((1-D)*[i_avg^2 + delta_i^2/12])."""
        from ..core.line_cycle import average_over_half_cycle
        off_duty = 1.0 - trace.duty
        i_d_avg = average_over_half_cycle(off_duty * trace.i_avg_phase, trace.theta)
        i_d_rms_sq = average_over_half_cycle(off_duty * trace.i_rms_local_sq, trace.theta)
        return float(vf * i_d_avg + rd * i_d_rms_sq)

    # ── Reverse recovery ────────────────────────────────────────────

    def _reverse_recovery(self, spec: DesignSpec, qrr: float, is_sic: bool,
                          t_j: float) -> float:
        """Reverse recovery loss.

        SiC Schottky: traditional Qrr ≈ 0 (Cj/Ec not modeled yet).
        Si fast-recovery: Prr = Qrr(Tj) * Vout * fsw.
        """
        if is_sic:
            return 0.0
        qrr_t = qrr * (1.0 + QRR_TEMP_COEFF * (t_j - 25.0))
        return float(qrr_t * spec.vout * spec.fsw)
