"""Totem-pole bridgeless PFC loss model (CCM, synchronous rectification).

Compares against the standard bridge + boost-diode PFC: the bridge
rectifier and the boost diode are eliminated. The HF half-bridge leg
(two devices, typically GaN/SiC MOSFETs) does the switching plus
synchronous rectification, and the line-frequency leg (two devices,
typically low-cost Si MOSFETs) carries the AC current with plain
conduction loss.

The per-phase current/duty waveforms are identical to the boost PFC,
so the same inductor design and line-cycle trace are reused — the
comparison is apples-to-apples on identical magnetics.

Per phase, over one switching cycle at angle theta:
  active switch (boost role):  conducts during D, hard turn-on at
                               I_valley, hard turn-off at I_peak
  SR device:                   conducts during 1-D through its channel,
                               hard turn-off at I_valley (low current),
                               body diode conducts during deadtime at
                               both commutation edges
  line leg:                    two devices, each conducts one half line
                               cycle -> total I_rms^2 * Rds_slow
"""

import numpy as np

from ..core.spec import DesignSpec, MosfetSpec, MosfetDatabase
from ..core.operating_point import OperatingPoint, compute_mathcad_operating_point
from ..core.line_cycle import build_line_cycle_trace, average_over_half_cycle
from ..magnetics.core_database import CoreDatabase
from ..magnetics.core_entry import CoreSpec
from .inductor import InductorDesigner, InductorLoss, InductorDesign
from .capacitor import CapacitorBank
from .base import LossResult
from .system import THERMAL_MAX_ITER, THERMAL_TOL_C, compute_inductor_metrics

DEFAULT_HF_MOSFET = "GS66516T"        # GaN — the CCM totem-pole workhorse
DEFAULT_SLOW_MOSFET = "IPW65R032M8"   # Si SJ — line-frequency leg
DEFAULT_DEADTIME_S = 100e-9
DEFAULT_BODY_VF = 2.0                 # GaN reverse-conduction drop (V)


class TotemPoleAnalyzer:
    """System loss model for the totem-pole bridgeless PFC."""

    def __init__(self, db: CoreDatabase | None = None):
        self.db = db or CoreDatabase()
        self.mosfet_db = MosfetDatabase()
        self.inductor_designer = InductorDesigner(self.db)
        self.inductor_loss = InductorLoss(self.db)
        self.capacitor = CapacitorBank()

    def analyze(self, spec: DesignSpec,
                op: OperatingPoint | None = None,
                preferred_core: CoreSpec | None = None,
                mosfet: MosfetSpec | None = None,
                slow_mosfet: MosfetSpec | None = None,
                n_cores: int = 2,
                deadtime_s: float = DEFAULT_DEADTIME_S,
                vf_body: float = DEFAULT_BODY_VF,
                design: InductorDesign | None = None) -> dict:
        """Run the totem-pole analysis (fixed-point thermal if enabled).

        Args:
            spec: design specification (thermal_model toggles the loop)
            op: operating point (default: low-line worst case)
            preferred_core: core for the boost inductor
            mosfet: HF leg device (default GS66516T GaN)
            slow_mosfet: line-frequency leg device (default IPW65R032M8)
            n_cores: stacked cores for the inductor
            deadtime_s: SR deadtime in seconds
            vf_body: body-diode / reverse-conduction drop (V)
            design: pre-designed inductor to reuse

        Returns:
            dict with losses, breakdown, efficiency, thermal data
        """
        if op is None:
            op = compute_mathcad_operating_point(spec)
        if mosfet is None:
            mosfet = self.mosfet_db.get(DEFAULT_HF_MOSFET)
        if slow_mosfet is None:
            slow_mosfet = self.mosfet_db.get(DEFAULT_SLOW_MOSFET)

        if design is None:
            design = self.inductor_designer.design(spec, op, preferred_core, n_cores)

        L_eff = design.L_eff_at_ipeak_uh
        trace = build_line_cycle_trace(op, spec, L_eff)
        inductor_metrics = compute_inductor_metrics(design, trace, spec)

        n = spec.n_phases
        rth_hf = (spec.rth_mosfet_jc + spec.rth_mosfet_cs
                  + spec.rth_mosfet_sa)

        # Thermal fixed point (same params as the boost model)
        t_hf, t_slow, t_winding = 80.0, 80.0, spec.t_ambient + 40.0

        def _losses():
            ind = self.inductor_loss.compute(design, spec, op, trace=trace,
                                             t_winding=t_winding)
            active = self._hf_active_loss(spec, trace, mosfet, t_hf)
            sr = self._hf_sr_loss(spec, trace, mosfet, t_hf,
                                  deadtime_s, vf_body)
            slow = self._slow_leg_loss(spec, op, slow_mosfet, t_slow)
            cap = self.capacitor.compute(spec, op)
            return ind, active, sr, slow, cap

        if spec.thermal_model:
            for _ in range(THERMAL_MAX_ITER):
                ind, active, sr, slow, cap = _losses()
                t_hf_new = spec.t_ambient + rth_hf * (active.power_loss_W
                                                      + sr.power_loss_W)
                t_slow_new = spec.t_ambient + rth_hf * slow.power_loss_W
                t_wind_new = spec.t_ambient + spec.rth_inductor * ind.power_loss_W
                if (abs(t_hf_new - t_hf) < THERMAL_TOL_C
                        and abs(t_slow_new - t_slow) < THERMAL_TOL_C
                        and abs(t_wind_new - t_winding) < THERMAL_TOL_C):
                    break
                t_hf, t_slow, t_winding = t_hf_new, t_slow_new, t_wind_new
            ind, active, sr, slow, cap = _losses()
        else:
            ind, active, sr, slow, cap = _losses()

        total_loss = (n * (ind.power_loss_W + active.power_loss_W
                           + sr.power_loss_W + slow.power_loss_W)
                      + cap.power_loss_W)
        efficiency = spec.pout_total / (spec.pout_total + total_loss)

        breakdown = {}
        for label, val in ind.sub_losses.items():
            breakdown[f"Inductor_{label}_total"] = n * val
        for label, val in active.sub_losses.items():
            breakdown[f"HF_active_{label}_total"] = n * val
        for label, val in sr.sub_losses.items():
            breakdown[f"HF_SR_{label}_total"] = n * val
        for label, val in slow.sub_losses.items():
            breakdown[f"Slow_leg_{label}_total"] = n * val
        for label, val in cap.sub_losses.items():
            breakdown[f"Capacitor_{label}_total"] = val

        thermal = {
            "enabled": spec.thermal_model,
            "t_ambient": spec.t_ambient,
            "t_j_hf": t_hf,
            "t_j_slow": t_slow,
            "t_winding": t_winding,
            "rth_hf_ja": rth_hf,
            "rth_inductor": spec.rth_inductor,
        }

        return {
            "spec": spec,
            "op": op,
            "mosfet": mosfet,
            "slow_mosfet": slow_mosfet,
            "inductor_design": design,
            "inductor_metrics": inductor_metrics,
            "losses": {
                "inductor": ind,
                "active": active,
                "sr": sr,
                "slow_leg": slow,
                "capacitor": cap,
            },
            "trace": trace,
            "total_loss": total_loss,
            "efficiency": efficiency,
            "thermal": thermal,
            "breakdown": breakdown,
            "params": {"deadtime_s": deadtime_s, "vf_body": vf_body},
        }

    # ── HF leg: active switch (boost role) ──────────────────────────

    def _hf_active_loss(self, spec: DesignSpec, trace,
                        mosfet: MosfetSpec, t_j: float) -> LossResult:
        rds = mosfet.rds_on_25c * (1 + mosfet.rds_alpha * (t_j - 25))
        p_cond = float(rds * average_over_half_cycle(
            trace.duty * trace.i_rms_local_sq, trace.theta))
        e_on = 0.5 * spec.vout * trace.i_valley * mosfet.tr
        e_off = 0.5 * spec.vout * trace.i_peak * mosfet.tf
        p_sw = float(spec.fsw * average_over_half_cycle(
            e_on + e_off, trace.theta))
        p_coss = 0.5 * mosfet.coss_er * spec.vout ** 2 * spec.fsw
        p_gate = mosfet.qg * mosfet.vgs * spec.fsw
        return LossResult(
            component="HF active switch (per phase)",
            power_loss_W=p_cond + p_sw + p_coss + p_gate,
            sub_losses={"conduction": p_cond, "switching": p_sw,
                        "Coss": p_coss, "gate_drive": p_gate},
            metadata={"part_number": mosfet.part_number, "T_j": t_j,
                      "role": "active_boost"},
        )

    # ── HF leg: synchronous rectifier ───────────────────────────────

    def _hf_sr_loss(self, spec: DesignSpec, trace, mosfet: MosfetSpec,
                    t_j: float, deadtime_s: float,
                    vf_body: float) -> LossResult:
        rds = mosfet.rds_on_25c * (1 + mosfet.rds_alpha * (t_j - 25))
        p_cond = float(rds * average_over_half_cycle(
            (1.0 - trace.duty) * trace.i_rms_local_sq, trace.theta))
        # SR hard turn-off at the valley current (low), then body diode
        # takes over during deadtime at both commutation edges.
        e_off = 0.5 * spec.vout * trace.i_valley * mosfet.tf
        p_sw_off = float(spec.fsw * average_over_half_cycle(e_off, trace.theta))
        p_dead = float(spec.fsw * vf_body * deadtime_s
                       * average_over_half_cycle(trace.i_valley + trace.i_peak,
                                                 trace.theta))
        p_coss = 0.5 * mosfet.coss_er * spec.vout ** 2 * spec.fsw
        p_gate = mosfet.qg * mosfet.vgs * spec.fsw
        return LossResult(
            component="HF SR switch (per phase)",
            power_loss_W=p_cond + p_sw_off + p_dead + p_coss + p_gate,
            sub_losses={"conduction": p_cond, "switch_off": p_sw_off,
                        "deadtime_body": p_dead, "Coss": p_coss,
                        "gate_drive": p_gate},
            metadata={"part_number": mosfet.part_number, "T_j": t_j,
                      "role": "synchronous_rectifier"},
        )

    # ── Line-frequency leg (two devices) ────────────────────────────

    def _slow_leg_loss(self, spec: DesignSpec, op: OperatingPoint,
                       slow_mosfet: MosfetSpec, t_j: float) -> LossResult:
        """Both line-frequency devices: total I_rms^2 * Rds.

        Each device conducts one half line cycle carrying the full phase
        current; the two devices together dissipate I_rms^2 * Rds on
        average. They switch once per line cycle at ~zero current.
        """
        rds = slow_mosfet.rds_on_25c * (1 + slow_mosfet.rds_alpha * (t_j - 25))
        i_rms = op.iin_rms
        p_cond = i_rms ** 2 * rds
        p_gate = 2.0 * slow_mosfet.qg * slow_mosfet.vgs * spec.f_line
        return LossResult(
            component="Line-frequency leg (per phase)",
            power_loss_W=p_cond + p_gate,
            sub_losses={"conduction": p_cond, "gate_drive": p_gate},
            metadata={"part_number": slow_mosfet.part_number, "T_j": t_j,
                      "role": "line_leg"},
        )
