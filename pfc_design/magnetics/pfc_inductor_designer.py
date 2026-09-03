"""PFC boost-inductor automatic design using powder-core DC-bias curves.

V7.3 intentionally starts with a transparent engineering model:
* Magnetics High Flux toroid / user-editable geometry;
* enamelled round copper wire;
* DC copper loss only by default (skin/proximity excluded by requirement);
* full-load L(I) droop from the manufacturer's permeability-vs-DC-bias fit;
* line-cycle averaged core loss from the manufacturer's core-loss-density fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .high_flux import HighFluxCoreGeometry, HighFluxMaterial

TopologyKind = Literal["ttpl", "vienna"]
RHO_CU_20 = 1.724e-8
CU_TEMP_COEFF = 0.00393


@dataclass(frozen=True)
class PFCInductorDesignRequest:
    topology: TopologyKind
    input_rms_v: float
    bus_voltage_v: float
    output_power_w: float
    switching_frequency_hz: float
    target_inductance_uh: float
    efficiency: float
    core: HighFluxCoreGeometry
    material: HighFluxMaterial
    n_cores: int = 1
    wire_copper_diameter_mm: float = 1.0
    enamel_build_mm: float = 0.05
    target_current_density_a_mm2: float = 5.0
    copper_temperature_c: float = 100.0
    max_fill_factor: float = 0.45
    winding_length_factor: float = 1.10
    curve_points: int = 161

    def validate(self) -> None:
        if self.topology not in ("ttpl", "vienna"):
            raise ValueError("topology must be 'ttpl' or 'vienna'")
        for name, value in (
            ("input_rms_v", self.input_rms_v),
            ("bus_voltage_v", self.bus_voltage_v),
            ("output_power_w", self.output_power_w),
            ("switching_frequency_hz", self.switching_frequency_hz),
            ("target_inductance_uh", self.target_inductance_uh),
            ("efficiency", self.efficiency),
            ("wire_copper_diameter_mm", self.wire_copper_diameter_mm),
            ("target_current_density_a_mm2", self.target_current_density_a_mm2),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.n_cores < 1:
            raise ValueError("n_cores must be >= 1")
        if not 0.0 < self.max_fill_factor < 1.0:
            raise ValueError("max_fill_factor must be between 0 and 1")


@dataclass(frozen=True)
class PFCInductorDesignResult:
    request: PFCInductorDesignRequest
    turns: int
    parallel_wires: int
    l_no_bias_uh: float
    l_full_load_peak_uh: float
    l_drop_percent: float
    phase_current_rms_a: float
    phase_current_peak_avg_a: float
    full_load_peak_with_ripple_a: float
    ripple_pp_at_current_peak_a: float
    total_current_rms_with_ripple_a: float
    h_peak_oe: float
    permeability_at_peak_percent: float
    b_dc_approx_peak_t: float
    b_ac_line_max_t: float
    core_loss_w: float
    copper_loss_w: float
    total_loss_w: float
    rdc_20_ohm: float
    rdc_hot_ohm: float
    copper_area_mm2: float
    current_density_a_mm2: float
    winding_length_m: float
    fill_factor: float
    window_ok: bool
    inductance_target_met: bool
    current_a: np.ndarray
    inductance_uh: np.ndarray
    line_angle_deg: np.ndarray
    line_b_ac_t: np.ndarray
    line_core_loss_w: np.ndarray
    line_ripple_pp_a: np.ndarray
    warnings: tuple[str, ...]



def _phase_rms_voltage(req: PFCInductorDesignRequest) -> float:
    return req.input_rms_v if req.topology == "ttpl" else req.input_rms_v / math.sqrt(3.0)


def _phase_current_rms(req: PFCInductorDesignRequest) -> float:
    if req.topology == "ttpl":
        return req.output_power_w / (req.efficiency * req.input_rms_v)
    return req.output_power_w / (math.sqrt(3.0) * req.efficiency * req.input_rms_v)


def _active_bus_level(req: PFCInductorDesignRequest) -> float:
    # TTPL boost switches between 0 and Vbus.  A Vienna phase switches between
    # the neutral point and one half of the split DC bus.
    return req.bus_voltage_v if req.topology == "ttpl" else 0.5 * req.bus_voltage_v


def _h_oe(turns: int, current_a: float, le_mm: float) -> float:
    le_cm = le_mm * 0.1
    return 0.4 * math.pi * turns * abs(current_a) / max(le_cm, 1.0e-12)


def _l_bias_uh(req: PFCInductorDesignRequest, turns: int, current_a: float) -> float:
    l0 = req.material.al_nh_per_t2 * req.n_cores * turns * turns / 1000.0
    pct = req.material.permeability_percent(_h_oe(turns, current_a, req.core.le_mm))
    return l0 * pct / 100.0


def _ripple_pp(req: PFCInductorDesignRequest, voltage_abs_v: float, l_uh: float) -> float:
    active_bus = _active_bus_level(req)
    if active_bus <= 0.0 or l_uh <= 0.0:
        return 0.0
    modulation = min(max(voltage_abs_v / active_bus, 0.0), 1.0)
    zero_or_on_duty = 1.0 - modulation
    return voltage_abs_v * zero_or_on_duty / (l_uh * 1.0e-6 * req.switching_frequency_hz)


def _resolve_peak(req: PFCInductorDesignRequest, turns: int) -> tuple[float, float, float]:
    vpk = math.sqrt(2.0) * _phase_rms_voltage(req)
    irms = _phase_current_rms(req)
    iavg_pk = math.sqrt(2.0) * irms
    i_peak = iavg_pk
    ripple = 0.0
    for _ in range(14):
        l = _l_bias_uh(req, turns, i_peak)
        ripple = _ripple_pp(req, vpk, l)
        i_new = iavg_pk + 0.5 * ripple
        if abs(i_new - i_peak) < 1.0e-7:
            i_peak = i_new
            break
        i_peak = i_new
    return i_peak, _l_bias_uh(req, turns, i_peak), ripple


def _wire_parallel_count(req: PFCInductorDesignRequest, irms_with_ripple: float) -> int:
    area_one = math.pi * (0.5 * req.wire_copper_diameter_mm) ** 2
    required = irms_with_ripple / req.target_current_density_a_mm2
    return max(1, int(math.ceil(required / max(area_one, 1.0e-12))))


def _fill_factor(req: PFCInductorDesignRequest, turns: int, n_parallel: int) -> float:
    od = req.wire_copper_diameter_mm + 2.0 * req.enamel_build_mm
    occupied = turns * n_parallel * math.pi * (0.5 * od) ** 2
    return occupied / max(req.core.window_area_mm2, 1.0e-12)


def _line_cycle(req: PFCInductorDesignRequest, turns: int, n_pts: int = 361):
    theta = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    vpk = math.sqrt(2.0) * _phase_rms_voltage(req)
    ipk = math.sqrt(2.0) * _phase_current_rms(req)
    active_bus = _active_bus_level(req)
    ae_total_m2 = req.core.ae_mm2 * req.n_cores * 1.0e-6
    ve_total_cm3 = req.core.ve_mm3 * req.n_cores * 1.0e-3

    ripple = np.zeros_like(theta)
    b_ac = np.zeros_like(theta)
    pcore = np.zeros_like(theta)
    i_sq = np.zeros_like(theta)
    for idx, angle in enumerate(theta):
        s = abs(math.sin(float(angle)))
        v = vpk * s
        iavg = ipk * s
        l = _l_bias_uh(req, turns, iavg)
        di = _ripple_pp(req, v, l)
        ripple[idx] = di
        # Flux excursion from the zero/on-state volt-seconds.  This is half of
        # the peak-to-peak switching flux swing used by Magnetics' loss curve.
        m = min(max(v / max(active_bus, 1e-12), 0.0), 1.0)
        d = 1.0 - m
        delta_b_pp = v * d / (
            turns * ae_total_m2 * req.switching_frequency_hz
        ) if turns > 0 else 0.0
        b_ac[idx] = 0.5 * delta_b_pp
        pv = req.material.core_loss_density_mw_cm3(b_ac[idx], req.switching_frequency_hz)
        pcore[idx] = pv * ve_total_cm3 / 1000.0
        i_sq[idx] = iavg * iavg + di * di / 12.0
    return theta, ripple, b_ac, pcore, math.sqrt(float(np.mean(i_sq)))


def design_pfc_inductor(req: PFCInductorDesignRequest) -> PFCInductorDesignResult:
    req.validate()

    # Search for an integer turn count that achieves the requested inductance at
    # the full-load current peak after DC-bias droop.  Powder-core L(N) is not
    # guaranteed monotonic at deep bias, so retain the best feasible candidate.
    best = None
    for turns in range(1, 401):
        i_peak, l_peak, ripple_peak = _resolve_peak(req, turns)
        theta, ripple, b_ac, pcore, irms_total = _line_cycle(req, turns, 181)
        n_parallel = _wire_parallel_count(req, irms_total)
        fill = _fill_factor(req, turns, n_parallel)
        h = _h_oe(turns, i_peak, req.core.le_mm)
        pct = req.material.permeability_percent(h)
        l0 = req.material.al_nh_per_t2 * req.n_cores * turns * turns / 1000.0
        ae_total_m2 = req.core.ae_mm2 * req.n_cores * 1.0e-6
        bdc = l_peak * 1.0e-6 * i_peak / max(turns * ae_total_m2, 1.0e-12)
        score = abs(l_peak - req.target_inductance_uh)
        feasible = fill <= req.max_fill_factor and bdc <= 0.80 * req.core.bs_t
        candidate = (feasible, l_peak >= req.target_inductance_uh, score, turns,
                     i_peak, l_peak, ripple_peak, irms_total, n_parallel, fill,
                     h, pct, l0, bdc, theta, ripple, b_ac, pcore)
        if feasible and l_peak >= req.target_inductance_uh:
            best = candidate
            break
        if best is None or (feasible and (not best[0] or score < best[2])):
            best = candidate

    if best is None:
        raise RuntimeError("No inductor candidate found")

    (feasible, target_met, _score, turns, i_peak, l_peak, ripple_peak,
     irms_total, n_parallel, fill, h, pct, l0, bdc, theta, ripple,
     b_ac, pcore) = best

    copper_d = req.wire_copper_diameter_mm
    area_one_mm2 = math.pi * (0.5 * copper_d) ** 2
    copper_area_mm2 = area_one_mm2 * n_parallel
    # Toroid turn length around the core cross-section.  Stacked cores increase
    # winding height; a small build factor accounts for insulation/layer growth.
    ht_total = req.core.ht_mm * req.n_cores
    mlt_mm = (2.0 * ht_total + (req.core.od_mm - req.core.id_mm)) * req.winding_length_factor
    length_m = turns * mlt_mm * 1.0e-3
    r20 = RHO_CU_20 * length_m / max(copper_area_mm2 * 1.0e-6, 1.0e-12)
    rhot = r20 * (1.0 + CU_TEMP_COEFF * (req.copper_temperature_c - 20.0))
    pcu = irms_total ** 2 * rhot
    pcore_mean = float(np.mean(pcore))

    i_curve = np.linspace(0.0, max(i_peak * 1.25, 1.0), req.curve_points)
    l_curve = np.array([_l_bias_uh(req, turns, float(i)) for i in i_curve])
    warnings: list[str] = []
    if not target_met:
        warnings.append("当前磁芯/导线约束下，满载峰值电感未达到目标值。")
    if not feasible or fill > req.max_fill_factor:
        warnings.append("绕组窗口填充超过设定上限，建议增加并联磁芯、减小匝数或调整导线。")
    if bdc > 0.80 * req.core.bs_t:
        warnings.append("估算 DC 磁通超过 80% Bsat，建议增加磁芯或降低磁导率。")
    if pct < 30.0:
        warnings.append("满载峰值磁导率低于初始值的 30%，处于较深 DC Bias 区域。")
    warnings.append("铜损默认仅计算漆包线 DC I²R；趋肤效应与邻近效应未计入。")

    return PFCInductorDesignResult(
        request=req,
        turns=turns,
        parallel_wires=n_parallel,
        l_no_bias_uh=l0,
        l_full_load_peak_uh=l_peak,
        l_drop_percent=(1.0 - l_peak / l0) * 100.0 if l0 > 0 else 0.0,
        phase_current_rms_a=_phase_current_rms(req),
        phase_current_peak_avg_a=math.sqrt(2.0) * _phase_current_rms(req),
        full_load_peak_with_ripple_a=i_peak,
        ripple_pp_at_current_peak_a=ripple_peak,
        total_current_rms_with_ripple_a=irms_total,
        h_peak_oe=h,
        permeability_at_peak_percent=pct,
        b_dc_approx_peak_t=bdc,
        b_ac_line_max_t=float(np.max(b_ac)),
        core_loss_w=pcore_mean,
        copper_loss_w=pcu,
        total_loss_w=pcore_mean + pcu,
        rdc_20_ohm=r20,
        rdc_hot_ohm=rhot,
        copper_area_mm2=copper_area_mm2,
        current_density_a_mm2=irms_total / max(copper_area_mm2, 1e-12),
        winding_length_m=length_m,
        fill_factor=fill,
        window_ok=fill <= req.max_fill_factor,
        inductance_target_met=target_met,
        current_a=i_curve,
        inductance_uh=l_curve,
        line_angle_deg=np.degrees(theta),
        line_b_ac_t=b_ac,
        line_core_loss_w=pcore,
        line_ripple_pp_a=ripple,
        warnings=tuple(warnings),
    )


__all__ = [
    "PFCInductorDesignRequest",
    "PFCInductorDesignResult",
    "design_pfc_inductor",
]
