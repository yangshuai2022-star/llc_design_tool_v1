"""System-level LLC analysis, hardware synthesis and feasibility aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.spec import LLCDesignSpec
from ..core.tank import GainNotReachableError, TankDesign, design_tank
from ..magnetics.core import CoreDatabase
from ..magnetics.resonant_inductor import (ResonantInductorDesign,
                                           ResonantInductorLoss,
                                           design_resonant_inductor)
from ..magnetics.transformer import (TransformerDesign, TransformerLoss,
                                     design_transformer)
from .capacitors import (BusCapacitorResult, ResonantCapacitorResult,
                         bus_capacitor_result, output_capacitor_result,
                         resonant_capacitor_result)
from .devices import DeviceDatabase
from .primary_bridge import PrimaryBridgeLoss, primary_bridge_loss
from .synchronous_rectifier import (SynchronousRectifierLoss,
                                     synchronous_rectifier_loss)
from ..core.waveform import OutputRippleResult


@dataclass(frozen=True)
class OperatingPointLoss:
    operating_point: LLCOperatingPoint
    primary: PrimaryBridgeLoss
    synchronous_rectifier: SynchronousRectifierLoss
    transformer: TransformerLoss
    resonant_inductor: ResonantInductorLoss
    resonant_capacitor: ResonantCapacitorResult
    output_capacitor: OutputRippleResult
    auxiliary_w: float
    total_loss_w: float
    efficiency: float

    @property
    def label(self) -> str:
        return f"{self.operating_point.vbus_v:.0f}V_{self.operating_point.load_fraction*100:.0f}pct"

    def breakdown(self) -> dict[str, float]:
        """Non-overlapping system loss buckets used for totals and plots."""
        return {
            "primary_conduction_w": self.primary.conduction_w,
            "primary_turnoff_w": self.primary.turnoff_w,
            "primary_gate_w": self.primary.gate_drive_w,
            "primary_coss_w": self.primary.residual_coss_w,
            "primary_deadtime_w": self.primary.deadtime_diode_w,
            "sr_conduction_w": self.synchronous_rectifier.conduction_w,
            "sr_deadtime_w": self.synchronous_rectifier.deadtime_diode_w,
            "sr_turnoff_w": self.synchronous_rectifier.turnoff_w,
            "sr_coss_w": self.synchronous_rectifier.coss_w,
            "sr_gate_w": self.synchronous_rectifier.gate_drive_w,
            "transformer_core_w": self.transformer.core_w,
            "transformer_primary_copper_w": self.transformer.primary_copper_w,
            "transformer_secondary_copper_w": self.transformer.secondary_copper_w,
            "resonant_inductor_core_w": self.resonant_inductor.core_w,
            "resonant_inductor_copper_w": self.resonant_inductor.copper_w,
            "resonant_capacitor_w": self.resonant_capacitor.loss_w,
            "output_capacitor_w": self.output_capacitor.capacitor_loss_w,
            "auxiliary_w": self.auxiliary_w,
        }

    def detailed_magnetics_breakdown(self) -> dict[str, float]:
        """Detailed non-overlapping magnetic-component loss mechanisms."""
        return {
            "transformer_core_w": self.transformer.core_w,
            "transformer_primary_dc_w": self.transformer.primary_dc_w,
            "transformer_primary_skin_w": self.transformer.primary_skin_w,
            "transformer_primary_proximity_w": self.transformer.primary_proximity_w,
            "transformer_primary_bundle_w": self.transformer.primary_bundle_w,
            "transformer_primary_termination_w": self.transformer.primary_termination_w,
            "transformer_secondary_dc_w": self.transformer.secondary_dc_w,
            "transformer_secondary_skin_w": self.transformer.secondary_skin_w,
            "transformer_secondary_proximity_w": self.transformer.secondary_proximity_w,
            "transformer_secondary_bundle_w": self.transformer.secondary_bundle_w,
            "transformer_secondary_termination_w": self.transformer.secondary_termination_w,
            "resonant_inductor_core_w": self.resonant_inductor.core_w,
            "resonant_inductor_dc_w": self.resonant_inductor.dc_copper_w,
            "resonant_inductor_skin_w": self.resonant_inductor.skin_effect_w,
            "resonant_inductor_proximity_w": self.resonant_inductor.external_proximity_w,
            "resonant_inductor_gap_fringing_w": self.resonant_inductor.gap_fringing_w,
            "resonant_inductor_bundle_w": self.resonant_inductor.bundle_circulating_w,
            "resonant_inductor_termination_w": self.resonant_inductor.termination_w,
        }



@dataclass(frozen=True)
class SystemAnalysis:
    spec: LLCDesignSpec
    tank: TankDesign
    transformer: TransformerDesign
    resonant_inductor: ResonantInductorDesign
    bus_capacitor: BusCapacitorResult
    operating_points: tuple[OperatingPointLoss, ...]
    feasible: bool
    feasibility_reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def nominal(self) -> OperatingPointLoss:
        return min(self.operating_points,
                   key=lambda x: abs(x.operating_point.vbus_v - self.spec.vbus_nom_v)
                   + 100.0 * abs(x.operating_point.load_fraction - 1.0))

    @property
    def worst_loss(self) -> OperatingPointLoss:
        return max(self.operating_points, key=lambda x: x.total_loss_w)

    @property
    def minimum_efficiency(self) -> OperatingPointLoss:
        return min(self.operating_points, key=lambda x: x.efficiency)


def default_work_points(spec: LLCDesignSpec) -> list[tuple[float, float]]:
    return [
        (spec.vbus_max_v, 1.00),
        (spec.vbus_nom_v, 1.00),
        (spec.vbus_min_normal_v, 1.00),
        (spec.vbus_hold_end_v, 1.00),
        (spec.vbus_nom_v, 0.50),
        (spec.vbus_nom_v, 0.25),
        (spec.vbus_nom_v, 0.10),
    ]


class LLCSystemAnalyzer:
    def __init__(self, core_database: CoreDatabase | None = None,
                 device_database: DeviceDatabase | None = None):
        self.core_db = core_database or CoreDatabase()
        self.device_db = device_database or DeviceDatabase()

    def analyze(self, spec: LLCDesignSpec,
                work_points: Iterable[tuple[float, float]] | None = None,
                preferred_transformer_core: str | None = None,
                preferred_inductor_core: str | None = None) -> SystemAnalysis:
        spec.validate()
        tank = design_tank(spec)
        point_requests = list(work_points or default_work_points(spec))
        solved: list[LLCOperatingPoint] = []
        solve_errors: list[str] = []
        for vbus, load in point_requests:
            try:
                solved.append(solve_operating_point(spec, tank, vbus, load))
            except GainNotReachableError as exc:
                solve_errors.append(f"{vbus:.0f} V / {load*100:.0f}% load: {exc}")
        if not solved:
            raise GainNotReachableError("no requested operating point can be solved")

        transformer = design_transformer(
            spec, tank, solved, self.core_db, preferred_transformer_core)
        resonant_inductor = design_resonant_inductor(
            spec, tank, solved, self.core_db, preferred_inductor_core)
        primary_device = self.device_db.get_primary(spec.primary_device)
        sr_device = self.device_db.get_sr(spec.sr_device)

        point_results: list[OperatingPointLoss] = []
        for op in solved:
            p_primary = primary_bridge_loss(spec, tank, op, primary_device)
            p_sr = synchronous_rectifier_loss(spec, op, sr_device)
            p_transformer = transformer.loss(spec, op)
            p_inductor = resonant_inductor.loss(spec, op)
            p_cr = resonant_capacitor_result(spec, tank, op)
            p_co = output_capacitor_result(spec, op)
            p_aux = 5.0 + 0.004 * op.pout_w
            total = (p_primary.total_w + p_sr.total_w + p_transformer.total_w
                     + p_inductor.total_w + p_cr.loss_w
                     + p_co.capacitor_loss_w + p_aux)
            efficiency = op.pout_w / (op.pout_w + total) if op.pout_w > 0 else 0.0
            point_results.append(OperatingPointLoss(
                operating_point=op, primary=p_primary,
                synchronous_rectifier=p_sr, transformer=p_transformer,
                resonant_inductor=p_inductor, resonant_capacitor=p_cr,
                output_capacitor=p_co, auxiliary_w=p_aux,
                total_loss_w=total, efficiency=efficiency))

        bus = bus_capacitor_result(spec)
        reasons: list[str] = list(solve_errors)
        warnings: list[str] = []
        full_load = [p for p in point_results if p.operating_point.load_fraction >= 0.999]
        hard_zvs_points = [p for p in point_results if p.operating_point.load_fraction >= 0.25]

        if not transformer.feasible:
            reasons.extend(f"transformer: {reason}" for reason in transformer.reasons)
        if not resonant_inductor.feasible:
            reasons.extend(f"resonant inductor: {reason}" for reason in resonant_inductor.reasons)
        for point in full_load:
            op = point.operating_point
            if op.input_phase_deg < spec.minimum_inductive_angle_deg:
                reasons.append(f"{point.label}: inductive angle is only {op.input_phase_deg:.2f}°")
        for point in hard_zvs_points:
            margin = min(point.primary.zvs_charge_margin,
                         point.primary.zvs_energy_margin)
            if margin < 1.0:
                reasons.append(f"{point.label}: primary ZVS margin {margin:.2f} < 1.0")
            elif margin < spec.primary_zvs_margin_required:
                warnings.append(f"{point.label}: primary ZVS margin {margin:.2f} is below preferred margin")

        max_primary_v = max(p.primary.voltage_stress_v for p in point_results)
        if max_primary_v > primary_device.vds_max_v * spec.primary_voltage_derating:
            reasons.append("primary MOSFET voltage derating failed")
        max_sr_v = max(p.synchronous_rectifier.voltage_stress_v for p in point_results)
        if max_sr_v > sr_device.vds_max_v * spec.sr_voltage_derating:
            reasons.append("SR MOSFET voltage derating failed")
        if max(p.primary.device_peak_current_a for p in point_results) > 0.8 * primary_device.id_cont_a:
            warnings.append("primary device peak-current screen exceeds 80% of reference rating")
        if max(p.synchronous_rectifier.device_peak_current_a for p in point_results) > 0.8 * sr_device.id_cont_a:
            warnings.append("SR device peak-current screen exceeds 80% of reference rating")

        for point in full_load:
            if point.resonant_capacitor.voltage_margin < 1.0:
                reasons.append(f"{point.label}: resonant capacitor voltage rating failed")
            if point.resonant_capacitor.current_margin < 1.0:
                reasons.append(f"{point.label}: resonant capacitor current rating failed")
            if point.output_capacitor.total_ripple_vpp > spec.output_ripple_limit_vpp:
                reasons.append(f"{point.label}: output ripple exceeds limit")

        if bus.hold_time_to_limit_s + 1e-12 < spec.requested_hold_time_s:
            reasons.append("installed bus capacitance does not meet requested hold-up time")
        if bus.predicted_end_voltage_v < spec.vbus_hold_end_v:
            reasons.append("bus voltage falls below LLC hold-up endpoint")

        if point_results[-1].operating_point.load_fraction <= 0.10:
            light = point_results[-1]
            margin = min(light.primary.zvs_charge_margin,
                         light.primary.zvs_energy_margin)
            if margin < spec.primary_zvs_margin_required:
                warnings.append("10% load point may require burst/skip control for robust ZVS")

        for point in point_results:
            if point.transformer.estimated_hotspot_c > spec.magnetic_hotspot_limit_c:
                reasons.append(f"{point.label}: transformer hotspot {point.transformer.estimated_hotspot_c:.1f} °C exceeds limit")
            if point.resonant_inductor.estimated_hotspot_c > spec.magnetic_hotspot_limit_c:
                reasons.append(f"{point.label}: resonant-inductor hotspot {point.resonant_inductor.estimated_hotspot_c:.1f} °C exceeds limit")
            warnings.extend(f"{point.label} transformer material: {w}" for w in point.transformer.material_warnings)
            warnings.extend(f"{point.label} inductor material: {w}" for w in point.resonant_inductor.material_warnings)

        warnings.append("Core geometry and ferrite coefficients are engineering reference records; verify the exact ordering code and fit manufacturer curves before release.")
        warnings.append("Magnetic core loss uses temperature-interpolated iGSE on reconstructed B(t); minor-loop and DC-bias corrections require measured material data.")
        warnings.append("Litz loss uses exact round-strand skin effect, harmonic layer-MMF proximity loss, transposition and termination penalties; 2D/3D FEA remains the final verification method.")
        warnings.append("Resonant-inductor gap-fringing winding loss is enabled through a distance-based calibrated field term; transformer leakage contribution to Lr remains separate.")

        return SystemAnalysis(
            spec=spec, tank=tank, transformer=transformer,
            resonant_inductor=resonant_inductor, bus_capacitor=bus,
            operating_points=tuple(point_results), feasible=not reasons,
            feasibility_reasons=tuple(reasons), warnings=tuple(warnings))
