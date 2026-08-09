"""Interactive LLC transformer synthesis from ferrite datasheet parameters.

This module is deliberately independent from the GUI.  It accepts the effective
magnetic/core/bobbin parameters that are normally copied from a ferrite data
sheet (Ae, Amin, le, Ve, AL, effective permeability, winding area and mean turn
length), synthesizes an integer turns pair, selects 0.1 mm Litz constructions in
configurable strand-count steps, and reuses the existing waveform-aware iGSE and
layered-Litz loss engines for the final loss calculation.

The default PQ35/35 preset is based on TDK B65881A/B65882B data.  The bundled
material Steinmetz coefficients remain engineering reference fits; production
release should use coefficients fitted to the exact material loss curves.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.spec import LLCDesignSpec
from ..core.tank import GainNotReachableError, TankDesign, design_tank
from ..models.system import default_work_points
from .core import CoreSpec
from .litz import LitzWire, dc_resistance_ohm, winding_layers
from .material import MaterialDatabase
from .transformer import MU0, TransformerDesign, TransformerLoss


@dataclass(frozen=True)
class FerriteCoreInput:
    preset_key: str = "TDK_PQ35_35_B65881A_N87"
    manufacturer: str = "TDK Electronics"
    part_number: str = "B65881A0000R087"
    shape: str = "PQ35/35"
    material_key: str = "TDK_N87_REF"
    material_grade: str = "N87"
    ae_mm2: float = 171.0
    amin_mm2: float = 161.0
    le_mm: float = 79.7
    ve_mm3: float = 13650.0
    sigma_l_over_a_per_mm: float = 0.465
    al_nh: float = 4500.0
    mu_e: float = 1670.0
    winding_area_mm2: float = 158.0
    mean_turn_length_mm: float = 76.0
    usable_winding_width_mm: float = 24.6
    ar_uohm: float = 16.5
    core_mass_g: float = 74.0
    thermal_resistance_k_per_w: float = 5.0
    datasheet_loss_ref_w: float = 8.75
    datasheet_loss_ref_frequency_hz: float = 100_000.0
    datasheet_loss_ref_b_t: float = 0.200
    datasheet_loss_ref_temperature_c: float = 100.0

    @property
    def effective_window_height_mm(self) -> float:
        return self.winding_area_mm2 / max(self.usable_winding_width_mm, 1e-9)

    def to_core_spec(self) -> CoreSpec:
        material = MaterialDatabase().get(self.material_key)
        return CoreSpec(
            part_number=self.part_number,
            shape=self.shape,
            family="PQ" if self.shape.upper().startswith("PQ") else "CUSTOM",
            manufacturer=self.manufacturer,
            standard=self.shape,
            material_key=self.material_key,
            purposes=("transformer",),
            ae_mm2=self.ae_mm2,
            amin_mm2=self.amin_mm2,
            aw_mm2=self.winding_area_mm2,
            ve_mm3=self.ve_mm3,
            le_mm=self.le_mm,
            window_width_mm=self.usable_winding_width_mm,
            window_height_mm=self.effective_window_height_mm,
            mlt_primary_mm=self.mean_turn_length_mm,
            mlt_secondary_mm=self.mean_turn_length_mm,
            center_leg_width_mm=2.0 * math.sqrt(max(self.amin_mm2, 1e-9) / math.pi),
            thermal_resistance_k_per_w=self.thermal_resistance_k_per_w,
            core_mass_g=self.core_mass_g,
            cost_usd=0.0,
            material_spec=material,
        )


@dataclass(frozen=True)
class TransformerSynthesisSettings:
    nominal_tank_gain: float = 1.0
    max_flux_density_t: float = 0.18
    strand_copper_diameter_mm: float = 0.10
    strand_outer_diameter_mm: float = 0.112
    strand_count_step: int = 50
    current_density_target_a_per_mm2: float = 6.0
    current_density_max_a_per_mm2: float = 8.0
    packing_factor: float = 0.55
    max_fill_factor: float = 0.60
    insulation_area_mm2: float = 28.0
    winding_layout: str = "P/2-S-P/2"
    max_secondary_turns_search: int = 40
    max_primary_turns_search: int = 500
    turn_ratio_tolerance: float = 0.04
    workpoint_scope: str = "all"  # all | normal | nominal


@dataclass(frozen=True)
class LitzSelection:
    strand_count: int
    strand_diameter_mm: float
    strand_outer_diameter_mm: float
    copper_area_mm2: float
    equivalent_outer_diameter_mm: float
    current_rms_a: float
    current_density_a_per_mm2: float
    parallel_sub_bundles: int
    strands_per_sub_bundle: int
    description: str


@dataclass(frozen=True)
class TransformerWorkpoint:
    vbus_v: float
    load_fraction: float
    switching_frequency_hz: float
    b_peak_t: float
    primary_rms_a: float
    secondary_rms_a: float
    core_loss_w: float
    primary_copper_w: float
    secondary_copper_w: float
    total_transformer_loss_w: float


@dataclass(frozen=True)
class TransformerSynthesisResult:
    core: FerriteCoreInput
    settings: TransformerSynthesisSettings
    spec: LLCDesignSpec
    tank: TankDesign
    primary_turns: int
    secondary_turns: int
    target_turns_ratio: float
    actual_turns_ratio: float
    turns_ratio_error_pct: float
    target_lm_uh: float
    ungapped_lm_uh: float
    target_al_nh: float
    estimated_gap_mm: float
    primary_litz: LitzSelection
    secondary_litz: LitzSelection
    primary_layers_per_half: int
    secondary_layers: int
    primary_turns_per_layer: int
    secondary_turns_per_layer: int
    fill_factor: float
    radial_build_mm: float
    primary_rdc_mohm: float
    secondary_rdc_mohm: float
    worst_b_peak_t: float
    nominal_loss: TransformerLoss
    workpoints: tuple[TransformerWorkpoint, ...]
    feasible: bool
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def total_nominal_loss_w(self) -> float:
        return self.nominal_loss.total_w


_PRESET_PATH = Path(__file__).parent.parent / "data" / "transformer_core_presets.json"


def load_transformer_core_presets(path: str | Path | None = None) -> dict[str, FerriteCoreInput]:
    source = Path(path) if path is not None else _PRESET_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    presets: dict[str, FerriteCoreInput] = {}
    for entry in data.get("presets", []):
        preset = FerriteCoreInput(**entry)
        presets[preset.preset_key] = preset
    return presets


def _round_up_multiple(value: int, step: int) -> int:
    step = max(int(step), 1)
    return int(math.ceil(max(int(value), 1) / step) * step)


def _select_discrete_litz(current_rms_a: float, settings: TransformerSynthesisSettings) -> tuple[LitzWire, LitzSelection]:
    d_m = settings.strand_copper_diameter_mm * 1e-3
    do_m = settings.strand_outer_diameter_mm * 1e-3
    area_strand_mm2 = math.pi * settings.strand_copper_diameter_mm**2 / 4.0
    raw = math.ceil(current_rms_a / max(settings.current_density_target_a_per_mm2 * area_strand_mm2, 1e-12))
    count = _round_up_multiple(raw, settings.strand_count_step)
    # A very large Litz is normally built from independently twisted sub-bundles.
    sub_bundles = max(1, int(math.ceil(count / 400.0)))
    wire = LitzWire(
        strand_count=count,
        strand_copper_diameter_m=d_m,
        strand_outer_diameter_m=do_m,
        packing_factor=settings.packing_factor,
        sub_bundle_count=sub_bundles,
    )
    density = current_rms_a / max(wire.copper_area_mm2, 1e-12)
    sel = LitzSelection(
        strand_count=count,
        strand_diameter_mm=settings.strand_copper_diameter_mm,
        strand_outer_diameter_mm=settings.strand_outer_diameter_mm,
        copper_area_mm2=wire.copper_area_mm2,
        equivalent_outer_diameter_mm=wire.equivalent_outer_diameter_mm,
        current_rms_a=current_rms_a,
        current_density_a_per_mm2=density,
        parallel_sub_bundles=sub_bundles,
        strands_per_sub_bundle=wire.strands_per_sub_bundle,
        description=wire.description,
    )
    return wire, sel


def _workpoint_requests(spec: LLCDesignSpec, scope: str) -> list[tuple[float, float]]:
    if scope == "nominal":
        return [(spec.vbus_nom_v, 1.0)]
    if scope == "normal":
        return [
            (spec.vbus_max_v, 1.0),
            (spec.vbus_nom_v, 1.0),
            (spec.vbus_min_normal_v, 1.0),
            (spec.vbus_nom_v, 0.50),
            (spec.vbus_nom_v, 0.25),
            (spec.vbus_nom_v, 0.10),
        ]
    return default_work_points(spec)


def _solve_workpoints(spec: LLCDesignSpec, tank: TankDesign, scope: str = "all") -> tuple[LLCOperatingPoint, ...]:
    points: list[LLCOperatingPoint] = []
    errors: list[str] = []
    for vbus, load in _workpoint_requests(spec, scope):
        try:
            points.append(solve_operating_point(spec, tank, vbus, load))
        except GainNotReachableError as exc:
            errors.append(f"{vbus:.0f} V/{load*100:.0f}%: {exc}")
    if errors:
        raise GainNotReachableError("; ".join(errors))
    return tuple(points)


def _candidate_turn_pairs(spec: LLCDesignSpec, settings: TransformerSynthesisSettings) -> Iterable[tuple[int, int, float]]:
    vsec = spec.vout_v + spec.rectifier_equivalent_drop_v
    target_ratio = (spec.bridge_gain * spec.vbus_nom_v * settings.nominal_tank_gain) / max(vsec, 1e-12)
    for ns in range(1, settings.max_secondary_turns_search + 1):
        center = target_ratio * ns
        for np_ in sorted({max(1, int(round(center)) + delta) for delta in (-1, 0, 1)}):
            if np_ > settings.max_primary_turns_search:
                continue
            error = abs(np_ / ns - target_ratio) / max(target_ratio, 1e-12)
            if error <= settings.turn_ratio_tolerance:
                yield np_, ns, target_ratio


def _estimate_gap_mm(core: FerriteCoreInput, primary_turns: int, target_lm_h: float) -> tuple[float, float, float]:
    al_target_h = target_lm_h / max(primary_turns**2, 1)
    ungapped_lm_h = core.al_nh * 1e-9 * primary_turns**2
    if al_target_h <= 0.0:
        return 0.0, ungapped_lm_h, 0.0
    # Reluctance model using the datasheet effective permeability.  This is a
    # first-order total-gap estimate; fringing and distributed gaps are not included.
    g_m = MU0 * core.ae_mm2 * 1e-6 / al_target_h - core.le_mm * 1e-3 / max(core.mu_e, 1.0)
    return max(g_m, 0.0) * 1e3, ungapped_lm_h, al_target_h * 1e9


def synthesize_transformer(
    base_spec: LLCDesignSpec,
    core_input: FerriteCoreInput,
    settings: TransformerSynthesisSettings | None = None,
) -> TransformerSynthesisResult:
    settings = settings or TransformerSynthesisSettings()
    core = core_input.to_core_spec()

    candidate_records: list[tuple[float, LLCDesignSpec, TankDesign, tuple[LLCOperatingPoint, ...], float]] = []
    solve_notes: list[str] = []
    for np_, ns, target_ratio in _candidate_turn_pairs(base_spec, settings):
        spec = base_spec.clone(primary_turns=np_, secondary_turns=ns)
        tank = design_tank(spec)
        try:
            ops = _solve_workpoints(spec, tank, settings.workpoint_scope)
        except GainNotReachableError as exc:
            if len(solve_notes) < 6:
                solve_notes.append(str(exc))
            continue
        b_values = [
            op.transformer_square_equivalent_v /
            (4.0 * np_ * core.amin_m2 * op.switching_frequency_hz)
            for op in ops
        ]
        worst_b = max(b_values)
        if worst_b > settings.max_flux_density_t * 1.20:
            # Keep the search compact: clearly under-turn candidates are discarded.
            continue
        ratio_err = abs(np_ / ns - target_ratio) / target_ratio
        flux_penalty = max(0.0, worst_b / settings.max_flux_density_t - 1.0)
        # Prefer the lowest practical turn count while strongly respecting flux and ratio.
        score = np_ + 0.35 * ns + 250.0 * ratio_err + 1000.0 * flux_penalty
        candidate_records.append((score, spec, tank, ops, worst_b))

    if not candidate_records:
        detail = solve_notes[0] if solve_notes else "no turn pair satisfied the search constraints"
        raise ValueError(f"automatic transformer turn search failed: {detail}")
    candidate_records.sort(key=lambda row: row[0])
    _, spec, tank, ops, worst_b = candidate_records[0]

    max_ip = max(op.resonant_current_rms_a for op in ops)
    max_is = max(op.secondary_current_rms_a for op in ops)
    p_wire, p_sel = _select_discrete_litz(max_ip, settings)
    s_wire, s_sel = _select_discrete_litz(max_is, settings)

    half_turns = int(math.ceil(spec.primary_turns / 2))
    p_layers, p_tpl = winding_layers(half_turns, p_wire, core.window_width_mm)
    s_layers, s_tpl = winding_layers(spec.secondary_turns, s_wire, core.window_width_mm)

    fill_area = (
        spec.primary_turns * p_wire.envelope_area_m2 * 1e6
        + spec.secondary_turns * s_wire.envelope_area_m2 * 1e6
        + settings.insulation_area_mm2
    )
    fill_factor = fill_area / max(core_input.winding_area_mm2, 1e-9)
    radial_build = (
        2.0 * p_layers * p_wire.equivalent_outer_diameter_mm
        + s_layers * s_wire.equivalent_outer_diameter_mm
        + 1.2
    )

    length_p = spec.primary_turns * core_input.mean_turn_length_mm * 1e-3
    length_s = spec.secondary_turns * core_input.mean_turn_length_mm * 1e-3
    rdc_p = dc_resistance_ohm(p_wire, length_p, spec.winding_temperature_c)
    rdc_s = dc_resistance_ohm(s_wire, length_s, spec.winding_temperature_c)

    gap_mm, ungapped_lm_h, target_al_nh = _estimate_gap_mm(
        core_input, spec.primary_turns, tank.lm_h
    )

    transformer = TransformerDesign(
        core=core,
        primary_turns=spec.primary_turns,
        secondary_turns=spec.secondary_turns,
        primary_wire=p_wire,
        secondary_wire=s_wire,
        primary_layers_per_half=p_layers,
        secondary_layers=s_layers,
        primary_turns_per_layer=p_tpl,
        secondary_turns_per_layer=s_tpl,
        fill_factor=fill_factor,
        radial_build_mm=radial_build,
        gap_total_mm=gap_mm,
        primary_rdc_ohm=rdc_p,
        secondary_rdc_ohm=rdc_s,
        worst_b_peak_t=worst_b,
        feasible=True,
        reasons=(),
    )

    nominal_op = min(
        ops,
        key=lambda op: abs(op.vbus_v - spec.vbus_nom_v) + 100.0 * abs(op.load_fraction - 1.0),
    )
    nominal_loss = transformer.loss(spec, nominal_op)
    workpoints: list[TransformerWorkpoint] = []
    for op in ops:
        loss = transformer.loss(spec, op)
        workpoints.append(TransformerWorkpoint(
            vbus_v=op.vbus_v,
            load_fraction=op.load_fraction,
            switching_frequency_hz=op.switching_frequency_hz,
            b_peak_t=loss.b_peak_min_area_t,
            primary_rms_a=op.resonant_current_rms_a,
            secondary_rms_a=op.secondary_current_rms_a,
            core_loss_w=loss.core_w,
            primary_copper_w=loss.primary_copper_w,
            secondary_copper_w=loss.secondary_copper_w,
            total_transformer_loss_w=loss.total_w,
        ))

    reasons: list[str] = []
    warnings: list[str] = []
    if worst_b > settings.max_flux_density_t:
        reasons.append(
            f"worst Bpk {worst_b:.3f} T exceeds design limit {settings.max_flux_density_t:.3f} T"
        )
    if p_sel.current_density_a_per_mm2 > settings.current_density_max_a_per_mm2:
        reasons.append("primary Litz current density exceeds maximum")
    if s_sel.current_density_a_per_mm2 > settings.current_density_max_a_per_mm2:
        reasons.append("secondary Litz current density exceeds maximum")
    if fill_factor > settings.max_fill_factor:
        reasons.append(
            f"window fill {fill_factor:.3f} exceeds {settings.max_fill_factor:.3f}"
        )
    if radial_build > core.window_height_mm:
        warnings.append(
            f"round-bundle radial-build screen {radial_build:.2f} mm exceeds equivalent window height "
            f"{core.window_height_mm:.2f} mm; consider parallel/flattened Litz bundles and verify the bobbin drawing"
        )
    if gap_mm <= 0.0 and tank.lm_h > ungapped_lm_h:
        reasons.append("target Lm is higher than the ungapped AL estimate")
    if gap_mm > 5.0:
        warnings.append(f"estimated total gap is large ({gap_mm:.2f} mm); verify fringing/leakage")
    warnings.extend(nominal_loss.material_warnings)
    warnings.append(
        "Core-loss calculation uses the bundled iGSE material reference fit.  The datasheet single-point Pv value is shown for cross-checking, not used as a full loss surface."
    )

    target_ratio = (spec.bridge_gain * spec.vbus_nom_v * settings.nominal_tank_gain) / (
        spec.vout_v + spec.rectifier_equivalent_drop_v
    )
    actual_ratio = spec.turns_ratio
    return TransformerSynthesisResult(
        core=core_input,
        settings=settings,
        spec=spec,
        tank=tank,
        primary_turns=spec.primary_turns,
        secondary_turns=spec.secondary_turns,
        target_turns_ratio=target_ratio,
        actual_turns_ratio=actual_ratio,
        turns_ratio_error_pct=100.0 * (actual_ratio - target_ratio) / target_ratio,
        target_lm_uh=tank.lm_h * 1e6,
        ungapped_lm_uh=ungapped_lm_h * 1e6,
        target_al_nh=target_al_nh,
        estimated_gap_mm=gap_mm,
        primary_litz=p_sel,
        secondary_litz=s_sel,
        primary_layers_per_half=p_layers,
        secondary_layers=s_layers,
        primary_turns_per_layer=p_tpl,
        secondary_turns_per_layer=s_tpl,
        fill_factor=fill_factor,
        radial_build_mm=radial_build,
        primary_rdc_mohm=rdc_p * 1e3,
        secondary_rdc_mohm=rdc_s * 1e3,
        worst_b_peak_t=worst_b,
        nominal_loss=nominal_loss,
        workpoints=tuple(workpoints),
        feasible=not reasons,
        warnings=tuple(dict.fromkeys(warnings)),
        reasons=tuple(reasons),
    )


def export_transformer_synthesis(result: TransformerSynthesisResult, output_directory: str | Path) -> dict[str, Path]:
    """Export a compact engineering calculation book for the transformer page."""
    from dataclasses import asdict
    import csv

    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "transformer_design.json"
    csv_path = out / "transformer_workpoints.csv"
    txt_path = out / "transformer_summary.txt"

    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        return value

    json_path.write_text(
        json.dumps(jsonable(asdict(result)), ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(result.workpoints[0]).keys()))
        writer.writeheader()
        for row in result.workpoints:
            writer.writerow(asdict(row))

    loss = result.nominal_loss
    lines = [
        "LLC Transformer Design",
        "=" * 72,
        f"Core: {result.core.shape} {result.core.material_grade} {result.core.part_number}",
        f"Np:Ns = {result.primary_turns}:{result.secondary_turns} (ratio={result.actual_turns_ratio:.6f})",
        f"Primary Litz: {result.primary_litz.description}",
        f"Secondary Litz: {result.secondary_litz.description}",
        f"Worst Bpk = {result.worst_b_peak_t*1e3:.3f} mT",
        f"Window fill = {result.fill_factor*100:.3f}%",
        f"Estimated total gap = {result.estimated_gap_mm:.4f} mm",
        f"Nominal core loss = {loss.core_w:.6f} W",
        f"Nominal primary copper = {loss.primary_copper_w:.6f} W",
        f"Nominal secondary copper = {loss.secondary_copper_w:.6f} W",
        f"Nominal transformer total = {loss.total_w:.6f} W",
        f"Estimated hotspot = {loss.estimated_hotspot_c:.2f} degC",
        f"Feasible = {result.feasible}",
    ]
    if result.reasons:
        lines += ["", "Reasons:"] + [f"- {x}" for x in result.reasons]
    if result.warnings:
        lines += ["", "Warnings:"] + [f"- {x}" for x in result.warnings]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": txt_path}
