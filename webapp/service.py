"""Pure-Python service layer used by the FastAPI web application.

The web layer deliberately calls the same engineering kernels as the desktop GUI.
No calculation is duplicated in JavaScript; the browser only submits parameters and
renders the returned engineering results.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from llc_design.core.spec import LLCDesignSpec, PrimaryTopology
from llc_design.core.tank import equivalent_ac_load_ohm, gain_vector, target_gain
from llc_design.models.devices import DeviceDatabase
from llc_design.models.system import LLCSystemAnalyzer, SystemAnalysis

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PACKAGE_ROOT / "llc_design" / "data"

# Public web API intentionally exposes a bounded subset of LLCDesignSpec.  This
# avoids allowing an anonymous request to select pathological solver sizes while
# still covering the normal electrical/tank/device design workflow.
_NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "vbus_nom_v": (50.0, 1500.0),
    "vbus_min_normal_v": (50.0, 1500.0),
    "vbus_max_v": (50.0, 1500.0),
    "vbus_hold_end_v": (20.0, 1500.0),
    "vout_v": (1.0, 1000.0),
    "pout_w": (1.0, 200_000.0),
    "efficiency_assumption": (0.5, 0.9999),
    "resonant_frequency_hz": (5_000.0, 2_000_000.0),
    "minimum_frequency_hz": (1_000.0, 2_000_000.0),
    "maximum_frequency_hz": (2_000.0, 3_000_000.0),
    "ln_ratio": (1.01, 50.0),
    "q_full_load": (0.01, 5.0),
    "primary_turns": (1.0, 500.0),
    "secondary_turns": (1.0, 500.0),
    "rectifier_equivalent_drop_v": (0.0, 20.0),
    "bus_capacitance_f": (1e-9, 0.2),
    "requested_hold_time_s": (0.0, 1.0),
    "output_capacitance_f": (1e-9, 0.2),
    "output_cap_esr_ohm": (0.0, 10.0),
    "primary_deadtime_s": (0.0, 20e-6),
    "primary_zvs_margin_required": (0.1, 20.0),
    "primary_parallel_devices": (1.0, 32.0),
    "sr_parallel_devices_per_position": (1.0, 32.0),
    "ambient_temperature_c": (-40.0, 180.0),
    "primary_junction_temperature_c": (-40.0, 250.0),
    "sr_junction_temperature_c": (-40.0, 250.0),
    "winding_temperature_c": (-40.0, 250.0),
}

_STRING_FIELDS = {"primary_device", "sr_device"}
_ENUM_FIELDS = {"primary_topology"}
_ALLOWED_FIELDS = set(_NUMERIC_LIMITS) | _STRING_FIELDS | _ENUM_FIELDS
_INTEGER_FIELDS = {"primary_turns", "secondary_turns", "primary_parallel_devices", "sr_parallel_devices_per_position"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def default_payload() -> dict[str, Any]:
    spec = LLCDesignSpec()
    fields = {name: _jsonable(getattr(spec, name)) for name in sorted(_ALLOWED_FIELDS)}
    db = DeviceDatabase()
    return {
        "spec": fields,
        "topologies": [item.value for item in PrimaryTopology],
        "primary_devices": [item.part_number for item in db.primary],
        "sr_devices": [item.part_number for item in db.sr],
    }


def spec_from_payload(payload: dict[str, Any]) -> LLCDesignSpec:
    if not isinstance(payload, dict):
        raise ValueError("request body must contain an object named 'spec'")
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"unsupported public web parameter(s): {', '.join(unknown)}")

    changes: dict[str, Any] = {}
    for name, raw in payload.items():
        if name in _NUMERIC_LIMITS:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            lower, upper = _NUMERIC_LIMITS[name]
            if value < lower or value > upper:
                raise ValueError(f"{name} must be within {lower:g}..{upper:g}")
            if name in _INTEGER_FIELDS:
                if abs(value - round(value)) > 1e-9:
                    raise ValueError(f"{name} must be an integer")
                value = int(round(value))
            changes[name] = value
        elif name == "primary_topology":
            changes[name] = PrimaryTopology(str(raw))
        elif name in _STRING_FIELDS:
            changes[name] = str(raw)

    spec = LLCDesignSpec().clone(**changes)
    # Validate device selections explicitly for a clean HTTP 422 rather than a
    # later KeyError inside the analysis engine.
    db = DeviceDatabase()
    db.get_primary(spec.primary_device)
    db.get_sr(spec.sr_device)
    spec.validate()
    return spec


def _operating_point_row(point_loss) -> dict[str, Any]:
    op = point_loss.operating_point
    zvs_margin = min(point_loss.primary.zvs_charge_margin, point_loss.primary.zvs_energy_margin)
    return {
        "label": point_loss.label,
        "vbus_v": op.vbus_v,
        "load_fraction": op.load_fraction,
        "pout_w": op.pout_w,
        "switching_frequency_hz": op.switching_frequency_hz,
        "normalized_frequency": op.normalized_frequency,
        "required_gain": op.required_gain,
        "achieved_gain": op.achieved_gain,
        "input_phase_deg": op.input_phase_deg,
        "resonant_current_rms_a": op.resonant_current_rms_a,
        "resonant_current_peak_a": op.resonant_current_peak_a,
        "magnetizing_current_peak_a": op.magnetizing_current_peak_a,
        "secondary_current_rms_a": op.secondary_current_rms_a,
        "commutation_current_a": op.commutation_current_a,
        "zvs_margin": zvs_margin,
        "total_loss_w": point_loss.total_loss_w,
        "efficiency": point_loss.efficiency,
        "branch": op.branch,
    }


def _gain_map(analysis: SystemAnalysis, samples: int = 360) -> dict[str, Any]:
    spec = analysis.spec
    tank = analysis.tank
    frequencies = np.linspace(spec.minimum_frequency_hz, spec.maximum_frequency_hz, samples)
    curves = []
    for load in (0.10, 0.25, 0.50, 0.75, 1.00):
        pout = spec.pout_w * max(load, spec.minimum_modeled_load_fraction)
        rac = equivalent_ac_load_ohm(
            spec.turns_ratio,
            spec.vout_v + spec.rectifier_equivalent_drop_v,
            pout * (1.0 + spec.rectifier_equivalent_drop_v / spec.vout_v),
        )
        curves.append({
            "load_fraction": load,
            "frequency_hz": frequencies.tolist(),
            "gain": gain_vector(tank, frequencies, rac).tolist(),
        })
    targets = [
        {"vbus_v": bus, "gain": target_gain(spec, bus)}
        for bus in (spec.vbus_hold_end_v, spec.vbus_min_normal_v, spec.vbus_nom_v, spec.vbus_max_v)
    ]
    return {"curves": curves, "targets": targets, "fr_hz": tank.fr_hz, "fm_hz": tank.fm_hz}


def analyze_llc(payload: dict[str, Any]) -> dict[str, Any]:
    spec = spec_from_payload(payload)
    analysis = LLCSystemAnalyzer().analyze(spec)
    nominal = analysis.nominal
    worst_loss = analysis.worst_loss
    min_eff = analysis.minimum_efficiency
    op = nominal.operating_point

    transformer = analysis.transformer
    inductor = analysis.resonant_inductor

    # The fast design path does not retain the loss-evaluation object, but the
    # ranked candidate summaries carry the full-evaluation results (hotspot and
    # core cost) for the selected core.  Guarded against an empty list so the
    # web layer never fabricates a thermal number.
    transformer_hotspot_c = (
        transformer.alternatives[0].nominal_hotspot_c if transformer.alternatives else None
    )
    inductor_hotspot_c = (
        inductor.alternatives[0].nominal_hotspot_c if inductor.alternatives else None
    )
    hotspots = [x for x in (transformer_hotspot_c, inductor_hotspot_c) if x is not None]
    max_hotspot_c = max(hotspots) if hotspots else None
    transformer_cost_usd = transformer.alternatives[0].cost_usd if transformer.alternatives else None
    inductor_cost_usd = inductor.alternatives[0].cost_usd if inductor.alternatives else None
    db = DeviceDatabase()
    primary_device = db.get_primary(spec.primary_device)
    sr_device = db.get_sr(spec.sr_device)
    device_cost_usd = (
        primary_device.price_usd * spec.primary_parallel_devices
        + sr_device.price_usd * spec.sr_parallel_devices_per_position
    )
    magnetic_cost_usd = sum(x for x in (transformer_cost_usd, inductor_cost_usd) if x is not None)
    bom_cost_usd = device_cost_usd + magnetic_cost_usd if magnetic_cost_usd else None

    return {
        "status": "PASS" if analysis.feasible else "FAIL",
        "feasible": analysis.feasible,
        "warnings": list(analysis.warnings),
        "feasibility_reasons": list(analysis.feasibility_reasons),
        "spec": _jsonable(asdict(spec)),
        "summary": {
            "lr_h": analysis.tank.lr_h,
            "cr_f": analysis.tank.cr_f,
            "lm_h": analysis.tank.lm_h,
            "zr_ohm": analysis.tank.zr_ohm,
            "rac_nom_ohm": analysis.tank.rac_nom_ohm,
            "turns_ratio": spec.turns_ratio,
            "nominal_switching_frequency_hz": op.switching_frequency_hz,
            "nominal_total_loss_w": nominal.total_loss_w,
            "nominal_efficiency": nominal.efficiency,
            "nominal_input_phase_deg": op.input_phase_deg,
            "nominal_zvs_margin": min(nominal.primary.zvs_charge_margin, nominal.primary.zvs_energy_margin),
            "worst_loss_w": worst_loss.total_loss_w,
            "worst_loss_label": worst_loss.label,
            "minimum_efficiency": min_eff.efficiency,
            "minimum_efficiency_label": min_eff.label,
            "transformer_core": transformer.core.part_number,
            "transformer_fill_factor": transformer.fill_factor,
            "transformer_feasible": transformer.feasible,
            "transformer_worst_b_peak_t": transformer.worst_b_peak_t,
            "transformer_gap_total_mm": transformer.gap_total_mm,
            "transformer_primary_turns": transformer.primary_turns,
            "transformer_secondary_turns": transformer.secondary_turns,
            "transformer_hotspot_c": transformer_hotspot_c,
            "resonant_inductor_core": inductor.core.part_number,
            "resonant_inductor_turns": inductor.turns,
            "resonant_inductor_layers": inductor.layers,
            "resonant_inductor_feasible": inductor.feasible,
            "resonant_inductor_hotspot_c": inductor_hotspot_c,
            "max_hotspot_c": max_hotspot_c,
            "transformer_cost_usd": transformer_cost_usd,
            "resonant_inductor_cost_usd": inductor_cost_usd,
            "device_cost_usd": device_cost_usd,
            "bom_cost_usd": bom_cost_usd,
            "zvs_required_margin": spec.primary_zvs_margin_required,
            "frequency_range_hz": [spec.minimum_frequency_hz, spec.maximum_frequency_hz],
            "primary_topology": spec.primary_topology.value,
            "secondary_topology": spec.secondary_topology.value,
        },
        "nominal_loss_breakdown": nominal.breakdown(),
        "operating_points": [_operating_point_row(item) for item in analysis.operating_points],
        "gain_map": _gain_map(analysis),
    }


def core_catalog() -> dict[str, Any]:
    """Read-only view of the bundled core / material databases.

    This is the server-side source for the Component Database browser; the
    front-end never hard-codes core records.
    """
    with open(_DATA_DIR / "cores.json", encoding="utf-8") as handle:
        cores_payload = json.load(handle)
    with open(_DATA_DIR / "materials.json", encoding="utf-8") as handle:
        materials_payload = json.load(handle)
    with open(_DATA_DIR / "devices.json", encoding="utf-8") as handle:
        devices_payload = json.load(handle)

    cores = [
        {
            "part_number": item["part_number"],
            "family": item["family"],
            "shape": item["shape"],
            "manufacturer": item["manufacturer"],
            "material_key": item["material_key"],
            "purposes": list(item["purposes"]),
            "ae_mm2": item["ae_mm2"],
            "ve_mm3": item["ve_mm3"],
            "le_mm": item["le_mm"],
            "core_mass_g": item["core_mass_g"],
            "cost_usd": item["cost_usd"],
            "thermal_resistance_k_per_w": item["thermal_resistance_k_per_w"],
        }
        for item in cores_payload["cores"]
    ]
    families = sorted({item["family"] for item in cores})
    materials = [
        {
            "key": item["key"],
            "manufacturer": item["manufacturer"],
            "grade": item["grade"],
            "mu_i_25": item["mu_i_25"],
            "frequency_range_hz": item.get("frequency_range_hz"),
        }
        for item in materials_payload["materials"]
    ]
    return {
        "cores": cores,
        "materials": materials,
        "families": families,
        "presets": list(transformer_core_presets()),
        "devices": {
            "primary": devices_payload["primary_mosfets"],
            "sr": devices_payload["sr_mosfets"],
        },
        "metadata": {
            "cores": cores_payload.get("metadata"),
            "materials": materials_payload.get("metadata"),
        },
    }


def transformer_core_presets() -> list[dict[str, Any]]:
    with open(_DATA_DIR / "transformer_core_presets.json", encoding="utf-8") as handle:
        return json.load(handle)["presets"]


def _linspace_list(lo: float, hi: float, steps: int) -> list[float]:
    if steps <= 1:
        return [lo]
    return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]


def optimize_llc(payload: dict[str, Any]) -> dict[str, Any]:
    """Bounded multi-objective sweep over Ln and Q (both feed design_tank).

    The full engineering kernel is re-run for every grid point so ranking uses
    the same loss/ZVS/thermal models as the single-point analysis.  The sweep
    is deliberately bounded (default 5x5) to keep one request ~10 s.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("spec"), dict):
        raise ValueError("optimizer requires an object named 'spec'")

    base = dict(payload["spec"])
    sweep = payload.get("sweep") or {}
    weights = payload.get("weights") or {}

    def _range_arg(name: str, default_lo: float, default_hi: float) -> list[float]:
        entry = sweep.get(name)
        if entry is None:
            return _linspace_list(default_lo, default_hi, 5)
        if isinstance(entry, list):
            values = [float(v) for v in entry]
            if len(values) > 8:
                raise ValueError(f"{name} sweep is bounded to 8 points")
            return values
        lo = float(entry.get("min", default_lo))
        hi = float(entry.get("max", default_hi))
        steps = int(entry.get("steps", 5))
        if not 2 <= steps <= 8:
            raise ValueError(f"{name} sweep steps must be 2..8")
        return _linspace_list(lo, hi, steps)

    ln_values = _range_arg("ln_ratio", 3.0, 8.0)
    q_values = _range_arg("q_full_load", 0.2, 1.0)
    w_eff = float(weights.get("efficiency", 0.5))
    w_cost = float(weights.get("cost", 0.25))
    w_size = float(weights.get("size", 0.25))
    total_w = w_eff + w_cost + w_size
    if total_w <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w_eff, w_cost, w_size = w_eff / total_w, w_cost / total_w, w_size / total_w

    population: list[dict[str, Any]] = []
    for ln in ln_values:
        for q in q_values:
            trial = dict(base)
            trial["ln_ratio"] = ln
            trial["q_full_load"] = q
            try:
                result = analyze_llc(trial)
            except (ValueError, KeyError):
                continue  # one infeasible grid point must not kill the sweep
            summary = result["summary"]
            cost = summary.get("bom_cost_usd")
            core_volume_mm3 = None
            if summary.get("transformer_core") and summary.get("resonant_inductor_core"):
                # Core volume is a stable, kernel-backed size proxy.
                core_volume_mm3 = _core_volume_mm3(summary["transformer_core"]) + _core_volume_mm3(
                    summary["resonant_inductor_core"]
                )
            population.append({
                "ln_ratio": ln,
                "q_full_load": q,
                "feasible": result["feasible"],
                "efficiency": summary["nominal_efficiency"],
                "minimum_efficiency": summary["minimum_efficiency"],
                "total_loss_w": summary["nominal_total_loss_w"],
                "zvs_margin": summary["nominal_zvs_margin"],
                "lr_h": summary["lr_h"],
                "lm_h": summary["lm_h"],
                "cr_f": summary["cr_f"],
                "switching_frequency_hz": summary["nominal_switching_frequency_hz"],
                "max_hotspot_c": summary["max_hotspot_c"],
                "transformer_core": summary["transformer_core"],
                "resonant_inductor_core": summary["resonant_inductor_core"],
                "cost_usd": cost,
                "core_volume_mm3": core_volume_mm3,
            })

    if not population:
        return {"results": [], "population": 0, "weights": {"efficiency": w_eff, "cost": w_cost, "size": w_size}}

    def _norm(key: str, invert: bool = False) -> dict[Any, float]:
        values = [row[key] for row in population if row[key] is not None]
        lo, hi = min(values), max(values)
        span = hi - lo
        mapping: dict[Any, float] = {}
        for row in population:
            value = row[key]
            if value is None:
                mapping[id(row)] = 1.0
                continue
            norm = 1.0 if span == 0.0 else (value - lo) / span
            mapping[id(row)] = 1.0 - norm if invert else norm
        return mapping

    eff_norm = _norm("efficiency")
    cost_norm = _norm("cost_usd", invert=True)
    size_norm = _norm("core_volume_mm3", invert=True)
    for row in population:
        score = w_eff * eff_norm[id(row)] + w_cost * cost_norm[id(row)] + w_size * size_norm[id(row)]
        row["score"] = score

    ranked = sorted(population, key=lambda row: (-row["feasible"], -row["score"]))
    top = ranked[:5]
    for index, row in enumerate(top, start=1):
        row["rank"] = index
    return {
        "results": top,
        "population": len(population),
        "weights": {"efficiency": w_eff, "cost": w_cost, "size": w_size},
        "sweep": {"ln_ratio": ln_values, "q_full_load": q_values},
    }


def _core_volume_mm3(part_number: str) -> float | None:
    """Resolve a core record's effective volume for size ranking."""
    catalog = core_catalog()
    for core in catalog["cores"]:
        if core["part_number"] == part_number:
            return core["ve_mm3"]
    return None
