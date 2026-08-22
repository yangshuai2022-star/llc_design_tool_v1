"""Adapter for the single-phase totem-pole PFC (TTPL) workspace.

The adapter is deliberately a thin boundary around the existing PFC Control
Lab algorithms.  It translates JSON-shaped nested configuration into the
actual frozen dataclasses, executes the requested operation, and converts the
result with :func:`web_api.serialization.jsonable`.  No control equations are
reimplemented here.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any

from llc_design import __version__ as TOOLKIT_VERSION
from llc_design.control.digital_loop import (
    ControllerKind,
    PIControllerConfig,
    PIFControllerConfig,
    TwoP2ZControllerConfig,
    controller_kind,
)
from pfc_design.control import (
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    default_current_controller,
    default_current_sense,
    default_vac_sense,
    default_vbus_sense,
    default_voltage_controller,
    export_pfc_control_lab,
    simulate_pfc_line_cycle,
    tune_pfc_current_loop,
)
from pfc_design.control.config import (
    ADCTimingConfig,
    DigitalFilterConfig,
    ExternalSenseConfig,
    LoadModel,
    PFCFirmwareAlgorithmConfig,
    PFCPowerStageConfig,
)
from pfc_design.magnetics import (
    HIGH_FLUX_254,
    HighFluxCoreGeometry,
    HighFluxMaterial,
    PFCInductorDesignRequest,
    design_pfc_inductor,
    high_flux_254_material,
)
from power_codegen import generate_ttpl_control_code
from web_api.serialization import jsonable

ALGORITHM_VERSION = "ttpl-control-lab-v7"

# Resource ceilings for the anonymous 1-vCPU/2-GB service boundary.  These
# retain the GUI/CLI ranges while stopping accidental multi-gigabyte arrays.
MAX_FREQUENCY_POINTS = 20_000
MAX_WAVEFORM_LINE_CYCLES = 20
MAX_WAVEFORM_INTEGRATION_RATE_HZ = 2.0e6
MAX_LINE_CYCLE_SAMPLES = 5_000_000
MAX_SWITCHING_CYCLES = 50
MAX_SWITCHING_SAMPLES_PER_CYCLE = 10_000
MAX_SWITCHING_SAMPLES = 400_000
MAX_INDUCTOR_CURVE_POINTS = 5_000


class TTPLAdapterError(ValueError):
    """Invalid TTPL request shape or operation option."""


class TTPLAdapterEnvelope(dict[str, Any]):
    """Public JSON payload plus non-public paths for ArtifactStore handoff.

    The mapping behavior preserves existing callers that index adapter results,
    while ``artifact_paths`` remains outside the JobResultPayload contract.
    """

    def __init__(
        self,
        payload: Mapping[str, Any],
        artifact_paths: Mapping[str, Path] | None = None,
    ) -> None:
        super().__init__(payload)
        self.payload = dict(payload)
        self.artifact_paths = dict(artifact_paths or {})


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TTPLAdapterError(f"{path} must be an object")
    return dict(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise TTPLAdapterError(f"{path} contains unknown field(s): {joined}")


def _field_names(cls: type[Any]) -> set[str]:
    return {item.name for item in fields(cls)}


def _enum(value: Any, enum_type: type[Enum], path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise TTPLAdapterError(f"{path} must be one of: {choices}") from exc


def _build_filter(value: Any, base: DigitalFilterConfig, path: str) -> DigitalFilterConfig:
    data = _mapping(value, path)
    _reject_unknown(data, _field_names(DigitalFilterConfig), path)
    return replace(base, **data)


def _build_timing(value: Any, base: ADCTimingConfig, path: str) -> ADCTimingConfig:
    data = _mapping(value, path)
    allowed = _field_names(ADCTimingConfig)
    _reject_unknown(data, allowed, path)
    filter_data = data.pop("digital_filter", None)
    if filter_data is not None:
        data["digital_filter"] = _build_filter(
            filter_data, base.digital_filter, f"{path}.digital_filter"
        )
    return replace(base, **data)


def _build_sense(
    value: Any,
    base: ExternalSenseConfig,
    path: str,
) -> ExternalSenseConfig:
    data = _mapping(value, path)
    _reject_unknown(data, _field_names(ExternalSenseConfig), path)
    timing_data = data.pop("timing", None)
    if timing_data is not None:
        data["timing"] = _build_timing(timing_data, base.timing, f"{path}.timing")
    return replace(base, **data)


def _build_controller(
    value: Any,
    base: PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig,
    path: str,
) -> PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig:
    data = _mapping(value, path)
    if "kind" in data and "controller_kind" in data:
        raise TTPLAdapterError(f"{path} must use either kind or controller_kind, not both")
    kind_value = data.pop("kind", data.pop("controller_kind", None))
    if kind_value is None:
        kind_value = controller_kind(base).value
    kind = _enum(kind_value, ControllerKind, f"{path}.kind")
    controller_type: type[PIControllerConfig | PIFControllerConfig | TwoP2ZControllerConfig]
    if kind is ControllerKind.PI:
        controller_type = PIControllerConfig
        default = PIControllerConfig(
            kp=base.kp if isinstance(base, (PIControllerConfig, PIFControllerConfig)) else 0.01,
            ti_s=base.ti_s if isinstance(base, (PIControllerConfig, PIFControllerConfig)) else 1.0e-3,
            sample_time_s=base.sample_time_s,
            output_min=base.output_min,
            output_max=base.output_max,
        )
    elif kind is ControllerKind.PIF:
        controller_type = PIFControllerConfig
        default = PIFControllerConfig(
            kp=base.kp if isinstance(base, (PIControllerConfig, PIFControllerConfig)) else 0.01,
            ti_s=base.ti_s if isinstance(base, (PIControllerConfig, PIFControllerConfig)) else 1.0e-3,
            lpf_cutoff_hz=base.lpf_cutoff_hz if isinstance(base, PIFControllerConfig) else 3500.0,
            sample_time_s=base.sample_time_s,
            output_min=base.output_min,
            output_max=base.output_max,
        )
    else:
        controller_type = TwoP2ZControllerConfig
        default = TwoP2ZControllerConfig(
            b0=base.b0 if isinstance(base, TwoP2ZControllerConfig) else 0.0,
            b1=base.b1 if isinstance(base, TwoP2ZControllerConfig) else 0.0,
            b2=base.b2 if isinstance(base, TwoP2ZControllerConfig) else 0.0,
            a1=base.a1 if isinstance(base, TwoP2ZControllerConfig) else 0.0,
            a2=base.a2 if isinstance(base, TwoP2ZControllerConfig) else 0.0,
            sample_time_s=base.sample_time_s,
            output_min=base.output_min,
            output_max=base.output_max,
        )
    _reject_unknown(data, _field_names(controller_type), path)
    return replace(default, **data)


def build_ttpl_config(value: Mapping[str, Any] | PFCControlLabConfig | None = None) -> PFCControlLabConfig:
    """Build and validate a complete TTPL config from nested request data.

    Every nested object is checked before construction.  Defaults are taken
    from the real ``PFCControlLabConfig`` factory and its three independent
    sensing-chain factories; the reference JSON is not treated as a config.
    """

    if isinstance(value, PFCControlLabConfig):
        value.validate()
        return value
    data = _mapping(value, "config")
    _reject_unknown(data, _field_names(PFCControlLabConfig), "config")
    base = PFCControlLabConfig()
    power_data = data.pop("power_stage", None)
    firmware_data = data.pop("firmware", None)
    current_controller = data.pop("current_controller", None)
    voltage_controller = data.pop("voltage_controller", None)
    current_sense = data.pop("current_sense", None)
    vac_sense = data.pop("vac_sense", None)
    vbus_sense = data.pop("vbus_sense", None)
    if power_data is not None:
        power_map = _mapping(power_data, "config.power_stage")
        _reject_unknown(power_map, _field_names(PFCPowerStageConfig), "config.power_stage")
        if "load_model" in power_map:
            power_map["load_model"] = _enum(
                power_map["load_model"], LoadModel, "config.power_stage.load_model"
            )
        data["power_stage"] = replace(base.power_stage, **power_map)
    if firmware_data is not None:
        firmware_map = _mapping(firmware_data, "config.firmware")
        _reject_unknown(firmware_map, _field_names(PFCFirmwareAlgorithmConfig), "config.firmware")
        data["firmware"] = replace(base.firmware, **firmware_map)
    if current_controller is not None:
        data["current_controller"] = _build_controller(
            current_controller, default_current_controller(), "config.current_controller"
        )
    if voltage_controller is not None:
        data["voltage_controller"] = _build_controller(
            voltage_controller, default_voltage_controller(), "config.voltage_controller"
        )
    if current_sense is not None:
        data["current_sense"] = _build_sense(
            current_sense, default_current_sense(), "config.current_sense"
        )
    if vac_sense is not None:
        data["vac_sense"] = _build_sense(vac_sense, default_vac_sense(), "config.vac_sense")
    if vbus_sense is not None:
        data["vbus_sense"] = _build_sense(
            vbus_sense, default_vbus_sense(), "config.vbus_sense"
        )
    config = replace(base, **data)
    config.validate()
    return config


def _build_core(value: Any, path: str = "config.inductor.core") -> HighFluxCoreGeometry:
    data = _mapping(value, path)
    _reject_unknown(data, _field_names(HighFluxCoreGeometry), path)
    return replace(HIGH_FLUX_254, **data)


def _build_material(value: Any, path: str = "config.inductor.material") -> HighFluxMaterial:
    data = _mapping(value, path)
    _reject_unknown(data, _field_names(HighFluxMaterial), path)
    if "permeability" in data:
        base = high_flux_254_material(int(data["permeability"]))
    else:
        base = high_flux_254_material(60)
    return replace(base, **data)


def build_inductor_request(
    value: Mapping[str, Any] | PFCInductorDesignRequest | None = None,
    *,
    config: PFCControlLabConfig | Mapping[str, Any] | None = None,
) -> PFCInductorDesignRequest:
    """Build a real ``PFCInductorDesignRequest`` with TTPL defaults."""

    if isinstance(value, PFCInductorDesignRequest):
        value.validate()
        return value
    data = _mapping(value, "config.inductor")
    if "inductor_request" in data:
        nested = data.pop("inductor_request")
        if data:
            raise TTPLAdapterError("config.inductor cannot mix inductor_request with sibling fields")
        data = _mapping(nested, "config.inductor_request")
    if "inductor" in data:
        nested = data.pop("inductor")
        if data:
            raise TTPLAdapterError("config.inductor cannot mix inductor with sibling fields")
        data = _mapping(nested, "config.inductor")
    config_obj = build_ttpl_config(config) if not isinstance(config, PFCControlLabConfig) else config
    stage = config_obj.power_stage
    allowed = _field_names(PFCInductorDesignRequest)
    _reject_unknown(data, allowed, "config.inductor")
    core_data = data.pop("core", None)
    material_data = data.pop("material", None)
    request = PFCInductorDesignRequest(
        topology=str(data.pop("topology", "ttpl")),
        input_rms_v=float(data.pop("input_rms_v", stage.vin_rms_v)),
        bus_voltage_v=float(data.pop("bus_voltage_v", stage.bus_voltage_v)),
        output_power_w=float(data.pop("output_power_w", stage.output_power_w)),
        switching_frequency_hz=float(
            data.pop("switching_frequency_hz", stage.switching_frequency_hz)
        ),
        target_inductance_uh=float(
            data.pop("target_inductance_uh", stage.boost_inductance_h * 1.0e6)
        ),
        efficiency=float(data.pop("efficiency", stage.efficiency)),
        core=_build_core(core_data) if core_data is not None else HIGH_FLUX_254,
        material=_build_material(material_data) if material_data is not None else high_flux_254_material(60),
        n_cores=int(data.pop("n_cores", 2)),
        wire_copper_diameter_mm=float(data.pop("wire_copper_diameter_mm", 1.0)),
        enamel_build_mm=float(data.pop("enamel_build_mm", 0.05)),
        target_current_density_a_mm2=float(data.pop("target_current_density_a_mm2", 5.0)),
        copper_temperature_c=float(data.pop("copper_temperature_c", 100.0)),
        max_fill_factor=float(data.pop("max_fill_factor", 0.45)),
        winding_length_factor=float(data.pop("winding_length_factor", 1.10)),
        curve_points=int(data.pop("curve_points", 161)),
    )
    if data:
        # The check above is intentionally repeated after popping nested fields
        # so a future dataclass field cannot be silently ignored by this mapper.
        raise TTPLAdapterError(f"config.inductor contains unknown field(s): {', '.join(sorted(data))}")
    request.validate()
    return request


def _bounded_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise TTPLAdapterError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TTPLAdapterError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise TTPLAdapterError(f"{name} must be an integer")
    integer = int(numeric)
    if integer > maximum:
        raise TTPLAdapterError(f"{name} exceeds resource limit {maximum}")
    return integer


def _bounded_real(value: Any, name: str, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TTPLAdapterError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise TTPLAdapterError(f"{name} must be finite")
    if numeric > maximum:
        raise TTPLAdapterError(f"{name} exceeds resource limit {maximum:g}")
    return numeric


def _enforce_resource_bounds(config: PFCControlLabConfig) -> None:
    """Reject allocation multipliers before entering numerical algorithms."""

    _bounded_int(config.frequency_points, "frequency_points", MAX_FREQUENCY_POINTS)
    line_cycles = _bounded_int(
        config.waveform_line_cycles, "waveform_line_cycles", MAX_WAVEFORM_LINE_CYCLES
    )
    integration_rate = _bounded_real(
        config.waveform_integration_rate_hz,
        "waveform_integration_rate_hz",
        MAX_WAVEFORM_INTEGRATION_RATE_HZ,
    )
    line_frequency = _bounded_real(
        config.power_stage.line_frequency_hz, "line_frequency_hz", math.inf
    )
    switching_cycles = _bounded_int(
        config.switching_cycles, "switching_cycles", MAX_SWITCHING_CYCLES
    )
    samples_per_cycle = _bounded_int(
        config.switching_samples_per_cycle,
        "switching_samples_per_cycle",
        MAX_SWITCHING_SAMPLES_PER_CYCLE,
    )
    line_samples = math.ceil(
        line_cycles * integration_rate / line_frequency
    ) + 1
    if line_samples > MAX_LINE_CYCLE_SAMPLES:
        raise TTPLAdapterError(
            f"line-cycle sample count {line_samples} exceeds resource limit {MAX_LINE_CYCLE_SAMPLES}"
        )
    switching_samples = switching_cycles * samples_per_cycle
    if switching_samples > MAX_SWITCHING_SAMPLES:
        raise TTPLAdapterError(
            f"switching sample count {switching_samples} exceeds resource limit {MAX_SWITCHING_SAMPLES}"
        )


def _enforce_switching_options(
    config: PFCControlLabConfig, options: Mapping[str, Any]
) -> None:
    cycles = options.get("cycles", 1 if "samples" in options else config.switching_cycles)
    samples_per_cycle = options.get(
        "samples_per_cycle", options.get("samples", config.switching_samples_per_cycle)
    )
    cycle_count = _bounded_int(cycles, "switching cycles", MAX_SWITCHING_CYCLES)
    sample_count = _bounded_int(
        samples_per_cycle,
        "switching samples per cycle",
        MAX_SWITCHING_SAMPLES_PER_CYCLE,
    )
    total = cycle_count * sample_count
    if total > MAX_SWITCHING_SAMPLES:
        raise TTPLAdapterError(
            f"switching sample count {total} exceeds resource limit {MAX_SWITCHING_SAMPLES}"
        )


def _enforce_inductor_bounds(request: PFCInductorDesignRequest) -> None:
    _bounded_int(request.curve_points, "inductor curve points", MAX_INDUCTOR_CURVE_POINTS)


def _job_output(job_dir: str | Path | None, operation: str) -> tuple[Path, Path]:
    if job_dir is None:
        raise TTPLAdapterError(f"{operation} requires an explicit job_dir output root")
    root = Path(job_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    output = (root / ("ttpl_control_lab" if operation == "full_export" else "generated_ttpl")).resolve()
    if root not in output.parents:
        raise TTPLAdapterError("resolved output escaped job_dir")
    return root, output


def _margins(value: Any) -> dict[str, Any]:
    return {
        "gain_crossovers_hz": value.gain_crossovers_hz,
        "phase_margins_deg": value.phase_margins_deg,
        "phase_crossovers_hz": value.phase_crossovers_hz,
        "gain_margins_db": value.gain_margins_db,
        "critical_gain_crossover_hz": value.critical_gain_crossover_hz,
        "phase_margin_deg": value.phase_margin_deg,
        "critical_phase_crossover_hz": value.critical_phase_crossover_hz,
        "gain_margin_db": value.gain_margin_db,
        "delay_margin_s": value.delay_margin_s,
    }


def _loop_metrics(loop: Any) -> dict[str, Any]:
    margins = _margins(loop.margins)
    return {
        "name": loop.name,
        "controller": loop.controller.name,
        "likely_stable": loop.likely_stable,
        "margins": margins,
        # Keep the common scalar margin names easy for API consumers.
        "crossover_hz": margins["critical_gain_crossover_hz"],
        "phase_margin_deg": margins["phase_margin_deg"],
        "gain_margin_db": margins["gain_margin_db"],
    }


def _analysis_payload(analysis: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = {
        "operating_point": asdict(analysis.operating_point),
        "current_loop": _loop_metrics(analysis.current_loop),
        "voltage_loop": _loop_metrics(analysis.voltage_loop),
        "sensing": {
            "current": asdict(analysis.current_sense_summary),
            "vac": asdict(analysis.vac_sense_summary),
            "vbus": asdict(analysis.vbus_sense_summary),
        },
        "angle_margins": {str(angle): _margins(margin) for angle, margin in analysis.angle_margins.items()},
    }
    series = {
        "frequencies_hz": analysis.frequencies_hz,
        "current_loop": analysis.current_loop.responses,
        "voltage_loop": analysis.voltage_loop.responses,
        "sensing": {
            "current": analysis.current_sense_response.total,
            "vac": analysis.vac_sense_response.total,
            "vbus": analysis.vbus_sense_response.total,
        },
    }
    tables = {
        "current_loop_margins": _margins(analysis.current_loop.margins),
        "voltage_loop_margins": _margins(analysis.voltage_loop.margins),
        "angle_margins": metrics["angle_margins"],
    }
    return metrics, series, tables


def _waveform_payload(waveform: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[str]]:
    metrics = asdict(waveform.metrics) if hasattr(waveform, "metrics") else {}
    series = {"time_s": waveform.time_s, "signals": waveform.signals}
    return metrics, series, dict(waveform.units), list(getattr(waveform, "warnings", ()))


def _config_snapshot(config: PFCControlLabConfig) -> dict[str, Any]:
    snapshot = jsonable(config)
    for name in ("current_controller", "voltage_controller"):
        snapshot[name]["kind"] = controller_kind(getattr(config, name)).value
    return snapshot


def _result(
    operation: str,
    config: PFCControlLabConfig,
    started: float,
    *,
    parameters: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    tables: Any = None,
    series: Any = None,
    units: Mapping[str, str] | None = None,
    feasibility: bool | None = None,
    warnings: list[str] | tuple[str, ...] = (),
    evidence: Any = None,
    artifact_paths: Mapping[str, Path] | None = None,
) -> TTPLAdapterEnvelope:
    payload = {
        "toolkit_version": TOOLKIT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "workspace": "ttpl",
        "operation": operation,
        "config_snapshot": _config_snapshot(config),
        "parameters": dict(parameters or {}),
        "stage": "complete",
        "elapsed_s": max(0.0, time.perf_counter() - started),
        "metrics": dict(metrics or {}),
        "tables": {} if tables is None else tables,
        "series": {} if series is None else series,
        "plots": {},
        "units": dict(units or {}),
        "feasibility": feasibility,
        "warnings": list(warnings),
        "evidence": [] if evidence is None else evidence,
        "artifacts": [],
    }
    return TTPLAdapterEnvelope(jsonable(payload), artifact_paths=artifact_paths)


def _effective_analysis(
    config: PFCControlLabConfig,
    options: Mapping[str, Any],
) -> tuple[PFCControlLabConfig, Any | None, bool, bool]:
    """Resolve explicit autotune/apply flags without mutating the request."""

    allowed = {
        "autotune",
        "apply",
        "desired_phase_margin_deg",
        "minimum_phase_margin_deg",
        "minimum_gain_margin_db",
        "maximum_crossover_hz",
    }
    _reject_unknown(options, allowed, "options")
    autotune = bool(options.get("autotune", False))
    apply = bool(options.get("apply", False))
    if apply and not autotune:
        raise TTPLAdapterError("options.apply requires options.autotune=true")
    if not autotune:
        return config, None, False, False
    tune_kwargs = {
        key: options[key]
        for key in (
            "desired_phase_margin_deg",
            "minimum_phase_margin_deg",
            "minimum_gain_margin_db",
            "maximum_crossover_hz",
        )
        if key in options
    }
    tune = tune_pfc_current_loop(config, **tune_kwargs)
    tuned_config = replace(config, current_controller=tune.controller)
    return (tuned_config if apply else config), tune, autotune, apply


def _split_inductor_config(config: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _mapping(config, "config")
    inductor_data: dict[str, Any] = {}
    for key in ("inductor", "inductor_request"):
        if key in data:
            if inductor_data:
                raise TTPLAdapterError("config cannot contain both inductor and inductor_request")
            inductor_data = _mapping(data.pop(key), f"config.{key}")
    return data, inductor_data


def run_ttpl_operation(
    operation: str,
    config: Mapping[str, Any] | PFCControlLabConfig | None = None,
    options: Mapping[str, Any] | None = None,
    *,
    export_options: Mapping[str, Any] | None = None,
    job_dir: str | Path | None = None,
) -> TTPLAdapterEnvelope:
    """Execute one TTPL workspace operation and return a JSON-safe payload."""

    normalized = str(operation).strip().lower()
    if normalized not in {
        "control_analysis",
        "autotune_current",
        "inductor_design",
        "line_cycle",
        "switching",
        "full_export",
        "codegen",
    }:
        raise TTPLAdapterError(f"unsupported TTPL operation: {operation!r}")
    merged_options = _mapping(options, "options")
    if export_options is not None:
        if merged_options:
            raise TTPLAdapterError("use options or export_options, not both")
        merged_options = _mapping(export_options, "export_options")
    started = time.perf_counter()

    job_root: Path | None = None
    job_output: Path | None = None
    if normalized in {"full_export", "codegen"}:
        job_root, job_output = _job_output(job_dir, normalized)

    if normalized == "inductor_design":
        if isinstance(config, PFCControlLabConfig):
            pfc_config = build_ttpl_config(config)
            inductor_data: dict[str, Any] = {}
        else:
            config_data, inductor_data = _split_inductor_config(
                config if isinstance(config, Mapping) else None
            )
            pfc_config = build_ttpl_config(config_data)
        _reject_unknown(merged_options, set(), "options")
        _enforce_resource_bounds(pfc_config)
        request = build_inductor_request(inductor_data, config=pfc_config)
        _enforce_inductor_bounds(request)
        design = design_pfc_inductor(request)
        result_metrics = {
            key: value
            for key, value in asdict(design).items()
            if key not in {"request", "current_a", "inductance_uh", "line_angle_deg", "line_b_ac_t", "line_core_loss_w", "line_ripple_pp_a", "warnings"}
        }
        result_metrics["request"] = asdict(request)
        candidate = {
            key: result_metrics[key]
            for key in (
                "turns",
                "parallel_wires",
                "l_full_load_peak_uh",
                "fill_factor",
                "window_ok",
                "inductance_target_met",
                "total_loss_w",
            )
        }
        return _result(
            normalized,
            pfc_config,
            started,
            parameters={"request": asdict(request)},
            metrics=result_metrics,
            tables={"candidates": [candidate]},
            series={
                "current_a": design.current_a,
                "inductance_uh": design.inductance_uh,
                "line_angle_deg": design.line_angle_deg,
                "line_b_ac_t": design.line_b_ac_t,
                "line_core_loss_w": design.line_core_loss_w,
                "line_ripple_pp_a": design.line_ripple_pp_a,
            },
            units={
                "current_a": "A",
                "inductance_uh": "uH",
                "line_angle_deg": "deg",
                "line_b_ac_t": "T",
                "line_core_loss_w": "W",
                "line_ripple_pp_a": "A",
            },
            feasibility=bool(design.inductance_target_met and design.window_ok),
            warnings=list(design.warnings),
            evidence=[
                "pfc_design.magnetics.pfc_inductor_designer.design_pfc_inductor",
                "Magnetics High Flux Core Data 254 fit embedded in pfc_design.magnetics.high_flux",
            ],
        )

    pfc_config = build_ttpl_config(config)
    _enforce_resource_bounds(pfc_config)

    if normalized == "control_analysis":
        effective, tune, autotune, apply = _effective_analysis(pfc_config, merged_options)
        analysis = build_pfc_control_lab_analysis(effective)
        metrics, series, tables = _analysis_payload(analysis)
        if tune is not None:
            metrics["autotune"] = {
                "accepted": tune.accepted,
                "message": tune.message,
                "controller": asdict(tune.controller),
                "applied": apply,
            }
        warnings = list(analysis.warnings)
        if tune is not None:
            warnings.extend(tune.controller and [tune.message] or [])
        return _result(
            normalized,
            effective,
            started,
            parameters={"autotune": autotune, "apply": apply, "applied": apply},
            metrics=metrics,
            tables=tables,
            series=series,
            warnings=warnings,
            evidence=["pfc_design.control.analysis.build_pfc_control_lab_analysis"],
        )

    if normalized == "autotune_current":
        allowed = {
            "apply",
            "desired_phase_margin_deg",
            "minimum_phase_margin_deg",
            "minimum_gain_margin_db",
            "maximum_crossover_hz",
        }
        _reject_unknown(merged_options, allowed, "options")
        apply = bool(merged_options.get("apply", False))
        tune_kwargs = {
            key: merged_options[key]
            for key in allowed
            if key != "apply" and key in merged_options
        }
        tune = tune_pfc_current_loop(pfc_config, **tune_kwargs)
        effective = replace(pfc_config, current_controller=tune.controller) if apply else pfc_config
        metrics = {
            "accepted": tune.accepted,
            "message": tune.message,
            "controller": asdict(tune.controller),
            "target_crossover_hz": tune.target_crossover_hz,
            "nominal_crossover_hz": tune.nominal_crossover_hz,
            "nominal_phase_margin_deg": tune.nominal_phase_margin_deg,
            "nominal_gain_margin_db": tune.nominal_gain_margin_db,
            "worst_phase_margin_deg": tune.worst_phase_margin_deg,
            "worst_gain_margin_db": tune.worst_gain_margin_db,
            "worst_point": None if tune.worst_point is None else asdict(tune.worst_point),
        }
        tables = {"envelope": [asdict(item) for item in tune.envelope]}
        if apply:
            analysis = build_pfc_control_lab_analysis(effective)
            analysis_metrics, _series, analysis_tables = _analysis_payload(analysis)
            tables["analysis"] = analysis_metrics
            tables["analysis_margins"] = analysis_tables
        return _result(
            normalized,
            effective,
            started,
            parameters={"apply": apply, "applied": apply},
            metrics=metrics,
            tables=tables,
            warnings=[tune.message],
            evidence=["pfc_design.control.autotune.tune_pfc_current_loop"],
        )

    if normalized == "line_cycle":
        _reject_unknown(merged_options, set(), "options")
        waveform = simulate_pfc_line_cycle(pfc_config)
        metrics, series, units, warnings = _waveform_payload(waveform)
        return _result(
            normalized,
            pfc_config,
            started,
            metrics=metrics,
            series=series,
            units=units,
            feasibility=metrics.get("power_factor", 0.0) > 0.0,
            warnings=warnings,
            evidence=["pfc_design.control.waveforms.simulate_pfc_line_cycle"],
        )

    if normalized == "switching":
        allowed = {"line_angle_deg", "samples", "cycles", "samples_per_cycle"}
        _reject_unknown(merged_options, allowed, "options")
        _enforce_switching_options(pfc_config, merged_options)
        line_cycle = simulate_pfc_line_cycle(pfc_config)
        waveform = build_pfc_switching_waveforms(
            pfc_config,
            line_cycle=line_cycle,
            **merged_options,
        )
        series = {"time_s": waveform.time_s, "signals": waveform.signals}
        metrics = {
            "line_angle_deg": waveform.line_angle_deg,
            "switching_frequency_hz": waveform.switching_frequency_hz,
            "source_time_s": waveform.source_time_s,
        }
        return _result(
            normalized,
            pfc_config,
            started,
            metrics=metrics,
            series=series,
            units=dict(waveform.units),
            evidence=[
                "pfc_design.control.waveforms.simulate_pfc_line_cycle",
                "pfc_design.control.waveforms.build_pfc_switching_waveforms",
            ],
        )

    if normalized == "full_export":
        _reject_unknown(merged_options, {"line_angle_deg"}, "options")
        assert job_output is not None and job_root is not None
        line_cycle = simulate_pfc_line_cycle(pfc_config)
        switching = build_pfc_switching_waveforms(
            pfc_config,
            line_cycle=line_cycle,
            line_angle_deg=merged_options.get("line_angle_deg"),
        )
        analysis = build_pfc_control_lab_analysis(pfc_config)
        paths = export_pfc_control_lab(analysis, line_cycle, switching, job_output)
        metrics, series, tables = _analysis_payload(analysis)
        line_metrics, line_series, units, warnings = _waveform_payload(line_cycle)
        tables["line_cycle_metrics"] = line_metrics
        tables["switching"] = {
            "line_angle_deg": switching.line_angle_deg,
            "switching_frequency_hz": switching.switching_frequency_hz,
        }
        series["line_cycle"] = line_series
        series["switching"] = {"time_s": switching.time_s, "signals": switching.signals}
        path_map = {name: Path(path).resolve() for name, path in paths.items()}
        return _result(
            normalized,
            pfc_config,
            started,
            parameters={"job_root_provided": True},
            metrics=metrics,
            tables=tables,
            series=series,
            units=units,
            feasibility=True,
            warnings=warnings + list(analysis.warnings),
            evidence=[
                "pfc_design.control.export.export_pfc_control_lab",
                "pfc_design.control.analysis.build_pfc_control_lab_analysis",
                "pfc_design.control.waveforms.simulate_pfc_line_cycle",
            ],
            artifact_paths=path_map,
        )

    # codegen
    allowed = {
        "duty_feedforward_enabled",
        "require_stable",
        "autotune",
        "apply",
    }
    _reject_unknown(merged_options, allowed, "options")
    assert job_output is not None and job_root is not None
    autotune = bool(merged_options.get("autotune", False))
    apply = bool(merged_options.get("apply", False))
    if apply and not autotune:
        raise TTPLAdapterError("options.apply requires options.autotune=true")
    effective = pfc_config
    tune = None
    if autotune:
        tune = tune_pfc_current_loop(pfc_config)
        if apply:
            effective = replace(pfc_config, current_controller=tune.controller)
    analysis = build_pfc_control_lab_analysis(effective)
    generated = generate_ttpl_control_code(
        analysis,
        job_output,
        duty_feedforward_enabled=bool(merged_options.get("duty_feedforward_enabled", True)),
        require_stable=bool(merged_options.get("require_stable", True)),
    )
    path_map = {name: Path(path).resolve() for name, path in generated.files.items()}
    warnings = list(generated.validation.warnings)
    if tune is not None:
        warnings.append(tune.message)
    return _result(
        normalized,
        effective,
        started,
        parameters={
            "job_root_provided": True,
            "autotune": autotune,
            "apply": apply,
            "applied": apply,
        },
        metrics={
            "validation": asdict(generated.validation),
            "analysis": _analysis_payload(analysis)[0],
        },
        warnings=warnings,
        feasibility=generated.validation.passed,
        evidence=["power_codegen.generator.generate_ttpl_control_code"],
        artifact_paths=path_map,
    )


def execute_ttpl_operation(
    operation: str,
    config: Mapping[str, Any] | PFCControlLabConfig | None = None,
    export_options: Mapping[str, Any] | None = None,
    *,
    job_dir: str | Path | None = None,
) -> TTPLAdapterEnvelope:
    """Compatibility entry point for job runners using ``export_options``."""

    return run_ttpl_operation(operation, config, export_options=export_options, job_dir=job_dir)


class TTPLAdapter:
    """Small object wrapper convenient for dependency-injected job runners."""

    workspace = "ttpl"

    @staticmethod
    def run(
        operation: str,
        config: Mapping[str, Any] | PFCControlLabConfig | None = None,
        options: Mapping[str, Any] | None = None,
        *,
        export_options: Mapping[str, Any] | None = None,
        job_dir: str | Path | None = None,
    ) -> TTPLAdapterEnvelope:
        return run_ttpl_operation(
            operation, config, options, export_options=export_options, job_dir=job_dir
        )


__all__ = [
    "ALGORITHM_VERSION",
    "MAX_FREQUENCY_POINTS",
    "MAX_INDUCTOR_CURVE_POINTS",
    "MAX_LINE_CYCLE_SAMPLES",
    "MAX_SWITCHING_CYCLES",
    "MAX_SWITCHING_SAMPLES",
    "MAX_SWITCHING_SAMPLES_PER_CYCLE",
    "MAX_WAVEFORM_INTEGRATION_RATE_HZ",
    "MAX_WAVEFORM_LINE_CYCLES",
    "TTPLAdapter",
    "TTPLAdapterEnvelope",
    "TTPLAdapterError",
    "build_inductor_request",
    "build_ttpl_config",
    "execute_ttpl_operation",
    "run_ttpl_operation",
]
