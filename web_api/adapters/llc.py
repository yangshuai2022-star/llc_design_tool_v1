"""Thin, JSON-safe adapter for the existing LLC numerical workflows.

The adapter deliberately owns no engineering algorithms.  It validates and
constructs the public configuration objects, calls the established LLC
functions, and converts their dataclass/NumPy/DataFrame results at the API
boundary.  Exporters are only called with a directory below the supplied job
directory so the job store can register the returned manifest safely.
"""

from __future__ import annotations

import json
import mimetypes
import time
from collections.abc import Mapping
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from llc_design import __version__ as TOOLKIT_VERSION
from llc_design.control.analysis import (
    SmallSignalAnalysis,
    build_small_signal_analysis,
    export_small_signal_analysis,
)
from llc_design.control.digital_loop import (
    ADCSamplingConfig,
    AnalogSenseConfig,
    CommandTimingConfig,
    ControllerConfig,
    FMLUTMode,
    FrequencyModulatorLUT,
    PIControllerConfig,
    PIFControllerConfig,
    PWMCountMode,
    TwoP2ZControllerConfig,
    build_digital_loop_analysis,
    controller_kind,
    export_digital_loop_analysis,
)
from llc_design.control.linearize import ControlInputKind
from llc_design.core.config import load_spec
from llc_design.core.q_zvs import build_q_zvs_analysis
from llc_design.core.spec import LLCDesignSpec, PrimaryTopology, SecondaryTopology
from llc_design.dynamics.export import export_waveform_bundle
from llc_design.dynamics.plant import DynamicPhasorModel
from llc_design.dynamics.switched import (
    SwitchedSimulationConfig,
    simulate_switched_steady_state,
)
from llc_design.dynamics.waveforms import (
    WaveformBundle,
    reconstruct_dynamic_phasor_waveforms,
)
from llc_design.magnetics.transformer_designer import (
    FerriteCoreInput,
    TransformerSynthesisSettings,
    export_transformer_synthesis,
    load_transformer_core_presets,
    synthesize_transformer,
)
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.optimization.sweep import LLCOptimizer, OptimizationConfig
from llc_design.report.export import export_calculation_book
from power_codegen.generator import CodegenResult, generate_llc_control_code
from web_api.serialization import jsonable

SUPPORTED_OPERATIONS = frozenset(
    {
        "system",
        "q_zvs",
        "transformer",
        "waveforms_fast",
        "waveforms_detailed",
        "small_signal",
        "digital_loop",
        "optimize",
        "codegen",
    }
)

_SPEC_FIELDS = frozenset(field.name for field in fields(LLCDesignSpec))
_T = TypeVar("_T")


class LLCAdapter:
    """Dispatch one validated LLC workspace operation."""

    def run(
        self,
        operation: str,
        config: Mapping[str, Any] | None = None,
        job_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        normalized = str(operation).strip()
        if normalized not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"unknown LLC operation: {normalized!r}; "
                f"supported operations are {sorted(SUPPORTED_OPERATIONS)}"
            )
        raw = _mapping(config, "config")
        spec, options = _split_config(raw)
        spec.validate()
        export_root = _job_operation_directory(job_dir, normalized)
        result = self._dispatch(normalized, spec, options, export_root)
        result["elapsed_s"] = float(time.perf_counter() - started)
        result["toolkit_version"] = TOOLKIT_VERSION
        result["algorithm_version"] = f"llc-{TOOLKIT_VERSION}"
        result["workspace"] = "llc"
        result["operation"] = normalized
        result["config_snapshot"] = jsonable(raw)
        return jsonable(result)

    execute = run

    def _dispatch(
        self,
        operation: str,
        spec: LLCDesignSpec,
        options: dict[str, Any],
        export_root: Path | None,
    ) -> dict[str, Any]:
        if operation == "system":
            return self._system(spec, options, export_root)
        if operation == "q_zvs":
            return self._q_zvs(spec, options, export_root)
        if operation == "transformer":
            return self._transformer(spec, options, export_root)
        if operation in {"waveforms_fast", "waveforms_detailed"}:
            return self._waveforms(spec, options, export_root, operation)
        if operation == "small_signal":
            return self._small_signal(spec, options, export_root)
        if operation == "digital_loop":
            return self._digital_loop(spec, options, export_root)
        if operation == "optimize":
            return self._optimize(spec, options, export_root)
        if operation == "codegen":
            return self._codegen(spec, options, export_root)
        raise AssertionError(f"unhandled LLC operation: {operation}")

    @staticmethod
    def _system(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = {
            "work_points",
            "preferred_transformer_core",
            "preferred_inductor_core",
        }
        _reject_unknown(options, allowed)
        analysis = LLCSystemAnalyzer().analyze(
            spec,
            work_points=options.get("work_points"),
            preferred_transformer_core=options.get("preferred_transformer_core"),
            preferred_inductor_core=options.get("preferred_inductor_core"),
        )
        paths = export_calculation_book(analysis, export_root) if export_root else {}
        points = [_system_point_row(point) for point in analysis.operating_points]
        nominal = analysis.nominal
        return _result(
            stage="analysis",
            feasibility=analysis.feasible,
            metrics={
                "nominal_efficiency": nominal.efficiency,
                "minimum_efficiency": analysis.minimum_efficiency.efficiency,
                "nominal_loss_w": nominal.total_loss_w,
                "worst_loss_w": analysis.worst_loss.total_loss_w,
                "nominal_frequency_hz": nominal.operating_point.switching_frequency_hz,
                "operating_point_count": len(points),
            },
            tables={"operating_points": points},
            series={},
            units={
                "nominal_efficiency": "1",
                "minimum_efficiency": "1",
                "nominal_loss_w": "W",
                "worst_loss_w": "W",
                "nominal_frequency_hz": "Hz",
            },
            warnings=[*analysis.warnings, *analysis.feasibility_reasons],
            evidence=["LLCSystemAnalyzer.analyze"],
            paths=paths,
        )

    @staticmethod
    def _q_zvs(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = {"load_fractions", "vbus_points", "frequency_points"}
        _reject_unknown(options, allowed)
        kwargs: dict[str, Any] = {}
        if "load_fractions" in options:
            kwargs["load_fractions"] = tuple(
                float(value) for value in options["load_fractions"]
            )
        if "vbus_points" in options:
            kwargs["vbus_points"] = tuple(
                float(value) for value in options["vbus_points"]
            )
        if "frequency_points" in options:
            kwargs["frequency_points"] = int(options["frequency_points"])
        analysis = build_q_zvs_analysis(spec, **kwargs)
        workpoints = [jsonable(asdict(item)) for item in analysis.workpoints]
        data = {
            "spec": jsonable(asdict(analysis.spec)),
            "map": jsonable(asdict(analysis.map)),
            "workpoints": workpoints,
            "warnings": list(analysis.warnings),
        }
        paths: dict[str, Path] = {}
        if export_root:
            path = export_root / "q_zvs_analysis.json"
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            paths["analysis_json"] = path
        qmap = analysis.map
        return _result(
            stage="analysis",
            feasibility=None,
            metrics={
                "frequency_points": len(qmap.frequencies_hz),
                "load_points": len(qmap.load_fractions),
                "workpoint_count": len(workpoints),
            },
            tables={"workpoints": workpoints},
            series={
                "frequencies_hz": qmap.frequencies_hz,
                "normalized_frequency": qmap.normalized_frequency,
                "load_fractions": qmap.load_fractions,
                "q_effective": qmap.q_effective,
                "gain": qmap.gain,
                "input_phase_deg": qmap.input_phase_deg,
                "zvs_margin": qmap.zvs_margin,
                "zvs_safe": qmap.zvs_safe,
                "zvs_warning": qmap.zvs_warning,
            },
            units={
                "frequencies_hz": "Hz",
                "normalized_frequency": "1",
                "load_fractions": "1",
                "input_phase_deg": "deg",
                "zvs_margin": "1",
            },
            warnings=list(analysis.warnings),
            evidence=["build_q_zvs_analysis"],
            paths=paths,
        )

    @staticmethod
    def _transformer(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = {"core", "core_input", "preset", "settings", "transformer_settings"}
        _reject_unknown(options, allowed)
        core_value = options.get(
            "core_input", options.get("core", options.get("preset"))
        )
        if core_value is None:
            core = FerriteCoreInput()
        elif isinstance(core_value, str):
            presets = load_transformer_core_presets()
            try:
                core = presets[core_value]
            except KeyError as exc:
                raise ValueError(
                    f"unknown transformer core preset: {core_value}"
                ) from exc
        else:
            core = _dataclass_from_mapping(FerriteCoreInput, core_value, "core_input")
        settings_value = options.get(
            "settings", options.get("transformer_settings", {})
        )
        settings = _dataclass_from_mapping(
            TransformerSynthesisSettings, settings_value, "transformer settings"
        )
        result = synthesize_transformer(spec, core, settings)
        paths = export_transformer_synthesis(result, export_root) if export_root else {}
        return _result(
            stage="synthesis",
            feasibility=result.feasible,
            metrics={
                "primary_turns": result.primary_turns,
                "secondary_turns": result.secondary_turns,
                "actual_turns_ratio": result.actual_turns_ratio,
                "target_turns_ratio": result.target_turns_ratio,
                "worst_b_peak_t": result.worst_b_peak_t,
                "fill_factor": result.fill_factor,
                "estimated_gap_mm": result.estimated_gap_mm,
                "nominal_loss_w": result.total_nominal_loss_w,
            },
            tables={
                "workpoints": [jsonable(asdict(item)) for item in result.workpoints]
            },
            series={},
            units={
                "worst_b_peak_t": "T",
                "fill_factor": "1",
                "estimated_gap_mm": "mm",
                "nominal_loss_w": "W",
            },
            warnings=[*result.warnings, *result.reasons],
            evidence=["synthesize_transformer", "FerriteCoreInput"],
            paths=paths,
        )

    @staticmethod
    def _waveforms(
        spec: LLCDesignSpec,
        options: dict[str, Any],
        export_root: Path | None,
        operation: str,
    ) -> dict[str, Any]:
        allowed = (
            _SMALL_SIGNAL_OPTIONS
            | {"cycles", "samples_per_cycle", "switched_config"}
            | _SWITCHED_CONFIG_FIELDS
        )
        _reject_unknown(options, allowed)
        small = _build_small_signal(spec, options)
        model = DynamicPhasorModel(small.parameters)
        cycles = int(options.get("cycles", 2))
        samples = int(
            options.get(
                "samples_per_cycle", 1024 if operation == "waveforms_fast" else 512
            )
        )
        if operation == "waveforms_fast":
            bundle = reconstruct_dynamic_phasor_waveforms(
                model,
                small.steady_state,
                cycles=cycles,
                samples_per_cycle=samples,
                spec=spec,
            )
            evidence = [
                "build_small_signal_analysis",
                "reconstruct_dynamic_phasor_waveforms",
            ]
        else:
            switched_options = options.get("switched_config", {})
            if not isinstance(switched_options, Mapping):
                raise TypeError("switched_config must be a mapping")
            switched_values = dict(switched_options)
            for key in _SWITCHED_CONFIG_FIELDS:
                if key in options and key not in switched_values:
                    switched_values[key] = options[key]
            switched_values.setdefault("samples_per_cycle", samples)
            switched_values.setdefault("output_cycles", cycles)
            switched = _dataclass_from_mapping(
                SwitchedSimulationConfig, switched_values, "switched_config"
            )
            bundle = simulate_switched_steady_state(model, small.steady_state, switched)
            evidence = ["build_small_signal_analysis", "simulate_switched_steady_state"]
        paths = export_waveform_bundle(bundle, export_root) if export_root else {}
        return _waveform_result(bundle, paths, evidence)

    @staticmethod
    def _small_signal(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        _reject_unknown(options, _SMALL_SIGNAL_OPTIONS)
        result = _build_small_signal(spec, options)
        paths = export_small_signal_analysis(result, export_root) if export_root else {}
        return _small_signal_result(result, paths, ["build_small_signal_analysis"])

    @staticmethod
    def _digital_loop(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = _SMALL_SIGNAL_OPTIONS | _DIGITAL_LOOP_OPTIONS
        _reject_unknown(options, allowed)
        small = _build_small_signal(spec, options)
        loop = _build_digital_loop(small, options)
        paths = export_digital_loop_analysis(loop, export_root) if export_root else {}
        margin = loop.margins_nominal_delay
        return _result(
            stage="analysis",
            feasibility=loop.likely_stable,
            metrics={
                "controller_kind": controller_kind(loop.controller_config).value,
                "pcmd": loop.fm_operating_point.command_pu,
                "frequency_hz": loop.fm_operating_point.frequency_hz,
                "gain_hz_per_pu": loop.fm_operating_point.gain_hz_per_pu,
                "phase_margin_deg": margin.phase_margin_deg,
                "gain_margin_db": margin.gain_margin_db,
                "likely_stable": loop.likely_stable,
                "discrete_stable": loop.discrete_approximation.stable,
                "sample_time_s": loop.controller.sample_time_s,
            },
            tables={
                "margins_nominal": asdict(margin),
                "margins_minimum": asdict(loop.margins_minimum_delay),
                "margins_maximum": asdict(loop.margins_maximum_delay),
            },
            series=_loop_series(loop),
            units={
                "frequency_hz": "Hz",
                "gain_hz_per_pu": "Hz/pu",
                "phase_margin_deg": "deg",
                "gain_margin_db": "dB",
                "sample_time_s": "s",
            },
            warnings=list(loop.warnings),
            evidence=[
                "build_small_signal_analysis",
                "build_digital_loop_analysis",
                "FrequencyModulatorLUT.firmware_default",
            ],
            paths=paths,
        )

    @staticmethod
    def _optimize(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = {"optimization_config", "optimization", "maximum_candidates"}
        _reject_unknown(options, allowed)
        config_value = options.get("optimization_config", options.get("optimization"))
        config = (
            OptimizationConfig.quick()
            if config_value is None
            else _dataclass_from_mapping(
                OptimizationConfig, config_value, "optimization_config"
            )
        )
        maximum = options.get("maximum_candidates")
        result = LLCOptimizer().run(
            spec, config, None if maximum is None else int(maximum)
        )
        paths: dict[str, Path] = {}
        if export_root:
            export_root.mkdir(parents=True, exist_ok=True)
            paths["all_csv"] = export_root / "optimization_all.csv"
            result.table.to_csv(paths["all_csv"], index=False)
            paths["pareto_csv"] = export_root / "optimization_pareto.csv"
            result.pareto.to_csv(paths["pareto_csv"], index=False)
            if result.best_analysis is not None:
                paths.update(
                    {
                        f"best_{key}": value
                        for key, value in export_calculation_book(
                            result.best_analysis, export_root / "best_design"
                        ).items()
                    }
                )
        return _result(
            stage="optimization",
            feasibility=(
                None if result.best_analysis is None else result.best_analysis.feasible
            ),
            metrics={
                "candidate_count": len(result.table),
                "feasible_count": int(result.table["feasible"].sum())
                if "feasible" in result.table
                else 0,
            },
            tables={
                "all": _dataframe_records(result.table),
                "pareto": _dataframe_records(result.pareto),
            },
            series={},
            units={},
            warnings=[],
            evidence=["LLCOptimizer.run", "OptimizationConfig"],
            paths=paths,
        )

    @staticmethod
    def _codegen(
        spec: LLCDesignSpec, options: dict[str, Any], export_root: Path | None
    ) -> dict[str, Any]:
        allowed = _SMALL_SIGNAL_OPTIONS | _DIGITAL_LOOP_OPTIONS | {"require_stable"}
        _reject_unknown(options, allowed)
        if export_root is None:
            raise ValueError("codegen requires a job_dir for generated artifacts")
        small = _build_small_signal(spec, options)
        loop = _build_digital_loop(small, options)
        require_stable = bool(options.get("require_stable", True))
        generated: CodegenResult = generate_llc_control_code(
            loop, export_root, require_stable=require_stable
        )
        validation = generated.validation
        return _result(
            stage="codegen",
            feasibility=validation.passed,
            metrics={
                "stability_gate_passed": validation.passed,
                "likely_stable": loop.likely_stable,
                "generated_file_count": len(generated.files),
            },
            tables={
                "validation": {
                    "passed": validation.passed,
                    "checks": list(validation.checks),
                    "warnings": list(validation.warnings),
                },
            },
            series={},
            units={},
            warnings=[*loop.warnings, *validation.warnings],
            evidence=[
                "build_small_signal_analysis",
                "build_digital_loop_analysis",
                "generate_llc_control_code",
            ],
            paths=generated.files,
        )


def run_llc_operation(
    operation: str,
    config: Mapping[str, Any] | None = None,
    job_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one LLC operation through the public adapter boundary."""
    return LLCAdapter().run(operation, config, job_dir)


_SMALL_SIGNAL_OPTIONS = frozenset(
    {
        "vbus_v",
        "load_fraction",
        "sample_time_s",
        "control_input_kind",
        "timer_clock_hz",
        "input_delay_samples",
        "series_resistance_ohm",
        "trim_frequency_to_output",
    }
)
_DIGITAL_LOOP_OPTIONS = frozenset(
    {
        "controller",
        "controller_config",
        "fm_lut",
        "fm_mode",
        "timer_clock_hz",
        "count_mode",
        "command_pu",
        "analog_sense",
        "adc_sampling",
        "command_timing",
        "frequencies_hz",
        "require_stable",
    }
)
_SWITCHED_CONFIG_FIELDS = frozenset(
    {
        "minimum_settling_cycles",
        "maximum_settling_cycles",
        "convergence_tolerance",
        "rectifier_smoothing_current_a",
        "use_periodic_shooting",
        "shooting_max_evaluations",
    }
)


def _mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _split_config(raw: dict[str, Any]) -> tuple[LLCDesignSpec, dict[str, Any]]:
    values = dict(raw)
    spec_value = values.pop("spec", values.pop("design_spec", None))
    if spec_value is None:
        spec_values = {
            key: values.pop(key) for key in list(values) if key in _SPEC_FIELDS
        }
    else:
        spec_values = _mapping(spec_value, "spec")
        direct_spec = {
            key: values.pop(key) for key in list(values) if key in _SPEC_FIELDS
        }
        spec_values.update(direct_spec)
    if "spec_path" in values:
        path = Path(values.pop("spec_path"))
        loaded = _mapping(asdict(load_spec(path)), "loaded spec")
        loaded.update(spec_values)
        spec_values = loaded
    spec = _dataclass_from_mapping(LLCDesignSpec, spec_values, "spec")
    return spec, values


def _dataclass_from_mapping(
    cls: type[_T], value: Mapping[str, Any] | None, label: str
) -> _T:
    values = _mapping(value, label)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown config field(s) for {label}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    enum_fields: dict[str, type[Enum]] = {
        "primary_topology": PrimaryTopology,
        "secondary_topology": SecondaryTopology,
        "control_input_kind": ControlInputKind,
        "mode": FMLUTMode,
        "count_mode": PWMCountMode,
    }
    for name, item in values.items():
        if name in enum_fields and not isinstance(item, enum_fields[name]):
            item = _enum_from_value(enum_fields[name], item, name)
        if isinstance(item, list):
            field = next(field for field in fields(cls) if field.name == name)
            default = field.default
            if isinstance(default, tuple) or name.endswith(("_values", "_families")):
                item = tuple(item)
        kwargs[name] = item
    return cls(**kwargs)


def _reject_unknown(
    options: Mapping[str, Any], allowed: set[str] | frozenset[str]
) -> None:
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")


def _enum_from_value(enum_type: type[Enum], value: Any, label: str) -> Enum:
    """Accept serialized enum values and their stable member names."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        try:
            return enum_type[str(value).upper()]
        except KeyError as exc:
            raise ValueError(f"invalid {label}: {value!r}") from exc


def _job_operation_directory(job_dir: str | Path | None, operation: str) -> Path | None:
    if job_dir is None:
        return None
    root = Path(job_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = (root / operation).resolve()
    if root not in output.parents:
        raise ValueError("operation export directory must remain inside job_dir")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _result(
    *,
    stage: str,
    feasibility: bool | None,
    metrics: Mapping[str, Any],
    tables: Any,
    series: Any,
    units: Mapping[str, str],
    warnings: list[str] | tuple[str, ...],
    evidence: Any,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    manifest = _artifact_manifest(paths or {})
    plot_paths = {
        key: item["path"]
        for key, item in ((item.get("key", ""), item) for item in manifest)
        if item["path"].lower().endswith((".png", ".svg"))
    }
    return {
        "stage": stage,
        "metrics": dict(metrics),
        "tables": tables,
        "series": series,
        "plots": plot_paths,
        "units": dict(units),
        "feasibility": feasibility,
        "warnings": list(warnings),
        "evidence": evidence,
        "artifacts": manifest,
    }


def _artifact_manifest(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for key, raw_path in paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"exporter did not produce artifact: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        manifest.append(
            {
                "key": str(key),
                "path": str(path),
                "name": path.name,
                "media_type": media_type,
                "size_bytes": path.stat().st_size,
            }
        )
    return manifest


def _system_point_row(point: Any) -> dict[str, Any]:
    op = point.operating_point
    return {
        "vbus_v": op.vbus_v,
        "load_fraction": op.load_fraction,
        "pout_w": op.pout_w,
        "switching_frequency_hz": op.switching_frequency_hz,
        "input_phase_deg": op.input_phase_deg,
        "resonant_current_rms_a": op.resonant_current_rms_a,
        "total_loss_w": point.total_loss_w,
        "efficiency": point.efficiency,
        "zvs_charge_margin": point.primary.zvs_charge_margin,
        "zvs_energy_margin": point.primary.zvs_energy_margin,
    }


def _build_small_signal(
    spec: LLCDesignSpec, options: Mapping[str, Any]
) -> SmallSignalAnalysis:
    kind = options.get("control_input_kind", ControlInputKind.FREQUENCY_HZ)
    kind = _enum_from_value(ControlInputKind, kind, "control_input_kind")
    return build_small_signal_analysis(
        spec,
        vbus_v=options.get("vbus_v"),
        load_fraction=float(options.get("load_fraction", 1.0)),
        sample_time_s=float(options.get("sample_time_s", 20e-6)),
        control_input_kind=kind,
        timer_clock_hz=(
            None
            if options.get("timer_clock_hz") is None
            else float(options["timer_clock_hz"])
        ),
        input_delay_samples=int(options.get("input_delay_samples", 0)),
        series_resistance_ohm=(
            None
            if options.get("series_resistance_ohm") is None
            else float(options["series_resistance_ohm"])
        ),
        trim_frequency_to_output=bool(options.get("trim_frequency_to_output", True)),
    )


def _build_digital_loop(small: SmallSignalAnalysis, options: Mapping[str, Any]):
    sample_time = small.sample_time_s
    controller_value = options.get("controller_config", options.get("controller"))
    if controller_value is None:
        controller: ControllerConfig = PIFControllerConfig(sample_time_s=sample_time)
    elif isinstance(controller_value, str):
        controller_type = controller_value.lower()
        if controller_type == "pi":
            controller = PIControllerConfig(sample_time_s=sample_time)
        elif controller_type == "pif":
            controller = PIFControllerConfig(sample_time_s=sample_time)
        elif controller_type in {"2p2z", "two_p_two_z"}:
            controller = TwoP2ZControllerConfig(sample_time_s=sample_time)
        else:
            raise ValueError(f"unknown controller: {controller_value}")
    else:
        controller_kind_name = str(controller_value.get("kind", "pif")).lower()
        controller_fields = dict(controller_value)
        controller_fields.pop("kind", None)
        controller_cls: type[ControllerConfig]
        controller_cls = {
            "pi": PIControllerConfig,
            "pif": PIFControllerConfig,
            "2p2z": TwoP2ZControllerConfig,
            "two_p_two_z": TwoP2ZControllerConfig,
        }.get(controller_kind_name)
        if controller_cls is None:
            raise ValueError(f"unknown controller: {controller_kind_name}")
        controller_fields.setdefault("sample_time_s", sample_time)
        controller = _dataclass_from_mapping(
            controller_cls, controller_fields, "controller_config"
        )
    if not hasattr(controller, "sample_time_s"):
        raise TypeError("controller_config must be a supported controller dataclass")
    fm = _build_lut(options)
    analog = _dataclass_from_mapping(
        AnalogSenseConfig, options.get("analog_sense", {}), "analog_sense"
    )
    adc_values = dict(_mapping(options.get("adc_sampling"), "adc_sampling"))
    adc_values.setdefault("control_sample_time_s", sample_time)
    adc = _dataclass_from_mapping(ADCSamplingConfig, adc_values, "adc_sampling")
    timing = _dataclass_from_mapping(
        CommandTimingConfig, options.get("command_timing", {}), "command_timing"
    )
    return build_digital_loop_analysis(
        small,
        controller_config=controller,
        fm_lut=fm,
        command_pu=(
            None if options.get("command_pu") is None else float(options["command_pu"])
        ),
        analog_sense=analog,
        adc_sampling=adc,
        command_timing=timing,
        frequencies_hz=options.get("frequencies_hz"),
    )


def _build_lut(options: Mapping[str, Any]) -> FrequencyModulatorLUT:
    value = options.get("fm_lut")
    if value is None:
        lut = FrequencyModulatorLUT.firmware_default()
        timer = options.get("timer_clock_hz")
        count_mode = options.get("count_mode")
        if timer is not None or count_mode is not None:
            lut = FrequencyModulatorLUT(
                lut.pcmd,
                lut.values,
                lut.mode,
                float(timer if timer is not None else lut.timer_clock_hz),
                PWMCountMode(count_mode) if count_mode is not None else lut.count_mode,
                lut.name,
            )
        return lut
    if isinstance(value, str):
        mode = _enum_from_value(
            FMLUTMode,
            options.get("fm_mode", FMLUTMode.PCMD_TO_TBPRD),
            "fm_mode",
        )
        count_mode = _enum_from_value(
            PWMCountMode,
            options.get("count_mode", PWMCountMode.UP_DOWN),
            "count_mode",
        )
        return FrequencyModulatorLUT.from_text(
            value,
            mode=mode,
            timer_clock_hz=float(options.get("timer_clock_hz", 120e6)),
            count_mode=count_mode,
        )
    values = _mapping(value, "fm_lut")
    mode = _enum_from_value(
        FMLUTMode,
        values.pop("mode", options.get("fm_mode", FMLUTMode.PCMD_TO_TBPRD)),
        "fm_lut.mode",
    )
    count_mode = _enum_from_value(
        PWMCountMode,
        values.pop("count_mode", options.get("count_mode", PWMCountMode.UP_DOWN)),
        "fm_lut.count_mode",
    )
    allowed = {"pcmd", "values", "timer_clock_hz", "name"}
    _reject_unknown(values, allowed)
    return FrequencyModulatorLUT(
        values["pcmd"],
        values["values"],
        mode=mode,
        timer_clock_hz=float(
            values.get("timer_clock_hz", options.get("timer_clock_hz", 120e6))
        ),
        count_mode=count_mode,
        name=str(values.get("name", "PCMD-FM-LUT")),
    )


def _waveform_result(
    bundle: WaveformBundle, paths: Mapping[str, Path], evidence: list[str]
) -> dict[str, Any]:
    statistics = {
        key: asdict(signal.statistics) for key, signal in bundle.signals.items()
    }
    return _result(
        stage="waveform",
        feasibility=None,
        metrics={
            "model_name": bundle.model_name,
            "switching_frequency_hz": bundle.switching_frequency_hz,
            "sample_count": len(bundle.time_s),
            "signal_count": len(bundle.signals),
        },
        tables={"statistics": statistics},
        series={
            "time_s": bundle.time_s,
            **{key: signal.values for key, signal in bundle.signals.items()},
        },
        units={
            "time_s": "s",
            **{key: signal.unit for key, signal in bundle.signals.items()},
        },
        warnings=list(bundle.warnings),
        evidence=evidence,
        paths=paths,
    )


def _small_signal_result(
    result: SmallSignalAnalysis, paths: Mapping[str, Path], evidence: list[str]
) -> dict[str, Any]:
    transfer = result.continuous_transfer
    discrete = result.discrete_plant
    return _result(
        stage="analysis",
        feasibility=result.stable,
        metrics={
            "sample_time_s": result.sample_time_s,
            "dc_gain": transfer.dc_gain,
            "continuous_stable": result.continuous_plant.stable,
            "discrete_stable": result.discrete_plant.stable,
            "output_voltage_v": result.steady_state.output_voltage_v,
            "switching_frequency_hz": result.operating_point.switching_frequency_hz,
        },
        tables={
            "operating_point": jsonable(asdict(result.operating_point)),
            "continuous_transfer": {
                "numerator": transfer.numerator,
                "denominator": transfer.denominator,
                "poles": transfer.poles,
                "zeros": transfer.zeros,
            },
            "discrete_transfer": {
                "numerator": discrete.numerator,
                "denominator": discrete.denominator,
                "poles": discrete.poles,
                "zeros": discrete.zeros,
            },
        },
        series={},
        units={
            "sample_time_s": "s",
            "dc_gain": f"{transfer.output_unit}/{transfer.input_unit}",
            "output_voltage_v": "V",
            "switching_frequency_hz": "Hz",
        },
        warnings=[],
        evidence=evidence,
        paths=paths,
    )


def _loop_series(loop: Any) -> dict[str, Any]:
    series: dict[str, Any] = {"frequencies_hz": loop.frequencies_hz}
    for name, response in loop.responses.items():
        series[f"{name}_magnitude"] = abs(response)
        series[f"{name}_phase_deg"] = np.unwrap(np.angle(response)) * 180.0 / np.pi
    return series


def _dataframe_records(frame: pd.DataFrame) -> dict[str, Any]:
    """Return explicit stable DataFrame columns/records for API clients."""
    return {
        "columns": [str(column) for column in frame.columns],
        "records": frame.to_dict(orient="records"),
    }


__all__ = ["SUPPORTED_OPERATIONS", "LLCAdapter", "run_llc_operation"]
