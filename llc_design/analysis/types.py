"""Common V8 analysis contracts for multi-fidelity LLC solvers.

The existing V7 modules remain available.  This module adds a stable adapter
layer so FHA, multi-harmonic balance and switched time-domain solvers expose
one result shape to the GUI, CLI, validation and future multiphase engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..core.spec import LLCDesignSpec
from ..dynamics.waveforms import WaveformBundle


class FidelityLevel(str, Enum):
    """Available steady-state model fidelities in V8.1."""

    FHA = "fha"
    HARMONIC_BALANCE = "harmonic_balance"
    SWITCHED_TIME_DOMAIN = "switched_time_domain"


@dataclass(frozen=True)
class LLCAnalysisRequest:
    """One LLC electrical work-point request.

    ``load_fraction`` always refers to rated output power at the requested
    target output voltage.  When ``frequency_hz`` is omitted and
    ``regulate_output`` is true, each solver is allowed to find its own
    switching frequency that regulates to ``target_output_voltage_v``.
    """

    spec: LLCDesignSpec
    vbus_v: float | None = None
    load_fraction: float = 1.0
    frequency_hz: float | None = None
    regulate_output: bool = True
    target_output_voltage_v: float | None = None
    waveform_cycles: int = 2
    samples_per_cycle: int = 2048

    def validate(self) -> None:
        self.spec.validate()
        if self.bus_voltage_v <= 0.0:
            raise ValueError("bus voltage must be positive")
        if not (0.0 < self.load_fraction <= 1.5):
            raise ValueError("load fraction must be within 0..1.5")
        if self.frequency_hz is not None and self.frequency_hz <= 0.0:
            raise ValueError("switching frequency must be positive")
        if self.output_voltage_target_v <= 0.0:
            raise ValueError("target output voltage must be positive")
        if self.waveform_cycles < 1:
            raise ValueError("waveform_cycles must be >= 1")
        if self.samples_per_cycle < 128:
            raise ValueError("samples_per_cycle must be >= 128")

    @property
    def bus_voltage_v(self) -> float:
        return self.spec.vbus_nom_v if self.vbus_v is None else float(self.vbus_v)

    @property
    def output_voltage_target_v(self) -> float:
        return (
            self.spec.vout_v
            if self.target_output_voltage_v is None
            else float(self.target_output_voltage_v)
        )

    @property
    def requested_output_power_w(self) -> float:
        return self.spec.pout_w * self.load_fraction

    @property
    def load_resistance_ohm(self) -> float:
        return self.output_voltage_target_v**2 / self.requested_output_power_w


@dataclass(frozen=True)
class SolverConvergence:
    """Numerical convergence status shared by all V8 solvers."""

    converged: bool
    residual_norm: float
    iterations: int
    method: str
    message: str = ""


@dataclass(frozen=True)
class ModelMetrics:
    """Comparable electrical measurements extracted from one model result."""

    switching_frequency_hz: float
    output_voltage_v: float
    output_power_w: float
    normalized_gain: float
    input_phase_deg: float
    input_power_w: float
    resonant_current_rms_a: float
    resonant_current_peak_a: float
    magnetizing_current_rms_a: float
    magnetizing_current_peak_a: float
    primary_load_current_rms_a: float
    primary_load_current_peak_a: float
    secondary_current_rms_a: float
    secondary_current_peak_a: float
    rectifier_current_average_a: float
    resonant_capacitor_rms_v: float
    resonant_capacitor_peak_v: float
    output_capacitor_current_rms_a: float
    power_balance_error_w: float
    power_balance_error_percent: float

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class LLCModelResult:
    """Uniform output of one V8 LLC steady-state solver."""

    fidelity: FidelityLevel
    request: LLCAnalysisRequest
    waveform: WaveformBundle
    metrics: ModelMetrics
    convergence: SolverConvergence
    harmonic_orders: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    native_result: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ModelComparisonRow:
    """One solver row plus errors relative to the selected reference model."""

    fidelity: FidelityLevel
    metrics: ModelMetrics
    converged: bool
    residual_norm: float
    frequency_error_percent: float
    gain_error_percent: float
    resonant_rms_error_percent: float
    resonant_peak_error_percent: float
    magnetizing_rms_error_percent: float
    secondary_rms_error_percent: float
    resonant_cap_peak_error_percent: float


@dataclass(frozen=True)
class MultiFidelityAnalysis:
    """FHA/HB/TD work-point solution and numerical comparison."""

    request: LLCAnalysisRequest
    results: Mapping[FidelityLevel, LLCModelResult]
    reference_level: FidelityLevel
    comparison_rows: tuple[ModelComparisonRow, ...]
    warnings: tuple[str, ...] = ()

    @property
    def reference(self) -> LLCModelResult:
        return self.results[self.reference_level]

    def result(self, level: FidelityLevel) -> LLCModelResult:
        try:
            return self.results[level]
        except KeyError as exc:
            raise KeyError(f"analysis did not include {level.value}") from exc
