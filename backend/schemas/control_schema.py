"""Pydantic schemas for the Control Loop Designer API.

The browser only ever receives these typed JSON shapes; it never touches
llc_design kernel objects directly.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

ComplexPair = tuple[float, float]


class ManualPlantParams(BaseModel):
    """User-supplied tank parameters for the control plant (FM, LLC)."""

    lr_h: float = Field(gt=0.0, description="Resonant inductance / H")
    cr_f: float = Field(gt=0.0, description="Resonant capacitance / F")
    lm_h: float = Field(gt=0.0, description="Magnetizing inductance / H")
    turns_ratio: float = Field(gt=0.0, description="Transformer primary:secondary ratio")
    vbus_v: float = Field(gt=0.0, description="Operating bus voltage / V")
    vout_v: float = Field(gt=0.0, description="Output voltage / V")
    pout_w: float = Field(gt=0.0, description="Output power / W")
    primary_turns: int | None = Field(default=None, ge=1, description="Primary turns (derived from ratio when omitted)")
    secondary_turns: int | None = Field(default=None, ge=1, description="Secondary turns (default 4)")
    primary_topology: str = "FULL_BRIDGE"


class PlantContext(BaseModel):
    """How to build the LLC plant for a control request."""

    spec: dict[str, Any] | None = Field(default=None, description="LLC design spec payload (same shape as /api/llc/analyze)")
    plant: ManualPlantParams | None = None
    vbus_v: float | None = Field(default=None, description="Operating bus override; defaults to spec vbus_nom")
    load_fraction: float = Field(default=1.0, ge=0.05, le=1.5)
    sample_time_s: float = Field(default=20e-6, gt=0.0)


class OperatingPointSchema(BaseModel):
    label: str
    vbus_v: float
    load_fraction: float
    pout_w: float
    switching_frequency_hz: float
    required_gain: float
    achieved_gain: float
    input_phase_deg: float


class TransferFunctionSchema(BaseModel):
    input_name: str
    input_unit: str
    output_name: str
    output_unit: str
    dc_gain: float
    numerator: list[float]
    denominator: list[float]
    poles: list[ComplexPair]
    zeros: list[ComplexPair]


class PlantModelResponse(BaseModel):
    mode: str
    control_input: str
    control_output: str
    model_name: str
    operating_point: OperatingPointSchema
    continuous: TransferFunctionSchema
    discrete_numerator: list[float]
    discrete_denominator: list[float]
    sample_time_s: float
    fm_gain_hz_per_pu: float
    warnings: list[str]


class ControllerConfigSchema(BaseModel):
    """Controller either given directly (kernel discrete forms) or auto-designed."""

    kind: str = Field(description="pi | 2p2z | designed")
    type: str | None = Field(default=None, description="designed only: pi | type_ii | type_iii")
    kp: float | None = None
    ti_s: float | None = None
    b0: float | None = None
    b1: float | None = None
    b2: float | None = None
    a1: float | None = None
    a2: float | None = None
    target_bandwidth_hz: float | None = Field(default=None, gt=0.0, description="designed only")
    target_phase_margin_deg: float | None = Field(default=None, gt=0.0, lt=90.0, description="designed only")
    output_min: float = 0.0
    output_max: float = 1.0


class LoopGainRequest(BaseModel):
    plant_context: PlantContext
    controller: ControllerConfigSchema
    frequencies_hz: list[float] | None = None


class StabilityMarginsSchema(BaseModel):
    crossover_hz: float | None = None
    phase_margin_deg: float | None = None
    gain_margin_db: float | None = None
    delay_margin_s: float | None = None
    stable: bool = False


class LoopGainResponse(BaseModel):
    margins: StabilityMarginsSchema
    margins_min_delay: StabilityMarginsSchema
    margins_max_delay: StabilityMarginsSchema
    controller_kind: str
    warnings: list[str]


class BodeResponse(BaseModel):
    frequencies_hz: list[float]
    responses: dict[str, dict[str, list[float]]] = Field(description="magnitude_db / phase_deg per named response")
    annotations: dict[str, float | None]


class CompensatorRequest(BaseModel):
    plant_context: PlantContext
    type: str = Field(description="pi | type_ii | type_iii")
    target_bandwidth_hz: float = Field(gt=0.0)
    target_phase_margin_deg: float = Field(gt=0.0, lt=90.0)


class CompensatorResponse(BaseModel):
    type: str
    continuous: dict[str, Any] = Field(description="Kp / Ki / zeros_hz / poles_hz of the analog form")
    discrete: dict[str, float] = Field(description="b0 b1 b2 a1 a2")
    sample_time_s: float
    achieved: StabilityMarginsSchema
    iterations: int
    c_code: str | None = None
    warnings: list[str]


class DigitalControllerRequest(BaseModel):
    plant_context: PlantContext
    controller: ControllerConfigSchema


class DigitalControllerResponse(BaseModel):
    kind: str
    coefficients: dict[str, float]
    difference_equation: str
    sample_time_s: float
    c_code: str
    warnings: list[str]


class ProtectionThresholds(BaseModel):
    ovp_v: float
    uvp_v: float
    ocp_a: float
    freq_min_hz: float
    freq_max_hz: float
    soft_start_start_hz: float
    soft_start_steps: int
    light_load_fraction: float
    burst_fraction: float


class ProtectionStateSchema(BaseModel):
    name: str
    description: str
    actions: list[str]


class ProtectionTransitionSchema(BaseModel):
    source: str
    target: str
    condition: str


class ProtectionResponse(BaseModel):
    thresholds: ProtectionThresholds
    states: list[ProtectionStateSchema]
    transitions: list[ProtectionTransitionSchema]
    note: str
