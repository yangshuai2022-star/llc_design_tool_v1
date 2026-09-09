from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLCSpecPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vbus_nom_v: float | None = None
    vbus_min_normal_v: float | None = None
    vbus_max_v: float | None = None
    vbus_hold_end_v: float | None = None
    vout_v: float | None = None
    pout_w: float | None = None
    efficiency_assumption: float | None = None
    resonant_frequency_hz: float | None = None
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None
    ln_ratio: float | None = None
    q_full_load: float | None = None
    user_lr_h: float | None = None
    user_cr_f: float | None = None
    user_lm_h: float | None = None
    primary_turns: int | None = None
    secondary_turns: int | None = None
    rectifier_equivalent_drop_v: float | None = None
    bus_capacitance_f: float | None = None
    requested_hold_time_s: float | None = None
    output_capacitance_f: float | None = None
    output_cap_esr_ohm: float | None = None
    primary_deadtime_s: float | None = None
    primary_zvs_margin_required: float | None = None
    primary_parallel_devices: int | None = None
    sr_parallel_devices_per_position: int | None = None
    ambient_temperature_c: float | None = None
    primary_junction_temperature_c: float | None = None
    sr_junction_temperature_c: float | None = None
    winding_temperature_c: float | None = None
    primary_device: str | None = None
    sr_device: str | None = None
    primary_topology: Literal["HALF_BRIDGE", "FULL_BRIDGE"] | None = None
    parameter_mode: Literal["AUTO_DESIGN", "USER_DEFINED"] | None = None

    def service_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class LLCAnalyzeRequest(BaseModel):
    spec: LLCSpecPayload = Field(default_factory=LLCSpecPayload)


class ReportRequest(LLCAnalyzeRequest):
    project: str = ""
    engineer: str = ""


class OptimizeRequest(LLCAnalyzeRequest):
    sweep: dict[str, Any] = Field(default_factory=dict)
    weights: dict[str, Any] = Field(default_factory=dict)

    def service_payload(self) -> dict[str, Any]:
        return {
            "spec": self.spec.service_payload(),
            "sweep": self.sweep,
            "weights": self.weights,
        }
