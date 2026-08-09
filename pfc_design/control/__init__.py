"""PFC nested current/voltage-loop modelling and waveform laboratory."""

from .analysis import (
    LoopResult,
    PFCControlLabAnalysis,
    PFCOperatingPoint,
    build_pfc_control_lab_analysis,
)
from .config import (
    ADCTimingConfig,
    ControllerConfig,
    ControllerKind,
    DigitalFilterConfig,
    ExternalSenseConfig,
    LoadModel,
    PFCControlLabConfig,
    PFCFirmwareAlgorithmConfig,
    PFCPowerStageConfig,
    default_current_controller,
    default_current_sense,
    default_vac_sense,
    default_vbus_sense,
    default_voltage_controller,
)
from .sensing import (
    SenseChainSummary,
    SenseFrequencyResponse,
    sense_frequency_response,
    summarize_sense_chain,
)
from .export import export_pfc_control_lab, export_pfc_controller_c99
from .waveforms import (
    PFCLineCycleWaveforms,
    PFCSwitchingWaveforms,
    PFCWaveformMetrics,
    build_pfc_switching_waveforms,
    simulate_pfc_line_cycle,
)

from .autotune import CurrentLoopEnvelopePoint, CurrentLoopTuneResult, tune_pfc_current_loop

__all__ = [
    "ADCTimingConfig",
    "ControllerConfig",
    "ControllerKind",
    "DigitalFilterConfig",
    "ExternalSenseConfig",
    "LoadModel",
    "LoopResult",
    "PFCControlLabAnalysis",
    "PFCControlLabConfig",
    "PFCFirmwareAlgorithmConfig",
    "PFCLineCycleWaveforms",
    "PFCOperatingPoint",
    "PFCPowerStageConfig",
    "PFCSwitchingWaveforms",
    "PFCWaveformMetrics",
    "SenseChainSummary",
    "SenseFrequencyResponse",
    "build_pfc_control_lab_analysis",
    "export_pfc_control_lab",
    "export_pfc_controller_c99",
    "build_pfc_switching_waveforms",
    "default_current_controller",
    "default_current_sense",
    "default_vac_sense",
    "default_vbus_sense",
    "default_voltage_controller",
    "sense_frequency_response",
    "simulate_pfc_line_cycle",
    "summarize_sense_chain",

    "CurrentLoopEnvelopePoint", "CurrentLoopTuneResult", "tune_pfc_current_loop",
]

