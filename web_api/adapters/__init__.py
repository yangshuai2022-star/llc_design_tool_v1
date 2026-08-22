"""Workspace adapters used by the web API job runner."""

from .ttpl import (
    MAX_FREQUENCY_POINTS,
    MAX_INDUCTOR_CURVE_POINTS,
    MAX_LINE_CYCLE_SAMPLES,
    MAX_SWITCHING_CYCLES,
    MAX_SWITCHING_SAMPLES,
    MAX_SWITCHING_SAMPLES_PER_CYCLE,
    MAX_WAVEFORM_INTEGRATION_RATE_HZ,
    MAX_WAVEFORM_LINE_CYCLES,
    TTPLAdapter,
    TTPLAdapterEnvelope,
    TTPLAdapterError,
    build_inductor_request,
    build_ttpl_config,
    execute_ttpl_operation,
    run_ttpl_operation,
)

__all__ = [
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
