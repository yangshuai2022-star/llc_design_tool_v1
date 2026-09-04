"""Power Design Toolkit V9 digital closed-loop runtime.

This package contains backend-independent execution primitives used by the
Control Tools / LLC small-signal chain: sampler/ADC behaviour, digital
controller runtime, LLC FM/TBPRD modulation, event scheduling and linear
closed-loop diagnostics.  No external circuit simulator is required.
"""

from .digital_control import (
    LLCFMMode,
    PWMCountMode,
    SamplerConfig,
    SamplerRuntime,
    ControllerLimitConfig,
    DigitalTransferRuntime,
    LLCFMConfig,
    LLCFMRuntime,
    LLCFMStep,
)
from .closed_loop import (
    StepProfile,
    ClosedLoopTiming,
    ClosedLoopScenario,
    ClosedLoopSample,
    ClosedLoopDiagnostics,
    ClosedLoopResult,
    LinearClosedLoopAnalysis,
    FirstOrderLLCPlant,
    analyze_linear_closed_loop,
    llc_small_signal_to_digital_plant,
    run_closed_loop,
)

__all__ = [
    "LLCFMMode", "PWMCountMode", "SamplerConfig", "SamplerRuntime",
    "ControllerLimitConfig", "DigitalTransferRuntime", "LLCFMConfig",
    "LLCFMRuntime", "LLCFMStep", "StepProfile", "ClosedLoopTiming",
    "ClosedLoopScenario", "ClosedLoopSample", "ClosedLoopDiagnostics",
    "ClosedLoopResult", "LinearClosedLoopAnalysis", "FirstOrderLLCPlant",
    "analyze_linear_closed_loop", "llc_small_signal_to_digital_plant",
    "run_closed_loop",
]
