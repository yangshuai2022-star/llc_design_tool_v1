"""Reusable digital-control design utilities for power-electronics firmware."""
from .models import (
    AnalogTransferFunction, DigitalTransferFunction, DiscretizationMethod,
    ControllerKind, FilterResponse, IIRFamily, StabilityClass, FilterDesignResult,
)
from .controllers import design_controller
from .discretize import discretize_transfer_function
from .filters import design_iir_filter, design_fir_filter, design_moving_average, design_dc_blocker
from .analysis import analyze_digital_filter, ControlResponseAnalysis
from .codegen import export_c99_filter, verify_c99_filter, render_c99_single_file, C99Verification

__all__ = [
    "AnalogTransferFunction", "DigitalTransferFunction", "DiscretizationMethod",
    "ControllerKind", "FilterResponse", "IIRFamily", "StabilityClass", "FilterDesignResult",
    "design_controller", "discretize_transfer_function", "design_iir_filter",
    "design_fir_filter", "design_moving_average", "design_dc_blocker",
    "analyze_digital_filter", "ControlResponseAnalysis", "export_c99_filter", "verify_c99_filter", "render_c99_single_file", "C99Verification",
]
