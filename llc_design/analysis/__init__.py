"""V8 multi-fidelity LLC steady-state analysis package."""

from .export import export_multifidelity_analysis, write_multifidelity_markdown
from .fha import solve_fha
from .golden import GoldenSolverConfig, LLCGoldenSolver, solve_multifidelity
from .harmonic_balance import (
    HarmonicBalanceConfig,
    HarmonicBalanceConvergenceError,
    HarmonicBalanceNativeSolution,
    HarmonicConvergenceStep,
    MultiHarmonicLLCSolver,
    exact_rectifier_projection,
    extract_peak_phasors,
    find_trigonometric_zero_crossings,
    odd_harmonics,
    solve_harmonic_balance,
    synthesize_real_waveform,
)
from .time_domain import TimeDomainConfig, TimeDomainNativeSolution, solve_time_domain
from .complementarity import ComplementarityTimeDomainConfig, solve_complementarity_time_domain
from .types import (
    FidelityLevel,
    LLCAnalysisRequest,
    LLCModelResult,
    ModelComparisonRow,
    ModelMetrics,
    MultiFidelityAnalysis,
    SolverConvergence,
)

__all__ = [
    "FidelityLevel",
    "GoldenSolverConfig",
    "HarmonicBalanceConfig",
    "HarmonicBalanceConvergenceError",
    "HarmonicBalanceNativeSolution",
    "HarmonicConvergenceStep",
    "LLCAnalysisRequest",
    "LLCGoldenSolver",
    "LLCModelResult",
    "ModelComparisonRow",
    "ModelMetrics",
    "MultiFidelityAnalysis",
    "MultiHarmonicLLCSolver",
    "SolverConvergence",
    "TimeDomainConfig",
    "ComplementarityTimeDomainConfig",
    "TimeDomainNativeSolution",
    "exact_rectifier_projection",
    "export_multifidelity_analysis",
    "extract_peak_phasors",
    "find_trigonometric_zero_crossings",
    "odd_harmonics",
    "solve_fha",
    "solve_harmonic_balance",
    "solve_multifidelity",
    "solve_time_domain",
    "solve_complementarity_time_domain",
    "synthesize_real_waveform",
    "write_multifidelity_markdown",
]
