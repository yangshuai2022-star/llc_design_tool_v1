"""V8 multi-fidelity orchestration for one LLC steady-state work point.

The orchestrator deliberately keeps the three solver layers independent:

* FHA is the fast design model and backward-compatible baseline.
* Multi-harmonic balance is the nonlinear frequency-domain verification layer.
* The switched periodic solver is the current ideal-topology Golden reference.

A failed high-fidelity solve does not erase lower-fidelity engineering results.
The highest *converged* available result is selected as the comparison reference,
unless strict mode is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .fha import solve_fha
from .harmonic_balance import (
    HarmonicBalanceConfig,
    HarmonicBalanceConvergenceError,
    solve_harmonic_balance,
)
from .metrics import relative_error_percent
from .time_domain import TimeDomainConfig, solve_time_domain
from .types import (
    FidelityLevel,
    LLCAnalysisRequest,
    LLCModelResult,
    ModelComparisonRow,
    MultiFidelityAnalysis,
)
from ..core.tank import TankDesign, design_tank
from ..models.system import LLCSystemAnalyzer, SystemAnalysis


@dataclass(frozen=True)
class GoldenSolverConfig:
    """Configuration for FHA/HB/switched-model orchestration."""

    include_fha: bool = True
    include_harmonic_balance: bool = True
    include_time_domain: bool = True
    harmonic_balance: HarmonicBalanceConfig = field(
        default_factory=HarmonicBalanceConfig)
    time_domain: TimeDomainConfig = field(default_factory=TimeDomainConfig)
    derive_time_domain_damping_from_system_loss: bool = False
    strict: bool = False

    def validate(self) -> None:
        if not (
            self.include_fha
            or self.include_harmonic_balance
            or self.include_time_domain
        ):
            raise ValueError("at least one LLC fidelity level must be enabled")
        self.harmonic_balance.validate()
        self.time_domain.validate()


class LLCGoldenSolver:
    """Solve and compare the enabled V8 steady-state model fidelities."""

    def __init__(self, config: GoldenSolverConfig | None = None):
        self.config = config or GoldenSolverConfig()
        self.config.validate()

    @staticmethod
    def _reference_level(
        results: Mapping[FidelityLevel, LLCModelResult],
    ) -> FidelityLevel:
        priority = (
            FidelityLevel.SWITCHED_TIME_DOMAIN,
            FidelityLevel.HARMONIC_BALANCE,
            FidelityLevel.FHA,
        )
        for level in priority:
            result = results.get(level)
            if result is not None and result.convergence.converged:
                return level
        # A non-converged switched orbit can still carry useful traces, but it
        # must never silently outrank a converged lower-fidelity result.
        for level in priority:
            if level in results:
                return level
        raise RuntimeError("multi-fidelity analysis produced no model result")

    @staticmethod
    def _comparison_rows(
        results: Mapping[FidelityLevel, LLCModelResult],
        reference_level: FidelityLevel,
    ) -> tuple[ModelComparisonRow, ...]:
        reference = results[reference_level].metrics
        order = (
            FidelityLevel.FHA,
            FidelityLevel.HARMONIC_BALANCE,
            FidelityLevel.SWITCHED_TIME_DOMAIN,
        )
        rows: list[ModelComparisonRow] = []
        for level in order:
            result = results.get(level)
            if result is None:
                continue
            metrics = result.metrics
            rows.append(ModelComparisonRow(
                fidelity=level,
                metrics=metrics,
                converged=result.convergence.converged,
                residual_norm=result.convergence.residual_norm,
                frequency_error_percent=relative_error_percent(
                    metrics.switching_frequency_hz,
                    reference.switching_frequency_hz,
                ),
                gain_error_percent=relative_error_percent(
                    metrics.normalized_gain,
                    reference.normalized_gain,
                ),
                resonant_rms_error_percent=relative_error_percent(
                    metrics.resonant_current_rms_a,
                    reference.resonant_current_rms_a,
                ),
                resonant_peak_error_percent=relative_error_percent(
                    metrics.resonant_current_peak_a,
                    reference.resonant_current_peak_a,
                ),
                magnetizing_rms_error_percent=relative_error_percent(
                    metrics.magnetizing_current_rms_a,
                    reference.magnetizing_current_rms_a,
                ),
                secondary_rms_error_percent=relative_error_percent(
                    metrics.secondary_current_rms_a,
                    reference.secondary_current_rms_a,
                ),
                resonant_cap_peak_error_percent=relative_error_percent(
                    metrics.resonant_capacitor_peak_v,
                    reference.resonant_capacitor_peak_v,
                ),
            ))
        return tuple(rows)

    def _build_system_analysis(
        self,
        request: LLCAnalysisRequest,
    ) -> SystemAnalysis:
        """Build the established system-loss model for TD damping calibration."""

        # Use the normal system work-point set so magnetic/device selection and
        # equivalent damping remain identical to the V7 regression path.
        return LLCSystemAnalyzer().analyze(request.spec)

    def solve(
        self,
        request: LLCAnalysisRequest,
        *,
        tank: TankDesign | None = None,
        system_analysis: SystemAnalysis | None = None,
    ) -> MultiFidelityAnalysis:
        """Run enabled solvers and compare them against the best reference."""

        request.validate()
        cfg = self.config
        shared_tank = tank or design_tank(request.spec)
        results: dict[FidelityLevel, LLCModelResult] = {}
        warnings: list[str] = []

        if cfg.include_fha:
            try:
                results[FidelityLevel.FHA] = solve_fha(request)
            except Exception as exc:
                if cfg.strict:
                    raise
                warnings.append(f"FHA solve failed: {exc}")

        if cfg.include_harmonic_balance:
            try:
                results[FidelityLevel.HARMONIC_BALANCE] = solve_harmonic_balance(
                    request,
                    cfg.harmonic_balance,
                )
            except (HarmonicBalanceConvergenceError, ValueError, RuntimeError) as exc:
                if cfg.strict:
                    raise
                warnings.append(f"multi-harmonic HB solve failed: {exc}")

        td_system = system_analysis
        if (
            cfg.include_time_domain
            and td_system is None
            and cfg.derive_time_domain_damping_from_system_loss
            and cfg.time_domain.series_resistance_ohm is None
        ):
            try:
                td_system = self._build_system_analysis(request)
            except Exception as exc:
                if cfg.strict:
                    raise
                warnings.append(
                    "system-loss damping calibration failed; TD uses the "
                    f"documented reference damping instead: {exc}"
                )

        if cfg.include_time_domain:
            try:
                results[FidelityLevel.SWITCHED_TIME_DOMAIN] = solve_time_domain(
                    request,
                    cfg.time_domain,
                    system_analysis=td_system,
                    tank=shared_tank,
                )
            except (ValueError, RuntimeError) as exc:
                if cfg.strict:
                    raise
                warnings.append(f"switched time-domain solve failed: {exc}")

        reference_level = self._reference_level(results)
        if (
            cfg.include_time_domain
            and reference_level is not FidelityLevel.SWITCHED_TIME_DOMAIN
        ):
            warnings.append(
                "The switched periodic result was unavailable or non-converged; "
                f"{reference_level.value} is used as the comparison reference."
            )
        for level, result in results.items():
            if not result.convergence.converged:
                warnings.append(
                    f"{level.value} returned a non-converged result "
                    f"(residual={result.convergence.residual_norm:.3e})."
                )

        return MultiFidelityAnalysis(
            request=request,
            results=results,
            reference_level=reference_level,
            comparison_rows=self._comparison_rows(results, reference_level),
            warnings=tuple(dict.fromkeys(warnings)),
        )


def solve_multifidelity(
    request: LLCAnalysisRequest,
    config: GoldenSolverConfig | None = None,
) -> MultiFidelityAnalysis:
    """Convenience entry point for the V8 FHA/HB/TD comparison."""

    return LLCGoldenSolver(config).solve(request)
