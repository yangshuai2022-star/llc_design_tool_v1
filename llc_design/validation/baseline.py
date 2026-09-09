from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.config import load_spec
from ..models.system import LLCSystemAnalyzer
from .provenance import EvidenceStatus


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    evidence_type: str
    status: EvidenceStatus
    expected: float | bool | None
    actual: float | bool | None
    tolerance: float | None
    passed: bool | None
    source: str
    verified_for_hardware: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class ValidationMatrix:
    baseline: str
    checks: tuple[ValidationCheck, ...]

    @property
    def software_regression_passed(self) -> bool:
        software = [item for item in self.checks if item.evidence_type == "software_regression"]
        return bool(software) and all(item.passed is True for item in software)

    @property
    def hardware_verified(self) -> bool:
        hardware = [item for item in self.checks if item.evidence_type == "hardware_measurement"]
        return bool(hardware) and all(
            item.verified_for_hardware
            and item.status is EvidenceStatus.VERIFIED
            and item.passed is True
            for item in hardware
        )

    @property
    def status(self) -> EvidenceStatus:
        if self.software_regression_passed and self.hardware_verified:
            return EvidenceStatus.VERIFIED
        return EvidenceStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "status": self.status.value,
            "software_regression_passed": self.software_regression_passed,
            "hardware_verified": self.hardware_verified,
            "checks": [item.to_dict() for item in self.checks],
        }


def _numeric_check(check_id: str, actual: float, expected: float,
                   tolerance: float) -> ValidationCheck:
    measured = float(actual)
    return ValidationCheck(
        check_id=check_id,
        evidence_type="software_regression",
        status=EvidenceStatus.VERIFIED,
        expected=expected,
        actual=measured,
        tolerance=tolerance,
        passed=bool(abs(measured - expected) <= tolerance),
        source="Repository-controlled baseline recomputed by the current Python implementation.",
    )


def run_builtin_baseline_validation() -> ValidationMatrix:
    baseline_path = Path(__file__).parent.parent / "examples" / "baseline_400V_53V_3kW.json"
    result = LLCSystemAnalyzer().analyze(load_spec(baseline_path))
    checks = (
        ValidationCheck(
            check_id="system_feasible",
            evidence_type="software_regression",
            status=EvidenceStatus.VERIFIED,
            expected=True,
            actual=result.feasible,
            tolerance=None,
            passed=result.feasible is True,
            source="Repository-controlled feasibility screen recomputed by the current Python implementation.",
        ),
        _numeric_check("tank_lr_h", result.tank.lr_h, 23.7811e-6, 0.03e-6),
        _numeric_check("tank_cr_f", result.tank.cr_f, 106.5145e-9, 0.12e-9),
        _numeric_check("tank_lm_h", result.tank.lm_h, 118.9054e-6, 0.12e-6),
        _numeric_check(
            "nominal_switching_frequency_hz",
            result.nominal.operating_point.switching_frequency_hz,
            99_689.0,
            75.0,
        ),
        _numeric_check("nominal_efficiency", result.nominal.efficiency, 0.9766, 0.005),
        ValidationCheck(
            check_id="plecs_or_ltspice_correlation",
            evidence_type="external_simulation",
            status=EvidenceStatus.UNKNOWN,
            expected=None,
            actual=None,
            tolerance=None,
            passed=None,
            source="No PLECS or LTspice evidence artifact is recorded in the repository.",
        ),
        ValidationCheck(
            check_id="bench_measurement_correlation",
            evidence_type="hardware_measurement",
            status=EvidenceStatus.UNKNOWN,
            expected=None,
            actual=None,
            tolerance=None,
            passed=None,
            source="No traceable bench measurement artifact is recorded in the repository.",
        ),
    )
    return ValidationMatrix("400V_to_53V_3kW", checks)
