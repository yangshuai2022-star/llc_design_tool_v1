from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy import signal


class DiscretizationMethod(str, Enum):
    TUSTIN = "tustin"
    PREWARP_TUSTIN = "prewarp_tustin"
    BACKWARD_EULER = "backward_euler"


class ControllerKind(str, Enum):
    INTEGRATOR = "integrator"
    PI = "pi"
    MODIFIED_PI = "modified_pi"
    LEAD = "lead"
    LAG = "lag"
    ONE_P_ONE_Z = "1p1z"
    TWO_P_TWO_Z = "2p2z"
    THREE_P_THREE_Z = "3p3z"
    GENERAL = "general"


class StabilityClass(str, Enum):
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"


class FilterResponse(str, Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    NOTCH = "notch"


class IIRFamily(str, Enum):
    BUTTERWORTH = "butterworth"
    BESSEL = "bessel"
    CHEBYSHEV1 = "chebyshev1"
    CHEBYSHEV2 = "chebyshev2"
    ELLIPTIC = "elliptic"


@dataclass(frozen=True)
class AnalogTransferFunction:
    numerator: tuple[float, ...]
    denominator: tuple[float, ...]
    name: str = "H(s)"

    def arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.asarray(self.numerator, dtype=float), np.asarray(self.denominator, dtype=float)

    def validate(self) -> None:
        b, a = self.arrays()
        if a.size == 0 or not np.any(np.abs(a) > 0):
            raise ValueError("analog denominator cannot be zero")
        if b.size == 0:
            raise ValueError("analog numerator cannot be empty")
        if len(b) - 1 > len(a) - 1:
            raise ValueError("controller/filter must be proper for causal discretization")


@dataclass(frozen=True)
class DigitalTransferFunction:
    b: tuple[float, ...]
    a: tuple[float, ...]
    sample_rate_hz: float
    name: str = "H(z)"
    source: str = ""

    def normalized(self) -> "DigitalTransferFunction":
        b = np.asarray(self.b, dtype=float)
        a = np.asarray(self.a, dtype=float)
        if a.size == 0 or abs(a[0]) < 1e-30:
            raise ValueError("digital denominator a0 must be non-zero")
        b = b / a[0]
        a = a / a[0]
        return DigitalTransferFunction(tuple(float(v) for v in b), tuple(float(v) for v in a), self.sample_rate_hz, self.name, self.source)

    @property
    def poles(self) -> NDArray[np.complex128]:
        n = self.normalized()
        return np.roots(np.asarray(n.a, dtype=float)).astype(complex)

    @property
    def zeros(self) -> NDArray[np.complex128]:
        n = self.normalized()
        b = np.trim_zeros(np.asarray(n.b, dtype=float), "f")
        return np.roots(b).astype(complex) if b.size > 1 else np.asarray([], dtype=complex)

    @property
    def stability_class(self) -> StabilityClass:
        p = self.poles
        if not p.size:
            return StabilityClass.STABLE
        r = np.abs(p)
        tol = 1e-9
        if np.any(r > 1.0 + tol):
            return StabilityClass.UNSTABLE
        if np.any(r >= 1.0 - tol):
            return StabilityClass.MARGINAL
        return StabilityClass.STABLE

    @property
    def stable(self) -> bool:
        return self.stability_class == StabilityClass.STABLE

    @property
    def implementable(self) -> bool:
        # Integrators / PI controllers intentionally place a pole at z=1.
        # They are marginal as standalone transfer functions, but are valid
        # controller building blocks and must remain exportable.
        return self.stability_class != StabilityClass.UNSTABLE

    @property
    def max_pole_radius(self) -> float:
        p = self.poles
        return float(np.max(np.abs(p))) if p.size else 0.0

    @property
    def sos(self) -> NDArray[np.float64]:
        n = self.normalized()
        return signal.tf2sos(np.asarray(n.b), np.asarray(n.a)).astype(float)


@dataclass(frozen=True)
class FilterDesignResult:
    digital: DigitalTransferFunction
    family: str
    response: str
    order: int
    description: str
