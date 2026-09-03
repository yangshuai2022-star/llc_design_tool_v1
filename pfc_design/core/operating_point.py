"""Operating point calculations for two-phase interleaved PFC.

Computes all time-dependent and averaged operating-point values:
currents, voltages, duty cycles over a line half-cycle.
"""

import numpy as np

from .spec import DesignSpec

# Number of time points for line-cycle integration
N_PTS = 200


class OperatingPoint:
    """Computed operating point for a PFC design at a given input voltage."""

    def __init__(self, spec: DesignSpec, vin_rms: float | None = None):
        self.spec = spec
        self.vin_rms = vin_rms if vin_rms is not None else spec.vin_min

        # Derived parameters
        vm = np.sqrt(2) * self.vin_rms  # peak line voltage
        self.vm = vm
        self.pout_per_phase = spec.pout_per_phase

        # Time array over half line cycle (0 to pi)
        self.omega = 2 * np.pi * spec.f_line
        self.t = np.linspace(0, 0.5 / spec.f_line, N_PTS)
        theta = self.omega * self.t  # 0 to pi

        # Instantaneous input voltage (rectified sine)
        self.vac_t = vm * np.abs(np.sin(theta))

        # Duty cycle: D(t) = 1 - vac(t) / Vout (for boost)
        self.duty_t = 1.0 - self.vac_t / spec.vout
        # Clamp to valid range
        self.duty_t = np.clip(self.duty_t, 0.02, 0.98)

        # Input current
        pin_per_phase = spec.pout_per_phase / spec.eta_target
        self.iin_rms = pin_per_phase / self.vin_rms
        self.iin_peak = np.sqrt(2) * self.iin_rms
        self.iin_t = self.iin_peak * np.abs(np.sin(theta))  # instantaneous

        # Output DC current per phase
        self.iout = spec.pout_per_phase / spec.vout

        # Ripple current (peak-to-peak)
        # delta_I(t) = vac(t) * (vout - vac(t)) / (L * fsw * vout)
        # We'll compute this once L is known; here just store the function
        self._delta_i_t = None  # set by inductor designer

    @property
    def d_min(self) -> float:
        """Minimum duty cycle (at line peak)."""
        return float(1.0 - self.vm / self.spec.vout)

    @property
    def d_max(self) -> float:
        """Maximum duty cycle (at zero crossing)."""
        return 0.98

    def rms(self, waveform: np.ndarray, duty_mod: np.ndarray | None = None) -> float:
        """Compute RMS of a waveform over half line cycle.

        If duty_mod is provided, the waveform is duty-cycle-modulated
        (e.g. transistor current = iin_t * sqrt(duty_t)).
        """
        if duty_mod is not None:
            integrand = waveform**2 * duty_mod
        else:
            integrand = waveform**2
        return float(np.sqrt(np.trapezoid(integrand, self.t) / (0.5 / self.spec.f_line)))

    def avg(self, waveform: np.ndarray, duty_mod: np.ndarray | None = None) -> float:
        """Compute average of a waveform over half line cycle."""
        if duty_mod is not None:
            integrand = waveform * duty_mod
        else:
            integrand = waveform
        return float(np.trapezoid(integrand, self.t) / (0.5 / self.spec.f_line))


def compute_mathcad_operating_point(spec: DesignSpec) -> OperatingPoint:
    """Create operating point matching the Mathcad reference conditions (Vin=176V)."""
    return OperatingPoint(spec, vin_rms=spec.vin_min)
