"""Steinmetz core loss equations for various waveforms.

Implements:
- OSE: Original Steinmetz Equation (sinusoidal)
- MSE: Modified Steinmetz Equation (rectangular/duty-cycle)
- iGSE: improved Generalized Steinmetz Equation (arbitrary waveform)
"""

import numpy as np


def steinmetz_ose(f: float, b_peak: float, k: float, alpha: float, beta: float) -> float:
    """Original Steinmetz Equation — sinusoidal excitation.

    Args:
        f: frequency in Hz
        b_peak: peak AC flux density in T (half-swing)
        k: material coefficient
        alpha: frequency exponent
        beta: flux density exponent

    Returns:
        Core loss density in W/m³
    """
    return k * f ** alpha * b_peak ** beta


def steinmetz_mse(f: float, b_peak: float, k: float, alpha: float, beta: float,
                  duty: float = 0.5) -> float:
    """Modified Steinmetz Equation — for non-sinusoidal PWM waveforms.

    Uses equivalent frequency approach. Good for rectangular voltage waveforms
    (e.g., buck/boost inductor with high ripple).

    Args:
        f: switching frequency in Hz
        b_peak: peak AC flux density in T
        k, alpha, beta: material Steinmetz coefficients
        duty: duty cycle of the rectangular waveform (0-1)

    Returns:
        Core loss density in W/m³
    """
    if duty <= 0 or duty >= 1:
        duty = 0.5
    f_eq = 2.0 / (np.pi ** 2 * duty * (1 - duty)) * f
    return k * f_eq ** (alpha - 1) * f * b_peak ** beta


def _igse_ki(k: float, alpha: float, beta: float) -> float:
    """Compute the iGSE normalization constant ki.

    ki = k / (2^(beta-1) * pi^(alpha-1) * integral)
    integral = ∫|cos(θ)|^α dθ from 0 to 2π

    (Expanding the textbook ki = k / [(2π)^(α−1) · 2^(β−α) · ∫|cosθ|^α dθ].)
    """
    n_pts = 200
    theta = np.linspace(0, 2 * np.pi, n_pts)
    integrand = np.abs(np.cos(theta)) ** alpha
    integral = np.trapezoid(integrand, theta)
    denom = 2 ** (beta - 1) * np.pi ** (alpha - 1) * integral
    return k / denom


def steinmetz_igse(f: float, b_peak: float, k: float, alpha: float, beta: float,
                   duty: float = 0.5) -> float:
    """improved Generalized Steinmetz Equation.

    Most accurate for PFC boost inductor where dB/dt varies over the line cycle.
    Accounts for the actual flux waveform shape rather than assuming sinusoid.

    For a triangular waveform with peak-to-peak swing 2*B_peak:

        Pv = ki * 2^beta * f^alpha * B_peak^beta
             * [D^(1-alpha) + (1-D)^(1-alpha)]

    which at D = 0.5 reduces to Pv = ki * 2^(alpha+beta) * f^alpha * B_peak^beta.

    Args:
        f: switching frequency in Hz
        b_peak: peak AC flux density in T
        k, alpha, beta: material Steinmetz coefficients (OSE)
        duty: duty cycle

    Returns:
        Core loss density in W/m³
    """
    if duty <= 0 or duty >= 1:
        duty = 0.5
    ki = _igse_ki(k, alpha, beta)
    shape = duty ** (1 - alpha) + (1 - duty) ** (1 - alpha)
    return ki * 2.0 ** beta * f ** alpha * b_peak ** beta * shape


class SteinmetzMaterial:
    """Material-specific Steinmetz loss model with cached coefficients."""

    def __init__(self, name: str, k: float, alpha: float, beta: float,
                 f_min_khz: float = 10, f_max_khz: float = 500):
        self.name = name
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.f_min_khz = f_min_khz
        self.f_max_khz = f_max_khz
        self._ki = _igse_ki(k, alpha, beta)

    def pv_ose(self, f: float, b_peak: float) -> float:
        """Core loss density (W/m³) using OSE."""
        return steinmetz_ose(f, b_peak, self.k, self.alpha, self.beta)

    def pv_igse(self, f: float, b_peak: float, duty: float = 0.5) -> float:
        """Core loss density (W/m³) using iGSE."""
        return steinmetz_igse(f, b_peak, self.k, self.alpha, self.beta, duty)

    def pv_mse(self, f: float, b_peak: float, duty: float = 0.5) -> float:
        """Core loss density (W/m³) using MSE."""
        return steinmetz_mse(f, b_peak, self.k, self.alpha, self.beta, duty)

    def core_loss(self, f: float, b_peak: float, ve_m3: float,
                  method: str = 'igse', duty: float = 0.5) -> float:
        """Total core loss in Watts.

        Args:
            f: frequency in Hz
            b_peak: peak AC flux density in T
            ve_m3: effective volume in m³
            method: 'ose', 'mse', or 'igse'
            duty: duty cycle (for MSE/iGSE)
        """
        if method == 'ose':
            pv = self.pv_ose(f, b_peak)
        elif method == 'mse':
            pv = self.pv_mse(f, b_peak, duty)
        else:
            pv = self.pv_igse(f, b_peak, duty)
        return pv * ve_m3
