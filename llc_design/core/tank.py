"""Fundamental-harmonic approximation (FHA) LLC resonant tank model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import brentq

from .spec import LLCDesignSpec


SQRT2 = math.sqrt(2.0)
FHA_RECTIFIER_K = 2.0 * SQRT2 / math.pi


@dataclass(frozen=True)
class TankDesign:
    lr_h: float
    cr_f: float
    lm_h: float
    rac_nom_ohm: float
    zr_ohm: float
    ln_ratio: float
    q_full_load: float
    fr_hz: float

    @property
    def fm_hz(self) -> float:
        return self.fr_hz / math.sqrt(1.0 + self.ln_ratio)


@dataclass(frozen=True)
class TankState:
    frequency_hz: float
    rac_ohm: float
    z_series_ohm: complex
    z_parallel_ohm: complex
    z_input_ohm: complex
    transfer: complex

    @property
    def gain(self) -> float:
        return abs(self.transfer)

    @property
    def input_phase_deg(self) -> float:
        return math.degrees(math.atan2(self.z_input_ohm.imag, self.z_input_ohm.real))


@dataclass(frozen=True)
class FrequencySolution:
    frequency_hz: float
    target_gain: float
    achieved_gain: float
    phase_deg: float
    roots_hz: tuple[float, ...]
    branch: str


class GainNotReachableError(RuntimeError):
    pass


def equivalent_ac_load_ohm(turns_ratio: float, vout_v: float, pout_w: float) -> float:
    """FHA full-wave rectifier load reflected to transformer primary."""
    if pout_w <= 0:
        raise ValueError("pout_w must be positive")
    r_load = vout_v * vout_v / pout_w
    return (8.0 / math.pi**2) * turns_ratio**2 * r_load


def design_tank(spec: LLCDesignSpec) -> TankDesign:
    """Synthesize Lr/Cr/Lm from fr, full-load Q and Ln."""
    spec.validate()
    rac = equivalent_ac_load_ohm(spec.turns_ratio, spec.vout_v, spec.pout_w)
    zr = spec.q_full_load * rac
    omega_r = 2.0 * math.pi * spec.resonant_frequency_hz
    lr = zr / omega_r
    cr = 1.0 / (omega_r * zr)
    lm = spec.ln_ratio * lr
    return TankDesign(lr_h=lr, cr_f=cr, lm_h=lm, rac_nom_ohm=rac,
                      zr_ohm=zr, ln_ratio=spec.ln_ratio,
                      q_full_load=spec.q_full_load,
                      fr_hz=spec.resonant_frequency_hz)


def tank_state(tank: TankDesign, frequency_hz: float, rac_ohm: float) -> TankState:
    if frequency_hz <= 0 or rac_ohm <= 0:
        raise ValueError("frequency and Rac must be positive")
    w = 2.0 * math.pi * frequency_hz
    z_series = 1j * w * tank.lr_h + 1.0 / (1j * w * tank.cr_f)
    z_lm = 1j * w * tank.lm_h
    z_parallel = 1.0 / (1.0 / rac_ohm + 1.0 / z_lm)
    z_input = z_series + z_parallel
    transfer = z_parallel / z_input
    return TankState(frequency_hz, rac_ohm, z_series, z_parallel, z_input, transfer)


def gain(tank: TankDesign, frequency_hz: float, rac_ohm: float) -> float:
    return tank_state(tank, frequency_hz, rac_ohm).gain


def gain_vector(tank: TankDesign, frequencies_hz: np.ndarray,
                rac_ohm: float) -> np.ndarray:
    """Vectorized |transfer| over a frequency array (same model as gain())."""
    w = 2.0 * math.pi * frequencies_hz
    z_series = 1j * w * tank.lr_h + 1.0 / (1j * w * tank.cr_f)
    z_lm = 1j * w * tank.lm_h
    z_parallel = 1.0 / (1.0 / rac_ohm + 1.0 / z_lm)
    z_input = z_series + z_parallel
    return np.abs(z_parallel / z_input)


def target_gain(spec: LLCDesignSpec, vbus_v: float) -> float:
    return (spec.turns_ratio * (spec.vout_v + spec.rectifier_equivalent_drop_v)
            / (spec.bridge_gain * vbus_v))


def bridge_fundamental_rms_v(spec: LLCDesignSpec, vbus_v: float) -> float:
    """RMS value of the bridge square-wave fundamental."""
    return FHA_RECTIFIER_K * spec.bridge_gain * vbus_v


def find_gain_roots(tank: TankDesign, rac_ohm: float, target: float,
                    fmin_hz: float, fmax_hz: float,
                    samples: int = 1600) -> list[float]:
    if target <= 0:
        raise ValueError("target gain must be positive")
    freqs = np.geomspace(fmin_hz, fmax_hz, samples)
    values = gain_vector(tank, freqs, rac_ohm) - target
    roots: list[float] = []
    for idx in range(len(freqs) - 1):
        f0, f1 = float(freqs[idx]), float(freqs[idx + 1])
        y0, y1 = float(values[idx]), float(values[idx + 1])
        if abs(y0) < 1e-10:
            roots.append(f0)
        if y0 * y1 < 0.0:
            root = brentq(lambda f: gain(tank, f, rac_ohm) - target, f0, f1,
                          xtol=1e-8, rtol=1e-11, maxiter=100)
            roots.append(float(root))
    if abs(values[-1]) < 1e-10:
        roots.append(float(freqs[-1]))
    # Deduplicate adjacent roots from grid-point coincidences.
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > max(0.1, 1e-7 * root):
            unique.append(root)
    return unique


def solve_frequency(tank: TankDesign, spec: LLCDesignSpec, rac_ohm: float,
                    required_gain: float | None = None) -> FrequencySolution:
    target = required_gain if required_gain is not None else target_gain(spec, spec.vbus_nom_v)
    roots = find_gain_roots(tank, rac_ohm, target,
                            spec.minimum_frequency_hz,
                            spec.maximum_frequency_hz)
    if not roots:
        frequencies = np.geomspace(spec.minimum_frequency_hz,
                                   spec.maximum_frequency_hz, 2000)
        gains = gain_vector(tank, frequencies, rac_ohm)
        raise GainNotReachableError(
            f"required gain {target:.4f} is outside available range "
            f"{gains.min():.4f}..{gains.max():.4f}")

    inductive = [f for f in roots
                 if tank_state(tank, f, rac_ohm).input_phase_deg
                 >= spec.minimum_inductive_angle_deg]
    candidates = inductive or roots

    # Preferred LLC operating branch: above fr for M<=1; closest root below fr
    # for boost operation. If constraints remove that branch, choose closest to fr.
    if target <= 1.0:
        preferred = [f for f in candidates if f >= tank.fr_hz * (1.0 - 1e-8)]
        chosen = min(preferred) if preferred else min(candidates, key=lambda f: abs(f - tank.fr_hz))
        branch = "above_resonance" if chosen >= tank.fr_hz else "below_resonance_fallback"
    else:
        preferred = [f for f in candidates if f <= tank.fr_hz * (1.0 + 1e-8)]
        chosen = max(preferred) if preferred else min(candidates, key=lambda f: abs(f - tank.fr_hz))
        branch = "boost_inductive" if chosen <= tank.fr_hz else "high_frequency_fallback"

    state = tank_state(tank, chosen, rac_ohm)
    return FrequencySolution(chosen, target, state.gain, state.input_phase_deg,
                             tuple(roots), branch)


def gain_range(tank: TankDesign, rac_ohm: float, frequencies_hz: Iterable[float]) -> tuple[float, float]:
    values = [gain(tank, float(f), rac_ohm) for f in frequencies_hz]
    return min(values), max(values)
