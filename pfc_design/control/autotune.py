"""Conservative PFC current-loop auto tuning.

The tuner is deliberately based on the *same* digital controller, sensing, ZOH
and delay model used by the PFC Bode workbench.  It does not use a generic
continuous-time PI rule and then hope that the sampled implementation remains
stable.

The recommended result is intended as a safe engineering starting point.  The
user can subsequently move bandwidth/phase-margin targets and re-tune.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from llc_design.control.digital_loop import PIControllerConfig, StabilityMargins

from .analysis import _current_loop_at_indu_comp
from .config import PFCControlLabConfig
from .sensing import sense_frequency_response


@dataclass(frozen=True)
class CurrentLoopEnvelopePoint:
    vin_rms_v: float
    load_ratio: float
    line_angle_deg: float
    current_reference_a: float
    indu_comp: float
    crossover_hz: float | None
    phase_margin_deg: float | None
    gain_margin_db: float | None


@dataclass(frozen=True)
class CurrentLoopTuneResult:
    controller: PIControllerConfig
    target_crossover_hz: float
    nominal_crossover_hz: float | None
    nominal_phase_margin_deg: float | None
    nominal_gain_margin_db: float | None
    worst_phase_margin_deg: float | None
    worst_gain_margin_db: float | None
    worst_point: CurrentLoopEnvelopePoint | None
    envelope: tuple[CurrentLoopEnvelopePoint, ...]
    accepted: bool
    message: str


def _interp_db(frequencies: np.ndarray, response: np.ndarray, frequency_hz: float) -> float:
    mag_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
    return float(np.interp(
        math.log10(frequency_hz),
        np.log10(frequencies),
        mag_db,
    ))


def _gm_value(margin: StabilityMargins) -> float:
    return float("inf") if margin.gain_margin_db is None else float(margin.gain_margin_db)


def _pm_value(margin: StabilityMargins) -> float:
    return -1.0e9 if margin.phase_margin_deg is None else float(margin.phase_margin_deg)


def _operating_envelope(
    config: PFCControlLabConfig,
    frequencies: np.ndarray,
    current_sense,
) -> tuple[CurrentLoopEnvelopePoint, ...]:
    """Evaluate the tuned return ratio over line/load/phase gain scheduling.

    The averaged Boost current plant with duty feed-forward is approximately
    independent of Vac.  Line/load/phase mainly move the firmware ``indu_comp``
    gain.  We still report the physical operating conditions so the result is
    useful to an engineer rather than exposing only an abstract gain sweep.
    """
    stage = config.power_stage
    fw = config.firmware
    vin_candidates = tuple(max(stage.vin_rms_v * scale, 20.0) for scale in (0.80, 1.00, 1.15))
    load_ratios = (0.25, 0.50, 1.00)
    angles = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    points: list[CurrentLoopEnvelopePoint] = []
    margin_cache: dict[float, StabilityMargins] = {}

    for vin in vin_candidates:
        vpk = math.sqrt(2.0) * vin
        for load in load_ratios:
            pout = stage.output_power_w * load
            conductance = pout / max(stage.efficiency * vin * vin, 1e-12)
            for angle in angles:
                iref = conductance * abs(vpk * math.sin(math.radians(angle)))
                kindu = min(max(fw.indu_comp_gain * iref, fw.indu_comp_min), fw.indu_comp_max)
                # Rounded cache key avoids recomputing virtually identical gain points.
                key = round(float(kindu), 6)
                margin = margin_cache.get(key)
                if margin is None:
                    _, margin, _ = _current_loop_at_indu_comp(
                        config, frequencies, current_sense, kindu)
                    margin_cache[key] = margin
                points.append(CurrentLoopEnvelopePoint(
                    vin_rms_v=float(vin),
                    load_ratio=float(load),
                    line_angle_deg=float(angle),
                    current_reference_a=float(iref),
                    indu_comp=float(kindu),
                    crossover_hz=margin.critical_gain_crossover_hz,
                    phase_margin_deg=margin.phase_margin_deg,
                    gain_margin_db=margin.gain_margin_db,
                ))
    return tuple(points)


def tune_pfc_current_loop(
    config: PFCControlLabConfig,
    *,
    desired_phase_margin_deg: float = 55.0,
    minimum_phase_margin_deg: float = 50.0,
    minimum_gain_margin_db: float = 10.0,
    maximum_crossover_hz: float | None = None,
) -> CurrentLoopTuneResult:
    """Return a conservative PI current-loop starting point.

    Search variables are target crossover and PI zero placement.  Kp is solved
    from the exact sampled open-loop magnitude at the candidate crossover.  The
    candidate is accepted only after checking both minimum and maximum
    ``indu_comp`` gain; the final result is then checked over a broader
    line/load/phase envelope.
    """
    config.validate()
    stage, fw = config.power_stage, config.firmware
    ts = 1.0 / fw.current_loop_rate_hz
    nyquist = 0.5 * fw.current_loop_rate_hz
    safe_high = min(
        0.08 * fw.current_loop_rate_hz,
        0.08 * stage.switching_frequency_hz,
        0.16 * nyquist,
        config.frequency_stop_hz * 0.80,
    )
    if maximum_crossover_hz is not None:
        safe_high = min(safe_high, float(maximum_crossover_hz))
    safe_low = max(200.0, 0.008 * fw.current_loop_rate_hz)
    if safe_high <= safe_low:
        safe_high = max(safe_low * 1.25, min(1000.0, config.frequency_stop_hz * 0.25))

    frequencies = np.geomspace(
        max(0.5, min(config.frequency_start_hz, 1.0)),
        min(config.frequency_stop_hz, 0.45 * fw.current_loop_rate_hz),
        max(config.frequency_points, 2200),
    )
    current_sense = sense_frequency_response(config.current_sense, frequencies)

    target_candidates = np.geomspace(safe_high, safe_low, 20)
    zero_ratios = (5.0, 6.0, 8.0, 10.0, 4.0, 3.0)
    best: tuple[float, float, float, StabilityMargins, StabilityMargins] | None = None
    fallback: tuple[float, float, float, StabilityMargins, StabilityMargins] | None = None
    fallback_score = -1.0e30

    for target_fc in target_candidates:
        for zero_ratio in zero_ratios:
            ti = zero_ratio / (2.0 * math.pi * float(target_fc))
            unity = PIControllerConfig(
                kp=1.0,
                ti_s=ti,
                sample_time_s=ts,
                output_min=config.current_controller.output_min,
                output_max=config.current_controller.output_max,
            )
            trial_unity = replace(config, current_controller=unity)
            unity_responses, _, _ = _current_loop_at_indu_comp(
                trial_unity, frequencies, current_sense, fw.indu_comp_max)
            mag_db = _interp_db(frequencies, unity_responses["open_current"], float(target_fc))
            kp = 10.0 ** (-mag_db / 20.0)
            if not (math.isfinite(kp) and 1e-8 < kp < 1e3):
                continue

            controller = PIControllerConfig(
                kp=kp,
                ti_s=ti,
                sample_time_s=ts,
                output_min=config.current_controller.output_min,
                output_max=config.current_controller.output_max,
            )
            trial = replace(config, current_controller=controller)
            _, margin_hi, _ = _current_loop_at_indu_comp(
                trial, frequencies, current_sense, fw.indu_comp_max)
            _, margin_lo, _ = _current_loop_at_indu_comp(
                trial, frequencies, current_sense, fw.indu_comp_min)
            worst_pm = min(_pm_value(margin_hi), _pm_value(margin_lo))
            worst_gm = min(_gm_value(margin_hi), _gm_value(margin_lo))
            score = worst_pm + 0.2 * min(worst_gm, 30.0) + 0.002 * float(target_fc)
            if score > fallback_score:
                fallback_score = score
                fallback = (float(target_fc), kp, ti, margin_hi, margin_lo)

            if (
                worst_pm >= minimum_phase_margin_deg
                and worst_gm >= minimum_gain_margin_db
                and _pm_value(margin_hi) >= desired_phase_margin_deg - 3.0
            ):
                # Candidates are searched high-bandwidth first.  The first
                # successful frequency is preferred; within it choose the zero
                # placement with the strongest phase margin.
                candidate = (float(target_fc), kp, ti, margin_hi, margin_lo)
                if best is None or target_fc > best[0] * 1.001 or (
                    abs(target_fc - best[0]) <= best[0] * 0.001
                    and min(_pm_value(margin_hi), _pm_value(margin_lo))
                    > min(_pm_value(best[3]), _pm_value(best[4]))
                ):
                    best = candidate
        if best is not None and abs(best[0] - target_fc) <= target_fc * 0.001:
            break

    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("PFC current-loop tuner could not find a finite PI candidate")

    target_fc, kp, ti, _, _ = chosen
    controller = PIControllerConfig(
        kp=kp,
        ti_s=ti,
        sample_time_s=ts,
        output_min=config.current_controller.output_min,
        output_max=config.current_controller.output_max,
    )
    tuned = replace(config, current_controller=controller)
    _, nominal, _ = _current_loop_at_indu_comp(
        tuned, frequencies, current_sense, stage.indu_comp)
    envelope = _operating_envelope(tuned, frequencies, current_sense)
    valid_pm = [p for p in envelope if p.phase_margin_deg is not None]
    worst_point = min(valid_pm, key=lambda p: p.phase_margin_deg) if valid_pm else None
    worst_pm = None if worst_point is None else worst_point.phase_margin_deg
    gm_values = [p.gain_margin_db for p in envelope if p.gain_margin_db is not None]
    worst_gm = min(gm_values) if gm_values else None

    accepted = bool(
        nominal.phase_margin_deg is not None
        and nominal.phase_margin_deg >= desired_phase_margin_deg - 3.0
        and worst_pm is not None
        and worst_pm >= minimum_phase_margin_deg
        and (worst_gm is None or worst_gm >= minimum_gain_margin_db)
    )
    if accepted:
        message = "STABLE — 推荐作为电流环调试起点，再由用户按动态/THD目标微调。"
    else:
        message = "CONSERVATIVE FALLBACK — 已给出最优候选，但当前硬件/延迟约束下未达到全部裕量目标。"

    return CurrentLoopTuneResult(
        controller=controller,
        target_crossover_hz=target_fc,
        nominal_crossover_hz=nominal.critical_gain_crossover_hz,
        nominal_phase_margin_deg=nominal.phase_margin_deg,
        nominal_gain_margin_db=nominal.gain_margin_db,
        worst_phase_margin_deg=worst_pm,
        worst_gain_margin_db=worst_gm,
        worst_point=worst_point,
        envelope=envelope,
        accepted=accepted,
        message=message,
    )


__all__ = [
    "CurrentLoopEnvelopePoint",
    "CurrentLoopTuneResult",
    "tune_pfc_current_loop",
]
