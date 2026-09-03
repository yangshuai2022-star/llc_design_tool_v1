"""Digital synchronous-rectifier timing, third-quadrant and recovery model.

This module consumes the V8 Golden secondary-current waveform.  It deliberately
separates the *physical* current-zero window from the firmware command timing:

    command on  = conduction start + driver/blanking/guard delay
    command off = current zero - propagation/guard advance

The loss model then replays those commands against the same current waveform,
so channel, third-quadrant, body-diode and reverse-recovery losses are mutually
consistent instead of being calculated from unrelated sinusoidal estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..analysis.types import LLCModelResult
from ..core.spec import LLCDesignSpec, MosfetSpec


@dataclass(frozen=True)
class SRTimingConfig:
    current_threshold_a: float = 0.02
    on_delay_s: float | None = None
    turnoff_advance_s: float | None = None
    propagation_s: float | None = None
    turnoff_propagation_s: float | None = None
    blanking_s: float | None = None
    guard_s: float | None = None
    recovery_enabled: bool = True
    enforce_safe_turnoff: bool = True


@dataclass(frozen=True)
class SRTimingPoint:
    switching_frequency_hz: float
    positive_on_delay_s: float
    positive_off_advance_s: float
    positive_conduction_s: float
    negative_on_delay_s: float
    negative_off_advance_s: float
    negative_conduction_s: float
    positive_zero_crossing_s: float
    negative_zero_crossing_s: float
    minimum_reverse_margin_a: float

    @property
    def on_delay_s(self) -> float:
        return 0.5 * (self.positive_on_delay_s + self.negative_on_delay_s)

    @property
    def off_advance_s(self) -> float:
        return 0.5 * (self.positive_off_advance_s + self.negative_off_advance_s)

    @property
    def conduction_s(self) -> float:
        return 0.5 * (self.positive_conduction_s + self.negative_conduction_s)


@dataclass(frozen=True)
class SRLossBreakdown:
    channel_w: float
    third_quadrant_w: float
    body_diode_w: float
    reverse_current_w: float
    reverse_recovery_w: float
    coss_w: float
    gate_drive_w: float
    total_w: float
    reverse_charge_c: float
    recovery_charge_c_per_cycle: float
    body_diode_charge_c_per_cycle: float
    device_rms_current_a: float
    device_peak_current_a: float
    voltage_stress_v: float


@dataclass(frozen=True)
class SRAnalysis:
    timing: SRTimingPoint
    loss: SRLossBreakdown
    gate_positive: NDArray[np.float64]
    gate_negative: NDArray[np.float64]
    body_diode_current_a: NDArray[np.float64]
    third_quadrant_current_a: NDArray[np.float64]
    reverse_current_a: NDArray[np.float64]
    recovery_current_a: NDArray[np.float64]
    warnings: tuple[str, ...] = ()


def _one_cycle(result: LLCModelResult, key: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    bundle = result.waveform
    fs = result.metrics.switching_frequency_hz
    if fs <= 0.0:
        raise ValueError("switching frequency must be positive")
    time = np.asarray(bundle.time_s, dtype=float)
    values = np.asarray(bundle.signal(key).values, dtype=float)
    if len(time) != len(values) or len(time) < 32:
        raise ValueError("Golden waveform is too short for SR timing analysis")
    period = 1.0 / fs
    dt = float(np.median(np.diff(time)))
    n = max(32, int(round(period / dt)))
    n = min(n, len(values))
    return time[:n] - time[0], values[:n]


def _periodic_interp_crossings(time: NDArray[np.float64], current: NDArray[np.float64], threshold: float) -> list[tuple[float, int]]:
    """Return polarity-changing zeroes as (time, direction), direction +/-1."""
    period = float((time[-1] - time[0]) + np.median(np.diff(time)))
    scale = max(float(np.max(np.abs(current))), 1.0)
    eps = max(abs(threshold), 1e-9 * scale)
    x = current.copy()
    x[np.abs(x) <= eps] = 0.0
    crossings: list[tuple[float, int]] = []
    n = len(x)
    for k in range(n):
        k1 = (k + 1) % n
        y0 = float(x[k]); y1 = float(x[k1])
        if y0 == 0.0:
            continue
        if y0 * y1 < 0.0:
            t0 = float(time[k])
            t1 = float(time[k1] if k1 else period)
            frac = abs(y0) / max(abs(y0) + abs(y1), 1e-15)
            tc = (t0 + frac * (t1 - t0)) % period
            crossings.append((tc, 1 if y1 > y0 else -1))
    crossings.sort(key=lambda item: item[0])
    # dedupe numerical duplicates
    unique: list[tuple[float, int]] = []
    for item in crossings:
        if not unique or abs(item[0] - unique[-1][0]) > period / (10 * n):
            unique.append(item)
    return unique


def _bridge_edges(result: LLCModelResult, period: float) -> list[float]:
    """Primary *command* commutation references for one switching period.

    SR timing is specified relative to primary gate commands, not the bridge
    voltage after deadtime.  Therefore the two references are exactly 0 and
    Ts/2 even when ``v_bridge`` contains a zero-voltage deadtime interval.
    """
    return [0.0, 0.5 * period]


def _next_periodic(event: float, reference: float, period: float) -> float:
    while event < reference:
        event += period
    return event


def _window_from_polarity(time: NDArray[np.float64], current: NDArray[np.float64], polarity: int, threshold: float) -> tuple[float, float]:
    period = float((time[-1] - time[0]) + np.median(np.diff(time)))
    crossings = _periodic_interp_crossings(time, current, threshold)
    if len(crossings) < 2:
        raise ValueError("secondary current does not contain two commutation zeroes")
    candidates = []
    for idx, (tc, direction) in enumerate(crossings):
        if direction != polarity:
            continue
        end = None
        for offset in range(1, len(crossings) + 1):
            tn, dn = crossings[(idx + offset) % len(crossings)]
            tn = _next_periodic(tn, tc + 1e-18, period)
            if dn == -polarity:
                end = tn
                break
        if end is not None:
            candidates.append((tc, end))
    if not candidates:
        raise ValueError("unable to identify requested SR conduction window")
    return max(candidates, key=lambda pair: pair[1] - pair[0])


def _periodic_mask(time: NDArray[np.float64], start: float, stop: float, period: float) -> NDArray[np.float64]:
    phase = np.mod(time, period)
    s = start % period; e = stop % period
    if (stop - start) >= period:
        return np.ones_like(time)
    if s <= e:
        return ((phase >= s) & (phase < e)).astype(float)
    return ((phase >= s) | (phase < e)).astype(float)


def solve_sr_timing(result: LLCModelResult, spec: LLCDesignSpec, config: SRTimingConfig | None = None) -> SRTimingPoint:
    cfg = config or SRTimingConfig()
    time, isec = _one_cycle(result, "i_transformer_secondary")
    fs = result.metrics.switching_frequency_hz
    period = 1.0 / fs
    pos_start, pos_zero = _window_from_polarity(time, isec, +1, cfg.current_threshold_a)
    neg_start, neg_zero = _window_from_polarity(time, isec, -1, cfg.current_threshold_a)
    edges = _bridge_edges(result, period)

    half = 0.5 * period
    def nearest_edge(t: float) -> float:
        return round(t / half) * half

    def next_edge(t: float) -> float:
        return math.ceil((t - 1e-18) / half) * half

    prop_on = spec.sr_driver_propagation_s if cfg.propagation_s is None else cfg.propagation_s
    prop_off = spec.sr_turnoff_propagation_s if cfg.turnoff_propagation_s is None else cfg.turnoff_propagation_s
    blank = spec.sr_blanking_s if cfg.blanking_s is None else cfg.blanking_s
    guard = spec.sr_timing_guard_s if cfg.guard_s is None else cfg.guard_s
    requested_on = spec.sr_turnon_delay_s if cfg.on_delay_s is None else cfg.on_delay_s
    requested_adv = spec.sr_turnoff_advance_s if cfg.turnoff_advance_s is None else cfg.turnoff_advance_s

    def derive(start: float, zero: float) -> tuple[float, float, float, float]:
        ref_on = nearest_edge(start)
        command_on = max(start + blank + guard - prop_on, ref_on + requested_on)
        physical_off_advance = (
            max(requested_adv, prop_off + guard)
            if cfg.enforce_safe_turnoff else requested_adv
        )
        command_off = zero - physical_off_advance
        if command_off <= command_on:
            command_off = 0.5 * (command_on + zero)
        on_delay = command_on - ref_on
        off_advance = next_edge(zero) - command_off
        return on_delay, off_advance, command_off - command_on, zero % period

    p_on, p_off, p_cond, p_zero = derive(pos_start, pos_zero)
    n_on, n_off, n_cond, n_zero = derive(neg_start, neg_zero)
    # Margin is measured at the commanded turn-off instant; positive means the
    # desired current polarity has not yet reversed.
    def current_at(tq: float) -> float:
        phase = np.mod(tq, period)
        return float(np.interp(phase, time, isec, period=period))
    p_cmd_off = pos_zero - (max(requested_adv, prop_off + guard) if cfg.enforce_safe_turnoff else requested_adv)
    n_cmd_off = neg_zero - (max(requested_adv, prop_off + guard) if cfg.enforce_safe_turnoff else requested_adv)
    margin = min(abs(current_at(p_cmd_off)), abs(current_at(n_cmd_off)))
    return SRTimingPoint(
        switching_frequency_hz=fs,
        positive_on_delay_s=p_on,
        positive_off_advance_s=p_off,
        positive_conduction_s=p_cond,
        negative_on_delay_s=n_on,
        negative_off_advance_s=n_off,
        negative_conduction_s=n_cond,
        positive_zero_crossing_s=p_zero,
        negative_zero_crossing_s=n_zero,
        minimum_reverse_margin_a=margin,
    )


def _triangular_recovery_current(time: NDArray[np.float64], event_times: Sequence[float], qrr_c: float, trr_s: float, period: float) -> NDArray[np.float64]:
    result = np.zeros_like(time)
    if qrr_c <= 0.0 or trr_s <= 0.0:
        return result
    peak = 2.0 * qrr_c / trr_s
    for event in event_times:
        age = np.mod(time - event, period)
        active = age < trr_s
        result[active] += peak * (1.0 - age[active] / trr_s)
    return result


def analyze_sr(result: LLCModelResult, spec: LLCDesignSpec, device: MosfetSpec, config: SRTimingConfig | None = None) -> SRAnalysis:
    cfg = config or SRTimingConfig()
    timing = solve_sr_timing(result, spec, cfg)
    time, isec = _one_cycle(result, "i_transformer_secondary")
    fs = result.metrics.switching_frequency_hz
    period = 1.0 / fs
    npar = max(spec.sr_parallel_devices_per_position, 1)
    rds = device.rds_at(spec.sr_junction_temperature_c) / npar
    tq = device.third_quadrant_rds_factor * rds

    pos_start, pos_zero = _window_from_polarity(time, isec, +1, cfg.current_threshold_a)
    neg_start, neg_zero = _window_from_polarity(time, isec, -1, cfg.current_threshold_a)
    edges = _bridge_edges(result, period)
    half=0.5*period
    def nearest_edge(t: float) -> float:
        return round(t/half)*half
    prop_on = spec.sr_driver_propagation_s if cfg.propagation_s is None else cfg.propagation_s
    prop_off = spec.sr_turnoff_propagation_s if cfg.turnoff_propagation_s is None else cfg.turnoff_propagation_s
    blank = spec.sr_blanking_s if cfg.blanking_s is None else cfg.blanking_s
    guard = spec.sr_timing_guard_s if cfg.guard_s is None else cfg.guard_s
    req_on = spec.sr_turnon_delay_s if cfg.on_delay_s is None else cfg.on_delay_s
    req_adv = spec.sr_turnoff_advance_s if cfg.turnoff_advance_s is None else cfg.turnoff_advance_s
    def commands(start: float, zero: float) -> tuple[float,float]:
        on=max(start+blank+guard-prop_on, nearest_edge(start)+req_on)
        advance=(max(req_adv,prop_off+guard) if cfg.enforce_safe_turnoff else req_adv)
        off=zero-advance
        if off<=on: off=0.5*(on+zero)
        return on,off
    p_on,p_off=commands(pos_start,pos_zero)
    n_on,n_off=commands(neg_start,neg_zero)
    gp=_periodic_mask(time,p_on,p_off,period)
    gn=_periodic_mask(time,n_on,n_off,period)

    expected_gate=np.where(isec>=0.0,gp,gn)
    desired_sign=np.where(gp>0.5,1.0,np.where(gn>0.5,-1.0,0.0))
    signed_expected=isec*desired_sign
    channel_mask=(expected_gate>0.5)&(signed_expected>=0.0)
    third_mask=(expected_gate>0.5)&(signed_expected<0.0)
    diode_mask=(expected_gate<=0.5)&(np.abs(isec)>cfg.current_threshold_a)
    reverse=np.where(third_mask,np.abs(isec),0.0)
    diode=np.where(diode_mask,np.abs(isec),0.0)
    third=np.where(third_mask,np.abs(isec),0.0)

    # Pair-current path: two SR devices conduct simultaneously. Parallel devices
    # split current at each bridge position.
    p_channel=2.0*np.mean(np.where(channel_mask,isec**2,0.0))*rds
    p_third=2.0*np.mean(third**2)*tq + 2.0*device.third_quadrant_v_offset_v*np.mean(third)
    p_diode=2.0*device.body_diode_vf_v*np.mean(diode)
    p_reverse=2.0*np.mean(reverse**2)*tq
    q_reverse=2.0*float(np.trapezoid(reverse,time))
    q_diode=2.0*float(np.trapezoid(diode,time))

    # Reverse recovery is attached to the two body-diode commutations per cycle.
    # Qrr is scaled sub-linearly with the actual pre-commutation diode current.
    event_times=[]
    qrr_cycle=0.0
    if cfg.recovery_enabled and device.qrr_c>0.0 and device.trr_s>0.0:
        for zero in (pos_zero,neg_zero):
            sample_t=(zero-0.5*device.trr_s)%period
            i_pre=abs(float(np.interp(sample_t,time,isec,period=period)))
            scale=math.sqrt(max(i_pre,0.0)/max(device.qrr_ref_i_a,1e-9)) if i_pre>0 else 0.0
            qrr_cycle += device.qrr_c*min(max(scale,0.1),2.0)
            event_times.append(zero%period)
    recovery=_triangular_recovery_current(time,event_times,qrr_cycle/max(len(event_times),1),device.trr_s,period)
    v_stress=spec.sr_voltage_overshoot_factor*spec.vout_v
    p_rr=qrr_cycle*v_stress*fs
    p_coss=4.0*npar*0.5*device.coss_er_f*v_stress**2*fs*spec.sr_coss_dissipation_factor
    p_gate=4.0*npar*device.qg_c*device.gate_voltage_v*fs
    total=p_channel+p_third+p_diode+p_reverse+p_rr+p_coss+p_gate
    rms=float(np.sqrt(np.mean(isec**2)))/(math.sqrt(2.0)*npar)
    peak=float(np.max(np.abs(isec)))/npar
    warnings=[]
    if timing.minimum_reverse_margin_a < spec.sr_reverse_current_limit_a:
        warnings.append("SR turn-off current margin is below the configured reverse-current guard.")
    if qrr_cycle>0 and np.max(recovery)>0.25*max(np.max(np.abs(isec)),1e-9):
        warnings.append("Reverse-recovery current is a significant fraction of the secondary current; verify device Qrr at temperature.")
    return SRAnalysis(
        timing=timing,
        loss=SRLossBreakdown(
            channel_w=float(p_channel), third_quadrant_w=float(p_third), body_diode_w=float(p_diode),
            reverse_current_w=float(p_reverse), reverse_recovery_w=float(p_rr), coss_w=float(p_coss),
            gate_drive_w=float(p_gate), total_w=float(total), reverse_charge_c=q_reverse,
            recovery_charge_c_per_cycle=float(qrr_cycle), body_diode_charge_c_per_cycle=q_diode,
            device_rms_current_a=rms, device_peak_current_a=peak, voltage_stress_v=v_stress,
        ),
        gate_positive=gp, gate_negative=gn, body_diode_current_a=diode,
        third_quadrant_current_a=third, reverse_current_a=reverse,
        recovery_current_a=recovery, warnings=tuple(warnings),
    )


def generate_sr_lut(
    solver: Callable[[float, float], LLCModelResult],
    spec: LLCDesignSpec,
    device: MosfetSpec,
    vin_values_v: Iterable[float],
    load_values: Iterable[float],
    config: SRTimingConfig | None = None,
) -> tuple[dict[str, float], ...]:
    """Generate a firmware-oriented Vin/load SR timing/loss table.

    ``solver(vin, load_fraction)`` is deliberately injected so callers can use
    HB, switched TD, or a cached Golden reference without coupling this module
    to one numerical backend.
    """
    rows=[]
    for vin in vin_values_v:
        for load in load_values:
            result=solver(float(vin),float(load))
            sr=analyze_sr(result,spec,device,config)
            rows.append({
                "vin_v":float(vin),"load_fraction":float(load),"fs_hz":sr.timing.switching_frequency_hz,
                "on_delay_ns":sr.timing.on_delay_s*1e9,"off_advance_ns":sr.timing.off_advance_s*1e9,
                "conduction_ns":sr.timing.conduction_s*1e9,"sr_loss_w":sr.loss.total_w,
                "reverse_charge_nc":sr.loss.reverse_charge_c*1e9,"qrr_nc_per_cycle":sr.loss.recovery_charge_c_per_cycle*1e9,
            })
    return tuple(rows)


def export_sr_lut_c99(rows: Sequence[dict[str,float]], path: str | Path, symbol: str="g_llc_sr_lut") -> Path:
    out=Path(path); out.parent.mkdir(parents=True,exist_ok=True)
    lines=[
        "/* Auto-generated by LLC Design Tool V8 SR timing solver. */",
        "#include <stdint.h>",
        "typedef struct { float vin_v, load_pu, fs_hz, on_delay_ns, off_advance_ns, conduction_ns; } llc_sr_lut_t;",
        f"const llc_sr_lut_t {symbol}[{len(rows)}] = {{",
    ]
    for r in rows:
        lines.append("    { %.6ff, %.6ff, %.6ff, %.6ff, %.6ff, %.6ff }," % (
            r['vin_v'],r['load_fraction'],r['fs_hz'],r['on_delay_ns'],r['off_advance_ns'],r['conduction_ns']))
    lines += ["};", f"const uint32_t {symbol}_count = {len(rows)}u;", ""]
    out.write_text("\n".join(lines),encoding="utf-8")
    return out
