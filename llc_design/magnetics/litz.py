"""Physics-based Litz-wire copper-loss and layered-winding field model.

The model separates five effects:
1. hot DC resistance;
2. exact round-strand internal skin effect;
3. external proximity loss from the layer-by-layer MMF field;
4. imperfect transposition / sub-bundle circulating-current penalty;
5. termination and lead resistance.

Currents and magnetic fields are decomposed harmonic-by-harmonic.  This is a
substantial improvement over applying one empirical Rac/Rdc multiplier to the
entire winding.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import iv


MU0 = 4.0e-7 * math.pi
COPPER_RESISTIVITY_20 = 1.724e-8
COPPER_ALPHA = 0.00393


@dataclass(frozen=True)
class LitzWire:
    strand_count: int
    strand_copper_diameter_m: float
    strand_outer_diameter_m: float
    packing_factor: float
    sub_bundle_count: int

    @property
    def copper_area_m2(self) -> float:
        return self.strand_count * math.pi * self.strand_copper_diameter_m**2 / 4.0

    @property
    def copper_area_mm2(self) -> float:
        return self.copper_area_m2 * 1e6

    @property
    def insulated_strand_area_m2(self) -> float:
        return self.strand_count * math.pi * self.strand_outer_diameter_m**2 / 4.0

    @property
    def envelope_area_m2(self) -> float:
        # The larger of copper-fill and insulated-strand packing constraints.
        return max(self.copper_area_m2 / self.packing_factor,
                   self.insulated_strand_area_m2 / 0.72)

    @property
    def equivalent_outer_diameter_m(self) -> float:
        return 2.0 * math.sqrt(self.envelope_area_m2 / math.pi)

    @property
    def equivalent_outer_diameter_mm(self) -> float:
        return self.equivalent_outer_diameter_m * 1e3

    @property
    def strands_per_sub_bundle(self) -> int:
        return int(math.ceil(self.strand_count / max(self.sub_bundle_count, 1)))

    @property
    def description(self) -> str:
        if self.sub_bundle_count <= 1:
            return f"{self.strand_count}×{self.strand_copper_diameter_m*1e3:.3f} mm"
        return (f"{self.sub_bundle_count} parallel sub-bundles × approximately "
                f"{self.strands_per_sub_bundle}×{self.strand_copper_diameter_m*1e3:.3f} mm")


@dataclass(frozen=True)
class HarmonicLoss:
    harmonic: int
    frequency_hz: float
    current_rms_a: float
    field_rms_a_per_m: float
    dc_component_w: float
    skin_increment_w: float
    proximity_w: float


@dataclass(frozen=True)
class WindingLossBreakdown:
    dc_copper_w: float
    skin_effect_w: float
    external_proximity_w: float
    bundle_circulating_w: float
    termination_w: float
    total_w: float
    effective_ac_factor: float
    current_rms_a: float
    harmonics: tuple[HarmonicLoss, ...]


@dataclass(frozen=True)
class StackLayer:
    label: str
    turns: int
    conductor_length_m: float
    wire: LitzWire
    current_waveform_a: tuple[float, ...]


@dataclass(frozen=True)
class _LayerSpectrum:
    layer: StackLayer
    current_phasors: tuple[complex, ...]
    current_rms_total: float
    rdc_ohm: float


def copper_resistivity(temperature_c: float) -> float:
    return COPPER_RESISTIVITY_20 * (1.0 + COPPER_ALPHA * (temperature_c - 20.0))


def skin_depth_m(frequency_hz: float, temperature_c: float = 20.0) -> float:
    if frequency_hz <= 0.0:
        return math.inf
    rho = copper_resistivity(temperature_c)
    return math.sqrt(rho / (math.pi * frequency_hz * MU0))


def round_wire_skin_factor(strand_diameter_m: float, frequency_hz: float,
                           temperature_c: float = 20.0) -> float:
    """Exact internal impedance of an isolated round conductor.

    Uses the modified-Bessel representation of the cylindrical diffusion
    solution.  The returned value is Rac/Rdc and approaches one continuously
    at low frequency.
    """
    if frequency_hz <= 0.0:
        return 1.0
    radius = strand_diameter_m / 2.0
    delta = skin_depth_m(frequency_hz, temperature_c)
    z = (1.0 + 1.0j) * radius / delta
    if abs(z) < 1e-4:
        # Series result: Rac/Rdc = 1 + (a/delta)^4 / 48 + ...
        return 1.0 + (radius / delta) ** 4 / 48.0
    ratio = (z / 2.0) * iv(0, z) / iv(1, z)
    return max(1.0, float(ratio.real))


def dc_resistance_ohm(wire: LitzWire, length_m: float,
                      temperature_c: float) -> float:
    return copper_resistivity(temperature_c) * length_m / wire.copper_area_m2


def _fft_rms_phasors(waveform: Sequence[float], max_harmonic: int) -> tuple[complex, ...]:
    values = np.asarray(waveform, dtype=float)
    if values.ndim != 1 or len(values) < 32:
        raise ValueError("waveform must contain at least 32 samples")
    coeffs = np.fft.rfft(values) / len(values)
    phasors: list[complex] = [complex(values.mean(), 0.0)]
    for harmonic in range(1, max_harmonic + 1):
        if harmonic >= len(coeffs):
            phasors.append(0.0j)
        else:
            # peak complex coefficient is 2*Ck; RMS phasor is peak/sqrt(2)
            phasors.append(math.sqrt(2.0) * complex(coeffs[harmonic]))
    return tuple(phasors)


def harmonic_rms_spectrum(waveform: Sequence[float], max_harmonic: int = 15) -> tuple[float, ...]:
    return tuple(abs(x) for x in _fft_rms_phasors(waveform, max_harmonic))


def transverse_field_loss_per_strand_w_per_m(
        strand_diameter_m: float, frequency_hz: float,
        h_rms_a_per_m: float, temperature_c: float = 100.0) -> float:
    """Eddy-current loss of one round strand in a transverse RMS H field.

    The low-frequency cylinder solution is exact when d << skin depth.  A
    bounded penetration correction extends it through the practical Litz range
    without treating the whole bundle as a solid conductor.
    """
    if frequency_hz <= 0.0 or h_rms_a_per_m <= 0.0:
        return 0.0
    rho = copper_resistivity(temperature_c)
    omega = 2.0 * math.pi * frequency_hz
    b_rms = MU0 * h_rms_a_per_m
    radius = strand_diameter_m / 2.0
    low_frequency = (math.pi * radius**4 * omega**2 * b_rms**2) / (4.0 * rho)
    penetration = round_wire_skin_factor(strand_diameter_m, frequency_hz, temperature_c)
    return low_frequency * min(penetration, 8.0)


def select_litz_wire(current_rms_a: float, strand_copper_diameter_m: float,
                     strand_outer_diameter_m: float, packing_factor: float,
                     current_density_a_per_mm2: float,
                     maximum_strands_per_sub_bundle: int = 400,
                     strand_rounding: int = 25) -> LitzWire:
    area_required_mm2 = current_rms_a / current_density_a_per_mm2
    area_per_strand_mm2 = math.pi * (strand_copper_diameter_m * 1e3)**2 / 4.0
    raw_count = max(1, math.ceil(area_required_mm2 / area_per_strand_mm2))
    strand_count = int(math.ceil(raw_count / strand_rounding) * strand_rounding)
    sub_bundles = int(math.ceil(strand_count / maximum_strands_per_sub_bundle))
    return LitzWire(strand_count, strand_copper_diameter_m,
                    strand_outer_diameter_m, packing_factor, sub_bundles)


def winding_layers(turns: int, wire: LitzWire, window_width_mm: float,
                   turn_spacing_mm: float = 0.15) -> tuple[int, int]:
    pitch = wire.equivalent_outer_diameter_mm + turn_spacing_mm
    turns_per_layer = max(1, int(window_width_mm // pitch))
    return int(math.ceil(turns / turns_per_layer)), turns_per_layer


def distribute_turns(turns: int, turns_per_layer: int) -> tuple[int, ...]:
    remaining = turns
    result: list[int] = []
    while remaining > 0:
        n = min(remaining, turns_per_layer)
        result.append(n)
        remaining -= n
    return tuple(result)


def layered_litz_stack_loss(
        layers: Sequence[StackLayer], fundamental_frequency_hz: float,
        window_width_m: float, temperature_c: float = 100.0,
        max_harmonic: int = 15, transposition_quality: float = 0.90,
        sub_bundle_coupling_factor: float = 0.12,
        termination_resistance_fraction: float = 0.03,
        extra_field_harmonics_a_per_m: Mapping[int, float] | None = None,
        calibration_factor: float = 1.0) -> dict[str, WindingLossBreakdown]:
    """Calculate losses for a complete physical winding stack.

    The cumulative complex ampere-turns at each layer boundary are used to
    obtain the average H^2 inside every layer.  This naturally captures the
    benefit of P/2-S-P/2 interleaving and the severe field in a two-layer
    inductor winding.
    """
    if not layers:
        return {}
    sample_count = len(layers[0].current_waveform_a)
    if any(len(layer.current_waveform_a) != sample_count for layer in layers):
        raise ValueError("all stack-layer waveforms must have equal length")
    if window_width_m <= 0.0:
        raise ValueError("window width must be positive")
    transposition_quality = min(max(transposition_quality, 0.0), 1.0)

    spectra: list[_LayerSpectrum] = []
    for layer in layers:
        phasors = _fft_rms_phasors(layer.current_waveform_a, max_harmonic)
        rms_total = float(np.sqrt(np.mean(np.asarray(layer.current_waveform_a) ** 2)))
        rdc = dc_resistance_ohm(layer.wire, layer.conductor_length_m, temperature_c)
        spectra.append(_LayerSpectrum(layer, phasors, rms_total, rdc))

    labels = sorted({layer.label for layer in layers})
    accum: dict[str, dict[str, object]] = {
        label: {"dc": 0.0, "skin": 0.0, "prox": 0.0, "bundle": 0.0,
                "term": 0.0, "rms_sq_length": 0.0, "rdc_loss": 0.0,
                "harmonics": []}
        for label in labels
    }

    extra = extra_field_harmonics_a_per_m or {}
    for harmonic in range(0, max_harmonic + 1):
        frequency = harmonic * fundamental_frequency_hz
        boundary_mmf: list[complex] = [0.0j]
        for spectrum in spectra:
            iph = spectrum.current_phasors[harmonic]
            boundary_mmf.append(boundary_mmf[-1] + spectrum.layer.turns * iph)

        for index, spectrum in enumerate(spectra):
            label = spectrum.layer.label
            current = spectrum.current_phasors[harmonic]
            i_rms = abs(current)
            if harmonic == 0:
                p_dc = i_rms**2 * spectrum.rdc_ohm
                accum[label]["dc"] = float(accum[label]["dc"]) + p_dc
                continue
            h0 = boundary_mmf[index] / window_width_m
            h1 = boundary_mmf[index + 1] / window_width_m
            # Integral of |linear complex field|^2 through the layer thickness.
            h_sq = ((abs(h0) ** 2 + (h0 * h1.conjugate()).real + abs(h1) ** 2) / 3.0
                    + float(extra.get(harmonic, 0.0)) ** 2)
            h_rms = math.sqrt(max(h_sq, 0.0))

            p_base = i_rms**2 * spectrum.rdc_ohm
            skin_factor = round_wire_skin_factor(
                spectrum.layer.wire.strand_copper_diameter_m, frequency, temperature_c)
            p_skin = p_base * (skin_factor - 1.0)
            p_prox = (transverse_field_loss_per_strand_w_per_m(
                spectrum.layer.wire.strand_copper_diameter_m, frequency,
                h_rms, temperature_c)
                * spectrum.layer.wire.strand_count * spectrum.layer.conductor_length_m)

            # Incomplete strand transposition and independently twisted parallel
            # sub-bundles allow residual circulating currents.  The penalty is
            # tied to bundle build and the physically calculated field loss.
            build_ratio = max(1.0, spectrum.layer.wire.equivalent_outer_diameter_m /
                              spectrum.layer.wire.strand_outer_diameter_m)
            imperfection = (1.0 - transposition_quality) * math.sqrt(build_ratio)
            sub_bundle = max(spectrum.layer.wire.sub_bundle_count - 1, 0)
            p_bundle = p_prox * (sub_bundle_coupling_factor * imperfection
                                 * math.log1p(sub_bundle))
            p_prox *= calibration_factor
            p_bundle *= calibration_factor

            accum[label]["dc"] = float(accum[label]["dc"]) + p_base
            accum[label]["skin"] = float(accum[label]["skin"]) + p_skin
            accum[label]["prox"] = float(accum[label]["prox"]) + p_prox
            accum[label]["bundle"] = float(accum[label]["bundle"]) + p_bundle
            cast_list = accum[label]["harmonics"]
            assert isinstance(cast_list, list)
            cast_list.append(HarmonicLoss(
                harmonic=harmonic, frequency_hz=frequency,
                current_rms_a=i_rms, field_rms_a_per_m=h_rms,
                dc_component_w=p_base, skin_increment_w=p_skin,
                proximity_w=p_prox + p_bundle,
            ))

    results: dict[str, WindingLossBreakdown] = {}
    for label in labels:
        item = accum[label]
        dc = float(item["dc"])
        skin = float(item["skin"])
        prox = float(item["prox"])
        bundle = float(item["bundle"])
        termination = termination_resistance_fraction * dc
        total = dc + skin + prox + bundle + termination
        harmonics = tuple(item["harmonics"])  # type: ignore[arg-type]
        # Sum actual waveform RMS over all layers is not useful; report the
        # common winding current from the first layer carrying this label.
        first = next(s for s in spectra if s.layer.label == label)
        factor = total / dc if dc > 0.0 else 1.0
        results[label] = WindingLossBreakdown(
            dc_copper_w=dc, skin_effect_w=skin,
            external_proximity_w=prox, bundle_circulating_w=bundle,
            termination_w=termination, total_w=total,
            effective_ac_factor=factor, current_rms_a=first.current_rms_total,
            harmonics=harmonics,
        )
    return results


def litz_ac_factor(wire: LitzWire, frequency_hz: float, layers: int,
                   severity: float, correction: float,
                   temperature_c: float = 100.0) -> float:
    """Compatibility wrapper for legacy callers.

    New magnetics code uses ``layered_litz_stack_loss``.  This wrapper retains
    a conservative single-frequency estimate for external users.
    """
    skin = round_wire_skin_factor(wire.strand_copper_diameter_m,
                                  frequency_hz, temperature_c)
    delta = skin_depth_m(frequency_hz, temperature_c)
    strand_ratio = wire.strand_copper_diameter_m / max(2.0 * delta, 1e-15)
    proximity = correction * severity * max(layers, 1) ** 2 * strand_ratio**4
    return max(1.0, skin + proximity)
