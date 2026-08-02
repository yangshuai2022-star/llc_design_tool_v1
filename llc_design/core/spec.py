"""User-facing LLC design specification and reference device records.

The defaults represent the agreed V1 baseline:
400 Vdc -> 53 V / 3 kW, full-bridge LLC, full-bridge synchronous rectifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class PrimaryTopology(str, Enum):
    HALF_BRIDGE = "HALF_BRIDGE"
    FULL_BRIDGE = "FULL_BRIDGE"


class SecondaryTopology(str, Enum):
    FULL_BRIDGE_SR = "FULL_BRIDGE_SR"


@dataclass(frozen=True)
class MosfetSpec:
    """Simplified MOSFET record used by the V1 loss model.

    Reference records bundled with the tool are engineering placeholders, not
    substitutes for a selected part's nonlinear datasheet curves.
    """

    part_number: str
    technology: str
    vds_max_v: float
    id_cont_a: float
    rds_on_25_ohm: float
    rds_on_hot_ohm: float
    hot_temperature_c: float
    qg_c: float
    coss_er_f: float
    qoss_c: float
    eoff_ref_j: float
    eoff_ref_v: float
    eoff_ref_i: float
    gate_voltage_v: float
    body_diode_vf_v: float = 1.0
    package: str = "REFERENCE"
    price_usd: float = 0.0

    def rds_at(self, temperature_c: float) -> float:
        """Linear interpolation/extrapolation between 25 °C and hot Rds values."""
        span = max(self.hot_temperature_c - 25.0, 1.0)
        k = (temperature_c - 25.0) / span
        return max(self.rds_on_25_ohm * 0.8,
                   self.rds_on_25_ohm + k * (self.rds_on_hot_ohm - self.rds_on_25_ohm))


@dataclass
class LLCDesignSpec:
    """Complete V1 LLC design input set.

    Voltages are DC values. Frequencies are Hz. Inductances/capacitances use SI.
    """

    # Electrical specification
    vbus_nom_v: float = 400.0
    vbus_min_normal_v: float = 360.0
    vbus_max_v: float = 420.0
    vbus_hold_end_v: float = 300.0
    vout_v: float = 53.0
    pout_w: float = 3000.0
    efficiency_assumption: float = 0.96

    primary_topology: PrimaryTopology = PrimaryTopology.FULL_BRIDGE
    secondary_topology: SecondaryTopology = SecondaryTopology.FULL_BRIDGE_SR

    # Resonant tank
    resonant_frequency_hz: float = 100_000.0
    minimum_frequency_hz: float = 60_000.0
    maximum_frequency_hz: float = 180_000.0
    ln_ratio: float = 5.0                 # Lm/Lr
    q_full_load: float = 0.35             # sqrt(Lr/Cr)/Rac at rated load
    primary_turns: int = 30
    secondary_turns: int = 4
    rectifier_equivalent_drop_v: float = 0.40

    # Hold-up and capacitors
    bus_capacitance_f: float = 1800e-6
    requested_hold_time_s: float = 20e-3
    output_capacitance_f: float = 1500e-6
    output_cap_esr_ohm: float = 1.5e-3
    output_ripple_limit_vpp: float = 0.50
    resonant_cap_esr_ohm: float = 4.0e-3
    resonant_cap_voltage_rating_v: float = 1000.0
    resonant_cap_rms_current_rating_a: float = 30.0

    # Switching and devices
    primary_device: str = "REF_650V_SIC_45M"
    primary_parallel_devices: int = 1
    primary_deadtime_s: float = 200e-9
    primary_zvs_margin_required: float = 1.20
    # Safety margin on the commutation (magnetizing) current used for Eoff.
    primary_turnoff_current_factor: float = 1.2

    sr_device: str = "REF_100V_SI_1P8M"
    sr_parallel_devices_per_position: int = 2
    sr_deadtime_s: float = 100e-9
    sr_turnoff_advance_s: float = 30e-9
    sr_coss_dissipation_factor: float = 0.35

    # Temperatures / thermal approximation
    ambient_temperature_c: float = 45.0
    primary_junction_temperature_c: float = 100.0
    sr_junction_temperature_c: float = 100.0
    winding_temperature_c: float = 100.0
    transformer_rth_k_per_w: float = 5.0
    resonant_inductor_rth_k_per_w: float = 6.0
    magnetic_thermal_max_iterations: int = 12
    magnetic_thermal_tolerance_c: float = 0.25
    magnetic_waveform_samples: int = 2048
    magnetic_hotspot_limit_c: float = 130.0
    magnetic_candidate_full_evaluation_limit: int = 32

    # Litz wire and winding constraints
    litz_strand_copper_diameter_m: float = 0.10e-3
    litz_strand_outer_diameter_m: float = 0.112e-3
    litz_packing_factor: float = 0.55
    litz_current_density_target_a_per_mm2: float = 5.0
    litz_current_density_max_a_per_mm2: float = 6.0
    transformer_winding_layout: str = "P/2-S-P/2"
    transformer_core_families: tuple[str, ...] = ("PQ", "EE", "EC", "EER", "ETD")
    resonant_inductor_core_families: tuple[str, ...] = ("PQ", "EE", "EC", "EER", "ETD")
    transformer_max_fill_factor: float = 0.60
    transformer_max_b_t: float = 0.20
    transformer_max_gap_mm: float = 4.0
    transformer_insulation_area_mm2: float = 28.0
    resonant_inductor_max_layers: int = 2
    resonant_inductor_max_fill_factor: float = 0.55
    resonant_inductor_max_b_t: float = 0.22
    resonant_inductor_max_gap_mm: float = 3.0

    # Harmonic Litz-wire and winding-field model
    litz_max_harmonic: int = 15
    litz_transposition_quality: float = 0.90
    litz_sub_bundle_coupling_factor: float = 0.12
    winding_termination_resistance_fraction: float = 0.03
    litz_proximity_correction: float = 1.0
    transformer_proximity_severity: float = 1.0  # retained as calibration input
    inductor_proximity_severity: float = 1.0     # retained as calibration input
    include_gap_fringing_loss: bool = True
    gap_to_winding_distance_mm: float = 3.0
    gap_fringing_calibration: float = 1.0

    # Feasibility margins
    minimum_inductive_angle_deg: float = 3.0
    primary_voltage_derating: float = 0.80
    sr_voltage_overshoot_factor: float = 1.35
    sr_voltage_derating: float = 0.80
    capacitor_voltage_derating: float = 0.80

    # Search/work-point configuration
    minimum_modeled_load_fraction: float = 0.10

    def clone(self, **changes) -> "LLCDesignSpec":
        return replace(self, **changes)

    @property
    def turns_ratio(self) -> float:
        return self.primary_turns / self.secondary_turns

    @property
    def output_current_a(self) -> float:
        return self.pout_w / self.vout_v

    @property
    def bridge_gain(self) -> float:
        """DC conversion normalization: full bridge 1, half bridge 0.5."""
        return 1.0 if self.primary_topology == PrimaryTopology.FULL_BRIDGE else 0.5

    @property
    def bridge_device_count(self) -> int:
        return 4 if self.primary_topology == PrimaryTopology.FULL_BRIDGE else 2

    @property
    def bridge_series_devices(self) -> int:
        return 2 if self.primary_topology == PrimaryTopology.FULL_BRIDGE else 1

    def validate(self) -> None:
        errors: list[str] = []
        if not (0 < self.vbus_hold_end_v <= self.vbus_min_normal_v <= self.vbus_nom_v <= self.vbus_max_v):
            errors.append("bus voltage ordering must be hold_end <= min_normal <= nominal <= maximum")
        if self.vout_v <= 0 or self.pout_w <= 0:
            errors.append("output voltage and power must be positive")
        if not (0 < self.minimum_frequency_hz < self.maximum_frequency_hz):
            errors.append("frequency range is invalid")
        if not (self.minimum_frequency_hz <= self.resonant_frequency_hz <= self.maximum_frequency_hz):
            errors.append("resonant frequency must lie inside the switching range")
        if self.ln_ratio <= 1.0:
            errors.append("Ln=Lm/Lr must exceed 1")
        if self.q_full_load <= 0:
            errors.append("full-load Q must be positive")
        if self.primary_turns <= 0 or self.secondary_turns <= 0:
            errors.append("transformer turns must be positive integers")
        if self.primary_parallel_devices < 1 or self.sr_parallel_devices_per_position < 1:
            errors.append("parallel device counts must be >= 1")
        if not self.transformer_core_families or not self.resonant_inductor_core_families:
            errors.append("magnetic core-family filters cannot be empty")
        if self.litz_strand_copper_diameter_m <= 0:
            errors.append("Litz strand copper diameter must be positive")
        if not (0.0 <= self.litz_transposition_quality <= 1.0):
            errors.append("Litz transposition quality must be within 0..1")
        if self.litz_max_harmonic < 1:
            errors.append("Litz maximum harmonic must be >= 1")
        if self.magnetic_waveform_samples < 128:
            errors.append("magnetic waveform samples must be >= 128")
        if errors:
            raise ValueError("; ".join(errors))
