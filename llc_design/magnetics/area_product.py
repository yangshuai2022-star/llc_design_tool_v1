"""Transformer core sizing by the optimum area-product (Ap) method.

Implements the Hurley/McLyman optimum-transformer area-product method from
Chap5 of 《应用于电力电子技术的变压器和电感器理论设计和应用》 (see
``BOOK_TRANSFORMER_INDUCTOR/Problem_5_1.m``), in SI units.

The book's worked example is a 50 Hz silicon-steel transformer; the formula
shell (B0, Ap1, Ap2, Jo) is frequency- and topology-agnostic once ``sumVA``
and ``Kv`` are supplied.  For the LLC resonant converter ``Kv = 4`` (square
wave, consistent with the ``4*N*Ae*f`` Faraday form already used in
``transformer.py``) and ``sumVA`` is taken from the *solved operating point*
rather than the book's 50 Hz topology factors — the LLC operating-point solver
already produces the real RMS winding voltages and currents.

This is an order-of-magnitude core-size pre-screen.  The empirical constants
``Kt, kc, kw, ka, h`` are geometry/thermal correlations fitted for a specific
core family; the final design still goes through the existing integer-turn /
Litz / harmonic-loss search in ``transformer_designer.py``.  The
``Problem_5_1.m`` regression in ``test_area_product.py`` validates the formula
transcription and SI unit handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.operating_point import LLCOperatingPoint, solve_operating_point
from ..core.spec import LLCDesignSpec
from ..core.tank import GainNotReachableError, TankDesign, design_tank
from .core import CoreDatabase, CoreSpec
from .material import MaterialDatabase
from .transformer_designer import FerriteCoreInput

MU0 = 4.0e-7 * math.pi
COPPER_RHO20 = 1.72e-8
COPPER_ALPHA = 0.00393

# Waveform factor for a square-wave-excited transformer (V = 4 f N B Ae).
# The book uses Kv = 4.44 for a sine wave; LLC bridges apply a square wave.
KV_SQUARE = 4.0


@dataclass(frozen=True)
class AreaProductResult:
    sum_va: float
    b0_t: float                 # optimum (uncapped) flux density
    b_design_t: float            # min(b0, b_sat_cap) — flux used for Ap
    ap1_m4: float                # first Ap estimate [m^4]
    ap2_m4: float                # Newton-corrected Ap [m^4] — use for matching
    ap_cm4: float                # ap2 in cm^4 (1 m^4 = 1e8 cm^4)
    jo_a_per_mm2: float | None   # heat-balance J; None until a core is matched
    intermediate: dict   # full intermediate terms for traceability

    @property
    def ap_target_m4(self) -> float:
        return self.ap2_m4


def area_product_sizing(
    sum_va: float,
    frequency_hz: float,
    *,
    kv: float,
    steinmetz_k: float,         # Kc: W/(m^3 * Hz^alpha * T^beta), same as project's k
    alpha: float,
    beta: float,
    b_sat_cap_t: float,         # design flux ceiling (use min(b0, cap) for Ap)
    temperature_c: float,
    delta_t_rise_c: float,
    kf: float = 0.95,
    ku: float = 0.40,
    kt: float = 48.2e3,
    kc: float = 5.6,
    kw: float = 10.0,
    ka: float = 40.0,
    h: float = 10.0,
    rho20: float = COPPER_RHO20,
    alpha20: float = COPPER_ALPHA,
) -> AreaProductResult:
    """Optimum area-product sizing, SI units (f Hz, B T, Ap m^4, J A/m^2).

    Follows Problem 5.1: ``B0`` is the loss-optimum flux; the design flux is
    ``min(B0, b_sat_cap)``; ``Ap1`` is the closed-form estimate and ``Ap2`` one
    Newton step correcting the core-loss/copper-loss balance.
    """
    if sum_va <= 0 or frequency_hz <= 0 or b_sat_cap_t <= 0:
        raise ValueError("sum_va, frequency_hz and b_sat_cap_t must be positive")
    # The optimum-flux / area-product derivation uses the cold (20 C) copper
    # resistivity, matching Problem 5.1 (the hot resistance only enters Jo,
    # computed separately in jo_from_heat_balance).  ``temperature_c`` is
    # retained on the signature for the Jo path and traceability.
    rho = rho20

    # Optimum flux density (Eq. 5.1 / Problem 5.1, Bo term).
    numer_b0 = (h * ka * delta_t_rise_c) ** (2.0 / 3.0)
    denom_b0 = (
        2.0 ** (2.0 / 3.0)
        * (rho * kw * ku) ** (1.0 / 12.0)
        * (kc * steinmetz_k * frequency_hz ** alpha) ** (7.0 / 12.0)
    )
    b0 = (numer_b0 / denom_b0) * ((kv * frequency_hz * kf * ku) / sum_va) ** (1.0 / 6.0)
    b_design = min(b0, b_sat_cap_t)

    # First area-product estimate (Ap1).
    ap1 = (
        math.sqrt(2.0) * sum_va
        / (kv * frequency_hz * b_design * kf * kt * math.sqrt(ku * delta_t_rise_c))
    ) ** (8.0 / 7.0)

    # Newton correction (Ap2): solve f(Ap) = a0*Ap^2 - a1*Ap^(7/4) + a2 = 0.
    a0 = (kc * steinmetz_k * frequency_hz ** alpha * b_sat_cap_t ** beta) / (rho * kw * ku)
    a1 = (h * ka * delta_t_rise_c) / (rho * kw * ku)
    a2 = (sum_va / (kv * frequency_hz * b_sat_cap_t * kf * ku)) ** 2
    f = a0 * ap1 ** 2 - a1 * ap1 ** (7.0 / 4.0) + a2
    df = 2.0 * a0 * ap1 - (7.0 / 4.0) * a1 * ap1 ** (3.0 / 4.0)
    ap2 = ap1 - f / df if df != 0 else ap1

    return AreaProductResult(
        sum_va=sum_va,
        b0_t=b0,
        b_design_t=b_design,
        ap1_m4=ap1,
        ap2_m4=ap2,
        ap_cm4=ap2 * 1e8,
        jo_a_per_mm2=None,
        intermediate={
            "rho_hot": rho, "a0": a0, "a1": a1, "a2": a2,
            "denom_b0": denom_b0, "numer_b0": numer_b0,
        },
    )


def jo_from_heat_balance(
    ap_m4: float,
    core_volume_m3: float,
    winding_volume_m3: float,
    *,
    frequency_hz: float,
    steinmetz_k: float,
    alpha: float,
    beta: float,
    b_t: float,
    delta_t_rise_c: float,
    temperature_c: float,
    h: float = 10.0,
    ka: float = 40.0,
    ku: float = 0.40,
    rho20: float = COPPER_RHO20,
    alpha20: float = COPPER_ALPHA,
) -> float:
    """Heat-balance optimum current density (Problem 5.1 ``Jo`` term) [A/m^2].

    ``Jo = sqrt((h*ka*sqrt(Ap)*dT - Vc*Kc*f^a*B^b) / (rho*Vw*ku))``.
    Requires the *matched* core's volume ``Vc`` and winding volume ``Vw``.
    """
    rho = rho20 * (1.0 + alpha20 * (temperature_c - 20.0))
    heat_in = h * ka * math.sqrt(ap_m4) * delta_t_rise_c
    core_loss = core_volume_m3 * steinmetz_k * frequency_hz ** alpha * b_t ** beta
    numer = heat_in - core_loss
    if numer <= 0:
        return 0.0
    return math.sqrt(numer / (rho * winding_volume_m3 * ku))


def sum_va_from_operating_point(op: LLCOperatingPoint, turns_ratio: float) -> float:
    """Winding VA sum from a solved LLC operating point.

    ``sumVA = Vp_rms*Ip_rms + Vs_rms*Is_rms`` where the secondary RMS winding
    voltage is ``Vp_rms / turns_ratio`` (the *transformer* turns ratio, NOT the
    resonant-tank gain ``achieved_gain`` which is ~1 at resonance).  The
    primary current already includes the magnetizing component, so the primary
    VA carries the reactive burden — this replaces the book's 50 Hz topology
    factors ``(1/kpp + 1/kps)*Po``.
    """
    vp = op.transformer_fundamental_rms_v
    vs = vp / turns_ratio if turns_ratio > 0 else 0.0
    return vp * op.resonant_current_rms_a + vs * op.secondary_current_rms_a


def _solve_nominal_op(spec: LLCDesignSpec, tank: TankDesign) -> LLCOperatingPoint:
    """Solve the nominal (Vbus_nom, full load) operating point for sumVA."""
    try:
        return solve_operating_point(spec, tank, spec.vbus_nom_v, 1.0)
    except GainNotReachableError:
        # Fall back to a lighter load if full-load nominal is not reachable
        # (should not happen for a valid spec, but keeps the recommender robust).
        return solve_operating_point(spec, tank, spec.vbus_nom_v, 0.5)


def ferrite_core_input_from_core_spec(core: CoreSpec) -> FerriteCoreInput:
    """Build a ``FerriteCoreInput`` (datasheet-oriented) from a library ``CoreSpec``.

    ``CoreSpec`` lacks the ungapped-AL / datasheet-loss reference fields that
    ``FerriteCoreInput`` carries; AL and mu_e are derived from the material and
    geometry, and the unused datasheet-reference fields are zeroed.
    """
    ae_m2 = core.ae_m2
    le_m = core.le_mm * 1e-3
    mu_r = core.mu_r if core.mu_r > 0 else 1500.0
    al_h = MU0 * mu_r * ae_m2 / le_m if le_m > 0 else 0.0
    return FerriteCoreInput(
        preset_key=f"AP_RECOMMENDED_{core.part_number}",
        manufacturer=core.manufacturer,
        part_number=core.part_number,
        shape=core.shape,
        material_key=core.material_key,
        material_grade=core.material,
        ae_mm2=core.ae_mm2,
        amin_mm2=core.amin_mm2,
        le_mm=core.le_mm,
        ve_mm3=core.ve_mm3,
        sigma_l_over_a_per_mm=0.465,
        al_nh=al_h * 1e9,
        mu_e=mu_r,
        winding_area_mm2=core.aw_mm2,
        mean_turn_length_mm=core.mlt_primary_mm,
        usable_winding_width_mm=core.window_width_mm,
        ar_uohm=0.0,
        core_mass_g=core.core_mass_g,
        thermal_resistance_k_per_w=core.thermal_resistance_k_per_w,
        datasheet_loss_ref_w=0.0,
        datasheet_loss_ref_frequency_hz=100_000.0,
        datasheet_loss_ref_b_t=0.200,
        datasheet_loss_ref_temperature_c=100.0,
    )


@dataclass(frozen=True)
class CoreRecommendation:
    core: FerriteCoreInput
    core_spec: CoreSpec
    sizing: AreaProductResult
    jo_a_per_mm2: float
    candidate_area_products_cm4: tuple[tuple[str, float], ...]   # (part, ae*aw cm^4)
    notes: tuple[str, ...]


def recommend_core_by_area_product(
    spec: LLCDesignSpec,
    *,
    material_key: str = "TDK_N87_REF",
    margin: float = 1.15,
    delta_t_rise_c: float = 55.0,
    b_sat_cap_t: float | None = None,
    families: tuple[str, ...] | None = None,
    temperature_c: float | None = None,
    kt: float = 48.2e3,
    kc: float = 5.6,
    kw: float = 10.0,
    ka: float = 40.0,
    h: float = 10.0,
    ku: float = 0.40,
    kf: float = 0.95,
) -> CoreRecommendation:
    """Recommend the smallest library core whose area product covers the Ap target.

    The sizing uses ``material_key``'s Steinmetz coefficients; the matched
    core may be of a different material (the pool is filtered by family, not
    material), so the result is an order-of-magnitude pre-screen.
    """
    mat = MaterialDatabase().get(material_key)
    temp_c = spec.winding_temperature_c if temperature_c is None else temperature_c
    b_cap = spec.transformer_max_b_t if b_sat_cap_t is None else b_sat_cap_t
    fams = families if families is not None else spec.transformer_core_families
    fsw_hz = spec.resonant_frequency_hz

    # Solve a nominal operating point on the spec's own tank to obtain real
    # winding RMS voltages and currents for sumVA.
    tank = design_tank(spec)
    op = _solve_nominal_op(spec, tank)
    sum_va = sum_va_from_operating_point(op, spec.turns_ratio)

    coeff = mat.coefficients_at(temp_c)
    sizing = area_product_sizing(
        sum_va,
        fsw_hz,
        kv=KV_SQUARE,
        steinmetz_k=coeff.k,
        alpha=coeff.alpha,
        beta=coeff.beta,
        b_sat_cap_t=b_cap,
        temperature_c=temp_c,
        delta_t_rise_c=delta_t_rise_c,
        kf=kf, ku=ku, kt=kt, kc=kc, kw=kw, ka=ka, h=h,
    )
    ap_target = sizing.ap2_m4 * margin

    db = CoreDatabase()
    candidates = db.for_purpose("transformer", fams)
    if not candidates:
        candidates = db.for_purpose("transformer")
    if not candidates:
        raise ValueError("no transformer-purpose cores in the library")

    scored = []
    for core in candidates:
        ap_cm4 = core.ae_mm2 * core.aw_mm2   # mm^4; convert below
        ap_m4 = ap_cm4 * 1e-12               # mm^4 -> m^4 (1 mm^4 = 1e-12 m^4)
        scored.append((core, ap_m4, ap_cm4))
    scored.sort(key=lambda row: row[1])

    feasible = [row for row in scored if row[1] >= ap_target]
    chosen_row = feasible[0] if feasible else scored[-1]
    chosen_core, chosen_ap_m4, chosen_ap_cm4 = chosen_row

    # Jo from the matched core's actual volume and winding volume.
    vc = chosen_core.ve_mm3 * 1e-9
    mlt_m = 0.5 * (chosen_core.mlt_primary_mm + chosen_core.mlt_secondary_mm) * 1e-3
    vwind = chosen_core.aw_mm2 * 1e-6 * mlt_m   # window area * mean turn length
    jo = jo_from_heat_balance(
        chosen_ap_m4, vc, vwind,
        frequency_hz=fsw_hz,
        steinmetz_k=coeff.k, alpha=coeff.alpha, beta=coeff.beta,
        b_t=sizing.b_design_t, delta_t_rise_c=delta_t_rise_c,
        temperature_c=temp_c, h=h, ka=ka, ku=ku,
    )

    notes = []
    if chosen_ap_m4 < ap_target:
        notes.append(
            f"largest library core {chosen_core.part_number} (Ap={chosen_ap_cm4:.0f} mm^4) "
            f"is below Ap target {ap_target*1e12:.0f} mm^4; consider a larger core."
        )
    notes.append(
        f"sized with {material_key} Steinmetz (k={coeff.k:.3g}, a={coeff.alpha:.3g}, "
        f"b={coeff.beta:.3g}); matched core material is {chosen_core.material}."
    )

    candidate_aps = tuple((c.part_number, apcm4) for (c, _ap, apcm4) in scored)
    return CoreRecommendation(
        core=ferrite_core_input_from_core_spec(chosen_core),
        core_spec=chosen_core,
        sizing=sizing,
        jo_a_per_mm2=jo / 1e6 if jo > 0 else 0.0,
        candidate_area_products_cm4=candidate_aps,
        notes=tuple(notes),
    )