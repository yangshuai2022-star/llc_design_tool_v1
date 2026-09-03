import pytest

from llc_design.core.operating_point import solve_operating_point
from llc_design.core.spec import LLCDesignSpec
from llc_design.core.tank import design_tank
from llc_design.models.devices import DeviceDatabase
from llc_design.models.primary_bridge import primary_bridge_loss


def test_primary_losses_are_positive_and_zvs_holds():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = solve_operating_point(spec, tank, spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_primary(spec.primary_device)
    result = primary_bridge_loss(spec, tank, op, device)
    assert result.conduction_w > 0.0
    assert result.turnoff_w > 0.0
    assert result.gate_drive_w > 0.0
    assert result.zvs_charge_margin > 1.0
    assert result.zvs_energy_margin > 1.0
    assert result.voltage_stress_v == op.vbus_v


def test_parallel_primary_devices_split_current_and_conduction():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = solve_operating_point(spec, tank, spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_primary(spec.primary_device)
    single = primary_bridge_loss(spec, tank, op, device)
    parallel = primary_bridge_loss(
        spec.clone(primary_parallel_devices=2), tank, op, device)
    assert parallel.device_rms_current_a == pytest.approx(
        single.device_rms_current_a / 2.0)
    assert parallel.conduction_w == pytest.approx(single.conduction_w / 2.0)


def test_turnoff_loss_scales_with_bus_voltage():
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    device = DeviceDatabase().get_primary(spec.primary_device)
    at_high = primary_bridge_loss(spec, tank,
                                  solve_operating_point(spec, tank, 420.0, 1.0),
                                  device)
    at_low = primary_bridge_loss(spec, tank,
                                 solve_operating_point(spec, tank, 360.0, 1.0),
                                 device)
    assert at_high.turnoff_w > at_low.turnoff_w


def test_turnoff_loss_uses_commutation_current_not_resonant_peak():
    """LLC turns off on the magnetizing/commutation current, so Eoff must
    scale with op.commutation_current_a, not op.resonant_current_peak_a
    (the old 0.85*Ipk factor overstated turn-off loss by 38-71%)."""
    spec = LLCDesignSpec()
    tank = design_tank(spec)
    op = solve_operating_point(spec, tank, spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_primary(spec.primary_device)
    result = primary_bridge_loss(spec, tank, op, device)

    # Eoff_device = eoff_ref * (V/Vref) * (I/Iref) — verify against the
    # exact commutation current (including the safety-margin factor).
    i_used = op.commutation_current_a * spec.primary_turnoff_current_factor
    eoff = (device.eoff_ref_j * op.vbus_v / device.eoff_ref_v
            * i_used / device.eoff_ref_i)
    assert result.turnoff_w == pytest.approx(
        spec.bridge_device_count * eoff * op.switching_frequency_hz,
        rel=1e-9)
    # The margin factor is a genuine safety margin (>= 1), not a peak
    # scaling factor.
    assert spec.primary_turnoff_current_factor >= 1.0
    assert op.commutation_current_a < op.resonant_current_peak_a
    # The old behavior (0.85 * resonant peak) would give a larger Eoff.
    old_style = (spec.bridge_device_count
                 * op.resonant_current_peak_a * spec.primary_turnoff_current_factor
                 * device.eoff_ref_j * op.vbus_v / device.eoff_ref_v
                 / device.eoff_ref_i * op.switching_frequency_hz)
    assert result.turnoff_w < old_style
