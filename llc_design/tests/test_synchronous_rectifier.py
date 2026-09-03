import pytest

from llc_design.core.operating_point import solve_operating_point
from llc_design.core.spec import LLCDesignSpec
from llc_design.core.tank import design_tank
from llc_design.models.devices import DeviceDatabase
from llc_design.models.synchronous_rectifier import synchronous_rectifier_loss


def test_sr_losses_and_stress():
    spec = LLCDesignSpec()
    op = solve_operating_point(spec, design_tank(spec), spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_sr(spec.sr_device)
    result = synchronous_rectifier_loss(spec, op, device)
    assert result.conduction_w > 0.0
    assert result.coss_w > 0.0
    assert result.gate_drive_w > 0.0
    assert result.voltage_stress_v == pytest.approx(
        spec.sr_voltage_overshoot_factor * spec.vout_v)


def test_parallel_sr_devices_keep_conduction_constant():
    spec = LLCDesignSpec()
    op = solve_operating_point(spec, design_tank(spec), spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_sr(spec.sr_device)
    single = synchronous_rectifier_loss(spec, op, device)
    parallel = synchronous_rectifier_loss(
        spec.clone(sr_parallel_devices_per_position=3), op, device)
    assert parallel.device_rms_current_a < single.device_rms_current_a
    assert parallel.conduction_w < single.conduction_w


def test_sr_deadtime_diode_loss_decreases_with_less_deadtime():
    spec = LLCDesignSpec()
    op = solve_operating_point(spec, design_tank(spec), spec.vbus_nom_v, 1.0)
    device = DeviceDatabase().get_sr(spec.sr_device)
    wide = synchronous_rectifier_loss(spec.clone(sr_deadtime_s=150e-9), op, device)
    narrow = synchronous_rectifier_loss(spec.clone(sr_deadtime_s=50e-9), op, device)
    assert narrow.deadtime_diode_w < wide.deadtime_diode_w
