from llc_design.core.spec import LLCDesignSpec
from llc_design.models.system import LLCSystemAnalyzer
from llc_design.optimization.sweep import LLCOptimizer, OptimizationConfig


def test_baseline_system_passes_v1_screen():
    result = LLCSystemAnalyzer().analyze(LLCDesignSpec())
    assert result.feasible
    assert not result.feasibility_reasons
    assert 0.96 < result.nominal.efficiency < 0.99
    assert result.bus_capacitor.hold_time_to_limit_s >= result.spec.requested_hold_time_s


def test_small_multivariable_sweep_returns_a_feasible_best_design():
    cfg = OptimizationConfig(
        ln_values=(5.0, 6.0), q_values=(0.30,),
        fr_values_hz=(100_000.0,), primary_turn_values=(30,),
        secondary_turn_values=(4,), primary_devices=("REF_650V_SIC_45M",),
        sr_parallel_values=(2,))
    result = LLCOptimizer().run(LLCDesignSpec(), cfg)
    assert len(result.table) == 2
    assert result.best_analysis is not None
    assert bool(result.table.iloc[0]["feasible"])


def test_half_bridge_mode_uses_half_turns_and_remains_solvable():
    from llc_design.core.spec import PrimaryTopology
    spec = LLCDesignSpec(primary_topology=PrimaryTopology.HALF_BRIDGE,
                         primary_turns=15)
    result = LLCSystemAnalyzer().analyze(spec)
    assert result.feasible
    assert result.nominal.operating_point.resonant_current_rms_a > 18.0
    assert result.nominal.efficiency < LLCSystemAnalyzer().analyze(LLCDesignSpec()).nominal.efficiency


def test_optimization_supports_custom_hold_up_voltage():
    spec = LLCDesignSpec().clone(vbus_hold_end_v=320.0, bus_capacitance_f=2200e-6)
    cfg = OptimizationConfig(
        ln_values=(5.0,), q_values=(0.35,), fr_values_hz=(100_000.0,),
        primary_turn_values=(30,), secondary_turn_values=(4,),
        primary_devices=("REF_650V_SIC_45M",), sr_parallel_values=(2,))
    result = LLCOptimizer().run(spec, cfg)
    row = result.table.iloc[0]
    assert bool(row["feasible"])
    assert row["f_hold_khz"] is not None and row["f_hold_khz"] > 0.0
    assert 0.0 < row["weighted_loss_w"] < 100.0


def test_work_point_weights_follow_spec_bus_levels():
    spec = LLCDesignSpec().clone(vbus_hold_end_v=320.0)
    assert LLCOptimizer._point_weight(spec, 320.0, 1.0) == 0.15
    assert LLCOptimizer._point_weight(spec, 400.0, 1.0) == 0.30
    assert LLCOptimizer._point_weight(spec, 400.0, 0.5) == 0.20
    assert LLCOptimizer._point_weight(spec, 400.0, 0.1) == 0.0
    assert LLCOptimizer._point_weight(spec, 380.0, 1.0) == 0.0
