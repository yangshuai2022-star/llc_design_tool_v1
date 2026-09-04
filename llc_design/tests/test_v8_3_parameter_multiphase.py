import numpy as np
from llc_design.core.spec import LLCDesignSpec,TankParameterMode
from llc_design.core.tank import design_tank
from llc_design.multiphase import (
    MultiphaseTopology, solve_interleaved_electrical_point, solve_system_gain_curve,
    three_phase_equivalent_ac_load_ohm,
)


def test_user_defined_tank_is_never_resynthesized():
    s=LLCDesignSpec(parameter_mode=TankParameterMode.USER_DEFINED,user_lr_h=22.0e-6,user_cr_f=120e-9,user_lm_h=132e-6,primary_turns=30,secondary_turns=4)
    t=design_tank(s)
    assert t.lr_h==22e-6
    assert t.cr_f==120e-9
    assert t.lm_h==132e-6
    assert abs(t.fr_hz-1/(2*np.pi*np.sqrt(22e-6*120e-9)))<1e-9*t.fr_hz
    assert abs(t.ln_ratio-6.0)<1e-12


def test_multiphase_common_frequency_and_fixed_angles():
    s=LLCDesignSpec()
    a=solve_interleaved_electrical_point(s,2)
    b=solve_interleaved_electrical_point(s,3)
    assert a.phase_offsets_deg==(0.0,90.0)
    assert b.phase_offsets_deg==(0.0,120.0,240.0)
    assert abs(sum(p.output_power_w for p in a.phases)-s.pout_w)<1e-4
    assert abs(sum(p.output_power_w for p in b.phases)-s.pout_w)<1e-4
    assert max(abs(p.share_percent-50) for p in a.phases)<1e-6
    assert max(abs(p.share_percent-100/3) for p in b.phases)<1e-6


def test_same_user_tank_produces_distinct_2p_3p_system_gain_curves():
    # Freeze the physical tank instead of re-synthesizing it for each phase.
    base=design_tank(LLCDesignSpec())
    s=LLCDesignSpec(parameter_mode=TankParameterMode.USER_DEFINED,user_lr_h=base.lr_h,user_cr_f=base.cr_f,user_lm_h=base.lm_h)
    g2=solve_system_gain_curve(s,2,points=36)
    g3=solve_system_gain_curve(s,3,points=36)
    mask=np.isfinite(g2.normalized_gain)&np.isfinite(g3.normalized_gain)
    assert mask.sum()>20
    assert np.max(np.abs(g2.normalized_gain[mask]-g3.normalized_gain[mask]))>1e-3


def test_tank_mismatch_solves_non_equal_power_share_at_one_common_frequency():
    s=LLCDesignSpec()
    base=s.clone(pout_w=s.pout_w/2)
    t=design_tank(base)
    manual=s.clone(parameter_mode=TankParameterMode.USER_DEFINED,user_lr_h=t.lr_h,user_cr_f=t.cr_f,user_lm_h=t.lm_h)
    r=solve_interleaved_electrical_point(manual,2,phase_spec_overrides={1:{'user_cr_f':t.cr_f*1.03}})
    assert abs(r.phases[0].output_power_w-r.phases[1].output_power_w)>1.0
    assert 60e3 <= r.switching_frequency_hz <= 180e3


def test_three_phase_y_llc_uses_per_phase_parallel_rectifier_fha_load():
    n=4.0; rload=2.0
    expected=(24.0/np.pi**2)*n*n*rload
    assert abs(three_phase_equivalent_ac_load_ohm(n,rload)-expected)<1e-12*expected


def test_three_phase_auto_design_is_topology_specific_and_not_single_phase_fanout():
    s=LLCDesignSpec()
    two=solve_interleaved_electrical_point(s,2)
    three=solve_interleaved_electrical_point(s,3)
    assert two.topology == MultiphaseTopology.TWO_PHASE_PARALLEL_90
    assert three.topology == MultiphaseTopology.THREE_PHASE_STAR_120
    # 3P Y uses three half-bridge legs and synthesizes a different tank/turns
    # operating condition.  The phase tank must not be a copied 2P tank.
    assert abs(two.phases[0].tank.lr_h-three.phases[0].tank.lr_h) > 1e-6
    assert abs(three.switching_frequency_hz-three.phases[0].tank.fr_hz) < 0.02*three.phases[0].tank.fr_hz
    assert abs(three.total_output_power_w-s.pout_w) < 1e-6*s.pout_w


def test_three_phase_y_mismatch_activates_floating_neutral_coupling():
    s=LLCDesignSpec()
    nominal=solve_interleaved_electrical_point(s,3)
    # Freeze topology-specific auto values into the customer-validation mode,
    # then perturb one Cr to emulate a measured production mismatch.
    t=nominal.phases[0].tank
    # AUTO 3P selected an effective half-bridge turns ratio close to 15:4 for
    # the default specification.  Use that ratio explicitly in USER mode.
    manual=s.clone(parameter_mode=TankParameterMode.USER_DEFINED,
                   user_lr_h=t.lr_h,user_cr_f=t.cr_f,user_lm_h=t.lm_h,
                   primary_turns=15,secondary_turns=4)
    r=solve_interleaved_electrical_point(manual,3,phase_spec_overrides={2:{'user_cr_f':t.cr_f*1.03}})
    shares=np.asarray([p.share_percent for p in r.phases])
    assert np.ptp(shares)>1e-4
    # Star coupling should keep all phases participating rather than silently
    # collapsing the mismatched phase into an independent-cell solution.
    assert min(shares)>1.0


def test_user_defined_turns_ratio_is_preserved_in_multiphase_validation():
    s=LLCDesignSpec(parameter_mode=TankParameterMode.USER_DEFINED,
                    user_lr_h=22e-6,user_cr_f=120e-9,user_lm_h=132e-6,
                    primary_turns=17,secondary_turns=3)
    two=solve_interleaved_electrical_point(s,2)
    # 3P USER_DEFINED must not silently replace the customer's transformer ratio.
    three=solve_interleaved_electrical_point(s,3)
    assert all(p.primary_turns==17 and p.secondary_turns==3 for p in two.phases)
    assert all(p.primary_turns==17 and p.secondary_turns==3 for p in three.phases)
    assert all(abs(p.turns_ratio-17/3)<1e-12 for p in three.phases)


def test_three_phase_waveform_uses_topology_specific_model_not_phase_fanout():
    from llc_design.multiphase import solve_interleaved_llc
    from llc_design.analysis import FidelityLevel
    r=solve_interleaved_llc(LLCDesignSpec(),3,fidelity=FidelityLevel.HARMONIC_BALANCE,samples_per_cycle=256)
    assert r.waveform.metadata['electrical_model']=='three_phase_star_floating_neutral_fha'
    assert all(p.electrical is None for p in r.phases)
    assert any('single-phase' in w and 'fan-out' in w for w in r.warnings)
