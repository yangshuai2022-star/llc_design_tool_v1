import math
import subprocess
from pathlib import Path
import numpy as np
from scipy import signal

from power_control_tools import (
    ControllerKind, DiscretizationMethod, FilterResponse, IIRFamily,
    design_controller, discretize_transfer_function, design_iir_filter,
    analyze_digital_filter, export_c99_filter, verify_c99_filter,
)


def test_tustin_pi_matches_scipy():
    a=design_controller(ControllerKind.PI,kp=0.7,ki=125.0)
    d=discretize_transfer_function(a,40_000,DiscretizationMethod.TUSTIN)
    b0,a0=a.arrays(); b,a,_=signal.cont2discrete((b0,a0),1/40_000,method='bilinear')
    b=np.asarray(b).reshape(-1); b/=a[0]; a=np.asarray(a)/a[0]
    np.testing.assert_allclose(d.b,b,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(d.a,a,rtol=1e-11,atol=1e-12)


def test_prewarped_tustin_hits_analog_at_warp_frequency():
    a=design_controller(ControllerKind.LEAD,gain=2.0,fz_hz=500.0,fp_hz=5000.0)
    fs=40_000.0; fw=3000.0
    d=discretize_transfer_function(a,fs,DiscretizationMethod.PREWARP_TUSTIN,prewarp_frequency_hz=fw)
    _,ha=signal.freqs(*a.arrays(),worN=[2*math.pi*fw])
    _,hd=signal.freqz(d.b,d.a,worN=[2*math.pi*fw/fs])
    assert abs(abs(ha[0])-abs(hd[0]))/abs(ha[0]) < 1e-10
    phase=lambda x: math.atan2(x.imag,x.real)
    assert abs(phase(ha[0])-phase(hd[0])) < 1e-10


def test_butterworth_analysis_has_pz_step_impulse():
    f=design_iir_filter(FilterResponse.LOWPASS,IIRFamily.BUTTERWORTH,sample_rate_hz=48_000,order=2,f1_hz=1080)
    r=analyze_digital_filter(f.digital,response_samples=200)
    assert r.stable
    assert len(r.poles)==2
    assert len(r.impulse)==200 and len(r.step)==200
    assert r.max_pole_radius < 1
    assert abs(r.dc_gain-1) < 1e-9


def test_notch_rejects_center():
    f=design_iir_filter(FilterResponse.NOTCH,IIRFamily.BUTTERWORTH,sample_rate_hz=40_000,f1_hz=1000,q=10)
    _,h=signal.freqz(f.digital.b,f.digital.a,worN=[2*np.pi*1000/40_000])
    assert abs(h[0]) < 1e-8


def test_c99_export_compiles(tmp_path):
    f=design_iir_filter(FilterResponse.LOWPASS,IIRFamily.BUTTERWORTH,sample_rate_hz=48_000,order=4,f1_hz=1080)
    out=export_c99_filter(f.digital,tmp_path,prefix='ADC_LPF')
    assert out.sections==2
    subprocess.run(['gcc','-std=c99','-Wall','-Wextra','-Werror','-c',str(out.source_path),'-I',str(tmp_path),'-o',str(tmp_path/'x.o')],check=True)
    verify=verify_c99_filter(f.digital,out,samples=128)
    assert verify.available and verify.passed
    assert verify.impulse_max_abs_error < 2e-5


def test_pi_integrator_is_marginal_but_exportable():
    from power_control_tools import StabilityClass
    a=design_controller(ControllerKind.PI,kp=0.7,ki=125.0)
    d=discretize_transfer_function(a,40_000,DiscretizationMethod.TUSTIN)
    assert d.stability_class == StabilityClass.MARGINAL
    assert d.implementable
    assert not d.stable


def test_pi_c99_export_is_allowed(tmp_path):
    a=design_controller(ControllerKind.PI,kp=0.7,ki=125.0)
    d=discretize_transfer_function(a,40_000,DiscretizationMethod.TUSTIN)
    out=export_c99_filter(d,tmp_path,prefix='PFC_VLOOP')
    verify=verify_c99_filter(d,out,samples=96)
    assert verify.available and verify.passed


def test_bessel_fir_moving_average_common_filter_families():
    from power_control_tools import design_fir_filter, design_moving_average
    bessel=design_iir_filter(FilterResponse.LOWPASS,IIRFamily.BESSEL,sample_rate_hz=40_000,order=2,f1_hz=1000)
    assert bessel.digital.stable
    fir=design_fir_filter(FilterResponse.LOWPASS,sample_rate_hz=40_000,num_taps=31,f1_hz=1000,window='hamming')
    assert fir.digital.stable and len(fir.digital.b)==31
    ma=design_moving_average(sample_rate_hz=40_000,length=16)
    assert abs(sum(ma.digital.b)-1.0)<1e-12


def test_general_third_order_discretizes_to_sos_and_c99(tmp_path):
    from power_control_tools import AnalogTransferFunction
    # Stable proper 3rd-order example; the exporter must implement it as SOS,
    # not a fragile monolithic direct-form 3rd-order section.
    a=AnalogTransferFunction((1.0, 100.0), (1.0, 6000.0, 8.0e6, 1.0e9), 'General 3rd')
    d=discretize_transfer_function(a,50_000,DiscretizationMethod.TUSTIN)
    assert d.sos.shape[0] == 2
    out=export_c99_filter(d,tmp_path,prefix='CTRL_3RD')
    verify=verify_c99_filter(d,out,samples=160)
    assert verify.available and verify.passed


def test_pi_uses_kp_ti_project_convention():
    kp=2.5; ti=0.004
    a=design_controller(ControllerKind.PI,kp=kp,ti_s=ti)
    np.testing.assert_allclose(a.numerator,[kp,kp/ti],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.denominator,[1.0,0.0],rtol=0,atol=0)


def test_pif_is_pi_followed_by_first_order_lpf():
    kp=1.7; ti=0.008; fp=7500.0
    a=design_controller(ControllerKind.PIF,kp=kp,ti_s=ti,lpf_pole_hz=fp)
    w=2*np.pi*fp
    expected_num=np.polymul([kp,kp/ti],[w])
    expected_den=np.polymul([1.0,0.0],[1.0,w])
    np.testing.assert_allclose(a.numerator,expected_num,rtol=1e-12)
    np.testing.assert_allclose(a.denominator,expected_den,rtol=1e-12)


def test_pid_and_pidf_tustin_are_implementable():
    pid=design_controller(ControllerKind.PID,kp=1.2,ti_s=0.01,td_s=2e-4)
    dz=discretize_transfer_function(pid,50_000,DiscretizationMethod.TUSTIN)
    assert dz.implementable
    pidf=design_controller(ControllerKind.PIDF,kp=1.2,ti_s=0.01,td_s=2e-4,lpf_pole_hz=12_000)
    df=discretize_transfer_function(pidf,50_000,DiscretizationMethod.TUSTIN)
    assert df.implementable


def test_type2_rc_matches_analytical_zero_pole_locations():
    r1=10e3; r2=47e3; c1=10e-9; c2=470e-12
    a=design_controller(ControllerKind.TYPE_II,type_input_mode='rc',r1_ohm=r1,r2_ohm=r2,c1_f=c1,c2_f=c2)
    z=np.roots(a.numerator); p=np.roots(a.denominator)
    wz=1/(r2*c1); wp=(c1+c2)/(r2*c1*c2)
    assert np.min(np.abs(z+wz)) < 1e-7*wz
    finite=[x for x in p if abs(x)>1e-9]
    assert len(finite)==1 and abs(finite[0]+wp)<1e-7*wp


def test_type3_rc_has_two_zeros_integrator_and_two_hf_poles():
    a=design_controller(ControllerKind.TYPE_III,type_input_mode='rc',r1_ohm=10e3,r2_ohm=47e3,r3_ohm=12e3,c1_f=10e-9,c2_f=470e-12,c3_f=1e-9)
    assert len(np.roots(a.numerator))==2
    p=np.roots(a.denominator)
    assert len(p)==3 and np.min(np.abs(p))<1e-6


def test_single_file_c99_export(tmp_path):
    a=design_controller(ControllerKind.PI,kp=0.7,ti_s=0.0056)
    d=discretize_transfer_function(a,40_000,DiscretizationMethod.TUSTIN)
    out=export_c99_filter(d,tmp_path,prefix='PFC_VLOOP')
    assert out.file_path.suffix=='.h'
    assert len(list(tmp_path.glob('pfc_vloop.*')))==1
    text=out.file_path.read_text()
    assert 'static inline float32_t PFC_VLOOP_Run' in text
    verify=verify_c99_filter(d,out,samples=96)
    assert verify.available and verify.passed
