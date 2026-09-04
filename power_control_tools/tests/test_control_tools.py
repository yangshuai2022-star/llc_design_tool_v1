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
