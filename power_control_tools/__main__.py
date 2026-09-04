from __future__ import annotations
import json
from pathlib import Path
import click
from .controllers import design_controller
from .discretize import discretize_transfer_function
from .analysis import analyze_digital_filter
from .codegen import export_c99_filter, verify_c99_filter
from .filters import design_iir_filter, design_fir_filter, design_moving_average, design_dc_blocker
from .models import ControllerKind,DiscretizationMethod,FilterResponse,IIRFamily

@click.group()
def cli():
    """Digital power control utilities: S2Z, filter design, response analysis and C99 export."""


def _payload(d, r):
    return {
        'b': d.b,
        'a': d.a,
        'sos': d.sos.tolist(),
        'stability': r.stability_class.value,
        'implementable': d.implementable,
        'max_pole_radius': r.max_pole_radius,
        'dc_gain': r.dc_gain,
        'step_overshoot_percent': r.step_overshoot_percent,
        'settling_time_s': r.settling_time_s,
    }


@cli.command('s2z')
@click.option('--kind',type=click.Choice([v.value for v in ControllerKind]),default='pi')
@click.option('--fs',type=float,default=40000.0)
@click.option('--kp',type=float,default=1.0)
@click.option('--ki',type=float,default=100.0)
@click.option('--gain',type=float,default=1.0)
@click.option('--fz',type=float,default=100.0)
@click.option('--fp',type=float,default=10000.0)
@click.option('--fz2',type=float,default=1000.0)
@click.option('--fp2',type=float,default=20000.0)
@click.option('--fz3',type=float,default=3000.0)
@click.option('--fp3',type=float,default=30000.0)
@click.option('--numerator',type=str,default=None,help='General H(s) numerator, descending powers, comma separated')
@click.option('--denominator',type=str,default=None,help='General H(s) denominator, descending powers, comma separated')
@click.option('--method',type=click.Choice([v.value for v in DiscretizationMethod]),default='tustin')
@click.option('--prewarp',type=float,default=None)
@click.option('--export-dir',type=click.Path(path_type=Path),default=None)
@click.option('--prefix',type=str,default='POWER_CTRL')
def s2z(kind,fs,kp,ki,gain,fz,fp,fz2,fp2,fz3,fp3,numerator,denominator,method,prewarp,export_dir,prefix):
    kwargs=dict(kp=kp,ki=ki,gain=gain,fz_hz=fz,fp_hz=fp,fz1_hz=fz,fp1_hz=fp,fz2_hz=fz2,fp2_hz=fp2,fz3_hz=fz3,fp3_hz=fp3)
    if ControllerKind(kind) == ControllerKind.GENERAL:
        if numerator is None or denominator is None:
            raise click.UsageError('--kind general requires --numerator and --denominator')
        parse=lambda text:[float(v.strip()) for v in text.replace(';',',').split(',') if v.strip()]
        kwargs['numerator']=parse(numerator);kwargs['denominator']=parse(denominator)
    a=design_controller(kind,**kwargs)
    d=discretize_transfer_function(a,fs,method,prewarp_frequency_hz=prewarp)
    r=analyze_digital_filter(d,analog=a)
    payload=_payload(d,r)
    if export_dir:
        out=export_c99_filter(d,export_dir,prefix=prefix)
        verify=verify_c99_filter(d,out)
        payload['c99']={'source':str(out.source_path),'verification':verify.message,
                        'impulse_max_abs_error':verify.impulse_max_abs_error,
                        'step_max_abs_error':verify.step_max_abs_error}
    click.echo(json.dumps(payload,indent=2))


@cli.command('filter')
@click.option('--implementation',type=click.Choice(['iir','fir','moving_average','dc_blocker']),default='iir')
@click.option('--response',type=click.Choice([v.value for v in FilterResponse]),default='lowpass')
@click.option('--family',type=click.Choice([v.value for v in IIRFamily]),default='butterworth')
@click.option('--fs',type=float,default=40000.0)
@click.option('--order',type=int,default=2)
@click.option('--fc',type=float,default=1000.0)
@click.option('--f2',type=float,default=None)
@click.option('--q',type=float,default=10.0)
@click.option('--rp',type=float,default=1.0)
@click.option('--rs',type=float,default=40.0)
@click.option('--taps',type=int,default=31)
@click.option('--window',type=click.Choice(['hamming','hann','blackman']),default='hamming')
@click.option('--pole-radius',type=float,default=0.995)
@click.option('--export-dir',type=click.Path(path_type=Path),default=None)
@click.option('--prefix',type=str,default='POWER_FILTER')
def filter_cmd(implementation,response,family,fs,order,fc,f2,q,rp,rs,taps,window,pole_radius,export_dir,prefix):
    if implementation == 'iir':
        result=design_iir_filter(response,family,sample_rate_hz=fs,order=order,f1_hz=fc,f2_hz=f2,q=q,passband_ripple_db=rp,stopband_atten_db=rs)
    elif implementation == 'fir':
        result=design_fir_filter(response,sample_rate_hz=fs,num_taps=taps,f1_hz=fc,f2_hz=f2,window=window)
    elif implementation == 'moving_average':
        result=design_moving_average(sample_rate_hz=fs,length=taps)
    else:
        result=design_dc_blocker(sample_rate_hz=fs,pole_radius=pole_radius)
    d=result.digital; r=analyze_digital_filter(d)
    payload=_payload(d,r); payload.update({'family':result.family,'response':result.response,'description':result.description})
    if export_dir:
        out=export_c99_filter(d,export_dir,prefix=prefix); verify=verify_c99_filter(d,out)
        payload['c99']={'source':str(out.source_path),'verification':verify.message,
                        'impulse_max_abs_error':verify.impulse_max_abs_error,
                        'step_max_abs_error':verify.step_max_abs_error}
    click.echo(json.dumps(payload,indent=2))

if __name__=='__main__': cli()
