from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from scipy import signal
from .models import DigitalTransferFunction


@dataclass(frozen=True)
class C99Export:
    header_path: Path
    source_path: Path
    coefficients_path: Path
    sections: int


@dataclass(frozen=True)
class C99Verification:
    available: bool
    passed: bool
    impulse_max_abs_error: float
    step_max_abs_error: float
    compiler: str = ""
    message: str = ""


def _ident(name: str) -> str:
    s=re.sub(r'[^A-Za-z0-9_]+','_',name.strip()).strip('_').upper()
    if not s: s='CTRL_FILTER'
    if s[0].isdigit(): s='CTRL_'+s
    return s


def export_c99_filter(digital: DigitalTransferFunction, directory: str|Path, *, prefix: str="CTRL_FILTER") -> C99Export:
    d=digital.normalized(); out=Path(directory);out.mkdir(parents=True,exist_ok=True); P=_ident(prefix); p=P.lower()
    sos=np.asarray(d.sos,dtype=float)
    # scipy rows are b0,b1,b2,a0,a1,a2
    for row in sos:
        row[:3]/=row[3]; row[4:]/=row[3]; row[3]=1.0
    coeff=out/f'{p}_coeff.h'; hdr=out/f'{p}.h'; src=out/f'{p}.c'
    lines=['#ifndef '+P+'_COEFF_H','#define '+P+'_COEFF_H','',f'#define {P}_FS_HZ ({d.sample_rate_hz:.9e}F)',f'#define {P}_SECTIONS ({len(sos)}U)','']
    for i,row in enumerate(sos):
        for label,val in zip(('B0','B1','B2','A1','A2'),(row[0],row[1],row[2],row[4],row[5])):
            lines.append(f'#define {P}_S{i+1}_{label} ({float(val):.9e}F)')
        lines.append('')
    lines += ['#endif','']
    coeff.write_text('\n'.join(lines),encoding='utf-8')
    hdr.write_text(f'''#ifndef {P}_H\n#define {P}_H\n\n#include <stdint.h>\n#include "{p}_coeff.h"\n\n#ifndef POWER_CTRL_FLOAT32_T_DEFINED\ntypedef float float32_t;\n#define POWER_CTRL_FLOAT32_T_DEFINED\n#endif\n\ntypedef struct {{ float32_t d1; float32_t d2; }} {P}_SECTION_T;\ntypedef struct {{ {P}_SECTION_T section[{P}_SECTIONS]; }} {P}_STATE_T;\n\nvoid {P}_Reset({P}_STATE_T *state);\nfloat32_t {P}_Run({P}_STATE_T *state, float32_t x);\n\n#endif\n''',encoding='utf-8')
    c=['#include <stddef.h>',f'#include "{p}.h"','',f'static const float32_t kB0[{P}_SECTIONS] = {{'+','.join(f'{v[0]:.9e}F' for v in sos)+'};',f'static const float32_t kB1[{P}_SECTIONS] = {{'+','.join(f'{v[1]:.9e}F' for v in sos)+'};',f'static const float32_t kB2[{P}_SECTIONS] = {{'+','.join(f'{v[2]:.9e}F' for v in sos)+'};',f'static const float32_t kA1[{P}_SECTIONS] = {{'+','.join(f'{v[4]:.9e}F' for v in sos)+'};',f'static const float32_t kA2[{P}_SECTIONS] = {{'+','.join(f'{v[5]:.9e}F' for v in sos)+'};','',f'void {P}_Reset({P}_STATE_T *state)','{','    uint32_t i;','    if (state == NULL) { return; }',f'    for (i = 0U; i < {P}_SECTIONS; ++i) {{ state->section[i].d1 = 0.0F; state->section[i].d2 = 0.0F; }}','}','','/* DF2T: y=b0*x+d1; d1=b1*x-a1*y+d2; d2=b2*x-a2*y. */',f'float32_t {P}_Run({P}_STATE_T *state, float32_t x)','{','    uint32_t i;','    float32_t y = x;','    if (state == NULL) { return x; }',f'    for (i = 0U; i < {P}_SECTIONS; ++i)','    {','        const float32_t in = y;','        y = kB0[i] * in + state->section[i].d1;','        state->section[i].d1 = kB1[i] * in - kA1[i] * y + state->section[i].d2;','        state->section[i].d2 = kB2[i] * in - kA2[i] * y;','    }','    return y;','}','']
    src.write_text('\n'.join(c),encoding='utf-8')
    return C99Export(hdr,src,coeff,len(sos))


def verify_c99_filter(digital: DigitalTransferFunction, exported: C99Export, *, samples: int = 160, compiler: str = "gcc") -> C99Verification:
    """Compile and execute generated float32_t DF2T against Python reference."""
    cc = shutil.which(compiler)
    if cc is None:
        return C99Verification(False, False, float("nan"), float("nan"), compiler, "C compiler not found")
    out = exported.source_path.parent
    P = _ident(exported.header_path.stem)
    harness = out / "_verify_filter.c"
    exe = out / "_verify_filter.exe"
    code = (
        f'#include <stdio.h>\n#include "{exported.header_path.name}"\n'
        'int main(void)\n{\n'
        f'    {P}_STATE_T s; unsigned int n; {P}_Reset(&s);\n'
        f'    for (n=0U;n<{samples}U;++n) {{ float32_t x=(n==0U)?1.0F:0.0F; printf("I %.9g\\n", (double){P}_Run(&s,x)); }}\n'
        f'    {P}_Reset(&s);\n'
        f'    for (n=0U;n<{samples}U;++n) {{ printf("S %.9g\\n", (double){P}_Run(&s,1.0F)); }}\n'
        '    return 0;\n}\n'
    )
    harness.write_text(code, encoding="utf-8")
    try:
        subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", str(exported.source_path), str(harness), "-I", str(out), "-o", str(exe)], check=True, capture_output=True, text=True)
        raw = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
        imp = np.asarray([float(x.split()[1]) for x in raw if x.startswith("I ")], dtype=float)
        step = np.asarray([float(x.split()[1]) for x in raw if x.startswith("S ")], dtype=float)
        d = digital.normalized()
        ximp = np.zeros(samples); ximp[0] = 1.0
        xstep = np.ones(samples)
        refi = signal.lfilter(np.asarray(d.b), np.asarray(d.a), ximp)
        refs = signal.lfilter(np.asarray(d.b), np.asarray(d.a), xstep)
        ei = float(np.max(np.abs(imp - refi)))
        es = float(np.max(np.abs(step - refs)))
        scale = max(1.0, float(np.max(np.abs(refi))), float(np.max(np.abs(refs))))
        passed = max(ei, es) <= 2e-5 * scale
        return C99Verification(True, passed, ei, es, cc, "PASS" if passed else "float32_t response mismatch")
    except Exception as exc:
        return C99Verification(True, False, float("inf"), float("inf"), cc, str(exc))
