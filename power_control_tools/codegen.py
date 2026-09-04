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
    file_path: Path
    sections: int

    # Backward-compatible aliases for callers from V8.3-alpha1. All three now
    # intentionally point to the single generated header-only file.
    @property
    def header_path(self) -> Path:
        return self.file_path

    @property
    def source_path(self) -> Path:
        return self.file_path

    @property
    def coefficients_path(self) -> Path:
        return self.file_path


@dataclass(frozen=True)
class C99Verification:
    available: bool
    passed: bool
    impulse_max_abs_error: float
    step_max_abs_error: float
    compiler: str = ""
    message: str = ""


def _ident(name: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_]+', '_', name.strip()).strip('_').upper()
    if not s:
        s = 'CTRL_FILTER'
    if s[0].isdigit():
        s = 'CTRL_' + s
    return s


def render_c99_single_file(digital: DigitalTransferFunction, *, prefix: str = "CTRL_FILTER") -> str:
    """Render one header-only C99 file using float32_t and DF2T/SOS.

    The exported coefficient convention is identical to the GUI:
      H(z) = (b0+b1 z^-1+b2 z^-2)/(1+a1 z^-1+a2 z^-2)
      y = b0*x + d1
      d1 = b1*x - a1*y + d2
      d2 = b2*x - a2*y
    """
    d = digital.normalized()
    P = _ident(prefix)
    sos = np.asarray(d.sos, dtype=float).copy()
    for row in sos:
        row[:3] /= row[3]
        row[4:] /= row[3]
        row[3] = 1.0

    lines = [
        f'#ifndef {P}_H',
        f'#define {P}_H',
        '',
        '#include <stdint.h>',
        '',
        '#ifndef POWER_CTRL_FLOAT32_T_DEFINED',
        'typedef float float32_t;',
        '#define POWER_CTRL_FLOAT32_T_DEFINED',
        '#endif',
        '',
        f'#define {P}_FS_HZ ({d.sample_rate_hz:.9e}F)',
        f'#define {P}_SECTIONS ({len(sos)}U)',
        '',
    ]
    for i, row in enumerate(sos, start=1):
        for label, val in zip(('B0','B1','B2','A1','A2'), (row[0],row[1],row[2],row[4],row[5])):
            lines.append(f'#define {P}_S{i}_{label} ({float(val):.9e}F)')
        lines.append('')
    lines += [
        f'typedef struct {{ float32_t d1; float32_t d2; }} {P}_SECTION_T;',
        f'typedef struct {{ {P}_SECTION_T section[{P}_SECTIONS]; }} {P}_STATE_T;',
        '',
        f'static inline void {P}_Reset({P}_STATE_T *state)',
        '{',
        '    uint32_t i;',
        '    if (state == (void *)0) { return; }',
        f'    for (i = 0U; i < {P}_SECTIONS; ++i)',
        '    {',
        '        state->section[i].d1 = 0.0F;',
        '        state->section[i].d2 = 0.0F;',
        '    }',
        '}',
        '',
        '/* DF2T/SOS implementation. Coefficient signs match displayed H(z). */',
        f'static inline float32_t {P}_Run({P}_STATE_T *state, float32_t x)',
        '{',
        '    float32_t y = x;',
        '    if (state == (void *)0) { return x; }',
    ]
    for i in range(1, len(sos)+1):
        lines += [
            '    {',
            '        const float32_t in = y;',
            f'        y = {P}_S{i}_B0 * in + state->section[{i-1}U].d1;',
            f'        state->section[{i-1}U].d1 = {P}_S{i}_B1 * in - {P}_S{i}_A1 * y + state->section[{i-1}U].d2;',
            f'        state->section[{i-1}U].d2 = {P}_S{i}_B2 * in - {P}_S{i}_A2 * y;',
            '    }',
        ]
    lines += ['    return y;', '}', '', '#endif', '']
    return '\n'.join(lines)


def export_c99_filter(digital: DigitalTransferFunction, destination: str | Path, *, prefix: str = "CTRL_FILTER") -> C99Export:
    """Export a single header-only C99 file.

    `destination` may be a directory (the file name is derived from prefix) or
    an explicit .h file path.
    """
    dest = Path(destination)
    P = _ident(prefix)
    if dest.suffix.lower() == '.h':
        path = dest
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f'{P.lower()}.h'
    text = render_c99_single_file(digital, prefix=P)
    path.write_text(text, encoding='utf-8')
    return C99Export(path, len(digital.sos))


def verify_c99_filter(digital: DigitalTransferFunction, exported: C99Export, *, samples: int = 160, compiler: str = "gcc") -> C99Verification:
    """Compile and execute generated header-only float32_t DF2T against Python."""
    cc = shutil.which(compiler)
    if cc is None:
        return C99Verification(False, False, float('nan'), float('nan'), compiler, 'C compiler not found')
    out = exported.file_path.parent
    P = _ident(exported.file_path.stem)
    harness = out / '_verify_filter.c'
    exe = out / '_verify_filter.exe'
    code = (
        f'#include <stdio.h>\n#include "{exported.file_path.name}"\n'
        'int main(void)\n{\n'
        f'    {P}_STATE_T s; unsigned int n; {P}_Reset(&s);\n'
        f'    for (n=0U;n<{samples}U;++n) {{ float32_t x=(n==0U)?1.0F:0.0F; printf("I %.9g\\n", (double){P}_Run(&s,x)); }}\n'
        f'    {P}_Reset(&s);\n'
        f'    for (n=0U;n<{samples}U;++n) {{ printf("S %.9g\\n", (double){P}_Run(&s,1.0F)); }}\n'
        '    return 0;\n}\n'
    )
    harness.write_text(code, encoding='utf-8')
    try:
        subprocess.run([cc, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror', str(harness), '-I', str(out), '-o', str(exe)], check=True, capture_output=True, text=True)
        raw = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
        imp = np.asarray([float(x.split()[1]) for x in raw if x.startswith('I ')], dtype=float)
        step = np.asarray([float(x.split()[1]) for x in raw if x.startswith('S ')], dtype=float)
        d = digital.normalized()
        ximp = np.zeros(samples); ximp[0] = 1.0
        xstep = np.ones(samples)
        refi = signal.lfilter(np.asarray(d.b), np.asarray(d.a), ximp)
        refs = signal.lfilter(np.asarray(d.b), np.asarray(d.a), xstep)
        ei = float(np.max(np.abs(imp - refi)))
        es = float(np.max(np.abs(step - refs)))
        scale = max(1.0, float(np.max(np.abs(refi))), float(np.max(np.abs(refs))))
        passed = max(ei, es) <= 2e-5 * scale
        return C99Verification(True, passed, ei, es, cc, 'PASS' if passed else 'float32_t response mismatch')
    except Exception as exc:
        return C99Verification(True, False, float('inf'), float('inf'), cc, str(exc))
