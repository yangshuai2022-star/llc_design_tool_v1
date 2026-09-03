# V8 Changelog

## V8.1.0 — 2026-09-03

### V8.0 architecture

- Added stable LLC multi-fidelity analysis contracts.
- Added common convergence, metric, diagnostics and Golden Waveform outputs.
- Kept the V7 FHA, operating-point, dynamic-phasor and switched modules compatible.
- Added common model-comparison and export infrastructure.

### V8.1 electrical solver

- Added self-consistent nonlinear odd-harmonic balance.
- Added exact trigonometric zero-crossing location and analytical rectifier-sign projection.
- Added smooth continuation seed and explicit light-load regularized fallback diagnostics.
- Added adaptive harmonic-order convergence.
- Added FHA/HB/switched-periodic reference selection and signed error metrics.
- Added periodic output RC-network reconstruction with sample-level KCL consistency.
- Added CLI `waveforms --mode fha|hb` and `model-compare`.
- Added GUI FHA/HB/TD comparison page and HB waveform command.
- Added V8 multi-fidelity regression tests.

### Deferred V8 work

- Magnetic winding/core/parasitic/thermal upgrade.
- Nonlinear semiconductor capacitance and switching loss.
- Digital SR timing, LUT and SR loss.
- Fixed 90° two-phase and fixed 120° three-phase interleaved LLC.
- Corner, sensitivity and measurement calibration.
