# LLC Design Tool V8.0 / V8.1 Implementation Notes

**Version:** 8.1.0  
**Date:** 2026-09-03  
**Implemented scope:** V8.0 architecture refactor + V8.1 multi-fidelity electrical solver  
**Deferred scope:** V8 magnetics enhancement, nonlinear device parasitics, digital SR timing/loss, interleaved LLC and robustness analysis

## 1. Delivery objective

This delivery changes the LLC electrical-analysis core from a collection of model-specific outputs into a stable multi-fidelity interface:

```text
LLC work-point request
        │
        ├── FHA adapter
        ├── nonlinear multi-harmonic HB
        └── switched periodic TD adapter
        │
        ▼
common convergence + metrics + Golden Waveform
        │
        ├── GUI comparison
        ├── CLI comparison/export
        └── future ZVS / SR / magnetics / multiphase modules
```

The V7 APIs and design pages are retained. V8.1 adds new modules rather than replacing the existing tank, operating-point, dynamic-phasor or switched-model implementations.

## 2. V8.0 common analysis contracts

New package:

```text
llc_design/analysis/
```

The central data structures are:

- `LLCAnalysisRequest`: one physical work point, including bus voltage, load fraction, optional fixed frequency, target output voltage and waveform resolution.
- `FidelityLevel`: `fha`, `harmonic_balance`, `switched_time_domain`.
- `SolverConvergence`: convergence flag, residual, iteration count, method and message.
- `ModelMetrics`: common switching frequency, gain, power, phase, current and capacitor-stress metrics.
- `LLCModelResult`: waveform, metrics, convergence, warnings, diagnostics and native solver result.
- `MultiFidelityAnalysis`: all enabled results, selected reference fidelity and signed model errors.

This interface is the V8 extension point for later magnetics, SR, ZVS and interleaved engines. Those modules should consume one `LLCModelResult`/Golden Waveform instead of reconstructing separate approximate currents.

## 3. Common Golden Waveform schema

FHA and HB results are reconstructed into the same `WaveformBundle` node set used by existing GUI/export code. Core traces include:

\[
v_{bridge}(t),\quad i_r(t),\quad v_{Cr}(t),\quad i_m(t),
\]

\[
i_{pri,load}(t)=i_r(t)-i_m(t),\quad i_{sec}(t),\quad v_o(t)
\]

and the existing ideal gate/VDS/SR visualization channels.

The output network is solved as a periodic frequency-domain RC network. Therefore the exported output-node waveforms satisfy, sample by sample within numerical tolerance,

\[
i_{rect}(t)=i_{load}(t)+i_{Cout}(t)+i_{ESR}(t).
\]

This replaces a constant-load-current integration shortcut and prevents artificial output-voltage drift in a periodic steady-state waveform.

## 4. FHA adapter

`llc_design/analysis/fha.py` wraps the existing V7 FHA tank and operating-point functions. It remains the fast design model and compatibility baseline.

It supports:

- fixed-frequency evaluation;
- regulated-frequency solution using the existing operating-point search;
- common waveform/metrics output;
- explicit convergence and diagnostics.

No V7 FHA equations were silently changed.

## 5. Nonlinear multi-harmonic harmonic balance

### 5.1 Harmonic state

For selected odd harmonics \(h\in\{1,3,5,\ldots,H_{max}\}\), the solver represents peak complex phasors using

\[
x(\theta)=\Re\left\{\sum_h X_h e^{jh\theta}\right\}.
\]

At each harmonic,

\[
I_{r,h}=\frac{V_{bridge,h}-V_{p,h}}
{R_s+jh\omega L_r+\frac{1}{jh\omega C_r}},
\]

\[
I_{m,h}=\frac{V_{p,h}}{R_m+jh\omega L_m},
\]

\[
I_{load,h}=I_{r,h}-I_{m,h}.
\]

### 5.2 Rectifier self-consistency

The primary-referred rectifier clamp is not assigned independent fixed harmonic amplitudes. It is generated from the polarity of the reconstructed load current:

\[
v_p(\theta)=n\left(V_o+V_d\right)
\operatorname{sgn}\!\left(i_{load}(\theta)\right).
\]

The voltage harmonics \(V_{p,h}\), current harmonics and commutation angles are therefore solved as one nonlinear problem. This avoids the incorrect procedure of independently solving H1/H3/H5/H7 linear circuits and simply adding the results.

The DC output-current balance is enforced simultaneously:

\[
\frac{1}{2\pi}\int_0^{2\pi}|i_{load}(\theta)|\,d\theta
=\frac{V_o}{nR_o}
\]

under the package's primary/secondary current convention.

### 5.3 Exact zero-crossing projection

For the exact continuous-clamp branch, V8.1:

1. reconstructs the finite trigonometric current polynomial;
2. uses dense samples only to bracket polarity-changing roots;
3. polishes every crossing with Brent's method;
4. integrates the Fourier coefficients of `sign(i)` analytically over constant-polarity intervals;
5. integrates the average absolute current analytically over the same intervals.

This removes FFT-grid quantization from rectifier commutation timing and provides a suitable basis for later SR timing work.

### 5.4 Numerical continuation and fallback

The nonlinear equations are initialized with a smooth `tanh(i/I_smooth)` rectifier projection. The regularized result seeds the exact zero-crossing solve. Harmonic order can be increased adaptively from lower order to the configured maximum.

At light-load/discontinuous topology transitions, the continuously clamped ideal-rectifier equations may cease to describe the physical branch or may have multiple nearby branches. In that case V8.1 can return the regularized solution **only with an explicit warning and diagnostic flag**. It is not labelled as an exact converged zero-crossing solution.

For applications where this fallback is unacceptable, set:

```python
HarmonicBalanceConfig(allow_regularized_fallback=False)
```

and handle `HarmonicBalanceConvergenceError`.

## 6. Switched periodic time-domain adapter

`llc_design/analysis/time_domain.py` adapts the existing piecewise switched LLC solver to the common V8 result contract.

It:

- uses the same `LLCAnalysisRequest` and tank data;
- supports fixed-frequency and regulated operating points;
- uses the existing periodic-orbit mismatch as its convergence residual;
- can optionally trim frequency around the seed solution;
- reconstructs common comparison metrics.

By default the HB and switched models use the resonant-capacitor ESR as the common series-damping baseline. An optional mode can derive switched-model damping from the existing aggregate conductive-loss analysis, but this is disabled for strict model-to-model electrical comparison.

The switched result is the current **ideal-topology** Golden reference. It is not yet a nonlinear-Coss, winding-capacitance or semiconductor-recovery model and must not be presented as SPICE-equivalent.

## 7. Multi-fidelity reference selection

`LLCGoldenSolver` runs the enabled solvers independently. Reference priority is:

```text
converged switched TD
    > converged nonlinear HB
    > converged FHA
```

A non-converged higher-fidelity trace never silently outranks a converged lower-fidelity result. Solver failures can be preserved as warnings while still returning usable lower-fidelity results; strict mode is available when callers prefer exceptions.

Signed errors are reported for:

- switching frequency;
- normalized gain;
- resonant-current RMS/peak;
- magnetizing-current RMS;
- secondary-current RMS;
- resonant-capacitor peak voltage.

## 8. CLI

### 8.1 HB waveform

```bash
python -m llc_design waveforms \
    --mode hb \
    --samples 1024 \
    --cycles 2 \
    --output output/llc_hb
```

### 8.2 FHA/HB/TD comparison

```bash
python -m llc_design model-compare \
    --vbus 400 \
    --load 1.0 \
    --max-harmonic 7 \
    --hb-samples 1024 \
    --td-samples 1024 \
    --output output/llc_v8_compare
```

Use `--skip-td` for a faster FHA/HB comparison. Use `--frequency` to evaluate all models at a fixed switching frequency instead of allowing each model to regulate independently.

The output directory contains:

- `model_comparison.csv`;
- `model_comparison.json`;
- `LLC_V8_model_comparison.md`;
- optional per-model waveform CSV/statistics/PNG/report files.

## 9. GUI

The LLC main window adds:

- a dedicated **FHA / HB / TD** tab;
- bus voltage, load, harmonic order, HB sample count and TD inclusion controls;
- convergence/reference/error table;
- solver warnings and diagnostics;
- a **多谐波 HB 波形** action in the waveform page and parameter dock.

The existing dynamic-phasor and detailed switched waveform modes remain available.

## 10. Regression tests added

`llc_design/tests/test_v8_multifidelity.py` validates:

1. harmonic phasor synthesis/extraction round-trip;
2. exact sign projection for a pure sine, including the expected \(4/\pi\) fundamental and \(2I_{pk}/\pi\) average absolute value;
3. full-load regulated exact HB convergence, H3/H5 content, output power balance and output-node KCL;
4. explicit light-load regularized-fallback reporting;
5. full-bridge and half-bridge HB regulation paths;
6. Golden reference selection and comparison export.

The final package also contains `TEST_REPORT_V8_1.txt` generated from the release-candidate build.

## 11. Known boundaries

The following are deliberately **not** claimed as completed in V8.1:

- nonlinear `Coss(V)`, `Qoss(V)` and deadtime commutation dynamics;
- transformer winding/interwinding capacitance and full leakage network;
- diode reverse recovery or SR MOSFET third-quadrant device model;
- discontinuous-rectifier complementarity/state-machine HB branch;
- the planned V8 high-frequency winding/core-loss model;
- SR timing/LUT/loss;
- two-phase 90° and three-phase 120° interleaved LLC pages;
- corner/sensitivity/measurement calibration.

These are later V8 stages and should consume the V8.0 common result/waveform interface.

## 12. Recommended next stage

The next implementation stage is the agreed V8 magnetics work:

1. derive primary and secondary harmonic current spectra directly from the Golden Waveform;
2. establish validated conductor geometry and winding-stack data structures;
3. implement DC resistance plus skin/proximity loss for round wire, Litz, foil and planar conductors;
4. improve nonsinusoidal core-loss calculation from actual \(B(t)\);
5. add leakage, winding capacitance, gap-fringing and lumped thermal models;
6. close the loop by feeding extracted parasitics back into the electrical solver.

This order prevents the magnetic-loss module from depending on a separate approximate sinusoidal current model.
