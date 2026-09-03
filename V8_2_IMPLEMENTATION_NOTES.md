# LLC Design Tool V8.2 Implementation Notes

**Baseline:** V8.1 multi-fidelity package
**Patch target:** V8.1 -> V8.2
**Scope:** the seven functions requested after V8.1.

## Implemented

1. **Secondary diode reverse-recovery process**
   - `MosfetSpec` now carries optional `Qrr`, `trr`, reference current and third-quadrant parameters.
   - SR replay generates a triangular reverse-recovery current pulse whose area is the scaled `Qrr`.
   - `Prr = Qrr_cycle * Vblock * Fsw` is reported separately.

2. **SR MOSFET third-quadrant model**
   - Golden `Isec(t)` is replayed against the commanded SR gate window.
   - Forward channel, third-quadrant channel, body-diode, reverse-current, recovery, Coss and gate losses are separated.
   - Deliberately late turn-off can be analyzed by disabling the safe turn-off clamp in `SRTimingConfig`.

3. **DCM rectifier complementarity solver**
   - Hybrid states are `CLAMP_POS`, `OPEN`, `CLAMP_NEG`.
   - In `OPEN`, secondary current is exactly zero and `Ir == Im` is projected after each event step.
   - Clamp turn-on occurs only when the open-circuit transformer voltage exceeds `n*(Vo+Vdrop)`.
   - This is separate from the legacy `tanh()` smooth rectifier and is exposed as `solve_complementarity_time_domain()` / CLI `v8-dcm`.

4. **V8 Golden-waveform magnetics**
   - Uses actual Golden primary/secondary current and primary-voltage waveforms.
   - iGSE core loss and harmonic Litz stack loss share the electrical waveform.
   - Adds 1-D field-energy leakage estimate, primary-secondary capacitance estimate and thermal iteration.

5. **Digital SR Timing / LUT / Loss**
   - Physical zero crossings are interpolated from Golden `Isec(t)`.
   - Firmware timing is referenced to primary gate command edges (not post-deadtime bridge voltage).
   - Outputs `ON_DELAY`, `OFF_ADVANCE`, conduction time, reverse margin and loss breakdown.
   - `generate_sr_lut()` supports Vin x Load sweeps; `export_sr_lut_c99()` emits a DSP-friendly C99 table.
   - GUI page: `SR Timing / Loss`; CLI: `v8-sr`.

6. **2-phase interleaved LLC**
   - Fixed phase offsets only: **0° / 90°**.
   - Per-phase LLC is designed at `Psystem/2` and reuses the V8 electrical model.
   - Computes rectified-current superposition, output-capacitor RMS/ripple, input-bus ripple and current sharing.

7. **3-phase interleaved LLC**
   - Fixed phase offsets only: **0° / 120° / 240°**.
   - Same system metrics as 2-phase with three phase cells.
   - GUI page: `2P / 3P Interleaved`; CLI: `interleaved --phases 2|3`.

## Engineering boundaries

- Reverse recovery is an engineering charge/time model, not a semiconductor drift-diffusion model. Replace reference Qrr/trr with selected-device hot datasheet or double-pulse data before hardware release.
- Third-quadrant MOSFET loss uses temperature-corrected Rds plus optional offset; it does not attempt SPICE-level nonlinear body-diode/channel sharing.
- DCM complementarity topology constraints are exact; event time resolution is bounded by the configured samples/cycle. At 100 kHz and 1024 samples/cycle this is below 10 ns.
- Leakage and winding-capacitance estimates are geometry-based first-order values. Measured impedance should override them when available.
- Interleaved system analysis assumes a common commanded switching frequency. Component-mismatch overrides are supported, and frequency mismatch is explicitly warned.

## New modules

```text
llc_design/
  analysis/complementarity.py
  dynamics/complementarity.py
  switching/sr.py
  magnetics/v8.py
  multiphase/interleaved.py
  gui/widgets/sr_design_view.py
  gui/widgets/interleaved_view.py
```

## Validation

New V8.2 regression file:

```text
llc_design/tests/test_v8_sr_dcm_multiphase.py
```

It verifies Qrr loss, forced third-quadrant conduction, C99 LUT generation, exact DCM OPEN-state current constraint, Golden magnetics/parasitics and fixed interleaving/ripple cancellation.
