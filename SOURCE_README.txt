Power Design Toolkit V8.1 source package

Install core:
  python -m pip install -e .

Install GUI:
  python -m pip install -e ".[gui]"

Run tests:
  python -m pytest

Launch GUI:
  python -m llc_design gui

The GUI opens with a function selector:
  - LLC resonant converter design, magnetics, waveforms and digital control
  - Single-phase TTPL and three-phase Vienna PFC control workspaces

LLC and PFC are independent top-level workspaces. Each window has toolbar
actions for switching to the other workspace or returning to function selection.

PFC control labs:
  pfc-control-lab --output output/ttpl
  vienna-control-lab --output output/vienna

C99/Float32 control-code generation:
  python -m power_codegen --help

PFC Bode behavior:
  - Current-loop page defaults to Li open loop only.
  - Voltage-loop page defaults to Lv open loop only.
  - Every plant/controller/sensing/ZOH/closed-loop/sensitivity transfer function
    has an independent show/hide check box.

PFC waveform behavior:
  - The simulator uses multiple AC periods for settling.
  - The GUI displays the final complete AC period with power-stage and control signals.
  - A separate detailed page shows configurable local switching periods.

LLC command examples:
  python -m llc_design waveforms --mode fast --output output/waveforms_fast
  python -m llc_design waveforms --mode fha --output output/waveforms_fha
  python -m llc_design waveforms --mode hb --output output/waveforms_hb
  python -m llc_design waveforms --mode detailed --output output/waveforms_detailed
  python -m llc_design model-compare --output output/llc_v8_compare
  python -m llc_design small-signal --sample-us 20 --output output/small_signal

V8.1 notes:
  - FHA, nonlinear multi-harmonic HB and switched periodic TD use one comparison API.
  - Light-load HB branch fallback is reported explicitly in warnings/diagnostics.
  - Detailed implementation and current boundaries are in V8_IMPLEMENTATION_NOTES.md.

Baseline LLC: 400 Vdc -> 53 V / 3 kW, full-bridge LLC + full-bridge SR.
MK2166-specific functions are not included.
