Power Design Toolkit V6 source package

Install core:
  python -m pip install -e .

Install GUI:
  python -m pip install -e ".[gui]"

Run tests:
  python -m pytest

Launch GUI:
  python -m llc_design gui

The GUI opens with a function selector:
  - LLC Design / Waveform / Digital Control
  - PFC Current Loop / Voltage Loop / Sensing / Waveforms

LLC and PFC are independent top-level workspaces. Each window has toolbar
actions for switching to the other workspace or returning to function selection.

PFC Control Lab:
  pfc-control-lab --output output/pfc_control_lab

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
  python -m llc_design waveforms --mode detailed --output output/waveforms_detailed
  python -m llc_design small-signal --sample-us 20 --output output/small_signal

Baseline LLC: 400 Vdc -> 53 V / 3 kW, full-bridge LLC + full-bridge SR.
MK2166-specific functions are not included.
