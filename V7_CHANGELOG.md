# Power Design Toolkit V7

## V7.1.4 — LLC transformer datasheet-driven design

- Added a dedicated **变压器** LLC sub-workspace.
- Added ferrite-datasheet parameter entry for Ae, Amin, le, Ve, AL, effective permeability, winding area AN, mean turn length lN, AR and usable winding width.
- Added TDK PQ35/35 B65881A/B65882B N87 and N97 presets using the user-provided October 2022 datasheet geometry/reference values.
- Added automatic integer Np/Ns search from nominal conversion ratio plus selected Bpk/work-point constraints.
- Added selectable design scope: full range including hold-up, normal-bus range, or nominal full-load only.
- Added discrete Litz selection with default 0.10 mm copper strands and strand counts rounded in user-selectable steps (default 50: 50/100/150/...).
- Added current-density, bundle/sub-bundle, winding-layer, turns-per-layer, window-fill, radial-build, DCR, target AL and estimated-gap results.
- Reused the existing waveform-aware iGSE ferrite-loss and layered-Litz harmonic copper-loss engines for core/DC/skin/proximity/bundle/termination loss calculation.
- Added per-workpoint Bpk/current/core/copper/total-loss table, winding-stack sketch and nominal loss-breakdown plot.
- Added one-click application of the recommended turns into the main LLC project and JSON/CSV/text export.
- Added `python -m llc_design transformer-design` CLI.
- The PQ35/35 datasheet single-point Pv value is shown as a cross-check; the full core-loss surface still uses the bundled N87/N97 engineering reference fit and is clearly flagged as such.

## V7.1 — LLC GUI information-architecture cleanup

- LLC global design inputs moved from a permanent splitter into a dockable panel (`F4`).
- Added focus mode (`F9`) and a hidden-by-default run-log dock (`F8`).
- Removed duplicate run commands from the top toolbar; `Ctrl+R` runs the analysis owned by the current page.
- LLC workspace tabs shortened to Design Overview / Gain / Q-ZVS / Waveform / Small Signal / Digital Control.
- Digital-control signal-flow diagram reduced to a compact two-row header instead of consuming a large blank canvas.
- Digital-control parameters changed from one long scrolling form to a context-sensitive stacked inspector: Work Point, Controller, FM LUT, PWM/Delay, Analog Sense, ADC.
- Clicking a control block now selects the matching parameter page and matching Bode group.
- Added an inner inspector show/hide button so Bode can use nearly the full workspace width.
- Bode defaults to **open-loop stability only**; component/closed-loop views remain available through the view selector.
- Detailed transfer-function text moved to a separate result tab so it no longer permanently consumes plot height.
- Vout sensing schematic now wraps responsively in narrow parameter inspectors instead of being clipped.

V7 is a structural reconstruction rather than a page-level patch.

## Workspace architecture

- Startup selector separates **LLC Design** and **PFC Design**.
- PFC contains independent **Single-Phase TTPL** and **Three-Phase Vienna** sub-workspaces.
- Workspaces can be switched without destroying entered state.

## LLC

- Multi-load effective-Q analysis.
- Gain/Q map with real bus/load operating-point trajectory.
- Theoretical inductive ZVS region plus engineering Qoss/Coss/deadtime commutation margin.
- Primary-device Coss/Qoss and required-ZVS-margin controls.
- Interactive digital-control block diagram and Vout sensing schematic.
- Block-to-parameter/Bode linkage and cursor phase-budget reporting.

## Single-phase TTPL PFC

- Firmware-shaped two-loop architecture: 50 kHz current, 25 kHz AMC reference layer, 10 kHz voltage.
- Explicit Duty Feedforward and `indu_comp` paths in the graphical control diagram.
- Current/Vac/Vbus sensing schematics and frequency responses.
- PFC Bode defaults to open loop only; every trace is independently selectable.
- Reworked averaged AC-cycle solver with signed input current, physical bus-energy dynamics and strict harmonic analysis.
- Switching waveform derived from final settled AC-cycle workpoint.
- Dedicated zero-crossing analyzer.

## Three-phase Vienna PFC

- New topology-specific power-stage and modulation model sharing PFC-common control/sensing infrastructure.
- DC-voltage outer loop + three ABC stationary-frame current inner loops.
- Separate split-bus midpoint-balance auxiliary loop.
- Optional common-mode/third-harmonic injection and R/L inductor-voltage-drop feed-forward.
- Correct Vienna center-switch zero-state duty `D0 = 1-|m|`.
- Split-bus capacitor and averaged midpoint-current dynamics.
- Ia/Ib/Ic, Va/Vb/Vc and Vdc+/Vdc- sensing with optional gain/offset mismatch diagnostics.
- Current/Vdc/balance open-loop Bode pages plus sampling-chain Bode.
- Full three-phase AC-cycle, sector, PF/THD and workpoint-derived switching views.
- Switching reconstruction includes center gates, three-level phase voltages, current ripple, upper/lower diode currents and midpoint/split-bus currents.

## V7.1.3 — LLC digital-control diagram interaction

- Replaced the fixed auto-fit-only diagram behaviour with a zoomable/pannable vector view.
- Added Fit / 100% / Zoom Out / Zoom In / Full-screen controls to the LLC digital-control signal chain.
- Added blank-space double-click to restore fit-to-window.
- Added selected-path highlighting: the active block and directly connected signal path stay prominent while unrelated paths are dimmed.
- Added context-sensitive enlarged detail diagrams in the LLC digital-control parameter inspector.
- Added a full-screen signal-chain window; block selections made there update the main parameter/Bode view.
- Numerical LLC, TTPL and Vienna solvers are unchanged by this GUI-only revision.

## V7.1.5 — PFC stability / Vienna robustness

- TTPL PFC control diagram rebuilt on the same interactive vector framework as LLC:
  orthogonal routing, zoom/pan/fit/fullscreen, selected-path highlighting and Bode linkage.
- Added conservative one-click TTPL current-loop PI auto tuning.  The tuner uses the exact
  sampled controller + sensing + ZOH + delay model and checks a line/load/phase `indu_comp`
  envelope before accepting a recommendation.
- TTPL GUI now starts from a stable current-loop design baseline while retaining a button to
  restore the original firmware PI (`Kp=0.01`, `Ti=75 us`) for comparison.
- Fixed local TTPL switching-current reconstruction: current is integrated continuously across
  PWM periods rather than being reset to a triangular ripple template each period.
- Vienna control diagram rebuilt into a clean main double-loop row plus separate sensing,
  feed-forward and midpoint-balance support paths; added zoom/pan/fit/fullscreen controls.
- Added Vienna numerical-pipeline validators for Bode, AC-cycle and switching results.
- Vienna worker errors now identify the failing stage and the GUI error dialog preserves the
  complete traceback.  GUI-thread result/plot exceptions are caught as well.
- Added regression coverage for auto tuning, TTPL switching continuity and the complete
  Vienna GUI-equivalent calculation pipeline.
- Corrected a GUI-only delay-modeling error in both TTPL and Vienna sensing chains: ADC/SOC
  delay remains in the sense block, while controller-computation/PWM-update delay is owned by
  the firmware/PWM block.  The same delay is no longer counted twice in loop gain.
- Vienna default outer-voltage PI was made conservative (`Kp=2e-5`, `Ti=80 ms`), giving about
  53° phase margin with the default 400 Vac / 700 V / 10 kW example.

## V7.2 — C99 Float32 Control Code Generator

- Added portable C99/single-precision control code generation for LLC, single-phase TTPL PFC and three-phase Vienna PFC.
- Generated code exposes `Init`, `Reset` and `ControlStep` APIs and deliberately excludes BSP ADC/PWM/GPIO/interrupt configuration.
- Added compile-ready ISR integration templates with semantic inputs/outputs.
- Added stability-gate reports and design snapshots for traceability.
- TTPL generator includes 50 kHz current loop, 25 kHz AMC/reference/Duty feedforward and 10 kHz voltage loop using generated multi-rate scheduling.
- Vienna generator includes ABC current loops, outer voltage loop, midpoint balance, reference generation, inductor voltage-drop feedforward and three-level zero-state duty modulation.
- LLC generator includes selected PI/PIF/2P2Z controller and PCMD-to-frequency/TBPRD LUT interpolation.
- Fixed Vienna GUI `self.fsw` spinbox/Figure attribute collision by renaming the switching Figure to `self.fsw_fig`.
