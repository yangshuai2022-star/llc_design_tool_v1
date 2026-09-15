# LLC_400V_53V_3kW

This case promotes the bundled `400 V -> 53 V / 3 kW` LLC baseline into a traceable Reference Design.

## Canonical input

Do not duplicate or silently edit the case parameters here.  The authoritative input is:

`llc_design/examples/baseline_400V_53V_3kW.json`

Key nominal values are 400 V bus, 53 V output, 3 kW, 100 kHz resonant frequency, full-bridge primary, and full-bridge synchronous rectification.

## Evidence chain

```text
canonical input
    -> Python regression
    -> internal FHA/HB/switched-TD comparison
    -> real ngspice execution path
    -> bench measurement: UNKNOWN
    -> hardware release: NOT APPROVED
```

The machine-readable authority for the statuses above is `evidence.json`.

## What is verified

- The input case is version controlled.
- Repository-controlled Python regression is reproducible within its maintained numerical tolerances.
- Internal multi-fidelity comparison artifacts exist and are regression-controlled.
- Real ngspice/libngspice execution is exercised by CI for simulator plumbing.

## What is not verified

- Vendor MOSFET nonlinear Coss/Qrr/switching-loss accuracy.
- Release-quality magnetic material and thermal accuracy.
- Insulation/safety margin.
- Bench correlation for this exact case.
- Hardware-release readiness.

## Falsification

Any VERIFIED item must be downgraded when its artifact no longer reproduces, a higher-fidelity result contradicts it outside tolerance, or future bench evidence contradicts it.  Missing evidence remains `UNKNOWN`; it is never inferred from adjacent tests.
