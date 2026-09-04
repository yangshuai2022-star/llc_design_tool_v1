# Power Design Toolkit V9 Changelog

## V9.1.0 — Formal Digital Control Baseline

### Architecture

- Removed the experimental GeckoCIRCUITS/Java backend, bridge, GUI tab, runtime probes and packaging hooks from the formal V9 baseline.
- Retained the backend-independent digital-control runtime: sampler/ADC, controller execution, LLC FM/TBPRD modulator, event scheduling and linear closed-loop diagnostics.
- Kept the project license at `GPL-3.0-only`.

### Control Tools → LLC loop linkage

- Added a direct canonical `H(z)` bridge from Control Tools to the LLC Digital Loop page.
- The exact Control Tools numerator/denominator coefficients are used; there is no PI/PID parameter re-fit.
- The LLC small-signal sample time follows the linked Control Tools sample rate.
- Added a visible `Use Control Tools current H(z)` selector and active-source indication.
- The user may disable the external controller and return to the LLC page's local PI/PIF/2P2Z controls.

### Closed-loop analysis

- Preserved LLC power-stage small-signal linearization, sensing/ADC, FM LUT/TBPRD, PWM/ZOH timing and delay-envelope analysis.
- Complete-loop outputs include `L`, `T`, `S`, PM, GM, crossover, discrete closed-loop poles, pole radius and output-impedance suppression.

### C99

- Control Tools remains the canonical one-file `float32_t` DF2T/SOS C99 exporter.
- When the LLC Digital Loop uses a linked Control Tools controller, its C99 export uses the same single-file generator and numerical verification path.

### Regression

- Added a direct Control Tools PIF → LLC small-signal loop regression that compares exact `b/a` coefficients and sample rate and verifies complete-loop response/poles.
