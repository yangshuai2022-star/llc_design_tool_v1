# Power Design Toolkit V9 — Digital Control & Closed-Loop Baseline

## 1. Scope

The formal V9 baseline removes the experimental GeckoCIRCUITS backend and concentrates on the Toolkit's native digital-control workflow:

`Control Tools -> exact H(z) -> small-signal plant -> sensing/sampler -> modulator -> closed-loop stability -> C99`.

No external Java or circuit simulator is required.

## 2. Canonical controller object

Control Tools owns the controller/filter design. Its discrete transfer function is

\[
C(z)=\frac{b_0+b_1z^{-1}+\cdots+b_mz^{-m}}
{1+a_1z^{-1}+\cdots+a_nz^{-n}}.
\]

The coefficient convention is identical in the GUI, LLC loop analysis and C99 DF2T/SOS export:

\[
y[k]=\sum_i b_i x[k-i]-\sum_{j=1}^n a_j y[k-j].
\]

V9 links the exact `b/a` arrays from Control Tools. It does not infer a new PI/PID from those coefficients.

## 3. LLC closed-loop chain

For a selected operating point, the LLC plant provides frequency-command to output-voltage small-signal dynamics. The complete loop is evaluated as

\[
L = C \cdot G_{FM} \cdot G_{vf} \cdot H_{sense} \cdot H_{ADC} \cdot H_{delay}.
\]

The analysis reports:

- open-loop Bode;
- gain crossover frequency;
- phase margin and gain margin;
- closed-loop complementary sensitivity \(T= L/(1+L)\);
- sensitivity \(S=1/(1+L)\);
- delay envelopes;
- discrete closed-loop poles and maximum pole radius;
- output-impedance suppression;
- FM command headroom and polarity warnings.

## 4. Sampler / ADC

The digital loop keeps the existing ADC/sampling model, including acquisition/conversion timing, multiple SOC samples, recursive averaging and the effective sample offset. The control sample time must equal the linked Control Tools sample time.

## 5. LLC frequency modulator

The FM block remains explicit rather than being hidden inside the plant. It supports PCMD-to-frequency and PCMD-to-TBPRD LUTs, timer-clock and Up/Up-Down counting semantics. The local slope at the operating point provides the small-signal modulator gain.

This is important because LLC frequency-control polarity is part of the loop sign.

## 6. Control Tools GUI linkage

When Control Tools recalculates a controller, `digital_design_updated` emits the exact digital transfer function. The workspace controller forwards it to the LLC window. The LLC Digital Loop page then:

1. displays the controller source;
2. enables `Use Control Tools current H(z)`;
3. selects it by default;
4. copies the Control Tools sample rate into the LLC control sample time;
5. passes the exact transfer function into `build_digital_loop_analysis`.

The user can uncheck the option to return to the LLC page's local PI/PIF/2P2Z controls.

## 7. C99

Control Tools exports one header-only C99 file using `float32_t` and DF2T/SOS. When the LLC loop is using an external Control Tools controller, the LLC loop page exports that same `H(z)` through the same single-file generator and verifies the generated C response against Python impulse/step responses when a C compiler is available.

## 8. Validation contract

The V9 regression suite includes a direct linkage test that designs a PIF in Control Tools, discretizes it, sends it into the LLC small-signal loop, and verifies:

- identical sample rate;
- identical numerator coefficients;
- identical denominator coefficients;
- finite complete-loop frequency response;
- computed discrete closed-loop poles;
- explicit Control Tools linkage warning/source metadata.

This test is intended to prevent future GUI refactors from silently breaking the controller-to-loop linkage.
