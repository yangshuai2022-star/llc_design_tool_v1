# LLC engineering validation status

The bundled `400 V -> 53 V / 3 kW` case is a reproducible software-regression baseline. Passing it confirms that the current Python implementation still produces the repository-controlled tank, operating-point and efficiency results within recorded tolerances. It does not validate component selection, magnetic loss accuracy, thermal performance, ZVS margin or hardware safety.

## Evidence levels

- `VERIFIED`: the named claim is reproduced from a traceable artifact within its stated scope.
- `UNKNOWN`: the required source, simulation or measurement artifact is absent or insufficient.

`verified_for_hardware` is separate from evidence status. A source transcription or software regression may be `VERIFIED` for that narrow scope while remaining unsuitable for a hardware release.

## Current matrix

| Evidence | Status | Hardware verified | Basis |
|---|---|---:|---|
| Python baseline calculation | VERIFIED | No | Recomputed from `llc_design/examples/baseline_400V_53V_3kW.json` and checked against repository-controlled tolerances |
| PLECS or LTspice correlation | UNKNOWN | No | No traceable external-simulation artifact is stored |
| Bench measurement correlation | UNKNOWN | No | No traceable measurement artifact is stored |
| Bundled MOSFET records | UNKNOWN | No | Synthetic reference parts, not manufacturer ordering codes |
| Bundled core geometry library | UNKNOWN | No | Generic family geometry without complete ordering-code sources |
| Bundled material curve fits | UNKNOWN | No | Original curves and fit records are not stored |
| PQ35/35 transformer presets | UNKNOWN | No | Named datasheet source exists, but helper geometry and full loss fit are not independently verified |

Run `llc_design.validation.validation_summary()` to obtain a JSON-safe report. `assert_hardware_release_ready()` intentionally raises while any bundled dataset remains unverified for hardware.

## Evidence required to close UNKNOWN items

External simulation evidence must record simulator/version, schematic or model revision, component models, input case, solver settings, exported measurements and comparison tolerances. Bench evidence must additionally record equipment, calibration state, probe location, operating conditions, raw captures and sample identity. Device and magnetic records require exact ordering codes, source document revisions, extraction or fitting method, units, temperature and tolerance conditions.

The current result is falsified if the baseline regression fails, if a claimed source cannot reproduce its recorded values, or if external simulation or measurement differs beyond an approved tolerance. Hardware release remains blocked until the exact selected records carry reviewed evidence and `verified_for_hardware` is explicitly set true.
