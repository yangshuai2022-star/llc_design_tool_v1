# Reference Designs

Reference Designs are **traceable engineering cases**, not marketing examples.  Each case links a frozen input artifact to reproducible software evidence and explicitly records which validation stages are still unknown.

## Evidence contract

Every case must contain an `evidence.json` file with:

- a stable `case_id`;
- the canonical input artifact path;
- evidence items with `status`, `scope`, `artifact`, and `verified_for_hardware`;
- explicit model boundaries;
- falsification conditions;
- measurement status kept `UNKNOWN` until raw bench evidence and operating conditions are committed.

Allowed evidence status values are deliberately conservative:

- `VERIFIED` — reproducible from a traceable artifact **within the stated scope**;
- `UNKNOWN` — required source, simulation, or measurement evidence is absent or insufficient.

`VERIFIED` never implies `verified_for_hardware=true` unless the corresponding hardware evidence satisfies the release policy in `docs/ENGINEERING_VALIDATION.md`.

## Current cases

| Case | Purpose | Software evidence | Bench evidence | Hardware release |
| --- | --- | --- | --- | --- |
| `LLC_400V_53V_3kW` | Bundled LLC regression/reference workflow | VERIFIED within repository-controlled software scope | UNKNOWN | No |

## Required directory shape

```text
reference_designs/<CASE_ID>/
├── README.md
├── evidence.json
└── measurement/
    └── README.md
```

Large/raw datasets may be stored elsewhere when appropriate, but `evidence.json` must point to a reproducible artifact and describe its scope.

## Falsification rule

A reference-design claim must be downgraded from `VERIFIED` when its regression no longer reproduces, its source artifact cannot reproduce the stated value, a higher-fidelity result contradicts it outside the documented tolerance, or the operating condition leaves the evidence range.
