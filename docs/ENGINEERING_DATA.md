# Engineering data policy

Power Design Toolkit keeps device, magnetic-core, and material data separate from the equations that consume them.  A populated JSON record is **not** automatically a release-qualified component record.

The cross-workspace index is `engineering_data/catalog.json`.

## Evidence states

- `VERIFIED`: the stated data claim can be reproduced from a traceable source within its documented conditions.
- `UNKNOWN`: source revision, extraction method, operating range, tolerance, or other required evidence is missing or insufficient.
- `verified_for_hardware`: separate release gate; defaults to `false` and must never be inferred from a part number or source URL.

## Dataset-level provenance

A release-quality dataset should record:

- schema version;
- exact source document and revision;
- extraction/fitting method;
- units and reference conditions;
- temperature/frequency/flux/current validity range as applicable;
- min/typ/max or tolerance semantics;
- evidence status;
- hardware-verification state;
- a warning when the data are generic, fitted, synthetic, or reference-only.

LLC datasets already carry embedded metadata and are validated by `llc_design.validation.provenance`.  The current PFC datasets predate that contract; the catalog therefore keeps them `UNKNOWN` until equivalent provenance is normalized.

## Record-level overrides

An individual part/material may override dataset-level status only when its own provenance is at least as traceable as the dataset contract.  A record with a manufacturer-like name or a `source_url` is not sufficient by itself.

## Semiconductor records

For hardware-release use, record the exact manufacturer ordering code and the source/revision for any derived quantities such as:

- `RDS(on)` versus temperature;
- `Coss`/`Qoss` definition and voltage point;
- `Eon`/`Eoff` reference voltage, current, gate resistance, and temperature;
- diode/Qrr data;
- package/current/thermal limits.

Interpolation or scaling laws must state their assumptions.

## Magnetic records

For cores/materials, release-quality evidence should identify:

- exact core/material ordering code;
- effective geometry source and tolerance;
- permeability / DC-bias curve source;
- Steinmetz or loss-map source and fitting range;
- temperature/frequency/flux validity range;
- gap/bobbin/winding assumptions when they affect the result.

## Promotion rule

A dataset can move from `UNKNOWN` to `VERIFIED` only when the cited source can reproduce the stored value/fit within the stated conditions.  A higher-fidelity source or bench result that contradicts the stored data falsifies the previous verification and requires downgrade or correction.

## Current boundary

The catalog is intentionally conservative.  Its purpose is to make incomplete provenance visible now, so future database growth does not silently convert engineering placeholders into apparent component sign-off data.
