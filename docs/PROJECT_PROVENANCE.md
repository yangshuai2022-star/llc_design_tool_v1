# Project JSON and provenance contract

Power Design Toolkit treats an exported engineering project as more than a bag of parameters.  A project should preserve enough provenance to answer four questions later:

1. **What exactly was calculated?**
2. **Which toolkit/model revision produced it?**
3. **Which source artifacts support the result?**
4. **What has and has not been verified for hardware?**

## Common envelope

New or migrated project formats should converge on the envelope defined in `schemas/project.schema.json` and validated by `llc_design.validation.project.validate_project_document`.

Minimum fields:

```json
{
  "schema_version": "1.0",
  "toolkit_version": "9.2.2",
  "workspace": "llc",
  "input": {},
  "results": {},
  "sources": [],
  "evidence": [],
  "model_boundaries": []
}
```

`created_utc` and source hashes are recommended when a project is written to disk.  Raw measurement files should remain raw; store a locator and checksum rather than rewriting evidence into an undocumented derived format.

## Evidence semantics

Only two default statuses are allowed:

- `VERIFIED`: reproducible from a traceable artifact within the stated `scope`;
- `UNKNOWN`: source/simulation/measurement evidence is absent or insufficient.

`verified_for_hardware` is a separate boolean.  It must never be inferred from `VERIFIED`; a software-regression claim can be VERIFIED while hardware verification remains false.

## Source records

A source record should at least contain:

- `id`: stable local identifier;
- `kind`: e.g. `repository`, `datasheet`, `measurement`, `simulation`, `standard`;
- `locator`: repository path, document identifier, or other reproducible locator;
- `sha256`: recommended for imported/raw artifacts.

## Model boundaries

Every project carrying calculated results should record the assumptions that can invalidate those results: fidelity level, linearization point, device-model limitations, sensing/timing assumptions, thermal/material data quality, and other applicable boundaries.

## Falsification

A result must be reclassified or regenerated when the source hash changes, the referenced regression fails, a higher-fidelity model contradicts it outside the approved tolerance, or new measured evidence contradicts the current claim.

This contract is intentionally topology-neutral so LLC, PFC, Vienna, FRA, control-tool, generated-code, and future Agent workflows can share one traceability model.
