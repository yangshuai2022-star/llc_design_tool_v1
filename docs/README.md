# Documentation Index

This directory contains the **maintained engineering documentation** for Power Design Toolkit.

The repository used to accumulate release snapshots, one-off migration notes, temporary audit reports and patch instructions in the project root. Those files were useful while a specific version was being developed, but they made it difficult to identify the current engineering contract. The documentation policy is now:

- `README.md` — product overview, quick start and current capability map.
- `CHANGELOG.md` — concise version history.
- `docs/` — current engineering architecture, model boundaries, validation, provenance, Agent integration and deployment.
- `reference_designs/` — traceable engineering cases and evidence matrices.
- `engineering_data/` — cross-workspace catalog for device/material/core data quality.
- GitHub Issues / Pull Requests / Actions / Releases — development history, CI logs and binary artifacts.
- `release_validation/` — numerical baselines that are part of regression evidence, not narrative documentation.

## Maintained documents

| Document | Scope |
| --- | --- |
| [DIGITAL_CONTROL_ARCHITECTURE.md](DIGITAL_CONTROL_ARCHITECTURE.md) | Exact H(z), mixed-domain loop model, sensing/ADC, FM/PWM, timing and C99 contract |
| [LLC_MODELING.md](LLC_MODELING.md) | FHA / HB / switched TD / shared-ngspice model hierarchy and appropriate use |
| [FRA_LOOP_DESIGNER.md](FRA_LOOP_DESIGNER.md) | FRA import semantics, controller de-embedding, stability metrics, Auto Design and model identification |
| [NGSPICE_CLOSED_LOOP.md](NGSPICE_CLOSED_LOOP.md) | Circuit IR, batch/shared ngspice architecture, exact digital-control execution and model boundary |
| [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md) | Evidence levels and what software regression does or does not prove |
| [PROJECT_PROVENANCE.md](PROJECT_PROVENANCE.md) | Common project JSON provenance/evidence envelope and falsification rules |
| [ENGINEERING_DATA.md](ENGINEERING_DATA.md) | Device, core and material database provenance/release policy |
| [AGENT_MCP.md](AGENT_MCP.md) | MCP v2 Agent interface and deterministic-kernel boundary |
| [WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md) | FastAPI/web architecture, local execution and deployment |
| [../reference_designs/README.md](../reference_designs/README.md) | Reference-design evidence contract and current cases |
| [../CHANGELOG.md](../CHANGELOG.md) | Current concise version history |

## Compatibility documents

A small number of historical paths are intentionally kept as short redirect files because the in-application Help/F1 system in V9.2.2 links to them. They are **not** independent sources of truth. New documentation should link directly into `docs/`.

The compatibility paths will be removed after the Help document links are migrated in a future UI change.

## Documentation rule

A document belongs in the maintained set only if it answers one of these questions:

1. What can the current product do?
2. How is a current engineering algorithm implemented?
3. What assumptions/model boundaries apply?
4. How is a current feature installed, validated or deployed?
5. What traceable evidence would prove or falsify an engineering claim?

Per-version debugging logs, temporary patches, wheel files and CI result snapshots do not belong in the source documentation tree.
