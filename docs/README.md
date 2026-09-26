# ObsidianChain Documentation

Welcome to the technical documentation for **ObsidianChain**—an AI-powered monitoring, forensic analysis, and evidence-fusion platform for Bitcoin transaction traffic (SIH 2026 / NTRO Problem Statement 26146).

---

## Current System

The authoritative documentation suite for the current production architecture, ingestion pipeline, intelligence stack, security model, and deployment footprint:

- **[System Architecture](ARCHITECTURE.md)**  
  End-to-end 17-stage analytical pipeline, deterministic dataflow from multi-format ingestion to alert ranking, and backend package topology (`api/`, `console/`, `pipeline/`, `ml/`, `cluster/`, `features/`, `network/`, `io/`, `alerts/`).

- **[System Workflow](SYSTEM_WORKFLOW.md)**  
  19-step casework lifecycle state machine (`DRAFT` $\to$ `VALIDATING` $\to$ `ANALYZING` $\to$ `ACTIVE` $\to$ `SUBMITTED` $\to$ `IN_REVIEW` $\to$ `APPROVED` $\to$ `CLOSED` $\to$ `ARCHIVED`), role restrictions, and two-person sign-off protocol.

- **[Data & Ingestion](DATA_AND_INGESTION.md)**  
  Multi-format parser specifications (CSV, JSON, XML), 14-column canonical schema, normalization rules, and strict truth-label isolation.

- **[ML Pipeline](ML_PIPELINE.md)**  
  Multi-layer intelligence stack: 31-feature extraction, tree ensemble scoring, probability calibration, Median Absolute Deviation (MAD) anomaly detection, and structural heuristics (peeling chains, equal-output mixers).

- **[Feature Catalog](feature_catalog.md)**  
  Authoritative specification of all 31 core model features across functional groups and 6 network context features, with as-of-$t$ extraction rules and missingness semantics.

- **[Network Evidence & Telemetry](NETWORK_EVIDENCE.md)**  
  P2P gossip propagation realities (multi-hop diffusion, observer vantage variation, NAT/CGNAT, VPN/Tor relays), multi-origin relay analysis, and strict cannot-link evidentiary abstention rules.

- **[Security & RBAC](SECURITY_AND_RBAC.md)**  
  Institutional security model, role-based access control (`ADMIN`, `INVESTIGATOR`, `REVIEWER`), standard-library `scrypt` key derivation, case isolation, and tamper-evident Merkle bundle verification.

- **[Model Card](MODEL_CARD.md)**  
  Authoritative model card for the production ObsidianChain Risk Model (internal identifier: `ps_native_v5`) and fallback model, detailing feature schema, Platt calibration, TreeSHAP explanations, intended use, and known operational boundaries.

- **[Local Demo Setup](LOCAL_DEMO_SETUP.md)**  
  Step-by-step instructions to initialize a clean demo database (`make demo-reset`), generate dynamic high-entropy credentials, and run the complete offline backend and web console.

- **[Deployment & Runtime Architecture](DEPLOYMENT.md)**  
  Deployment footprint audit establishing the minimal ~9.7 MB production runtime package (`deploy-data/`) with offline DB-IP GeoIP resolution, container configuration, and operational requirements.

- **[Technical Write-Up](TECHNICAL_WRITEUP.md)**  
  Comprehensive technical report detailing the problem statement, system architecture, 17-stage pipeline, 31-feature model, graph analytics, network telemetry, evaluation results, and engineering validation.

---

## Demo & Visual Evidence

Comprehensive visual walkthroughs, high-resolution console captures, and presentation materials:

- **[Walkthrough Video](demo/obsidianchain_walkthrough.mp4)**  
  Full 5-minute narrated walkthrough video demonstrating investigator triage, the 17-stage analytical pipeline, P2P network correlation panel ($p = 5.5 \times 10^{-6}$), money-flow path traversal, reviewer sign-off, and admin Merkle audit export.
- **[Production UI Screenshots](screenshots/)**  
  High-resolution screenshots of the production web console:
  - [`login.png`](screenshots/login.png): Institutional authentication view
  - [`home.png`](screenshots/home.png): Case management dashboard
  - [`alert.png`](screenshots/alert.png): Ranked alert queue with severity badges
  - [`entity-full.png`](screenshots/entity-full.png): Comprehensive entity forensic detail
  - [`tx-full.png`](screenshots/tx-full.png): Transaction inspector with inputs/outputs
  - [`graph.png`](screenshots/graph.png): Interactive transaction graph visualization
  - [`path.png`](screenshots/path.png): Money-flow path traversal between entities
  - [`network.png`](screenshots/network.png): P2P network telemetry intelligence and propagation timing
  - [`correlation.png`](screenshots/correlation.png): Multimodal blockchain <-> network correlation panel ($p = 1.1 \times 10^{-19}$)
  - [`models-full.png`](screenshots/models-full.png): Model registry, champion status, and gate reports
  - [`review.png`](screenshots/review.png): Two-person sign-off and reviewer oversight workflow
  - [`audit.png`](screenshots/audit.png): Append-only immutable audit ledger with Merkle digest
  - [`rbac.png`](screenshots/rbac.png): Role-based access control and user administration

---

## Research Decisions

- **[Research Decisions & Empirical Journey](RESEARCH_DECISIONS.md)**  
  Detailed narrative documenting the empirical progression from initial research questions to production safeguards: co-spend clustering, change heuristics, temporal evaluation protocols, and rejected designs.
- **[Research Directory](../research/README.md)**  
  Master index of scientific experiments, synthetic benchmark suites, and reproduction scripts.

---

## Archive

Historical specifications, internal audits, and superseded planning documents are preserved in **[`archive/`](archive/)** for auditability and institutional memory:

- **[Architectural Decision Records (ADRs)](archive/decisions/)**: Formal decision records governing model pathways, feature evolution, and evaluation protocols (`0001`–`0004`).
- **[System Truth Audit](archive/audit/2026-09-25-system-truth-audit.md)**: Reality audit verifying exact code locations for network analysis, ML scoring, and scaling bounds.
- **[Network Layer Engineering Plan](archive/plans/2026-09-25-network-layer.md)**: Implementation plan tracking network layer engineering phases.
- **[Production Freeze Record](archive/PRODUCTION_FREEZE.md)**: Cryptographic baseline freeze record documenting digests and frozen stages.
- **[Deployed Acceptance Record](archive/DEMO_FREEZE_2026-09-25.md)**: 54/54 end-to-end acceptance run record and demo freeze verification.
- **[Results Register](archive/results_register.md)**: Register of valid, invalid, and holdout research metrics across iterations.
- **[Production Runbook](archive/runbook.md)**: Operational procedures and promotion gate protocols.
- **[Threat Model](archive/security.md)**: Early security threat model and test verification matrix.
- **[Data Contracts](archive/data_contract.md)**: Early code contract specifications.
- **[Production Gate Spec](archive/production_gate_spec.json)**: Automated gate criteria specification.
- **[Design System](archive/design-system.md)**: Frontend design tokens and styling guidelines.
- **Earlier Engineering Milestones**:
  - [`PHASE4_ARCHITECTURE_AUDIT.md`](archive/PHASE4_ARCHITECTURE_AUDIT.md)
  - [`PHASE5_ARTIFACT_API_MAP.md`](archive/PHASE5_ARTIFACT_API_MAP.md)
  - [`PHASE6_SPEC.md`](archive/PHASE6_SPEC.md)
  - [`TECHNICAL_WRITEUP.md`](archive/TECHNICAL_WRITEUP.md)
  - [`latency_walkthrough.md`](archive/latency_walkthrough.md)
