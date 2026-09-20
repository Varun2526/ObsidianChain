# ObsidianChain — Documentation Index
**Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
**Repository Layer:** `docs/` (System Specifications, Analytical Architecture & Operating Procedures)

---

## 1. Documentation Overview

This index provides a direct navigational map of the technical documentation for ObsidianChain. Every document outlines a distinct subsystem, design decision, or operational procedure without redundancy.

---

## 2. Directory Index

### System Architecture & Ingestion
- [Architecture](file:///Users/varun/dev/obsidianchain/docs/ARCHITECTURE.md)  
  Explains the end-to-end multi-stage pipeline, data flow from ingestion to alert ranking, and backend package boundaries across the platform.
- [System Workflow](file:///Users/varun/dev/obsidianchain/docs/SYSTEM_WORKFLOW.md)  
  Details the 19-step casework lifecycle from investigator authentication and dataset validation to review sign-off, archival, and Merkle proof verification.
- [Data & Ingestion](file:///Users/varun/dev/obsidianchain/docs/DATA_AND_INGESTION.md)  
  Documents supported input formats (CSV, JSON, XML), canonical schemas, validation rules, type coercion policies, and strict truth-label isolation.

### Intelligence & Forensic Analytics
- [ML Pipeline](file:///Users/varun/dev/obsidianchain/docs/ML_PIPELINE.md)  
  Specifies the production 30-feature supervised Random Forest model, isotonic calibration, MAD anomaly scoring, structural heuristics, and model-selection history.
- [Network Evidence](file:///Users/varun/dev/obsidianchain/docs/NETWORK_EVIDENCE.md)  
  Explains transaction propagation telemetry, observer vantage points, P2P network limitations (NAT/VPN/Tor), and evidentiary abstention safeguards.

### Security & Operational Integrity
- [Security & RBAC](file:///Users/varun/dev/obsidianchain/docs/SECURITY_AND_RBAC.md)  
  Covers role-based access control (Admin, Investigator, Reviewer), scrypt password derivation, case boundary isolation, and tamper-evident Merkle bundle export.
- [Production Freeze](file:///Users/varun/dev/obsidianchain/docs/PRODUCTION_FREEZE.md)  
  Provides the authoritative freeze record for SIH 2026, including model hashes, schema versions, frozen pipeline stages, and verification guarantees.

### Research & Empirical Decisions
- [Research Decisions](file:///Users/varun/dev/obsidianchain/docs/RESEARCH_DECISIONS.md)  
  Details the empirical progression from research questions to production safeguards, covering clustering, change heuristics, temporal splits, and rejected designs.
- [Research Directory](file:///Users/varun/dev/obsidianchain/research/README.md)  
  Outlines the isolated research directory layout, benchmarking scripts, ablation experiments, and canonical dataset reproduction pipelines.

### Operations & Deployment
- [Local Demo Setup](file:///Users/varun/dev/obsidianchain/docs/LOCAL_DEMO_SETUP.md)  
  Provides reproducible instructions to reset the SQLite database, generate dynamic temporary credentials, and run the complete offline demo stack.

---

## 3. Historical Engineering Records

Earlier development specifications, preliminary milestone audits, and historical walkthroughs are preserved in [`docs/archive/`](file:///Users/varun/dev/obsidianchain/docs/archive/):
- [PHASE4_ARCHITECTURE_AUDIT.md](file:///Users/varun/dev/obsidianchain/docs/archive/PHASE4_ARCHITECTURE_AUDIT.md): Initial architectural gap analysis and additive safety fixes.
- [PHASE5_ARTIFACT_API_MAP.md](file:///Users/varun/dev/obsidianchain/docs/archive/PHASE5_ARTIFACT_API_MAP.md): Artifact-to-API contract mapping during early backend implementation.
- [PHASE6_SPEC.md](file:///Users/varun/dev/obsidianchain/docs/archive/PHASE6_SPEC.md): Preliminary specification for earlier pipeline iterations.
- [TECHNICAL_WRITEUP.md](file:///Users/varun/dev/obsidianchain/docs/archive/TECHNICAL_WRITEUP.md): Initial technical write-up prior to the PS-native Random Forest model-selection freeze.
- [latency_walkthrough.md](file:///Users/varun/dev/obsidianchain/docs/archive/latency_walkthrough.md): Export serialization benchmark and Merkle profiler notes.

