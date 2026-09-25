# ObsidianChain — Production Freeze Record

> **Superseded (2026-09-25).** The production model is **`ps_native_v5`,
> LightGBM** (31 features, schema `ps_native_features/5`, 300 trees), served
> from `data/models/ps_native/registry.json` (champion) with
> `ps_native_v5_fallback_no_g` as fallback. The Random Forest described
> below is the legacy `ps_native_v1`: registered, holding no role, and not
> used by any production path (`docs/audit/2026-09-25-system-truth-audit.md`
> section 7). Current record: the Models page, or `obsidianchain model list`.

**Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
**Repository Layer:** `docs/PRODUCTION_FREEZE.md` (Authoritative Engineering Baseline & State of Record)

---

## 1. Freeze Notice & System State

As of **September 20, 2026** (ahead of the SIH 2026 target of September 25, 2026), the analytical and operational core of **ObsidianChain is officially FROZEN**. 

No further modifications, hyperparameter tunings, feature alterations, model retraining, or pipeline refactorings may occur within the production boundary (`src/obsidianchain/` and `frontend/`). All production artifacts, model weights, and schemas are locked and verified by cryptographic digests.

---

## 2. Core Metadata & Cryptographic Hashes

| Attribute | Frozen Value |
| :--- | :--- |
| **System Version** | `1.0.0-sih2026.frozen` |
| **Freeze Date** | `2026-09-20` (Target: `2026-09-25`) |
| **Git Commit Reference** | `eeb5ffd403645232085b7f1d4c1b57a3a8be60dd` |
| **Database Schema** | SQLite Schema `v4` (`obsidianchain.sqlite3`) |
| **Model Version** | `ps_native_v1` |
| **Feature Schema Version** | `ps_native_features/1` |
| **Pipeline Conductor** | 17 Deterministic Stages (`orchestrator.py`) |

### Model Artifact Hashes (Directory: `data/models/ps_native/v1/`)

```json
{
  "schema": "obsidianchain.ps_model_manifest/1",
  "model_version": "ps_native_v1",
  "training_dataset_sha256": "72170b671a68856fd50ad80bed5d64e3159455cc62758d36ce289097e74a328d",
  "artifacts": {
    "model.joblib": "7d9c1b404872fd1b05ecb4ff99648904c89f6f5ee1502e6c2915bc6f4d7ad65d",
    "feature_schema.json": "8ef0f929e6a7bbf502e1dfc413b0688fb8b836e84bd0198f3c3cf1303f6dab5c",
    "calibration.json": "953a7f9075c16c5ce43ef3dbf2962e6fa6466ef27c2e7b905b909135edc9b810",
    "metrics.json": "a9dc5055271271049ec2badd3c192018d911c39b94451242951f2d2acf9279da"
  }
}
```

---

## 3. Supervised Model & Calibration Specification

### Production Architecture
- **Model Type:** `RandomForestClassifier` (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters:**
  ```python
  RandomForestClassifier(
      n_estimators=120,
      max_depth=14,
      class_weight="balanced_subsample",
      random_state=42,
      n_jobs=-1
  )
  ```
- **Calibration Engine:** `IsotonicRegression` fit strictly on the Validation split (timesteps 35–41).
  - Raw Brier Score: `0.04186` $\longrightarrow$ Calibrated Brier Score: `0.03612`

### Calibrated Severity Bands

| Band | Threshold | Target Precision | Validation Precision | Validation Support |
| :--- | :--- | :--- | :--- | :--- |
| **`CRITICAL`** | $\ge 0.6697$ | 90.0% | **90.0%** | 840 |
| **`HIGH`** | $\ge 0.2149$ | 75.0% | **75.0%** | 1,452 |
| **`MEDIUM`** | $\ge 0.1139$ | 50.0% | **50.0%** | 2,602 |
| **`LOW`** | $< 0.1139$ | — | — | Unflagged baseline |

### Authoritative Performance Metrics
- **Validation PR-AUC (Timesteps 35–41):** **0.5702** (vs. random baseline 0.0626) | **Precision@100: 99.0%**
- **Held-Out Test PR-AUC (Timesteps 42–49):** **0.2079** (vs. random baseline 0.0463) | **Precision@100: 97.0%**
- **Held-Out Test ROC-AUC:** **0.7367**

---

## 4. Frozen Analytical Pipeline (17 Stages)

The complete pipeline is coordinated by `obsidianchain.pipeline.orchestrator.ConductorOrchestrator`:
1. `Ingest`
2. `Validate & Deduplicate`
3. `GeoIP / ASN Provider`
4. `Blockchain Analysis`
5. `Network Analysis`
6. `Blockchain ↔ Network Correlation`
7. `Blockchain Transaction Graph`
8. `Entity Clustering (Union-Find)`
9. `Features & Compatibility Check`
10. `Supervised ML Risk`
11. `Unsupervised Anomaly (MAD per Address)`
12. `Peeling & Mixing Patterns`
13. `Evidence Fusion`
14. `Ranked Alerts`
15. `Forensic Explanations`
16. `Investigation Graph Projection`
17. `Reporting & Run Integrity Manifest`

---

## 5. Security & Access Control State

- **Authentication:** Bearer session tokens; standard-library `scrypt` ($N=2^{15}, r=8, p=1$, salt 16 bytes).
- **Institutional Roles:** `ADMIN`, `INVESTIGATOR`, `REVIEWER` with code-enforced capabilities (`src/obsidianchain/console/rbac.py`).
- **Casework Integrity:**
  - Append-only audit logging in `audit_events`.
  - Anti-circular Merkle tree generation (`integrity.py`) with domain separation (`0x00` leaf, `0x01` branch).
  - Merkle inclusion proofs for case items and `X-Bundle-SHA256` HTTP download verification.
- **Air-Gapped Assurance:** Operates with `--network none`; zero external calls; zero remote CDNs.

---

## 6. Demarcation of Research vs. Production

| Repository Component | Production Status | Operational Boundary |
| :--- | :--- | :--- |
| `src/obsidianchain/` | **PRODUCTION (FROZEN)** | Production analytical engine, API, console, and security. |
| `frontend/` | **PRODUCTION (FROZEN)** | Single-page forensic application with zero external CDN assets. |
| `data/models/ps_native/v1/` | **PRODUCTION (FROZEN)** | Calibrated Random Forest model weights and manifest. |
| `data/reference/` | **PRODUCTION (FROZEN)** | Offline GeoIP and ASN mapping tables. |
| `research/experiments/` | **RESEARCH ONLY** | Feature ablation scripts (`ablation_study.py`). |
| `research/reports/` | **RESEARCH ONLY** | Historical model validation reports. |
| `research/benchmarks/` | **RESEARCH ONLY** | Export latency benchmarks and profilers. |
| `research/reproduction/` | **RESEARCH ONLY** | Offline dataset builders and training pipelines. |

---

## 7. Known System Boundaries & Limitations

1. **Investigative Signals Only:** Predictions prioritize where human investigators focus attention; they do not establish legal guilt or replace judicial process.
2. **Network Telemetry Caveat:** Announcing peer IPs provide propagation context; they do not prove private key ownership or sender identity.
3. **No Live Sniffer:** The platform processes bulk forensic metadata captures; it is not a live network wiretap.
4. **Offline Reference Resolution:** Unregistered or private IP addresses correctly resolve to `NaN` rather than fabricated geographical locations.

---

## 8. Verification Results at Freeze

- **Backend Unit & Integration Tests:** **1,702 passed**, 4 skipped (`pytest tests/`)
- **Frontend Unit & Component Tests:** **101 passed** (`vitest`)
- **Frontend Production Build:** **Clean build** (`npm run build` in 935ms)
- **Offline Assurance Tests:** **Passed** (`tests/test_offline.py`, `tests/test_frontend_offline.py`)
- **Ground-Truth Isolation:** **Passed** (`tests/test_truth_isolation.py`, `tests/test_api_boundary.py`)
