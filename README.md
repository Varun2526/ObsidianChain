# ObsidianChain

**AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
*SIH 2026 Problem Statement 26146 | Air-Gapped Forensic Platform*

---

## 1. Problem

Cryptocurrency transaction graphs are inherently pseudonymous, high-volume, and cross-jurisdictional. Financial intelligence units and law enforcement agencies face critical bottlenecks when investigating illicit flows (e.g., ransomware, mixing services, darknet markets, money laundering):

- **Transitive Clustering Contamination:** Traditional co-spend heuristics merge addresses transitively. A single incorrect merge welds unrelated entities together into massive, false "super-clusters."
- **Lack of Evidence Separation:** Existing tools conflate on-chain graph connectivity with real-world identity, asserting ownership without corroborating multi-layer evidence.
- **Air-Gap & Privacy Requirements:** Operational intelligence environments cannot upload proprietary case evidence or blockchain telemetry to third-party cloud APIs. Analysis must execute 100% offline.

---

## 2. What ObsidianChain Does

ObsidianChain is an offline institutional forensic intelligence platform. It transforms raw blockchain transaction streams and network gossip telemetry into prioritized, evidence-backed investigative leads:

```
Raw Telemetry (CSV/JSON/XML)
  ↓
Ingestion & Normalization
  ↓
Common-Input Co-Spend Clustering (Union-Find)
  ↓
Behavioural, Graph & Structural Pattern Feature Extraction
  ↓
Network Propagation Observation & Context
  ↓
Supervised Risk Scoring & Robust Outlier Detection
  ↓
Cluster Risk Aggregation & Alert Ranking
  ↓
Interactive Investigation Console (Graph, Timeline, Evidence Funnel)
  ↓
Independent Review, Disposition Sign-Off & Merkle Integrity Export
```

> [!IMPORTANT]
> **No Absolute Identity or Criminality Claims:** ObsidianChain produces calibrated risk scores, behavioral anomaly flags, and evidentiary trails. It explicitly does **not** claim that an IP address proves wallet ownership, nor does it assert definitive criminal guilt.

---

## 3. Core Design Principle

> **"Do not force an answer when the evidence does not support one."**

- **Evidentiary Abstention:** Where a feature cannot be computed or network observations are unavailable, the platform explicitly stamps `INSUFFICIENT_EVIDENCE` or `UNOBSERVED` rather than fabricating neutral default values (e.g., zeroes) that could be misread as measurements.
- **Multimodal Signal Separation:** On-chain cospend evidence (`BLOCKCHAIN_CONTEXT`), structural heuristics (`PATTERN_CONTEXT`), anomaly deviations (`ANOMALY_CONTEXT`), supervised predictions (`MODEL_SIGNAL`), and peer announcements (`NETWORK_CONTEXT`) are held in distinct, unmerged evidentiary blocks.
- **Immutability:** Analytical predictions and investigator decisions are recorded side-by-side. An investigator disposition never mutates an analytical risk score, and an analytical rerun never erases an investigator's notes.

---

## 4. Key Capabilities

- **Air-Gapped Operation:** Runs 100% offline with zero external network connectivity, zero external CDN dependencies, and pinned offline wheels.
- **PS-Native Supervised Risk Model:** Evaluates addresses as-of transaction timestamp $t$ across 30 behavioral, topological, and structural features.
- **Isotonic Calibration:** Raw classifier probabilities are calibrated to true empirical risk percentiles, eliminating artificial score inflation.
- **Robust Anomaly Detection:** Utilizes Median Absolute Deviation (MAD) robust Z-scoring to isolate heavy-tailed transaction velocity outliers.
- **Structural Pattern Detection:** Identifies deterministic transaction topology signatures including peeling chains and Equal-Output CoinJoin mixers.
- **Network Telemetry Integration:** Maps peer announcement dispersion without making unverified IP-to-wallet ownership claims.
- **Role-Based Casework & Review:** Full investigator, reviewer, and administrator lifecycle with independent two-person sign-off.
- **Merkle Tree Integrity:** Exports case bundles with anti-circular Merkle tree verification, inclusion proofs, and immutable audit trails.

---

## 5. Production ML

The production analytical engine is **strictly frozen**:

- **Production Supervised Model:** `RandomForestClassifier` (120 estimators, `max_depth=14`, `random_state=42`, `class_weight='balanced_subsample'`).
- **Probability Calibration:** Monotonic `IsotonicRegression` fit exclusively on the chronological validation split (timesteps 35–41).
- **Outlier Engine:** Non-parametric Median Absolute Deviation (MAD) anomaly detector.
- **Model Storage:** `data/models/ps_native/v1/model.joblib` verified at startup via SHA-256 against `manifest.json`.
- **Model Selection Research (Historical):** During architecture selection, Random Forest (Validation PR-AUC = 0.5702, Precision@100 = 99.0%) decisively outperformed Logistic Regression (0.2654) and LightGBM (0.5228). LightGBM remains strictly research exploration and is not part of production inference.

---

## 6. Network Evidence

ObsidianChain models network propagation telemetry to provide investigative context:

- **What Network Evidence Contributes:** Identifies peer observation diversity, announcement dispersion across autonomous systems (ASNs), and temporal clustering of transaction broadcasts.
- **What Network Evidence Does NOT Establish:**
  - An IP address does **not** identify a wallet owner.
  - An announcing peer is **not** assumed to be the transaction sender (due to Bitcoin P2P multi-hop gossip, Tor/VPN relays, and NAT/CGNAT multiplexing).
  - Identical broadcast origins do **not** imply common entity ownership.

All network data is presented under explicit evidentiary caveats as `NETWORK_CONTEXT`.

---

## 7. Security / Offline Design

- **Air-Gap Enforcement:** Docker containers execute with `--network none`. Frontend assets contain zero external Google Fonts, CDNs, or remote scripts.
- **RBAC & Isolation:** Server-side capability checks enforce strict isolation between `INVESTIGATOR`, `REVIEWER`, and `ADMIN` roles.
- **Append-Only Audit Ledger:** Every login, case transition, alert view, note addition, and export is recorded immutably in an append-only SQLite log.
- **Cryptographic Merkle Export:** Exported case packages compute a canonical Merkle tree over all case records, generating verifiable inclusion proofs and anti-circular hash bindings.

---

## 8. Repository Structure

```
obsidianchain/
├── src/obsidianchain/   # Core Python package: API, console, pipeline, ML, clustering
├── frontend/            # React 18 / Vite / TypeScript air-gapped web console
├── research/            # Isolated research experiments, reports, benchmarks & reproduction
├── scripts/             # Operational dataset verification tooling (make verify)
├── tests/               # Full test suite: 1,702 backend pytest + 101 frontend Vitest tests
├── docs/                # System, architectural, security, and operational documentation
├── data/                # Local data root: raw CSVs, models, SQLite state, run outputs
└── vendor/              # Vendored offline wheels and system debs (gitignored)
```

---

## 9. Quick Start

### Prerequisites
- Python 3.11+ (or Docker for air-gapped container execution)
- Node.js 18+ (for frontend console)

### Local Native Execution

1. **Install Backend:**
   ```bash
   pip install -e .
   ```

2. **Verify Dataset (Optional / if raw Elliptic++ files are present):**
   ```bash
   python3 scripts/verify_dataset.py --data-root data
   ```
   *(Or inside Docker container: `make verify`)*

3. **Reset Database to Clean Demo State:**
   ```bash
   make demo-reset
   ```
   *(Initializes clean schema v4, creates `admin`, `investigator`, and `reviewer` accounts, and prints temporary passwords.)*

4. **Launch Backend API (Port 8000):**
   ```bash
   python3 -m uvicorn obsidianchain.api.app:create_app --factory --port 8000
   ```

5. **Launch Frontend Console (Port 5173):**
   ```bash
   npm --prefix frontend run dev
   ```
   Open `http://localhost:5173` in your browser.

---

## 10. Demo Workflow

To perform a complete end-to-end demonstration from a clean state:

1. Run `make demo-reset` and note the temporary credentials.
2. Sign in at `http://localhost:5173` as `investigator`.
3. Create a new Investigation (`DRAFT`).
4. Upload `tests/data/synthetic_acceptance_capture.json` and validate (`VALIDATING`).
5. Progress case to `ANALYZING` and click **Run 17-Stage Analysis**.
6. View live pipeline progress across all 17 stages until `COMPLETE`.
7. Browse the ranked Alert Queue, select an alert, and inspect **Why Flagged**, **Feature Breakdown**, **Transaction Graph**, and **Timeline**.
8. Record an Investigator Note and submit case for review (`SUBMITTED`).
9. Log in as `reviewer`, inspect the findings, record an approval disposition, and sign off on the forensic report (`APPROVED`).
10. Log in as `admin` to verify immutable audit logs and export the case package with Merkle root verification.

---

## 11. Documentation Map

| Area | Document | Description |
| :--- | :--- | :--- |
| **System Overview** | [`docs/README.md`](docs/README.md) | Complete documentation index and reading guide. |
| **Architecture** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline stages, data flows, and package breakdown. |
| **Workflow** | [`docs/SYSTEM_WORKFLOW.md`](docs/SYSTEM_WORKFLOW.md) | 20-step investigator, reviewer, and admin workflow. |
| **Intelligence** | [`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md) | Random Forest model, calibration, features, and ablation. |
| **Network Context** | [`docs/NETWORK_EVIDENCE.md`](docs/NETWORK_EVIDENCE.md) | Network observation principles, limits, and abstention. |
| **Security** | [`docs/SECURITY_AND_RBAC.md`](docs/SECURITY_AND_RBAC.md) | RBAC capabilities, case isolation, and Merkle proofs. |
| **Data Ingestion** | [`docs/DATA_AND_INGESTION.md`](docs/DATA_AND_INGESTION.md) | Ingest formats (JSON/CSV/XML), schema normalisation, and limits. |
| **Research Decisions** | [`docs/RESEARCH_DECISIONS.md`](docs/RESEARCH_DECISIONS.md) | Why components were chosen, rejected, or safeguarded. |
| **Production Freeze** | [`docs/PRODUCTION_FREEZE.md`](docs/PRODUCTION_FREEZE.md) | Authoritative SIH 2026 freeze record and SHA-256 hashes. |
| **Demo Setup** | [`docs/LOCAL_DEMO_SETUP.md`](docs/LOCAL_DEMO_SETUP.md) | Air-gapped deployment and demonstration runbook. |
| **Research Index** | [`research/README.md`](research/README.md) | Experiments, benchmarks, validation reports, and reproduction. |

---

## 12. Current Status

- **Status:** **FROZEN & VERIFIED** (Target Build Freeze: September 25, 2026).
- **Backend Verification:** 1,702 / 1,702 unit and integration tests passing (`pytest tests/`).
- **Frontend Verification:** 101 / 101 unit tests passing (`vitest`); production Vite bundle built offline.
- **Air-Gap Compliance:** Zero external network calls, zero remote fonts/CDNs, 100% local model loading.
