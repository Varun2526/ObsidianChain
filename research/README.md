# ObsidianChain — Research & Experimental Foundation
**Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
**Repository Layer:** `research/` (Isolated Research, Benchmarks, Reports & Reproduction)

---

## 1. Purpose of the Research Directory

The `research/` directory houses the experimental foundation, model-selection evaluations, latency/export benchmarks, and offline reproduction pipelines for ObsidianChain.

In an institutional intelligence platform, research exploration must be strictly demarcated from operational casework:
- **Research Scope:** Exploratory modeling (benchmarking LightGBM, Logistic Regression, and Random Forest), feature group ablation studies, latency profiling, and canonical dataset construction from raw blockchain transaction graphs.
- **Production Scope:** Immutable analytical pipelines, hardened RBAC, deterministic casework lifecycle management, append-only cryptographic audit ledgers, and air-gapped forensic reporting.

This architectural separation guarantees that research exploration remains 100% reproducible and verifiable without allowing experimental state, uncalibrated heuristics, or exploratory models to bleed into production inference.

---

## 2. Distinction Between Research and Production

| Aspect | Research Layer (`research/`) | Production Engine (`src/obsidianchain/`) |
| :--- | :--- | :--- |
| **Supervised Model** | Evaluated multiple architectures (Logistic Regression, Random Forest, LightGBM) | **FROZEN Random Forest** (120 trees, `max_depth=14`, balanced subsample) with Isotonic Calibration |
| **Model Artifact** | Generation scripts (`research/reproduction/train_ps_model.py`) | Stored in `data/models/ps_native/v1/model.joblib` and SHA-256 verified against `manifest.json` |
| **Pipeline Execution** | Exploratory ablation and profiling harnesses | Deterministic 17-stage orchestration (`src/obsidianchain/pipeline/orchestrator.py`) |
| **Ground Truth Access** | Allowed strictly for training, cross-validation, and PR-AUC measurement | **STRICTLY BLOCKED** (`boundary.assert_no_truth_fields()`); zero truth labels reach console or APIs |
| **Network Telemetry** | Observed announcement modeling and synthetic graph stress testing | Treated strictly as `NETWORK_CONTEXT` with evidentiary caveats; never an ownership or identity claim |
| **Operational Interface** | CLI scripts and analytical benchmarks | Hardened FastAPI REST endpoints, air-gapped web console, and cryptographic Merkle export |

---

## 3. Empirical Research Findings & Production Consequences

| Research Finding | Production Consequence |
| :--- | :--- |
| **Model Selection:** Random Forest achieved superior Validation PR-AUC (0.5702 vs. 0.5228 for LightGBM) and Top-100 Precision (99.0% vs. 71.0%). | Random Forest was selected and **FROZEN** as the production model; LightGBM is historical research only. |
| **Probability Calibration:** Raw voting proportions clustered away from true posterior probabilities; Isotonic Regression reduced Brier loss from 0.04186 to 0.03612. | All raw risk scores pass through the isotonic calibrator before severity thresholds (`CRITICAL >= 0.6697`, `HIGH >= 0.2149`, `MEDIUM >= 0.1139`) are stamped. |
| **Feature Engineering:** Graph topology and structural pattern features delivered decisive discriminatory lift (+14.2% PR-AUC) over transaction features alone. | Implemented the 30-feature PS-native feature engine (`features_ps.py`) directly in the 17-stage analytical pipeline. |
| **Robust Anomaly Scoring:** Extreme transaction values distorted standard Gaussian Z-scores. | Implemented Median Absolute Deviation (MAD) robust Z-scoring (`src/obsidianchain/ml/anomaly.py`) for outlier flagging. |
| **Gossip Diffusion Realities:** Over 84% of transactions are announced by multiple peers; IP reflects relay vantage, not wallet client. | System prohibits equating IP with wallet ownership; network data is stamped strictly as `NETWORK_CONTEXT`. |
| **Cluster Collapse Risk:** Heuristic change guesses on batch payments risk merging entire exchanges into super-clusters. | Change detection is gated to strictly 2-output transactions with high confidence thresholds and separate accounting. |
| **Temporal Lookahead:** Scoring historical transactions with future graph metrics creates artificial performance inflation. | All 30 features are extracted strictly as-of event timestamp $t$ with zero forward leakage. |

---

## 4. Operational Boundaries

To maintain institutional compliance and analytical integrity:
1. **Research Ground Truth is Evaluation-Only:** Ground-truth labels from Elliptic++ or synthetic datasets exist solely to evaluate models in `research/`. They are never imported, served, or consulted during live casework.
2. **Research Experiments Do Not Run in Production:** Ablation studies, micro-benchmarks, and training pipelines are standalone offline tools. They do not execute during standard 17-stage casework analysis.
3. **Production Inference Uses Frozen Artifacts:** The production engine does not train models on the fly; it loads frozen, SHA-256 verified weights from `data/models/ps_native/v1/`.
4. **Reproduction Scripts are Not Runtime Code:** Scripts in `research/reproduction/` are maintainer tools for audit verification, not production API dependencies.

---

## 5. Directory Structure

```
research/
├── README.md                                  # This document
├── experiments/
│   └── ablation_study.py                      # Feature group ablation experiment across temporal splits
├── reports/
│   └── PS_NATIVE_MODEL_VALIDATION_REPORT.md   # Formal 30-feature model validation and calibration report
├── benchmarks/
│   ├── benchmark_export.py                    # Latency, memory, and payload size export benchmark
│   └── profile_export.py                      # 9-stage fine-grained profiler for Merkle export pipeline
└── reproduction/
    ├── build_ps_dataset.py                    # Deterministic raw -> canonical train/val/test parquet builder
    └── train_ps_model.py                      # Complete training, calibration, and manifest generation pipeline
```

---

## 6. How to Reproduce the Major Research Experiments

All reproduction scripts are self-contained and run offline in an air-gapped environment using the local `data/` directory:

### A. Rebuilding the PS-Native Dataset Splits
Transforms raw Elliptic++ CSV files in `data/raw/` into chronological, leak-free Parquet splits (`train.parquet`, `validation.parquet`, `test.parquet`):
```bash
python3 research/reproduction/build_ps_dataset.py
```
*Output:* Writes to `data/models/ps_native/datasets/` and updates `manifest.json`.

### B. Training and Calibrating the Production Model
Trains candidate models on the train split, validates PR-AUC on the validation split, fits the isotonic calibrator, and evaluates held-out test performance:
```bash
python3 research/reproduction/train_ps_model.py
```
*Output:* Generates `data/models/ps_native/v1/model.joblib`, `calibration.json`, `metrics.json`, and `manifest.json`.

### C. Running the Feature Ablation Study
Measures PR-AUC, ROC-AUC, Brier score, and Precision@100 across feature subsets (A: Behavior only, B: Behavior + Graph, C: Full PS-Native):
```bash
python3 research/experiments/ablation_study.py
```
*Output:* Writes results to `data/models/ps_native/v1/ablation.json`.

### D. Running Export & Merkle Benchmarks
Measures endpoint latency, serialization speed, and peak memory for standard (50 alerts) and large (500 alerts) cases:
```bash
python3 research/benchmarks/benchmark_export.py
python3 research/benchmarks/profile_export.py
```
