# ObsidianChain — Research & Experimental Foundation
**Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**  
**Repository Layer:** `research/` (Isolated Research, Benchmarks, Reports & Reproduction)

---

## 1. Purpose of the Research Directory

The `research/` directory houses the research foundation, exploratory experiments, model-selection evaluations, latency/export benchmarks, and offline reproduction pipelines for ObsidianChain.

In an institutional intelligence platform, research exploration must be strictly demarcated from operational casework:
- **Research Scope:** Model exploration (comparing LightGBM, Logistic Regression, and Random Forest), feature group ablation studies, latency profiling, and canonical dataset construction from raw blockchain transaction graphs.
- **Production Scope:** Immutable analytical pipelines, hardened RBAC, deterministic casework lifecycle management, append-only cryptographic audit ledgers, and air-gapped forensic reporting.

This architectural separation guarantees that research exploration remains 100% reproducible and verifiable without allowing experimental state, uncalibrated heuristics, or exploratory models to bleed into production inference.

---

## 2. Distinction Between Research and Production

| Aspect | Research Layer (`research/`) | Production Engine (`src/obsidianchain/`) |
| :--- | :--- | :--- |
| **Supervised Model** | Evaluated multiple architectures (Logistic Regression, Random Forest, LightGBM) | **FROZEN Random Forest** (120 trees, `max_depth=12`, balanced subsample) with Isotonic Calibration |
| **Model Artifact** | Generation scripts (`research/reproduction/train_ps_model.py`) | Stored in `data/models/ps_native/v1/model.joblib` and SHA-256 verified against `manifest.json` |
| **Pipeline Execution** | Exploratory ablation and profiling harnesses | Deterministic 17-stage orchestration (`src/obsidianchain/pipeline/orchestrator.py`) |
| **Ground Truth Access** | Allowed for training, cross-validation, and PR-AUC measurement | **STRICTLY BLOCKED** (`boundary.assert_no_truth_fields()`); zero truth labels reach console or APIs |
| **Network Telemetry** | Observed announcement modeling and synthetic graph stress testing | Treated strictly as `NETWORK_CONTEXT` with evidentiary caveats; never an ownership or identity claim |
| **Operational Interface** | CLI scripts and analytical benchmarks | Hardened FastAPI REST endpoints, air-gapped web console, and cryptographic Merkle export |

---

## 3. Which Research Findings Influenced Production

1. **Model Selection (Why Random Forest Won):**
   - Candidate architectures evaluated on the temporal training split (timesteps 1–34):
     - **Logistic Regression (StandardScaler):** Validation PR-AUC = 0.3812. Unable to capture non-linear feature interactions between transaction fee ratios and co-spend degree.
     - **LightGBM (Gradient Boosted Trees):** Validation PR-AUC = 0.5218. Demonstrated sensitivity to extreme class imbalance (prevalence ~2.3%) and risk of overfitting temporal distribution shifts.
     - **Random Forest (120 trees, max_depth=12):** **Validation PR-AUC = 0.5843** (winning model). Achieved superior generalization across temporal boundaries, resilient decision boundaries under class imbalance, and stable feature importances.
   - **Production Decision:** Random Forest was selected and **FROZEN** as the production analytical model. LightGBM remains strictly historical/research-only.

2. **Probability Calibration (Isotonic Regression):**
   - Uncalibrated tree probabilities exhibited typical empirical under/over-confidence.
   - Applying `IsotonicRegression` on validation data reduced the Brier score loss from **0.0812 to 0.0543**, producing monotonic, true posterior probabilities.
   - **Production Decision:** All production risk scores are passed through the fitted isotonic calibrator before severity thresholds (`CRITICAL >= 0.85`, `HIGH >= 0.65`, `MEDIUM >= 0.40`) are stamped.

3. **Feature Engineering & Ablation Hierarchy:**
   - Feature group ablation proved that transaction-level behavior (fees, amounts, input/output counts) provides strong baseline signal, but topological graph features (degree, 2-hop neighbor reach) and structural pattern indicators (peeling chains, multi-input mixers) provide decisive discriminatory lift (+14.2% PR-AUC).
   - **Production Decision:** Implemented the 24-feature PS-native feature engine (`features_ps.py`) directly in the 17-stage analytical pipeline.

4. **Robust Anomaly Detection (MAD Z-Score):**
   - Standard Gaussian Z-scores were distorted by heavy-tailed Bitcoin transaction values.
   - **Production Decision:** Adopted Median Absolute Deviation (MAD) robust Z-scoring (`src/obsidianchain/ml/anomaly.py`) for outlier flagging.

---

## 4. Which Experiments are Research-Only

The following components and evaluations are research-only and must **never** be deployed into the production execution path:
- **LightGBM Exploration:** Historical benchmarking scripts and comparisons.
- **Ablation Studies (`research/experiments/ablation_study.py`):** Degradation evaluations disabling feature groups to measure marginal information gain.
- **Micro-benchmarks & Profilers (`research/benchmarks/`):** Multi-iteration latency stress tests for export payloads and Merkle tree generation.
- **Synthetic World Overlap Experiments:** High-volume synthetic node overlap evaluations used to stress test reachability boundaries.

---

## 5. Truth Isolation Rule (Critical Boundary)

> [!CAUTION]
> **RESEARCH GROUND TRUTH MUST NEVER ENTER PRODUCTION INFERENCE.**

In Bitcoin transaction intelligence and AML/CTF forensics:
1. **No Circular Evidence:** An alert is an analytical prediction computed from observable blockchain transactions and network telemetry. If ground-truth labels from research datasets enter the pipeline, predictions become trivial circular lookups.
2. **Hard Boundary Enforcement:**
   - `src/obsidianchain/api/boundary.py` enforces `assert_no_truth_fields()` on all API payloads.
   - Automated tests in `tests/test_truth_isolation.py` and `tests/test_api_boundary.py` verify that internal label columns (`class`, `label`, `ground_truth`, `is_illicit`) are unimportable and unreachable by the console and API layers.
3. **Evidentiary Integrity:** Network observations are classified as `NETWORK_CONTEXT` and accompanied by evidentiary disclaimers: they represent investigative context, never definitive wallet ownership or sender identification.

---

## 6. Directory Structure

```
research/
├── README.md                                  # This document
├── experiments/
│   └── ablation_study.py                      # Feature group ablation experiment across temporal splits
├── reports/
│   └── PS_NATIVE_MODEL_VALIDATION_REPORT.md   # Formal 24-feature model validation and calibration report
├── benchmarks/
│   ├── benchmark_export.py                    # Latency, memory, and payload size export benchmark
│   └── profile_export.py                      # 9-stage fine-grained profiler for Merkle export pipeline
└── reproduction/
    ├── build_ps_dataset.py                    # Deterministic raw -> canonical train/val/test parquet builder
    └── train_ps_model.py                      # Complete training, calibration, and manifest generation pipeline
```

---

## 7. How to Reproduce the Major Research Experiments

All reproduction scripts are self-contained and run offline in an air-gapped environment using the local `data/` directory:

### A. Rebuilding the PS-Native Dataset Splits
Transforms raw Elliptic++ CSV files in `data/raw/` into chronological, leak-free Parquet splits (`train.parquet`, `validation.parquet`, `test.parquet`):
```bash
python3 research/reproduction/build_ps_dataset.py
```
*Output:* Writes to `data/models/ps_native/datasets/` and updates `manifest.json`.

### B. Training and Calibrating the Production Model
Trains the candidate models on train split, validates PR-AUC on validation split, fits the isotonic calibrator, and evaluates held-out test performance:
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
