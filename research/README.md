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
| **Supervised Model** | Evaluated multiple architectures (Logistic Regression, Random Forest, LightGBM, Stacking) | **FROZEN Champion LightGBM** (ObsidianChain Risk Model, internal identifier: `ps_native_v5`, 300 trees, 31 features, schema `ps_native_features/5`; fallback: Risk Model Fallback / `ps_native_v5_fallback_no_g`) with Platt Calibration |
| **Model Artifact** | Attested scripts (`research/reproduction/train_ps_production_model.py`) | Stored in `data/models/ps_native/v5/model.joblib` and verified against `registry.json` and `manifest.json` |
| **Pipeline Execution** | Exploratory ablation, temporal drift, and profiling harnesses | Deterministic 17-stage orchestration (`src/obsidianchain/pipeline/orchestrator.py`) |
| **Ground Truth Access** | Allowed strictly for training, cross-validation, and PR-AUC measurement | **STRICTLY BLOCKED** (`boundary.assert_no_truth_fields()`); zero truth labels reach console or APIs |
| **Network Telemetry** | Observed announcement modeling and synthetic graph stress testing | Treated strictly as `NETWORK_CONTEXT` with evidentiary caveats; never an ownership or identity claim |
| **Operational Interface** | CLI scripts and analytical benchmarks | Hardened FastAPI REST endpoints, air-gapped web console, and cryptographic Merkle export |

---

## 3. Empirical Research Findings & Production Consequences

| Research Finding | Production Consequence |
| :--- | :--- |
| **Model Selection & Evolution:** While early baseline testing explored Random Forest (Legacy Random Forest Baseline / `ps_native_v1`), the 23-experiment Production Model Research Archive (`research/autoresearch_2026_09_23/`) demonstrated LightGBM (ObsidianChain Risk Model / `ps_native_v5`) achieved superior holdout generalization under Time-Ordered Evaluation Protocol (Protocol B; Holdout nAP 0.548, ROC-AUC 0.946, pooled P@100 1.00) with exact per-row TreeSHAP attributions and native missingness handling. | LightGBM was selected, gate-validated, and **FROZEN** as production champion (`ps_native_v5`) in `registry.json`; `ps_native_v5_fallback_no_g` serves as fallback; `ps_native_v1` is retained for historical lineage. |
| **Probability Calibration:** Raw ensemble voting proportions clustered away from true posterior probabilities under severe class imbalance; Platt scaling on out-of-fold logit scores reduced Expected Calibration Error (ECE) to 0.010 on holdout. | Raw scores pass through Platt calibration for display and severity budget ranking (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`). |
| **Feature Engineering:** Graph topology (Group C — Graph Topology), role indicators (Group F — Address Role), and upstream flow features (Group G — Upstream Flow Dynamics) delivered decisive discriminatory lift (+0.047 nAP, p = 0.013) over transaction features alone. | Implemented the 31-feature PS-native feature engine (`features_ps.py`, schema `ps_native_features/5`) directly in the 17-stage analytical pipeline. |
| **Robust Anomaly Scoring:** Extreme transaction values distorted standard Gaussian Z-scores. | Implemented Median Absolute Deviation (MAD) robust Z-scoring (`src/obsidianchain/ml/anomaly.py`) for outlier flagging. |
| **Gossip Diffusion Realities:** Over 84% of transactions are announced by multiple peers; IP reflects relay vantage, not wallet client. | System prohibits equating IP with wallet ownership; network data is stamped strictly as `NETWORK_CONTEXT`. |
| **Cluster Collapse Risk:** Heuristic change guesses on batch payments risk merging entire exchanges into super-clusters. | Change detection is gated to strictly 2-output transactions with high confidence thresholds and separate accounting. |
| **Temporal Lookahead:** Scoring historical transactions with future graph metrics creates artificial performance inflation. | All 31 features are extracted strictly as-of event timestamp $t$ with zero forward leakage. |

---

## 4. Operational Boundaries

To maintain institutional compliance and analytical integrity:
1. **Research Ground Truth is Evaluation-Only:** Ground-truth labels from Elliptic++ or synthetic datasets exist solely to evaluate models in `research/`. They are never imported, served, or consulted during live casework.
2. **Research Experiments Do Not Run in Production:** Ablation studies, micro-benchmarks, and training pipelines are standalone offline tools. They do not execute during standard 17-stage casework analysis.
3. **Production Inference Uses Frozen Artifacts:** The production engine does not train models on the fly; it loads frozen, SHA-256 verified weights from `data/models/ps_native/registry.json`.
4. **Reproduction Scripts are Not Runtime Code:** Scripts in `research/reproduction/` are maintainer tools for audit verification, not production API dependencies.

---

## 5. Research Directory Structure

```
research/
├── README.md                                  # This document
├── autoresearch_2026_09_23/                   # Production Model Research Archive (23 experiments: exp01-exp23)
│   ├── 00_research_protocol.md ... 20_production_program.md
│   ├── scripts/                               # Experiment harnesses for ablation, calibration, leakage
│   └── results/                               # Preserved experiment outputs, metrics, and JSON logs
├── cycle2_2026_09_24/                         # Cycle 2 research investigations and evaluation
├── network_2026_09_25/                        # Pre-registered network telemetry experiments (exp_net1, exp_net2)
├── experiments/
│   ├── ablation_study.py                      # Baseline feature group ablation experiment
│   └── 2026_09_26_typology_robustness/       # Typology robustness pre-registration and results
├── benchmark/                                 # Specialized benchmarks (feature signal, temporal, threshold, v5)
├── benchmarks/                                # Production export & profiler benchmarks
│   ├── benchmark_export.py                    # Latency, memory, and payload size export benchmark
│   └── profile_export.py                      # 9-stage fine-grained profiler for Merkle export pipeline
├── audit_2026_09_22/                          # Historical ML audit reports and test scripts
├── protocol_2026_09_22/                       # Evaluation protocol and decision rule scripts
├── reports/
│   └── PS_NATIVE_MODEL_VALIDATION_REPORT.md   # Baseline 30-feature model validation and calibration report
└── reproduction/
    ├── build_ps_dataset.py                    # Canonical dataset split builder
    ├── train_ps_production_model.py           # Attested training script for ps_native_v5 champion
    ├── train_ps_model.py                      # Historical training script for ps_native_v1 baseline
    ├── train_ps_model_v2.py                   # Intermediate v2 training script
    ├── train_ps_stacker_v2.py                 # Personalized PageRank seed-propagation stacker builder
    ├── evaluate_holdout.py                    # Single-run sealed holdout evaluator (ADR 0003)
    ├── production_gate.py                     # 19-criterion production promotion gate verification
    └── build_monitoring_baseline.py           # Feature drift baseline builder
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

### B. Training and Gate-Evaluating the Production Champion (`ps_native_v5`)
Trains champion LightGBM on the committed stream, fits out-of-fold Platt calibrator, and verifies the 19-criterion promotion gate:
```bash
python3 research/reproduction/train_ps_production_model.py
python3 research/reproduction/production_gate.py ps_native_v5
```
*Output:* Generates `data/models/ps_native/v5/` artifacts, verified in `registry.json`.

### C. Training the Baseline Model (`ps_native_v1`)
Trains the historical Random Forest model on the train split, fits isotonic calibration, and evaluates test performance:
```bash
python3 research/reproduction/train_ps_model.py
```
*Output:* Generates `data/models/ps_native/v1/model.joblib`, `calibration.json`, and `metrics.json`.

### D. Running Feature Ablation & Robustness Studies
Measures performance across feature subsets and typology representations:
```bash
python3 research/experiments/ablation_study.py
python3 research/experiments/2026_09_26_typology_robustness/run.py
```

### E. Running Export & Merkle Benchmarks
Measures endpoint latency, serialization speed, and peak memory for standard (50 alerts) and large (500 alerts) cases:
```bash
python3 research/benchmarks/benchmark_export.py
python3 research/benchmarks/profile_export.py
```
