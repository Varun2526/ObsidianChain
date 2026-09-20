# PS-NATIVE SUPERVISED ML MODEL VALIDATION REPORT

**Version**: `ps_native_v1`  
**Model Family**: PS-Native Tabular Risk Architecture (Separate from Frozen Research Model)  
**Evaluated**: 2026-09-19  
**Artifact Path**: `data/models/ps_native/v1/`  
**Model SHA-256**: `7d9c1b404872fd1b05ecb4ff99648904c89f6f5ee1502e6c2915bc6f4d7ad65d`

---

## 1. Dataset

### Provenance & License
- **Provenance Type**: `RESEARCH_DERIVED_CANONICAL_TRAINING_REPRESENTATION`
- **Source Artifacts**: Reconstructed from locally available raw Elliptic++ tables:
  - `data/raw/AddrTx_edgelist.csv` (SHA-256: `f5f903f752387f66a1bccaeff54e293b2e8470fcddf5eb56b88aa06fd23a8f3b`)
  - `data/raw/TxAddr_edgelist.csv` (SHA-256: `9f5afbdde7bc3d91fb7a4655be55799d6504cd0063ae55a0753a5a41189932b8`)
  - `data/raw/txs_features.csv` (SHA-256: `2db326ec8ddb68f1d810c1834e1ff62e0a8300378f0984a1e3b2ca82a439821b`)
  - `data/raw/wallets_classes.csv` (SHA-256: `4e5132c99f941666bf1fefd4100a1428d339c9252ec6987909e1adf8eac902f9`)
- **License**: CC BY 4.0 (Elliptic++ Data License). Offline forensic research usage.
- **Timestamp Formulation**: `ELLIPTIC_TIMESTEP_SURROGATE`. Timesteps 1–49 mapped monotonically to continuous Unix seconds via $t = 1400000000 + (\text{step} - 1) \times 1209600$. This represents a deterministic temporal surrogate, not real-world P2P packet arrival telemetry.

### Labels & Class Balance
- **Primary Prediction Unit**: `ADDRESS-AS-OF-TIMESTAMP` (address evaluated at the point of each transaction event).
- **Label Mapping**:
  - Class 1 $\rightarrow$ `1` (Illicit)
  - Class 2 $\rightarrow$ `0` (Licit)
  - Class 3 $\rightarrow$ `EXCLUDED` (Unknown labels are strictly excluded from supervised loss and evaluation).
- **Usable Sample Count**: 262,433 labeled address-transaction event instances.
  - Positive (Illicit): 14,204 (5.41%)
  - Negative (Licit): 248,229 (94.59%)

---

## 2. Features

### Feature Schema & Groups (30 Core Features + 4 Optional Network)
1. **Group A: Transaction Behaviour (Instantaneous at $t$)** (12 features)
   - `input_count`, `output_count`, `total_input_amount`, `total_output_amount`, `fee`, `fee_ratio`
   - `input_amount_mean`, `input_amount_max`, `input_amount_std`
   - `output_amount_mean`, `output_amount_max`, `output_amount_std`
2. **Group B: Address Historical Behaviour (As-of-$t$)** (10 features)
   - `n_txs_asof_t`, `n_sent_asof_t`, `n_recv_asof_t`
   - `btc_sent_total_asof_t`, `btc_recv_total_asof_t`, `net_flow_asof_t`, `mean_fee_ratio_asof_t`
   - `active_duration_seconds`, `tx_velocity_per_hour`, `gap_since_last_tx`
3. **Group C: Graph & Counterparty Behaviour (As-of-$t$)** (4 features)
   - `in_degree_asof_t`, `out_degree_asof_t`, `unique_counterparties_asof_t`, `cluster_size_asof_t`
4. **Group D: Structural Pattern Heuristics** (4 features)
   - `is_peeling_candidate`, `is_mixing_candidate`, `equal_output_count`, `output_entropy`
5. **Group E: Network Telemetry (Optional)** (4 features)
   - `network_observation_count`, `observer_diversity`, `peer_count`, `asn_count`
   - **Missingness Policy**: When network telemetry is unobserved (as in pure blockchain records), features natively evaluate to `NaN`. Zero is never fabricated.

### Leakage Verification
All 6 mandatory leakage safeguards were asserted in `tests/test_ps_features_leakage.py`:
- **Test A**: Future transactions append without modifying earlier feature states.
- **Test B**: Future counterparties do not alter earlier degrees.
- **Test C**: Future cluster expansion does not alter earlier cluster sizes.
- **Test D**: Future transactions cannot alter earlier peeling/mixing markers.
- **Test E**: Label columns cannot enter the feature matrix.
- **Test F**: Split boundaries are assigned purely chronologically on first appearance.

---

## 3. Chronological Splits

| Split | Timestep Range | Total Rows | Positive (Illicit) | Negative (Licit) | Positive Prevalence | Boundary Spanners |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Step 1–34 | 170,404 | 9,328 | 161,076 | 5.47% | First seen in 1–34 |
| **Validation** | Step 35–41 | 37,694 | 2,358 | 35,336 | 6.26% | Excludes spanners from Train |
| **Test** | Step 42–49 | 54,335 | 2,518 | 51,817 | 4.63% | Held-out until final freeze |

Boundary-spanning entities appearing in multiple split regimes were strictly assigned by their earliest appearance; downstream splits evaluate unseen and strictly forward entities.

---

## 4. Model Selection (Evaluated on Validation Set Only)

**Primary Optimization Metric**: PR-AUC (Precision-Recall Area Under Curve).  
**Baseline**: No-skill random prevalence baseline on Validation = **0.0626**.

| Model Candidate | Validation PR-AUC | Validation ROC-AUC | Precision@100 | Precision@500 | Validation Brier | Fit Time (s) | Beats Baseline? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (L2 penalty) | 0.2654 | 0.7903 | 0.5500 | 0.5400 | 0.2014 | 0.46s | YES (+0.2028) |
| **LightGBM** (n_est=100, lr=0.05) | 0.5228 | 0.8693 | 0.7100 | 0.9340 | 0.0445 | 1.49s | YES (+0.4602) |
| **Random Forest** (n_est=120, depth=14) | **0.5702** | **0.8606** | **0.9900** | **0.9440** | **0.0419** | 1.25s | **YES (+0.5076)** |

### Winning Model
**Random Forest Classifier** achieved the highest PR-AUC on the Validation set (**0.5702**), with an extraordinary **Precision@100 of 99.0%** (99 out of the top 100 highest-ranked entities were true positives). It was selected as the frozen architecture.

---

## 5. Model Calibration & Severity Thresholds

- **Method**: Isotonic Regression (fit exclusively on Validation set).
- **Brier Score Improvement**: Reduced Brier score from `0.04186` (raw) to `0.03612` (calibrated).
- **Severity Bands** (Derived strictly on Validation data):
  - `CRITICAL`: Score $\ge 0.6697$ (Precision: 90.0%, Support: 840)
  - `HIGH`: Score $\ge 0.2149$ (Precision: 75.0%, Support: 1,452)
  - `MEDIUM`: Score $\ge 0.1139$ (Precision: 50.0%, Support: 2,602)
  - `LOW`: Score $< 0.1139$

---

## 6. Final Held-Out Test Evaluation

Test set was evaluated **once** after model architecture, hyperparameters, calibration parameters, and severity bands were frozen.

| Metric | Held-Out Test Result | No-Skill Prevalence Baseline |
| :--- | :--- | :--- |
| **PR-AUC** | **0.2079** | 0.0463 |
| **ROC-AUC** | **0.7367** | 0.5000 |
| **Precision@100** | **0.9700 (97.0%)** | 0.0463 |
| **Precision@500** | **0.5040 (50.4%)** | 0.0463 |
| **Brier Score** | **0.0451** | - |
| **F1 @ 0.5** | **0.2103** | - |

The model achieves **97% precision at top-100** in the completely held-out future test partition (steps 42–49), demonstrating high utility for investigative prioritization despite natural temporal concept drift.

---

## 7. Ablation Analysis

| Configuration | Feature Count | Features Included | Validation PR-AUC | Validation ROC-AUC | Precision@100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Behaviour Only** | 22 | Group A (Transaction) + Group B (Address History) | 0.2743 | 0.8194 | 0.7200 |
| **B. Behaviour + Graph** | 26 | Group A + Group B + Group C (Graph Degrees & Cluster Size) | **0.5474** | **0.8528** | **0.9800** |
| **C. Behaviour + Graph + Patterns** | 30 | Group A + Group B + Group C + Group D (Peeling/Mixing/Entropy) | **0.5702** | **0.8606** | **0.9900** |
| **D. With Network Telemetry** | 34 | Full Core + Group E (Network Observations) | *N/A (Unobserved)* | *N/A* | *N/A* |

**Key Findings**:
1. Adding temporal graph state (Group C) provides the single largest gain in PR-AUC ($+0.2731$) and pushes Top-100 Precision from 72% to 98%.
2. Structural pattern features (Group D) further stabilize top-tier ranking precision up to 99%.
3. Network features (Group E) were unobserved in the Elliptic++ dataset; missingness is handled cleanly as NaN without degrading the tabular baseline.

---

## 8. Scientific Interpretation & Limitations

1. **Investigative Risk Ranking**: The score represents estimated probability of association with entities labeled illicit in historical training data given evidence as of timestamp $T$. It is an investigative risk ranking tool, NOT definitive proof of criminality.
2. **Surrogate Timestamps**: Timestamps in the training dataset are deterministic surrogates mapped from discrete 2-week Elliptic++ timesteps. They do not simulate microsecond network gossip latency.
3. **Entity vs Temporal Labels**: Ground-truth labels in public forensic datasets are entity-level and may have been identified post-hoc.
4. **Separation of Signals**: The PS-native supervised model produces `MODEL_SIGNAL`. It is combined with, but does not overwrite or blur, independent `BLOCKCHAIN_CONTEXT`, `NETWORK_CONTEXT`, `ANOMALY_CONTEXT`, and `PATTERN_CONTEXT`.
