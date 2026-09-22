# ObsidianChain — Independent ML Audit
**Date:** 2026-09-22 · **Scope:** PS-native supervised risk model `ps_native_v1`, its dataset, evaluation and alerting layer
**Method:** read-only code audit + independent re-execution against the frozen artifacts, using the repo's own vendored wheels (scikit-learn 1.9.0, pandas 3.0.5, numpy 2.4.6, lightgbm 4.7.0) so versions match the freeze exactly.
**Nothing in the production pipeline was modified.** All scripts in this directory are additive and reproduce every number below.

---

## 0. Executive summary

The ML system is **real, reproducible and fundamentally sound in the place most student projects fail**: there is no label leakage, no temporal leakage, and no train/test entity overlap. The feature engine is genuinely forward-only. That is worth stating plainly, because it is the hard part and it was done correctly.

The serious problems are elsewhere, and they are all in **measurement and reporting**, not in the model:

1. **`docs/ML_PIPELINE.md` does not describe the shipped model.** Every documented hyperparameter is wrong, one headline test metric is overstated by 31 points, and three capabilities are claimed that the code does not implement.
2. **The evaluation is a single 8-timestep window.** Under 7 rolling windows, the between-window standard deviation is **38x** the seed-to-seed standard deviation. Every reported point estimate is one draw from a very wide distribution, quoted to four decimals.
3. **The severity bands do not deliver their advertised precision, even on the split they were fitted on** — a rank-derived cutoff is applied as a value threshold under massive ties.
4. **Isotonic calibration improves Brier but degrades the ranking**, which is the thing the product actually sells.
5. **Six of thirty features carry no information**, because synthetic amount construction makes them constant or exact duplicates of other features. Peeling-chain detection can never fire.

**Recommendation: do not change the model family.** Change the evaluation protocol, the alert-ranking policy, the documentation, and the explanation path. Evidence for each is below, including the experiments that failed.

---

## 1. Current pipeline (as built)

```
data/raw/{AddrTx,TxAddr}_edgelist.csv + txs_features.csv + wallets_classes.csv
   -> research/reproduction/build_ps_dataset.py
        surrogate clock  t = 1400000000 + (step-1)*1209600
        amounts synthesised:each input gets total_in/len(inputs), likewise outputs
        PsTemporalFeatureEngine (forward-only, incremental union-find)
        keep LAST observation per address; split by FIRST-appearance step;
        drop boundary-spanning addresses; join Elliptic wallet class
   -> train/validation/test.parquet   (170,404 / 37,694 / 54,335 rows)
   -> research/reproduction/train_ps_model.py
        fit LR, RF, LightGBM on TRAIN; select on VALIDATION PR-AUC (raw scores)
        isotonic calibration fitted on VALIDATION
        severity bands derived from calibrated VALIDATION scores
        single TEST evaluation (calibrated scores)
   -> data/models/ps_native/v1/{model.joblib, metrics.json, calibration.json, manifest.json}
```

**Two parallel alerting systems exist**, which is itself a defect:

| | Phase 7 artifact path (`alerts/build.py`) | Pipeline/console path (`pipeline/alerts.py`) |
|---|---|---|
| Model | LightGBM + isotonic (`ml/model.py`) | frozen RF `ps_native_v1` |
| Unit | one alert per co-spend cluster | one alert per cluster |
| Ranking | `AGG-TOPK` mean of top-3 member risks | fused score: anomaly 0.40 + patterns 0.40 + **ML 0.20** |
| Severity | worst member band | hardcoded 0.80/0.60/0.40/0.20 |
| Explanations | **real TreeSHAP** (`pred_contrib`) | `importance x abs(value)` proxy |
| Serves | the React dashboard / `api/alerts.py` | `data/runs/*/alerts.json` |

The model the documentation is entirely about (`ps_native_v1`) is **not** the model behind the dashboard, and contributes only 20% of the console path's ranking score. The MAD anomaly detector — a rule, not a model — carries double the weight of the ML model there.

---

## 2. Reproduction (Phase 10)

| Check | Result |
|---|---|
| `model.joblib` SHA-256 vs manifest | **matches** (`7d9c1b40…`) |
| Refit from `train.parquet` + `train_ps_model.py` config | **reproduces**: val AP 0.5701758 vs frozen 0.5701973 (Δ 2e-5) |
| `metrics.json` validation block | **exact match** — computed on **raw** scores |
| `metrics.json` test PR-AUC / ROC-AUC / Brier | **exact match** — computed on **calibrated** scores |
| `metrics.json` test P@100 / P@500 | 0.97 / 0.504 reported vs 0.98 / 0.508 recomputed — differ because `train_ps_model.py` uses an unstable `np.argsort` under heavy ties |
| `calibration.json` severity band support/precision | **does not reproduce** (see §5) |

Reproduction initially failed because `min_samples_leaf=20` is set in the training script but appears in no documentation. It is the single most important undocumented hyperparameter in the repo.

**Validation metrics use raw scores; test metrics use calibrated scores.** The headline "0.5702 → 0.2079 degradation" therefore compares two different score types. Like-for-like raw test AP is 0.2182.

---

## 3. Data and label audit (Phases 2–3)

**Labels are sound.** `wallets_classes.csv` class 1 → 1, class 2 → 0, class 3 excluded. Genuine Elliptic++ ground truth, joined *after* feature extraction, never touching the feature matrix. No rule-generated labels, no target leakage. Verified by code path and by confirming no feature correlates with the label through a definitional identity.

**Splits are clean on the dimension that usually breaks.**

| | train | validation | test |
|---|---|---|---|
| rows | 170,404 | 37,694 | 54,335 |
| positives | 9,328 (5.47%) | 2,358 (6.26%) | 2,518 (4.63%) |
| duplicate rows | 0 | 0 | 0 |
| address overlap with other splits | **0** | **0** | **0** |
| txid overlap with other splits | **0** | **0** | **0** |

**But the split design manufactures three different populations.** Addresses spanning a boundary are dropped entirely, so each split retains only addresses whose *whole life* fits inside it: train allows 34 timesteps, validation 7, test 8.

| feature (mean) | train | validation | test |
|---|---|---|---|
| `active_duration_seconds` | 811,648 | 20,088 | 38,491 |
| `n_txs_asof_t` | 0.214 | 0.106 | 0.095 |
| `output_count` | 990.5 | 13.6 | 283.2 |
| `cluster_size_asof_t` | 312.1 | 1,278.7 | 1,421.6 |

The model is trained on a population containing long-lived addresses and applied to populations from which they were structurally removed. 89–92% of all rows have `n_txs_asof_t == 0`, so the ten Group B "historical behaviour" features are zero for roughly nine rows in ten.

**The surrogate clock** gives every transaction in a timestep an identical timestamp. `active_duration_seconds` and `gap_since_last_tx` therefore take only 34 distinct values each, and `gap_since_last_tx == 0` conflates "no previous transaction" with "same fortnight". `ps_model.scores()` then applies `np.nan_to_num(..., nan=0.0)`, which the docs explicitly promise never happens.

---

## 4. Feature audit (Phase 4) — six features carry no information

Verified across all 262,433 pooled rows:

| Feature | Finding | Mechanism |
|---|---|---|
| `is_peeling_candidate` | **constant 0, always** | requires `out_max >= 0.8*total_out` with `out_count==2`; synthesised equal outputs make `out_max` exactly `0.5*total_out` |
| `input_amount_std` | max value **2.8e-14** (float noise) | all inputs assigned an equal share |
| `output_amount_std` | max value **2.3e-13** | all outputs assigned an equal share |
| `equal_output_count` | **identical to `output_count`** | all outputs equal ⇒ every output is a "matching" output |
| `output_entropy` | **≡ log2(output_count)** to 1.8e-15 | entropy of a uniform distribution |
| `in_degree_asof_t` | **identical to `n_recv_asof_t`** | both count distinct receiving txids |
| `out_degree_asof_t` | **identical to `n_sent_asof_t`** | both count distinct sending txids |
| `is_mixing_candidate` | **≡ `(input_count>=3) AND (output_count>=3)`**, exactly | `max_equal == out_count` always, so the equality test is vacuous |

Consequences: Group D "Structural Patterns" detects no structural patterns. Group C "Graph & Topology" contains two transaction counters mislabelled as graph degrees; only `unique_counterparties_asof_t` and `cluster_size_asof_t` are genuinely topological. **The PS requirement for peeling-chain and mixing detection is not met by these features.**

**Group E (network) is not in the model at all.** The frozen manifest lists 30 features, none of them network. `include_network=True` is passed by no caller in `src/`. If it were, the implementation is degenerate — `obs_div = 1.0 if row.get("observer_id") is not None else 1.0` returns 1.0 unconditionally. **GeoIP/ASN never reaches either ML model**; it is display-only evidence in `pipeline/alerts.py`.

---

## 5. Leakage audit (Phase 5) — the good news, and one real bug

**No temporal leakage.** `PsTemporalFeatureEngine` sorts by timestamp, reads prior state into the feature row, and only then updates state. `cluster_size_asof_t` uses a forward-replay incremental union-find, not a globally-built cluster index. This is correct and was verified by reading the state-update ordering directly.

**No entity leakage** (zero address overlap). **No label leakage** (labels joined after extraction).

**Contamination hypothesis tested and refuted.** The frozen model scores validation better (AP 0.5702) than my train-only refit did before I found `min_samples_leaf`, which raised the possibility it had been trained on validation. Controls settle it: a model trained on train+validation scores validation at AP 0.8438 with 33% of positives above 0.9; the frozen model scores 0.5702 with 11.3%, matching the train-only control. **The frozen model did not see validation data.**

**Residual, minor:** transactions within one timestep update each other's state, ordered by `txId` — an arbitrary order with no real-world meaning. Unavoidable given the surrogate clock, but it should be documented rather than described as "strictly forward-only".

**The repo's own leakage tests overstate their coverage.** `tests/test_ps_features_leakage.py` exercises the real engine but only on 1–2 row fixtures. In test A, 18 of 30 feature assertions are `0 == 0` and would pass under any leakage that does not make them non-zero. Test F's docstring claims it verifies split-assignment-before-labelling; its body asserts only two `first_seen` values and never constructs a split. That invariant is untested. (`tests/test_phase6_leakage.py` is substantially stronger and does check split disjointness on real data.)

---

## 6. Metric and threshold audit (Phases 7–8)

`ml/metrics.py` is well designed: accuracy is deliberately never reported, and PR-AUC is always paired with its prevalence baseline via normalised AP. That is better practice than most production systems.

**The defect is ties.** Isotonic regression maps ~10,000 distinct raw scores onto **55–59 distinct calibrated values**. Production ranks on calibrated scores.

| | validation | test |
|---|---|---|
| distinct raw scores | 8,428 | 10,155 |
| distinct **calibrated** scores | **36** | **55** |
| rows tied at the rank-100 cutoff | 253 | 171 |
| P@100 range over tie orderings | 0.99 – 1.00 | **0.89 – 1.00** |

Reported test P@100 = 0.97 sits inside a ±5-point band determined by arbitrary row order. It is not wrong; it is quoted with false precision.

**Severity bands are derived by rank position and applied as a value threshold.** `_derive_severity_bands` records `support = cum_total[best_idx]` — a rank — then production runs `calibrated_score >= threshold`, which sweeps in the entire tie block:

| Band | advertised (calibration.json) | actually admitted on **validation** | on **test** |
|---|---|---|---|
| CRITICAL | 840 rows @ 90.0% | **1,280 rows @ 82.2%** | **991 rows @ 37.2%** |
| HIGH (cum.) | 1,452 @ 75.0% | 2,148 @ 57.7% | 1,814 @ 24.7% |
| MEDIUM (cum.) | 2,602 @ 50.0% | 5,255 @ 29.3% | 4,912 @ 20.1% |

The band metadata is unreproducible from the frozen artifacts, and the advertised precision is not delivered **even on the split it was fitted on**.

**A simple rank-based queue beats the bands outright.** Test-set precision@K for the frozen model: P@50 = 1.000, P@100 = 0.980, P@200 = 0.950, P@300 = 0.740, P@500 = 0.508. A top-200 queue gives 95% precision; the CRITICAL band gives 37% over 991 alerts. Recall is the real ceiling: 500 alerts capture 10% of illicit addresses, 2,000 capture 19%.

---

## 7. Experiments (Phases 11–15)

Every experiment changes **one** variable. Results are reported whether or not they favour the current system.

### 7.1 Hypotheses that were REFUTED

| Hypothesis | Test | Outcome |
|---|---|---|
| P@100 = 99% is a tie-breaking artifact of raw RF scores | tie structure + random/best/worst tie orderings | **Refuted.** Raw RF gives 8,428 distinct scores; P@K identical under all tie orderings. The top of the ranking is genuinely almost all illicit. (The tie problem is real but belongs to the *calibrated* scores — §6.) |
| The frozen model was trained on validation | train-only vs train+val controls, memorisation fingerprint | **Refuted** (§5). |
| Trees decay faster with forward horizon | per-timestep nAP across test steps 42–49 | **Refuted.** No monotone decay for any model. Per-window nAP swings from 0.003 to 0.79, driven by prevalence (0.29% at step 45, 14.8% at step 49). The docs' "temporal degradation" claim is unsupported — the best windows are the latest ones. |
| **Logistic Regression is the better model** | single-window test + paired bootstrap | **Refuted by the follow-up experiment — see 7.2. This is the most important result in the audit.** |

### 7.2 The LR result, and why it does not survive

On the official test split, LR — the model the team **rejected** — decisively beats the shipped RF:

| Model | test AP | test ROC | P@200 | P@500 |
|---|---|---|---|---|
| LogisticRegression | **0.2925** | **0.8413** | 0.9500 | **0.7880** |
| RandomForest (shipped) | 0.2183 | 0.7356 | 0.9650 | 0.4340 |
| LightGBM | 0.2214 | 0.7733 | 0.9500 | 0.6180 |

Paired bootstrap, 800 resamples: **LR − RF = +0.0742, 95% CI [+0.0551, +0.0933], P(LR better) = 1.000.** Spearman correlation between validation AP and test AP across the three families is **−1.00** — validation ranked them in exactly the wrong order.

That looks conclusive. **It is not.** Under rolling-origin evaluation (train ≤ k, evaluate k+1…k+4, seven windows):

| Model | mean nAP | sd | folds won |
|---|---|---|---|
| RandomForest | **0.4748** | 0.175 | 3 / 7 |
| LightGBM | 0.4440 | 0.219 | 4 / 7 |
| LogisticRegression | 0.2913 | 0.253 | **0 / 7** |

LR wins **zero of seven** windows and collapses to P@200 = 0.07 in one. Mechanism confirmed: on that fold, 12.9% of evaluation rows sit beyond |z| > 10 relative to the training scaler (max |z| = 606, on `n_sent_asof_t`). Unbounded heavy-tailed count features make a linear model extrapolate arbitrarily far.

**The bootstrap CI was correct and still misleading.** Resampling within one window measures sampling noise inside that window; it cannot see between-window variance, which is far larger.

### 7.3 Variance decomposition — the central finding

| Source of variation | magnitude (AP / nAP units) |
|---|---|
| random seed, fixed window | sd = **0.005** |
| bootstrap resampling within one window | 95% CI half-width ≈ **0.015** |
| **choice of evaluation window** | sd = **0.175** |

**Window choice dominates by roughly 38x.** Every four-decimal metric in `metrics.json` and `docs/ML_PIPELINE.md` is one draw from a distribution with sd ≈ 0.175. The val→test "collapse" from 0.5702 to 0.2079 is well within that; it is not evidence of drift, overfitting, or model failure.

### 7.4 Other controlled experiments

| Experiment | Result | Decision |
|---|---|---|
| **Remove 7 degenerate/duplicate features** | LR test AP 0.2925 → 0.2921; RF within seed noise; P@500 0.788 → 0.790 | **Adopt.** Free simplification: −7 features, no cost, removes three false capability claims. |
| **Ensemble (rank-mean)** | LR alone 0.2925; LR+LGBM 0.2771; all three 0.2675 | **Reject.** Ensembling actively hurts. Do not add it for architectural appearance. |
| **Train on train+validation** | RF 0.2183 → **0.3246**; LGBM 0.2214 → 0.2868; LR 0.2925 → 0.3075 | **Adopt at freeze time.** Consistent gain across all three families; largest single effect measured (+0.106 for RF) — larger than any model-family difference. Requires a replacement selection protocol. |
| **Isotonic calibration** | Brier improves (RF 0.04555 → 0.04507; LGBM 0.04951 → 0.04109) but **AP degrades for all three** (LR 0.2925 → **0.1642**), resolution collapses 10,000 → 58 values | **Change the policy.** Rank on raw scores; carry calibrated probability as a display field only. |
| **RF vs LightGBM**, 6 folds × 3 seeds | RF wins **9 / 18**; mean diff +0.042, sd 0.121; RF sd 0.162 vs LGBM 0.219; LGBM collapses to 0.049 on fold 44 where RF holds 0.325 | **Statistically indistinguishable; keep RF** for lower variance and no collapse. The original choice was right, for the wrong reason. |
| **Seed stability of shipped config** | test AP sd 0.005; but **P@200 sd 0.124** (one seed fell to 0.56) | Report P@K as a distribution over seeds, never a single value. |

---

## 8. Explainability audit (Phase 16)

A **correct** TreeSHAP implementation exists (`ml/model.py`, LightGBM `pred_contrib`) and backs the dashboard's parquet artifacts. That path is fine.

The frozen model's path (`ml/ps_model.py`) is not. It computes `contribution = global feature_importances_ × abs(raw feature value)` and sets `direction = "INCREASES_RISK" if calibrated_prob >= 0.5 else "DECREASES_RISK"`. Both are structurally wrong:

- Multiplying by raw magnitude makes the attribution a ranking of **feature units**, not influence. Across all 365 explained addresses in `data/runs/*/alerts.json` (1,095 explanation rows), the top-3 is an amount or count feature in ~97% of rows; a single triple covers 44% of addresses; the top five triples cover 82%. **No binary flag ever appears**, because `|value| ≤ 1`.
- Direction is per-row constant by construction — all three features of an alert always carry the same label. HIGH and MEDIUM alerts (thresholds 0.215 and 0.114, both < 0.5) therefore display **three features all labelled "DECREASES_RISK"** on an alert the system is escalating.
- Additional bug: `pipeline/alerts.py:331` reads `e.get('feature', '')` but `PsFeatureExplanation` serialises `feature_name`. Every run artifact renders *"Key contributing features:  (decreases risk),  (decreases risk),  (decreases risk)."* — names blank.

The PS requires "why a wallet/transaction was flagged". On the console path, that requirement is currently not met.

---

## 9. Final comparison table

All figures on the frozen held-out test split (54,335 rows, 4.63% prevalence), single seed 20260919, raw scores unless noted. **Read the last column first.**

| Model | AP | ROC | P@100 | P@200 | P@500 | Brier (raw) | Brier (cal) | Fit | Rolling-origin nAP (7 windows) |
|---|---|---|---|---|---|---|---|---|---|
| **RandomForest** (shipped) | 0.218 | 0.736 | 0.94 | 0.965 | 0.434 | 0.0456 | 0.0451 | 8.3 s | **0.475 ± 0.175 — most stable** |
| LightGBM | 0.221 | 0.773 | 0.91 | 0.950 | 0.618 | 0.0495 | 0.0411 | 3.2 s | 0.444 ± 0.219 (one collapse) |
| LogisticRegression | **0.293** | **0.841** | 0.90 | 0.950 | **0.788** | 0.1936 | 0.0469 | 3.9 s | 0.291 ± 0.253 — **0/7 folds won** |
| RF trained on train+val | **0.325** | — | — | 0.955 | — | — | — | 9 s | not evaluated |

Single-window ranking (LR first) and multi-window ranking (LR last) **disagree completely**. That disagreement is the audit's main result: no reported number from a single window can support a model choice on this dataset.

---

## 10. Recommended pipeline — strictly from evidence

**Keep:** the Random Forest, the feature engine's forward-only discipline, `ml/metrics.py`'s normalised-AP design, the offline/provenance machinery, the real TreeSHAP in `ml/model.py`.

**Change, in priority order:**

1. **Replace single-window evaluation with rolling-origin evaluation.** Report mean ± sd over ≥5 windows and the per-window table. Never quote a four-decimal single-window figure again. *(Fixes the root cause.)*
2. **Rank on raw model scores; keep the calibrated probability as a display field.** Isotonic calibration improves Brier and damages ranking, and the product is a ranked queue.
3. **Replace the severity bands with a rank-based top-K queue** (top-200 on this data: 95% precision). If bands must persist, derive and apply them the same way, and validate the delivered precision on a window they were not fitted on.
4. **Correct `docs/ML_PIPELINE.md`.** Real config is `n_estimators=100, max_depth=12, min_samples_leaf=20, class_weight=None, random_state=20260919`; LightGBM is 250 trees; test P@500 is 0.504, not 81.2%; the model has 30 features and no network features; the feature module is `pipeline/features_ps.py`.
5. **Stop claiming peeling-chain and mixing detection**, or fix the amount synthesis that makes them impossible. `is_peeling_candidate` is 0 on all 262,433 rows.
6. **Drop the 7 dead/duplicate features** — measured free.
7. **Fix the explanation path**: reuse the existing TreeSHAP, and fix the `feature` / `feature_name` key bug that blanks every explanation in `data/runs/*/alerts.json`.
8. **Retrain on train+validation at freeze time** — the single largest measured improvement — with selection moved to rolling-origin.
9. **Reconcile the two alerting systems**, or document explicitly which one is the product.

**Do not:** add an ensemble (measured harmful), switch to LightGBM (indistinguishable, higher variance), or switch to Logistic Regression (wins one window, loses seven).

---

## 11. Limitations of this audit

- Raw Elliptic++ CSVs were not re-derived, so rolling-origin folds inherit the original spanner-drop at boundaries 34 and 41. Folds are comparable to each other but are not a clean re-derivation. Rebuilding from raw with per-fold boundaries would strengthen §7.2.
- Labels are address-level and time-invariant; "was this address ever illicit" is not the same target as "is this transaction suspicious now". The system cannot score an address until its last observed transaction, which limits its use as a live monitor.
- No real IP/port/ASN data reaches either model, so the PS's cross-layer correlation requirement is unmet at the ML layer regardless of anything measured here.
- Recall is bounded: ~10% at 500 alerts, ~19% at 2,000. No configuration tested changed this materially.
- `active_duration_seconds`, `gap_since_last_tx` and `tx_velocity_per_hour` are artifacts of the surrogate clock and would behave differently on real timestamps.

## 12. Reproducing this audit

Scripts in this directory, run against the frozen artifacts with the repo's vendored wheels:

| Script | Produces |
|---|---|
| `a1_data.py` | shapes, overlap, degeneracy, duplicate-feature pairs (§3, §4) |
| `a2_repro.py` `a8_repro2.py` | artifact hash + metric reproduction (§2) |
| `a3_ties.py` `a4_cal_ops.py` | tie structure, P@K sensitivity, band precision (§6) |
| `a5_shift.py` | threshold provenance, population shift (§3, §6) |
| `a7_contam.py` | contamination controls (§5) |
| `a9_cmp.py` `a10_boot.py` | seeded comparison, paired bootstrap (§7.2) |
| `a11_final.py` | horizon, queue, ensemble, calibration, pruning (§7.4) |
| `a12_rolling.py` `a13_var.py` | rolling-origin, variance decomposition (§7.2, §7.3) |
