# ps_native_v5 decision-threshold analysis (2026-09-26)

Analysis only. The protocol and selection rules were committed in `PLAN.md`
(b124dbe) before any validation metric was computed. Validation results and
the frozen operating points were committed in 54ec81d before the holdout was
read. Artifacts are alongside: `validation.json`, `selection.json`,
`holdout_check.json`, `threshold_analysis.py`.

## 1. Validation dataset used
The frozen `ps_native_v5` (model SHA-256 `974d37f2…07`, verified before
scoring) was scored on addresses **first seen in t2-25** of Elliptic++. This
covers three holdout-shaped windows (8 steps each; addresses first seen in the
window, each at its last event at or before the window end): t2-9, t10-17,
t18-25. That is 126,318 labelled addresses, 6,302 of them positive
(prevalence 4.99%). The pipeline is identical to the published benchmark
(`build_canonical_frame`, `PsTemporalFeatureEngine`, `raw_scores`, then
`calibrate`), and labels were joined after features.

## 2. Why it is valid for threshold selection, and where it is weak
- **Not in v5's training set.** v5 was fit on addresses first seen in
  t26-41. No address first seen before t26 is in it.
- **Not the holdout.** t42-49 was not read until the thresholds were frozen.
- **Rejected alternatives.**
  - `datasets/validation.parquet` (t35-41): all 37,694 of its addresses are
    in v5's training set, so scoring it is in-sample.
  - Out-of-fold predictions: none are stored, and producing them would mean
    retraining.
- **Weaknesses.**
  - It is earlier in time than the training data, so it tests the model
    backward.
  - These steps served as evaluation windows during model selection (not
    for thresholds).
  - It is much easier than t42-49: pooled nAP 0.797 against 0.548. The
    windows also differ widely (nAP 0.464 on t2-9, 0.879 on t10-17, 0.882
    on t18-25), so any threshold chosen here transfers with uncertainty.

## 3. Threshold table (validation, pooled, calibrated probability >= t)

| t | Precision | Recall | F1 | Balanced acc. | FPR | Predicted + |
|---|---|---|---|---|---|---|
| 0.10 | 41.9% | 88.6% | 56.9% | 91.1% | 6.45% | 13,330 |
| 0.15 | 53.2% | 86.3% | 65.9% | 91.2% | 3.98% | 10,220 |
| 0.20 | 63.1% | 83.7% | 71.9% | 90.6% | 2.57% | 8,364 |
| 0.25 | 69.7% | 78.7% | 73.9% | 88.5% | 1.80% | 7,114 |
| **0.30** | 75.2% | 73.5% | **74.3%** | 86.1% | 1.27% | 6,154 |
| 0.35 | 78.7% | 64.8% | 71.1% | 82.0% | 0.92% | 5,192 |
| 0.40 | 82.1% | 58.9% | 68.6% | 79.1% | 0.67% | 4,522 |
| 0.45 | 85.7% | 56.2% | 67.9% | 77.9% | 0.49% | 4,138 |
| 0.50 | 87.7% | 50.8% | 64.3% | 75.2% | 0.37% | 3,646 |
| 0.55 | 89.7% | 46.5% | 61.2% | 73.1% | 0.28% | 3,268 |
| 0.60 | 91.2% | 42.0% | 57.5% | 70.9% | 0.21% | 2,900 |
| 0.65 | 93.4% | 40.7% | 56.7% | 70.3% | 0.15% | 2,747 |
| 0.70 | 94.8% | 39.7% | 56.0% | 69.8% | 0.11% | 2,640 |

Accuracy stays within 93.3-97.5% across the whole grid and is not used for
selection. Per-window tables are in `validation.json`.

## 4. Top-K (ranking; independent of any threshold)

| | P@100 | R@100 | P@500 | R@500 | P@1000 | R@1000 |
|---|---|---|---|---|---|---|
| Validation (t2-25, 6,302 positives) | 99.0% | 1.6% | 99.8% | 7.9% | 99.8% | 15.8% |
| t42-49, REUSED_HOLDOUT (2,518 positives) | 100% | 4.0% | 95.0% | 18.9% | 69.3% | 27.5% |

## 5. Operating points and severity bands (rules fixed in PLAN.md)
- **Balanced point = 0.30.** Highest validation F1 (74.3%, against 64.3% at
  0.50): recall rises 50.8% -> 73.5%, precision falls 87.7% -> 75.2%.
- **High-recall point = 0.15.** Lowest threshold with validation precision
  >= 50%: recall 86.3% at precision 53.2%, with 2.8x the alerts of 0.50.
- **Bands** (from validation precision; nothing adjusted afterwards):

| Band | Calibrated probability | Validation precision at the cutoff |
|---|---|---|
| Critical | >= 0.60 | 91.2% (first grid t with precision >= 90%) |
| High | 0.30 - 0.60 | 75.2% at 0.30 |
| Medium | 0.15 - 0.30 | 53.2% at 0.15 |
| Low | < 0.15 | |

Neither point is "optimal". 0.30 is the best validation F1. 0.15 buys recall
by accepting that roughly half of the alerts are false.

## 6. Holdout check (t42-49, applied once after freezing; REUSED_HOLDOUT)
Under ADR 0004, t42-49 is SEEN (opened once for v5). These numbers are a
veto-style check, not an unbiased estimate. The ranking is unchanged by any
threshold: nAP 0.5475 and P@100 100% in every row.

| Threshold | Precision | Recall | F1 | Balanced acc. | FPR | Flagged | nAP | P@100 |
|---|---|---|---|---|---|---|---|---|
| 0.50 (current) | 81.1% | 25.8% | 39.1% | 62.7% | 0.29% | 800 | 0.5475 | 100% |
| 0.30 (balanced) | 54.8% | 29.1% | 38.0% | 64.0% | 1.17% | 1,338 | 0.5475 | 100% |
| 0.15 (high-recall) | 45.7% | 62.0% | 52.6% | 79.2% | 3.58% | 3,417 | 0.5475 | 100% |

The 0.50 row reproduces `research/benchmark/v5_accuracy/RESULT.md` exactly.

## 7. Did recall improve?
Yes, at 0.15: holdout recall goes 25.8% -> **62.0%** (2.4x), F1 39.1% ->
52.6%, balanced accuracy 62.7% -> 79.2%.

No, not meaningfully, at the validation-F1 choice 0.30: recall goes 25.8% ->
29.1% while precision drops 26 points, and F1 falls slightly. The balanced
point **did not transfer**. On t42-49 the model's positives score lower than
on validation (the documented drift and per-step collapse), so most of the
recall appears only below 0.30. This is an observation about the holdout. It
was not used to choose anything, and no threshold was re-picked because of it.

## 8. Precision trade-off
At 0.15, precision falls 81.1% -> 45.7%. That is below the 50% floor the
point was chosen to meet on validation, so the floor did not hold out of
time. Alerts rise from 800 to 3,417 (about 1,855 false among them, against
151). Accuracy falls 96.3% -> 94.8%. **Accuracy did not improve** at any
threshold, and nothing here should be claimed as an accuracy gain. What
changed is where the alert line sits on an unchanged ranking: nAP, AUC and
P@K are identical.

## 9. Did any production code or model change?
No. The model, feature schema, registry, calibration, holdout artifacts,
Docker and deployment are untouched, and nothing was retrained or tuned.
Only files in `research/benchmark/threshold_analysis/` were added.

Note for product use: v5 in production assigns severity by rank budget per
run (top 1% CRITICAL, 5% HIGH, 15% MEDIUM; `ml/ps_model.py`
`SEVERITY_BUDGET`), not by a probability threshold. The 0.50 cutoff exists
only in the benchmark. The probability bands above would be an alternative
presentation. Adopting them is a product decision with the trade-off in
section 8 (more recall, about half the Medium-band alerts false). If adopted,
it should be monitored on delayed labels, because the validation->holdout gap
shows the cutoffs drift.
