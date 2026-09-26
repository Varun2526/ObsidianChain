# Threshold analysis plan (fixed before any validation metric was computed)

Date: 2026-09-26. Analysis only: no model, feature schema, registry, holdout
artifact, deployment or production code is changed. No retraining.

## Validation set
`research/benchmark/threshold_analysis/` builds it from existing development
data (raw Elliptic++ files); nothing new is collected.

- Why not `data/models/ps_native/datasets/validation.parquet`: all 37,694 of
  its addresses were first seen in t26-41, which is v5's final training set
  ("addresses first seen in t26-41, snapshot as of t41"). Scoring it is
  in-sample.
- Why not out-of-fold predictions: none are stored. Regenerating them means
  refitting fold models, which is excluded.
- Used instead: the frozen `ps_native_v5` scored on addresses **first seen in
  t2-25**, which were never in its training set. There are three windows
  shaped like the holdout (8 steps, addresses first seen in the window,
  snapshotted at their last event at or before the window end): t2-9 (as of
  t9), t10-17 (as of t17), t18-25 (as of t25). t1 is left out so that every
  window has 8 steps.
- Pipeline: identical to `research/benchmark/v5_accuracy/RESULT.md`, i.e.
  `build_canonical_frame(raw, max_step=hi)`, then
  `PsTemporalFeatureEngine().process_records`, then
  `PsNativeRiskModel.raw_scores`, then `calibrate`. Labels come from
  `wallets_classes.csv` (1 = positive, 2 = negative, 3 excluded), joined after
  features.
- Known weaknesses: the period is earlier than the training data; these
  steps were evaluation windows during model selection (not for thresholds);
  and the prevalence differs from t42-49.

## Metrics (validation, pooled over the three windows)
For each calibrated-probability threshold in {0.10, 0.15, ..., 0.70}:
precision, recall, F1, balanced accuracy, false-positive rate, predicted
positives. Also reported per window. Top-K: precision and recall at K = 100,
500, 1000, ranked by calibrated probability.

## Operating-point rules (fixed now)
- Reference: 0.50 (current).
- **Balanced point:** the grid threshold with the highest validation F1; ties
  go to the higher threshold.
- **High-recall point:** the lowest grid threshold with validation precision
  >= 0.50 (at least as many true alerts as false ones).
- Severity bands, from validation precision on the grid (each band's cutoff
  is the lowest grid threshold meeting it):
  - Critical: precision >= 0.90
  - High: the balanced point
  - Medium: the high-recall point
  - Low: below Medium

  If two bands land on the same threshold, they are merged and reported as
  merged. No cutoff is adjusted afterwards.

## Holdout (t42-49), once, after the points above are written down
Apply 0.50, the balanced point and the high-recall point once, with the same
pipeline (max_step = 49). Report precision, recall, F1, balanced accuracy,
nAP and P@100. Under ADR 0004 t42-49 is SEEN, so these numbers are labelled
`REUSED_HOLDOUT`. They are a check, never used to change a threshold.
