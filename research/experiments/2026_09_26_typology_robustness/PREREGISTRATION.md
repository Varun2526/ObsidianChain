# Pre-specified development-only experiment: typology robustness

## Status

Exploratory development-only comparison. This specification is frozen before
the E0/E1/E2 runs in this directory. It cannot be confirmatory generalization
evidence because the development period and the holdout diagnosis were already
seen. No `t42-49` record may be read by the runner.

## Question

Across the fixed twelve Protocol-B development folds, do causal typology
features and transaction-cluster-balanced training improve temporal ranking
relative to the current v5 design?

## Fixed data and protocol

- Data: Elliptic++ raw source restricted to timesteps `t1-41` before feature
  generation.
- Features: the unchanged `PsTemporalFeatureEngine` in
  `src/obsidianchain/pipeline/features_ps.py`; labels are joined only after
  all causal features are produced.
- Folds: `obsidianchain.ml.protocol.rolling_origin_folds()` exactly as-is:
  expanding train ends `t16` through `t38`, two-step test windows `t17-18`
  through `t39-40`.
- Training: 16-step Protocol-B rolling training window, new addresses only,
  snapshot at the training-window end; test addresses are new in the two-step
  test window, snapshot at that window's end.
- Metrics: address-level nAP is primary; P@100, R@100, cold-start-slice nAP,
  and weak-upstream-slice nAP are secondary. Slice nAP is computed on all
  rows in the named slice (both classes); it is unmeasurable if either class
  is absent.

## Fixed experiments

- **E0:** the 31 `ps_native_features/5` CORE columns and v5 LightGBM
  hyperparameters: 300 estimators, learning rate 0.05, 31 leaves,
  `min_child_samples=50`, `subsample=0.8`, `subsample_freq=1`,
  `colsample_bytree=0.8`, seed `20260919`, and `n_jobs=-1`.
- **E1:** E0 plus these six causal research-only columns:
  1. `research_cold_start = (n_txs_asof_t == 0)`;
  2. `research_weak_upstream = (upstream_funded_share < 0.50)`;
  3. `research_log_fan_in = log1p(input_count)`;
  4. `research_relationship_density = unique_counterparties_asof_t / max(n_txs_asof_t, 1)`;
  5. `research_cold_x_weak_upstream = research_cold_start * research_weak_upstream`;
  6. `research_cold_x_upstream_share = research_cold_start * upstream_funded_share`.
- **E2:** exactly E1 features. For training rows only, a positive row with
  snapshot transaction ID `txid` gets raw weight `1 / (# positive training
  rows with that txid)`; every negative row gets raw weight `1`. All weights
  are multiplied by `n_train / sum(raw_weights)`. Thus each positive
  transaction event contributes one raw unit and the total effective training
  mass equals `n_train`. Test labels never enter weights.

## Fixed decision and uncertainty rules

- Winner is selected exclusively from the twelve development folds by:
  highest mean nAP, then higher worst-fold nAP, then higher mean measurable
  cold-start-slice nAP, then higher mean P@100.
- Paired uncertainty is a 10,000-resample bootstrap over the twelve fixed
  per-fold nAP differences, seed `20260926`; report its percentile 95% CI.
- No threshold is evaluated, tuned, or applied. No final model/artifact is
  produced. All models exist only in memory during their fold.

## Disconfirming results

The hypothesis that robustness helps is disconfirmed if E1's mean nAP is no
higher than E0's. The hypothesis that cluster balancing adds value is
disconfirmed if E2's mean nAP is no higher than E1's. Fold-level dispersion
and bootstrap intervals will be reported regardless.
