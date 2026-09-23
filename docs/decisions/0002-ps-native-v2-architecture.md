# ADR 0002 — PS-native v2: schema /3, LightGBM, propagation, fusion, monitoring

Date: 2026-09-23. Status: accepted, holdout evaluation pending.
Scope: the PS-native path only (ADR 0001). Phase 6 artifacts are unchanged.

## Context

Uploaded-dataset runs had no risk score: the frozen RandomForest v1 declared
schema /1 and the pipeline emitted /2, so stage 10 returned
`MODEL_UNAVAILABLE_FOR_SCHEMA`. The dataset audit
(`research/autoresearch_2026_09_23/17_dataset_metrics_audit.md`) also found:
83% of development rows duplicate another row's feature vector (group-A
features belong to the transaction), redundant columns, strong drift, no
calibrator beating the raw score, and an explanation proxy whose directions
were wrong 66% of the time. The problem statement additionally asks for risk
propagation from seed wallets and graph embeddings, neither of which existed.

## Decisions

1. **Feature schema /3** (`pipeline/features_ps.py`).
   - One chain event per txid. Repeated network observations of a txid were
     replayed as repeated transactions, doubling history counts. They are now
     aggregated into group E.
   - Missing fee and unmeasurable fee ratio are NaN, not 0.0.
   - Removed four restatements (Spearman >= 0.995): `total_output_amount`,
     `tx_velocity_per_hour`, `btc_sent_total_asof_t`, `mean_fee_ratio_asof_t`.
   - New group F (role): `addr_is_sender`, `addr_is_self_change`,
     `counterparty_max_n_txs_asof_t`, `counterparty_mean_n_txs_asof_t`.
     Transaction-weighted nAP +0.053, p = 0.022, 11/12 folds.
   - Group E computed from real observations (the v2 stub set two columns to
     a literal 1.0), plus `dominant_peer_share` and `arrival_spread_seconds`.
2. **Model v2: LightGBM** (`data/models/ps_native/v2/`). Chosen from the
   statistically tied tier for exact TreeSHAP and native NaN handling, not
   for ranking. Trained on a 16-step window (exp18): 12-fold nAP 0.648
   (sd 0.160, worst 0.409), tx-weighted 0.678, address P@100 0.94,
   R@500 0.54 (exp17).
   Platt calibration fitted out-of-fold; ranking uses the raw score.
   Explanations are `pred_contrib` with per-feature sign. Severity is a rank
   budget with a base-rate floor.
3. **Risk propagation** (`ml/propagation.py`): personalized PageRank from
   seed wallets, hubs over 500 participants excluded, path to the nearest
   seed attached. Seeds come offline from the OFAC SDN list and
   `data/watchlists/*.csv` (`io/watchlist.py`). A frozen stacker
   (`ml/stacking.py`, `v2/stacker.json`) combines it with the model when a
   seed is present: exp14 tx-weighted nAP +0.053, p = 0.0496.
4. **Graph embeddings** (`ml/embeddings.py`): counterparty-profile SVD,
   producing cross-cluster link SUGGESTIONS only. Measured weak on
   Elliptic++ labelled entities (12% precision at cosine >= 0.90, 8x the
   base rate). Never merged, caveat printed on every suggestion.
5. **Fusion** (`pipeline/alerts.py`): noisy-OR over evidence lines with
   stated policy weights (model 1.0, propagation 0.8, pattern 0.6,
   anomaly 0.5), budget severity (2% / 8% / 20%) with a 0.20 floor, and a
   corroboration count. Replaces a weighted mean that a missing line
   dragged down, plus hand-set "reinforcement" bonuses.
6. **Monitoring** (`ml/monitoring.py`): every run writes `monitoring.json`
   with per-feature PSI against a training reference, unseen missingness,
   score PSI and cold-start share; the manifest carries `model_trust`.

## Consequences

- Uploaded runs are scored again. v1 is kept frozen and still loadable.
- The sealed holdout is still schema /1 and has not been opened. No holdout
  metric exists for v2. Rebuilding it at /3 requires
  `OBSIDIANCHAIN_REGENERATE_HOLDOUT=1` and a written reason, and is a human
  decision.
- The development training data has no missing fee, so LightGBM learned no
  missing branch for it: a missing fee is scored like a zero fee. The drift
  report names this (`unseen_missingness_features`).
- Fusion weights for the rule lines are policy. Only the model line (and the
  model + propagation stack) has labels behind it.
- World v2 (`world/noisy.py`) is a SYNTHETIC_CONTROL benchmark. Its result
  (exp15: network features raise relay-vs-negative AUC 0.803 -> 0.930 in the
  SIGNAL world and do nothing in the NULL world) shows the pipeline extracts
  a network signal that exists. It says nothing about real traffic.

## Amendment (same day): ps_native_v3, schema /4

- Causal event order inside Elliptic++ timesteps (`event_order` = spend-DAG
  level). The previous txId order leaked same-step future transactions into
  history features and inflated every v2 metric by ~0.06 nAP.
- Group G (upstream flow) admitted to CORE: nAP +0.120, p = 0.0001.
- `data/models/ps_native/v3/` is the default model (fallback v2, then v1).
  12-fold nAP 0.713 (worst 0.433), address P@100 0.94, R-precision 0.67,
  ceiling-normalised R@500 0.80, ECE 0.041. Holdout still unevaluated.
- Full record: `research/autoresearch_2026_09_23/18_architecture_upgrade.md`,
  Addendum 2.
