# 19 — Leakage and data-validity audit (2026-09-23)

Scope: the PS-native feature engine (`pipeline/features_ps.py`), the dataset
builder, the evaluation units, propagation, embeddings, calibration and
monitoring. The per-feature record is `docs/feature_catalog.md`, generated
from `contracts/features.py`, which the feature contract enforces.
Automated guard: `tests/test_leakage_audit.py`, which fails the build.

## Findings, in the order found

| # | Leak / validity defect | How found | Effect | Fix | Guard |
|---|---|---|---|---|---|
| L1 | Elliptic++ gives every transaction in a timestep one timestamp; the engine used txId order inside a step, so history features could count later same-step events | red-teaming a +0.12 group-G jump | inflated every v2/v3 number, about +0.06 nAP | causal `event_order` = spend-DAG level (all 100% of spend edges are same-step) | `test_every_spend_edge_goes_forward_in_event_order`, `test_elliptic_later_levels_in_a_step_never_change_earlier_levels` |
| L2 | Level written into timestamps (60 s/level) | CORE jumped to 0.802 | chain depth leaked into every "seconds" feature | order carried separately from the timestamp; never a feature value | catalog `same_step`; prefix tests |
| L3 | Strict "earlier timestamp" rule for upstream funding | gain vanished | discarded all real flow (not a leak, a validity error) | (timestamp, event_order) comparison | constructed tie test |
| L4 | Simultaneous events (same timestamp and level) saw each other's state updates in input order | writing the tie-order invariance test | history, counterparty and upstream features depended on row order | group-deferred updates, applied in canonical txid order | `test_simultaneous_events_do_not_see_each_other` |
| L5 | `groupby("address").last()` spliced the last NON-NULL value of each column from different transactions; unstable timestamp sort chose among same-time rows arbitrarily | reading the orchestrator | mixed rows in training and in production scoring | `last_snapshot_per_address`: one whole row in causal order | `test_the_snapshot_row_is_one_whole_row` |
| L6 | Final-event snapshot as evaluation unit | decomposing a +0.18 jump after L5 | hindsight: an address is scored only at its last event in the whole dataset (pass-through wallets' spend) | protocol B: addresses first seen in the window, snapshot at window end, disjoint from training | exp22; production training uses protocol B |
| L7 | Repeated network observations of one txid replayed as repeated transactions | reading the engine | history counts inflated on real captures | one chain event per txid; observations aggregated | `test_repeated_observations_of_one_txid_are_one_chain_event` |
| V1 | Conflicting chain facts across observation rows of a txid silently resolved by "first row" | writing the capture contract | an unknowable choice became a feature value | quarantine the transaction | `test_conflicting_and_invalid_transactions_are_quarantined` |

## Items checked and found clean

- **Labels:** no feature reads `wallets_classes`, a truth directory or any
  label column. Label columns in the input change nothing
  (`test_label_columns_in_the_input_change_nothing`). The placebo (training
  labels permuted within timestep) gives nAP 0.03.
- **Future timesteps:** appending later timesteps never changes an earlier
  row, on real Elliptic++ and on the synthetic capture with real timestamps.
- **Seed propagation:** in evaluation, seeds are labels of addresses first
  seen at or before the fold's train_end; the graph extends to the window end
  (structure, not labels). In production, seeds are OFAC / watchlist entries.
  Inference time is the end of the capture: propagation uses the whole
  capture graph, which is all in the past at that moment.
- **Graph embeddings:** within-run similarity only, never a model feature
  (SVD coordinates are not comparable across runs).
- **Calibration:** per-fold calibration is fitted on the four previous folds'
  out-of-fold scores only; the frozen calibrator on folds 9-12. The holdout
  never fitted anything.
- **Normalisation / aggregation:** the model uses raw feature values (tree
  model); no statistic is fitted over evaluation data.
- **Holdout:** never read before the single ADR 0003 evaluation. The
  protocol-A holdout file is unchanged (MD5 checked before and after).

## Residual assumptions (stated, not solved)

- Elliptic++ timestamps are two-week surrogates. Within a step, time order is
  known only along spend edges. Unrelated same-level transactions are treated
  as simultaneous, which is conservative.
- Per-address amounts on Elliptic++ are even splits of transaction totals
  (the source has no per-output values). `btc_recv_total_asof_t` and
  `net_flow_asof_t` inherit that approximation; real captures carry true
  amounts, which is a train/serve difference.
- The development data has no missing fee, so the model treats a missing fee
  as zero. The drift report names it (`unseen_missingness_features`).
