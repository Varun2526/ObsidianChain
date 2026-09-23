# 18 — Architecture Upgrade: Results Record (2026-09-23)

Six steps, in order, each measured under `ml/protocol.py` (12 rolling folds,
paired verdicts, MDE 0.164 nAP). Sealed holdout not opened; MD5 unchanged
(`a15500c94b9808cd42d584ad4b5c3017`). Decision record: ADR 0002.

## Step 1 — schema /3 and model v2

| | nAP (row) | nAP (tx-weighted) | P@100 distinct tx |
|---|---|---|---|
| v2, schema /3 | 0.619 ± 0.173 | 0.662 | 0.882 |
| same without group F | 0.584 | — | — |

Group F (role + counterparty history): row nAP +0.036 (p = 0.123, 9/12
folds, indistinguishable); tx-weighted nAP +0.053 (p = 0.022, 11/12 folds,
favours F). Source: `data/models/ps_native/v2/evaluation.json`.

Also found and fixed: repeated network observations of one txid were
replayed as repeated transactions (history counts doubled on real captures).

Correction to doc 17: the 7% fee == 0 rows are genuine zero-fee transactions
in the source, not missing values.

Known limit: no fee is missing in development data, so LightGBM treats a
missing fee as 0.0. The drift report flags it per run.

## Step 2 — risk propagation (exp14)

| | nAP | tx-weighted nAP |
|---|---|---|
| LightGBM | 0.620 | 0.662 |
| propagation alone | 0.280 | 0.251 |
| stack (LR on both) | 0.664 | 0.715 |

Stack vs LightGBM: row +0.044 (p = 0.061, indistinguishable); tx-weighted
+0.053 (p = 0.0496, favours stack). Seeds are labels from t <= train_end only.
Frozen as `v2/stacker.json` (coef 0.708 on model logit, 0.791 on
log1p(1000·p)). Production seeds: 532 OFAC SDN Bitcoin addresses plus
watchlist CSVs.

## Step 3 — network layer, world v2 (exp15, SYNTHETIC_CONTROL)

| world | nAP CORE | nAP CORE+E | relay AUC CORE | relay AUC CORE+E |
|---|---|---|---|---|
| SIGNAL | 0.946 | 0.963 (p = 0.0007) | 0.803 | 0.930 (p < 0.001) |
| NULL | 0.947 | 0.947 (p = 0.995) | 0.808 | 0.810 (p = 0.85) |

The pipeline extracts a network signal when one exists and adds nothing when
none exists. Caveats: CORE nAP 0.946 is close to the 0.95 saturation line
(MIXING_LIKE and RAPID_MOVEMENT are easy), RELAY is 1.5% of rows, and the
signal was put in the generator on purpose. The first diagnostic (top-10%
recall) was invalid at 26% prevalence and was replaced; the generator was not
changed after any result.

## Step 4 — graph embeddings (exp16, Elliptic++ labelled entities)

Three runs, each fixing an artefact the previous one exposed:

| rule set | eligible cross-cluster pairs | AUC | cosine >= 0.99 | cosine >= 0.90 |
|---|---|---|---|---|
| none | 64,304 | 0.58 | 127 at 79.5% | 425 at 29.2% |
| + >= 2 counterparties | 29,034 | 0.74 | 108 at 79.6% | 305 at 34.1% |
| + >= 2 transactions | 12,955 | 0.61 | 12 at 0% | 132 at 12.1% |

The apparent 80% precision was 85 payout addresses of one mining pool, each
seen in a single transaction. On addresses with real multi-transaction
history, similarity is a weak hint (8x the 1.5% base rate). Shipped as
suggestions only at cosine >= 0.90 with the measured precision printed.
Cross-check: the unfiltered run counts exactly 1,974 same-entity pairs across
clusters, the Phase 1.5 false-split count.

## Step 5 — fusion, explanations, severity

Noisy-OR over evidence lines, rank-budget severity with a floor,
corroboration count, TreeSHAP explanations. Alert build made linear: 84 s to
24 s end to end on a 7,303-transaction capture (fusion 69 s to 10.5 s).

## Step 6 — monitoring

Every run writes `monitoring.json` (PSI per feature, score PSI, unseen
missingness, cold-start share) and `manifest.provenance.model_trust`.

## Open items

1. Holdout at /3: needs a human decision to rebuild under a written reason,
   then one sealed evaluation of v2 + stacker.
2. Frontend: uploaded-run alerts, evidence paths and link suggestions are
   produced but not displayed.
3. Fusion weights for rule lines are policy, not fitted; a labelled
   cluster-level set would be needed to learn them.
4. World v2 is near-saturated on CORE; a harder variant should be designed
   and pre-registered before it is run, not tuned against results.

## Addendum — operational scorecard and temporal stability (exp17, exp18)

Prompted by an external review: report the investigator workload curve
(P@K and Recall@K), calibration reliability, and fix the temporal swing.

**exp18 (pre-registered).** Of five candidates, a 16-step training window
(W) met the stated rule best: fold-nAP sd 0.173 -> 0.160, worst fold
0.243 -> 0.409, mean 0.620 -> 0.648 (Holm p ~0.52, not significant). Recency
weighting also passed; per-step ranks passed on spread but lost mean;
transaction weighting failed; a 3-per-transaction queue cap was rejected (it
demoted duplicated TRUE positives too, lowering R@500 0.523 -> 0.441).
Adopted: `TRAIN_WINDOW = 16` in `train_ps_model_v2.py`; v2 and the stacker
were retrained. Caveat: choosing W among three passing candidates on the same
folds is mildly optimistic.

**Diagnosis of the old worst fold (t39-40).** Three licit transactions filled
55 of the top-100 address slots (29 + 16 + 10 identical rows). With the
window, address P@100 there went 0.36 -> 0.87.

**exp17 scorecard, adopted v2 (12-fold mean, worst fold in brackets):**

| K | addr precision | addr recall | addr FP | tx precision | tx recall |
|---|---|---|---|---|---|
| 50 | 0.99 (0.88) | 0.10 | 0.5 | 0.95 (0.80) | 0.18 |
| 100 | 0.94 (0.58) | 0.18 | 6.2 | 0.88 (0.60) | 0.34 |
| 500 | 0.69 (0.28) | 0.54 | 153 | 0.46 (0.26) | 0.77 |
| 1000 | 0.46 (0.14) | 0.67 | 541 | 0.29 (0.15) | 0.91 |

Transaction level = one entry per txid, positive if any of its scored
addresses is; it is the lenient view the earlier "P@100 = 0.88" used.

Calibration (Platt, production recipe): mean ECE 0.048, Brier 0.053; worst
ECE 0.149 at t27-28, where the calibration window prevalence (22.8%) was
twice the evaluation window's (10.2%). Pooled reliability, predicted ->
observed: 0.043 -> 0.041, 0.086 -> 0.100, 0.166 -> 0.188, 0.501 -> 0.485.

## Addendum 2 — causal ordering, group G, ps_native_v3 (exp19-exp21)

**Leak found and fixed.** Elliptic++ gives every transaction of a timestep one
timestamp, and every tx->tx spend edge lies inside one step. The engine
processed a step in txId order, which is not time order: history features
could count same-step transactions that happened later. The builder now
passes each transaction's spend-DAG level as `event_order` (order only, never
a feature value). Effect on CORE: 12-fold nAP 0.658 -> 0.594. **Every v2
number reported earlier (0.62-0.65) was inflated by ~0.06 through this leak.**

Two further artefacts caught on the way, both withdrawn:
- Writing the level into the timestamp (60 s per level) lifted CORE to 0.802:
  chain depth leaked into every "seconds" feature. Replaced by `event_order`.
- A strict "earlier timestamp" rule for group G removed all real flow (all
  spend edges are same-step). 80.4% of same-step funding pairs are true spend
  edges, 1.4% ran backwards; causal order keeps the first and drops the second.

**exp20 — group G (upstream flow), causal data:** nAP 0.594 -> 0.714
(+0.120, p = 0.0001); confirmation folds 0.541 -> 0.682; worst fold
0.227 -> 0.446. Admitted to CORE (schema /4).

**exp21 — re-selection on causal data with G:** the grid winner and the
three-model ensemble both lost to the current configuration on the
confirmation folds' worst fold (0.359 and 0.416 vs 0.433), so neither is
adopted. Production: LightGBM, 16-step window, 31 leaves, min_child 50.

**ps_native_v3 scorecard (12 folds, mean, worst fold in brackets):**

| | value |
|---|---|
| nAP | 0.713 (0.433), sd 0.158 |
| tx-weighted nAP | 0.736 |
| R-precision | 0.666 (0.427) |
| address P@50 / P@100 | 0.97 (0.86) / 0.94 (0.73) |
| address P@500 / R@500 | 0.70 / 0.55 |
| ceiling-normalised R@500 | 0.80 (0.45) |
| address R@1000 | 0.72 |
| transaction R@1000 | 0.93 |
| ECE / Brier (Platt) | 0.041 / 0.045 |

Seeded scenario (propagation stacker, seeds known): nAP 0.80 tuning / 0.72
confirmation (exp21, reported, not used for selection).
