# 17 — Dataset and Metrics Audit (exp13, 2026-09-23)

> **STATUS (2026-09-23, final): historical.** Absolute metric values in this
> document were computed before the leakage fixes L1-L6
> (`research/autoresearch_2026_09_23/19_leakage_audit.md`) and are
> **INVALID or WITHDRAWN**. Current valid numbers: `docs/results_register.md`.
> Current model: `docs/model_card.md`.


Scope: PS-native development data (`train.parquet` + `validation.parquet`,
t1-41) and the raw Elliptic++ files it is built from. The sealed holdout
(`test.parquet`) was **not** opened. Scripts: `scripts/exp13_*.py`.
Model for fold checks: LightGBM (300 trees, lr 0.05), the 12
`protocol.rolling_origin_folds()`, seed 0.

## 1. Class balance and prevalence

| | value |
|---|---|
| Raw wallets | 822,942 — 67.8% unknown (class 3), 30.5% licit, 1.7% illicit |
| Usable labeled rows | 208,098 (train 170,404 / val 37,694) |
| Prevalence | train 5.47%, val 6.26% |
| Per-timestep prevalence | 0.41% (t1) to 30.3% (t26) |
| Per-fold prevalence | 4.8% to 22.8% |

- Positives are among **labeled** addresses only; the operational base rate is unknown.
- nAP correlates with fold prevalence (r = 0.48): nAP does not fully remove prevalence effects.

## 2. The biggest issue: rows are not independent — transaction duplication

All Group A features (10 columns) are **transaction-level**. Every address in
the same transaction gets the same Group A values, and 88.6% of rows are
cold-start (all history features = 0). The result:

- **83.1%** of rows sit in groups of exact-duplicate feature vectors.
- **93.2%** of rows share their `txid` with another address; one txid has 7,866 addresses.
- **One transaction holds 434 positives**; the top 10 txids hold **17.4% of all positives**.
- Per fold, 141-1,813 positive rows collapse to only **117-569 distinct positive transactions**.
- The top 100 of a ranked list contains 26-91 distinct transactions, not 100 leads.

Consequences for the metrics:

- **Effective sample size is roughly 3-5x smaller than the row count.** This is
  a direct mechanism behind the fold sd of about 0.18 nAP.
- A single large transaction that is scored right or wrong moves nAP a lot.
  Transaction-weighted nAP (each txid counts once) disagrees strongly with row
  nAP on some folds: t33-34 is 0.506 row vs 0.246 tx-weighted; t21-22 is 0.383 vs 0.625.
- **Precision@K is inflated as an investigator metric.** P@100 = 1.00 on t25-26
  contains only 26 distinct transactions.
- Train AUC is 0.994-1.000 vs eval AUC 0.70-0.96. The model memorises the
  duplicated transaction vectors.
- 244 duplicate-vector groups (2,662 rows, 975 positives) carry **conflicting
  labels** — identical inputs, different labels. This is irreducible noise for any model.

## 3. Data quality defects

| Defect | Evidence | Effect |
|---|---|---|
| Zero-fee cluster (CORRECTED 2026-09-23, see note) | fee==0 on **7.0% of rows** from only 25 transactions; raw `fees` is exactly 0.0 for those txids (in_BTC_total == out_BTC_total), illicit rate 0.007% | Not a missing-value artefact: the zero fees are in the source. They are still a 25-transaction shortcut the model can memorise. Separately, `build_ps_dataset.py:122` did encode a missing fee as 0.0; that path touches 0 usable rows (all NaN-fee txs are unlabelled), and v3 now keeps it NaN |
| Time features quantised | `active_duration_seconds`, `gap_since_last_tx` take only 34 values, all multiples of 1,209,600 s (one timestep) | "Seconds" and "per hour" features are step counts in disguise; `tx_velocity_per_hour` is not a velocity |
| Redundant features | Spearman ≥ 0.995: `total_input_amount`/`total_output_amount`, `n_txs_asof_t`/`tx_velocity_per_hour`/`unique_counterparties_asof_t`, `n_sent_asof_t`/`btc_sent_total_asof_t`/`mean_fee_ratio_asof_t` | Removing 6 redundant columns changes nAP by −0.0005 (not significant). Simpler model, cleaner SHAP |
| Near-constant flag | `is_mixing_candidate` = 1 on 0.3% of rows, AUC 0.500 | No signal |
| `first_t` misnamed | Experiments derive `first_t` from the **last** snapshot timestamp (`exp01_model_family.py:50`); `protocol.development()` docstring says first appearance | No leakage (features are as of that step), but the fold assignment differs from the documented rule |
| Boundary spanners dropped | 2,921 of 265,354 labeled addresses (1.1%); illicit rate 2.1% vs 5.4% | Small selection effect; multi-step addresses (7.5%) are 3x less illicit than single-step ones |

## 4. Drift

- **Fee market drift:** median fee rises from 0.0001 BTC (t1) to 0.007-0.013 (t39-41).
  Absolute fee and amount features are not stationary.
- **Univariate direction flips between train and val:** `output_amount_mean`
  AUC 0.661 train vs 0.378 val; `cluster_size_asof_t` 0.537 vs 0.398;
  `output_count` 0.183 vs 0.419. The same feature points in different
  directions in different periods.
- Adversarial validation (train t≤34 vs val t35-41) AUC = 0.999. Random-CV
  duplicates inflate this number, but the split is clearly separable.
  `mean_fee_ratio_asof_t` carries 83% of the gain.
- Per-step rank normalisation of fee/amount features: +0.011 nAP, 8/12 folds
  won, p = 0.43. Direction is right, but the effect is not significant.

## 5. Calibration (12 folds, calibrator fit on the last 2 train timesteps)

| | raw | isotonic | Platt |
|---|---|---|---|
| Brier (mean) | 0.058 | 0.060 | 0.059 |
| ECE (mean) | 0.040 | 0.042 | 0.047 |

- **No calibrator beats raw LightGBM on average.** The earlier single-fold
  result in `08_calibration_analysis.md` (isotonic worse) is general, not a
  worst-fold artefact.
- Cause: **prevalence shift.** The calibration window prevalence differs from
  the eval window by up to 2x (t27-28: calibration 22.8%, eval 10.2%;
  isotonic ECE 0.157 vs raw 0.047). A static calibrator maps to the wrong base rate.
- Isotonic collapses scores to 25-64 levels. This creates the precision@K tie
  bands documented in 08.

## 6. Tested fixes (paired over 12 folds, protocol `paired_verdict`)

| Variant | nAP | tx-weighted nAP | verdict vs base |
|---|---|---|---|
| base | 0.581 ± 0.170 | 0.602 ± 0.193 | — |
| train weight 1/txsize | 0.567 | 0.610 | indistinguishable |
| fee NaN + missing flag | 0.580 | 0.604 | indistinguishable |
| per-step rank features | 0.592 | 0.611 | indistinguishable (8/12 wins) |
| rank + tx weight | 0.555 | **0.616** | indistinguishable |
| pruned (−6 redundant) | 0.581 | 0.604 | indistinguishable |

No data fix is significant against a 0.164 MDE. This confirms the program's
main finding: fold choice dominates. The data fixes are still correct to make
for validity, simplicity and explanation quality, not for score.

## 7. Recommendations, in priority order

1. **Change the evaluation unit.** Report transaction-grouped metrics next to
   row nAP: tx-weighted nAP, and P@K over **distinct transactions/clusters**.
   Deduplicate the worklist so one transaction cannot fill the top K.
2. **Fee handling.** Keep a missing fee as NaN (done in schema /3). The
   zero-fee cluster is real data, so no flag is needed; the transaction-weighted
   metric in item 1 is what stops 25 transactions from dominating.
3. **Rename or rebuild the time features.** Use `steps_active` and
   `steps_since_last_tx`. Drop `tx_velocity_per_hour`, or compute it only from
   real timestamps.
4. **Drop the 6 redundant columns** (`total_output_amount`,
   `tx_velocity_per_hour`, `btc_sent_total_asof_t`, `mean_fee_ratio_asof_t`,
   `gap_since_last_tx`, `is_mixing_candidate`).
5. **Make features drift-robust:** per-step percentile ranks of fee and amount,
   or fee relative to the step median.
6. **Calibration:** rank on the raw score. For displayed probabilities, use a
   prevalence-adjusted calibrator (recalibrate the intercept to recent
   prevalence) instead of static isotonic. Do not report calibrated P@K.
7. **Increase statistical power:** more folds are impossible, so reduce
   variance instead. Use grouped (txid) bootstrap CIs per fold, and add a
   seed×fold repeat only when variants are close.
8. **Add signal for cold-start (88.6% of rows):** address-level features that
   distinguish addresses inside one transaction — input vs output role, share
   of tx value, and counterparty history (exp08, +0.024 so far). Without these,
   all addresses in one tx are indistinguishable by construction.
9. **Label noise:** flag the 244 conflicting duplicate groups. Report them
   separately; do not train on contradictory copies.
10. Fix the `first_t` naming (call it `last_t`) or align with the documented
    first-appearance rule, then re-verify the fold assignment.

## Correction (same day)

The first version of this report said the 7% fee==0 rows came from missing
fees encoded as zero. That was wrong. Checked against `txs_features.csv`:
all 25 transactions carry `fees == 0.0` in the source, with input total equal
to output total. The missing-to-zero code path in the builder was real but
affected no usable row. The "fee NaN + missing flag" variant in section 6
therefore tested a change that was a near no-op, which is consistent with its
null result.
