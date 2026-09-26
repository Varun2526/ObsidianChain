# ps_native_v5 Temporal Generalization / Concept-Drift Diagnosis

## 1. Executive summary

The weak later result is concentrated in **three abrupt windows** (`t43`, `t45`, `t47`), not a smooth decline. Their recorded nAPs are `0.135`, `0.017`, and `0.324`, while adjacent holdout windows include `0.704` (`t42`), `0.687` (`t44`), `0.981` (`t46`), and `0.717` (`t49`). Read-only diagnostics show that the failing windows contain lower-prevalence, differently shaped positive examples and that their positive scores are much less separated from negative scores. This supports a **positive-class typology/relationship change plus sparse clustered labels**, not a threshold or feature-pipeline defect.

The current product pipeline also has broad input PSI changes over time, but the retained `exp23_relative_drift` result marked every holdout step — including failures — `WITHIN_BASELINE`. Therefore ordinary covariate drift is present but is insufficient to explain why only specific windows collapse.

## 2. Performance by window (recorded results)

Development fold values are the recorded refit-fold result in `v5/evaluation.json`; holdout values are the published fixed-v5 result in `holdout/ps_native_v5.json`. They are not interchangeable evaluation regimes.

| Window | Train period | Test period | nAP | P@100 | Recall@100 | Positive prevalence | Positives | Test addresses |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t<=16 -> t17-18 | t<=16 | t17-18 | 0.820 | 0.91 | 0.450 | 4.880% | 202 | 4,139 |
| t<=18 -> t19-20 | t<=18 | t19-20 | 0.791 | 1.00 | 0.183 | 7.620% | 546 | 7,165 |
| t<=20 -> t21-22 | t<=20 | t21-22 | 0.477 | 1.00 | 0.076 | 6.335% | 1321 | 20,853 |
| t<=22 -> t23-24 | t<=22 | t23-24 | 0.906 | 1.00 | 0.107 | 14.527% | 936 | 6,443 |
| t<=24 -> t25-26 | t<=24 | t25-26 | 0.975 | 1.00 | 0.055 | 21.982% | 1810 | 8,234 |
| t<=26 -> t27-28 | t<=26 | t27-28 | 0.950 | 0.98 | 0.676 | 10.773% | 145 | 1,346 |
| t<=28 -> t29-30 | t<=28 | t29-30 | 0.553 | 0.76 | 0.145 | 9.375% | 525 | 5,600 |
| t<=30 -> t31-32 | t<=30 | t31-32 | 0.850 | 1.00 | 0.111 | 10.295% | 903 | 8,771 |
| t<=32 -> t33-34 | t<=32 | t33-34 | 0.979 | 1.00 | 0.139 | 7.987% | 719 | 9,002 |
| t<=34 -> t35-36 | t<=34 | t35-36 | 0.978 | 1.00 | 0.098 | 7.310% | 1018 | 13,927 |
| t<=36 -> t37-38 | t<=36 | t37-38 | 0.825 | 1.00 | 0.163 | 9.211% | 612 | 6,644 |
| t<=38 -> t39-40 | t<=38 | t39-40 | 0.591 | 1.00 | 0.176 | 4.780% | 569 | 11,903 |
| t42 | fixed ps_native_v5 (trained t26-41) | t42 addresses; snapshots <=t49 | 0.704 | 0.98 | not recorded | 4.343% | 399 | 9,188 |
| t43 | fixed ps_native_v5 (trained t26-41) | t43 addresses; snapshots <=t49 | 0.135 | 0.18 | not recorded | 1.636% | 97 | 5,928 |
| t44 | fixed ps_native_v5 (trained t26-41) | t44 addresses; snapshots <=t49 | 0.687 | 1.00 | not recorded | 3.542% | 279 | 7,877 |
| t45 | fixed ps_native_v5 (trained t26-41) | t45 addresses; snapshots <=t49 | 0.017 | 0.02 | not recorded | 0.304% | 27 | 8,871 |
| t46 | fixed ps_native_v5 (trained t26-41) | t46 addresses; snapshots <=t49 | 0.981 | 0.98 | not recorded | 11.719% | 508 | 4,335 |
| t47 | fixed ps_native_v5 (trained t26-41) | t47 addresses; snapshots <=t49 | 0.324 | 0.28 | not recorded | 2.915% | 196 | 6,724 |
| t48 | fixed ps_native_v5 (trained t26-41) | t48 addresses; snapshots <=t49 | 0.484 | 0.44 | not recorded | 5.375% | 381 | 7,089 |
| t49 | fixed ps_native_v5 (trained t26-41) | t49 addresses; snapshots <=t49 | 0.717 | 1.00 | not recorded | 14.596% | 631 | 4,323 |

- Best recorded development fold: `t<=32 -> t33-34` (`nAP=0.979`).
- Best holdout step: `t46` (`nAP=0.981`); it is not independent evidence of broad generalization because 505 of 508 positives arise from one transaction.
- Worst holdout step: `t45` (`nAP=0.017`, `P@100=0.02`).
- Degradation starts suddenly at `t43`, recovers at `t44`, collapses at `t45`, recovers at `t46`, and weakens again at `t47–48`: it is **episodic, not gradual**.
- `Recall@100` is not retained in the per-step holdout scorecard; it is reported as `not recorded` rather than recomputed from sealed artifacts.

## 3. Feature drift (all 31 current `/5` features)

PSI is the project's existing metric, using the frozen v5 training reference bins. Reading: `<0.10` stable, `0.10–0.25` shifted, `>=0.25` major. Cohorts are produced by the unchanged causal feature engine; labels are joined only after feature generation.

| Feature (ranked by degraded-period PSI) | PSI: good t33-34 | PSI: degraded t43/45/47 | PSI: all holdout t42-49 |
| --- | ---: | ---: | ---: |
| input_count | 1.007 | 0.942 | 0.218 |
| counterparty_max_n_txs_asof_t | 0.813 | 0.424 | 0.127 |
| upstream_mean_input_count | 0.062 | 0.421 | 0.243 |
| output_count | 1.115 | 0.339 | 0.120 |
| output_spread | 0.478 | 0.227 | 0.093 |
| fee_ratio | 0.423 | 0.215 | 0.094 |
| input_amount_mean | 0.304 | 0.214 | 0.104 |
| output_amount_mean | 0.975 | 0.178 | 0.084 |
| cluster_size_asof_t | 0.471 | 0.167 | 0.042 |
| total_input_amount | 0.488 | 0.144 | 0.105 |
| counterparty_mean_n_txs_asof_t | 0.687 | 0.132 | 0.095 |
| fee | 1.387 | 0.116 | 0.116 |
| input_spread | 0.523 | 0.116 | 0.072 |
| upstream_peel_share | 0.157 | 0.102 | 0.080 |
| upstream_funded_share | 0.495 | 0.054 | 0.019 |
| upstream_mean_output_count | 0.635 | 0.051 | 0.111 |
| is_peeling_candidate | 0.000 | 0.037 | 0.001 |
| upstream_chain_depth | 0.353 | 0.014 | 0.049 |
| upstream_min_hold_seconds | 0.141 | 0.012 | 0.006 |
| n_recv_asof_t | 0.011 | 0.009 | 0.005 |
| n_txs_asof_t | 0.012 | 0.007 | 0.004 |
| unique_counterparties_asof_t | 0.012 | 0.007 | 0.004 |
| net_flow_asof_t | 0.041 | 0.002 | 0.003 |
| btc_recv_total_asof_t | 0.036 | 0.001 | 0.003 |
| addr_is_sender | 0.343 | 0.000 | 0.002 |
| n_sent_asof_t | 0.000 | 0.000 | 0.000 |
| active_duration_seconds | 0.000 | 0.000 | 0.000 |
| gap_since_last_tx | 0.000 | 0.000 | 0.000 |
| is_mixing_candidate | 0.000 | 0.000 | 0.000 |
| addr_is_self_change | 0.000 | 0.000 | 0.000 |
| upstream_mix_share | 0.000 | 0.000 | 0.000 |

The largest degraded-period shifts are the first rows above. They are principally transaction shape, fee/amount, counterparty, and upstream-flow features. But these shifts are not unique to failure windows: the recorded runtime monitor rated all holdout steps within the development baseline, including `t43`, `t45`, and `t47`.

## 4. Positive/negative distribution comparison

For each high-importance model feature, the table gives a one-feature ROC-AUC with its direction fixed from training (`0.5` = no class separation). This directly tests whether an earlier predictive relationship remains discriminative later; it does not train any model.

| Feature | v5 importance | Train AUC | Earlier-good AUC | Degraded AUC | All-holdout AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| fee | 1214 | 0.516 | 0.836 | 0.251 | 0.491 |
| output_spread | 695 | 0.876 | 0.969 | 0.770 | 0.856 |
| input_amount_mean | 691 | 0.721 | 0.921 | 0.766 | 0.846 |
| fee_ratio | 684 | 0.696 | 0.900 | 0.718 | 0.791 |
| upstream_mean_output_count | 668 | 0.643 | 0.643 | 0.759 | 0.556 |
| input_spread | 557 | 0.537 | 0.752 | 0.395 | 0.523 |
| output_amount_mean | 502 | 0.496 | 0.801 | 0.308 | 0.409 |
| total_input_amount | 485 | 0.679 | 0.802 | 0.773 | 0.763 |
| upstream_mean_input_count | 413 | 0.523 | 0.631 | 0.528 | 0.510 |
| btc_recv_total_asof_t | 392 | 0.787 | 0.917 | 0.586 | 0.591 |

- Positive behavior changed: retained diagnostics show failing positives have lower fan-in and upstream-funded share, and more cold starts than development positives (for `t43/t45/t47`: fan-in `4/3/8` vs development `103`; upstream-funded share `0.65/0.56/0.78` vs `0.91`; no-prior-transaction `0.52/0.59/0.72` vs `0.23`).
- Negative behavior also shifts on several inputs (the PSI table), but the score/separation evidence specifically shows the class relationship weakening in failure windows.
- Label/prevalence shift is substantial: `t43=1.6%`, `t45=0.3%`, and `t47=2.9%`, versus every recorded development fold at or above `4.8%`. The holdout positives are clustered into very few independent transactions (`24`, `5`, and `22` respectively), so per-step estimates are high variance.

## 5. Score-distribution comparison (frozen v5; diagnostic only)

Raw scores are the model's ranking score. `Positive ≤ negative p95` is a simple overlap measure: lower is better; a large percentage means many positives fall below a high negative-score cutoff. All values below were computed with the frozen artifact; no threshold was applied or selected.

| Cohort | n | Prevalence | Median raw score | Positive raw median | Negative raw median | Positive calibrated median | Negative calibrated median | Raw-score AUC | Positive ≤ negative p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train t26-41@41 | 65,161 | 8.246% | 0.0001 | 0.9971 | 0.0001 | 0.9799 | 0.0018 | 1.000 | 0.0% |
| good t33-34@34 | 9,002 | 7.987% | 0.0000 | 0.9994 | 0.0000 | 0.9928 | 0.0010 | 0.999 | 0.3% |
| degraded t43/t45/t47@49 | 21,523 | 1.487% | 0.0001 | 0.0678 | 0.0001 | 0.1371 | 0.0019 | 0.923 | 44.1% |
| holdout t42-49@49 | 54,335 | 4.634% | 0.0002 | 0.1273 | 0.0001 | 0.2031 | 0.0022 | 0.946 | 29.1% |

| Holdout first-seen step | n | Positives | Prevalence | Positive median raw score | Negative median raw score | Raw-score AUC | Positive ≤ negative p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t42 | 9,188 | 399 | 4.343% | 0.5149 | 0.0010 | 0.965 | 21.6% |
| t43 | 5,928 | 97 | 1.636% | 0.0441 | 0.0002 | 0.910 | 55.7% |
| t44 | 7,877 | 279 | 3.542% | 0.9987 | 0.0006 | 0.948 | 30.5% |
| t45 | 8,871 | 27 | 0.304% | 0.0729 | 0.0001 | 0.916 | 77.8% |
| t46 | 4,335 | 508 | 11.719% | 0.1995 | 0.0001 | 0.998 | 0.0% |
| t47 | 6,724 | 196 | 2.915% | 0.0726 | 0.0001 | 0.941 | 32.7% |
| t48 | 7,089 | 381 | 5.375% | 0.0574 | 0.0000 | 0.959 | 22.0% |
| t49 | 4,323 | 631 | 14.596% | 0.0677 | 0.0001 | 0.956 | 31.4% |

Later failure-step positives receive substantially less separable scores than well-performing periods; this is ranking failure, so a fixed classification threshold cannot restore nAP or P@100. Calibrated probabilities are monotone in these raw scores, so calibration cannot change the ranking conclusion.

## 6. Evidence by possible cause

| Possible cause | Evidence | Assessment |
| --- | --- | --- |
| A. Covariate drift | Major absolute PSI appears in many features, including the degraded cohort. However all `t42-49` steps were `WITHIN_BASELINE` on the retained relative-drift monitor, regardless of performance. | Present, but not sufficient as the primary explanation. |
| B. Concept / relationship drift | Earlier-important feature separation and frozen-model score separation weaken in degraded steps; failing positives have a different structural typology. | Strongest supported cause. |
| C. Label / prevalence shift | Prevalence falls to 1.6%, 0.3%, and 2.9%; only 24, 5, and 22 independent illicit transactions underpin `t43/t45/t47`. Labels were previously audited as internally consistent. | Strong contributor and makes step results high variance. |
| D. Protocol/evaluation difference | Development uses two-step rolling refit windows; holdout uses one fixed model and a single eight-step window with snapshots at `t49`. This accounts for some aggregate comparison difference, but cannot explain abrupt `t43/t45/t47` collapses within the same holdout protocol. | Partial contributor, not root cause of episodic collapse. |
| E. Implementation mismatch | The same current causal production engine and `/5` feature contract generated development and holdout diagnostics; artifact hash is unchanged; holdout scorecard matches the published result. | No evidence. |

## 7. Most likely cause and confidence

**Most likely cause:** a change in the positive class's transaction/graph typology (concept/relationship drift) combined with a severe prevalence and independent-event-count collapse. The current model recognizes the dominant development-era consolidation/upstream-funded positive pattern, but does not rank the small, weakly funded, cold-start-heavy positive patterns in `t43/t45/t47` as well.

**Confidence: moderate.** The direction is supported by recorded results, retained feature diagnostics, and this fresh frozen-model score diagnosis. It is not high because `t42-49` is a seen/reused holdout, and the failure steps have very few independent illicit transactions.

## 8. Is it fixable without retraining?

- **Threshold problem:** no. nAP/P@100 and raw-score separation degrade; changing a classification cutoff cannot repair ranking.
- **Calibration problem:** not the main issue. Calibration is monotone and the pooled holdout calibration record is good (`ECE=0.010`); it cannot improve rank separation.
- **Feature robustness / model retraining:** likely requires a future, prospectively evaluated model/data strategy, not a serving-time patch. No fix is implemented here.
- **Protocol problem:** it explains part of the headline comparison, so future experiments should retain matched evaluation units; it does not remove the within-holdout collapse.

## 9. Recommended next experiment

Pre-register a **development-only** typology-robustness experiment: split the existing `t17-41` protocol-B confirmation folds by independent transaction/event clusters and cold-start/upstream-funded strata, then test a deliberately specified robustness feature or training objective only on those folds. Do not choose it from `t42-49`; use delayed labels from new production traffic for an unbiased final generalization test.

## 10. Integrity statement

**No production code, model, registry, feature schema, threshold, or sealed holdout artifact was changed or inspected.** This report reused recorded holdout/development performance and, under ADR 0004's diagnosis allowance, generated read-only causal features from the existing raw source for score/distribution diagnosis. Labels were joined only after feature generation; no model was trained or retrained.
