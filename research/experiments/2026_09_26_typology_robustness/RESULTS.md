# E0/E1/E2 Typology-Robustness Development Experiment

## Scope and integrity

This is a **pre-specified exploratory, development-only** comparison. It used
only `t1-41` and the twelve fixed Protocol-B folds. `t42-49`, sealed/reused
holdout artifacts, the champion artifact, registry, production code, feature
contract, and classification thresholds were not read or changed. All models
were transient per-fold LightGBM fits; no model artifact was saved.

The complete fixed specification is [PREREGISTRATION.md](PREREGISTRATION.md).

## Winner rule result

Winner by the pre-specified lexicographic development-only rule: **E0**.
This is not an adoption or production-promotion decision.

## Compact comparison

| Experiment | Mean nAP | Std | Worst Fold | Cold-start nAP | Weak-upstream nAP | P@100 | Delta vs E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.8078 | 0.1757 | 0.4771 | 0.5629 | 0.5440 | 0.9708 | +0.0000 |
| E1 | 0.8068 | 0.1799 | 0.4701 | 0.5506 | 0.5313 | 0.9717 | -0.0010 |
| E2 | 0.7903 | 0.1566 | 0.5209 | 0.5036 | 0.5038 | 0.9633 | -0.0175 |

## Per-fold performance

| Fold | E0 nAP | E1 nAP | E2 nAP | E0 P@100 | E1 P@100 | E2 P@100 | Test positives | Independent positive test tx clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t<=16 -> t17-18 | 0.8196 | 0.8374 | 0.8335 | 0.91 | 0.92 | 0.91 | 202 | 161 |
| t<=18 -> t19-20 | 0.7907 | 0.7900 | 0.8012 | 1.00 | 1.00 | 1.00 | 546 | 368 |
| t<=20 -> t21-22 | 0.4771 | 0.4701 | 0.5209 | 1.00 | 1.00 | 1.00 | 1321 | 288 |
| t<=22 -> t23-24 | 0.9059 | 0.9084 | 0.8431 | 1.00 | 1.00 | 1.00 | 936 | 201 |
| t<=24 -> t25-26 | 0.9748 | 0.9757 | 0.9284 | 1.00 | 1.00 | 1.00 | 1810 | 245 |
| t<=26 -> t27-28 | 0.9495 | 0.9466 | 0.9482 | 0.98 | 0.97 | 0.98 | 145 | 124 |
| t<=28 -> t29-30 | 0.5534 | 0.5312 | 0.5681 | 0.76 | 0.77 | 0.76 | 525 | 414 |
| t<=30 -> t31-32 | 0.8497 | 0.8383 | 0.7705 | 1.00 | 1.00 | 0.99 | 903 | 417 |
| t<=32 -> t33-34 | 0.9794 | 0.9803 | 0.9668 | 1.00 | 1.00 | 0.99 | 719 | 71 |
| t<=34 -> t35-36 | 0.9779 | 0.9782 | 0.9216 | 1.00 | 1.00 | 1.00 | 1018 | 231 |
| t<=36 -> t37-38 | 0.8252 | 0.8328 | 0.8143 | 1.00 | 1.00 | 1.00 | 612 | 167 |
| t<=38 -> t39-40 | 0.5908 | 0.5928 | 0.5672 | 1.00 | 1.00 | 0.93 | 569 | 226 |

## Aggregate metrics and uncertainty

| Experiment | Mean nAP | SD | Median nAP | Worst nAP | Cold-start nAP | Weak-upstream nAP | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.8078 | 0.1757 | 0.8374 | 0.4771 | 0.5629 | 0.5440 | 0.9708 | 0.1983 |
| E1 | 0.8068 | 0.1799 | 0.8378 | 0.4701 | 0.5506 | 0.5313 | 0.9717 | 0.1983 |
| E2 | 0.7903 | 0.1566 | 0.8239 | 0.5209 | 0.5036 | 0.5038 | 0.9633 | 0.1970 |

- E1 − E0: mean paired nAP delta `-0.0010`, bootstrap 95% CI `-0.0064` to `+0.0041`, fold wins/losses/ties `7/5/0`.
- E2 − E0: mean paired nAP delta `-0.0175`, bootstrap 95% CI `-0.0377` to `+0.0021`, fold wins/losses/ties `4/8/0`.

Bootstrap resamples fixed at 10,000 over the 12 paired fold deltas (seed `20260926`). These intervals describe variability across the fixed development windows; they do not establish unseen future-period generalization.

## Exact methods

- **E0 schema:** 31 `ps_native_features/5` CORE columns.
- **E1/E2 schema:** E0 plus six fixed, research-only causal columns (`research_cold_start`, `research_weak_upstream`, `research_log_fan_in`, `research_relationship_density`, `research_cold_x_weak_upstream`, `research_cold_x_upstream_share`). Production schema remains unchanged.
- **All models:** LightGBM with seed `20260919` and `{"colsample_bytree": 0.8, "learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 300, "n_jobs": -1, "num_leaves": 31, "random_state": 20260919, "subsample": 0.8, "subsample_freq": 1, "verbose": -1}`.
- **E2 clusters/weights:** positive snapshot transaction ID is the deterministic event cluster; its positive training rows share one raw unit. Negative rows retain unit weight. Weights are normalized to total training-row count. Per-fold training and test sizes, positive counts, and independent positive-cluster counts are in `per_fold.csv`.
- **Causality and labels:** product features were generated from the raw t1-41 stream before the label join. Each snapshot is at its fold boundary; test labels were used only after scoring for metrics, never in fit or E2 weights.
- **Commit:** `d0ca55134fdd995cff6353ecfd5e3f813140e129`.

## Conclusion

**Neither helped on mean nap.** E1 is mixed and near-neutral (7 wins, 5 losses; its bootstrap interval spans zero). E2 is lower on mean nAP in 8 of 12 folds despite a better worst fold, so that safety improvement is not accompanied by consistent mean improvement. The winner is only the pre-specified best development configuration; it must not be selected, threshold-tuned, or promoted using `t42-49`.

## Generated artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| PREREGISTRATION.md | `b5b3fe84d11a877bce7c235bd704724328b532b807ce21bb01a9229ac4de0cb0` |
| run.py | `81e92d9920ef98c84faef0f3691c34d0a86c47b5762cded3e5bc01b23802e78a` |
| per_fold.csv | `66d10c0c04ba6c4afeee70313e5ca2a5d2b6f71da6e4629dab8986a1d0bed0b3` |
| metrics.json | `ac58e61ab70754961eb04e998b614a9d9d04e3ba97d4845d34aeddc9a1bcde57` |
