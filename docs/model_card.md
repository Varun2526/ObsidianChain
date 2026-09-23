# Model card — ps_native_v5 (champion)

| | |
|---|---|
| Model | LightGBM binary classifier, 300 trees, 31 leaves, min_child_samples 50, lr 0.05, subsample 0.8, colsample 0.8, seed 20260919 |
| Features | 31, schema `ps_native_features/5` (`docs/feature_catalog.md`) |
| Training data | Elliptic++ addresses first seen in t26-41, each at its last event at or before t41; labels classes 1 (illicit) / 2 (licit), unknown excluded |
| Evaluation unit | protocol B: rank addresses NEW in a window, as of the window's end |
| Ranking score | raw model probability; order unaffected by calibration |
| Display probability | Platt on logit(raw), fitted on out-of-fold scores of folds 9-12 |
| Explanations | exact per-row TreeSHAP (`pred_contrib`), per-feature sign; associations, not causes |
| Severity | rank budget per run (1% / 5% / 15%), a row above LOW only if above base rate |
| Registry | `data/models/ps_native/registry.json`; attested source commit 5431241; holdout result sha256 3e9e2f4b... |
| Fallback | ps_native_v5_fallback_no_g (no group G), same schema; an emergency mode that FAILS the champion gate |

## Intended use

Prioritising addresses in an offline Bitcoin capture for an investigator's
review: a ranked worklist with evidence. **Not intended** for automated
action, sanctioning, or any decision about a person without human review.

## Performance (see `docs/results_register.md` for status of every number)

- Development + confirmation (12 folds): nAP 0.808 (worst 0.477), address
  P@100 0.97, ECE 0.025.
- **Holdout t42-49:** nAP 0.548, ROC-AUC 0.946, P@100 1.00 pooled, ECE 0.010.

## Known failure modes

1. **Regime change.** On holdout windows t43, t45 and t47, P@100 was 0.18,
   0.02 and 0.28. After the known t43 dark-market closure, new illicit
   behaviour is not recognised. **Input drift monitoring does not detect
   this** (exp23). Only delayed-label review can.
2. **Weak slices** (holdout nAP): receive-only addresses 0.17, high-value
   transactions 0.14, first-seen addresses 0.43.
3. **Train/serve differences:**
   - per-address amounts on Elliptic++ are even splits;
   - a missing fee is scored as zero;
   - real captures have real timestamps, while Elliptic++ has two-week
     surrogates.
4. **Labels** are Elliptic++ wallet classes: a labelled subset, static per
   address, not a ground truth of criminality.

## Determinism and batch behaviour

- **The model score is batch-invariant.** The same row, model and features
  give the same raw score, alone or in any batch
  (`test_a_score_does_not_depend_on_the_batch`). Runs are reproducible
  byte-for-byte in `predictions.parquet`.
- **Relative to the capture by design:**
  - severity, which is a rank budget within the run;
  - watchlist propagation and embedding link suggestions, which depend on the
    capture's graph;
  - the fused alert score, which includes propagation when seeds are present.

  None of these is a model output; each is labelled with its evidence class.

## Ethical and operational notes

A high score means "resembles addresses labelled illicit in Elliptic++ and
is structurally close to them". It is not evidence of a crime. Mixing and
privacy-preserving transactions have legitimate uses. The model's precision
on any given capture is unknown until labels arrive.
