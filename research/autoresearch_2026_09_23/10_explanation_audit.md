# 10 — Explanation Audit (exp05_explanation_audit, completed 2026-09-23)

Full results: `results/exp05_explanation_audit.json`. Same fold/model as
exp03/exp04. **This is the single most operationally serious finding in
this research program.**

## Method

`ps_model.py::predict_address_features` (the current production explanation
method) computes, per prediction: `contribution = global_feature_importances_
× |raw_feature_value|`, takes the top 3 by that score, and assigns every one
of those 3 features the **same direction label** — `"INCREASES_RISK"` if the
row's overall calibrated probability is ≥ 0.5, `"DECREASES_RISK"` otherwise
(`ml/ps_model.py:180-186`, confirmed by direct reading in `01_repo_audit.md`).

This experiment computes **exact TreeSHAP** for the same fitted LightGBM
model on the same rows, via `Booster.predict(pred_contrib=True)` — LightGBM's
own built-in, exact implementation (not an approximation, not the external
`shap` package, which is not installed — see `01_repo_audit.md`). Verified
exact before trusting it: `sum(shap_values) + bias = raw_margin` to within
1.91e-14 (floating-point precision) across the eval set — this is local
fidelity **by construction** for TreeSHAP, and the check confirms the
computation itself is correct before using it as ground truth.

## Result 1: top-3 feature selection is frequently wrong

Sampled 500 eval rows, compared each row's current-method top-3 features
against its true TreeSHAP top-3 (by `|contribution|`):

- Mean overlap: **1.40 of 3** features in common.
- **9.0% of rows share zero features** between the two methods' "top 3
  reasons."
- Only **4.8% of rows** have full (3/3) agreement.

## Result 2: the direction label is wrong more often than right

1,500 (feature, row) pairs (500 rows × top-3 TreeSHAP features each):
current method's row-level direction label agrees with that specific
feature's own TreeSHAP sign only **33.9% of the time** — 992 of 1,500
disagreements. **This is worse than the 50% a coin flip would give.**

**Mechanism, confirmed by the method itself, not inferred:** the current
method assigns one direction to all 3 reported features based solely on
whether the *overall* row crossed the 0.5 probability threshold. A feature
that is individually pushing risk *down* (negative SHAP value) on a row
whose *other* features pushed the overall score above 0.5 gets labeled
`"INCREASES_RISK"` anyway — and this is not a rare edge case, it is the
majority behavior (66.1% of sampled feature-level directions are mislabeled
this way).

## What this means

An investigator reading a current `ps_model.py` explanation is told "these
3 features, all pushing risk up" (or all down) when in reality:
- Roughly 1.6 of those 3 features (on average) are not even in the model's
  actual top-3 most-influential features for that specific prediction.
- The direction claim is more often wrong than right for any individual
  feature in that list.

This is not "explainability with rough edges" — the current mechanism
routinely produces attributions that would mislead an investigator's
mental model of *why* an address was flagged, which is a direct
operational/evidentiary risk for a system whose stated design goal is
investigative usefulness and explainability (this program's own Phase 2
protocol objectives).

## Is this "explainable AI"? (Phase 9's explicit question)

No, not as currently implemented — global feature importance weighted by a
row's own feature magnitude is a **plausible-looking heuristic**, not a
local attribution method, and this experiment shows the gap between
"looks like an explanation" and "is a faithful explanation" is large and
measurable, not theoretical.

## Decision

- **REJECT** the current `ps_model.py` explanation method as faithful to
  the model's actual behavior — confirmed by direct, exact comparison, not
  assumed.
- **KEEP** (recommend adopting): TreeSHAP via `Booster.predict(pred_contrib=True)`
  for any LightGBM-family candidate — it is exact, already available without
  new dependencies (no `shap` package needed for LightGBM specifically —
  this only covers LightGBM; if a non-LightGBM model is eventually chosen,
  a different TreeSHAP path or the external `shap` package would be needed,
  and `shap` is not currently vendored — see `01_repo_audit.md`).
- **Scope limit, stated explicitly**: this experiment used LightGBM, one
  member of exp01's tied top tier. RandomForest and HistGradientBoosting
  would each need their own TreeSHAP path checked (`sklearn`'s
  `RandomForestClassifier` does not have a built-in exact TreeSHAP the way
  LightGBM does; would require the external `shap` package, not installed).
  This is a concrete, evidenced reason to prefer LightGBM specifically if
  Phase 17 has to pick one model from the tied tier and explanation fidelity
  is a deciding factor — the first time in this program that a
  model-family choice has evidence behind it beyond ranking quality alone.
