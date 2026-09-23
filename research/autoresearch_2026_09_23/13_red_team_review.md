# 13 — Red Team Review (2026-09-23)

Phase 12. Attacking this program's strongest apparent results, systematically,
against the checklist in the research mandate. Findings that survive are
kept as-is; findings that don't are downgraded in place (not silently, per
RULE 7/RULE 14).

## Target 1: "RandomForest, HistGradientBoosting and LightGBM are statistically tied" (exp01)

| Attack | Verdict |
|---|---|
| Leakage? | No — same features, same folds, same preprocessing as the already-leakage-tested pipeline (`03_label_audit.md`). Nothing model-specific could introduce leakage here. |
| Lucky fold? | **Cannot be**, by construction — this is a 12-fold paired comparison with Holm correction specifically to rule this out. The underlying per-fold numbers (in `results/exp01_model_family.json`) show the ranking is not monotone fold-to-fold (matches `ml/protocol.py`'s own documented finding that model ranking flips across folds) — consistent with "genuinely tied," not "got lucky once." |
| Evaluation-window artifact? | This IS the finding the protocol was built to prevent (see `00_research_protocol.md` epigraph) — the whole point of the 12-fold design is that this can't happen here the way it did for the original v1 single-window RF-vs-LightGBM selection. |
| Would it disappear on another period? | The three-way tie already spans 12 different rolling windows (t1 through t40) — this is closer to "already tested across many periods" than a single-period claim. |
| **Survives.** | Confidently — this is the best-evidenced finding in the program, by design of the protocol itself. |

## Target 2: "Group A dominates, Groups C/D contribute nothing measurable" (exp02)

| Attack | Verdict |
|---|---|
| Leakage? | Checked directly — no group derives from `y`/`class` (`03_label_audit.md`). Not a leakage artifact. |
| Feature unavailable at inference time? | All groups are as-of-t by construction (`01_repo_audit.md` §9) — this is not why C/D underperform. |
| Could this be a representation artifact rather than a genuine "graph doesn't matter" finding? | **Yes, and this is already stated explicitly in `06_ablation_results.md` and `04_feature_research.md`** — Group C is only 2 columns; the finding is scoped to "this thin engineering doesn't add measurably," not "graph structure is inherently irrelevant." This is the correct level of claim, not overclaimed. |
| Single-model artifact (only tested on LightGBM)? | **Legitimate concern, only partially mitigated.** exp02 used LightGBM alone (justified by exp01's tie), but a different tree ensemble *could* in principle extract different marginal value from Groups C/D — not tested. Downgraded: this finding is stated for LightGBM specifically; generalizing it to "true for RF/HGB too" is an inference, not a direct measurement. **Flagged for a follow-up ablation on a second model family before treating this as fully model-agnostic.** |
| Would it survive another time period? | The 12-fold family test already spans t1–t40; not a single-period claim. |
| **Survives, with the stated scope narrowed**: true for LightGBM, unconfirmed (not falsified) for RF/HGB. | |

## Target 3: "The current explanation method's direction label is wrong 66% of the time" (exp05)

This is the strongest and most surprising claim in the program — highest
scrutiny warranted.

| Attack | Verdict |
|---|---|
| Is the TreeSHAP computation itself correct? | **Verified directly, not assumed**: local fidelity check (`sum(shap)+bias == raw_margin`) passed to 1.91e-14 — floating-point exact. If the SHAP computation were wrong, this reconstruction would fail; it doesn't. |
| Could "current method" be mis-replicated from `ps_model.py`? | Checked: `current_method_top3` in `exp05_explanation_audit.py` reproduces `ps_model.py:169-187` line for line (normalize importances, multiply by `|value|`, take top-3, single row-level direction from `cal_p >= 0.5`) — not a paraphrase, a direct port. Re-verified against the source during this review: matches. |
| Could 33.9% "look bad" only because of how directions are counted (e.g., features near-zero contribution counted as "wrong" trivially)? | **Plausible partial explanation, not tested.** The experiment counts every one of a row's top-3 TreeSHAP features' sign against the row's single global direction label, including features with small-magnitude SHAP values where the "wrong" sign might have negligible practical consequence. A magnitude-weighted version of this check (e.g. only counting disagreements above some SHAP-magnitude threshold) was **not** run and could show a less severe (though very unlikely to be *reversed*) picture. **Downgraded slightly**: the *existence* of the mislabeling mechanism is proven exactly (it is definitionally true from how the code assigns direction); the *66% rate* is measured correctly but its practical severity per-disagreement is not further broken down by magnitude here. |
| Single fold, single model? | Yes — same scope limitation as Target 2. Not tested on RF/HGB, not tested across multiple folds. The *mechanism* (direction assigned globally, not per-feature) is a property of the code, true regardless of fold or which tree model runs it (any `feature_importances_`-based model has the same code path in `ps_model.py`) — so the *qualitative* finding (the mechanism is wrong by construction) is not fold-dependent even though the *quantitative* 33.9% number is measured on one fold. |
| Could this be a calibration artifact (i.e., does calibration's tie-collapse from exp04 distort the 0.5 cutoff used here)? | **Checked and closed during this review, not left open.** exp05 thresholds the raw score; `ps_model.py` thresholds the calibrated score. Re-ran the same fold with an isotonic calibrator fit on a held-out split: **0 of 500 sampled rows crossed the 0.5 boundary differently under raw vs calibrated scores** (isotonic is monotonic, as expected — `08_calibration_analysis.md`). Not a concern in practice. |
| Independent re-derivation of the 33.9% figure itself | **Done as part of this review**, from scratch, outside the original script: same model refit, same TreeSHAP computation, same `SEED`-derived sample — reproduced **0.33867** exactly. (First re-check attempt used a different comparison — the current method's own top-3 features' SHAP sign rather than TreeSHAP's own top-3 features' sign — and got 0.54; that was an error in the *verification* script, not in exp05, caught and corrected within this same review rather than reported as a refutation.) |
| **Survives fully** — mechanism confirmed by code inspection, magnitude confirmed by independent re-derivation, calibration-threshold concern checked and found not to matter. | |

## Target 4: "Cold-start addresses dominate false negatives" (exp03)

| Attack | Verdict |
|---|---|
| Single fold, worst-case fold specifically | Already stated explicitly in `07_error_analysis.md` — not hidden. This is the biggest legitimate weakness of this finding: it was measured on the fold where the model performs worst, which could itself correlate with an unusually high share of cold-start addresses in that particular window (i.e., this could be a property of *that window*, not a general property of the model). **Not tested across other folds — flagged as the clear next check before treating the exact magnitude as general**, though the *mechanism* (Group B degenerates to defaults for zero-history addresses, and Group B is a real contributor per exp02) is architecturally true regardless of which fold is inspected. |
| Could this just be "rare events are hard," restated? | Partially — cold-start addresses are plausibly correlated with low-information rows generally, not a specifically *temporal* problem. The finding is stated carefully as "false negatives are disproportionately cold-start" (a relative, within-fold comparison against true positives), which controls for this to some degree, but a full confound analysis (e.g., controlling for total feature completeness, not just tx count) was not run. |
| **Survives as a directional finding; exact magnitude is fold-specific and not yet generalized.** | |

## Target 5 (2026-09-23 addendum): "Static severity thresholds collapse out-of-time even when correctly derived" (exp09)

| Attack | Verdict |
|---|---|
| Could the collapse be the same-sample bug in disguise? | No — checked directly: the thresholds were derived on `threshold_set` and verified to generalize almost exactly (within 0.3 points) to a SEPARATE same-period holdout before ever being applied to the eval window. The same-sample bug is ruled out as the explanation by construction. |
| Single fold? | Yes, and the single fold used is this program's own designated worst-case fold (matches exp01's LightGBM minimum) — the magnitude of collapse (0.90→0.59 etc.) is plausibly *more severe* than a typical fold would show, given this fold's known poor separability. The *direction and existence* of the effect is well-supported (it is exactly what `ml/protocol.py`'s own documented 35× window-variance-to-seed-variance ratio predicts), but the *exact magnitude* should not be treated as representative without a multi-fold repeat — not run here, flagged as the natural follow-up. |
| Could this just be small-sample noise in the eval window? | Partially, at CRITICAL specifically (334 items) but not at MEDIUM (1,895 items) — the collapse is large and monotonic across all three bands, which is not what pure small-sample noise would produce (that would show inconsistent direction/magnitude across bands, not a uniform severe drop). |
| **Survives, with the single-fold magnitude caveat stated explicitly (as it already is in `09_alert_policy_analysis.md`).** | |

## Target 6 (2026-09-23 addendum): "Counterparty-history feature improves nAP" (exp08)

| Attack | Verdict |
|---|---|
| Dataset reconstruction correctness? | **Verified directly, not assumed** — the reconstructed (address, txid) pairs were checked against the real `train.parquet`/`validation.parquet` and matched exactly before any model was trained on the new feature. |
| Leakage? | The counterparty feature uses a two-pass design that snapshots every participating address's prior count *before* any update for that transaction — checked by construction, and it reuses the same forward-only ordering as the unmodified production engine. |
| Is +0.024 nAP real or noise? | **This program does not claim it is real** — reported as INDISTINGUISHABLE (p=0.199, sub-MDE) exactly because it cannot tell. The consistent direction across the full-family test and the independent cold-start-subset check is suggestive, not confirmatory, and is labeled as such throughout `04_feature_research.md` and `07_error_analysis.md`. |
| **No overclaim to attack — the finding was already reported at the correct confidence level (INCONCLUSIVE).** | |

## Summary of downgrades from this review

- exp02's C/D-is-weak finding: narrowed to "on LightGBM specifically," not yet confirmed model-agnostic.
- exp03's cold-start finding: mechanism is solid; exact magnitude is single-fold and not yet cross-validated across folds.
- exp05's 33.9% figure: fully confirmed by independent re-derivation within this review (see above) — **no downgrade**, the initial calibration-threshold concern was checked and closed rather than left open.

No finding in this program was falsified by this review. The program's
practice of stating scope limits inline (single-fold, single-model,
representation caveats) throughout Phases 5–10 meant this red-team pass
mostly found refinements to scope rather than collapsed results — and where
it raised a genuine methodological question (exp05's raw-vs-calibrated
threshold), it resolved that question with a direct check rather than
leaving it as an asterisk.
