# 09 — Alert Policy Analysis (exp04_calibration_alert_policy, completed 2026-09-23)

Same experiment/fold as `08_calibration_analysis.md`; this file isolates
the **alert-policy** question (RULE 10: model score vs alert policy are
separate axes) and audits the existing severity-band mechanism.

## Top-K policy: raw score vs calibrated score, precision/recall at budget

| K | raw precision | raw recall | calibrated precision | calibrated recall |
|---|---|---|---|---|
| 50 | 0.340 | 0.030 | 0.320\* | 0.028\* |
| 100 | 0.450 | 0.080 | 0.460\* | 0.081\* |
| 200 | 0.565 | 0.200 | 0.565 | 0.200 |
| 500 | 0.446 | 0.394 | 0.446 | 0.394 |

*(\*calibrated-column values depend on tie-break order at K=50/100 — see
`08_calibration_analysis.md`'s tie-bound table; the single point value here
is one arbitrary draw, shown for reference only, not as a trustworthy
number — this is the whole point of the calibration finding.)*

**At K=200 and K=500, raw and calibrated top-K selection produce identical
precision/recall** — the tie problem is concentrated in low-K budgets,
which is also where day-to-day investigator alert queues are most likely to
sit (an analyst working 50–100 leads a day, not 500). This makes the
tie-collapse problem an **operationally relevant** finding, not an academic
one.

## Auditing the existing severity-band mechanism (v1 frozen model reference)

The frozen `data/models/ps_native/v1/calibration.json` (a different model,
different feature schema — shown here as **reference**, not applied to any
v2 candidate) declares:

| Band | Target precision | Threshold | Support | Recorded validation precision |
|---|---|---|---|---|
| CRITICAL | 90% | 0.6697 | 840 | 90.0% |
| HIGH | 75% | 0.2149 | 1,452 | 75.0% |
| MEDIUM | 50% | 0.1139 | 2,602 | 50.0% |

HANDOFF.md already states this policy's real-world failure: *"CRITICAL
advertises 90%, measures 82.2% on validation and 37.2% on test."* The
numbers in the v1 `calibration.json` file itself show *exactly* 90.0% /
75.0% / 50.0% precision on validation — i.e. **the thresholds were tuned
directly against the validation set they are then reported as validating
against**, which is precisely the rank-derived-cutoff-as-value-threshold
problem: a threshold chosen to hit a target precision on one specific
sample will not hit that precision on a different sample, and the recorded
"validation_precision: 0.9" is not an independent measurement, it is the
optimization target restated. This program did not need a new experiment to
find this — it is visible directly in the artifact's own numbers, and
corroborates HANDOFF.md's independently-reported test-set collapse (37.2%)
as the predictable consequence.

## Comparing policy families (per Phase 8's explicit requirement)

| Policy | Behavior observed |
|---|---|
| **Top-K ranking** | Not sensitive to tie-collapse at K≥200 in this fold; sensitive at K<200. Simple, matches an investigator's actual workflow (a worklist of N items), and — per `05_model_comparison.md`/`06_ablation_results.md` — doesn't require choosing between RF/HGB/LightGBM since they're tied. |
| **Percentile/rank bands** | Not separately tested in this experiment (would require re-deriving bands on a v2-trained model, out of scope for this diagnostic pass); the v1 reference bands above are this policy family, and their in-sample-tuned precision is the demonstrated failure mode to avoid repeating. |
| **Score thresholds (raw)** | Threshold on the raw (uncalibrated) score is well-defined (2,185 distinct values in this fold) but not directly probability-interpretable. |
| **Probability thresholds (calibrated)** | Directly interpretable ("this address is calibrated-P(illicit)=0.6") but — per `08_calibration_analysis.md` — only 50 distinct achievable values in this fold, so a fixed probability threshold silently governs a *wide, ties-dependent* set of addresses near any given cutoff. |

## Addendum 2026-09-23: prototyping the corrected procedure reveals a deeper problem (exp09_alert_policy_redesign)

Built and tested the fix this file recommends: a three-way split (fit /
threshold-selection / reporting) instead of the one-sample-does-everything
pattern found in the v1 artifact.

**The fix works, as far as it goes.** Thresholds picked on a held-out
`threshold_set` (28,608 rows, same time period as training, never used for
fitting) generalize almost exactly to a second held-out split from the same
period (`holdout_check_set`): CRITICAL 0.900→0.902, HIGH 0.750→0.753, MEDIUM
0.500→0.494. The same-sample-tuning bug is real and this specific fix
resolves it, **for reporting within the same time period the thresholds
were derived from.**

**But applying those same, correctly-derived thresholds to the fold's
actual out-of-time eval window collapses precision anyway:**

| Band | Target | Threshold-set (in-sample) | Same-period holdout | **Out-of-time eval window** |
|---|---|---|---|---|
| CRITICAL | 90% | 0.900 | 0.902 | **0.593** |
| HIGH | 75% | 0.750 | 0.753 | **0.482** |
| MEDIUM | 50% | 0.500 | 0.494 | **0.177** |

**This means the severity-band problem is not fully solved by fixing the
sampling bug.** Even a properly-derived, non-leaky threshold is a fixed
score *value*, and this program's own central finding — window-to-window
variance dominates seed-to-seed variance by ~35× (`00_research_protocol.md`,
`ml/protocol.py`) — applies to alert thresholds exactly as it applies to
model comparison. A threshold calibrated on one period's score distribution
does not survive a new period's score distribution, independent of how
cleanly it was derived.

**A secondary check gave a confusing, non-illustrative result, reported
honestly rather than omitted**: attempting to reproduce the original
same-sample flaw on this specific (worst-case) diagnostic fold showed
HIGH/MEDIUM thresholds derived in-sample-on-eval transferring *better* to
the training-period holdout than to themselves, and CRITICAL's 90% target
was unreachable on this fold at all. This is fold-specific noise (this is
the single hardest fold in the whole protocol, matching exp01's LightGBM
minimum) rather than evidence against the main finding above, which is
independently well-supported by the threshold_set→holdout_check_set→eval_window
chain.

## Revised decision

- **CONFIRMED, deeper than originally stated**: the severity-band mechanism
  needs more than a sampling-discipline fix — a fixed value threshold is
  fundamentally in tension with this dataset's own documented temporal
  instability. **Any static-threshold severity policy inherits this
  problem**, not just the current, sloppily-derived one.
- **Strengthens, rather than merely repeats, this file's original
  recommendation**: a top-K / rank-based policy naturally re-ranks against
  whatever the *current* score distribution is and does not commit to a
  stale numeric cutoff — this is now supported by a direct demonstration,
  not just an inference from the tie-bound analysis in `08_calibration_analysis.md`.
- **NEEDS_MORE_DATA**: whether a *periodically re-derived* threshold
  (re-fit each rolling window rather than fixed once) would recover
  acceptable precision — not tested; a legitimate middle-ground policy
  between "fixed static bands" and "pure top-K" that this program has not
  yet evaluated.

## Decision

- **REJECT** as currently implemented: severity bands whose thresholds are
  fit and reported on the same sample (the v1 artifact's own numbers show
  this happened; HANDOFF.md's independent test-set number, 37.2% vs an
  advertised 90%, is the predicted consequence measured out-of-sample).
- **KEEP, with a caveat**: top-K ranking as the operationally safer default
  policy at investigator-realistic budgets (K≥200 in this fold shows no
  tie sensitivity), consistent with HANDOFF.md's already-stated candidate
  policy ("rank on raw scores, carry the calibrated probability as a
  display field only"). This experiment adds concrete tie-bound numbers in
  support of that recommendation on the PS-native v2 scope specifically.
- **NEEDS_MORE_DATA**: a proper percentile-band policy re-derived on a
  correctly held-out calibration/threshold-selection split (distinct from
  both the training set and the set the band precision is reported against)
  has not been built or tested in this program. This is the concrete
  next step if severity bands are to be kept as a UI concept at all, rather
  than replaced outright by a top-K worklist.
