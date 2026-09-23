# 08 — Calibration Analysis (exp04_calibration_alert_policy, completed 2026-09-23)

Full results: `results/exp04_calibration_alert_policy.json`. Same
diagnostic fold as exp03 (`t<=38 -> t39-40`) for direct comparability.
Calibrator fit on a held-out 15% split of the fold's own training data
(never the eval window), matching correct calibration practice.

## Reproducing the known isotonic-collapse finding — now measured on PS-native v2 specifically

HANDOFF.md already stated (Phase 6 / M0–M4 scope, ps_native_v1's own
`calibration.json`) that isotonic calibration collapses resolution. This
experiment measures the **same phenomenon on the PS-native v2 scope, on
this program's own freshly-trained LightGBM**, not by re-reading the old
artifact:

- Raw score: **2,185 distinct values** over 11,764 eval rows.
- Isotonic-calibrated score: **50 distinct values.**

## Ranking quality: small aggregate change, large tie-bound consequence

- nAP: raw 0.2835 → calibrated 0.2707 (small drop; isotonic is
  rank-preserving except at ties, so this gap is entirely a tie artefact,
  consistent with `protocol.py`'s own explanation).
- **Precision@K tie-bound width is where the real damage shows**, using
  the protocol's own `precision_at_k_bounds` (worst/best over all valid tie
  orderings, not a single arbitrary value):

| K | raw score bound | calibrated score bound | calibrated width |
|---|---|---|---|
| 50 | [0.320, 0.340] (width 0.020) | **[0.160, 0.420] (width 0.260)** | 13× wider |
| 100 | [0.450, 0.450] (width 0.000) | [0.420, 0.460] (width 0.040) | — |
| 200 | [0.565, 0.565] (width 0.000) | [0.565, 0.565] (width 0.000) | — |
| 500 | [0.446, 0.446] (width 0.000) | [0.440, 0.452] (width 0.012) | — |

At K=50 specifically, **a reported precision@50 under the calibrated score
could honestly be anywhere from 16% to 42% depending on arbitrary row
order** — the exact failure mode HANDOFF.md and `protocol.py` describe in
the abstract, now quantified concretely: a 26-point precision swing from
tie-breaking alone, at a budget size (K=50) squarely inside what an
investigator would plausibly use.

## Calibration quality (Brier): an unresolved anomaly, flagged rather than smoothed over

This experiment measured isotonic calibration making Brier **slightly
worse**, not better, on this fold: raw 0.03922 → calibrated 0.04019. This
**contradicts** the frozen v1 model's own recorded result
(`data/models/ps_native/v1/calibration.json`: raw Brier 0.04186 →
calibrated 0.03612, an improvement) and is not dismissed as noise without
comment (RULE 8):

- **Plausible explanation 1 — calibration-set size/positives.** The
  calibrator here was fit on a 15% held-out slice of one fold's *training*
  data (a few thousand rows at ~5% prevalence → likely only ~100–200
  positive examples), much smaller than whatever produced the v1 artifact's
  calibration curve. Isotonic regression is a step function fit
  nonparametrically; with few positives its steps are coarse and can
  overfit local noise.
- **Plausible explanation 2 — this is deliberately the worst-performing
  fold** (matches exp01's LightGBM minimum, nAP 0.3033 single-fold vs
  0.5914 family mean). A calibrator fit on in-fold training data may not
  transfer well to an unusually hard evaluation window; this would be
  consistent with the project's own headline finding that window choice
  dominates other sources of variation by ~35×.
- **Not tested, and this experiment does NOT claim to have resolved it:**
  whether the same isotonic fit on a typical (non-worst) fold reproduces
  the v1 artifact's improvement. This is the natural next check before
  treating either result (v1's improvement or this fold's regression) as
  representative.

## What Phase 8 concludes

- **Confirmed, with a new concrete measurement:** isotonic calibration's
  resolution collapse is real on PS-native v2 and has a quantifiable,
  material effect on precision@K reporting at operationally relevant
  budgets (K=50).
- **INCONCLUSIVE:** whether isotonic calibration reliably improves Brier on
  this scope — this experiment's single-fold result disagrees with the
  frozen v1 artifact's own recorded result, and neither is run across
  enough folds here to adjudicate.
- **Actionable, consistent with HANDOFF.md's already-proposed policy:**
  separate ranking from calibration explicitly — rank on the raw score,
  carry the calibrated probability as a *display-only* field, and do not
  let the calibrated score's tie structure determine which addresses cross
  an alert-budget cutoff. This experiment's tie-bound numbers are the first
  concrete evidence *for* that specific policy on the PS-native v2 scope,
  not just the Phase 6 scope it was originally proposed for.
