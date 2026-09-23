# 15 — Final Recommendation (2026-09-23, interim — development research is not yet complete)

> **STATUS (2026-09-23, final): historical.** Absolute metric values in this
> document were computed before the leakage fixes L1-L6
> (`research/autoresearch_2026_09_23/19_leakage_audit.md`) and are
> **INVALID or WITHDRAWN**. Current valid numbers: `docs/results_register.md`.
> Current model: `docs/model_card.md`.


**Updated same day, cycle 2**: a round-2 model search (exp06), a
cross-model ablation confirmation (exp07), a counterparty-history feature
test (exp08), and an alert-policy redesign prototype (exp09) have been
added since this document was first written. Answers below are updated in
place where new evidence changes them; see the addendum at the bottom for
what's new this cycle specifically.

**Status: this is a checkpoint report, not the program's final output.**
Per `14_final_holdout.md`, several Phase-16 preconditions are unmet
(counterparty-history feature untested, C/D-weak finding not confirmed
beyond LightGBM, alert policy not re-implemented, no v2 model artifact
trained). This document answers the research mandate's Phase 17 questions
with what is actually evidenced so far, marking every open item as such —
it does not manufacture a finished answer where the research isn't finished
(RULE 15).

## 1. What is actually the strongest validated pipeline?

**Not yet fully determined — but strongly narrowed, and the model axis is
now close to exhausted.** Features: Group A (transaction shape) is
necessary and dominant, **confirmed on two independent model families**
(exp02 LightGBM, exp07 RandomForest); Group B (address history) is a real,
smaller contributor, especially for cold-start recall — and cold-start is
now known to be 90.9% of the dataset (exp08), not a rare case. Model: one
of {RandomForest, HistGradientBoosting, LightGBM} — statistically tied
(exp01), and **a round-2 search of 6 justified variants (class-weighting,
capacity, a native ranking objective, more ExtraTrees estimators) found
nothing that displaces this tied tier** (exp06). **LightGBM remains the
best-evidenced single choice** if a single model must be named today, on
explainability grounds (exp05/exp10) — unchanged from cycle 1, now on a
firmer footing since the model-search space has been searched more broadly
without finding an alternative. Not yet a final pipeline: the alert policy
is confirmed to need more than the originally-proposed fix (exp09 — a
correctly-derived static threshold still fails across time), and the
counterparty-history feature (exp08) is a promising but unconfirmed
addition to the feature set.

## 2. Is the bottleneck model, features, labels, representation, or alert policy?

**Not model** (exp01: three architecturally different tree ensembles tie).
**Partially features** — but asymmetrically: Group A/B matter, Group C/D
(as currently, thinly engineered) do not measurably, though this is
confirmed on LightGBM only (`13_red_team_review.md`). **Partially
representation** — the address-last-snapshot, boundary-spanner-dropping
dataset construction (`02_data_audit.md`) is untested as a lever; it could
be masking real graph/temporal signal rather than the signal being absent.
**Labels**: no defects found (`03_label_audit.md`) — not the bottleneck.
**Alert policy**: confirmed broken as currently implemented
(`09_alert_policy_analysis.md`), independent of the ranking model's
quality — this is a real, separate bottleneck on top of whatever ranking
model is chosen.

## 3. Which changes produced statistically defensible improvement?

None yet, in the sense of "a change that beat the existing frozen
approach" — this program has not yet trained and compared a full v2
candidate against anything frozen (the v1 artifact can't even run against
v2 data — `01_repo_audit.md` §5). What IS statistically defensible: the
exp01 tree-ensemble tier beating LogReg/ExtraTrees (p_holm<0.03), and
Group A/B's necessity over removing them (p_holm<0.05 in both ablations).

## 4. Which changes failed?

ExtraTrees and both LogisticRegression variants, as sole candidates
(exp01). Groups C and D, as currently engineered, failed to show a
defensible marginal contribution on top of A+B (exp02) — with the explicit
caveat that this may be a statement about their current engineering, not
their ceiling.

## 5. Which results were inconclusive because of MDE/power limitations?

- RandomForest vs HistGradientBoosting vs LightGBM ranking order (exp01) —
  all pairwise diffs 0.009–0.019 nAP, MDE is 0.164.
- Group B's exact marginal-contribution size (exp02) — significant but
  sub-MDE (0.022 nAP).
- The combined C+D contribution via the `A_plus_B` follow-up — nominally
  significant uncorrected (p=0.036) but sub-MDE and not run through the
  full corrected family.
- Isotonic calibration's effect on Brier (exp04) — this fold showed a
  regression, contradicting the frozen v1 artifact's recorded improvement;
  not resolved with the data available.

## 6. What false positives remain?

Not deeply characterized beyond `07_error_analysis.md`'s aggregate counts
(46 at K=100 on the diagnostic fold). No systematic false-positive
breakdown by amount/degree/cluster was completed — flagged as an
incomplete part of Phase 7, not run to conserve program time in this pass.

## 7. What false negatives remain?

**Well-characterized and the clearest actionable finding in the program**:
cold-start addresses (zero or near-zero prior transaction history) —
511 of 565 positives missed at K=100 in the diagnostic fold, with false
negatives having a median of 0 prior transactions vs 1 for true positives
caught at that budget (`07_error_analysis.md`).

## 8. What features genuinely matter?

Group A (input/output counts and amounts, fee, fee_ratio, spreads) —
dominant. Group B (velocity, duration, gap, net flow, history counts) —
real, smaller, and specifically load-bearing for cold-start recall. Groups
C/D as currently engineered — not shown to matter on top of A+B, though not
proven irrelevant in principle (`06_ablation_results.md`).

## 9. Does network-layer data add measurable value?

**Untested — not "no," but "unmeasurable with current data."** The
PS-native development dataset has zero network-layer columns
(`include_network=False` at generation time). Answering this question
requires regenerating the dataset with network telemetry routed through
`network/boundary.py`, not yet done (`01_repo_audit.md`, `04_feature_research.md`).

## 10. Does graph context add measurable value?

**Not shown to, as currently (thinly) engineered — 2 columns, no adjacency
structure.** `06_ablation_results.md`'s `minus_C`/`only_C` results are the
direct evidence. Whether a *richer* graph representation would add value is
explicitly untested (`11_graph_temporal_research.md`) and is the clearest
"needs more engineering before the question can be answered" item in the
program.

## 11. Does temporal modeling add measurable value?

Simple temporal aggregates (Group B) do; a genuine sequence/temporal-graph
model was not built or tested, and the current dataset representation
(one collapsed snapshot per address) would need to change before one
could be — see `11_graph_temporal_research.md`.

## 12. Does calibration help ranking or only probability interpretation?

**Only probability interpretation, and even that is contested on this
fold.** Isotonic calibration measurably *hurts* precision@K reporting
reliability (tie-collapse, `08_calibration_analysis.md`) and did not
improve Brier on the tested fold (contradicting the v1 artifact's own
recorded result — unresolved). Ranking quality (nAP) is essentially
unaffected in aggregate (calibration is rank-preserving up to ties) but the
tie artifacts it introduces directly degrade the *reportability* of
ranking-adjacent metrics like precision@K.

## 13. Is the current alert policy valid?

**No, and the problem is deeper than originally diagnosed.** The frozen v1
severity bands' recorded validation precision matches their own
optimization target exactly (90.0%/75.0%/50.0%) and HANDOFF.md independently
reports the predictable out-of-sample consequence (37.2% on test vs 90%
advertised for CRITICAL). This program's tie-bound measurement
(`08_calibration_analysis.md`) added a second line of evidence. **exp09
added a third, more fundamental one**: even a correctly-derived,
properly-held-out threshold (not just the same-sample-tuned one) collapses
against a genuinely new time window (90%→59%, 75%→48%, 50%→18% on the
tested fold). Fixing the sampling bug alone is not sufficient — any
static-value severity threshold is in tension with this dataset's known
temporal instability.

## 14. Are explanations faithful?

**No — the most confidently answered question in this entire program.**
33.9% direction agreement with true TreeSHAP (worse than chance), 1.4/3
mean feature overlap, independently re-derived and confirmed during red-team
review (`10_explanation_audit.md`, `13_red_team_review.md`).

## 15. What pipeline gaps remain?

See `12_pipeline_gap_analysis.md` in full; top of the priority list:
explanation fidelity, alert policy re-derivation, the v1/v2 schema blocker,
and the confidence-semantics gap (track-record-based vs first-transaction
flags are not distinguished anywhere in the output).

## 16. What should be promoted to production?

**Nothing yet, by this program's own standard** — no v2 model has been
trained, and the standing `MODEL_UNAVAILABLE_FOR_SCHEMA` blocker remains
exactly where HANDOFF.md left it. What this program adds: a specific,
evidenced path to resolving it (LightGBM as the leading candidate, subject
to the open items in `14_final_holdout.md`).

## 17. What should NOT be promoted?

- The current `ps_model.py` explanation method, unmodified (§14 above).
- The current severity-band mechanism, unmodified (§13 above).
- ExtraTrees or LogisticRegression as the PS-native ranking model
  (exp01, decisive).
- Any v1-schema-trained artifact against v2 features (already structurally
  prevented — `01_repo_audit.md` §5).

## 18. What additional data would most improve the research?

In priority order, from this program's own findings:
1. **Counterparty-side history data/features** — directly motivated by the
   cold-start false-negative finding (`07_error_analysis.md`), cheapest to
   engineer within the existing tabular framework.
2. **A no-collapse dataset variant** (retaining boundary-spanning
   addresses, or the full per-transaction row stream rather than
   last-snapshot-only) — would resolve whether the representation itself,
   not the graph/temporal signal, is the current ceiling
   (`02_data_audit.md`, `11_graph_temporal_research.md`).
3. **Network telemetry joined into the PS-native dataset** — the only way
   to answer Phase 4's Group F/G questions at all.
4. **Real mainnet capture** (already planned for October per HANDOFF.md) —
   would let σ and the network-layer timing model be re-fit from real
   data rather than the synthetic surrogate.

## What this program did NOT do (stated explicitly, per RULE 7)

**As of cycle 1 (superseded items struck through, resolved this cycle):**
~~run the full `make test` suite~~ (still not done — remains open); ~~test
Groups C/D's weakness on RF/HGB (only LightGBM)~~ **done — exp07 confirms
on RandomForest**; ~~engineer or test the counterparty-history feature~~
**done — exp08, INCONCLUSIVE**; ~~re-derive severity bands~~ **prototyped —
exp09, revealed a deeper problem than expected**; wire TreeSHAP into
`ps_model.py` (still not done — remains open, method is validated, not
integrated); train a v2 model artifact (still not done); touch the sealed
holdout (correctly still not done).

**New, as of cycle 2**: did not search hyperparameters *within* LightGBM
beyond the single "deeper" variant tested; did not test the
counterparty-history feature with additional richness (diversity, not just
max/mean); did not test whether periodic re-thresholding recovers alert-policy
precision; did not repeat exp09's alert-policy finding across multiple
folds to confirm its magnitude generalizes beyond the single worst-case
fold tested. Every one of these remains a concrete, actionable next step.
