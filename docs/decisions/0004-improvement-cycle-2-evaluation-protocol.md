# ADR 0004 — Improvement cycle 2: no clean holdout exists; how evidence is judged

Date: 2026-09-24. Status: accepted **before** any cycle-2 diagnostic read of
t42-49 data and before any cycle-2 model was trained.

## The situation

ADR 0003 opened the protocol-B holdout (t42-49) exactly once, for
`ps_native_v5` and its fallback. The result is published: nAP 0.548,
P@100 1.00, ECE 0.010, with per-step collapses at t43 (0.135), t45 (0.017)
and t47 (0.324).

ADR 0003 also says t42-49 may become ordinary development data "only once a
newer period exists to serve as that cycle's holdout". Elliptic++ ends at
t49. **No newer period exists.** So this cycle has no clean holdout, and
this record says what follows from that instead of pretending otherwise.

## Decisions

1. **t42-49 status becomes SEEN.** From this cycle on, any number measured
   on t42-49 for any model other than v5 and its fallback is labelled
   `REUSED_HOLDOUT` everywhere: artifacts, API, UI, documents. It is an
   optimistic estimate, not a generalisation estimate.

2. **Diagnosis is allowed; selection on it is not.** The cycle may read
   t42-49 labels to explain why v5 failed (error analysis of an already
   published result). A proposed fix is admissible only if the mechanism it
   addresses is **also demonstrated in development data (t1-41)**, and it is
   selected using development/confirmation folds only.

3. **Selection and adoption use protocol B folds 1-12 only** (tuning 1-6,
   confirmation 7-12, as exp22). A candidate replaces the champion only if,
   on the confirmation folds:
   - mean nAP is not lower than the champion's (tolerance 0.005), **and**
   - worst-fold nAP is not lower than the champion's, **and**
   - it improves the metric it was designed for (declared in its experiment
     before running) with a paired comparison across folds, or on pooled
     confirmation predictions with an address-level bootstrap whose 95% CI
     excludes zero.

4. **The reused holdout is a veto, never a reason to promote.** After a
   candidate passes rule 3, it is scored once on t42-49. If its
   `REUSED_HOLDOUT` nAP is more than 0.02 below v5's published 0.548, or its
   ECE is above 0.03, it is rejected. A better reused-holdout number
   earns nothing.

5. **Rejected changes stay recorded.** Every experiment in this cycle writes
   its design before running and its result after, into
   `research/cycle2_2026_09_24/`, including those rejected.

6. **What cannot be claimed.** This cycle cannot produce an unbiased
   estimate of any new model's performance after t41. The honest
   generalisation evidence for a cycle-2 model is its confirmation-fold
   record plus live delayed-label monitoring. The documents and the demo
   say so.

## Why not re-split

Moving the holdout earlier (for example t36-41) would require those
timesteps to have been unseen by every choice made so far. They are inside
the development period of cycles 1 and 2 and were used for confirmation, so
they are not a holdout either. A re-split would launder, not restore, an
unbiased estimate.
