# exp-net2: does blockchain <-> network coherence improve alert ranking? (pre-registered 2026-09-25)

Written and committed before the method was implemented or run. Follows
exp-net1 (network features in the model: MECHANISM_DEMONSTRATED on
synthetic control, production model unchanged).

## The gap this addresses

Until now the two layers were joined only by exact TXID match. Network
evidence was displayed beside the alert and excluded from the fused score
(`"fused": False`), so the analytical result never asked whether the
network layer AGREES with the on-chain flow. The problem statement asks
for exactly that correlation.

## Method (fixed now)

For each alert cluster C:

1. **On-chain hop pairs.** Pairs of transactions (t1, t2) where t2 spends
   an output of t1 (an output address of t1 is an input address of t2) and
   at least one of them has a member of C as an input or an output. Pairs
   where either transaction has no timed network observation are dropped.
2. **Shared first relay.** A pair *coheres* when the two transactions'
   first-seen peer sets (ties kept, `network/propagation.py`) intersect.
3. **Chance rate from the capture itself.** `q = sum_p f_p^2`, where `f_p`
   is the share of observed transactions whose first-seen set contains
   peer p. This is the probability two independent transactions share a
   first-seen peer, so a relay that first-announces everything raises the
   baseline instead of creating links.
4. **Test.** With m pairs and k coherent, `p = P(X >= k)`, X ~ Binomial(m, q).
5. **Evidence line** `CROSS_LAYER_CONTEXT` / `cross_layer_relay_coherence`
   is PRESENT iff `k >= 2` and `p <= 0.05`, with score
   `min(1, -log10(p) / 4)` (p = 0.05 -> 0.33; p = 1e-4 -> 1.0), and
   otherwise NO_EVIDENCE with the counts. Evidence class RULE. Its
   explanation names the relay, k, m, q and p, and says a relay is a
   vantage point, never the sender.
6. **Fusion (the arm under test).** The line enters the existing noisy-OR
   with weight `W = 0.5`, a policy weight like the other rule lines, not
   fitted.

The ML model and its features are not changed in either arm.

## Worlds and labels

`data/synthetic_world_v2/{signal,null}/capture.csv` as generated on
2026-09-23 (identical except `network_signal`). Each capture is run through
the production pipeline (`pipeline/orchestrator.run_pipeline`, registry
champion) once per arm. An alert is positive when any member address has
`y = 1` in the quarantined `world_truth/labels.csv`, joined after scoring.

## Arms, metric, uncertainty

- BASE: current fusion (no cross-layer line in the score).
- CROSS: BASE plus the cross-layer line at W = 0.5.
- Metric: nAP of the alert ranking (`ml.protocol.normalised_average_precision`
  over alerts ordered by fused score). Secondary, reported only: P@50,
  P@200, and recall of RELAY_LAUNDERING entities in the top 200.
- Uncertainty: paired bootstrap over alerts, 2000 resamples, seed 20260925;
  95% percentile interval of delta nAP (CROSS - BASE).

## Decision rule (fixed now)

1. **Void** if either world's run fails or has fewer than 50 positive
   alerts.
2. **Promote** the line into the production fused score iff the SIGNAL
   interval lies entirely above 0 **and** the NULL interval does not lie
   entirely above 0.
3. **Method manufactures signal** if the NULL interval lies entirely above
   0: the line stays display-only and is flagged.
4. Otherwise **not demonstrated**: the line is computed and shown as
   evidence, but not fused.

## What a result can and cannot mean

A promotion rests on a synthetic control whose network signal was planted
through the label: it shows the method extracts coherent cross-layer
structure where it exists and stays neutral where it does not. It does not
show that real Bitcoin traffic carries such structure. Whatever the
outcome, the cross-layer links and the evidence line are part of the
analytical output, and the documentation says which of the above happened.
