# exp-net2 result (2026-09-25): cross-layer coherence is not a risk signal

Pre-registration: `PREREGISTRATION_exp_net2.md` (commit 0776854, before the
method was implemented or run). Output: `results/exp_net2_cross_layer_fusion.json`.
SYNTHETIC_CONTROL.

| World | Alerts (positive) | Line present (positive) | nAP BASE -> CROSS | delta nAP, 95% CI | P@200 | RELAY_LAUNDERING recall @200 |
|---|---|---|---|---|---|---|
| SIGNAL | 20,082 (4,785) | 1,360 (274) | 0.0713 -> 0.0619 | -0.0094 [-0.0122, -0.0066] | 0.14 -> 0.19 | 1.9% -> 5.0% |
| NULL | 20,462 (4,882) | 1,161 (222) | 0.0766 -> 0.0678 | -0.0087 [-0.0113, -0.0062] | 0.125 -> 0.15 | 2.4% -> 2.0% |

**Conclusion under the pre-registered rule: NOT_DEMONSTRATED.** The SIGNAL
interval is below zero, not above. The cross-layer line is therefore **not
fused**: `pipeline/alerts.py` keeps `FUSE_CROSS_LAYER = False`, and a test
pins it.

## What the numbers say

The line does find what it was designed to find. In the SIGNAL world it
lifts relay-laundering recall in the top 200 from 1.9% to 5.0%, and it
does nothing of the kind in NULL. But it fires on 1,360 alerts, and only
20% of them are positive, which is below the 24% base rate. Relay
coherence marks **one operator broadcasting its own chain**. The benign
well-connected services in these worlds (exchange batching, hot-wallet
shuffles) do that too. Added to the score, it promotes those services as
much as the launderers, and ranking quality falls in both worlds.

So coherence between the two layers is evidence about **who controls a
flow**, not about **whether the flow is illicit**. The product uses it
that way.

## What ships

- The `CROSS_LAYER_CONTEXT` line on every alert: k of m hops first
  announced by the same relay, the capture's chance rate, the p-value, and
  the relay. Shown, labelled as not part of the risk score.
- `SAME_FIRST_RELAY` edges (tx -> tx along a coherent hop) in the run graph.
- **Relay flows** (`cross_layer.relay_flows`), added after this experiment
  as a lead-only use. For relay r: of the hops whose parent r first
  announced, how often r also first announced the child, against r's
  overall share (binomial, Bonferroni across relays). A significant relay
  ties the clusters of its coherent hops into one flow, and each alert
  lists the other alerts in its flow. `CROSS_LAYER_LINK` edges in the
  graph. Never merged, never scored.
  - This test was changed after seeing one output. The first version
    compared k_r with all hops in the capture, and exchange address reuse
    diluted it. It now conditions on the parent. Because that change was
    made while looking at data, it carries its own null calibration test:
    random relays over chains produce a flow in 4.5% of captures, within
    the 5% family-wise bound (`tests/test_cross_layer.py`).
  - The per-alert line is calibrated too: 1.25% of null chains flagged at
    the 5% level.

## exp-net2b: are flows worth anything as ownership links? Uninformative here

`exp_net2b_flow_precision.py` measures link precision against the world's
true owners. It finds 619/619 (SIGNAL) and 603/603 (NULL) linked cluster
pairs sharing an owner, against 0.1% for random pairs. But every
hop-adjacent cluster pair in these worlds shares an owner too (2,371/2,371),
because the v2 generator never moves value between entities. The worlds
therefore cannot tell whether the network layer adds anything to linking.
Measuring that needs a capture where value crosses owners.

## Production consequence

The fused risk score, ranking and severities are unchanged, as is the ML
model. What changed is that each alert now carries the blockchain <->
network correlation as a tested, explained result. It links alerts the
network layer ties together, which gives the investigator a lead. It does
not assert risk.
