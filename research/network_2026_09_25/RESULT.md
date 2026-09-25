# exp-net1 result (2026-09-25)

Pre-registration: `PREREGISTRATION.md` (commit 45690d3, before the run).
Output: `results/exp_net1_network_features.json`. SYNTHETIC_CONTROL.

| World | CORE nAP | CORE+E nAP | RELAY_LAUNDERING AUC | Paired verdict (12 folds) |
|---|---|---|---|---|
| SIGNAL | 0.9349 | 0.9541 | 0.814 -> 0.926 | favours CORE+E, +0.0192, p = 0.0007 |
| NULL | 0.9385 | 0.9360 | 0.822 -> 0.812 | indistinguishable, -0.0025, p = 0.50 |

**Conclusion under the pre-registered rule: MECHANISM_DEMONSTRATED.**
CORE is below the 0.95 saturation bar in both worlds, so the comparison is
not void; network features help only where a network signal was planted
and add nothing where none was.

What this shows: the production ingest and feature engine (schema /5,
v5 hyperparameters, protocol B folds) turn network observations into model
signal when such signal exists, and do not manufacture signal when it does
not.

What it does not show: that real Bitcoin traffic carries such a signal. In
the SIGNAL world the behaviour was planted through the illicit label, by
construction.

**Production consequence: none.** The production model stays
blockchain-only until labelled captures with real network observations
exist to repeat this comparison on.
