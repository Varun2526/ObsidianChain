# exp-net1: do network features add signal? (pre-registered 2026-09-25)

Written and committed before the script was run. Phase 7 of
`docs/archive/plans/2026-09-25-network-layer.md`.

## Question

When network observations carry information the chain does not, does the
production feature engine turn it into model signal; and when they carry
none, does adding them stay neutral?

## Why this can only be a synthetic-control experiment

No labelled dataset in this repository has real network fields. Elliptic++
has none, and its overlay draws each origin at random (no signal). So the
only worlds with a known answer are the two generated v2 worlds, identical
except `network_signal`:

- **SIGNAL**: the `RELAY_LAUNDERING` behaviour (a positive class) broadcasts
  from well-connected nodes (`world/noisy.py:97, 177-178`). The signal is
  planted **through the label**, so a gain here demonstrates that the
  pipeline can extract a network signal. It says nothing about Bitcoin.
- **NULL**: the same world with no network signal. A gain here would mean
  the network features manufacture signal: a false positive of the method.

## Data (unchanged, verified by manifest)

`data/synthetic_world_v2/{signal,null}/capture.csv` as generated on
2026-09-23. The script checks each world manifest and does not regenerate
the worlds. Labels come from the quarantined `world_truth/labels.csv`,
joined last.

## Features and model

- **CORE**: `CORE_PS_FEATURE_COLUMNS` (schema `ps_native_features/5`, the
  31 production features), filtered by `ml.diagnostics.healthy_features`.
- **CORE+E**: CORE plus `GROUP_E_NETWORK` (6 features:
  `network_observation_count, observer_diversity, peer_count, asn_count,
  dominant_peer_share, arrival_spread_seconds`), computed by the production
  engine with `include_network=True`, through the production ingest path
  (which since 2026-09-25 keeps observer identity).
- **Model**: LightGBM with the `ps_native_v5` hyperparameters from
  `data/models/ps_native/v5/manifest.json`, fixed seed.

## Evaluation (protocol B, as for v5)

The 12 rolling folds of `ml.protocol` over world timesteps 1-41 (tuning
1-6, confirmation 7-12). Training uses addresses first seen in
(te-16, te], snapshotted at their last event at or before te. Evaluation
uses addresses first seen in the fold window, snapshotted at the window
end. Metric: nAP; paired verdict across folds (`protocol.paired_verdict`).
Diagnostic only: ROC-AUC of `RELAY_LAUNDERING` addresses against negatives.

## Decision rule (fixed now)

1. **Void** if CORE mean nAP is ≥ 0.95 in either world (saturated; nothing
   to add).
2. **Mechanism demonstrated** if and only if the paired verdict favours
   CORE+E in SIGNAL **and** does not favour CORE+E in NULL.
3. **Method manufactures signal** if CORE+E is favoured in NULL. That would
   bar network features from any model until explained.
4. Otherwise **not demonstrated**.

**Production consequence, whatever the outcome:** the production model is
not changed. A mechanism result on a world whose signal was planted through
the label cannot justify scoring real traffic with network features. That
requires labelled captures with real network observations, which do not
exist here. The result is recorded either way.
