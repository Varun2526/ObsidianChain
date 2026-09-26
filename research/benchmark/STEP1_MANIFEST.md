# STEP 1 — Benchmark Manifest

## Status

**NOT FOUND** — no retained repository artifact records an exact benchmark of
blockchain-only nAP approximately `0.935` on the `SIGNAL` world across 12
time-ordered folds.

The closest retained result is `exp15_noisy_world_network_ablation`, whose
`SIGNAL` / blockchain-only (`CORE`) mean nAP is **0.9460** across 12 folds.

## Closest retained benchmark

- **Benchmark script:** `research/autoresearch_2026_09_23/scripts/exp15_noisy_world_network_ablation.py`
- **Recorded result artifact:** `research/autoresearch_2026_09_23/results/exp15_noisy_world_network_ablation.json`
- **Dataset path:** `data/synthetic_world_v2/signal/capture.csv`
- **Quarantined labels used only for evaluation:** `data/synthetic_world_v2/signal/world_truth/labels.csv`
- **Fold / protocol:** `obsidianchain.ml.protocol.rolling_origin_folds()`; 12 rolling, time-ordered development folds: train through `t16`..`t38`, evaluate successive windows `t17-18`..`t39-40`; `t42-49` excluded.
- **Metric:** mean address-level normalized average precision, `nap_CORE`; recorded value `0.9460` (12 folds).
- **Model / artifact:** LightGBM using `research/reproduction/train_ps_model_v2.py`'s `_model()` configuration; this is a `SYNTHETIC_CONTROL` research benchmark, not the production artifact `data/models/ps_native/v5/model.joblib`.
- **Exact command to reproduce the closest retained benchmark:**

  ```bash
  python3 research/autoresearch_2026_09_23/scripts/exp15_noisy_world_network_ablation.py
  ```

## Missing external research artifact

The missing artifact is the original benchmark script/result/configuration
that produced the claimed `SIGNAL` blockchain-only mean nAP of approximately
`0.935` over 12 folds. It is not present in the retained research results or
reproduction scripts; the available `exp15` artifact records `0.9460` instead.
