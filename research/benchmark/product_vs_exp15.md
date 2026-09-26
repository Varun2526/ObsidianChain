# Product vs exp15

1. **exp15 research nAP:** `0.9460`
2. **Current product pipeline nAP:** `-0.0142`
3. **Difference:** `-0.9602` (`-0.0142311789 - 0.9460`)
4. **Product model artifact:** `data/models/ps_native/v5/model.joblib` — `ps_native_v5`, SHA-256 `974d37f22e2f1e7df4e03cb5a13fdb35bf487e43f4d5db3ea3bb7191ebd3e607`.
5. **Product feature pipeline:** `src/obsidianchain/pipeline/features_ps.py`: `extract_ps_features(frame)` followed by `last_snapshot_per_address(...)`; the artifact requires the `ps_native_features/5` 31-feature CORE+G contract in `src/obsidianchain/contracts/features.py`.
6. **Same model/protocol:** No. Both measurements use the 12 `rolling_origin_folds()` time boundaries and the address-level `normalised_average_precision` metric. exp15 uses the last-address-snapshot unit and refits its research v2 LightGBM CORE model on each fold. This run used the frozen product artifact for prediction only on the same exp15 fold evaluation sets.
7. **Concrete reason for the difference:** exp15 is a `SYNTHETIC_CONTROL` benchmark whose `0.9460` is produced by a fold-specific v2 model trained on each SIGNAL training window with the 24-feature `/3` CORE set. The product artifact is a fixed `/5` 31-feature CORE+G LightGBM, trained on the separate production source under `protocol_B_new_addresses_at_window_end/1` (new addresses and window-end snapshots; final training set `t26-41`). It is neither the exp15 model nor retrained on SIGNAL. The SIGNAL capture passed the current product feature contract, so no code change was needed.
