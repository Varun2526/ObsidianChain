# Improvement cycle 2 — PAUSED (2026-09-24)

Paused by the owner to make the product demo-ready. No model was trained,
selected or changed in this cycle. ADR 0004 governs any resumption.

State at pause:
- `scripts/common.py` — shared data layer (dev cache t1-41, full cache t1-49).
- `results/events_full_t1_49.parquet` — full-period event features, built by
  the unchanged production engine (gitignored; rebuild with `common.events_full()`).
- One diagnostic fact recorded, from raw labels (not from any model):
  illicit-labelled transactions per timestep fall from 239 (t42) to 24 (t43),
  5 (t45) and 2 (t46), recovering to 56 by t49 — a regime break coinciding
  with v5's holdout collapses. Not yet tied to wallet-level causes; no fix
  has been proposed or tested.
