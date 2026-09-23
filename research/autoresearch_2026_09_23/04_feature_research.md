# 04 — Feature and Representation Research (synthesis, 2026-09-23)

This file synthesizes the feature-group investigation spread across
`02_data_audit.md`, `06_ablation_results.md` and `07_error_analysis.md`
into the Groups-A–G structure the research mandate specified. It does not
duplicate their detail — it maps what was actually found onto that
structure and states, per group, what is evidenced vs. untested.

## GROUP A — Amount behavior → maps to `A_transaction` (10 cols)

**Evidenced, dominant.** `06_ablation_results.md`: removing this group
collapses nAP from 0.591 to 0.245; alone it recovers 0.472 (80% of full).
Covers total/mean input/output amount, fee, fee_ratio, and the v2-real
`input_spread`/`output_spread` (replacing the v1 fabricated std/max/entropy
columns — see `01_repo_audit.md` §3). Amount concentration, entropy,
unusually-large/small-transfer flags beyond spread were **not** engineered
as separate columns and were not tested as additions — a candidate
follow-up, not run here.

## GROUP B — Temporal behavior → maps to `B_address_history` (10 cols)

**Evidenced, small-but-real contributor.** `06_ablation_results.md`: removing
it costs 0.022 nAP (p_holm=0.048, sub-MDE effect but statistically
significant). `07_error_analysis.md` shows *where* it matters: cold-start
addresses (no prior history) are the dominant false-negative population,
and B is exactly the group that goes to its degenerate default for them.
Covers tx velocity, active duration, gap-since-last-tx, net flow, mean fee
ratio. Change-point features, acceleration, burstiness beyond velocity were
**not** engineered as separate columns — untested.

## GROUP C — Graph behavior → maps to `C_graph` (2 cols only)

**Evidenced weak marginal contributor, with an explicit representation
caveat.** `06_ablation_results.md`: `minus_C` is not statistically
significant after correction (p_holm=0.126); `only_C` alone is weak (0.129)
but far from zero. Current engineering is thin: only
`unique_counterparties_asof_t` and `cluster_size_asof_t` (from a
transaction-local `IncrementalUnionFind`, not the project's global
569,513-cluster Phase-1 structure — see `01_repo_audit.md` §1). In-degree,
out-degree, degree growth, fan-in/fan-out counts, two-hop neighborhood
stats, motif features, community/clustering-coefficient information are
**not implemented** in the PS-native scope at all (some exist in the
separate Phase 6 M0–M4 feature set, out of this scope per
`00_research_protocol.md`). **This experiment cannot separate "graph
signal doesn't matter" from "this particular 2-column graph representation
doesn't capture what matters"** — stated as an open question in
`06_ablation_results.md` and not resolved here.

## GROUP D — Address lifecycle → overlaps Group B

Address age (`active_duration_seconds`), first/last seen (implicit in the
last-snapshot-per-address dataset construction — see `02_data_audit.md`),
transaction count (`n_txs_asof_t`), counterparty diversity
(`unique_counterparties_asof_t`, technically Group C) are already covered
under B/C above.

**Update 2026-09-23 (exp08_counterparty_history_feature):** counterparty-side
history (does *this* transaction touch a counterparty with its own track
record?) has now been engineered and tested, as an experimental addition
(`counterparty_max_n_txs_asof_t`, `counterparty_mean_n_txs_asof_t` — not in
production `pipeline/features_ps.py`). Result: **directionally positive,
statistically INCONCLUSIVE.** Full-feature-set nAP rose from 0.5914 to
0.6155 (+0.024) with the two new columns added, consistent in direction on
both the full 12-fold protocol and a cold-start-only subset check
(0.2387→0.2464), but the improvement is below this design's MDE (0.164) and
not statistically significant at n=2 candidates (p=0.199). **Not confirmed,
not refuted** — the next step would be a richer counterparty feature set
(diversity of counterparty history, not just max/mean) or accepting this as
a small, real-but-unconfirmable-at-this-power effect. See
`07_error_analysis.md`'s correction: cold-start is 90.9% of this dataset's
rows, not a rare subpopulation — so a counterparty-history feature's
ceiling is inherently the dominant case, not an edge case, which raises the
stakes of resolving this INCONCLUSIVE result properly rather than dropping
it.

## GROUP E — Behavioral patterns → maps to `D_patterns` (2 cols)

**Evidenced weakest marginal contributor.** `06_ablation_results.md`:
`minus_D` changes nothing measurable (diff 0.0018, p=0.653); `only_D` alone
is barely above random (0.0151). Only `is_peeling_candidate` and
`is_mixing_candidate` exist, both binary and both already fixed from
mathematically-broken v1 definitions (`01_repo_audit.md` §3). Consolidation,
rapid forwarding, repeated transfer chains, unusual structural transitions
are **not implemented** as features in this scope.

## GROUP F — Network layer

**Cannot be tested against this dataset — confirmed structural gap, not a
result.** `01_repo_audit.md` §3/§9, `02_data_audit.md`: current
`train.parquet`/`validation.parquet` were built with `include_network=False`.
IP/port/ASN/country/observer-diversity/peer-diversity features exist in
`pipeline/features_ps.py`'s `GROUP_E_NETWORK` definition and in the
project's separate `network/` synthetic-world infrastructure, but neither
is joined into the PS-native development dataset used by this program.
Testing this group requires either (a) regenerating the PS-native dataset
with `include_network=True` against real or synthetic network telemetry
routed through `network/boundary.py` (the sanctioned load path — HANDOFF
invariant 1), or (b) working in the separate Phase 6 scope, which is not
comparable to PS-native results (`ScopeMismatchError`). Not attempted in
this program — a genuine data-availability gap, tracked in
`12_pipeline_gap_analysis.md`.

## GROUP G — Cross-layer correlation (blockchain-only vs +network vs +graph vs +graph+network)

**Cannot be tested, for the same reason as Group F.** The
blockchain-only vs blockchain+graph comparison *is* partially answered by
this program (`06_ablation_results.md`'s `minus_C`/`only_C` results ARE the
blockchain-vs-blockchain+graph ablation, within the PS-native scope's thin
current graph representation). The network-inclusive comparisons
(blockchain+network, blockchain+graph+network) require the same dataset
regeneration as Group F and were not run.

## A noted tension: global TreeSHAP importance vs. ablation marginal contribution (from exp10's v2 dev candidate)

Training the v2 development candidate (`v2_dev_candidate/manifest.json`)
and computing global mean-|TreeSHAP| importance across a 5,000-row sample
produced a ranking where **`cluster_size_asof_t` (Group C) is the #2 most
important feature overall** (mean |SHAP| 0.474, behind only `output_count`
at 1.705) — despite Groups C's ablation-measured marginal contribution
being statistically indistinguishable from zero on top of A+B
(`06_ablation_results.md`, confirmed on two model families in
`07_ablation_rf_confirmation`).

**This is not a contradiction, and is reported as a real, informative
tension rather than smoothed over**: global importance measures how much a
feature is *used* when present; ablation measures what happens when it is
*removed*. A feature can be heavily used by the model while still being
redundant with other present features (e.g. `cluster_size_asof_t` may
substitute for information `n_txs_asof_t`/`unique_counterparties_asof_t`
already carry, so the model leans on whichever is present, but removing it
alone costs nothing because the others compensate). This is a legitimate,
well-known divergence between the two measurement types, not a bug in
either measurement — but it means **"Group C doesn't matter" is a claim
about marginal/removal value, not about whether the model uses it**, and
that distinction should be stated explicitly in any external-facing summary
of this program's findings.

## Summary table

| Group | Tested? | Verdict |
|---|---|---|
| A (amount/transaction) | Yes | **Dominant, necessary** |
| B (temporal/history) | Yes | Small, real, necessary for cold-start recall |
| C (graph, current 2-col engineering) | Yes | Weak marginal, representation-limited |
| D (lifecycle) | Overlaps B/C | See above |
| E (patterns, current 2-col engineering) | Yes | Negligible marginal |
| F (network) | **No — data unavailable** | Structural gap |
| G (cross-layer) | Partially (blockchain-vs-+graph only) | Network legs untested |
