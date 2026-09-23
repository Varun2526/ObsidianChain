# 11 — Graph / Temporal Research (2026-09-23)

Phase 10. Investigates whether the graph representation itself is limiting
performance, and whether more sophisticated graph/temporal architectures
(GNN, temporal graph snapshots, diffusion features) are justified.

## Current representation, restated precisely

- **Graph**: `cluster_size_asof_t` (component size from a transaction-local
  `IncrementalUnionFind`, replayed in strict timestamp order — see
  `01_repo_audit.md` §1) and `unique_counterparties_asof_t`. Two scalar
  columns. No adjacency structure, no message passing, no motif or
  community detection in the PS-native scope.
- **Temporal**: per-address scalar aggregates as-of-t (velocity, gap,
  duration, counts) — no sequence model, no explicit time-series structure
  beyond these summary statistics. The unit of representation is a single
  row (the address's LAST snapshot before its split boundary — see
  `02_data_audit.md`), not a sequence.

## Does the current data/evaluation design support a fair GNN or graph-transformer evaluation?

**No, not without substantial new engineering, and this program does not
force it (RULE 12).** Concretely:

1. **The dataset is one row per address** (last-snapshot-only,
   boundary-spanners dropped — `02_data_audit.md`). A GNN needs a graph
   object (nodes + edges) as input, not a flat feature table; building one
   would mean re-deriving the address/transaction graph from the raw
   Elliptic++ edgelists directly, bypassing `PsTemporalFeatureEngine`
   entirely — a different pipeline, not a drop-in model swap.
2. **The evaluation protocol is fold-based on tabular rows.**
   `ml/protocol.py::evaluate_candidate` expects a `fit_predict(train, eval,
   features, seed) -> scores` callable over a `DataFrame` — compatible with
   any tabular model (which is why LightGBM/RF/ExtraTrees/HGB/LogReg all
   plugged in without protocol changes) but not with a graph-batched
   training loop without an adapter layer that itself would need to respect
   the same forward-only, no-leakage guarantees `PsTemporalFeatureEngine`
   currently enforces by construction.
3. **Address-level split-boundary dropping compounds the graph problem.**
   Any address active across a split boundary is excluded from the tabular
   dataset (`02_data_audit.md`) — a GNN training procedure would need its
   own leakage-safe policy for graph edges that cross the same boundaries,
   which is a genuinely new design problem, not a parameter choice.
4. **No evidence yet that more graph signal would help.** `06_ablation_results.md`
   found the *current, thin* 2-column graph representation contributes
   weakly once amount/history features are present — but explicitly could
   not separate "graph doesn't matter" from "this thin representation
   doesn't capture what matters" (see `04_feature_research.md`). Building a
   GNN to test a hypothesis this program has not yet even weakly confirmed
   (that *richer* graph structure would help) would be exactly the
   "complex model because it sounds impressive" pattern RULE 2/RULE 12
   prohibit.

## What WOULD justify moving to a richer graph/temporal representation

In priority order, cheapest-to-test first:

1. **Richer scalar graph features within the current tabular framework**
   (in/out-degree split, two-hop neighborhood size, counterparty-side
   history — directly motivated by `07_error_analysis.md`'s cold-start
   finding) — still tabular, still fits `protocol.evaluate_candidate`
   without new infrastructure. **Recommended next step, not yet run.**
2. **A no-collapse dataset variant** that doesn't drop split-boundary-
   spanning addresses and doesn't collapse to last-snapshot-only
   (`02_data_audit.md`'s flagged follow-up) — would test whether the
   *representation* choice, not the graph richness, is the bigger lever.
3. **Only after (1) and (2) show a graph-signal ceiling that scalar
   features can't reach**, a graph-native model (GraphSAGE-style
   neighborhood aggregation is the natural first step before a full GNN/
   transformer, and is closer to what `cluster_size_asof_t` is already a
   crude proxy for) would be justified — with its own leakage-safe protocol
   built first, matching the rigor already present in
   `PsTemporalFeatureEngine`.

## Temporal models (LSTM/sequence)

Same conclusion, same reasoning: the current unit of representation is one
snapshot per address, not a sequence. A sequence model needs the full
per-transaction row stream `PsTemporalFeatureEngine.process_records`
already produces (before `build_ps_dataset.py` collapses it to
last-snapshot-only) — that collapse is the actual blocker, shared with the
graph case above. Not attempted; same "no forced complexity without
justification" reasoning applies.

## Decision

- **NOT ATTEMPTED, and documented as such rather than forced**: GNN,
  graph-transformer, LSTM/temporal-sequence models. The data representation
  and evaluation protocol do not currently support a fair evaluation of any
  of them, and no experiment in this program has yet shown a graph/temporal
  signal ceiling that would justify building the new infrastructure required.
- **Concrete, evidence-backed next steps** if graph/temporal investment is
  wanted: richer scalar graph/counterparty features first (cheap, fits
  existing protocol), then the no-collapse dataset variant, and only then
  reconsider a graph-native architecture.
