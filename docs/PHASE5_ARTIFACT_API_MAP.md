# Phase 5 — Artifact-to-API mapping

**Read-only inspection of the repository at `d6f62d4`. No code written.**

Hard constraint honoured throughout: *the API reads precomputed artifacts
only.* Every "can this be served" answer below is judged against that. Filtering
and aggregating rows already on disk is reading; re-running `load_cospend_graph`,
union-find, `separation_evidence`, or any statistic is **not**, and is marked as
a violation wherever an endpoint would need it.

No field below is invented. Where the requested response needs something that
does not exist on disk, it is marked **GAP** and left unfilled.

---

## 0. Four structural gaps that decide most of this document

These are not per-endpoint problems. They are missing artifacts that several
endpoints each depend on, and they should be read before the endpoint table.

### GAP-1 — the address ↔ code map is not persisted

`evidence_funnel.parquet.node_a/node_b`, `phase33_decisions.component_a/
component_b` and `contaminated_clusters.cluster_id` are **int address codes**,
not addresses. Sample row:

```
provenance_type,cluster_id,representative_address,size,...
PRODUCTION,419015,1FXb9kuLLSG48K6uBGZobxspR5STWwM5qP,14885,...
```

`419015` is a `pd.factorize` code assigned inside `load_cospend_graph()`
(`io/elliptic.py:227-294`) over `wallets_classes.csv ++ AddrTx_edgelist.csv`.
It is deterministic given identical inputs but **exists only in memory** —
`grep` for a writer of `universe_codes`/`addresses` returns nothing.

Consequence: the API cannot turn any evidence row into an address, and cannot
look an address up. Rebuilding the map costs ~2 s and ~444 MB peak — acceptable
**once at startup**, forbidden per request. `contaminated_clusters.csv` is the
only bridge and it covers **68 of 569,513** clusters.

### GAP-2 — cluster membership is not persisted

Nothing writes `ClusterRun.roots`. There is no `address → cluster_id` artifact
anywhere. 569,513 clusters exist in the terminal report; 68 have a durable row.

Consequence: `/api/entity/{id}` and `/api/graph` have no membership source.

### GAP-3 — there is no risk model, no alerts, no severity

`grep -rniE "risk|alert|threat|suspicious"` over `src/` returns **one hit**, a
docstring in `cluster/change.py:374`. Confirmed in the Phase 4 audit §4.5:
`RiskScorer` does not exist. Nothing scores, ranks, or triages.

Consequence: `/api/alerts` is a 100% gap. Every field it asks for — risk,
severity, reason — would have to be invented.

### GAP-4 — the chain-only vs fused comparison is never written

`run --mode fused` and `fusion-summary` make **zero** write calls (verified by
scanning both command bodies). The comparison exists only as terminal output.

Consequence: `/api/compare` has no backing artifact.

---

## 1. Artifact inventory (production namespace)

| Artifact | Rows | Key | `provenance_type` | `is_measurement` |
|---|---|---|---|---|
| `evolution.csv` | 98 | (heuristics, timestep) | PRODUCTION | — |
| `contaminated_clusters.csv` | 68 | cluster_id *(code)* | PRODUCTION | — |
| `entity_labels.csv` | 39,209 | address | PRODUCTION | **true** |
| `entity_resolution_multi-input.csv` | 117 | entity_norm | PRODUCTION | — |
| `evidence_funnel.parquet` | 253,429 | edge_index | PRODUCTION | **false** |
| `network/arrival_vectors.parquet` | 202,804 | txid | PRODUCTION | **false** |
| `network/observations.parquet` | 1,589,863 | (txid, observer_id) | *(unstamped — see below)* | — |
| `network/manifest.json` | — | — | — | — |
| `phase33.csv` | 5 | regime | SYNTHETIC_CONTROL | false |
| `phase33_decisions.csv` | 1,267,145 | (regime, decision_id) | SYNTHETIC_CONTROL | false |
| `world_diagnostics.csv` | 5 | regime | SYNTHETIC_CONTROL | false |
| `reach_stress/…/phase33_decisions.csv` | 28,500 | (regime, decision_id) | SYNTHETIC_CONTROL | false |
| `demo/output/scenarios.json` | 5 scenarios | key | DEMO | false |

`observations.parquet` carries no provenance columns **by design** — stamping it
would change `c405493d…` and break HANDOFF invariant 8. Its provenance is
`network/manifest.json`. The API must special-case this rather than treat the
absence as "unknown".

`entity_labels.csv` is the only artifact with `is_measurement: true` — real
Elliptic++ address labels, no inference. Everything network-derived is `false`.

---

## 2. Per-endpoint mapping

### `GET /api/overview`

**1. Sources** `evolution.csv` (last row per heuristics — PRODUCTION),
`network/manifest.json`, `evidence_funnel.parquet` + sidecar,
`contaminated_clusters.csv`, and separately `phase33.csv` (SYNTHETIC_CONTROL).

**2. Available**

```
evolution.csv  t=49, multi-input:
  n_clusters 569,513 · largest 14,885 · coverage 34.5965
  addresses_seen 284,709 · contaminated 68 · cospend_merges 253,429
manifest.json:
  transaction_count 202,804 · record_count 1,589,863 · observer_count 8
evidence_funnel.parquet:
  253,429 rows; min_pooled ≥25 on 40 of them
phase33.csv (SYNTHETIC_CONTROL, per regime):
  abstention_rate, evaluated, abstained, blocked, contested
```

**3. Without recomputation** Yes. Counting funnel rows by `min_pooled` is a
scan of a precomputed column, not a re-run of the statistic. Precompute once at
startup regardless.

**4. Unavailable** *Alert count* (GAP-3). *Address universe = 822,942* — not in
any processed artifact; `addresses_seen` is 284,709 clustered addresses, a
different quantity, and substituting one for the other would be wrong.

**5. Joins** None; four independent reads.

**6. Size** < 8 KB.

**7. Risk** Production and controlled-world figures must be **two labelled
blocks**, not one merged set of counts. Regime A happens to show
`clusters=569,513` identical to production only because `blocked=0`; presenting
that row as "the overview" would silently relabel SYNTHETIC_CONTROL as fact.

---

### `GET /api/alerts`

**1. Sources** **None.**

**2. Available** Nothing that is an alert. Closest proxies, none of which is a
risk score: `arrival_vectors.evidence` + `no_evidence_reason` (per transaction),
`evidence_funnel.min_pooled`/`dof` (per union), `phase33.csv.contested`
(**0 in every production regime**).

**3. Without recomputation** N/A — nothing to read.

**4. Unavailable** `risk`, `severity`, `reason`, ranking. All of it (GAP-3).

**5. Joins** N/A.

**6. Size** N/A.

**7. Risk** **The highest-consequence endpoint in the set.** An alert is the
artifact an analyst acts on; inventing a severity here is how a synthetic
demonstration becomes an operational claim about a person. Audit §6.4 already
recorded the two properties a scorer must have from its first commit: `score()`
returns `None` rather than a default "low", and `confidence` is `str | None`,
never a default "medium".

> **Recommendation: do not ship `/api/alerts` in Phase 5.** If a queue is
> needed, ship `/api/decisions` — ranked by *evidence strength* (`min_pooled`,
> `dof`), explicitly not risk, and named so the UI cannot render it as triage.

---

### `GET /api/entity/{id}`

**1. Sources** `contaminated_clusters.csv` (68 rows), `entity_labels.csv`
(39,209 address→label rows), `entity_resolution_multi-input.csv` (117 labelled
entities).

**2. Available** For the 68 contaminated clusters: `cluster_id`,
`representative_address`, `size`, `n_illicit`, `n_licit`, `n_unknown`,
`n_labelled`, `label_coverage`, `purity`, `entropy`, `majority_class`.
For a known address: `entity_norm`, `entity_display`, `category`, `source`,
`in_elliptic`.

**3. Without recomputation** **No.** Membership requires GAP-2; resolving codes
requires GAP-1.

**4. Unavailable** *members* (GAP-2) · *transactions* (raw `AddrTx` only, not in
`processed/`) · *temporal behaviour per entity* (`evolution.csv` is global per
timestep, never per cluster) · *evidence timeline* (decisions key on root codes
that change as union-find proceeds — see the risk note) · *blocked merges*
(production = 0) · *contested relationships* (production = 0).

**5. Joins** `contaminated_clusters.cluster_id` → *(missing code map)* →
`entity_labels.address`.

**6. Size** Bounded by cluster size; the largest is 14,885 members and must
never be returned whole.

**7. Risk** `component_a`/`component_b` in the decisions file are union-find
**roots at decision time**, not stable identities — a root changes when its
component is absorbed. Treating them as durable entity ids would produce an
"evidence timeline" that silently refers to different components over time.

---

### `GET /api/transaction/{id}` — the most complete endpoint

**1. Sources** `network/arrival_vectors.parquet` (202,804 rows, keyed `txid`),
`network/observations.parquet` (1,589,863 rows, keyed `txid` + `observer_id`).

**2. Available**

```
arrival_vectors:  txid, offset_ms__obs-00..07, rank__obs-00..07,
                  first_observer, spread_ms, n_observed, is_flat,
                  evidence (USABLE | NO_EVIDENCE), no_evidence_reason,
                  provenance_type, synthetic_network
observations:     txid, observer_id, peer_ip, peer_port, peer_asn, timestamp_ms
```

`evidence` + `no_evidence_reason` is exactly the requested "evidence quality",
already computed and already on disk.

**4. Unavailable** *inputs / outputs* — `raw/AddrTx_edgelist.csv` (477,117 rows)
and `raw/TxAddr_edgelist.csv` (837,124) only, not in `processed/`.
*timestamp / block height* — only in `raw/txs_features.csv` (663 MB) and
`raw/wallets_features.csv` (578 MB); `change.py:153` streams the latter in
chunks precisely because reading it whole peaks over 1.2 GB.
*connected entities* — GAP-1 + GAP-2.

**3. Without recomputation** Yes for propagation and evidence quality, given a
txid index built at startup. No for inputs/outputs/timestamp.

**5. Joins** `arrival_vectors ⋈ observations` on `txid` (1 row ⋈ ~8 rows).

**6. Size** ~2–4 KB.

**7. Risk** `peer_ip` **must not be rendered as the origin.** HANDOFF invariant
7: for ordinary transactions the generator deliberately makes the relaying peer
*not* identify the origin; only known-broadcaster transactions relay directly,
and those are exactly the ones classified `NO_EVIDENCE`. A UI column headed
"origin" over `peer_ip` would invert the design.

---

### `GET /api/graph`

**1. Sources** `evidence_funnel.parquet` — the only persisted edge list.

**2. Available** `edge_index`, `node_a`, `node_b` *(codes)*, `size_a`, `size_b`
(component sizes at merge time), `min_size`, plus the evidence columns.
253,429 proposed unions in edge-application order.

**3. Without recomputation** **Partially, and only in code space.** An edge list
can be served; nodes cannot be labelled (GAP-1) and components cannot be
aggregated into summary nodes (GAP-2).

**4. Unavailable** Node labels · cluster membership · any neighbourhood beyond
one funnel row · aggregation counts.

**5. Joins** `evidence_funnel.node_a/node_b` → *(missing code map)*.

**6. Size** Capped by design. The full funnel is 4 MB, which is above any
sensible cap and must never be returned unfiltered.

**7. Risk** The stated rule — *the 14,885-member cluster is never sent whole* —
cannot currently be **enforced**, because there is no membership artifact to cap
against. The cap has to be built on top of the Stage-0 artifact, not asserted in
the handler.

---

### `GET /api/evidence/{id}` — the "why was this flagged?" path

**1. Sources** `evidence_funnel.parquet` (253,429, **PRODUCTION**),
`phase33_decisions.csv` (1,267,145, **SYNTHETIC_CONTROL**),
`reach_stress/…/phase33_decisions.csv` (28,500, SYNTHETIC_CONTROL).

**2. Available**

```
decisions: regime, decision_id, component_a, component_b, evidence_state,
           chi2, p_value, effect_size, pooled_n_a, pooled_n_b,
           proposing_edges, blocked, provenance_type, synthetic_network
funnel:    edge_index, node_a, node_b, pooled_a, pooled_b, min_pooled,
           size_a, size_b, min_size, dof, chi2, p_value, effect
sidecar:   production_rule {min_pooled_observations 25, min_observer 5,
           alpha 1e-4, min_effect 0.05}, dataset_sha256, dataset_id, notes
```

The sidecar's `production_rule` is what makes a record self-describing — the
rule in force is recorded next to the number, which was the Phase 4.1 fix.

**3. Without recomputation** Yes, for the fields above.

**4. Unavailable** `reason` — `SeparationEvidence.reason` exists in memory and
is **dropped at write time**; the "effect below floor" vs "not significant"
distinction is not on disk. `quality`, `score`, `limitations` — the audit's
`EvidenceRecord` fields, none of which exist (§4.6). **No global evidence id** —
`decision_id` is unique only within `(fixture, regime)`; the funnel keys on
`edge_index`. An id scheme has to be defined.

**5. Joins** `id` → artifact + row. Cross-artifact join is *not* possible:
funnel `edge_index` and decisions `decision_id` index different things (edges vs
component boundaries) and there is no key between them.

**6. Size** < 1 KB per record.

**7. Risk — the most serious in this document.** `phase33_decisions.csv`
columns 15–17 are `truth_category`, `entities_a`, `entities_b`. **These are
ground truth**, read through `load_entity_truth_FOR_EVALUATION_ONLY`. Serving
them would put generated entity labels into a UI as findings, and would be the
first time in this project that truth crossed into a presentation path.
A column denylist is mandatory, and `tests/test_truth_isolation.py:20-40` uses a
**manual whitelist** — a new `api/` package defaults to *unchecked* (audit §6.5).
Invert that list before the first handler is written.

---

### `GET /api/compare`

**1. Sources** **None** (GAP-4).

**2. Available** `phase33.csv` gives per-regime `clusters`, `largest`,
`coverage`, `blocked`, `contested`, `evaluated`, `abstained` — but that is the
*fused* run across controlled worlds, not chain-only vs fused for production.

**3. Without recomputation** **No.** Producing it means running both modes.

**4. Unavailable** The entire side-by-side.

**7. Risk** On production the two are identical (`blocked = 0`), and that
equality *is the finding*. An endpoint showing two identical columns must say
why — "the veto never fired because 40 of 253,429 unions reached the pooled
minimum and none separated" — or it reads as a broken feature.

---

### `GET /api/demo/scenarios` — ships as-is

**1. Sources** `data/demo/output/scenarios.json` (20,230 bytes).

**2. Available** Complete. Top level: `banner`, `demo`, `provenance`,
`provenance_type`, `not_a_measurement`, `seed`, `namespace`, `statements`
{mechanism, frozen_dataset}, `fixture`, `rule`, `totals`, `scenarios`.
Per scenario: `key`, `title`, `situation`, `chain_story`, `network_story`,
`reads`, `outcome`, `expected_outcome`, `matches_expectation`, `verdict`,
`addresses`, `pooled`, `announcements_seen`, `usable_observations`,
`chain_only_components`, `fused_components`, `contested`, `decisions`,
`contradiction`. Per decision: `address_a`, `address_b`, `verdict`,
`pooled_a/b`, `chi2`, `p_value`, `effect`, `reason`, `merged`.

Note `reason` **is** present here — the demo payload keeps what the production
CSV drops.

**3. Without recomputation** Yes. Serve near-verbatim.

**4. Unavailable** Nothing.

**5. Joins** None.

**6. Size** ~20 KB.

**7. Risk** Low, and already mitigated: every object carries `demo: true` +
`provenance: SYNTHETIC_DEMONSTRATION`, and `api.assert_demo_flagged` refuses an
unflagged payload.

> **Correction to the brief:** these are the Phase **3.4** demonstration
> scenarios, not "the Phase 3.3 fixtures". Phase 3.3's fixtures are the
> controlled worlds A–E, which are `SYNTHETIC_CONTROL`, **not** `DEMO`.
> Serving worlds under `/api/demo/` would relabel a real experiment as a
> demonstration — precisely the distinction Phase 4.1 introduced the three-type
> enum to preserve.

---

## 3. The evidence-classification enum — language risk

Requested: `CONFIRMED · OBSERVED · INFERRED · CANNOT_LINK · NO_EVIDENCE ·
CONTESTED`. What exists: `Verdict = {SEPARATED, NOT_SEPARATED, NO_EVIDENCE}`
plus `CONTESTED` as a **component** property, not a pairwise verdict.

| Requested | Backing | Assessment |
|---|---|---|
| `CONFIRMED` | none | **Nothing in this system is confirmed.** Co-spend is the common-input-ownership *heuristic*. Offering CONFIRMED invites the UI to use it. |
| `OBSERVED` | `observations.parquet` rows | Sound — a sighting is observed. |
| `INFERRED` | co-spend merges | Sound if it means "heuristic merge", not "network-supported". |
| `CANNOT_LINK` | `Verdict.SEPARATED` | Clean 1:1. |
| `NO_EVIDENCE` | `Verdict.NO_EVIDENCE` | Clean 1:1. |
| `CONTESTED` | component-level flag | Not a decision verdict; must be carried on the entity, not the evidence record. |
| — | **`NOT_SEPARATED` has no target** | **The dangerous one.** It is 40 of 40 decidable production decisions. Mapping it to `INFERRED` converts "we looked and found nothing" into a positive claim. It needs its own member. |

---

## 4. Implementation order

Ordered so the **"why was this flagged?" evidence path** is reachable earliest,
and so nothing ships that would require inventing a field.

**Stage 0 — precompute the two missing artifacts (pipeline, not API).**
A new CLI command writing, once, with Phase 4.1 provenance:
`address_index.parquet` (code → address, 822,942 rows, closes GAP-1) and
`clusters.parquet` (code → cluster_id + sizes, closes GAP-2). This is a pipeline
addition; the API still only reads. Everything below except Stages 1–2 is
blocked on it.

**Stage 1 — `/api/demo/scenarios`.** Zero gaps, zero joins, ~20 KB, already
flagged. It is also the clearest end-to-end telling of *why something was
flagged* (A→E), so it validates the response envelope before harder endpoints.

**Stage 2 — `/api/evidence/{id}`.** The core of the brief. Needs, in order:
a global evidence-id scheme; a **truth-column denylist** (`truth_category`,
`entities_a`, `entities_b`); sidecar provenance passthrough; and the inverted
truth-isolation test so `api/` is checked by default.

**Stage 3 — `/api/transaction/{id}`.** Second-most-complete. Needs a txid index
over two parquet files. Ships propagation + evidence quality; marks
inputs/outputs/timestamp as gaps rather than reading 663 MB per request.

**Stage 4 — `/api/overview`.** Two labelled blocks (PRODUCTION /
SYNTHETIC_CONTROL). Alert count omitted, not zeroed.

**Stage 5 — `/api/compare`.** Requires `fusion-summary` to persist its table
(small pipeline change, closes GAP-4).

**Stage 6 — `/api/entity/{id}`.** Unblocked by Stage 0.

**Stage 7 — `/api/graph`.** Unblocked by Stage 0; caps enforced against real
membership.

**Not in Phase 5 — `/api/alerts`.** No risk model exists. Shipping it means
inventing one.

## 5. Constraints Phase 5 must not break

- 566 tests pass. `test_frozen_metrics.py` pins Phase 1 and the frozen hash;
  `test_truth_isolation.py` pins the truth boundary; `test_provenance.py` pins
  the artifact marking. An `api/` package must be added to the isolation test.
- No new storage engine. Parquet + DuckDB only.
- The API must not import `cluster.pipeline`, `cluster.replay`,
  `network.separation.build_oracle`, or anything under `eval/` that recomputes.
  A source-level test can enforce that, the same way truth isolation is enforced.
