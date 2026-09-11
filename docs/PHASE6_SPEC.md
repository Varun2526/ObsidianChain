# Phase 6.0 — Entity Risk Ranking: frozen specification

**Status: FROZEN. Approved for implementation.**

M1 implementation-readiness check passed: the address-address counterparty
graph implies 3,359,487 pairs, mean address tx-incidence 1.60, p99 6, max
2,924 — tractable. One correction was required and applied: `cluster_size_final`
is removed from the M1 feature list, because final cluster membership must not
be a predictive feature. It remains describable in investigation mode (§4.3)
but never enters M0–M3.

This document exists because the ML design decisions that matter are not the
model. They are the prediction unit, the label policy, and the leakage
boundary. Those three are decidable today from the data, and each one has a
wrong answer that produces a spectacular number.

Everything below is grounded in measurements against `data/raw/`, recorded in
section 0 so the spec can be checked rather than believed. Every empirical
number names the script that produced it, so any claim here can be re-derived
rather than trusted.

Claims are tagged by kind and the kinds are not interchangeable:

| tag | meaning |
|---|---|
| `FACT` | a file schema or dataset property, requiring no inference |
| `MEASURED` | a number produced from the data, with the method named |
| `INFERENCE` | a conclusion drawn from facts or measurements |
| `DESIGN` | a choice with no uniquely correct answer, fixed deliberately |
| `PREDICTION` | falsifiable, registered before any model is fitted |
| `UNMEASURED` | named, and explicitly not yet measured |

---

## 0. Measured facts the spec rests on

Taken from the Elliptic++ files on disk, September 2026.

**Method.** §0.1–§0.6 are direct reads of the named files — label counts by
value count over `txs_classes.csv` / `wallets_classes.csv`, column inventories
from the CSV headers, per-timestep counts by grouping `txs_features.csv` on
`Time step`, boundary spans by min/max `Time step` per address, and cluster
coverage by joining `data/processed/address_clusters.parquet` to
`wallets_classes.csv`. §0.7, §0.8 and §5.6 name the specific script behind
each number. No frozen artifact was read or written outside
`address_clusters.parquet`, which was read only.

### 0.1 Labels

| | transactions | addresses |
|---|---|---|
| total | 203,769 | 822,942 |
| class 1 (illicit) | 4,545 (2.23%) | 14,266 (1.73%) |
| class 2 (licit) | 42,019 (20.6%) | 251,088 (30.5%) |
| class 3 (unknown) | 157,205 (77.1%) | 557,588 (67.8%) |
| labeled | 46,564 (22.9%) | 265,354 (32.2%) |
| **illicit share of labeled** | **9.76%** | **5.38%** |

`wallets_classes.csv` holds 822,942 rows and 822,942 unique addresses, and no
address carries two distinct classes. **A label is static per address: it has
no time dimension.**

### 0.2 Features actually available

`txs_features.csv`, 184 columns:

| block | count | interpretable? |
|---|---|---|
| `txId`, `Time step` | 2 | keys |
| **named transaction economics** | **17** | **yes** |
| `Local_feature_1..93` | 93 | no — anonymised |
| `Aggregate_feature_1..72` | 72 | no — anonymised |

The 17 named: `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`,
`num_input_addresses`, `num_output_addresses`, `in_BTC_{min,max,mean,median,total}`,
`out_BTC_{min,max,mean,median,total}`.

`wallets_features.csv`, 57 columns, all named: transaction counts, BTC
transacted/sent/received (total/min/max/mean/median), fees and fee shares,
blocks-between-transactions statistics, counterparty statistics, lifetime.

### 0.3 THE LEAKAGE FINDING — `wallets_features.csv` is not a time series

`wallets_features.csv` has 1,268,260 rows over 822,942 unique addresses, with
a `Time step` column, which reads as a per-timestep observation. It is not.

Every feature tested is **constant across all of an address's timesteps**:

```
num_timesteps_appeared_in    constant for 100.0% of multi-timestep addresses
lifetime_in_blocks           constant for 100.0%
last_block_appeared_in       constant for 100.0%
first_block_appeared_in      constant for 100.0%
total_txs                    constant for 100.0%
btc_transacted_total         constant for 100.0%
```

Verbatim, one address:

```
address                            Time step  lifetime_in_blocks  total_txs  num_timesteps_appeared_in
1111DAYXhoxZx2tsRnzimfozo783x1yC2         25             46370.0        8.0                        6.0
1111DAYXhoxZx2tsRnzimfozo783x1yC2         29             46370.0        8.0                        6.0
1111DAYXhoxZx2tsRnzimfozo783x1yC2         39             46370.0        8.0                        6.0
1111DAYXhoxZx2tsRnzimfozo783x1yC2         43             46370.0        8.0                        6.0
1111DAYXhoxZx2tsRnzimfozo783x1yC2         47             46370.0        8.0                        6.0
1111DAYXhoxZx2tsRnzimfozo783x1yC2         48             46370.0        8.0                        6.0
```

At timestep 25 the row already states that this address will transact 8 times
across 46,370 blocks and appear in 6 timesteps. The file is a **static
lifetime profile replicated across every timestep the address appears in.**

Consequence: **training any model on `wallets_features.csv` under a
chronological split leaks the future into the training set.** The resulting
metrics would be invalid, and invalid in the direction that looks like
success.

Of the 55 non-key columns, exactly **three** are past-facing at any
`t >= first appearance` and therefore survive: `first_block_appeared_in`,
`first_sent_block`, `first_received_block`. **52 of 55 are unusable.**

There are also 246,545 `(address, Time step)` pairs carrying more than one
row (max 568), so the file requires de-duplication before any use at all.

### 0.4 Illicit rate by timestep

Transaction-level, illicit as a share of labeled:

```
t1-t6    0.4 - 2.1%     quiet opening
t7-t34   3.2 - 36.0%    high and volatile
t35-t42  1.9 - 14.7%
t43-t47  1.8, 1.5, 0.4, 0.3, 2.6%   <-- collapse
t48-t49  7.6, 11.8%
```

The collapse at t43 is the documented dark-market shutdown. It is a genuine
regime change, not noise, and any test set spanning it measures two different
worlds.

Illicit transactions by candidate split: **t1–34: 3,462 · t35–41: 675 ·
t42–49: 408.**

### 0.5 Boundary-spanning addresses

Labels are static but addresses appear in several timesteps, so an address can
sit on both sides of a temporal cut and be memorised.

```
boundary t34/t35:  2,499 of 265,354 labeled addresses span it (0.9%), 49 illicit
boundary t41/t42:  2,101 of 265,354 (0.8%), 30 illicit
```

**Under 1%.** Excluding them entirely costs almost nothing, so the
entity-disjoint guarantee is cheap and becomes mandatory rather than a
stress test.

### 0.6 Cluster label coverage

Against `data/processed/address_clusters.parquet` (284,709 addresses):

```
class 1 (illicit)    8,849
class 2 (licit)    105,222
class 3 (unknown)  170,638

clusters with >=1 labeled address   9,795
clusters with >=1 illicit             578
clusters with >=2 labeled           8,441
largest labeled cluster            11,001 labeled addresses
```

578 positive clusters is the ceiling for cluster-level evaluation. 8,441
clusters with two or more labeled members is what gives the aggregation
choice anything to bite on.

The 11,001-member cluster is the Phase 1 super-cluster, already known from
Phase 1.1 to be contaminated at 6 illicit / 10,995 licit. `MEASURED`: its
illicit share among labeled members is **0.0545%**. `INFERENCE`: it is a
concrete stress case for max aggregation, because one high-risk member can
dominate the score of a very large cluster despite overwhelming licit
membership. This implies **no** measured precision for AGG-MAX, which is
unmeasured.

### 0.7 Aggregate-feature provenance — UNRESOLVED

`MEASURED` (`inv1a.py`, `inv1b.py`, `inv1c.py`, `inv1d.py`):

| test | result |
|---|---|
| timestep span of `txs_edgelist.csv` edges | **234,355 of 234,355 (100.000%) lie within one timestep**; 0 forward, 0 backward |
| reconstruction of the 72 aggregates from that graph | 1,116 candidates ({max,min,mean,std} × {pred,succ,both} × 93 local features) over timesteps 1–10: **2 of 72** reach \|r\| ≥ 0.95; **58 of 72** fall below 0.80 |
| completeness of `txs_edgelist.csv` | independent `AddrTx→TxAddr` reconstruction yields **13,574,231** distinct transaction pairs; only **233,260 (1.7%)** appear in the edgelist; **4,837,988** omitted links cross forward in time, max jump **+48** timesteps |
| isolated-transaction test | **INAPPLICABLE** — 0 of 203,769 transactions are isolated |

`INFERENCE.` If the aggregates were computed over the dataset's own graph they
could not reach a later timestep. But they are not reproducible from it, and
that graph is 1.7% of the real one. Their true neighbourhood cannot be bounded
from local data, so they can be shown neither safe nor unsafe. See §3.3.

### 0.8 PEEL-1 feasibility benchmark

`MEASURED` (`inv2b.py`). Deterministic timestep-prefix subsets; chains run
forward, so a prefix holds complete chain beginnings. Hops strictly forward.

| subset | txs | candidates | pairs examined | fwd hops | same-t hops | chains ≥3 | max depth | sec | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| t1..t4 (12.1%) | 24,738 | 14,952 | 139,982 | 20,678 | 100,947 | 369 | 4 | 0.0 | 289 MB |
| t1..t9 (25.2%) | 51,370 | 31,580 | 943,435 | 284,271 | 374,895 | 1,885 | 9 | 0.2 | 524 MB |
| t1..t23 (51.2%) | 104,358 | 62,891 | 3,939,699 | 969,327 | 2,000,366 | 5,280 | 22 | 0.7 | 1,392 MB |
| **t1..t49 (100%)** | **203,769** | **124,962** | **9,917,510** | **3,286,789** | **3,345,860** | **12,793** | **43** | **2.4** | **3,157 MB** |

`INFERENCE.` Full-scale enumeration is computationally trivial — 2.4 s. The
RAM is dominated by a pandas merge on raw address strings and drops sharply by
factorising addresses to `int32` first, as Phase 1 already does. **The
bottleneck is semantic, not computational** (§5.4).

`FACT.` This benchmark used the *withdrawn* dominance-based candidate rule
(`num_output_addresses == 2 AND dominance >= 0.80`, 124,962 candidates). The
adopted PEEL-1 rule (§5.5) drops dominance and admits 2–3 outputs, giving
**167,808** admissible transactions and different chain counts. The benchmark
is retained here as the feasibility evidence it is; the binding chain counts
are in §5.7.

---

## 1. Target and label policy — FROZEN

**Prediction unit: the address.** Not the transaction, not the cluster.

```
Address
  ├── class 1 illicit   ->  y = 1
  ├── class 2 licit     ->  y = 0
  └── class 3 unknown   ->  EXCLUDED from supervised training and from
                            supervised evaluation. Never y = 0.
```

Rationale for the address over the transaction: 5.7× the labeled examples
(265,354 vs 46,564), 3.1× the positives (14,266 vs 4,545), and it is the unit
the existing pipeline already emits. It is **not** because transaction
features are uninterpretable — `txs_features.csv` carries 17 named economic
features, and an earlier draft of this reasoning was wrong about that.

Rationale for excluding unknown: HANDOFF invariant from Phase 1.1, *"Never
treat unknown as licit."* 67.8% of addresses are unknown. Mapping them to 0
would fabricate 557,588 negative labels.

### 1.1 Two prevalences, both reported, never interchanged

- **labeled prevalence 5.38%** — the PR-AUC no-skill baseline for every
  number this project reports.
- **population prevalence 1.73%** — what the rate would be if unknown were
  all licit, which is exactly what we refuse to assume.

Every PR-AUC in every table is accompanied by the 5.38% baseline. Quoting a
PR-AUC against the 1.73% baseline inflates the apparent lift roughly 3×.

### 1.2 The selection bias is stated, not solved

Training on 32.2% of addresses means the model is fitted to whatever made an
address labelable. Phase 6.0 does not attempt PU-learning. It states the bias
in the artifact and in the report as a known limitation.

### 1.3 A static label against time-varying features

The label has no time dimension, so the model predicts "is this address ever
illicit" from "how this address behaved up to t". This is the honest reading
and it must be the wording used. It is **not** "is this address illicit right
now".

---

## 2. Address → cluster aggregation — FROZEN AS AN EXPERIMENT

The cluster is an **inference**, never ground truth. The chain is:

```
address label (ground truth)
      -> address risk           (model output)
      -> cluster risk           (aggregation, chosen by experiment)
      -> ranked alert
```

Cluster risk is **not** assumed to be the mean. Four candidates are evaluated
on the same temporal test set:

| id | aggregation | why it is a candidate |
|---|---|---|
| AGG-MAX | `max(risk_i)` | one bad member condemns the entity |
| AGG-TOPK | mean of top-k, k = 3 | robust to a single outlier |
| AGG-QUANT | 90th percentile | scale-free version of top-k |
| AGG-WMEAN | mean weighted by address activity | a dormant member should not count as much as an active one |

### 2.1 The derived cluster label

`DESIGN.` Primary cluster evaluation label:
**`illicit_share_among_labeled >= 0.10`**, over the 9,795 clusters holding at
least one labeled address. This yields **537 positive and 9,258 negative
clusters**.

> **THIS IS A DERIVED EVALUATION CONSTRUCT. IT IS NOT ENTITY GROUND TRUTH.**
> The only ground truth in this project is the address label. A cluster is an
> inference produced by co-spend clustering, and this threshold is a statement
> about how we choose to score that inference — not a claim about any entity.

`MEASURED` (`inv3.py`, `inv3b.py`). The distribution is almost perfectly
bimodal: of the 578 clusters with ≥1 illicit member, **510 (88.2%) are purely
illicit**, and `illicit_share` is already 1.000 at the 25th percentile of that
group. Candidate definitions therefore differ little:

| definition | positives | super-cluster 419015 admitted? |
|---|---|---|
| ≥1 illicit | 578 | **yes — 6 illicit / 11,001 labeled** |
| ≥2 illicit | 464 | **yes — 6 ≥ 2** |
| **share ≥ 0.10** | **537** | no |
| share ≥ 0.50 | 519 | no |

`INFERENCE.` The threshold is not a tuning knob. Because the distribution is
bimodal, any threshold in (0, 1) selects nearly the same set; it does exactly
one job, which is excluding trace contamination. That is why it can be fixed
now, before any model exists.

`INFERENCE.` The decisive case is cluster 419015 at `illicit_share = 0.000545`.
Both count-based definitions admit it; every share-based definition excludes
it. Alerting on it would send an investigator after 11,001 addresses to find
6, which is a clustering artefact rather than an entity finding.

`DESIGN.` No minimum-labeled-count guard is added. 1,354 clusters (13.8%) carry
exactly one labeled address; whether they belong in the evaluation population
is a question about the population and must not be smuggled into the label
definition.

`DESIGN.` **`>=1` illicit (578 positives) is retained as a registered
sensitivity analysis**, reported alongside the primary. Both are fixed before
fitting and **must not be revisited against model performance.**

---

## 3. Feature contract — FROZEN

### 3.1 The central consequence of section 0.3

Because `wallets_features.csv` is whole-life, **address features must be
recomputed as-of-timestep** from the transaction layer. The inputs exist:

```
txs_features.csv     txId, Time step, total_BTC, fees, size,
                     num_input_addresses, num_output_addresses, in/out_BTC_*
AddrTx_edgelist.csv  input address  -> tx      (address spends)
TxAddr_edgelist.csv  tx -> output address      (address receives)
```

For address `a` at timestep `t`: take every transaction involving `a` with
`Time step <= t`, and aggregate. This is the main engineering task of Phase 6
and it is what makes a temporal split honest.

### 3.2 Feature families

**M0 — address behaviour, as-of-t** (recomputed, §3.1)

```
n_txs_asof_t, n_txs_as_sender_asof_t, n_txs_as_receiver_asof_t
btc_sent_{total,min,max,mean,median}_asof_t
btc_received_{total,min,max,mean,median}_asof_t
fees_{total,mean,median}_asof_t, fees_as_share_mean_asof_t
blocks_btwn_txs_{mean,median,min,max}_asof_t
active_timesteps_asof_t, blocks_since_first_seen_asof_t
first_block_appeared_in, first_sent_block, first_received_block   (safe, §0.3)
```

**M1 adds — graph**, over the address↔transaction bipartite graph
(`AddrTx` + `TxAddr` + `txs_features`) restricted to `Time step <= t`

```
in_degree_asof_t, out_degree_asof_t
weighted_in_degree_asof_t, weighted_out_degree_asof_t
unique_counterparties_asof_t, counterparty_growth_rate
local_clustering_coefficient_asof_t
cluster_size_asof_t                              (final size EXCLUDED, §4.3)
```

`FACT.` **Not** over `AddrAddr_edgelist.csv`. That file carries no timestep
(§5.2), so "restricted to `Time step <= t`" is not expressible on it — the
same void assumption withdrawn from §5. Counterparties are therefore derived
through the transaction layer, where `Time step` exists: address `a` is a
counterparty of address `b` as of `t` if some transaction with
`Time step <= t` has one as an input and the other as an output. `INFERENCE`:
this keeps M1 on the same substrate as M2 and leaves
`test_loader_never_reads_addraddr` untouched (§5.8).

**M2 adds — PEEL-1 chain structure** (§5), all label-blind, all nullable

```
in_chain                    bool
chain_depth_max             int    longest chain containing the address
position_in_chain           int    first position at which it appears
chain_count                 int    distinct chains containing it
chain_fanout_mean           float  mean num_output_addresses over the chain
hop_gap_median              float  median Time step delta between hops
value_retention_ratio       float  NULL unless the chain qualifies (§5.6)
value_retention_available   bool
```

**M3 adds — network**, from the existing separation oracle

```
pooled_observations, observer_coverage
network_divergence, origin_consistency
network_evidence_state
```

### 3.3 Explicitly excluded

| excluded | reason |
|---|---|
| 52 of 55 `wallets_features.csv` columns | whole-life aggregates, §0.3 |
| `Aggregate_feature_1..72` | **Provenance UNRESOLVED — excluded until established.** Not claimed forward-looking; not claimed safe. See §0.7 for the four measurements: the dataset's graph is 100% within-timestep, but the aggregates are not reproducible from it (2 of 72 at \|r\| ≥ 0.95) and that graph is only 1.7% of the real one, with 4,837,988 omitted links crossing forward. `INFERENCE`: their true neighbourhood cannot be bounded from local data, so they cannot be shown safe. Excluded on unresolvable provenance, **not** on a demonstrated leak. |
| k-hop illicit-neighbour counts | §4.2 |
| anything derived from an alert or investigation outcome | circularity |
| `peer_ip` → owner, or any identity claim from network data | HANDOFF invariant 1 and 7 |

`INFERENCE.` The aggregate-feature exclusion is reversible only by
establishing provenance — for example by obtaining Elliptic's feature
construction definition, or by reproducing the aggregates from a graph whose
timestep span is known. **Performance on a held-out split is not evidence of
provenance and may not be used to readmit them.**

`Local_feature_1..93` are transaction-intrinsic and not known to be
forward-looking, but they are anonymised and therefore unexplainable. They are
**not** in M0–M3. They may be used only in a separate, clearly-labelled
"anonymised-features comparison" run whose purpose is to bound how much
performance the interpretable feature set gives up.

---

## 4. Leakage rules — FROZEN

### 4.1 The as-of-t rule

Every feature for address `a` at timestep `t` is computed **only** from
transactions with `Time step <= t`. A feature that cannot be expressed this
way does not enter the model. This is enforced by construction in the feature
builder and asserted by test, in the manner of `test_truth_isolation.py`.

### 4.2 k-hop illicit exposure is excluded from Phase 6.0

Elliptic has 49 timesteps with **no intra-timestep ordering**, so "illicit
neighbours known as of t" is undefined for neighbours inside the same
timestep. A same-timestep neighbour is not known to precede the target.

A k-hop illicit-exposure feature is the most likely source of a large,
wrong-for-the-right-reason result in this entire design. It is excluded from
M0–M3 and deferred to a separate experiment with its own temporal
justification.

### 4.3 Final-state features are flagged, not silently mixed

`cluster_size_final` uses the completed Phase 1 clustering, which is built
from the whole chain. It is therefore an **investigation-mode** feature, not a
monitoring-mode one. Phase 6.0 reports both modes separately:

- **monitoring mode** — strictly as-of-t features only;
- **investigation mode** — final-state features permitted, clearly labelled.

Numbers from the two modes are never presented in the same column.

### 4.4 Entity disjointness is mandatory, not a stress test

Because labels are static and addresses recur, a boundary-spanning address is
memorised rather than predicted. At 0.9% of labeled addresses (§0.5) the cost
of exclusion is negligible, so:

> Any labeled address active on both sides of a split boundary is **removed
> from the dataset entirely.**

This makes train/val/test address-disjoint by construction.

---

## 5. Laundering structure — FROZEN AS DETECTION, NOT CLASSIFICATION

### 5.1 Why this is detection and not classification

`INFERENCE.` Elliptic++ labels addresses and transactions, never subgraphs.
Deriving a subgraph label (">=k illicit members") and then predicting it from
features correlated with member illicitness measures only our ability to
recover a rule we invented. That is the component-keyed-entities error and the
manufactured-54.2%-precision error in a new costume.

Detection and scoring are therefore separated, and the detector is
**LABEL-BLIND**: it may not open `wallets_classes.csv`, `txs_classes.csv`, any
illicit or licit label, or any model prediction, at any point during structure
discovery. This is asserted by test in the manner of
`tests/test_truth_isolation.py`.

```
AddrTx + TxAddr + txs_features,  Time step <= t
            |
            v
   chain detector (PEEL-1)        <-- LABEL-BLIND
            |
            +--> chain objects    --> the investigation unit (dashboard)
            |
            +--> membership+shape --> address features for M2
```

PEEL-1 is **structural candidate generation, not laundering classification.**
It emits candidates; it never asserts that a candidate is laundering.

The scientific question stays measurable rather than assumed:

> Does membership in the detected structure predict address illicitness on the
> held-out temporal test set, better than chance?

A negative answer is a result, exactly as the Phase 3.3 network finding was.

### 5.2 Substrate — corrected

`FACT.` `AddrAddr_edgelist.csv` has exactly two columns, `input_address` and
`output_address`. It carries **no value and no timestep**. The Phase 6.0 draft
specified a detector over AddrAddr requiring value decay and block-time
ordering; neither quantity exists there. **That specification was void and is
withdrawn.**

The substrate is:

| file | provides |
|---|---|
| `AddrTx_edgelist.csv` | `input_address -> txId` (address spends in tx) |
| `TxAddr_edgelist.csv` | `txId -> output_address` (address receives from tx) |
| `txs_features.csv` | `Time step`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in_BTC_*`, `out_BTC_*` |

`AddrAddr_edgelist.csv` is **not required by Phase 6.0 and is not read.**

### 5.3 What is observed, what is inferred, what is unavailable

The distinction is structural to this section and must survive into the
artifact.

| quantity | status | basis |
|---|---|---|
| transaction `Time step` | **OBSERVED** | `txs_features.csv` |
| transaction `total_BTC`, `in_BTC_*`, `out_BTC_*`, `fees`, `size` | **OBSERVED** | `txs_features.csv` |
| `num_input_addresses`, `num_output_addresses` | **OBSERVED** | `txs_features.csv` |
| address ∈ inputs(tx), address ∈ outputs(tx) | **OBSERVED** | `AddrTx`, `TxAddr` |
| which output address received which amount | **UNAVAILABLE** | `TxAddr` carries no per-output value |
| continuing-output identity | **INFERRED** | from chain continuation, never observed |
| peeled-output identity | **INFERRED** | the complement of the continuing output |
| per-hop value retention | **CONDITIONALLY OBSERVED** | see §5.6 |

### 5.4 Limitations recorded in the specification, not discovered later

`FACT (A).` `TxAddr_edgelist.csv` has no per-output BTC value, so the system
cannot observe which specific output is the continuing one.

`INFERENCE (B).` Continuing-vs-peeled is therefore an inference at transaction
level, derived from which output address goes on to appear as an input of a
later admissible transaction. It is never presented as an observation.

`FACT (C).` Elliptic++ has 49 timesteps with **no intra-timestep ordering**.
Only strictly forward transitions `t2 > t1` may order a chain.

`MEASURED (D)` (`inv2b.py`). 3,345,860 of 6,632,649 candidate hops at full
scale are same-timestep — **50.4%** — and are therefore unusable for ordered
chains. The detector observes roughly half of all potential chain steps.

`MEASURED (E)` (`inv2a.py`). **A dominant output is not a laundering
signature.** 162,544 of 203,769 transactions (79.77%) have exactly 2 outputs,
and 124,962 of those (76.88%) show `out_BTC_max / out_BTC_total >= 0.80` —
**61.3% of all transactions.** That is the ordinary payment-plus-change shape.

`DESIGN.` Consequently this specification does **not** define "dominant output
= peeling", and no dominance filter appears in PEEL-1. Small fan-out is a
necessary filter, never a signature. All discriminative weight rests on the
structure of repeated ordered transitions.

`FACT.` `out_BTC_*` are normalised in Elliptic (observed range
0.000 – 5960.558), so any ratio built from them indicates relative magnitude,
not a true value share. This is recorded on every feature derived from them.

### 5.5 The detector — PEEL-1

`DESIGN.` Deliberately simple, deterministic, bounded and interpretable. It is
a candidate generator, not a laundering detector.

**Admissible transaction** (OBSERVED): `num_output_addresses ∈ {2, 3}`.

**Hop** `tx_i -> tx_{i+1}` (OBSERVED): some address `a` with
`a ∈ outputs(tx_i)` and `a ∈ inputs(tx_{i+1})`, and
`Time step(tx_{i+1}) > Time step(tx_i)`. Strict inequality guarantees a DAG
and assumes no intra-timestep ordering.

**Chain**: a path over admissible transactions joined by hops. Depth is the
longest-path depth in the DAG. A chain is identified by its **terminal
transaction** and its maximal depth; "chains at depth >= d" counts nodes whose
longest inbound path has depth >= d.

**Admissible chain** (DESIGN): `depth >= 5`. See §5.7.

**As-of-t rule**: a chain visible at timestep `t` is truncated to hops with
`Time step <= t`. Chain features are recomputed per `t` and never taken from
the completed chain, or §4.1 is violated.

**Address features emitted**: the M2 block in §3.2. All label-blind, all
nullable.

### 5.6 Value retention — conditionally observed, never imputed

`INFERENCE.` `total_BTC(tx_{i+1}) / total_BTC(tx_i)` is **not** a retention
measure: `tx_{i+1}` may draw inputs from addresses unrelated to `tx_i`, so the
ratio compares two unrelated totals.

There is one case where retention is attributable. If
`num_input_addresses(tx_{i+1}) == 1`, all of that transaction's input value
arrived through the single address `a`, which received from `tx_i`. On a chain
where every transaction from position 2 onward has exactly one input address,
`in_BTC_total` is attributable along the whole path and a retention ratio is
defensible.

`MEASURED` (`g2.py`, the G2 freeze-blocking measurement). Share of PEEL-1
chains satisfying that condition:

| depth | PEEL-1 chains | single-input chains | coverage |
|---|---|---|---|
| ≥ 3 | 16,043 | 12,114 | 75.51% |
| **≥ 5 (primary)** | **11,338** | **8,056** | **71.05%** |
| ≥ 10 (sensitivity) | 6,556 | 4,362 | 66.53% |

`DESIGN.` The pre-registered rule was: retain `value_retention_ratio` if
coverage ≥ 10%, remove it below. Coverage at the primary depth is **71.05%**,
so **`value_retention_ratio` and `value_retention_available` are RETAINED** in
M2. The rule was fixed before the measurement and no model was consulted.

`DESIGN.` `value_retention_ratio` is emitted **only** for qualifying chains,
with `value_retention_available = false` and a NULL ratio otherwise. It is
never imputed, never zero-filled, and never averaged across available and
unavailable chains. This follows the existing house rule from the evidence
contract, where `dof_production` is a nullable integer precisely so that
"not computed" cannot read as "computed, and the answer was zero".

### 5.7 Minimum chain depth — primary and sensitivity

`MEASURED` (`g2.py`, PEEL-1 candidate rule). The adopted detector yields
**16,043** chains at depth ≥ 3, **11,338** at depth ≥ 5, and **6,556** at
depth ≥ 10, over a DAG of 30,790 nodes built from 167,808 admissible
transactions and 4,481,802 strictly-forward hops.

`DESIGN.` **Primary threshold `depth >= 5`. Sensitivity analysis at
`depth >= 10`.** The reasoning is structural and statistical, fixed before any
model is fitted, and explicitly not performance-based:

1. Because 50.4% of hops are discarded as same-timestep (D), an observed depth
   of `d` **understates** the true chain length. The observable threshold
   should therefore sit below the conceptual one, not above it.
2. `depth >= 10` cuts the candidate set to 6,556 across 284,709 clustered
   addresses, making an already sparse feature sparser and its measured effect
   noisier.
3. Depths 1–2 are ordinary transaction sequences; 5 is the smallest depth
   requiring repetition sustained across at least five distinct timesteps.

These are **structural candidates, not labels and not positives.**

`INFERENCE.` Max observed depth is 43 across 49 timesteps — one hop per
timestep — which is as consistent with a long-lived reused address as with
peeling. This is an interpretation caveat carried on every M2 result, not a
resolved question (§8).

### 5.8 Boundary note — withdrawn

The draft proposed narrowing `test_loader_never_reads_addraddr` to permit
`AddrAddr_edgelist.csv` for features. `INFERENCE.` PEEL-1 does not read that
file (§5.2), so **no change to that guard or to the HANDOFF trap table is
required by Phase 6.0.** The guard stands exactly as written.

---

## 6. Splits, evaluation, and pre-registered predictions — FROZEN

### 6.1 Split

```
TRAIN       t1  - t34      3,462 illicit transactions
VALIDATION  t35 - t41        675
TEST        t42 - t49        408
```

Chosen for comparability with published Elliptic work. Addresses are assigned
by first appearance; boundary-spanning addresses are dropped (§4.4).

**Test metrics are additionally reported split at t43**, because the
dark-market shutdown (§0.4) makes t42 and t43–t49 different regimes.
Reporting one blended test number would hide the regime change.

### 6.2 Metrics

| tier | metric |
|---|---|
| primary | **PR-AUC (average precision)**, always beside the 5.38% no-skill baseline |
| secondary | precision, recall, F1 on the illicit class; ROC-AUC; confusion matrix |
| operational (address) | **Precision@10 / @50 / @100** |
| operational (cluster) | **Precision@10 / @50 / @100** and **Recall@10 / @50 / @100** |
| calibration | reliability curve, Brier score, log loss |

Accuracy is not reported. At 5.38% prevalence it is not informative.

`DESIGN.` Cluster recall is reported beside cluster precision so an
aggregation cannot score well merely by being conservative. The address-level
metric set is unchanged.

### 6.3 Calibration and severity

Raw LightGBM scores are calibrated on the **validation** split, never on
training data. Severity bands are then derived from validation precision
targets and frozen.

### 6.3.1 Threshold selection — deterministic, validation only

`DESIGN.` One procedure serves all three populated bands, with precision
targets **CRITICAL 0.90, HIGH 0.75, MEDIUM 0.50**. Scores are the calibrated
validation probabilities. **The test split is never consulted.**

For a band with precision target `p`:

1. **Candidate thresholds** — every distinct calibrated score present in the
   validation split. No grid and no rounding, so the procedure is reproducible
   from the data alone.
2. **Monotone-suffix constraint** — retain candidate `s` only if validation
   precision at `>= s'` meets `p` **for every** `s' >= s`. Precision is not
   monotone in the threshold, and taking the first crossing point would select
   a noisy local maximum that does not survive on new data.
3. **Minimum support** — retain `s` only if at least **50** validation
   addresses score `>= s`. A band defined on a handful of samples is not a
   band.
4. **Selection and tie handling** — among retained candidates take the
   **minimum** `s`: the largest alert volume, hence maximum recall, subject to
   the precision target. Ties cannot occur, because candidates are distinct
   scores and the minimum is unique; a single retained point is taken as-is.
5. **No feasible threshold** — the band is recorded as **UNPOPULATED** and
   emits no alerts. It is **not** filled by lowering `p`, relaxing the support
   minimum, or dropping the monotone-suffix constraint.

**Band assignment**, applied top-down so every score lands in exactly one band:

```
CRITICAL   score >= t_CRITICAL
HIGH       t_HIGH    <= score < t_CRITICAL
MEDIUM     t_MEDIUM  <= score < t_HIGH
LOW        score < t_MEDIUM           (context only, never an alert)
```

An UNPOPULATED band is skipped and the next band extends upward in its place.

`INFERENCE.` The monotone-suffix rule guarantees
`t_CRITICAL >= t_HIGH >= t_MEDIUM`, since a higher precision target can only
shrink the retained set. **This ordering is asserted at build time and the run
fails loudly if violated** rather than silently emitting inverted bands.

`DESIGN.` The selected thresholds are written into the artifact provenance
alongside the validation precision and support actually achieved at each, and
are **never recomputed on the test split.** Test-set band precision is a
reported outcome, never an input to selection.

### 6.4 The ablation

| model | features |
|---|---|
| M0 | address behaviour, as-of-t |
| M1 | M0 + graph |
| M2 | M1 + PEEL-1 chain structure |
| M3 | M2 + network |

Identical split, identical calibration procedure, identical metrics.
Reported as `Δ(M1−M0)`, `Δ(M2−M1)`, `Δ(M3−M2)` on PR-AUC and on
Precision@50.

`DESIGN.` M2 is run at PEEL-1 `depth >= 5` (primary) and `depth >= 10`
(sensitivity), per §5.7. Both thresholds are fixed before fitting. The cluster
evaluation is run under the primary label `illicit_share >= 0.10` and the
registered sensitivity label `>=1 illicit`, per §2.1.

### 6.5 Pre-registered predictions

Recorded **before** any model is fitted. Confirmed or refuted, both are
reported.

**P1 — `PREDICTION`: `Δ(M3−M2)` on PR-AUC will be within ±0.005 of zero.**

`MEASURED.` Of 284,709 clustered addresses, 17.4% carry any pooled network
observation, 2.3% reach pooled ≥ 5, and **1.64% reach the production minimum
of 25.**

`INFERENCE.` Low coverage **motivates** the prediction of limited aggregate
contribution, but **does not imply it**: a sparse feature can be highly
discriminative on the rows where it is present, and a model may exploit it
there without moving an aggregate metric — or may move the metric
substantially if those rows are concentrated among positives. P1 is therefore
an **empirical, falsifiable prediction, not a mathematical consequence of
coverage.**

If P1 holds, the finding is *"network observations are investigative context
under this data regime; reach, not the rule, is the limit"* — the Phase 3.3
conclusion arriving by an independent route. If P1 is refuted, that is direct
quantitative evidence the network layer carries predictive signal, and it is
the more interesting outcome.

**P2 — `PREDICTION`: AGG-MAX will rank below AGG-TOPK on cluster
Precision@50.**

`INFERENCE.` Motivated by the 11,001-member cluster (§0.6, §2.1), where `max`
over many members is dominated by the model's false-positive tail. Under the
primary cluster label that cluster is a **negative**, so an aggregation that
ranks it highly is penalised.

**P3 — `PREDICTION`: `nAP(t42) − nAP(t43–t49) >= 0.10`**, where
`nAP = (AP − prevalence) / (1 − prevalence)`.

`INFERENCE — why raw PR-AUC is invalid here.` Labeled illicit prevalence is
239/2,154 = **11.09%** at t42 and 169/6,687 = **2.53%** across t43–t49
(transaction-level, §0.4). The no-skill baseline therefore falls by 0.086 on
prevalence alone, so `PR-AUC(t43–49) < PR-AUC(t42)` would be satisfied by
construction with a perfectly stable model.

`INFERENCE — why lift was rejected.` `lift = AP / prevalence` removes the
no-skill floor but not the ceiling: maximum attainable lift is `1/prevalence`,
which is **9.02 at t42 and 39.57 at t43–t49, a 4.39× difference.** A fixed
ratio threshold on lift means something different in each period.

`DESIGN.` Normalised average precision fixes both endpoints — no-skill = 0 and
perfect = 1 in **both** periods — so the difference is comparable.

`DESIGN — why 0.10.` t43–t49 carries 169 labeled illicit transactions, on
which a precision estimate has a binomial standard error of roughly ±3.8
percentage points at p = 0.5. A 0.10 absolute drop in a [0,1]-bounded quantity
sits well outside that noise while stopping far short of requiring total
collapse. It is registered before any model is fitted and may not be adjusted
afterwards.

`UNMEASURED — pending freeze condition.` **The prevalences above are
transaction-level.** The model is address-level, so the address-level
baselines `prevalence(t42)` and `prevalence(t43–t49)` **must be recomputed
during Phase 6 dataset construction** before P3 is considered fully frozen.
The *form* and the *0.10 threshold* are frozen now; only the baselines are
pending.

`DESIGN.` Within the test split, an address is assigned to the t42 or
t43–t49 sub-period by its **first appearance inside the test window**,
consistent with §4.4.

`INFERENCE — stated caveat.` Both sub-periods carry few positives (239 and
169 transaction-level), so both nAP estimates are noisy. P3 is a coarse
directional test, reported with its confidence interval, never as a point
claim.

**P4 — `PREDICTION`: `Δ(M1−M0)` > `Δ(M2−M1)`.**

`INFERENCE.` Graph degree and counterparty structure are dense and available
for most addresses; PEEL-1 membership is sparse — 11,338 chains at depth ≥ 5
over 284,709 clustered addresses (§5.7) — in the manner of every other
structural signal measured in this project so far. Stated with low confidence:
this is the prediction most likely to be wrong, and the one worth being wrong
about.

---

## 7. What Phase 6.0 deliberately does not do

GNNs, graph transformers, Bayesian ensembles, temporal attention, PU-learning,
anomaly-detection stacks, k-hop label propagation, Kafka, distributed
training. Each is a possible later comparison. None is required for a working,
defensible system, and every one of them costs explainability that this
project's whole credibility rests on.

---

## 8. Open questions

The three original questions are closed. What remains is recorded here as
limitations, not as blockers, and each states explicitly whether it blocks.

### 8.1 Closed by the pre-freeze investigations

| was | outcome |
|---|---|
| Are `Aggregate_feature_1..72` forward-looking? | **CLOSED — UNDETERMINED.** Not resolvable from local data. All 72 excluded on unresolved provenance (§0.7, §3.3). |
| Peeling enumeration cost at full scale | **CLOSED — feasible.** 2.4 s, 3.16 GB peak (§0.8). The bottleneck is semantic, not computational (§5.4). |
| Cluster label definition | **CLOSED.** `illicit_share >= 0.10`, 537 positives, with `>=1` illicit as registered sensitivity (§2.1). |
| G2 — value-retention coverage | **CLOSED — 71.05%** at depth ≥ 5, above the pre-registered 10% rule, so the feature is retained (§5.6). |

### 8.2 Open limitations — none blocking

| # | question | status | blocks freeze? |
|---|---|---|---|
| 1 | Provenance of `Aggregate_feature_1..72` | **UNDETERMINED**, all 72. Needs Elliptic's construction definition; not obtainable from local data. | **No.** Exclusion is a safe default, correctly justified. |
| 2 | Address-level no-skill baselines for P3 | **UNMEASURED** — only transaction-level figures exist. | **No.** P3's form and threshold are frozen; baselines are computed during dataset construction (§6.5). |
| 3 | Whether PEEL-1's depth-43 chains are genuine structure or address reuse | **UNRESOLVED.** One hop per timestep is as consistent with a long-lived reused address as with peeling. | **No.** Carried as an interpretation caveat on every M2 result (§5.7). |
| 4 | Whether 1-labeled-address clusters belong in the cluster evaluation population | **OPEN DESIGN QUESTION.** 1,354 clusters (13.8%). Deliberately kept out of the label definition (§2.1). | **No.** A population choice, to be made and recorded before cluster metrics are computed. |
| 5 | Cost of PEEL-1's per-`t` recomputation | **UNMEASURED.** §0.8 computed chains once over all 49 timesteps; §5.5 requires truncation per `t`. | **No.** Worst case ≈ 49 × 2.4 s ≈ 2 minutes, and the DAG's incremental structure makes it far cheaper. Measure during implementation. |

`INFERENCE.` No item above has been converted from a blocker into a design
decision. The one blocker that existed — G2 — was measured, and the
pre-registered decision rule was applied to its result (§5.6).

---

## 9. Sign-off

Sections 1–6 must be signed off before implementation begins. §8.2 holds no
blockers, so no item there gates the freeze.

| section | decision | status |
|---|---|---|
| 1 | target + label policy | awaiting sign-off |
| 2 | address → cluster aggregation | awaiting sign-off |
| 2.1 | derived cluster label, `illicit_share >= 0.10` | awaiting sign-off — **new** |
| 3 | feature contract | awaiting sign-off |
| 3.3 | aggregate features excluded as UNDETERMINED | awaiting sign-off — **revised** |
| 4 | leakage rules | awaiting sign-off — unchanged |
| 5 | PEEL-1 detection, not classification | awaiting sign-off — **rewritten end to end** |
| 6.2 | metrics incl. cluster Recall@K | awaiting sign-off — **revised** |
| 6.3.1 | severity threshold procedure | awaiting sign-off — **new** |
| 6.5 | P1, P2, P3 | awaiting sign-off — **P1 and P3 revised** |

`DESIGN.` §5 was rewritten end to end rather than patched — substrate, hop
definition, value criterion, depth threshold and boundary note all changed —
so it should be read in full before it becomes binding.
