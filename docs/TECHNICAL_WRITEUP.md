# ObsidianChain — technical write-up

**SIH 2026 / NTRO** — AI-Powered Monitoring & Analysis of Bitcoin Transaction
Traffic. Offline prototype, Linux containers, no network at build or run time.

---

## 1. Approach

The system answers one question: **which entities should an investigator look
at first, and why?** It gets there in five stages, each producing a durable,
fingerprinted artifact rather than a number held in memory.

```
ingest (CSV/JSON/XML)  ->  correlate  ->  features  ->  model  ->  alerts
```

**Ingest.** `io/ingest.py` parses CSV, JSON and XML into one canonical record
(`timestamp, src_ip, dst_ip, src_port, dst_port, txid, input_addresses[],
output_addresses[], input_amounts[], output_amounts[], fee, script_type,
geo_country, asn`) and reports what was missing, coerced or rejected. Absent
fields are reported as absent; nothing is filled in.

**Correlate.** Network observations (`txid, observer_id, peer_ip, peer_port,
peer_asn, timestamp_ms`) join the blockchain layer on `txid`, and the
address↔transaction edgelists complete the chain:

```
IP  --announced-->  TRANSACTION  --involves-->  WALLET
```

Every edge is an observation. None is an ownership claim.

**Entity clustering.** Addresses are grouped by the common-input-ownership
heuristic — union-find over star edges per transaction (k−1 edges, not
k(k−1)/2). 569,513 clusters over 822,942 addresses; largest 14,885; coverage
34.60%.

**Features.** 43 features in four groups, all computed *as of* an address's
last active timestep from transactions at or before it:

| group | content |
|---|---|
| M0 | address behaviour: value sent/received, fees, fan-in/out, activity gaps |
| M1 | graph: degree, counterparties, clustering coefficient, cluster size as-of-t |
| M2 | PEEL-1 chain structure: depth, position, value retention |
| M3 | network: pooled observations, observer coverage, arrival dispersion |

**Model → alerts.** Address risk is aggregated to the cluster, banded by
severity, ranked, and published with its explanation and evidence.

---

## 2. Model choice

**LightGBM, gradient-boosted trees.** Chosen over a graph neural network
deliberately:

- The features are tabular and engineered, which is what boosted trees handle
  best.
- Every feature has a name an investigator can read. `output_fanout_mean` is
  explainable in a sentence; a learned embedding dimension is not, and the
  deliverable requires explaining *why* an entity was flagged.
- It trains in seconds, so the whole pipeline is reproducible on a laptop
  with no GPU and no network.

**Calibration.** Isotonic regression fitted on the **validation split only** —
never on training scores, which are optimistic by construction. This is what
makes the output a probability rather than an arbitrary score.

**Severity bands** are derived from validation precision targets (CRITICAL
≥ 0.90, HIGH ≥ 0.75, MEDIUM ≥ 0.50) by a deterministic rule: the lowest
threshold at which precision meets the target *for every higher threshold*,
with a minimum support of 50. A band that cannot meet its target is recorded
UNPOPULATED rather than filled by lowering the bar. The test split is never
consulted.

### Evaluation

Chronological split by first appearance — train t1–34, validation t35–41,
test t42–49 — with boundary-spanning addresses removed entirely, so the three
splits are address-disjoint. Labels are static per address, so an address in
both train and test would be memorised rather than predicted.

Ablation on the test split (262,433 labeled addresses, 4.63% prevalence):

| stage | PR-AUC | baseline | ROC-AUC |
|---|---|---|---|
| M0 | 0.2600 | 0.0463 | 0.7780 |
| + M1 graph | **0.3524** | 0.0463 | 0.8508 |
| + M2 peeling | 0.3514 | 0.0463 | 0.8534 |
| + M3 network | 0.3582 | 0.0463 | 0.8444 |

PR-AUC is the primary metric and is always quoted with its no-skill baseline;
accuracy is not reported, because at 4.63% prevalence it is not informative.

**Two negative results, reported rather than hidden.** Peeling structure moved
PR-AUC by −0.0010, and the network contribution (+0.0067 at depth 5, −0.0096
at depth 10) flips sign across the sensitivity run, so it is reported as *not
a reliable contribution*. Both were pre-registered predictions.

---

## 3. Explainability method

**TreeSHAP**, computed through LightGBM's own `pred_contrib`. Exact Shapley
values for tree ensembles — the same algorithm as the `shap` package, run
inside the booster so the offline build needs no extra dependency.

Contributions are reported **in log-odds**, and the UI says so. A contribution
of +3.32 is a shift in the model's raw margin, not 332 percentage points of
anything.

The design rule that matters more than the algorithm: **a model signal and a
ledger fact are never shown as the same kind of claim.** Every explanation row
carries two categories:

| category | meaning |
|---|---|
| `MODEL_SIGNAL` | a SHAP contribution — a statement about the model |
| `BLOCKCHAIN_CONTEXT` | an observed on-chain quantity — true of the ledger |
| `NETWORK_CONTEXT` | a summary of synthetic announcement observations |
| `INSUFFICIENT_EVIDENCE` | not computable for this address |

`INSUFFICIENT_EVIDENCE` is returned instead of a zero wherever a quantity
could not be established, because "not measured" and "measured as none" are
different answers and only one of them is honest.

---

## 4. What the system does not claim

These are enforced by tests, not convention.

- **An IP does not own a wallet.** A peer IP is where an *observer first heard
  an announcement*. 170,899 of 202,804 transactions (84.3%) were announced by
  more than one peer — that is gossip relay. A peer is a vantage point, never
  an originator or a sender.
- **Two addresses sharing network characteristics are not the same party.**
  The network layer emits cannot-link only: it can say two groups look
  *different*, never that they are the same. One Electrum server broadcasts
  for tens of thousands of unrelated users.
- **A cluster is not a person.** It is an inference from a heuristic.
- **PEEL-1 is not a laundering classifier.** It is label-blind structural
  candidate generation. Whether chain membership predicts illicitness is the
  question it exists to *measure*, and on this data the answer was no.
- **All network data is SYNTHETIC.** It demonstrates the mechanism and
  validates nothing about Bitcoin.
- **GeoIP returns no country for this dataset, correctly.** The synthetic
  addresses are RFC 5737 documentation ranges and the ASNs are RFC 6996
  private-use. A real database returns nothing for them; so does ours, and it
  says why. A vendored CIDR→country CSV at `data/reference/geoip_country.csv`
  resolves real captures offline with no code change.

---

## 5. Reproducibility

Every artifact carries a provenance sidecar and a composite run fingerprint
over the raw inputs plus the configuration that lives in code. The fingerprint
is written into the parquet footer as well as the sidecar, so a half-finished
regeneration is detectable rather than silently served. Alert ids are
`<run fingerprint>:<cluster id>`; an id from an earlier run returns **409**
rather than resolving against different rows.

```bash
make build                      # offline, from vendored wheels
make run ARGS="cospend"         # entity clustering
make run ARGS="evidence-funnel" # network evidence
make run ARGS="phase6-dataset"  # features
make run ARGS="phase6-experiment"
make run ARGS="phase7-alerts"   # alert artifacts
make serve                      # read-only API on localhost:8000
cd frontend && npm run dev      # dashboard on localhost:5173
```

1,175 backend tests and 46 frontend tests, no skips.
