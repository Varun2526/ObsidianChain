# 02 — Data Audit (2026-09-23)

Direct, executed checks against `data/models/ps_native/datasets/{train,validation}.parquet`
via `research/reproduction/build_ps_dataset.py` (the generator) and pandas
queries run live during this audit (not inferred from documentation).

## The unit of evaluation — clarified, and it is not what the feature engine's own docstring implies

`pipeline/features_ps.py`'s module docstring states: *"Primary prediction
unit is ADDRESS-AS-OF-TIMESTAMP"* — read naturally, this suggests the
dataset carries one row per (address, transaction) pair, since
`PsTemporalFeatureEngine.process_records` does emit exactly that (confirmed
by reading `process_records`: it appends one `feature_dict` per participating
address per transaction).

**But the training/evaluation datasets do not use that grain.**
`build_ps_dataset.py` (lines ~163-183) does:

```python
features_last = features.sort_values(by=["timestamp"]).groupby("address").last().reset_index()
```

— i.e. it collapses every address down to **exactly one row: its LAST
observed as-of-t feature snapshot**, then assigns that one row to a split by
the address's FIRST-seen timestep. Confirmed directly: `train.parquet` has
170,404 rows and 170,404 distinct addresses (checked live —
`train.duplicated(subset=['address','txid']).sum() == 0`, and separately
`len(set(train.address)) == len(train)`).

**Consequence for how every metric in this program must be read:** nAP,
precision@k etc. all rank **addresses**, each represented once, by its
*final* known state within its split window — not transaction-events, and
not an address's full trajectory. A model "ranking suspicious activity" in
this dataset is really ranking *"how suspicious did this address look the
last time it was observed inside this window,"* which is a reasonable
investigative framing (an analyst wants a ranked address worklist, not a
per-transaction stream) but is a materially different claim than
per-transaction risk scoring, and should be stated as such in any final
report (Phase 17 must not conflate the two).

## Boundary-spanning addresses are dropped entirely — a real selection effect, quantified

`build_ps_dataset.py` computes each address's first and last active
timestep and **excludes any address whose activity spans a split
boundary**:

```python
spans_train_val = (addr_first_step <= TRAIN_END) & (addr_last_step > TRAIN_END)
spans_val_test  = (addr_first_step <= VALIDATION_END) & (addr_last_step > VALIDATION_END)
split_series[spans_train_val | spans_val_test] = pd.NA   # dropped from `usable`
```

This is confirmed to be exactly why **address overlap between
`train.parquet` and `validation.parquet` is 0%** — checked directly
(`len(set(train.address) & set(val.address)) == 0`). It is a deliberate
anti-leakage design decision (an address active on both sides of a split
cannot contribute a "before" feature row to one split and a labeled outcome
to another), not a bug — but it is a **selection effect that has never been
quantified in this project's documentation**, and it matters for external
validity: addresses that persist across a 14-day-step boundary are, by
construction, the more temporally *active* ones, and they are structurally
absent from both training and evaluation. Whether short-lived,
single-window addresses are representative of the addresses an investigator
actually cares about (which often includes long-lived, recidivist activity)
is an open question this program flags but does not resolve — see
`12_pipeline_gap_analysis.md`.

**Not yet quantified in this audit:** the exact count/fraction of addresses
dropped as spanners (would require re-running `build_ps_dataset.py`'s
intermediate `features` frame, which is not persisted to disk — only
`features_last` post-filter is saved). Flagged as a follow-up measurement,
not fabricated here (RULE 14/15).

## Class balance

| split | rows (=addresses) | y=1 (illicit) | prevalence |
|---|---|---|---|
| train | 170,404 | 9,328 | 5.47% |
| validation | 37,694 | 2,358 | 6.26% |
| test (sealed, not read) | 54,335 | 2,518 | 4.63% *(from dataset manifest only — not opened by this audit)* |

Per-timestep prevalence (measured live, train+validation, `first_t` 1–41)
swings from **0.41% (t1) to 30.3% (t26)** — a ~74× range across single
timesteps, and confirms at finer grain what `protocol.py`'s own docstring
already states about window-level prevalence swings (3.3%–21.0% *per fold*,
which pools multiple timesteps). This is the direct mechanistic reason
`normalised_average_precision` (rather than raw AP or accuracy) is the
correct primary metric — see `00_research_protocol.md`.

## Duplicate / integrity checks (all pass, checked live)

- Duplicate `(address, txid)` rows within train: **0 of 170,404**.
- `txid` overlap between train and validation: **0** (disjoint time
  windows, as expected — train is step ≤34, validation is step 35–41, and
  a txid belongs to exactly one step).
- Label consistency: **0 addresses** have more than one distinct `y` value
  across their (pre-collapse) observation rows — labels are a stable
  per-address property in this dataset (expected: Elliptic++ wallet class
  is an entity attribute, not a per-transaction one), so the
  last-snapshot-per-address collapse cannot accidentally mix labels.

## Train/test (holdout) entity overlap — NOT checked in this audit

Checking address overlap against `test.parquet` would require reading the
sealed holdout, which this program's protocol (`00_research_protocol.md`)
explicitly forbids before Phase 16. The `spans_val_test` filter in
`build_ps_dataset.py` gives structural confidence that no address is
double-counted across validation and test (any spanner is dropped from
both), but this has not been, and will not be, directly verified against
`test.parquet` contents until the one sanctioned Phase 16 read.

## Super-cluster contamination / artificial correlations

Not applicable to the PS-native scope in the way it applies to the
co-spend-clustering engine (`src/obsidianchain/cluster/`) — PS-native's
`cluster_size_asof_t` feature comes from `IncrementalUnionFind`, a **separate,
transaction-local co-spend replay scoped to the frame being processed**, not
the global 569,513-cluster structure from Phase 1. It has not been checked
for its own contamination properties (e.g. whether `cluster_size_asof_t`
itself correlates with label in a way that's really tracking dataset
construction order rather than genuine graph structure) — flagged as a
candidate check for Phase 4 (feature research) rather than answered here.

## Duplicate entities across the broader project (Elliptic++ vs synthetic worlds)

Out of scope for this dataset: PS-native training data is Elliptic++-only,
does not join any `network/` synthetic world table. No cross-dataset entity
duplication risk to check here.
