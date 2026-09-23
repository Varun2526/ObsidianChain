# 03 — Label Audit (2026-09-23)

## Label construction

Source: Elliptic++ `wallets_classes.csv`. Three classes: 1 = illicit,
2 = licit, 3 = unknown. `build_ps_dataset.py`:

```python
features_last["y"] = np.where(
    features_last["class"] == 1, 1,
    np.where(features_last["class"] == 2, 0, np.nan)
)
```

Class-3 (unknown) rows are **excluded from the usable frame entirely**
(`usable = features_last[... & features_last["y"].notna()]`) — not treated
as negatives, not imputed. This is correct practice (an unlabeled entity is
not evidence of licit behavior) but means the observed prevalence (5.47%
train / 6.26% validation) is prevalence **among labeled addresses only**,
not among all addresses in the raw graph. The true population prevalence
(including unknowns) is unmeasured here and out of this project's control —
it is an Elliptic++ dataset property, not something this program can fix.

## Address-level vs transaction-level labeling

Elliptic++ labels are an **entity attribute** (a wallet is licit/illicit as
a whole), joined onto the address's single collapsed row
(`features_last.merge(labels, on="address", how="left")`) — confirmed live
that no address carries more than one distinct `y` across its raw
observation rows (§02_data_audit.md). There is no *temporal* label
(no "this address became illicit at time t") — a label is static for the
address's entire lifetime in this dataset. This means a model trained here
is learning "does this address's *most recent* as-of-t behavioral snapshot
look like a wallet that is *ever* labeled illicit," not "did this address
just start behaving illicitly." That distinction matters for how a
production alert should be worded (Phase 8/17), and is not currently
surfaced anywhere in the pipeline's severity/alert language — flagged for
`12_pipeline_gap_analysis.md`.

## Temporal label behavior

Not applicable in the usual "label drift over time" sense — see above, labels
are static per address. The temporal dimension that *is* real here is which
**feature snapshot** (as-of which transaction) gets used to represent the
address, which is always its last pre-split-boundary snapshot
(§02_data_audit.md).

## Synthetic label generation

None. PS-native labels are real Elliptic++ ground truth, not synthetic.
(Contrast with the network-layer / demo-scenario data elsewhere in the repo,
which *is* synthetic — see `01_repo_audit.md` §10.)

## Duplicate entities / duplicate transactions

Checked live in `02_data_audit.md`: 0 duplicate `(address, txid)` rows, 0
addresses with inconsistent labels. No further duplicate-entity risk
identified for this scope.

## Train/test entity and transaction overlap

Structural guarantee via the boundary-spanner filter (§02); the sealed
holdout itself not directly inspected, per protocol.

## Temporal leakage / future information leakage

This is the one area with **executable, passing tests** rather than just
documentation: `tests/test_ps_features_leakage.py` (`TestTemporalLeakageSafeguards`,
parts A–E) constructs adversarial cases — a "future" transaction that
should not be able to alter an earlier feature row — and asserts the
invariant holds. This audit did not re-run `pytest tests/test_ps_features_leakage.py`
as a fresh execution (flagged as a fast, cheap follow-up rather than
assumed); the code path itself (`PsTemporalFeatureEngine.process_records`
sorts strictly by timestamp before iterating, and updates `AddressState`
only *after* emitting that transaction's feature row) is consistent with
the tests' claims on direct reading.

## Graph leakage

`cluster_size_asof_t` comes from `IncrementalUnionFind`, updated only from
`in_addrs[1:]` co-spend unions **processed in the same strictly-increasing
timestamp order** as everything else in `process_records` — no evidence of
forward leakage on direct reading. Not independently stress-tested by this
audit beyond the existing `test_c_future_cluster_expansion_does_not_change_earlier_cluster_size`
test referenced in `01_repo_audit.md`.

## Target leakage

None of the 24 v2 CORE feature columns derive from `y`/`class` — confirmed
by column-name inspection (`pipeline/features_ps.py` never references
`wallets_classes` or `class` inside `PsTemporalFeatureEngine`; labels are
joined in a wholly separate step in `build_ps_dataset.py`, *after*
`features_last` is finalized). This matches the P0 constraint stated in the
feature engine's own docstring ("Label-blind... labels... NEVER enter the
feature matrix").

## Feature availability at time t

By construction, every Group A/B/C/D feature is derived from state at or
before the row's own transaction (§ above and `01_repo_audit.md` §9). Group
E (network) is **absent from the dataset**, so "availability at time t" is
moot for it here — it simply isn't in the frame (see `01_repo_audit.md` §3,
§9).

## Label noise

Not independently measurable from inside this project — Elliptic++'s
labeling process is external ground truth. No noise-injection or
label-quality stress test was run in this audit; flagged as a candidate
Phase 7 (error analysis) technique (e.g. checking whether high-confidence
false positives cluster around addresses with only 1-2 total transactions,
where "wallet type" is inherently hard to infer) rather than a data-audit
item to resolve here.

## Quantified effect of the one serious issue found

The **boundary-spanning-address exclusion** (§02) is the one construction
choice in this audit that plausibly has a measurable effect on results and
has never been quantified. Recommended as the first Phase-3
"run a controlled test" item once the model-family and ablation experiments
(Phase 5/6) are further along: rebuild a variant dataset that does **not**
drop spanning addresses (assign them to their *first*-seen split, same as
non-spanners, instead of dropping) and re-run the protocol to see whether
nAP or the fold-to-fold spread changes materially. Not run in this audit —
it requires a new dataset-generation variant, which is exactly the kind of
"single change, protocol-locked before running" experiment this program's
inner loop is for, not something to improvise inline. Logged as a candidate
hypothesis for the next inner-loop cycle.
