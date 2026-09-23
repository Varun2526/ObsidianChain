# 16 — Synthetic World Dataset Investigation (2026-09-23, cycle 3)

**New research direction, separate from the Elliptic++-based PS-native
program (exp01–exp10).** Investigates whether the current dataset is
limiting this program's conclusions, and whether a purpose-built
synthetic dataset with blockchain + network-layer metadata should become a
separate controlled benchmark. **Nothing in this document changes, replaces,
or touches exp01–exp10's results, artifacts, or conclusions** — those remain
exactly as recorded (`00`–`15`, `v2_dev_candidate/`). This is additive, kept
under its own experiment IDs (exp11, exp12) and its own dataset paths,
clearly separated per the research mandate.

## The question, and why it's answerable now

`01_repo_audit.md` and `02_data_audit.md` already established (H2) that the
real PS-native development dataset has **zero network-layer columns** —
`include_network=False` at generation time — so Group F (network) and Group
G (cross-layer) have been structurally untestable since this program's
first audit. This cycle investigates whether that gap should be closed with
a purpose-built synthetic dataset.

**Discovery: the infrastructure to do exactly this already exists in the
repository**, and was not built by this research program —
`src/obsidianchain/world/` (`generate.py`, `behaviours.py`, `overlap.py`),
with a 120-entity instance already generated at `data/synthetic_world/`.
Its own module docstring states its purpose in almost exactly the terms of
this investigation's question:

> *"Elliptic++ gives a real chain layer with real labels and no network
> layer. `network/synthetic.py` gives a network layer whose announcements
> are attached to Elliptic txids after the fact. The two are consistent in
> their identifiers and in nothing else: no entity in the chain data
> **caused** an announcement, so a question like 'does a network-derived
> constraint prevent a false merge' cannot be asked of them, only asserted
> about them. This package generates both layers from ONE set of synthetic
> transactions... Chain structure and network structure share a cause, so a
> controlled experiment over them measures something."*

This was built for the Phase 2/3 **clustering / cannot-link** research
(HANDOFF.md), not for this program's supervised ranking question — but the
underlying data (chain layer in Elliptic++-compatible file format, network
layer via the same frozen `network/synthetic.py` generator, quarantined
`world_truth/` with entity/behavior ground truth) is directly reusable for
an ML ranking evaluation. This program did not need to build a dataset from
scratch — it needed to determine whether this existing one is fit for this
specific purpose. **It is not, as currently configured — with clear,
evidenced reasons why, below.**

## exp11: building the PS-native feature set on the synthetic world, and two audit findings

Reused the unmodified production `PsTemporalFeatureEngine` (same trusted
code path validated against real data in exp08) over
`data/synthetic_world/raw/*` (already on disk, Elliptic++-compatible
format — no regeneration needed). Two concrete findings surfaced before any
ranking comparison was run:

### Finding 1 (confirmed, not merely read from source): the production Group E feature code is a stub that discards real network structure

`pipeline/features_ps.py`'s `include_network` branch was already flagged in
`01_repo_audit.md` as thin; this experiment **confirms it empirically**,
not just by reading the source. The synthetic world's
`processed/network/observations.parquet` carries genuine multi-observer
structure per transaction (mean 5.67 distinct peer IPs, mean 5.61 distinct
ASNs per observed transaction — real variation). The current production
code, run over this exact data, produces:

| Column | Production stub's actual output | What the raw data actually supports |
|---|---|---|
| `observer_diversity` | **always 1.0** whenever any observation exists | 8.0 (constant in this world's config, but a real count, not a hardcoded stub) |
| `peer_count` | **always 1.0** | ranges 1–8, mean 5.67 |
| `asn_count` | **always 1.0** | ranges 1–8, mean 5.61 |

The code (`features_ps.py`, confirmed by direct reading and now by this
execution): `obs_div = 1.0 if row.get("observer_id") is not None else 1.0` —
this expression is **always 1.0 regardless of the condition**, a dead
branch. `peer_cnt = 1.0` is a literal constant. This is a genuine,
previously-undocumented production defect, structurally identical in kind
to the v1 fabricated-feature bugs `ml/diagnostics.py` already catalogues for
Groups A/D — Group E has the same class of problem and no test currently
catches it (`tests/test_ps_feature_correctness.py`/`test_ps_features_leakage.py`
do not exercise `include_network=True`). **New pipeline gap, added to
`12_pipeline_gap_analysis.md`'s list below.**

### Finding 2: the label is shape-adjacent to existing detectors, but not fully tautological at the single-feature level

The world's own labeling convention (`POSITIVE_CLASS = PEELING, MIXING_LIKE,
RAPID_MOVEMENT`, a stated "labelling convention for a controlled experiment,"
per the module's own docstring) is defined in terms of transaction shapes
that Groups A/D's detectors (`is_peeling_candidate`, `is_mixing_candidate`)
were built to catch. Single-feature ROC-AUC, checked directly before
trusting any full-model result:

| Feature | Single-feature ROC-AUC |
|---|---|
| `is_peeling_candidate` | 0.755 |
| `is_mixing_candidate` | 0.745 |
| `tx_velocity_per_hour` | 0.588 |
| `n_txs_asof_t` | 0.580 |

**Not tautological at the single-feature level** (0.75, not 0.99+) — no
single existing feature trivially solves the label on its own, because the
label is a three-way disjunction and no single feature covers all three
behaviors. This matters for correctly scoping the next finding.

## exp12: the full feature set saturates the task completely — the decisive finding

Running the full 24-column CORE feature set (Groups A–D, unmodified) through
the same rolling-fold protocol structure used throughout this program:

| Fold | n_eval | prevalence | nAP |
|---|---|---|---|
| t≤34→t35-36 | 376 | 0.340 | **1.0000** |
| t≤36→t37-38 | 38 | 0.632 | **1.0000** |
| t≤38→t39-40 | 46 | 0.696 | **1.0000** |

**Perfect ranking in every usable fold, for every feature variant tested**
(CORE-only, CORE+correctly-aggregated network, CORE+stub network — all
three identical, nAP=1.0000, sd=0.0000). This is not a floating-point
artifact: `nAP = (AP − prevalence)/(1 − prevalence)` equals exactly 1.0 if
and only if `AP = 1.0`, i.e. every true positive is scored strictly above
every true negative — an exact, verifiable condition, checked directly
against fold sizes large enough (376 rows in the first fold) that this
cannot be a small-sample fluke.

**Investigated as an anomaly before being trusted (per this program's own
discipline), using the quarantined `world_truth/transactions.csv` scenario
labels for POST-HOC DIAGNOSTIC READING ONLY** — never fed into any feature
or model:

- Is this driven by "easy" negatives (plain `NORMAL` transactions)
  dominating the negative class? **No** — `NORMAL` is a small minority of
  the negative population in the usable data (24 of ~944 rows); the
  adversarial behaviors (`EXCHANGE_BATCH`, `CONSOLIDATION`, `MERCHANT_SWEEP`,
  `UNIFORM_PAYOUT`, `BENIGN_EQUAL_SPLIT`) dominate the negative class, as
  the generator's own design intends.
- Does the model at least struggle on the behavior the generator's authors
  explicitly designed as the hardest adversarial case — `MERCHANT_SWEEP`,
  described in `generate.py`'s own comment as *"the shape most likely to be
  mistaken for a collaborative spend"*? **No.** Per-scenario score
  breakdown in every fold: `MERCHANT_SWEEP` scores clearly higher than every
  other negative (mean 0.041, max 0.140 in the first fold) — the model
  "notices" it is the hardest case — but still separates it from every true
  positive by a wide margin (min positive score 0.940 vs max negative score
  0.140 in that same fold — a ~7× gap, consistent across all three folds).

**Conclusion: this is not a bug or an artifact. It is a real, mechanistic
property of this world's current parameterization.** The generator
(`behaviours.py`) uses exact, fixed shape parameters
(`mixing_denomination: 0.1` precisely, `peel_fraction: 0.08` precisely, a
tight `UNIFORM_SPREAD = 0.02` tolerance) with no noise/overlap between
positive- and negative-behavior parameter distributions — deliberately, per
the module's own stated design goal ("the world must be inspectable by eye
before it is trusted at any scale"). A deterministic, low-noise generator
matched against detectors built to catch exactly those deterministic shapes
produces a task with **no room left for any additional feature group,
including network telemetry, to show incremental value** — not because
network evidence doesn't matter in principle, but because the ranking
metric is already saturated at its ceiling before Group E is even
considered.

## Does this support introducing a purpose-built synthetic benchmark? Evidenced answer: not this instance, as-is — but a specific, well-scoped future one, yes

**The existing `data/synthetic_world/` instance should NOT become this
program's Group F/G benchmark as currently configured.** Two independent,
compounding reasons, both evidenced directly rather than assumed:

1. **Ceiling effect** (exp12): the task is fully saturated by Groups A–D
   alone; no feature-group comparison can show anything on this data at
   this parameterization.
2. **Statistical power** (exp11): only 3 of the standard protocol's 12
   rolling folds are usable at this instance's current size (944 usable
   rows total, most transactions concentrated in a handful of early
   timesteps) — even if the ceiling effect were fixed, this instance
   couldn't support a properly-powered comparison under this program's own
   MDE/Holm-correction discipline.

**What a genuinely useful version would need — a concrete, evidenced
specification, not a vague "make it better":**

- **Noise/overlap in behavior parameters.** Positive and negative behavior
  shapes need distributions that overlap (e.g. a range of peel fractions
  and mixing denominations spanning both what current detectors catch and
  what they narrowly miss), not single fixed values — this is what would
  make Groups A–D's signal genuinely imperfect, leaving room to measure
  whether something else (network evidence, in particular) recovers what
  chain-only features miss.
- **Deliberately chain-ambiguous, network-resolvable cases.** None of the
  current 10 behaviors is designed so that chain-layer shape is
  *insufficient* to decide and network evidence is the deciding factor —
  every current behavior is fully decidable from the chain layer alone,
  by design (that was Phase 2/3's question: does network evidence
  *corroborate or veto* a chain-visible merge, not "can the chain layer see
  anything at all"). A benchmark for *this* program's Group F/G question
  would need at least one new behavior class explicitly constructed so
  that two entities look identical on-chain but are separable only via
  network origin (or vice versa: look different on-chain but are the same
  entity, correlated only via shared network origin) — genuinely new
  generator work, not a parameter tweak.
- **Larger scale** — enough entities/timesteps per split band to support
  the standard 12-fold, MDE-aware protocol this program uses everywhere
  else, so any resulting comparison could be held to the same evidentiary
  standard as exp01–exp10 rather than reported as descriptive-only.

**This program does not undertake that engineering in this cycle.** It is a
substantial, well-motivated new-behavior-design task (not a parameter bump),
and per RULE 12/RULE 2, this program does not build new infrastructure
merely because a question is open — only when evidence justifies the
investment and the scope is genuinely understood. Both conditions are now
met for *documenting the requirement*; neither is met yet for treating a
next-generation synthetic benchmark as a lightweight follow-up. **Recorded
as a concrete, scoped recommendation for a future cycle**, not built.

## What this investigation does NOT change

- `exp01`–`exp10`'s results, the tied model tier, the feature-group
  findings, the counterparty-history INCONCLUSIVE result, the alert-policy
  findings, and `v2_dev_candidate/` are **all unchanged** — this was a
  separate, additive investigation on a separate dataset, run under
  `SCOPE_UNDECLARED` specifically so `protocol.py`'s own scope machinery
  would refuse any accidental pooling with `SCOPE_PS_NATIVE` numbers.
- `data/synthetic_world/` itself was **not modified** — read-only access
  throughout (its own raw files and processed/network parquet), no
  regeneration, no new instance created. `src/obsidianchain/world/` and
  `src/obsidianchain/pipeline/features_ps.py` were **not edited** — the
  Group E stub bug is documented, not fixed, consistent with this program's
  standing practice of measuring rather than silently patching production
  code (see `10_explanation_audit.md`'s identical treatment of the
  explanation-method bug).

## Decision

- **REJECT**: using `data/synthetic_world/`'s current instance as a Group
  F/G benchmark — saturated (nAP=1.0 ceiling) and underpowered (3/12 folds),
  independently confirmed reasons.
- **CONFIRMED (new finding)**: production `features_ps.py`'s Group E code
  is a stub (constant `observer_diversity`/`peer_count` regardless of real
  variation) — a genuine defect, added to the pipeline gap list.
- **NEEDS_MORE_ENGINEERING, well-scoped**: a next-generation synthetic
  benchmark, if built, needs noisy/overlapping behavior parameters,
  at least one genuinely chain-ambiguous/network-resolvable behavior class,
  and enough scale for the standard 12-fold protocol. Not built this cycle.
- **UNCHANGED**: every prior finding in this program (`00`–`15`).
