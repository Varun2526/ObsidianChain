# ObsidianChain — Handoff

**SIH 2026 / NTRO PS 26146** — "AI-Powered Monitoring & Analysis of Bitcoin
Transaction Traffic". Offline Bitcoin forensics prototype.

Read this top to bottom before changing anything. The **Invariants** and
**Traps** sections exist because each item in them cost a wrong result that
looked correct.

---

## 0. Current state (2026-09-23)

Sections 1-11 below describe Phases 1-4 and remain accurate for that work.
Phases 5-11 were built afterwards and are summarised here.

| Layer | Where | State |
|---|---|---|
| Analytical API | `api/` | read-only over frozen artifacts; every route needs a session except `POST /auth/login`, `POST /auth/logout`, `GET /health` |
| Application layer | `console/` | SQLite: users, sessions, RBAC, investigations, datasets, AnalysisRuns, alert references, dispositions, notes, reports, audit |
| Frontend | `frontend/` | React/TS console; no identity or case state in browser storage |
| Synthetic world | `world/` | coherent TXID-correlated chain+network world, `SYNTHETIC_CONTROL` |
| ML — Phase 6 | `features/`, `ml/model.py` | LightGBM + isotonic + TreeSHAP over M0-M4; serves alert run `043ea584e99daf99` |
| ML — PS-native | `pipeline/`, `ml/ps_model.py` | RandomForest; executed by the orchestrator for uploaded datasets |
| **Evaluation protocol** | **`ml/protocol.py`** | **the only sanctioned way to compare models** |

### 2026-09-23 (latest): default model is ps_native_v3, schema /4

v3 supersedes v2 below: causal within-step order (the old txId order leaked
and inflated v2 by ~0.06 nAP), group G upstream-flow features in CORE, 12-fold
nAP 0.713. Rebuild: `build_ps_dataset.py`, `train_ps_model_v2.py` (writes
v3), `train_ps_stacker_v2.py`. See ADR 0002 amendment.

### 2026-09-23 (later): PS-native v2 shipped — read ADR 0002

The "PS-native inference is blocked" note below is RESOLVED. Summary
(`docs/decisions/0002-ps-native-v2-architecture.md`,
`research/autoresearch_2026_09_23/18_architecture_upgrade.md`):

- Feature schema `ps_native_features/3`; train/validation rebuilt. The holdout
  is still `/1`, sealed, MD5 `a15500c94b9808cd42d584ad4b5c3017`, unopened.
- `data/models/ps_native/v2/`: LightGBM, Platt, TreeSHAP explanations, rank
  budget severity, drift reference, `stacker.json`. `holdout_evaluated: false`.
  Rebuild: `python research/reproduction/train_ps_model_v2.py` then
  `python research/reproduction/train_ps_stacker_v2.py`.
- New: `ml/propagation.py` + `io/watchlist.py` (OFAC SDN / watchlist seeds),
  `ml/embeddings.py` (link suggestions only), `ml/stacking.py`,
  `ml/monitoring.py` (`monitoring.json` per run), `world/noisy.py`
  (SYNTHETIC_CONTROL world v2, output under `data/synthetic_world_v2/`).
- Fusion is noisy-OR with budget severity; the alert build is linear, not
  O(clusters x transactions).
- Uploaded-run alerts (`alerts.json`) are still not shown in the frontend;
  the console shows run progress only.

### Two production model paths — read ADR 0001 first

`docs/decisions/0001-two-production-model-paths.md`. Both are production with
declared, non-overlapping scopes: Phase 6 owns the Elliptic++ reference
artifacts, PS-native owns uploaded-dataset runs. **Numbers are never compared
across them** - `protocol.compare()` raises `ScopeMismatchError`.

### The sealed holdout

Timesteps **42-49 are sealed**. `protocol.development()` is the only accessor
a model developer uses; `protocol.holdout()` raises unless `break_seal(reason)`
is called with a written reason. A `Fold` whose evaluation window reaches t>=42
**cannot be constructed**. The builder refuses to regenerate `test.parquet`
without `OBSIDIANCHAIN_REGENERATE_HOLDOUT=1`.

Current holdout MD5: `a15500c94b9808cd42d584ad4b5c3017`, schema
`ps_native_features/1`. Development parquets are `/2`. That asymmetry is
correct and deliberate.

### PENDING: PS-native inference is blocked

The generator fix moved the pipeline to `ps_native_features/2` (24 columns).
The frozen `ps_native_v1` model declares `/1` (30 columns), so `ps_model.py`
refuses rather than scoring misaligned columns. The orchestrator degrades
honestly - all 17 stages complete, `ml_status = MODEL_UNAVAILABLE_FOR_SCHEMA`,
no crash - but **uploaded-dataset runs get no risk score until a v2 model is
trained under the protocol**. This is the known cost of fixing the data
without retraining.

### Installed skills (checked 2026-09-23)

Present: the **superpowers** pack (`using-superpowers`, `brainstorming`,
`writing-plans`, `test-driven-development`, `systematic-debugging`,
`verification-before-completion`, `subagent-driven-development`,
`executing-plans`, `requesting-code-review`, `receiving-code-review`,
`dispatching-parallel-agents`, `using-git-worktrees`,
`finishing-a-development-branch`, `writing-skills`, `skill-creator`), plus
`llm-council`, `docs`, `xlsx`, `pptx`, `pdf`, `docx`, `webapp-testing`,
`langsmith-fetch`, `changelog-generator`, `connect`/`connect-apps` and
assorted unrelated utility skills.

**NOT present:** Probabl, scikit-learn, statistical-analysis, statsmodels,
SHAP, Orchestra Autoresearch. The ML audits were done without them.
**Libraries:** scikit-learn 1.9.0 and scipy 1.17.1 ARE installed;
**statsmodels and shap are not, and are not in the vendored wheel set** -
adding either means re-vendoring and touching the offline build guarantee.

---

## 1. The idea

Co-spend clustering merges Bitcoin addresses that appear as inputs to the same
transaction (common-input-ownership). A single wrong merge is permanent and
propagates, producing super-clusters. The thesis: **network-layer evidence can
veto a merge before it happens** — "these two groups were not broadcast from
the same machine, do not merge them."

Two non-negotiable rules:

- The network layer emits **CANNOT-LINK only**, never must-link. One Electrum
  server broadcasts for tens of thousands of users, so co-origination is not
  co-ownership.
- A constraint fires only on **repeated, consistent separation**, never a
  single observation. This is structural, not a threshold: evidence is pooled
  per component and a minimum pooled count gates every decision.

---

## 2. Environment

Fully offline. Linux containers, `linux/amd64` primary (Windows teammates via
WSL2), `linux/arm64` supported for Apple-silicon dev.

```bash
make vendor      # ONCE per architecture, online. Wheels + libgomp1 .deb
make build       # docker build --network none
make test        # pytest inside the container, --network none
make help        # every target
```

Vendored wheels live in `vendor/linux-amd64/` and `vendor/linux-arm64/`,
gitignored. `make build` fails loudly if they are missing or the wrong arch.

**The offline guarantee is narrow and must be stated that way:** *once the
`python:3.11-slim` base image and vendored artifacts are present locally, the
application image builds and runs with Docker networking disabled.* It is NOT
"air-gapped from a fresh machine".

**566 tests pass.** Run `make test` before and after any change.

---

## 3. Where things are

```
src/obsidianchain/
  io/elliptic.py          Elliptic++ loader, star co-spend edge derivation
  io/entity_labels.py     addresses.csv normalisation
  cluster/unionfind.py    union-find (find/union written by Varun)
  cluster/change.py       change-address detection + scoring
  cluster/constrained.py  CANNOT-LINK union-find (union written by Varun)
  cluster/pipeline.py     run_clustering / run_fused
  network/synthetic.py    SYNTHETIC announcement generator, FROZEN config
  network/arrivals.py     arrival-vector instrumentation
  network/boundary.py     THE information boundary. Only sanctioned load path
  network/separation.py   group-level separation evidence (cannot-link only)
  network/worlds.py       controlled worlds A–E
  network/reach_stress.py reach-stress fixture (own dense synthetic chain)
  network/audit.py        Phase 2 audit
  eval/purity.py          licit/illicit contamination
  eval/entity_resolution.py  entity-level scoring
  eval/evidence_funnel.py    where evidence disappears
  eval/phase33.py         Phase 3.3 experiment, DECISION-level scoring
  eval/world_diagnostics.py  regime verification
  eval/compare.py, eval/evolution.py
  demo/scenarios.py       five DEMO fixtures, seeded. Namespace guard
  demo/audit.py           post-hoc leave-one-out contradiction audit
  demo/runner.py          runs the unchanged engine, asserts each scenario
  demo/api.py             DEMO-flagged JSON envelope + flag enforcement
  demo/report.py          terminal report + self-contained HTML page

  --- Phases 5-11, added after this section was first written ---
  api/                    read-only HTTP layer over frozen artifacts
    app.py                routes + PUBLIC_ROUTES (the whole unauth surface)
    alerts.py evidence.py separation.py patterns.py evaluation.py
    boundary.py           modules this package may NOT import
    provenance_gate.py    refuses a non-PRODUCTION or torn artifact
  console/                MUTABLE investigator state (SQLite)
    db.py rbac.py users.py sessions.py investigations.py
    datasets.py runs.py casework.py reports.py audit.py
    routes_auth.py routes_investigations.py routes_casework.py
  features/               M0-M4 feature groups
    incidence.py behaviour.py graph.py peel.py netfeat.py
    mixing.py             M4 - mixing/CoinJoin-like structure
    dataset.py            FEATURE_GROUPS (frozen) + OPTIONAL (M4)
  ml/
    model.py              LightGBM + isotonic + severity bands
    experiment.py         M0-M3 ablation, P1-P4
    metrics.py            nAP, precision@k
    protocol.py           THE canonical evaluation protocol + holdout seal
    diagnostics.py        constant / duplicate / dependent feature detection
    ps_model.py anomaly.py
  pipeline/               PS-native path (orchestrator, 17 stages)
    features_ps.py        ps_native_features/2 schema
    orchestrator.py blockchain.py patterns.py alerts.py
  world/                  coherent TXID-correlated synthetic world
    generate.py behaviours.py overlap.py

frontend/                 React/TS investigation console
docs/decisions/           ADRs. 0001 = the two production model paths
research/
  audit_2026_09_22/       first ML audit (report + a1-a13 scripts)
  protocol_2026_09_22/    protocol runs, decision-rule validation, power design
  reproduction/           build_ps_dataset.py, train_ps_model.py

data/models/ps_native/
  v1/                     FROZEN model. Declares ps_native_features/1
  datasets/train.parquet        /2, regenerated
  datasets/validation.parquet   /2, regenerated
  datasets/test.parquet         /1, SEALED — do not regenerate
data/obsidianchain.sqlite3      console state (gitignored)
data/uploads/                   content-addressed dataset storage (gitignored)
data/synthetic_world/           SYNTHETIC_CONTROL world (gitignored)
data/processed/tx_mixing.parquet  additive scan, no run fingerprint

data/demo/                       DEMONSTRATION namespace (gitignored)
  raw/ processed/                its own chain + observations
  output/scenarios.json          DEMO-flagged payload
  output/index.html              the page a judge looks at
data/raw/                        Elliptic++ (gitignored)
  address_labels/addresses.csv   entity labels
  bitcoinheist/                  deferred
  ofac_sdn/sdn_xml.zip           deferred
data/processed/network/          Phase-3-visible observations
data/processed/network_truth/    QUARANTINED ground truth
data/processed/worlds/{A..E}/    controlled worlds
data/processed/worlds_truth/     QUARANTINED
data/reach_stress/               self-contained fixture (own raw/ + processed/)
```

Git: branch **`main`**. The `phase-3.3-checkpoint` note above is stale.

---

## 4. FROZEN results — these must not change

If any of these move, something broke. Check before and after every change.

### Phase 1 — co-spend baseline
```
input rows 477,117 · transactions 202,804 · star edges 274,313
co-spend merges 253,429 · clusters 569,513
LARGEST CLUSTER 14,885 · coverage 34.60% · peak 444.5 MB
```
`make run ARGS="cospend"`

### Phase 1.1 — purity vs licit/illicit
68 contaminated clusters, observed/expected **0.042** (~24× purer than a
shuffled-label null). Largest cluster is contaminated (6 illicit / 10,995
licit).

### Phase 1.2 — change detection
`--heuristics multi-input+change`: 562,894 clusters, largest 14,921,
coverage 35.68%, **6,619 change merges**.
Reachable confidence bands only: 0 / 0.15 / 0.35 / 0.65 / 0.85 / 0.98.
`self_reference` yields **zero** new merges (those addresses are already inputs).

### Phase 1.3 — temporal evolution
Onset t=30, local exponent 3.42, **global exponent 0.42 — step-like, NOT
runaway collapse**. Largest single jump +3,706 at t=36.

### Phase 1.5 — entity resolution
macro F1 **0.1348** (headline) · micro precision **0.8269** · micro F1 0.3484
TP 559 · false merges 117 · false splits 1,974
recall raw 0.2207 / ceiling 0.3269 / conditional 0.6751
Baseline 21–30× the shuffled null, p = 0.0010.
Change detection: **identical to 4 decimals** — it changes nothing here.

### Phase 2 — synthetic network
Frozen dataset hash **`c405493d5bd904c0e17c840dca79386a`**, 1,589,863 records,
σ = **1.1506** (Decker & Wattenhofer 2013).
Per-transaction separability **1.003** (four metrics agree — no single-vector
signal). Per-origin centroids **6.563 SE** apart — signal is aggregate only.

### Phase 3.1 — evidence funnel
```
proposed unions 253,429
  >=1 pooled  8,415   >=2  1,515   >=5  348   >=10  131   >=25  40
separated share 41.5% -> 7.2% -> 0%   FALLS = small-sample noise
median min(A,B) pooled = 0
```

### Phase 3.3 — normal worlds A–E
**Zero blocked merges in every regime.** Reach, not the rule, was the limit.

### Phase 3.3 — reach-stress, decision level (LATEST)
```
regime  unique  decidable  NO_EVID  SEPARATED  NOT_SEP   blocked decisions
A        5,700        474    5,226          0      474   0
B        5,700        461    5,239          1      460   1 MIXED
C        5,700        626    5,074          1      625   1 MIXED
D        5,700        604    5,096          2      602   1 PURE_SAME + 1 MIXED
E        5,700        469    5,231          0      469   0
```
**D produced 1 false split. Precision 0% on the only binary-scorable decision.
Zero PURE_CROSS_ENTITY blocks across 28,500 decisions in five regimes.**

---

## 5. THE headline finding

> When network evidence was strong enough to act, it distinguished **network
> behaviour** without ever distinguishing **entity identity**.

The engine never once correctly separated two genuinely different entities,
including in regimes B and E where entity→origin affinity was injected and is
detectable in aggregate (2.41 and 2.16 SE). The one scorable action it took
was wrong: it cut entity 13 across its own three origins.

This is the architectural warning to carry forward. A cannot-link rule built
on origin dissimilarity **cannot express** that one entity may legitimately
broadcast from several origins.

---

## 5.5 The demonstration — `make demo`

Five deterministic scenarios that show the mechanism working end to end,
built as **fixtures** and marked as such at every level. They are not results
and no number in them says anything about Bitcoin.

```
A  blockchain evidence proposes a merge                -> CANDIDATE
B  network evidence present but insufficient           -> merge proceeds
C  network evidence strong and separating              -> merge BLOCKED
D  contradiction appears after transitive clustering   -> CONTESTED
E  no usable network evidence                          -> abstain
```

One chain, five disjoint sub-graphs, one engine run. Evidence is carried by
single-input transactions (one announcement, no co-spend edge); merges are
proposed by separate multi-input transactions, so each address's pooled count
is set exactly without tangling structure and volume.

**The threshold is not lowered.** `SeparationConfig()` is the production rule
untouched — 25 pooled per side, alpha 1e-4, effect floor 0.05 — and
`runner.assert_production_rule` fails the run if it has been edited. What the
fixture raises is the evidence: 320 usable announcements per side, because
that is what a genuine origin difference needs to clear a 1e-4 gate at
sigma 1.1506. Measured, not guessed: the same two origins return
NOT_SEPARATED at 30 and at 60 pooled.

Observed output:

```
     outcome    verdict        pooled     p          effect
  A  CANDIDATE  NO_EVIDENCE    0/0        1.0e+00    0.0000
  B  MERGED     NOT_SEPARATED  321/320    3.0e-01    0.0281
  C  BLOCKED    SEPARATED      321/320    4.5e-11    0.0768
  D  CONTESTED  SEPARATED      321/256    6.3e-11    0.0804
  E  ABSTAINED  NO_EVIDENCE    0/0        1.0e+00    0.0000

clusters chain-only 5 -> fused 6   blocked 1   contested 1
decisions evaluated 4   abstained 30
```

**B and C are identical except for the origin** — same pooled counts, same
volume, opposite decisions. That contrast is the demonstration; the veto
tracks the origin difference, not the amount of data.

**A and E are different kinds of silence.** A's transactions have zero
announcement records (the capture never saw them); E's have 320 per side and
every one is discarded as `known_broadcaster`. Same engine verdict
`NO_EVIDENCE`, different operational meaning — one is a coverage problem, the
other is not.

**D is the blind spot, made visible.** Every merge decision has a side below
the pooled minimum, so all thirteen abstain and the component forms. Only
once it is complete do two comparable pools exist, and by then union-find has
no split. `demo/audit.py` finds it with a leave-one-out comparison — each
member with enough evidence of its own against the rest of its component,
same rule, both gates, all observable — and feeds it back through the
engine's existing `add_cannot_link` / `audit_existing_constraints` path. It is
a demo-side audit, not part of the production engine, and it cannot see a
contradiction between two thin halves. Say that out loud.

### Honesty contract, enforced by tests

- **Namespace.** Everything lives under `data/demo/`.
  `scenarios.assert_demo_namespace` refuses any root whose final component is
  not `demo`, or that sits inside a production directory.
- **Frozen production untouched.**
  `test_demo_writes_nothing_into_the_production_namespace` hashes every file
  under `processed/` before and after a run and asserts they are identical.
  Checked on the real dataset too: hash still `c405493d…`, Phase 1 still
  14,885 / 34.60%.
- **The DEMO flag is structural.** Every object in the payload carries
  `demo: true` and `provenance: SYNTHETIC_DEMONSTRATION`;
  `api.assert_demo_flagged` walks the whole tree and both renderers call it
  before emitting a character. An unflagged payload does not render at all,
  so the marking cannot be dropped by editing a template.
- **Both statements in one envelope.** `statements.mechanism` and
  `statements.frozen_dataset` are fields, side by side and in the same visual
  weight on the page. Neither renders without the other.
- **The demo asserts its own claims.** Each `ScenarioSpec` declares the
  outcome and verdict it expects; the runner raises `DemoExpectationError` if
  the engine stops producing it. A demonstration that silently shows the
  wrong thing is worse than one that fails.

---

## 5.6 Phase 4.1 — additive safety fixes (no methodology change)

Four fixes from `docs/archive/PHASE4_ARCHITECTURE_AUDIT.md`. Nothing scientific moved:
Phase 1, Phase 3.1, Phase 3.3 and Phase 3.4 all reproduce exactly, and the
frozen dataset hash is unchanged. E1/E3 are NOT implemented.

**1. Provenance on durable outputs** (`src/obsidianchain/provenance.py`).
Every CSV/Parquet written by an evaluation, funnel, phase33 or demo run now
carries a `provenance_type` column in **every row** plus a sibling
`<name>.meta.json` with dataset id and hash, world, generator version, the
production rule that was in force, and the git revision when discoverable.
Three types, because `DEMO` alone is wrong — controlled worlds are synthetic
but are not demo scenarios:

```
PRODUCTION         production pipeline on the frozen dataset
SYNTHETIC_CONTROL  controlled worlds A-E, reach-stress
DEMO               the five demonstration scenarios
```

`synthetic_network` is a second row column, added only where the numbers came
from generated announcements. It answers a different question from
`provenance_type`, so it is not duplicated metadata: a `PRODUCTION` funnel row
is still `synthetic_network=true`, and `is_measurement` in the sidecar is
derived from both rather than settable.

**The frozen `observations.parquet` is deliberately NOT stamped.** Adding a
column would change `c405493d…` and break invariant 8. Its `manifest.json`
sibling already carries its provenance.

**2. Compatibility guard moved into the production fusion path**
(`separation.build_oracle`). It was only ever called from `eval/phase33.py`,
leaving `run --mode fused`, `fusion-summary` and `evidence-funnel` unguarded —
the three commands most likely to be pointed at a real capture with the wrong
chain. It now runs inside `build_oracle` on the frames already loaded (no
second file read), and the duplicate call in `run_regime` is gone.
`DatasetMismatchError` / `MIN_TXID_OVERLAP` / `assert_datasets_compatible`
moved to `network/separation.py` and are re-exported from `phase33` under the
same names.

**3. One pooled-evidence implementation** (`src/obsidianchain/cluster/replay.py`).
`replay_unions()` is the single public walker; funnel, phase33 and demo all use
it. Pooled aggregation now lives only in `ConstrainedUnionFind`; the funnel's
private stats dict over a plain `UnionFind` is gone. `_evaluate_cannot_link` is
now public `evaluate_cannot_link` and nothing reaches into a private method.

Verified **bit-identical on all 253,429 x 13 funnel values** on the real
dataset, plus golden-master tests in `tests/test_replay.py` that hold verbatim
copies of both replaced loops.

**Two behaviours preserved on purpose, because they look like bugs:**
- *Double evaluation.* The walker records via `evaluate_cannot_link` and then
  `union` evaluates again, so `evaluated`/`abstained` count each decision
  twice. The demo's published "evaluated 4 / abstained 30" over 17 decisions
  IS that doubling. Fixing it would silently change a reported figure — own
  phase.
- *Boundary de-duplication.* One boundary re-proposed by many edges is one
  decision. This is the 166-false-splits trap.

**4. Enum-derived state export.** `phase33.to_frame` iterated the literal
`("NO_EVIDENCE","SEPARATED","NOT_SEPARATED")`, so a fourth state could exist
in code and vanish from `phase33.csv` with no error. It now iterates `Verdict`.

> **One deliberate side effect:** the `state_*` column ORDER in `phase33.csv`
> now follows the enum (`SEPARATED, NOT_SEPARATED, NO_EVIDENCE`) instead of
> the old literal order. Every value is identical; only the order moved.

**What "B3 remaining" does and does not mean.** The B3 *defect* was one
thing: a state present in the enum vanishing from an export with no error.
That is fixed and only ever had one site. What remains are six **semantic
branch points** — places that are correct for the three-state space and that
need a decision about what INCONCLUSIVE *means* before they can change:

```
cluster/constrained.py:182        union()            does INCONCLUSIVE merge?
cluster/constrained.py:404-415    note_evidence()    abstention or evaluated?
eval/phase33.py:404               _score_against_truth()  is it decidable?
eval/evidence_funnel.py:104-121   verdicts_at()      no slot; folds silently
eval/evidence_funnel.py:246-253   _production_verdict()   re-derives; can't
                                                     express a 4th state (B2)
demo/runner.py:174-195            _derive_outcome()  needs a 6th DemoOutcome
```

None is a bug today: the suite passes, every frozen number reproduces, and no
output loses information in the three-state space. None can be made
behaviour-preserving either, because there is no fourth state whose behaviour
could be preserved — any change would be speculative E3 implementation. Two of
them (`verdicts_at`, `_production_verdict`) are downstream of B2, the
re-derivation, not of B3.

So: **B3 the defect is closed. B3 the design consequence is not, and cannot be
until Phase 3.7 picks a method.**

---

## 6. Invariants — never break these

1. **Truth never reaches inference.** `network/boundary.py` is the only load
   path. Inference sees exactly six columns: `txid, observer_id, peer_ip,
   peer_port, peer_asn, timestamp_ms`. Truth accessors are named
   `*_FOR_EVALUATION_ONLY` so misuse is greppable.
   Enforced by `tests/test_truth_isolation.py` (source-level).
2. **σ = 1.1506 in every regime.** It is the one parameter with a published
   source. Tuning it to improve a Phase 3 result makes the demonstration
   circular.
3. **Production rule fixed**: min pooled 25, alpha 1e-4, effect floor 0.05.
   Never tune per regime.
4. **NO_EVIDENCE transactions never reach a constraint builder.**
   `arrivals.assert_no_constraint_permitted` exists for this.
5. **Cannot-link only.** `Verdict` has exactly three members and no must-link.
6. **Score DECISIONS, not edges** (see Traps).
7. **`peer_ip` must not identify the origin** for ordinary transactions, or
   the next phase can cheat by string comparison.
8. Frozen dataset hash and Phase 1 numbers unchanged.
9. **The holdout (t42-49) is sealed.** No development decision may be
   informed by it. Enforced by `ml/protocol.py` at fold construction and at
   read time; `tests/test_ml_protocol.py` pins both.
10. **No model is superior on a point estimate.** Comparison goes through
    `protocol.compare_family()`: paired by fold, Holm-corrected, and reported
    with its confidence interval. Fold-to-fold sd on this problem is ~35x the
    seed-to-seed sd.
11. **No cross-scope comparison.** Phase 6 and PS-native numbers describe
    different feature sets over different streams. `ScopeMismatchError`.
12. **No fabricated feature.** `ml/diagnostics.assert_healthy()` refuses a
    feature set containing a constant, an exact duplicate or a fixed
    transform of another column. A new candidate must pass it.
13. **Analytical artifacts never enter SQLite.** The console stores
    identifiers (`alert_id`, `run_fingerprint`) only; a risk score lives in
    exactly one place. `tests/test_console_boundary.py` checks the DDL.

---

## 7. Traps — bugs already found, do not reintroduce

| Trap | What happened | Guard |
|---|---|---|
| **Edge scoring** | One component boundary re-proposed by 166 edges was counted as 166 false splits. D's "116 false splits" was really **1**. | `tests/test_decision_scoring.py` |
| **Mixed-entity boundaries** | Boundaries with several entities per side were forced into a binary verdict, manufacturing 54.2% precision from unanswerable questions. | `TruthCategory.MIXED_ENTITY` |
| **20-block cap** | `blocked_examples[:20]` — every precision figure computed on ≤20 decisions. Renamed `blocked_merges`, holds all. | — |
| **Component-keyed entities** | Entities assigned per co-spend component ⇒ zero cross-entity edges ⇒ precision structurally untestable. Now sub-component (52,102 cross-entity edges, 18.99%). | `tests/test_world_ground_truth.py` |
| **651× observation inflation** | Per-address pooling counted each transaction once per input. One component pooled 6,035 "observations" from 26 transactions. Fixed: one representative address per transaction. | `tests/test_separation.py` |
| **Self-loop ordering** | Filtering self-loops *before* the per-tx winner promoted the runner-up, asserting the payment output was change in 17,656 transactions. Filter **after**. | `tests/test_change.py` |
| **AddrAddr misuse** | `AddrAddr_edgelist.csv` is money flow (`input_address, output_address`), NOT co-spend. Clustering on it unions every payer with every payee. Co-spend comes from `AddrTx_edgelist.csv` grouped on `txId`. | `test_loader_never_reads_addraddr` |
| **Dataset mismatch** | Pairing a chain with the wrong network data resolved 0 addresses and reported a silent 100% abstention. | `tests/test_dataset_compatibility.py` |
| **Single-window model selection** | RandomForest was selected over LightGBM on one validation window (0.5702 vs 0.5228). Under 12 rolling folds the difference is −0.019 and **the sign reverses**. Window sd 0.175 vs seed sd 0.005. | `ml/protocol.py`, `tests/test_ml_protocol.py` |
| **The "1 sd" decision rule** | `\|mean diff\| > 1 sd` was reasoned, not measured. Simulation put its false-positive rate at **8.7%**. Replaced by a paired t-test at α=.05 (measured 5.1%). | `test_the_decision_rule_is_a_calibrated_test` |
| **Folds reaching the holdout** | An audit's own rolling CV built folds `41-44` and `45-48` — evaluating on sealed timesteps. A `Fold` that does this now raises at construction. | `test_a_fold_that_reaches_the_holdout_cannot_be_constructed` |
| **Fabricated per-output amounts** | `build_ps_dataset.py` synthesised per-input/output values by even division. Consequences: `*_std` ≡ 0, `*_max` ≡ `*_mean`, `equal_output_count` ≡ `output_count`, `output_entropy` ≡ log2(`output_count`), **`is_peeling_candidate` mathematically unable to fire** (`0.5 >= 0.8`), `is_mixing_candidate` reduced to a fan-in/fan-out rule firing on 34%. Nine of thirty features carried no information. | `ml/diagnostics.py`, `tests/test_feature_health.py`, `tests/test_ps_feature_schema.py` |
| **Permutation test at n=5** | With 5 folds the smallest reachable two-sided p is 2/2^5 = 0.0625, so the exact test **cannot reject at .05** however large the effect. Reported as `permutation_usable: False` rather than as a non-significant p. | `test_the_permutation_check_declares_itself_unusable_below_seven_folds` |
| **localStorage as a security boundary** | Identity, ownership and case state lived in the browser; any string logged you in as anyone. Now server-side sessions, scrypt, HttpOnly cookie. | `tests/test_console_auth.py`, `tests/test_api_access.py` |

---

## 8. Known limitations — say these out loud

- **All network data is SYNTHETIC.** It demonstrates the mechanism and
  validates nothing. The generator was written by the same people as the
  analysis and contains the structure the analysis looks for. Real validation
  needs mainnet capture + controlled ground truth (October).
- **Reach-stress is not an Elliptic++ result.** Its 300-component synthetic
  chain exists so the threshold is reachable at all. Never quote its numbers
  as Elliptic++ performance.
- **No super-cluster exists in Elliptic++.** Largest co-spend cluster is
  14,885 = 1.81% of addresses. The README's cluster-collapse framing has no
  collapse to point at in this data.
- **Entity resolution covers 361 of 822,942 addresses (0.044%)**, with a
  structural recall ceiling of 32.7%.
- **Change detection has shown no benefit** on three independent measures:
  contamination, cluster structure, entity resolution.
- **The five demo scenarios are fixtures, not results.** They were designed
  to reach each engine state, and they do. They say nothing about how often
  those states occur on real data — on Elliptic++ the answer is "almost
  never", which is the finding, not a defect in the demo.
- Phase 3.3 conclusions rest on **1 binary-scorable decision**. The structural
  finding (zero PURE_CROSS blocks in 28,500 decisions) is the evidence, not
  the 0%.

---

## 9. Open questions / next steps

### Current blockers (2026-09-23) — these come first

1. **Train a v2 PS-native model** under `ml/protocol.py`. Until then
   uploaded-dataset runs report `MODEL_UNAVAILABLE_FOR_SCHEMA`. This is the
   one item blocking the product, and it is also the first legitimate use of
   the protocol.
2. **RandomForest vs LightGBM is unresolved and may be unresolvable.**
   0.019 nAP apart against a minimum detectable effect of 0.164. Detecting
   0.05 would need 110 folds; the development period supplies 12. Do not
   pick one on a point estimate.
3. **Severity bands do not deliver their advertised precision.** A
   rank-derived cutoff is applied as a value threshold under heavy score
   ties: CRITICAL advertises 90%, measures 82.2% on validation and 37.2% on
   test. A plain top-K queue beats the bands outright (P@200 = 0.95).
   Fixing it is a ranking-policy decision, not a model change.
4. **Isotonic calibration degrades ranking.** Brier improves, AP falls
   (LR 0.2925 → 0.1642) and resolution collapses from ~10,000 distinct
   scores to ~55. Candidate policy: rank on raw scores, carry the calibrated
   probability as a display field only.
5. **M3 network features are not as-of-t.** `netfeat.build()` takes no
   cutoff while M0/M1/M2 do. Currently benign because M3 contributes ~nothing,
   but a future model that can exploit it would get an unearned lift. Fix
   before any retrain that includes M3.
6. **`ps_model.py` explanations are not SHAP.** It computes
   `global feature_importances_ × |raw value|` and derives direction from
   whether the calibrated probability exceeds 0.5. The Phase 6 path's
   TreeSHAP (`pred_contrib`) is correct; this one is not.
7. **statsmodels / shap are not vendored.** Decide whether the offline build
   guarantee is worth re-vendoring for. Everything needed so far has been
   done with scipy.

### Older Phase 3.7 / Phase 4 questions

**Phase 4 architecture audit: `docs/archive/PHASE4_ARCHITECTURE_AUDIT.md`.** Read it
before any Phase 4 structural work. Headline: E1+E3 can replace the current
oracle, but four blockers must clear first (~120 lines, 5 files), and three
boundary fixes are worth doing regardless of which method Phase 3.7 picks —
result files carry no provenance, the production rule is CLI-tunable without
trace, and a mismatched chain/network pair abstains silently on the production
path.

1. **The architectural question.** Origin dissimilarity cannot express
   "one entity, several origins". Does the cannot-link design need entity-aware
   evidence, or a different signal entirely? This is the decision to make.
2. **Reach on real data.** 40 of 253,429 unions reached the threshold on
   Elliptic++. Even a perfect rule cannot act. Options: pool differently, or
   accept the mechanism only demonstrates on denser chains.
3. **Why zero PURE_CROSS blocks in B/E?** Aggregate signal exists (2.41 /
   2.16 SE) but never produced a correct separation. Worth understanding
   before redesigning.
4. **BitcoinHeist / OFAC** deferred. BitcoinHeist shares provenance with
   `addresses.csv` (13,310 overlapping addresses) so it is NOT independent
   corroboration. OFAC overlaps Elliptic++ by **2 addresses**.
5. **Mainnet capture** — October. Replace `DEFAULT_TX_MEDIAN_MS` (an
   explicitly flagged assumption) and re-fit σ from real transactions.

---

## 10. Command reference

```bash
# Phase 1
make run ARGS="cospend"
make run ARGS="cospend --heuristics multi-input+change"
make run ARGS="purity"
make run ARGS="evolution"
make run ARGS="entity-labels"
make run ARGS="entity-resolution"

# Phase 2
make run ARGS="network-generate"      # frozen config, default
make run ARGS="network-arrivals"
make run ARGS="network-audit"

# Phase 3
make run ARGS="evidence-funnel"
python -m obsidianchain run --mode chain-only     # in-container
python -m obsidianchain run --mode fused
make run ARGS="world-generate"
make run ARGS="world-diagnostics"
make run ARGS="world-experiment"                  # normal A–E
make run ARGS="reach-stress-build"

# reach-stress experiment (explicit roots — a mismatch is refused)
docker run --rm --network none --platform linux/amd64 \
  -v "$PWD/data:/data" -e OBSIDIANCHAIN_DATA=/data \
  obsidianchain:latest world-experiment \
  --chain-root /data/reach_stress --network-root /data/reach_stress

# Demonstration (SYNTHETIC fixtures, confined to data/demo)
make demo                          # build + run + render, then open
                                   # data/demo/output/index.html
make run ARGS="demo"               # reuse an existing fixture
make run ARGS="demo --rebuild"
```

---

### Application layer, synthetic world, evaluation

```bash
# Console accounts (there is no self-registration)
make run ARGS="console-init"
make run ARGS="console-user-add alice --role INVESTIGATOR"

# Additive structural scan - does NOT touch any alert artifact or fingerprint
make run ARGS="mixing-scan"

# Coherent synthetic world + degradation sweep (SYNTHETIC_CONTROL)
make run ARGS="synthetic-world-build"
make run ARGS="synthetic-world-overlap"

# Rebuild the PS DEVELOPMENT parquets. The sealed holdout is skipped.
python3 research/reproduction/build_ps_dataset.py

# The canonical model comparison. Reads train+validation only.
python3 research/protocol_2026_09_22/run_protocol.py

# Supporting evidence for the protocol's constants
python3 research/protocol_2026_09_22/validate_decision_rule.py
python3 research/protocol_2026_09_22/power_design.py
```

**Never** set `OBSIDIANCHAIN_REGENERATE_HOLDOUT=1` without a written reason
recorded alongside the run.

Test suite: `1745 passed, 4 skipped` (backend, excluding
`test_phase6_leakage.py` which is container-only), `101 passed` (frontend).

---

## 11. Working agreement that has served well

- Claude writes loaders, instrumentation, evaluation, CLI, tests.
- **Varun writes the core algorithms** — union-find `find`/`union`,
  `select_change_rows`, `ConstrainedUnionFind.union` — because he must explain
  them to judges without notes. Claude supplies the skeleton, contracts, and
  the tests that act as the spec.
- Every new measurement gets an independent cross-check before it is believed.
  Four confident-but-wrong numbers have been caught this way.
- Nothing is committed by Claude without being asked.
