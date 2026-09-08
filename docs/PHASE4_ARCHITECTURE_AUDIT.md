# Phase 4 — Architecture Audit

**Scope.** Read-only audit of `obsidianchain` at branch `phase-3.3-checkpoint`
(514 tests passing). No features written, no working code refactored. Every
claim below carries a `file:line` reference and was checked against source, not
inferred from documentation.

**Standing constraints honoured throughout.** Air-gapped, one laptop, stdlib +
numpy/pandas/scipy. No queue, no database, no cluster, no graph library, no
cloud. Nothing in this document proposes any of those, and §8 says why the
temptation is real anyway.

**One input is missing.** The brief says *"Edge cases currently unhandled (list
below)"* and no list followed. §5 therefore contains the edge cases found by
inspection. If there was a specific list intended, send it and §5 gets
re-scored against it — the rest of the audit does not depend on it.

---

## 0. Headline answers

### 0.1 The Phase 3.7 question

> *If E1 + E3 becomes the chosen methodology, can it replace the current
> `ArrivalSeparationOracle` without changing the rest of the system?*

**No. Four blockers, all small, all in known places. Total ≈ 120 lines across
5 files.** None of them is a redesign; every one is a case of a concrete type
being named where an abstract one belongs.

| # | Blocker | Where | Why E1/E3 breaks on it |
|---|---|---|---|
| B1 | The clusterer manipulates the oracle's **internal sufficient statistic** by name | `cluster/constrained.py:48, 161-164, 279-283, 299-301` | `GroupStats` is chi-square sufficient statistics (`dim_count/dim_sum/dim_sumsq`). A likelihood-ratio provider pools different state. The clusterer must hold an *opaque* evidence state it can only `combine()`. |
| B2 | The evidence record **is** a chi-square result | `network/separation.py:114-142` | `chi2`, `dof`, `p_value`, `effect` are fields, not payload. An LR has no `dof`; its effect is a ratio, not an RMS centroid difference. Read by name in 4 places incl. a CSV schema. |
| B3 | `Verdict` is a closed 3-set and one consumer enumerates it **as string literals** | `network/separation.py:71-83`; `eval/phase33.py:723` | Adding `INCONCLUSIVE` compiles fine and then silently vanishes from `phase33.csv`, because the column loop is `for state in ("NO_EVIDENCE", "SEPARATED", "NOT_SEPARATED")`. A new state that disappears from the output file is worse than one that crashes. |
| B4 | `build_oracle` is a god function welding **four responsibilities** together | `network/separation.py:331-457` (127 lines) | It loads the network source, loads the blockchain source, runs the arrival transform, *and* accumulates evidence. E1 needs to replace only the fourth. Nothing can be reused piecewise. |

**And a multiplier:** the pooled-evidence loop — "walk the edges, keep per-root
state, merge on union, ask the oracle" — exists **four times**:

- `cluster/constrained.py:168-211` (production)
- `eval/evidence_funnel.py:157-201` (instrumentation)
- `eval/phase33.py:317-360` (decision scoring)
- `demo/runner.py:169-197` (demonstration)

E1 would require editing all four, and two of them (`phase33.py:330`,
`demo/runner.py:179`) reach into the **private** `forest._evaluate_cannot_link`
to do it.

**Verdict: the *shape* of the current contract is right, the *types* are wrong.**
The clusterer already assumes exactly what E1 needs — that evidence is a
commutative monoid over components plus a comparator. That is the correct
abstraction and it was arrived at for good reasons (`constrained.py:157-160`).
Only the names are concrete. This is a rename-and-narrow job, not a rewrite.

### 0.2 The modularity test

| Scenario | Correct answer | Actual answer | Verdict |
|---|---|---|---|
| Real P2P capture arrives tomorrow | 1 adapter + config | **0 files** if pre-normalised to the six-column schema; **no slot at all** for the normaliser if not | **Partial pass** |
| Another blockchain dataset | 1 adapter + config | **6+ files** | **Fail** |
| LightGBM replaces the rule-based scorer | 1 adapter + config | Neither exists — greenfield | **N/A (cheapest to get right)** |
| Different clustering algorithm | 1 adapter + config | **2 files** (`pipeline.py`, `worlds.py`) | **Near pass** |

Worked through in §4.6.

---

## 1. Current architecture, as it actually is

Not the intended design. Solid arrows are imports that exist today; the two
marked `!` are the ones that should not.

```
                        ┌──────────────────────────────────────┐
   data/raw/            │ cli.py  (1342 lines, 20 commands)    │
   AddrTx_edgelist.csv  │ every command imports concrete       │
   TxAddr_edgelist.csv  │ modules inline; no registry, no      │
   wallets_classes.csv  │ config file, no DI                   │
        │               └───┬──────────────────────────────┬───┘
        │                   │                              │
        ▼                   ▼                              ▼
┌──────────────────┐  ┌──────────────┐            ┌─────────────────┐
│ io/elliptic.py   │  │ cluster/     │            │ eval/           │
│  ADDR_TX  = ...  │  │  unionfind   │            │  phase33        │
│  TX_ADDR  = ...  │◀─┤  change      │            │  evidence_funnel│
│  WALLETS  = ...  │  │  pipeline    │            │  purity         │
│  CoSpendGraph    │  │  constrained │            │  entity_resolut │
│  (3 filenames    │  └──────┬───────┘            │  evolution      │
│   pinned :35-37) │         │                    │  world_diagnost │
└────────┬─────────┘         │                    │  compare        │
         │                   │                    └────────┬────────┘
         │                   │ GroupStats  (!)             │
         │                   │ by name                     │ reads truth
         │                   ▼                             │ (whitelisted)
         │          ┌────────────────────────────┐         │
         │          │ network/separation.py      │         ▼
         └─────────▶│                            │  data/processed/
           (!)      │  Verdict          :71-83   │  network_truth/
        build_      │  SeparationConfig :86-112  │  worlds_truth/
        oracle      │  SeparationEvidence:114-142│   (QUARANTINED)
        reads the   │  GroupStats       :144-192 │
        chain       │  separation_evidence:194   │
        directly    │  SeparationOracle :282-314 │
                    │  build_oracle     :331-457 │ ◀── GOD FUNCTION
                    │    ├ loads network source  │     4 responsibilities
                    │    ├ loads chain source    │     in one call
                    │    ├ arrival transform     │
                    │    └ accumulates evidence  │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼─────────────────┐
                    │ network/boundary.py        │  THE information
                    │  PHASE3_ALLOWED_FIELDS     │  boundary. Six
                    │  = {txid, observer_id,     │  columns. Enforced
                    │     peer_ip, peer_port,    │  by column screen
                    │     peer_asn, timestamp_ms}│  + exact schema pin.
                    │  world: str|None  ◀────────┼── the ONLY source
                    └──────────┬─────────────────┘   selector that exists
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
      processed/network/  processed/worlds/{A..E}/  data/demo/processed/
      (frozen, hash        (controlled regimes)     data/reach_stress/
       c405493d…)                                   (own roots)
              ▲                ▲                    ▲
              │                │                    │
       ┌──────┴────────────────┴────────────────────┴──────┐
       │ network/synthetic.py   generator, sigma 1.1506    │
       │ network/worlds.py      A–E regimes  (!)imports    │
       │ network/reach_stress.py dense fixture   cluster/  │
       │ demo/scenarios.py      five scenarios   pipeline  │
       └───────────────────────────────────────────────────┘
         all four WRITE Elliptic's filenames verbatim,
         because there is no other way into the loader
```

**What the diagram shows that a design document would not:**

1. `network/separation.py` imports `io/elliptic.py` (`separation.py:383`). The
   evidence provider reads the blockchain. That is the single edge that makes
   "swap the network source" and "swap the blockchain source" the *same*
   change.
2. `network/boundary.py` imports `network/synthetic.py` (`boundary.py:48`) —
   the screen depends on the generator, for directory-name constants. The
   boundary cannot be reasoned about without the generator in scope.
3. `network/worlds.py` imports `cluster/pipeline.py` (`worlds.py:233`) — the
   data generator runs the clusterer, to assign entities to co-spend
   components. Generation and inference are not layered.
4. Four fixture generators independently emit `AddrTx_edgelist.csv` +
   `wallets_classes.csv` (`reach_stress.py:143-144`,
   `demo/scenarios.py:600-601`, and the two chain builders above them). They
   impersonate Elliptic because impersonation is the only interface.

---

## 2. What is already modular

This section matters as much as §3. Several things are **right**, and the
refactor must not damage them.

### 2.1 `network/boundary.py` — the information boundary (keep exactly as is)

`PHASE3_ALLOWED_FIELDS` (`boundary.py:51-55`) is a six-column contract,
enforced twice: a forbidden-substring screen (`:57-75`, `:77-99`) and an exact
schema pin (`:164-169`). `_reject_truth_path` (`:102-110`) refuses truth
directories by path. This is the best-engineered boundary in the repository and
it is *already* the `NetworkSource` data contract — see §4.2.

The screen is substring-based and case-insensitive, so a renamed leak
(`origin_id_x`, `true_origin`) is still caught. That design survives new
adapters unchanged.

### 2.2 `ClusterRun` / `FusedRun` — an accidental but correct contract

`cluster/pipeline.py:26-63` and `:109-133`. Seven consumers
(`compare`, `entity_resolution`, `evolution`, `worlds`, `phase33`,
`demo/runner`, `cli`) touch only `.roots`, `.sizes`, `.n_clusters`, `.largest`,
`.coverage`, `.blocked`, `.contested`. **None reaches into the union-find.**
This is already the `EntityCluster` contract in all but name.

### 2.3 The evidence monoid

`GroupStats.combine` (`separation.py:169-175`) is associative and commutative,
tested for both (`test_separation.py:62-73`). `ConstrainedUnionFind` merges
component state in O(1) on union (`constrained.py:299-301`). **This is exactly
the algebra E1 needs.** The abstraction is present; only its type is concrete.

### 2.4 Truth quarantine

`test_truth_isolation.py` is a *source-level* test — it scans executable code,
stripping docstrings (`:56-73`), so a module may discuss truth at length and
still pass. That catches the failure runtime tests cannot: a truth join on a
branch nobody exercised. Caveat in §6.5.

### 2.5 Determinism and offline posture

Every generator takes an explicit seed and records a manifest with a SHA-256
(`synthetic.py:548-582`, `worlds.py:528`, `reach_stress.py:305-326`,
`demo/scenarios.py:631-660`). `Dockerfile:28-38` pins `PYTHONHASHSEED=0` and
single-threaded BLAS/OMP. `Makefile:49` runs `--network none`. Nothing here
needs touching.

### 2.6 The `demo/` package's flag discipline

`demo/api.py:76-104` enforces a provenance flag on **every object** in the
payload and both renderers refuse unflagged input
(`demo/report.py:56`, `:401`). This is the pattern §6 recommends generalising —
the demonstration path is currently more rigorous about provenance than the
production evaluation path.

---

## 3. What is tightly coupled

Ordered by how much it costs at Phase 3.7.

### 3.1 The clusterer knows the oracle's internal statistic — **critical**

```
cluster/constrained.py:48        from ...separation import GroupStats
cluster/constrained.py:161-164   self._stats: dict[int, GroupStats] = {}
                                 for code, stats in oracle.address_stats.items():
cluster/constrained.py:279-283   def pooled_stats(self, root) -> GroupStats
cluster/constrained.py:282       return GroupStats.empty(1)
cluster/constrained.py:299-301   .combine(self.pooled_stats(absorbed_root))
```

The clusterer imports a chi-square sufficient statistic, reads
`oracle.address_stats` (a public attribute holding provider internals), and
constructs `GroupStats.empty(1)` itself. Three separate leaks of one type.

Same leak in `eval/evidence_funnel.py:140-141, 196` and
`demo/audit.py:57, 87-98`.

**Cost at 3.7:** every one of these must change when the statistic changes.

### 3.2 The evidence record is a chi-square result — **critical**

`network/separation.py:114-142`. `chi2`, `dof`, `p_value`, `effect` are
first-class fields. Consumed by name in:

```
eval/evidence_funnel.py:79-81, 181-188, 212-214   -> CSV/parquet schema
eval/evidence_funnel.py:244-250                   -> re-derives the verdict
eval/phase33.py:166-168, 349-351                  -> Decision dataclass
demo/runner.py:95-96, 292-294, 380-382, 400-402   -> API payload
demo/report.py:132, 190, 319, 355                 -> rendered to the UI
```

`evidence_funnel.py:244-250` is the worst of these: it **re-implements the
production decision rule** from recorded columns rather than asking the
provider. Under E1 the columns would not exist and the function would be
silently wrong rather than absent.

### 3.3 `build_oracle` is a god function — **high**

`network/separation.py:331-457`, 127 lines, four responsibilities:

```
:354   boundary.load_phase3_inputs(...)          NetworkSource
:358   arrivals.build(...)  :365 classify_evidence   arrival transform
:383   elliptic.load_input_edges(data_root)      BlockchainSource   (!)
:426-435  np.bincount accumulation               evidence accumulation
```

Line 397 is the coupling that matters: the evidence provider opens the
blockchain. It needs the address↔transaction map, which is a legitimate need —
but it takes it by *importing the Elliptic loader*, so a new chain format
breaks the network layer.

### 3.4 Four copies of the pooled-evidence loop — **high**

| Copy | Location | Reaches into private state? |
|---|---|---|
| production | `cluster/constrained.py:168-211` | — |
| instrumentation | `eval/evidence_funnel.py:157-201` | reads `oracle.address_stats` |
| decision scoring | `eval/phase33.py:317-360` | **`forest._evaluate_cannot_link`** (`:330`) |
| demonstration | `demo/runner.py:169-197` | **`forest._evaluate_cannot_link`** (`:179`) |

Two callers depend on a private method to observe a decision without making it.
That is a missing public API (`§4.4`, `probe()`), not a discipline failure —
there is currently no other way to ask "what would you decide?"

### 3.5 The production rule is CLI-tunable and the value is never recorded — **high**

```
cli.py:284-288    run            --min-pooled / --alpha / --min-effect
cli.py:367-369    fusion-summary --min-pooled / --alpha / --min-effect
cli.py:714-718    evidence-funnel --min-pooled / --alpha / --min-effect
```

HANDOFF invariant 3 says *"Production rule fixed: min pooled 25, alpha 1e-4,
effect floor 0.05. Never tune per regime."* Three commands accept overrides,
and **none of them prints the config it used or writes it to the output file.**
A `merges BLOCKED = 47` line produced at `--min-pooled 2` is byte-identical to
one produced at 25. See §6.1.

(`eval/phase33.py` is clean here — `run_regime` takes an explicit config and
`cli.py:482` constructs it from defaults with no flags. The demo path is clean
too: `demo/runner.py:48-79` hard-fails on a modified rule.)

### 3.6 The blockchain loader is Elliptic-shaped — **medium**

`io/elliptic.py:35-37` pins three filenames; `load_cospend_graph:227-294` pins
the two-file factorize. Direct consumers: `cluster/change.py:138-152`,
`network/synthetic.py:288-298`, `network/separation.py:383`,
`eval/purity.py`, `eval/entity_resolution.py`, `io/entity_labels.py`, `cli.py`.

Evidence that this hurts: `reach_stress.py:143-144` and
`demo/scenarios.py:600-601` both write `AddrTx_edgelist.csv` and
`wallets_classes.csv` *for synthetic data that has nothing to do with
Elliptic*, purely to get through the loader.

### 3.7 `network/worlds.py` imports the clusterer — **medium**

`worlds.py:233`, used to assign entities per co-spend component. Data generation
depending on inference is a layering inversion. It has already caused one
documented bug (component-keyed entities → zero cross-entity edges → precision
structurally untestable; HANDOFF §7).

### 3.8 `network/boundary.py` imports `network/synthetic.py` — **low**

`boundary.py:48`, for `OBSERVATIONS_DIR` / `TRUTH_DIR` / `WORLDS_DIR` string
constants (`synthetic.py:104-111`). Harmless today; wrong direction. The screen
should not depend on any producer.

### 3.9 `cli.py` at 1342 lines with no registry — **low, but growing**

Every command imports concrete modules inline. There is **no configuration
layer at all**: `grep` for config loading returns only
`os.environ.get("OBSIDIANCHAIN_DATA")` (`cli.py:48`, `elliptic.py:45`). So
"plus configuration" in the modularity test currently means "plus a CLI flag
and an `if`".

`tests/test_cli.py` is 28 lines / 3 tests (version, info, help). The CLI is the
least-tested surface and the one a refactor touches most.

---

## 4. The exact modules that need interfaces

Five interfaces, no more. Each is a `typing.Protocol` — structural, no base
class, no registry framework, no dependency-injection container. Adapters are
plain modules.

> These are **specifications, not code to write now.** Ship them when the
> corresponding swap is actually needed (§7 orders that).

### 4.1 `BlockchainSource`

```python
class BlockchainSource(Protocol):
    """Everything the clusterer and the evidence provider need from a chain."""

    def cospend_graph(self) -> CoSpendGraph: ...

    def input_edges(self) -> pd.DataFrame:
        """Columns: input_address (str), txid. The address<->tx map that
        evidence attribution needs. Separate from cospend_graph() because
        build_oracle needs this and not the edge list."""

    def output_edges(self) -> pd.DataFrame | None:
        """None when the dataset has no output side. Change detection is
        skipped rather than failing."""

    def address_labels(self) -> pd.Series | None:
        """None when the dataset carries no licit/illicit class."""

    def describe(self) -> SourceManifest: ...
```

Adapters: `EllipticSource` (wraps today's `io/elliptic.py` unchanged),
`CsvBlockchainSource` (generic two-column `address,txid`).

`output_edges` and `address_labels` return `None` deliberately — a real chain
dump may have neither, and today `cluster/change.py` and `eval/purity.py` would
simply fail.

### 4.2 `NetworkSource`

```python
class NetworkSource(Protocol):
    def observations(self) -> pd.DataFrame:
        """EXACTLY boundary.PHASE3_ALLOWED_FIELDS. The adapter normalises;
        the boundary screen still runs on the result."""

    def observers(self) -> pd.DataFrame:
        """observer_id, asn. No clock offsets - those are estimated, never
        known."""

    def broadcaster_ips(self) -> set[str]:
        """Public infrastructure list. Empty set is valid."""

    def describe(self) -> SourceManifest: ...
```

Adapters: `SyntheticNetworkSource`, `ParquetNetworkSource` (today's
`boundary.load_phase3_inputs`), later `PcapNetworkSource` / `ZmqNetworkSource`.

**This one is nearly free.** `boundary.Phase3Inputs` (`boundary.py:113-127`)
already has this exact shape. The interface is a rename plus a place to put an
adapter. The boundary screen stays *downstream* of every adapter, so a bad
adapter is caught, not trusted.

### 4.3 `Clusterer`

```python
class Clusterer(Protocol):
    def cluster(
        self,
        graph: CoSpendGraph,
        evidence: EvidenceProvider | None = None,
    ) -> EntityClustering: ...
```

Adapters: `CoSpendClusterer` (today's `run_clustering`),
`CoSpendPlusChangeClusterer` (today's `run_clustering(change_edges=...)`).

`EntityClustering` = today's `ClusterRun`/`FusedRun` merged. This is the
**cheapest** interface: §2.2 shows all seven consumers already respect it.
Only `pipeline.py:149-158` (hardwired `ConstrainedUnionFind` + a probe call)
and `worlds.py:233` need to change.

### 4.4 `EvidenceProvider` — the one that decides Phase 3.7

```python
class EvidenceState(Protocol):
    """Opaque per-component evidence. The clusterer NEVER inspects this;
    it only pools it. Must be associative and commutative under combine()
    so union order cannot change a decision."""

    count: int          # pooled observations. The ONLY field the clusterer
                        # may read, and only for reporting.

    def combine(self, other: EvidenceState) -> EvidenceState: ...


class EvidenceProvider(Protocol):
    def initial_state(self, code: int) -> EvidenceState:
        """Per-address seed. Empty state for an address with no evidence."""

    def empty_state(self) -> EvidenceState: ...

    def compare(self, a: EvidenceState, b: EvidenceState) -> EvidenceRecord:
        """The whole decision. The provider owns thresholds, calibration,
        and interpretation. The clusterer only reads .verdict."""

    def describe(self) -> EvidenceManifest:
        """Method name, parameters, calibration provenance, known limits.
        Flows into every output file - see section 6."""
```

Adapters: `ArrivalSeparationOracle` (today's `SeparationOracle`, renamed),
later `CalibratedLikelihoodRatioProvider` (E1).

**This is the minimum interface that makes E1 pluggable.** Note what it does
*not* require: no `chi2`, no `dof`, no `p_value`, no per-observer dimensions,
no calibration hook. Those are E1's business, behind `compare()`.

The single behavioural contract the clusterer needs and must state explicitly:

> `combine` is associative and commutative; `compare` is a pure function of two
> states. Union order must not change any decision.

That contract is already tested for `GroupStats`
(`test_separation.py:62-73`) — the test generalises to any provider.

### 4.5 `RiskScorer` — greenfield

```python
class RiskScorer(Protocol):
    def score(
        self,
        cluster: EntityCluster,
        evidence: Sequence[EvidenceRecord],
    ) -> Alert | None:
        """None when nothing is worth raising. Never a default 'low risk'
        alert - see section 6.4."""

    def describe(self) -> ScorerManifest: ...
```

**Nothing like this exists.** `grep -rniE "risk|alert|threat|suspicious"` over
`src/` returns exactly one hit, a docstring in `cluster/change.py:374`. There is
no scorer to unpick, which makes this the one interface that can be defined
correctly on the first try — and the one where a wrong data contract would be
most expensive, because `Alert` is the object an analyst acts on.

### 4.6 Data contracts

Required fields are the minimum a real dataset must supply. **Optional means
optional** — every `| None` below is a dataset that genuinely may lack it.

```python
@dataclass(frozen=True)
class TransactionRecord:
    txid: str
    input_addresses: tuple[str, ...]        # required: this IS co-spend
    output_addresses: tuple[str, ...] = ()  # absent in input-only dumps
    block_height: int | None = None
    timestamp: int | None = None
    fee: int | None = None


@dataclass(frozen=True)
class NetworkObservation:
    txid: str
    observer_id: str
    timestamp_ms: float
    peer_ip: str | None = None       # a capture may not record the peer
    peer_port: int | None = None
    peer_asn: int | None = None
    # NOTHING ELSE. This is boundary.PHASE3_ALLOWED_FIELDS and must stay
    # identical to it, or the screen and the contract can drift apart.


@dataclass(frozen=True)
class ArrivalEvidence:
    """One transaction's arrival pattern. Raw observation layer."""
    txid: str
    observer_ids: tuple[str, ...]
    offsets_ms: np.ndarray            # NaN where unobserved
    ranks: np.ndarray                 # bias-robust; see section 5.3
    spread_ms: float
    n_observed: int
    usable: bool
    excluded_reason: str | None = None


@dataclass(frozen=True)
class EntityCluster:
    cluster_id: int
    address_codes: np.ndarray
    size: int
    contested: bool = False
    provenance: Provenance | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    """What a provider returns. Method-agnostic by construction."""

    verdict: Verdict                  # SEPARATED / NOT_SEPARATED /
                                      # INCONCLUSIVE / NO_EVIDENCE
    coverage_a: int                   # pooled observations, side A
    coverage_b: int
    provenance: Provenance            # NOT optional - see section 6

    score: float | None = None        # E1: calibrated LR. None for a
                                      # method that has no scalar score.
    raw_score: float | None = None    # pre-calibration
    quality: float | None = None      # provider-defined evidence quality
    statistics: Mapping[str, float] = field(default_factory=dict)
                                      # {"chi2":.., "dof":.., "p_value":..}
                                      # today; {"llr":.., "n_calib":..}
                                      # under E1. A BAG, never fields.
    reason: str = ""
    limitations: tuple[str, ...] = ()  # e.g. ("synthetic propagation model",
                                       #       "calibration set n=340")


@dataclass(frozen=True)
class Provenance:
    """Travels with every record into every file. Section 6."""
    source: str                # "elliptic++" / "synthetic-world-D" / "demo"
    synthetic: bool
    dataset_sha256: str | None
    seed: int | None
    method: str                # "chi2-centroid" / "calibrated-LR"
    method_params: Mapping[str, float]
    is_measurement: bool       # False for every fixture-derived number


@dataclass(frozen=True)
class Alert:
    alert_id: str
    cluster_id: int
    score: float
    reasons: tuple[str, ...]
    supporting_evidence: tuple[EvidenceRecord, ...]
    provenance: Provenance
    confidence: str | None = None   # None when the scorer is uncalibrated -
                                    # NOT a default "medium"
```

Three deliberate choices:

- **`statistics` is a `Mapping`, not fields.** This is the single change that
  unblocks E1 (blocker B2). Chi-square puts `chi2/dof/p_value/effect` in it;
  E1 puts `llr/calibration_n/prior`. Consumers that want to *render* a number
  iterate the mapping; nothing reads `.chi2` by name.
- **`Provenance` is required on `EvidenceRecord` and `Alert`**, optional on
  `EntityCluster`. §6 explains why the first two cannot be optional.
- **`confidence: str | None`** — an uncalibrated scorer must be able to say
  *nothing*, not "medium". Same reasoning as `INCONCLUSIVE`.

### 4.7 `Verdict` under E3

```python
class Verdict(str, Enum):
    SEPARATED     = "SEPARATED"      # cannot-link fires
    NOT_SEPARATED = "NOT_SEPARATED"  # looked, found nothing
    INCONCLUSIVE  = "INCONCLUSIVE"   # E3: enough data, cannot decide
    NO_EVIDENCE   = "NO_EVIDENCE"    # not enough data to look
    # Still no MUST_LINK / SAME_ORIGIN. The asymmetry is the design.
```

`INCONCLUSIVE` and `NO_EVIDENCE` are **not** the same and merging them would
lose the distinction E3 exists to make: *"the measurement was adequate and the
answer is still no"* versus *"the measurement was inadequate"*. That is exactly
the A-versus-E distinction the demonstration already draws at the coverage
layer (`demo/scenarios.py:158-183`) — E3 draws it at the decision layer.

`CONTESTED` is deliberately **not** in this enum. It is a property of a
*component after clustering*, not of a pairwise comparison — it lives on
`EntityCluster.contested`, where `constrained.py:362-372` already puts it.
Adding it to `Verdict` would let a provider claim a component-level state it
cannot observe.

**Phase 4.1 status.** The B3 *defect* - a state that exists in the enum
disappearing from an export with no error - was one site, and it is fixed:
`phase33.to_frame` now iterates `Verdict`. Nothing else in the table below was
ever a defect. Every remaining row is a **semantic branch point**: correct for
the three-state space, and needing a decision about *what INCONCLUSIVE means*
before it could change. None can be made behaviour-preserving, because there is
no fourth state today whose behaviour could be preserved - so all of them are
Phase 3.7 / 4.2 work by construction, and Phase 4.1 left them alone
deliberately rather than by omission.

**What breaks when the fourth member is added** (all of it, exhaustively):

| Site | Behaviour | Action | Phase 4.1 |
|---|---|---|---|
| `constrained.py:181` | `if verdict is SEPARATED: block` — everything else merges | Correct for `INCONCLUSIVE`, but **accidental**. Make it explicit. | branch point, open (now `:182`) |
| `constrained.py:357` | counts `NO_EVIDENCE` as abstention, all else as evaluated | `INCONCLUSIVE` would count as *evaluated*. Wrong — needs a third counter. | branch point, open (now `:404-415`) |
| `phase33.py:723` | `for state in ("NO_EVIDENCE","SEPARATED","NOT_SEPARATED")` | **Silently drops the column.** Must derive from the enum. | **FIXED** (now `:670`) |
| `phase33.py:462` | `!= NO_EVIDENCE` ⇒ decidable | `INCONCLUSIVE` counted decidable. Arguable; must be a decision, not a default. | branch point, open (now `:404`) |
| `evidence_funnel.py:231` | `{v.value: 0 for v in Verdict}` | Already enum-derived. **Correct as written.** | still correct (now `:233`) |
| `evidence_funnel.py:244-250` | re-derives verdict from chi-square columns | Cannot express `INCONCLUSIVE`. Delete; call the provider. | branch point, open (now `:246-253`) — this is B2 |
| `demo/runner.py:197-218` | outcome derivation | Add one branch. | branch point, open (now `:174-195`) |
| `evidence_funnel.py:104-121` | `verdicts_at` returns a hard two-key `{SEPARATED, NOT_SEPARATED}` dict | A third *decidable* state has no slot and folds into NOT_SEPARATED. Same harm as B3, different cause: re-derivation, not enumeration. | branch point, open — **was missing from this table**; a consequence of B2 |

### 4.8 The modularity test, worked

**(a) A real P2P capture arrives tomorrow.**

*If it is normalised to the six columns and written to
`processed/network/observations.parquet`:* **zero files change.** Point
`--data-root` at it. `boundary.load_observations` reads parquet or CSV
(`boundary.py:151-152`), screens the columns, and everything downstream is
already source-agnostic. This is a genuinely good result and it is worth
saying so plainly: the boundary did its job.

*If it arrives as pcap / ZMQ / a peer log:* the normaliser has **nowhere to
live**. It would land in `boundary.py` — which is the *screen*, not a loader
registry — or in an ad-hoc script outside the package. Also `separation.py:383`
means the capture must be paired with an Elliptic-shaped chain.

**Verdict: partial pass.** The data contract is right; the adapter slot is
missing. Fix = §4.2 + a `sources/` package. Cost: low.

**(b) Another blockchain dataset.**

Files that change today:

```
io/elliptic.py:35-37, 224-292     filenames + two-file factorize
network/synthetic.py:288-298      load_transaction_ids -> elliptic
network/separation.py:383         build_oracle -> elliptic
cluster/change.py:138-152         output side -> elliptic
eval/purity.py                    class labels -> elliptic
eval/entity_resolution.py         -> elliptic
io/entity_labels.py:92            raw/ layout
cli.py                            ~8 commands
```

**Verdict: fail — 6+ files.** Root cause is §3.6. The proof is already in the
tree: two synthetic fixtures write Elliptic's filenames because impersonation
is the only interface.

**(c) LightGBM replaces the rule-based scorer.**

Neither exists (§4.5). **Verdict: N/A.** This is the cheapest interface to get
right *and* the one where getting the contract wrong is most expensive, because
`Alert` is what an analyst acts on. Define `Alert` before writing either
scorer.

**(d) A different clustering algorithm.**

`ClusterRun` is already the contract (§2.2). Changes needed:
`pipeline.py:149-158` (hardwired `ConstrainedUnionFind` + a probe that
instantiates it), `worlds.py:233` (imports `run_clustering` directly).

**Verdict: near pass — 2 files.** Note the real constraint is algorithmic, not
architectural: the cannot-link veto requires an *incremental* clusterer that
decides per-merge. A batch algorithm cannot be vetoed mid-run and would need a
different constraint model. That is a genuine limit, not a coupling defect, and
the `Clusterer` protocol should say so rather than pretend otherwise.

---

## 5. Edge cases currently unhandled

The brief's list did not arrive. These are what inspection found, each with the
file that would have to change. Severity is about *silence*: a case that fails
loudly is not on this list.

### 5.1 A mismatched chain/network pair abstains silently on the production path — **critical**

`assert_datasets_compatible` (`phase33.py:81-119`, `MIN_TXID_OVERLAP = 0.5`)
exists precisely because this failure has bitten twice. It is called from
**one** place: `phase33.py:396`.

`build_oracle` has no such guard. If no transaction resolves, `:384-387`
filters to an empty frame, `address_stats` ends up empty, and the run reports
100% abstention — a clean-looking negative result. The three production
commands that call `build_oracle` are all unguarded:

```
cli.py:270   run --mode fused
cli.py:316   fusion-summary
cli.py:751   evidence-funnel
```

The guard lives in the evaluation harness, not the production path. Under
Phase 4 (real capture + real chain, pairing no longer guaranteed by a shared
generator) this becomes the most likely way to produce a confident wrong
answer. **Fix: move the check into `build_oracle`.** ~10 lines, no new
concepts.

### 5.2 A single-observer deployment abstains 100% with no diagnostic — **high**

`arrivals.classify_evidence:264` marks `n_observed < 2` as
`too_few_observers`. A one-vantage-point capture therefore yields *every*
transaction `NO_EVIDENCE`. Correct behaviour; **no warning anywhere.** The
operator sees the same output as a two-observer capture with unlucky coverage.
`arrivals.build:161-162` raises only on *zero* observers.

### 5.3 The bias-robust representation is computed and never used — **high**

`arrivals.py:12-23` argues that rank vectors survive per-observer clock bias
while offsets do not. `ArrivalVectors.ranks` is computed (`:226-238`) and
written to the wide table (`:131`).

The oracle uses **offsets**: `separation.py:374` `_unit_normalise(vectors.offsets_ms[...])`.
`ranks` is consumed only by `network/audit.py:216,225` and
`eval/world_diagnostics.py:206-212` — both evaluation.

Today the generator models clock bias as a *fixed* per-observer offset, and a
fixed offset shifts a whole vector without changing its shape, so unit
normalisation absorbs most of it. **A real capture has drift, not a fixed
offset**, and drift changes the shape. The one representation immune to it is
sitting unused. Not a bug now; a live risk the moment real data arrives, and
worth stating before anyone reads a Phase 4 result.

### 5.4 Evidence for out-of-range address codes is dropped silently — **medium**

```
constrained.py:164     if 0 <= code < n:        # else: discarded, no warning
constrained.py:394-395 if a >= self.n_nodes ... continue
```

A provider whose code space is larger than the graph's loses evidence with no
diagnostic. Harmless while one function builds both; **exactly the failure a
pluggable `EvidenceProvider` introduces.** Fix: raise, or count and report.

### 5.5 An empty-but-present observation file is untested — **medium**

`boundary.load_observations` raises `FileNotFoundError` when the file is
missing (`:154-160`), but a present, zero-row parquet passes the screen. It
then flows into `arrivals.build`, which raises only if the *observer axis* is
empty. No test covers this; the likely outcome is a confusing downstream error
rather than "this capture is empty".

### 5.6 The post-hoc contradiction audit cannot see two thin halves — **medium, documented**

`demo/audit.py:109-152` compares one member against the rest of its component.
A component split into two halves that each pool below the minimum stays
invisible. Documented at `demo/audit.py:30-38`; listed here because it is a
real coverage limit of the CONTESTED path, not only of the demonstration.

### 5.7 `_unit_normalise` has no post-normalisation degeneracy guard — **low**

`separation.py:316-330` substitutes a scale of 1.0 when the max offset is
non-positive or non-finite. An all-zero offset vector normalises to all zeros
and is then indistinguishable from a genuinely simultaneous arrival. In
practice `classify_evidence`'s 250 ms flat filter (`arrivals.py:83`) catches
these first — but that filter runs on raw `spread_ms`, *before* normalisation,
so the two guards are not composed and the coverage is incidental.

### 5.8 Not edge cases — verified handled

Listed so nobody re-checks them: duplicate announcements collapse to earliest
(`arrivals.py:173-179`); a dead observer mid-capture is handled per-dimension
by `dim_count`; zero-variance observers return `NO_EVIDENCE` rather than
dividing (`separation.py:241-247`); int32 code-space overflow raises
(`elliptic.py:255-258`); `build_oracle` raises when address labels were
discarded (`separation.py:378-382`).

---

## 6. Scientific-boundary findings

The question: **anywhere an inference could be displayed or stored as a fact.**

Truth *isolation* is in good shape — `test_truth_isolation.py` passes and no
inference module reads `network_truth/`. Every finding below is the *other*
direction: an inference leaving the system without its provenance attached.

### 6.1 Result files carry no provenance at all — **critical**

Every terminal renderer carries a SYNTHETIC banner:
`separation.py:464`, `arrivals.py:319`, `synthetic.py:596`, `phase33.py:500-513`,
`world_diagnostics.py:398`, `evidence_funnel.py:273`, `cli.py:584,638`.

**No output file carries anything.**

```
cli.py:527   phase33.csv            <- to_frame:708-731, no provenance column
cli.py:532   phase33_decisions.csv  <- decisions_to_frame:693-706, none
cli.py:700   world_diagnostics.csv  <- none
cli.py:936   arrival_vectors.parquet<- none
evidence_funnel.py:430-441  evidence_funnel.parquet <- none
```

A row of `phase33_decisions.csv` reads:

```
regime,decision_id,component_a,component_b,evidence_state,chi2,p_value,blocked
D,417,88213,90114,SEPARATED,64.4,6.33e-11,True
```

Opened in a spreadsheet, detached from the terminal that produced it, that is
indistinguishable from a measurement about two real Bitcoin entities. It is a
decision made by a synthetic engine on a synthetic chain against a synthetic
origin assignment. **The banner is in the wrong artifact** — it is on the
ephemeral output and absent from the durable one.

This is the highest-severity finding in the audit, and it is not hypothetical:
these CSVs are the ones a teammate opens six weeks from now.

**Fix:** `Provenance` (§4.6) as required columns on every written frame —
`source`, `synthetic`, `is_measurement`, `dataset_sha256`, `method`, `seed`.
`demo/api.py:76-104` already does exactly this per-object and refuses to render
without it; generalise that pattern rather than inventing a second one.

### 6.2 The production rule can be changed from the CLI and the change leaves no trace — **critical**

Detailed at §3.5. `cli.py:284-288`, `:367-369`, `:714-718` accept
`--min-pooled / --alpha / --min-effect`; none prints the value used; none
records it. Combined with §6.1, a blocked-merge count produced under a loosened
rule is unrecoverable from the artifact.

HANDOFF invariant 3 is currently enforced by discipline alone on three
commands. `demo/runner.py:48-79` shows the enforceable version
(`assert_production_rule`, hard failure) and `eval/phase33.py` shows the safe
version (no flags at all).

**Fix, smallest possible:** echo the effective config in every report header
and write it into every output file. Optionally gate non-default values behind
an explicit `--experimental-rule` flag so an override is a deliberate act.

### 6.3 `evidence_funnel` re-derives the production verdict — **high**

`evidence_funnel.py:244-250` reimplements the decision rule from recorded
columns. The comment at `:229-230` states the intent — *"Reproduce the
production rule exactly, through the production function, so the headline count
is not a re-derivation that could drift from it"* — and then the code
immediately does re-derive it. Two implementations of one rule; the comment is
already inconsistent with the code beneath it.

Under E1 this function cannot express the method at all and would report
confident wrong verdicts rather than failing.

### 6.4 `RiskScorer` does not exist, and its absence is the safest state — **note**

No default risk level exists to leak, because no scorer exists. When one is
built, two properties must hold from the first commit:

- `score()` returns `None` rather than a default "low" — an uncalibrated
  scorer must be able to say nothing (this is E3's logic applied at the alert
  layer);
- `Alert.confidence` is `str | None`, never a default `"medium"`.

An alert is the artifact an analyst acts on. It is the one place where a
displayed inference becomes an operational fact about a person.

### 6.5 The truth-isolation whitelist is manual, and new modules default to unchecked — **high**

`test_truth_isolation.py:20-40` lists `INFERENCE_MODULES` and
`EVALUATION_MODULES` by hand. Modules in **neither** list:

```
eval/evidence_funnel.py     builds an oracle, replays edges  -> inference
demo/runner.py              builds an oracle, clusters       -> inference
demo/audit.py               calls the oracle                 -> inference
demo/scenarios.py           writes ground truth              -> generator
network/worlds.py           writes ground truth              -> generator
network/reach_stress.py     writes ground truth              -> generator
network/synthetic.py        writes ground truth              -> generator
```

None currently violates the boundary — I checked. The defect is the *default*:
a module nobody adds to the list is silently exempt. Under a plugin
architecture, where new `EvidenceProvider` adapters are the expected growth
direction, an unchecked-by-default inference path is the wrong failure mode.

**Fix:** invert it. Enumerate `src/obsidianchain/**/*.py`, subtract the declared
evaluation and generator lists, and require the remainder to be truth-free.
A new module is then inference-by-default and must be *explicitly* excused.
~15 lines in the existing test file, no production change.

### 6.6 Two callers observe decisions through a private method — **medium**

`phase33.py:330` and `demo/runner.py:179` call
`forest._evaluate_cannot_link(root_a, root_b)`. Both then call `union()`, which
evaluates *again*. The recorded evidence and the applied decision come from two
separate invocations of a method that mutates counters (`constrained.py:243`
calls `note_evidence`).

Today both calls are deterministic and agree, so nothing is wrong. But the
*recorded* decision is not provably the *applied* one, and the counters are
double-incremented on the observed path. **Fix:** a public
`probe(a, b) -> EvidenceRecord` that is side-effect free, and have `union()`
return the record it acted on.

### 6.7 Correctly handled — do not "fix" these

- `blocked_merges` holds every refused merge, not a 20-item sample
  (`pipeline.py:118-129`) — a previous truncation silently computed every
  precision figure on ≤20 decisions.
- `TruthCategory.MIXED_ENTITY` (`phase33.py:138-140`) refuses a binary verdict
  where none exists, rather than manufacturing precision.
- Decision-level rather than edge-level scoring (`phase33.py:152-161`).
- `demo/` provenance enforcement (§2.6).

These are the project's four best boundary decisions and a refactor must not
erode any of them.

---

## 7. Minimal refactoring plan, ordered by risk

Ordered lowest-risk-first. **Stages 1-3 are worth doing whether or not E1 is
ever chosen** — they are boundary and correctness fixes, not architecture.
Stages 4-6 are the E1/E3 enablers and should wait until Phase 3.7 actually
selects a method.

`make test` (514 tests) is the gate after every stage. Nothing below changes a
frozen number; §8 says how that is verified.

| # | Change | Files | Risk | Reversible? | Blocks E1? |
|---|---|---|---|---|---|
| 1 | Provenance on every output file | `phase33.py`, `evidence_funnel.py`, `world_diagnostics.py`, `arrivals.py`, `cli.py` | **very low** — additive columns | yes | no |
| 2 | Echo + record the effective rule; invert the truth whitelist | `cli.py`, `test_truth_isolation.py` | **very low** | yes | no |
| 3 | Move `assert_datasets_compatible` into `build_oracle`; warn on single-observer | `separation.py`, `arrivals.py` | **low** — new failure mode on genuinely broken input | yes | no |
| 4 | `EvidenceRecord.statistics` becomes a mapping; delete `_production_verdict` | `separation.py`, `evidence_funnel.py`, `phase33.py`, `demo/*` | **medium** — touches 4 output schemas | yes, mechanical | **B2** |
| 5 | Extract `EvidenceState` protocol; `constrained.py` stops naming `GroupStats`; add public `probe()` | `constrained.py`, `separation.py`, `evidence_funnel.py`, `phase33.py`, `demo/runner.py` | **medium-high** — touches the core algorithm | yes, but retest hard | **B1, B4** |
| 6 | Split `build_oracle` into `NetworkSource` + arrival transform + accumulator | `separation.py`, `boundary.py`, new `sources/` | **medium** | yes | **B4** |
| 7 | `Verdict.INCONCLUSIVE` + fix the 6 branch points in §4.7 | `separation.py`, `constrained.py`, `phase33.py`, `evidence_funnel.py`, `demo/runner.py` | **medium** | yes | **B3** |
| 8 | `BlockchainSource`; `EllipticSource` wraps today's loader | `io/*`, new `sources/`, 6 consumers | **high** — widest blast radius | yes, but slow | no |
| 9 | `Clusterer` protocol; unhardwire `pipeline.run_fused` | `pipeline.py`, `worlds.py` | **low** | yes | no |

**Sequencing notes.**

- **Do 1-3 now.** They are the §6 findings, they are additive, and they are
  worth doing even if the architecture never changes. Stage 1 in particular
  should not wait for a methodology decision: the files being written today are
  the ones that will be misread later.
- **Stage 4 before stage 5.** Widening the record is mechanical and reversible;
  it also shrinks stage 5, because once nothing reads `.chi2` by name the
  provider swap touches fewer call sites.
- **Stage 5 is the real one.** It changes `ConstrainedUnionFind`, which
  Varun wrote and must explain to judges without notes. Do it with him, not
  for him. The union-by-rank logic (`constrained.py:196-202`) must not move at
  all — only the *type* of what `_stats` holds changes.
- **Stage 8 is deferrable indefinitely** unless a second chain dataset is
  actually planned. It is the largest change and it unblocks nothing on the
  E1 path.
- **Stage 9 is nearly free** and can be done any time; §2.2 shows the contract
  already holds.

**Verification protocol for stages 4-7** (the ones that touch inference):

1. `make test` — 514 pass, before and after.
2. `make run ARGS="cospend"` — largest cluster 14,885, coverage 34.60%.
3. `make run ARGS="world-experiment"` — zero blocked merges in all five
   regimes.
4. reach-stress decision table byte-identical to HANDOFF §4.
5. `make demo` — all five scenarios still reach their stated outcome
   (`demo/runner.py` fails the run itself if not).
6. Frozen dataset hash still `c405493d5bd904c0e17c840dca79386a`.

Steps 3-5 are the ones that catch a semantic change the unit tests miss.

---

## 8. Risks and tradeoffs

### 8.1 The main risk is that this audit gets over-implemented

Every interface in §4 is a cost paid now against a benefit that arrives only if
the corresponding swap happens. Of the four swaps in §4.8, exactly **one** is
scheduled (the October mainnet capture) and it currently costs **zero files**
if the capture is normalised to six columns.

The honest recommendation: **do stages 1-3 now, and hold 4-9 until Phase 3.7
picks a method.** Building `BlockchainSource` before a second chain exists is
speculative generality, and this codebase's own history (HANDOFF §7) is a list
of abstractions that were wrong until the data said otherwise.

### 8.2 Protocols are the right weight; a plugin framework is not

`typing.Protocol` is structural — no base class, no registration, no import at
definition time, zero runtime cost, and an existing concrete class satisfies it
without being modified. That is the whole reason `SeparationOracle` can become
`ArrivalSeparationOracle` and satisfy `EvidenceProvider` with a rename.

Anything heavier — an entry-point registry, a DI container, a config-driven
factory — would add moving parts to a system whose selling point is that it
runs offline on one laptop and is explicable to a judge without notes.

### 8.3 Widening `EvidenceRecord` weakens type checking

Moving `chi2/dof/p_value/effect` into `statistics: Mapping[str, float]` trades a
compile-time guarantee for runtime flexibility. A typo in a key becomes a
`KeyError` at render time rather than an editor squiggle.

Mitigation: providers declare their key set in `describe()`, and renderers
iterate rather than index. Accepted — the alternative is a record that cannot
represent the next method, which is the blocker being removed.

### 8.4 `INCONCLUSIVE` will make results look worse, and that is the point

E3 splits today's `NOT_SEPARATED` into "looked, found nothing" and "looked,
cannot decide". Every headline that currently reads *"the rule abstained
honestly"* will partly become *"the rule could not decide"*. That is a more
accurate statement of the same underlying behaviour and it should be presented
as a reporting improvement, not a regression. Deciding that in advance, in
writing, is cheaper than deciding it under a judge's question.

### 8.5 Stage 5 touches the one class that must stay explicable

`ConstrainedUnionFind.union` and `_evaluate_cannot_link` are Varun's, by the
working agreement, because he has to explain them without notes. An
`EvidenceState` protocol makes the class *more* abstract and therefore
marginally harder to explain — "it holds evidence it cannot inspect" is a
subtler sentence than "it holds pooled arrival statistics".

Tradeoff accepted only because the alternative is worse: rewriting that method
again at Phase 3.7. But the abstraction should be introduced with him, and the
docstring should keep a concrete example alongside the abstract contract.

### 8.6 What this audit deliberately did not do

- **No statistical redesign.** Whether E1 is the right method is not an
  architecture question and is not answered here.
- **No recommendation of E1.** §4.4 defines the minimum interface that makes
  *any* provider pluggable. It would equally admit a method that is not E1, and
  that neutrality is deliberate.
- **No calibration, no threshold tuning, no new features.**
- **No style refactoring.** Working code was read, not touched.

### 8.7 One thing worth flagging that is outside the audit's scope

`cli.py` is 1342 lines with 3 tests. It is the least-tested surface in the repo
and the one every stage above touches. Before stage 4, adding CLI smoke tests
that assert *output content* (not just exit codes) for `run`, `fusion-summary`,
`world-experiment` and `evidence-funnel` would convert the riskiest part of the
refactor into the cheapest. That is test work, not architecture, which is why
it is a note rather than a stage.

---

## Appendix — audit method

Every claim was checked against source. Evidence gathered by: full internal
import graph (28 modules); `grep` sweeps for `GroupStats` / `address_stats` /
`pooled_stats` / `combine` leakage, private-attribute reach-through, `Verdict`
consumers, chi-square field consumers, every write path (`to_csv` /
`to_parquet` / `write_text` / `savefig` / `json.dump`), hardcoded dataset
filenames, `world=` threading, and configuration loading; plus targeted reads
of `separation.py`, `constrained.py`, `pipeline.py`, `boundary.py`,
`arrivals.py`, `elliptic.py`, `synthetic.py`, `worlds.py`, `reach_stress.py`,
`phase33.py`, `evidence_funnel.py`, `unionfind.py`, `change.py`, `cli.py`,
`test_truth_isolation.py`, `test_cli.py`, `Dockerfile`, `Makefile`.

Two claims were verified by a second targeted query rather than trusted from
reading: that `ArrivalVectors.ranks` is never consumed by inference (§5.3), and
that `assert_datasets_compatible` is called from exactly one site (§5.1).
