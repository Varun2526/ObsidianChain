"""Phase 3.3: the unchanged Phase 3 engine across all five controlled worlds.

The production rule is fixed. ``SeparationConfig`` defaults - 25 pooled
observations per side, alpha 1e-4, effect floor 0.05 - apply identically to
every regime. Tuning per world would answer a different question: it would
show that some rule can fit each world, which nobody doubts, rather than
whether one rule behaves correctly across all of them.

Four things are reported per regime.

``REACH``
    Decidable unions at each pooled threshold - the Phase 3.1 funnel,
    re-run on this world's observations.

``SIGNAL``
    Separated share against pooled count. Rising means evidence is real but
    starved; falling means the apparent separations are small-sample noise.
    This is the diagnostic that told us the frozen dataset was noise.

``DECISION``
    What the engine actually did: merges blocked, clusters contested,
    unions abstained.

``TRUTH``
    Scored against hidden ground truth. **This module is the only place
    truth is read.** Inference has already finished by the time it is
    touched, and the accessor is the shouted one so misuse is visible in a
    grep.

A limitation of the experiment, discovered before running it
------------------------------------------------------------
Entities were assigned per baseline co-spend component, and a co-spend edge
never crosses a component. Measured: **zero of 274,313 edges connect two
different entities.** So the engine is never asked to separate two genuine
entities, and every block it emits is necessarily a false split.

Consequences, stated plainly:

* **constraint precision is not testable here.** Of separations emitted, the
  share between genuinely different entities is 0 by construction, not by
  failure. Reporting it as a performance figure would be misleading.
* **the false-split rate is fully testable**, and equals blocked over
  proposed. That is exactly the regime D question, which is the one that
  matters most: a system that splits a legitimate multi-origin entity sends
  an investigator after two halves of one target.
* **abstention is fully testable.**

Testing precision would need entities defined finer than co-spend
components, so that some edges are genuinely wrong. That is a change to the
world design, not to the engine, and it is not made here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.eval import evidence_funnel
from obsidianchain.network import separation, worlds
from obsidianchain.network.separation import Verdict

FUNNEL_THRESHOLDS = (1, 2, 5, 10, 25)

#: Share of a network dataset's transactions that must exist in the chain
#: before the pair is considered the same experiment. A correct pairing sits
#: at 1.0; a mismatched one measured 0.00095 (57 coincidentally equal integer
#: ids out of 60,000), which is worse than zero because those 57 are
#: different transactions that happen to share a number.
MIN_TXID_OVERLAP = 0.5


class DatasetMismatchError(RuntimeError):
    """Raised when a chain and a network dataset are not the same experiment."""


def assert_datasets_compatible(
    chain_root: Path, network_root: Path, world: str
) -> float:
    """Fail loudly when a chain and a network dataset do not belong together.

    Without this the failure is silent, not loud. ``build_oracle`` maps
    observations onto the chain and drops whatever does not resolve, so an
    incompatible pair yields an oracle with no statistics, an abstention rate
    of 100%, and a report that looks like a clean negative result. That is
    the same shape of failure as an evaluation that cannot come out any way
    but "pass", and it has bitten this project twice already.

    Returns the resolved fraction so callers can report it.
    """
    from obsidianchain.io import elliptic
    from obsidianchain.network import boundary

    observations = boundary.load_observations(
        Path(network_root) / "processed", world=world
    )
    observed = set(observations["txid"].unique().tolist())
    if not observed:
        raise DatasetMismatchError(
            f"network dataset {network_root} regime {world} has no transactions"
        )

    chain = set(elliptic.load_input_edges(chain_root)["txId"].unique().tolist())
    resolved = len(observed & chain) / len(observed)

    if resolved < MIN_TXID_OVERLAP:
        raise DatasetMismatchError(
            f"chain {chain_root} and network dataset {network_root} "
            f"(regime {world}) are not the same experiment: only "
            f"{resolved:.4%} of {len(observed):,} observed transactions exist "
            f"in the chain's {len(chain):,}. Pairing them would produce an "
            f"oracle with no evidence and a silent 100% abstention. Point "
            f"--chain-root at the blockchain this network data was generated "
            f"over."
        )
    return resolved

class TruthCategory(str, Enum):
    """What ground truth says about the two sides of one decision.

    Only the two PURE categories admit a binary right/wrong answer. A
    boundary whose sides each contain several entities has no correct
    merge decision: some address pairs across it belong together and some
    do not, so scoring it as either a hit or a false split manufactures a
    precision number out of an unanswerable question.
    """

    PURE_SAME_ENTITY = "PURE_SAME_ENTITY"
    """Both sides are one and the same entity. Blocking is a false split."""

    PURE_CROSS_ENTITY = "PURE_CROSS_ENTITY"
    """Each side is one entity, and they differ. Blocking is correct."""

    MIXED_ENTITY = "MIXED_ENTITY"
    """At least one side holds several entities. Not binary-scorable."""

    UNRESOLVED = "UNRESOLVED"
    """Truth is missing for at least one side. Not scorable at all."""


BINARY_CATEGORIES = (
    TruthCategory.PURE_SAME_ENTITY,
    TruthCategory.PURE_CROSS_ENTITY,
)


@dataclass
class Decision:
    """One proposed merge between two components - the unit of judgement.

    A component boundary can be re-proposed by many address edges; in the
    reach-stress fixture one boundary was proposed 166 times. Scoring the
    edges instead of the boundary inflated a single decision into 166
    "false splits". ``proposing_edges`` keeps that provenance without
    letting it into the metric.
    """

    decision_id: int
    component_a: int
    component_b: int
    evidence_state: str
    chi2: float
    p_value: float
    effect_size: float
    pooled_n_a: int
    pooled_n_b: int
    proposing_edges: int = 1
    blocked: bool = False
    truth_category: str = TruthCategory.UNRESOLVED.value
    entities_a: tuple = ()
    entities_b: tuple = ()

    @property
    def binary_scorable(self) -> bool:
        return self.truth_category in {c.value for c in BINARY_CATEGORIES}

    @property
    def is_false_split(self) -> bool:
        return self.blocked and (
            self.truth_category == TruthCategory.PURE_SAME_ENTITY.value
        )

    @property
    def is_correct_separation(self) -> bool:
        return self.blocked and (
            self.truth_category == TruthCategory.PURE_CROSS_ENTITY.value
        )


def classify_truth(entities_a: set, entities_b: set) -> TruthCategory:
    """Categorise a boundary from the entities present on each side."""
    if not entities_a or not entities_b:
        return TruthCategory.UNRESOLVED
    if len(entities_a) > 1 or len(entities_b) > 1:
        return TruthCategory.MIXED_ENTITY
    return (
        TruthCategory.PURE_SAME_ENTITY
        if entities_a == entities_b
        else TruthCategory.PURE_CROSS_ENTITY
    )


PREDICTIONS = {
    "A": "no separation, abstention high -> confirms no spurious signal",
    "B": "separation rises with pooling, precision above null",
    "C": "abstention rises, separation does NOT -> ambiguity handled",
    "D": "false splits near zero -> THE CRITICAL ONE",
    "E": "partial separation, precision between A and B",
}


@dataclass
class RegimeOutcome:
    regime: str = ""
    name: str = ""

    # REACH
    proposed: int = 0
    redundant: int = 0
    cleared: dict[int, int] = field(default_factory=dict)
    decidable: dict[int, int] = field(default_factory=dict)

    # SIGNAL
    separated_share: dict[int, float] = field(default_factory=dict)
    separated_count: dict[int, int] = field(default_factory=dict)
    signal_direction: str = ""

    # DECISION
    blocked: int = 0
    contested: int = 0
    evaluated: int = 0
    abstained: int = 0
    clusters: int = 0
    largest: int = 0
    coverage: float = 0.0

    # TRUTH (evaluation only) - DECISION level, one row per boundary
    unique_decisions: int = 0
    decidable_decisions: int = 0
    state_counts: dict = field(default_factory=dict)
    blocked_by_truth: dict = field(default_factory=dict)
    false_split_decisions: int = 0
    correct_separation_decisions: int = 0
    binary_scorable_blocked: int = 0
    abstention_rate: float = float("nan")
    decisions: list = field(default_factory=list)

    @property
    def decision_precision(self) -> float:
        """Correct separations over binary-scorable blocks. NaN when none."""
        if self.binary_scorable_blocked == 0:
            return float("nan")
        return self.correct_separation_decisions / self.binary_scorable_blocked




def _signal_direction(shares: dict[int, float]) -> str:
    """Does the separated share rise or fall as pooling increases?"""
    points = [(t, s) for t, s in sorted(shares.items()) if not np.isnan(s)]
    if len(points) < 2:
        return "undetermined"
    first, last = points[0][1], points[-1][1]
    if last > first + 0.05:
        return "RISES  (evidence real but starved)"
    if last < first - 0.05:
        return "FALLS  (apparent separations are small-sample noise)"
    return "FLAT   (no trend)"


def load_entity_truth_FOR_EVALUATION_ONLY(
    processed_root: Path, world: str, n_addresses: int
) -> np.ndarray:
    """Address-to-entity ground truth. Evaluation only, after inference.

    Named to shout, like the accessor in :mod:`obsidianchain.network.boundary`,
    so that a call from inside an inference path is obvious in review and in a
    grep. The entity map is ground truth every bit as much as the origin map:
    it is generated, never observed, and a real capture supplies neither.

    Returns an array indexed by address code, -1 where truth is absent.
    """
    path = Path(processed_root) / "worlds_truth" / world / "entity_assignment.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; regenerate the world before scoring it."
        )
    assignment = pd.read_csv(path)
    entity_of_address = np.full(n_addresses, -1, dtype=np.int64)
    entity_of_address[assignment["address_code"].to_numpy()] = assignment[
        "true_entity_id"
    ].to_numpy()
    return entity_of_address


def replay_decisions(graph, oracle, entity_of_address) -> list[Decision]:
    """Replay the edges and record ONE row per component boundary.

    The engine itself is untouched: this uses the same
    :class:`ConstrainedUnionFind` and the same oracle, and simply observes
    what happens rather than changing it. Component membership is tracked
    alongside so each boundary can be attributed to the entities actually
    present on either side - which the engine has no reason to know and must
    never be told.
    """
    from obsidianchain.cluster.constrained import ConstrainedUnionFind

    forest = ConstrainedUnionFind(graph.n_addresses, oracle=oracle)
    members: dict[int, list[int]] = {i: [i] for i in range(graph.n_addresses)}
    seen: dict[tuple[int, int], Decision] = {}
    next_id = 0

    for a, b in graph.edges.tolist():
        root_a, root_b = forest.find(a), forest.find(b)
        if root_a == root_b:
            continue  # already connected: no decision to make

        key = (min(root_a, root_b), max(root_a, root_b))
        if key in seen:
            # The same boundary re-proposed by another edge. Provenance, not
            # a second decision.
            seen[key].proposing_edges += 1
            forest.union(a, b)
            continue

        evidence = forest._evaluate_cannot_link(root_a, root_b)
        side_a, side_b = list(members[root_a]), list(members[root_b])
        merged = forest.union(a, b)

        entities_a = {
            int(entity_of_address[m])
            for m in side_a
            if entity_of_address[m] >= 0
        }
        entities_b = {
            int(entity_of_address[m])
            for m in side_b
            if entity_of_address[m] >= 0
        }
        decision = Decision(
            decision_id=next_id,
            component_a=int(key[0]),
            component_b=int(key[1]),
            evidence_state=evidence.verdict.value,
            chi2=float(evidence.chi2),
            p_value=float(evidence.p_value),
            effect_size=float(evidence.effect),
            pooled_n_a=int(evidence.n_a),
            pooled_n_b=int(evidence.n_b),
            blocked=not merged,
            truth_category=classify_truth(entities_a, entities_b).value,
            entities_a=tuple(sorted(entities_a)),
            entities_b=tuple(sorted(entities_b)),
        )
        seen[key] = decision
        next_id += 1

        if merged:
            survivor = forest.find(a)
            absorbed = root_b if survivor == root_a else root_a
            members[survivor] = side_a + side_b
            if absorbed != survivor:
                members.pop(absorbed, None)

    return list(seen.values())


def run_regime(
    regime: worlds.Regime,
    graph,
    chain_root: Path,
    config: separation.SeparationConfig,
    network_root: Path | None = None,
) -> RegimeOutcome:
    """Run the unchanged engine on one world and score it.

    ``chain_root`` holds ``raw/`` - the blockchain the graph was built from.
    ``network_root`` holds ``processed/worlds/`` and ``processed/worlds_truth/``.
    They are the same directory for the normal A-E worlds and differ only
    when a self-contained fixture supplies both, which is why they are named
    separately rather than derived from one another: a mismatched pair fails
    silently, so the pairing has to be explicit.
    """
    from obsidianchain.cluster.pipeline import run_fused

    key = regime.value
    network_root = Path(network_root) if network_root is not None else Path(chain_root)
    chain_root = Path(chain_root)
    outcome = RegimeOutcome(regime=key, name=worlds.REGIME_NAMES[regime])
    processed = network_root / "processed"

    assert_datasets_compatible(chain_root, network_root, key)

    # --- INFERENCE SIDE: no truth is read anywhere below --------------
    oracle = separation.build_oracle(
        graph, processed_root=processed, data_root=chain_root,
        config=config, world=key,
    )

    funnel = evidence_funnel.build_funnel(
        graph, oracle, thresholds=FUNNEL_THRESHOLDS, production_config=config
    )
    outcome.proposed = funnel.n_proposed
    outcome.redundant = funnel.n_redundant
    for threshold in FUNNEL_THRESHOLDS:
        outcome.cleared[threshold] = int(len(funnel.cleared(threshold)))
        outcome.decidable[threshold] = int(len(funnel.decidable(threshold)))
        verdicts = funnel.verdicts_at(threshold)
        decidable = verdicts["SEPARATED"] + verdicts["NOT_SEPARATED"]
        outcome.separated_count[threshold] = verdicts["SEPARATED"]
        outcome.separated_share[threshold] = (
            verdicts["SEPARATED"] / decidable if decidable else float("nan")
        )
    outcome.signal_direction = _signal_direction(outcome.separated_share)

    fused = run_fused(graph, oracle, label=f"fused-{key}")
    outcome.blocked = fused.blocked
    outcome.contested = fused.contested
    outcome.evaluated = fused.evaluated
    outcome.abstained = fused.abstained
    outcome.clusters = fused.run.n_clusters
    outcome.largest = fused.run.largest
    outcome.coverage = fused.run.coverage
    total = fused.evaluated + fused.abstained
    outcome.abstention_rate = fused.abstained / total if total else float("nan")

    # --- EVALUATION SIDE: inference has finished; truth may be read ---
    outcome = _score_against_truth(
        outcome, fused, graph, network_root, key, oracle
    )
    return outcome


def _score_against_truth(
    outcome: RegimeOutcome, fused, graph, data_root: Path, key: str, oracle
) -> RegimeOutcome:
    """Score DECISIONS against hidden entity truth, one row per boundary.

    Called after inference has produced its output. Two things this must not
    do, both learned the hard way: score address edges rather than component
    boundaries, which turned one decision into 166; and force a mixed-entity
    boundary into a binary verdict, which manufactures a precision figure for
    a question that has no answer.
    """
    from obsidianchain.network import boundary

    entity_of_address = load_entity_truth_FOR_EVALUATION_ONLY(
        data_root / "processed", key, graph.n_addresses
    )

    decisions = replay_decisions(graph, oracle, entity_of_address)
    outcome.decisions = decisions
    outcome.unique_decisions = len(decisions)

    states = collections.Counter(d.evidence_state for d in decisions)
    outcome.state_counts = dict(states)
    outcome.decidable_decisions = sum(
        count for state, count in states.items() if state != Verdict.NO_EVIDENCE.value
    )

    blocked = [d for d in decisions if d.blocked]
    outcome.blocked_by_truth = dict(
        collections.Counter(d.truth_category for d in blocked)
    )
    outcome.false_split_decisions = sum(1 for d in blocked if d.is_false_split)
    outcome.correct_separation_decisions = sum(
        1 for d in blocked if d.is_correct_separation
    )
    outcome.binary_scorable_blocked = sum(1 for d in blocked if d.binary_scorable)
    return outcome


# ---- rendering --------------------------------------------------------

_W = 78


def format_experiment(
    outcomes: dict[str, RegimeOutcome],
    config: separation.SeparationConfig,
    fixture: str = "controlled-world",
    chain_description: str = "Elliptic++",
) -> str:
    out: list[str] = []
    add = out.append
    order = [k for k in ("A", "B", "C", "D", "E") if k in outcomes]
    reach_stress = fixture == "reach-stress"

    add("=" * _W)
    if reach_stress:
        add("OBSIDIANCHAIN - PHASE 3.3  REACH-STRESS MECHANISM TEST")
    else:
        add("OBSIDIANCHAIN - PHASE 3.3  ENGINE ACROSS FIVE CONTROLLED WORLDS")
    add("=" * _W)
    if reach_stress:
        add("  *** REACH-STRESS FIXTURE - NOT AN ELLIPTIC++ RESULT ***")
        add("")
        add("  This runs on a SYNTHETIC blockchain built to be dense enough")
        add("  that the production threshold is reachable at all. It answers")
        add("  one question only: given sufficient evidence, what does the")
        add("  unchanged rule do? It says nothing about how often the rule can")
        add("  act on real data - the normal A-E worlds answer that, and their")
        add(f"  answer was 47 of 253,429 unions.")
        add("")
        add("  Numbers here must NOT be quoted as Elliptic++ performance, and")
        add("  the cluster counts below are this fixture's, not the 569,513 of")
        add("  the Elliptic++ baseline.")
    else:
        add("  SYNTHETIC worlds over the Elliptic++ co-spend graph.")
        add("  Demonstrates mechanism; validates nothing about Bitcoin.")
    add("")
    add(f"  chain: {chain_description}")
    add("  The engine is unchanged and the production rule is FIXED:")
    add(f"  min pooled {config.min_pooled_observations} per side, "
        f"alpha {config.alpha:.0e}, effect floor {config.min_effect}.")
    add("  No threshold was tuned per regime.")
    add("")

    add("-- REACH: decidable unions at each pooled threshold " + "-" * (_W - 53))
    add(f"  {'':<4}{'proposed':>11}" + "".join(f"{'>=' + str(t):>10}" for t in FUNNEL_THRESHOLDS))
    for key in order:
        o = outcomes[key]
        row = "".join(f"{o.decidable.get(t, 0):>10,}" for t in FUNNEL_THRESHOLDS)
        add(f"  {key:<4}{o.proposed:>11,}{row}")
    add("  (decidable = cleared the pooled gate AND has a computable statistic)")
    add("")

    add("-- SIGNAL: separated share vs pooled count " + "-" * (_W - 44))
    add(f"  {'':<4}" + "".join(f"{'>=' + str(t):>10}" for t in FUNNEL_THRESHOLDS)
        + "   direction")
    for key in order:
        o = outcomes[key]
        cells = "".join(
            f"{o.separated_share.get(t, float('nan')) * 100:>9.1f}%"
            if not np.isnan(o.separated_share.get(t, float("nan")))
            else f"{'-':>10}"
            for t in FUNNEL_THRESHOLDS
        )
        add(f"  {key:<4}{cells}   {o.signal_direction}")
    add("  RISES = evidence real but starved.  FALLS = small-sample noise.")
    add("")

    add("-- DECISION: what the engine did " + "-" * (_W - 34))
    add(f"  {'':<4}{'clusters':>11}{'largest':>9}{'coverage':>10}"
        f"{'blocked':>9}{'contested':>11}{'evaluated':>11}{'abstained':>11}")
    for key in order:
        o = outcomes[key]
        add(f"  {key:<4}{o.clusters:>11,}{o.largest:>9,}{o.coverage:>9.2f}%"
            f"{o.blocked:>9,}{o.contested:>11,}{o.evaluated:>11,}"
            f"{o.abstained:>11,}")
    add("")

    add("-- TRUTH: DECISIONS, not edges " + "-" * (_W - 33))
    add("  One row per component boundary. A boundary re-proposed by many")
    add("  address edges is one decision: counting the edges turned a single")
    add("  refusal into 166 in an earlier run of this same fixture.")
    add("")
    add(f"  {'':<4}{'unique':>9}{'decidable':>11}{'NO_EVID':>10}"
        f"{'SEPARATED':>11}{'NOT_SEP':>10}")
    for key in order:
        o = outcomes[key]
        add(f"  {key:<4}{o.unique_decisions:>9,}{o.decidable_decisions:>11,}"
            f"{o.state_counts.get('NO_EVIDENCE', 0):>10,}"
            f"{o.state_counts.get('SEPARATED', 0):>11,}"
            f"{o.state_counts.get('NOT_SEPARATED', 0):>10,}")
    add("")

    add("-- BLOCKED DECISIONS by truth category " + "-" * (_W - 40))
    add("  Only the two PURE categories admit a right/wrong answer. A mixed")
    add("  boundary has entities on both sides, so some address pairs across")
    add("  it belong together and some do not - there is no correct merge,")
    add("  and scoring it either way would invent a precision number.")
    add("")
    add(f"  {'':<4}{'PURE_SAME':>12}{'PURE_CROSS':>12}{'MIXED':>9}"
        f"{'UNRESOLVED':>12}{'total':>8}")
    for key in order:
        o = outcomes[key]
        by = o.blocked_by_truth
        add(f"  {key:<4}"
            f"{by.get('PURE_SAME_ENTITY', 0):>12,}"
            f"{by.get('PURE_CROSS_ENTITY', 0):>12,}"
            f"{by.get('MIXED_ENTITY', 0):>9,}"
            f"{by.get('UNRESOLVED', 0):>12,}"
            f"{sum(by.values()):>8,}")
    add("")

    add("-- THE SCIENTIFIC QUESTION " + "-" * (_W - 28))
    add("  When evidence is strong enough to act, does it distinguish")
    add("  genuinely different ENTITIES, or merely different NETWORK")
    add("  BEHAVIOUR? Only binary-scorable decisions can answer that.")
    add("")
    add(f"  {'':<4}{'false splits':>14}{'correct sep':>13}"
        f"{'binary-scorable':>17}{'precision':>12}")
    for key in order:
        o = outcomes[key]
        precision = (
            f"{o.decision_precision:>11.1%}"
            if o.binary_scorable_blocked else f"{'n/a':>12}"
        )
        add(f"  {key:<4}{o.false_split_decisions:>14,}"
            f"{o.correct_separation_decisions:>13,}"
            f"{o.binary_scorable_blocked:>17,}{precision}")
    add("  false split  = PURE_SAME_ENTITY boundary that was blocked")
    add("  correct sep  = PURE_CROSS_ENTITY boundary that was blocked")
    add("")

    add("-- PRE-REGISTERED PREDICTIONS vs OUTCOME " + "-" * (_W - 42))
    for key in order:
        o = outcomes[key]
        add(f"  {key}  predicted: {PREDICTIONS[key]}")
        add(f"     observed:  decisions blocked "
            f"{sum(o.blocked_by_truth.values()):,}   "
            f"false splits {o.false_split_decisions:,}   "
            f"correct sep {o.correct_separation_decisions:,}")
        add(f"     {_verdict_line(o)}")
    add("")

    critical = outcomes.get("D")
    add("-- REGIME D, THE CRITICAL ONE " + "-" * (_W - 31))
    if critical is None:
        add("  not run")
    else:
        add(f"  false split DECISIONS   {critical.false_split_decisions:>12,}")
        add(f"  correct separations     {critical.correct_separation_decisions:>12,}")
        add(f"  mixed (unscorable)      "
            f"{critical.blocked_by_truth.get('MIXED_ENTITY', 0):>12,}")
        add(f"  block EVENTS (edges)    {critical.blocked:>12,}"
            f"   provenance only, not decisions")
        add(f"  within-entity sub-centroid separation was measured at 6.48 SE,")
        add(f"  so the engine HAD the evidence to split this entity.")
        add("")
        if critical.false_split_decisions == 0:
            add("  RESULT: zero false splits. The engine did not split a")
            add("  legitimate multi-origin entity even though the arrival")
            add("  vectors of its three origins are 6.5 standard errors apart.")
            add("  The pooled minimum is what prevented it - reach never")
            add("  extended far enough for the split to be proposed.")
            add("")
            add("  Read this carefully: the correct outcome was reached for a")
            add("  reason unrelated to entity awareness. The rule has no")
            add("  concept of an entity; it abstained because evidence was")
            add("  scarce, not because it recognised one owner behind three")
            add("  origins. On a dataset with more reach the same rule could")
            add("  well split D, and that must be re-tested before any claim")
            add("  that the design is safe against false splits.")
        else:
            add(f"  RESULT: {critical.false_split_decisions:,} FALSE SPLIT "
                f"DECISION(S). The production")
            add("  rule split a legitimate multi-origin entity. This is worse")
            add("  than never separating anything: a false split sends an")
            add("  investigator after two halves of one target.")
            add("")
            add("  THE PRODUCTION RULE IS WRONG for multi-origin entities and")
            add("  must not be shipped in this form. It needs evidence that an")
            add("  entity may legitimately span several origins - which is")
            add("  precisely what a cannot-link rule built on origin")
            add("  dissimilarity cannot express on its own.")
    add("=" * _W)
    return "\n".join(out)


def _verdict_line(outcome: RegimeOutcome) -> str:
    """One-line read of whether the prediction held, where it is testable."""
    key = outcome.regime
    if key == "A":
        ok = outcome.false_split_decisions == 0
        return f"{'CONFIRMED' if ok else 'NOT CONFIRMED'}: no false split"
    if key == "B":
        rises = outcome.signal_direction.startswith("RISES")
        return (
            f"{'partially confirmed' if rises else 'NOT CONFIRMED'}: "
            f"signal {outcome.signal_direction.split()[0]}; precision untestable"
        )
    if key == "C":
        return (
            f"{'CONFIRMED' if outcome.false_split_decisions == 0 else 'NOT CONFIRMED'}: "
            f"no false split on an ambiguous boundary"
        )
    if key == "D":
        return (
            f"{'CONFIRMED' if outcome.false_split_decisions == 0 else 'FAILED'}: "
            f"false split decisions = {outcome.false_split_decisions:,}"
        )
    if key == "E":
        return "precision untestable; separation behaviour reported above"
    return ""


def decisions_to_frame(outcomes: dict[str, RegimeOutcome]) -> pd.DataFrame:
    """Every decision, one row each, with its evidence and truth category."""
    rows = []
    for key, outcome in outcomes.items():
        for decision in outcome.decisions:
            row = {"regime": key}
            row.update(
                {k: v for k, v in vars(decision).items() if k != "decisions"}
            )
            row["entities_a"] = ";".join(str(e) for e in decision.entities_a)
            row["entities_b"] = ";".join(str(e) for e in decision.entities_b)
            rows.append(row)
    return pd.DataFrame(rows)


def to_frame(outcomes: dict[str, RegimeOutcome]) -> pd.DataFrame:
    rows = []
    for key in ("A", "B", "C", "D", "E"):
        outcome = outcomes.get(key)
        if outcome is None:
            continue
        row = {
            k: v for k, v in vars(outcome).items()
            if not isinstance(v, (dict, list))
        }
        row["decision_precision"] = outcome.decision_precision
        for category in TruthCategory:
            row[f"blocked_{category.value}"] = outcome.blocked_by_truth.get(
                category.value, 0
            )
        for state in ("NO_EVIDENCE", "SEPARATED", "NOT_SEPARATED"):
            row[f"state_{state}"] = outcome.state_counts.get(state, 0)
        for threshold in FUNNEL_THRESHOLDS:
            row[f"decidable_ge{threshold}"] = outcome.decidable.get(threshold, 0)
            row[f"sep_share_ge{threshold}"] = outcome.separated_share.get(
                threshold, float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)
