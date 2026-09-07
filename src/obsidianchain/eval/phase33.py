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
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.eval import evidence_funnel
from obsidianchain.network import separation, worlds

FUNNEL_THRESHOLDS = (1, 2, 5, 10, 25)

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

    # TRUTH (evaluation only)
    blocks_cross_entity: int = 0
    blocks_same_entity: int = 0
    constraint_precision: float = float("nan")
    false_split_rate: float = float("nan")
    abstention_rate: float = float("nan")
    precision_testable: bool = False

    @property
    def n_scored_blocks(self) -> int:
        return self.blocks_cross_entity + self.blocks_same_entity


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


def run_regime(
    regime: worlds.Regime,
    graph,
    data_root: Path,
    config: separation.SeparationConfig,
) -> RegimeOutcome:
    """Run the unchanged engine on one world and score it."""
    from obsidianchain.cluster.pipeline import run_fused

    key = regime.value
    outcome = RegimeOutcome(regime=key, name=worlds.REGIME_NAMES[regime])
    processed = data_root / "processed"

    # --- INFERENCE SIDE: no truth is read anywhere below --------------
    oracle = separation.build_oracle(
        graph, processed_root=processed, data_root=data_root,
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
    outcome = _score_against_truth(outcome, fused, graph, data_root, key)
    return outcome


def _score_against_truth(
    outcome: RegimeOutcome, fused, graph, data_root: Path, key: str
) -> RegimeOutcome:
    """Score emitted separations against hidden entity ground truth.

    Called after inference has produced its output. The accessor name is
    deliberately conspicuous; if it ever appears inside an inference path,
    that result is invalid.
    """
    from obsidianchain.network import boundary

    truth = boundary.load_ground_truth_FOR_EVALUATION_ONLY(
        data_root / "processed", world=key
    )
    entity_map = pd.read_csv(
        data_root / "processed" / "worlds_truth" / key / "entity_map.csv"
    )

    # Address -> entity, via the baseline component the entity was assigned to.
    from obsidianchain.cluster.pipeline import run_clustering

    baseline = run_clustering(graph, "truth-alignment")
    lookup = entity_map.set_index("component_root")["true_entity_id"]
    entity_of_address = lookup.reindex(baseline.roots).to_numpy()

    cross = same = 0
    for blocked in fused.blocked_examples:
        entity_a = entity_of_address[blocked.a]
        entity_b = entity_of_address[blocked.b]
        if pd.isna(entity_a) or pd.isna(entity_b):
            continue
        if entity_a == entity_b:
            same += 1
        else:
            cross += 1

    outcome.blocks_cross_entity = cross
    outcome.blocks_same_entity = same
    scored = cross + same
    if scored:
        outcome.constraint_precision = cross / scored
    outcome.false_split_rate = (
        outcome.blocked / outcome.proposed if outcome.proposed else float("nan")
    )

    # Precision is only testable if the engine could ever have been asked to
    # separate two genuine entities.
    edge_entities_a = entity_of_address[graph.edges[:, 0]]
    edge_entities_b = entity_of_address[graph.edges[:, 1]]
    outcome.precision_testable = bool(
        np.nansum(edge_entities_a != edge_entities_b) > 0
    )
    return outcome


# ---- rendering --------------------------------------------------------

_W = 78


def format_experiment(
    outcomes: dict[str, RegimeOutcome], config: separation.SeparationConfig
) -> str:
    out: list[str] = []
    add = out.append
    order = [k for k in ("A", "B", "C", "D", "E") if k in outcomes]

    add("=" * _W)
    add("OBSIDIANCHAIN - PHASE 3.3  ENGINE ACROSS FIVE CONTROLLED WORLDS")
    add("=" * _W)
    add("  SYNTHETIC worlds. Demonstrates mechanism; validates nothing about")
    add("  Bitcoin. The engine is unchanged and the production rule is FIXED:")
    add(f"  min pooled {config.min_pooled_observations} per side, "
        f"alpha {config.alpha:.0e}, effect floor {config.min_effect}.")
    add("  No threshold was tuned per regime.")
    add("")

    add("-- REACH: decidable unions at each pooled threshold " + "-" * (_W - 53))
    add(f"  {'':<4}{'proposed':>11}" + "".join(f"{'>=' + str(t):>10}" for t in FUNNEL_THRESHOLDS))
    for key in order:
        o = outcomes[key]
        row = "".join(f"{o.decidable[t]:>10,}" for t in FUNNEL_THRESHOLDS)
        add(f"  {key:<4}{o.proposed:>11,}{row}")
    add("  (decidable = cleared the pooled gate AND has a computable statistic)")
    add("")

    add("-- SIGNAL: separated share vs pooled count " + "-" * (_W - 44))
    add(f"  {'':<4}" + "".join(f"{'>=' + str(t):>10}" for t in FUNNEL_THRESHOLDS)
        + "   direction")
    for key in order:
        o = outcomes[key]
        cells = "".join(
            f"{o.separated_share[t] * 100:>9.1f}%"
            if not np.isnan(o.separated_share[t]) else f"{'-':>10}"
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

    add("-- TRUTH: against hidden ground truth, EVALUATION ONLY " + "-" * (_W - 56))
    testable = any(o.precision_testable for o in outcomes.values())
    if not testable:
        add("  CONSTRAINT PRECISION IS NOT TESTABLE IN THIS EXPERIMENT.")
        add("")
        add("  Entities were assigned per baseline co-spend component, and a")
        add("  co-spend edge never crosses a component: zero of 274,313 edges")
        add("  connect two different entities. So the engine is never asked to")
        add("  separate two genuine entities, and any block it emits is a false")
        add("  split by construction rather than by failure.")
        add("")
        add("  The pre-registered predictions for B and E mention precision")
        add("  above null. That cannot be evaluated here, and reporting a")
        add("  precision figure would be misleading. What IS testable:")
        add("")
    add(f"  {'':<4}{'blocked':>9}{'cross-entity':>14}{'same-entity':>13}"
        f"{'false-split rate':>18}{'abstention':>12}")
    for key in order:
        o = outcomes[key]
        rate = (
            f"{o.false_split_rate * 100:>17.4f}%"
            if not np.isnan(o.false_split_rate) else f"{'-':>18}"
        )
        abst = (
            f"{o.abstention_rate * 100:>11.2f}%"
            if not np.isnan(o.abstention_rate) else f"{'-':>12}"
        )
        add(f"  {key:<4}{o.blocked:>9,}{o.blocks_cross_entity:>14,}"
            f"{o.blocks_same_entity:>13,}{rate}{abst}")
    add("  false-split rate = blocked / proposed; every block is same-entity,")
    add("  so this is the full picture of harm the engine could cause here.")
    add("")

    add("-- PRE-REGISTERED PREDICTIONS vs OUTCOME " + "-" * (_W - 42))
    for key in order:
        o = outcomes[key]
        add(f"  {key}  predicted: {PREDICTIONS[key]}")
        add(f"     observed:  blocked {o.blocked:,}   "
            f"abstention {o.abstention_rate * 100:.2f}%   "
            f"signal {o.signal_direction.split()[0]}")
        add(f"     {_verdict_line(o)}")
    add("")

    critical = outcomes.get("D")
    add("-- REGIME D, THE CRITICAL ONE " + "-" * (_W - 31))
    if critical is None:
        add("  not run")
    else:
        add(f"  false splits            {critical.blocked:>12,}")
        add(f"  false-split rate        "
            f"{critical.false_split_rate * 100:>11.4f}%")
        add(f"  within-entity sub-centroid separation was measured at 6.48 SE,")
        add(f"  so the engine HAD the evidence to split this entity.")
        add("")
        if critical.blocked == 0:
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
            add(f"  RESULT: {critical.blocked:,} FALSE SPLITS. The production")
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
        ok = outcome.blocked == 0
        return f"{'CONFIRMED' if ok else 'NOT CONFIRMED'}: no spurious separation"
    if key == "B":
        rises = outcome.signal_direction.startswith("RISES")
        return (
            f"{'partially confirmed' if rises else 'NOT CONFIRMED'}: "
            f"signal {outcome.signal_direction.split()[0]}; precision untestable"
        )
    if key == "C":
        return (
            f"{'CONFIRMED' if outcome.blocked == 0 else 'NOT CONFIRMED'}: "
            f"abstained rather than separated an ambiguous pair"
        )
    if key == "D":
        return (
            f"{'CONFIRMED' if outcome.blocked == 0 else 'FAILED'}: "
            f"false splits = {outcome.blocked:,}"
        )
    if key == "E":
        return "precision untestable; separation behaviour reported above"
    return ""


def to_frame(outcomes: dict[str, RegimeOutcome]) -> pd.DataFrame:
    rows = []
    for key in ("A", "B", "C", "D", "E"):
        outcome = outcomes.get(key)
        if outcome is None:
            continue
        row = {
            k: v for k, v in vars(outcome).items()
            if not isinstance(v, dict)
        }
        for threshold in FUNNEL_THRESHOLDS:
            row[f"decidable_ge{threshold}"] = outcome.decidable.get(threshold, 0)
            row[f"sep_share_ge{threshold}"] = outcome.separated_share.get(
                threshold, float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)
