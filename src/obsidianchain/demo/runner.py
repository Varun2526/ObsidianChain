"""Run the unchanged engine over the demonstration fixture.

Nothing in this module is a new algorithm. It builds the oracle through
:func:`obsidianchain.network.separation.build_oracle`, clusters with
:class:`obsidianchain.cluster.constrained.ConstrainedUnionFind`, and reads
back what happened. The production :class:`SeparationConfig` is used exactly
as it stands - 25 pooled per side, alpha 1e-4, effect floor 0.05 - and
:func:`assert_production_rule` fails the run if it has been edited, because a
demonstration run against a loosened rule would show a mechanism that is not
the one shipping.

Two passes, because the contrast is the point
---------------------------------------------
``chain-only`` is plain co-spend union-find: it merges everything the chain
proposes, which is the behaviour the constraint layer exists to improve on.
``fused`` is the same edges in the same order with the veto able to fire. The
two differ in exactly one thing - whether the oracle is consulted - so any
difference in their output is attributable to the constraints and not to edge
ordering.

Self-check
----------
The per-decision replay and the ordinary :func:`run_fused` pipeline are run
independently and their blocked counts compared. Four confident-but-wrong
numbers have been caught in this project by cross-checking a measurement
against a second path to it, so the demonstration does the same rather than
trusting one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from obsidianchain.cluster.constrained import ConstrainedUnionFind
from obsidianchain.cluster import replay
from obsidianchain.cluster.pipeline import run_clustering, run_fused
from obsidianchain.demo import api, audit, scenarios
from obsidianchain.demo.scenarios import DemoOutcome
from obsidianchain.io import elliptic
from obsidianchain.network import boundary, separation
from obsidianchain.network.separation import SeparationConfig, Verdict

#: The production rule, as of the last time this was checked against
#: :class:`SeparationConfig`. Held here as literals so that loosening the rule
#: to make a demonstration prettier breaks the demonstration instead.
PRODUCTION_RULE = {
    "min_pooled_observations": 25,
    "min_observer_observations": 5,
    "alpha": 1e-4,
    "min_effect": 0.05,
}


class DemoRuleError(RuntimeError):
    """Raised when the demonstration is about to run against a changed rule."""


class DemoExpectationError(RuntimeError):
    """Raised when a scenario stops behaving as it is described."""


def assert_production_rule(config: SeparationConfig) -> None:
    """Fail unless ``config`` is the production rule, unmodified."""
    actual = {
        "min_pooled_observations": config.min_pooled_observations,
        "min_observer_observations": config.min_observer_observations,
        "alpha": config.alpha,
        "min_effect": config.min_effect,
    }
    if actual != PRODUCTION_RULE:
        raise DemoRuleError(
            f"the demonstration must run against the production rule.\n"
            f"  expected {PRODUCTION_RULE}\n"
            f"  got      {actual}\n"
            f"Tuning the rule to make a scenario fire would demonstrate a "
            f"different system."
        )


@dataclass
class DecisionRecord:
    """One proposed union, and what the engine did with it."""

    scenario: str
    address_a: str
    address_b: str
    side_a: int
    """Members on side A at decision time."""
    side_b: int
    verdict: str
    pooled_a: int
    pooled_b: int
    chi2: float
    p_value: float
    effect: float
    reason: str
    merged: bool


@dataclass
class ScenarioResult:
    """Everything the demonstration reports about one scenario."""

    spec: scenarios.ScenarioSpec
    outcome: DemoOutcome
    verdict: str
    decisions: list[DecisionRecord] = field(default_factory=list)
    pooled: dict[str, int] = field(default_factory=dict)
    observations_seen: int = 0
    usable_observations: int = 0
    contradiction: audit.Contradiction | None = None
    contested: bool = False
    chain_only_components: int = 0
    fused_components: int = 0

    @property
    def matches_expectation(self) -> bool:
        return (
            self.outcome is self.spec.expected_outcome
            and self.verdict == self.spec.expected_verdict
        )


@dataclass
class DemoRun:
    """The whole demonstration: fixture, engine output, and the scenarios."""

    results: list[ScenarioResult]
    manifest: dict
    oracle: separation.SeparationOracle
    config: SeparationConfig
    chain_only_clusters: int
    fused_clusters: int
    blocked: int
    contested: int
    evaluated: int
    abstained: int
    n_addresses: int
    n_edges: int
    demo_root: Path


# ---- the run -----------------------------------------------------------


def _scenario_of_address(chain: scenarios.DemoChain) -> dict[str, str]:
    return {
        address: key
        for key, addresses in chain.scenario_addresses.items()
        for address in addresses
    }


def _replay(graph, oracle) -> tuple[ConstrainedUnionFind, list]:
    """Cluster, recording one row per component boundary.

    The loop itself lives in :func:`obsidianchain.cluster.replay.replay_unions`
    - one implementation shared with the Phase 3.1 funnel and the Phase 3.3
    decision scorer. Before that extraction this function was a fourth copy
    of it, and it reached into the clusterer's private evidence method to
    observe a decision without making it.
    """
    result = replay.replay_unions(graph, oracle, veto=True)
    return result.forest, result.decisions


def _components_over(roots: np.ndarray, codes: list[int]) -> int:
    return int(len({int(roots[c]) for c in codes}))


def _derive_outcome(
    result_verdict: str,
    blocked: bool,
    contested: bool,
    observations_seen: int,
) -> DemoOutcome:
    """Read the outcome off the engine's state, never off the scenario spec.

    Derivation order matters: a contested component was merged and then found
    to be contradictory, so CONTESTED outranks the merge that produced it.
    """
    if contested:
        return DemoOutcome.CONTESTED
    if blocked:
        return DemoOutcome.BLOCKED
    if result_verdict == Verdict.NO_EVIDENCE.value:
        # The distinction the operator cares about: nothing to consult
        # versus consulted and unusable.
        return (
            DemoOutcome.ABSTAINED if observations_seen else DemoOutcome.CANDIDATE
        )
    return DemoOutcome.MERGED


def run(demo_root, rebuild: bool = False) -> DemoRun:
    """Build (if needed) and run the whole demonstration.

    Returns everything the API layer needs. Raises
    :class:`DemoExpectationError` if any scenario stops producing the outcome
    it claims - a demonstration that silently shows the wrong thing is worse
    than one that fails.
    """
    root = scenarios.assert_demo_namespace(demo_root)
    chain = scenarios.build_chain()
    if rebuild or not scenarios.fixture_exists(root):
        manifest = scenarios.build_fixture(root, chain)
    else:
        import json

        manifest = json.loads(
            (
                root / "processed" / "network" / "manifest.json"
            ).read_text(encoding="utf-8")
        )

    config = SeparationConfig()
    assert_production_rule(config)

    graph = elliptic.load_cospend_graph(root, keep_labels=True)
    code_of = {str(a): i for i, a in enumerate(graph.addresses)}
    scenario_of = _scenario_of_address(chain)

    oracle = separation.build_oracle(
        graph,
        processed_root=root / "processed",
        data_root=root,
        config=config,
    )

    # How much of each scenario the capture actually saw. Observable: it is
    # a count of announcement records, not a generator fact.
    inputs = boundary.load_phase3_inputs(root / "processed")
    observed_txids = set(inputs.observations["txid"].tolist())

    chain_only = run_clustering(graph, label="demo-chain-only")
    forest, rows = _replay(graph, oracle)

    contradictions = audit.find_contradictions(forest, oracle, graph.n_addresses)
    contested_components = audit.record_contradictions(forest, contradictions)
    contested_roots = forest.contested_roots()

    # Independent path to the same headline counters.
    fused = run_fused(graph, oracle, label="demo-fused")
    if fused.blocked != forest.counters.blocked:
        raise DemoExpectationError(
            f"cross-check failed: run_fused blocked {fused.blocked} merges, "
            f"the decision replay blocked {forest.counters.blocked}. The two "
            f"paths must agree."
        )

    fused_roots = forest.roots()

    results: list[ScenarioResult] = []
    for spec in scenarios.SCENARIOS:
        codes = [code_of[a] for a in chain.scenario_addresses[spec.key]]
        decisions = [
            DecisionRecord(
                scenario=spec.key,
                address_a=str(graph.addresses[row.node_a]),
                address_b=str(graph.addresses[row.node_b]),
                side_a=row.size_a,
                side_b=row.size_b,
                verdict=row.evidence.verdict.value,
                pooled_a=row.evidence.n_a,
                pooled_b=row.evidence.n_b,
                chi2=float(row.evidence.chi2),
                p_value=float(row.evidence.p_value),
                effect=float(row.evidence.effect),
                reason=row.evidence.reason,
                merged=row.merged,
            )
            for row in rows
            if scenario_of.get(str(graph.addresses[row.node_a])) == spec.key
        ]
        contradiction = next(
            (c for c in contradictions if c.member in codes), None
        )
        contested = any(int(fused_roots[c]) in contested_roots for c in codes)

        blocked = any(not d.merged for d in decisions)
        seen = sum(
            1 for t in chain.scenario_txids[spec.key] if int(t) in observed_txids
        )
        usable = sum(
            oracle.stats_for(c).count for c in codes
        )
        if contradiction is not None:
            verdict = contradiction.evidence.verdict.value
        elif blocked:
            verdict = next(d.verdict for d in decisions if not d.merged)
        elif decisions:
            verdict = decisions[0].verdict
        else:
            verdict = Verdict.NO_EVIDENCE.value

        outcome = _derive_outcome(verdict, blocked, contested, seen)
        result = ScenarioResult(
            spec=spec,
            outcome=outcome,
            verdict=verdict,
            decisions=decisions,
            pooled={
                str(graph.addresses[c]): int(oracle.stats_for(c).count)
                for c in codes
            },
            observations_seen=seen,
            usable_observations=int(usable),
            contradiction=contradiction,
            contested=contested,
            chain_only_components=_components_over(chain_only.roots, codes),
            fused_components=_components_over(fused_roots, codes),
        )
        if not result.matches_expectation:
            raise DemoExpectationError(
                f"scenario {spec.key} ({spec.title}) no longer behaves as "
                f"described.\n"
                f"  expected outcome {spec.expected_outcome.value} / verdict "
                f"{spec.expected_verdict}\n"
                f"  observed outcome {outcome.value} / verdict {verdict}\n"
                f"Either the fixture or the engine changed. Fix one of them "
                f"rather than the description."
            )
        results.append(result)

    return DemoRun(
        results=results,
        manifest=manifest,
        oracle=oracle,
        config=config,
        chain_only_clusters=int(chain_only.n_clusters),
        fused_clusters=int(len(set(fused_roots.tolist()))),
        blocked=int(forest.counters.blocked),
        contested=int(contested_components),
        evaluated=int(forest.counters.evaluated),
        abstained=int(forest.counters.abstained),
        n_addresses=int(graph.n_addresses),
        n_edges=int(graph.n_edges),
        demo_root=root,
    )


# ---- API projection ----------------------------------------------------


def _decision_record(decision: DecisionRecord) -> dict:
    return api.record(
        address_a=decision.address_a,
        address_b=decision.address_b,
        members_a=decision.side_a,
        members_b=decision.side_b,
        verdict=decision.verdict,
        pooled_a=decision.pooled_a,
        pooled_b=decision.pooled_b,
        chi2=round(decision.chi2, 4),
        p_value=decision.p_value,
        effect=round(decision.effect, 6),
        reason=decision.reason,
        merged=decision.merged,
    )


def _scenario_record(result: ScenarioResult) -> dict:
    spec = result.spec
    contradiction = None
    if result.contradiction is not None:
        evidence = result.contradiction.evidence
        contradiction = api.record(
            member_code=result.contradiction.member,
            counterpart_code=result.contradiction.counterpart,
            component_size=result.contradiction.component_size,
            verdict=evidence.verdict.value,
            pooled_member=evidence.n_a,
            pooled_rest=evidence.n_b,
            chi2=round(float(evidence.chi2), 4),
            p_value=float(evidence.p_value),
            effect=round(float(evidence.effect), 6),
            found_by="leave-one-out audit, after clustering finished",
            resolution="recorded as CONTESTED; union-find has no split",
        )
    return api.record(
        key=spec.key,
        title=spec.title,
        situation=spec.situation,
        chain_story=spec.chain_story,
        network_story=spec.network_story,
        reads=spec.reads,
        outcome=result.outcome.value,
        expected_outcome=spec.expected_outcome.value,
        matches_expectation=result.matches_expectation,
        verdict=result.verdict,
        addresses=list(result.pooled),
        pooled=[
            api.record(address=address, observations=count)
            for address, count in result.pooled.items()
        ],
        announcements_seen=result.observations_seen,
        usable_observations=result.usable_observations,
        chain_only_components=result.chain_only_components,
        fused_components=result.fused_components,
        contested=result.contested,
        decisions=[_decision_record(d) for d in result.decisions],
        contradiction=contradiction,
    )


def to_envelope(run_result: DemoRun) -> dict:
    """Project a run into the flagged API payload the UI consumes."""
    manifest = run_result.manifest
    fixture = api.record(
        seed=manifest.get("seed", scenarios.DEMO_SEED),
        fixture_version=manifest.get(
            "demo_fixture_version", scenarios.DEMO_FIXTURE_VERSION
        ),
        dataset_sha256=manifest.get("dataset_sha256", ""),
        addresses=run_result.n_addresses,
        cospend_edges=run_result.n_edges,
        chain_transactions=manifest.get("chain_transaction_count", 0),
        announced_transactions=manifest.get("transaction_count", 0),
        unannounced_transactions=manifest.get("unobserved_transaction_count", 0),
        announcement_records=manifest.get("record_count", 0),
        observers=manifest.get("observer_count", 0),
        usable_transactions=run_result.oracle.n_usable_transactions,
        discarded_no_evidence=run_result.oracle.n_excluded_no_evidence,
        sigma=manifest.get("sigma", 0.0),
        sigma_source="Decker & Wattenhofer 2013",
    )
    config = run_result.config
    rule = api.record(
        min_pooled_observations=config.min_pooled_observations,
        min_observer_observations=config.min_observer_observations,
        alpha=config.alpha,
        min_effect=config.min_effect,
        unchanged_from_production=True,
        note=(
            "The threshold is not lowered for the demonstration. The fixture "
            "raises the evidence to meet it."
        ),
        cannot_link_only=(
            "The network layer emits cannot-link only. A shared origin never "
            "implies a shared entity: one Electrum server broadcasts for tens "
            "of thousands of unrelated users."
        ),
    )
    totals = api.record(
        scenarios=len(run_result.results),
        chain_only_clusters=run_result.chain_only_clusters,
        fused_clusters=run_result.fused_clusters,
        merges_blocked=run_result.blocked,
        components_contested=run_result.contested,
        decisions_evaluated=run_result.evaluated,
        decisions_abstained=run_result.abstained,
        all_scenarios_as_described=all(
            r.matches_expectation for r in run_result.results
        ),
    )
    return api.build_envelope(
        scenarios=[_scenario_record(r) for r in run_result.results],
        fixture=fixture,
        rule=rule,
        totals=totals,
        seed=scenarios.DEMO_SEED,
        namespace=str(run_result.demo_root),
    )
