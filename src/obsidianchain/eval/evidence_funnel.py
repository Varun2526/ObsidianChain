"""Where network evidence disappears: a funnel over every proposed union.

Phase 3 produced zero blocked merges out of 253,429 proposed unions, with
only 40 reaching the pooled minimum. That could mean two very different
things, and the difference decides what to do next:

* **the ceiling is noise** - unions that did clear the threshold mostly come
  back NOT_SEPARATED, so even with more data there is nothing to find; or
* **the ceiling is sample size** - those unions skew SEPARATED, so the
  evidence is real but starved, and the fix is more observations per group
  rather than a different statistic.

This module measures which. It is diagnosis: it adds no mechanism, changes
no threshold, and does not touch the frozen generator, the production
observations, ``network_truth``, the Phase 1 baseline, or the merge-time
constraint logic. It replays the same edges through its own instrumented
copy of the bookkeeping and records what the evidence looked like at each
proposed union.

Measured on the chain-only trajectory
-------------------------------------
Every union is applied; none is blocked. This matters for interpretation.
If the funnel blocked merges, a lower threshold would refuse merges a higher
one allowed, the components would diverge, and the rows of the funnel would
describe different clusterings - "how many unions cleared 5" and "how many
cleared 25" would not be answers about the same sequence of events. Holding
the trajectory fixed makes the thresholds comparable, at the cost of
describing the baseline's merge sequence rather than each threshold's own.

Two gates, reported separately
------------------------------
A union can fail to yield a decision for two different reasons, and lumping
them together would hide half the story:

* **pooled count** - fewer than ``min_pooled_observations`` transactions on
  one side. This is the gate the headline funnel varies.
* **degrees of freedom** - no observer has enough paired observations for a
  variance estimate, so the statistic is undefined however many
  transactions are pooled. Note that clearing a pooled minimum of 1 can
  never produce a decision: variance needs at least two samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.io.elliptic import CoSpendGraph
from obsidianchain.network.separation import (
    SeparationConfig,
    SeparationOracle,
    Verdict,
    separation_evidence,
)

DEFAULT_THRESHOLDS = (1, 2, 5, 10, 25)

#: Permissive config used to extract the raw statistic wherever it is
#: computable at all, so the distribution analysis is not confounded by the
#: production gates. The production config is applied separately.
PROBE_CONFIG = SeparationConfig(
    min_pooled_observations=1, min_observer_observations=2
)

#: Schema /2 adds the production-evaluation columns. Imported rather than
#: redeclared so the artifact and the API cannot drift apart.
from obsidianchain import evidence_contract as contract

RECORD_COLUMNS = [
    "edge_index",
    "node_a",
    "node_b",
    "pooled_a",
    "pooled_b",
    "min_pooled",
    "size_a",
    "size_b",
    "min_size",
    "dof",
    "chi2",
    "p_value",
    "effect",
]


@dataclass
class FunnelResult:
    """Per-union diagnostics plus the aggregate funnel."""

    records: pd.DataFrame
    n_edges: int = 0
    n_redundant: int = 0
    n_proposed: int = 0
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
    production_config: SeparationConfig = field(default_factory=SeparationConfig)
    production_verdicts: dict[str, int] = field(default_factory=dict)

    def cleared(self, threshold: int) -> pd.DataFrame:
        return self.records[self.records["min_pooled"] >= threshold]

    def decidable(self, threshold: int) -> pd.DataFrame:
        cleared = self.cleared(threshold)
        return cleared[cleared["dof"] >= 1]

    def verdicts_at(self, threshold: int) -> dict[str, int]:
        """Decisions the production rule would reach at this pooled minimum.

        Only the pooled gate moves; alpha and the effect floor stay at their
        production values, because the question is where evidence vanishes,
        not what a different rule would conclude.
        """
        config = self.production_config
        frame = self.decidable(threshold)
        if frame.empty:
            return {"SEPARATED": 0, "NOT_SEPARATED": 0}
        separated = (frame["p_value"] < config.alpha) & (
            frame["effect"] >= config.min_effect
        )
        return {
            "SEPARATED": int(separated.sum()),
            "NOT_SEPARATED": int((~separated).sum()),
        }


def build_funnel(
    graph: CoSpendGraph,
    oracle: SeparationOracle,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    production_config: SeparationConfig | None = None,
    production_statistics_config: SeparationConfig | None = None,
) -> FunnelResult:
    """Replay the co-spend edges and record evidence at every proposed union.

    The walk is :func:`obsidianchain.cluster.replay.replay_unions` - one
    implementation shared with the Phase 3.3 decision scorer and the
    demonstration runner. This function previously kept its own pooled
    statistics dictionary over a plain :class:`UnionFind`, a second
    implementation of the aggregation :class:`ConstrainedUnionFind` already
    performs; that duplication is what the Phase 4 audit found.

    Two things about how it is called matter and are unchanged:

    * ``veto=False`` - every union is applied, so all thresholds describe
      the same baseline sequence of proposed merges. See the module
      docstring on why a blocking funnel would not be comparable across
      thresholds.
    * ``config=PROBE_CONFIG`` - the raw statistic is extracted wherever it is
      computable at all, so the distribution analysis is not confounded by
      the production gates. The production rule is applied separately, below.
    """
    from obsidianchain.cluster import replay

    production_config = production_config or SeparationConfig()

    edge_index: list[int] = []
    node_a: list[int] = []
    node_b: list[int] = []
    pooled_a: list[int] = []
    pooled_b: list[int] = []
    size_a: list[int] = []
    size_b: list[int] = []
    dof: list[int] = []
    chi2: list[float] = []
    p_value: list[float] = []
    effect: list[float] = []

    def record(decision) -> None:
        # Recorded BEFORE any evidence test is interpreted, as specified.
        edge_index.append(decision.edge_index)
        node_a.append(int(decision.node_a))
        node_b.append(int(decision.node_b))
        pooled_a.append(int(decision.evidence.n_a))
        pooled_b.append(int(decision.evidence.n_b))
        size_a.append(int(decision.size_a))
        size_b.append(int(decision.size_b))

        # The statistic wherever it is computable at all. Two pooled
        # observations per side is the floor for a variance estimate, so
        # below that there is nothing to compute and the union is recorded
        # with zero degrees of freedom rather than a fabricated number.
        if min(decision.evidence.n_a, decision.evidence.n_b) >= 2:
            dof.append(int(decision.evidence.dof))
            chi2.append(float(decision.evidence.chi2))
            p_value.append(float(decision.evidence.p_value))
            effect.append(float(decision.evidence.effect))
        else:
            dof.append(0)
            chi2.append(np.nan)
            p_value.append(np.nan)
            effect.append(np.nan)

    # collect=False: a quarter of a million retained records would cost far
    # more memory than the thirteen scalar lists above, for no benefit - the
    # funnel never looks at a decision twice.
    walk = replay.replay_unions(
        graph,
        oracle,
        veto=False,
        config=PROBE_CONFIG,
        collect=False,
        on_decision=record,
    )
    n_redundant = walk.n_redundant

    # ---- schema /2: the production evaluation ------------------------
    #
    # A SECOND walk of the same trajectory, not a re-derivation from the
    # columns above. `union_without_veto` never consults evidence, so with
    # veto=False the union sequence is a pure function of the graph and is
    # identical between the two walks - which means row i of one aligns with
    # row i of the other. That alignment is asserted, not assumed.
    #
    # One oracle serves both: SeparationConfig reaches only the oracle's
    # `config` attribute, never its per-address statistics, so the pooled
    # numbers are config-independent.
    production = None
    if production_statistics_config is not None:
        production = _production_evaluation(
            graph, oracle, production_statistics_config,
            expected_edge_index=edge_index,
            expected_pooled=(pooled_a, pooled_b),
        )

    records = pd.DataFrame(
        {
            "edge_index": edge_index,
            "node_a": node_a,
            "node_b": node_b,
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
            "min_pooled": np.minimum(pooled_a, pooled_b) if pooled_a else [],
            "size_a": size_a,
            "size_b": size_b,
            "min_size": np.minimum(size_a, size_b) if size_a else [],
            "dof": dof,
            "chi2": chi2,
            "p_value": p_value,
            "effect": effect,
        },
        columns=RECORD_COLUMNS,
    )
    if production is not None:
        for column in contract.PRODUCTION_COLUMNS:
            values = production[column]
            if column == "dof_production":
                # Nullable INTEGER, not float. A plain list with Nones makes
                # pandas infer float64, and the column would land as `double`
                # with 8.0 instead of 8 - the trap the contract calls out.
                records[column] = pd.array(values, dtype="Int64")
            else:
                records[column] = values

    result = FunnelResult(
        records=records,
        n_edges=int(graph.n_edges),
        n_redundant=n_redundant,
        n_proposed=int(len(records)),
        thresholds=tuple(thresholds),
        production_config=production_config,
    )

    # Reproduce the production rule exactly, through the production function,
    # so the headline count is not a re-derivation that could drift from it.
    verdicts = {v.value: 0 for v in Verdict}
    production_min = production_config.min_pooled_observations
    candidate = records[records["min_pooled"] >= production_min]
    for row in candidate.itertuples():
        # Statistics are recomputed from the recorded gates rather than
        # cached, because the production per-observer minimum differs from
        # the probe's and can change the degrees of freedom.
        verdicts[_production_verdict(row, production_config).value] += 1
    verdicts[Verdict.NO_EVIDENCE.value] += int(len(records) - len(candidate))
    result.production_verdicts = verdicts
    return result


def _production_evaluation(
    graph, oracle, config: SeparationConfig, *, expected_edge_index,
    expected_pooled,
) -> dict[str, list]:
    """Walk the same trajectory again under the production configuration.

    Returns the seven schema /2 columns. Every value is the direct output of
    ``separation_evidence`` under ``config`` - nothing here derives a verdict
    from a persisted number.

    Short-circuited rows carry NULL statistics, not the dataclass defaults.
    All three NO_EVIDENCE gates return early with chi2=0.0, dof=0,
    p_value=1.0, effect=0.0, and those are not results: persisting them raw
    would put a chi-square of exactly zero on a quarter of a million rows
    that were never evaluated. ``reason_code_production`` is what
    distinguishes the three gates, since they are numerically identical.
    """
    from obsidianchain.cluster import replay as _replay

    verdicts: list[str] = []
    codes: list[str] = []
    reasons: list[str] = []
    dof: list[float | None] = []
    chi2: list[float | None] = []
    p_value: list[float | None] = []
    effect: list[float | None] = []
    seen_edges: list[int] = []
    seen_pooled_a: list[int] = []
    seen_pooled_b: list[int] = []

    def record_production(decision) -> None:
        evidence = decision.evidence
        seen_edges.append(decision.edge_index)
        seen_pooled_a.append(int(evidence.n_a))
        seen_pooled_b.append(int(evidence.n_b))
        verdicts.append(evidence.verdict.value)
        # Total mapping: an unmapped reason aborts generation rather than
        # being coerced to a default.
        codes.append(contract.reason_code(evidence.reason))
        reasons.append(evidence.reason)
        if evidence.verdict is Verdict.NO_EVIDENCE:
            dof.append(None)
            chi2.append(None)
            p_value.append(None)
            effect.append(None)
        else:
            dof.append(int(evidence.dof))
            chi2.append(float(evidence.chi2))
            p_value.append(float(evidence.p_value))
            effect.append(float(evidence.effect))

    _replay.replay_unions(
        graph, oracle, veto=False, config=config, collect=False,
        on_decision=record_production,
    )

    # Alignment is the whole basis for joining the two walks row-wise, so it
    # is checked rather than trusted.
    if seen_edges != list(expected_edge_index):
        raise RuntimeError(
            "the production walk did not follow the same trajectory as the "
            "probe walk: edge_index sequences differ. The two walks must be "
            "row-aligned or the production columns would describe different "
            "proposed merges."
        )
    if (seen_pooled_a, seen_pooled_b) != (
        list(expected_pooled[0]), list(expected_pooled[1])
    ):
        raise RuntimeError(
            "the production walk observed different pooled counts from the "
            "probe walk; the oracle is not config-independent as assumed."
        )

    import numpy as _np

    def _round(values):
        return [
            None if v is None else float(_np.round(v, contract.ROUNDING_DECIMALS))
            for v in values
        ]

    return {
        "verdict_production": verdicts,
        "reason_code_production": codes,
        "reason_production": reasons,
        "dof_production": dof,
        # Rounded to the same 6 decimals the probe columns use, so the two
        # blocks are comparable at face value.
        "chi2_production": _round(chi2),
        "p_value_production": p_value,
        "effect_production": _round(effect),
    }


def _production_verdict(row, config: SeparationConfig) -> Verdict:
    """Verdict under the production alpha and effect floor."""
    if row.dof < 1 or not np.isfinite(row.p_value):
        return Verdict.NO_EVIDENCE
    if row.p_value < config.alpha and row.effect >= config.min_effect:
        return Verdict.SEPARATED
    return Verdict.NOT_SEPARATED


# ---- reporting --------------------------------------------------------

_W = 78


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:6.2f}%" if whole else "     -"


def format_funnel(result: FunnelResult) -> str:
    """Render the funnel, the statistic distribution, and the size profile."""
    records = result.records
    total = result.n_proposed
    out: list[str] = []
    add = out.append

    add("=" * _W)
    add("OBSIDIANCHAIN - PHASE 3.1 EVIDENCE FUNNEL")
    add("=" * _W)
    add("  Diagnosis of the frozen dataset. No mechanism added, no threshold")
    add("  changed, no generator touched. Network data remains SYNTHETIC.")
    add("")
    add("  Measured on the chain-only trajectory: every union is applied, so")
    add("  all thresholds below describe the same sequence of proposed merges.")
    add("")

    add("-- funnel " + "-" * (_W - 11))
    add(f"  co-spend edges                        {result.n_edges:>12,}")
    add(f"  ...already connected (redundant)      {result.n_redundant:>12,}")
    add(f"  proposed unions                       {total:>12,}")
    add("")
    for threshold in result.thresholds:
        cleared = int((records["min_pooled"] >= threshold).sum())
        marker = "  <- current setting" if (
            threshold == result.production_config.min_pooled_observations
        ) else ""
        add(f"  ...with >= {threshold:<3} usable tx on both sides "
            f"{cleared:>12,}   {_pct(cleared, total)}{marker}")
    add("")

    add("-- the second gate: is the statistic even computable? " + "-" * (_W - 55))
    add("  Clearing a pooled minimum is not enough - a variance estimate")
    add("  needs at least two observations, and the per-observer minimum")
    add("  can leave zero usable dimensions however much is pooled.")
    add("")
    add(f"  {'threshold':>10}{'cleared':>12}{'decidable':>12}{'lost to dof':>14}")
    for threshold in result.thresholds:
        cleared = int((records["min_pooled"] >= threshold).sum())
        decidable = int(len(result.decidable(threshold)))
        add(f"  {threshold:>10}{cleared:>12,}{decidable:>12,}"
            f"{cleared - decidable:>14,}")
    add("")

    add("-- decisions at each threshold " + "-" * (_W - 32))
    add("  Only the pooled gate moves; alpha and the effect floor stay at")
    add(f"  production values (alpha={result.production_config.alpha:.0e}, "
        f"effect>={result.production_config.min_effect}).")
    add("")
    add(f"  {'threshold':>10}{'decidable':>12}{'SEPARATED':>12}"
        f"{'NOT_SEP':>10}{'sep share':>12}")
    for threshold in result.thresholds:
        verdicts = result.verdicts_at(threshold)
        decidable = verdicts["SEPARATED"] + verdicts["NOT_SEPARATED"]
        share = (
            f"{verdicts['SEPARATED'] / decidable * 100:10.1f}%"
            if decidable
            else f"{'-':>11}"
        )
        add(f"  {threshold:>10}{decidable:>12,}{verdicts['SEPARATED']:>12,}"
            f"{verdicts['NOT_SEPARATED']:>10,}{share}")
    add("")

    add("-- production rule, reproduced exactly " + "-" * (_W - 40))
    for name, count in result.production_verdicts.items():
        add(f"  {name:<20}{count:>12,}")
    add("")

    # ---- statistic distribution over decidable unions -----------------
    add("-- separation statistic where it could be computed " + "-" * (_W - 52))
    decidable_all = records[records["dof"] >= 1]
    add(f"  decidable unions (any pooled count)   {len(decidable_all):>12,}")
    if len(decidable_all):
        for label, column in (
            ("chi2", "chi2"),
            ("effect", "effect"),
            ("min_pooled", "min_pooled"),
        ):
            values = decidable_all[column].dropna().to_numpy()
            if not values.size:
                continue
            add(f"  {label:<12} median {np.median(values):>10.4f}"
                f"   p95 {np.percentile(values, 95):>10.4f}"
                f"   max {values.max():>12.4f}")
        log_p = -np.log10(
            np.maximum(decidable_all["p_value"].dropna().to_numpy(), 1e-300)
        )
        add(f"  {'-log10 p':<12} median {np.median(log_p):>10.2f}"
            f"   p95 {np.percentile(log_p, 95):>10.2f}"
            f"   max {log_p.max():>12.2f}")
    add("")

    threshold = result.production_config.min_pooled_observations
    at_production = result.decidable(threshold)
    add(f"-- the {len(at_production)} unions that cleared {threshold} " + "-" * (_W - 34))
    if len(at_production):
        verdicts = result.verdicts_at(threshold)
        separated = verdicts["SEPARATED"]
        decidable = separated + verdicts["NOT_SEPARATED"]
        add(f"  SEPARATED       {separated:>6,} of {decidable:,}"
            f"   {_pct(separated, decidable)}")
        add(f"  NOT_SEPARATED   {verdicts['NOT_SEPARATED']:>6,} of {decidable:,}"
            f"   {_pct(verdicts['NOT_SEPARATED'], decidable)}")
        add("")
        add(f"  {'min_pooled':>12}{'dof':>6}{'chi2':>12}{'effect':>10}"
            f"{'-log10 p':>11}  verdict")
        shown = at_production.sort_values("min_pooled", ascending=False).head(15)
        for row in shown.itertuples():
            verdict = _production_verdict(row, result.production_config)
            lp = -np.log10(max(row.p_value, 1e-300))
            add(f"  {row.min_pooled:>12,}{row.dof:>6}{row.chi2:>12.1f}"
                f"{row.effect:>10.4f}{lp:>11.1f}  {verdict.value}")
        if len(at_production) > 15:
            add(f"  ... {len(at_production) - 15} more")
    else:
        add("  none")
    add("")

    # ---- size profile at merge time -----------------------------------
    add("-- group size at merge time " + "-" * (_W - 29))
    add("  The binding constraint on pooling is whichever side is smaller.")
    add("")
    for label, column in (
        ("pooled observations", "min_pooled"),
        ("addresses in group", "min_size"),
    ):
        values = records[column].to_numpy()
        if not values.size:
            continue
        add(f"  min(A, B) by {label}")
        add(f"    median {np.median(values):>10.1f}"
            f"    p95 {np.percentile(values, 95):>10.1f}"
            f"    p99 {np.percentile(values, 99):>10.1f}"
            f"    max {values.max():>10,}")
        add(f"    mean   {values.mean():>10.2f}"
            f"    share at 1: {_pct(int((values <= 1).sum()), len(values))}")
    add("")

    # ---- reading -------------------------------------------------------
    verdicts = result.verdicts_at(threshold)
    decidable = verdicts["SEPARATED"] + verdicts["NOT_SEPARATED"]
    add("-- reading " + "-" * (_W - 12))
    if decidable == 0:
        add("  Nothing cleared the threshold with a computable statistic, so")
        add("  neither hypothesis is tested. The funnel rows above show where")
        add("  the evidence is lost; that is the finding.")
    else:
        share = verdicts["SEPARATED"] / decidable
        if share >= 0.5:
            add(f"  Of the unions that could be judged, {share * 100:.0f}% came back")
            add("  SEPARATED. The ceiling is SAMPLE SIZE, not noise: where there")
            add("  is enough pooled evidence the statistic does find structure.")
            add("  More observations per group would extend reach; a different")
            add("  statistic would not be the lever.")
        elif share <= 0.1:
            add(f"  Of the unions that could be judged, only {share * 100:.0f}% came")
            add("  back SEPARATED. The ceiling is NOISE: even with enough pooled")
            add("  evidence there is little structure to find, which is what the")
            add("  Phase 2 audit predicted from a per-transaction separability")
            add("  of 1.003. More data would not help on its own.")
        else:
            add(f"  {share * 100:.0f}% of judgeable unions came back SEPARATED -")
            add("  between the two clean readings. Sample size limits reach, but")
            add("  structure is only sometimes present when reach allows a look.")
    add("=" * _W)
    return "\n".join(out)


def write_records(result: FunnelResult, path: Path, provenance=None) -> int:
    """Write the per-union records. Returns the row count.

    ``provenance`` attaches the record two ways - a marker column in every
    row and a sibling ``.meta.json``. Optional so the function keeps working
    for callers that only want the frame on disk; every CLI path passes it,
    and ``tests/test_provenance.py`` reads the artifacts back to check.
    """
    path = Path(path)
    frame = result.records.copy()
    for column in ("chi2", "effect"):
        frame[column] = frame[column].round(6)
    if provenance is not None:
        from obsidianchain import provenance as prov

        prov.write_frame(frame, path, provenance)
        return int(len(frame))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return int(len(frame))
