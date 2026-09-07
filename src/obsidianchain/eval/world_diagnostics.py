"""Verify the five controlled worlds produce the distributions intended.

This runs before any Phase 3 inference. Its job is to confirm the generator
did what the regime specification says, so that a later inference result can
be read against a known input rather than a hoped-for one.

Two families of measurement, and both matter:

**Origin structure**, read from ground truth. How many origins an entity
actually uses, how peaked its distribution is, and how much two entities
overlap. This checks the specification was implemented, not that anything is
detectable.

**Arrival-vector separability by entity**, read from the observations. This
is the measurement that decides whether the regime is usable: the Phase 2
audit's methodology, re-pointed from origins to entities, with the
corrections that audit needed - centroids over the full usable set, and a
denominator of the centroid's own standard error rather than the scatter of
individual transactions.

Regime A is the control. It replicates the frozen null under this new code
path, so if A shows separability the generator has introduced signal and
every other regime's result is suspect. A is checked first for that reason.

Nothing here is validation. The worlds are synthetic and were written by the
same people as the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import arrivals, boundary, worlds

#: Transactions sampled for pairwise distance work. The question is about
#: distributions, so a sample answers it; the full pairwise problem is
#: quadratic and unnecessary.
SAMPLE = 4000

#: Support threshold: an origin counts as used by an entity when its
#: probability exceeds half of uniform. Avoids counting Dirichlet dust.
SUPPORT_FLOOR_MULTIPLE = 0.5


@dataclass
class RegimeDiagnostics:
    regime: str = ""
    name: str = ""
    expectation: str = ""

    # dataset shape
    n_transactions: int = 0
    n_records: int = 0
    n_entities: int = 0
    n_origins_used: int = 0
    dataset_sha256: str = ""

    # origin structure, from ground truth
    origins_per_entity_mean: float = 0.0
    origins_per_entity_median: float = 0.0
    origin_entropy_mean: float = 0.0
    entity_overlap_median: float = 0.0
    entity_overlap_min: float = 0.0
    identical_entity_pairs: int = 0

    # separability by entity, from observations
    usable_transactions: int = 0
    per_transaction_ratio: float = float("nan")
    rank_ratio: float = float("nan")
    first_observer_lift: float = float("nan")
    centroid_separation_se: float = float("nan")
    within_entity_scatter: float = float("nan")
    near_duplicate_vectors: int = 0

    notes: list[str] = field(default_factory=list)


def _unit(matrix: np.ndarray) -> np.ndarray:
    """Offsets scaled by each row's own maximum; missing filled by row max.

    A missing sighting is not a fast one. Filling with the row maximum stops
    an absent observer masquerading as an early arrival, which would invent
    the very signal these regimes are meant to control.
    """
    filled = matrix.copy()
    row_max = np.nanmax(
        np.where(np.isnan(filled), -np.inf, filled), axis=1, keepdims=True
    )
    row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
    filled = np.where(np.isnan(filled), row_max, filled)
    return filled / row_max


def _describe_overlap(distribution: np.ndarray) -> tuple[float, float, int]:
    """Median and minimum total-variation distance between entity pairs.

    Total variation is 0 for identical distributions and 1 for disjoint
    support, so it reads directly as "how confusable are these two
    entities". Regime C should show a cluster of pairs at ~0.
    """
    from scipy.spatial.distance import pdist

    if distribution.shape[0] < 2:
        return float("nan"), float("nan"), 0
    tv = 0.5 * pdist(distribution, metric="cityblock")
    identical = int((tv < 1e-9).sum())
    return float(np.median(tv)), float(tv.min()), identical


def diagnose_regime(
    regime: worlds.Regime,
    processed_root: Path,
    sample: int = SAMPLE,
    seed: int = 0,
) -> RegimeDiagnostics:
    """Measure one regime's generated data against its specification."""
    from scipy.spatial.distance import pdist

    key = regime.value
    report = RegimeDiagnostics(
        regime=key,
        name=worlds.REGIME_NAMES[regime],
        expectation=worlds.REGIME_EXPECTATIONS[regime],
    )

    inputs = boundary.load_phase3_inputs(processed_root, world=key)
    report.n_records = inputs.n_records
    report.dataset_sha256 = str(inputs.manifest.get("dataset_sha256", ""))[:32]

    observer_ids = sorted(inputs.observers["observer_id"].tolist()) or sorted(
        inputs.observations["observer_id"].unique().tolist()
    )
    vectors = arrivals.build(inputs.observations, observer_ids=observer_ids)
    report.n_transactions = vectors.n_transactions

    broadcaster_txids = set(
        inputs.observations.loc[
            inputs.observations["peer_ip"].isin(inputs.broadcaster_ips), "txid"
        ]
    )
    labels = arrivals.classify_evidence(vectors, broadcaster_txids)
    usable_mask = (labels["evidence"] == arrivals.Evidence.USABLE.value).to_numpy()
    report.usable_transactions = int(usable_mask.sum())

    # --- ground truth, used openly for description only ----------------
    truth = boundary.load_ground_truth_FOR_EVALUATION_ONLY(processed_root, world=key)
    aligned = truth.set_index("txid").reindex(vectors.txids)
    entities = aligned["true_entity_id"].to_numpy()
    origins = aligned["true_origin_id"].to_numpy()
    report.n_entities = int(pd.Series(entities).nunique())
    report.n_origins_used = int(pd.Series(origins).nunique())

    truth_dir = (
        Path(processed_root) / "worlds_truth" / key / "origin_distribution.csv"
    )
    if truth_dir.is_file():
        distribution = np.loadtxt(truth_dir, delimiter=",")
        distribution = np.atleast_2d(distribution)
        floor = SUPPORT_FLOOR_MULTIPLE / distribution.shape[1]
        support = (distribution > floor).sum(axis=1)
        report.origins_per_entity_mean = float(support.mean())
        report.origins_per_entity_median = float(np.median(support))
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(
                np.where(distribution > 0, distribution * np.log2(distribution), 0.0),
                axis=1,
            )
        report.origin_entropy_mean = float(entropy.mean())
        median_tv, min_tv, identical = _describe_overlap(distribution)
        report.entity_overlap_median = median_tv
        report.entity_overlap_min = min_tv
        report.identical_entity_pairs = identical

    # --- separability by entity ----------------------------------------
    eligible = np.flatnonzero(usable_mask & (vectors.n_observed >= 2))
    if eligible.size < 2:
        report.notes.append("too few usable transactions to measure separability")
        return report

    full_unit = _unit(vectors.offsets_ms[eligible])
    full_entities = entities[eligible]

    rng = np.random.default_rng(seed)
    picked = (
        rng.choice(eligible.size, size=sample, replace=False)
        if eligible.size > sample
        else np.arange(eligible.size)
    )
    unit = full_unit[picked]
    sub_entities = full_entities[picked]
    codes = pd.factorize(pd.Series(sub_entities))[0].astype(np.float64)
    same = pdist(codes.reshape(-1, 1), metric="hamming") == 0

    if same.any() and (~same).any():
        dist = pdist(unit)
        within, between = dist[same], dist[~same]
        if np.median(within) > 0:
            report.per_transaction_ratio = float(
                np.median(between) / np.median(within)
            )
        ranks = pdist(
            vectors.ranks[eligible][picked].astype(float), metric="hamming"
        )
        if np.median(ranks[same]) > 0:
            report.rank_ratio = float(
                np.median(ranks[~same]) / np.median(ranks[same])
            )
        first = vectors.ranks[eligible][picked].argmin(axis=1).astype(float)
        agree = pdist(first.reshape(-1, 1), metric="hamming") == 0
        p_same, p_diff = float(agree[same].mean()), float(agree[~same].mean())
        if p_diff > 0:
            report.first_observer_lift = p_same / p_diff

    # Aggregate view over EVERY usable transaction, not the sample: group
    # means are cheap, and sampling would inflate the standard error by the
    # square root of the sampling fraction and understate the signal.
    frame = pd.DataFrame(full_unit)
    frame["entity"] = pd.factorize(pd.Series(full_entities))[0]
    grouped = frame.groupby("entity")
    centroids = grouped.mean().to_numpy()
    per_entity_n = grouped.size().to_numpy()
    if len(centroids) >= 2 and per_entity_n.min() > 0:
        between_c = float(np.median(pdist(centroids)))
        scatter = float(
            np.median(
                np.linalg.norm(full_unit - grouped.transform("mean").to_numpy(), axis=1)
            )
        )
        report.within_entity_scatter = scatter
        standard_error = scatter / np.sqrt(float(np.median(per_entity_n)))
        if standard_error > 0:
            report.centroid_separation_se = between_c / standard_error

    bucketed = np.round(full_unit / 0.05).astype(np.int64)
    keys = [tuple(row) for row in bucketed]
    counts = pd.Series(keys).value_counts()
    report.near_duplicate_vectors = int(counts[counts > 1].sum())
    return report


# ---- verdicts against the specification -------------------------------

#: Centroid separation, in standard errors, above which a regime is treated
#: as carrying recoverable aggregate signal. Two standard errors is the
#: conventional line and is not tuned to any of these results.
SIGNAL_SE = 2.0


def check_expectations(
    reports: dict[str, RegimeDiagnostics], config: worlds.WorldConfig
) -> list[tuple[str, str, bool, str]]:
    """Score each regime against what its specification promised.

    Returns (regime, check, passed, detail). These are assertions about the
    generated data, not about any inference method.
    """
    out: list[tuple[str, str, bool, str]] = []

    def get(key: str) -> RegimeDiagnostics | None:
        return reports.get(key)

    a = get("A")
    if a is not None:
        passed = (
            np.isfinite(a.centroid_separation_se)
            and a.centroid_separation_se < SIGNAL_SE
        )
        out.append((
            "A", "no aggregate signal (control)", bool(passed),
            f"centroid separation {a.centroid_separation_se:.2f} SE, "
            f"want < {SIGNAL_SE}",
        ))
        passed_pt = np.isfinite(a.per_transaction_ratio) and abs(
            a.per_transaction_ratio - 1.0
        ) < 0.05
        out.append((
            "A", "per-transaction ratio near 1.0", bool(passed_pt),
            f"{a.per_transaction_ratio:.3f}",
        ))
        out.append((
            "A", "every entity uses all origins equally", 
            a.origins_per_entity_mean >= config.n_origins * 0.9,
            f"{a.origins_per_entity_mean:.1f} of {config.n_origins} origins",
        ))

    b = get("B")
    if b is not None:
        out.append((
            "B", "recoverable aggregate signal",
            bool(np.isfinite(b.centroid_separation_se)
                 and b.centroid_separation_se >= SIGNAL_SE),
            f"centroid separation {b.centroid_separation_se:.2f} SE, "
            f"want >= {SIGNAL_SE}",
        ))
        out.append((
            "B", "NOT one entity per origin",
            b.origins_per_entity_mean > 1.5,
            f"{b.origins_per_entity_mean:.1f} origins per entity",
        ))
        out.append((
            "B", "real overlap between entities",
            bool(np.isfinite(b.entity_overlap_median)
                 and b.entity_overlap_median < 0.999),
            f"median total-variation {b.entity_overlap_median:.3f} "
            f"(1.0 = disjoint)",
        ))

    c = get("C")
    if c is not None:
        out.append((
            "C", "entity pairs are indistinguishable",
            c.identical_entity_pairs > 0,
            f"{c.identical_entity_pairs} entity pairs share a distribution",
        ))
        out.append((
            "C", "minimum pairwise overlap at zero",
            bool(np.isfinite(c.entity_overlap_min) and c.entity_overlap_min < 1e-6),
            f"min total-variation {c.entity_overlap_min:.2e}",
        ))

    d = get("D")
    if d is not None:
        out.append((
            "D", f"exactly {config.origins_per_entity} origins per entity",
            abs(d.origins_per_entity_median - config.origins_per_entity) < 0.01,
            f"median {d.origins_per_entity_median:.1f}",
        ))
        if a is not None and np.isfinite(d.within_entity_scatter) and np.isfinite(
            a.within_entity_scatter
        ):
            out.append((
                "D", "within-entity scatter present (split trap)",
                d.within_entity_scatter > 0,
                f"scatter {d.within_entity_scatter:.4f} vs A "
                f"{a.within_entity_scatter:.4f}",
            ))

    e = get("E")
    if e is not None and b is not None and a is not None:
        se_a, se_b, se_e = (
            a.centroid_separation_se,
            b.centroid_separation_se,
            e.centroid_separation_se,
        )
        ordered = (
            np.isfinite(se_a) and np.isfinite(se_b) and np.isfinite(se_e)
            and se_a < se_e <= se_b * 1.05
        )
        out.append((
            "E", "signal between A and B",
            bool(ordered),
            f"A {se_a:.2f} < E {se_e:.2f} <= B {se_b:.2f} SE",
        ))

    # Cross-regime invariants: the whole point of the design.
    sigmas = {r.regime: r for r in reports.values()}
    entity_counts = {r.n_entities for r in sigmas.values()}
    out.append((
        "all", "same entity count in every regime",
        len(entity_counts) == 1,
        f"{sorted(entity_counts)}",
    ))
    tx_counts = {r.n_transactions for r in sigmas.values()}
    out.append((
        "all", "same transaction count in every regime",
        len(tx_counts) == 1,
        f"{sorted(tx_counts)}",
    ))
    hashes = {r.dataset_sha256 for r in sigmas.values() if r.dataset_sha256}
    out.append((
        "all", "each regime is a distinct dataset",
        len(hashes) == len(sigmas),
        f"{len(hashes)} distinct hashes across {len(sigmas)} regimes",
    ))
    return out


# ---- rendering --------------------------------------------------------

_W = 78


def format_diagnostics(
    reports: dict[str, RegimeDiagnostics],
    config: worlds.WorldConfig,
    checks: list[tuple[str, str, bool, str]],
) -> str:
    out: list[str] = []
    add = out.append

    add("=" * _W)
    add("OBSIDIANCHAIN - CONTROLLED WORLD DIAGNOSTICS (pre-Phase 3.3)")
    add("=" * _W)
    add("  SYNTHETIC worlds. These verify the generator produced the intended")
    add("  distributions. They are not validation of anything: the worlds were")
    add("  written by the same people as the analysis and contain the structure")
    add("  the analysis looks for. No inference has been run.")
    add("")
    add(f"  sigma held at {config.describe()['sigma']:.4f} in every regime "
        f"(Decker & Wattenhofer)")
    add("  topology, observers, clock offsets and ENTITY ASSIGNMENT are shared")
    add("  across regimes; only the entity-origin distribution differs.")
    add("")
    add(f"  entities {config.n_entities}   origins {config.n_origins}   "
        f"observers {config.n_observers}   "
        f"missing-rate {config.missing_observation_rate}")
    add("")

    add("-- regimes " + "-" * (_W - 12))
    for key in ("A", "B", "C", "D", "E"):
        report = reports.get(key)
        if report is None:
            continue
        add(f"  {key}  {report.name}")
        add(f"      expect: {report.expectation}")
    add("")

    add("-- dataset shape " + "-" * (_W - 18))
    add(f"  {'':<4}{'records':>12}{'tx':>10}{'usable':>10}{'entities':>10}"
        f"{'origins':>9}  sha256")
    for key in ("A", "B", "C", "D", "E"):
        r = reports.get(key)
        if r is None:
            continue
        add(f"  {key:<4}{r.n_records:>12,}{r.n_transactions:>10,}"
            f"{r.usable_transactions:>10,}{r.n_entities:>10,}"
            f"{r.n_origins_used:>9,}  {r.dataset_sha256[:16]}")
    add("")

    add("-- origin structure, from ground truth " + "-" * (_W - 40))
    add(f"  {'':<4}{'origins/entity':>16}{'entropy':>10}"
        f"{'overlap med':>13}{'overlap min':>13}{'identical':>11}")
    for key in ("A", "B", "C", "D", "E"):
        r = reports.get(key)
        if r is None:
            continue
        add(f"  {key:<4}{r.origins_per_entity_mean:>16.1f}"
            f"{r.origin_entropy_mean:>10.2f}"
            f"{r.entity_overlap_median:>13.3f}{r.entity_overlap_min:>13.2e}"
            f"{r.identical_entity_pairs:>11,}")
    add("  overlap = total-variation distance between entity distributions")
    add("  (0 = indistinguishable, 1 = disjoint support)")
    add("")

    add("-- separability BY ENTITY, from observations " + "-" * (_W - 46))
    add(f"  {'':<4}{'per-tx ratio':>14}{'rank ratio':>12}{'first-obs':>11}"
        f"{'centroid SE':>13}{'scatter':>10}")
    for key in ("A", "B", "C", "D", "E"):
        r = reports.get(key)
        if r is None:
            continue
        add(f"  {key:<4}{r.per_transaction_ratio:>14.3f}"
            f"{r.rank_ratio:>12.3f}{r.first_observer_lift:>11.3f}"
            f"{r.centroid_separation_se:>13.2f}{r.within_entity_scatter:>10.4f}")
    add("  per-tx / rank ratio near 1.0 = a single transaction carries nothing")
    add("  centroid SE = between-entity distance in standard errors, over ALL")
    add("  usable transactions; >= 2.0 is recoverable aggregate signal")
    add("")

    add("-- specification checks " + "-" * (_W - 25))
    passed = sum(1 for _, _, ok, _ in checks if ok)
    for regime, name, ok, detail in checks:
        add(f"  [{'PASS' if ok else 'FAIL'}] {regime:<4}{name:<42}{detail}")
    add("")
    add(f"  {passed} of {len(checks)} checks passed")
    add("")

    add("-- status " + "-" * (_W - 11))
    control = reports.get("A")
    control_clean = (
        control is not None
        and np.isfinite(control.centroid_separation_se)
        and control.centroid_separation_se < SIGNAL_SE
    )
    if not control_clean:
        add("  CONTROL FAILED. Regime A shows aggregate signal, which means the")
        add("  new generator introduced structure that the frozen dataset does")
        add("  not have. Every other regime's result is suspect until this is")
        add("  explained. Do not proceed to Phase 3.3.")
    elif passed == len(checks):
        add("  All checks passed and the control is clean. The regimes carry the")
        add("  structure their specification describes, so a Phase 3.3 result")
        add("  can be attributed to the method rather than to the input.")
    else:
        add(f"  {len(checks) - passed} check(s) failed. Read them above before")
        add("  proceeding: a regime that does not match its specification will")
        add("  produce an uninterpretable Phase 3.3 result.")
    add("=" * _W)
    return "\n".join(out)


def to_frame(reports: dict[str, RegimeDiagnostics]) -> pd.DataFrame:
    """Diagnostics as a table, for the record."""
    rows = []
    for key in ("A", "B", "C", "D", "E"):
        report = reports.get(key)
        if report is None:
            continue
        row = {k: v for k, v in vars(report).items() if k != "notes"}
        rows.append(row)
    return pd.DataFrame(rows)
